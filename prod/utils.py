import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SiameseNetwork(nn.Module):
    """Red neuronal siamesa para comparación de cartas de Yu-Gi-Oh!"""
    
    def __init__(self, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        
        # Cargar ResNet-101 pre-entrenado
        import torchvision.models as models
        from torchvision.models import ResNet101_Weights
        
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        
        # Remover la capa de clasificación final
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Obtener la dimensión de entrada para la capa de embedding
        # ResNet-101 tiene 2048 características de salida
        backbone_output_dim = 2048
        
        # Capa de embedding
        self.embedding = nn.Linear(backbone_output_dim, embedding_dim)
        
        # Congelar capas del backbone para transfer learning
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        logger.info(f"Siamese Network inicializada con embedding_dim={embedding_dim}")

    def forward_once(self, x):
        """Forward pass para una sola imagen."""
        # Forward pass a través del backbone
        x = self.resnet(x)
        
        # Flatten: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
        x = x.view(x.size(0), -1)
        
        # Proyección a embedding
        x = self.embedding(x)
        
        # Normalización L2 para estabilidad
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, x):
        """Forward pass para compatibilidad con la aplicación."""
        return self.forward_once(x)

class CardEmbeddingManager:
    """Gestor de embeddings optimizado con tensores de 4 dimensiones"""
    
    def __init__(self, model_path: str, dataset_path: str, embedding_cache_path: str = "data/embeddings_cache.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.dataset_path = dataset_path
        self.embedding_cache_path = embedding_cache_path
        
        # Transformaciones para las imágenes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar o generar embeddings
        self.embeddings_tensor, self.card_names = self._load_or_generate_embeddings()
        
    def _load_model(self, model_path: str) -> SiameseNetwork:
        """Cargar el modelo entrenado"""
        model = SiameseNetwork()
        
        # Cargar el checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Verificar si es un checkpoint completo o solo los pesos
        if 'model_state_dict' in checkpoint:
            # Es un checkpoint completo, extraer solo los pesos del modelo
            state_dict = checkpoint['model_state_dict']
            logger.info(f"Checkpoint completo cargado - epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            # Son solo los pesos del modelo
            state_dict = checkpoint
            logger.info("Pesos del modelo cargados directamente")
        
        # Cargar los pesos
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        logger.info(f"Modelo cargado desde {model_path}")
        return model
    
    def _load_or_generate_embeddings(self) -> Tuple[torch.Tensor, List[str]]:
        """Cargar embeddings desde cache o generarlos"""
        if os.path.exists(self.embedding_cache_path):
            logger.info("Cargando embeddings desde cache...")
            with open(self.embedding_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                embeddings_tensor = cache_data['embeddings']
                card_names = cache_data['card_names']
                logger.info(f"Embeddings cargados: {embeddings_tensor.shape}")
                return embeddings_tensor, card_names
        else:
            logger.info("Generando embeddings...")
            return self._generate_embeddings()
    
    def _generate_embeddings(self) -> Tuple[torch.Tensor, List[str]]:
        """Generar embeddings para todas las cartas del dataset"""
        card_images = []
        card_names = []
        
        # Cargar todas las imágenes del dataset
        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.dataset_path, filename)
                try:
                    image = Image.open(image_path).convert('RGB')
                    card_images.append(image)
                    card_names.append(filename)
                except Exception as e:
                    logger.warning(f"Error cargando {filename}: {e}")
        
        logger.info(f"Procesando {len(card_images)} imágenes...")
        
        # Procesar en batches para optimizar memoria
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(card_images), batch_size):
                batch_images = card_images[i:i+batch_size]
                batch_tensors = torch.stack([self.transform(img) for img in batch_images])
                batch_tensors = batch_tensors.to(self.device)
                
                # Generar embeddings para el batch
                batch_embeddings = self.model(batch_tensors)
                all_embeddings.append(batch_embeddings.cpu())
                
                logger.info(f"Procesado batch {i//batch_size + 1}/{(len(card_images) + batch_size - 1)//batch_size}")
        
        # Concatenar todos los embeddings en un tensor de 4 dimensiones
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        
        # Guardar en cache
        cache_data = {
            'embeddings': embeddings_tensor,
            'card_names': card_names
        }
        os.makedirs(os.path.dirname(self.embedding_cache_path), exist_ok=True)
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Embeddings generados y guardados: {embeddings_tensor.shape}")
        return embeddings_tensor, card_names
    
    def find_similar_cards(self, query_image: Image.Image, top_k: int = 10, 
                          similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Encontrar cartas similares usando el tensor de embeddings precalculado"""
        # Preprocesar imagen de consulta
        query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generar embedding de la imagen de consulta
            query_embedding = self.model(query_tensor)
            
            # Calcular similitud con todas las cartas del dataset (operación vectorizada)
            similarities = torch.mm(query_embedding, self.embeddings_tensor.t()).squeeze()
            
            # Obtener top-k resultados
            top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            # Filtrar por umbral de similitud
            results = []
            for sim, idx in zip(top_similarities, top_indices):
                if sim >= similarity_threshold:
                    card_name = self.card_names[idx]
                    results.append((card_name, sim.item()))
            
            return results

def preprocess_real_photo(image_path: str) -> Optional[Image.Image]:
    """Preprocesar foto real para extraer la carta usando OpenCV y warpear a 624x624"""
    try:
        TARGET_SIZE = (624, 624)
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return None
        
        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_contour = None
        max_area = 0
        best_quad = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    best_quad = approx
        if best_quad is not None:
            # Ordenar los puntos del cuadrilátero
            pts = best_quad.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            dst = np.array([
                [0, 0],
                [TARGET_SIZE[0] - 1, 0],
                [TARGET_SIZE[0] - 1, TARGET_SIZE[1] - 1],
                [0, TARGET_SIZE[1] - 1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image_rgb, M, TARGET_SIZE)
            pil_image = Image.fromarray(warped)
            logger.info(f"Carta warpeada exitosamente a {TARGET_SIZE}")
            return pil_image
        else:
            # Si no hay cuadrilátero, usar bounding box más grande
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                if w > 50 and h > 50:
                    margin = 20
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(image_rgb.shape[1] - x, w + 2 * margin)
                    h = min(image_rgb.shape[0] - y, h + 2 * margin)
                    crop = image_rgb[y:y+h, x:x+w]
                    resized = cv2.resize(crop, TARGET_SIZE)
                    pil_image = Image.fromarray(resized)
                    logger.info(f"Carta recortada por bounding box y estirada a {TARGET_SIZE}")
                    return pil_image
            # Fallback: detección por color
            logger.warning("No se encontró cuadrilátero, usando fallback por color o imagen original")
            return _detect_card_by_color_and_resize(image_rgb, TARGET_SIZE)
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        return None

def _detect_card_by_color_and_resize(image_rgb: np.ndarray, target_size=(624, 624)) -> Image.Image:
    """Fallback: detección por color y resize a target_size"""
    try:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_or(mask_yellow, mask_white)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                margin = 30
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image_rgb.shape[1] - x, w + 2 * margin)
                h = min(image_rgb.shape[0] - y, h + 2 * margin)
                crop = image_rgb[y:y+h, x:x+w]
                resized = cv2.resize(crop, target_size)
                pil_image = Image.fromarray(resized)
                logger.info(f"Carta detectada por color y estirada a {target_size}")
                return pil_image
        logger.warning("No se pudo detectar carta por color, usando imagen original")
        resized = cv2.resize(image_rgb, target_size)
        return Image.fromarray(resized)
    except Exception as e:
        logger.error(f"Error en detección por color: {e}")
        resized = cv2.resize(image_rgb, target_size)
        return Image.fromarray(resized)

def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    """Cargar imagen desde path"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error cargando imagen {image_path}: {e}")
        return None 