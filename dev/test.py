"""
Yu-Gi-Oh! Card Recognition - Testing and Evaluation Module
==========================================================

Este módulo implementa las pruebas y evaluación del modelo siamesa entrenado
para reconocimiento de cartas de Yu-Gi-Oh!.

Características:
- Carga y validación de modelos entrenados
- Evaluación de similitud entre cartas
- Búsqueda de cartas más similares
- Visualización de resultados
- Métricas de rendimiento
- Pruebas de robustez
"""

import os
import sys
import hashlib
import logging
import argparse
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import SiameseNetwork
from dataset import CustomImageDataset

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración por defecto
DEFAULT_IMAGE_SIZE = (255, 255)
DEFAULT_EMBEDDING_DIM = 64
IMAGES_DIR = "../data/yugioh_card_images"


class ModelEvaluator:
    """
    Evaluador del modelo siamesa para cartas de Yu-Gi-Oh!.
    
    Características:
    - Carga y validación de modelos
    - Cálculo de embeddings
    - Evaluación de similitud
    - Búsqueda de cartas similares
    - Visualización de resultados
    """
    
    def __init__(self, model_path: str, images_dir: str = IMAGES_DIR,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE):
        """
        Inicializa el evaluador del modelo.
        
        Args:
            model_path: Ruta al modelo entrenado
            images_dir: Directorio con las imágenes de las cartas
            embedding_dim: Dimensión del embedding
            image_size: Tamaño de las imágenes
        """
        self.model_path = model_path
        self.images_dir = images_dir
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Configurar transformaciones
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelo
        self.model = self._load_model()
        
        # Cache de embeddings
        self.embeddings_cache = {}
        
        logger.info("ModelEvaluator inicializado correctamente")

    def _load_model(self) -> SiameseNetwork:
        """
        Carga y valida el modelo entrenado.
        
        Returns:
            Modelo cargado y configurado
            
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe
            RuntimeError: Si hay error al cargar el modelo
        """
        # Verificar que el archivo existe
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {self.model_path}")
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(self.model_path)
        logger.info(f"Tamaño del archivo: {file_size / (1024**2):.2f} MB")
        
        # Calcular hash MD5 para verificación
        logger.info("Calculando hash MD5...")
        hash_md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        logger.info(f"MD5: {hash_md5.hexdigest()}")
        
        # Cargar modelo
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info("✅ Archivo cargado CORRECTAMENTE")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo: {str(e)}")
        
        # Crear modelo
        model = SiameseNetwork(embedding_dim=self.embedding_dim)
        
        # Determinar si es un checkpoint completo o solo state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Es un checkpoint completo
            logger.info("📦 Cargando checkpoint completo...")
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"   Época: {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"   Loss: {checkpoint.get('loss', 'N/A')}")
        else:
            # Es solo el state_dict del modelo
            logger.info("📦 Cargando state_dict del modelo...")
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info("✅ Modelo cargado y listo para usar")
        return model

    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Carga y transforma una imagen.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Tensor de la imagen transformada
        """
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img).unsqueeze(0)  # Añadir dimensión de batch
        except Exception as e:
            logger.error(f"Error cargando imagen {image_path}: {e}")
            raise

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Obtiene el embedding de una imagen.
        
        Args:
            image_tensor: Tensor de la imagen
            
        Returns:
            Embedding de la imagen
        """
        with torch.no_grad():
            embedding = self.model.forward_once(image_tensor.to(self.device))
            return embedding

    def calculate_similarity(self, embedding1: torch.Tensor, 
                           embedding2: torch.Tensor) -> float:
        """
        Calcula la similitud coseno entre dos embeddings.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        return similarity.item()

    def find_most_similar_cards(self, input_image_path: str, 
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Encuentra las cartas más similares a una imagen de entrada.
        
        Args:
            input_image_path: Ruta de la imagen de entrada
            top_k: Número de resultados más similares
            
        Returns:
            Lista de tuplas (card_id, similarity_score)
        """
        logger.info(f"🔍 Buscando cartas similares a: {os.path.basename(input_image_path)}")
        
        # Procesar imagen de entrada
        input_embedding = self.get_embedding(self.load_image(input_image_path))
        
        # Obtener lista de imágenes del dataset
        dataset_images = [f for f in os.listdir(self.images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"📁 Comparando con {len(dataset_images)} cartas en el dataset")
        
        # Comparar con cada carta del dataset
        similarities = []
        
        for card_image in tqdm(dataset_images, desc="Comparando cartas"):
            try:
                card_path = os.path.join(self.images_dir, card_image)
                card_embedding = self.get_embedding(self.load_image(card_path))
                
                similarity = self.calculate_similarity(input_embedding, card_embedding)
                card_id = card_image.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                
                similarities.append((card_id, similarity))
                
            except Exception as e:
                logger.warning(f"Error procesando {card_image}: {e}")
                continue
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ Encontradas {len(similarities)} coincidencias")
        return similarities[:top_k]

    def evaluate_model_robustness(self, test_images: List[str]) -> Dict:
        """
        Evalúa la robustez del modelo con diferentes imágenes.
        
        Args:
            test_images: Lista de imágenes de prueba
            
        Returns:
            Diccionario con métricas de robustez
        """
        logger.info("🧪 Evaluando robustez del modelo...")
        
        results = {
            'self_similarity': [],
            'cross_similarity': [],
            'processing_times': []
        }
        
        # Cargar embeddings de las imágenes de prueba
        embeddings = {}
        for img_name in test_images:
            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path):
                try:
                    start_time = time.time()
                    img_tensor = self.load_image(img_path)
                    embedding = self.get_embedding(img_tensor)
                    processing_time = time.time() - start_time
                    
                    embeddings[img_name] = embedding
                    results['processing_times'].append(processing_time)
                    
                    logger.info(f"✅ {img_name} procesada en {processing_time:.3f}s")
                except Exception as e:
                    logger.error(f"❌ Error procesando {img_name}: {e}")
            else:
                logger.warning(f"⚠️  {img_name} no encontrada")
        
        # Calcular similitudes
        image_names = list(embeddings.keys())
        
        # Similitud consigo misma (debería ser 1.0)
        for img_name in image_names:
            same_similarity = self.calculate_similarity(
                embeddings[img_name], 
                embeddings[img_name]
            )
            results['self_similarity'].append(same_similarity)
        
        # Similitud entre cartas diferentes
        for i in range(len(image_names)):
            for j in range(i + 1, len(image_names)):
                cross_similarity = self.calculate_similarity(
                    embeddings[image_names[i]], 
                    embeddings[image_names[j]]
                )
                results['cross_similarity'].append(cross_similarity)
        
        # Calcular estadísticas
        stats = {
            'avg_self_similarity': np.mean(results['self_similarity']),
            'std_self_similarity': np.std(results['self_similarity']),
            'avg_cross_similarity': np.mean(results['cross_similarity']),
            'std_cross_similarity': np.std(results['cross_similarity']),
            'avg_processing_time': np.mean(results['processing_times']),
            'total_images_processed': len(embeddings)
        }
        
        logger.info("📊 Estadísticas de robustez:")
        logger.info(f"  Similitud consigo misma: {stats['avg_self_similarity']:.4f} ± {stats['std_self_similarity']:.4f}")
        logger.info(f"  Similitud entre cartas: {stats['avg_cross_similarity']:.4f} ± {stats['std_cross_similarity']:.4f}")
        logger.info(f"  Tiempo promedio: {stats['avg_processing_time']:.3f}s")
        
        return stats

    def visualize_similarity_matrix(self, test_images: List[str]):
        """
        Visualiza la matriz de similitud entre imágenes de prueba.
        
        Args:
            test_images: Lista de imágenes de prueba
        """
        logger.info("📊 Generando matriz de similitud...")
        
        # Cargar embeddings
        embeddings = {}
        valid_images = []
        
        for img_name in test_images:
            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path):
                try:
                    img_tensor = self.load_image(img_path)
                    embedding = self.get_embedding(img_tensor)
                    embeddings[img_name] = embedding
                    valid_images.append(img_name)
                except Exception as e:
                    logger.warning(f"Error procesando {img_name}: {e}")
        
        if len(valid_images) < 2:
            logger.error("Se necesitan al menos 2 imágenes válidas para la matriz")
            return
        
        # Crear matriz de similitud
        n = len(valid_images)
        similarity_matrix = np.zeros((n, n))
        
        for i, img1 in enumerate(valid_images):
            for j, img2 in enumerate(valid_images):
                similarity = self.calculate_similarity(
                    embeddings[img1], 
                    embeddings[img2]
                )
                similarity_matrix[i, j] = similarity
        
        # Visualizar matriz
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Similitud Coseno')
        plt.title('Matriz de Similitud entre Cartas de Prueba')
        plt.xlabel('Carta')
        plt.ylabel('Carta')
        plt.xticks(range(n), [img.replace('.jpg', '') for img in valid_images], rotation=45)
        plt.yticks(range(n), [img.replace('.jpg', '') for img in valid_images])
        
        # Añadir valores en las celdas
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()

    def demo_card_search(self, test_image: str, top_k: int = 5):
        """
        Demuestra la búsqueda de cartas similares.
        
        Args:
            test_image: Imagen de prueba
            top_k: Número de resultados a mostrar
        """
        logger.info(f"🎯 DEMOSTRACIÓN: Búsqueda de cartas similares a {test_image}")
        
        test_image_path = os.path.join(self.images_dir, test_image)
        if not os.path.exists(test_image_path):
            logger.error(f"Imagen de prueba no encontrada: {test_image_path}")
            return
        
        # Encontrar cartas similares
        similar_cards = self.find_most_similar_cards(test_image_path, top_k)
        
        # Mostrar resultados
        print(f"\n🎴 Top {top_k} cartas más similares a {test_image}:")
        print("-" * 60)
        
        for i, (card_id, similarity) in enumerate(similar_cards, 1):
            print(f"{i:2d}. Carta ID: {card_id:>10} - Similitud: {similarity:.4f}")
        
        # Visualizar resultados
        self._visualize_search_results(test_image_path, similar_cards)

    def _visualize_search_results(self, query_image_path: str, 
                                similar_cards: List[Tuple[str, float]]):
        """
        Visualiza los resultados de búsqueda.
        
        Args:
            query_image_path: Ruta de la imagen consulta
            similar_cards: Lista de cartas similares
        """
        fig, axes = plt.subplots(1, len(similar_cards) + 1, figsize=(15, 3))
        
        # Mostrar imagen consulta
        query_img = Image.open(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Consulta")
        axes[0].axis('off')
        
        # Mostrar cartas similares
        for i, (card_id, similarity) in enumerate(similar_cards):
            card_path = os.path.join(self.images_dir, f"{card_id}.jpg")
            if os.path.exists(card_path):
                card_img = Image.open(card_path)
                axes[i + 1].imshow(card_img)
                axes[i + 1].set_title(f"ID: {card_id}\nSim: {similarity:.3f}")
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Función principal para ejecutar las pruebas."""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Pruebas del modelo siamesa para Yu-Gi-Oh!')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado')
    parser.add_argument('--images_dir', type=str, default=IMAGES_DIR,
                       help='Directorio con las imágenes de las cartas')
    parser.add_argument('--test_images', nargs='+', 
                       default=["62121.jpg", "88472456.jpg", "31339260.jpg"],
                       help='Imágenes de prueba')
    parser.add_argument('--demo_image', type=str, default="62121.jpg",
                       help='Imagen para demostración de búsqueda')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Número de resultados similares a mostrar')
    
    args = parser.parse_args()
    
    # Crear evaluador
    try:
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            images_dir=args.images_dir
        )
    except Exception as e:
        logger.error(f"Error inicializando evaluador: {e}")
        return
    
    # Ejecutar pruebas de robustez
    print("\n🧪 PRUEBAS DE ROBUSTEZ")
    print("=" * 50)
    robustness_stats = evaluator.evaluate_model_robustness(args.test_images)
    
    # Visualizar matriz de similitud
    print("\n📊 MATRIZ DE SIMILITUD")
    print("=" * 50)
    evaluator.visualize_similarity_matrix(args.test_images)
    
    # Demostración de búsqueda
    print("\n🎯 DEMOSTRACIÓN DE BÚSQUEDA")
    print("=" * 50)
    evaluator.demo_card_search(args.demo_image, args.top_k)
    
    print("\n✅ Todas las pruebas completadas exitosamente!")


if __name__ == '__main__':
    import time
    main()

