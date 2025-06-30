"""
Yu-Gi-Oh! Card Recognition Dataset
==================================

Este módulo implementa un dataset personalizado para el entrenamiento de la red neuronal siamesa.
Incluye aumentación de datos realista para simular condiciones del mundo real.

Características:
- Carga de imágenes de cartas de Yu-Gi-Oh!
- Aumentación de datos realista (blur, iluminación, ruido)
- Generación de triplets (anchor, positive, negative)
- Transformaciones compatibles con PyTorch
"""

import os
import random
import logging
from typing import Tuple, List, Optional
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración por defecto
IMAGES_DIR = "../data/yugioh_card_images"
DEFAULT_IMAGE_SIZE = (255, 255)

class CustomImageDataset(Dataset):
    """
    Dataset personalizado para cartas de Yu-Gi-Oh! con aumentación de datos.
    
    Este dataset genera triplets (anchor, positive, negative) para entrenamiento
    de redes neuronales siamesas con triplet loss.
    
    Args:
        image_dir (str): Directorio que contiene las imágenes de las cartas
        augment_prob (float): Probabilidad de aplicar aumentación (0.0-1.0)
        transform: Transformaciones de PyTorch a aplicar
        image_size (tuple): Tamaño de las imágenes (width, height)
        max_images (int): Número máximo de imágenes a cargar (None = todas)
    """
    
    def __init__(self, 
                 image_dir: str = IMAGES_DIR, 
                 augment_prob: float = 0.5, 
                 transform = None,
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                 max_images: Optional[int] = None):
        
        self.image_dir = image_dir
        self.augment_prob = augment_prob
        self.image_size = image_size
        self.transform = transform
        
        # Verificar que el directorio existe
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directorio de imágenes no encontrado: {image_dir}")
        
        # Obtener lista de archivos de imagen
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Limitar número de imágenes si se especifica
        if max_images and len(self.image_files) > max_images:
            self.image_files = self.image_files[:max_images]
            logger.info(f"Limitado a {max_images} imágenes para entrenamiento rápido")
        
        logger.info(f"Dataset inicializado con {len(self.image_files)} imágenes desde {image_dir}")
        
        # Verificar que hay suficientes imágenes
        if len(self.image_files) < 2:
            raise ValueError("Se necesitan al menos 2 imágenes para generar triplets")

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        """
        Aplica aumentación de datos realista a una imagen.
        
        Las aumentaciones incluyen:
        - Desenfoque (simula cámara fuera de foco)
        - Ajustes de iluminación (brillo y contraste)
        - Cambios de color (simula diferentes condiciones de luz)
        - Ruido de sensor (simula ruido de cámara)
        
        Args:
            img: Imagen PIL a procesar
            
        Returns:
            Imagen con aumentación aplicada
        """
        
        # 1. Aplicar desenfoque (cámara fuera de foco o movimiento)
        if random.random() < self.augment_prob:
            blur_types = [ImageFilter.GaussianBlur, ImageFilter.BoxBlur]
            blur = random.choice(blur_types)(radius=random.uniform(1.0, 5.0))
            img = img.filter(blur)
        
        # 2. Ajustes de iluminación
        if random.random() < self.augment_prob:
            # Brillo
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
            
            # Contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        
        # 3. Cambios de color (diferentes condiciones de luz)
        if random.random() < self.augment_prob:
            r, g, b = img.split()
            factors = [random.uniform(0.7, 1.3) for _ in range(3)]
            r = r.point(lambda i: i * factors[0])
            g = g.point(lambda i: i * factors[1])
            b = b.point(lambda i: i * factors[2])
            img = Image.merge("RGB", (r, g, b))

        # 4. Ruido de sensor (simula ruido de cámara)
        if random.random() < self.augment_prob:
            img = self._add_realistic_noise(img)
        
        return img

    def _add_realistic_noise(self, img: Image.Image) -> Image.Image:
        """
        Añade ruido realista de sensor a la imagen.
        
        Incluye:
        - Ruido de Poisson (ruido de fotones)
        - Ruido Gaussiano (ruido de lectura del sensor)
        
        Args:
            img: Imagen PIL
            
        Returns:
            Imagen con ruido añadido
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Ruido de Poisson (ruido de fotones)
        poisson_noise = np.random.poisson(img_array * 20) / 20.0
        img_array = 0.5 * img_array + 0.5 * poisson_noise
        
        # Ruido Gaussiano (ruido de lectura del sensor)
        gaussian_noise = np.random.normal(0, 0.05, img_array.shape)
        img_array += gaussian_noise
        
        # Recortar valores y convertir de vuelta
        img_array = np.clip(img_array, 0, 1) * 255
        return Image.fromarray(img_array.astype(np.uint8))

    def __len__(self) -> int:
        """Retorna el número de imágenes en el dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int, test: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Obtiene un triplet (anchor, positive, negative) del dataset.
        
        Args:
            idx: Índice de la imagen anchor
            test: Si es True, muestra la imagen positiva (para debugging)
            
        Returns:
            Tupla con tres pares (imagen, card_id):
            - (anchor_image, anchor_id)
            - (positive_image, positive_id) 
            - (negative_image, negative_id)
        """
        # Obtener imagen anchor
        anchor_filename = self.image_files[idx]
        anchor_path = os.path.join(self.image_dir, anchor_filename)
        
        try:
            image_anchor = Image.open(anchor_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error cargando imagen {anchor_path}: {e}")
            # Retornar una imagen aleatoria como fallback
            fallback_idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(fallback_idx, test)
        
        # Generar imagen positiva (versión aumentada de la anchor)
        image_positive = self._apply_augmentations(image_anchor.copy())
        
        # Mostrar imagen positiva si estamos en modo test
        if test:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image_anchor)
            plt.title(f"Anchor: {anchor_filename}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(image_positive)
            plt.title(f"Positive (augmented): {anchor_filename}")
            plt.axis('off')
            plt.show()
        
        # Generar imagen negative (carta diferente)
        max_attempts = 10
        for attempt in range(max_attempts):
            negative_idx = random.randint(0, len(self.image_files) - 1)
            negative_filename = self.image_files[negative_idx]
            
            # Asegurar que es una carta diferente
            if negative_filename != anchor_filename:
                negative_path = os.path.join(self.image_dir, negative_filename)
                try:
                    image_negative = Image.open(negative_path).convert("RGB")
                    break
                except Exception as e:
                    logger.warning(f"Error cargando imagen negativa {negative_path}: {e}")
                    continue
        else:
            # Si no se pudo encontrar una imagen negativa válida
            logger.error("No se pudo generar imagen negativa válida")
            raise RuntimeError("No se pudo generar triplet válido")
        
        # Aplicar transformaciones si están definidas
        if self.transform:
            image_anchor = self.transform(image_anchor)
            image_positive = self.transform(image_positive)
            image_negative = self.transform(image_negative)
        
        # Extraer IDs de las cartas
        anchor_id = anchor_filename.split(".")[0]
        negative_id = negative_filename.split(".")[0]

        return (image_anchor, anchor_id), (image_positive, anchor_id), (image_negative, negative_id)

    def get_sample_images(self, num_samples: int = 5) -> List[Image.Image]:
        """
        Obtiene una muestra de imágenes del dataset para visualización.
        
        Args:
            num_samples: Número de imágenes a mostrar
            
        Returns:
            Lista de imágenes PIL
        """
        sample_indices = random.sample(range(len(self.image_files)), 
                                     min(num_samples, len(self.image_files)))
        
        images = []
        for idx in sample_indices:
            filename = self.image_files[idx]
            img_path = os.path.join(self.image_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Error cargando imagen de muestra {img_path}: {e}")
        
        return images

    def visualize_augmentations(self, num_examples: int = 3):
        """
        Visualiza ejemplos de aumentación de datos.
        
        Args:
            num_examples: Número de ejemplos a mostrar
        """
        if num_examples > len(self.image_files):
            num_examples = len(self.image_files)
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
        
        for i in range(num_examples):
            idx = random.randint(0, len(self.image_files) - 1)
            filename = self.image_files[idx]
            img_path = os.path.join(self.image_dir, filename)
            
            try:
                original = Image.open(img_path).convert("RGB")
                augmented1 = self._apply_augmentations(original.copy())
                augmented2 = self._apply_augmentations(original.copy())
                
                if num_examples == 1:
                    axes[0].imshow(original)
                    axes[0].set_title(f"Original: {filename}")
                    axes[0].axis('off')
                    
                    axes[1].imshow(augmented1)
                    axes[1].set_title("Augmented 1")
                    axes[1].axis('off')
                    
                    axes[2].imshow(augmented2)
                    axes[2].set_title("Augmented 2")
                    axes[2].axis('off')
                else:
                    axes[i, 0].imshow(original)
                    axes[i, 0].set_title(f"Original: {filename}")
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(augmented1)
                    axes[i, 1].set_title("Augmented 1")
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(augmented2)
                    axes[i, 2].set_title("Augmented 2")
                    axes[i, 2].axis('off')
                    
            except Exception as e:
                logger.error(f"Error en visualización de aumentación: {e}")
        
        plt.tight_layout()
        plt.show()


def create_dataloader(image_dir: str = IMAGES_DIR,
                     batch_size: int = 32,
                     augment_prob: float = 0.5,
                     image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                     num_workers: int = 4,
                     shuffle: bool = True,
                     max_images: Optional[int] = None) -> DataLoader:
    """
    Crea un DataLoader configurado para el dataset de cartas de Yu-Gi-Oh!.
    
    Args:
        image_dir: Directorio con las imágenes
        batch_size: Tamaño del batch
        augment_prob: Probabilidad de aumentación
        image_size: Tamaño de las imágenes
        num_workers: Número de workers para carga paralela
        shuffle: Si mezclar los datos
        max_images: Número máximo de imágenes a cargar
        
    Returns:
        DataLoader configurado
    """
    # Transformaciones por defecto
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Crear dataset
    dataset = CustomImageDataset(
        image_dir=image_dir,
        augment_prob=augment_prob,
        transform=transform,
        image_size=image_size,
        max_images=max_images
    )
    
    # Crear dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Ejemplo de uso y testing
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear dataset
    dataset = CustomImageDataset(
        image_dir=os.path.abspath(IMAGES_DIR),
        augment_prob=0.5,
        max_images=100  # Solo 100 imágenes para testing rápido
    )
    
    print(f"Dataset creado con {len(dataset)} imágenes")
    
    # Visualizar aumentaciones
    print("Visualizando ejemplos de aumentación...")
    dataset.visualize_augmentations(num_examples=3)
    
    # Crear dataloader
    dataloader = create_dataloader(
        image_dir=os.path.abspath(IMAGES_DIR),
        batch_size=16,
        max_images=100
    )
    
    # Probar un batch
    print("Probando dataloader...")
    for batch_idx, ((anchors, anchor_ids), (positives, pos_ids), (negatives, neg_ids)) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Anchors shape: {anchors.shape}")
        print(f"  Positives shape: {positives.shape}")
        print(f"  Negatives shape: {negatives.shape}")
        print(f"  Anchor IDs: {anchor_ids[:3]}...")
        print(f"  Positive IDs: {pos_ids[:3]}...")
        print(f"  Negative IDs: {neg_ids[:3]}...")
        break
    
    print("✅ Dataset y DataLoader funcionando correctamente!")