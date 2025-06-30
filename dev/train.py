"""
Yu-Gi-Oh! Card Recognition - Training Module
============================================

Este módulo implementa el entrenamiento de la red neuronal siamesa para reconocimiento
de cartas de Yu-Gi-Oh! utilizando Triplet Loss.

Características:
- Red neuronal siamesa basada en ResNet-101
- Triplet Loss con margen configurable
- Aumentación de datos realista
- Guardado de checkpoints
- Visualización del progreso de entrenamiento
- Soporte para GPU/CPU
"""

import os
import time
import logging
import argparse
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import CustomImageDataset, create_dataloader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración por defecto
IMAGES_DIR = "../data/yugioh_card_images"
DEFAULT_IMAGE_SIZE = (255, 255)
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_MARGIN = 1.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10


class SiameseNetwork(nn.Module):
    """
    Red neuronal siamesa para reconocimiento de cartas de Yu-Gi-Oh!.
    
    Arquitectura:
    - Backbone: ResNet-101 pre-entrenado (transfer learning)
    - Embedding Layer: Capa lineal que reduce características a embedding_dim
    - Normalización: L2-normalization para estabilidad
    
    Args:
        embedding_dim (int): Dimensión del embedding de salida
        freeze_backbone (bool): Si congelar las capas del backbone
    """
    
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, 
                 freeze_backbone: bool = True):
        super(SiameseNetwork, self).__init__()
        
        # Cargar ResNet-101 pre-entrenado
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        
        # Remover la capa de clasificación final
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Obtener la dimensión de entrada para la capa de embedding
        # ResNet-101 tiene 2048 características de salida
        backbone_output_dim = 2048
        
        # Capa de embedding
        self.embedding = nn.Linear(backbone_output_dim, embedding_dim)
        
        # Congelar capas del backbone si se especifica
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            logger.info("Backbone ResNet-101 congelado para transfer learning")
        
        logger.info(f"Siamese Network inicializada con embedding_dim={embedding_dim}")

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para una sola imagen.
        
        Args:
            x: Tensor de entrada [batch_size, 3, height, width]
            
        Returns:
            Embedding normalizado [batch_size, embedding_dim]
        """
        # Forward pass a través del backbone
        x = self.resnet(x)
        
        # Flatten: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
        x = x.view(x.size(0), -1)
        
        # Proyección a embedding
        x = self.embedding(x)
        
        # Normalización L2 para estabilidad
        x = F.normalize(x, p=2, dim=1)
        
        return x

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, 
                input3: torch.Tensor) -> tuple:
        """
        Forward pass para un triplet (anchor, positive, negative).
        
        Args:
            input1: Imagen anchor
            input2: Imagen positive
            input3: Imagen negative
            
        Returns:
            Tupla con tres embeddings: (anchor_emb, positive_emb, negative_emb)
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3


class TripletLoss(nn.Module):
    """
    Triplet Loss para entrenamiento de redes siamesas.
    
    El triplet loss aprende a:
    - Acercar embeddings de la misma clase (anchor y positive)
    - Separar embeddings de clases diferentes (anchor y negative)
    
    Args:
        margin (float): Margen mínimo entre distancias positiva y negativa
    """
    
    def __init__(self, margin: float = DEFAULT_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin
        logger.info(f"Triplet Loss inicializado con margin={margin}")

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Calcula el triplet loss.
        
        Args:
            anchor: Embedding de la imagen anchor
            positive: Embedding de la imagen positive
            negative: Embedding de la imagen negative
            
        Returns:
            Loss promedio del batch
        """
        # Calcular distancias euclidianas al cuadrado
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


class TrainingManager:
    """
    Gestor de entrenamiento con funcionalidades avanzadas.
    
    Características:
    - Guardado automático de checkpoints
    - Visualización del progreso
    - Early stopping
    - Logging detallado
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, device: torch.device,
                 save_dir: str = "models"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        
        # Crear directorio de guardado
        os.makedirs(save_dir, exist_ok=True)
        
        # Historial de entrenamiento
        self.train_losses = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 5  # Early stopping patience
        
        logger.info(f"TrainingManager inicializado. Guardando en: {save_dir}")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Guarda un checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses
        }
        
        # Guardar checkpoint regular
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Mejor modelo guardado en {best_path}")
        
        logger.info(f"Checkpoint guardado: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Carga un checkpoint del modelo."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            logger.info(f"Checkpoint cargado desde {checkpoint_path}")
            return checkpoint['epoch']
        else:
            logger.warning(f"Checkpoint no encontrado: {checkpoint_path}")
            return 0

    def plot_training_progress(self):
        """Visualiza el progreso del entrenamiento."""
        if len(self.train_losses) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, 'b-', label='Training Loss')
            plt.title('Progreso del Entrenamiento')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'))
            plt.show()

    def should_stop_early(self, current_loss: float) -> bool:
        """Determina si se debe aplicar early stopping."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping activado después de {self.patience} épocas sin mejora")
                return True
            return False


def train_model(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                num_epochs: int = DEFAULT_EPOCHS,
                save_dir: str = "models",
                resume_from: Optional[str] = None) -> Dict:
    """
    Entrena el modelo siamesa.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con los datos de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: Dispositivo (CPU/GPU)
        num_epochs: Número de épocas
        save_dir: Directorio para guardar checkpoints
        resume_from: Ruta del checkpoint para continuar entrenamiento
        
    Returns:
        Diccionario con métricas de entrenamiento
    """
    
    # Inicializar gestor de entrenamiento
    manager = TrainingManager(model, optimizer, criterion, device, save_dir)
    
    # Cargar checkpoint si se especifica
    start_epoch = 0
    if resume_from:
        start_epoch = manager.load_checkpoint(resume_from)
    
    model.train()
    logger.info(f"Iniciando entrenamiento por {num_epochs} épocas desde la época {start_epoch}")
    
    training_stats = {
        'train_losses': [],
        'epochs_completed': 0,
        'best_loss': float('inf'),
        'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        # Barra de progreso para la época
        pbar = tqdm(dataloader, desc=f'Época {epoch+1}/{num_epochs}')
        
        for batch_idx, ((anchors, anchor_ids), (positives, pos_ids), (negatives, neg_ids)) in enumerate(pbar):
            # Mover datos al dispositivo
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives)
            
            # Calcular loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Actualizar estadísticas
            running_loss += loss.item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calcular loss promedio de la época
        epoch_loss = running_loss / len(dataloader)
        manager.train_losses.append(epoch_loss)
        training_stats['train_losses'].append(epoch_loss)
        
        # Tiempo de la época
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        logger.info(f"Época [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Tiempo: {epoch_time:.2f}s")
        
        # Guardar checkpoint
        is_best = epoch_loss < training_stats['best_loss']
        if is_best:
            training_stats['best_loss'] = epoch_loss
        
        manager.save_checkpoint(epoch + 1, epoch_loss, is_best)
        
        # Early stopping
        if manager.should_stop_early(epoch_loss):
            logger.info("Entrenamiento detenido por early stopping")
            break
    
    # Tiempo total de entrenamiento
    total_time = time.time() - start_time
    training_stats['training_time'] = total_time
    training_stats['epochs_completed'] = epoch + 1
    
    # Visualizar progreso
    manager.plot_training_progress()
    
    logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")
    logger.info(f"Mejor loss: {training_stats['best_loss']:.4f}")
    
    return training_stats


def main():
    """Función principal para ejecutar el entrenamiento."""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de red siamesa para Yu-Gi-Oh!')
    parser.add_argument('--images_dir', type=str, default=IMAGES_DIR,
                       help='Directorio con las imágenes de las cartas')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                       help='Tasa de aprendizaje')
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                       help='Margen del triplet loss')
    parser.add_argument('--embedding_dim', type=int, default=DEFAULT_EMBEDDING_DIM,
                       help='Dimensión del embedding')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directorio para guardar modelos')
    parser.add_argument('--resume', type=str, default=None,
                       help='Ruta del checkpoint para continuar entrenamiento')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Número máximo de imágenes para entrenamiento rápido')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    # Crear modelo
    model = SiameseNetwork(embedding_dim=args.embedding_dim)
    model = model.to(device)
    
    # Crear optimizador y función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripletLoss(margin=args.margin)
    
    # Crear dataloader
    dataloader = create_dataloader(
        image_dir=args.images_dir,
        batch_size=args.batch_size,
        max_images=args.max_images
    )
    
    logger.info(f"Dataloader creado con {len(dataloader.dataset)} imágenes")
    
    # Entrenar modelo
    training_stats = train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        resume_from=args.resume
    )
    
    # Guardar modelo final
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Modelo final guardado en: {final_model_path}")
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*50)
    print(f"Épocas completadas: {training_stats['epochs_completed']}")
    print(f"Mejor loss: {training_stats['best_loss']:.4f}")
    print(f"Tiempo total: {training_stats['training_time']:.2f} segundos")
    print(f"Modelo guardado en: {args.save_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
