"""
Yu-Gi-Oh! Card Recognition - Web Application
===========================================

Aplicación web Flask para reconocimiento de cartas de Yu-Gi-Oh!.
Permite subir una imagen y encontrar la carta más similar en el dataset.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import time

# Importar módulos del proyecto
import sys
sys.path.append('dev')
from train import SiameseNetwork

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = Flask(__name__)
app.config['SECRET_KEY'] = 'yugioh-card-recognition-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configuración del modelo
MODEL_PATH = 'models/best_model.pth'
IMAGES_DIR = 'data/yugioh_card_images'
EMBEDDING_DIM = 64
IMAGE_SIZE = (255, 255)
SIMILARITY_THRESHOLD = 0.7  # Umbral para considerar una carta como "encontrada"

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar transformaciones
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class CardRecognizer:
    """Clase para reconocimiento de cartas."""
    
    def __init__(self):
        """Inicializar el reconocedor de cartas."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.dataset_embeddings = {}
        self.dataset_images = []
        self._load_dataset_info()
        
    def _load_model(self):
        """Cargar el modelo entrenado."""
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
            
            # Determinar si es checkpoint completo o solo state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Modelo cargado desde checkpoint (época {checkpoint.get('epoch', 'N/A')})")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Modelo cargado desde state_dict")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def _load_dataset_info(self):
        """Cargar información del dataset."""
        try:
            self.dataset_images = [f for f in os.listdir(IMAGES_DIR) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"Dataset cargado: {len(self.dataset_images)} imágenes")
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            self.dataset_images = []
    
    def get_embedding(self, image_path):
        """Obtener embedding de una imagen."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.forward_once(img_tensor)
                return embedding
        except Exception as e:
            logger.error(f"Error obteniendo embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calcular similitud coseno entre dos embeddings."""
        try:
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
            return similarity.item()
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0
    
    def find_similar_cards(self, uploaded_image_path, top_k=5):
        """Encontrar cartas similares a la imagen subida."""
        start_time = time.time()
        
        # Obtener embedding de la imagen subida
        query_embedding = self.get_embedding(uploaded_image_path)
        if query_embedding is None:
            return {"error": "No se pudo procesar la imagen"}
        
        # Comparar con todas las cartas del dataset
        similarities = []
        
        for card_image in self.dataset_images:
            try:
                card_path = os.path.join(IMAGES_DIR, card_image)
                card_embedding = self.get_embedding(card_path)
                
                if card_embedding is not None:
                    similarity = self.calculate_similarity(query_embedding, card_embedding)
                    card_id = card_image.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                    
                    similarities.append({
                        'card_id': card_id,
                        'image_name': card_image,
                        'similarity': similarity
                    })
            except Exception as e:
                logger.warning(f"Error procesando {card_image}: {e}")
                continue
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Tomar los top_k resultados
        top_results = similarities[:top_k]
        
        # Determinar si se encontró la carta
        best_match = top_results[0] if top_results else None
        found_card = best_match and best_match['similarity'] >= SIMILARITY_THRESHOLD
        
        processing_time = time.time() - start_time
        
        return {
            'found_card': found_card,
            'best_match': best_match,
            'top_results': top_results,
            'processing_time': processing_time,
            'total_cards_compared': len(similarities)
        }

# Inicializar reconocedor
try:
    recognizer = CardRecognizer()
    logger.info("✅ Reconocedor de cartas inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando reconocedor: {e}")
    recognizer = None

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesar imagen subida y encontrar cartas similares."""
    if recognizer is None:
        return jsonify({'error': 'Reconocedor no disponible'})
    
    try:
        # Verificar si se subió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se subió ningún archivo'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'})
        
        # Verificar tipo de archivo
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Solo se permiten archivos JPG, JPEG o PNG'})
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Procesar imagen
        results = recognizer.find_similar_cards(filepath, top_k=5)
        
        # Limpiar archivo temporal
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        return jsonify({'error': f'Error procesando imagen: {str(e)}'})

@app.route('/card/<card_id>')
def get_card_image(card_id):
    """Obtener imagen de una carta específica."""
    try:
        # Buscar la imagen en el dataset
        for filename in os.listdir(IMAGES_DIR):
            if filename.startswith(card_id) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return send_from_directory(IMAGES_DIR, filename)
        
        return jsonify({'error': 'Carta no encontrada'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Obtener estadísticas del sistema."""
    try:
        total_cards = len(recognizer.dataset_images) if recognizer else 0
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) if os.path.exists(MODEL_PATH) else 0
        
        return jsonify({
            'total_cards': total_cards,
            'model_size_mb': round(model_size, 2),
            'device': str(recognizer.device) if recognizer else 'N/A',
            'similarity_threshold': SIMILARITY_THRESHOLD
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if recognizer is None:
        print("❌ No se pudo inicializar el reconocedor. Verifica que el modelo existe.")
        exit(1)
    
    print("🎴 Yu-Gi-Oh! Card Recognition Web App")
    print("=" * 50)
    print(f"📊 Dataset: {len(recognizer.dataset_images)} cartas")
    print(f"🤖 Modelo: {MODEL_PATH}")
    print(f"💻 Dispositivo: {recognizer.device}")
    print(f"🎯 Umbral de similitud: {SIMILARITY_THRESHOLD}")
    print("=" * 50)
    print("🌐 Servidor iniciando en http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 