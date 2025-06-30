# Yu-Gi-Oh! Card Recognition Configuration
# ================================================

# Directorios
IMAGES_DIR = "data/yugioh_card_images"
MODELS_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "results"

# Configuración del modelo
EMBEDDING_DIM = 64
IMAGE_SIZE = (255, 255)
MARGIN = 1.0

# Configuración de entrenamiento
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Configuración de datos
AUGMENT_PROB = 0.5
MAX_IMAGES = None  # None = todas las imágenes

# Configuración de pruebas
TOP_K_RESULTS = 5

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Configuración de API
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"
IMAGE_TAG = "image_url_cropped"
FORMAT_TAG = "GOAT"
TIMEOUT = 10

# Configuración de aumentación de datos
BLUR_RADIUS_RANGE = (1.0, 5.0)
BRIGHTNESS_RANGE = (0.6, 1.4)
CONTRAST_RANGE = (0.6, 1.4)
COLOR_FACTOR_RANGE = (0.7, 1.3)
NOISE_STD = 0.05

# Configuración de early stopping
EARLY_STOPPING_PATIENCE = 5

# Configuración de checkpoints
SAVE_CHECKPOINTS = True
SAVE_BEST_MODEL = True

# Configuración de visualización
FIGURE_SIZE = (10, 6)
DEFAULT_CMAP = 'viridis'

# Configuración de dispositivos
USE_CUDA = True
NUM_WORKERS = 4

# Configuración de normalización (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225] 