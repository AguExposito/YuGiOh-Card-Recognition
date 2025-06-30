# Yu-Gi-Oh! Card Recognition System

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de reconocimiento de cartas de Yu-Gi-Oh! utilizando una **Red Neuronal Siamesa** con **Triplet Loss**. El sistema es capaz de identificar cartas de Yu-Gi-Oh! a partir de imágenes, incluso cuando estas presentan variaciones como:

- Diferentes condiciones de iluminación
- Desenfoque de cámara
- Ruido de sensor
- Cambios de color/contraste
- Rotaciones y distorsiones

## 🏗️ Arquitectura del Sistema

### Red Neuronal Siamesa
- **Backbone**: ResNet-101 pre-entrenado (transfer learning)
- **Embedding Layer**: Capa lineal que reduce las características a 64 dimensiones
- **Normalización**: L2-normalization para estabilidad numérica

### Triplet Loss
El modelo utiliza **Triplet Loss** que aprende a:
- Acercar embeddings de la misma carta (anchor y positive)
- Separar embeddings de cartas diferentes (anchor y negative)
- Margin: 1.0 (distancia mínima entre clases)

## 📁 Estructura del Proyecto

```
YuGiOh-Card-Recognition/
├── data/
│   ├── cards_downloader.py          # Descargador de imágenes de cartas
│   └── yugioh_card_images/          # Dataset de imágenes (2000+ cartas)
├── dev/
│   ├── dataset.py                   # Dataset personalizado con aumentación
│   ├── train.py                     # Entrenamiento del modelo
│   └── test.py                      # Pruebas y evaluación
├── models/                          # Modelos entrenados
├── requirements.txt                 # Dependencias del proyecto
└── README.md                       # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd YuGiOh-Card-Recognition
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el Dataset
```bash
cd data
python cards_downloader.py
```

**Nota**: El script descargará automáticamente más de 2000 imágenes de cartas de Yu-Gi-Oh! en formato GOAT desde la API oficial de Yu-Gi-Oh!.

## 🎯 Cómo Ejecutar el Proyecto

### Paso 1: Preparación del Dataset
```bash
# Navegar al directorio de datos
cd data

# Ejecutar el descargador de cartas
python cards_downloader.py
```

**Verificación**: Deberías ver mensajes como:
```
Images will be saved to: /path/to/data/yugioh_card_images
Found 2000+ cards in total from API
Filtered 2000+ cards in GOAT format.
Total GOAT images to download: 2000+
Downloaded (1/2000): 10012614.jpg
...
Download completed!
```

### Paso 2: Entrenamiento del Modelo
```bash
# Navegar al directorio de desarrollo
cd dev

# Ejecutar el entrenamiento
python train.py
```

**Verificación**: Deberías ver:
```
Epoch [1/10], Loss: 0.XXXX
Epoch [2/10], Loss: 0.XXXX
...
Epoch [10/10], Loss: 0.XXXX
Training complete!
```

**Archivos generados**:
- `modelo_entrenado.pth`: Modelo entrenado

### Paso 3: Pruebas y Evaluación
```bash
# Ejecutar las pruebas
python test.py
```

**Verificación**: Deberías ver:
```
Tamaño del archivo: XX.XX MB
MD5: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
✅ Archivo cargado CORRECTAMENTE
Usando dispositivo: cuda/cpu
✅ Modelo cargado y listo para usar
🧪 Probando el modelo con imágenes del dataset:
✅ 62121.jpg procesada
...
📊 Similitudes entre imágenes:
62121.jpg vs 88472456.jpg: 0.XXXX
...
✅ Prueba completada exitosamente!
```

## 🔍 Verificación del Funcionamiento Correcto

### 1. Verificar Descarga de Datos
```bash
# Verificar que se descargaron las imágenes
ls data/yugioh_card_images/ | wc -l
# Debería mostrar más de 2000 archivos
```

### 2. Verificar Entrenamiento
- **Loss decreciente**: El loss debería disminuir durante el entrenamiento
- **Archivo de modelo**: Se debe generar `modelo_entrenado.pth`
- **Tamaño del modelo**: Debería ser ~100-200 MB

### 3. Verificar Inferencia
- **Similitud consigo misma**: Debería ser cercana a 1.0
- **Similitud entre cartas diferentes**: Debería ser menor a 0.5
- **Sin errores**: No deberían aparecer errores de carga o procesamiento

## 📊 Métricas de Rendimiento

### Durante el Entrenamiento
- **Loss inicial**: ~0.8-1.2
- **Loss final**: ~0.1-0.3
- **Convergencia**: Debería estabilizarse después de 5-7 épocas

### Durante las Pruebas
- **Similitud intra-clase**: >0.8 (misma carta)
- **Similitud inter-clase**: <0.5 (cartas diferentes)
- **Tiempo de inferencia**: <1 segundo por imagen

## 🛠️ Personalización

### Modificar Hiperparámetros
En `dev/train.py`:
```python
# Tamaño del embedding
embedding_dim = 64

# Margen del Triplet Loss
margin = 1.0

# Tasa de aprendizaje
lr = 0.001

# Tamaño del batch
batch_size = 32

# Número de épocas
num_epochs = 10
```

### Modificar Aumentación de Datos
En `dev/dataset.py`:
```python
# Probabilidad de aplicar aumentación
augment_prob = 0.5

# Tipos de aumentación disponibles:
# - Blur (desenfoque)
# - Brightness/Contrast
# - Color shifts
# - Sensor noise
```

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"
```bash
# Reducir batch_size en train.py
batch_size = 16  # o menor
```

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "Images not found"
```bash
# Verificar que se ejecutó cards_downloader.py
ls data/yugioh_card_images/
```

### Error: "Model file not found"
```bash
# Verificar que se completó el entrenamiento
ls dev/modelo_entrenado.pth
```

## 📈 Mejoras Futuras

1. **Data Augmentation**: Más técnicas de aumentación
2. **Modelo más grande**: ResNet-152 o EfficientNet
3. **Ensemble**: Combinar múltiples modelos
4. **API REST**: Servicio web para reconocimiento
5. **Mobile App**: Aplicación móvil para reconocimiento en tiempo real

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

Desarrollado como proyecto de Redes Neuronales para reconocimiento de cartas de Yu-Gi-Oh!.

---

**¡Disfruta reconociendo cartas de Yu-Gi-Oh! con IA! 🎴✨**