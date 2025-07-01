# 🎴 Yu-Gi-Oh! Card Recognition System

Sistema de reconocimiento de cartas de Yu-Gi-Oh! utilizando una red neuronal siamesa con fine-tuning de ResNet18. La aplicación permite identificar cartas a partir de imágenes reales mediante comparación de embeddings optimizada.

## 🚀 Características

- **Red Neuronal Siamesa**: Arquitectura basada en ResNet18 con fine-tuning
- **Optimización de Rendimiento**: Tensor de embeddings precalculado (4ª dimensión)
- **Preprocesamiento Automático**: Extracción de cartas de fotos reales con OpenCV
- **Interfaz Web Intuitiva**: Aplicación Streamlit con controles configurables
- **Cache de Embeddings**: Sistema de cache para evitar recálculos

## 📋 Requisitos Previos

- Python 3.8 o superior
- Git
- Acceso a internet para descargar dependencias

## 🛠️ Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/YuGiOh-Card-Recognition.git
cd YuGiOh-Card-Recognition/prod
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Preparar el Modelo y Dataset

Asegúrate de que los siguientes archivos estén en su lugar:

```
prod/
├── modelo.pth                    # Modelo entrenado (pesos de la red neuronal)
├── app.py                       # Aplicación principal
├── utils.py                     # Funciones auxiliares
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo

../data/
└── yugioh_card_images/          # Carpeta con imágenes del dataset
    ├── carta1.jpg
    ├── carta2.png
    └── ...
```

### 4. Ejecutar la Aplicación

#### Ejecución Local

```bash
streamlit run app.py
```

La aplicación estará disponible en: `http://localhost:8501`

#### Despliegue en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio de GitHub
4. Configura la ruta del archivo principal: `prod/app.py`
5. ¡Listo! Tu app estará disponible online

## 🎯 Uso de la Aplicación

### Interfaz Principal

1. **Subir Imagen**: Arrastra o selecciona una imagen de carta de Yu-Gi-Oh!
2. **Configurar Parámetros** (barra lateral):
   - **Umbral de Similitud**: Controla la precisión de los resultados (0.0 - 1.0)
   - **Número de Resultados**: Cuántas cartas similares mostrar (1 - 20)
   - **Preprocesamiento Automático**: Extrae automáticamente la carta de la imagen

3. **Buscar Cartas Similares**: Haz clic en el botón para obtener resultados

### Resultados

- **Tabla de Resultados**: Muestra las cartas más similares con sus puntuaciones
- **Visualización**: Imágenes de las cartas encontradas
- **Información Detallada**: Nombre de la carta y porcentaje de similitud

## 🔧 Configuración Avanzada

### Parámetros del Modelo

Los parámetros del modelo se pueden ajustar en `utils.py`:

```python
# Dimensiones del embedding
embedding_dim = 128

# Tamaño de imagen de entrada
image_size = (224, 224)

# Batch size para procesamiento
batch_size = 32
```

### Cache de Embeddings

Los embeddings se guardan automáticamente en `data/embeddings_cache.pkl` para optimizar el rendimiento. Si modificas el dataset, elimina este archivo para regenerar los embeddings.

## 📊 Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Imagen Input  │───▶│  Preprocesamiento│───▶│  Red Neuronal   │
│   (Real/Clara)  │    │   (OpenCV)       │    │   (Siamese)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Resultados    │◀───│  Comparación     │◀───│   Embeddings    │
│   (Streamlit)   │    │  Vectorizada     │    │   (Cache)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🐛 Solución de Problemas

### Error: "No se encontró el modelo"
- Verifica que `modelo.pth` esté en la carpeta `prod/`
- Asegúrate de que el archivo no esté corrupto

### Error: "No se encontró el dataset"
- Verifica que la carpeta `data/yugioh_card_images/` exista
- Asegúrate de que contenga imágenes válidas (.jpg, .png, .jpeg)

### Rendimiento Lento
- Los embeddings se generan automáticamente en la primera ejecución
- Verifica que tienes suficiente RAM disponible
- Considera usar GPU si está disponible

### Resultados Pobres
- Ajusta el umbral de similitud en la barra lateral
- Usa imágenes más claras y bien iluminadas
- Activa el preprocesamiento automático para fotos reales

## 📈 Optimizaciones Implementadas

1. **Tensor de Embeddings**: Almacenamiento en 4ª dimensión para comparaciones vectorizadas
2. **Cache Inteligente**: Los embeddings se guardan y cargan automáticamente
3. **Procesamiento por Batches**: Optimización de memoria para datasets grandes
4. **Preprocesamiento Automático**: Extracción de cartas de fotos reales
5. **Interfaz Responsiva**: Streamlit con controles configurables

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- [Tu Nombre] - Desarrollo inicial
- [Otros Contribuidores]

## 🙏 Agradecimientos

- Dataset de cartas de Yu-Gi-Oh! de la comunidad
- PyTorch y Streamlit por las herramientas de desarrollo
- OpenCV por las funciones de procesamiento de imágenes 