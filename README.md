# 🎴 Yu-Gi-Oh! Card Recognition System

Sistema de reconocimiento de cartas de Yu-Gi-Oh! utilizando una red neuronal siamesa con fine-tuning de ResNet18. Este proyecto cumple con los requisitos de la asignatura de Redes Neuronales Profundas.

## 📁 Estructura del Proyecto

```
YuGiOh-Card-Recognition/
├── data/                          # Datasets y datos preprocesados
│   ├── yugioh_card_images/        # Imágenes de cartas del dataset
│   └── cards_downloader.py        # Script de descarga de datos
├── dev/                           # Desarrollo experimental
│   ├── model_dev.ipynb            # Notebook de experimentación
│   ├── dataset.py                 # Clase del dataset
│   ├── train.py                   # Script de entrenamiento
│   └── test.py                    # Script de evaluación
└── prod/                          # Aplicación de producción
    ├── app.py                     # Aplicación Streamlit principal
    ├── modelo.pth                 # Modelo entrenado
    ├── utils.py                   # Funciones auxiliares
    ├── requirements.txt           # Dependencias
    └── README.md                  # Documentación de producción
```

## 🚀 Características Principales

### Componente Inteligente
- **Red Neuronal Siamesa**: Arquitectura basada en ResNet18 con fine-tuning
- **Triplet Loss**: Función de pérdida optimizada para comparación de similitud
- **Embeddings de 128 dimensiones**: Representación compacta de las cartas

### Optimizaciones de Rendimiento
- **Tensor de embeddings precalculado**: Almacenamiento en 4ª dimensión para comparaciones vectorizadas
- **Cache inteligente**: Los embeddings se guardan automáticamente para evitar recálculos
- **Procesamiento por batches**: Optimización de memoria para datasets grandes

### Preprocesamiento Avanzado
- **Detección automática de cartas**: OpenCV para extraer cartas de fotos reales
- **Bounding box inteligente**: Identificación de contornos de cartas
- **Normalización robusta**: Adaptación a diferentes condiciones de iluminación

### Aplicación Web
- **Streamlit**: Interfaz moderna y responsiva
- **Controles configurables**: Umbral de similitud, número de resultados
- **Visualización mejorada**: Tablas, imágenes y métricas claras

## 🛠️ Instalación y Uso

### Requisitos Previos
- Python 3.8 o superior
- Git
- Acceso a internet para descargar dependencias

### Instalación Rápida

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/YuGiOh-Card-Recognition.git
   cd YuGiOh-Card-Recognition
   ```

2. **Instalar dependencias**:
   ```bash
   cd prod
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**:
   ```bash
   streamlit run app.py
   ```

La aplicación estará disponible en: `http://localhost:8501`

### Despliegue en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio de GitHub
4. Configura la ruta del archivo principal: `prod/app.py`
5. ¡Listo! Tu app estará disponible online

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

## 🔧 Desarrollo y Experimentación

### Notebook de Desarrollo
El archivo `dev/model_dev.ipynb` contiene:
- Exploración del dataset
- Desarrollo de la arquitectura de la red
- Entrenamiento y evaluación
- Visualización de resultados
- Pruebas con imágenes reales

### Scripts de Desarrollo
- `dev/train.py`: Entrenamiento completo del modelo
- `dev/test.py`: Evaluación y métricas de rendimiento
- `dev/dataset.py`: Clase personalizada del dataset

## 📈 Optimizaciones Implementadas

1. **Tensor de Embeddings**: Almacenamiento en 4ª dimensión para comparaciones vectorizadas
2. **Cache Inteligente**: Los embeddings se guardan y cargan automáticamente
3. **Procesamiento por Batches**: Optimización de memoria para datasets grandes
4. **Preprocesamiento Automático**: Extracción de cartas de fotos reales
5. **Interfaz Responsiva**: Streamlit con controles configurables

## 🐛 Solución de Problemas

### Error: "No se encontró el modelo"
- Verifica que `prod/modelo.pth` exista
- Asegúrate de que el archivo no esté corrupto

### Error: "No se encontró el dataset"
- Verifica que `data/yugioh_card_images/` exista
- Asegúrate de que contenga imágenes válidas

### Rendimiento Lento
- Los embeddings se generan automáticamente en la primera ejecución
- Verifica que tienes suficiente RAM disponible
- Considera usar GPU si está disponible

### Resultados Pobres
- Ajusta el umbral de similitud en la barra lateral
- Usa imágenes más claras y bien iluminadas
- Activa el preprocesamiento automático para fotos reales

## 📋 Cumplimiento de Consignas

✅ **Componente Inteligente**: Red neuronal siamesa con fine-tuning de ResNet18  
✅ **Dataset**: Dataset público de cartas de Yu-Gi-Oh!  
✅ **Entrenamiento PyTorch**: Modelo entrenado y documentado  
✅ **Aplicación Web**: Streamlit funcional y moderna  
✅ **Hosting**: Preparado para Streamlit Cloud  
✅ **Estructura del Repositorio**: Exactamente como solicitado  

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