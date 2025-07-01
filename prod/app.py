import streamlit as st
import os
import tempfile
from PIL import Image
import sys
import logging

# Agregar el directorio actual al path para importar utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import CardEmbeddingManager, preprocess_real_photo, load_image_from_path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Yu-Gi-Oh! Card Recognition",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🎴 Yu-Gi-Oh! Card Recognition System")
st.markdown("""
Este sistema utiliza una red neuronal siamesa para identificar cartas de Yu-Gi-Oh! 
a partir de imágenes. Puedes subir una foto de una carta y el sistema te mostrará 
las cartas más similares del dataset.
""")

# Configuración en sidebar
st.sidebar.header("⚙️ Configuración")

# Parámetros configurables
similarity_threshold = st.sidebar.slider(
    "Umbral de similitud", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.3, 
    step=0.05,
    help="Solo se mostrarán cartas con similitud mayor a este valor"
)

top_k = st.sidebar.slider(
    "Número de resultados", 
    min_value=1, 
    max_value=20, 
    value=10,
    help="Número máximo de cartas similares a mostrar"
)

enable_preprocessing = st.sidebar.checkbox(
    "Preprocesamiento automático", 
    value=True,
    help="Extraer automáticamente la carta de la imagen usando OpenCV"
)

# Información del sistema
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Información del Sistema")
st.sidebar.markdown("""
- **Modelo**: Red Neuronal Siamesa
- **Arquitectura**: ResNet18 (Fine-tuned)
- **Embedding**: 128 dimensiones
- **Optimización**: Tensor de embeddings precalculado
""")

@st.cache_resource
def load_card_manager():
    """Cargar el gestor de cartas con cache para evitar recargas"""
    try:
        model_path = "modelo.pth"
        dataset_path = "../data/yugioh_card_images"
        
        # Verificar que los archivos existan
        if not os.path.exists(model_path):
            st.error(f"❌ No se encontró el modelo en: {model_path}")
            st.info("💡 Asegúrate de que el archivo 'modelo.pth' esté en la carpeta 'prod/'")
            return None
            
        if not os.path.exists(dataset_path):
            st.error(f"❌ No se encontró el dataset en: {dataset_path}")
            st.info("💡 Asegúrate de que las imágenes estén en: data/yugioh_card_images/")
            return None
        
        with st.spinner("🔄 Cargando modelo y embeddings..."):
            manager = CardEmbeddingManager(model_path, dataset_path)
            st.success("✅ Modelo cargado exitosamente!")
            return manager
            
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        logger.error(f"Error en load_card_manager: {e}")
        return None

def main():
    """Función principal de la aplicación"""
    
    # Cargar el gestor de cartas
    card_manager = load_card_manager()
    
    if card_manager is None:
        st.stop()
    
    # Mostrar información del dataset
    st.info(f"📚 Dataset cargado: {len(card_manager.card_names)} cartas disponibles")
    
    # Área de carga de archivos
    st.header("📤 Subir Imagen")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de carta de Yu-Gi-Oh!",
        type=['png', 'jpg', 'jpeg'],
        help="Puedes subir una foto de una carta real o una imagen limpia del dataset"
    )
    
    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.subheader("🖼️ Imagen Subida")
        
        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Cargar la imagen
            original_image = load_image_from_path(temp_path)
            
            if original_image is None:
                st.error("❌ No se pudo cargar la imagen")
                return
            
            # Mostrar imagen original
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(original_image, caption="Imagen Original", use_container_width=True)
            
            # Preprocesar si está habilitado
            processed_image = None
            if enable_preprocessing:
                with st.spinner("🔍 Procesando imagen..."):
                    processed_image = preprocess_real_photo(temp_path)
                
                if processed_image is not None:
                    with col2:
                        st.image(processed_image, caption="Imagen Procesada", use_container_width=True)
                else:
                    st.warning("⚠️ No se pudo procesar la imagen automáticamente")
                    processed_image = original_image
            else:
                processed_image = original_image
            
            # Botón para buscar cartas similares
            if st.button("🔍 Buscar Cartas Similares", type="primary"):
                with st.spinner("🔍 Buscando cartas similares..."):
                    try:
                        # Buscar cartas similares
                        similar_cards = card_manager.find_similar_cards(
                            processed_image, 
                            top_k=top_k, 
                            similarity_threshold=similarity_threshold
                        )
                        
                        if similar_cards:
                            st.subheader(f"🎯 Resultados ({len(similar_cards)} cartas encontradas)")
                            
                            # Mostrar resultados en una tabla
                            results_data = []
                            for i, (card_name, similarity) in enumerate(similar_cards, 1):
                                results_data.append({
                                    "Posición": i,
                                    "Carta": card_name,
                                    "Similitud": f"{similarity:.3f}",
                                    "Porcentaje": f"{similarity * 100:.1f}%"
                                })
                            
                            st.dataframe(
                                results_data,
                                column_config={
                                    "Posición": st.column_config.NumberColumn("Pos", width="small"),
                                    "Carta": st.column_config.TextColumn("Nombre de la Carta", width="medium"),
                                    "Similitud": st.column_config.TextColumn("Similitud", width="small"),
                                    "Porcentaje": st.column_config.TextColumn("Porcentaje", width="small")
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Mostrar las imágenes de las cartas más similares
                            st.subheader("🖼️ Cartas Más Similares")
                            
                            # Crear columnas para mostrar las imágenes
                            cols = st.columns(min(5, len(similar_cards)))
                            
                            for i, (card_name, similarity) in enumerate(similar_cards):
                                col_idx = i % 5
                                with cols[col_idx]:
                                    # Cargar y mostrar la imagen de la carta
                                    card_path = os.path.join(card_manager.dataset_path, card_name)
                                    if os.path.exists(card_path):
                                        card_image = Image.open(card_path)
                                        st.image(
                                            card_image, 
                                            caption=f"{card_name}\nSimilitud: {similarity:.3f}",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error(f"Imagen no encontrada: {card_name}")
                                    
                                    # Mostrar solo las primeras 5 cartas en la primera fila
                                    if i == 4 and len(similar_cards) > 5:
                                        st.info(f"... y {len(similar_cards) - 5} cartas más")
                                        break
                        
                        else:
                            st.warning("⚠️ No se encontraron cartas similares con el umbral especificado")
                            st.info("💡 Intenta bajar el umbral de similitud o subir una imagen más clara")
                    
                    except Exception as e:
                        st.error(f"❌ Error durante la búsqueda: {str(e)}")
                        logger.error(f"Error en búsqueda: {e}")
            
            # Limpiar archivo temporal
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error procesando la imagen: {str(e)}")
            logger.error(f"Error procesando imagen: {e}")
            # Limpiar archivo temporal en caso de error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 📖 Instrucciones de Uso")
    st.markdown("""
    1. **Sube una imagen** de una carta de Yu-Gi-Oh! (formato PNG, JPG, JPEG)
    2. **Ajusta los parámetros** en la barra lateral si es necesario:
       - Umbral de similitud: Controla qué tan similares deben ser las cartas
       - Número de resultados: Cuántas cartas similares mostrar
       - Preprocesamiento: Extrae automáticamente la carta de la imagen
    3. **Haz clic en "Buscar Cartas Similares"** para obtener resultados
    4. **Revisa los resultados** en la tabla y las imágenes mostradas
    """)
    
    st.markdown("### 💡 Consejos")
    st.markdown("""
    - **Para mejores resultados**: Usa imágenes claras y bien iluminadas
    - **Fotos reales**: El preprocesamiento automático ayuda a extraer la carta
    - **Umbral bajo**: Si no encuentras resultados, baja el umbral de similitud
    - **Umbral alto**: Para resultados más precisos, sube el umbral
    """)

if __name__ == "__main__":
    main() 