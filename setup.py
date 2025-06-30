"""
Setup script for Yu-Gi-Oh! Card Recognition Project
==================================================

Este script facilita la configuración inicial del proyecto.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_requirements():
    """Instala las dependencias del proyecto."""
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def create_directories():
    """Crea los directorios necesarios."""
    print("\n📁 Creando directorios...")
    directories = [
        "models",
        "logs",
        "data/yugioh_card_images",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def check_data_directory():
    """Verifica si existe el directorio de datos."""
    data_dir = Path("data/yugioh_card_images")
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"✅ Directorio de datos encontrado con {len(list(data_dir.glob('*.jpg')))} imágenes")
        return True
    else:
        print("⚠️  Directorio de datos no encontrado o vacío")
        print("   Ejecuta: cd data && python cards_downloader.py")
        return False

def create_config_file():
    """Crea un archivo de configuración."""
    config_content = """# Yu-Gi-Oh! Card Recognition Configuration
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
"""
    
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("✅ Archivo de configuración creado: config.py")

def create_run_scripts():
    """Crea scripts para facilitar la ejecución."""
    
    # Script para Windows
    windows_script = """@echo off
echo Yu-Gi-Oh! Card Recognition - Setup
echo =================================

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python no encontrado
    pause
    exit /b 1
)

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt

REM Crear directorios
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "results" mkdir results

echo.
echo Setup completado!
echo.
echo Para descargar datos: cd data && python cards_downloader.py
echo Para entrenar: cd dev && python train.py
echo Para probar: cd dev && python test.py --model_path ../models/best_model.pth
echo.
pause
"""
    
    with open("setup.bat", "w", encoding="utf-8") as f:
        f.write(windows_script)
    
    # Script para Linux/Mac
    linux_script = """#!/bin/bash
echo "Yu-Gi-Oh! Card Recognition - Setup"
echo "================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 no encontrado"
    exit 1
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip3 install -r requirements.txt

# Crear directorios
mkdir -p models logs results

echo ""
echo "Setup completado!"
echo ""
echo "Para descargar datos: cd data && python3 cards_downloader.py"
echo "Para entrenar: cd dev && python3 train.py"
echo "Para probar: cd dev && python3 test.py --model_path ../models/best_model.pth"
echo ""
"""
    
    with open("setup.sh", "w", encoding="utf-8") as f:
        f.write(linux_script)
    
    # Hacer ejecutable el script de Linux
    if os.name != 'nt':  # No Windows
        os.chmod("setup.sh", 0o755)
    
    print("✅ Scripts de setup creados: setup.bat (Windows), setup.sh (Linux/Mac)")

def main():
    """Función principal del setup."""
    print("🎴 Yu-Gi-Oh! Card Recognition - Setup")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return
    
    # Instalar dependencias
    if not install_requirements():
        return
    
    # Crear directorios
    create_directories()
    
    # Crear archivo de configuración
    create_config_file()
    
    # Crear scripts de ejecución
    create_run_scripts()
    
    # Verificar directorio de datos
    check_data_directory()
    
    print("\n" + "=" * 50)
    print("✅ Setup completado exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Descargar datos: cd data && python cards_downloader.py")
    print("2. Entrenar modelo: cd dev && python train.py")
    print("3. Probar modelo: cd dev && python test.py --model_path ../models/best_model.pth")
    print("\n📚 Para más información, consulta el README.md")
    print("=" * 50)

if __name__ == "__main__":
    main() 