"""
Yu-Gi-Oh! Card Recognition - Verification Script
===============================================

Script para verificar que el proyecto esté configurado correctamente.
"""

import os
import sys
import importlib
from pathlib import Path

def print_header():
    """Imprime el encabezado de verificación."""
    print("🔍 Yu-Gi-Oh! Card Recognition - Verificación")
    print("=" * 50)

def check_python_version():
    """Verifica la versión de Python."""
    print("\n🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere 3.8+")
        return False

def check_dependencies():
    """Verifica las dependencias principales."""
    print("\n📦 Verificando dependencias...")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'requests': 'Requests',
        'tqdm': 'TQDM',
        'sklearn': 'Scikit-learn'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            importlib.import_module(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Verifica la estructura del proyecto."""
    print("\n📁 Verificando estructura del proyecto...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "dev/dataset.py",
        "dev/train.py",
        "dev/test.py",
        "data/cards_downloader.py"
    ]
    
    required_dirs = [
        "data",
        "dev",
        "models"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ {file_path}")
        else:
            print(f"✅ {file_path}")
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path}/")
        else:
            print(f"✅ {dir_path}/")
    
    if missing_files or missing_dirs:
        print(f"\n⚠️  Faltan archivos/directorios: {len(missing_files) + len(missing_dirs)}")
        return False
    
    return True

def check_data_directory():
    """Verifica el directorio de datos."""
    print("\n📊 Verificando directorio de datos...")
    
    data_dir = Path("data/yugioh_card_images")
    if not data_dir.exists():
        print("❌ Directorio de datos no encontrado")
        print("   Ejecuta: cd data && python cards_downloader.py")
        return False
    
    image_files = list(data_dir.glob("*.jpg"))
    if len(image_files) == 0:
        print("⚠️  Directorio de datos vacío")
        print("   Ejecuta: cd data && python cards_downloader.py")
        return False
    
    print(f"✅ {len(image_files)} imágenes encontradas")
    return True

def check_models_directory():
    """Verifica el directorio de modelos."""
    print("\n🤖 Verificando directorio de modelos...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Directorio de modelos no encontrado")
        print("   Se creará automáticamente durante el entrenamiento")
        return True
    
    model_files = list(models_dir.glob("*.pth"))
    if len(model_files) == 0:
        print("ℹ️  No hay modelos entrenados")
        print("   Ejecuta: cd dev && python train.py")
        return True
    
    print(f"✅ {len(model_files)} modelos encontrados")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   - {model_file.name} ({size_mb:.1f} MB)")
    
    return True

def test_imports():
    """Prueba las importaciones de los módulos del proyecto."""
    print("\n🔧 Probando importaciones...")
    
    try:
        # Cambiar al directorio dev para importar módulos
        original_dir = os.getcwd()
        os.chdir("dev")
        
        # Probar importaciones
        import dataset
        print("✅ dataset.py")
        
        import train
        print("✅ train.py")
        
        import test
        print("✅ test.py")
        
        # Volver al directorio original
        os.chdir(original_dir)
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        os.chdir(original_dir)
        return False

def check_gpu_support():
    """Verifica el soporte de GPU."""
    print("\n🚀 Verificando soporte de GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU disponible: {gpu_name}")
            print(f"   Dispositivos CUDA: {gpu_count}")
        else:
            print("ℹ️  GPU no disponible - Usando CPU")
        return True
    except ImportError:
        print("❌ PyTorch no instalado")
        return False

def generate_summary():
    """Genera un resumen de la verificación."""
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    checks = [
        ("Python 3.8+", check_python_version()),
        ("Dependencias", check_dependencies()),
        ("Estructura del proyecto", check_project_structure()),
        ("Directorio de datos", check_data_directory()),
        ("Directorio de modelos", check_models_directory()),
        ("Importaciones", test_imports()),
        ("Soporte GPU", check_gpu_support())
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n✅ Verificaciones pasadas: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ¡Proyecto listo para usar!")
        print("\n📋 Próximos pasos:")
        print("1. Para demo rápido: python demo.py")
        print("2. Para entrenamiento: cd dev && python train.py")
        print("3. Para pruebas: cd dev && python test.py --model_path ../models/best_model.pth")
    else:
        print(f"\n⚠️  {total - passed} verificaciones fallaron")
        print("Revisa los errores arriba y corrígelos antes de continuar")
    
    print("=" * 50)

def main():
    """Función principal de verificación."""
    print_header()
    generate_summary()

if __name__ == "__main__":
    main() 