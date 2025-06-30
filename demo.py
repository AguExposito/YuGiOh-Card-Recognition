"""
Yu-Gi-Oh! Card Recognition - Demo Script
========================================

Script de demostración rápida para probar el proyecto.
Este script ejecuta un entrenamiento rápido con pocas imágenes para verificar
que todo funciona correctamente.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header():
    """Imprime el encabezado del demo."""
    print("🎴 Yu-Gi-Oh! Card Recognition - Demo")
    print("=" * 50)
    print("Este script ejecutará una demostración rápida del proyecto")
    print("usando un subconjunto pequeño de imágenes para verificar")
    print("que todo funciona correctamente.")
    print("=" * 50)

def check_requirements():
    """Verifica que las dependencias estén instaladas."""
    print("\n🔍 Verificando dependencias...")
    
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 
        'matplotlib', 'requests', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing_packages)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def check_data():
    """Verifica que existan datos para el demo."""
    print("\n📁 Verificando datos...")
    
    data_dir = Path("data/yugioh_card_images")
    if not data_dir.exists():
        print("❌ Directorio de datos no encontrado")
        print("Ejecuta: cd data && python cards_downloader.py")
        return False
    
    image_files = list(data_dir.glob("*.jpg"))
    if len(image_files) < 10:
        print(f"⚠️  Pocas imágenes encontradas: {len(image_files)}")
        print("Ejecuta: cd data && python cards_downloader.py")
        return False
    
    print(f"✅ {len(image_files)} imágenes encontradas")
    return True

def run_quick_training():
    """Ejecuta un entrenamiento rápido."""
    print("\n🚀 Ejecutando entrenamiento rápido...")
    print("(Usando solo 100 imágenes para demo)")
    
    try:
        # Cambiar al directorio dev
        os.chdir("dev")
        
        # Ejecutar entrenamiento con pocas imágenes
        cmd = [
            sys.executable, "train.py",
            "--max_images", "100",
            "--epochs", "3",
            "--batch_size", "16",
            "--save_dir", "../models"
        ]
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Entrenamiento completado exitosamente")
            return True
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")
        return False
    finally:
        # Volver al directorio original
        os.chdir("..")

def run_quick_testing():
    """Ejecuta pruebas rápidas."""
    print("\n🧪 Ejecutando pruebas rápidas...")
    
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return False
    
    try:
        # Cambiar al directorio dev
        os.chdir("dev")
        
        # Ejecutar pruebas
        cmd = [
            sys.executable, "test.py",
            "--model_path", f"../{model_path}",
            "--test_images", "62121.jpg", "88472456.jpg",
            "--demo_image", "62121.jpg",
            "--top_k", "3"
        ]
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pruebas completadas exitosamente")
            return True
        else:
            print(f"❌ Error en pruebas: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando pruebas: {e}")
        return False
    finally:
        # Volver al directorio original
        os.chdir("..")

def show_results():
    """Muestra los resultados del demo."""
    print("\n📊 Resultados del Demo")
    print("=" * 50)
    
    # Verificar archivos generados
    model_files = list(Path("models").glob("*.pth"))
    if model_files:
        print(f"✅ Modelos generados: {len(model_files)}")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   - {model_file.name} ({size_mb:.1f} MB)")
    else:
        print("❌ No se generaron modelos")
    
    # Verificar logs
    if os.path.exists("dev/training.log"):
        print("✅ Log de entrenamiento generado")
    
    print("\n🎯 Demo completado!")
    print("\n📋 Próximos pasos:")
    print("1. Para entrenamiento completo: cd dev && python train.py")
    print("2. Para pruebas completas: cd dev && python test.py --model_path ../models/best_model.pth")
    print("3. Para personalizar: edita config.py")

def main():
    """Función principal del demo."""
    start_time = time.time()
    
    print_header()
    
    # Verificar dependencias
    if not check_requirements():
        return
    
    # Verificar datos
    if not check_data():
        return
    
    # Ejecutar entrenamiento rápido
    if not run_quick_training():
        return
    
    # Ejecutar pruebas rápidas
    if not run_quick_testing():
        return
    
    # Mostrar resultados
    show_results()
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Tiempo total del demo: {total_time:.1f} segundos")
    print("=" * 50)

if __name__ == "__main__":
    main() 