#!/usr/bin/env python
"""
Script de ejemplo para probar la nueva funcionalidad de corrección automática 
por diferencias significativas en duraciones del script detect_markers.py

Este script muestra cómo usar el nuevo parámetro --max-duration-diff para 
controlar cuándo se abre automáticamente el visualizador para corrección manual.

Ejemplo de uso:
    # Usar umbral por defecto (1.0 segundo)
    python detect_markers.py --subject 14 --session vr --task 01 --run 006 --acq b
    
    # Usar umbral más estricto (0.5 segundos)
    python detect_markers.py --subject 14 --session vr --task 01 --run 006 --acq b --max-duration-diff 0.5
    
    # Usar umbral más relajado (5.0 segundos)
    python detect_markers.py --subject 14 --session vr --task 01 --run 006 --acq b --max-duration-diff 5.0
"""

import subprocess
import sys
from pathlib import Path

def run_detect_markers_example():
    """
    Ejecuta el script detect_markers.py con diferentes configuraciones de 
    --max-duration-diff para demostrar la nueva funcionalidad.
    """
    
    # Comando base
    base_cmd = [
        "python", "scripts/preprocessing/detect_markers.py",
        "--subject", "14",
        "--session", "vr", 
        "--task", "01",
        "--run", "006",
        "--acq", "b"
    ]
    
    print("="*60)
    print("DEMOSTRACIÓN: Nueva funcionalidad de corrección por duración")
    print("="*60)
    
    print("\nEsta funcionalidad detecta automáticamente cuando hay diferencias")
    print("significativas entre las duraciones de eventos originales y nuevas anotaciones.")
    print("\nCuando las diferencias superan el umbral especificado por --max-duration-diff,")
    print("se abre automáticamente el visualizador MNE para corrección manual.")
    
    print("\n" + "-"*60)
    print("CONFIGURACIONES DE EJEMPLO:")
    print("-"*60)
    
    # Diferentes configuraciones de umbral
    configs = [
        {
            "threshold": 0.5,
            "description": "Umbral estricto - detecta diferencias > 0.5s",
            "use_case": "Análisis que requiere precisión temporal alta"
        },
        {
            "threshold": 1.0,
            "description": "Umbral por defecto - detecta diferencias > 1.0s",
            "use_case": "Uso general recomendado"
        },
        {
            "threshold": 5.0,
            "description": "Umbral relajado - detecta diferencias > 5.0s",
            "use_case": "Análisis exploratorio o datos con menos precisión"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. --max-duration-diff {config['threshold']}")
        print(f"   {config['description']}")
        print(f"   Caso de uso: {config['use_case']}")
        
        # Mostrar comando completo
        cmd = base_cmd + ["--max-duration-diff", str(config['threshold'])]
        print(f"   Comando: {' '.join(cmd)}")
    
    print("\n" + "-"*60)
    print("FLUJO DE CORRECCIÓN AUTOMÁTICA:")
    print("-"*60)
    
    print("\n1. El script fusiona eventos originales con nuevas anotaciones")
    print("2. Compara las duraciones evento por evento")
    print("3. Si encuentra diferencias > --max-duration-diff:")
    print("   ✓ Muestra advertencias detalladas")
    print("   ✓ Abre automáticamente el visualizador MNE")
    print("   ✓ Permite corrección manual interactiva")
    print("   ✓ Re-intenta la fusión con los datos corregidos")
    print("4. Si las diferencias persisten, pregunta si continuar")
    
    print("\n" + "-"*60)
    print("EJEMPLO DE ADVERTENCIA DETECTADA:")
    print("-"*60)
    print("¡ADVERTENCIA! Diferencia significativa en la duración del evento 2:")
    print("  Original: 104.00s")
    print("  Nueva: 87.15s")
    print("🔄 Se requiere edición manual adicional debido a diferencias significativas en duraciones")
    print("Abriendo visualizador para corrección manual...")
    
    print("\n" + "-"*60)
    print("PARA PROBAR:")
    print("-"*60)
    print("Ejecuta uno de los comandos de arriba con tus datos.")
    print("Si tienes el caso específico mencionado (sub-14 task-01 run-006),")
    print("verás la funcionalidad en acción automáticamente.")

if __name__ == "__main__":
    run_detect_markers_example() 