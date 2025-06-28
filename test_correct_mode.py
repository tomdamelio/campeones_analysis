#!/usr/bin/env python
"""
Ejemplo de uso del modo de corrección --correct-file

Este script muestra cómo usar la nueva funcionalidad para corregir
archivos de eventos ya procesados, sobrescribiendo el archivo original.
"""

import sys
from pathlib import Path

def main():
    print("🔧 MODO CORRECCIÓN DE ARCHIVOS - detect_markers.py")
    print("=" * 60)
    print()
    
    print("📋 DESCRIPCIÓN:")
    print("   El flag --correct-file permite editar manualmente archivos")
    print("   de eventos ya procesados, sobrescribiendo el archivo original")
    print("   con un backup automático.")
    print()
    
    print("🚀 EJEMPLO DE USO:")
    print("   python scripts/preprocessing/detect_markers.py \\")
    print("       --subject 14 \\")
    print("       --session vr \\")
    print("       --task 01 \\")
    print("       --run 006 \\")
    print("       --acq b \\")
    print("       --correct-file")
    print()
    
    print("⚙️  PARÁMETROS ADICIONALES:")
    print("   --correct-file-dir merged_events  # Directorio donde buscar")
    print("   --correct-file-desc merged        # Descripción del archivo")
    print("   --force-save                      # Guardar sin confirmar")
    print()
    
    print("🔄 FLUJO:")
    print("   1. Carga el archivo especificado")
    print("   2. Muestra estadísticas y eventos línea por línea")
    print("   3. Abre ventana interactiva de MNE para edición")
    print("   4. Detecta cambios y muestra resumen")
    print("   5. Crea backup (.tsv.backup y .json.backup)")
    print("   6. Sobrescribe archivo original con correcciones")
    print()
    
    print("📁 ARCHIVOS GENERADOS:")
    print("   Archivo original actualizado:")
    print("   data/derivatives/merged_events/sub-14/ses-vr/eeg/")
    print("   ├── sub-14_ses-vr_task-01_acq-b_run-006_desc-merged_events.tsv")
    print("   ├── sub-14_ses-vr_task-01_acq-b_run-006_desc-merged_events.json")
    print("   └── Backups:")
    print("       ├── sub-14_ses-vr_task-01_acq-b_run-006_desc-merged_events.tsv.backup")
    print("       └── sub-14_ses-vr_task-01_acq-b_run-006_desc-merged_events.json.backup")
    print()
    
    print("✨ VENTAJAS:")
    print("   • Sobrescribe el archivo original (no necesitas reemplazar manualmente)")
    print("   • Backup automático para seguridad")
    print("   • Historial de cambios en el JSON")
    print("   • Detección inteligente de modificaciones")
    print("   • Resumen detallado de cambios (Δ onsets y duraciones)")
    print()
    
    print("⚠️  IMPORTANTE:")
    print("   • Se crea backup automáticamente antes de sobrescribir")
    print("   • Los archivos .backup pueden eliminarse una vez confirmados los cambios")
    print("   • El historial de procesamiento se mantiene en el JSON")
    print()
    
    # Verificar si el archivo objetivo existe
    target_file = Path("data/derivatives/merged_events/sub-14/ses-vr/eeg/sub-14_ses-vr_task-01_acq-b_run-006_desc-merged_events.tsv")
    
    if target_file.exists():
        print(f"✅ Archivo objetivo encontrado: {target_file}")
        print("   ¡Listo para usar el modo corrección!")
    else:
        print(f"❌ Archivo objetivo no encontrado: {target_file}")
        print("   Asegúrate de que el archivo existe antes de usar --correct-file")
    
    print()
    print("🎯 ¿CUÁNDO USAR ESTE MODO?")
    print("   • Cuando veas diferencias significativas en duraciones")
    print("   • Para ajustar onsets manualmente")
    print("   • Para eliminar eventos falsos positivos")
    print("   • Para añadir eventos que se perdieron en la detección automática")

if __name__ == "__main__":
    main() 