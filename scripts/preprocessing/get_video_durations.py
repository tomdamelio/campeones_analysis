#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para obtener la duración precisa de los videos en ./stimuli
y guardarla en un archivo CSV.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import cv2

# Buscar la raíz del repositorio
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

# Asegurarse de que podemos importar módulos del proyecto
sys.path.insert(0, str(repo_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Obtiene la duración precisa de los videos en stimuli"
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta personalizada para guardar el CSV (por defecto: stimuli/video_durations.csv)")
    
    return parser.parse_args()


def get_video_duration_cv2(video_path):
    """
    Obtiene la duración precisa de un video utilizando OpenCV.
    
    Args:
        video_path (str): Ruta al archivo de video.
        
    Returns:
        float: Duración del video en segundos con alta precisión.
    """
    try:
        # Abrir el video
        cap = cv2.VideoCapture(str(video_path))
        
        # Verificar que se haya abierto correctamente
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return None
        
        # Obtener el número total de frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Obtener la tasa de frames por segundo (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calcular la duración
        duration = frame_count / fps
        
        # Liberar el recurso
        cap.release()
        
        # Redondear a 3 decimales
        return round(duration, 3)
    
    except Exception as e:
        print(f"Error al procesar {video_path}: {e}")
        return None


def main():
    """Función principal."""
    args = parse_args()
    
    # Definir rutas
    stimuli_dir = repo_root / "stimuli"
    
    if not stimuli_dir.exists():
        print(f"Error: El directorio {stimuli_dir} no existe.")
        sys.exit(1)
    
    # Obtener lista de archivos de video
    video_files = [f for f in stimuli_dir.glob("*.mp4")]
    
    if not video_files:
        print(f"No se encontraron archivos .mp4 en {stimuli_dir}")
        sys.exit(1)
    
    # Obtener duración de cada video
    print(f"Procesando {len(video_files)} archivos de video...")
    
    filenames = []
    durations = []
    
    for video_file in video_files:
        print(f"Procesando {video_file.name}...")
        duration = get_video_duration_cv2(str(video_file))
        
        if duration is not None:
            filenames.append(video_file.name)
            durations.append(duration)
    
    # Crear DataFrame y guardar como CSV
    df = pd.DataFrame({
        'filename': filenames,
        'duration': durations
    })
    
    # Ordenar por nombre de archivo
    df = df.sort_values('filename').reset_index(drop=True)
    
    # Guardar CSV
    output_path = args.output if args.output else stimuli_dir / "video_durations.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Información de duración guardada en {output_path}")
    print(f"Se procesaron {len(filenames)} videos de {len(video_files)} totales.")


if __name__ == "__main__":
    main() 