#!/usr/bin/env python
"""
Script para verificar que las anotaciones se guardaron correctamente después
de usar visualize_events.py.

Este script:
1. Carga el archivo de anotaciones guardado
2. Muestra las 3 columnas de anotaciones (onset, duration, trial_type)
3. Visualiza los datos con las anotaciones de forma interactiva

Uso:
    python verify_annotations.py --subject 16 --session vr --task 02 --run 003 --acq a
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt

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
        description="Verifica las anotaciones guardadas por visualize_events.py"
    )
    parser.add_argument("--subject", type=str, required=True, 
                        help="ID del sujeto (e.g., '16')")
    parser.add_argument("--session", type=str, default="vr",
                        help="ID de la sesión (default: 'vr')")
    parser.add_argument("--task", type=str, required=True,
                        help="ID de la tarea (e.g., '02')")
    parser.add_argument("--run", type=str, required=True,
                        help="ID del run (e.g., '003')")
    parser.add_argument("--acq", type=str, default=None,
                        help="Parámetro de adquisición (e.g., 'a')")
    parser.add_argument("--show-all-channels", action="store_true",
                        help="Mostrar todos los canales en vez de solo AUDIO/PHOTO/joystick_x")
    parser.add_argument("--source_dir", type=str, default="merged_events",
                        help="Directorio dentro de derivatives donde buscar los eventos. Opciones: merged_events, edited_events, auto_events, aligned_events, events_manual, events (default: merged_events)")
    
    return parser.parse_args()


def load_data_and_annotations(subject, session, task, run, acq=None, source_dir="merged_events"):
    """
    Carga los datos raw y las anotaciones guardadas.
    
    Parameters
    ----------
    subject : str
        ID del sujeto
    session : str
        ID de la sesión
    task : str
        ID de la tarea
    run : str
        ID del run
    acq : str, optional
        Parámetro de adquisición
    source_dir : str, optional
        Nombre del directorio dentro de derivatives donde buscar los eventos
        
    Returns
    -------
    tuple
        (raw, annotations_df, annot_path) con los datos raw, anotaciones y la ruta de anotaciones
    """
    print(f"\n=== Verificando anotaciones para sub-{subject} ses-{session} task-{task} run-{run} ===\n")
    
    # Definir rutas
    bids_root = repo_root / 'data' / 'raw'
    
    # Primero intentamos cargar desde aligned_events (anotaciones alineadas)
    deriv_root = repo_root / 'data' / 'derivatives' / source_dir
    
    # Asegurar formato correcto de parámetros
    if task and task.isdigit():
        task = task.zfill(2)
    if run and run.isdigit():
        run = run.zfill(3)
    if acq:
        acq = acq.lower()
    
    # Crear BIDSPath para datos raw
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        root=bids_root,
        extension='.vhdr'
    )
    
    print(f"Ruta de datos raw: {bids_path.fpath}")
    
    # Crear BIDSPath para anotaciones - ajustar descripción según el directorio
    if source_dir == "merged_events":
        description = 'merged'  # Para merged_events, usar desc-merged
    elif source_dir == "aligned_events":
        description = 'withann'  # Para aligned_events, usar desc-withann
    elif source_dir == "edited_events":
        description = 'edited'  # Para edited_events, usar desc-edited
    elif source_dir == "auto_events":
        description = 'autoann'  # Para auto_events, usar desc-autoann
    else:
        description = None  # Para events originales, sin descripción
    
    annot_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description=description,
        extension='.tsv',
        root=deriv_root,
        check=False
    )
    
    # Si no existen anotaciones en el directorio especificado, buscar en otros directorios
    if not annot_path.fpath.exists():
        print(f"No se encontraron anotaciones en: {annot_path.fpath}")
        
        # Lista de directorios y descripciones a probar como fallback
        fallback_options = [
            ('edited_events', 'edited'),
            ('auto_events', 'autoann'),
            ('aligned_events', 'withann'),
            ('events_manual', 'manual'),
            ('events', None)  # events originales sin descripción
        ]
        
        found = False
        for fallback_dir, fallback_desc in fallback_options:
            if fallback_dir == source_dir:
                continue  # Saltar el directorio que ya probamos
                
            print(f"Buscando anotaciones en {fallback_dir}...")
            
            deriv_root = repo_root / 'data' / 'derivatives' / fallback_dir
            annot_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run=run,
                acquisition=acq,
                datatype='eeg',
                suffix='events',
                description=fallback_desc,
                extension='.tsv',
                root=deriv_root,
                check=False
            )
            
            if annot_path.fpath.exists():
                print(f"¡Encontradas anotaciones en {fallback_dir}!")
                found = True
                break
            else:
                print(f"No se encontraron en: {annot_path.fpath}")
        
        if not found:
            print("No se encontraron anotaciones en ningún directorio de derivados.")
    
    print(f"Ruta de anotaciones: {annot_path.fpath}")
    
    # Verificar que los archivos existen
    if not bids_path.fpath.exists():
        raise FileNotFoundError(f"Archivo raw no encontrado: {bids_path.fpath}")
    if not annot_path.fpath.exists():
        raise FileNotFoundError(f"Archivo de anotaciones no encontrado: {annot_path.fpath}")
    
    # Cargar datos raw
    raw = read_raw_bids(bids_path, verbose=False)
    print(f"Datos raw cargados: {raw}")
    
    # Cargar anotaciones
    annotations_df = pd.read_csv(annot_path.fpath, sep='\t')
    
    # Mostrar resumen de las anotaciones
    print(f"\nSe encontraron {len(annotations_df)} anotaciones:")
    
    # Convertir onset/duration a formato tiempo legible
    if len(annotations_df) > 0:
        print("\nResumen de anotaciones (mm:ss):")
        for i, row in annotations_df.iterrows():
            onset_min = int(row['onset'] // 60)
            onset_sec = int(row['onset'] % 60)
            
            duration_min = int(row['duration'] // 60)
            duration_sec = int(row['duration'] % 60)
            
            end_time = row['onset'] + row['duration']
            end_min = int(end_time // 60)
            end_sec = int(end_time % 60)
            
            print(f"  {i+1}. {row['trial_type']}: {onset_min:02d}:{onset_sec:02d} - {end_min:02d}:{end_sec:02d} (duración: {duration_min:02d}:{duration_sec:02d})")
    
    return raw, annotations_df, annot_path.fpath


def verify_annotations(raw, annotations_df, annotations_path, show_all_channels=False):
    """
    Verifica y visualiza las anotaciones.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    annotations_df : pandas.DataFrame
        DataFrame con las anotaciones
    annotations_path : str
        Ruta al archivo de anotaciones
    show_all_channels : bool, optional
        Si es True, muestra todos los canales en vez de solo AUDIO/PHOTO/joystick_x
    """
    print("\n=== Verificando anotaciones ===\n")
    
    # Mostrar información sobre las anotaciones
    print(f"Archivo de anotaciones: {annotations_path}")
    print(f"Número de anotaciones: {len(annotations_df)}")
    print("\nContenido de las anotaciones:")
    print(annotations_df)
    
    # Crear anotaciones MNE desde el DataFrame
    annot = mne.Annotations(
        onset=annotations_df['onset'].values,
        duration=annotations_df['duration'].values,
        description=annotations_df['trial_type'].values
    )
    
    # Añadir anotaciones al raw
    raw_plot = raw.copy()
    raw_plot.set_annotations(annot)
    
    # Si solo queremos ver canales específicos, seleccionar esos canales
    if not show_all_channels:
        # Verificar que los canales existen
        available_channels = raw_plot.ch_names
        required_channels = ['AUDIO', 'PHOTO', 'joystick_x']
        
        # Lista para almacenar los canales que realmente existen
        channels_to_pick = []
        
        for channel in required_channels:
            if channel in available_channels:
                channels_to_pick.append(channel)
            else:
                print(f"¡ADVERTENCIA! Canal {channel} no encontrado en los datos.")
                
                # Intentar encontrar canales similares
                similar_channels = [ch for ch in available_channels if channel.lower() in ch.lower()]
                if similar_channels:
                    print(f"Canales similares encontrados para {channel}: {similar_channels}")
                    channels_to_pick.extend(similar_channels[:1])  # Usar el primer canal similar
        
        if not channels_to_pick:
            print("No se encontraron los canales requeridos ni similares. Se usarán los primeros tres canales disponibles.")
            channels_to_pick = available_channels[:3]
        
        print(f"Mostrando los canales: {channels_to_pick}")
        raw_plot = raw_plot.pick_channels(channels_to_pick)
    else:
        print("Mostrando todos los canales disponibles")
    
    # Visualizar con todas las anotaciones
    print("\nVisualizando datos con anotaciones...")
    print("Presiona 'h' en el visualizador para ver atajos de teclado")
    print("Presiona 'p' para desplazarte por el registro página a página")
    print("Presiona 'j'/'k' para ajustar la escala vertical")
    print("Cierra la ventana para finalizar.")
    
    fig = raw_plot.plot(
        title="Verificación de anotaciones",
        scalings='auto',
        duration=180,  # 3 minutos por página
        start=0,
        show=True,
        block=True,
        decim=64  # Aplicar decimación para reducir la resolución
    )
    
    print("\nVisualizador cerrado.")
    print(f"Se encontraron {len(annot)} anotaciones.")


def main():
    """Función principal del script."""
    args = parse_args()
    
    try:
        # Cargar datos y anotaciones
        raw, annotations_df, annot_path = load_data_and_annotations(
            args.subject, args.session, args.task, args.run, args.acq,
            source_dir=args.source_dir
        )
        
        # Verificar y visualizar anotaciones
        verify_annotations(raw, annotations_df, annot_path, args.show_all_channels)
        
        print("\n=== Verificación completada exitosamente ===")
        print("\nPróximos pasos:")
        print("1. Si las anotaciones son correctas, puedes continuar con el análisis")
        print("2. Si encuentras errores, vuelve a usar visualize_events.py para corregir las anotaciones")
        print("3. Valida la estructura BIDS con 'bids-validator data/derivatives/aligned_events'")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 