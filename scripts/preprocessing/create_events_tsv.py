#!/usr/bin/env python
"""
Este script construye el events.tsv inicial a partir de las planillas de órdenes.
Sigue las fases descritas en el documento W3.txt para crear anotaciones BIDS-compliant.

1. Localiza planillas de órdenes
2. Filtra filas relevantes
3. Asigna duraciones basadas en el catálogo de estímulos
4. Inicializa anotaciones en formato estandarizado (tipo/id)
5. Exporta a BIDS con metadatos completos

El archivo events.tsv generado incluye las siguientes columnas:
- onset: Tiempo de inicio del evento en segundos
- duration: Duración del evento en segundos
- trial_type: Tipo de evento (video, video_luminance, fixation, calm, practice)
- stim_id: Identificador único del estímulo (001-014, 101-112, 500, 901-902, 991-994)
- condition: Condición experimental (affective, luminance, baseline, calm, practice)
- stim_file: Ruta relativa al archivo de estímulo

Tipos de eventos soportados:
- video: Videos afectivos (originales) - IDs 001-014
- video_luminance: Videos con luminancia aumentada - IDs 101, 103, 107, 109, 112
- fixation: Cruz de fijación (baseline) - ID 500
- calm: Clips audiovisuales calmantes - IDs 901-902
- practice: Estímulos de práctica - IDs 991-994
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
import datetime
import re
import json
import argparse

# Buscar la raíz del repositorio si este script es llamado desde cualquier lugar
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

# Asegurarse de que podemos importar módulos del proyecto
sys.path.insert(0, str(repo_root))
from src.campeones_analysis.utils.bids_compliance import make_bids_basename


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crea archivos events.tsv en formato BIDS a partir de planillas de órdenes"
    )
    
    parser.add_argument("--subjects", type=str, nargs='+',
                       help="Lista de IDs de sujetos a procesar (ej: 16 17 18)")
    parser.add_argument("--session", type=str, default="vr",
                       help="ID de la sesión (default: 'vr')")
    parser.add_argument("--task", type=str,
                       help="ID de la tarea específica a procesar (ej: '01')")
    parser.add_argument("--acq", type=str,
                       help="Parámetro de adquisición (ej: 'a')")
    parser.add_argument("--run", type=str,
                       help="ID del run específico a procesar (ej: '003')")
    parser.add_argument("--all-runs", action="store_true",
                       help="Procesar todas las runs disponibles para cada sujeto")
    
    return parser.parse_args()


def time_str_to_seconds(time_str):
    """
    Convierte una cadena de tiempo en formato 'H:MM:SS' a segundos.
    
    Parameters
    ----------
    time_str : str
        Cadena de tiempo en formato 'H:MM:SS' o 'MM:SS'
    
    Returns
    -------
    float
        Tiempo en segundos
    """
    if isinstance(time_str, datetime.time):
        return time_str.hour * 3600 + time_str.minute * 60 + time_str.second
    
    if ':' not in time_str:
        return float(time_str)
    
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:  # H:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Formato de tiempo no reconocido: {time_str}")


def explore_excel_file(file_path):
    """
    Explora el contenido de un archivo Excel para entender su estructura.
    
    Parameters
    ----------
    file_path : str or Path
        Ruta al archivo Excel
    """
    try:
        df = pd.read_excel(file_path)
        print(f"\nEstructura del archivo: {file_path.name}")
        print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"Columnas: {df.columns.tolist()}")
        
        # Mostrar las primeras filas
        print("\nPrimeras 5 filas:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error al explorar el archivo: {e}")
        return None


def load_order_matrix(file_path):
    """
    Carga la planilla de órdenes desde un archivo Excel.
    
    Parameters
    ----------
    file_path : str or Path
        Ruta al archivo Excel con la planilla de órdenes
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con las filas de la planilla
    """
    print(f"Cargando planilla desde: {file_path}")
    try:
        df = pd.read_excel(file_path)
        print(f"Columnas encontradas: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None


def filter_relevant_rows(df):
    """
    Filtra las filas relevantes de la planilla según la columna 'path'.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la planilla completa
    
    Returns
    -------
    pandas.DataFrame
        DataFrame filtrado con solo las filas relevantes
    """
    # Verificar qué columna contiene las rutas de archivos
    path_column = None
    possible_columns = ['path', 'file', 'filepath', 'file_path', 'Path', 'File', 'FilePath', 'File_Path', 'filename']
    
    for col in possible_columns:
        if col in df.columns:
            path_column = col
            break
    
    if path_column is None:
        print(f"No se encontró una columna de path. Columnas disponibles: {df.columns.tolist()}")
        # Intentar inferir cual podría ser la columna de path
        for col in df.columns:
            if 'path' in col.lower() or 'file' in col.lower() or 'mp4' in str(df[col].iloc[0]).lower():
                path_column = col
                print(f"Se utilizará la columna '{col}' como columna de path.")
                break
        
        if path_column is None:
            return None
    
    print(f"Usando columna '{path_column}' para extraer información de los archivos")
    
    # Función para extraer el nombre del archivo sin ruta ni extensión
    def extract_filename(path_str):
        if pd.isna(path_str) or not isinstance(path_str, str):
            return ""
        
        # Obtener el nombre del archivo sin la ruta
        filename = path_str.split('/')[-1].split('\\')[-1]
        
        # Quitar la extensión (considerar cualquier extensión, no solo .mp4)
        filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Limpiar el nombre para asegurarnos de que no tiene caracteres especiales
        filename = filename.strip()
        
        print(f"Nombre extraído: '{filename}' del path '{path_str}'")
        
        return filename
    
    # Aplicar la función a la columna path para obtener solo los nombres de archivo
    df['extracted_filename'] = df[path_column].apply(extract_filename)
    
    # Mapear nombres de archivos a tipos de eventos
    filename_to_event_type = {
        # Videos (1-14)
        **{str(i): "video" for i in range(1, 15)},
        
        # Videos de luminancia
        'green_intensity_video_1': 'video_luminance',
        'green_intensity_video_3': 'video_luminance',
        'green_intensity_video_7': 'video_luminance',
        'green_intensity_video_9': 'video_luminance',
        'green_intensity_video_12': 'video_luminance',
        
        # Fijación
        'fixation_cross': 'fixation',
        
        # Calma
        '901': 'calm',
        '902': 'calm',
        
        # Práctica
        '991': 'practice',
        '992': 'practice',
        '993': 'practice',
        '994': 'practice'
    }
    
    # Crear columna de tipo de evento basada en el nombre extraído
    df['event_type'] = df['extracted_filename'].map(lambda x: filename_to_event_type.get(x, None))
    
    # Filtrar filas que tienen un tipo de evento asignado
    filtered_df = df[df['event_type'].notna()].copy()
    
    print(f"Filas filtradas: {len(filtered_df)} de {len(df)}")
    
    # Si no hay filas relevantes, mostrar los nombres extraídos para depuración
    if len(filtered_df) == 0:
        print("No se encontraron archivos con los nombres esperados. Mostrando nombres extraídos:")
        unique_filenames = df['extracted_filename'].unique()
        print(unique_filenames)
        
        # Intentar hacer una búsqueda más flexible
        print("Intentando búsqueda más flexible...")
        
        # Verificar si hay coincidencias parciales
        for filename in unique_filenames:
            if not pd.isna(filename) and filename:
                # Comprobar coincidencias con videos numerados
                if filename.isdigit() and 1 <= int(filename) <= 14:
                    print(f"  Posible video: {filename}")
                    df.loc[df['extracted_filename'] == filename, 'event_type'] = 'video'
                
                # Comprobar coincidencias con videos de luminancia
                elif 'green' in filename.lower() or 'intensity' in filename.lower():
                    print(f"  Posible video de luminancia: {filename}")
                    df.loc[df['extracted_filename'] == filename, 'event_type'] = 'video_luminance'
                
                # Comprobar coincidencias con fijación
                elif 'fix' in filename.lower() or 'cross' in filename.lower():
                    print(f"  Posible fijación: {filename}")
                    df.loc[df['extracted_filename'] == filename, 'event_type'] = 'fixation'
                
                # Comprobar coincidencias con calm
                elif filename in ['901', '902'] or filename.startswith('9') and len(filename) == 3:
                    print(f"  Posible calm: {filename}")
                    df.loc[df['extracted_filename'] == filename, 'event_type'] = 'calm'
                
                # Comprobar coincidencias con practice
                elif filename in ['991', '992', '993', '994'] or filename.startswith('99'):
                    print(f"  Posible practice: {filename}")
                    df.loc[df['extracted_filename'] == filename, 'event_type'] = 'practice'
        
        # Volver a filtrar después de la búsqueda flexible
        filtered_df = df[df['event_type'].notna()].copy()
        print(f"Filas filtradas después de búsqueda flexible: {len(filtered_df)} de {len(df)}")
    
    # Añadir mapeo para stim_id basado en el nombre del archivo
    def map_to_stim_id(filename, event_type):
        if pd.isna(filename) or pd.isna(event_type):
            return None
        
        if event_type == 'video':
            # Para videos, convertir el número a formato 001-014
            try:
                num = int(filename)
                if 1 <= num <= 14:
                    return f"{num:03d}"
            except ValueError:
                pass
        elif event_type == 'video_luminance':
            # Para videos de luminancia, extraer el número y convertirlo a formato 101, 103, etc.
            try:
                num = int(filename.split('_')[-1])
                if num in [1, 3, 7, 9, 12]:
                    return str(num + 100)
            except (ValueError, IndexError):
                # Si no podemos extraer el número, asignar un ID basado en el nombre completo
                if '1' in filename:
                    return '101'
                elif '3' in filename:
                    return '103'
                elif '7' in filename:
                    return '107'
                elif '9' in filename:
                    return '109'
                elif '12' in filename:
                    return '112'
        elif event_type == 'fixation':
            return '500'
        elif event_type == 'calm':
            # Para calm, usar el número tal cual (901, 902)
            if filename in ['901', '902']:
                return filename
        elif event_type == 'practice':
            # Para practice, usar el número tal cual (991-994)
            if filename in ['991', '992', '993', '994']:
                return filename
        
        return None
    
    # Añadir stim_id basado en el nombre del archivo y tipo de evento
    filtered_df['stim_id'] = filtered_df.apply(
        lambda row: map_to_stim_id(row['extracted_filename'], row['event_type']), 
        axis=1
    )
    
    # Mostrar resultados
    print("\nEventos identificados:")
    for event_type in filtered_df['event_type'].unique():
        count = filtered_df[filtered_df['event_type'] == event_type].shape[0]
        print(f"  {event_type}: {count} eventos")
    
    return filtered_df


def assign_durations(df, durations_dict=None):
    """
    Asigna duraciones a los eventos basados en el catálogo de estímulos.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las filas relevantes
    durations_dict : dict, optional
        Diccionario con los tiempos de inicio y fin para cada tipo de evento (obsoleto)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con las duraciones calculadas
    """
    # Crear columna de duración si no existe
    if 'duration' not in df.columns:
        df['duration'] = 0.0
    
    # Cargar las duraciones desde el archivo CSV
    video_durations_path = repo_root / "data" / "raw" / "stimuli" / "video_durations.csv"
    
    if video_durations_path.exists():
        print(f"Cargando duraciones de videos desde {video_durations_path}")
        video_durations_df = pd.read_csv(video_durations_path)
        
        # Crear un diccionario de duraciones {filename: duration}
        durations_dict = dict(zip(video_durations_df['filename'], video_durations_df['duration']))
        
        print(f"Se cargaron {len(durations_dict)} duraciones de videos")
    else:
        print(f"¡ADVERTENCIA! No se encontró el archivo de duraciones: {video_durations_path}")
        print("Se utilizarán valores predeterminados para las duraciones")
        
        # Definir las duraciones para cada tipo de estímulo (valores predeterminados)
        durations_dict = {
            "1.mp4": 103, "2.mp4": 229, "3.mp4": 94, "4.mp4": 60, "5.mp4": 81, 
            "9.mp4": 154, "10.mp4": 173, "11.mp4": 103, "12.mp4": 61, "13.mp4": 216, "14.mp4": 116,
            "901.mp4": 104, "902.mp4": 93,
            "991.mp4": 32, "992.mp4": 40, "993.mp4": 31, "994.mp4": 29,
            "fixation_cross.mp4": 300,
            "green_intensity_video_1.mp4": 60, "green_intensity_video_3.mp4": 60,
            "green_intensity_video_7.mp4": 60, "green_intensity_video_9.mp4": 60,
            "green_intensity_video_12.mp4": 60
        }
    
    # Verificar que tenemos las columnas necesarias
    if 'event_type' not in df.columns or 'stim_id' not in df.columns:
        print("Error: Faltan columnas 'event_type' o 'stim_id' en el DataFrame. No se pueden asignar duraciones.")
        return df
    
    # Asignar duraciones basadas en el tipo de evento y stim_id
    for idx, row in df.iterrows():
        event_type = row['event_type']
        stim_id = str(row['stim_id'])
        
        if event_type == 'video':
            # Para videos afectivos, buscar por nombre de archivo (1.mp4, 2.mp4, etc.)
            filename = f"{int(stim_id):d}.mp4"
            if filename in durations_dict:
                df.loc[idx, 'duration'] = durations_dict[filename]
                print(f"Asignando duración {durations_dict[filename]}s al video {stim_id} ({filename})")
            else:
                # Si no está en el diccionario, usar un valor por defecto
                print(f"Advertencia: No se encontró duración para el video {stim_id} ({filename}). Asignando duración por defecto.")
                df.loc[idx, 'duration'] = 103.0  # Duración por defecto
        
        elif event_type == 'video_luminance':
            # Para videos de luminancia, usar el formato green_intensity_video_X.mp4
            original_id = int(stim_id) - 100
            filename = f"green_intensity_video_{original_id}.mp4"
            if filename in durations_dict:
                df.loc[idx, 'duration'] = durations_dict[filename]
                print(f"Asignando duración {durations_dict[filename]}s al video de luminancia {stim_id} ({filename})")
            else:
                # Si no está en el diccionario, usar un valor por defecto
                print(f"Advertencia: No se encontró duración para el video de luminancia {stim_id} ({filename}). Asignando duración por defecto.")
                df.loc[idx, 'duration'] = 60.0  # Duración por defecto para luminancia
        
        elif event_type == 'fixation':
            # Para fijación, usar fixation_cross.mp4
            filename = "fixation_cross.mp4"
            if filename in durations_dict:
                df.loc[idx, 'duration'] = durations_dict[filename]
                print(f"Asignando duración {durations_dict[filename]}s a la fijación")
            else:
                # Si no está en el diccionario, usar un valor por defecto
                print(f"Advertencia: No se encontró duración para fixation_cross.mp4. Asignando duración por defecto.")
                df.loc[idx, 'duration'] = 300.0  # Duración por defecto para fijación
        
        elif event_type == 'calm':
            # Para calm, usar 901.mp4 o 902.mp4
            filename = f"{stim_id}.mp4"
            if filename in durations_dict:
                df.loc[idx, 'duration'] = durations_dict[filename]
                print(f"Asignando duración {durations_dict[filename]}s al estímulo calm {stim_id} ({filename})")
            else:
                # Si no está en el diccionario, usar un valor por defecto
                print(f"Advertencia: No se encontró duración para el estímulo calm {stim_id} ({filename}). Asignando duración por defecto.")
                df.loc[idx, 'duration'] = 100.0  # Duración por defecto para calm
        
        elif event_type == 'practice':
            # Para practice, usar 991.mp4, 992.mp4, etc.
            filename = f"{stim_id}.mp4"
            if filename in durations_dict:
                df.loc[idx, 'duration'] = durations_dict[filename]
                print(f"Asignando duración {durations_dict[filename]}s al estímulo de práctica {stim_id} ({filename})")
            else:
                # Si no está en el diccionario, usar un valor por defecto
                print(f"Advertencia: No se encontró duración para el estímulo de práctica {stim_id} ({filename}). Asignando duración por defecto.")
                df.loc[idx, 'duration'] = 30.0  # Duración por defecto para práctica
    
    return df


def create_annotations(df):
    """
    Crea anotaciones MNE a partir del DataFrame con duraciones.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con duraciones asignadas y columnas 'event_type' y 'stim_id'
    
    Returns
    -------
    mne.Annotations
        Objeto de anotaciones para agregar al raw
    """
    # Verificar que las columnas necesarias existen
    required_columns = ['event_type', 'stim_id', 'duration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Faltan columnas requeridas en el DataFrame: {missing_columns}")
        # Si falta stim_id pero tenemos event_type, podemos intentar recrearlo
        if 'stim_id' in missing_columns and 'event_type' in df.columns:
            print("Intentando recrear stim_id a partir de event_type...")
            # Recrear el mapeo de tipos de evento a IDs de estímulo
            event_type_to_default_id = {
                'video': '001',
                'video_luminance': '101',
                'fixation': '500',
                'calm': '901',
                'practice': '991'
            }
            df['stim_id'] = df['event_type'].map(event_type_to_default_id)
        else:
            return None
    
    # Inicializar onsets y duraciones
    onsets = np.zeros(len(df))
    durations = df['duration'].values
    
    # Crear descripciones en formato tipo/id para las anotaciones
    descriptions = []
    for i, row in df.iterrows():
        event_type = row['event_type']
        stim_id = row['stim_id']
        
        # Formatear la descripción como tipo/id
        if event_type == 'fixation':
            # Para fixation, no incluimos ID
            description = 'fixation'
        else:
            # Para otros tipos, incluimos el ID
            description = f"{event_type}/{stim_id}"
        
        descriptions.append(description)
        print(f"Creando anotación: {description} (duración: {row['duration']}s)")
    
    # Crear anotaciones
    annot = mne.Annotations(onsets, durations, descriptions)
    print(f"Creadas {len(annot)} anotaciones")
    
    return annot


def export_to_bids(raw, annot, bids_path):
    """
    Exporta los datos raw y las anotaciones a formato BIDS en derivatives.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    annot : mne.Annotations
        Objeto Annotations con las anotaciones
    bids_path : mne_bids.BIDSPath
        Ruta BIDS donde exportar los datos
        
    Returns
    -------
    tuple
        (events_df, event_id) con el DataFrame de eventos y el diccionario de IDs
    """
    # Crear el catálogo de estímulos y el diccionario event_id
    stimulus_catalog = create_stimulus_catalog()
    event_id = create_event_id_dict()
    
    # Agregar anotaciones al raw (solo para procesar, no modificamos el original)
    raw_copy = raw.copy()
    raw_copy.set_annotations(annot)
    
    # Convertir anotaciones a eventos
    events, event_id_auto = mne.events_from_annotations(raw_copy)
    print(f"Anotaciones convertidas a {len(events)} eventos con {len(event_id_auto)} tipos de evento")
    
    # Crear directorio de derivados
    deriv_root = Path(bids_path.root) / '..' / 'derivatives' / 'events'
    deriv_root.mkdir(exist_ok=True, parents=True)
    
    # Crear BIDSPath para los derivados
    deriv_path = BIDSPath(
        subject=bids_path.subject,
        session=bids_path.session,
        task=bids_path.task,
        run=bids_path.run,
        acquisition=bids_path.acquisition,
        datatype=bids_path.datatype,
        suffix='events',
        extension='.tsv',
        root=str(deriv_root),
        check=False
    )
    
    # Asegurarnos de que el directorio existe
    Path(deriv_path.directory).mkdir(exist_ok=True, parents=True)
    
    # Mapear las descripciones de anotaciones a claves en el catálogo de estímulos
    # Asumimos que las descripciones son del tipo "video", "calm_901", etc.
    # y las mapeamos a "video/001", "calm/901", etc.
    descriptions = []
    for desc in annot.description:
        # Manejar caso de calm_901 -> calm/901
        if desc.startswith('calm_'):
            stim_id = desc.split('_')[1]
            mapped_desc = f"calm/{stim_id}"
        # Manejar caso de luminance -> video_luminance/101 (simplificación)
        elif desc == 'luminance':
            # Asignar un ID de luminancia por defecto
            mapped_desc = "video_luminance/101"
        # Manejar caso de video -> video/001 (simplificación)
        elif desc == 'video':
            # Asignar un ID de video por defecto
            mapped_desc = "video/001"
        # Para otros casos, intentamos usar la descripción directamente
        else:
            # Si la descripción está en el catálogo, la usamos directamente
            if desc in stimulus_catalog:
                mapped_desc = desc
            else:
                # Si no, usamos una descripción por defecto
                print(f"Advertencia: Descripción '{desc}' no encontrada en el catálogo, usando 'fixation'")
                mapped_desc = "fixation"
        descriptions.append(mapped_desc)
    
    # Crear el DataFrame de eventos con todos los campos requeridos
    events_df = pd.DataFrame()
    
    # Columnas obligatorias
    events_df['onset'] = annot.onset
    events_df['duration'] = annot.duration
    
    # Extraer información del catálogo para cada descripción
    trial_types = []
    stim_ids = []
    conditions = []
    stim_files = []
    
    for desc in descriptions:
        if desc in stimulus_catalog:
            info = stimulus_catalog[desc]
            trial_types.append(info["trial_type"])
            stim_ids.append(info["stim_id"])
            conditions.append(info["condition"])
            stim_files.append(info["stim_file"])
        else:
            # Si la descripción no está en el catálogo, usamos valores por defecto
            parts = desc.split('/')
            if len(parts) > 1 and parts[0] in ["video", "video_luminance", "calm", "practice"]:
                trial_types.append(parts[0])
                try:
                    stim_id = int(parts[1])
                    stim_ids.append(stim_id)
                    
                    # Determinar el nombre de archivo basado en el tipo y el ID
                    if parts[0] == "video":
                        # Para videos, usar simplemente el número
                        stim_file = f"stimuli/{stim_id % 100}.mp4"
                    elif parts[0] == "video_luminance":
                        # Para luminancia, usar el formato green_intensity_video_X
                        orig_id = stim_id - 100
                        stim_file = f"stimuli/green_intensity_video_{orig_id}.mp4"
                    elif parts[0] == "calm":
                        # Para calm, usar directamente el ID
                        stim_file = f"stimuli/{stim_id}.mp4"
                    elif parts[0] == "practice":
                        # Para practice, usar directamente el ID
                        stim_file = f"stimuli/{stim_id}.mp4"
                    else:
                        stim_file = f"stimuli/{stim_id}.mp4"
                    
                    stim_files.append(stim_file)
                except ValueError:
                    stim_ids.append(0)
                    stim_files.append("stimuli/unknown.mp4")
                
                if parts[0] == "video":
                    conditions.append("affective")
                elif parts[0] == "video_luminance":
                    conditions.append("luminance")
                elif parts[0] == "calm":
                    conditions.append("calm")
                elif parts[0] == "practice":
                    conditions.append("practice")
                else:
                    conditions.append("unknown")
            else:
                if parts[0] == "fixation":
                    trial_types.append("fixation")
                    stim_ids.append(500)
                    conditions.append("baseline")
                    stim_files.append("stimuli/fixation_cross.mp4")
                else:
                    trial_types.append("unknown")
                    stim_ids.append(0)
                    conditions.append("unknown")
                    stim_files.append("stimuli/unknown.mp4")
    
    # Añadir columnas al DataFrame
    events_df['trial_type'] = trial_types
    events_df['stim_id'] = stim_ids
    events_df['condition'] = conditions
    events_df['stim_file'] = stim_files
    
    # Guardar events.tsv
    events_tsv_path = str(deriv_path.fpath)
    events_df.to_csv(events_tsv_path, sep='\t', index=False)
    print(f"Archivo events.tsv guardado en: {events_tsv_path}")
    
    # Crear el archivo events.json con metadatos según el estándar BIDS
    events_json = {
        "trial_type": {
            "Description": "Broad class of event",
            "Levels": {
                "video": "Affective video clip (original colours)",
                "video_luminance": "Video clip with luminance-enhanced green channel",
                "fixation": "Central fixation cross (baseline)",
                "calm": "Calming audiovisual clip",
                "practice": "Practice stimulus used before main task"
            }
        },
        "stim_id": {
            "LongName": "Stimulus identifier",
            "Description": "001-014 = affective videos; 101/103/107/109/112 = luminance videos; 500 = fixation; 901–902 = calm; 991-994 = practice"
        },
        "condition": {
            "Description": "Experimental condition of the event",
            "Levels": {
                "affective": "Emotionally engaging content",
                "luminance": "Green-channel luminance manipulation",
                "baseline": "Baseline with fixation",
                "calm": "Calm/relaxation stimulus",
                "practice": "Pre-experiment practice clip"
            }
        },
        "stim_file": {
            "Description": "Relative path to the stimulus file inside the stimuli/ folder"
        }
    }
    
    # Añadir información de HED Tags como un campo adicional en cada trial_type
    # Nota: No usar 'HED' como clave directa en el sidecar JSON
    for event_type, hed_tag in {
        "video": "Sensory-event,Visual,Video,Stimulus/Affective",
        "video_luminance": "Sensory-event,Visual,Video,Attribute/Luminance-increase,Stimulus/Affective",
        "fixation": "Task,Instruction,Fixation,Baseline",
        "calm": "Sensory-event,Audio-Visual,Relaxation",
        "practice": "Task,Instruction,Practice"
    }.items():
        if event_type in events_json["trial_type"]["Levels"]:
            events_json["trial_type"]["Levels"][event_type] += f" [HED: {hed_tag}]"
    
    # Guardar events.json
    events_json_path = events_tsv_path.replace('.tsv', '.json')
    with open(events_json_path, 'w', encoding='utf-8') as f:
        json.dump(events_json, f, indent=4, ensure_ascii=False)
    print(f"Archivo events.json guardado en: {events_json_path}")
    
    # Crear README.md si no existe
    readme_path = deriv_root / 'README.md'
    if not readme_path.exists():
        readme_content = """# CAMPEONES Events Derivatives

Este directorio contiene archivos de eventos derivados del proyecto CAMPEONES.

## Contenido

- `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`: Archivos de eventos con marcas temporales
- `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.json`: Metadatos de los eventos

Los eventos fueron generados a partir de las planillas de órdenes y alineados con los registros EEG.

## Estructura de eventos

Cada archivo events.tsv contiene las siguientes columnas:

| Columna      | Descripción                                           |
|--------------|-------------------------------------------------------|
| onset        | Tiempo de inicio del evento en segundos               |
| duration     | Duración del evento en segundos                       |
| trial_type   | Tipo de evento (video, video_luminance, fixation, etc)|
| stim_id      | Identificador único del estímulo                      |
| condition    | Condición experimental                                |
| stim_file    | Ruta relativa al archivo de estímulo                  |
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"Archivo README.md creado en: {readme_path}")
    
    # Crear dataset_description.json para el derivado
    dataset_desc = {
        'Name': 'CAMPEONES events',
        'BIDSVersion': '1.8.0',
        'DatasetType': 'derivative',
        'Authors': [
            'D\'Amelio, Tomás Ariel',
            'Rodriguez Cuello, Jerónimo',
            'Aboitiz, Julieta',
            'Bruno, Nicolás Marcelo',
            'Cavanna, Federico',
            'de La Fuente, Laura Alethia',
            'Müller, Stephanie Andrea',
            'Pallavicini, Carla',
            'Engemann, Denis-Alexander',
            'Vidaurre, Diego',
            'Tagliazucchi, Enzo'
        ],
        'GeneratedBy': [
            {
                'Name': 'create_events_tsv.py',
                'Version': '1.0.0',
                'Description': 'Script para crear archivos events.tsv a partir de planillas de órdenes'
            }
        ],
        'SourceDatasets': [
            {
                'Path': '../../raw',
                'Description': 'Raw BIDS dataset'
            }
        ]
    }
    
    # Guardar dataset_description.json
    desc_path = deriv_root / 'dataset_description.json'
    with open(desc_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_desc, f, indent=4, ensure_ascii=False)
    print(f"Archivo dataset_description.json creado/actualizado en: {desc_path}")
    
    print(f"Datos exportados a {deriv_path.directory}")
    return events_df, event_id


def map_block_to_task(block_num):
    """
    Mapea el número de bloque al formato de número de tarea BIDS.
    
    Parameters
    ----------
    block_num : str
        Número de bloque (e.g., '1', '2')
    
    Returns
    -------
    str
        Número de tarea en formato BIDS (e.g., '01', '02')
    """
    try:
        return str(int(block_num)).zfill(2)
    except (ValueError, TypeError):
        return None


def extract_order_info(filename):
    """
    Extrae información del nombre de un archivo de orden.
    
    Parameters
    ----------
    filename : str
        Nombre del archivo (e.g., 'order_matrix_12_A_block1_VR.xlsx')
    
    Returns
    -------
    dict
        Diccionario con la información extraída (subject, condition, block)
    """
    info = {}
    
    # Extraer el número de sujeto
    subject_match = re.search(r'order_matrix_(\d+)', filename)
    if subject_match:
        info['subject'] = subject_match.group(1)
    
    # Extraer la condición (A o B)
    condition_match = re.search(r'_([AB])_', filename)
    if condition_match:
        info['condition'] = condition_match.group(1).lower()
    
    # Extraer el número de bloque
    block_match = re.search(r'block(\d+)', filename)
    if block_match:
        info['block'] = block_match.group(1)
        # Mapear bloque a tarea (e.g., block1 -> task-01)
        info['task'] = map_block_to_task(info['block'])
    
    return info


def extract_eeg_info(filename):
    """
    Extrae información del nombre de un archivo EEG.
    
    Parameters
    ----------
    filename : str
        Nombre del archivo (e.g., 'sub-16_ses-vr_task-02_acq-a_run-003_eeg.eeg')
    
    Returns
    -------
    dict
        Diccionario con la información extraída (subject, session, task, acq, run)
    """
    info = {}
    
    # Usar una expresión regular más precisa para extraer todos los componentes
    pattern = r'sub-(\w+)_ses-(\w+)_task-(\w+)_acq-(\w+)_run-(\d+)_'
    match = re.search(pattern, filename)
    
    if match:
        info['subject'] = match.group(1)
        info['session'] = match.group(2)
        info['task'] = match.group(3)
        info['acq'] = match.group(4)
        info['run'] = match.group(5)
    else:
        # Fallback a extracción individual de componentes
        subject_match = re.search(r'sub-(\w+)', filename)
        if subject_match:
            info['subject'] = subject_match.group(1)
        
        session_match = re.search(r'ses-(\w+)', filename)
        if session_match:
            info['session'] = session_match.group(1)
        
        task_match = re.search(r'task-(\w+)', filename)
        if task_match:
            info['task'] = task_match.group(1)
        
        acq_match = re.search(r'acq-(\w+)', filename)
        if acq_match:
            info['acq'] = acq_match.group(1)
        
        # Extraer solo los dígitos del run
        run_match = re.search(r'run-(\d+)', filename)
        if run_match:
            info['run'] = run_match.group(1)
    
    print(f"Información extraída del archivo: {info}")
    return info


def find_matching_order_file(subject, task_num, condition, order_dir):
    """
    Encuentra el archivo de orden que coincide con el sujeto, tarea y condición.
    
    Parameters
    ----------
    subject : str
        ID del sujeto (e.g., '13')
    task_num : str
        Número de tarea (e.g., '01')
    condition : str
        Condición (e.g., 'a' o 'b')
    order_dir : Path
        Directorio donde buscar los archivos de orden
    
    Returns
    -------
    Path or None
        Ruta al archivo de orden correspondiente, o None si no se encuentra
    """
    # Convertir el número de tarea a un número de bloque (sin ceros a la izquierda)
    try:
        block_num = str(int(task_num))
    except (ValueError, TypeError):
        print(f"No se pudo convertir el número de tarea '{task_num}' a número de bloque")
        return None
    
    # Convertir condición a mayúscula para el patrón de búsqueda
    condition_upper = condition.upper() if condition else '[AB]'
    
    # Patrón para encontrar archivos que coincidan con el sujeto, bloque y condición
    pattern = f"order_matrix_{subject}_{condition_upper}_block{block_num}_VR.xlsx"
    matching_files = list(order_dir.glob(pattern))
    
    # Si no se encuentra con la condición específica, buscar sin especificar condición
    if not matching_files and condition:
        print(f"No se encontraron archivos con condición {condition_upper}, buscando cualquier condición...")
        pattern = f"order_matrix_{subject}_*_block{block_num}_VR.xlsx"
        matching_files = list(order_dir.glob(pattern))
    
    if not matching_files:
        print(f"No se encontraron archivos que coincidan con el patrón: {pattern}")
        # Intentar buscar todos los archivos de matriz de orden para este sujeto
        all_files = list(order_dir.glob(f"order_matrix_{subject}_*.xlsx"))
        if all_files:
            print(f"Archivos disponibles para el sujeto {subject}:")
            for f in all_files:
                print(f"  - {f.name}")
        return None
    
    # Si hay más de un archivo, mostrar las opciones y usar el primero
    if len(matching_files) > 1:
        print(f"Se encontraron múltiples archivos que coinciden con el patrón:")
        for i, f in enumerate(matching_files):
            print(f"  {i+1}. {f.name}")
        
        print(f"Usando el archivo: {matching_files[0].name}")
    
    return matching_files[0]


def find_matching_eeg_file(bids_root, subject, session, task, acq=None, run=None):
    """
    Encuentra el archivo EEG que coincide con los parámetros dados.
    
    Parameters
    ----------
    bids_root : Path
        Ruta raíz del directorio BIDS
    subject : str
        ID del sujeto (e.g., '16')
    session : str
        ID de la sesión (e.g., 'vr')
    task : str
        ID de la tarea (e.g., '02')
    acq : str, optional
        Parámetro de adquisición (e.g., 'a')
    run : str, optional
        Número de ejecución (e.g., '003')
    
    Returns
    -------
    tuple
        (Path, dict) con la ruta al archivo EEG y un diccionario con los parámetros encontrados
    """
    # Directorio donde buscar los archivos EEG
    eeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg"
    
    if not eeg_dir.exists():
        print(f"Directorio EEG no encontrado: {eeg_dir}")
        return None, None
    
    # Si el run no está especificado, intentar calcularlo a partir del task
    if run is None and task is not None:
        try:
            # Formato: task-XX → run-0XX+1 (ej: task-02 → run-003)
            task_num = int(task)
            run = str(task_num + 1).zfill(3)
            print(f"Run calculado a partir de task: task-{task} → run-{run}")
        except (ValueError, TypeError):
            pass
    
    # Construir un patrón de búsqueda con los parámetros que conocemos
    pattern = f"sub-{subject}_ses-{session}"
    if task:
        # Asegurarse de que task tiene formato XX (dos dígitos)
        task_fmt = task
        if task_fmt.isdigit():
            task_fmt = str(int(task_fmt)).zfill(2)
        pattern += f"_task-{task_fmt}"
    if acq:
        # Asegurarse de que acq está en minúsculas
        pattern += f"_acq-{acq.lower()}"
    if run:
        # Asegurarse de que run tiene formato XXX (tres dígitos)
        run_fmt = run
        if run_fmt.isdigit():
            run_fmt = str(int(run_fmt)).zfill(3)
        pattern += f"_run-{run_fmt}"
    pattern += "_eeg.eeg"
    
    print(f"Buscando archivos con patrón: {pattern}")
    matching_files = list(eeg_dir.glob(pattern))
    
    # Si no se encuentra con todos los parámetros, intentar ajustar el run
    if not matching_files and task:
        print(f"No se encontraron archivos con el patrón completo, intentando con diferentes formatos de run...")
        
        # Probar con diferentes formatos de run
        for run_fmt in [str(int(task)+1).zfill(3), str(int(task)).zfill(3), str(int(task)+1)]:
            alt_pattern = f"sub-{subject}_ses-{session}_task-{task.zfill(2) if task.isdigit() else task}"
            if acq:
                alt_pattern += f"_acq-{acq.lower()}"
            alt_pattern += f"_run-{run_fmt}_eeg.eeg"
            
            print(f"Intentando con patrón alternativo: {alt_pattern}")
            alt_files = list(eeg_dir.glob(alt_pattern))
            if alt_files:
                matching_files = alt_files
                break
    
    # Si aún no se encuentra, mostrar todos los archivos disponibles
    if not matching_files:
        print(f"No se encontraron archivos EEG para sub-{subject} ses-{session} task-{task}")
        all_files = list(eeg_dir.glob(f"sub-{subject}_ses-{session}*.eeg"))
        if all_files:
            print(f"Archivos disponibles para sub-{subject} ses-{session}:")
            for f in all_files:
                print(f"  - {f.name}")
        return None, None
    
    # Si hay más de un archivo, mostrar las opciones
    if len(matching_files) > 1:
        print(f"Se encontraron múltiples archivos EEG:")
        for i, f in enumerate(matching_files):
            print(f"  {i+1}. {f.name}")
        print(f"Usando el primer archivo encontrado: {matching_files[0].name}")
    else:
        print(f"Archivo encontrado: {matching_files[0].name}")
    
    # Extraer la información del archivo seleccionado
    selected_file = matching_files[0]
    file_info = extract_eeg_info(selected_file.name)
    
    return selected_file, file_info


def process_subject(subject, session=None, task=None, acq=None, run=None):
    """
    Procesa un sujeto específico siguiendo todos los pasos.
    
    Parameters
    ----------
    subject : str
        ID del sujeto (e.g., '16')
    session : str, optional
        ID de la sesión (e.g., 'vr')
    task : str, optional
        ID de la tarea (e.g., '02')
    acq : str, optional
        Parámetro de adquisición (e.g., 'a')
    run : str, optional
        Número de ejecución (e.g., '003')
    """
    # Imprimir valores de entrada para depuración
    print(f"\n=== Iniciando proceso para sub-{subject} ses-{session} task-{task} acq-{acq} run-{run} ===\n")
    
    if session is None:
        session = 'vr'  # Usar 'vr' como valor predeterminado
    
    # 1. Localizar planilla de órdenes
    order_matrix_dir = repo_root / 'data' / 'sourcedata' / 'xdf' / f'sub-{subject}'
    
    # Si no se proporcionó task, listar todas las planillas disponibles y pedir elegir
    if task is None:
        all_order_files = list(order_matrix_dir.glob(f"order_matrix_{subject}_*.xlsx"))
        if all_order_files:
            print(f"Planillas disponibles para el sujeto {subject}:")
            for i, f in enumerate(all_order_files):
                info = extract_order_info(f.name)
                task_str = f"task-{info['task']}" if 'task' in info else 'desconocido'
                cond_str = f"cond-{info['condition']}" if 'condition' in info else 'desconocida'
                print(f"  {i+1}. {f.name} ({task_str}, {cond_str})")
            
            # Por defecto, usar la primera
            order_file = all_order_files[0]
            info = extract_order_info(order_file.name)
            if 'task' in info:
                task = info['task']
                print(f"Usando tarea: {task}")
            if 'condition' in info and acq is None:
                acq = info['condition']
                print(f"Usando condición: {acq}")
        else:
            print(f"No se encontraron planillas para el sujeto {subject}")
            return
    else:
        # Buscar archivo que coincida con el sujeto/tarea/condición
        order_file = find_matching_order_file(subject, task, acq, order_matrix_dir)
        
        if order_file is None:
            print(f"No se pudo encontrar una planilla de órdenes adecuada para sub-{subject} task-{task}")
            return
    
    print(f"Usando planilla: {order_file}")
    
    # Extraer información del nombre del archivo de orden
    order_info = extract_order_info(order_file.name)
    print(f"Información extraída del archivo de órdenes: {order_info}")
    
    # Actualizar parámetros si no se proporcionaron
    if acq is None and 'condition' in order_info:
        acq = order_info['condition']
        print(f"Usando condición detectada como parámetro de adquisición: acq={acq}")
    
    if task is None and 'task' in order_info:
        task = order_info['task']
        print(f"Usando tarea detectada: task={task}")
    
    # Explorar el contenido del archivo Excel para entender su estructura
    excel_df = explore_excel_file(order_file)
    if excel_df is None:
        return
    
    # 2. Cargar y filtrar filas relevantes
    df = load_order_matrix(order_file)
    if df is None:
        return
    
    filtered_df = filter_relevant_rows(df)
    if filtered_df is None or filtered_df.empty:
        print("No se encontraron filas relevantes, verificar la estructura de la planilla")
        return
    
    # 3. Asignar duraciones usando el catálogo de estímulos
    df_with_durations = assign_durations(filtered_df)
    
    # 4. Inicializar anotaciones
    annot = create_annotations(df_with_durations)
    
    # 5. Encontrar y cargar el archivo raw correspondiente
    bids_root = repo_root / 'data' / 'raw'
    eeg_file, eeg_info = find_matching_eeg_file(bids_root, subject, session, task, acq, run)
    
    if eeg_file is None:
        print(f"No se pudo encontrar un archivo EEG adecuado")
        return
    
    print(f"Usando archivo EEG: {eeg_file}")
    print(f"Información extraída del archivo EEG: {eeg_info}")
    
    # Actualizar los parámetros con la información del archivo encontrado
    # Usar sólo información válida del archivo EEG, evitando sobrescribir parámetros correctos
    if 'acq' in eeg_info:
        acq = eeg_info['acq']
    if 'run' in eeg_info:
        run = eeg_info['run']
    
    # Formatear correctamente los parámetros para el BIDSPath
    # Los parámetros deben estar limpios, sin "-", "_" o "/" adicionales
    print(f"Parámetros antes de formatear: subject={subject}, session={session}, task={task}, acq={acq}, run={run}")
    
    # Limpiar los parámetros para asegurar que son válidos
    if task and isinstance(task, str) and task.isdigit():
        task_fmt = str(int(task)).zfill(2)
    else:
        task_fmt = task
    
    if run and isinstance(run, str) and run.isdigit():
        run_fmt = str(int(run)).zfill(3)
    else:
        run_fmt = run
    
    if acq and isinstance(acq, str):
        # Asegurar que solo contiene caracteres alfanuméricos válidos
        acq_fmt = re.sub(r'[^a-zA-Z0-9]', '', acq).lower()
    else:
        acq_fmt = None
    
    print(f"Parámetros formateados: subject={subject}, session={session}, task={task_fmt}, acq={acq_fmt}, run={run_fmt}")
    
    # Crear un BIDSPath con los parámetros validados
    try:
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task_fmt,
            run=run_fmt,
            acquisition=acq_fmt,
            datatype='eeg',
            root=bids_root,
            extension='.vhdr'  # Cambiado a .vhdr para cargar correctamente
        )
        print(f"BIDSPath creado correctamente: {bids_path}")
    except ValueError as e:
        print(f"Error al crear BIDSPath: {e}")
        # Intentar crear un BIDSPath con parámetros mínimos
        print("Intentando crear BIDSPath con parámetros mínimos...")
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            datatype='eeg',
            root=bids_root,
            extension='.vhdr'  # Cambiado a .vhdr para cargar correctamente
        )
    
    try:
        # Cargar archivo BrainVision
        vhdr_file = eeg_file.with_suffix('.vhdr')
        print(f"Intentando cargar archivo raw desde: {vhdr_file}")
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        print(f"Archivo raw cargado correctamente: {vhdr_file}")
    except Exception as e:
        print(f"Error al cargar el archivo raw: {e}")
        return
    
    # 6. Exportar a derivatives sin modificar los archivos originales
    events_df, event_id = export_to_bids(raw, annot, bids_path)
    
    print(f"Procesamiento completado para sub-{subject} ses-{session} task-{task} run-{run}")
    
    # 7. Crear dataset description en derivatives si no existe
    deriv_root = repo_root / 'data' / 'derivatives' / 'events'
    if not (deriv_root / 'dataset_description.json').exists():
        print("Creando dataset_description.json para los derivados")
        try:
            make_dataset_description(
                path=str(deriv_root),
                name="CAMPEONES events",
                dataset_type="derivative",
                bids_version="1.8.0",
                pipeline_name="create_events_tsv",
                source_datasets=[{
                    "path": "../../raw",
                    "description": "Raw BIDS dataset"
                }],
                overwrite=True
            )
            print(f"dataset_description.json creado en {deriv_root}")
        except Exception as e:
            print(f"Error al crear dataset_description.json: {e}")
    
    return True


def main():
    """Función principal"""
    args = parse_args()
    
    print("\n*** CREANDO ARCHIVOS EVENTS.TSV EN DERIVATIVES (SIN MODIFICAR RAW) ***\n")
    
    # Definir la relación entre task y run
    task_run_map = {
        '01': '002',
        '02': '003',
        '03': '004',
        '04': '005'
    }
    
    # Valores de adquisición disponibles
    acq_values = ['a', 'b']
    
    # Lista de sujetos a procesar
    subjects_to_process = args.subjects if args.subjects else ['16']
    print(f"Sujetos a procesar: {subjects_to_process}")
    
    successful_runs = 0
    failed_runs = 0
    
    for subject in subjects_to_process:
        print(f"\n=== Procesando sujeto {subject} ===\n")
        
        # Si se especificó una tarea/run específica
        if args.task and args.run:
            run = args.run
            task = args.task
            acq_list = [args.acq] if args.acq else acq_values
            
            for acq in acq_list:
                print(f"Procesando sub-{subject} task-{task} acq-{acq} run-{run}")
                if process_subject(subject, args.session, task, acq, run):
                    successful_runs += 1
                else:
                    failed_runs += 1
        
        # Si se debe procesar todas las runs
        elif args.all_runs or (not args.task and not args.run):
            for task, run in task_run_map.items():
                acq_list = [args.acq] if args.acq else acq_values
                
                for acq in acq_list:
                    print(f"Procesando sub-{subject} task-{task} acq-{acq} run-{run}")
                    if process_subject(subject, args.session, task, acq, run):
                        successful_runs += 1
                    else:
                        failed_runs += 1
        
        # Si solo se especificó una tarea
        elif args.task:
            task = args.task
            run = task_run_map.get(task, f"00{int(task)+1}")
            acq_list = [args.acq] if args.acq else acq_values
            
            for acq in acq_list:
                print(f"Procesando sub-{subject} task-{task} acq-{acq} run-{run}")
                if process_subject(subject, args.session, task, acq, run):
                    successful_runs += 1
                else:
                    failed_runs += 1
        
        # Si solo se especificó un run
        elif args.run:
            run = args.run
            # Deducir la tarea a partir del run
            task = None
            for t, r in task_run_map.items():
                if r == run:
                    task = t
                    break
            
            if task:
                acq_list = [args.acq] if args.acq else acq_values
                
                for acq in acq_list:
                    print(f"Procesando sub-{subject} task-{task} acq-{acq} run-{run}")
                    if process_subject(subject, args.session, task, acq, run):
                        successful_runs += 1
                    else:
                        failed_runs += 1
            else:
                print(f"No se pudo deducir la tarea para el run {run}")
                failed_runs += 1
    
    print("\n=== Resumen de procesamiento ===")
    print(f"Runs procesadas exitosamente: {successful_runs}")
    print(f"Runs con errores: {failed_runs}")
    
    if successful_runs > 0:
        print("\nProcesamiento completado:")
        deriv_path = repo_root / 'data' / 'derivatives' / 'events'
        print(f"Los archivos events.tsv y events.json se han creado en: {deriv_path}")
        print("\nValidando la estructura BIDS:")
        print("Para validar, ejecuta en la terminal: bids-validator data/derivatives/events")
    else:
        print("\nEl procesamiento falló. Revisa los mensajes de error para más detalles.")


def create_stimulus_catalog():
    """
    Crea un catálogo completo de estímulos con la información de stim_id, condition y stim_file.
    
    Returns
    -------
    dict
        Diccionario con la información de cada estímulo
    """
    # Cargar las duraciones desde el archivo CSV
    video_durations_path = repo_root / "data" / "raw" / "stimuli" / "video_durations.csv"
    
    # Diccionario para almacenar las duraciones
    durations_dict = {}
    
    if video_durations_path.exists():
        print(f"Cargando duraciones de videos para el catálogo desde {video_durations_path}")
        video_durations_df = pd.read_csv(video_durations_path)
        
        # Crear un diccionario de duraciones {filename: duration}
        durations_dict = dict(zip(video_durations_df['filename'], video_durations_df['duration']))
        
        print(f"Se cargaron {len(durations_dict)} duraciones de videos para el catálogo")
    else:
        print(f"¡ADVERTENCIA! No se encontró el archivo de duraciones: {video_durations_path}")
        print("Se utilizarán valores predeterminados para el catálogo de estímulos")
        
        # Valores predeterminados
        durations_dict = {
            "1.mp4": 103, "2.mp4": 229, "3.mp4": 94, "4.mp4": 60, "5.mp4": 81, 
            "9.mp4": 154, "10.mp4": 173, "11.mp4": 103, "12.mp4": 61, "13.mp4": 216, "14.mp4": 116,
            "901.mp4": 104, "902.mp4": 93,
            "991.mp4": 32, "992.mp4": 40, "993.mp4": 31, "994.mp4": 29,
            "fixation_cross.mp4": 300,
            "green_intensity_video_1.mp4": 60, "green_intensity_video_3.mp4": 60,
            "green_intensity_video_7.mp4": 60, "green_intensity_video_9.mp4": 60,
            "green_intensity_video_12.mp4": 60
        }
    
    # Crear el catálogo completo de estímulos
    catalog = {}
    
    # Videos afectivos (001-014)
    for i in range(1, 15):
        stim_id = f"{i:03d}"
        filename = f"{i}.mp4"
        
        # Obtener la duración del diccionario o usar un valor predeterminado
        duration = durations_dict.get(filename, 100.0)
        
        catalog[f"video/{stim_id}"] = {
            "trial_type": "video",
            "stim_id": int(stim_id),
            "condition": "affective",
            "duration": duration,
            "stim_file": f"stimuli/{filename}"  # Usar el número simple como nombre de archivo
        }
    
    # Videos de luminancia (101, 103, 107, 109, 112)
    luminance_ids = [101, 103, 107, 109, 112]
    for stim_id in luminance_ids:
        orig_id = stim_id - 100
        filename = f"green_intensity_video_{orig_id}.mp4"
        
        # Obtener la duración del diccionario o usar un valor predeterminado
        duration = durations_dict.get(filename, 60.0)
        
        catalog[f"video_luminance/{stim_id}"] = {
            "trial_type": "video_luminance",
            "stim_id": stim_id,
            "condition": "luminance",
            "duration": duration,
            "stim_file": f"stimuli/{filename}"  # Usar el nombre original
        }
    
    # Fijación (500)
    filename = "fixation_cross.mp4"
    duration = durations_dict.get(filename, 300.0)
    
    catalog["fixation"] = {
        "trial_type": "fixation",
        "stim_id": 500,
        "condition": "baseline",
        "duration": duration,
        "stim_file": f"stimuli/{filename}"
    }
    
    # Calma (901-902)
    for stim_id in ["901", "902"]:
        filename = f"{stim_id}.mp4"
        duration = durations_dict.get(filename, 100.0)
        
        catalog[f"calm/{stim_id}"] = {
            "trial_type": "calm",
            "stim_id": int(stim_id),
            "condition": "calm",
            "duration": duration,
            "stim_file": f"stimuli/{filename}"
        }
    
    # Práctica (991-994)
    for stim_id in ["991", "992", "993", "994"]:
        filename = f"{stim_id}.mp4"
        duration = durations_dict.get(filename, 30.0)
        
        catalog[f"practice/{stim_id}"] = {
            "trial_type": "practice",
            "stim_id": int(stim_id),
            "condition": "practice",
            "duration": duration,
            "stim_file": f"stimuli/{filename}"
        }
    
    return catalog


def create_event_id_dict():
    """
    Crea un diccionario event_id para usar con MNE-BIDS.
    
    Returns
    -------
    dict
        Diccionario event_id compatible con MNE-BIDS
    """
    # Videos afectivos 001-014 → códigos numéricos 1-14
    affective_ids = {f"video/{i:03d}": i for i in range(1, 15)}
    
    # Videos de luminancia → 101, 103, 107, 109, 112
    luminance_ids = {f"video_luminance/{i}": i for i in [101, 103, 107, 109, 112]}
    
    # Otras categorías
    other_ids = {
        "fixation": 500,
        "calm/901": 901,
        "calm/902": 902,
        "practice/991": 991,
        "practice/992": 992,
        "practice/993": 993,
        "practice/994": 994,
    }
    
    event_id = {**affective_ids, **luminance_ids, **other_ids}
    return event_id


if __name__ == "__main__":
    main()