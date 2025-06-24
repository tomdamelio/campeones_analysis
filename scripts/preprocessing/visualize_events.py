#!/usr/bin/env python
"""
Este script implementa la Fase B del proceso de análisis:
- Carga eventos desde cualquier directorio de derivatives (events, merged_events, aligned_events, etc.)
- Visualiza canales AUDIO/PHOTO/joystick_x junto con anotaciones existentes
- Aplica z-score a las señales para facilitar su visualización conjunta
- Permite agregar anotaciones de forma manual (modo interactivo)
- Modo solo lectura para visualización sin modificaciones
- Guarda las anotaciones alineadas en derivatives/aligned_events con metadatos BIDS

Ejemplos de uso:

    # Visualizar eventos desde derivatives/events (por defecto)
    python visualize_events.py --subject 16 --session vr --task 02 --run 003
    
    # Visualizar merged_events en modo solo lectura
    python visualize_events.py --subject 16 --session vr --task 02 --run 003 --events-dir merged_events --events-desc merged --read-only
    
    # Visualizar eventos alineados previamente
    python visualize_events.py --subject 16 --session vr --task 02 --run 003 --events-dir aligned_events --events-desc withann --read-only
    
    # Modo interactivo con guardado automático
    python visualize_events.py --subject 16 --session vr --task 02 --run 003 --force-save
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids, make_dataset_description
import matplotlib.pyplot as plt
from scipy import stats

# Configurar matplotlib para usar backend interactivo si está disponible
plt.ion()  # Activa el modo interactivo de matplotlib

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
        description="Visualiza canales AUDIO/PHOTO/joystick_x con anotaciones de eventos y permite añadir anotaciones manualmente"
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
    parser.add_argument("--force-save", action="store_true",
                        help="Forzar guardado de anotaciones sin preguntar")
    parser.add_argument("--no-zscore", action="store_true",
                        help="No aplicar z-score a las señales")
    parser.add_argument("--save-dir", type=str, default="aligned_events",
                        help="Directorio dentro de derivatives donde guardar los eventos (default: aligned_events)")
    parser.add_argument("--events-dir", type=str, default="events",
                        help="Directorio dentro de derivatives donde buscar los eventos (default: events)")
    parser.add_argument("--events-desc", type=str, default=None,
                        help="Descripción de los eventos a cargar (e.g., 'merged', 'withann')")
    parser.add_argument("--read-only", action="store_true",
                        help="Modo solo lectura: visualizar sin permitir modificaciones ni guardado")
    
    return parser.parse_args()


def load_events_and_raw(subject, session, task, run, acq=None, events_dir="events", events_desc=None):
    """
    Carga los eventos y los datos raw originales.
    
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
    events_dir : str, optional
        Directorio dentro de derivatives donde buscar los eventos
    events_desc : str, optional
        Descripción de los eventos a cargar
    
    Returns
    -------
    tuple
        (raw, events_df, bids_path, events_path) con los datos raw, eventos, la ruta BIDS y la ruta de eventos
    """
    print(f"\n=== Cargando datos para sub-{subject} ses-{session} task-{task} run-{run} ===\n")
    
    # Definir rutas
    bids_root = repo_root / 'data' / 'raw'
    deriv_root = repo_root / 'data' / 'derivatives' / events_dir
    
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
    
    # Crear BIDSPath para eventos
    events_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description=events_desc,
        extension='.tsv',
        root=deriv_root,
        check=False
    )
    
    print(f"Buscando eventos en: {events_path.fpath}")
    
    # Verificar que el archivo raw existe
    if not bids_path.fpath.exists():
        raise FileNotFoundError(f"Archivo raw no encontrado: {bids_path.fpath}")
    
    # Verificar que el archivo de eventos existe
    if not events_path.fpath.exists():
        # Si no se encuentra con la descripción especificada, intentar sin descripción
        if events_desc:
            print(f"No se encontró archivo con descripción '{events_desc}', intentando sin descripción...")
            alt_events_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run=run,
                acquisition=acq,
                datatype='eeg',
                suffix='events',
                extension='.tsv',
                root=deriv_root,
                check=False
            )
            if alt_events_path.fpath.exists():
                events_path = alt_events_path
                print(f"Archivo encontrado sin descripción: {events_path.fpath}")
            else:
                # Listar archivos disponibles
                print(f"Archivos de eventos disponibles en {deriv_root}:")
                pattern = f"sub-{subject}_ses-{session}_task-{task}_*_events.tsv"
                for file in deriv_root.glob(pattern):
                    print(f"  - {file.name}")
                raise FileNotFoundError(f"Archivo de eventos no encontrado: {events_path.fpath}")
        else:
            # Listar archivos disponibles
            print(f"Archivos de eventos disponibles en {deriv_root}:")
            pattern = f"sub-{subject}_ses-{session}_task-{task}_*_events.tsv"
            for file in deriv_root.glob(pattern):
                print(f"  - {file.name}")
            raise FileNotFoundError(f"Archivo de eventos no encontrado: {events_path.fpath}")
    
    # Cargar datos raw
    raw = read_raw_bids(bids_path, verbose=False)
    print(f"Datos raw cargados: {raw}")
    
    # Cargar eventos
    events_df = pd.read_csv(events_path.fpath, sep='\t')
    print(f"Eventos cargados: {len(events_df)} eventos desde {events_dir}")
    print(f"Columnas disponibles: {list(events_df.columns)}")
    print("\nPrimeros eventos:")
    print(events_df.head())
    
    return raw, events_df, bids_path, events_path


def apply_zscore_to_raw(raw):
    """
    Aplica z-score a los datos raw para normalizar las señales.
    Maneja casos problemáticos como señales constantes, NaN o varianza cero.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
        
    Returns
    -------
    mne.io.Raw
        Objeto Raw con los datos normalizados
    """
    import warnings
    
    # Crear una copia del raw para no modificar el original
    raw_zscore = raw.copy()
    
    # Cargar los datos en memoria
    data = raw_zscore.get_data()
    
    # Lista para almacenar canales problemáticos
    problematic_channels = []
    
    # Aplicar z-score a cada canal
    for i in range(data.shape[0]):
        channel_name = raw.ch_names[i]
        channel_data = data[i]
        
        # Verificar si hay NaN en el canal
        if np.isnan(channel_data).any():
            print(f"¡ADVERTENCIA! Canal {channel_name} contiene valores NaN. Usando nan_policy='omit'.")
            problematic_channels.append(f"{channel_name} (NaN)")
        
        # Verificar si el canal es constante (varianza cero)
        if np.var(channel_data) == 0:
            print(f"¡ADVERTENCIA! Canal {channel_name} tiene varianza cero (señal constante). Manteniendo valores originales.")
            problematic_channels.append(f"{channel_name} (varianza cero)")
            # Mantener los valores originales para canales constantes
            continue
        
        # Aplicar z-score con manejo de NaN
        try:
            zscore_data = stats.zscore(channel_data, nan_policy='omit')
            
            # Verificar si el resultado contiene NaN o infinitos
            if np.isnan(zscore_data).any() or np.isinf(zscore_data).any():
                print(f"¡ADVERTENCIA! Canal {channel_name} produjo NaN/Inf después del z-score. Manteniendo valores originales.")
                problematic_channels.append(f"{channel_name} (NaN/Inf post-zscore)")
                # Mantener los valores originales
                continue
            
            # Si todo está bien, usar los datos z-scoreados
            data[i] = zscore_data
            
        except Exception as e:
            print(f"¡ERROR! No se pudo aplicar z-score al canal {channel_name}: {e}")
            print(f"Manteniendo valores originales para el canal {channel_name}.")
            problematic_channels.append(f"{channel_name} (error: {str(e)})")
            # Mantener los valores originales
            continue
    
    # Mostrar resumen de canales problemáticos
    if problematic_channels:
        print(f"\nCanales con problemas en z-score ({len(problematic_channels)} de {data.shape[0]}):")
        for ch in problematic_channels:
            print(f"  - {ch}")
        print("Estos canales mantuvieron sus valores originales.\n")
    
    # Crear un nuevo objeto Raw con los datos procesados
    info = raw.info
    raw_zscore = mne.io.RawArray(data, info)
    
    # Copiar las anotaciones del raw original
    raw_zscore.set_annotations(raw.annotations)
    
    return raw_zscore


def visualize_channels_with_annotations(raw, events_df, apply_zscore=True, read_only=False):
    """
    Visualiza los canales AUDIO, PHOTO y joystick_x junto con las anotaciones.
    Permite agregar anotaciones de forma manual si no está en modo solo lectura.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    events_df : pandas.DataFrame
        DataFrame con los eventos
    apply_zscore : bool, optional
        Si es True, aplica z-score a las señales para normalizarlas
    read_only : bool, optional
        Si es True, modo solo lectura sin permitir modificaciones
        
    Returns
    -------
    tuple
        (updated_annotations, has_changes) con las anotaciones actualizadas y un flag que indica si hubo cambios
    """
    mode_str = "SOLO LECTURA" if read_only else "INTERACTIVO"
    print(f"\n=== Visualizando canales AUDIO/PHOTO/joystick_x y anotaciones - MODO {mode_str} ===\n")
    
    # Verificar que los canales existen
    available_channels = raw.ch_names
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
    
    print(f"Se usarán los siguientes canales: {channels_to_pick}")
    
    # Crear una copia del raw para no modificar el original
    raw_plot = raw.copy()
    
    # Crear anotaciones desde los eventos y añadirlas al raw_plot
    annot = mne.Annotations(
        onset=events_df['onset'].values,
        duration=events_df['duration'].values,
        description=events_df['trial_type'].values
    )
    
    # Guardar el número original de anotaciones para comparar después
    original_annotations = annot.copy()
    
    # Añadir anotaciones al raw_plot
    raw_plot.set_annotations(annot)
    
    # Seleccionar solo los canales requeridos
    raw_plot = raw_plot.pick_channels(channels_to_pick)
    
    # Aplicar z-score si se solicita
    if apply_zscore:
        print("Aplicando z-score a las señales para normalización...")
        raw_plot = apply_zscore_to_raw(raw_plot)
        print("Z-score aplicado. Las señales están ahora en unidades de desviación estándar.")
    
    # Mostrar información de los eventos
    print(f"\nEventos cargados: {len(events_df)}")
    if 'stim_id' in events_df.columns:
        print("Tipos de estímulos encontrados:")
        for trial_type in events_df['trial_type'].unique():
            subset = events_df[events_df['trial_type'] == trial_type]
            if 'stim_id' in subset.columns:
                stim_ids = subset['stim_id'].unique()
                print(f"  - {trial_type}: {len(subset)} eventos (IDs: {sorted(stim_ids)})")
            else:
                print(f"  - {trial_type}: {len(subset)} eventos")
    
    # Instrucciones para el usuario
    if read_only:
        print("\n--- MODO SOLO LECTURA ---")
        print("Visualización sin posibilidad de modificación.")
        print("1. Usa las teclas de navegación para explorar los datos")
        print("2. Presiona 'j'/'k' para ajustar la escala vertical")
        print("3. Usa las flechas o barra espaciadora para navegar en el tiempo")
        print("4. Cierra la ventana para finalizar\n")
    else:
        print("\n--- MODO INTERACTIVO ---")
        print("1. Presiona 'a' y arrastra para seleccionar una región")
        print("2. Ingresa un nombre para la anotación en la ventana emergente")
        print("   - Para videos, usa 'video'")
        print("   - Para luminancia, usa 'luminance'")
        print("   - Para otros eventos, usa nombres descriptivos")
        print("3. Para eliminar una anotación: Haz clic derecho sobre ella")
        print("4. Para ajustar los límites: Arrastra los bordes de una anotación")
        print("5. Presiona 'j'/'k' para ajustar la escala vertical")
        print("6. Cierra la ventana para finalizar y guardar los cambios\n")
    
    # Definir escalados específicos para evitar problemas con canales misc
    scalings = {}
    for ch_name in raw_plot.ch_names:
        ch_type = raw_plot.get_channel_types(picks=ch_name)[0]
        if ch_type == 'misc':
            # Para canales misc, usar un escalado fijo que funcione bien con datos z-scoreados
            scalings[ch_type] = 2.0  # Escalado conservador para datos normalizados
        elif ch_type == 'eeg':
            scalings[ch_type] = 20e-6
        elif ch_type == 'eog':
            scalings[ch_type] = 150e-6
        elif ch_type == 'ecg':
            scalings[ch_type] = 5e-4
        else:
            # Para otros tipos, usar un escalado genérico
            scalings[ch_type] = 1.0

    # Visualizar
    title_suffix = " (SOLO LECTURA)" if read_only else ""
    print("Abriendo visualizador de MNE. Cierra la ventana para continuar.")
    fig = raw_plot.plot(
        title=f"Canales {', '.join(channels_to_pick)} con anotaciones{title_suffix}",
        scalings=scalings,
        duration=180,
        start=0,
        show=True,
        block=True,
        decim=32  # Aplicar decimación para reducir la resolución
    )
    
    # En modo solo lectura, no hay cambios
    if read_only:
        print("\nVisualizador cerrado (modo solo lectura).")
        return original_annotations, False
    
    # Obtener las anotaciones actualizadas después de cerrar el visualizador
    updated_annotations = raw_plot.annotations
    
    # Mostrar información sobre las anotaciones
    print("\nVisualizador cerrado.")
    print(f"Anotaciones originales: {len(original_annotations)}")
    print(f"Anotaciones actualizadas: {len(updated_annotations)}")
    
    # Verificar si realmente hubo cambios en las anotaciones
    has_changes = False
    
    # Si el número de anotaciones es diferente, definitivamente hay cambios
    if len(original_annotations) != len(updated_annotations):
        has_changes = True
    else:
        # Comparar cada anotación para ver si hay cambios
        for i in range(len(original_annotations)):
            if (original_annotations.onset[i] != updated_annotations.onset[i] or
                original_annotations.duration[i] != updated_annotations.duration[i] or
                original_annotations.description[i] != updated_annotations.description[i]):
                has_changes = True
                break
    
    return updated_annotations, has_changes


def save_annotations_aligned(updated_annotations, events_path, subject, session, task, run, acq=None, save_dir="aligned_events"):
    """
    Guarda las anotaciones actualizadas en un archivo TSV en derivatives/aligned_events.
    
    Parameters
    ----------
    updated_annotations : mne.Annotations
        Anotaciones actualizadas
    events_path : mne_bids.BIDSPath
        Ruta BIDS del archivo de eventos original
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
    save_dir : str, optional
        Nombre del directorio dentro de derivatives donde guardar los eventos
        
    Returns
    -------
    str
        Ruta del archivo guardado
    """
    # Crear BIDSPath para guardar las anotaciones actualizadas
    aligned_root = repo_root / 'data' / 'derivatives' / save_dir
    
    # Asegurar formato correcto de parámetros
    if task and task.isdigit():
        task = task.zfill(2)
    if run and run.isdigit():
        run = run.zfill(3)
    if acq:
        acq = acq.lower()
        
    output_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description='withann',  # Entidad desc-withann para distinguir esta versión
        extension='.tsv',
        root=aligned_root,
        check=False
    )
    
    # Crear el directorio si no existe
    os.makedirs(output_path.directory, exist_ok=True)
    
    # Convertir anotaciones a DataFrame
    annotations_df = pd.DataFrame({
        'onset': updated_annotations.onset,
        'duration': updated_annotations.duration,
        'trial_type': updated_annotations.description
    })
    
    # Guardar el DataFrame como TSV
    annotations_df.to_csv(output_path.fpath, sep='\t', index=False)
    
    print(f"Anotaciones guardadas en: {output_path.fpath}")
    
    # Crear el archivo JSON asociado
    json_path = output_path.fpath.with_suffix('.json')
    
    # Intentar cargar el JSON original si existe
    original_json_path = events_path.fpath.with_suffix('.json')
    original_json = {}
    if original_json_path.exists():
        try:
            with open(original_json_path, 'r') as f:
                original_json = json.load(f)
        except json.JSONDecodeError:
            print(f"Error al leer el archivo JSON original: {original_json_path}")
    
    # Crear el contenido del JSON con los campos requeridos
    # Asegurarse de que el formato cumple con el esquema BIDS
    json_content = {
        # Mantener los campos estándar de columnas si existen en el original
        "onset": original_json.get("onset", {"Description": "Event onset in seconds"}),
        "duration": original_json.get("duration", {"Description": "Event duration in seconds"}),
        "trial_type": original_json.get("trial_type", {"Description": "Event description/type"}),
        
        # Añadir los metadatos como objetos para cumplir con el esquema BIDS
        "SidecarDescription": {"Description": "Eventos alineados con señales fisiológicas"},
        "SourceData": {"Sources": [str(events_path.fpath.relative_to(repo_root)).replace('\\', '/')]},
        "ProcessingMethod": {"OffsetApplied": "manual AUDIO/PHOTO (s)"},
        "GeneratedBy": {
            "Name": "visualize_events.py",
            "Version": "1.0",
            "Description": "Alineación manual de eventos AUDIO/PHOTO"
        },
        "MetadataDate": {"DateCreated": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")}
    }
    
    # Guardar el JSON correctamente
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)
    
    # Crear dataset_description.json si no existe
    create_dataset_description(aligned_root, events_path)
    
    return str(output_path.fpath)


def create_dataset_description(aligned_root, events_path):
    """
    Crea el archivo dataset_description.json para la carpeta de derivados.
    
    Parameters
    ----------
    aligned_root : pathlib.Path
        Ruta a la carpeta de derivados
    events_path : mne_bids.BIDSPath
        Ruta BIDS del archivo de eventos original
    """
    dataset_desc_path = aligned_root / 'dataset_description.json'
    
    if not dataset_desc_path.exists():
        print(f"Creando dataset_description.json en {aligned_root}")
        
        # Crear el README si no existe (requerido por BIDS)
        readme_path = aligned_root / 'README'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# Aligned Events\n\n")
                f.write("Este directorio contiene eventos alineados manualmente con señales fisiológicas.\n")
                f.write("Los eventos fueron alineados usando el script visualize_events.py.\n")
        
        # Crear manualmente el dataset_description.json para asegurar el formato correcto
        dataset_desc = {
            "Name": "aligned_events",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "Authors": [
                "D'Amelio, Tomás Ariel",
                "..."
            ],
            "GeneratedBy": [{
                "Name": "visualize_events.py",
                "Version": "1.0",
                "Description": "Alineación manual de eventos AUDIO/PHOTO"
            }],
            "SourceDatasets": [{
                "URL": "file:///../../raw"  # URL en formato URI válido
            }]
        }
        
        # Guardar el JSON correctamente
        with open(dataset_desc_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_desc, f, indent=4)
        
        print(f"Archivo dataset_description.json creado en: {dataset_desc_path}")


def main():
    """Función principal del script."""
    args = parse_args()
    
    try:
        # Cargar datos
        raw, events_df, bids_path, events_path = load_events_and_raw(
            args.subject, args.session, args.task, args.run, args.acq, args.events_dir, args.events_desc
        )
        
        # Visualizar canales con anotaciones y obtener anotaciones actualizadas
        updated_annotations, has_changes = visualize_channels_with_annotations(
            raw, events_df, apply_zscore=not args.no_zscore, read_only=args.read_only
        )
        
        # En modo solo lectura, no procesar guardado
        if args.read_only:
            print("\n=== Visualización completada (modo solo lectura) ===")
            print("\nNo se realizaron modificaciones en modo solo lectura.")
        else:
            # Verificar si las anotaciones han cambiado
            if has_changes:
                print("\n¡Se detectaron cambios en las anotaciones!")
                
                # Si se especificó --force-save, guardar sin preguntar
                if args.force_save:
                    saved_path = save_annotations_aligned(
                        updated_annotations, events_path,
                        args.subject, args.session, args.task, args.run, args.acq,
                        save_dir=args.save_dir
                    )
                    print(f"\nAnotaciones guardadas exitosamente en: {saved_path}")
                else:
                    # Preguntar al usuario si desea guardar los cambios
                    while True:
                        response = input("\n¿Deseas guardar las anotaciones actualizadas? (yes/no): ").strip().lower()
                        if response in ['yes', 'y', 'si', 's']:
                            saved_path = save_annotations_aligned(
                                updated_annotations, events_path,
                                args.subject, args.session, args.task, args.run, args.acq,
                                save_dir=args.save_dir
                            )
                            print(f"\nAnotaciones guardadas exitosamente en: {saved_path}")
                            break
                        elif response in ['no', 'n']:
                            print("\nLos cambios en las anotaciones NO han sido guardados.")
                            break
                        else:
                            print("Por favor, responde 'yes' o 'no'.")
            else:
                print("\nNo se detectaron cambios en las anotaciones.")
                
                # Preguntar si se quiere forzar el guardado aunque no haya cambios
                if not args.force_save:
                    response = input("\n¿Deseas guardar las anotaciones de todos modos? (yes/no): ").strip().lower()
                    if response in ['yes', 'y', 'si', 's']:
                        saved_path = save_annotations_aligned(
                            updated_annotations, events_path,
                            args.subject, args.session, args.task, args.run, args.acq,
                            save_dir=args.save_dir
                        )
                        print(f"\nAnotaciones guardadas exitosamente en: {saved_path}")
                elif args.force_save:
                    saved_path = save_annotations_aligned(
                        updated_annotations, events_path,
                        args.subject, args.session, args.task, args.run, args.acq,
                        save_dir=args.save_dir
                    )
                    print(f"\nAnotaciones guardadas exitosamente en: {saved_path}")
            
            print("\n=== Proceso completado exitosamente ===")
            print("\nPróximos pasos:")
            print("1. Revisar las anotaciones guardadas si se realizaron cambios")
            print("2. Ejecutar scripts posteriores para análisis basados en estas anotaciones")
            print("3. Validar la estructura BIDS con 'bids-validator data/derivatives/aligned_events'")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 