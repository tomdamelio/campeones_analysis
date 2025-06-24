# Standard library imports
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import cast

# Third-party imports
import mne
import numpy as np
import pandas as pd
import pyxdf
from mne.io.constants import FIFF
from mne_bids import BIDSPath, write_raw_bids
from mnelab.io.xdf import read_raw_xdf

# Local imports
from ..utils.bids_compliance import (
    create_bids_metadata,
)
from ..utils.preprocessing_helpers import (
    make_joystick_mapping,
    set_chs_montage,
)

# Configuración del logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configura el logger solo si no está ya configurado
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Nivel por defecto, se puede cambiar mediante argumento

# Add project root to Python path for imports when running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))



def read_and_process_xdf(subject, session, task, run, acq="b", xdf_path=None, debug=False):
    """
    Read and process XDF files, converting them to BIDS format.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier (e.g., "VR")
    task : str
        Task identifier
    run : str
        Run identifier
    acq : str, optional
        Acquisition identifier, by default "a"
    xdf_path : Path or str, optional
        Direct path to the XDF file. If provided, other path parameters are ignored.
    debug : bool, optional
        If True, sets logging level to DEBUG for more verbose output

    Returns
    -------
    mne.io.Raw
        Processed raw data object
    """
    # Variable para almacenar eventos para añadir más tarde
    saved_events = None
    
    # Ajustar nivel de logging según el parámetro debug
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo debug activado")
    else:
        logger.setLevel(logging.INFO)
        
    # Get project root (two levels up from current file)
    project_root = Path(__file__).resolve().parents[3]
    logger.info(f"Directorio raíz del proyecto: {project_root}")

    # Configuration of paths using pathlib
    data_folder = project_root / "data" / "sourcedata" / "xdf"
    bids_root = project_root / "data" / "raw"
    
    # Asegurar que las carpetas de datos existen
    data_folder.mkdir(parents=True, exist_ok=True)
    bids_root.mkdir(parents=True, exist_ok=True)

    # Create or update BIDS metadata files
    create_bids_metadata(bids_root)

    # Update participant information (optional)
    #update_participant_info(
    #    bids_root,
    #    subject,
    #    birth_date=None,
    #    age=None,
    #    gender=None,
    #    handedness=None,
    #    education_level=None,
    #    university_career=None,
    #    nationality=None,
    #    ethnicity=None,
    #    residence_location=None,
    #    vr_experience_count=None,
    #    medical_conditions_current=None,
    #    medical_conditions_past=None,
    #    psychological_diagnosis=None,
    #    current_treatment=None,
    #)

    # Build path for original file (non-BIDS) if not provided
    if xdf_path is None:
        xdf_path = (
            data_folder
            / f"sub-{subject}"
            / f"ses-{session}"
            / "physio"
            / f"sub-{subject}_ses-{session}_day-{acq}_task-{task}_run-{run}_eeg.xdf"
        )
        # Ensure xdf_path is a Path object if it's a string
    elif isinstance(xdf_path, str):
            xdf_path = Path(xdf_path)
            
    logger.info("\nBuscando archivo en:")
    logger.info(f"Path relativo: {xdf_path}")
    logger.info(f"Path absoluto: {xdf_path.absolute()}")

    # Verify if file exists
    if not xdf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {xdf_path}")

    logger.info(f"\nProcesando archivo: {xdf_path}")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    
    # Print detailed information about the XDF file structure
    logger.debug("\n===== ESTRUCTURA COMPLETA DEL ARCHIVO XDF =====")
    logger.debug(f"Número total de streams: {len(streams)}")
    logger.debug(f"Información del header: {header}")
    
    # Examine each stream in detail
    for i, stream in enumerate(streams):
        logger.debug(f"\n----- Stream {i+1} -----")
        logger.debug(f"Tipo: {stream['info']['type'][0] if 'type' in stream['info'] else 'No especificado'}")
        logger.debug(f"Nombre: {stream['info']['name'][0] if 'name' in stream['info'] else 'No especificado'}")
        logger.debug(f"Stream ID: {stream['info']['stream_id'] if 'stream_id' in stream['info'] else 'No especificado'}")
        
        # Print stream info structure - muy detallado, a nivel debug
        logger.debug("\nEstructura de 'info':")
        for key in stream['info']:
            logger.debug(f"  - {key}: {stream['info'][key]}")
        
        # Print time series information - muy detallado, a nivel debug
        logger.debug("\nInformación de time_series:")
        logger.debug(f"  - Tipo: {type(stream['time_series'])}")
        logger.debug(f"  - Longitud: {len(stream['time_series'])}")
        
        if len(stream['time_series']) > 0:
            logger.debug("  - Primeros 5 elementos:")
            for j in range(min(5, len(stream['time_series']))):
                logger.debug(f"    {j}: {stream['time_series'][j]}")
        
        # Print timestamp information - muy detallado, a nivel debug
        logger.debug("\nInformación de time_stamps:")
        logger.debug(f"  - Tipo: {type(stream['time_stamps'])}")
        logger.debug(f"  - Longitud: {len(stream['time_stamps'])}")
        
        if len(stream['time_stamps']) > 0:
            logger.debug("  - Primeros 5 timestamps:")
            for j in range(min(5, len(stream['time_stamps']))):
                logger.debug(f"    {j}: {stream['time_stamps'][j]}")
        
        # Check for other keys in the stream - muy detallado, a nivel debug
        logger.debug("\nOtras claves en el stream:")
        for key in stream:
            if key not in ['info', 'time_series', 'time_stamps']:
                logger.debug(f"  - {key}: {type(stream[key])}, longitud: {len(stream[key]) if hasattr(stream[key], '__len__') else 'N/A'}")
                
                # If it's a list with elements, show the first few
                if isinstance(stream[key], list) and len(stream[key]) > 0:
                    logger.debug("    Primeros 5 elementos:")
                    for j in range(min(5, len(stream[key]))):
                        logger.debug(f"      {j}: {stream[key][j]}")
    
    logger.debug("\n=================================================")

    # Process and save events if they exist
    # Mensaje claro sobre filtrado de marcadores
    logger.info("\n=============================================")
    logger.info("IMPORTANTE: Filtrando marcas con problemas de sincronización")
    logger.info("Se preservarán solo las marcas dentro del rango válido del EEG")
    logger.info("=============================================\n")
    
    # Comentando todo el procesamiento de marcadores
    """
    for stream in streams:
        if stream["info"]["type"][0] == "Markers":
    """
    # Establecer saved_events como None para que no se intente añadir annotaciones
    saved_events = None

    # Initialize raw as None
    raw = None
    eeg_stream = None

    # First find EEG stream and show all streams
    logger.info("\nStreams encontrados:")
    for i, stream in enumerate(streams):
        logger.info(f"\nStream {i + 1}:")
        logger.info(f"Tipo: {stream['info']['type'][0]}")
        logger.info(f"Nombre: {stream['info']['name'][0]}")
        logger.info(f"Stream ID: {stream['info']['stream_id']}")
        if stream["info"]["type"][0] == "EEG":
            eeg_stream = stream
            logger.info("¡Este es el stream de EEG!")
        if stream["info"]["name"][0] == "T.16000MAxes":
            logger.info("\nInformación de los ejes del joystick (X e Y):")
            logger.info(f"Número total de muestras: {len(stream['time_series'])}")
            srate_joystick = float(stream["info"]["nominal_srate"][0])
            logger.info(f"Frecuencia de muestreo: {srate_joystick} Hz")

            # Detailed joystick timing analysis
            logger.info("\nAnálisis detallado de timing del joystick:")
            logger.info(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
            logger.info(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
            duration_joystick = stream["time_stamps"][-1] - stream["time_stamps"][0]
            logger.info(f"Duración total: {duration_joystick:.2f} segundos")
            logger.info(
                f"Intervalo entre muestras promedio: {duration_joystick / len(stream['time_stamps']):.6f} segundos"
            )

            # Calculate and show joystick duration in more readable format
            hours_joystick = int(duration_joystick // 3600)
            minutes_joystick = int((duration_joystick % 3600) // 60)
            seconds_joystick = duration_joystick % 60
            logger.info(
                f"Duración total: {hours_joystick:02d}:{minutes_joystick:02d}:{seconds_joystick:05.2f} (HH:MM:SS)"
            )

            # Analyze joystick data
            np.array(stream["time_series"])
            time_stamps = np.array(stream["time_stamps"])
            _ = time_stamps - time_stamps[0]  # Calculate relative time

        if stream["info"]["type"][0] == "EEG":
            logger.info("\nInformación del stream de EEG:")
            logger.info(f"Número total de muestras: {len(stream['time_series'])}")
            srate_eeg = float(stream["info"]["nominal_srate"][0])
            logger.info(f"Frecuencia de muestreo: {srate_eeg} Hz")
            logger.info(f"Número de canales: {stream['info']['channel_count'][0]}")

            # Detailed EEG timing analysis
            logger.info("\nAnálisis detallado de timing del EEG:")
            logger.info(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
            logger.info(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
            duration = stream["time_stamps"][-1] - stream["time_stamps"][0]
            logger.info(f"Duración total: {duration:.2f} segundos")
            logger.info(
                f"Intervalo entre muestras promedio: {duration / len(stream['time_stamps']):.6f} segundos"
            )

            # Calculate and show duration in more readable format
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            logger.info(
                f"Duración total: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)"
            )

    # If we found EEG, process it
    if eeg_stream is None:
        raise ValueError("No se encontró stream de EEG en el archivo")

    logger.info("\nProcesando stream EEG...")
    raw_result = read_raw_xdf(
        str(xdf_path), stream_ids=[eeg_stream["info"]["stream_id"]], preload=True
    )
    # Cast raw to the correct type to avoid type errors
    raw = cast(mne.io.Raw, raw_result)
    logger.info("Info del raw EEG:")
    logger.info(raw.info)

    # Añadir las annotations guardadas si existen
    """
    if saved_events is not None:
        logger.info("\n=== AÑADIENDO ANNOTATIONS AL RAW ===")
        logger.info(f"Tiempo de inicio del stream de eventos: {saved_events['stream_start_time']:.4f}")
        
        # Obtener el tiempo absoluto de inicio del EEG
        eeg_abs_start = eeg_stream["time_stamps"][0]
        eeg_abs_end = eeg_stream["time_stamps"][-1]
        logger.info(f"Tiempo absoluto de inicio del EEG: {eeg_abs_start:.4f}")
        logger.info(f"Tiempo absoluto de fin del EEG: {eeg_abs_end:.4f}")
        logger.info(f"Duración absoluta del EEG: {eeg_abs_end - eeg_abs_start:.4f} segundos")
        
        # Ajustar los timestamps relativos al inicio absoluto del EEG
        logger.info(f"Ajustando timestamps relativos al inicio absoluto del EEG")
        
        # Mostrar los onsets originales
        logger.info("\nOnsets originales:")
        for i, (onset, desc) in enumerate(zip(saved_events["onsets"], saved_events["descriptions"])):
            logger.info(f"{i}: '{desc}' en tiempo {onset:.4f}")
            # Verificar si el onset está dentro del rango del EEG
            if onset < eeg_abs_start or onset > eeg_abs_end:
                logger.warning(f"¡ADVERTENCIA! El marcador '{desc}' con tiempo {onset:.4f} está fuera del rango del EEG ({eeg_abs_start:.4f} - {eeg_abs_end:.4f})")
                if onset < eeg_abs_start:
                    logger.warning(f"   El marcador ocurre {eeg_abs_start - onset:.4f} segundos ANTES del inicio del EEG")
                else:
                    logger.warning(f"   El marcador ocurre {onset - eeg_abs_end:.4f} segundos DESPUÉS del fin del EEG")
        
        # Calcular los onsets ajustados usando el tiempo absoluto del EEG
        adjusted_onsets = [t - eeg_abs_start for t in saved_events["onsets"]]
        
        # Verificar si los onsets ajustados están dentro del rango válido para MNE (0 a duración)
        eeg_duration = raw.times[-1]
        logger.info(f"Duración del Raw EEG en MNE: {eeg_duration:.4f} segundos")
        
        valid_indices = []
        for i, (onset, desc) in enumerate(zip(adjusted_onsets, saved_events["descriptions"])):
            if 0 <= onset <= eeg_duration:
                valid_indices.append(i)
                logger.info(f"{i}: '{desc}' en tiempo {onset:.4f} (DENTRO del rango válido)")
            else:
                logger.warning(f"{i}: '{desc}' en tiempo {onset:.4f} (FUERA del rango válido 0-{eeg_duration:.4f})")
        
        # Usar solo los marcadores válidos
        if len(valid_indices) < len(adjusted_onsets):
            logger.warning(f"Se omitirán {len(adjusted_onsets) - len(valid_indices)} marcadores fuera del rango válido")
            filtered_onsets = [adjusted_onsets[i] for i in valid_indices]
            filtered_durations = [saved_events["durations"][i] for i in valid_indices]
            filtered_descriptions = [saved_events["descriptions"][i] for i in valid_indices]
        else:
            filtered_onsets = adjusted_onsets
            filtered_durations = saved_events["durations"]
            filtered_descriptions = saved_events["descriptions"]
        
        logger.info("\nOnsets ajustados finales que se usarán:")
        for i, (onset, desc) in enumerate(zip(filtered_onsets, filtered_descriptions)):
            logger.info(f"{i}: '{desc}' en tiempo {onset:.4f}")
        
        # Crear y añadir annotations
        annot = mne.Annotations(
            onset=filtered_onsets,
            duration=filtered_durations,
            description=filtered_descriptions
        )
        raw.set_annotations(annot)
        logger.info(f"Se añadieron {len(filtered_onsets)} annotations al objeto raw")
    """
    logger.info("\nSe utilizarán solo las anotaciones dentro del rango válido del EEG")

    # Resample EEG to 250 Hz if necessary
    target_sfreq = 250
    if raw.info["sfreq"] != target_sfreq:
        logger.info(f"\nResampleando EEG de {raw.info['sfreq']} Hz a {target_sfreq} Hz...")
        raw = raw.resample(target_sfreq, npad="auto")

    # Process joystick stream if it exists
    joystick_raw = None
    for stream in streams:
        if stream["info"]["name"][0] == "T.16000MAxes":
            logger.info("\nProcesando joystick...")
            joystick_data = np.array(stream["time_series"])
            joystick_timestamps = stream["time_stamps"]
            
            # Calcular duración del EEG y joystick
            eeg_duration = raw.times[-1]
            joystick_duration = joystick_timestamps[-1] - joystick_timestamps[0]
            
            logger.info(f"Duración EEG: {eeg_duration:.2f}s, Duración Joystick: {joystick_duration:.2f}s")
            
            # Verificar si hay una diferencia significativa en la duración
            if abs(eeg_duration - joystick_duration) > 1.0:  # Diferencia mayor a 1 segundo
                logger.warning(f"ADVERTENCIA: Diferencia significativa en la duración entre EEG y joystick: {abs(eeg_duration - joystick_duration):.2f}s")
                
                # Determinar qué datos son más cortos
                if eeg_duration < joystick_duration:
                    logger.warning("Recortando datos del joystick para que coincidan con la duración del EEG")
                    # Encontrar el índice donde el joystick alcanza la duración del EEG
                    end_idx = np.searchsorted(joystick_timestamps - joystick_timestamps[0], eeg_duration)
                    joystick_data = joystick_data[:end_idx]
                    joystick_timestamps = joystick_timestamps[:end_idx]
                else:
                    logger.warning("El EEG es más largo que el joystick, se rellenarán los datos faltantes del joystick")
            
            # Crear el objeto Raw para el joystick, remuestreado a la frecuencia del EEG
            joystick_raw_result = make_joystick_mapping(
                joystick_data, joystick_timestamps, raw.info["sfreq"]
            )
            # Cast joystick_raw to the correct type
            joystick_raw = cast(mne.io.Raw, joystick_raw_result)
            
            # Verificar que las dimensiones coincidan
            eeg_samples = len(raw.times)
            joystick_samples = len(joystick_raw.times)
            
            logger.info(f"Muestras EEG: {eeg_samples}, Muestras Joystick: {joystick_samples}")
            
            # Si todavía hay diferencia en el número de muestras, ajustar
            if eeg_samples != joystick_samples:
                logger.info(f"Ajustando longitud de datos: diferencia de {abs(eeg_samples - joystick_samples)} muestras")
                
                if eeg_samples > joystick_samples:
                    # Rellenar el joystick con valores constantes al final
                    pad_length = eeg_samples - joystick_samples
                    pad_data = np.zeros((2, pad_length))
                    # Usar los últimos valores para el relleno
                    pad_data[0, :] = joystick_raw.get_data()[0, -1]
                    pad_data[1, :] = joystick_raw.get_data()[1, -1]
                    
                    # Concatenar los datos originales con el relleno
                    new_data = np.hstack((joystick_raw.get_data(), pad_data))
                    
                    # Crear un nuevo objeto Raw con los datos ajustados
                    info = mne.create_info(
                        ch_names=['joystick_x', 'joystick_y'],
                        sfreq=raw.info["sfreq"],
                        ch_types=['misc', 'misc']
                    )
                    joystick_raw = mne.io.RawArray(new_data, info)
                    
                    logger.info(f"Joystick rellenado con {pad_length} muestras adicionales")
                else:
                    # Recortar el joystick
                    logger.info(f"Recortando {joystick_samples - eeg_samples} muestras del joystick")
                    joystick_raw = joystick_raw.crop(tmax=raw.times[-1])
            
            logger.info(f"Dimensiones finales - EEG: {len(raw.times)}, Joystick: {len(joystick_raw.times)}")

    # If joystick data was found, add it to the main Raw object
    if joystick_raw is not None:
        logger.info("\nAgregando canales de joystick al objeto Raw...")
        try:
            raw.add_channels([joystick_raw])
            logger.info("Canales de joystick agregados correctamente")
        except ValueError as e:
            logger.error(f"Error al agregar canales de joystick: {e}")
            logger.info("Intentando ajustar dimensiones...")
            
            # Último intento de ajuste si las dimensiones aún no coinciden
            if len(raw.times) != len(joystick_raw.times):
                min_length = min(len(raw.times), len(joystick_raw.times))
                logger.info(f"Recortando ambos conjuntos de datos a {min_length} muestras")
                
                # Recortar EEG
                if len(raw.times) > min_length:
                    raw = raw.crop(tmax=(min_length - 1) / raw.info['sfreq'])
                
                # Recortar joystick
                if len(joystick_raw.times) > min_length:
                    joystick_raw = joystick_raw.crop(tmax=(min_length - 1) / joystick_raw.info['sfreq'])
                
                # Intentar agregar de nuevo
                try:
                    raw.add_channels([joystick_raw])
                    logger.info("Canales de joystick agregados correctamente después del ajuste")
                except ValueError as e2:
                    logger.error(f"Error persistente al agregar canales de joystick: {e2}")
                    logger.info("Continuando sin datos de joystick")

    # Apply EEG channel montage
    logger.info("\nAplicando montaje de canales EEG...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        raw = set_chs_montage(raw)

    # Document channel mapping and units
    logger.info("\nDocumentando información de canales...")
    channels_info = pd.DataFrame(
        {
            "name": raw.ch_names,
            "type": [raw.info["chs"][i]["kind"] for i in range(len(raw.ch_names))],
            "units": [
                raw.info["chs"][i].get("unit", "n/a") for i in range(len(raw.ch_names))
            ],
            "sampling_frequency": [raw.info["sfreq"]] * len(raw.ch_names),
            "low_cutoff": [raw.info["highpass"]] * len(raw.ch_names),
            "high_cutoff": [raw.info["lowpass"]] * len(raw.ch_names),
        }
    )

    # Corregir unidades para canales misc/aux
    logger.info("\n=== UNIDADES DE CANALES ANTES DE CAMBIOS ===")
    for i, ch in enumerate(raw.info["chs"]):
        ch_name = raw.ch_names[i]
        ch_type = "EEG" if ch["kind"] == FIFF.FIFFV_EEG_CH else "MISC" if ch["kind"] == FIFF.FIFFV_MISC_CH else "OTHER"
        unit = ch.get("unit", "None")
        unit_mul = ch.get("unit_mul", "None")
        logger.info(f"{ch_name} ({ch_type}): unidad={unit}, multiplicador={unit_mul}")
    
    # Aplicar cambios a las unidades
    logger.info("\n=== APLICANDO CAMBIOS A UNIDADES ===")
    for i, ch in enumerate(raw.info["chs"]):
        ch_name = raw.ch_names[i]
        original_unit = ch.get("unit")
        
        # Verificar si es un canal misc o si es GSR (que tiene unidad S)
        if ch["kind"] == FIFF.FIFFV_MISC_CH or ch_name == "GSR" or ch_name == "RESP":
            # Cambiar todas las unidades no soportadas a voltios (µV)
            # Esto es necesario porque el formato BrainVision solo admite µV
            # La información real de las unidades se guardará en el archivo JSON
            logger.info(f"Cambiando unidad de {ch_name} de {original_unit} a FIFF_UNIT_V con multiplicador -6 (µV)")
            ch["unit"] = FIFF.FIFF_UNIT_V  # Usar voltios como unidad base
            ch["unit_mul"] = -6  # Multiplicador para µV (-6 = 10^-6 = µV)
    
    # Verificar cambios
    logger.info("\n=== UNIDADES DE CANALES DESPUÉS DE CAMBIOS ===")
    for i, ch in enumerate(raw.info["chs"]):
        ch_name = raw.ch_names[i]
        ch_type = "EEG" if ch["kind"] == FIFF.FIFFV_EEG_CH else "MISC" if ch["kind"] == FIFF.FIFFV_MISC_CH else "OTHER"
        unit = ch.get("unit", "None")
        unit_mul = ch.get("unit_mul", "None")
        logger.info(f"{ch_name} ({ch_type}): unidad={unit}, multiplicador={unit_mul}")

    # Create BIDSPath for EEG
    bids_path = BIDSPath(
        subject=subject,
        session=session.lower(),  # Use lowercase for BIDS
        task=task,
        run=run,
        acquisition=acq,
        root=bids_root,
        datatype="eeg",
        extension=".vhdr",
        check=True,
    )

    # Create necessary directories
    Path(bids_path.directory).mkdir(parents=True, exist_ok=True)

    # Save channels.tsv
    channels_path = bids_path.copy().update(suffix="channels", extension=".tsv")
    channels_info.to_csv(channels_path, sep="\t", index=False)
    logger.info(f"Información de canales guardada en: {channels_path}")

    # Create sidecar JSON for channels
    channels_json_path = str(channels_path.fpath).replace('.tsv', '.json')
    channels_json = {
        "units": {
            "description": "Las unidades reales de los canales. Nota: Todos los canales se almacenan en µV en el archivo BrainVision por compatibilidad, pero aquí se documenta la unidad real."
        }
    }
    
    # Documentar unidades reales para todos los canales
    for ch_name in raw.ch_names:
        ch_idx = raw.ch_names.index(ch_name)
        ch = raw.info["chs"][ch_idx]
        
        # Asignar unidades reales según el tipo de canal
        if ch["kind"] == FIFF.FIFFV_EEG_CH:
            channels_json["units"][ch_name] = "µV"
        elif ch["kind"] == FIFF.FIFFV_MISC_CH:
            if 'joystick' in ch_name:
                channels_json["units"][ch_name] = "arbitrary"
            elif ch_name == "GSR":
                channels_json["units"][ch_name] = "µS"  # Microsiemens
            elif ch_name == "RESP":
                channels_json["units"][ch_name] = "Ω"   # Ohms para respiración
            elif ch_name == "TRIG":
                channels_json["units"][ch_name] = "boolean"
            else:
                channels_json["units"][ch_name] = "arbitrary"
        elif ch["kind"] == FIFF.FIFFV_ECG_CH:
            channels_json["units"][ch_name] = "mV"
        elif ch["kind"] == FIFF.FIFFV_EOG_CH:
            channels_json["units"][ch_name] = "µV"
        else:
            channels_json["units"][ch_name] = "unknown"
    
    # Guardar el sidecar JSON
    with open(channels_json_path, 'w') as f:
        json.dump(channels_json, f, indent=4)
    logger.info(f"Metadata de canales guardada en: {channels_json_path}")

    # Save EEG data in BIDS format
    logger.info("\nGuardando datos EEG en formato BIDS...")
    
    # En lugar de eliminar todas las anotaciones, filtrar solo las que están fuera de rango
    if raw.annotations and len(raw.annotations) > 0:
        logger.info(f"El objeto raw contiene {len(raw.annotations)} anotaciones")
        
        # Verificar si hay anotaciones fuera de rango
        eeg_duration = raw.times[-1]
        valid_mask = (raw.annotations.onset >= 0) & (raw.annotations.onset <= eeg_duration)
        invalid_count = (~valid_mask).sum()
        
        # Crear un resumen de las anotaciones originales
        logger.info("\n=== DETALLE DE ANOTACIONES ORIGINALES ===")
        logger.info(f"{'Índice':<7} | {'Tiempo (s)':<10} | {'Descripción':<50} | {'Estado':<10}")
        logger.info(f"{'-'*7:<7} | {'-'*10:<10} | {'-'*50:<50} | {'-'*10:<10}")
        
        for i, (onset, duration, desc) in enumerate(zip(raw.annotations.onset, 
                                                     raw.annotations.duration, 
                                                     raw.annotations.description)):
            status = "VÁLIDA" if valid_mask[i] else "FUERA RANGO"
            logger.info(f"{i:<7} | {onset:<10.2f} | {desc:<50} | {status:<10}")
        
        if invalid_count > 0:
            logger.info(f"\nFiltrando {invalid_count} anotaciones fuera del rango válido (0-{eeg_duration:.2f}s)")
            # Crear nuevas anotaciones solo con las válidas
            new_annotations = mne.Annotations(
                onset=raw.annotations.onset[valid_mask],
                duration=raw.annotations.duration[valid_mask],
                description=raw.annotations.description[valid_mask]
            )
            raw.set_annotations(new_annotations)
            logger.info(f"Se conservaron {len(new_annotations)} anotaciones dentro del rango válido")
        else:
            logger.info(f"\nTodas las anotaciones ({len(raw.annotations)}) están dentro del rango válido")
        
        # Mostrar resumen detallado de las anotaciones conservadas
        logger.info("\n=== ANOTACIONES CONSERVADAS PARA EL ARCHIVO BIDS ===")
        logger.info(f"{'Índice':<7} | {'Tiempo (s)':<10} | {'Descripción':<50}")
        logger.info(f"{'-'*7:<7} | {'-'*10:<10} | {'-'*50:<50}")
        
        for i, (onset, duration, desc) in enumerate(zip(raw.annotations.onset, 
                                                     raw.annotations.duration, 
                                                     raw.annotations.description)):
            logger.info(f"{i:<7} | {onset:<10.2f} | {desc:<50}")
        
        # Agrupar por tipo de anotación para mostrar un resumen
        annotation_types = {}
        for desc in raw.annotations.description:
            desc_str = str(desc)
            if desc_str not in annotation_types:
                annotation_types[desc_str] = 0
            annotation_types[desc_str] += 1
        
        logger.info("\n=== RESUMEN POR TIPO DE ANOTACIÓN ===")
        logger.info(f"{'Tipo de anotación':<50} | {'Cantidad':<10}")
        logger.info(f"{'-'*50:<50} | {'-'*10:<10}")
        for desc_type, count in annotation_types.items():
            logger.info(f"{desc_type:<50} | {count:<10}")
    
    # No convertir eventos a formato BrainVision
    """
    # Convert events to BrainVision format before saving
    events_array = None
    event_id = None
    if raw.annotations:
        events, event_id = mne.events_from_annotations(raw)
        if len(events) > 0:
            events_array = events
    """
    
    # Usar eventos y event_id del raw actual si existen
    events_array = None
    event_id = None
    if raw.annotations and len(raw.annotations) > 0:
        try:
            events, event_id = mne.events_from_annotations(raw)
            if len(events) > 0:
                events_array = events
                logger.info(f"\nSe usarán {len(events)} eventos extraídos de las anotaciones")
                
                # Mostrar los detalles de los eventos generados
                if event_id:
                    logger.info("\n=== MAPEO DE EVENTOS ===")
                    logger.info(f"{'Descripción':<50} | {'Código de evento':<15}")
                    logger.info(f"{'-'*50:<50} | {'-'*15:<15}")
                    for desc, code in event_id.items():
                        logger.info(f"{desc:<50} | {code:<15}")
        except Exception as e:
            logger.warning(f"Error al convertir anotaciones a eventos: {e}")
            # Asegurar que events_array y event_id sean None en caso de error
            events_array = None
            event_id = None

    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        events=events_array,
        event_id=event_id,
    )

    logger.info("\nProcesamiento BIDS completado exitosamente.")
    return raw


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process XDF files and convert to BIDS format. "
        "Automatically skips files that have already been processed unless --force is used."
    )
    parser.add_argument(
        "--subject", type=str, help="Process specific subject (e.g., '01')"
    )
    parser.add_argument(
        "--session", type=str, help="Process specific session (e.g., 'VR')"
    )
    parser.add_argument(
        "--test", action="store_true", help="Process only the first XDF file found (for testing)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for detailed output"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue processing other files if one fails"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing of files that already exist in BIDS format"
    )
    args = parser.parse_args()

    # Configure logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modo debug activado")
    else:
        logger.setLevel(logging.INFO)

    # Get project root
    project_root = Path(__file__).resolve().parents[3]
    data_folder = project_root / "data" / "sourcedata" / "xdf"
    bids_root = project_root / "data" / "raw"

    # Function to check if a file has already been processed
    def is_already_processed(subject, session, task, run, acq):
        """
        Check if an XDF file has already been processed by looking for the corresponding BIDS files.
        
        Parameters
        ----------
        subject : str
            Subject identifier
        session : str  
            Session identifier
        task : str
            Task identifier
        run : str
            Run identifier
        acq : str
            Acquisition identifier
            
        Returns
        -------
        bool
            True if the file has already been processed, False otherwise
        """
        try:
            # Create BIDSPath to check for existing files
            bids_path = BIDSPath(
                subject=subject,
                session=session.lower(),  # Use lowercase for BIDS
                task=task,
                run=run,
                acquisition=acq,
                root=bids_root,
                datatype="eeg",
                extension=".vhdr",
                check=False  # Don't check validity, just check existence
            )
            
            # Check if the main files exist (.vhdr, .eeg, .vmrk)
            vhdr_file = bids_path.fpath
            eeg_file = vhdr_file.with_suffix('.eeg')
            vmrk_file = vhdr_file.with_suffix('.vmrk')
            
            # All three files must exist for it to be considered processed
            files_exist = vhdr_file.exists() and eeg_file.exists() and vmrk_file.exists()
            
            if files_exist:
                logger.debug(f"BIDS files already exist for sub-{subject}_ses-{session}_task-{task}_run-{run}_acq-{acq}")
                return True
            else:
                logger.debug(f"BIDS files do not exist for sub-{subject}_ses-{session}_task-{task}_run-{run}_acq-{acq}")
                return False
                
        except Exception as e:
            logger.debug(f"Error checking if file is already processed: {e}")
            return False

    # Function to find all available subjects
    def find_subjects():
        subjects = []
        for path in data_folder.glob("sub-*"):
            if path.is_dir():
                subject = path.name.replace("sub-", "")
                subjects.append(subject)
        return sorted(subjects)

    # Function to find sessions for a subject
    def find_sessions(subject):
        sessions = []
        subject_dir = data_folder / f"sub-{subject}"
        for path in subject_dir.glob("ses-*"):
            if path.is_dir():
                session = path.name.replace("ses-", "")
                sessions.append(session)
        return sorted(sessions)

    # Function to find XDF files for a subject and session
    def find_xdf_files(subject, session):
        xdf_files = []
        session_dir = data_folder / f"sub-{subject}" / f"ses-{session}" / "physio"
        
        # Verificar si la carpeta existe
        if not session_dir.exists():
            logger.warning(f"La carpeta {session_dir} no existe. Creando estructura de directorios...")
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Carpeta creada. Por favor, coloque los archivos XDF en: {session_dir}")
            return []

        # Buscar archivos XDF
        pattern = f"sub-{subject}_ses-{session}_*.xdf"
        for path in session_dir.glob(pattern):
            xdf_files.append(path)
        
        if not xdf_files:
            logger.warning(f"No se encontraron archivos XDF en {session_dir} con el patrón: {pattern}")
        
        return sorted(xdf_files)

    # Process all subjects or a specific one
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = find_subjects()

    if not subjects:
        logger.error("No subjects found in the sourcedata directory.")
        sys.exit(1)

    logger.info(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Process each subject
    for subject in subjects:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing subject: {subject}")
        logger.info(f"{'=' * 80}")

        if args.session:
            sessions = [args.session]
        else:
            sessions = find_sessions(subject)

        if not sessions:
            logger.error(f"No sessions found for subject {subject}.")
            continue

        logger.info(f"Found {len(sessions)} sessions: {', '.join(sessions)}")

        # Process each session
        for session in sessions:
            logger.info(f"\n{'-' * 60}")
            logger.info(f"Processing session: {session}")
            logger.info(f"{'-' * 60}")

            xdf_files = find_xdf_files(subject, session)

            if not xdf_files:
                logger.error(f"No XDF files found for subject {subject}, session {session}.")
                continue

            logger.info(f"Found {len(xdf_files)} XDF files.")
            
            # If test mode, process only the first file
            if args.test:
                xdf_files = xdf_files[:1]
                logger.info("TEST MODE: Processing only the first XDF file.")

            # Process each XDF file
            processed_count = 0
            skipped_count = 0
            
            for xdf_file in xdf_files:
                filename = xdf_file.name
                logger.info(f"\nEvaluating file: {filename}")
                
                try:
                    # Extract parameters from filename
                    parts = filename.split("_")
                    
                    # Default values
                    task = "unknown"
                    run = "01"
                    acq = "a"
                    
                    # Extract parameters from filename parts
                    for part in parts:
                        if part.startswith("task-"):
                            task = part.replace("task-", "")
                        elif part.startswith("run-"):
                            run = part.replace("run-", "")
                        elif part.startswith("day-"):
                            acq = part.replace("day-", "")
                    
                    # Check if file has already been processed (unless force flag is used)
                    if not args.force and is_already_processed(subject, session, task, run, acq):
                        logger.info(f"⏭️  OMITIENDO: {filename} - Ya fue procesado")
                        logger.info(f"   Los archivos BIDS correspondientes ya existen")
                        logger.info(f"   Usa --force para reprocesar archivos existentes")
                        skipped_count += 1
                        continue
                    
                    # Process the file
                    if args.force and is_already_processed(subject, session, task, run, acq):
                        logger.info(f"🔄 REPROCESANDO: {filename} - Forzando reescritura")
                    else:
                        logger.info(f"🔄 PROCESANDO: {filename}")
                    
                    raw = read_and_process_xdf(subject, session, task, run, acq, xdf_path=xdf_file, debug=args.debug)
                    logger.info(f"✅ Successfully processed {filename}")
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    if not args.continue_on_error:
                        sys.exit(1)
            
            # Summary for this session
            logger.info(f"\n📊 RESUMEN SESIÓN {session}:")
            logger.info(f"   Archivos procesados: {processed_count}")
            logger.info(f"   Archivos omitidos: {skipped_count}")
            logger.info(f"   Total archivos encontrados: {len(xdf_files)}")
