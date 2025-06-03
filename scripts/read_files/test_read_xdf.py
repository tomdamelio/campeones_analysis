# %%
# Standard library imports
import json
import os
import sys
import warnings
from pathlib import Path

# Third-party imports
import mne
import numpy as np
import pandas as pd
import pyxdf
from mne_bids import BIDSPath, write_raw_bids
from mnelab.io.xdf import read_raw_xdf

from campeones_analysis.utils.bids_compliance import (
    create_bids_metadata,
    update_participant_info,
)

# Local imports
from campeones_analysis.utils.preprocessing_helpers import (
    correct_channel_types,
    make_joystick_mapping,
    set_chs_montage,
)

# %%


# %%
# Obtener el directorio raíz del proyecto (subiendo un nivel desde tests)
project_root = Path(os.getcwd()).parent
print(f"Directorio actual: {os.getcwd()}")
print(f"Directorio raíz del proyecto: {project_root}")

# Cambiar al directorio raíz del proyecto
os.chdir(project_root)
print(f"Directorio de trabajo cambiado a: {os.getcwd()}")

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(str(project_root))

# Configuración de paths usando pathlib
data_folder = project_root / "data" / "sourcedata" / "xdf"
bids_root = project_root / "data" / "raw"

# Crear o actualizar archivos de metadatos BIDS
create_bids_metadata(bids_root)

subject = "16"
session = "VR"  # Mantener mayúsculas para compatibilidad con archivos originales
task = "02"
run = "003"
acq = "a"  # Reemplazamos day por acq

# Actualizar información del participante (opcional)
update_participant_info(
    bids_root, subject, age=None, sex=None, hand=None, weight=None, height=None
)

# %%
# Construir el path para el archivo original (no BIDS)
xdf_path = (
    data_folder
    / f"sub-{subject}"
    / f"ses-{session}"
    / "physio"
    / f"sub-{subject}_ses-{session}_day-{acq}_task-{task}_run-{run}_eeg.xdf"
)
print("\nBuscando archivo en:")
print(f"Path relativo: {xdf_path}")
print(f"Path absoluto: {xdf_path.absolute()}")

# Verificar si el archivo existe
if not xdf_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {xdf_path}")

print(f"\nProcesando archivo: {xdf_path}")
streams, header = pyxdf.load_xdf(str(xdf_path))

# Inicializar raw como None
raw = None
eeg_stream = None

# %%
# Primero encontrar el stream de EEG y mostrar todos los streams
print("\nStreams encontrados:")
for i, stream in enumerate(streams):
    print(f"\nStream {i + 1}:")
    print(f"Tipo: {stream['info']['type'][0]}")
    print(f"Nombre: {stream['info']['name'][0]}")
    print(f"Stream ID: {stream['info']['stream_id']}")
    if stream["info"]["type"][0] == "EEG":
        eeg_stream = stream
        print("¡Este es el stream de EEG!")
    if stream["info"]["name"][0] == "T.16000MAxes":
        print("\nInformación de los ejes del joystick (X e Y):")
        print(f"Número total de muestras: {len(stream['time_series'])}")
        srate_joystick = float(stream["info"]["nominal_srate"][0])
        print(f"Frecuencia de muestreo: {srate_joystick} Hz")

        # Análisis detallado de timing del joystick
        print("\nAnálisis detallado de timing del joystick:")
        print(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
        print(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
        duration_joystick = stream["time_stamps"][-1] - stream["time_stamps"][0]
        print(f"Duración total: {duration_joystick:.2f} segundos")
        print(
            f"Intervalo entre muestras promedio: {duration_joystick / len(stream['time_stamps']):.6f} segundos"
        )

        # Calcular y mostrar la duración del joystick en formato más legible
        hours_joystick = int(duration_joystick // 3600)
        minutes_joystick = int((duration_joystick % 3600) // 60)
        seconds_joystick = duration_joystick % 60
        print(
            f"Duración total: {hours_joystick:02d}:{minutes_joystick:02d}:{seconds_joystick:05.2f} (HH:MM:SS)"
        )

        # Convertir a array numpy para análisis
        time_series_array = np.array(stream["time_series"])
        time_stamps = np.array(stream["time_stamps"])

        # Calcular tiempo relativo (desde 0 hasta duración total)
        tiempo_relativo = time_stamps - time_stamps[0]

        # Análisis estadístico de los canales X e Y
        canales = {"X": 0, "Y": 1}

    if stream["info"]["type"][0] == "EEG":
        print("\nInformación del stream de EEG:")
        print(f"Número total de muestras: {len(stream['time_series'])}")
        srate_eeg = float(stream["info"]["nominal_srate"][0])
        print(f"Frecuencia de muestreo: {srate_eeg} Hz")
        print(f"Número de canales: {stream['info']['channel_count'][0]}")

        # Análisis detallado de timing del EEG
        print("\nAnálisis detallado de timing del EEG:")
        print(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
        print(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
        duration = stream["time_stamps"][-1] - stream["time_stamps"][0]
        print(f"Duración total: {duration:.2f} segundos")
        print(
            f"Intervalo entre muestras promedio: {duration / len(stream['time_stamps']):.6f} segundos"
        )

        # Calcular y mostrar la duración en formato más legible
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        print(f"Duración total: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)")


# %%
# Si encontramos EEG, procesarlo
if eeg_stream is None:
    raise ValueError("No se encontró stream de EEG en el archivo")

print("\nProcesando stream EEG...")
raw = read_raw_xdf(
    str(xdf_path), stream_ids=[eeg_stream["info"]["stream_id"]], preload=True
)
print("Info del raw EEG:")
print(raw.info)

# Corregir tipos de canales
raw = correct_channel_types(raw)

# Resamplear EEG a 250 Hz si es necesario
target_sfreq = 250
if raw.info["sfreq"] != target_sfreq:
    print(f"\nResampleando EEG de {raw.info['sfreq']} Hz a {target_sfreq} Hz...")
    raw = raw.resample(target_sfreq, npad="auto")

# Procesar el stream de joystick si existe
for stream in streams:
    if stream["info"]["name"][0] == "T.16000MAxes":
        print("\nProcesando datos del joystick...")
        # Crear un objeto Raw para el joystick
        joystick_data = np.array(stream["time_series"]).T
        print(f"Forma de los datos del joystick: {joystick_data.shape}")

        # Inspeccionar los datos del joystick
        print("\nInformación del joystick:")
        print(f"Número de ejes: {joystick_data.shape[0]}")
        print(f"Número de muestras: {joystick_data.shape[1]}")
        joystick_sfreq = float(stream["info"]["nominal_srate"][0])
        print(f"Frecuencia de muestreo original: {joystick_sfreq} Hz")

        # Crear nombres de canales para todos los ejes
        n_channels = joystick_data.shape[0]
        ch_names = [f"joystick_axis_{i + 1}" for i in range(n_channels)]

        # Crear info para los canales del joystick
        joystick_info = mne.create_info(
            ch_names=ch_names, sfreq=joystick_sfreq, ch_types=["misc"] * n_channels
        )

        # Crear objeto Raw para el joystick
        joystick_raw = mne.io.RawArray(joystick_data, joystick_info)

        # Aplicar mapeo del joystick
        print("Aplicando mapeo del joystick...")
        joystick_raw = make_joystick_mapping(joystick_raw)

        # Resamplear joystick a 250 Hz si es necesario
        if joystick_sfreq != target_sfreq:
            print(
                f"Resampleando joystick de {joystick_sfreq} Hz a {target_sfreq} Hz..."
            )
            # Asegurarnos de que las unidades sean numéricas antes de resamplear
            for ch in joystick_raw.info["chs"]:
                ch["unit"] = 0  # 0 = arbitrary units
                ch["unit_mul"] = 0
            joystick_raw = joystick_raw.resample(target_sfreq, npad="auto")

        # Verificar que las longitudes coincidan
        print("\nVerificando longitudes de señales:")
        print(f"Longitud EEG: {len(raw.times)} muestras")
        print(f"Longitud Joystick: {len(joystick_raw.times)} muestras")

        # Ajustar la longitud del joystick si es necesario
        if len(joystick_raw.times) != len(raw.times):
            print("\nAjustando longitud del joystick...")
            # Calcular el factor de escala
            scale_factor = len(raw.times) / len(joystick_raw.times)
            # Resamplear el joystick para que coincida con la longitud del EEG
            joystick_raw = joystick_raw.resample(
                target_sfreq * scale_factor, npad="auto"
            )
            print(f"Nueva longitud Joystick: {len(joystick_raw.times)} muestras")

        # Agregar los canales del joystick al objeto raw de EEG
        print("Agregando canales del joystick al objeto raw...")
        raw.add_channels([joystick_raw], force_update_info=True)

# Aplicar el montaje de canales EEG
print("\nAplicando montaje de canales EEG...")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = set_chs_montage(raw)

# Documentar el mapeo de canales y unidades
print("\nDocumentando información de canales...")
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

# Crear BIDSPath para EEG
bids_path = BIDSPath(
    subject=subject,
    session=session.lower(),  # Usar minúsculas para BIDS
    task=task,
    run=run,
    acquisition=acq,
    root=bids_root,
    datatype="eeg",
    extension=".vhdr",
    check=True,
)

# Crear directorios necesarios
Path(bids_path.directory).mkdir(parents=True, exist_ok=True)

# Guardar channels.tsv
channels_path = bids_path.copy().update(suffix="channels", extension=".tsv")
channels_info.to_csv(channels_path, sep="\t", index=False)
print(f"Información de canales guardada en: {channels_path}")

# Guardar datos EEG en formato BIDS
print("\nGuardando datos EEG en formato BIDS...")
# Convertir eventos a formato BrainVision antes de guardar
events_array = None
event_id = None
if raw.annotations:
    events, event_id = mne.events_from_annotations(raw)
    if len(events) > 0:
        events_array = events

write_raw_bids(
    raw=raw,
    bids_path=bids_path,
    format="BrainVision",
    allow_preload=True,
    overwrite=True,
    events=events_array,
    event_id=event_id,
)

# Procesar y guardar los eventos si existen
for stream in streams:
    if stream["info"]["type"][0] == "Markers":
        print("\nProcesando eventos...")
        # Convertir los valores de los markers a números
        marker_values = {
            "test": 0,
            "self_report_post_start": 1,
            "self_report_post_end": 2,
            "instruction_start": 3,
            "instruction_end": 4,
            "video_start": 5,
            "video_end": 6,
            "baseline_start": 7,
            "baseline_end": 8,
            "luminance_start": 9,
            "luminance_end": 10,
            "audio_response_start": 11,
            "audio_response_end": 12,
            "calm_video_start": 13,
            "calm_video_end": 14,
            "SUPRABLOCK_START": 15,
            "SUPRABLOCK_END": 16,
            "BLOCK_START": 17,
            "BLOCK_END": 18,
        }

        # Convertir los valores de los markers a números
        event_values = np.array(
            [marker_values.get(val[0], 0) for val in stream["time_series"]]
        )

        # Convertir timestamps a muestras relativas al inicio de los datos
        event_times = np.array(stream["time_stamps"])

        # Verificar si hay eventos
        if len(event_times) == 0:
            print("Warning: No events found in the stream")
            continue

        event_samples = np.round(
            (event_times - event_times[0]) * raw.info["sfreq"]
        ).astype(int)

        # Filtrar eventos que están dentro del rango de los datos
        valid_events = (event_samples >= 0) & (event_samples < len(raw.times))
        event_samples = event_samples[valid_events]
        event_values = event_values[valid_events]

        if len(event_samples) == 0:
            print("Warning: No valid events found within the data range")
            continue

        # Crear el canal de estimulación si no existe
        if "STI 014" not in raw.ch_names:
            stim_data = np.zeros((1, len(raw.times)))
            raw.add_channels(
                [
                    mne.io.RawArray(
                        stim_data,
                        mne.create_info(
                            ["STI 014"], raw.info["sfreq"], ch_types=["misc"]
                        ),
                    )
                ]
            )
            raw.set_channel_types({"STI 014": "stim"})

        # Crear eventos
        events = np.column_stack(
            (event_samples, np.zeros(len(event_samples)), event_values)
        )

        # Agregar eventos al objeto raw
        raw.add_events(events, stim_channel="STI 014")

        # Guardar eventos en formato BIDS
        events_df = pd.DataFrame(events, columns=["onset", "duration", "description"])
        events_path = bids_path.copy().update(suffix="events", extension=".tsv")
        events_df.to_csv(events_path, sep="\t", index=False)

        # Guardar metadatos de eventos
        events_metadata = {
            "onset": {"Description": "Event onset", "Units": "seconds"},
            "duration": {"Description": "Event duration", "Units": "seconds"},
            "description": {
                "Description": "Event description",
                "event_id": marker_values,
            },
        }
        events_metadata_path = events_path.copy().update(extension=".json")
        with open(events_metadata_path.fpath, "w") as f:
            json.dump(events_metadata, f, indent=4)

print("\nProcesamiento BIDS completado exitosamente.")


# %%
