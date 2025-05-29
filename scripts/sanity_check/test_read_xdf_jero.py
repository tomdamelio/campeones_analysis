#!/usr/bin/env python

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pyxdf
from mnelab.io.xdf import read_raw_xdf

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root():
    """Obtener el directorio raíz del proyecto."""
    # Si estamos en tests/sanity_check, subir dos niveles
    current_dir = Path(__file__).parent
    if current_dir.name == "sanity_check":
        return current_dir.parent.parent
    return current_dir


def process_subject(subject, skip_if_exists=False, data_folder=None):
    """Procesar archivos XDF de un sujeto."""
    # Obtener el directorio raíz del proyecto
    project_root = get_project_root()

    # Configuración de paths usando pathlib
    if data_folder is None:
        data_folder = project_root / "data"
    else:
        data_folder = Path(data_folder)
    test_outputs = project_root / "tests" / "test_outputs"

    # Encontrar todos los archivos xdf para el sujeto específico
    subject_pattern = f"sub-{subject}"
    xdf_files = list(data_folder.glob(f"{subject_pattern}/**/*.xdf"))
    logger.info(f"Encontrados {len(xdf_files)} archivos xdf para el sujeto {subject}")

    # Regex para intentar extraer info, pero no filtra
    pattern = (
        r"sub-(\d+)_ses-([\w-]+)(?:_day-([a-zA-Z]))?_task-([\w-]+)_run-(\d+)_(\w+)\.xdf"
    )

    if not xdf_files:
        logger.error(f"No se encontraron archivos XDF para el sujeto {subject}")
        return False

    for file in xdf_files:
        logger.info(f"Procesando archivo: {file}")

        match = re.search(pattern, file.name)
        if match:
            subject_id, session, day, task, run, modality = match.groups()
            logger.info(
                f"Subject: {subject_id}, Session: {session}, Day: {day}, Task: {task}, Run: {run}, Modality: {modality}"
            )
        else:
            logger.warning(
                f"No se pudo extraer información del nombre: {file.name}. Se procesará igual."
            )
            # Usar valores por defecto o el subject recibido
            subject_id = subject
            session = "unk"
            day = "unk"
            task = "unk"
            run = "unk"
            modality = "unk"

        # Verificar si el archivo FIF ya existe
        output_dir = (
            test_outputs / f"sub-{subject_id}" / f"ses-{session.upper()}" / "eeg"
        )
        output_file = (
            output_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_eeg.fif"
        )

        if skip_if_exists and output_file.exists():
            logger.info(f"El archivo FIF ya existe: {output_file}")
            logger.info("Saltando procesamiento...")
            continue

        streams, header = pyxdf.load_xdf(str(file))

        for stream in streams:
            if stream["info"]["type"][0] == "EEG":
                logger.info("Procesando stream EEG...")
                raw = read_raw_xdf(
                    str(file), stream_ids=[stream["info"]["stream_id"]], preload=True
                )
                logger.info(raw.info)

                # Downsample the data to 500 Hz
                raw.resample(500, npad="auto")

                # Crear estructura de directorios BIDS en test_outputs
                output_dir.mkdir(parents=True, exist_ok=True)

                # Guardar archivo FIF con nombre BIDS
                raw.save(str(output_file), overwrite=True)
                logger.info(f"Archivo guardado en: {output_file}")

            elif stream["info"]["type"][0] == "Markers":
                logger.info("Procesando stream Markers...")
                # Crear directorio para markers
                markers_dir = (
                    test_outputs
                    / f"sub-{subject_id}"
                    / f"ses-{session.upper()}"
                    / "events"
                )
                markers_dir.mkdir(parents=True, exist_ok=True)

                # Guardar markers en formato numpy
                markers_file = (
                    markers_dir
                    / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_events.npy"
                )
                np.save(
                    str(markers_file),
                    {
                        "time_stamps": stream["time_stamps"],
                        "data": stream["time_series"],
                    },
                )
                logger.info(f"Markers guardados en: {markers_file}")
                logger.info(f"Shape Markers: {stream['time_stamps'].shape}")

    logger.info("Procesamiento completado!")
    return True


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Procesar archivos XDF a formato BIDS")
    parser.add_argument(
        "--subject", type=str, help="ID del sujeto a procesar (ej: 16)", required=True
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Saltar la creación de archivos FIF si ya existen",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Ruta alternativa para buscar los datos (opcional)",
    )
    args = parser.parse_args()

    process_subject(args.subject, args.skip_if_exists, args.data_folder)


if __name__ == "__main__":
    main()

# %%
