#!/usr/bin/env python

import argparse
import logging
import os
from pathlib import Path

from test_check_eda import process_subject as process_eda

# Importar funciones de los otros scripts
from test_read_xdf_jero import process_subject as process_xdf

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root():
    """Obtener el directorio raíz del proyecto."""
    # Si estamos en tests/sanity_check, subir dos niveles
    current_dir = Path(__file__).parent
    if current_dir.name == "sanity_check":
        return current_dir.parent.parent
    return current_dir


def get_subject_dirs(data_dir):
    """Obtener lista de directorios de sujetos."""
    return [
        d
        for d in os.listdir(data_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(data_dir, d))
    ]


def process_subject(subject, data_dir, test_outputs_dir):
    """Procesar un sujeto específico."""
    logger.info(f"Procesando sujeto {subject}")

    # Procesar XDF
    logger.info("Procesando archivos XDF...")
    try:
        process_xdf(subject, skip_if_exists=True, data_folder=data_dir)
    except Exception as e:
        logger.error(f"Error procesando XDF: {e}")
        return False

    # Procesar EDA
    logger.info("Procesando EDA...")
    try:
        process_eda(subject)
    except Exception as e:
        logger.error(f"Error procesando EDA: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Procesar datos de sujetos")
    parser.add_argument("--subject", help="ID del sujeto a procesar (ej: 16)")
    parser.add_argument(
        "--test", action="store_true", help="Usar directorio de datos de test"
    )
    args = parser.parse_args()

    # Obtener directorio raíz del proyecto
    project_root = get_project_root()

    # Determinar directorio de datos
    data_dir = (
        project_root / "tests" / "test_data" if args.test else project_root / "data"
    )
    test_outputs_dir = project_root / "tests" / "test_outputs"

    # Crear directorio de outputs si no existe
    os.makedirs(test_outputs_dir, exist_ok=True)

    # Obtener lista de sujetos
    if args.subject:
        subjects = [f"sub-{args.subject}"]
    else:
        subjects = get_subject_dirs(data_dir)

    if not subjects:
        logger.error(f"No se encontraron sujetos en {data_dir}")
        return

    # Procesar cada sujeto
    results = {}
    for subject in subjects:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Procesando {subject}")
        logger.info(f"{'=' * 50}\n")

        success = process_subject(subject, data_dir, test_outputs_dir)
        results[subject] = "Completado" if success else "Error"

    # Mostrar resumen
    logger.info("\nResumen de procesamiento:")
    logger.info("=" * 50)
    for subject, status in results.items():
        logger.info(f"{subject}: {status}")


if __name__ == "__main__":
    main()
