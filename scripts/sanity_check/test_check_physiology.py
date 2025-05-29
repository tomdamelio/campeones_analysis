# %%
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import pandas as pd


def process_eda(raw, output_dir, file_name):
    """Procesa y genera plots de EDA."""
    print("\nProcesando señal EDA...")

    # Obtener datos y tiempos
    data, times = raw.get_data(return_times=True)

    # Crear DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["Time"] = times

    # Procesar señal EDA
    signals, info = nk.eda_process(df["GSR"], sampling_rate=raw.info["sfreq"])

    # Crear y guardar plot de EDA
    plt.figure(figsize=(15, 10))
    nk.eda_plot(signals, info=info)
    plt.title(f"EDA Analysis - {file_name}")

    # Guardar plot
    plot_file = output_dir / "eda" / f"{file_name}_eda_plot.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot EDA guardado en: {plot_file}")


def process_ecg(raw, output_dir, file_name):
    """Procesa y genera plots de ECG."""
    print("\nProcesando señal ECG...")

    # Obtener datos y tiempos
    data, times = raw.get_data(return_times=True)

    # Crear DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["Time"] = times

    # Procesar señal ECG
    signals, info = nk.ecg_process(df["ECG"], sampling_rate=raw.info["sfreq"])

    # Crear y guardar plot de ECG
    plt.figure(figsize=(15, 10))
    nk.ecg_plot(signals, info=info)
    plt.title(f"ECG Analysis - {file_name}")

    # Guardar plot
    plot_file = output_dir / "ecg" / f"{file_name}_ecg_plot.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot ECG guardado en: {plot_file}")


def process_resp(raw, output_dir, file_name):
    """Procesa y genera plots de RESP."""
    print("\nProcesando señal RESP...")

    # Obtener datos y tiempos
    data, times = raw.get_data(return_times=True)

    # Crear DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["Time"] = times

    # Procesar señal RESP
    signals, info = nk.rsp_process(df["RESP"], sampling_rate=raw.info["sfreq"])

    # Crear y guardar plot de RESP
    plt.figure(figsize=(15, 10))
    nk.rsp_plot(signals, info=info)
    plt.title(f"RESP Analysis - {file_name}")

    # Guardar plot
    plot_file = output_dir / "resp" / f"{file_name}_resp_plot.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot RESP guardado en: {plot_file}")


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Generar plots de señales fisiológicas"
    )
    parser.add_argument(
        "--subject", type=str, help="ID del sujeto a procesar (ej: 16)", required=True
    )
    args = parser.parse_args()

    # Obtener el directorio raíz del proyecto
    project_root = Path(os.getcwd())
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Directorio raíz del proyecto: {project_root}")

    # Configuración de paths
    test_outputs = project_root / "tests" / "test_outputs"
    subject_dir = test_outputs / f"sub-{args.subject}"

    # Encontrar todos los archivos FIF para el sujeto
    fif_files = list(subject_dir.rglob("*.fif"))
    print(f"\nEncontrados {len(fif_files)} archivos FIF para el sujeto {args.subject}")

    if not fif_files:
        print(f"No se encontraron archivos FIF para el sujeto {args.subject}")
        sys.exit(1)

    # Crear directorio para los plots
    plots_dir = test_outputs / f"sub-{args.subject}" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Procesar cada archivo FIF
    for fif_file in fif_files:
        print(f"\nProcesando archivo: {fif_file}")

        # Extraer información del nombre del archivo
        file_name = fif_file.stem
        print(f"Nombre del archivo: {file_name}")

        # Leer archivo raw
        print("Leyendo archivo raw")
        raw = mne.io.read_raw(fif_file, verbose=True, preload=True)

        print("Cargando datos raw")
        raw.load_data()

        # Procesar cada tipo de señal
        process_eda(raw, plots_dir, file_name)
        process_ecg(raw, plots_dir, file_name)
        process_resp(raw, plots_dir, file_name)

    print("\nProcesamiento completado!")


if __name__ == "__main__":
    main()
