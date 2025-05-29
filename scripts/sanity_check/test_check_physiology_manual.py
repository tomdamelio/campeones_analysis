import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import pandas as pd


def process_physiology_data(fif_file):
    """Procesa los datos fisiológicos de un archivo FIF."""
    print(f"\nProcesando archivo: {fif_file}")

    # Leer archivo raw
    print("Leyendo archivo raw")
    raw = mne.io.read_raw(fif_file, verbose=True, preload=True)

    print("Cargando datos raw")
    raw.load_data()

    # Obtener datos y tiempos
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["Time"] = times

    # Procesar EDA
    print("\nProcesando señal EDA...")
    signals_eda, info_eda = nk.eda_process(df["GSR"], sampling_rate=raw.info["sfreq"])
    plt.figure()
    nk.eda_plot(signals_eda, info=info_eda)
    plt.show()

    # Procesar ECG
    print("\nProcesando señal ECG...")
    ecg = df["ECG"]
    signals_ecg, info_ecg = nk.ecg_process(ecg, sampling_rate=raw.info["sfreq"])
    plt.figure()
    nk.ecg_plot(signals_ecg, info=info_ecg)
    plt.show()

    # Procesar RESP
    print("\nProcesando señal RESP...")
    signals_resp, info_resp = nk.rsp_process(
        df["RESP"], sampling_rate=raw.info["sfreq"]
    )
    plt.figure()
    nk.rsp_plot(signals_resp, info=info_resp)
    plt.show()


def main():
    # Configuración de paths
    script_dir = Path(__file__).parent
    test_outputs = script_dir.parent / "test_outputs"

    # Configurar parámetros del archivo a inspeccionar
    subject = "06"  # ID del sujeto
    session = "vr"  # Sesión
    task = "a"  # Tarea
    run = "001"  # Run

    # Construir path del archivo
    fif_file = (
        test_outputs
        / f"sub-{subject}"
        / f"ses-{session}"
        / "eeg"
        / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.fif"
    )

    process_physiology_data(fif_file)


if __name__ == "__main__":
    main()


def test_check_physiology_manual():
    # Configuración de paths
    project_root = Path(os.getcwd()).parent
    data_folder = project_root / "data" / "sourcedata" / "xdf"

    # Parámetros de prueba
    subject = "16"
    session = "VR"
    task = "02"
    run = "003"
    acq = "a"

    # Construir path del archivo
    xdf_path = (
        data_folder
        / f"sub-{subject}"
        / f"ses-{session}"
        / "physio"
        / f"sub-{subject}_ses-{session}_day-{acq}_task-{task}_run-{run}_eeg.xdf"
    )

    # Verificar que el archivo existe
    assert xdf_path.exists(), f"El archivo {xdf_path} no existe"

    # Verificar que es un archivo XDF
    assert xdf_path.suffix == ".xdf", f"El archivo {xdf_path} no es un archivo XDF"
