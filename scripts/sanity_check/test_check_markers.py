# %%
import os
from pathlib import Path

import mne

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)


# %%
subjects = ["18"]

# Chequeo de audio para ver marcadores
for subject in subjects:
    fname = (
        f"./test_outputs/sub-{subject}/ses-VR/eeg/sub-{subject}_ses-VR_task-02_eeg.fif"
    )

    print(f"Leyendo archivo raw de {subject}")
    raw = mne.io.read_raw(fname, verbose=True, preload=True)

    print("Cargando datos raw")
    data = raw.load_data()

    canal_audio = raw.copy().pick_channels(["AUDIO"])
    canal_audio.plot(scalings={"misc": 5e-4})
# %%


def test_check_markers():
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
