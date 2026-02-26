import sys
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("src").resolve()))
from scripts.modeling.config import EEG_CHANNELS

def interactive_view(run_id: str):
    runs_info = {
        "002": {"acq": "a", "vid": 12, "onset": 1452.6, "rej": "27.7%"},
        "003": {"acq": "a", "vid": 9, "onset": 939.7, "rej": "35.9%"},
        "004": {"acq": "a", "vid": 3, "onset": 1028.8, "rej": "54.5%"},
        "006": {"acq": "a", "vid": 7, "onset": 1258.0, "rej": "6.4%"},
        "007": {"acq": "b", "vid": 12, "onset": 1355.2, "rej": "4.5%"},
        "009": {"acq": "b", "vid": 9, "onset": 1281.2, "rej": "29.5%"},
        "010": {"acq": "b", "vid": 7, "onset": 1047.8, "rej": "38.3%"},
    }
    
    if run_id not in runs_info:
        print(f"Error: Run {run_id} no encontrado. Opciones: {list(runs_info.keys())}")
        return
        
    info = runs_info[run_id]
    print(f"Loading data for Run {run_id} (Video {info['vid']}) - Rechazo esperado: {info['rej']}...")
    
    # Path logic logic based on 16_eeg_qa_autoreject
    task_map = {"002": "01", "003": "02", "004": "03", "006": "04", "007": "01", "009": "03", "010": "04"}
    task_str = task_map.get(run_id, "01")
    vhdr_path = Path(f"data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-{task_str}_acq-{info['acq']}_run-{run_id}_desc-preproc_eeg.vhdr")
    
    if not vhdr_path.exists():
        print(f"Archivo no encontrado: {vhdr_path}")
        return
    
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    raw_cropped = raw.copy().crop(tmin=info['onset'], tmax=info['onset'] + 60.0)
    
    sfreq = raw_cropped.info["sfreq"]
    epoch_onsets_s = np.arange(0, 60.0 - 1.0 + 0.1, 0.1)
    events = np.zeros((len(epoch_onsets_s), 3), dtype=int)
    events[:, 0] = np.round(epoch_onsets_s * sfreq).astype(int) + raw_cropped.first_samp
    events[:, 2] = 1
    
    epochs = mne.Epochs(raw_cropped, events=events, tmin=0, tmax=1.0, baseline=None, preload=True)
    montage = mne.channels.read_custom_montage("scripts/preprocessing/BC-32_FCz_modified.bvef")
    epochs.set_montage(montage, on_missing="ignore")
    epochs.pick(EEG_CHANNELS)
    
    print("\nFitting AutoReject (esto toma unos segundos)...")
    n_interpolate = np.array([1, 4, 8])
    consensus = np.linspace(0.25, 1.0, 11)
    
    ar = AutoReject(n_interpolate=n_interpolate, consensus=consensus, random_state=42, verbose=False, n_jobs=-1)
    ar.fit(epochs)
    
    reject_log = ar.get_reject_log(epochs)
    
    print("\n" + "="*80)
    print("INSTRUCCIONES PARA EL VISOR INTERACTIVO:")
    print(" - Hacé clic sostenido en la barra inferior y arrastrá izq/der para moverte por las épocas.")
    print(" - O usá las flechas izquierda/derecha del teclado.")
    print(" - Épocas (bloques verticales sombreados en ROJO) = Descartadas.")
    print(" - Canales (líneas rojas) = Canal muy ruidoso.")
    print(" - Clic en una época rechazada te mostrará qué canales contribuyeron.")
    print(" - Cerrá la ventana para terminar el script.")
    print("="*80 + "\n")
    
    # Forcing MNE to use 'matplotlib' instead of the new 'qt' browser backend 
    # to prevent AutoReject's custom patches (red backgrounds) from being deleted on scroll/click
    mne.viz.set_browser_backend("matplotlib")
    
    fig = reject_log.plot_epochs(epochs)
    plt.show(block=True)
    
    # Previene que el script termine y cierre la ventana en Windows
    input("\nPresioná ENTER en esta consola para cerrar el script y el visor...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ver interactivamente los rechazos de AutoReject.")
    parser.add_argument("--run", type=str, default="006", help="ID del run a ver (ej: 002, 003, 004, 006, 007, 009, 010)")
    args = parser.parse_args()
    
    interactive_view(args.run)
