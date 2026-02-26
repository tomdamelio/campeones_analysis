import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

sys.path.insert(0, str(Path("src").resolve()))
from scripts.modeling.config import EEG_CHANNELS

def get_autoreject_params():
    mne.set_log_level("WARNING")
    
    runs = [
        {"run": "002", "acq": "a", "vid": 12, "onset": 1452.6},
        {"run": "003", "acq": "a", "vid": 9, "onset": 939.7},
        {"run": "004", "acq": "a", "vid": 3, "onset": 1028.8},
        {"run": "006", "acq": "a", "vid": 7, "onset": 1258.0},
        {"run": "007", "acq": "b", "vid": 12, "onset": 1355.2},
        {"run": "009", "acq": "b", "vid": 9, "onset": 1281.2},
        {"run": "010", "acq": "b", "vid": 7, "onset": 1047.8},
    ]
    
    # 0.25 consensus config mapped to qa.py
    n_interpolate = np.array([1, 4, 8])
    consensus = np.linspace(0.25, 1.0, 11)
    
    montage = mne.channels.read_custom_montage("scripts/preprocessing/BC-32_FCz_modified.bvef")
    
    results = []
    
    for r in runs:
        vhdr_path = Path(f"data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-0{r['run'][-1] if int(r['run']) < 5 else int(r['run']) - 2 if r['run'] == '006' else int(r['run'][-2:]) - 4 if r['acq'] == 'b' and r['run'] != '009' else '03'}_acq-{r['acq']}_run-{r['run']}_desc-preproc_eeg.vhdr")
        
        # Hardcoding the proper vhdr path logic based on the 16_eeg_qa script
        if r["run"] == "002": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-01_acq-a_run-002_desc-preproc_eeg.vhdr"
        elif r["run"] == "003": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-02_acq-a_run-003_desc-preproc_eeg.vhdr"
        elif r["run"] == "004": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-03_acq-a_run-004_desc-preproc_eeg.vhdr"
        elif r["run"] == "006": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-04_acq-a_run-006_desc-preproc_eeg.vhdr"
        elif r["run"] == "007": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-01_acq-b_run-007_desc-preproc_eeg.vhdr"
        elif r["run"] == "009": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-03_acq-b_run-009_desc-preproc_eeg.vhdr"
        elif r["run"] == "010": vhdr_path = "data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-04_acq-b_run-010_desc-preproc_eeg.vhdr"
        
        vhdr_path = Path(vhdr_path)
        
        print(f"Loading Run {r['run']}...")
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        raw_cropped = raw.copy().crop(tmin=r['onset'], tmax=r['onset'] + 60.0)
        
        sfreq = raw_cropped.info["sfreq"]
        epoch_onsets_s = np.arange(0, 60.0 - 1.0 + 0.1, 0.1)
        events = np.zeros((len(epoch_onsets_s), 3), dtype=int)
        events[:, 0] = np.round(epoch_onsets_s * sfreq).astype(int) + raw_cropped.first_samp
        events[:, 2] = 1
        
        epochs = mne.Epochs(raw_cropped, events=events, tmin=0, tmax=1.0, baseline=None, preload=True)
        epochs.set_montage(montage, on_missing="ignore")
        epochs.pick(EEG_CHANNELS)
        
        ar = AutoReject(n_interpolate=n_interpolate, consensus=consensus, random_state=42, verbose=False, n_jobs=-1)
        ar.fit(epochs)
        
        results.append({
            "Run": r["run"],
            "Video": r["vid"],
            "n_interpolate": ar.n_interpolate_["eeg"],
            "consensus": ar.consensus_["eeg"]
        })
        
    print("\n--- RESULTS ---")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    get_autoreject_params()
