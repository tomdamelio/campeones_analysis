import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

sys.path.insert(0, str(Path("src").resolve()))
from campeones_analysis.luminance.qa import compute_rejection_percentage

def test_autoreject_params():
    print("Loading data for one run to test AutoReject parameters...")
    vhdr_path = Path("data/derivatives/campeones_preproc/sub-27/ses-vr/eeg/sub-27_ses-vr_task-01_acq-a_run-002_desc-preproc_eeg.vhdr")
    events_path = Path("data/derivatives/merged_events/sub-27/ses-vr/eeg/sub-27_ses-vr_task-01_acq-a_run-002_desc-merged_events.tsv")

    if not vhdr_path.exists():
        print("Data not found.")
        return

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    events_df = pd.read_csv(events_path, sep="\t")
    
    # Just grab the first video
    video_events = events_df[events_df["trial_type"] == "video_luminance"].iloc[0]
    onset = video_events["onset"]
    dur = video_events["duration"]
    
    # Crop
    raw_cropped = raw.copy().crop(tmin=onset, tmax=onset + dur)
    
    # Make epochs
    epoch_duration = 1.0
    epoch_step = 0.1
    n_samples = raw_cropped.n_times
    sfreq = raw_cropped.info["sfreq"]
    
    epoch_onsets_s = np.arange(0, dur - epoch_duration + epoch_step, epoch_step)
    events = np.zeros((len(epoch_onsets_s), 3), dtype=int)
    events[:, 0] = np.round(epoch_onsets_s * sfreq).astype(int) + raw_cropped.first_samp
    events[:, 2] = 1
    
    epochs = mne.Epochs(
        raw_cropped,
        events=events,
        tmin=0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
    )
    
    montage = mne.channels.read_custom_montage("scripts/preprocessing/BC-32_FCz_modified.bvef")
    epochs.set_montage(montage, on_missing="ignore")
    from scripts.modeling.config import EEG_CHANNELS
    epochs.pick(EEG_CHANNELS)
    
    print(f"\nCreated {len(epochs)} epochs.")
    
    configs = [
        {"name": "Default (Strict)", "n_interpolate": np.array([1, 4, 32]), "consensus": np.linspace(0, 1.0, 11)},
        {"name": "Relaxed A", "n_interpolate": np.array([1, 4, 8, 12, 16]), "consensus": np.linspace(0, 1.0, 11)},
        {"name": "Very Relaxed B", "n_interpolate": np.array([4, 8, 16, 24]), "consensus": np.linspace(0.5, 1.0, 11)},
        {"name": "Extreme C (Try anything to save)", "n_interpolate": np.array([16, 24, 28]), "consensus": np.linspace(0.7, 1.0, 11)},
    ]
    
    for cfg in configs:
        print(f"\n--- Testing {cfg['name']} ---")
        print(f"n_interpolate grid: {cfg['n_interpolate']}")
        print(f"consensus grid len: {len(cfg['consensus'])}")
        
        ar = AutoReject(
            n_interpolate=cfg["n_interpolate"],
            consensus=cfg["consensus"],
            random_state=42,
            verbose=False,
            n_jobs=-1
        )
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)
        
        n_rej = int(np.sum(reject_log.bad_epochs))
        pct = (n_rej / len(epochs)) * 100
        print(f"Best n_interpolate chosen by CV: {ar.n_interpolate_}")
        print(f"Best consensus chosen by CV: {ar.consensus_}")
        print(f"Epochs rejected: {n_rej}/{len(epochs)} ({pct:.1f}%)")

if __name__ == "__main__":
    test_autoreject_params()
