import sys
from pathlib import Path

# Add project paths to import modules
sys.path.insert(0, str(Path("scripts/modeling").resolve()))
sys.path.insert(0, str(Path("src").resolve()))

from config_luminance import RUNS_CONFIG
from scripts.qa import __init__ # just to verify package
import importlib.util

# Load the qa script dynamically since it starts with a number
spec = importlib.util.spec_from_file_location("qa_script", "scripts/qa/16_eeg_qa_autoreject.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

def debug_epochs():
    run_config = RUNS_CONFIG[0]
    vhdr = qa.resolve_eeg_path(run_config)
    print(f"Loading EEG: {vhdr}")
    raw = qa.load_eeg_raw(vhdr)
    print(f"Annotations: {raw.annotations}")
    
    events_path = qa.resolve_events_path(run_config)
    evs = qa.load_events_df(events_path)
    ev = evs[evs['trial_type'] == 'video_luminance'].iloc[0]
    print(f"Event: {ev['trial_type']} at {ev['onset']}s for {ev['duration']}s")
    
    try:
        video_raw = raw.copy().crop(tmin=float(ev['onset']), tmax=float(ev['onset'])+float(ev['duration']))
        import numpy as np
        import mne
        sfreq = video_raw.info['sfreq']
        epoch_dur = 0.5
        n_samples_segment = video_raw.n_times
        n_samples_epoch = int(round(epoch_dur * sfreq))
        n_samples_step = int(round(0.1 * sfreq))
        epoch_onset_samples = np.arange(0, n_samples_segment - n_samples_epoch + 1, n_samples_step)
        events_array = np.column_stack([
            epoch_onset_samples,
            np.zeros(len(epoch_onset_samples), dtype=int),
            np.ones(len(epoch_onset_samples), dtype=int),
        ])
        eeg_picks = mne.pick_types(video_raw.info, eeg=True, exclude="bads")
        print(f"Creating Epochs with {len(events_array)} events...")
        eps = mne.Epochs(video_raw, events=events_array, event_id=1, tmin=0.0, tmax=0.5 - 1.0/sfreq, picks=eeg_picks, baseline=None, preload=True, verbose=False)
        print(f"Created epochs: {len(eps)}")
        print("Drop log sample:")
        for i, d in enumerate(eps.drop_log[:20]):
            print(f"  Epoch {i}: {d}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_epochs()
