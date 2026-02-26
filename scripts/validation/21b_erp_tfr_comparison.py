"""Comparative ERP and TFR Analysis: Change vs No-Change luminance conditions.

Extracts two conditions per video:
1. 'Change': Top N moments of largest absolute luminance change.
2. 'No-Change': N moments of smallest absolute luminance change (stable periods),
   ensuring they don't temporally overlap with the 'Change' moments.

Computes ERPs and Time-Frequency Representations (TFRs) for both conditions,
and visualizes the contrasts (Change - No-Change).
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import EEG_CHANNELS
from config_luminance import (
    DERIVATIVES_PATH,
    ERP_N_CHANGES,
    ERP_TMAX,
    ERP_TMIN,
    LUMINANCE_CSV_MAP,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
)
from campeones_analysis.luminance.sync import load_luminance_csv

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Override TMIN to allow for -1000ms baseline
ERP_TMIN = -1.0

# Output directory
RESULTS_PATH = PROJECT_ROOT / "results" / "validation" / "erp_tfr_comparison"

ROIS = {
    "Frontal": ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'FCz'],
    "Temporal": ['T7', 'T8', 'FT9', 'FT10', 'TP9', 'TP10'],
    "Parietal": ['P3', 'P4', 'P7', 'P8', 'Pz', 'CP1', 'CP2', 'CP5', 'CP6'],
    "Occipital": ['O1', 'O2']
}

TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)  # 3 to 40 Hz logarithmic
TFR_N_CYCLES = TFR_FREQS / 2.0  # variable time-window


def detect_contrast_events(luminance_values: np.ndarray, n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect top N changes and bottom N stable moments."""
    if len(luminance_values) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    luminance_diff = np.diff(luminance_values)
    abs_diff = np.abs(luminance_diff)

    # Top N (Change)
    n_top = min(n_events, len(abs_diff))
    top_indices = np.argsort(abs_diff)[::-1][:n_top]

    # Bottom N (No-Change)
    bottom_candidates = np.argsort(abs_diff)
    selected_bottom = []
    
    for idx in bottom_candidates:
        # Must be at least 30 frames (~1 sec) away from any Top N change
        dist_to_top = np.min(np.abs(top_indices - idx)) if len(top_indices) > 0 else 1000
        if dist_to_top > 30:
            # Must also be at least 15 frames (~0.5 sec) away from other No-Change events
            dist_to_bottom = np.min(np.abs(np.array(selected_bottom) - idx)) if selected_bottom else 1000
            if dist_to_bottom > 15:
                selected_bottom.append(idx)
        
        if len(selected_bottom) == n_events:
            break
            
    return top_indices, np.array(selected_bottom)


def select_roi_channels(available: list[str], roi: list[str]) -> list[str]:
    """Select requested channels available in the EEG."""
    return [ch for ch in roi if ch in available]


def _resolve_eeg_path(run_config: dict) -> Path | None:
    path = DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg" / \
           f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-preproc_eeg.vhdr"
    return path if path.exists() else None


def _resolve_events_path(run_config: dict) -> Path | None:
    merged = PROJECT_ROOT / "data" / "derivatives" / "merged_events" / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg" / \
             f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-merged_events.tsv"
    if merged.exists():
        return merged
    regular = DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg" / \
              f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-preproc_events.tsv"
    return regular if regular.exists() else None


def process_video_segment_contrast(
    eeg_raw: mne.io.Raw, onset_s: float, duration_s: float, video_id: int,
    luminance_df: pd.DataFrame, sfreq: float, roi_channels: list[str]
) -> mne.Epochs | None:
    t_stop = onset_s + duration_s
    try:
        segment_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=t_stop)
    except ValueError:
        return None

    lum_vals = luminance_df["luminance"].values
    lum_times = luminance_df["timestamp"].values

    top_idx, bottom_idx = detect_contrast_events(lum_vals, ERP_N_CHANGES)
    if len(top_idx) == 0 or len(bottom_idx) == 0:
        return None

    # Map to EEG samples
    all_indices = np.concatenate([top_idx, bottom_idx])
    event_ids = np.concatenate([np.ones(len(top_idx), dtype=int), np.full(len(bottom_idx), 2, dtype=int)])
    
    change_times_s = lum_times[all_indices + 1]
    change_samples = np.round((change_times_s - segment_raw.times[0]) * sfreq).astype(int)
    
    valid_mask = (change_samples >= 0) & (change_samples < len(segment_raw.times))
    change_samples = change_samples[valid_mask]
    event_ids = event_ids[valid_mask]
    
    change_samples = change_samples + segment_raw.first_samp

    mne_events = np.column_stack([change_samples, np.zeros(len(change_samples), dtype=int), event_ids])
    mne_events = mne_events[np.argsort(mne_events[:, 0])]

    try:
        epochs = mne.Epochs(
            segment_raw, events=mne_events,
            event_id={"Change": 1, "NoChange": 2},
            tmin=ERP_TMIN, tmax=ERP_TMAX, picks=roi_channels,
            baseline=None, preload=True, verbose=False
        )
        return epochs
    except Exception:
        return None


def plot_erp_comparisons(epochs: mne.Epochs, output_dir: Path):
    time_pts = epochs.times * 1000  # in ms
    
    roi_plot_data = {}
    global_min, global_max = float('inf'), float('-inf')
    
    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs: continue
        
        roi_plot_data[roi_name] = {}
        for cond, color in [("Change", "C0"), ("NoChange", "C1")]:
            cond_data = epochs[cond].copy().pick(valid_chs).get_data() # (epochs, channels, times)
            
            # 1. Average across channels -> shape (epochs, times)
            cond_data_roi = cond_data.mean(axis=1) * 1e6 # convert to uV
            
            # 2. Compute mean and SEM across epochs -> shape (times,)
            mean_erp = cond_data_roi.mean(axis=0)
            sem_erp = cond_data_roi.std(axis=0, ddof=1) / np.sqrt(cond_data_roi.shape[0])
            
            roi_plot_data[roi_name][cond] = {
                "color": color,
                "n": cond_data_roi.shape[0],
                "mean": mean_erp,
                "sem": sem_erp
            }
            
            global_min = min(global_min, np.min(mean_erp - sem_erp))
            global_max = max(global_max, np.max(mean_erp + sem_erp))
            
    # Add a small margin to y-limits
    margin = (global_max - global_min) * 0.05
    ylim = (global_min - margin, global_max + margin)
            
    for roi_name, cond_data_dict in roi_plot_data.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        
        for cond, cp in cond_data_dict.items():
            ax.plot(time_pts, cp["mean"], label=f"{cond} (n={cp['n']})", color=cp["color"])
            ax.fill_between(time_pts, cp["mean"] - cp["sem"], cp["mean"] + cp["sem"], color=cp["color"], alpha=0.3)
        
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"ERP Contrast: {roi_name} ROI (sub-{SUBJECT})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_erp_contrast_{roi_name}.png", dpi=150)
        plt.close(fig)


def plot_tfr_comparisons(epochs: mne.Epochs, output_dir: Path):
    logger.info("Computing Time-Frequency Representations (Morlet)...")
    tfr_change = tfr_morlet(epochs["Change"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)
    tfr_nochange = tfr_morlet(epochs["NoChange"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)

    # 1. Apply baseline from -1000ms to -200ms
    tfr_change.apply_baseline(baseline=(-1.0, -0.2), mode="percent")
    tfr_nochange.apply_baseline(baseline=(-1.0, -0.2), mode="percent")

    # 2. Contrast Change - NoChange
    tfr_diff = tfr_change.copy()
    tfr_diff.data = tfr_change.data - tfr_nochange.data
    
    roi_tfr_data = {}
    global_vmax = 0
    
    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in tfr_diff.ch_names]
        if not valid_chs: continue
        
        # Average across channels manually
        ch_indices = [tfr_diff.ch_names.index(c) for c in valid_chs]
        roi_data = tfr_diff.data[ch_indices, :, :].mean(axis=0) # shape (freqs, times)
        
        roi_tfr_data[roi_name] = roi_data
        
        vmax = np.percentile(np.abs(roi_data), 98)
        global_vmax = max(global_vmax, vmax)
        
    for roi_name, roi_data in roi_tfr_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        times = tfr_diff.times * 1000
        freqs = tfr_diff.freqs
        
        im = ax.pcolormesh(times, freqs, roi_data, cmap='RdBu_r', shading='gouraud', vmin=-global_vmax, vmax=global_vmax)
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"TFR Contrast (Change - NoChange): {roi_name} ROI")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power diff (% points)")
        
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_tfr_contrast_{roi_name}.png", dpi=150)
        plt.close(fig)


def run_pipeline():
    print("=" * 60)
    print(f"21b — ERP & TFR Comparison (Change vs No-Change) — sub-{SUBJECT}")
    print("=" * 60)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    roi_channels = select_roi_channels(EEG_CHANNELS, EEG_CHANNELS)
    if not roi_channels: return

    all_epochs = []
    
    for run_config in RUNS_CONFIG:
        vhdr = _resolve_eeg_path(run_config)
        events = _resolve_events_path(run_config)
        if not vhdr or not events: continue

        eeg_raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        events_df = pd.read_csv(events, sep="\t")
        rec_roi = select_roi_channels(eeg_raw.ch_names, EEG_CHANNELS)
        
        lum_evs = events_df[events_df["trial_type"] == "video_luminance"]
        
        for _, row in lum_evs.iterrows():
            vid = int(row["stim_id"]) - 100
            csv = LUMINANCE_CSV_MAP.get(vid)
            if not csv: continue
            try: lum_df = load_luminance_csv(STIMULI_PATH / csv)
            except FileNotFoundError: continue

            eps = process_video_segment_contrast(eeg_raw, float(row["onset"]), float(row["duration"]), vid, lum_df, eeg_raw.info["sfreq"], rec_roi)
            if eps: all_epochs.append(eps)

    if not all_epochs:
        print("No epochs created.")
        return

    grand_epochs = mne.concatenate_epochs(all_epochs)
    print(f"Grand total epochs: {len(grand_epochs)} (Change: {len(grand_epochs['Change'])}, NoChange: {len(grand_epochs['NoChange'])})")

    plot_erp_comparisons(grand_epochs, RESULTS_PATH)
    plot_tfr_comparisons(grand_epochs, RESULTS_PATH)

    print("Comparative pipeline complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
