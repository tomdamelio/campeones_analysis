#!/usr/bin/env python
"""ERP + t-test por timepoint — Sanity check 3.3.

Compara CHANGE_PHOTO vs NO_CHANGE_PHOTO: ERP promedio ± SEM en función
del tiempo, con t-test independiente por timepoint (corrección FDR).

ROIs: occipital (O1/O2) y temporal (T7/T8).

Grid 2×1:
  Panel superior: occipital
  Panel inferior: temporal

Usage
-----
    micromamba run -n campeones python scripts/validation/38c_erp_ttest.py --subject 27
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "sanity_checks"
SESSION = "vr"

TMIN, TMAX = -1.5, 1.5
BASELINE = (-1.5, -1.0)
VIS_TMIN, VIS_TMAX = -0.3, 0.7   # ventana de visualización

ROIS = {
    "Occipital (O1/O2)": ["O1", "O2"],
    "Temporal (T7/T8)":  ["T7", "T8"],
}

COND_COLORS = {"CHANGE_PHOTO": "C3", "NO_CHANGE_PHOTO": "C2"}
EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

RUNS_CONFIG = {
    "27": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "006", "acq": "a", "task": "04"},
        {"run": "007", "acq": "b", "task": "01"},
        {"run": "009", "acq": "b", "task": "03"},
        {"run": "010", "acq": "b", "task": "04"},
    ],
}
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]


def load_all_epochs(subject: str) -> mne.Epochs:
    all_list = []
    for rc in RUNS_CONFIG[subject]:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
        vhdr = eeg_dir / (
            f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
            f"_run-{run_id}_desc-preproc_eeg.vhdr"
        )
        tsv = (
            PHOTO_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
            / f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
              f"_run-{run_id}_desc-photo_events.tsv"
        )
        if not vhdr.exists() or not tsv.exists():
            continue
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"].isin(EVENT_ID)]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        eids = rows["trial_type"].map(EVENT_ID).values
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets, eids = onsets[valid], eids[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int), eids])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        run_eid = {k: v for k, v in EVENT_ID.items() if k in rows["trial_type"].unique()}
        try:
            ep = mne.Epochs(raw, mne_ev, event_id=run_eid,
                            tmin=TMIN, tmax=TMAX, picks=avail,
                            baseline=BASELINE, preload=True, verbose=False)
            all_list.append(ep)
            n_c = (ep.events[:, 2] == 1).sum()
            n_nc = (ep.events[:, 2] == 2).sum()
            print(f"  run-{run_id}: {n_c} CHANGE + {n_nc} NO_CHANGE")
        except Exception as e:
            print(f"  run-{run_id}: {e}")
    if not all_list:
        raise RuntimeError("No epochs loaded")
    return mne.concatenate_epochs(all_list, verbose=False)


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean reject array."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.where(below)[0].max()
    reject = np.zeros(n, dtype=bool)
    reject[order[:max_k + 1]] = True
    return reject


def plot_erp_panel(ax, data_c, data_nc, times_ms, roi_name):
    """ERP comparison + t-test FDR in a single axis."""
    # data_c, data_nc: (n_epochs, n_times) — already averaged over ROI channels
    mean_c  = data_c.mean(axis=0) * 1e6
    sem_c   = data_c.std(axis=0, ddof=1) / np.sqrt(data_c.shape[0]) * 1e6
    mean_nc = data_nc.mean(axis=0) * 1e6
    sem_nc  = data_nc.std(axis=0, ddof=1) / np.sqrt(data_nc.shape[0]) * 1e6

    # t-test per timepoint
    n_t = data_c.shape[1]
    p_vals = np.array([
        ttest_ind(data_c[:, t], data_nc[:, t]).pvalue for t in range(n_t)
    ])
    reject = fdr_bh(p_vals, alpha=0.05)

    # shading for significant timepoints
    sig_times = times_ms[reject]
    if len(sig_times) > 0:
        # group contiguous significant timepoints into spans
        diffs = np.diff(sig_times)
        breaks = np.where(diffs > (times_ms[1] - times_ms[0]) * 1.5)[0] + 1
        spans = np.split(sig_times, breaks)
        for span in spans:
            if len(span) > 0:
                ax.axvspan(span[0], span[-1], alpha=0.18, color="gold", zorder=0,
                           label="p<0.05 FDR" if span is spans[0] else None)

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)

    ax.fill_between(times_ms, mean_c - sem_c,  mean_c + sem_c,
                    alpha=0.25, color="C3")
    ax.fill_between(times_ms, mean_nc - sem_nc, mean_nc + sem_nc,
                    alpha=0.25, color="C2")
    ax.plot(times_ms, mean_c,  color="C3", lw=2.0, label="CHANGE")
    ax.plot(times_ms, mean_nc, color="C2", lw=2.0, label="NO_CHANGE")

    n_sig = reject.sum()
    ax.set_title(f"{roi_name}   ({n_sig}/{n_t} timepoints p<0.05 FDR)", fontsize=10)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Amplitude (µV)", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading epochs for sub-{sub}...")
    epochs = load_all_epochs(sub)
    epochs_vis = epochs.copy().crop(tmin=VIS_TMIN, tmax=VIS_TMAX)
    times_ms = epochs_vis.times * 1000

    n_c  = len(epochs_vis["CHANGE_PHOTO"])
    n_nc = len(epochs_vis["NO_CHANGE_PHOTO"])
    print(f"  CHANGE: {n_c}, NO_CHANGE: {n_nc}")

    roi_list = list(ROIS.items())
    fig, axes = plt.subplots(len(roi_list), 1, figsize=(12, 4 * len(roi_list)))
    if len(roi_list) == 1:
        axes = [axes]

    fig.suptitle(
        f"sub-{sub} — ERP: CHANGE vs NO_CHANGE\n"
        "Sanity check 3.3  |  Shading = p<0.05 FDR (Benjamini-Hochberg)",
        fontsize=11,
    )

    data_c_all  = epochs_vis["CHANGE_PHOTO"].get_data()    # (n, ch, t)
    data_nc_all = epochs_vis["NO_CHANGE_PHOTO"].get_data()

    for ax, (roi_name, roi_chs) in zip(axes, roi_list):
        avail = [ch for ch in roi_chs if ch in epochs_vis.ch_names]
        if not avail:
            ax.set_visible(False)
            continue
        ch_idx = [epochs_vis.ch_names.index(ch) for ch in avail]
        data_c_roi  = data_c_all[:, ch_idx, :].mean(axis=1)   # (n, t)
        data_nc_roi = data_nc_all[:, ch_idx, :].mean(axis=1)
        plot_erp_panel(ax, data_c_roi, data_nc_roi, times_ms, roi_name)

    plt.tight_layout()
    out_path = out_dir / f"sub-{sub}_38c_erp_ttest.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
