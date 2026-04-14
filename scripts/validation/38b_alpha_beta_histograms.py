#!/usr/bin/env python
"""Alpha/beta power histograms — Sanity check 3.2.

Para cada época CHANGE_PHOTO, extrae ventanas pre [-0.55, -0.05]s y
post [+0.05, +0.55]s, calcula potencia alfa (8-13 Hz) y beta (13-30 Hz)
usando Welch, y plotea histogramas superpuestos pre vs post.

ROIs: occipital (O1/O2) y temporal (T7/T8) — el segundo para capturar
la respuesta auditiva (tono sincrónico al cambio de luminancia).

Grid 2×2: alfa-occipital | beta-occipital
          alfa-temporal  | beta-temporal

Usage
-----
    micromamba run -n campeones python scripts/validation/38b_alpha_beta_histograms.py --subject 27
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
from scipy.signal import welch as scipy_welch
from scipy.stats import mannwhitneyu
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "sanity_checks"
SESSION = "vr"

WIDE_TMIN, WIDE_TMAX = -1.5, 1.5
BASELINE = (-1.5, -1.0)
PRE_START, PRE_END = -0.55, -0.05
POST_START, POST_END = 0.05, 0.55

ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

ROIS = {
    "Occipital (O1/O2)": ["O1", "O2"],
    "Temporal (T7/T8)":  ["T7", "T8"],
}

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


def load_pre_post_data(subject: str):
    """Returns (X_pre, X_post, sfreq, ch_names) concatenated across runs."""
    pre_list, post_list = [], []
    sfreq_out = None
    ch_names_out = None

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
        rows = df[df["trial_type"] == "CHANGE_PHOTO"]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets = onsets[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int),
                                   np.ones(len(onsets), int)])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        try:
            ep = mne.Epochs(raw, mne_ev, event_id={"CHANGE_PHOTO": 1},
                            tmin=WIDE_TMIN, tmax=WIDE_TMAX, picks=avail,
                            baseline=BASELINE, preload=True, verbose=False)
            pre_list.append(ep.copy().crop(PRE_START, PRE_END).get_data())
            post_list.append(ep.copy().crop(POST_START, POST_END).get_data())
            sfreq_out = sfreq
            ch_names_out = ep.ch_names
            print(f"  run-{run_id}: {len(ep)} onsets")
        except Exception as e:
            print(f"  run-{run_id}: {e}")

    X_pre = np.concatenate(pre_list, axis=0)   # (n, ch, t)
    X_post = np.concatenate(post_list, axis=0)
    return X_pre, X_post, sfreq_out, ch_names_out


def band_power(data: np.ndarray, sfreq: float, flo: float, fhi: float) -> np.ndarray:
    """Returns mean band power per epoch. data: (n, ch, t) → (n,)"""
    n_ep, n_ch, n_t = data.shape
    powers = np.empty((n_ep, n_ch))
    for i in range(n_ep):
        for c in range(n_ch):
            freqs, psd = scipy_welch(data[i, c], fs=sfreq, nperseg=n_t)
            mask = (freqs >= flo) & (freqs <= fhi)
            powers[i, c] = np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
    return powers.mean(axis=1)  # average over channels in ROI → (n,)


def plot_hist(ax, pre_vals, post_vals, band_name, roi_name):
    """Overlapping histogram + KDE + Mann-Whitney p-value."""
    stat, p = mannwhitneyu(post_vals, pre_vals, alternative="two-sided")
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"

    bins = np.histogram_bin_edges(np.concatenate([pre_vals, post_vals]), bins=20)

    ax.hist(pre_vals,  bins=bins, alpha=0.45, color="steelblue", density=True,
            label="pre")
    ax.hist(post_vals, bins=bins, alpha=0.45, color="tomato",    density=True,
            label="post")

    # KDE overlay
    for vals, color in [(pre_vals, "steelblue"), (post_vals, "tomato")]:
        if len(np.unique(vals)) > 1:
            xg = np.linspace(vals.min(), vals.max(), 200)
            kde = gaussian_kde(vals)
            ax.plot(xg, kde(xg), color=color, lw=1.8)

    ax.set_title(f"{band_name} — {roi_name}\n{p_str} (Mann-Whitney)", fontsize=9)
    ax.set_xlabel("Power (µV²)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pre/post epochs for sub-{sub}...")
    X_pre, X_post, sfreq, ch_names = load_pre_post_data(sub)
    print(f"  Total: {X_pre.shape[0]} pre + {X_post.shape[0]} post epochs")

    roi_names = list(ROIS.keys())
    bands = [("Alpha (8–13 Hz)", *ALPHA_BAND), ("Beta (13–30 Hz)", *BETA_BAND)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"sub-{sub} — Alpha/Beta power: pre vs post (27b windows)\n"
        "Sanity check 3.2",
        fontsize=11,
    )

    for row, (band_name, flo, fhi) in enumerate(bands):
        for col, roi_name in enumerate(roi_names):
            roi_chs = [ch for ch in ROIS[list(ROIS.keys())[col]] if ch in ch_names]
            if not roi_chs:
                axes[row, col].set_visible(False)
                continue
            ch_idx = [ch_names.index(ch) for ch in roi_chs]
            pre_roi  = X_pre[:, ch_idx, :]
            post_roi = X_post[:, ch_idx, :]
            pre_pow  = band_power(pre_roi,  sfreq, flo, fhi)
            post_pow = band_power(post_roi, sfreq, flo, fhi)
            plot_hist(axes[row, col], pre_pow, post_pow, band_name, roi_name)

    plt.tight_layout()
    out_path = out_dir / f"sub-{sub}_38b_alpha_beta_histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
