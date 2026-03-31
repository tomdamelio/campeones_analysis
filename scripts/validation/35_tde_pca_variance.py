#!/usr/bin/env python
"""Plot TDE-PCA explained variance vs number of components (1 to 100).

Loads all runs for a subject, applies TDE (±10 lags), fits PCA(n_components=100)
on the concatenated TDE data, and plots:
  - Cumulative explained variance (%)
  - Marginal explained variance per component (%)
  - Vertical line at current setting (TDE_PCA_COMPONENTS=17)

Usage
-----
    micromamba run -n campeones python scripts/validation/35_tde_pca_variance.py --subject 27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "tde_pca_variance"

SESSION = "vr"
WIDE_TMIN = -1.5
WIDE_TMAX = 2.0
BASELINE = (-1.5, -1.0)
TDE_WINDOW_HALF = 10
TDE_PCA_COMPONENTS_CURRENT = 17   # current setting to mark on plot
N_COMPONENTS_MAX = 100

EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]

RUNS_CONFIG: dict[str, list[dict]] = {
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


def load_tde_data(subject: str) -> np.ndarray:
    """Load all runs, apply TDE, return concatenated (n_timepoints, n_tde_features)."""
    from campeones_analysis.luminance.tde_glhmm import apply_tde_only

    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    all_tde_segments = []
    total_t = 0

    for rc in runs:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        label = f"task-{task}_acq-{acq}_run-{run_id}"

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
            print(f"  SKIP {label}")
            continue

        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        events_df = pd.read_csv(tsv, sep="\t")
        change_rows = events_df[events_df["trial_type"] == "CHANGE_PHOTO"]

        if change_rows.empty:
            continue

        sfreq = raw.info["sfreq"]
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets_s = change_rows["onset"].astype(float).values
        samples = np.round(onsets_s * sfreq).astype(int) + raw.first_samp

        last_sample = raw.first_samp + raw.n_times
        valid = (samples >= raw.first_samp) & (samples < last_sample)
        samples = samples[valid]
        if len(samples) == 0:
            continue

        mne_events = np.column_stack([
            samples,
            np.zeros(len(samples), dtype=int),
            np.ones(len(samples), dtype=int),
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        try:
            epochs = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=WIDE_TMIN, tmax=WIDE_TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            data_wide = epochs.get_data()  # (n_epochs, n_ch, n_times)
            n_epochs = data_wide.shape[0]

            for i in range(n_epochs):
                epoch_data = data_wide[i].T  # (n_times, n_ch)
                n_t = epoch_data.shape[0]
                indices = np.array([[0, n_t]])
                tde_data, _ = apply_tde_only(epoch_data, indices, TDE_WINDOW_HALF)
                all_tde_segments.append(tde_data)

            n_t_tde = sum(s.shape[0] for s in all_tde_segments[-n_epochs:])
            total_t += n_t_tde
            print(f"  {label}: {n_epochs} epochs → {n_t_tde} TDE timepoints")

        except Exception as exc:
            print(f"  {label}: error — {exc}")

    if not all_tde_segments:
        print("No TDE data loaded.")
        sys.exit(1)

    X = np.vstack(all_tde_segments)
    print(f"\n  Total TDE data: {X.shape[0]} timepoints × {X.shape[1]} features")
    return X


def run(subject: str) -> None:
    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TDE data for sub-{subject}...")
    X = load_tde_data(subject)

    n_max = min(N_COMPONENTS_MAX, X.shape[0] - 1, X.shape[1])
    print(f"\nFitting PCA({n_max}) on {X.shape[0]} × {X.shape[1]} TDE matrix...")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=n_max)
    pca.fit(X_sc)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    margvar = pca.explained_variance_ratio_ * 100
    n_comp = np.arange(1, n_max + 1)

    # Find elbow: largest drop in marginal variance
    drops = np.diff(margvar)
    elbow_idx = int(np.argmin(drops)) + 1  # 1-indexed component after which drop is largest

    # Find where cumvar first exceeds common thresholds
    thresholds = [80, 90, 95, 99]
    thresh_crossings = {}
    for t in thresholds:
        idx = np.searchsorted(cumvar, t)
        thresh_crossings[t] = int(idx) + 1 if idx < len(cumvar) else n_max

    print(f"\n  Current setting: {TDE_PCA_COMPONENTS_CURRENT} PCs → "
          f"{cumvar[TDE_PCA_COMPONENTS_CURRENT-1]:.1f}% cumulative variance")
    print(f"  Variance thresholds:")
    for t, n in thresh_crossings.items():
        print(f"    {t}%: {n} PCs")
    print(f"  Largest marginal drop after PC {elbow_idx} "
          f"({margvar[elbow_idx-1]:.2f}% → {margvar[elbow_idx]:.2f}%)")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: cumulative
    ax = axes[0]
    ax.plot(n_comp, cumvar, color="steelblue", lw=2)
    ax.axvline(TDE_PCA_COMPONENTS_CURRENT, color="crimson", ls="--", lw=1.5,
               label=f"Current: {TDE_PCA_COMPONENTS_CURRENT} PCs "
                     f"({cumvar[TDE_PCA_COMPONENTS_CURRENT-1]:.1f}%)")
    for t, n in thresh_crossings.items():
        ax.axhline(t, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(n_max * 0.98, t + 0.5, f"{t}% @ {n} PCs",
                ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"TDE-PCA explained variance — sub-{subject} (all runs, standardised)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)

    # Bottom: marginal
    ax = axes[1]
    ax.bar(n_comp, margvar, color="steelblue", alpha=0.7, width=0.8)
    ax.axvline(TDE_PCA_COMPONENTS_CURRENT, color="crimson", ls="--", lw=1.5,
               label=f"Current: {TDE_PCA_COMPONENTS_CURRENT} PCs")
    ax.axvline(elbow_idx, color="darkorange", ls=":", lw=1.5,
               label=f"Largest drop after PC {elbow_idx}")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Marginal explained variance (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = output_dir / f"sub-{subject}_tde_pca_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out}")

    # Save CSV with variance values
    import csv
    csv_path = output_dir / f"sub-{subject}_tde_pca_variance.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_components", "marginal_var_pct", "cumulative_var_pct"])
        for i in range(n_max):
            w.writerow([i + 1, round(float(margvar[i]), 4), round(float(cumvar[i]), 4)])
    print(f"  CSV saved: {csv_path}")
    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TDE-PCA explained variance vs number of components"
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.subject)
