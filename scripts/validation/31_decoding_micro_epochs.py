#!/usr/bin/env python
"""Decode CHANGE vs NO_CHANGE using 50ms micro-epochs (Enzo's design).

Implements Task 10 from the research diary.

For each CHANGE_PHOTO onset (0 ms), extracts:
  - 4 CHANGE windows: [50,100], [100,150], [150,200], [200,250] ms post-onset
  - 4 NO_CHANGE windows: [-50,-100], [-100,-150], [-150,-200], [-200,-250] ms pre-onset

Strategy for bandpower: first epoch the signal with a long window
(TMIN=-2.5, TMAX=2.0 with baseline correction), then bandpass-filter
in each spectral band over the full epoch, and finally segment into
50ms micro-windows to compute power (variance). This gives full
spectral resolution despite the short micro-epoch duration.

Leave-One-Run-Out CV ensures all micro-epochs from the same stimulus
stay in the same fold (guaranteed by run-level splitting).

Feature sets:
  1. bandpower_filtered: filter on long epoch → segment → variance per band/ch = 160 features
  2. raw_signal: segment baseline-corrected epoch → vectorize = 32 ch × ~13 samples

Usage
-----
    micromamba run -n campeones python scripts/validation/31_decoding_micro_epochs.py --subject 27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_micro"

SESSION = "vr"

# Long epoch parameters (same as Task 9 — for baseline correction & filtering)
TMIN = -2.5
TMAX = 2.0
BASELINE = (-2.5, -1.5)

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

# Micro-epoch window definitions (seconds relative to onset at t=0)
CHANGE_WINDOWS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25)]
NO_CHANGE_WINDOWS = [(-0.25, -0.20), (-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05)]

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

EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]

SPECTRAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Epoch loading (long epochs, same as Task 9)
# ---------------------------------------------------------------------------

def load_long_epochs_per_run(subject: str) -> list[tuple[mne.Epochs, str]]:
    """Load preprocessed EEG and create long epochs per run.

    Uses CHANGE_PHOTO onsets only. Returns list of (epochs, run_label).
    Only CHANGE_PHOTO events are epoched — the micro-epoch windows
    around each onset define CHANGE (post) and NO_CHANGE (pre).
    """
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    all_runs: list[tuple[mne.Epochs, str]] = []

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
            print(f"  SKIP {label} — files not found")
            continue

        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        events_df = pd.read_csv(tsv, sep="\t")
        # Only CHANGE_PHOTO onsets — we derive both classes from these
        change_rows = events_df[events_df["trial_type"] == "CHANGE_PHOTO"]

        if change_rows.empty:
            print(f"  SKIP {label} — no CHANGE_PHOTO events")
            continue

        sfreq = raw.info["sfreq"]
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets_s = change_rows["onset"].astype(float).values
        samples = np.round(onsets_s * sfreq).astype(int) + raw.first_samp

        n_total = raw.n_times
        last_sample = raw.first_samp + n_total
        valid = (samples >= raw.first_samp) & (samples < last_sample)
        samples = samples[valid]

        if len(samples) == 0:
            continue

        # All events get id=1 (CHANGE_PHOTO) — we only need the onset
        mne_events = np.column_stack([
            samples,
            np.zeros(len(samples), dtype=int),
            np.ones(len(samples), dtype=int),
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        try:
            epochs = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=TMIN, tmax=TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            n_ep = len(epochs)
            print(f"  {label}: {n_ep} CHANGE_PHOTO onsets (→ {n_ep*4} CHANGE "
                  f"+ {n_ep*4} NO_CHANGE micro-epochs)")
            all_runs.append((epochs, label))
        except Exception as exc:
            print(f"  {label}: epoch error — {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Micro-epoch segmentation from long epochs
# ---------------------------------------------------------------------------

def _time_to_sample(t: float, sfreq: float, tmin: float) -> int:
    """Convert time (relative to onset) to sample index within epoch."""
    return int(np.round((t - tmin) * sfreq))


def segment_micro_epochs(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Segment long epochs into micro-epochs.

    For each long epoch (centered on CHANGE_PHOTO onset):
      - 4 CHANGE windows (post-onset)
      - 4 NO_CHANGE windows (pre-onset)

    Returns:
        X: (n_micro, n_ch, win_samples)
        y: (n_micro,)  — 1=CHANGE, 0=NO_CHANGE
    """
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin

    # Fixed window length in samples
    win_samples = int(np.round((CHANGE_WINDOWS[0][1] - CHANGE_WINDOWS[0][0]) * sfreq))

    micro_list = []
    label_list = []

    for ep_idx in range(data.shape[0]):
        ep_data = data[ep_idx]  # (n_ch, n_times)

        for t_start, _t_end in CHANGE_WINDOWS:
            s0 = _time_to_sample(t_start, sfreq, tmin)
            s1 = s0 + win_samples
            if s1 <= ep_data.shape[1]:
                micro_list.append(ep_data[:, s0:s1])
                label_list.append(1)

        for t_start, _t_end in NO_CHANGE_WINDOWS:
            s0 = _time_to_sample(t_start, sfreq, tmin)
            s1 = s0 + win_samples
            if s1 <= ep_data.shape[1]:
                micro_list.append(ep_data[:, s0:s1])
                label_list.append(0)

    return np.array(micro_list), np.array(label_list)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_bandpower_filtered(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass-filter long epoch per band, then segment into micro-epochs.

    For each spectral band:
      1. Filter the full long epoch (plenty of samples for good filter)
      2. Segment into micro-windows
      3. Compute variance (= power) per channel per micro-window

    This gives full spectral resolution despite 50ms micro-epochs.

    Returns:
        features: (n_micro, n_ch * n_bands)
        y: (n_micro,)
    """
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin
    band_list = list(SPECTRAL_BANDS.items())
    n_bands = len(band_list)
    n_ch = len(epochs.ch_names)

    # Fixed window length
    win_samples = int(np.round((CHANGE_WINDOWS[0][1] - CHANGE_WINDOWS[0][0]) * sfreq))

    # First pass: determine n_micro from one epoch
    data_raw = epochs.get_data()
    n_epochs_long = data_raw.shape[0]

    # Count micro-epochs per long epoch
    windows_all = list(CHANGE_WINDOWS) + list(NO_CHANGE_WINDOWS)
    labels_per_window = [1] * len(CHANGE_WINDOWS) + [0] * len(NO_CHANGE_WINDOWS)
    n_windows = len(windows_all)
    n_micro = n_epochs_long * n_windows

    features = np.empty((n_micro, n_ch * n_bands))
    y = np.empty(n_micro, dtype=int)

    # Build y labels
    for ep_idx in range(n_epochs_long):
        for w_idx, lbl in enumerate(labels_per_window):
            y[ep_idx * n_windows + w_idx] = lbl

    # For each band: filter full epochs, then segment
    for b_idx, (band_name, (flo, fhi)) in enumerate(band_list):
        # Filter all epochs at once (MNE operates on the full time series)
        filtered = epochs.copy().filter(
            l_freq=flo, h_freq=fhi, verbose=False,
        )
        filt_data = filtered.get_data()  # (n_epochs, n_ch, n_times)

        for ep_idx in range(n_epochs_long):
            for w_idx, (t_start, _t_end) in enumerate(windows_all):
                s0 = _time_to_sample(t_start, sfreq, tmin)
                s1 = s0 + win_samples
                micro_idx = ep_idx * n_windows + w_idx
                # Power = variance of filtered signal in this window
                segment = filt_data[ep_idx, :, s0:s1]  # (n_ch, win_samples)
                features[micro_idx, b_idx * n_ch:(b_idx + 1) * n_ch] = (
                    np.var(segment, axis=1)
                )

    return features, y


def extract_raw_micro(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Segment baseline-corrected long epochs into raw micro-epochs.

    Returns:
        features: (n_micro, n_ch * win_samples)
        y: (n_micro,)
    """
    X, y = segment_micro_epochs(epochs)
    return X.reshape(X.shape[0], -1), y


# ---------------------------------------------------------------------------
# Classification with Leave-One-Run-Out CV
# ---------------------------------------------------------------------------

def _build_pipeline(c_val: float):
    """Build sklearn pipeline: StandardScaler + LogisticRegression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_val, max_iter=1000, random_state=RANDOM_SEED),
    )


def _select_best_c(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Inner stratified CV to select best C."""
    n_splits = min(3, max(2, len(np.unique(y_train))))
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=RANDOM_SEED)
    best_c, best_score = C_GRID[0], -1.0
    for c_val in C_GRID:
        scores = []
        try:
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                pipe = _build_pipeline(c_val)
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                scores.append(pipe.score(X_train[val_idx], y_train[val_idx]))
        except ValueError:
            continue
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val
    return best_c


def run_loro(
    runs_data: list[tuple[np.ndarray, np.ndarray, str]],
    feature_name: str,
) -> dict:
    """Leave-One-Run-Out CV.

    Args:
        runs_data: list of (features, labels, run_label)
        feature_name: name for reporting
    """
    n_runs = len(runs_data)
    if n_runs < 2:
        print(f"    {feature_name}: need >= 2 runs, got {n_runs}")
        return {}

    n_feat = runs_data[0][0].shape[1]
    print(f"    N features: {n_feat}")

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []

    for test_idx in range(n_runs):
        X_train = np.vstack([runs_data[i][0] for i in range(n_runs) if i != test_idx])
        y_train = np.concatenate([runs_data[i][1] for i in range(n_runs) if i != test_idx])
        X_test = runs_data[test_idx][0]
        y_test = runs_data[test_idx][1]

        best_c = _select_best_c(X_train, y_train)

        pipe = _build_pipeline(best_c)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        train_acc = float(pipe.score(X_train, y_train))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_metrics.append({
            "fold": runs_data[test_idx][2],
            "C": best_c,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "train_accuracy": train_acc,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        })

    y_true = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    return {
        "feature": feature_name,
        "n_features": int(n_feat),
        "accuracy": float(accuracy_score(y_true, y_pred_arr)),
        "precision": float(precision_score(y_true, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_arr, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob_arr)),
        "n_change": int(y_true.sum()),
        "n_no_change": int((1 - y_true).sum()),
        "folds": fold_metrics,
    }


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_summary(all_results: list[dict], output_dir: Path, subject: str) -> None:
    """Bar chart of metrics across feature sets."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = [r["feature"] for r in all_results]
    x = np.arange(len(metrics))
    width = 0.8 / len(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall metrics
    ax = axes[0]
    for i, r in enumerate(all_results):
        vals = [r[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=r["feature"])
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_ylabel("Score")
    ax.set_title(f"Micro-epoch decoding (sub-{subject})")
    ax.legend(fontsize=8)

    # Right: per-fold accuracy
    ax = axes[1]
    for i, r in enumerate(all_results):
        fold_accs = [f["accuracy"] for f in r["folds"]]
        fold_names = [f["fold"].split("_")[0] for f in r["folds"]]
        x_folds = np.arange(len(fold_accs))
        ax.bar(x_folds + i * width, fold_accs, width, label=r["feature"])
    ax.set_xticks(x_folds + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(fold_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-fold accuracy (LORO)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{subject}_micro_decoding_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Summary plot saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    print("=" * 60)
    print(f"31 — Micro-epoch decoding — sub-{subject}")
    print(f"     Long epoch: TMIN={TMIN}, TMAX={TMAX}, BASELINE={BASELINE}")
    print(f"     CHANGE windows: {CHANGE_WINDOWS}")
    print(f"     NO_CHANGE windows: {NO_CHANGE_WINDOWS}")
    print(f"     Channels: {len(EEG_CHANNELS)}")
    print("=" * 60)

    long_epochs = load_long_epochs_per_run(subject)
    if len(long_epochs) < 2:
        print("Need at least 2 runs for LORO CV.")
        sys.exit(1)

    total_onsets = sum(len(ep) for ep, _ in long_epochs)
    print(f"\n  Total CHANGE_PHOTO onsets: {total_onsets}")
    print(f"  Expected micro-epochs: {total_onsets * 4} CHANGE + "
          f"{total_onsets * 4} NO_CHANGE = {total_onsets * 8}\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # --- 1. Bandpower (filtered on long epoch, then segmented) ---
    print("  Feature set: bandpower_filtered")
    print("    Filtering long epochs per band, then segmenting...")
    bp_runs = []
    for epochs, label in long_epochs:
        feats, y = extract_bandpower_filtered(epochs)
        bp_runs.append((feats, y, label))
        n_ch = int(y.sum())
        n_nc = int((1 - y).sum())
        print(f"    {label}: {n_ch} CHANGE + {n_nc} NO_CHANGE")

    res = run_loro(bp_runs, "bandpower_filtered")
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} train_acc={f['train_accuracy']:.3f} "
                  f"(C={f['C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # --- 2. Raw signal (baseline-corrected, then segmented) ---
    print(f"\n  Feature set: raw_signal")
    raw_runs = []
    for epochs, label in long_epochs:
        feats, y = extract_raw_micro(epochs)
        raw_runs.append((feats, y, label))

    res = run_loro(raw_runs, "raw_signal")
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} train_acc={f['train_accuracy']:.3f} "
                  f"(C={f['C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # Save results JSON
    json_path = output_dir / f"sub-{subject}_micro_decoding_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    if all_results:
        plot_summary(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode CHANGE vs NO_CHANGE from 50ms micro-epochs",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
