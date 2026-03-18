#!/usr/bin/env python
"""Decode CHANGE vs NO_CHANGE per post-onset window (Task 10.1).

For each of the 4 post-onset windows (50-100, 100-150, 150-200, 200-250 ms),
trains an independent model using:
  - CHANGE: 74 micro-epochs from that specific window (one per onset)
  - NO_CHANGE: 74 micro-epochs sampled from the 4 pre-onset windows
    (balanced: ~18-19 from each pre-window, preserving run structure)

This identifies which post-stimulus moment maximizes decoding performance.

Feature sets:
  1. bandpower_filtered: filter on long epoch → segment → variance = 160 features
  2. raw_signal: baseline-corrected long epoch → segment → vectorize = 384 features

Usage
-----
    micromamba run -n campeones python scripts/validation/32_decoding_per_window.py --subject 27
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
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_per_window"

SESSION = "vr"

# Long epoch parameters (same as Task 9/10)
TMIN = -2.5
TMAX = 2.0
BASELINE = (-2.5, -1.5)

# Post-onset windows to evaluate independently
POST_WINDOWS = [
    (0.05, 0.10, "50-100ms"),
    (0.10, 0.15, "100-150ms"),
    (0.15, 0.20, "150-200ms"),
    (0.20, 0.25, "200-250ms"),
]

# Pre-onset windows (pool for NO_CHANGE sampling)
PRE_WINDOWS = [
    (-0.25, -0.20),
    (-0.20, -0.15),
    (-0.15, -0.10),
    (-0.10, -0.05),
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
FIXED_C = 0.001
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Epoch loading (long epochs, same as Task 10)
# ---------------------------------------------------------------------------

def load_long_epochs_per_run(subject: str) -> list[tuple[mne.Epochs, str]]:
    """Load preprocessed EEG and create long epochs per run.

    Only CHANGE_PHOTO onsets are epoched.
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
        change_rows = events_df[events_df["trial_type"] == "CHANGE_PHOTO"]

        if change_rows.empty:
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
            print(f"  {label}: {len(epochs)} CHANGE_PHOTO onsets")
            all_runs.append((epochs, label))
        except Exception as exc:
            print(f"  {label}: epoch error — {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Feature extraction per window
# ---------------------------------------------------------------------------

def _time_to_sample(t: float, sfreq: float, tmin: float) -> int:
    """Convert time (relative to onset) to sample index within epoch."""
    return int(np.round((t - tmin) * sfreq))


def extract_single_window_bandpower(
    epochs: mne.Epochs,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Filter long epoch per band, extract variance in one window.

    Returns: (n_epochs, n_ch * n_bands)
    """
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin
    win_samples = int(np.round((t_end - t_start) * sfreq))
    s0 = _time_to_sample(t_start, sfreq, tmin)
    s1 = s0 + win_samples

    band_list = list(SPECTRAL_BANDS.items())
    n_bands = len(band_list)
    n_ch = len(epochs.ch_names)
    n_ep = len(epochs)
    features = np.empty((n_ep, n_ch * n_bands))

    for b_idx, (band_name, (flo, fhi)) in enumerate(band_list):
        filtered = epochs.copy().filter(l_freq=flo, h_freq=fhi, verbose=False)
        filt_data = filtered.get_data()  # (n_ep, n_ch, n_times)
        for i in range(n_ep):
            segment = filt_data[i, :, s0:s1]  # (n_ch, win_samples)
            features[i, b_idx * n_ch:(b_idx + 1) * n_ch] = np.var(segment, axis=1)

    return features


def extract_single_window_raw(
    epochs: mne.Epochs,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Extract raw signal from one window of baseline-corrected epoch.

    Returns: (n_epochs, n_ch * win_samples)
    """
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin
    win_samples = int(np.round((t_end - t_start) * sfreq))
    s0 = _time_to_sample(t_start, sfreq, tmin)
    s1 = s0 + win_samples

    data = epochs.get_data()  # (n_ep, n_ch, n_times)
    segments = data[:, :, s0:s1]  # (n_ep, n_ch, win_samples)
    return segments.reshape(segments.shape[0], -1)


# ---------------------------------------------------------------------------
# Build per-window datasets with matched NO_CHANGE
# ---------------------------------------------------------------------------

def build_per_window_datasets(
    long_epochs_runs: list[tuple[mne.Epochs, str]],
    feature_type: str,
) -> dict[str, list[tuple[np.ndarray, np.ndarray, str]]]:
    """For each post-onset window, build CHANGE vs NO_CHANGE dataset.

    CHANGE: micro-epochs from that specific post-onset window (1 per onset).
    NO_CHANGE: same number of micro-epochs sampled from the 4 pre-onset
    windows, balanced across windows, preserving run structure.

    Optimized: filters each run once per band, then extracts all windows.

    Returns: {window_label: [(features, labels, run_label), ...]}
    """
    rng = np.random.RandomState(RANDOM_SEED)

    # All windows we need to extract
    all_windows = (
        [(pw[0], pw[1], pw[2], "change") for pw in POST_WINDOWS]
        + [(pre[0], pre[1], f"pre_{i}", "nochange")
           for i, pre in enumerate(PRE_WINDOWS)]
    )

    # Pre-extract features for all windows per run (filter once per band)
    # Structure: run_feats[run_idx][window_key] = (n_ep, n_feat)
    run_feats_all: list[dict[str, np.ndarray]] = []
    run_labels_list: list[str] = []
    run_n_epochs: list[int] = []

    for epochs, run_label in long_epochs_runs:
        sfreq = epochs.info["sfreq"]
        tmin = epochs.tmin
        n_ep = len(epochs)
        n_ch = len(epochs.ch_names)
        run_labels_list.append(run_label)
        run_n_epochs.append(n_ep)

        window_feats: dict[str, np.ndarray] = {}

        # Fixed window length (avoid rounding inconsistencies)
        win_samples = int(np.round(
            (POST_WINDOWS[0][1] - POST_WINDOWS[0][0]) * sfreq
        ))

        if feature_type == "bandpower_filtered":
            band_list = list(SPECTRAL_BANDS.items())
            n_bands = len(band_list)

            # Filter once per band, extract all windows
            for b_idx, (band_name, (flo, fhi)) in enumerate(band_list):
                filtered = epochs.copy().filter(
                    l_freq=flo, h_freq=fhi, verbose=False)
                filt_data = filtered.get_data()

                for t_start, t_end, w_key, _ in all_windows:
                    s0 = _time_to_sample(t_start, sfreq, tmin)
                    s1 = s0 + win_samples

                    if w_key not in window_feats:
                        window_feats[w_key] = np.empty((n_ep, n_ch * n_bands))

                    for i in range(n_ep):
                        segment = filt_data[i, :, s0:s1]
                        window_feats[w_key][
                            i, b_idx * n_ch:(b_idx + 1) * n_ch
                        ] = np.var(segment, axis=1)

        else:  # raw_signal
            data = epochs.get_data()
            for t_start, t_end, w_key, _ in all_windows:
                s0 = _time_to_sample(t_start, sfreq, tmin)
                s1 = s0 + win_samples
                segments = data[:, :, s0:s1]
                window_feats[w_key] = segments.reshape(n_ep, -1)

        run_feats_all.append(window_feats)

    # Now assemble per-window datasets
    result: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {}
    pre_keys = [f"pre_{i}" for i in range(len(PRE_WINDOWS))]

    for pw_start, pw_end, pw_label in POST_WINDOWS:
        runs_data = []

        for r_idx in range(len(long_epochs_runs)):
            n_ep = run_n_epochs[r_idx]
            run_label = run_labels_list[r_idx]
            feats = run_feats_all[r_idx]

            change_feats = feats[pw_label]  # (n_ep, n_feat)

            # Pool all pre-window features
            pre_pool = np.vstack([feats[pk] for pk in pre_keys])
            indices = rng.choice(len(pre_pool), size=n_ep, replace=False)
            nochange_feats = pre_pool[indices]

            X = np.vstack([change_feats, nochange_feats])
            y = np.concatenate([np.ones(n_ep), np.zeros(n_ep)]).astype(int)
            runs_data.append((X, y, run_label))

        result[pw_label] = runs_data

    return result



# ---------------------------------------------------------------------------
# Classification (LORO CV)
# ---------------------------------------------------------------------------

def _build_pipeline(c_val: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_val, max_iter=1000, random_state=RANDOM_SEED),
    )


def _select_best_c(X_train: np.ndarray, y_train: np.ndarray,
                   fixed_c: float | None = None) -> float:
    if fixed_c is not None:
        return fixed_c
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
    window_label: str,
    fixed_c: float | None = None,
) -> dict:
    """Leave-One-Run-Out CV."""
    n_runs = len(runs_data)
    if n_runs < 2:
        return {}

    n_feat = runs_data[0][0].shape[1]
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []

    for test_idx in range(n_runs):
        X_train = np.vstack([runs_data[i][0] for i in range(n_runs) if i != test_idx])
        y_train = np.concatenate([runs_data[i][1] for i in range(n_runs) if i != test_idx])
        X_test = runs_data[test_idx][0]
        y_test = runs_data[test_idx][1]

        best_c = _select_best_c(X_train, y_train, fixed_c=fixed_c)
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
        "window": window_label,
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
    """Line plot: accuracy and AUC per window, one line per feature set."""
    feature_names = sorted(set(r["feature"] for r in all_results))
    window_labels = [pw[2] for pw in POST_WINDOWS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax, title in [
        ("accuracy", axes[0], "Accuracy"),
        ("auc_roc", axes[1], "AUC-ROC"),
    ]:
        for feat in feature_names:
            vals = []
            for wl in window_labels:
                r = next(
                    (r for r in all_results
                     if r["feature"] == feat and r["window"] == wl),
                    None,
                )
                vals.append(r[metric] if r else np.nan)
            ax.plot(window_labels, vals, "o-", label=feat, linewidth=2, markersize=8)

        ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax.set_ylim(0.35, 0.85)
        ax.set_xlabel("Post-onset window")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per window (sub-{subject})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{subject}_per_window_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Summary plot saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str, fixed_c: float | None = FIXED_C) -> None:
    print("=" * 60)
    print(f"32 — Per-window micro-epoch decoding — sub-{subject}")
    print(f"     Post-onset windows: {[pw[2] for pw in POST_WINDOWS]}")
    print(f"     Pre-onset pool: {PRE_WINDOWS}")
    if fixed_c is not None:
        print(f"     Fixed C={fixed_c} (no inner CV)")
    else:
        print(f"     C grid search: {C_GRID} (inner CV)")
    print("=" * 60)

    long_epochs = load_long_epochs_per_run(subject)
    if len(long_epochs) < 2:
        print("Need at least 2 runs for LORO CV.")
        sys.exit(1)

    total_onsets = sum(len(ep) for ep, _ in long_epochs)
    print(f"\n  Total CHANGE_PHOTO onsets: {total_onsets}")
    print(f"  Per-window: {total_onsets} CHANGE + {total_onsets} NO_CHANGE "
          f"= {total_onsets * 2} per model\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for feat_type in ["bandpower_filtered", "raw_signal"]:
        print(f"  === Feature set: {feat_type} ===")
        print(f"  Building per-window datasets...")
        datasets = build_per_window_datasets(long_epochs, feat_type)

        for pw_start, pw_end, pw_label in POST_WINDOWS:
            print(f"\n    Window {pw_label}:")
            runs_data = datasets[pw_label]
            res = run_loro(runs_data, feat_type, pw_label, fixed_c=fixed_c)
            if res:
                all_results.append(res)
                print(f"      Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  "
                      f"AUC={res['auc_roc']:.3f}  "
                      f"({res['n_change']} CH + {res['n_no_change']} NC)")
                for f in res["folds"]:
                    print(f"        {f['fold']}: acc={f['accuracy']:.3f} "
                          f"train={f['train_accuracy']:.3f} "
                          f"(C={f['C']}, n_tr={f['n_train']}, n_te={f['n_test']})")

    # Save results JSON
    json_path = output_dir / f"sub-{subject}_per_window_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    if all_results:
        plot_summary(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-window micro-epoch decoding (Task 10.1)",
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--fixed-c", type=float, default=FIXED_C,
                        help="Fixed C for LogisticRegression (default: 0.001). "
                             "Set to 0 to enable inner CV grid search instead.")
    parser.add_argument("--inner-cv", action="store_true",
                        help="Enable inner CV grid search for C (overrides --fixed-c).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fixed_c = None if args.inner_cv else args.fixed_c
    run_pipeline(subject=args.subject, fixed_c=fixed_c)
