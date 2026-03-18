#!/usr/bin/env python
"""Decode CHANGE_PHOTO vs NO_CHANGE_PHOTO from EEG epochs.

Implements Task 9 from the research diary.

Re-epochs the preprocessed EEG with decoding-specific parameters
(TMIN=-2.5, TMAX=2.0, BASELINE=(-2.5, -1.5)), then runs binary
classification using all 32 EEG channels with Leave-One-Run-Out CV.

Feature sets (aligned with scripts/modeling/ pipeline):
  1. Bandpower (Welch):  5 spectral bands × 32 channels (script 11 approach)
  2. TDE + Cov:  GLHMM TDE → PCA(20) → covariance upper triangle (script 13)
     PCA is fit on train data only within each CV fold to avoid leakage.
  3. Raw + PCA:  raw EEG vectorized → PCA(100) inside sklearn pipeline (script 10)

Classifier: LogisticRegression (L2, C grid search via inner CV).

Usage
-----
    micromamba run -n campeones python scripts/validation/27_decoding_photo_change.py --subject 27
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
from scipy.signal import welch as scipy_welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path for project library imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding"

SESSION = "vr"

# Decoding-specific epoch parameters (wide epochs)
TMIN = -2.5
TMAX = 2.0
BASELINE = (-2.5, -1.5)

# Focused epoch parameters (1s windows centred on discriminative signal)
FOCUSED_TMIN = -1.5
FOCUSED_TMAX = 1.5
FOCUSED_BASELINE = (-1.5, -1.0)
FOCUSED_CROP_START = 0.05
FOCUSED_CROP_END = 1.05  # default: 1s post-onset

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

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

# Classifier hyperparameters
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RANDOM_SEED = 42

# TDE parameters (from config_luminance.py / script 13)
TDE_WINDOW_HALF = 10  # ±10 timepoints → 21 total
TDE_PCA_COMPONENTS = 20

# PCA for raw signal model (from script 10)
RAW_PCA_COMPONENTS = 100


# ---------------------------------------------------------------------------
# Epoch loading — re-epoch from raw with decoding parameters
# ---------------------------------------------------------------------------

def load_epochs_per_run(subject: str) -> list[tuple[mne.Epochs, str]]:
    """Load preprocessed EEG and create epochs per run.

    Returns list of (epochs, run_label) tuples for Leave-One-Run-Out CV.
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
        photo_rows = events_df[events_df["trial_type"].isin(EVENT_ID.keys())]

        if photo_rows.empty:
            continue

        sfreq = raw.info["sfreq"]
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets_s = photo_rows["onset"].astype(float).values
        event_ids = photo_rows["trial_type"].map(EVENT_ID).values
        samples = np.round(onsets_s * sfreq).astype(int) + raw.first_samp

        n_total = raw.n_times
        last_sample = raw.first_samp + n_total
        valid = (samples >= raw.first_samp) & (samples < last_sample)
        samples, event_ids = samples[valid], event_ids[valid]

        if len(samples) == 0:
            continue

        mne_events = np.column_stack([
            samples, np.zeros(len(samples), dtype=int), event_ids,
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        present = set(photo_rows["trial_type"].unique())
        run_event_id = {k: v for k, v in EVENT_ID.items() if k in present}

        try:
            epochs = mne.Epochs(
                raw, events=mne_events, event_id=run_event_id,
                tmin=TMIN, tmax=TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            all_runs.append((epochs, label))
            n_ch = sum(1 for e in epochs.events[:, 2] if e == 1)
            n_nc = sum(1 for e in epochs.events[:, 2] if e == 2)
            print(f"  {label}: {n_ch} CHANGE + {n_nc} NO_CHANGE")
        except Exception as exc:
            print(f"  {label}: epoch error — {exc}")

    return all_runs


def load_epochs_per_run_focused(subject: str,
                                crop_end: float = FOCUSED_CROP_END,
                                ) -> list[tuple[mne.Epochs, str]]:
    """Load epochs with focused windows around the discriminative signal.

    Epochs wide (-1.5 to 1.5s) with baseline (-1.5, -1.0), then
    both conditions cropped to [0.05, crop_end]s post-onset.
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
        photo_rows = events_df[events_df["trial_type"].isin(EVENT_ID.keys())]

        if photo_rows.empty:
            continue

        sfreq = raw.info["sfreq"]
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets_s = photo_rows["onset"].astype(float).values
        event_ids = photo_rows["trial_type"].map(EVENT_ID).values
        samples = np.round(onsets_s * sfreq).astype(int) + raw.first_samp

        n_total = raw.n_times
        last_sample = raw.first_samp + n_total
        valid = (samples >= raw.first_samp) & (samples < last_sample)
        samples, event_ids = samples[valid], event_ids[valid]

        if len(samples) == 0:
            continue

        mne_events = np.column_stack([
            samples, np.zeros(len(samples), dtype=int), event_ids,
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        present = set(photo_rows["trial_type"].unique())
        run_event_id = {k: v for k, v in EVENT_ID.items() if k in present}

        try:
            # Epoch wide, apply baseline, then crop to focused window
            epochs_wide = mne.Epochs(
                raw, events=mne_events, event_id=run_event_id,
                tmin=FOCUSED_TMIN, tmax=FOCUSED_TMAX, picks=available_chs,
                baseline=FOCUSED_BASELINE, preload=True, verbose=False,
            )

            # Crop both conditions to same window post-onset
            epochs = epochs_wide.crop(
                tmin=FOCUSED_CROP_START, tmax=crop_end)

            win_ms = int(round((crop_end - FOCUSED_CROP_START) * 1000))

            all_runs.append((epochs, label))
            n_ch = sum(1 for e in epochs.events[:, 2] if e == 1)
            n_nc = sum(1 for e in epochs.events[:, 2] if e == 2)
            print(f"  {label}: {n_ch} CHANGE + {n_nc} NO_CHANGE "
                  f"(focused {win_ms}ms epochs)")
        except Exception as exc:
            print(f"  {label}: epoch error — {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_bandpower_welch(epochs: mne.Epochs) -> np.ndarray:
    """Welch bandpower per channel per epoch (script 11 approach).

    Returns: (n_epochs, n_channels * n_bands)
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_ep, n_ch, n_times = data.shape
    sfreq = epochs.info["sfreq"]
    band_list = list(SPECTRAL_BANDS.values())
    n_bands = len(band_list)
    features = np.empty((n_ep, n_ch * n_bands))

    for i in range(n_ep):
        feat_idx = 0
        for ch in range(n_ch):
            freqs, psd = scipy_welch(data[i, ch], fs=sfreq, nperseg=n_times)
            for flo, fhi in band_list:
                mask = (freqs >= flo) & (freqs <= fhi)
                features[i, feat_idx] = (
                    np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
                )
                feat_idx += 1
    return features


def extract_tde_cov_train_test(
    train_epochs_list: list[mne.Epochs],
    test_epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray, int]:
    """TDE → Global PCA (fit on train) → covariance upper triangle.

    PCA is fit on concatenated TDE data from train epochs only,
    then applied to both train and test to avoid data leakage.

    Returns: (X_train, X_test, n_features)
    """
    from campeones_analysis.luminance.tde_glhmm import (
        apply_tde_only, fit_global_pca, apply_global_pca,
    )
    from campeones_analysis.luminance.features import compute_epoch_covariance

    def _epochs_to_tde_segments(epochs: mne.Epochs) -> list[np.ndarray]:
        """Apply TDE to each epoch, return list of TDE arrays."""
        data = epochs.get_data()  # (n_ep, n_ch, n_times)
        segments = []
        for i in range(data.shape[0]):
            epoch_data = data[i].T  # (n_times, n_channels)
            n_t = epoch_data.shape[0]
            indices = np.array([[0, n_t]])
            tde_data, _ = apply_tde_only(epoch_data, indices, TDE_WINDOW_HALF)
            segments.append(tde_data)
        return segments

    def _segments_to_cov(segments: list[np.ndarray], pca_model) -> np.ndarray:
        """Project TDE segments with PCA, compute covariance per epoch."""
        n_ep = len(segments)
        n_feat = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
        features = np.empty((n_ep, n_feat))
        for i, seg in enumerate(segments):
            pca_data = apply_global_pca(seg, pca_model)
            features[i] = compute_epoch_covariance(pca_data)
        return features

    # Collect TDE segments from all train runs
    train_tde_segments = []
    train_run_boundaries = []  # track which segments belong to which run
    for ep in train_epochs_list:
        segs = _epochs_to_tde_segments(ep)
        train_run_boundaries.append(len(segs))
        train_tde_segments.extend(segs)

    # Fit global PCA on train TDE data only
    pca_model = fit_global_pca(train_tde_segments, TDE_PCA_COMPONENTS)

    # Extract features for train
    X_train = _segments_to_cov(train_tde_segments, pca_model)

    # Extract features for test (using train PCA)
    test_tde_segments = _epochs_to_tde_segments(test_epochs)
    X_test = _segments_to_cov(test_tde_segments, pca_model)

    n_features = X_train.shape[1]
    return X_train, X_test, n_features


def extract_raw_features(epochs: mne.Epochs) -> np.ndarray:
    """Raw EEG vectorized (for PCA inside sklearn pipeline).

    Returns: (n_epochs, n_channels * n_times)
    """
    data = epochs.get_data()
    return data.reshape(data.shape[0], -1)


# ---------------------------------------------------------------------------
# Classification with Leave-One-Run-Out CV
# ---------------------------------------------------------------------------

def _build_pipeline(use_pca: bool, c_val: float, n_samples: int, n_features: int):
    """Build sklearn pipeline with optional PCA."""
    steps = [StandardScaler()]
    if use_pca:
        n_comp = min(RAW_PCA_COMPONENTS, n_samples, n_features)
        steps.append(PCA(n_components=n_comp))
    steps.append(LogisticRegression(C=c_val, max_iter=1000,
                                    random_state=RANDOM_SEED))
    return make_pipeline(*steps)



def _select_best_c(X_train, y_train, use_pca: bool,
                   fixed_c: float | None = None) -> float:
    """Inner CV to select best C, or return fixed_c if provided."""
    if fixed_c is not None:
        return fixed_c
    inner_cv = StratifiedKFold(n_splits=min(3, max(2, len(np.unique(y_train)))),
                               shuffle=True, random_state=RANDOM_SEED)
    best_c, best_score = C_GRID[0], -1.0
    for c_val in C_GRID:
        scores = []
        try:
            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                pipe = _build_pipeline(use_pca, c_val,
                                       X_train[tr_idx].shape[0],
                                       X_train[tr_idx].shape[1])
                pipe.fit(X_train[tr_idx], y_train[tr_idx])
                scores.append(pipe.score(X_train[val_idx], y_train[val_idx]))
        except ValueError:
            continue
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val
    return best_c




def run_loro_standard(
    runs_data: list[tuple[mne.Epochs, str]],
    feature_name: str,
    extract_fn,
    fixed_c: float | None = None,
) -> dict:
    """LORO CV for bandpower_welch and raw_pca (simple extract → classify).

    For raw_pca, PCA is applied on train data before inner CV to keep
    the grid search fast (36k features → 100 components).
    """
    use_pca = (feature_name == "raw_pca")

    # Pre-extract features per run
    run_features = []
    run_labels = []
    run_names = []
    for epochs, run_label in runs_data:
        X = extract_fn(epochs)
        y = (epochs.events[:, 2] == 1).astype(int)  # 1=CHANGE, 0=NO_CHANGE
        run_features.append(X)
        run_labels.append(y)
        run_names.append(run_label)

    n_runs = len(run_features)
    if n_runs < 2:
        print(f"    {feature_name}: need >= 2 runs, got {n_runs}")
        return {}

    n_features_raw = run_features[0].shape[1]
    print(f"    N features (raw): {n_features_raw}")

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []
    n_features_effective = n_features_raw

    for test_idx in range(n_runs):
        X_train = np.vstack([run_features[i] for i in range(n_runs) if i != test_idx])
        y_train = np.concatenate([run_labels[i] for i in range(n_runs) if i != test_idx])
        X_test = run_features[test_idx]
        y_test = run_labels[test_idx]

        # For raw_pca: apply PCA on train, transform test, then do inner CV
        # without PCA in the pipeline (already reduced)
        pca_model_outer = None
        if use_pca:
            n_comp = min(RAW_PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
            scaler_outer = StandardScaler()
            X_train_sc = scaler_outer.fit_transform(X_train)
            X_test_sc = scaler_outer.transform(X_test)
            pca_model_outer = PCA(n_components=n_comp)
            X_train = pca_model_outer.fit_transform(X_train_sc)
            X_test = pca_model_outer.transform(X_test_sc)
            n_features_effective = n_comp

        best_c = _select_best_c(X_train, y_train, use_pca=False,
                               fixed_c=fixed_c)

        pipe = _build_pipeline(False, best_c, X_train.shape[0], X_train.shape[1])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_metrics.append({
            "fold": run_names[test_idx], "C": best_c,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        })

    y_true = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    return {
        "feature": feature_name,
        "n_features": int(n_features_effective),
        "n_features_raw": int(n_features_raw),
        "accuracy": float(accuracy_score(y_true, y_pred_arr)),
        "precision": float(precision_score(y_true, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_arr, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob_arr)),
        "n_change": int(y_true.sum()),
        "n_no_change": int((1 - y_true).sum()),
        "folds": fold_metrics,
    }



def run_loro_tde_cov(
    runs_data: list[tuple[mne.Epochs, str]],
    fixed_c: float | None = None,
) -> dict:
    """LORO CV for tde_cov — PCA fit on train only per fold.

    Pre-computes TDE segments once, then only re-fits PCA per fold.
    """
    from campeones_analysis.luminance.tde_glhmm import (
        apply_tde_only, fit_global_pca, apply_global_pca,
    )
    from campeones_analysis.luminance.features import compute_epoch_covariance

    feature_name = "tde_cov"
    n_runs = len(runs_data)
    if n_runs < 2:
        print(f"    {feature_name}: need >= 2 runs, got {n_runs}")
        return {}

    # Pre-compute TDE segments for all runs (expensive, do once)
    print("    Pre-computing TDE segments...")
    run_tde_segments: list[list[np.ndarray]] = []
    run_labels: list[np.ndarray] = []
    for epochs, label in runs_data:
        data = epochs.get_data()  # (n_ep, n_ch, n_times)
        segments = []
        for i in range(data.shape[0]):
            epoch_data = data[i].T  # (n_times, n_channels)
            n_t = epoch_data.shape[0]
            indices = np.array([[0, n_t]])
            tde_data, _ = apply_tde_only(epoch_data, indices, TDE_WINDOW_HALF)
            segments.append(tde_data)
        run_tde_segments.append(segments)
        run_labels.append((epochs.events[:, 2] == 1).astype(int))

    n_features = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
    print(f"    N features: {n_features}")

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []

    for test_idx in range(n_runs):
        print(f"    Fold {test_idx+1}/{n_runs}: {runs_data[test_idx][1]}")
        # Collect train TDE segments
        train_segments = []
        for i in range(n_runs):
            if i != test_idx:
                train_segments.extend(run_tde_segments[i])

        # Fit PCA on train only
        pca_model = fit_global_pca(train_segments, TDE_PCA_COMPONENTS)

        # Extract cov features for train
        X_train_parts = []
        y_train_parts = []
        for i in range(n_runs):
            if i != test_idx:
                n_ep = len(run_tde_segments[i])
                feats = np.empty((n_ep, n_features))
                for j, seg in enumerate(run_tde_segments[i]):
                    pca_data = apply_global_pca(seg, pca_model)
                    feats[j] = compute_epoch_covariance(pca_data)
                X_train_parts.append(feats)
                y_train_parts.append(run_labels[i])

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Extract cov features for test
        test_segs = run_tde_segments[test_idx]
        X_test = np.empty((len(test_segs), n_features))
        for j, seg in enumerate(test_segs):
            pca_data = apply_global_pca(seg, pca_model)
            X_test[j] = compute_epoch_covariance(pca_data)
        y_test = run_labels[test_idx]

        best_c = _select_best_c(X_train, y_train, use_pca=False,
                               fixed_c=fixed_c)

        pipe = _build_pipeline(False, best_c, X_train.shape[0], X_train.shape[1])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_metrics.append({
            "fold": runs_data[test_idx][1], "C": best_c,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        })

    y_true = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    return {
        "feature": feature_name,
        "n_features": int(n_features),
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

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(all_results):
        vals = [r[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=r["feature"])

    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_ylabel("Score")
    ax.set_title(f"Photo decoding: CHANGE vs NO_CHANGE (sub-{subject}, LORO CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{subject}_decoding_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Summary plot saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str, fixed_c: float | None = None,
                 focused: bool = False, crop_end: float = FOCUSED_CROP_END) -> None:
    win_ms = int(round((crop_end - FOCUSED_CROP_START) * 1000))
    mode_label = f"FOCUSED ({win_ms}ms)" if focused else "WIDE (4.5s)"
    if focused:
        ep_tmin, ep_tmax = FOCUSED_CROP_START, crop_end
        ep_baseline = FOCUSED_BASELINE
    else:
        ep_tmin, ep_tmax = TMIN, TMAX
        ep_baseline = BASELINE

    print("=" * 60)
    print(f"27 — Photo decoding — sub-{subject} — {mode_label}")
    print(f"     Epoch window: {ep_tmin} to {ep_tmax}")
    print(f"     Baseline: {ep_baseline}")
    print(f"     Channels: {len(EEG_CHANNELS)}")
    if fixed_c is not None:
        print(f"     Fixed C={fixed_c} (no inner CV)")
    else:
        print(f"     C grid search: {C_GRID} (inner 3-fold CV)")
    print("=" * 60)

    if focused:
        runs_data = load_epochs_per_run_focused(subject, crop_end=crop_end)
    else:
        runs_data = load_epochs_per_run(subject)
    if len(runs_data) < 2:
        print("Need at least 2 runs for LORO CV.")
        sys.exit(1)

    total_ch = sum(
        sum(1 for e in ep.events[:, 2] if e == 1) for ep, _ in runs_data
    )
    total_nc = sum(
        sum(1 for e in ep.events[:, 2] if e == 2) for ep, _ in runs_data
    )
    print(f"\n  Total: {total_ch} CHANGE + {total_nc} NO_CHANGE across "
          f"{len(runs_data)} runs\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    if focused:
        output_dir = (RESULTS_ROOT.parent / f"photo_decoding_focused_{win_ms}ms"
                      / f"sub-{subject}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # --- 1. Bandpower (Welch) ---
    print(f"  Feature set: bandpower_welch")
    res = run_loro_standard(runs_data, "bandpower_welch", extract_bandpower_welch,
                            fixed_c=fixed_c)
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(C={f['C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # --- 2. TDE + Cov (PCA fit on train per fold) ---
    print(f"\n  Feature set: tde_cov")
    res = run_loro_tde_cov(runs_data, fixed_c=fixed_c)
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(C={f['C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # --- 3. Raw + PCA ---
    print(f"\n  Feature set: raw_pca")
    res = run_loro_standard(runs_data, "raw_pca", extract_raw_features,
                            fixed_c=fixed_c)
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(C={f['C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # Save results JSON
    suffix = f"_focused_{win_ms}ms" if focused else ""
    json_path = output_dir / f"sub-{subject}_decoding_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    if all_results:
        plot_summary(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode CHANGE vs NO_CHANGE from photo epochs",
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--fixed-c", type=float, default=1.0,
                        help="Fixed C for LogisticRegression (default: 1.0). "
                             "Set to 0 to enable inner CV grid search instead.")
    parser.add_argument("--inner-cv", action="store_true",
                        help="Enable inner CV grid search for C (overrides --fixed-c).")
    parser.add_argument("--focused", action="store_true",
                        help="Use focused epochs (50ms to crop-end post-onset) "
                             "instead of wide 4.5s epochs.")
    parser.add_argument("--crop-end", type=float, default=FOCUSED_CROP_END,
                        help="End of crop window in seconds post-onset "
                             f"(default: {FOCUSED_CROP_END}). "
                             "E.g. 0.55 for 500ms, 1.05 for 1s.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fixed_c = None if args.inner_cv else args.fixed_c
    run_pipeline(subject=args.subject, fixed_c=fixed_c, focused=args.focused,
                 crop_end=args.crop_end)
