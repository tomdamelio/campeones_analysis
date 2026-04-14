#!/usr/bin/env python
"""Decode POST-stimulus vs PRE-stimulus — Yongjie-style pipeline (PRE vs POST design).

Adapts Yongjie Duan's best-practice pipeline to sub-27 single-subject data.
For each CHANGE_PHOTO onset (onset of 1-second 3Hz flash + tone stimulus):
  PRE  (class 0): [-1.25, -0.05] s — 1.2 s of fixation-cross baseline
  POST (class 1): [+0.05, +1.25] s — 1.2 s capturing the full flash response

This design ensures every run contributes balanced classes (N PRE + N POST),
avoiding folds with a single class in LORO-CV.

No baseline correction is applied during epoching: the PRE window would overlap
with a standard pre-stimulus baseline, so we let MVNN handle normalization.

Pipeline:
  - Raw time-series (channels × time) as features, flattened
  - MVNN whitening (Guggenmos et al. 2018): per-timepoint noise covariance
    normalization estimated on training data within each LORO fold
  - LinearSVC (C=1.0)
  - Leave-One-Run-Out CV (7 folds), global accuracy (concatenated predictions)
  - Resampled to 100 Hz → 32 channels × 120 timepoints = 3 840 features

References
----------
  Guggenmos et al. 2018, J. Neurosci. Methods — MVNN
  Duan et al. (unpublished) — brightness_loso.py pipeline

Usage
-----
    micromamba run -n campeones python scripts/validation/41_decoding_mvnn_svc.py --subject 27
    micromamba run -n campeones python scripts/validation/41_decoding_mvnn_svc.py --subject 27 --no-perm
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_mvnn_svc"

SESSION = "vr"

# Wide epoch (no baseline correction — PRE window would overlap any pre-stimulus baseline)
WIDE_TMIN = -1.5
WIDE_TMAX = 1.5

# PRE window: baseline before the flash onset
PRE_START = -1.25
PRE_END = -0.05

# POST window: response to the 1-second flash stimulus
POST_START = 0.05
POST_END = 1.25

# Resample to Yongjie's sampling rate
TARGET_SFREQ = 100  # Hz

# Classifier (matching Yongjie exactly)
C_VALUE = 1.0
MAX_ITER = 10_000

# Permutation test
N_PERMUTATIONS = 100
RANDOM_SEED = 42

SPECTRAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

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

# Occipital + temporal subset — visual cortex + lateral temporal sites
OCC_TEMP_CHANNELS = ['O1', 'O2', 'T7', 'T8', 'FT9', 'FT10', 'TP9', 'TP10']

CHANNEL_SUBSETS = {
    "all": EEG_CHANNELS,
    "occ_temp": OCC_TEMP_CHANNELS,
}


# ---------------------------------------------------------------------------
# MVNN Whitener
# ---------------------------------------------------------------------------

class MVNNWhitener:
    """Multivariate Noise Normalization (per-timepoint covariance whitening).

    For each timepoint t, estimates the noise covariance matrix Sigma_t from
    within-class residuals (trial minus class mean) on training data, then
    computes the whitening matrix W_t = Sigma_t^(-1/2) via eigendecomposition.

    Applying W_t to the data at each timepoint transforms the noise covariance
    to identity, so any remaining structure reflects signal rather than noise.
    Data are subsequently flattened to (n_trials, n_channels * n_timepoints).

    Reference: Guggenmos et al. 2018, J. Neurosci. Methods.

    Parameters
    ----------
    regularization : float
        Ridge term added to the diagonal of Sigma before inversion,
        for numerical stability when some eigenvalues are near zero.
    """

    def __init__(self, regularization: float = 1e-8):
        self.regularization = regularization
        self.whitening_matrices_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MVNNWhitener":
        """Estimate per-timepoint whitening matrices from training data.

        Parameters
        ----------
        X : (n_trials, n_channels, n_timepoints)
        y : (n_trials,) int class labels
        """
        n_trials, n_ch, n_tp = X.shape
        classes = np.unique(y)
        self.whitening_matrices_ = []

        for t in range(n_tp):
            X_t = X[:, :, t]  # (n_trials, n_channels)

            # Within-class residuals: subtract class mean from each trial
            residuals = []
            for c in classes:
                mask = (y == c)
                X_c = X_t[mask]
                residuals.append(X_c - X_c.mean(axis=0))

            R = np.vstack(residuals)  # (n_trials, n_channels)

            # Noise covariance (n_ch x n_ch), normalized by n_trials - 1
            Sigma = (R.T @ R) / max(R.shape[0] - 1, 1)

            # Ridge regularization for numerical stability
            Sigma += self.regularization * np.eye(n_ch)

            # Sigma^(-1/2) via symmetric eigendecomposition
            # Sigma = V @ diag(lambda) @ V^T  =>  Sigma^(-1/2) = V @ diag(lambda^-0.5) @ V^T
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # clamp for stability
            W = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

            self.whitening_matrices_.append(W)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply per-timepoint whitening and flatten.

        Parameters
        ----------
        X : (n_trials, n_channels, n_timepoints)

        Returns
        -------
        (n_trials, n_channels * n_timepoints) — whitened and flattened
        """
        if self.whitening_matrices_ is None:
            raise RuntimeError("Call fit() before transform().")

        n_trials, n_ch, n_tp = X.shape
        X_whitened = np.empty_like(X)

        for t in range(n_tp):
            # W is (n_ch, n_ch) symmetric
            # X[:, :, t] is (n_trials, n_ch)
            # result: (n_trials, n_ch) @ (n_ch, n_ch) = (n_trials, n_ch)
            X_whitened[:, :, t] = X[:, :, t] @ self.whitening_matrices_[t]

        return X_whitened.reshape(n_trials, -1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Epoch loading
# ---------------------------------------------------------------------------

def load_epochs_per_run(subject: str, channel_list: list[str] | None = None) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load PRE and POST epochs from CHANGE_PHOTO onsets for all runs.

    For each CHANGE_PHOTO onset (= onset of the 1-second 3Hz flash stimulus):
      PRE  (class 0): [PRE_START,  PRE_END]  — fixation-cross baseline
      POST (class 1): [POST_START, POST_END] — flash response

    Both windows are 1.2 s at TARGET_SFREQ = 120 timepoints × 32 ch = 3 840 features.
    Every run contributes N PRE + N POST trials (perfectly balanced).

    No baseline correction: PRE window overlaps with any standard pre-stimulus
    baseline, so we leave it uncorrected and rely on MVNN for normalization.

    Returns
    -------
    list of (X, y, run_label) where:
      X : (2*n_onsets, n_channels, n_timepoints) float64
      y : (2*n_onsets,) int  — 0 = PRE, 1 = POST
    """
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    result = []

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

        # Keep only CHANGE_PHOTO events (onset of each flash presentation)
        change_rows = events_df[events_df["trial_type"] == "CHANGE_PHOTO"]

        if change_rows.empty:
            print(f"  SKIP {label} — no CHANGE_PHOTO events in TSV")
            continue

        sfreq = raw.info["sfreq"]
        ch_pool = channel_list if channel_list is not None else EEG_CHANNELS
        available_chs = [ch for ch in ch_pool if ch in raw.ch_names]

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
            np.ones(len(samples), dtype=int),   # event_id = 1 for all CHANGE_PHOTO
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        try:
            # Load wide epoch, no baseline correction
            epochs = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=WIDE_TMIN, tmax=WIDE_TMAX, picks=available_chs,
                baseline=None, preload=True, verbose=False,
            )

            # Resample before crop
            if abs(epochs.info["sfreq"] - TARGET_SFREQ) > 1.0:
                epochs = epochs.resample(TARGET_SFREQ, verbose=False)

            n_onsets = len(epochs)

            # PRE: fixation-cross baseline before flash onset
            X_pre = epochs.copy().crop(tmin=PRE_START, tmax=PRE_END).get_data()

            # POST: flash response window
            X_post = epochs.copy().crop(tmin=POST_START, tmax=POST_END).get_data()

            # Stack PRE (class 0) + POST (class 1) — balanced by construction
            X = np.concatenate([X_pre, X_post], axis=0)  # (2*n_onsets, n_ch, n_tp)
            y = np.concatenate([
                np.zeros(n_onsets, dtype=int),   # PRE = 0
                np.ones(n_onsets, dtype=int),    # POST = 1
            ])

            print(f"  {label}: {n_onsets} onsets → "
                  f"{n_onsets} PRE + {n_onsets} POST | X={X.shape}")
            result.append((X, y, label))

        except Exception as exc:
            print(f"  {label}: epoch error — {exc}")

    return result


# ---------------------------------------------------------------------------
# LORO-CV with MVNN + LinearSVC
# ---------------------------------------------------------------------------

def run_loro(
    runs_data: list[tuple[np.ndarray, np.ndarray, str]],
) -> dict:
    """LORO cross-validation: MVNN whitening + LinearSVC per fold.

    For each fold, MVNN is fit exclusively on training runs to prevent
    leakage of noise structure from test data.

    Global accuracy is computed by concatenating all fold predictions.
    Classifier coefficients (coef_) are collected from each fold and averaged
    to produce a stable interpretability map (shape: n_channels × n_timepoints).
    """
    n_runs = len(runs_data)
    if n_runs < 2:
        raise ValueError(f"Need >= 2 runs, got {n_runs}")

    n_ch = runs_data[0][0].shape[1]
    n_tp = runs_data[0][0].shape[2]

    all_y_true = []
    all_y_pred = []
    fold_accs = []
    fold_details = []
    coef_list = []   # collect coef_ from each fold

    for test_idx in range(n_runs):
        train_idx = [i for i in range(n_runs) if i != test_idx]

        X_train = np.concatenate([runs_data[i][0] for i in train_idx])
        y_train = np.concatenate([runs_data[i][1] for i in train_idx])
        X_test, y_test, test_label = runs_data[test_idx]

        # MVNN: fit on train, transform both
        mvnn = MVNNWhitener()
        X_train_flat = mvnn.fit_transform(X_train, y_train)
        X_test_flat = mvnn.transform(X_test)

        # LinearSVC (C=1.0, dual='auto' handles p >> n via primal formulation)
        clf = LinearSVC(C=C_VALUE, max_iter=MAX_ITER, dual="auto")
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)

        # Collect coefficients (shape: n_ch * n_tp,)
        coef_list.append(clf.coef_[0].copy())

        fold_acc = float(np.mean(y_pred == y_test))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_accs.append(fold_acc)
        fold_details.append({
            "run": test_label,
            "accuracy": fold_acc,
            "n_test": len(y_test),
            "n_post": int(y_test.sum()),
            "n_pre": int((y_test == 0).sum()),
        })
        print(f"  Fold {test_idx + 1}/{n_runs} (test={test_label}): "
              f"{fold_acc*100:.1f}%  "
              f"(post={int(y_test.sum())}, pre={int((y_test==0).sum())})")

    # Global accuracy: concatenate all test predictions across folds
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    global_acc = float(np.mean(y_true_all == y_pred_all))

    # Average coef_ across folds → (n_ch, n_tp) interpretability map
    mean_coef_2d = np.mean(coef_list, axis=0).reshape(n_ch, n_tp)

    return {
        "global_accuracy": global_acc,
        "fold_mean_accuracy": float(np.mean(fold_accs)),
        "fold_std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
        "fold_details": fold_details,
        "n_trials_total": len(y_true_all),
        "n_post_total": int(y_true_all.sum()),
        "n_pre_total": int((y_true_all == 0).sum()),
        "mean_coef_2d": mean_coef_2d,  # (n_ch, n_tp)
    }


# ---------------------------------------------------------------------------
# Bandpower feature extraction + LORO-CV
# ---------------------------------------------------------------------------

def extract_bandpower(X: np.ndarray) -> np.ndarray:
    """Welch bandpower per channel per epoch.

    Parameters
    ----------
    X : (n_trials, n_channels, n_timepoints) at TARGET_SFREQ

    Returns
    -------
    (n_trials, n_channels * n_bands) — 32 ch × 5 bands = 160 features
    """
    n_ep, n_ch, n_tp = X.shape
    band_list = list(SPECTRAL_BANDS.values())
    n_bands = len(band_list)
    features = np.empty((n_ep, n_ch * n_bands))

    for i in range(n_ep):
        feat_idx = 0
        for ch in range(n_ch):
            freqs, psd = scipy_welch(X[i, ch], fs=TARGET_SFREQ, nperseg=n_tp)
            for flo, fhi in band_list:
                mask = (freqs >= flo) & (freqs <= fhi)
                features[i, feat_idx] = (
                    np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
                )
                feat_idx += 1
    return features


def run_loro_bandpower(
    runs_data: list[tuple[np.ndarray, np.ndarray, str]],
) -> dict:
    """LORO-CV with bandpower features + StandardScaler + LinearSVC.

    Bandpower is extracted once per run (no train/test leakage: Welch PSD
    on each epoch independently). StandardScaler is fit on train data only.
    Global accuracy computed on concatenated fold predictions.
    """
    n_runs = len(runs_data)
    if n_runs < 2:
        raise ValueError(f"Need >= 2 runs, got {n_runs}")

    # Pre-extract bandpower features for each run (no leakage: per-epoch operation)
    runs_bp = []
    for X, y, label in runs_data:
        X_bp = extract_bandpower(X)
        runs_bp.append((X_bp, y, label))

    all_y_true = []
    all_y_pred = []
    fold_accs = []
    fold_details = []

    for test_idx in range(n_runs):
        train_idx = [i for i in range(n_runs) if i != test_idx]

        X_train = np.concatenate([runs_bp[i][0] for i in train_idx])
        y_train = np.concatenate([runs_bp[i][1] for i in train_idx])
        X_test, y_test, test_label = runs_bp[test_idx]

        # StandardScaler fit on train only
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        clf = LinearSVC(C=C_VALUE, max_iter=MAX_ITER, dual="auto")
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)

        fold_acc = float(np.mean(y_pred == y_test))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_accs.append(fold_acc)
        fold_details.append({
            "run": test_label,
            "accuracy": fold_acc,
            "n_test": len(y_test),
            "n_post": int(y_test.sum()),
            "n_pre": int((y_test == 0).sum()),
        })
        print(f"  Fold {test_idx + 1}/{n_runs} (test={test_label}): {fold_acc*100:.1f}%")

    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    global_acc = float(np.mean(y_true_all == y_pred_all))

    return {
        "global_accuracy": global_acc,
        "fold_mean_accuracy": float(np.mean(fold_accs)),
        "fold_std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
        "fold_details": fold_details,
        "n_features": runs_bp[0][0].shape[1],
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    runs_data: list[tuple[np.ndarray, np.ndarray, str]],
    observed_acc: float,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Build null distribution by shuffling labels and re-running LORO-CV.

    Labels are shuffled independently within each run to preserve the
    run-level class balance as closely as possible.
    Uses global_accuracy (concatenated predictions) to match observed_acc.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    null_accs = []

    for perm_idx in range(n_permutations):
        perm_runs = [
            (X, rng.permutation(y), label)
            for X, y, label in runs_data
        ]
        result = run_loro(perm_runs)
        null_accs.append(result["global_accuracy"])

        if (perm_idx + 1) % 20 == 0:
            print(f"  [{perm_idx + 1}/{n_permutations}] "
                  f"null mean so far: {np.mean(null_accs)*100:.1f}%")

    null_accs = np.array(null_accs)
    p_value = float((null_accs >= observed_acc).mean())

    return {
        "p_value": p_value,
        "n_permutations": n_permutations,
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "null_distribution": null_accs.tolist(),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_interpretability_mvnn(
    mean_coef_2d: np.ndarray,
    out_path: Path,
    channel_names: list[str] | None = None,
) -> None:
    """Plot interpretability map for MVNN + LinearSVC.

    mean_coef_2d : (n_channels, n_timepoints) — averaged coef_ across LORO folds.
    Positive values push toward POST (flash response), negative toward PRE (baseline).

    Note: coefficients are in the MVNN-whitened space. The activation pattern
    in the original EEG space would require Haufe et al. 2014 transformation
    (A = Σ_x @ w.T). Here we plot the whitened-space coefficients as a first
    approximation (valid because MVNN makes noise covariance ≈ identity,
    so Σ_x_whitened ≈ I and A_whitened ≈ w).

    Three panels:
      1. Channel × time heatmap (signed coef: red=POST, blue=PRE)
      2. Temporal profile: mean |coef| across channels — when is the signal discriminative?
      3. Spatial profile: mean |coef| across time — which channels matter most?
    """
    n_ch, n_tp = mean_coef_2d.shape
    ch_names = channel_names if channel_names is not None else EEG_CHANNELS
    times = np.linspace(0, POST_END - POST_START, n_tp)   # 0 to 1.2 s within epoch

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1], hspace=0.45, wspace=0.4)

    ax_heatmap = fig.add_subplot(gs[0, :])   # full top row: channel × time heatmap
    ax_time    = fig.add_subplot(gs[1, 0])   # bottom left: temporal profile
    ax_space   = fig.add_subplot(gs[1, 1])   # bottom right: spatial profile

    # --- Panel 1: Channel × time heatmap ---
    vmax = np.percentile(np.abs(mean_coef_2d), 98)
    im = ax_heatmap.imshow(
        mean_coef_2d,
        aspect="auto",
        origin="upper",
        extent=[times[0], times[-1], n_ch - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    ax_heatmap.set_yticks(np.arange(n_ch))
    ax_heatmap.set_yticklabels(ch_names, fontsize=9)
    ax_heatmap.set_xlabel("Time within epoch (s)")
    ax_heatmap.set_ylabel("Channel")
    ax_heatmap.set_title(
        "MVNN + LinearSVC — classifier weights (whitened space)\n"
        "Red = pushes toward POST (flash), Blue = pushes toward PRE (baseline)"
    )

    # Mark expected SSVEP peaks at 3Hz (333ms, 667ms, 1000ms)
    for t_ssvep in [1/3, 2/3, 1.0]:
        ax_heatmap.axvline(t_ssvep, color="k", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.colorbar(im, ax=ax_heatmap, label="coef weight", fraction=0.02, pad=0.01)

    # --- Panel 2: Temporal profile ---
    temporal_profile = np.mean(np.abs(mean_coef_2d), axis=0)   # (n_tp,)
    ax_time.plot(times, temporal_profile, color="steelblue", linewidth=1.5)
    for t_ssvep in [1/3, 2/3, 1.0]:
        ax_time.axvline(t_ssvep, color="k", linestyle="--", alpha=0.4, linewidth=0.8,
                        label=f"3Hz peak ({t_ssvep*1000:.0f}ms)" if t_ssvep == 1/3 else None)
    ax_time.set_xlabel("Time within epoch (s)")
    ax_time.set_ylabel("Mean |coef| across channels")
    ax_time.set_title("Temporal discriminability profile")
    ax_time.legend(fontsize=7)
    ax_time.set_xlim(times[0], times[-1])

    # --- Panel 3: Spatial profile ---
    spatial_profile = np.mean(np.abs(mean_coef_2d), axis=1)    # (n_ch,)
    sorted_idx = np.argsort(spatial_profile)[::-1]
    ch_names_sorted = [ch_names[i] for i in sorted_idx]
    y_pos = np.arange(n_ch)
    ax_space.barh(y_pos, spatial_profile[sorted_idx], color="steelblue", alpha=0.8)
    ax_space.set_yticks(y_pos)
    ax_space.set_yticklabels(ch_names_sorted, fontsize=9)
    ax_space.invert_yaxis()
    ax_space.set_xlabel("Mean |coef| across time")
    ax_space.set_title("Spatial importance (channels)")

    fig.suptitle(
        f"Interpretability — MVNN + LinearSVC  |  POST vs PRE  |  "
        f"mean over {n_ch} ch, 7 LORO folds",
        fontsize=11, y=1.01,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  Interpretability plot: {out_path}")


def plot_results(
    mvnn_results: dict,
    bp_results: dict,
    perm_results: dict | None,
    out_path: Path,
) -> None:
    """Save a figure comparing MVNN+SVC vs bandpower+SVC fold accuracies."""
    n_folds = len(mvnn_results["fold_accuracies"])
    x = np.arange(n_folds)
    width = 0.35

    n_cols = 3 if perm_results is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    # Panel 1: MVNN fold accuracies
    ax = axes[0]
    ax.bar(x, [a * 100 for a in mvnn_results["fold_accuracies"]],
           color="steelblue", alpha=0.8)
    ax.axhline(mvnn_results["global_accuracy"] * 100, color="navy",
               linestyle="--",
               label=f"Global {mvnn_results['global_accuracy']*100:.1f}%")
    ax.axhline(50, color="gray", linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}" for i in x])
    ax.set_xlabel("LORO fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MVNN + LinearSVC\n(raw 3 840 features)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Panel 2: Bandpower fold accuracies
    ax2 = axes[1]
    ax2.bar(x, [a * 100 for a in bp_results["fold_accuracies"]],
            color="darkorange", alpha=0.8)
    ax2.axhline(bp_results["global_accuracy"] * 100, color="saddlebrown",
                linestyle="--",
                label=f"Global {bp_results['global_accuracy']*100:.1f}%")
    ax2.axhline(50, color="gray", linestyle=":", label="Chance")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"F{i+1}" for i in x])
    ax2.set_xlabel("LORO fold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Bandpower Welch + LinearSVC\n(160 features)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 100)

    # Panel 3: null distribution for MVNN (if permutation test was run)
    if perm_results is not None:
        ax3 = axes[2]
        null = np.array(perm_results["null_distribution"]) * 100
        ax3.hist(null, bins=20, color="lightgray", edgecolor="gray", label="Null (MVNN)")
        ax3.axvline(mvnn_results["global_accuracy"] * 100, color="navy",
                    linestyle="--",
                    label=f"MVNN {mvnn_results['global_accuracy']*100:.1f}%")
        ax3.axvline(bp_results["global_accuracy"] * 100, color="darkorange",
                    linestyle="--",
                    label=f"BP {bp_results['global_accuracy']*100:.1f}%")
        ax3.set_xlabel("Accuracy (%)")
        ax3.set_ylabel("Count")
        p = perm_results["p_value"]
        ax3.set_title(f"Permutation test — MVNN\n(p={p:.3f})")
        ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CHANGE vs NO_CHANGE decoding with MVNN + LinearSVC"
    )
    parser.add_argument("--subject", default="27")
    parser.add_argument("--no-perm", action="store_true",
                        help="Skip permutation test (faster, for debugging)")
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    parser.add_argument("--channel-subset", choices=list(CHANNEL_SUBSETS.keys()),
                        default="all",
                        help="Channel subset to use (default: all 32 channels)")
    args = parser.parse_args()

    channel_list = CHANNEL_SUBSETS[args.channel_subset]
    subset_tag = args.channel_subset   # used in output filenames

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    n_ch = len(channel_list)
    win_dur = POST_END - POST_START
    n_tp = int(round(win_dur * TARGET_SFREQ))
    n_features = n_ch * n_tp

    print(f"\n{'='*60}")
    print("Decoding POST vs PRE — MVNN + LinearSVC (Yongjie-style)")
    print(f"Subject:     {args.subject}")
    print(f"PRE window:  [{PRE_START}, {PRE_END}] s  ({win_dur:.1f} s of baseline)")
    print(f"POST window: [{POST_START}, {POST_END}] s  ({win_dur:.1f} s of flash response)")
    print(f"Sfreq:       {TARGET_SFREQ} Hz")
    print(f"Features:    {n_ch} ch × {n_tp} tp = {n_features}")
    print(f"Classifier:  LinearSVC (C={C_VALUE})")
    print(f"Norm:        MVNN (Guggenmos et al. 2018)")
    print(f"CV:          LORO (global accuracy on concatenated predictions)")
    print(f"{'='*60}\n")

    print(f"Channel subset: {args.channel_subset} ({n_ch} channels: {channel_list})")
    print("Loading epochs...")
    runs_data = load_epochs_per_run(args.subject, channel_list=channel_list)

    if len(runs_data) < 2:
        print("ERROR: fewer than 2 runs loaded. Exiting.")
        sys.exit(1)

    total_trials = sum(len(y) for _, y, _ in runs_data)
    total_change = sum(int(y.sum()) for _, y, _ in runs_data)
    total_nochange = total_trials - total_change
    print(f"\nTotal: {total_trials} trials "
          f"({total_change} CHANGE, {total_nochange} NO_CHANGE)\n")

    # --- Model 1: MVNN + LinearSVC ---
    print("Running LORO-CV — MVNN + LinearSVC...")
    mvnn_results = run_loro(runs_data)
    mvnn_acc = mvnn_results["global_accuracy"]

    print(f"\nMVNN + LinearSVC Results:")
    print(f"  Global accuracy: {mvnn_acc*100:.1f}%")
    print(f"  Fold-mean:       {mvnn_results['fold_mean_accuracy']*100:.1f}%  "
          f"(std={mvnn_results['fold_std_accuracy']*100:.1f}%)")
    accs_str = ", ".join(f"{a*100:.1f}%" for a in mvnn_results["fold_accuracies"])
    print(f"  Per-fold:        [{accs_str}]")

    # Interpretability plot for MVNN (extract array before JSON serialization)
    mean_coef_2d = mvnn_results.pop("mean_coef_2d")
    coef_path = RESULTS_ROOT / f"sub-{args.subject}_{subset_tag}_mvnn_coef.npy"
    np.save(coef_path, mean_coef_2d)
    print(f"  Coefficients saved: {coef_path}")
    interp_path = RESULTS_ROOT / f"sub-{args.subject}_{subset_tag}_mvnn_interpretability.png"
    plot_interpretability_mvnn(mean_coef_2d, interp_path, channel_names=channel_list)

    # --- Model 2: Bandpower Welch + LinearSVC ---
    print("\nRunning LORO-CV — Bandpower Welch + LinearSVC...")
    bp_results = run_loro_bandpower(runs_data)
    bp_acc = bp_results["global_accuracy"]

    print(f"\nBandpower Welch + LinearSVC Results:")
    print(f"  Global accuracy: {bp_acc*100:.1f}%")
    print(f"  Fold-mean:       {bp_results['fold_mean_accuracy']*100:.1f}%  "
          f"(std={bp_results['fold_std_accuracy']*100:.1f}%)")
    accs_str = ", ".join(f"{a*100:.1f}%" for a in bp_results["fold_accuracies"])
    print(f"  Per-fold:        [{accs_str}]")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("Summary (same PRE/POST epochs, same LinearSVC C=1.0):")
    print(f"  Chance level:              50.0%")
    print(f"  MVNN + LinearSVC:          {mvnn_acc*100:.1f}%  ({n_features} features)")
    print(f"  Bandpower + LinearSVC:     {bp_acc*100:.1f}%  ({bp_results['n_features']} features)")
    print(f"{'='*50}")

    # --- Permutation test (MVNN only) ---
    perm_results = None
    if not args.no_perm:
        n_perm = args.n_perm
        print(f"\nRunning permutation test for MVNN ({n_perm} permutations)...")
        print("  (use --no-perm to skip)")
        perm_results = permutation_test(runs_data, mvnn_acc, n_permutations=n_perm)
        print(f"\nPermutation test (MVNN):")
        print(f"  p-value:   {perm_results['p_value']:.4f}")
        print(f"  Null mean: {perm_results['null_mean']*100:.1f}% "
              f"± {perm_results['null_std']*100:.1f}%")

    # Save JSON
    output = {
        "subject": args.subject,
        "channel_subset": args.channel_subset,
        "channels": channel_list,
        "config": {
            "pre_window": [PRE_START, PRE_END],
            "post_window": [POST_START, POST_END],
            "window_duration_s": win_dur,
            "target_sfreq": TARGET_SFREQ,
            "n_channels": n_ch,
            "n_timepoints": n_tp,
            "C": C_VALUE,
            "classifier": "LinearSVC",
            "baseline_correction": "none (MVNN handles normalization)",
            "cv": "LORO (global accuracy on concatenated predictions)",
            "n_runs": len(runs_data),
        },
        "data_summary": {
            "total_trials": total_trials,
            "n_post": total_change,
            "n_pre": total_nochange,
        },
        "mvnn_svc": {
            "n_features": n_features,
            "normalization": "MVNN (Guggenmos et al. 2018)",
            "results": mvnn_results,
            "permutation": perm_results,
        },
        "bandpower_svc": {
            "n_features": bp_results["n_features"],
            "normalization": "StandardScaler",
            "results": bp_results,
        },
    }

    json_path = RESULTS_ROOT / f"sub-{args.subject}_{subset_tag}_mvnn_vs_bandpower.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save plot
    plot_path = RESULTS_ROOT / f"sub-{args.subject}_{subset_tag}_mvnn_vs_bandpower.png"
    plot_results(mvnn_results, bp_results, perm_results, plot_path)

    print("Done.")


if __name__ == "__main__":
    main()
