#!/usr/bin/env python
"""3-class decoding of luminance changes in 60-second luminance epochs.

Task 3 from research diary 03_04_diario_tareas.md.

For each run's ~60s video_luminance segment, detects ChangeUp/ChangeDown events
from the frame-level derivative of the green channel, then classifies 250ms EEG
windows as ChangeUp / ChangeDown / NoChange with Leave-One-Run-Out CV.

Event detection:
  ChangeUp   (class 1): derivative crosses above +2.0 (onset of rising luminance)
  ChangeDown (class 2): derivative crosses below -2.0 (onset of falling luminance)
  NoChange   (class 0): |derivative| < 1.0 throughout window AND prior 1s

Windows per change event: 6 x 250ms at offsets [0, 50, 100, 150, 200, 250ms]
  → last window ends at 500ms post-onset (tope derecho = 500ms)
NoChange windows: 1 x 250ms per sampled stable frame.
Class balance: N_NC sampled to match N_up_windows + N_down_windows.

Feature sets:
  bandpower_welch : 32 ch x 5 bands = 160 features (Welch PSD)
  tde_cov         : TDE(±10 lags) -> PCA(17) -> cov upper triangle = 153 features
  raw_pca         : flatten -> PCA(160) = 160 features

Classifier: LogisticRegression (L2, C=1.0 fixed, lbfgs),
Leave-One-Run-Out CV outer loop.

Outputs per subject:
  results/validation/luminance_3class/sub-{sub}/
    sub-{sub}_3class_results.json
    sub-{sub}_3class_confusion.png
    sub-{sub}_3class_per_class_acc.png
    diagnostic_plots/  ← derivative timeline per run with highlighted windows

Usage
-----
    micromamba run -n campeones python scripts/validation/36_decoding_luminance_3class.py --subject 27
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch as scipy_welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from campeones_analysis.luminance.tde_glhmm import apply_tde_only

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
MERGED_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "merged_events"
STIMULI_PATH = PROJECT_ROOT / "stimuli" / "luminance"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "luminance_3class"

SESSION = "vr"

# ---------------------------------------------------------------------------
# Luminance detection parameters (aligned with script 21d)
# ---------------------------------------------------------------------------
DERIV_THRESHOLD = 1.5   # |ΔL/frame| > 1.5 → change event
NC_THRESHOLD = 1.5      # |ΔL/frame| < 1.5 throughout NC epoch (same as DERIV_THRESHOLD)
NC_PRIOR_S = 1.0        # seconds of stability required before a NoChange window

# ---------------------------------------------------------------------------
# Window parameters
# ---------------------------------------------------------------------------
WIN_SIZE_S = 0.250      # 250ms window
WIN_OFFSETS_S = [0.0, 0.050, 0.100, 0.150, 0.200, 0.250]  # 6 windows per event
# last window: offset 0.250 + size 0.250 = 0.500s (tope derecho)
GUARD_POST_S = WIN_OFFSETS_S[-1] + WIN_SIZE_S  # 0.500s — must fit in segment

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
LABEL_NC   = 0
LABEL_UP   = 1
LABEL_DOWN = 2
CLASS_NAMES = {0: "NoChange", 1: "ChangeUp", 2: "ChangeDown"}
N_CLASSES = 3

# ---------------------------------------------------------------------------
# Feature extraction (same constants as script 34)
# ---------------------------------------------------------------------------
SPECTRAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FIXED_C = 1.0
RANDOM_SEED = 42
TDE_WINDOW_HALF = 10
TDE_PCA_COMPONENTS = 17   # → 17*18//2 = 153 cov features
RAW_PCA_COMPONENTS = 160

# ---------------------------------------------------------------------------
# Thread-safe permutation progress counter
# ---------------------------------------------------------------------------

class _PermCounter:
    def __init__(self, total: int, feature_name: str, report_every: int = 0):
        self._n = 0
        self._lock = threading.Lock()
        self.total = total
        self.feature = feature_name
        self.report_every = report_every if report_every > 0 else max(1, total // 10)

    def tick(self) -> None:
        with self._lock:
            self._n += 1
            n = self._n
        if n % self.report_every == 0 or n == self.total:
            print(f"    [{self.feature}] {n}/{self.total} permutaciones "
                  f"({n / self.total * 100:.0f}%)", flush=True)


# ---------------------------------------------------------------------------
# Luminance CSV map (video_id → filename)
# ---------------------------------------------------------------------------
LUMINANCE_CSV_MAP: dict[int, str] = {
    3:  "green_intensity_video_3.csv",
    7:  "green_intensity_video_7.csv",
    9:  "green_intensity_video_9.csv",
    12: "green_intensity_video_12.csv",
}

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

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _events_path(subject: str, task: str, acq: str, run: str) -> Path | None:
    base = f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}_run-{run}"
    merged = (MERGED_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
              / f"{base}_desc-merged_events.tsv")
    if merged.exists():
        return merged
    preproc = (PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
               / f"{base}_desc-preproc_events.tsv")
    return preproc if preproc.exists() else None


def _eeg_path(subject: str, task: str, acq: str, run: str) -> Path | None:
    base = (PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
            / f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
              f"_run-{run}_desc-preproc_eeg.vhdr")
    return base if base.exists() else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(subject: str) -> list[dict]:
    """Load EEG segments and luminance CSVs for all runs.

    Returns list of dicts with keys:
      run_label, eeg_segment (n_ch, n_samples), sfreq,
      lum_times, lum_vals, lum_deriv, fps, video_id
    """
    runs_cfg = RUNS_CONFIG.get(subject)
    if not runs_cfg:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    result = []
    vid_count: dict[int, int] = {}

    for rc in runs_cfg:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        run_label = f"task-{task}_acq-{acq}_run-{run_id}"

        eeg_path = _eeg_path(subject, task, acq, run_id)
        events_path = _events_path(subject, task, acq, run_id)

        if not eeg_path or not events_path:
            print(f"  SKIP {run_label} — files missing")
            continue

        raw = mne.io.read_raw_brainvision(str(eeg_path), preload=True, verbose=False)
        events_df = pd.read_csv(events_path, sep="\t")
        lum_evs = events_df[events_df["trial_type"] == "video_luminance"]

        if lum_evs.empty:
            print(f"  SKIP {run_label} — no video_luminance events")
            continue

        sfreq = raw.info["sfreq"]
        available_chs = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]

        for _, ev_row in lum_evs.iterrows():
            stim_id = int(ev_row["stim_id"])
            video_id = stim_id - 100
            onset_s = float(ev_row["onset"])
            duration_s = float(ev_row["duration"])

            csv_name = LUMINANCE_CSV_MAP.get(video_id)
            if csv_name is None:
                continue
            csv_path = STIMULI_PATH / csv_name
            if not csv_path.exists():
                print(f"  SKIP {run_label} — CSV not found: {csv_path.name}")
                continue

            vid_count[video_id] = vid_count.get(video_id, 0) + 1
            pres = vid_count[video_id]
            segment_label = f"{run_label}_vid{video_id}_pres{pres}"

            # Crop EEG to luminance segment
            try:
                seg = raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)
            except ValueError as e:
                print(f"  SKIP {segment_label} — crop error: {e}")
                continue

            eeg_data = seg.get_data(picks=available_chs)  # (n_ch, n_samples)

            # Load luminance signal
            lum_df = pd.read_csv(csv_path)
            lum_times = lum_df["timestamp"].values.astype(float)
            lum_vals = lum_df["luminance"].values.astype(float)
            fps = 1.0 / np.median(np.diff(lum_times))
            lum_deriv = np.diff(lum_vals)  # frame-level ΔL (len = len(lum_vals) - 1)

            result.append({
                "run_label":   segment_label,
                "eeg_segment": eeg_data,
                "sfreq":       sfreq,
                "lum_times":   lum_times,
                "lum_vals":    lum_vals,
                "lum_deriv":   lum_deriv,
                "fps":         fps,
                "video_id":    video_id,
                "n_ch":        len(available_chs),
            })
            print(f"  Loaded {segment_label}: EEG {eeg_data.shape}, "
                  f"lum {len(lum_vals)} frames @ {fps:.1f}fps")

    return result


# ---------------------------------------------------------------------------
# Event detection (aligned with script 21d)
# ---------------------------------------------------------------------------

def detect_events(
    lum_deriv: np.ndarray,
    lum_times: np.ndarray,
    fps: float,
    threshold: float = DERIV_THRESHOLD,
) -> dict:
    """Detect non-overlapping ChangeUp/ChangeDown events from frame derivative.

    Returns dict with:
      ChangeUp / ChangeDown: arrays of crossing frame indices (in lum_deriv coords)
      occupied: set of frame indices reserved by change events
    """
    guard_frames = int(GUARD_POST_S * fps)  # 500ms in frames

    above = lum_deriv > threshold
    crossings_up = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    below = lum_deriv < -threshold
    crossings_down = np.where(np.diff(below.astype(int)) == 1)[0] + 1

    all_events = sorted(
        [(idx, "ChangeUp")   for idx in crossings_up] +
        [(idx, "ChangeDown") for idx in crossings_down]
    )

    selected_up: list[int] = []
    selected_down: list[int] = []
    occupied: set[int] = set()

    for idx, cond in all_events:
        event_end = idx + guard_frames
        if event_end >= len(lum_deriv):
            continue  # not enough room for all 6 windows
        epoch_range = set(range(idx, event_end))
        if epoch_range & occupied:
            continue
        occupied |= epoch_range
        (selected_up if cond == "ChangeUp" else selected_down).append(idx)

    return {
        "ChangeUp":   np.array(selected_up,   dtype=int),
        "ChangeDown": np.array(selected_down, dtype=int),
        "occupied":   occupied,
    }


def sample_nochange_frames(
    lum_deriv: np.ndarray,
    lum_times: np.ndarray,
    fps: float,
    occupied: set[int],
    n_target: int,
    nc_threshold: float = NC_THRESHOLD,
    prior_s: float = NC_PRIOR_S,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample NoChange event onsets from stable regions.

    A frame f is a valid NoChange event onset if:
    1. |lum_deriv| < nc_threshold for all frames in [f - prior_frames, f + guard_frames]
       (stable for 1s prior AND full 500ms epoch window)
    2. The range [f, f + guard_frames] doesn't overlap with any change event
    3. NC events are non-overlapping (greedy selection with 500ms guard)

    Each selected frame generates 6 windows (same offsets as change events).
    Returns array of n_target event onset frame indices.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    guard_frames = int(GUARD_POST_S * fps)   # 500ms — full NC epoch span
    prior_frames = int(prior_s * fps)        # 1s prior stability
    n_deriv = len(lum_deriv)

    stable = np.abs(lum_deriv) < nc_threshold

    candidates = []
    for f in range(prior_frames, n_deriv - guard_frames):
        # Stability: 1s prior AND 500ms forward
        if not np.all(stable[f - prior_frames: f + guard_frames]):
            continue
        # No overlap with change events
        if set(range(f, f + guard_frames)) & occupied:
            continue
        candidates.append(f)

    if not candidates:
        return np.array([], dtype=int)

    # Greedy non-overlapping selection (random order, fixed seed)
    candidates_arr = np.array(candidates, dtype=int)
    order = rng.permutation(len(candidates_arr))
    selected: list[int] = []
    sel_occupied: set[int] = set()

    for i in order:
        if len(selected) >= n_target:
            break
        f = int(candidates_arr[i])
        epoch_range = set(range(f, f + guard_frames))
        if epoch_range & sel_occupied:
            continue
        selected.append(f)
        sel_occupied |= epoch_range

    return np.sort(np.array(selected, dtype=int))


# ---------------------------------------------------------------------------
# EEG window extraction
# ---------------------------------------------------------------------------

def extract_windows_for_run(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract all EEG windows (raw) and labels for one run.

    For each ChangeUp/ChangeDown event: 6 windows at offsets [0,50,...,250ms].
    For each NoChange event onset: 6 windows at same offsets.
    Class balance: n_nc_events = (n_up_events + n_down_events) // 2
      → NoChange windows ≈ ChangeUp windows ≈ ChangeDown windows.

    Returns (windows, labels):
      windows: (n_win, n_ch, win_samples)
      labels:  (n_win,) with values in {0, 1, 2}
    """
    eeg = run["eeg_segment"]    # (n_ch, n_samples)
    sfreq = run["sfreq"]
    lum_deriv = run["lum_deriv"]
    lum_times = run["lum_times"]
    fps = run["fps"]

    events = detect_events(lum_deriv, lum_times, fps)
    n_ch, n_samples = eeg.shape
    win_samples = int(WIN_SIZE_S * sfreq)

    windows: list[np.ndarray] = []
    labels:  list[int] = []

    def _add_event_windows(frame_indices: np.ndarray, label: int) -> None:
        for f in frame_indices:
            event_t = lum_times[f]
            for offset_s in WIN_OFFSETS_S:
                start_smp = int((event_t + offset_s) * sfreq)
                end_smp = start_smp + win_samples
                if start_smp < 0 or end_smp > n_samples:
                    continue
                windows.append(eeg[:, start_smp:end_smp])
                labels.append(label)

    _add_event_windows(events["ChangeUp"],   LABEL_UP)
    _add_event_windows(events["ChangeDown"], LABEL_DOWN)

    # Balance: n_nc_events = (n_up + n_down) // 2 → equal windows per class
    n_up_events   = len(events["ChangeUp"])
    n_down_events = len(events["ChangeDown"])
    n_nc_target   = (n_up_events + n_down_events) // 2

    nc_frames = sample_nochange_frames(
        lum_deriv, lum_times, fps,
        events["occupied"],
        n_target=n_nc_target,
    )

    # NC: 6 windows per event (same offsets as change events)
    _add_event_windows(nc_frames, LABEL_NC)

    run["_events"]    = events
    run["_nc_frames"] = nc_frames

    if not windows:
        return np.empty((0, n_ch, win_samples)), np.empty(0, dtype=int)

    return np.stack(windows), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_bandpower(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Welch PSD per channel per band → 160 features."""
    n_ch, n_samp = window.shape
    nperseg = min(n_samp, 64)
    freqs, psd = scipy_welch(window, fs=sfreq, nperseg=nperseg, axis=-1)
    feats = []
    for band_lo, band_hi in SPECTRAL_BANDS.values():
        mask = (freqs >= band_lo) & (freqs < band_hi)
        vals = psd[:, mask].mean(axis=-1) if mask.any() else np.zeros(n_ch)
        feats.append(vals)
    return np.concatenate(feats)


def build_feature_matrix_bandpower(
    windows: np.ndarray, sfreq: float
) -> np.ndarray:
    """(n_win, n_ch, win_samp) → (n_win, 160)."""
    return np.array([_extract_bandpower(w, sfreq) for w in windows])


def build_feature_matrix_raw(windows: np.ndarray) -> np.ndarray:
    """Flatten (n_win, n_ch, win_samp) → (n_win, n_ch*win_samp)."""
    n_win = windows.shape[0]
    return windows.reshape(n_win, -1)


def build_feature_matrix_tde(
    windows: np.ndarray,
    eeg_segment: np.ndarray,
    sfreq: float,
    window_start_samples: np.ndarray,
    pca_model: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """TDE(±10) on full segment → global PCA → covariance per window.

    pca_model: if provided, use it (test fold); if None, fit on this data (train fold).
    Returns (features (n_win, 153), fitted_pca).
    """
    n_ch, n_seg_samples = eeg_segment.shape
    n_tde_half = TDE_WINDOW_HALF
    win_samp = windows.shape[-1]

    # Apply TDE to the full 60s segment
    seg_time_major = eeg_segment.T                          # (n_samples, n_ch)
    indices = np.array([[0, n_seg_samples]])
    tde_data, _ = apply_tde_only(
        eeg_data=seg_time_major,
        indices=indices,
        tde_lags=n_tde_half,
    )
    # tde_data: (n_seg_samples - 2*n_tde_half, n_ch*(2*n_tde_half+1)) = (N, 672)

    # Extract TDE rows for each window
    tde_wins = []
    n_tde = tde_data.shape[0]
    for s in window_start_samples:
        # TDE row for original sample s = s - n_tde_half
        tde_start = s - n_tde_half
        tde_end = tde_start + win_samp
        if tde_start < 0 or tde_end > n_tde:
            tde_wins.append(None)
        else:
            tde_wins.append(tde_data[tde_start:tde_end])  # (win_samp, 672)

    # Stack valid TDE windows for PCA
    valid_idx = [i for i, w in enumerate(tde_wins) if w is not None]
    if not valid_idx:
        n_feats = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
        return np.zeros((len(windows), n_feats)), PCA(n_components=TDE_PCA_COMPONENTS)

    stacked = np.vstack([tde_wins[i] for i in valid_idx])  # (sum_wins*win_samp, 672)

    if pca_model is None:
        n_comp = min(TDE_PCA_COMPONENTS, stacked.shape[0] - 1, stacked.shape[1])
        pca_model = PCA(n_components=n_comp).fit(stacked)

    n_feats = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
    features = np.zeros((len(windows), n_feats))

    for pos, i in enumerate(valid_idx):
        pca_proj = pca_model.transform(tde_wins[i])        # (win_samp, n_comp)
        cov = np.cov(pca_proj.T)                           # (n_comp, n_comp)
        idx_u = np.triu_indices(cov.shape[0])
        features[pos] = cov[idx_u][:n_feats]

    return features, pca_model


# ---------------------------------------------------------------------------
# LORO CV
# ---------------------------------------------------------------------------

def _run_one_fold(
    test_idx: int,
    run_features: list[np.ndarray],
    run_labels: list[np.ndarray],
    use_pca: bool,
    run_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_runs = len(run_features)
    X_train = np.vstack([run_features[i] for i in range(n_runs) if i != test_idx])
    y_train = np.concatenate([run_labels[i] for i in range(n_runs) if i != test_idx])
    X_test = run_features[test_idx]
    y_test = run_labels[test_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if use_pca:
        n_comp = min(RAW_PCA_COMPONENTS, X_train_sc.shape[0] - 1, X_train_sc.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_sc = pca.fit_transform(X_train_sc)
        X_test_sc = pca.transform(X_test_sc)

    clf = LogisticRegression(C=FIXED_C, max_iter=500, solver="lbfgs")
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)

    fold_acc = float(accuracy_score(y_test, y_pred))
    fold_info = {
        "fold": run_names[test_idx],
        "accuracy": fold_acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    return y_test, y_pred, y_prob, fold_info


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES.keys())).tolist()
    per_class = {
        CLASS_NAMES[c]: float(
            accuracy_score(y_true[y_true == c], y_pred[y_true == c])
        ) if (y_true == c).any() else float("nan")
        for c in range(N_CLASSES)
    }
    return {"accuracy": acc, "f1_macro": f1, "auc_roc_macro": auc,
            "per_class_accuracy": per_class, "confusion_matrix": cm}


def run_loro(
    run_features: list[np.ndarray],
    run_labels: list[np.ndarray],
    run_names: list[str],
    feature_name: str,
) -> dict:
    """LORO CV for any feature set."""
    use_pca = (feature_name == "raw_pca")
    n_runs = len(run_features)

    fold_results = Parallel(n_jobs=-3)(
        delayed(_run_one_fold)(i, run_features, run_labels, use_pca, run_names)
        for i in range(n_runs)
    )

    all_y_true, all_y_pred, all_y_prob, fold_metrics = [], [], [], []
    for y_test, y_pred, y_prob, info in fold_results:
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())
        fold_metrics.append(info)
        print(f"      {info['fold']}: acc={info['accuracy']:.3f}")

    m = _metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob))
    return {"feature": feature_name, **m, "folds": fold_metrics}


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def save_diagnostic_plot(run: dict, output_dir: Path) -> None:
    """Save derivative timeline with highlighted windows for one run."""
    lum_times  = run["lum_times"]
    lum_vals   = run["lum_vals"]
    lum_deriv  = run["lum_deriv"]
    events     = run.get("_events", {})
    nc_frames  = run.get("_nc_frames", np.array([], dtype=int))
    run_label  = run["run_label"]
    fps        = run["fps"]
    sfreq      = run["sfreq"]

    t_deriv = lum_times[1:]  # derivative is one shorter

    fig, (ax_d, ax_l) = plt.subplots(2, 1, figsize=(20, 6), sharex=True)

    ax_d.plot(t_deriv, lum_deriv, color="0.4", lw=0.5)
    ax_d.axhline(DERIV_THRESHOLD, color="C3", ls=":", lw=0.8)
    ax_d.axhline(-DERIV_THRESHOLD, color="C0", ls=":", lw=0.8)
    ax_d.axhline(NC_THRESHOLD, color="C2", ls=":", lw=0.6, alpha=0.5)
    ax_d.axhline(-NC_THRESHOLD, color="C2", ls=":", lw=0.6, alpha=0.5)
    ax_d.set_ylabel("ΔLuminance / frame")

    ax_l.plot(lum_times, lum_vals, color="0.4", lw=0.6)
    ax_l.set_ylabel("Luminance (green ch.)")
    ax_l.set_xlabel("Time (s)")

    cond_cfg = {
        "ChangeUp":   (events.get("ChangeUp",   np.array([], int)), "C3", LABEL_UP),
        "ChangeDown": (events.get("ChangeDown", np.array([], int)), "C0", LABEL_DOWN),
    }
    plotted = set()
    for cname, (frames, color, _) in cond_cfg.items():
        for f in frames:
            t_evt = lum_times[f]
            lbl = cname if cname not in plotted else None
            for ax in (ax_d, ax_l):
                for o in WIN_OFFSETS_S:
                    ax.axvspan(t_evt + o, t_evt + o + WIN_SIZE_S,
                               alpha=0.12, color=color,
                               label=lbl if (ax is ax_d and lbl) else None)
                    lbl = None
            plotted.add(cname)

    # NoChange
    for f in nc_frames:
        t_nc = lum_times[f]
        lbl = "NoChange" if "NoChange" not in plotted else None
        for ax in (ax_d, ax_l):
            ax.axvspan(t_nc, t_nc + WIN_SIZE_S, alpha=0.10, color="C2",
                       label=lbl if ax is ax_d else None)
        plotted.add("NoChange")

    n_up   = len(events.get("ChangeUp",   []))
    n_down = len(events.get("ChangeDown", []))
    n_nc   = len(nc_frames)
    ax_d.set_title(
        f"{run_label} | thr={DERIV_THRESHOLD} | "
        f"Up={n_up} ({n_up*6} wins), Down={n_down} ({n_down*6} wins), "
        f"NC={n_nc} wins"
    )
    ax_d.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = output_dir / f"{run_label}_diagnostic.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Diagnostic plot: {out.name}")


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(all_results: list[dict], output_dir: Path, subject: str) -> None:
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    class_labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    for ax, res in zip(axes, all_results):
        cm = np.array(res["confusion_matrix"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(class_labels)
        for r in range(N_CLASSES):
            for c in range(N_CLASSES):
                ax.text(c, r, f"{cm[r,c]}", ha="center", va="center", fontsize=9)
        ax.set_title(f"{res['feature']}\nAcc={res['accuracy']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"3-class luminance decoding (sub-{subject}, LORO CV)", fontsize=10)
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_3class_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Confusion matrix saved: {out.name}")


def plot_per_class_accuracy(all_results: list[dict], output_dir: Path, subject: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(N_CLASSES)
    width = 0.8 / len(all_results)
    class_labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    for k, res in enumerate(all_results):
        vals = [res["per_class_accuracy"].get(CLASS_NAMES[c], float("nan")) for c in range(N_CLASSES)]
        ax.bar(x + k * width, vals, width, label=res["feature"])
    ax.axhline(1 / N_CLASSES, color="gray", ls="--", lw=1, label="chance (33.3%)")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class accuracy (sub-{subject}, LORO CV, 3-class luminance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_3class_per_class_acc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Per-class accuracy saved: {out.name}")


# ---------------------------------------------------------------------------
# Permutation test helpers
# ---------------------------------------------------------------------------

def _precompute_splits(
    run_features: list[np.ndarray],
    run_labels: list[np.ndarray],
    feature_name: str,
) -> list[tuple]:
    """Pre-compute scaled (and optionally PCA-reduced) train/test splits for all folds.

    Returns list of (X_train_sc, X_test_sc, y_test, train_run_sizes, train_run_indices).
    """
    n_runs = len(run_features)
    use_pca = (feature_name == "raw_pca")
    splits = []
    for test_idx in range(n_runs):
        train_indices = [i for i in range(n_runs) if i != test_idx]
        X_train = np.vstack([run_features[i] for i in train_indices])
        X_test = run_features[test_idx]
        train_run_sizes = [len(run_features[i]) for i in train_indices]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        if use_pca:
            n_comp = min(RAW_PCA_COMPONENTS, X_train_sc.shape[0] - 1, X_train_sc.shape[1])
            pca = PCA(n_components=n_comp)
            X_train_sc = pca.fit_transform(X_train_sc)
            X_test_sc = pca.transform(X_test_sc)

        splits.append((X_train_sc, X_test_sc, run_labels[test_idx],
                       train_run_sizes, train_indices))
    return splits


def _run_one_permutation(
    splits: list[tuple],
    run_labels: list[np.ndarray],
    rng: np.random.Generator,
    counter: _PermCounter | None = None,
) -> float:
    """One permutation: shuffle labels within each run, return LORO accuracy."""
    all_y_true, all_y_pred = [], []
    for X_train_sc, X_test_sc, y_test, train_run_sizes, train_run_indices in splits:
        # Shuffle labels within each training run independently
        y_train_parts = []
        for run_i, size in zip(train_run_indices, train_run_sizes):
            perm = rng.permutation(size)
            y_train_parts.append(run_labels[run_i][perm])
        y_train_perm = np.concatenate(y_train_parts)

        clf = LogisticRegression(C=FIXED_C, max_iter=500, solver="lbfgs")
        clf.fit(X_train_sc, y_train_perm)
        y_pred = clf.predict(X_test_sc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    if counter is not None:
        counter.tick()
    return float(accuracy_score(all_y_true, all_y_pred))


def run_permutation_test(
    run_features: list[np.ndarray],
    run_labels: list[np.ndarray],
    run_names: list[str],
    feature_name: str,
    obs_acc: float,
    n_perm: int,
) -> dict:
    """Permutation test: label-shuffle within runs, n_perm iterations."""
    import time as _t

    print(f"\n  Permutation test: {feature_name} (n={n_perm})")
    print(f"    Pre-computing scaled fold splits (one-time cost)...")
    splits = _precompute_splits(run_features, run_labels, feature_name)

    # Time one permutation sequentially
    rng0 = np.random.default_rng(RANDOM_SEED)
    t0 = _t.time()
    acc0 = _run_one_permutation(splits, run_labels, rng0)
    t_one = _t.time() - t0

    n_workers = max(1, (os.cpu_count() or 4) + 3 - 1)  # approximation of n_jobs=-3
    est_min = t_one * (n_perm - 1) / n_workers / 60
    print(f"    Fixed C={FIXED_C}, lbfgs, permutations parallelizadas (n_jobs=-3, prefer=threads)")
    print(f"    Timing 1ra permutacion: {t_one:.1f}s  →  estimado n={n_perm}: ~{est_min:.1f} min")

    counter = _PermCounter(n_perm - 1, feature_name)

    def _perm_job(seed: int) -> float:
        return _run_one_permutation(splits, run_labels,
                                    np.random.default_rng(seed), counter)

    rest = Parallel(n_jobs=-3, prefer="threads")(
        delayed(_perm_job)(RANDOM_SEED + i + 1) for i in range(n_perm - 1)
    )

    acc_perm = np.array([acc0] + rest)
    p_val = float((acc_perm >= obs_acc).mean())
    null_mean = float(acc_perm.mean())
    null_std = float(acc_perm.std())
    z = float((obs_acc - null_mean) / (null_std + 1e-9))

    print(f"\n    === Permutation test result ({feature_name}) ===")
    print(f"    Observed acc : {obs_acc:.4f}")
    print(f"    Null mean±std: {null_mean:.4f} ± {null_std:.4f}")
    print(f"    p-value      : {p_val:.4f}  (n_perm={n_perm})")
    print(f"    z-score      : {z:.2f}")

    return {
        "feature": feature_name,
        "observed_acc": float(obs_acc),
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_val,
        "z_score": z,
        "n_perm": n_perm,
        "null_distribution": acc_perm.tolist(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str, features: list[str], n_permutations: int = 0) -> None:
    print("=" * 60)
    print(f"36 -- 3-class luminance decoding -- sub-{subject}")
    print(f"     Threshold: ChangeUp/Down > {DERIV_THRESHOLD} | NC < {NC_THRESHOLD}")
    print(f"     Windows: {int(WIN_SIZE_S*1000)}ms x6 offsets, tope={int(GUARD_POST_S*1000)}ms")
    print(f"     NC prior: {NC_PRIOR_S}s stability required")
    print(f"     Classifier: LogisticRegression(C={FIXED_C}, lbfgs)")
    print("=" * 60)

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    diag_dir = output_dir / "diagnostic_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading EEG + luminance data per run...")
    runs = load_run_data(subject)
    if len(runs) < 2:
        print("Need at least 2 runs for LORO CV. Aborting.")
        sys.exit(1)

    print(f"\nExtracting windows for {len(runs)} runs...")
    run_raw_windows: list[np.ndarray] = []
    run_labels_all:  list[np.ndarray] = []
    run_names:       list[str] = []

    for run in runs:
        windows, labels = extract_windows_for_run(run)
        if len(windows) == 0:
            print(f"  WARNING: {run['run_label']} produced 0 windows, skipping")
            continue
        counts = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(N_CLASSES)}
        print(f"  {run['run_label']}: {len(windows)} windows {counts}")
        run_raw_windows.append(windows)
        run_labels_all.append(labels)
        run_names.append(run["run_label"])
        save_diagnostic_plot(run, diag_dir)

    if len(run_raw_windows) < 2:
        print("Not enough valid runs for LORO CV. Aborting.")
        sys.exit(1)

    all_results = []
    n_features_total = len(features)

    for feat_idx, feat in enumerate(features, start=1):
        print(f"\n  [{feat_idx}/{n_features_total}] Feature set: {feat}")
        import time as _t; t0 = _t.time()

        if feat == "bandpower_welch":
            run_feats = [
                build_feature_matrix_bandpower(w, run_raw_windows[i][:, 0, :].shape[0]
                                               if False else runs[i]["sfreq"])
                for i, w in enumerate(run_raw_windows)
            ]
        elif feat == "raw_pca":
            run_feats = [build_feature_matrix_raw(w) for w in run_raw_windows]
        elif feat == "tde_cov":
            run_feats = _build_tde_features(runs, run_raw_windows, run_names)
        else:
            print(f"  Unknown feature set: {feat}, skipping")
            continue

        # Check all runs have at least some features
        valid = all(f.shape[0] > 0 for f in run_feats)
        if not valid:
            print(f"  Skipping {feat} — some runs have 0 windows")
            continue

        res = run_loro(run_feats, run_labels_all, run_names, feat)
        t_elapsed = _t.time() - t0
        all_results.append(res)
        print(f"    ✓ LORO listo ({t_elapsed:.1f}s) — "
              f"Acc={res['accuracy']:.3f}  F1={res['f1_macro']:.3f}  "
              f"AUC={res['auc_roc_macro']:.3f}")
        print(f"    Per-class: {res['per_class_accuracy']}")

        if n_permutations > 0:
            perm_res = run_permutation_test(
                run_feats, run_labels_all, run_names,
                feat, res["accuracy"], n_permutations,
            )
            res["permutation_test"] = perm_res
            print(f"    ✓ Feature set {feat_idx}/{n_features_total} permutation done "
                  f"(p={perm_res['p_value']:.4f}, z={perm_res['z_score']:.2f})")

    if not all_results:
        print("No results produced. Check data.")
        sys.exit(1)

    json_path = output_dir / f"sub-{subject}_3class_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {json_path.name}")

    if n_permutations > 0:
        perm_results = [r["permutation_test"] for r in all_results if "permutation_test" in r]
        if perm_results:
            perm_json = output_dir / f"sub-{subject}_3class_permutation.json"
            with open(perm_json, "w") as f:
                json.dump(perm_results, f, indent=2)
            print(f"  Permutation results saved: {perm_json.name}")

    plot_confusion_matrices(all_results, output_dir, subject)
    plot_per_class_accuracy(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")

    # Print summary
    print("\n" + "=" * 65)
    print(f"  SUMMARY — sub-{subject} — 3-class luminance decoding")
    print(f"  Chance level: {1/N_CLASSES:.3f} (33.3%)")
    print("=" * 65)
    if n_permutations > 0:
        print(f"  {'Feature':<20} {'Acc':>6} {'F1':>6} {'AUC':>6} {'p-val':>8} {'z':>6}")
        print("  " + "-" * 55)
        for res in all_results:
            pt = res.get("permutation_test", {})
            p_str = f"{pt['p_value']:.4f}" if pt else "   —"
            z_str = f"{pt['z_score']:.2f}" if pt else "  —"
            print(f"  {res['feature']:<20} {res['accuracy']:>6.3f} "
                  f"{res['f1_macro']:>6.3f} {res['auc_roc_macro']:>6.3f} "
                  f"{p_str:>8} {z_str:>6}")
    else:
        print(f"  {'Feature':<20} {'Acc':>6} {'F1':>6} {'AUC':>6}")
        print("  " + "-" * 40)
        for res in all_results:
            print(f"  {res['feature']:<20} {res['accuracy']:>6.3f} "
                  f"{res['f1_macro']:>6.3f} {res['auc_roc_macro']:>6.3f}")
    print("=" * 65)


def _build_tde_features(
    runs: list[dict],
    run_raw_windows: list[np.ndarray],
    run_names: list[str],
) -> list[np.ndarray]:
    """Build TDE covariance features for all runs (LORO-aware PCA fit)."""
    # We need to compute TDE per segment and store, then fit PCA per fold.
    # For simplicity: precompute TDE time series per run, then do fold-wise PCA.
    n_runs = len(run_raw_windows)
    n_feats = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2

    # Map run_names back to runs list
    name_to_run = {r["run_label"]: r for r in runs}

    # For each run, get the window start samples (in the cropped segment)
    def _get_start_samples(run_name: str, run_wins: np.ndarray) -> np.ndarray:
        run = name_to_run[run_name]
        sfreq = run["sfreq"]
        # Recover events from run
        events = run.get("_events", {})
        nc_frames = run.get("_nc_frames", np.array([], int))
        lum_times = run["lum_times"]
        starts = []
        for label_src, frames in [("up", events.get("ChangeUp", [])),
                                   ("down", events.get("ChangeDown", []))]:
            for f in frames:
                t_evt = lum_times[f]
                for o in WIN_OFFSETS_S:
                    s = int((t_evt + o) * sfreq)
                    if 0 <= s + int(WIN_SIZE_S * sfreq) <= run["eeg_segment"].shape[1]:
                        starts.append(s)
        for f in nc_frames:
            t_nc = lum_times[f]
            for o in WIN_OFFSETS_S:
                s = int((t_nc + o) * sfreq)
                if 0 <= s + int(WIN_SIZE_S * sfreq) <= run["eeg_segment"].shape[1]:
                    starts.append(s)
        return np.array(starts, dtype=int)

    # Precompute TDE per run
    print(f"    Pre-computing TDE per run...")
    tde_per_run = []
    starts_per_run = []
    for i, run_name in enumerate(run_names):
        run = name_to_run[run_name]
        seg = run["eeg_segment"]
        n_ch, n_samp = seg.shape
        seg_tm = seg.T                             # (n_samp, n_ch)
        indices = np.array([[0, n_samp]])
        tde_data, _ = apply_tde_only(seg_tm, indices, TDE_WINDOW_HALF)
        tde_per_run.append(tde_data)               # (n_tde, 672)
        starts = _get_start_samples(run_name, run_raw_windows[i])
        starts_per_run.append(starts)
        print(f"      {run_name}: TDE {tde_data.shape}")

    # For each LORO fold, fit PCA on training runs, transform test run
    run_features_tde: list[np.ndarray] = [None] * n_runs  # type: ignore

    for test_idx in range(n_runs):
        # Stack all training TDE timepoints for global PCA
        train_tde = np.vstack([tde_per_run[i] for i in range(n_runs) if i != test_idx])
        n_comp = min(TDE_PCA_COMPONENTS, train_tde.shape[0] - 1, train_tde.shape[1])
        pca = PCA(n_components=n_comp).fit(train_tde)
        print(f"      Fold {test_idx+1}/{n_runs} PCA: "
              f"{n_comp} components ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")

        # Build features for the test run
        tde_test = tde_per_run[test_idx]
        starts = starts_per_run[test_idx]
        n_tde = tde_test.shape[0]
        win_samp = int(WIN_SIZE_S * runs[0]["sfreq"])

        feats = np.zeros((len(starts), n_feats))
        for j, s in enumerate(starts):
            tde_s = s - TDE_WINDOW_HALF
            tde_e = tde_s + win_samp
            if tde_s < 0 or tde_e > n_tde:
                continue
            pca_proj = pca.transform(tde_test[tde_s:tde_e])  # (win_samp, n_comp)
            cov = np.cov(pca_proj.T)
            idx_u = np.triu_indices(cov.shape[0])
            feats[j, :len(idx_u[0])] = cov[idx_u]

        # Also build for all other runs (training), but LORO only needs test features
        # We return features for test run, but LORO also needs training features
        # → We'll compute all folds' features and then reconstruct per-run matrices
        if run_features_tde[test_idx] is None:
            run_features_tde[test_idx] = feats

    # Above gives test features per fold — but we need a consistent per-run matrix.
    # Since PCA varies per fold, we use the leave-one-out PCA for each run's test features.
    return run_features_tde


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-class luminance decoding (Task 3)"
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument(
        "--features", nargs="+",
        choices=["bandpower_welch", "tde_cov", "raw_pca"],
        default=["bandpower_welch", "tde_cov", "raw_pca"],
    )
    parser.add_argument(
        "--permute", type=int, default=0, metavar="N",
        help="Run permutation test with N iterations per feature set (0 = skip)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.subject, args.features, n_permutations=args.permute)
