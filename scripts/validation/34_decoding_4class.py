#!/usr/bin/env python
"""4-class temporal decoding: Baseline / ChangeUp / Luminance / ChangeDown.

Task 10.3 from the research diary.

For each CHANGE_PHOTO trial, extracts 100ms sliding windows (50ms step) from
4 temporal regions and classifies them with LORO CV:

  Class 0 - Baseline:   [-250, 0ms] + [1500, 1750ms] (pre-onset + post-return, pooled)
  Class 1 - ChangeUp:   [0, 500ms]  (onset of luminance increase)
  Class 2 - Luminance:  [500, 1000ms] (sustained luminance)
  Class 3 - ChangeDown: [1000, 1500ms] (return to baseline)

Windows per trial: Baseline=8, ChangeUp=9, Luminance=9, ChangeDown=9 (minor imbalance).

Feature sets (~160 features each):
  bandpower_welch : 32 ch x 5 bands = 160 features
  tde_cov         : TDE(±10 lags) -> PCA(17) -> cov upper triangle = 153 features
  raw_pca         : vectorize -> PCA(160) = 160 features

Classifier: LogisticRegressionCV (L2, C cross-validated per feature set),
Leave-One-Run-Out CV outer loop.

Usage
-----
    micromamba run -n campeones python scripts/validation/34_decoding_4class.py --subject 27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch as scipy_welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_4class"

SESSION = "vr"

# Wide epoch: must cover [-250ms, +1750ms] for class windows + TDE margin
WIDE_TMIN = -1.5
WIDE_TMAX = 2.0
BASELINE = (-1.5, -1.0)

# Sliding window parameters
WIN_SIZE_S = 0.250   # 250ms
WIN_STEP_S = 0.050   # 50ms step → 80% overlap

# Class definitions: label -> list of (tmin, tmax) segments to draw windows from
# All classes produce 6 windows (perfectly balanced):
#   250ms window + 50ms step over 500ms segment → 6 windows
CLASS_SEGMENTS = {
    0: [(-0.500, 0.000)],   # Baseline: 500ms pre-onset
    1: [(0.000, 0.500)],    # ChangeUp
    2: [(0.500, 1.000)],    # Luminance
    3: [(1.000, 1.500)],    # ChangeDown
}
CLASS_NAMES = {0: "Baseline", 1: "ChangeUp", 2: "Luminance", 3: "ChangeDown"}
N_CLASSES = 4

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
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
INNER_CV_FOLDS = 5
RANDOM_SEED = 42
TDE_WINDOW_HALF = 10
TDE_PCA_COMPONENTS = 17   # -> 17*18//2 = 153 cov features  (≈ bandpower 160)
RAW_PCA_COMPONENTS = 160  # ≈ bandpower 160


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def get_class_windows() -> list[tuple[float, float, int]]:
    """Return all (t_start, t_end, class_label) for one trial.

    Baseline:   8 windows (4 pre + 4 post)
    ChangeUp:   9 windows
    Luminance:  9 windows
    ChangeDown: 9 windows
    """
    windows = []
    for label, segments in CLASS_SEGMENTS.items():
        for seg_start, seg_end in segments:
            t = seg_start
            while t + WIN_SIZE_S <= seg_end + 1e-9:
                windows.append((round(t, 6), round(t + WIN_SIZE_S, 6), label))
                t = round(t + WIN_STEP_S, 6)
    return windows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_epochs_per_run(subject: str) -> list[tuple[np.ndarray, float, str]]:
    """Load wide epochs per run.

    Returns list of (data_wide, sfreq, run_label).
    data_wide: (n_onsets, n_ch, n_times_wide)
    """
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    all_runs = []
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
            print(f"  SKIP {label} -- files not found")
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
            epochs_wide = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=WIDE_TMIN, tmax=WIDE_TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            data_wide = epochs_wide.get_data()  # (n, ch, times)
            n = data_wide.shape[0]
            print(f"  {label}: {n} onsets")
            all_runs.append((data_wide, float(sfreq), label))
        except Exception as exc:
            print(f"  {label}: error -- {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_bandpower_window(segment: np.ndarray, sfreq: float) -> np.ndarray:
    """Bandpower per channel per band for a single window.

    Args:
        segment: (n_ch, n_times)
    Returns: (n_ch * n_bands,)
    """
    n_ch, n_times = segment.shape
    band_list = list(SPECTRAL_BANDS.values())
    features = np.empty(n_ch * len(band_list))
    feat_idx = 0
    for ch in range(n_ch):
        freqs, psd = scipy_welch(segment[ch], fs=sfreq, nperseg=n_times)
        for flo, fhi in band_list:
            mask = (freqs >= flo) & (freqs <= fhi)
            features[feat_idx] = (
                np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
            )
            feat_idx += 1
    return features


def _window_sample_indices(t_start: float, sfreq: float) -> tuple[int, int]:
    """Convert window start time (relative to WIDE_TMIN) to sample indices.

    Uses a fixed window length to avoid floating-point rounding mismatches.
    """
    n_win = int(round(WIN_SIZE_S * sfreq))
    s_start = int(round((t_start - WIDE_TMIN) * sfreq))
    return s_start, s_start + n_win


def _build_run_features_standard(
    data_wide: np.ndarray,
    sfreq: float,
    feature_name: str,
    class_windows: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract window-level features for one run (bandpower or raw).

    Returns: X (n_windows_total, n_features_raw), y (n_windows_total,)
    """
    n_trials = data_wide.shape[0]
    feats, labels = [], []
    for i in range(n_trials):
        epoch = data_wide[i]  # (n_ch, n_times_wide)
        for t_start, t_end, label in class_windows:
            s0, s1 = _window_sample_indices(t_start, sfreq)
            segment = epoch[:, s0:s1]  # (n_ch, n_win_samples)
            if feature_name == "bandpower_welch":
                feats.append(_extract_bandpower_window(segment, sfreq))
            else:
                feats.append(segment.reshape(-1))
            labels.append(label)
    return np.array(feats), np.array(labels)


# ---------------------------------------------------------------------------
# Classifier pipeline
# ---------------------------------------------------------------------------

def _build_pipeline() -> object:
    steps = [StandardScaler()]
    steps.append(LogisticRegressionCV(
        Cs=C_GRID, cv=INNER_CV_FOLDS, l1_ratios=(0,), solver="saga",
        max_iter=5000, random_state=RANDOM_SEED, scoring="accuracy",
        use_legacy_attributes=False, n_jobs=1,  # outer folds are already parallel
    ))
    return make_pipeline(*steps)


def _get_cov_mask(k: int, mode: str) -> np.ndarray:
    """Boolean mask over the upper-triangle (including diagonal) of a k×k matrix.

    mode:
        "full"    → all k*(k+1)//2 elements (default)
        "diag"    → only the k diagonal elements (power per PC)
        "offdiag" → only the k*(k-1)//2 off-diagonal elements (connectivity)
    """
    rows, cols = np.triu_indices(k)
    if mode == "full":
        return np.ones(len(rows), dtype=bool)
    elif mode == "diag":
        return rows == cols
    elif mode == "offdiag":
        return rows != cols
    raise ValueError(f"Unknown cov_mode: {mode!r}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    per_class = {
        CLASS_NAMES[c]: float(cm[c, c] / cm[c].sum()) if cm[c].sum() > 0 else 0.0
        for c in range(N_CLASSES)
    }
    return {"accuracy": acc, "f1_macro": f1, "auc_roc_macro": auc,
            "per_class_accuracy": per_class, "confusion_matrix": cm.tolist()}


# ---------------------------------------------------------------------------
# LORO CV — bandpower_welch and raw_pca
# ---------------------------------------------------------------------------

def _run_one_fold_standard(
    test_idx: int,
    run_features: list,
    run_labels: list,
    run_names: list,
    use_pca: bool,
) -> tuple:
    """Execute one LORO fold for bandpower_welch or raw_pca.

    Returns (y_test, y_pred, y_prob, fold_info_dict).
    """
    n_runs = len(run_features)
    X_train = np.vstack([run_features[i] for i in range(n_runs) if i != test_idx])
    y_train = np.concatenate([run_labels[i] for i in range(n_runs) if i != test_idx])
    X_test = run_features[test_idx]
    y_test = run_labels[test_idx]

    if use_pca:
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        n_comp = min(RAW_PCA_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train = pca.fit_transform(X_train_sc)
        X_test = pca.transform(X_test_sc)

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    best_c = float(np.atleast_1d(pipe[-1].C_)[0])
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    fold_acc = float(accuracy_score(y_test, y_pred))
    fold_info = {
        "fold": run_names[test_idx],
        "best_C": best_c,
        "accuracy": fold_acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    return y_test, y_pred, y_prob, fold_info


def run_loro_standard(runs_data: list, feature_name: str) -> dict:
    """LORO CV for bandpower_welch and raw_pca (folds run in parallel)."""
    use_pca = (feature_name == "raw_pca")
    class_windows = get_class_windows()

    print(f"    Extracting {feature_name} windows per run...")
    run_features, run_labels, run_names = [], [], []
    for data_wide, sfreq, run_label in runs_data:
        X, y = _build_run_features_standard(data_wide, sfreq, feature_name, class_windows)
        run_features.append(X)
        run_labels.append(y)
        run_names.append(run_label)

    n_runs = len(run_features)
    n_features_raw = run_features[0].shape[1]
    n_features_eff = (min(RAW_PCA_COMPONENTS, n_features_raw) if use_pca
                      else n_features_raw)
    print(f"    N features (raw): {n_features_raw}")

    fold_results = Parallel(n_jobs=-1)(
        delayed(_run_one_fold_standard)(
            test_idx, run_features, run_labels, run_names, use_pca
        )
        for test_idx in range(n_runs)
    )

    all_y_true, all_y_pred, all_y_prob, fold_metrics = [], [], [], []
    for y_test, y_pred, y_prob, fold_info in fold_results:
        all_y_true.extend(y_test.tolist() if hasattr(y_test, "tolist") else y_test)
        all_y_pred.extend(y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred)
        all_y_prob.extend(y_prob.tolist())
        fold_metrics.append(fold_info)
        print(f"      {fold_info['fold']}: acc={fold_info['accuracy']:.3f} "
              f"(best_C={fold_info['best_C']})")

    y_true = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    m = _metrics(y_true, y_pred_arr, y_prob_arr)
    return {"feature": feature_name, "n_features": int(n_features_eff),
            "n_features_raw": int(n_features_raw), **m, "folds": fold_metrics}


# ---------------------------------------------------------------------------
# LORO CV — tde_cov
# ---------------------------------------------------------------------------

def _run_one_fold_tde(
    test_idx: int,
    run_tde: list,
    run_names: list,
    tde_pca_components: int,
    cov_mode: str = "full",
) -> tuple:
    """Execute one LORO fold for tde_cov.

    Imports tde/feature helpers inside the function so joblib workers can
    pickle and call this without capturing module-level state.

    Returns (y_test, y_pred, y_prob, fold_info_dict).
    """
    from campeones_analysis.luminance.tde_glhmm import fit_global_pca, apply_global_pca
    from campeones_analysis.luminance.features import compute_epoch_covariance

    n_runs = len(run_tde)
    print(f"    Fold {test_idx+1}/{n_runs}: {run_names[test_idx]}")

    train_tde_segments = [seg for i in range(n_runs) if i != test_idx
                          for seg, _ in run_tde[i]]
    pca_model = fit_global_pca(train_tde_segments, tde_pca_components)

    mask = _get_cov_mask(tde_pca_components, cov_mode)

    X_train_parts, y_train_parts = [], []
    for i in range(n_runs):
        if i == test_idx:
            continue
        feats, labels = [], []
        for tde_seg, label in run_tde[i]:
            pca_proj = apply_global_pca(tde_seg, pca_model, standardise_pc=False)
            feats.append(compute_epoch_covariance(pca_proj)[mask])
            labels.append(label)
        X_train_parts.append(np.array(feats))
        y_train_parts.append(np.array(labels))

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    X_test_parts, y_test_parts = [], []
    for tde_seg, label in run_tde[test_idx]:
        pca_proj = apply_global_pca(tde_seg, pca_model, standardise_pc=False)
        X_test_parts.append(compute_epoch_covariance(pca_proj)[mask])
        y_test_parts.append(label)

    X_test = np.array(X_test_parts)
    y_test = np.array(y_test_parts)

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    best_c = float(np.atleast_1d(pipe[-1].C_)[0])
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    fold_acc = float(accuracy_score(y_test, y_pred))
    fold_info = {
        "fold": run_names[test_idx],
        "best_C": best_c,
        "accuracy": fold_acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    return y_test, y_pred, y_prob, fold_info


def run_loro_tde_cov(runs_data: list, cov_mode: str = "full") -> dict:
    """LORO CV for tde_cov (folds run in parallel).

    TDE applied to full wide epoch per trial; PCA fit on train TDE segments
    per fold; covariance computed per 250ms window after PCA projection.

    cov_mode: "full" (153), "diag" (17), or "offdiag" (136)
    """
    from campeones_analysis.luminance.tde_glhmm import apply_tde_only

    k = TDE_PCA_COMPONENTS
    feature_name = f"tde_cov_{cov_mode}" if cov_mode != "full" else "tde_cov"
    n_features = int(_get_cov_mask(k, cov_mode).sum())
    class_windows = get_class_windows()
    print(f"    N features: {n_features}  (cov_mode={cov_mode})")

    # Pre-compute raw TDE windows for all runs (sequential — I/O bound)
    # run_tde[run_idx] = list of (tde_segment, class_label)
    # tde_segment: (n_win_samples, n_ch*(2*half+1)) before PCA
    print("    Pre-computing TDE windows per run...")
    run_tde: list[list[tuple[np.ndarray, int]]] = []
    run_names = []

    for data_wide, sfreq, run_label in runs_data:
        n_trials = data_wide.shape[0]
        run_windows: list[tuple[np.ndarray, int]] = []

        for i in range(n_trials):
            epoch_data = data_wide[i].T  # (n_times_wide, n_ch)
            n_t = epoch_data.shape[0]
            indices = np.array([[0, n_t]])
            tde_data, _ = apply_tde_only(epoch_data, indices, TDE_WINDOW_HALF)
            # tde_data: (n_t - 2*half, n_ch*(2*half+1))

            tde_offset = TDE_WINDOW_HALF
            n_win = int(round(WIN_SIZE_S * sfreq))
            for t_start, t_end, label in class_windows:
                s0 = int(round((t_start - WIDE_TMIN) * sfreq)) - tde_offset
                s1 = s0 + n_win
                s0 = max(0, s0)
                s1 = min(tde_data.shape[0], s1)
                if s1 > s0:
                    run_windows.append((tde_data[s0:s1], label))

        run_tde.append(run_windows)
        run_names.append(run_label)
        print(f"      {run_label}: {n_trials} trials -> {len(run_windows)} TDE windows")

    n_runs = len(run_tde)

    fold_results = Parallel(n_jobs=-1)(
        delayed(_run_one_fold_tde)(test_idx, run_tde, run_names, TDE_PCA_COMPONENTS, cov_mode)
        for test_idx in range(n_runs)
    )

    all_y_true, all_y_pred, all_y_prob, fold_metrics = [], [], [], []
    for y_test, y_pred, y_prob, fold_info in fold_results:
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())
        fold_metrics.append(fold_info)
        print(f"      {fold_info['fold']}: acc={fold_info['accuracy']:.3f} "
              f"(best_C={fold_info['best_C']})")

    y_true = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)

    m = _metrics(y_true, y_pred_arr, y_prob_arr)
    return {"feature": feature_name, "n_features": n_features, **m,
            "folds": fold_metrics}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(all_results: list[dict], output_dir: Path, subject: str) -> None:
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    class_labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]

    for ax, r in zip(axes, all_results):
        cm = np.array(r["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(class_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = r["accuracy"]
        auc = r["auc_roc_macro"]
        ax.set_title(f"{r['feature']}\nAcc={acc:.3f}  AUC={auc:.3f}", fontsize=9)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"4-class decoding (sub-{subject}, LORO CV, 100ms windows)", fontsize=10)
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_4class_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix plot saved: {out.name}")


def plot_per_class_accuracy(all_results: list[dict], output_dir: Path, subject: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(N_CLASSES)
    width = 0.8 / len(all_results)
    class_labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]

    for k, r in enumerate(all_results):
        vals = [r["per_class_accuracy"][CLASS_NAMES[c]] for c in range(N_CLASSES)]
        ax.bar(x + k * width, vals, width, label=r["feature"])

    ax.axhline(1 / N_CLASSES, color="gray", ls="--", lw=1, label="chance (25%)")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class accuracy (sub-{subject}, LORO CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_4class_per_class_acc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-class accuracy plot saved: {out.name}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str, features: list[str] | None = None) -> None:
    if features is None:
        features = ["bandpower_welch", "tde_cov", "tde_cov_diag", "tde_cov_offdiag", "raw_pca"]
    print("=" * 60)
    print(f"34 -- 4-class decoding -- sub-{subject}")
    print(f"     Window: {int(WIN_SIZE_S*1000)}ms, step {int(WIN_STEP_S*1000)}ms")
    print(f"     Classes: {list(CLASS_NAMES.values())}")
    print(f"     Channels: {len(EEG_CHANNELS)}")
    print(f"     C grid (LogisticRegressionCV, {INNER_CV_FOLDS}-fold inner): {C_GRID}")
    print(f"     TDE_PCA_COMPONENTS={TDE_PCA_COMPONENTS} -> {TDE_PCA_COMPONENTS*(TDE_PCA_COMPONENTS+1)//2} cov features")
    print(f"     RAW_PCA_COMPONENTS={RAW_PCA_COMPONENTS}")
    print("=" * 60)

    runs_data = load_epochs_per_run(subject)
    if len(runs_data) < 2:
        print("Need at least 2 runs for LORO CV.")
        sys.exit(1)

    total_trials = sum(d.shape[0] for d, _, _ in runs_data)
    n_windows_per_trial = len(get_class_windows())
    print(f"\n  Total: {total_trials} trials -> ~{total_trials * n_windows_per_trial} windows "
          f"({n_windows_per_trial}/trial) across {len(runs_data)} runs\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for feat in features:
        print(f"\n  Feature set: {feat}")
        if feat.startswith("tde_cov"):
            cov_mode = feat[len("tde_cov_"):] if "_" in feat[7:] else "full"
            res = run_loro_tde_cov(runs_data, cov_mode=cov_mode)
        else:
            res = run_loro_standard(runs_data, feat)
        if res:
            all_results.append(res)
            print(f"    Acc={res['accuracy']:.3f}  F1={res['f1_macro']:.3f}  "
                  f"AUC={res['auc_roc_macro']:.3f}")
            print(f"    Per-class: {res['per_class_accuracy']}")

    # Save JSON
    json_path = output_dir / f"sub-{subject}_4class_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    if all_results:
        plot_results(all_results, output_dir, subject)
        plot_per_class_accuracy(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-class temporal decoding (Task 10.3)"
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument(
        "--features", nargs="+",
        choices=["bandpower_welch", "tde_cov", "tde_cov_diag", "tde_cov_offdiag", "raw_pca"],
        default=["bandpower_welch", "tde_cov", "tde_cov_diag", "tde_cov_offdiag", "raw_pca"],
        help="Feature sets to run (default: all five)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.subject, features=args.features)
