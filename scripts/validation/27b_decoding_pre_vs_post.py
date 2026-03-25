#!/usr/bin/env python
"""Decode post-stimulus vs pre-stimulus epochs around CHANGE_PHOTO onsets.

Implements Task 9.2 from the research diary.

For each CHANGE_PHOTO onset:
  - CHANGE (class 1): [+0.05, +0.55]s post-onset (500ms)
  - NO_CHANGE (class 0): [-0.55, -0.05]s pre-onset (500ms)

Both classes come from the SAME onset, so the contrast is purely
temporal: post-stimulus vs pre-stimulus activity.

Feature sets: bandpower_welch, tde_cov, raw_pca.
Classifier: LogisticRegressionCV (L2, C cross-validated per feature set),
Leave-One-Run-Out CV outer loop.

Usage
-----
    micromamba run -n campeones python scripts/validation/27b_decoding_pre_vs_post.py --subject 27
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_pre_vs_post"

SESSION = "vr"

# Epoch parameters: wide epoch to apply baseline, then crop
WIDE_TMIN = -1.5
WIDE_TMAX = 1.5
BASELINE = (-1.5, -1.0)

# Crop windows (500ms each)
POST_CROP_START = 0.05   # post-stimulus start
POST_CROP_END = 0.55     # post-stimulus end
PRE_CROP_START = -0.55   # pre-stimulus start
PRE_CROP_END = -0.05     # pre-stimulus end

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

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
INNER_CV_FOLDS = 5
RANDOM_SEED = 42
TDE_WINDOW_HALF = 10
TDE_PCA_COMPONENTS = 17   # → 17*18//2 = 153 cov features (≈ bandpower 160)
RAW_PCA_COMPONENTS = 160  # ≈ bandpower 160


# ---------------------------------------------------------------------------
# Epoch loading: create paired post/pre epochs from CHANGE_PHOTO onsets
# ---------------------------------------------------------------------------

def load_pre_post_epochs_per_run(
    subject: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load paired post-stimulus and pre-stimulus epochs per run.

    For each CHANGE_PHOTO onset, creates:
      - post epoch: [+0.05, +0.55]s  (class 1)
      - pre epoch:  [-0.55, -0.05]s  (class 0)

    Returns list of (X_post, X_pre, post_data_wide, pre_data_wide, run_label)
    where X_post/X_pre are (n_onsets, n_ch, n_times_500ms) cropped arrays,
    and *_wide are (n_onsets, n_ch, n_times_wide) for TDE processing.
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
        # Only use CHANGE_PHOTO onsets
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

        # Create MNE events (all as event_id=1 for CHANGE_PHOTO)
        mne_events = np.column_stack([
            samples,
            np.zeros(len(samples), dtype=int),
            np.ones(len(samples), dtype=int),
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        try:
            # Wide epoch with baseline
            epochs_wide = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=WIDE_TMIN, tmax=WIDE_TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )

            # Crop to post-stimulus window
            epochs_post = epochs_wide.copy().crop(
                tmin=POST_CROP_START, tmax=POST_CROP_END)
            X_post = epochs_post.get_data()  # (n, ch, times)

            # Crop to pre-stimulus window
            epochs_pre = epochs_wide.copy().crop(
                tmin=PRE_CROP_START, tmax=PRE_CROP_END)
            X_pre = epochs_pre.get_data()  # (n, ch, times)

            # Wide data for TDE (full epoch before cropping)
            data_wide = epochs_wide.get_data()  # (n, ch, times_wide)

            n = X_post.shape[0]
            print(f"  {label}: {n} CHANGE_PHOTO onsets -> "
                  f"{n} post + {n} pre epochs (500ms each)")

            all_runs.append((X_post, X_pre, data_wide, epochs_wide.info["sfreq"], label))
        except Exception as exc:
            print(f"  {label}: epoch error -- {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_bandpower_welch(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Welch bandpower per channel per epoch.

    Args:
        data: (n_epochs, n_channels, n_times)
        sfreq: sampling frequency

    Returns: (n_epochs, n_channels * n_bands)
    """
    n_ep, n_ch, n_times = data.shape
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


def extract_raw_features(data: np.ndarray) -> np.ndarray:
    """Raw EEG vectorized. Returns: (n_epochs, n_channels * n_times)"""
    return data.reshape(data.shape[0], -1)


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def _build_pipeline(use_pca: bool, n_samples: int, n_features: int):
    steps = [StandardScaler()]
    if use_pca:
        n_comp = min(RAW_PCA_COMPONENTS, n_samples, n_features)
        steps.append(PCA(n_components=n_comp))
    steps.append(LogisticRegressionCV(
        Cs=C_GRID, cv=INNER_CV_FOLDS, l1_ratios=(0,), solver="saga",
        max_iter=2000, random_state=RANDOM_SEED, scoring="accuracy",
        use_legacy_attributes=False,
    ))
    return make_pipeline(*steps)


# ---------------------------------------------------------------------------
# LORO CV
# ---------------------------------------------------------------------------

def run_loro_standard(
    runs_data: list[tuple],
    feature_name: str,
) -> dict:
    """LORO CV for bandpower_welch and raw_pca."""
    use_pca = (feature_name == "raw_pca")

    # Pre-extract features per run
    run_features = []
    run_labels = []
    run_names = []
    for X_post, X_pre, _data_wide, sfreq, run_label in runs_data:
        if feature_name == "bandpower_welch":
            feat_post = extract_bandpower_welch(X_post, sfreq)
            feat_pre = extract_bandpower_welch(X_pre, sfreq)
        else:  # raw_pca
            feat_post = extract_raw_features(X_post)
            feat_pre = extract_raw_features(X_pre)

        X = np.vstack([feat_post, feat_pre])
        y = np.concatenate([np.ones(len(feat_post)), np.zeros(len(feat_pre))])
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

        pipe = _build_pipeline(False, X_train.shape[0], X_train.shape[1])
        pipe.fit(X_train, y_train)
        best_c = float(np.atleast_1d(pipe[-1].C_)[0])
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_metrics.append({
            "fold": run_names[test_idx], "best_C": best_c,
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
        "n_post": int(y_true.sum()),
        "n_pre": int((1 - y_true).sum()),
        "folds": fold_metrics,
    }


def run_loro_tde_cov(
    runs_data: list[tuple],
) -> dict:
    """LORO CV for tde_cov with PCA fit on train only per fold."""
    from campeones_analysis.luminance.tde_glhmm import (
        apply_tde_only, fit_global_pca, apply_global_pca,
    )
    from campeones_analysis.luminance.features import compute_epoch_covariance

    feature_name = "tde_cov"
    n_runs = len(runs_data)
    if n_runs < 2:
        print(f"    {feature_name}: need >= 2 runs, got {n_runs}")
        return {}

    # Pre-compute TDE segments for all runs on wide epochs
    print("    Pre-computing TDE segments...")
    run_tde_post: list[list[np.ndarray]] = []
    run_tde_pre: list[list[np.ndarray]] = []
    run_n_onsets: list[int] = []

    for X_post, X_pre, data_wide, sfreq, label in runs_data:
        n_onsets = X_post.shape[0]
        run_n_onsets.append(n_onsets)

        # Compute TDE on wide epoch, then we'll extract post/pre windows
        # from the TDE-transformed data
        post_segments = []
        pre_segments = []

        for i in range(n_onsets):
            epoch_data = data_wide[i].T  # (n_times_wide, n_channels)
            n_t = epoch_data.shape[0]
            indices = np.array([[0, n_t]])
            tde_data, _ = apply_tde_only(epoch_data, indices, TDE_WINDOW_HALF)
            # tde_data: (n_t - 2*TDE_WINDOW_HALF, n_ch * (2*TDE_WINDOW_HALF+1))

            # Map crop times to TDE indices
            # Wide epoch: WIDE_TMIN to WIDE_TMAX
            # TDE removes TDE_WINDOW_HALF samples from each end
            tde_offset = TDE_WINDOW_HALF  # samples lost at start

            # Post window: POST_CROP_START to POST_CROP_END
            post_start_idx = int(round((POST_CROP_START - WIDE_TMIN) * sfreq)) - tde_offset
            post_end_idx = int(round((POST_CROP_END - WIDE_TMIN) * sfreq)) - tde_offset
            post_start_idx = max(0, post_start_idx)
            post_end_idx = min(tde_data.shape[0], post_end_idx)
            post_segments.append(tde_data[post_start_idx:post_end_idx])

            # Pre window: PRE_CROP_START to PRE_CROP_END
            pre_start_idx = int(round((PRE_CROP_START - WIDE_TMIN) * sfreq)) - tde_offset
            pre_end_idx = int(round((PRE_CROP_END - WIDE_TMIN) * sfreq)) - tde_offset
            pre_start_idx = max(0, pre_start_idx)
            pre_end_idx = min(tde_data.shape[0], pre_end_idx)
            pre_segments.append(tde_data[pre_start_idx:pre_end_idx])

        run_tde_post.append(post_segments)
        run_tde_pre.append(pre_segments)

    n_features = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
    print(f"    N features: {n_features}")

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []

    for test_idx in range(n_runs):
        print(f"    Fold {test_idx+1}/{n_runs}: {runs_data[test_idx][4]}")

        # Collect all train TDE segments (post + pre) for PCA fitting
        train_segments = []
        for i in range(n_runs):
            if i != test_idx:
                train_segments.extend(run_tde_post[i])
                train_segments.extend(run_tde_pre[i])

        pca_model = fit_global_pca(train_segments, TDE_PCA_COMPONENTS)

        # Extract cov features for train
        X_train_parts = []
        y_train_parts = []
        for i in range(n_runs):
            if i != test_idx:
                n = run_n_onsets[i]
                # Post epochs (class 1)
                feats_post = np.empty((n, n_features))
                for j, seg in enumerate(run_tde_post[i]):
                    pca_data = apply_global_pca(seg, pca_model)
                    feats_post[j] = compute_epoch_covariance(pca_data)
                # Pre epochs (class 0)
                feats_pre = np.empty((n, n_features))
                for j, seg in enumerate(run_tde_pre[i]):
                    pca_data = apply_global_pca(seg, pca_model)
                    feats_pre[j] = compute_epoch_covariance(pca_data)

                X_train_parts.append(np.vstack([feats_post, feats_pre]))
                y_train_parts.append(np.concatenate([np.ones(n), np.zeros(n)]))

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Extract cov features for test
        n_test = run_n_onsets[test_idx]
        feats_post = np.empty((n_test, n_features))
        for j, seg in enumerate(run_tde_post[test_idx]):
            pca_data = apply_global_pca(seg, pca_model)
            feats_post[j] = compute_epoch_covariance(pca_data)
        feats_pre = np.empty((n_test, n_features))
        for j, seg in enumerate(run_tde_pre[test_idx]):
            pca_data = apply_global_pca(seg, pca_model)
            feats_pre[j] = compute_epoch_covariance(pca_data)

        X_test = np.vstack([feats_post, feats_pre])
        y_test = np.concatenate([np.ones(n_test), np.zeros(n_test)])

        pipe = _build_pipeline(False, X_train.shape[0], X_train.shape[1])
        pipe.fit(X_train, y_train)
        best_c = float(np.atleast_1d(pipe[-1].C_)[0])
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_metrics.append({
            "fold": runs_data[test_idx][4], "best_C": best_c,
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
        "n_post": int(y_true.sum()),
        "n_pre": int((1 - y_true).sum()),
        "folds": fold_metrics,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_summary(all_results: list[dict], output_dir: Path, subject: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
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
    ax.set_title(f"Pre vs Post decoding (sub-{subject}, LORO CV, 500ms)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{subject}_decoding_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Summary plot saved.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    print("=" * 60)
    print(f"27b -- Pre vs Post decoding -- sub-{subject}")
    print(f"     Post window: [{POST_CROP_START}, {POST_CROP_END}]s")
    print(f"     Pre window:  [{PRE_CROP_START}, {PRE_CROP_END}]s")
    print(f"     Baseline: {BASELINE}")
    print(f"     Channels: {len(EEG_CHANNELS)}")
    print(f"     C grid (LogisticRegressionCV, {INNER_CV_FOLDS}-fold inner): {C_GRID}")
    print("=" * 60)

    runs_data = load_pre_post_epochs_per_run(subject)
    if len(runs_data) < 2:
        print("Need at least 2 runs for LORO CV.")
        sys.exit(1)

    total_onsets = sum(X_post.shape[0] for X_post, *_ in runs_data)
    print(f"\n  Total: {total_onsets} onsets -> {total_onsets} post + "
          f"{total_onsets} pre = {total_onsets * 2} epochs across "
          f"{len(runs_data)} runs\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Bandpower Welch
    print(f"  Feature set: bandpower_welch")
    res = run_loro_standard(runs_data, "bandpower_welch")
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(best_C={f['best_C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # 2. TDE + Cov
    print(f"\n  Feature set: tde_cov")
    res = run_loro_tde_cov(runs_data)
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(best_C={f['best_C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # 3. Raw + PCA
    print(f"\n  Feature set: raw_pca")
    res = run_loro_standard(runs_data, "raw_pca")
    if res:
        all_results.append(res)
        print(f"    Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['auc_roc']:.3f}")
        for f in res["folds"]:
            print(f"      {f['fold']}: acc={f['accuracy']:.3f} "
                  f"(best_C={f['best_C']}, n_train={f['n_train']}, n_test={f['n_test']})")

    # Save JSON
    json_path = output_dir / f"sub-{subject}_pre_vs_post_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    if all_results:
        plot_summary(all_results, output_dir, subject)

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode post-stimulus vs pre-stimulus (Task 9.2)",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.subject)
