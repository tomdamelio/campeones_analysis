#!/usr/bin/env python
"""Systematic sweep of post-onset windows for micro-epoch decoding (Task 10.2).

Slides a 50ms window from 0ms to 500ms post-onset in 10ms steps,
training an independent LORO CV model at each position.

For each window position:
  - CHANGE: 74 micro-epochs (one per CHANGE_PHOTO onset)
  - NO_CHANGE: 74 micro-epochs sampled from 4 pre-onset windows [-250,-50]ms

Feature sets: bandpower_filtered and raw_signal (same as Task 10).

Usage
-----
    micromamba run -n campeones python scripts/validation/33_decoding_window_sweep.py --subject 27
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Add src to path for TDE imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_decoding_sweep"

SESSION = "vr"
TMIN = -2.5
TMAX = 2.0
BASELINE = (-2.5, -1.5)

# Sweep parameters
WINDOW_DURATION = 0.050  # 50 ms
STEP_SIZE = 0.010        # 10 ms
SWEEP_START = 0.000      # 0 ms post-onset
SWEEP_END = 0.500        # 500 ms post-onset (last window starts here - WINDOW_DURATION)

# Pre-onset windows for NO_CHANGE pool
PRE_WINDOWS = [(-0.25, -0.20), (-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05)]

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

# TDE parameters (from config_luminance.py / script 13)
TDE_WINDOW_HALF = 10  # ±10 timepoints → 21 total
TDE_PCA_COMPONENTS = 20


# ---------------------------------------------------------------------------
# Epoch loading
# ---------------------------------------------------------------------------

def load_long_epochs_per_run(subject: str) -> list[tuple[mne.Epochs, str]]:
    """Load preprocessed EEG and create long epochs around CHANGE_PHOTO onsets."""
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
        valid = (samples >= raw.first_samp) & (samples < raw.first_samp + raw.n_times)
        samples = samples[valid]
        if len(samples) == 0:
            continue

        mne_events = np.column_stack([
            samples, np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int),
        ])
        mne_events = mne_events[np.argsort(mne_events[:, 0])]

        try:
            epochs = mne.Epochs(
                raw, events=mne_events, event_id={"CHANGE_PHOTO": 1},
                tmin=TMIN, tmax=TMAX, picks=available_chs,
                baseline=BASELINE, preload=True, verbose=False,
            )
            print(f"  {label}: {len(epochs)} onsets")
            all_runs.append((epochs, label))
        except Exception as exc:
            print(f"  {label}: error — {exc}")

    return all_runs


# ---------------------------------------------------------------------------
# Pre-compute filtered data (filter once per band per run)
# ---------------------------------------------------------------------------

def _time_to_sample(t: float, sfreq: float, tmin: float) -> int:
    return int(np.round((t - tmin) * sfreq))


def precompute_filtered_data(
    long_epochs_runs: list[tuple[mne.Epochs, str]],
) -> list[dict]:
    """Filter each run once per band. Return raw + filtered data per run.

    Returns list of dicts:
        {"raw_data": (n_ep, n_ch, n_times),
         "filtered": {band_name: (n_ep, n_ch, n_times)},
         "sfreq": float, "tmin": float, "n_ep": int, "label": str}
    """
    band_list = list(SPECTRAL_BANDS.items())
    result = []

    for epochs, label in long_epochs_runs:
        sfreq = epochs.info["sfreq"]
        tmin_ep = epochs.tmin
        raw_data = epochs.get_data()
        n_ep = raw_data.shape[0]

        filtered = {}
        for band_name, (flo, fhi) in band_list:
            filt_epochs = epochs.copy().filter(l_freq=flo, h_freq=fhi, verbose=False)
            filtered[band_name] = filt_epochs.get_data()

        result.append({
            "raw_data": raw_data,
            "filtered": filtered,
            "sfreq": sfreq,
            "tmin": tmin_ep,
            "n_ep": n_ep,
            "label": label,
        })

    return result


# ---------------------------------------------------------------------------
# Feature extraction from pre-computed data
# ---------------------------------------------------------------------------

def extract_window_bandpower(
    run_data: dict, t_start: float, win_samples: int,
) -> np.ndarray:
    """Extract bandpower features for one window from pre-filtered data.

    Returns: (n_ep, n_ch * n_bands)
    """
    sfreq = run_data["sfreq"]
    tmin = run_data["tmin"]
    n_ep = run_data["n_ep"]
    band_names = list(SPECTRAL_BANDS.keys())
    n_bands = len(band_names)
    n_ch = run_data["raw_data"].shape[1]

    s0 = _time_to_sample(t_start, sfreq, tmin)
    s1 = s0 + win_samples

    features = np.empty((n_ep, n_ch * n_bands))
    for b_idx, bname in enumerate(band_names):
        filt_data = run_data["filtered"][bname]
        for i in range(n_ep):
            features[i, b_idx * n_ch:(b_idx + 1) * n_ch] = np.var(
                filt_data[i, :, s0:s1], axis=1)
    return features


def extract_window_raw(
    run_data: dict, t_start: float, win_samples: int,
) -> np.ndarray:
    """Extract raw features for one window.

    Returns: (n_ep, n_ch * win_samples)
    """
    sfreq = run_data["sfreq"]
    tmin = run_data["tmin"]
    s0 = _time_to_sample(t_start, sfreq, tmin)
    s1 = s0 + win_samples
    segments = run_data["raw_data"][:, :, s0:s1]
    return segments.reshape(segments.shape[0], -1)


# ---------------------------------------------------------------------------
# TDE pre-computation (TDE on long epoch, PCA fit per LORO fold)
# ---------------------------------------------------------------------------

def precompute_tde_data(
    precomputed: list[dict],
) -> list[list[np.ndarray]]:
    """Apply TDE to each epoch of each run. Returns TDE segments per run.

    Returns: list of lists, run_tde[run_idx][epoch_idx] = (n_valid_times, n_tde_feat)
    """
    from campeones_analysis.luminance.tde_glhmm import apply_tde_only

    all_run_tde: list[list[np.ndarray]] = []
    for rd in precomputed:
        raw_data = rd["raw_data"]  # (n_ep, n_ch, n_times)
        n_ep = raw_data.shape[0]
        epoch_tde = []
        for i in range(n_ep):
            ep = raw_data[i].T  # (n_times, n_ch)
            n_t = ep.shape[0]
            indices = np.array([[0, n_t]])
            tde_data, _ = apply_tde_only(ep, indices, TDE_WINDOW_HALF)
            epoch_tde.append(tde_data)
        all_run_tde.append(epoch_tde)
    return all_run_tde


def extract_window_tde_pca(
    run_tde: list[np.ndarray],
    pca_model: PCA,
    t_start: float,
    win_samples: int,
    sfreq: float,
    tmin: float,
) -> np.ndarray:
    """Project TDE data with PCA, then extract variance per component in window.

    TDE removes TDE_WINDOW_HALF samples from each end, so sample indices
    need to be offset accordingly.

    Returns: (n_ep, TDE_PCA_COMPONENTS)
    """
    from campeones_analysis.luminance.tde_glhmm import apply_global_pca

    s0_epoch = _time_to_sample(t_start, sfreq, tmin)
    # TDE trims TDE_WINDOW_HALF from start → offset
    s0_tde = s0_epoch - TDE_WINDOW_HALF
    s1_tde = s0_tde + win_samples

    n_ep = len(run_tde)
    features = np.empty((n_ep, TDE_PCA_COMPONENTS))
    for i in range(n_ep):
        pca_data = apply_global_pca(run_tde[i], pca_model)
        n_valid = pca_data.shape[0]
        # Clamp indices
        s0_c = max(0, s0_tde)
        s1_c = min(n_valid, s1_tde)
        if s1_c <= s0_c:
            features[i] = 0.0
        else:
            features[i] = np.var(pca_data[s0_c:s1_c], axis=0)
    return features


# ---------------------------------------------------------------------------
# Pre-compute NO_CHANGE pool (fixed across sweep)
# ---------------------------------------------------------------------------

def precompute_nochange_pool(
    precomputed: list[dict], win_samples: int, feature_type: str,
) -> list[np.ndarray]:
    """Extract NO_CHANGE features from 4 pre-onset windows per run.

    Returns list of (n_ep * 4, n_feat) arrays, one per run.
    """
    pool_per_run = []
    for rd in precomputed:
        parts = []
        for pre_start, _pre_end in PRE_WINDOWS:
            if feature_type == "bandpower_filtered":
                f = extract_window_bandpower(rd, pre_start, win_samples)
            else:
                f = extract_window_raw(rd, pre_start, win_samples)
            parts.append(f)
        pool_per_run.append(np.vstack(parts))
    return pool_per_run


# ---------------------------------------------------------------------------
# LORO CV (lightweight — no per-fold printing)
# ---------------------------------------------------------------------------

def _build_pipeline(c_val: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_val, max_iter=1000, random_state=RANDOM_SEED),
    )


def _select_best_c(X_train: np.ndarray, y_train: np.ndarray) -> float:
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
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_c = c_val
    return best_c


def run_loro_quick(
    runs_data: list[tuple[np.ndarray, np.ndarray, str]],
) -> dict:
    """LORO CV returning accuracy, AUC, and train/test sizes."""
    n_runs = len(runs_data)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    n_train_total = 0
    n_test_total = 0

    for test_idx in range(n_runs):
        X_train = np.vstack([runs_data[i][0] for i in range(n_runs) if i != test_idx])
        y_train = np.concatenate([runs_data[i][1] for i in range(n_runs) if i != test_idx])
        X_test = runs_data[test_idx][0]
        y_test = runs_data[test_idx][1]

        n_train_total += len(y_train)
        n_test_total += len(y_test)

        best_c = _select_best_c(X_train, y_train)
        pipe = _build_pipeline(best_c)
        pipe.fit(X_train, y_train)

        all_y_true.extend(y_test)
        all_y_pred.extend(pipe.predict(X_test))
        all_y_prob.extend(pipe.predict_proba(X_test)[:, 1])

    y_true = np.array(all_y_true)
    return {
        "accuracy": float(accuracy_score(y_true, np.array(all_y_pred))),
        "auc_roc": float(roc_auc_score(y_true, np.array(all_y_prob))),
        "n_total": len(y_true),
        "n_train_avg": int(n_train_total / n_runs),
        "n_test_avg": int(n_test_total / n_runs),
    }


# ---------------------------------------------------------------------------
# Sweep one feature type
# ---------------------------------------------------------------------------

def run_sweep(
    precomputed: list[dict],
    nochange_pool: list[np.ndarray],
    feature_type: str,
    win_samples: int,
    sweep_positions: list[float],
) -> list[dict]:
    """Run LORO CV at each sweep position for one feature type."""
    rng = np.random.RandomState(RANDOM_SEED)
    results = []

    for pos_idx, t_start in enumerate(sweep_positions):
        t_end = t_start + WINDOW_DURATION
        label = f"{int(t_start*1000)}-{int(t_end*1000)}ms"

        # Build dataset per run
        runs_data = []
        for r_idx, rd in enumerate(precomputed):
            n_ep = rd["n_ep"]

            if feature_type == "bandpower_filtered":
                change_feats = extract_window_bandpower(rd, t_start, win_samples)
            else:
                change_feats = extract_window_raw(rd, t_start, win_samples)

            # Sample n_ep from pre-onset pool
            pool = nochange_pool[r_idx]
            indices = rng.choice(len(pool), size=n_ep, replace=False)
            nochange_feats = pool[indices]

            X = np.vstack([change_feats, nochange_feats])
            y = np.concatenate([np.ones(n_ep), np.zeros(n_ep)]).astype(int)
            runs_data.append((X, y, rd["label"]))

        res = run_loro_quick(runs_data)
        res["window"] = label
        res["t_start_ms"] = int(t_start * 1000)
        res["t_end_ms"] = int(t_end * 1000)
        res["feature"] = feature_type
        res["n_features"] = runs_data[0][0].shape[1]
        results.append(res)

        print(f"    {label}: acc={res['accuracy']:.3f}  AUC={res['auc_roc']:.3f}")

    return results


# ---------------------------------------------------------------------------
# TDE sweep (PCA fit per LORO fold)
# ---------------------------------------------------------------------------

def run_sweep_tde(
    precomputed: list[dict],
    all_run_tde: list[list[np.ndarray]],
    nochange_pool_tde: list[np.ndarray],
    win_samples: int,
    sweep_positions: list[float],
) -> list[dict]:
    """Run LORO CV sweep for TDE features with PCA fit per fold."""
    from campeones_analysis.luminance.tde_glhmm import fit_global_pca

    rng = np.random.RandomState(RANDOM_SEED)
    n_runs = len(precomputed)
    results = []

    for t_start in sweep_positions:
        t_end = t_start + WINDOW_DURATION
        label = f"{int(t_start*1000)}-{int(t_end*1000)}ms"

        all_y_true, all_y_pred, all_y_prob = [], [], []
        n_train_total = 0
        n_test_total = 0

        for test_idx in range(n_runs):
            # Fit PCA on train TDE segments only
            train_tde_all = []
            for i in range(n_runs):
                if i != test_idx:
                    train_tde_all.extend(all_run_tde[i])

            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                pca_model = fit_global_pca(train_tde_all, TDE_PCA_COMPONENTS)

            # Extract CHANGE features for this window
            train_parts_X, train_parts_y = [], []
            for i in range(n_runs):
                if i == test_idx:
                    continue
                rd = precomputed[i]
                change_f = extract_window_tde_pca(
                    all_run_tde[i], pca_model, t_start, win_samples,
                    rd["sfreq"], rd["tmin"])

                # NO_CHANGE: extract from pre-windows with this PCA
                pre_parts = []
                for pre_start, _pre_end in PRE_WINDOWS:
                    pf = extract_window_tde_pca(
                        all_run_tde[i], pca_model, pre_start, win_samples,
                        rd["sfreq"], rd["tmin"])
                    pre_parts.append(pf)
                pre_pool = np.vstack(pre_parts)
                n_ep = rd["n_ep"]
                indices = rng.choice(len(pre_pool), size=n_ep, replace=False)
                nochange_f = pre_pool[indices]

                X_run = np.vstack([change_f, nochange_f])
                y_run = np.concatenate([np.ones(n_ep), np.zeros(n_ep)]).astype(int)
                train_parts_X.append(X_run)
                train_parts_y.append(y_run)

            X_train = np.vstack(train_parts_X)
            y_train = np.concatenate(train_parts_y)

            # Test set
            rd_test = precomputed[test_idx]
            n_ep_test = rd_test["n_ep"]
            change_f_test = extract_window_tde_pca(
                all_run_tde[test_idx], pca_model, t_start, win_samples,
                rd_test["sfreq"], rd_test["tmin"])
            pre_parts_test = []
            for pre_start, _pre_end in PRE_WINDOWS:
                pf = extract_window_tde_pca(
                    all_run_tde[test_idx], pca_model, pre_start, win_samples,
                    rd_test["sfreq"], rd_test["tmin"])
                pre_parts_test.append(pf)
            pre_pool_test = np.vstack(pre_parts_test)
            indices_test = rng.choice(len(pre_pool_test), size=n_ep_test, replace=False)
            nochange_f_test = pre_pool_test[indices_test]

            X_test = np.vstack([change_f_test, nochange_f_test])
            y_test = np.concatenate([np.ones(n_ep_test), np.zeros(n_ep_test)]).astype(int)

            n_train_total += len(y_train)
            n_test_total += len(y_test)

            best_c = _select_best_c(X_train, y_train)
            pipe = _build_pipeline(best_c)
            pipe.fit(X_train, y_train)

            all_y_true.extend(y_test)
            all_y_pred.extend(pipe.predict(X_test))
            all_y_prob.extend(pipe.predict_proba(X_test)[:, 1])

        y_true = np.array(all_y_true)
        res = {
            "accuracy": float(accuracy_score(y_true, np.array(all_y_pred))),
            "auc_roc": float(roc_auc_score(y_true, np.array(all_y_prob))),
            "n_total": len(y_true),
            "n_train_avg": int(n_train_total / n_runs),
            "n_test_avg": int(n_test_total / n_runs),
            "window": label,
            "t_start_ms": int(t_start * 1000),
            "t_end_ms": int(t_end * 1000),
            "feature": "tde_pca_var",
            "n_features": TDE_PCA_COMPONENTS,
        }
        results.append(res)
        print(f"    {label}: acc={res['accuracy']:.3f}  AUC={res['auc_roc']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(all_results: list[dict], output_dir: Path, subject: str) -> None:
    """Line plot of accuracy and AUC vs post-onset window position."""
    feature_names = sorted(set(r["feature"] for r in all_results))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for feat in feature_names:
        subset = [r for r in all_results if r["feature"] == feat]
        subset.sort(key=lambda r: r["t_start_ms"])
        x = [r["t_start_ms"] + 25 for r in subset]  # center of window
        acc = [r["accuracy"] for r in subset]
        auc = [r["auc_roc"] for r in subset]

        axes[0].plot(x, acc, "o-", label=feat, markersize=4, linewidth=1.5)
        axes[1].plot(x, auc, "o-", label=feat, markersize=4, linewidth=1.5)

    for ax, metric_name in [(axes[0], "Accuracy"), (axes[1], "AUC-ROC")]:
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.35, 0.85)

    axes[1].set_xlabel("Post-onset window center (ms)")
    axes[0].set_title(
        f"Decoding sweep: 50ms windows, 10ms steps (sub-{subject}, LORO CV)")

    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{subject}_sweep_results.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Sweep plot saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    print("=" * 60)
    print(f"33 — Window sweep decoding — sub-{subject}")
    print(f"     Window: {WINDOW_DURATION*1000:.0f}ms, step: {STEP_SIZE*1000:.0f}ms")
    print(f"     Sweep: {SWEEP_START*1000:.0f}ms to {SWEEP_END*1000:.0f}ms")
    print("=" * 60)

    long_epochs = load_long_epochs_per_run(subject)
    if len(long_epochs) < 2:
        print("Need at least 2 runs.")
        sys.exit(1)

    total_onsets = sum(len(ep) for ep, _ in long_epochs)
    print(f"\n  Total onsets: {total_onsets}")

    # Pre-compute filtered data (expensive, done once)
    print("\n  Pre-computing filtered data (5 bands × 7 runs)...")
    precomputed = precompute_filtered_data(long_epochs)

    sfreq = precomputed[0]["sfreq"]
    win_samples = int(np.round(WINDOW_DURATION * sfreq))
    print(f"  Window: {win_samples} samples at {sfreq} Hz")

    # Sweep positions
    sweep_positions = []
    t = SWEEP_START
    while t + WINDOW_DURATION <= SWEEP_END + 1e-9:
        sweep_positions.append(round(t, 4))
        t += STEP_SIZE
    print(f"  Sweep positions: {len(sweep_positions)} windows\n")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for feat_type in ["bandpower_filtered", "raw_signal"]:
        print(f"  === {feat_type} ===")
        nc_pool = precompute_nochange_pool(precomputed, win_samples, feat_type)
        results = run_sweep(precomputed, nc_pool, feat_type, win_samples,
                            sweep_positions)
        all_results.extend(results)

    # --- TDE (PCA fit per LORO fold) ---
    print(f"\n  === tde_pca_var ===")
    print("  Pre-computing TDE segments...")
    all_run_tde = precompute_tde_data(precomputed)
    print("  Running sweep with per-fold PCA...")
    tde_results = run_sweep_tde(precomputed, all_run_tde, None,
                                win_samples, sweep_positions)
    all_results.extend(tde_results)

    # Save JSON
    json_path = output_dir / f"sub-{subject}_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    plot_sweep(all_results, output_dir, subject)

    # Print peak windows
    for feat in ["bandpower_filtered", "raw_signal", "tde_pca_var"]:
        subset = [r for r in all_results if r["feature"] == feat]
        if not subset:
            continue
        best_acc = max(subset, key=lambda r: r["accuracy"])
        best_auc = max(subset, key=lambda r: r["auc_roc"])
        print(f"\n  {feat}:")
        print(f"    Peak accuracy: {best_acc['accuracy']:.3f} at {best_acc['window']}")
        print(f"    Peak AUC:      {best_auc['auc_roc']:.3f} at {best_auc['window']}")
        print(f"    n_train_avg: {best_acc['n_train_avg']}, n_test_avg: {best_acc['n_test_avg']}")

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Systematic window sweep for micro-epoch decoding (Task 10.2)",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
