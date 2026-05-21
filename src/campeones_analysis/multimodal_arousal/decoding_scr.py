"""Decoding: predict SCR-onset (real) vs silent-EDA (control) from EEG epochs.

Uses the ONSET-ALIGNED epochs from erp_scr.py (window -5..+3 s; NOT the peak-aligned
condition added in analyze_scr_peak.py).

Two complementary decoders, both with stratified 5-fold cross-validation:
  1. Temporal MVPA (mne.decoding.SlidingEstimator with logistic regression):
     trains one classifier per timepoint on the 32-channel feature vector. Returns
     accuracy(t) -- WHEN in the epoch the EEG distinguishes the two conditions.
  2. Spectral decoder: PSD per band per channel (5 bands x 32 ch = 160 features),
     logistic regression with L2. Returns a single accuracy + confusion matrix per
     subject; coefficients diagnose WHICH band-channel features drive the decoding.

Outputs (research_diary/context/05_02/figures/decoding/):
  Y3_decoding_temporal_<sub>.png         -- accuracy(t) per subject with 95% chance band
  Y3_decoding_temporal_grandaverage.png  -- per-subject curves + mean across subjects
  Y3_decoding_spectral_<sub>.png         -- confusion matrix + top-N LR weights
  Y3_decoding_spectral_summary.png       -- bar chart of per-subject CV accuracy

CSVs:
  decoding_temporal.csv  -- subject, timepoint, mean_acc, std_acc
  decoding_spectral.csv  -- subject, mean_cv_acc, std_cv_acc, n_real, n_silent, top_feats

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_scr
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse epoch construction from tfr_psd_scr.py (which itself uses erp_scr.py helpers)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    NPZ_DIR,
    OUT,
    SUBJECTS,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- output folder (separate from earlier analyses) ---
FIG_DIR = OUT / "figures" / "decoding"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- decoding parameters ---
N_FOLDS = 5
N_JOBS = 1  # parallel jobs for SlidingEstimator
RNG_SEED = 42
PSD_FMIN, PSD_FMAX = 1.0, 40.0
PSD_NFFT = 512
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
TOP_N_FEATS = 15


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def build_dataset(real_ep: mne.Epochs, silent_ep: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Concatenate real + silent epochs into (X, y, times, ch_names).

    X shape: (n_total_epochs, n_channels, n_times)
    y: 1 for real, 0 for silent
    """
    Xr = real_ep.get_data()
    Xs = silent_ep.get_data()
    X = np.concatenate([Xr, Xs], axis=0)
    y = np.concatenate([np.ones(len(Xr), dtype=int), np.zeros(len(Xs), dtype=int)])
    times = real_ep.times.copy()
    return X, y, times, list(real_ep.ch_names)


# -----------------------------------------------------------------------------
# Temporal MVPA decoder
# -----------------------------------------------------------------------------
def temporal_decoding(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SlidingEstimator with LR + standard scaling. Returns scores (n_folds, n_times)."""
    base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    sliding = SlidingEstimator(base, scoring="accuracy", n_jobs=N_JOBS, verbose="ERROR")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_multiscore(sliding, X, y, cv=cv, n_jobs=N_JOBS, verbose="ERROR")
    return scores  # (n_folds, n_times)


def plot_temporal_decoding_subject(sub: str, scores: np.ndarray, times: np.ndarray,
                                     n_real: int, n_silent: int, out_png: Path) -> None:
    """Per-subject accuracy(t) with chance level."""
    n_per_fold = (n_real + n_silent) // N_FOLDS
    # 95% binomial CI around chance (0.5): +/- 1.96 * sqrt(0.25 / n_per_fold)
    chance_ci = 1.96 * np.sqrt(0.25 / max(1, n_per_fold))

    mean_t = scores.mean(axis=0)
    sem_t = scores.std(axis=0, ddof=1) / np.sqrt(N_FOLDS)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--", label="chance (50%)")
    ax.axhspan(0.5 - chance_ci, 0.5 + chance_ci, color="0.5", alpha=0.18, lw=0,
               label=f"95% binomial CI of chance (n/fold={n_per_fold})")
    ax.fill_between(times * 1000, mean_t - sem_t, mean_t + sem_t, color="C3", alpha=0.30, lw=0)
    ax.plot(times * 1000, mean_t, color="C3", lw=1.6, label=f"CV accuracy (mean of {N_FOLDS} folds)")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time from SCR onset (ms)")
    ax.set_ylabel("decoding accuracy")
    ax.set_title(f"{sub}  --  temporal MVPA  (real SCR n={n_real} vs silent n={n_silent})")
    ax.set_ylim(0.30, 1.00)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_temporal_decoding_grandaverage(per_sub_scores: dict, times: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--", label="chance (50%)")

    curves = []
    for sub, scores in per_sub_scores.items():
        mean_t = scores.mean(axis=0)
        curves.append(mean_t)
        ax.plot(times * 1000, mean_t, lw=0.9, alpha=0.55, label=f"{sub}")
    arr = np.array(curves)
    ga = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(ga)
    ax.fill_between(times * 1000, ga - sd, ga + sd, color="C3", alpha=0.25, lw=0)
    ax.plot(times * 1000, ga, color="C3", lw=2.0, label=f"grand average (mean ± SD, N={len(curves)})")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time from SCR onset (ms)")
    ax.set_ylabel("decoding accuracy")
    ax.set_title(f"Temporal MVPA grand average  (N={len(curves)} subjects)")
    ax.set_ylim(0.30, 1.00)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Spectral decoder
# -----------------------------------------------------------------------------
def epochs_to_band_features(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """Compute PSD per epoch -> average within bands -> log-transform.

    Returns (X, feat_names) where X.shape = (n_epochs, n_channels * n_bands).
    feat_names is a list of f"{ch}|{band}" for interpretability.
    """
    spectrum = epochs.compute_psd(method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX,
                                    n_fft=PSD_NFFT, verbose="ERROR")
    data = spectrum.get_data()  # (n_epochs, n_channels, n_freqs)
    freqs = spectrum.freqs
    ch_names = spectrum.ch_names
    bands_list = []
    feat_names = []
    for bname, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        band_pow = data[:, :, m].mean(axis=2)  # (n_epochs, n_channels) -- power in band
        band_db = 10.0 * np.log10(band_pow + 1e-30)  # log-transform for stability
        bands_list.append(band_db)
        for ch in ch_names:
            feat_names.append(f"{ch}|{bname}")
    X = np.concatenate(bands_list, axis=1)  # (n_epochs, n_channels * n_bands)
    return X, feat_names


def spectral_decoding(real_ep: mne.Epochs, silent_ep: mne.Epochs) -> dict:
    """LR with L2 on PSD-band features. Returns dict with CV scores, confusion matrix, weights."""
    Xr, feat_names = epochs_to_band_features(real_ep)
    Xs, _ = epochs_to_band_features(silent_ep)
    X = np.concatenate([Xr, Xs], axis=0)
    y = np.concatenate([np.ones(len(Xr), dtype=int), np.zeros(len(Xs), dtype=int)])

    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="liblinear"))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=N_JOBS)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=N_JOBS)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])  # rows=true (silent, real), cols=pred

    # fit on all data to extract feature weights
    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_.ravel()
    # top-N features by |coef|
    order = np.argsort(np.abs(coef))[::-1][:TOP_N_FEATS]
    top_feats = [(feat_names[i], float(coef[i])) for i in order]

    return dict(
        scores=scores,
        confusion_matrix=cm,
        coef=coef,
        feat_names=feat_names,
        top_feats=top_feats,
        n_real=len(Xr),
        n_silent=len(Xs),
    )


def plot_spectral_decoding_subject(sub: str, result: dict, out_png: Path) -> None:
    cm = result["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    top = result["top_feats"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1, 1.8]))
    # confusion matrix
    ax = axes[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["silent", "real"])
    ax.set_yticklabels(["silent", "real"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center",
                     color="black" if cm_norm[i, j] < 0.5 else "white", fontsize=10)
    ax.set_title(f"Confusion matrix  (CV acc = {result['scores'].mean()*100:.1f}% ± {result['scores'].std()*100:.1f}%)")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # top features bar chart
    ax = axes[1]
    feats = [f for f, _ in top]
    coefs = [c for _, c in top]
    colors = ["C3" if c > 0 else "C0" for c in coefs]
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, coefs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_xlabel("LR coefficient (positive -> drives 'real' class)")
    ax.set_title(f"Top {TOP_N_FEATS} discriminative features (by |coef|)")

    fig.suptitle(f"{sub}  --  spectral decoding (PSD bands x channels, LR + L2)  "
                  f"n_real={result['n_real']} n_silent={result['n_silent']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_spectral_summary(per_sub_results: dict, out_png: Path) -> None:
    subs = list(per_sub_results.keys())
    means = [per_sub_results[s]["scores"].mean() for s in subs]
    stds = [per_sub_results[s]["scores"].std() for s in subs]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subs))
    ax.bar(x, means, yerr=stds, color="C3", alpha=0.85, edgecolor="black", capsize=4)
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--", label="chance (50%)")
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + 0.02, f"{m*100:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(subs)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("CV accuracy (mean ± SD across 5 folds)")
    ax.set_title("Spectral decoding accuracy per subject")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(f"decoding_scr  ::  output -> {FIG_DIR}")
    print("=" * 78)

    per_sub_temporal: dict = {}
    per_sub_spectral: dict = {}
    common_times: np.ndarray | None = None

    rows_temporal = []
    rows_spectral = []

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print("  no epochs")
            continue
        n_real = len(real_ep)
        n_silent = len(silent_ep)
        print(f"  epochs real={n_real}  silent={n_silent}")

        # --- temporal MVPA ---
        X, y, times, ch_names = build_dataset(real_ep, silent_ep)
        print(f"  X shape={X.shape}  y mean={y.mean():.2f}")
        if common_times is None:
            common_times = times

        print("  fitting temporal MVPA (sliding LR) ...")
        scores_t = temporal_decoding(X, y)  # (n_folds, n_times)
        print(f"    max acc(t) = {scores_t.mean(axis=0).max():.3f} at t = "
              f"{times[scores_t.mean(axis=0).argmax()]*1000:.0f} ms")
        out_temp_png = FIG_DIR / f"Y3_decoding_temporal_{sub}.png"
        plot_temporal_decoding_subject(sub, scores_t, times, n_real, n_silent, out_temp_png)
        print(f"    -> {out_temp_png.name}")
        per_sub_temporal[sub] = scores_t
        for ti, t in enumerate(times):
            rows_temporal.append(dict(
                subject=sub, t_sec=float(t),
                mean_acc=float(scores_t[:, ti].mean()),
                std_acc=float(scores_t[:, ti].std(ddof=1)) if scores_t.shape[0] > 1 else 0.0,
            ))

        # --- spectral decoder ---
        print("  fitting spectral decoder (PSD bands x channels, LR + L2) ...")
        spec = spectral_decoding(real_ep, silent_ep)
        print(f"    CV acc = {spec['scores'].mean()*100:.1f}% ± {spec['scores'].std()*100:.1f}%  "
              f"top feat: {spec['top_feats'][0][0]}  (coef={spec['top_feats'][0][1]:+.3f})")
        out_spec_png = FIG_DIR / f"Y3_decoding_spectral_{sub}.png"
        plot_spectral_decoding_subject(sub, spec, out_spec_png)
        print(f"    -> {out_spec_png.name}")
        per_sub_spectral[sub] = spec
        rows_spectral.append(dict(
            subject=sub,
            cv_acc_mean=float(spec["scores"].mean()),
            cv_acc_std=float(spec["scores"].std(ddof=1)),
            n_real=spec["n_real"],
            n_silent=spec["n_silent"],
            top_feat=spec["top_feats"][0][0],
            top_feat_coef=spec["top_feats"][0][1],
            top_feats_pos=";".join([f for f, c in spec["top_feats"] if c > 0]),
            top_feats_neg=";".join([f for f, c in spec["top_feats"] if c < 0]),
        ))

    # --- grand-average plots ---
    if per_sub_temporal:
        out_ga_temp = FIG_DIR / "Y3_decoding_temporal_grandaverage.png"
        plot_temporal_decoding_grandaverage(per_sub_temporal, common_times, out_ga_temp)
        print(f"\nGrand-avg temporal -> {out_ga_temp.name}")
    if per_sub_spectral:
        out_summary = FIG_DIR / "Y3_decoding_spectral_summary.png"
        plot_spectral_summary(per_sub_spectral, out_summary)
        print(f"Spectral summary -> {out_summary.name}")

    if rows_temporal:
        csv = NPZ_DIR / "decoding_temporal.csv"
        pd.DataFrame(rows_temporal).to_csv(csv, index=False)
        print(f"Temporal CSV -> {csv.name}  ({len(rows_temporal)} rows)")
    if rows_spectral:
        csv = NPZ_DIR / "decoding_spectral.csv"
        pd.DataFrame(rows_spectral).to_csv(csv, index=False)
        print(f"Spectral CSV -> {csv.name}")


if __name__ == "__main__":
    main()
