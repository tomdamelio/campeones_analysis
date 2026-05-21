"""Tarea 4 -- SPoC (Source Power Comodulation) decoding of continuous Y1s from EEG.

Fourth model after the three implemented in ``decoding_y1_3models.py`` (base_raw,
spectral, raw_tde). All three of those operate on sensor-space features
(Vectorize+PCA / log-bandpower+PCA / TDE+covariance). This script adds a
*supervised* spatial-filter model:

    SPoC (Dahne et al. 2014, NeuroImage) -- find spatial filters w such that
    var(w . X_epoch), measured per epoch in a given frequency band, maximally
    correlates with the continuous target Y. Differs from log-bandpower+Ridge in
    two ways: (a) w is learned with knowledge of Y (vs unsupervised PCA on
    log-bandpower), (b) ``patterns_`` are physiologically interpretable as a
    forward projection of the underlying source (Haufe-style correction is
    applied internally by mne.decoding.SPoC). Standard method in the EEG <-> BOLD
    and EEG <-> autonomic regression literature.

What is reused (DO NOT duplicate the logic in those files):
  - SUBJECTS, NPZ_DIR, OUT, runs_for, run_label, attach_montage_and_drop_no_pos
    from ``erp_scr``
  - EEG_WIN_S, EEG_FS_TARGET, SUBSAMPLE_S, LAG_S, ALPHA_GRID, RNG_SEED
    from ``decoding_y1_3models``

What is **locally reimplemented** (justification below):
  - ``build_subject_dataset_banded`` mirrors ``decoding_y1_3models.build_subject_dataset``
    but inserts a continuous-raw band-pass to the chosen SPoC band BEFORE the
    epochs are sliced. Filtering 2 s epochs at delta (1-4 Hz) directly with the
    MNE default FIR (length 825 samples for 250 Hz, vs only 500 samples in a 2 s
    epoch) produces strong edge-of-window distortion that SPoC would happily
    capture as Y-correlated variance -- an artifact, not signal. Filtering the
    raw continuous trace removes this confound entirely. The reimplementation is
    ~30 lines and the only change is the added ``raw.filter(band[0], band[1])``;
    everything else (montage, channel drop, resample, window centering, lag,
    subsampling) is identical to the upstream function.

Cohort: sub-23, sub-24, sub-33 (sub-27 excluded -- see diary 05_2 banner).

Target: Y1s = Gaussian-smoothed (sigma=3 s) SMNA-AUC, same as decoding_y1_3models.

Lag: EEG(t) -> Y(t+2s). The EEG window is centered at (t_y - 2.0 s).

CV: Leave-One-Run-Out per subject (8 folds). Inner alpha selection via
LeaveOneGroupOut over the 7 remaining training runs.

Band: delta (1-4 Hz) in this first pass. Justification from diary 05_2 PSD/TFR
results (sub-24 Temporal real-silent = +11.8 dB delta; sub-33 +3.1 dB).
TODO (future pass): theta (4-8 Hz). To rerun in theta, change SPOC_BAND to
(4.0, 8.0) and SPOC_BAND_NAME to ``"theta"``; no other change required.

Outputs (created on first run):
  figures/spoc/Y4_spoc_<sub>.png         per-subject: scatter / topomap / bars
  figures/spoc/Y4_spoc_summary.png       cross-subject summary (4 metrics)
  y_candidates/decoding_spoc_per_fold.csv     per-fold metrics
  y_candidates/decoding_spoc_per_subject.csv  per-subject means+SDs
  y_candidates/spoc_patterns.npz              per-subject patterns_[0,:] + ch_names

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.spoc_decoding
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
from mne.decoding import SPoC
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from src.campeones_analysis.multimodal_arousal.decoding_y1_3models import (
    ALPHA_GRID,
    EEG_FS_TARGET,
    EEG_WIN_S,
    LAG_S,
    RNG_SEED,
    SUBSAMPLE_S,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    NPZ_DIR,
    OUT,
    SUBJECTS,
    attach_montage_and_drop_no_pos,
    run_label,
    runs_for,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# SPoC hyperparameters
# -----------------------------------------------------------------------------
SPOC_BAND: tuple[float, float] = (1.0, 4.0)
SPOC_BAND_NAME: str = "delta"
SPOC_N_COMPONENTS: int = 4
SPOC_LOG: bool = True
SPOC_REG: str = "oas"  # Oracle Approximating Shrinkage (Gaussian-optimal shrinkage,
                       # related-but-not-equivalent to Ledoit-Wolf).
SPOC_RANK: str = "full"  # raw is filtered+resampled only -- no projectors -> full rank

FIG_DIR: Path = OUT / "figures" / "spoc"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset construction (band-pass at the *raw* level, then epoch)
# -----------------------------------------------------------------------------
def build_subject_dataset_banded(
    sub: str, band: tuple[float, float], band_name: str
) -> dict | None:
    """Per subject: load preproc EEG, band-pass the *raw* trace, epoch -> (X, y, groups).

    Mirrors decoding_y1_3models.build_subject_dataset but applies an extra
    raw.filter(band[0], band[1]) AFTER the wideband 1-40 Hz filter and BEFORE
    epoch slicing. Same lag (EEG(t)->Y(t+LAG_S)), same EEG window length, same
    subsampling, same montage handling.
    """
    cont_npz = NPZ_DIR / f"{sub}_y_candidates.npz"
    if not cont_npz.exists():
        print(f"  missing {cont_npz.name}; skipping")
        return None
    yc = np.load(cont_npz, allow_pickle=True)
    runs_in_npz = [str(r) for r in yc["runs"]]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    g_list: list[str] = []
    c_list: list[float] = []  # t_y per epoch (centers of Y1s window, in seconds within run)
    ch_names: list[str] | None = None
    sfreq: float | None = None

    eeg_win_samples = int(round(EEG_WIN_S * EEG_FS_TARGET))
    half_win_s = EEG_WIN_S / 2.0

    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        if f"{label}__centers" not in yc.files or f"{label}__y1s" not in yc.files:
            continue
        centers = np.asarray(yc[f"{label}__centers"], dtype=float)
        y1s = np.asarray(yc[f"{label}__y1s"], dtype=float)
        valid = np.isfinite(y1s)
        centers = centers[valid]
        y1s = y1s[valid]

        step = int(round(SUBSAMPLE_S / (centers[1] - centers[0]))) if len(centers) > 1 else 1
        idx = np.arange(0, len(centers), step)
        idx = idx[1:]  # drop first sample (lag would push EEG window before run start)
        if len(idx) == 0:
            continue
        centers_sel = centers[idx]
        y_sel = y1s[idx]

        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            # Wideband first (matches the unbanded pipeline), then the SPoC band
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.filter(l_freq=band[0], h_freq=band[1], verbose="ERROR")
            raw.resample(EEG_FS_TARGET, verbose="ERROR")
            if ch_names is None:
                ch_names = list(raw.ch_names)
                sfreq = float(raw.info["sfreq"])
            duration = float(raw.times[-1])

            data_full = raw.get_data()
            n_total_samples = data_full.shape[1]
            n_kept = 0
            for t_y, y_val in zip(centers_sel, y_sel):
                t_center_eeg = t_y - LAG_S
                t_start = t_center_eeg - half_win_s
                t_end = t_center_eeg + half_win_s
                if t_start < 0 or t_end > duration:
                    continue
                s_start = int(round(t_start * sfreq))
                s_end = s_start + eeg_win_samples
                if s_end > n_total_samples:
                    continue
                X_list.append(data_full[:, s_start:s_end])
                y_list.append(float(y_val))
                g_list.append(label)
                c_list.append(float(t_y))
                n_kept += 1
            print(f"  {label} ({band_name}): kept {n_kept}/{len(centers_sel)} epochs")
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    if not X_list:
        return None
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=float)
    groups = np.array(g_list)
    centers = np.array(c_list, dtype=float)
    return dict(X=X, y=y, groups=groups, centers=centers, ch_names=ch_names, sfreq=sfreq)


# -----------------------------------------------------------------------------
# Score helper -- identical to decoding_y1_3models._score
# -----------------------------------------------------------------------------
def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    has_var = np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12
    return dict(
        pearson_r=float(pearsonr(y_true, y_pred).statistic) if has_var else 0.0,
        spearman_rho=float(spearmanr(y_true, y_pred).correlation) if has_var else 0.0,
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# -----------------------------------------------------------------------------
# Pipeline factory
# -----------------------------------------------------------------------------
def _make_pipeline() -> Pipeline:
    return Pipeline([
        ("spoc", SPoC(
            n_components=SPOC_N_COMPONENTS,
            log=SPOC_LOG,
            reg=SPOC_REG,
            rank=SPOC_RANK,
            transform_into="average_power",
        )),
        ("ridge", Ridge()),
    ])


# -----------------------------------------------------------------------------
# Per-fold fit + eval
# -----------------------------------------------------------------------------
def fit_eval_spoc_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
) -> dict:
    """Fit SPoC+Ridge with inner LOGO alpha grid. Return metrics + y_pred + best_alpha."""
    pipe = _make_pipeline()
    unique_groups = np.unique(groups_train)
    if len(unique_groups) >= 2:
        inner_cv = LeaveOneGroupOut()
        gs = GridSearchCV(
            pipe,
            param_grid={"ridge__alpha": ALPHA_GRID},
            cv=list(inner_cv.split(X_train, y_train, groups=groups_train)),
            scoring="r2",
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_alpha = float(gs.best_params_["ridge__alpha"])
        fitted = gs.best_estimator_
    else:
        pipe.set_params(ridge__alpha=10.0)
        pipe.fit(X_train, y_train)
        best_alpha = 10.0
        fitted = pipe
    y_pred = fitted.predict(X_test)
    out = _score(y_test, y_pred)
    out["best_alpha"] = best_alpha
    out["y_pred"] = y_pred
    return out


# -----------------------------------------------------------------------------
# Refit on full data to extract patterns_ for visualization
# -----------------------------------------------------------------------------
def refit_for_patterns(X: np.ndarray, y: np.ndarray, alpha: float) -> SPoC:
    """Refit SPoC+Ridge on ALL data (no CV) and return the fitted SPoC step.

    Note: alpha only affects Ridge weights downstream; patterns_ come entirely
    from the SPoC generalized eigendecomposition, which sees no alpha. We still
    fit the full pipeline so the returned object reflects the actual model
    morphology used during CV.
    """
    pipe = _make_pipeline()
    pipe.set_params(ridge__alpha=alpha)
    pipe.fit(X, y)
    return pipe.named_steps["spoc"]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _build_topo_info(ch_names: list[str], sfreq: float) -> mne.Info:
    """Build an Info object with the standard_1020 montage, dropping channels w/o positions."""
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    try:
        mont = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(mont, match_case=False, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    return info


def plot_subject(
    sub: str,
    rows: list[dict],
    per_run_preds: dict[str, dict],
    spoc_fitted: SPoC,
    ch_names: list[str],
    sfreq: float,
    out_png: Path,
) -> None:
    df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35, height_ratios=[1.0, 0.85])

    # Panel 1: scatter y_true vs y_pred across all folds, color-coded by fold
    ax1 = fig.add_subplot(gs[0, 0])
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    runs_sorted = sorted(per_run_preds.keys())
    cmap = plt.get_cmap("tab10")
    for i, run_held in enumerate(runs_sorted):
        yt = per_run_preds[run_held]["y_test"]
        yp = per_run_preds[run_held]["y_pred"]
        ax1.scatter(yt, yp, s=10, alpha=0.45, color=cmap(i % 10), label=run_held)
        y_true_all.append(yt)
        y_pred_all.append(yp)
    if y_true_all:
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        lo = float(min(y_true_cat.min(), y_pred_cat.min()))
        hi = float(max(y_true_cat.max(), y_pred_cat.max()))
        ax1.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--")
        r_cf = float(pearsonr(y_true_cat, y_pred_cat).statistic) if np.std(y_pred_cat) > 1e-12 else 0.0
        r2_cf = float(r2_score(y_true_cat, y_pred_cat))
    else:
        r_cf, r2_cf = 0.0, 0.0
    ax1.set_xlabel("y_true (Y1s)")
    ax1.set_ylabel("y_pred")
    ax1.set_title(f"Cross-fold scatter\nr = {r_cf:+.3f}   R² = {r2_cf:+.3f}", fontsize=10)
    ax1.legend(fontsize=6, loc="best", ncol=2)

    # Panel 2: topomap of pattern of component 0 (Haufe-corrected internally by MNE)
    ax2 = fig.add_subplot(gs[0, 1])
    info = _build_topo_info(ch_names, sfreq)
    pattern0 = spoc_fitted.patterns_[0, :]  # NB: row = component (verified)
    pos_ch = [
        ch for ch, ch_info in zip(info.ch_names, info["chs"])
        if not (np.isnan(ch_info["loc"][:3]).any() or np.allclose(ch_info["loc"][:3], 0.0))
    ]
    if len(pos_ch) >= 3 and len(pos_ch) == len(info.ch_names):
        vmax = float(np.max(np.abs(pattern0)))
        im, _ = mne.viz.plot_topomap(
            pattern0, info, axes=ax2, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), sensors=True, contours=4,
        )
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f"Spatial pattern -- {SPOC_BAND_NAME} comp 1", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "topomap unavailable\n(montage mismatch)",
                 ha="center", va="center", fontsize=9, transform=ax2.transAxes)
        ax2.set_axis_off()

    # Panel 3: bar chart of [Pearson r, R^2, RMSE] (mean +/- SD across folds)
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ["pearson_r", "r2", "rmse"]
    means = [float(df[m].mean()) for m in metrics]
    stds = [float(df[m].std(ddof=1)) for m in metrics]
    x = np.arange(len(metrics))
    bars = ax3.bar(x, means, yerr=stds, color=["C0", "C2", "C3"],
                   alpha=0.85, edgecolor="black", capsize=4)
    for xi, mn, sd, m in zip(x, means, stds, metrics):
        y_text = mn + (sd if sd > 0 else 0.0) + 0.02 * (1 if mn >= 0 else -1)
        ax3.text(xi, y_text, f"{mn:.3f}±{sd:.3f}", ha="center", fontsize=9)
        vals = df[m].to_numpy()
        ax3.scatter(np.full(len(vals), xi), vals, color="black", s=14, alpha=0.55, zorder=3)
    ax3.axhline(0, color="0.4", lw=0.8, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in metrics])
    ax3.set_title("Per-fold metrics", fontsize=10)
    _ = bars  # silence linter

    # Panel 4 (full-width row 1): time-series y_true vs y_pred for the best fold
    ax_ts = fig.add_subplot(gs[1, :])
    if per_run_preds:
        best_run = df.loc[df["pearson_r"].idxmax(), "run_held_out"]
        best_r = float(df.loc[df["pearson_r"].idxmax(), "pearson_r"])
        yt = per_run_preds[best_run]["y_test"]
        yp = per_run_preds[best_run]["y_pred"]
        t_axis = np.arange(len(yt)) * SUBSAMPLE_S
        ax_ts.plot(t_axis, yt, color="0.35", lw=1.4, label="y_true (Y1s)")
        ax_ts.plot(t_axis, yp, color="C0", lw=1.6, label=f"y_pred (SPoC {SPOC_BAND_NAME})")
        ax_ts.set_xlabel(f"time within held-out run {best_run} (s, step = {SUBSAMPLE_S:.0f} s)")
        ax_ts.set_ylabel("Y1s (smoothed SMNA-AUC)")
        ax_ts.set_title(
            f"Best fold = {best_run}   |   Pearson r = {best_r:+.3f}",
            fontsize=10,
        )
        ax_ts.legend(fontsize=9, loc="upper right")
        ax_ts.grid(alpha=0.25)
    else:
        ax_ts.text(0.5, 0.5, "no folds available", ha="center", va="center",
                   transform=ax_ts.transAxes)
        ax_ts.set_axis_off()

    fig.suptitle(
        f"{sub}  --  SPoC ({SPOC_BAND_NAME} {SPOC_BAND[0]:.0f}-{SPOC_BAND[1]:.0f} Hz)  "
        f"K={SPOC_N_COMPONENTS}, LORO-CV, EEG(t)->Y(t+{LAG_S:.0f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_summary(all_rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(all_rows)
    subs = sorted(df["subject"].unique())
    metrics = ("pearson_r", "spearman_rho", "r2", "rmse")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for sub in subs:
            sub_df = df[df["subject"] == sub]
            means.append(float(sub_df[metric].mean()))
            stds.append(float(sub_df[metric].std(ddof=1)))
        x = np.arange(len(subs))
        colors = [f"C{i}" for i in range(len(subs))]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, edgecolor="black", capsize=4)
        for xi, mn, sd in zip(x, means, stds):
            y_text = mn + (sd if sd > 0 else 0.0) + 0.02 * (1 if mn >= 0 else -1)
            ax.text(xi, y_text, f"{mn:.3f}", ha="center", fontsize=9)
        ga = float(df[metric].mean())
        ax.axhline(ga, color="black", lw=1.5, ls="-")
        ax.text(len(subs) - 0.5, ga, f" GA={ga:.3f}", va="center", fontsize=9, fontweight="bold")
        if metric in ("pearson_r", "spearman_rho", "r2"):
            ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(subs)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
    fig.suptitle(
        f"SPoC ({SPOC_BAND_NAME} {SPOC_BAND[0]:.0f}-{SPOC_BAND[1]:.0f} Hz) -- "
        f"per-subject means ± SD across LORO folds (grand average as black line)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(
        f"spoc_decoding :: band={SPOC_BAND} ({SPOC_BAND_NAME}), "
        f"K={SPOC_N_COMPONENTS}, LORO per subject"
    )
    print(f"  EEG_WIN_S={EEG_WIN_S}  SUBSAMPLE_S={SUBSAMPLE_S}  LAG_S={LAG_S}")
    print(f"  output -> {FIG_DIR}")
    print("=" * 78)

    all_rows: list[dict] = []
    patterns_per_sub: dict[str, np.ndarray] = {}
    ch_names_per_sub: dict[str, list[str]] = {}

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        ds = build_subject_dataset_banded(sub, SPOC_BAND, SPOC_BAND_NAME)
        if ds is None:
            continue
        X = ds["X"]
        y = ds["y"]
        groups = ds["groups"]
        ch_names = ds["ch_names"]
        sfreq = ds["sfreq"]
        print(
            f"  X shape={X.shape}  y range=[{y.min():.4f},{y.max():.4f}] "
            f"mean={y.mean():.4f}"
        )

        unique_runs = list(np.unique(groups))
        rows: list[dict] = []
        per_run_preds: dict[str, dict] = {}
        for run_held in unique_runs:
            test_mask = groups == run_held
            train_mask = ~test_mask
            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())
            if n_test < 5 or n_train < 20:
                print(f"  fold test={run_held}: too few epochs ({n_test},{n_train}) -- skipping")
                continue
            groups_train = groups[train_mask]
            print(f"  fold test={run_held}  n_train={n_train}  n_test={n_test} ...")
            r = fit_eval_spoc_fold(
                X[train_mask], y[train_mask],
                X[test_mask], y[test_mask],
                groups_train,
            )
            per_run_preds[run_held] = {
                "y_test": y[test_mask],
                "y_pred": r.pop("y_pred"),
            }
            rows.append(dict(
                subject=sub,
                model=f"spoc_{SPOC_BAND_NAME}",
                run_held_out=run_held,
                best_alpha=r["best_alpha"],
                pearson_r=r["pearson_r"],
                spearman_rho=r["spearman_rho"],
                r2=r["r2"],
                rmse=r["rmse"],
                n_train=n_train,
                n_test=n_test,
            ))
            print(
                f"    r={r['pearson_r']:+.3f}  rho={r['spearman_rho']:+.3f}  "
                f"R2={r['r2']:+.3f}  rmse={r['rmse']:.4f}  alpha={r['best_alpha']:.1f}"
            )

        all_rows.extend(rows)

        if rows:
            df_sub = pd.DataFrame(rows)
            alpha_for_refit = float(np.median(df_sub["best_alpha"].to_numpy()))
            print(f"  refit on full data with alpha={alpha_for_refit:.2f} (median across folds)")
            spoc_fitted = refit_for_patterns(X, y, alpha_for_refit)
            patterns_per_sub[sub] = spoc_fitted.patterns_[0, :].astype(float)
            ch_names_per_sub[sub] = list(ch_names)
            out_png = FIG_DIR / f"Y4_spoc_{sub}.png"
            plot_subject(sub, rows, per_run_preds, spoc_fitted, ch_names, sfreq, out_png)
            print(f"  -> {out_png.name}")

    if all_rows:
        out_summary = FIG_DIR / "Y4_spoc_summary.png"
        plot_summary(all_rows, out_summary)
        print(f"\nSummary figure -> {out_summary.name}")

        df_fold = pd.DataFrame(all_rows)
        out_csv_fold = NPZ_DIR / "decoding_spoc_per_fold.csv"
        df_fold.to_csv(out_csv_fold, index=False)
        print(f"Per-fold CSV   -> {out_csv_fold.name}  ({len(df_fold)} rows)")

        rows_subj: list[dict] = []
        for sub in df_fold["subject"].unique():
            sub_df = df_fold[df_fold["subject"] == sub]
            rows_subj.append(dict(
                subject=sub,
                model=f"spoc_{SPOC_BAND_NAME}",
                n_folds=int(len(sub_df)),
                mean_pearson_r=float(sub_df["pearson_r"].mean()),
                sd_pearson_r=float(sub_df["pearson_r"].std(ddof=1)),
                mean_spearman_rho=float(sub_df["spearman_rho"].mean()),
                mean_r2=float(sub_df["r2"].mean()),
                sd_r2=float(sub_df["r2"].std(ddof=1)),
                mean_rmse=float(sub_df["rmse"].mean()),
                median_alpha=float(np.median(sub_df["best_alpha"].to_numpy())),
            ))
        out_csv_subj = NPZ_DIR / "decoding_spoc_per_subject.csv"
        pd.DataFrame(rows_subj).to_csv(out_csv_subj, index=False)
        print(f"Per-subject CSV -> {out_csv_subj.name}")

        if patterns_per_sub:
            out_patt = NPZ_DIR / "spoc_patterns.npz"
            np.savez(
                out_patt,
                band=np.array(SPOC_BAND),
                band_name=SPOC_BAND_NAME,
                **{f"{sub}__pattern0": patterns_per_sub[sub] for sub in patterns_per_sub},
                **{f"{sub}__ch_names": np.array(ch_names_per_sub[sub], dtype=object)
                   for sub in patterns_per_sub},
            )
            print(f"Patterns NPZ    -> {out_patt.name}")


if __name__ == "__main__":
    main()
