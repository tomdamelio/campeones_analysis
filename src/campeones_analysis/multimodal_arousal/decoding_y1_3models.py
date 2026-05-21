"""Continuous-Y regression decoding of Y1s (Gaussian-smoothed SMNA AUC) from EEG.

Tarea 3 of the 2026-05 diary (X = EEG, Y = single signal, Ridge + LORO-CV) -- analog
to the luminance pipeline (`10/11/13_luminance_*` from the pivote-xy-doc worktree) but
adapted for SMNA prediction with an explicit central->peripheral lag.

LAG: EEG(t) -> Y(t + 2 s). The EEG window centered at (t_y - 2 s) predicts the SMNA
AUC at t_y. Motivation: SCR rise time is 1-3 s (Bach et al. 2010), so cortical activity
must precede the autonomic response. The lag is FIXED in this first pass (no grid search
over lag) so that any positive R^2 directly tests the hypothesis "EEG precedes EDA".

Three models, all with Ridge regression and GridSearchCV alpha on inner LOGO:
  1. base_raw    -- Vectorizer -> StandardScaler -> PCA(100) -> Ridge
  2. spectral    -- band-power per (channel, band) -> StandardScaler -> PCA(50) -> Ridge
  3. raw_tde     -- per-epoch TDE (lags +-10) + per-fold global PCA(20) on training pool
                    + covariance triu -> StandardScaler -> Ridge

CV: Leave-One-Run-Out per subject (8 folds per subject). Inner CV for alpha: LOGO over
the 6 non-test, non-held-out training runs (each fold's training pool).

Outputs:
  figures/decoding_y1/Y3_decoding_y1_<sub>.png       per-subject (scatters + timecourse + bars)
  figures/decoding_y1/Y3_decoding_y1_summary.png    grand-avg + per-subject bar chart
  y_candidates/decoding_y1_per_fold.csv             per-fold metrics
  y_candidates/decoding_y1_per_subject.csv          per-subject means

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_y1_3models
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
from mne.decoding import Vectorizer
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.decoding_scr_3models import (
    BANDS,
    apply_tde,
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

FIG_DIR = OUT / "figures" / "decoding_y1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Time / lag parameters ---
EEG_WIN_S = 2.0          # length of EEG window per epoch
EEG_FS_TARGET = 250.0    # resample EEG to this for analysis
SUBSAMPLE_S = 2.0        # subsample Y1 grid; one "epoch" = SUBSAMPLE_S seconds
LAG_S = 2.0              # EEG(t) predicts Y(t + LAG_S). The EEG window centered at (t_y - LAG_S).

# --- Model hyperparameters (mirror luminance) ---
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
PCA_BASE = 100
PCA_SPECTRAL = 50
PCA_TDE = 20
TDE_LAG = 10
RNG_SEED = 42
PSD_NFFT_LOCAL = 256  # must be <= EEG_WIN_S * EEG_FS_TARGET = 500


# -----------------------------------------------------------------------------
# Local spectral feature extractor (decoding_scr_3models.compute_band_power_features
# hardcodes n_fft=512 which is incompatible with our 500-sample EEG window)
# -----------------------------------------------------------------------------
def compute_band_power_features_local(X: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-epoch log-band-power. Returns (n_epochs, n_channels * n_bands).

    Uses MNE Welch with n_fft=PSD_NFFT_LOCAL (=256), compatible with 2 s windows at 250 Hz.
    """
    info = mne.create_info(ch_names=[f"ch{i}" for i in range(X.shape[1])], sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, verbose="ERROR")
    spectrum = epochs.compute_psd(
        method="welch", fmin=1.0, fmax=40.0,
        n_fft=PSD_NFFT_LOCAL, n_per_seg=PSD_NFFT_LOCAL, verbose="ERROR",
    )
    data = spectrum.get_data()
    freqs = spectrum.freqs
    n_epochs, n_channels, _ = data.shape
    feats = np.zeros((n_epochs, n_channels * len(BANDS)))
    for i, (lo, hi) in enumerate(BANDS.values()):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            band_pow = data[:, :, m].mean(axis=2)
            feats[:, i::len(BANDS)] = 10.0 * np.log10(band_pow + 1e-30)
    return feats


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def build_subject_dataset(sub: str) -> dict | None:
    """Per subject: load preproc EEG + Y1s, build (X, y, groups, ch_names, sfreq)."""
    cont_npz = NPZ_DIR / f"{sub}_y_candidates.npz"
    if not cont_npz.exists():
        print(f"  missing {cont_npz.name}; skipping")
        return None
    yc = np.load(cont_npz, allow_pickle=True)
    runs_in_npz = [str(r) for r in yc["runs"]]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    g_list: list[str] = []
    ch_names = None
    sfreq = None

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

        # Subsample Y1 every SUBSAMPLE_S seconds (centers are at 1 Hz).
        # centers come at 1 s spacing; take every SUBSAMPLE_S-th sample.
        step = int(round(SUBSAMPLE_S / (centers[1] - centers[0]))) if len(centers) > 1 else 1
        idx = np.arange(0, len(centers), step)
        # Drop first sample (lag would push EEG window before run start)
        idx = idx[1:]
        if len(idx) == 0:
            continue
        centers_sel = centers[idx]
        y_sel = y1s[idx]

        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.resample(EEG_FS_TARGET, verbose="ERROR")
            if ch_names is None:
                ch_names = list(raw.ch_names)
                sfreq = float(raw.info["sfreq"])
            duration = float(raw.times[-1])

            data_full = raw.get_data()  # (n_channels, n_samples)
            n_total_samples = data_full.shape[1]
            n_kept = 0
            for t_y, y_val in zip(centers_sel, y_sel):
                # EEG window centered at (t_y - LAG_S)
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
                n_kept += 1
            print(f"  {label}: kept {n_kept}/{len(centers_sel)} epochs")
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    if not X_list:
        return None
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=float)
    groups = np.array(g_list)
    return dict(X=X, y=y, groups=groups, ch_names=ch_names, sfreq=sfreq)


# -----------------------------------------------------------------------------
# Score helper
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
# Inner-CV alpha selection
# -----------------------------------------------------------------------------
def _fit_ridge_grid(pipeline_for_alpha_search: Pipeline, X_train, y_train, groups_train) -> tuple[float, Pipeline]:
    """GridSearch alpha via LOGO over groups_train. Returns (best_alpha, fitted_pipeline)."""
    unique_groups = np.unique(groups_train)
    if len(unique_groups) >= 2:
        cv = LeaveOneGroupOut()
        param_grid = {"ridge__alpha": ALPHA_GRID}
        gs = GridSearchCV(
            pipeline_for_alpha_search, param_grid, cv=cv.split(X_train, y_train, groups=groups_train),
            scoring="r2", n_jobs=1, refit=True,
        )
        gs.fit(X_train, y_train)
        return float(gs.best_params_["ridge__alpha"]), gs.best_estimator_
    # fallback: fit with median alpha
    pipeline_for_alpha_search.set_params(ridge__alpha=10.0)
    pipeline_for_alpha_search.fit(X_train, y_train)
    return 10.0, pipeline_for_alpha_search


# -----------------------------------------------------------------------------
# Per-model fit/eval
# -----------------------------------------------------------------------------
def fit_eval_base(X_train, y_train, X_test, y_test, groups_train) -> dict:
    n_pca = min(PCA_BASE, X_train.shape[0] - 1)
    pipe = Pipeline([
        ("vec", Vectorizer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RNG_SEED)),
        ("ridge", Ridge()),
    ])
    best_alpha, fitted = _fit_ridge_grid(pipe, X_train, y_train, groups_train)
    y_pred = fitted.predict(X_test)
    out = _score(y_test, y_pred)
    out["best_alpha"] = best_alpha
    out["y_pred"] = y_pred
    return out


def fit_eval_spectral(X_train_feat, y_train, X_test_feat, y_test, groups_train) -> dict:
    n_pca = min(PCA_SPECTRAL, X_train_feat.shape[0] - 1, X_train_feat.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RNG_SEED)),
        ("ridge", Ridge()),
    ])
    best_alpha, fitted = _fit_ridge_grid(pipe, X_train_feat, y_train, groups_train)
    y_pred = fitted.predict(X_test_feat)
    out = _score(y_test, y_pred)
    out["best_alpha"] = best_alpha
    out["y_pred"] = y_pred
    return out


TDE_POOL_TARGET_SAMPLES = 200_000  # cap pool size for global-PCA fit to control memory


def _tde_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch TDE -> global PCA fitted on subsampled training pool -> per-epoch covariance triu.

    Memory-conscious: we never materialize all TDE-embedded epochs at once. We collect a
    subsampled pool for PCA fit, then re-compute TDE per epoch on the fly for the final
    covariance-triu features.
    """
    # Phase 1: build a subsampled training pool for PCA fit
    target_per_epoch = max(10, TDE_POOL_TARGET_SAMPLES // max(1, len(X_train)))
    pool_chunks: list[np.ndarray] = []
    for i in range(len(X_train)):
        tde_ep = apply_tde(X_train[i])             # (n_feat, n_valid)
        n_cols = tde_ep.shape[1]
        n_take = min(target_per_epoch, n_cols)
        if n_take < n_cols:
            sel = np.linspace(0, n_cols - 1, n_take).astype(int)
            pool_chunks.append(tde_ep[:, sel].T.astype(np.float32))
        else:
            pool_chunks.append(tde_ep.T.astype(np.float32))
    pool = np.concatenate(pool_chunks, axis=0).astype(np.float64)
    del pool_chunks

    scaler = StandardScaler().fit(pool)
    n_pca = min(PCA_TDE, pool.shape[1], pool.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=RNG_SEED).fit(scaler.transform(pool))
    del pool

    # Phase 2: per-epoch transform (TDE -> scale -> PCA -> cov triu) inline
    def to_features(X_set: np.ndarray) -> np.ndarray:
        out = np.empty((len(X_set), n_pca * (n_pca + 1) // 2), dtype=np.float64)
        for i in range(len(X_set)):
            tde_ep = apply_tde(X_set[i])
            ep = scaler.transform(tde_ep.T)
            ep_pca = pca.transform(ep)
            cov = np.cov(ep_pca.T)
            cov = np.atleast_2d(cov)
            out[i] = cov[np.triu_indices(cov.shape[0])]
        return out

    return to_features(X_train), to_features(X_test)


def fit_eval_tde(X_train, y_train, X_test, y_test, groups_train) -> dict:
    Xf_train, Xf_test = _tde_features(X_train, X_test)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    best_alpha, fitted = _fit_ridge_grid(pipe, Xf_train, y_train, groups_train)
    y_pred = fitted.predict(Xf_test)
    out = _score(y_test, y_pred)
    out["best_alpha"] = best_alpha
    out["y_pred"] = y_pred
    return out


# -----------------------------------------------------------------------------
# LORO loop per subject
# -----------------------------------------------------------------------------
def loro_subject(sub: str, ds: dict) -> tuple[list[dict], dict]:
    X = ds["X"]; y = ds["y"]; groups = ds["groups"]; sfreq = ds["sfreq"]
    unique_runs = list(np.unique(groups))
    print(f"  X shape={X.shape}  y range=[{y.min():.4f},{y.max():.4f}] mean={y.mean():.4f}")
    print(f"  computing spectral features once ...")
    X_spec = compute_band_power_features_local(X, sfreq)
    print(f"  X_spec shape={X_spec.shape}")

    rows = []
    per_run_preds: dict[str, dict] = {}  # for the example time-series plot
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

        r_base = fit_eval_base(X[train_mask], y[train_mask], X[test_mask], y[test_mask], groups_train)
        r_spec = fit_eval_spectral(X_spec[train_mask], y[train_mask], X_spec[test_mask], y[test_mask], groups_train)
        r_tde = fit_eval_tde(X[train_mask], y[train_mask], X[test_mask], y[test_mask], groups_train)

        per_run_preds[run_held] = {
            "y_test": y[test_mask],
            "base_pred": r_base.pop("y_pred"),
            "spectral_pred": r_spec.pop("y_pred"),
            "tde_pred": r_tde.pop("y_pred"),
        }

        for model_name, r in (("base_raw", r_base), ("spectral", r_spec), ("raw_tde", r_tde)):
            rows.append(dict(
                subject=sub, model=model_name, run_held_out=run_held,
                best_alpha=r["best_alpha"],
                pearson_r=r["pearson_r"], spearman_rho=r["spearman_rho"],
                r2=r["r2"], rmse=r["rmse"],
                n_train=n_train, n_test=n_test,
            ))
        print(f"    base r={r_base['pearson_r']:+.3f} R2={r_base['r2']:+.3f} alpha={r_base['best_alpha']:.1f}  "
              f"spec r={r_spec['pearson_r']:+.3f} R2={r_spec['r2']:+.3f} alpha={r_spec['best_alpha']:.1f}  "
              f"tde r={r_tde['pearson_r']:+.3f} R2={r_tde['r2']:+.3f} alpha={r_tde['best_alpha']:.1f}")

    return rows, per_run_preds


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
MODEL_ORDER = ["base_raw", "spectral", "raw_tde"]
MODEL_COLOR = {"base_raw": "C0", "spectral": "C2", "raw_tde": "C3"}


def plot_subject(sub: str, rows: list[dict], per_run_preds: dict, out_png: Path) -> None:
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, height_ratios=[1.1, 1.0, 0.9])

    # ROW 0: scatter y_true vs y_pred (across folds concatenated) per model
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[0, col])
        y_true_all, y_pred_all = [], []
        for run_held, preds in per_run_preds.items():
            y_true_all.append(preds["y_test"])
            y_pred_all.append(preds[f"{'base' if model == 'base_raw' else model.replace('raw_', '').replace('spectral', 'spectral')}_pred"])
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        ax.scatter(y_true_all, y_pred_all, s=8, alpha=0.35, color=MODEL_COLOR[model])
        lo, hi = min(y_true_all.min(), y_pred_all.min()), max(y_true_all.max(), y_pred_all.max())
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--", label="identity")
        sub_df = df[df["model"] == model]
        ax.set_title(f"{model}  --  mean r={sub_df['pearson_r'].mean():.3f}  R2={sub_df['r2'].mean():.3f}", fontsize=10)
        ax.set_xlabel("y_true (Y1s)")
        ax.set_ylabel("y_pred")
        ax.legend(fontsize=8, loc="upper left")

    # ROW 1: example time-series of one test run (best Pearson r of base_raw for choice) per model
    if per_run_preds:
        # pick the test run with best Pearson r for spectral (typically best)
        spec_rows = df[df["model"] == "spectral"]
        if not spec_rows.empty:
            best_run = spec_rows.loc[spec_rows["pearson_r"].idxmax(), "run_held_out"]
        else:
            best_run = next(iter(per_run_preds))
    else:
        best_run = None

    if best_run is not None and best_run in per_run_preds:
        y_true_run = per_run_preds[best_run]["y_test"]
        t_axis = np.arange(len(y_true_run)) * SUBSAMPLE_S
        for col, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[1, col])
            key = "base_pred" if model == "base_raw" else ("spectral_pred" if model == "spectral" else "tde_pred")
            ax.plot(t_axis, y_true_run, color="0.4", lw=1.2, label="y_true")
            ax.plot(t_axis, per_run_preds[best_run][key], color=MODEL_COLOR[model], lw=1.4, label=f"y_pred ({model})")
            ax.set_xlabel("time within held-out run (s)")
            ax.set_ylabel("Y1s")
            ax.set_title(f"{model} -- best fold = {best_run}", fontsize=9)
            ax.legend(fontsize=8)

    # ROW 2: bar charts of metrics per model
    metrics = ["pearson_r", "r2", "rmse"]
    for col, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[2, col])
        x = np.arange(len(MODEL_ORDER))
        means = [df[df["model"] == m][metric].mean() for m in MODEL_ORDER]
        stds = [df[df["model"] == m][metric].std(ddof=1) for m in MODEL_ORDER]
        ax.bar(x, means, yerr=stds, color=[MODEL_COLOR[m] for m in MODEL_ORDER],
               alpha=0.85, edgecolor="black", capsize=4)
        for i, m in enumerate(MODEL_ORDER):
            vals = df[df["model"] == m][metric].values
            ax.scatter(np.full(len(vals), i), vals, color="black", s=18, alpha=0.55, zorder=3)
        if metric in ("pearson_r", "r2"):
            ax.axhline(0, color="0.4", lw=0.8, ls="--")
        for xi, mn in zip(x, means):
            ax.text(xi, mn + (max(stds) * 0.3 if max(stds) > 0 else 0.05), f"{mn:.3f}", ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())

    fig.suptitle(
        f"{sub}  --  Y1s regression decoding  (EEG(t) -> Y(t+{LAG_S:.0f}s), LORO-CV)  "
        f"3 models: {', '.join(MODEL_ORDER)}",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_summary(all_rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(all_rows)
    subs = sorted(df["subject"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric in zip(axes, ("pearson_r", "spearman_rho", "r2", "rmse")):
        x = np.arange(len(MODEL_ORDER))
        width = 0.22
        for i, sub in enumerate(subs):
            df_sub = df[df["subject"] == sub]
            means = [df_sub[df_sub["model"] == m][metric].mean() for m in MODEL_ORDER]
            stds = [df_sub[df_sub["model"] == m][metric].std(ddof=1) for m in MODEL_ORDER]
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=sub, capsize=3,
                   color=f"C{i}", alpha=0.85, edgecolor="black")
        for j, m in enumerate(MODEL_ORDER):
            ga = df[df["model"] == m][metric].mean()
            ax.hlines(ga, j - 0.45, j + 0.45, colors="black", lw=2.2)
            ax.text(j, ga + (0.01 if metric != "rmse" else 0.0), f"GA={ga:.3f}", ha="center", fontsize=9, fontweight="bold")
        if metric in ("pearson_r", "spearman_rho", "r2"):
            ax.axhline(0, color="0.4", lw=0.8, ls="--", label="chance / no skill" if metric == "pearson_r" else None)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        if metric == "pearson_r":
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Y1s regression decoding -- 3 models summary  (EEG(t) -> Y(t+{LAG_S:.0f}s), LORO per subject, "
        f"bars = mean ± SD across folds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(f"decoding_y1_3models  ::  output -> {FIG_DIR}")
    print(f"  EEG_WIN_S={EEG_WIN_S}  SUBSAMPLE_S={SUBSAMPLE_S}  LAG_S={LAG_S}  (EEG(t) -> Y(t+{LAG_S}s))")
    print("=" * 78)

    all_rows = []
    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        ds = build_subject_dataset(sub)
        if ds is None:
            continue
        rows, per_run_preds = loro_subject(sub, ds)
        all_rows.extend(rows)
        if rows:
            out_png = FIG_DIR / f"Y3_decoding_y1_{sub}.png"
            plot_subject(sub, rows, per_run_preds, out_png)
            print(f"  -> {out_png.name}")

    if all_rows:
        out_summary = FIG_DIR / "Y3_decoding_y1_summary.png"
        plot_summary(all_rows, out_summary)
        print(f"\nSummary -> {out_summary.name}")

        df_fold = pd.DataFrame(all_rows)
        out_csv_fold = NPZ_DIR / "decoding_y1_per_fold.csv"
        df_fold.to_csv(out_csv_fold, index=False)
        print(f"Per-fold CSV -> {out_csv_fold.name}  ({len(df_fold)} rows)")

        rows_subj = []
        for sub in df_fold["subject"].unique():
            for m in MODEL_ORDER:
                sub_m = df_fold[(df_fold["subject"] == sub) & (df_fold["model"] == m)]
                rows_subj.append(dict(
                    subject=sub, model=m,
                    mean_pearson_r=float(sub_m["pearson_r"].mean()),
                    sd_pearson_r=float(sub_m["pearson_r"].std(ddof=1)),
                    mean_spearman_rho=float(sub_m["spearman_rho"].mean()),
                    mean_r2=float(sub_m["r2"].mean()),
                    sd_r2=float(sub_m["r2"].std(ddof=1)),
                    mean_rmse=float(sub_m["rmse"].mean()),
                    n_folds=int(len(sub_m)),
                ))
        out_csv_subj = NPZ_DIR / "decoding_y1_per_subject.csv"
        pd.DataFrame(rows_subj).to_csv(out_csv_subj, index=False)
        print(f"Per-subject CSV -> {out_csv_subj.name}")


if __name__ == "__main__":
    main()
