"""3-model decoding of SCR onset (real) vs silent-EDA (control) -- analog to the
luminance decoding work (10/11/13 luminance scripts) but adapted to classification.

Models (each is a feature-extractor + LR classifier with L2):

  1. base_raw     -- Raw EEG epochs vectorized.
                     Pipeline: Vectorizer -> StandardScaler -> PCA(100) -> LR.
                     Mirrors `10_luminance_base_model.py` (with LR instead of Ridge).

  2. spectral     -- Log-band-power per (channel, band).
                     Bands: delta theta alpha beta gamma.
                     Pipeline: StandardScaler -> PCA(50) -> LR.
                     Mirrors `11_luminance_spectral_model.py`.

  3. raw_tde      -- Per-epoch Time-Delay Embedding (lags +/- 10) -> PCA(20)
                     fit on training-fold TDE pool -> per-epoch covariance triu
                     (210 features). Pipeline: StandardScaler -> LR.
                     Mirrors `13_luminance_raw_tde_model.py` (per-epoch variant
                     instead of continuous-segment variant, because our events
                     are discrete SCRs).

CV: Leave-One-Run-Out per subject (8 folds per subject; 7 runs train, 1 run test).
Channels: all 32 EEG.
Metrics per fold: accuracy, AUC-ROC, F1.

Outputs (research_diary/context/05_02/figures/decoding_3models/):
  Y3_decoding_3models_<sub>.png      -- per-subject bar chart of CV scores per model
  Y3_decoding_3models_summary.png    -- grand-average bar chart across subjects

CSVs:
  decoding_3models_per_fold.csv      -- subject, run_held_out, model, acc, auc, f1, n_train, n_test
  decoding_3models_per_subject.csv   -- subject, model, mean_acc, std_acc, mean_auc, std_auc

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_scr_3models
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.erp_scr import (
    BASELINE,
    EDA_FS,
    NPZ_DIR,
    OUT,
    SUBJECTS,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "decoding_3models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (mirror the luminance setup)
PCA_BASE = 100
PCA_SPECTRAL = 50
PCA_TDE = 20
TDE_LAG = 10
LR_C = 1.0
LR_SOLVER = "liblinear"
LR_MAX_ITER = 2000
RNG_SEED = 42

# spectral bands (same as throughout the pipeline)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
PSD_NFFT = 512


# -----------------------------------------------------------------------------
# Per-run epoch construction (keeps run labels for LORO grouping)
# -----------------------------------------------------------------------------
def build_per_run_epochs(sub: str) -> dict[str, dict] | None:
    """Return {run_label: {"real": Epochs, "silent": Epochs}} per subject."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])

    rng = np.random.default_rng(RNG_SEED)
    per_run: dict[str, dict] = {}
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.resample(250.0, verbose="ERROR")
            duration = float(raw.times[-1])

            eda_phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets_all = detect_scr_onsets_s(eda_phasic, EDA_FS)
            onsets_all = onsets_all[onsets_all < duration]
            onsets_clean = filter_clean_onsets(onsets_all, eda_phasic, EDA_FS)
            silent_t = sample_silent_controls(
                n_target=len(onsets_clean), duration_s=duration,
                phasic=eda_phasic, fs=EDA_FS, rng=rng,
            )
            ep_real = epoch_one_run(raw, onsets_clean, code=1, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            ep_silent = epoch_one_run(raw, silent_t, code=2, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            if ep_real is None or ep_silent is None:
                continue
            if len(ep_real) < 1 or len(ep_silent) < 1:
                continue
            per_run[label] = {"real": ep_real, "silent": ep_silent}
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")
    return per_run if per_run else None


def assemble_X_y_groups(per_run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Concatenate per-run epochs into (X, y, groups) for LORO.

    X shape: (n_total, n_channels, n_times).  y: 1=real, 0=silent.  groups: run_label per epoch.
    Returns also (ch_names, sfreq).
    """
    X_list, y_list, g_list = [], [], []
    sfreq = None
    ch_names = None
    for run_lab, eps in per_run.items():
        for cls_label, eps_obj in (("real", eps["real"]), ("silent", eps["silent"])):
            data = eps_obj.get_data()
            X_list.append(data)
            y_list.append(np.full(len(eps_obj), 1 if cls_label == "real" else 0, dtype=int))
            g_list.append(np.array([run_lab] * len(eps_obj)))
            if sfreq is None:
                sfreq = float(eps_obj.info["sfreq"])
                ch_names = list(eps_obj.ch_names)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    return X, y, groups, ch_names, sfreq


# -----------------------------------------------------------------------------
# Feature extractors
# -----------------------------------------------------------------------------
def compute_band_power_features(X: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-epoch log-band-power. Returns (n_epochs, n_channels * n_bands).

    Uses MNE PSD (Welch). Faster than per-channel scipy.welch loops.
    """
    info = mne.create_info(ch_names=[f"ch{i}" for i in range(X.shape[1])], sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, verbose="ERROR")
    spectrum = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0, n_fft=PSD_NFFT, verbose="ERROR")
    data = spectrum.get_data()  # (n_epochs, n_channels, n_freqs)
    freqs = spectrum.freqs
    n_epochs, n_channels, _ = data.shape
    feats = np.zeros((n_epochs, n_channels * len(BANDS)))
    for i, (lo, hi) in enumerate(BANDS.values()):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            band_pow = data[:, :, m].mean(axis=2)  # (n_epochs, n_channels)
            feats[:, i::len(BANDS)] = 10.0 * np.log10(band_pow + 1e-30)
    return feats


def apply_tde(epoch_data: np.ndarray, lag: int = TDE_LAG) -> np.ndarray:
    """Per-epoch TDE.

    epoch_data: (n_channels, n_times). Returns (n_channels * (2*lag + 1), n_times - 2*lag).
    For each valid sample t in [lag, n_times - lag), stack lags [t-lag, ..., t+lag] across channels.
    """
    n_channels, n_times = epoch_data.shape
    n_valid = n_times - 2 * lag
    if n_valid <= 0:
        raise ValueError(f"epoch has {n_times} samples but TDE needs > {2 * lag}")
    n_feat = n_channels * (2 * lag + 1)
    out = np.empty((n_feat, n_valid), dtype=np.float64)
    for shift_idx, shift in enumerate(range(-lag, lag + 1)):
        # rows for this lag: shift_idx * n_channels .. (shift_idx + 1) * n_channels
        # times: from lag + shift to lag + shift + n_valid
        t_start = lag + shift
        out[shift_idx * n_channels:(shift_idx + 1) * n_channels, :] = epoch_data[:, t_start:t_start + n_valid]
    return out


def tde_pipeline_fit_transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TDE + global PCA (fit on training) + per-epoch covariance triu.

    Returns (Xf_train, Xf_test) where Xf shape: (n_epochs, n_pca * (n_pca + 1) // 2).
    """
    # 1. per-epoch TDE
    tde_train = [apply_tde(X_train[i]) for i in range(len(X_train))]   # list of (n_feat, n_valid)
    tde_test = [apply_tde(X_test[i]) for i in range(len(X_test))]

    # 2. pool training TDE across epochs (rows = samples across time and epochs; cols = features)
    pool = np.concatenate([t.T for t in tde_train], axis=0)  # (n_total_samples, n_feat)
    scaler = StandardScaler().fit(pool)
    pca = PCA(n_components=min(PCA_TDE, pool.shape[1], pool.shape[0] - 1), random_state=RNG_SEED).fit(scaler.transform(pool))

    # 3. transform each epoch to PCA space then compute covariance triu
    def to_features(tde_list: list[np.ndarray]) -> np.ndarray:
        feats = []
        for tde_ep in tde_list:
            ep = scaler.transform(tde_ep.T)         # (n_valid, n_feat)
            ep_pca = pca.transform(ep)               # (n_valid, n_pca)
            cov = np.cov(ep_pca.T)                   # (n_pca, n_pca)
            cov = np.atleast_2d(cov)
            triu = cov[np.triu_indices(cov.shape[0])]
            feats.append(triu)
        return np.array(feats)

    return to_features(tde_train), to_features(tde_test)


# -----------------------------------------------------------------------------
# Per-fold model fitting
# -----------------------------------------------------------------------------
def fit_eval_base(X_train, y_train, X_test, y_test) -> dict:
    n_pca = min(PCA_BASE, X_train.shape[0] - 1)
    pipe = make_pipeline(
        Vectorizer(),
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=n_pca, random_state=RNG_SEED),
        LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver=LR_SOLVER),
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return _score(y_test, y_pred, y_proba)


def fit_eval_spectral(X_train_feat, y_train, X_test_feat, y_test) -> dict:
    n_pca = min(PCA_SPECTRAL, X_train_feat.shape[0] - 1, X_train_feat.shape[1])
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_pca, random_state=RNG_SEED),
        LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver=LR_SOLVER),
    )
    pipe.fit(X_train_feat, y_train)
    y_pred = pipe.predict(X_test_feat)
    y_proba = pipe.predict_proba(X_test_feat)[:, 1]
    return _score(y_test, y_pred, y_proba)


def fit_eval_tde(X_train, y_train, X_test, y_test) -> dict:
    Xf_train, Xf_test = tde_pipeline_fit_transform(X_train, X_test)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver=LR_SOLVER),
    )
    pipe.fit(Xf_train, y_train)
    y_pred = pipe.predict(Xf_test)
    y_proba = pipe.predict_proba(Xf_test)[:, 1]
    return _score(y_test, y_pred, y_proba)


def _score(y_true, y_pred, y_proba) -> dict:
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = np.nan
    return dict(
        acc=float(accuracy_score(y_true, y_pred)),
        auc=auc,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        n_test=len(y_true),
    )


# -----------------------------------------------------------------------------
# Per-subject LORO loop
# -----------------------------------------------------------------------------
def loro_subject(sub: str) -> list[dict]:
    print(f"\n=== {sub} ===")
    per_run = build_per_run_epochs(sub)
    if per_run is None:
        print("  no per-run epochs")
        return []
    X, y, groups, ch_names, sfreq = assemble_X_y_groups(per_run)
    print(f"  X={X.shape}  y mean={y.mean():.2f}  groups: {len(np.unique(groups))} runs")

    # Pre-compute spectral features once (they don't depend on the fold)
    print("  precomputing spectral features ...")
    X_spec = compute_band_power_features(X, sfreq)
    print(f"  X_spec={X_spec.shape}")

    rows = []
    unique_runs = list(np.unique(groups))
    for run_held in unique_runs:
        test_mask = groups == run_held
        train_mask = ~test_mask
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_test < 2 or len(np.unique(y[test_mask])) < 2:
            print(f"  test run {run_held}: too few or single-class -- skipping fold")
            continue
        if len(np.unique(y[train_mask])) < 2:
            print(f"  test run {run_held}: training is single-class -- skipping fold")
            continue

        print(f"  fold test={run_held}  n_train={n_train}  n_test={n_test} ...")
        # base
        r_base = fit_eval_base(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        # spectral
        r_spec = fit_eval_spectral(X_spec[train_mask], y[train_mask], X_spec[test_mask], y[test_mask])
        # tde
        r_tde = fit_eval_tde(X[train_mask], y[train_mask], X[test_mask], y[test_mask])

        for model_name, r in (("base_raw", r_base), ("spectral", r_spec), ("raw_tde", r_tde)):
            rows.append(dict(
                subject=sub, run_held_out=run_held, model=model_name,
                acc=r["acc"], auc=r["auc"], f1=r["f1"], n_train=n_train, n_test=n_test,
            ))
        print(f"    base acc={r_base['acc']:.3f} auc={r_base['auc']:.3f}  "
              f"spec acc={r_spec['acc']:.3f} auc={r_spec['auc']:.3f}  "
              f"tde acc={r_tde['acc']:.3f} auc={r_tde['auc']:.3f}")

    return rows


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
MODEL_ORDER = ["base_raw", "spectral", "raw_tde"]
MODEL_COLOR = {"base_raw": "C0", "spectral": "C2", "raw_tde": "C3"}


def plot_subject_3models(sub: str, per_fold_rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(per_fold_rows)
    df_sub = df[df["subject"] == sub]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("acc", "auc")):
        x = np.arange(len(MODEL_ORDER))
        means = [df_sub[df_sub["model"] == m][metric].mean() for m in MODEL_ORDER]
        stds = [df_sub[df_sub["model"] == m][metric].std(ddof=1) for m in MODEL_ORDER]
        bars = ax.bar(x, means, yerr=stds, color=[MODEL_COLOR[m] for m in MODEL_ORDER],
                       alpha=0.85, edgecolor="black", capsize=4)
        # individual fold dots
        for i, m in enumerate(MODEL_ORDER):
            vals = df_sub[df_sub["model"] == m][metric].values
            ax.scatter(np.full(len(vals), i) + 0.0, vals, color="black", s=18, zorder=3, alpha=0.55)
        ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + 0.03, f"{m*100:.1f}%", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} -- LORO across {df_sub['run_held_out'].nunique()} runs")
    fig.suptitle(f"{sub}  --  3-model decoding (LORO CV, dots = individual folds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_summary(all_rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(all_rows)
    subs = sorted(df["subject"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("acc", "auc")):
        x = np.arange(len(MODEL_ORDER))
        width = 0.22
        for i, sub in enumerate(subs):
            df_sub = df[df["subject"] == sub]
            means = [df_sub[df_sub["model"] == m][metric].mean() for m in MODEL_ORDER]
            stds = [df_sub[df_sub["model"] == m][metric].std(ddof=1) for m in MODEL_ORDER]
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=sub, capsize=3,
                    color=f"C{i}", alpha=0.85, edgecolor="black")
        # grand-average horizontal markers per model
        for j, m in enumerate(MODEL_ORDER):
            ga = df[df["model"] == m][metric].mean()
            ax.hlines(ga, j - 0.45, j + 0.45, colors="black", lw=2.2, linestyle="-")
            ax.text(j, ga + 0.025, f"GA={ga*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="0.4", lw=0.8, ls="--", label="chance (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} per subject and model")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("3-model decoding summary  (LORO per subject, bars = mean ± SD across folds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(f"decoding_scr_3models  ::  output -> {FIG_DIR}")
    print("=" * 78)

    all_rows = []
    for sub in SUBJECTS:
        rows = loro_subject(sub)
        all_rows.extend(rows)
        if rows:
            out_png = FIG_DIR / f"Y3_decoding_3models_{sub}.png"
            plot_subject_3models(sub, rows, out_png)
            print(f"  -> {out_png.name}")

    if all_rows:
        out_png = FIG_DIR / "Y3_decoding_3models_summary.png"
        plot_summary(all_rows, out_png)
        print(f"\nSummary -> {out_png.name}")

        df_fold = pd.DataFrame(all_rows)
        df_fold.to_csv(NPZ_DIR / "decoding_3models_per_fold.csv", index=False)
        print(f"Per-fold CSV -> decoding_3models_per_fold.csv  ({len(df_fold)} rows)")

        rows_subj = []
        for sub in df_fold["subject"].unique():
            for m in MODEL_ORDER:
                sub_m = df_fold[(df_fold["subject"] == sub) & (df_fold["model"] == m)]
                rows_subj.append(dict(
                    subject=sub, model=m,
                    mean_acc=float(sub_m["acc"].mean()), std_acc=float(sub_m["acc"].std(ddof=1)),
                    mean_auc=float(sub_m["auc"].mean()), std_auc=float(sub_m["auc"].std(ddof=1)),
                    mean_f1=float(sub_m["f1"].mean()), std_f1=float(sub_m["f1"].std(ddof=1)),
                    n_folds=int(len(sub_m)),
                ))
        pd.DataFrame(rows_subj).to_csv(NPZ_DIR / "decoding_3models_per_subject.csv", index=False)
        print(f"Per-subject CSV -> decoding_3models_per_subject.csv")


if __name__ == "__main__":
    main()
