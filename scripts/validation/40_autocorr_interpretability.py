#!/usr/bin/env python
"""Autocorrelation interpretability — coeficientes del clasificador (task 27b).

Entrena LogisticRegression(C=0.001) en TODOS los datos (sin CV) y mapea
los 800 coeficientes al espacio canal x lag para entender que estructura
temporal discrimina PRE vs POST.

Plots generados:
  40a_coef_heatmap.png       heatmap (canales x lags) de coeficientes
  40b_lag_profile.png        perfil de importancia por lag (ms)
  40c_channel_profile.png    perfil de importancia por canal
  40d_top_features.png       top-15 features por |beta|
  40_coefs.json              coeficientes numericos

Usage
-----
    micromamba run -n campeones python scripts/validation/40_autocorr_interpretability.py --subject 27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT      = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "interpretability"
SESSION = "vr"

RUNS_CONFIG = {
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
# Anatomical order: frontal -> occipital
CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FCz', 'FC2', 'FC6',
    'FT9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'FT10',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

N_LAGS   = 25
FIXED_C  = 0.001   # CV-optimal from 27b LORO
SFREQ_MS = 4.0     # ms per sample at 250 Hz
RANDOM_SEED = 42

# Log-spaced lags: covers 4-100ms at 250Hz with equal log-scale spacing.
# lag 12 (48ms) ≈ beta period (20Hz); lag 25 (100ms) ≈ alpha period (10Hz)
LOG_LAGS = [1, 2, 3, 4, 7, 12, 20, 25]
LOG_LAG_MS = [int(k * SFREQ_MS) for k in LOG_LAGS]
# Approximate target frequency for each lag: f ≈ 1/(2*lag_s) = 500/lag_ms Hz
LOG_LAG_FREQ = [round(500 / ms) for ms in LOG_LAG_MS]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_autocorr(data: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """(n_epochs, n_ch, n_times) -> (n_epochs, n_ch * n_lags).

    R(k) = E[(x_t - mu)(x_{t+k} - mu)] / Var(x), k = 1..n_lags.
    Channel-major ordering: [ch0_lag1,..,ch0_lagN, ch1_lag1,..,chM_lagN].
    """
    n_ep, n_ch, _ = data.shape
    feats = np.empty((n_ep, n_ch * n_lags), dtype=np.float64)
    for i in range(n_ep):
        for ch in range(n_ch):
            x    = data[i, ch]
            x_dm = x - x.mean()
            var  = np.mean(x_dm ** 2)
            if var == 0:
                feats[i, ch * n_lags : (ch + 1) * n_lags] = 0.0
                continue
            for k in range(1, n_lags + 1):
                feats[i, ch * n_lags + (k - 1)] = np.mean(x_dm[:-k] * x_dm[k:]) / var
    return feats


# ---------------------------------------------------------------------------
# Data loader (27b: pre vs post CHANGE_PHOTO)
# ---------------------------------------------------------------------------

def load_27b(subject: str):
    """Returns X (n_epochs, n_ch, n_times), y, sfreq, ch_names."""
    pre_list, post_list = [], []
    sfreq_out = ch_out = None

    for rc in RUNS_CONFIG[subject]:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
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
        df  = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"] == "CHANGE_PHOTO"]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        valid  = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets = onsets[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int),
                                   np.ones(len(onsets),  int)])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        try:
            ep = mne.Epochs(raw, mne_ev, event_id={"CHANGE_PHOTO": 1},
                            tmin=-1.5, tmax=1.5, picks=avail,
                            baseline=(-1.5, -1.0), preload=True, verbose=False)
            pre_list.append(ep.copy().crop(-0.55, -0.05).get_data())
            post_list.append(ep.copy().crop(0.05,  0.55).get_data())
            sfreq_out, ch_out = sfreq, ep.ch_names
        except Exception as e:
            print(f"  run-{run_id}: {e}")

    X_pre  = np.concatenate(pre_list,  axis=0)
    X_post = np.concatenate(post_list, axis=0)
    X = np.concatenate([X_pre, X_post], axis=0)
    y = np.concatenate([np.zeros(len(X_pre)), np.ones(len(X_post))])
    return X, y, sfreq_out, ch_out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _reorder(mat: np.ndarray, ch_names: list[str]):
    """Reorder matrix rows to anatomical order. Returns (mat_r, ch_r)."""
    avail = [ch for ch in CHANNEL_ORDER if ch in ch_names]
    idx   = [list(ch_names).index(ch) for ch in avail]
    return mat[idx], avail


def plot_heatmap(coef_mat: np.ndarray, ch_names_r: list[str],
                 out_path: Path, subject: str) -> None:
    """Plot A: heatmap channels x lags."""
    lag_ms = [int(k * SFREQ_MS) for k in range(1, N_LAGS + 1)]
    vmax   = np.abs(coef_mat).max()

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(coef_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N_LAGS))
    ax.set_xticklabels([f"{ms}" for ms in lag_ms], fontsize=7, rotation=45)
    ax.set_yticks(range(len(ch_names_r)))
    ax.set_yticklabels(ch_names_r, fontsize=7)
    ax.set_xlabel("Lag (ms)", fontsize=9)
    ax.set_ylabel("Canal (frontal → occipital)", fontsize=9)
    ax.set_title(
        f"sub-{subject} — Coeficientes autocorr (27b: pre vs post)\n"
        f"Rojo = predice POST (post-flash), Azul = predice PRE",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="beta")

    # Mark beta (~48ms = lag12) and alpha (~100ms = lag25) with vertical lines
    ax.axvline(11, color="gold",  lw=1.5, ls="--", alpha=0.8, label="~beta (48ms)")
    ax.axvline(24, color="lime",  lw=1.5, ls="--", alpha=0.8, label="~alpha (100ms)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_lag_profile(coef_mat: np.ndarray, out_path: Path, subject: str) -> None:
    """Plot B: mean |coeff| across channels per lag."""
    lag_ms      = np.array([k * SFREQ_MS for k in range(1, N_LAGS + 1)])
    mean_abs    = np.mean(np.abs(coef_mat), axis=0)  # (n_lags,)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lag_ms, mean_abs, "o-", color="steelblue", lw=2, ms=5)
    ax.axvline(48,  color="gold", lw=1.5, ls="--", alpha=0.8, label="~beta (48ms, lag12)")
    ax.axvline(100, color="lime", lw=1.5, ls="--", alpha=0.8, label="~alpha (100ms, lag25)")
    ax.set_xlabel("Lag (ms)", fontsize=10)
    ax.set_ylabel("Mean |beta| across channels", fontsize=10)
    ax.set_title(
        f"sub-{subject} — Importancia temporal por lag\n"
        f"Picos en lags de beta/alpha confirman el mecanismo ERD",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_channel_profile(coef_mat: np.ndarray, ch_names_r: list[str],
                         out_path: Path, subject: str) -> None:
    """Plot C: mean |coeff| across lags per channel."""
    mean_abs = np.mean(np.abs(coef_mat), axis=1)  # (n_ch,)

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["C3" if "O" in ch or "P" in ch else "steelblue"
              for ch in ch_names_r]
    ax.bar(range(len(ch_names_r)), mean_abs, color=colors)
    ax.set_xticks(range(len(ch_names_r)))
    ax.set_xticklabels(ch_names_r, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |beta| across lags", fontsize=9)
    ax.set_title(
        f"sub-{subject} — Importancia espacial por canal\n"
        f"(rojo = occipital/parietal, azul = resto)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_top_features(coef_1d: np.ndarray, ch_names: list[str],
                      out_path: Path, subject: str, n_top: int = 15) -> None:
    """Plot D: top-N features por |beta| con etiqueta canal-lag(ms)."""
    lag_ms = [int(k * SFREQ_MS) for k in range(1, N_LAGS + 1)]
    feat_labels = [f"{ch}-lag{k+1}({lag_ms[k]}ms)"
                   for ch in ch_names for k in range(N_LAGS)]

    idx_sorted = np.argsort(np.abs(coef_1d))[::-1][:n_top]
    top_labels = [feat_labels[i] for i in idx_sorted]
    top_vals   = coef_1d[idx_sorted]
    colors     = ["C3" if v > 0 else "steelblue" for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(n_top), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("beta coefficient", fontsize=9)
    ax.set_title(
        f"sub-{subject} — Top-{n_top} features autocorr por |beta|\n"
        f"Rojo = predice POST, Azul = predice PRE",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def extract_autocorr_log(data: np.ndarray) -> np.ndarray:
    """Autocorrelation at LOG_LAGS indices. Returns (n_epochs, n_ch * 8)."""
    n_ep, n_ch, _ = data.shape
    n_lags = len(LOG_LAGS)
    feats = np.empty((n_ep, n_ch * n_lags), dtype=np.float64)
    for i in range(n_ep):
        for ch in range(n_ch):
            x    = data[i, ch]
            x_dm = x - x.mean()
            var  = np.mean(x_dm ** 2)
            if var == 0:
                feats[i, ch * n_lags : (ch + 1) * n_lags] = 0.0
                continue
            for j, k in enumerate(LOG_LAGS):
                feats[i, ch * n_lags + j] = np.mean(x_dm[:-k] * x_dm[k:]) / var
    return feats


def plot_log_heatmap(coef_mat: np.ndarray, ch_names_r: list[str],
                     out_path: Path, subject: str) -> None:
    """Plot E: heatmap channels x 8 log-spaced lags."""
    xlabels = [f"lag{k}\n{ms}ms\n~{freq}Hz"
               for k, ms, freq in zip(LOG_LAGS, LOG_LAG_MS, LOG_LAG_FREQ)]
    vmax = np.abs(coef_mat).max()

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(coef_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(LOG_LAGS)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(len(ch_names_r)))
    ax.set_yticklabels(ch_names_r, fontsize=7)
    ax.set_xlabel("Lag (log-espaciado)", fontsize=9)
    ax.set_ylabel("Canal (frontal → occipital)", fontsize=9)
    ax.set_title(
        f"sub-{subject} — Coeficientes autocorr LOG-LAGS (27b)\n"
        f"Cada lag es independiente: comparacion directa beta vs alpha",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="beta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_log_lag_comparison(coef_mat: np.ndarray, out_path: Path,
                             subject: str) -> None:
    """Plot F: mean |coeff| por lag (8 barras) — comparacion directa beta vs alpha."""
    mean_abs = np.mean(np.abs(coef_mat), axis=0)  # (8,)
    xlabels  = [f"lag{k}\n{ms}ms\n~{freq}Hz"
                for k, ms, freq in zip(LOG_LAGS, LOG_LAG_MS, LOG_LAG_FREQ)]

    # Color por zona de frecuencia
    colors = []
    for freq in LOG_LAG_FREQ:
        if freq >= 30:
            colors.append("#9B59B6")  # purple = gamma
        elif freq >= 13:
            colors.append("#E74C3C")  # red = beta
        elif freq >= 8:
            colors.append("#2ECC71")  # green = alpha
        else:
            colors.append("#3498DB")  # blue = theta/delta

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(LOG_LAGS)), mean_abs, color=colors, width=0.6, edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(LOG_LAGS)))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean |beta| across channels", fontsize=10)
    ax.set_title(
        f"sub-{subject} — Importancia por lag log-espaciado\n"
        f"Morado=gamma, Rojo=beta, Verde=alpha. "
        f"Comparacion directa sin redundancia entre lags.",
        fontsize=10,
    )

    # Annotate each bar with value
    for bar, val in zip(bars, mean_abs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub  = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("=== Loading 27b (pre vs post CHANGE_PHOTO) ===")
    X, y, sfreq, ch_names = load_27b(sub)
    print(f"  X: {X.shape}  y: {np.bincount(y.astype(int))}")

    # --- Extract features ---
    print(f"\nExtracting autocorr features (n_lags={N_LAGS})...")
    F = extract_autocorr(X, N_LAGS)
    print(f"  Feature matrix: {F.shape}")

    # --- Fit full-data model ---
    print(f"\nFitting LogisticRegression(C={FIXED_C}) on all data...")
    scaler = StandardScaler()
    F_sc   = scaler.fit_transform(F)
    clf    = LogisticRegression(C=FIXED_C, max_iter=5000,
                                random_state=RANDOM_SEED, solver="saga")
    clf.fit(F_sc, y)
    coef_1d = clf.coef_[0]  # (800,)
    print(f"  coef shape: {coef_1d.shape}  |coef|_max: {np.abs(coef_1d).max():.4f}")

    # --- Reshape to (n_ch, n_lags) ---
    n_ch    = len(ch_names)
    coef_mat = coef_1d.reshape(n_ch, N_LAGS)   # (32, 25)

    # Reorder channels anatomically
    coef_mat_r, ch_names_r = _reorder(coef_mat, ch_names)

    # --- Save JSON ---
    json_out = {
        "task": "27b_autocorr",
        "n_lags": N_LAGS,
        "sfreq": sfreq,
        "fixed_C": FIXED_C,
        "ch_names": list(ch_names),
        "lag_ms": [int(k * SFREQ_MS) for k in range(1, N_LAGS + 1)],
        "coef_1d": coef_1d.tolist(),
        "coef_matrix_ch_x_lag": coef_mat.tolist(),
        "mean_abs_per_lag": np.mean(np.abs(coef_mat), axis=0).tolist(),
        "mean_abs_per_channel": np.mean(np.abs(coef_mat), axis=1).tolist(),
    }
    json_path = out_dir / f"sub-{sub}_40_autocorr_coefs.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved JSON: {json_path.name}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_heatmap(coef_mat_r, ch_names_r,
                 out_dir / f"sub-{sub}_40a_autocorr_heatmap.png", sub)
    plot_lag_profile(coef_mat_r,
                     out_dir / f"sub-{sub}_40b_lag_profile.png", sub)
    plot_channel_profile(coef_mat_r, ch_names_r,
                         out_dir / f"sub-{sub}_40c_channel_profile.png", sub)
    plot_top_features(coef_1d, list(ch_names),
                      out_dir / f"sub-{sub}_40d_top_features.png", sub)

    # ------------------------------------------------------------------
    # Log-spaced lags: interpretabilidad directa beta vs alpha
    # ------------------------------------------------------------------
    print(f"\n=== Log-spaced lags {LOG_LAGS} ({len(LOG_LAGS)} lags) ===")
    F_log = extract_autocorr_log(X)
    print(f"  Feature matrix: {F_log.shape}")

    scaler_log = StandardScaler()
    F_log_sc   = scaler_log.fit_transform(F_log)
    clf_log    = LogisticRegression(C=FIXED_C, max_iter=5000,
                                    random_state=RANDOM_SEED, solver="saga")
    clf_log.fit(F_log_sc, y)
    coef_log_1d  = clf_log.coef_[0]   # (256,)
    coef_log_mat = coef_log_1d.reshape(len(ch_names), len(LOG_LAGS))  # (32, 8)
    coef_log_r, _ = _reorder(coef_log_mat, ch_names)

    print(f"  |coef|_max: {np.abs(coef_log_1d).max():.4f}")
    print(f"  Mean |beta| per lag:")
    for k, ms, freq, v in zip(LOG_LAGS, LOG_LAG_MS, LOG_LAG_FREQ,
                               np.mean(np.abs(coef_log_mat), axis=0)):
        print(f"    lag{k:2d} ({ms:3d}ms ~{freq:3d}Hz): {v:.5f}")

    # Save to JSON
    json_out["log_lags"] = {
        "lags": LOG_LAGS,
        "lag_ms": LOG_LAG_MS,
        "lag_approx_freq_hz": LOG_LAG_FREQ,
        "coef_1d": coef_log_1d.tolist(),
        "mean_abs_per_lag": np.mean(np.abs(coef_log_mat), axis=0).tolist(),
        "mean_abs_per_channel": np.mean(np.abs(coef_log_mat), axis=1).tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    plot_log_heatmap(coef_log_r, ch_names_r,
                     out_dir / f"sub-{sub}_40e_log_heatmap.png", sub)
    plot_log_lag_comparison(coef_log_r,
                            out_dir / f"sub-{sub}_40f_log_lag_comparison.png", sub)

    print("\nDone.")


if __name__ == "__main__":
    main()
