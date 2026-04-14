#!/usr/bin/env python
"""Bandpower interpretability — Coeficientes del clasificador Welch.

Entrena LogisticRegression(C=1.0) en TODOS los datos (sin CV) para
extraer los coeficientes β y mapearlos a canal × banda.

El accuracy no se reporta aquí (viene del LORO CV de cada script);
el modelo full-data es solo para interpretabilidad.

Tareas analizadas:
  27b  — pre vs post CHANGE_PHOTO (500ms windows)
  27   — CHANGE_PHOTO vs NO_CHANGE_PHOTO (1000ms window)
         ⚠ caveat: confound temporal (NO_CHANGE = fixation al inicio del run)
  34   — 4 clases: Baseline / ChangeUp / Luminance / ChangeDown (250ms windows)

Outputs → results/validation/interpretability/sub-{sub}/
  39a_coef_heatmap_27b_vs_27.png   heatmaps + correlación β(27b) vs β(27)
  39b_coef_4class.png              4 heatmaps (una por clase)
  39c_top_features.png             barplots top-15 features por tarea
  39_coefs.json                    coeficientes numéricos

Usage
-----
    micromamba run -n campeones python scripts/validation/39_interpretability_bandpower.py --subject 27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch as scipy_welch
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
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
SPECTRAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
N_BANDS = len(SPECTRAL_BANDS)
BAND_NAMES = list(SPECTRAL_BANDS.keys())

# Anatomical channel order for heatmap (frontal → occipital)
CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FCz', 'FC2', 'FC6',
    'FT9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'FT10',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

# 4-class definitions (from script 34)
CLASS_SEGMENTS = {
    0: [(-0.500,  0.000)],
    1: [( 0.000,  0.500)],
    2: [( 0.500,  1.000)],
    3: [( 1.000,  1.500)],
}
CLASS_NAMES = {0: "Baseline", 1: "ChangeUp", 2: "Luminance", 3: "ChangeDown"}
WIN_SIZE_S = 0.250
WIN_STEP_S = 0.050
WIDE_TMIN_34 = -1.5
WIDE_TMAX_34 = 2.0
BASELINE_34  = (-1.5, -1.0)

EVENT_ID_27 = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}
FIXED_C = 1.0
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def bandpower_epoch(data: np.ndarray, sfreq: float) -> np.ndarray:
    """(n_ch, n_times) → (n_ch * n_bands,) bandpower features."""
    n_ch, n_t = data.shape
    band_list = list(SPECTRAL_BANDS.values())
    feats = np.empty(n_ch * N_BANDS)
    idx = 0
    for c in range(n_ch):
        freqs, psd = scipy_welch(data[c], fs=sfreq, nperseg=n_t)
        for flo, fhi in band_list:
            mask = (freqs >= flo) & (freqs <= fhi)
            feats[idx] = np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
            idx += 1
    return feats


def bandpower_array(X: np.ndarray, sfreq: float) -> np.ndarray:
    """(n_epochs, n_ch, n_times) → (n_epochs, n_ch * n_bands)."""
    return np.array([bandpower_epoch(X[i], sfreq) for i in range(X.shape[0])])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_raw_epochs(subject, event_id, tmin, tmax, baseline, crop=None):
    """Generic loader for CHANGE_PHOTO ± NO_CHANGE_PHOTO epochs."""
    all_ep = []
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
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"].isin(event_id)]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        eids = rows["trial_type"].map(event_id).values
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets, eids = onsets[valid], eids[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int), eids])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        run_eid = {k: v for k, v in event_id.items() if k in rows["trial_type"].unique()}
        try:
            ep = mne.Epochs(raw, mne_ev, event_id=run_eid,
                            tmin=tmin, tmax=tmax, picks=avail,
                            baseline=baseline, preload=True, verbose=False)
            if crop:
                ep.crop(*crop)
                ep.baseline = None  # already applied; clear to allow concatenation
            all_ep.append(ep)
        except Exception as e:
            print(f"  run-{run_id}: {e}")
    if not all_ep:
        raise RuntimeError("No epochs loaded")
    epochs = mne.concatenate_epochs(all_ep, verbose=False)
    return epochs.get_data(), epochs.events[:, 2], epochs.info["sfreq"], epochs.ch_names


def load_27b(subject):
    """Pre (label=0) vs post (label=1) from CHANGE_PHOTO onsets."""
    pre_list, post_list = [], []
    sfreq_out, ch_out = None, None
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
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"] == "CHANGE_PHOTO"]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets = onsets[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int),
                                   np.ones(len(onsets), int)])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        try:
            ep = mne.Epochs(raw, mne_ev, event_id={"CHANGE_PHOTO": 1},
                            tmin=-1.5, tmax=1.5, picks=avail,
                            baseline=(-1.5, -1.0), preload=True, verbose=False)
            pre_list.append(ep.copy().crop(-0.55, -0.05).get_data())
            post_list.append(ep.copy().crop(0.05, 0.55).get_data())
            sfreq_out, ch_out = sfreq, ep.ch_names
        except Exception as e:
            print(f"  run-{run_id}: {e}")
    X_pre  = np.concatenate(pre_list,  axis=0)
    X_post = np.concatenate(post_list, axis=0)
    X = np.concatenate([X_pre, X_post], axis=0)
    y = np.concatenate([np.zeros(len(X_pre)), np.ones(len(X_post))])
    return X, y, sfreq_out, ch_out


def load_27(subject):
    """CHANGE_PHOTO (1) vs NO_CHANGE_PHOTO (0), focused window [0.05, 1.05]s."""
    X, y_raw, sfreq, ch = _load_raw_epochs(
        subject, EVENT_ID_27,
        tmin=-2.5, tmax=2.0, baseline=(-2.5, -1.5),
        crop=(0.05, 1.05),
    )
    # remap: CHANGE_PHOTO=1 stays 1, NO_CHANGE_PHOTO=2 → 0
    y = np.where(y_raw == 1, 1, 0)
    return X, y, sfreq, ch


def load_34(subject):
    """4-class sliding windows from CHANGE_PHOTO trials."""
    X_all, y_all = [], []
    sfreq_out, ch_out = None, None

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
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"] == "CHANGE_PHOTO"]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets = onsets[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int),
                                   np.ones(len(onsets), int)])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        try:
            ep = mne.Epochs(raw, mne_ev, event_id={"CHANGE_PHOTO": 1},
                            tmin=WIDE_TMIN_34, tmax=WIDE_TMAX_34, picks=avail,
                            baseline=BASELINE_34, preload=True, verbose=False)
            data_wide = ep.get_data()
            n_win = int(round(WIN_SIZE_S * sfreq))
            for trial in data_wide:
                for label, segs in CLASS_SEGMENTS.items():
                    for seg_start, seg_end in segs:
                        t = seg_start
                        while t + WIN_SIZE_S <= seg_end + 1e-9:
                            s0 = int(round((t - WIDE_TMIN_34) * sfreq))
                            segment = trial[:, s0:s0 + n_win]
                            X_all.append(bandpower_epoch(segment, sfreq))
                            y_all.append(label)
                            t = round(t + WIN_STEP_S, 6)
            sfreq_out, ch_out = sfreq, ep.ch_names
        except Exception as e:
            print(f"  run-{run_id}: {e}")

    return np.array(X_all), np.array(y_all), sfreq_out, ch_out


# ---------------------------------------------------------------------------
# Model training & coefficient extraction
# ---------------------------------------------------------------------------

def fit_and_get_coefs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Train full-data model, return coef array (n_classes_minus_1_or_n_classes, n_features)."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=FIXED_C, max_iter=5000,
                           random_state=RANDOM_SEED, solver="lbfgs"),
    )
    pipe.fit(X, y)
    coef = pipe.named_steps["logisticregression"].coef_
    return coef  # (1, 160) for binary, (4, 160) for 4-class


def coef_to_matrix(coef_1d: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """(n_ch * n_bands,) → (n_ch, n_bands) in original channel order."""
    n_ch = len(ch_names)
    return coef_1d.reshape(n_ch, N_BANDS)


def reorder_channels(mat: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reorder matrix rows to anatomical CHANNEL_ORDER."""
    avail = [ch for ch in CHANNEL_ORDER if ch in ch_names]
    idx = [ch_names.index(ch) for ch in avail]
    return mat[idx], avail


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_heatmap(ax, mat: np.ndarray, ch_names: list[str],
                 title: str, vmax: float | None = None) -> None:
    if vmax is None:
        vmax = np.abs(mat).max()
    vmax = max(vmax, 1e-10)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N_BANDS))
    ax.set_xticklabels(BAND_NAMES, fontsize=8)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Band", fontsize=8)
    ax.set_ylabel("Channel", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="β")


def plot_top_features(ax, coef_1d, ch_names, title, n_top=15):
    bands = BAND_NAMES
    feat_names = [f"{ch}-{b}" for ch in ch_names for b in bands]
    idx_sorted = np.argsort(np.abs(coef_1d))[::-1][:n_top]
    top_names = [feat_names[i] for i in idx_sorted]
    top_vals = coef_1d[idx_sorted]
    colors = ["C3" if v > 0 else "steelblue" for v in top_vals]
    ax.barh(range(n_top), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_names[::-1], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("β coefficient", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & extract features ---
    print("=== Loading 27b (pre vs post) ===")
    X_27b, y_27b, sfreq_27b, ch_27b = load_27b(sub)
    F_27b = bandpower_array(X_27b, sfreq_27b)
    print(f"  X: {F_27b.shape}, classes: {np.bincount(y_27b.astype(int))}")

    print("\n=== Loading 27 (CHANGE vs NO_CHANGE) ===")
    X_27, y_27, sfreq_27, ch_27 = load_27(sub)
    F_27 = bandpower_array(X_27, sfreq_27)
    print(f"  X: {F_27.shape}, classes: {np.bincount(y_27.astype(int))}")

    print("\n=== Loading 34 (4-class) ===")
    F_34, y_34, sfreq_34, ch_34 = load_34(sub)
    print(f"  X: {F_34.shape}, classes: {np.bincount(y_34)}")

    # --- Fit models ---
    print("\nFitting models...")
    coef_27b = fit_and_get_coefs(F_27b, y_27b)[0]   # (160,)
    coef_27  = fit_and_get_coefs(F_27,  y_27)[0]    # (160,)
    coef_34  = fit_and_get_coefs(F_34,  y_34)        # (4, 160)

    # --- Reshape to (n_ch, n_bands) ---
    mat_27b = coef_to_matrix(coef_27b, ch_27b)
    mat_27  = coef_to_matrix(coef_27,  ch_27)
    mat_34  = np.stack([coef_to_matrix(coef_34[c], ch_34)
                        for c in range(4)], axis=0)  # (4, n_ch, n_bands)

    # Reorder channels anatomically
    mat_27b_r, chs_r = reorder_channels(mat_27b, ch_27b)
    mat_27_r,  _     = reorder_channels(mat_27,  ch_27)
    mat_34_r = np.stack([reorder_channels(mat_34[c], ch_34)[0]
                         for c in range(4)], axis=0)

    # --- Save JSON ---
    json_out = {
        "task_27b": {"coef": coef_27b.tolist(), "ch_names": ch_27b,
                     "band_names": BAND_NAMES},
        "task_27":  {"coef": coef_27.tolist(),  "ch_names": ch_27,
                     "band_names": BAND_NAMES},
        "task_34":  {"coef": coef_34.tolist(),  "ch_names": ch_34,
                     "band_names": BAND_NAMES, "class_names": CLASS_NAMES},
    }
    with open(out_dir / f"sub-{sub}_39_coefs.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print("Saved JSON coefs.")

    # ==========================================================
    # Figure A: heatmaps 27b vs 27 + correlation
    # ==========================================================
    vmax_ab = max(np.abs(mat_27b_r).max(), np.abs(mat_27_r).max())
    r, p_corr = pearsonr(coef_27b, coef_27)

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.suptitle(
        f"sub-{sub} — Coeficientes bandpower: 27b vs 27\n"
        f"Correlación β(27b) vs β(27): r={r:.3f}, p={p_corr:.3f}",
        fontsize=12,
    )
    plot_heatmap(axes[0], mat_27b_r, chs_r,
                 "27b — pre vs post\n(CHANGE_PHOTO onsets, 500ms)", vmax_ab)
    plot_heatmap(axes[1], mat_27_r, chs_r,
                 "27 — CHANGE vs NO_CHANGE\n(1000ms window) ⚠ confound", vmax_ab)

    # Correlation scatter
    ax = axes[2]
    ax.scatter(coef_27b, coef_27, alpha=0.4, s=20, c="steelblue")
    lim = max(np.abs(coef_27b).max(), np.abs(coef_27).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("β(27b) — pre vs post", fontsize=9)
    ax.set_ylabel("β(27)  — CHANGE vs NO_CHANGE", fontsize=9)
    ax.set_title(f"Correlación entre tareas\nr={r:.3f}  p={p_corr:.3f}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    out_a = out_dir / f"sub-{sub}_39a_coef_heatmap_27b_vs_27.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_a.name}")

    # ==========================================================
    # Figure B: 4-class heatmaps
    # ==========================================================
    vmax_34 = np.abs(mat_34_r).max()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"sub-{sub} — Coeficientes bandpower: 4 clases (script 34)",
                 fontsize=12)
    for idx, ax in enumerate(axes.flatten()):
        plot_heatmap(ax, mat_34_r[idx], chs_r,
                     CLASS_NAMES[idx], vmax_34)
    plt.tight_layout()
    out_b = out_dir / f"sub-{sub}_39b_coef_4class.png"
    fig.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_b.name}")

    # ==========================================================
    # Figure C: top-15 features
    # ==========================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"sub-{sub} — Top-15 features por magnitud |β|", fontsize=12)
    plot_top_features(axes[0], coef_27b, ch_27b,
                      "27b — pre vs post\n(rojo=predice POST, azul=predice PRE)")
    plot_top_features(axes[1], coef_27, ch_27,
                      "27 — CHANGE vs NO_CHANGE\n(rojo=predice CHANGE, azul=predice NO_CHANGE)")
    plt.tight_layout()
    out_c = out_dir / f"sub-{sub}_39c_top_features.png"
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_c.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
