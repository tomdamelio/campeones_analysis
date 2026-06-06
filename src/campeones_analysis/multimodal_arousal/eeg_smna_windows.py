"""Construye el dataset por-ventana EEG-banda x SMNA-AUC para evaluar acoplamiento EEG<->EDA.

Reusa la maquinaria probada de lag_sweep_3tasks/decoding_y1_3models:
  - build_subject_dataset(sub, lag) -> ventanas de 2 s: X (n_win, n_ch, 500), y = SMNA AUC (y1s),
    groups (run), ch_names, sfreq.
  - compute_band_power_features_local(X, sfreq) -> log band-power por canal (5 bandas, canal-major).
Agrega a ROIs {PO, central, frontal, edge, all} para {delta, alpha, gamma}. Guarda por sujeto:
  y (SMNA AUC), run (label), tnorm (posición temporal normalizada dentro del run, 0-1),
  bp__{band}__{roi} (n_win,). Esto es la parte PESADA (carga raw + filtra) -> correr una vez,
  secuencial. Las lentes del workflow después leen el cache (liviano).

Cache -> {sub}_eegsmna_win.npz en NPZ_DIR (cohort6/y_candidates).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eeg_smna_windows
  ... --subjects sub-19   (smoke)
"""

from __future__ import annotations

import argparse
import warnings

import mne
import numpy as np

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR
from src.campeones_analysis.multimodal_arousal.lag_sweep_3tasks import build_subject_dataset
from src.campeones_analysis.multimodal_arousal.decoding_y1_3models import (
    BANDS,
    compute_band_power_features_local,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

ROIS = {
    "PO": ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"],
    "central": ["C3", "Cz", "C4", "CP1", "CP2"],
    "frontal": ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
    "edge": ["FT9", "TP9", "T7", "T8", "P7", "P8"],   # EMG-ish (control)
    "all": None,                                       # all channels mean
}
USE_BANDS = ["delta", "alpha", "gamma"]
BAND_NAMES = list(BANDS)  # order from the extractor (delta,theta,alpha,beta,gamma)
BAND_IDX = {b: BAND_NAMES.index(b) for b in USE_BANDS}


def _tnorm(groups):
    """Within-run normalized position 0..1 (proxy de tiempo-en-sesión)."""
    out = np.zeros(len(groups))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n = len(idx)
        out[idx] = np.arange(n) / max(1, n - 1)
    return out


def build_subject(sub: str) -> dict | None:
    ds = build_subject_dataset(sub, 0.0)  # lag 0 = ventana EEG sincronica con la ventana SMNA
    if ds is None:
        return None
    X, y, groups, ch = ds["X"], ds["y"], ds["groups"], ds["ch_names"]
    feats = compute_band_power_features_local(X, ds["sfreq"])  # (n_win, n_ch*n_bands), canal-major
    n_win = feats.shape[0]
    n_ch = len(ch)
    feats = feats.reshape(n_win, n_ch, len(BAND_NAMES))  # (n_win, n_ch, band)
    out = {"y": y.astype(float), "run": groups.astype(str), "tnorm": _tnorm(groups), "ch": ch}
    for band in USE_BANDS:
        bi = BAND_IDX[band]
        for roi, chs in ROIS.items():
            idx = list(range(n_ch)) if chs is None else [ch.index(c) for c in chs if c in ch]
            out[f"bp__{band}__{roi}"] = feats[:, idx, bi].mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()
    subs = args.subjects if args.subjects else list(COHORT)
    print("=" * 78)
    print(f"eeg_smna_windows :: bandas={USE_BANDS} ROIs={list(ROIS)}  subjects={subs}")
    print("=" * 78, flush=True)
    for sub in subs:
        print(f"\n=== {sub} ===", flush=True)
        d = build_subject(sub)
        if d is None:
            print(f"  {sub}: no data", flush=True)
            continue
        path = NPZ_DIR / f"{sub}_eegsmna_win.npz"
        np.savez_compressed(path, **{k: v for k, v in d.items() if k != "ch"},
                            ch=np.array(d["ch"]))
        n = len(d["y"])
        r = np.corrcoef(d["bp__delta__PO"], d["y"])[0, 1]
        print(f"  {sub}: n_win={n}  runs={len(np.unique(d['run']))}  "
              f"delta-PO~y pearson={r:+.3f}  -> {path.name}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
