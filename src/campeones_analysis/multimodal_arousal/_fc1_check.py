"""Diagnostico ad-hoc (GATE E follow-up, observación del usuario): ¿FC1 es un canal malo?
(a) ¿el preproc lo marcó/interpoló? -> parsea logs_preprocessing_details_all_subjects_eeg.json
(b) ¿es anómalo en potencia en el cache post-preproc? -> stats por canal desde panel_psd.npz
NO es parte del pipeline.

Run: micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._fc1_check
"""
from __future__ import annotations

import json
import warnings
from collections import Counter

import numpy as np

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO
from src.campeones_analysis.multimodal_arousal import decoding_panel as d

warnings.filterwarnings("ignore")

LOG = REPO / "data" / "derivatives" / "campeones_preproc" / "logs_preprocessing_details_all_subjects_eeg.json"
TARGET = "FC1"


def _walk_find(obj, key_names, out):
    """Recursively collect lists stored under any key in key_names."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in key_names and isinstance(v, list):
                out.append((k, v))
            _walk_find(v, key_names, out)
    elif isinstance(obj, list):
        for it in obj:
            _walk_find(it, key_names, out)


def check_logs():
    print("=" * 70)
    print("(a) PREPROC LOGS — ¿FC1 marcado/interpolado?")
    print("=" * 70)
    if not LOG.exists():
        print(f"  log no encontrado: {LOG}"); return
    data = json.loads(LOG.read_text(encoding="utf-8"))
    # Try a per-subject structure first; fall back to a recursive walk.
    found = []
    _walk_find(data, {"bad_channels", "interpolated_channels"}, found)
    n_runs_with_bads = 0
    bad_counter = Counter()
    interp_counter = Counter()
    fc1_bad = 0
    fc1_interp = 0
    n_bad_lists = 0
    n_interp_lists = 0
    for key, lst in found:
        if key == "bad_channels":
            n_bad_lists += 1
            if lst:
                n_runs_with_bads += 1
            for c in lst:
                bad_counter[c] += 1
            if TARGET in lst:
                fc1_bad += 1
        else:
            n_interp_lists += 1
            for c in lst:
                interp_counter[c] += 1
            if TARGET in lst:
                fc1_interp += 1
    print(f"  listas bad_channels encontradas: {n_bad_lists}  (con ≥1 bad: {n_runs_with_bads})")
    print(f"  listas interpolated_channels:    {n_interp_lists}")
    print(f"\n  FC1 en bad_channels:         {fc1_bad} / {n_bad_lists} runs")
    print(f"  FC1 en interpolated_channels: {fc1_interp} / {n_interp_lists} runs")
    print(f"\n  Top-10 canales MÁS marcados bad (sobre {n_bad_lists} runs):")
    for c, n in bad_counter.most_common(10):
        print(f"    {c:6s} {n:3d} runs ({n / max(1, n_bad_lists) * 100:.0f}%)")
    print(f"\n  FC1 ranking entre los bad: posición "
          f"{[c for c, _ in bad_counter.most_common()].index(TARGET) + 1 if TARGET in bad_counter else 'NO aparece'}")


def check_cache():
    print("\n" + "=" * 70)
    print("(b) CACHE POST-PREPROC — ¿FC1 anómalo en potencia? (panel_psd.npz)")
    print("=" * 70)
    data, freqs, ch = d.load_cache("uniform")
    ch = list(ch)
    fc1 = ch.index(TARGET)
    print(f"  canal {TARGET} = índice {fc1} de {len(ch)}")
    print(f"\n  {'sub':8s} {'FC1 logP':>9s} {'z(FC1)':>7s} {'rank|z|':>8s} {'corr FC1~vecinos':>16s}")
    # neighbors of FC1 (fronto-central left): FCz, Fz, C3, Cz, FC5, F3, CP1
    neigh = [c for c in ["FCz", "Fz", "Cz", "C3", "FC5", "F3", "CP1", "FC2"] if c in ch]
    nidx = [ch.index(c) for c in neigh]
    for s in COHORT:
        if s not in data:
            continue
        psd = data[s][0]                       # (n_ep, n_ch, n_freq)
        logp = np.log10(psd.mean(axis=2) + 1e-30)   # (n_ep, n_ch) mean power per epoch per ch
        chmean = logp.mean(axis=0)                  # (n_ch,) avg over epochs
        z = (chmean - chmean.mean()) / chmean.std()
        rank = int(np.sum(np.abs(z) >= abs(z[fc1])))  # 1 = most extreme
        # correlation of FC1 power-over-epochs with mean-of-neighbors power-over-epochs
        fc1_series = logp[:, fc1]
        neigh_series = logp[:, nidx].mean(axis=1)
        r = np.corrcoef(fc1_series, neigh_series)[0, 1]
        flag = "  <-- |z|>2 OUTLIER" if abs(z[fc1]) > 2 else ""
        print(f"  {s:8s} {chmean[fc1]:9.2f} {z[fc1]:+7.2f} {rank:4d}/{len(ch):<3d} {r:16.2f}{flag}")
    print("\n  Lectura: z|FC1| alto + rank bajo = potencia anómala (no interpolado o mal interpolado).")
    print("           corr FC1~vecinos ALTA (~>0.9) = consistente/interpolado (smooth, sin info propia).")
    print("           corr BAJA + z alto = FC1 desacoplado de vecinos = canal genuinamente roto.")


if __name__ == "__main__":
    check_logs()
    check_cache()
