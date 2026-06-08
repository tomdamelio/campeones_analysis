"""2.6-A — Blindaje espectral del delta (sin gamma + corte ≤2 Hz + topografía + offset-indep).

Frente-DELTA. Tras 2.5 (delta sobrevive el deconfound LINEAL contra movimiento/blink/EOG, pero es
el más golpeado: 81%→57%, y la vulnerabilidad es movimiento 0.5-2 Hz +360% = banda scalp-sweat),
2.6 aporta la evidencia CONVERGENTE no-regresión de que el delta surviving es neural:

  (1) Corte ≤2 Hz: ¿el δ↑ sobrevive con delta=2-4 Hz (sin los bins 1-2 donde viven sweat+movimiento)?
      Métrica = potencia periódica channel-mean (F12, NO flattened-por-ROI). Signo 6/6.
  (2) Sin gamma: efecto por banda (delta/theta/alfa/beta) con el 1/f ajustado en 1-30 (excluye
      gamma) → δ↑ y desync θαβ viven sub-gamma. (El decoding 1-30≈1-40 ya está en 1.4.)
  (3) Topografía del delta PERIÓDICO (1/f removido): ¿PO/posterior (cortical) o edge (músculo/mov)?
      Resuelve la tensión 2.4-crudo(algo edge) vs 1.3-periódico(PO).
  (4) Independencia del offset: corr(d_offset, d_delta) entre sujetos (eco de 1.2 corr≈−0.43):
      el δ↑ NO es el broadband/offset filtrándose.

Todo desde el cache panel_psd (29 ch, esquema uniforme) → liviano, sin permutaciones. Hereda Track B
(EEG-CAR). La ventana pre-onset (2.6-B) va en script aparte (necesita re-epochar).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.delta_robustness_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.campeones_analysis.multimodal_arousal.cohort import CENTRAL, COHORT, EMG_EDGE, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import _make_info, _topo, load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic

warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_6_delta_robustness"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

RANGE = "1-30"   # 1/f ajustado SIN gamma (el punto de Enzo)
BANDS = {"delta_1_4": (1, 4), "delta_2_4": (2, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]   # PO/cortical, no-edge


def feat_band(psd, freqs, band):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES[RANGE])
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)   # (n_ep, n_ch)


def feat_offset(psd, freqs):
    off, _, _, _ = _linear_aperiodic(psd, freqs, RANGES[RANGE])
    return off


def _region(dch, ch, region):
    idx = [ch.index(c) for c in region if c in ch]
    return float(np.mean(dch[idx])) if idx else np.nan


def main():
    print("=" * 78)
    print("delta_robustness_scr :: 2.6-A blindaje espectral del delta (cache, sin gamma)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    per_band = {b: [] for b in BANDS}          # d channel-mean por sujeto
    d_offset = []
    dch14, dch24 = [], []                       # per-channel d (delta 1-4 / 2-4) por sujeto

    for sub in COHORT:
        psd, y, tn = cache[sub]
        for b, band in BANDS.items():
            cm = feat_band(psd, freqs, band).mean(axis=1)        # channel-mean por época
            per_band[b].append(float(cm[y == 1].mean() - cm[y == 0].mean()))
        off = feat_offset(psd, freqs).mean(axis=1)
        d_offset.append(float(off[y == 1].mean() - off[y == 0].mean()))
        f14 = feat_band(psd, freqs, (1, 4)); dch14.append(f14[y == 1].mean(0) - f14[y == 0].mean(0))
        f24 = feat_band(psd, freqs, (2, 4)); dch24.append(f24[y == 1].mean(0) - f24[y == 0].mean(0))

    # --- tabla por banda: signo + media ---
    rows = []
    for b in BANDS:
        v = np.array(per_band[b])
        rows.append(dict(banda=b, media_d=round(float(v.mean()), 4),
                         n_pos=f"{int((v > 0).sum())}/6", n_neg=f"{int((v < 0).sum())}/6",
                         por_sujeto=", ".join(f"{x:+.3f}" for x in v)))
    band_df = pd.DataFrame(rows)
    band_df.to_csv(TBL_DIR / "delta_byband.csv", index=False)

    # --- topografía del delta periódico (GA) + índices edge/central/posterior ---
    ga14, ga24 = np.mean(dch14, axis=0), np.mean(dch24, axis=0)
    topo_rows = []
    for nm, ga in [("delta_1_4", ga14), ("delta_2_4", ga24)]:
        e, c, p = _region(ga, ch, EMG_EDGE), _region(ga, ch, CENTRAL), _region(ga, ch, POSTERIOR)
        topo_rows.append(dict(banda=nm, edge=round(e, 4), central=round(c, 4), posterior=round(p, 4),
                              posterior_minus_edge=round(p - e, 4)))
    topo_df = pd.DataFrame(topo_rows)
    topo_df.to_csv(TBL_DIR / "delta_topography.csv", index=False)

    # --- independencia del offset (corr entre sujetos) ---
    d_off = np.array(d_offset)
    r14 = spearmanr(d_off, per_band["delta_1_4"]).correlation
    r24 = spearmanr(d_off, per_band["delta_2_4"]).correlation

    print("\n=== (1)+(2) Efecto por banda (channel-mean periódico, 1/f sin gamma) ===")
    print(band_df.to_string(index=False))
    print("\n=== (3) Topografía del delta periódico (GA): edge vs central vs posterior ===")
    print(topo_df.to_string(index=False))
    print(f"\n=== (4) Independencia del offset (corr entre sujetos) ===")
    print(f"  corr(d_offset, d_delta_1_4) = {r14:+.2f}   corr(d_offset, d_delta_2_4) = {r24:+.2f}")
    print("  (negativo = el delta NO es el offset/broadband filtrándose; eco de 1.2 ≈ −0.43)")

    # --- veredicto del corte ≤2 Hz ---
    d14, d24 = np.array(per_band["delta_1_4"]), np.array(per_band["delta_2_4"])
    print(f"\n=== VEREDICTO corte ≤2 Hz ===")
    print(f"  delta 1-4: {int((d14>0).sum())}/6 positivos (media {d14.mean():+.3f})")
    print(f"  delta 2-4: {int((d24>0).sum())}/6 positivos (media {d24.mean():+.3f}) "
          f"<- ¿sobrevive sin la banda 1-2 Hz (sweat/movimiento)?")

    _plot_band(per_band)
    _plot_topo(ga14, ga24, ch)
    _plot_offset_indep(d_off, d14, r14)
    print(f"\nTablas/figuras -> {OUT_DIR}\n[2.6-A] done", flush=True)


def _plot_band(per_band):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(BANDS))
    for k, sub in enumerate(COHORT):
        ax.plot(x + np.linspace(-0.2, 0.2, len(COHORT))[k],
                [per_band[b][k] for b in BANDS], "o", color=SUBJ_COLORS[sub], ms=6, label=sub)
    ax.plot(x, [np.mean(per_band[b]) for b in BANDS], "_", color="k", ms=28, mew=2.5, label="media")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(list(BANDS.keys()), fontsize=10)
    ax.set_ylabel("SCR − no-SCR (potencia periódica channel-mean)")
    ax.set_title("2.6-A Efecto por banda (1/f sin gamma): delta↑ sobrevive el corte ≤2 Hz · θαβ↓ desync",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(FIG_DIR / "delta_byband.png", dpi=120); plt.close(fig)


def _plot_topo(ga14, ga24, ch):
    info = _make_info(ch)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    _topo(axes[0], ga14, info, "delta 1-4 Hz periódico\n(SCR − no-SCR)")
    _topo(axes[1], ga24, info, "delta 2-4 Hz periódico\n(corte ≤2 Hz)")
    fig.suptitle("2.6-A Topografía del delta PERIÓDICO: ¿PO/posterior (cortical) o edge (músculo)?",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG_DIR / "delta_topo.png", dpi=120); plt.close(fig)


def _plot_offset_indep(d_off, d14, r14):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for k, sub in enumerate(COHORT):
        ax.scatter(d_off[k], d14[k], color=SUBJ_COLORS[sub], s=70, label=sub)
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("d_offset (SCR − no-SCR)"); ax.set_ylabel("d_delta_1_4 (SCR − no-SCR)")
    ax.set_title(f"2.6-A Independencia del offset: corr = {r14:+.2f}\n"
                 "(negativo = delta NO es el offset filtrándose)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "delta_offset_indep.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
