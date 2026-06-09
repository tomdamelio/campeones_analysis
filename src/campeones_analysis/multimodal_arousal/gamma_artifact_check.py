"""Gamma artifact deep-check (lectura escéptica, 2026-06-08).

Tras la observación del usuario sobre el topomap GA de gamma ("veo bastante artefacto, no lo veo
difuso"), este script formaliza la sospecha mirando POR SUJETO (no solo el GA, que promedia
heterogeneidad):

  (a) topomap per-sujeto del gamma 30-40 CRUDO vs PERIÓDICO (1/f removido), SCR−no-SCR.
      Si el crudo es edge/temporal (EMG broadband) y el periódico se va o cambia -> el edge es
      aperiódico (EMG), consistente con artefacto.
  (b) índice edge/central per-sujeto (crudo y periódico) + consistencia de signo.

Pregunta: ¿la "difusión" del GA es una fuente difusa real, o el promedio de topografías
idiosincrásicas por sujeto (algunas claramente temporales = EMG)? Lo segundo apoya artefacto.

Desde el cache panel_psd (29 ch). Hereda Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.gamma_artifact_check
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import CENTRAL, COHORT, EMG_EDGE, REPO
from src.campeones_analysis.multimodal_arousal.decoding_panel import _make_info, _topo, load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic

warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_4_edge_central" / "gamma_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GBAND = (30.0, 40.0)


def raw_gamma_db(psd, freqs):
    m = (freqs >= GBAND[0]) & (freqs < GBAND[1])
    return 10.0 * np.log10(psd[:, :, m].mean(axis=2) + 1e-30)   # (n_ep, n_ch)


def periodic_gamma(psd, freqs):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-40"])
    m = (f >= GBAND[0]) & (f < GBAND[1])
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)         # (n_ep, n_ch)


def _idx(ch, region):
    return [ch.index(c) for c in region if c in ch]


def main():
    print("=" * 78)
    print("gamma_artifact_check :: topografía per-sujeto crudo vs periódico (escéptico)")
    print("=" * 78, flush=True)
    cache, freqs, ch = load_cache("uniform")
    info = _make_info(ch)
    e_idx, c_idx = _idx(ch, EMG_EDGE), _idx(ch, CENTRAL)

    d_raw, d_per, rows = {}, {}, []
    for sub in COHORT:
        psd, y, tn = cache[sub]
        r = raw_gamma_db(psd, freqs); p = periodic_gamma(psd, freqs)
        dr = r[y == 1].mean(0) - r[y == 0].mean(0)
        dp = p[y == 1].mean(0) - p[y == 0].mean(0)
        d_raw[sub], d_per[sub] = dr, dp
        rows.append(dict(
            subject=sub,
            raw_edge=round(float(dr[e_idx].mean()), 3), raw_central=round(float(dr[c_idx].mean()), 3),
            raw_edge_minus_central=round(float(dr[e_idx].mean() - dr[c_idx].mean()), 3),
            raw_peak_ch=ch[int(np.argmax(dr))],
            per_edge=round(float(dp[e_idx].mean()), 4), per_central=round(float(dp[c_idx].mean()), 4),
            per_edge_minus_central=round(float(dp[e_idx].mean() - dp[c_idx].mean()), 4),
            per_peak_ch=ch[int(np.argmax(dp))],
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gamma_persubject.csv", index=False)
    ga_raw = np.mean([d_raw[s] for s in COHORT], axis=0)
    ga_per = np.mean([d_per[s] for s in COHORT], axis=0)

    print("\n=== Per-sujeto: edge−central (crudo dB / periódico) + canal pico ===")
    print(df.to_string(index=False))
    n_raw_edge = int((df["raw_edge_minus_central"] > 0).sum())
    n_per_edge = int((df["per_edge_minus_central"] > 0).sum())
    print(f"\n  CRUDO: edge>central en {n_raw_edge}/6   PERIÓDICO: edge>central en {n_per_edge}/6")
    print(f"  Canal pico CRUDO por sujeto: {list(df['raw_peak_ch'])}")
    print(f"  -> ¿temporal/edge (T7/T8/FT9/FT10/P7/P8) = EMG, o central (Cz/C3/C4) ?")

    # --- figura: overview per-sujeto (filas) x [crudo, periódico] (col) + GA ---
    subs = COHORT + ["GA"]
    fig, axes = plt.subplots(len(subs), 2, figsize=(6.5, 2.6 * len(subs)))
    for i, s in enumerate(subs):
        dr = ga_raw if s == "GA" else d_raw[s]
        dp = ga_per if s == "GA" else d_per[s]
        _topo(axes[i, 0], dr, info, f"{s} crudo")
        _topo(axes[i, 1], dp, info, f"{s} periódico")
    axes[0, 0].set_title("gamma 30-40 CRUDO (SCR−no-SCR)", fontsize=10)
    axes[0, 1].set_title("gamma 30-40 PERIÓDICO (1/f removido)", fontsize=10)
    fig.suptitle("Gamma per-sujeto: ¿el edge/temporal (EMG) sobrevive al remover el 1/f?\n"
                 "(rojo edge/temporal = EMG · el GA difumina la heterogeneidad)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "gamma_persubject_topo.png", dpi=120)
    plt.close(fig)
    print(f"\n-> {OUT_DIR / 'gamma_persubject_topo.png'}\n[gamma-check a] done", flush=True)


if __name__ == "__main__":
    main()
