"""Prueba (b) — ¿la gamma periódica sobrevive un covariado muscular MÁS FUERTE?

La crítica del usuario: el covariado de músculo en 2.5 es un ratio edge/central ÚNICO y casi-GA,
que NO captura la heterogeneidad per-sujeto del EMG -> el "sobrevive 63%" puede dejar músculo
residual. Acá se re-corre Q-gamma (VD = gamma periódico) con un covariado muscular RICO:
  - LIGHT (= 2.5 full): gamma_EOG + sp_hf + heog_gamma + edge_central_ratio + var_jerk + tiempo.
  - STRICT: LIGHT + potencia CRUDA de gamma 30-40 en CADA canal edge (7 ch, captura el EMG
    broadband per-sujeto/per-canal) + offset (nivel aperiódico/broadband) channel-mean del edge.
Si la gamma periódica COLAPSA a azar con STRICT -> el "sobrevive" era músculo residual (el usuario
tiene razón). Si SOBREVIVE igual -> es robusto a un control muscular directo.

Reusa la maquinaria probada de confound_model_scr. Desde el cache (29 ch). Hereda Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.gamma_strict_confound --nperm 1000
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import CENTRAL, COHORT, EMG_EDGE, REPO
from src.campeones_analysis.multimodal_arousal.confound_model_scr import (
    OUT_DIR,
    _load_covariates,
    _perm_p,
    evaluate,
    feat_band,
    feat_offset,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES

warnings.filterwarnings("ignore")
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
GBAND = (30.0, 40.0)
LIGHT_COLS = ["gamma_EOG", "sp_hf", "heog_gamma_30_40", "edge_gamma", "var_jerk", "tnorm", "tnorm2"]


def raw_gamma_edge(psd, freqs, e_idx):
    m = (freqs >= GBAND[0]) & (freqs < GBAND[1])
    g = 10.0 * np.log10(psd[:, :, m].mean(axis=2) + 1e-30)   # (n_ep, n_ch)
    return g[:, e_idx]                                        # (n_ep, n_edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78)
    print("gamma_strict_confound :: (b) Q-gamma vs covariado muscular FUERTE (edge crudo per-canal)")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    e_idx = [ch.index(c) for c in EMG_EDGE if c in ch]

    Xs, ys, C_light, C_strict = [], [], [], []
    for sub in cache:
        psd, y, tn = cache[sub]
        Xs.append(feat_band(psd, freqs, RANGES["1-40"], GBAND))   # VD = gamma periódico
        ys.append(y.astype(int))
        cl = np.column_stack([covs[sub][c] for c in LIGHT_COLS])
        edge_raw = raw_gamma_edge(psd, freqs, e_idx)              # (n_ep, n_edge) EMG broadband per-canal
        edge_off = feat_offset(psd, freqs, RANGES["1-40"])[:, e_idx].mean(axis=1, keepdims=True)
        C_light.append(cl)
        C_strict.append(np.column_stack([cl, edge_raw, edge_off]))

    print(f"  covariados: LIGHT={C_light[0].shape[1]}  STRICT={C_strict[0].shape[1]} "
          f"(+{len(e_idx)} edge crudo +1 offset edge)", flush=True)

    rows = []
    for name, C in [("light (=2.5)", C_light), ("strict (+edge crudo)", C_strict)]:
        ri, rl, groups = evaluate(Xs, ys, C, deconf=False)
        di, dl, _ = evaluate(Xs, ys, C, deconf=True)
        pi, pl = _perm_p(Xs, ys, C, True, di, dl, args.nperm, groups)
        rows.append(dict(covariados=name, raw_intra=round(ri, 4), deconf_intra=round(di, 4),
                         raw_loso=round(rl, 4), deconf_loso=round(dl, 4),
                         p_deconf_intra=pi, p_deconf_loso=pl))
        print(f"  {name:22s}: intra {ri:.3f}->{di:.3f}  loso {rl:.3f}->{dl:.3f} "
              f"(p_dec_loso={pl})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "gamma_strict_confound.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, df["raw_loso"], w, label="raw", color="C0")
    ax.bar(x + w / 2, df["deconf_loso"], w, label="deconfounded", color="C1")
    for xi, (d, p) in enumerate(zip(df["deconf_loso"], df["p_deconf_loso"])):
        ax.annotate(f"p={p:.3f}", (xi + w / 2, d + 0.005), ha="center", fontsize=8,
                    color="darkred" if p < 0.05 else "0.4")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(df["covariados"]); ax.set_ylim(0.45, 0.7)
    ax.set_ylabel("LOSO AUC (gamma periódico)")
    ax.set_title("(b) ¿La gamma periódica sobrevive un covariado muscular MÁS FUERTE?\n"
                 "strict = + potencia gamma cruda per-canal edge (EMG broadband) + offset edge", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gamma_strict_confound.png", dpi=120); plt.close(fig)
    print(f"\n-> {TBL_DIR / 'gamma_strict_confound.csv'}\n[gamma-check b] done", flush=True)


if __name__ == "__main__":
    main()
