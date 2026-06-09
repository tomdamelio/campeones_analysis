"""Strict-test de la DESINCRONIZACIÓN ALFA — control positivo (2026-06-08).

Gamma y delta COLAPSARON bajo control de artefacto fuerte (per-canal) -> sus subidas eran
artefacto residual. La alfa es distinta: es una CAÍDA de potencia (desync), y ningún artefacto
aditivo (músculo/blink/microsacada/movimiento/sudor) puede FABRICAR una caída (todos SUMAN). Si la
desync alfa SOBREVIVE el mismo control fuerte que tumbó a gamma/delta -> es la señal cortical
genuina del Bloque 2 (posterior, signo opuesto al EMG).

Mismo esquema que delta_strict_confound: VD = alfa periódico (8-13) all-channel + posterior-only;
LIGHT (=2.5 Q-alfa) vs STRICT (+ alfa cruda per-canal en sitios de artefacto + movimiento per-eje).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_strict_confound --nperm 1000
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import EMG_EDGE, REPO
from src.campeones_analysis.multimodal_arousal.confound_model_scr import (
    COV22,
    OUT_DIR,
    _load_covariates,
    _perm_p,
    evaluate,
    feat_band,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES

warnings.filterwarnings("ignore")
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
ABAND = (8.0, 13.0)
ARTIFACT_CH = ["Fp1", "Fp2", "F7", "F8"] + EMG_EDGE
POSTERIOR_CH = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]   # alfa posterior, DISJUNTA de ARTIFACT_CH
LIGHT_COLS = ["blink_slow", "veog_slow_0p5_8", "tnorm", "tnorm2"]
MOTION_EXTRA = ["var_jerk_x", "var_jerk_y", "var_jerk_z", "var_jerk_0p5_8", "var_jerk"]


def raw_alpha(psd, freqs, idx):
    m = (freqs >= ABAND[0]) & (freqs < ABAND[1])
    g = 10.0 * np.log10(psd[:, :, m].mean(axis=2) + 1e-30)
    return g[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78)
    print("alpha_strict_confound :: control positivo — ¿la desync alfa sobrevive el control fuerte?")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    d22 = pd.read_csv(COV22)
    a_idx = [ch.index(c) for c in ARTIFACT_CH if c in ch]
    p_idx = [ch.index(c) for c in POSTERIOR_CH if c in ch]

    Xs_all, Xs_post, ys, C_light, C_strict = [], [], [], [], []
    for sub in cache:
        psd, y, tn = cache[sub]
        aper = feat_band(psd, freqs, RANGES["1-30"], ABAND)
        Xs_all.append(aper); Xs_post.append(aper[:, p_idx]); ys.append(y.astype(int))
        cl = np.column_stack([covs[sub][c] for c in LIGHT_COLS])
        m = d22[d22.subject == sub].reset_index(drop=True)
        motion = np.column_stack([m[c].to_numpy() for c in MOTION_EXTRA])
        art = raw_alpha(psd, freqs, a_idx)
        C_light.append(cl); C_strict.append(np.column_stack([cl, art, motion]))

    print(f"  covariados: LIGHT={C_light[0].shape[1]}  STRICT={C_strict[0].shape[1]}; "
          f"VD posterior={len(p_idx)} ch", flush=True)

    rows = []
    for name, X, C in [("light all-ch (=2.5)", Xs_all, C_light),
                       ("strict all-ch", Xs_all, C_strict),
                       ("strict posterior-only (disjunto)", Xs_post, C_strict)]:
        ri, rl, groups = evaluate(X, ys, C, deconf=False)
        di, dl, _ = evaluate(X, ys, C, deconf=True)
        pi, pl = _perm_p(X, ys, C, True, di, dl, args.nperm, groups)
        rows.append(dict(covariados=name, raw_loso=round(rl, 4), deconf_loso=round(dl, 4),
                         deconf_intra=round(di, 4), p_deconf_loso=pl))
        print(f"  {name:34s}: loso {rl:.3f}->{dl:.3f} intra->{di:.3f} (p_dec_loso={pl})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_strict_confound.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, df["raw_loso"], w, label="raw", color="C0")
    ax.bar(x + w / 2, df["deconf_loso"], w, label="deconfounded", color="C1")
    for xi, (d, p) in enumerate(zip(df["deconf_loso"], df["p_deconf_loso"])):
        ax.annotate(f"p={p:.3f}", (xi + w / 2, d + 0.005), ha="center", fontsize=8,
                    color="darkred" if p < 0.05 else "0.4")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(df["covariados"], fontsize=8); ax.set_ylim(0.45, 0.7)
    ax.set_ylabel("LOSO AUC (alfa periódico, desync)")
    ax.set_title("Control positivo: ¿la desync ALFA sobrevive el control de artefacto FUERTE?\n"
                 "(sobrevive = señal cortical genuina · una CAÍDA no la fabrica un artefacto aditivo)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "alpha_strict_confound.png", dpi=120); plt.close(fig)
    print(f"\n-> {TBL_DIR / 'alpha_strict_confound.csv'}\n[alpha-strict] done", flush=True)


if __name__ == "__main__":
    main()
