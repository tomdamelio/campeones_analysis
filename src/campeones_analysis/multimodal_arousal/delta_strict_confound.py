"""Strict-test del DELTA (paralelo exacto al gamma deep-check, 2026-06-08).

El gamma colapsó bajo un covariado muscular fuerte (gamma cruda per-canal edge) → el "63% que
sobrevivía" con el ratio único era músculo residual. La MISMA lógica ("covariado grosero infla la
supervivencia") puede aplicar al delta. Acá se re-corre Q-delta con un covariado de artefacto RICO:

  - LIGHT (= 2.5 full Q-delta): blink_slow + blink_2hz_pre + veog_slow_0p5_8 + veog_slow_2hz_pre +
    var_jerk_0p5_2 + tiempo (todos escalares/channel-mean).
  - STRICT: LIGHT + potencia de delta 1-4 CRUDA en cada canal de ARTEFACTO (Fp1/Fp2/F7/F8 =
    blink/ocular frontal; EMG_EDGE = movimiento/edge) + movimiento per-EJE (var_jerk_x/y/z,
    0.5-8, broadband) -> captura el artefacto lento heterogéneo per-sujeto/per-canal.

VD = delta periódico all-channel (feat_band 1-4, 1/f en 1-30), igual que 2.5. Si el delta COLAPSA
con STRICT -> era artefacto residual (como gamma); si SOBREVIVE -> es la señal genuina, separable del
artefacto lento. Desde el cache (29 ch). Hereda Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.delta_strict_confound --nperm 1000
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, EMG_EDGE, REPO
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
DBAND = (1.0, 4.0)
ARTIFACT_CH = ["Fp1", "Fp2", "F7", "F8"] + EMG_EDGE          # blink/ocular frontal + movimiento/edge
POSTERIOR_CH = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]  # señal candidata, DISJUNTA de ARTIFACT_CH
LIGHT_COLS = ["blink_slow", "blink_2hz_pre", "veog_slow_0p5_8", "veog_slow_2hz_pre",
              "var_jerk_0p5_2", "tnorm", "tnorm2"]
MOTION_EXTRA = ["var_jerk_x", "var_jerk_y", "var_jerk_z", "var_jerk_0p5_8", "var_jerk"]


def raw_delta(psd, freqs, idx):
    m = (freqs >= DBAND[0]) & (freqs < DBAND[1])
    g = 10.0 * np.log10(psd[:, :, m].mean(axis=2) + 1e-30)
    return g[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78)
    print("delta_strict_confound :: strict-test del DELTA (paralelo al gamma deep-check)")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    d22 = pd.read_csv(COV22)
    a_idx = [ch.index(c) for c in ARTIFACT_CH if c in ch]

    p_idx = [ch.index(c) for c in POSTERIOR_CH if c in ch]
    Xs_all, Xs_post, ys, C_light, C_strict = [], [], [], [], []
    for sub in cache:
        psd, y, tn = cache[sub]
        dper = feat_band(psd, freqs, RANGES["1-30"], DBAND)           # delta periódico per-canal
        Xs_all.append(dper)                                           # VD all-channel (= 2.5)
        Xs_post.append(dper[:, p_idx])                                # VD posterior-only (disjunto)
        ys.append(y.astype(int))
        cl = np.column_stack([covs[sub][c] for c in LIGHT_COLS])
        m = d22[d22.subject == sub].reset_index(drop=True)
        motion = np.column_stack([m[c].to_numpy() for c in MOTION_EXTRA])
        art = raw_delta(psd, freqs, a_idx)                            # delta cruda per-canal artefacto
        C_light.append(cl)
        C_strict.append(np.column_stack([cl, art, motion]))

    print(f"  covariados: LIGHT={C_light[0].shape[1]}  STRICT={C_strict[0].shape[1]} "
          f"(+{len(a_idx)} delta cruda artefacto +{len(MOTION_EXTRA)} movimiento per-eje); "
          f"VD posterior={len(p_idx)} ch", flush=True)

    rows = []
    configs = [("light all-ch (=2.5)", Xs_all, C_light),
               ("strict all-ch", Xs_all, C_strict),
               ("strict posterior-only (disjunto)", Xs_post, C_strict)]
    for name, X, C in configs:
        ri, rl, groups = evaluate(X, ys, C, deconf=False)
        di, dl, _ = evaluate(X, ys, C, deconf=True)
        pi, pl = _perm_p(X, ys, C, True, di, dl, args.nperm, groups)
        rows.append(dict(covariados=name, raw_intra=round(ri, 4), deconf_intra=round(di, 4),
                         raw_loso=round(rl, 4), deconf_loso=round(dl, 4),
                         p_deconf_intra=pi, p_deconf_loso=pl))
        print(f"  {name:34s}: intra {ri:.3f}->{di:.3f}  loso {rl:.3f}->{dl:.3f} "
              f"(p_dec_loso={pl})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "delta_strict_confound.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, df["raw_loso"], w, label="raw", color="C0")
    ax.bar(x + w / 2, df["deconf_loso"], w, label="deconfounded", color="C1")
    for xi, (d, p) in enumerate(zip(df["deconf_loso"], df["p_deconf_loso"])):
        ax.annotate(f"p={p:.3f}", (xi + w / 2, d + 0.005), ha="center", fontsize=8,
                    color="darkred" if p < 0.05 else "0.4")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(df["covariados"]); ax.set_ylim(0.45, 0.7)
    ax.set_ylabel("LOSO AUC (delta periódico)")
    ax.set_title("Strict-test DELTA: ¿sobrevive un covariado de artefacto FUERTE?\n"
                 "strict = + delta cruda per-canal (Fp/F/edge) + movimiento per-eje", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "delta_strict_confound.png", dpi=120); plt.close(fig)
    print(f"\n-> {TBL_DIR / 'delta_strict_confound.csv'}\n[delta-strict] done", flush=True)


if __name__ == "__main__":
    main()
