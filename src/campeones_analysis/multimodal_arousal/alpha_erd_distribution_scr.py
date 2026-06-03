"""Distribución por-época del ERD alfa PO: box + scatter por sujeto (follow-up usuario
2026-06-03). Para ver no solo las medias sino la DISPERSIÓN de los valores de ERD por época.

Cada punto = una época. Por sujeto, dos columnas: real (SCR, rojo) y silent (gris), lado a
lado, sujetos ordenados de izquierda a derecha. Valor por época = ERD alfa (8-13 Hz) parieto-
occipital en dB respecto al baseline edge-safe (-4.5,-3.5), promediado sobre la ventana de
interés. Baseline común por condición (mean épocas × ventana de baseline), consistente con
alpha_erd_baseline_scr. Épocas con padding (-7,+4) para TFR limpio de borde.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_erd_distribution_scr
  ... --woi post     (ventana [0,1] post-onset en vez de la completa (-4,+3))
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from scipy import stats

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
from src.campeones_analysis.multimodal_arousal.alpha_erd_baseline_scr import (
    PARIETOOCCIPITAL,
    _alpha_power,
    _build_padded,
    _roi,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "qa_artifact_vs_signal" / "alpha_desync"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BL_SAFE = (-4.5, -3.5)
WOI = {"full": (-4.0, 3.0), "post": (0.0, 1.0)}
JIT = np.random.default_rng(0)


def per_epoch_erd(times, roi_pow, bl, woi):
    """Per-epoch dB ERD vs common baseline, averaged over WOI. Returns array[n_epochs]."""
    mb = (times >= bl[0]) & (times <= bl[1])
    base = roi_pow[:, mb].mean()  # common baseline scalar
    db = 10.0 * np.log10(roi_pow / base + 1e-30)  # [n_ep, n_t]
    mw = (times >= woi[0]) & (times <= woi[1])
    return db[:, mw].mean(axis=1)  # [n_ep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woi", choices=list(WOI), default="full")
    args = ap.parse_args()
    woi = WOI[args.woi]

    print("=" * 78)
    print(f"alpha_erd_distribution :: per-epoch ERD alfa PO  WOI={args.woi}{woi}  bl={BL_SAFE}")
    print("=" * 78, flush=True)

    data = {}  # sub -> (erd_real[n], erd_silent[n])
    rows = []
    for sub in COHORT:
        real_ep, silent_ep = _build_padded(sub)
        if real_ep is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        t, pr, ch = _alpha_power(real_ep)
        _, ps, _ = _alpha_power(silent_ep)
        er = per_epoch_erd(t, _roi(pr, ch, PARIETOOCCIPITAL), BL_SAFE, woi)
        si = per_epoch_erd(t, _roi(ps, ch, PARIETOOCCIPITAL), BL_SAFE, woi)
        data[sub] = (er, si)
        u, p = stats.mannwhitneyu(er, si, alternative="two-sided")
        rows.append(dict(subject=sub, n_real=len(er), n_silent=len(si),
                         median_real=round(float(np.median(er)), 3),
                         median_silent=round(float(np.median(si)), 3),
                         mean_real=round(float(er.mean()), 3),
                         mean_silent=round(float(si.mean()), 3),
                         iqr_real=round(float(np.subtract(*np.percentile(er, [75, 25]))), 2),
                         iqr_silent=round(float(np.subtract(*np.percentile(si, [75, 25]))), 2),
                         mannwhitney_p=round(float(p), 4)))
        print(f"  {sub}: median real={rows[-1]['median_real']:+.2f} silent={rows[-1]['median_silent']:+.2f} "
              f"IQR_real={rows[-1]['iqr_real']:.1f} MWU p={rows[-1]['mannwhitney_p']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / f"alpha_erd_distribution_{args.woi}.csv", index=False)

    # ---- combined box + scatter, subjects left to right, real|silent per subject ----
    fig, ax = plt.subplots(figsize=(15, 7))
    subs = list(data.keys())
    box_data, positions, colors, xticks, xticklabels = [], [], [], [], []
    for i, sub in enumerate(subs):
        er, si = data[sub]
        base_x = i * 3.0
        for j, (vals, cond, col) in enumerate([(er, "real", "C3"), (si, "silent", "0.45")]):
            pos = base_x + j
            box_data.append(vals); positions.append(pos); colors.append(col)
            jx = pos + JIT.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(jx, vals, s=6, color=col, alpha=0.25, lw=0, zorder=1)
        xticks.append(base_x + 0.5)
        md = df.loc[df.subject == sub, "median_real"].values[0] - df.loc[df.subject == sub, "median_silent"].values[0]
        xticklabels.append(f"{sub}\nΔmed={md:+.2f}")

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, zorder=2,
                    medianprops=dict(color="k", lw=1.6), whiskerprops=dict(color="0.3"),
                    capprops=dict(color="0.3"), boxprops=dict(alpha=0.55))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    ax.set_ylabel(f"ERD alfa PO por época (dB, baseline {BL_SAFE}, WOI {args.woi} {woi})")
    ax.set_title("Distribución por-época del ERD alfa parieto-occipital: real (rojo) vs silent (gris) "
                 "por sujeto.\nCada punto = una época. Negativo = desync. (whiskers sin outliers)",
                 fontsize=11)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="C3", alpha=0.55, label="real (SCR)"),
                       Patch(facecolor="0.45", alpha=0.55, label="silent")], fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"alpha_erd_distribution_{args.woi}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("\n" + df.to_string(index=False), flush=True)
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
