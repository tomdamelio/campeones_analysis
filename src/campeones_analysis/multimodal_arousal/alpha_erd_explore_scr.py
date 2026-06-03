"""Visualización de primera mano del alfa-ERD: curvas real vs silent por sujeto + barrido de
baseline (follow-up usuario 2026-06-03). Responde "¿cómo se obtiene el desync y cuánto cambia
moviendo el baseline?".

Computa el Morlet alfa UNA vez por sujeto (potencia absoluta, épocas con padding -7..+4) y
reaplica distintos baselines (barato). Dos figuras:
  (1) alpha_erd_persubject.png : 2x3, por sujeto ERD_real(t) y ERD_silent(t) en dB con SEM,
      baseline edge-safe (-4.5,-3.5) sombreado, WOI [0,1] marcado. El "desync" = cuánto cae
      real respecto a su baseline vs cuánto cae silent.
  (2) alpha_erd_baseline_sweep.png : real-silent (full window) por sujeto + GA, deslizando la
      ventana de baseline de -6 a -2.5 s. Muestra la sensibilidad a la elección de baseline.

Reusa _build_padded/_alpha_power/_roi/PARIETOOCCIPITAL/PAD/DISPLAY de alpha_erd_baseline_scr.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_erd_explore_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.alpha_erd_baseline_scr import (
    DISPLAY,
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
WOI_POST = (0.0, 1.0)
WOI_FULL = (-4.0, 3.0)
# ventanas de baseline para el barrido (1 s c/u, dentro del padding -7..+4, pre-onset)
SWEEP_BL = [(-6.0, -5.0), (-5.5, -4.5), (-5.0, -4.0), (-4.5, -3.5), (-4.0, -3.0), (-3.5, -2.5)]
DOMINANT = "sub-33"


def _erd_db_perepoch(times, roi_pow, bl):
    """dB ERD por época vs baseline escalar (mean épocas×ventana). Devuelve dB[n_ep, n_t]."""
    m = (times >= bl[0]) & (times <= bl[1])
    base = roi_pow[:, m].mean()
    return 10.0 * np.log10(roi_pow / base + 1e-30)


def _crop(times, arr):
    m = (times >= DISPLAY[0]) & (times <= DISPLAY[1])
    return times[m], arr[..., m]


def _woi(times, tc, w):
    m = (times >= w[0]) & (times <= w[1])
    return float(np.nanmean(tc[m]))


def main():
    print("=" * 78)
    print(f"alpha_erd_explore :: per-subject curves + baseline sweep  -> {FIG_DIR}")
    print("=" * 78, flush=True)

    # compute alpha power ONCE per subject (absolute), store ROI power
    store = {}  # sub -> (times, roi_real[n_ep,n_t], roi_silent[n_ep,n_t])
    for sub in COHORT:
        real_ep, silent_ep = _build_padded(sub)
        if real_ep is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        t, pr, ch = _alpha_power(real_ep)
        _, ps, _ = _alpha_power(silent_ep)
        store[sub] = (t, _roi(pr, ch, PARIETOOCCIPITAL), _roi(ps, ch, PARIETOOCCIPITAL))
        print(f"  {sub}: n_real={len(real_ep)} n_silent={len(silent_ep)} alpha power computed",
              flush=True)

    # ---- Figure 1: per-subject real vs silent ERD (edge-safe baseline) ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, sub in zip(axes.ravel(), COHORT):
        if sub not in store:
            ax.set_visible(False)
            continue
        t, rr, rs = store[sub]
        erd_r = _erd_db_perepoch(t, rr, BL_SAFE)   # [n_ep, n_t] dB
        erd_s = _erd_db_perepoch(t, rs, BL_SAFE)
        tc, erd_r = _crop(t, erd_r)
        _, erd_s = _crop(t, erd_s)
        rm, rsem = erd_r.mean(0), erd_r.std(0, ddof=1) / np.sqrt(erd_r.shape[0])
        sm, ssem = erd_s.mean(0), erd_s.std(0, ddof=1) / np.sqrt(erd_s.shape[0])
        ax.fill_between(tc, rm - rsem, rm + rsem, color="C3", alpha=0.2, lw=0)
        ax.fill_between(tc, sm - ssem, sm + ssem, color="0.4", alpha=0.2, lw=0)
        ax.plot(tc, rm, color="C3", lw=1.6, label="real (SCR)")
        ax.plot(tc, sm, color="0.4", lw=1.4, ls="--", label="silent")
        ax.axvspan(*BL_SAFE, color="cyan", alpha=0.12)
        ax.axvspan(*WOI_POST, color="gold", alpha=0.15)
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        diff_full = _woi(tc, rm - sm, WOI_FULL)
        ax.set_title(f"{sub}  (real−silent full={diff_full:+.2f} dB)", fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("time from SCR onset (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("alpha dB vs baseline (-4.5,-3.5)")
    fig.suptitle("Alfa parieto-occipital ERD por sujeto: real vs silent (baseline edge-safe, "
                 "cyan; WOI [0,1], dorado). Negativo = desync.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "alpha_erd_persubject.png", dpi=130)
    plt.close(fig)
    print("  -> alpha_erd_persubject.png", flush=True)

    # ---- Figure 2: baseline sweep ----
    rows = []
    sweep = {sub: [] for sub in store}
    ga_vals, ga_drop_vals = [], []
    for bl in SWEEP_BL:
        per_sub = {}
        for sub, (t, rr, rs) in store.items():
            erd_r = _erd_db_perepoch(t, rr, bl).mean(0)
            erd_s = _erd_db_perepoch(t, rs, bl).mean(0)
            tc, d = _crop(t, erd_r - erd_s)
            v = _woi(tc, d, WOI_FULL)
            per_sub[sub] = v
            sweep[sub].append(v)
            rows.append(dict(baseline=f"{bl[0]},{bl[1]}", subject=sub, diff_full_dB=round(v, 3)))
        vals = np.array(list(per_sub.values()))
        ga_vals.append(vals.mean())
        ga_drop_vals.append(np.array([per_sub[s] for s in per_sub if s != DOMINANT]).mean())

    pd.DataFrame(rows).to_csv(TBL_DIR / "alpha_erd_baseline_sweep.csv", index=False)
    xlab = [f"{b[0]:.1f}\n{b[1]:.1f}" for b in SWEEP_BL]
    x = np.arange(len(SWEEP_BL))
    fig, ax = plt.subplots(figsize=(11, 6))
    for sub in store:
        ax.plot(x, sweep[sub], marker="o", ms=4, color=SUBJ_COLORS.get(sub, "0.6"),
                lw=1.0, alpha=0.7, label=sub)
    ax.plot(x, ga_vals, color="k", lw=2.6, marker="s", label="GA")
    ax.plot(x, ga_drop_vals, color="C1", lw=2.0, ls="--", marker="s", label=f"GA sin {DOMINANT}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_xlabel("ventana de baseline (s, pre-onset)")
    ax.set_ylabel("real − silent, ventana completa (dB)")
    ax.set_title("Sensibilidad del alfa-ERD a la ventana de baseline (PO, full window).\n"
                 "Negativo = desync. Si fuera robusto, las líneas serían planas y negativas.",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_erd_baseline_sweep.png", dpi=130)
    plt.close(fig)
    print("  -> alpha_erd_baseline_sweep.png", flush=True)

    # resumen sweep
    df = pd.DataFrame(rows).pivot(index="subject", columns="baseline", values="diff_full_dB")
    print("\nBarrido de baseline (real-silent full, dB):")
    print(df.to_string(), flush=True)
    print(f"\nGA por baseline: {[round(v,3) for v in ga_vals]}", flush=True)
    print(f"GA sin {DOMINANT}: {[round(v,3) for v in ga_drop_vals]}", flush=True)
    print(f"\nOutputs -> {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
