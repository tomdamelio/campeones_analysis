"""plot 3 bis: scatter por sujeto de potencia DELTA parieto-occipital vs amplitud del SCR.
Idéntico al scatter alfa-PO (figura 1 de band_scr_amplitude_scr) pero para delta = el lead.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._scr_amplitude_delta_plot
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.band_scr_amplitude_scr import (
    FIG_DIR,
    subject_amp_power,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

BAND, ROI = "delta", "parieto-occipital"


def main():
    print(f"plot 3 bis :: scatter {BAND}-{ROI} vs amplitud SCR")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    used = []
    for sub in COHORT:
        d = subject_amp_power(sub)
        if d is None:
            continue
        used.append((sub, d))
    for ax, (sub, d) in zip(axes.ravel(), used):
        x, y = d["amp"], d[(BAND, ROI)]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, alpha=0.4, color=SUBJ_COLORS.get(sub, "C0"))
        if m.sum() > 3:
            b1, b0 = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5)
            rho, p = stats.spearmanr(x[m], y[m])
            ax.set_title(f"{sub}  delta-PO  rho={rho:+.2f} (p={p:.3f})", fontsize=9)
            print(f"  {sub}: rho={rho:+.3f} p={p:.3f} n={int(m.sum())}")
        ax.set_xlabel("amplitud SCR (phasic peak)")
        ax.set_ylabel("log delta PO power")
    fig.suptitle("Acoplamiento graduado: potencia DELTA parieto-occipital por época vs amplitud del SCR "
                 "(real). Positivo = más delta con SCR mayor.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "scr_amplitude_deltaPO_scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
