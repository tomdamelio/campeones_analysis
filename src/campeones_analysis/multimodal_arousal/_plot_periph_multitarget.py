"""Figura del multitarget periférico: rho EEG-banda <-> {SMNA,phasic,HR,RVT}, PO vs edge, N=6.
Lee periph_multitarget_perSubject.csv. Read-only plotting.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import OUT

TBL = OUT / "eeg_smna_coupling" / "tables"
df = pd.read_csv(TBL / "periph_multitarget_perSubject.csv")
TARGETS = ["smna", "phasic", "hr", "rvt"]
BANDS = ["delta", "alpha", "gamma"]
TLAB = {"smna": "SMNA", "phasic": "EDA-phasic", "hr": "HR", "rvt": "Respiración (RVT)"}

fig, axes = plt.subplots(1, len(TARGETS), figsize=(17, 4.3), sharey=True)
x = np.arange(len(BANDS))
w = 0.36
for ax, t in zip(axes, TARGETS):
    po = [df[(df.target == t) & (df.band == b) & (df.roi == "PO")]["rho"] for b in BANDS]
    ed = [df[(df.target == t) & (df.band == b) & (df.roi == "edge")]["rho"] for b in BANDS]
    po_m = [s.mean() for s in po]
    ed_m = [s.mean() for s in ed]
    ax.bar(x - w / 2, po_m, w, color="crimson", label="PO (parieto-occ.)")
    ax.bar(x + w / 2, ed_m, w, color="0.55", label="edge (proxy EMG)")
    # puntos por sujeto (PO)
    for j, s in enumerate(po):
        ax.scatter(np.full(len(s), x[j] - w / 2) + np.random.uniform(-0.05, 0.05, len(s)),
                   s, s=12, color="k", alpha=0.5, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_title(TLAB[t])
    if t == "smna":
        ax.set_ylabel("Spearman rho (N=6, runs concat, z intra-run)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Acoplamiento EEG band-power <-> periferia por target — PO vs edge (la firma es AMPLIA: PO≈edge en todo target; HR engancha como la EDA; respiración ~nula)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = OUT / "eeg_smna_coupling" / "figures" / "periph_multitarget.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Guardado -> {out}")
