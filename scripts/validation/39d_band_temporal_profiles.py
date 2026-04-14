#!/usr/bin/env python
"""Perfiles temporales de bandas para la tarea 4-clases (script 34).

Carga los coeficientes de 39_coefs.json y visualiza:

  A) Perfil temporal: para cada banda espectral, la media de β (promediada
     sobre todos los canales) como función de la clase (Baseline → ChangeUp
     → Luminance → ChangeDown). Muestra qué bandas suben/bajan en cada
     ventana temporal.

  B) Alpha y beta en canales ROI (O1, O2, Cz, Fz, T7, T8): barplots
     mostrando cómo varía β a través de las 4 clases para cada canal.

  C) Mini-heatmap banda × clase para cada canal ROI.

Usage
-----
    micromamba run -n campeones python scripts/validation/39d_band_temporal_profiles.py --subject 27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "interpretability"

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CLASS_LABELS = [
    "Baseline\n(−500→0ms)",
    "ChangeUp\n(0→+500ms)",
    "Luminance\n(+500→+1000ms)",
    "ChangeDown\n(+1000→+1500ms)",
]
CLASS_SHORT = ["BL\n−500→0", "CU\n0→+500", "LU\n+500→+1s", "CD\n+1→+1.5s"]
ROI_CHANNELS = ["O1", "O2", "Cz", "Fz", "T7", "T8"]
BAND_COLORS = {
    "delta": "#4C72B0",
    "theta": "#DD8452",
    "alpha": "#55A868",
    "beta":  "#C44E52",
    "gamma": "#8172B2",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    json_path = out_dir / f"sub-{sub}_39_coefs.json"
    print(f"Loading coefficients from {json_path.name}...")

    with open(json_path) as f:
        data = json.load(f)

    coef_34 = np.array(data["task_34"]["coef"])  # (4, n_feat)
    ch_names = data["task_34"]["ch_names"]
    n_ch = len(ch_names)
    n_bands = len(BAND_NAMES)

    # Reshape to (4_classes, n_ch, n_bands)
    mat_34 = coef_34.reshape(4, n_ch, n_bands)

    # -----------------------------------------------------------------------
    # Figure A: Perfil temporal — media β y media |β| por banda
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"sub-{sub} — Perfil temporal de coeficientes β (4 clases)\n"
        "β > 0 significa que potencia alta en esa banda/clase predice ESA clase",
        fontsize=11,
    )

    # Left: signed mean β
    ax = axes[0]
    ax.set_title("β promedio (sobre canales) — con signo", fontsize=10)
    for b_idx, b_name in enumerate(BAND_NAMES):
        vals = mat_34[:, :, b_idx].mean(axis=1)  # (4,)
        ax.plot(range(4), vals, marker="o", label=b_name,
                color=BAND_COLORS[b_name], lw=2, markersize=8)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6,
                        color=BAND_COLORS[b_name])
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_LABELS, fontsize=8)
    ax.set_ylabel("Mean β (z-score scale)", fontsize=9)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: unsigned mean |β|
    ax = axes[1]
    ax.set_title("β promedio |absoluto| (sobre canales) — magnitud de contribución", fontsize=10)
    for b_idx, b_name in enumerate(BAND_NAMES):
        vals = np.abs(mat_34[:, :, b_idx]).mean(axis=1)
        ax.plot(range(4), vals, marker="o", label=b_name,
                color=BAND_COLORS[b_name], lw=2, markersize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_LABELS, fontsize=8)
    ax.set_ylabel("Mean |β|", fontsize=9)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_a = out_dir / f"sub-{sub}_39d_band_temporal_profile.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_a.name}")

    # -----------------------------------------------------------------------
    # Figure B: Alpha y beta en canales ROI — barplot por clase
    # -----------------------------------------------------------------------
    avail_roi = [ch for ch in ROI_CHANNELS if ch in ch_names]
    n_roi = len(avail_roi)
    alpha_idx = BAND_NAMES.index("alpha")
    beta_idx  = BAND_NAMES.index("beta")

    fig, axes = plt.subplots(2, n_roi, figsize=(n_roi * 2.8, 6.5), sharey=False)
    fig.suptitle(
        f"sub-{sub} — α y β en canales ROI: ¿cómo varía β a lo largo del tiempo?\n"
        "β > 0 (rojo): potencia alta → vota por esa clase  |  β < 0 (azul): potencia alta → vota CONTRA esa clase",
        fontsize=10,
    )

    for col, ch in enumerate(avail_roi):
        ch_idx = ch_names.index(ch)
        for row, (b_idx, b_name, b_color) in enumerate([
            (alpha_idx, "alpha", BAND_COLORS["alpha"]),
            (beta_idx,  "beta",  BAND_COLORS["beta"]),
        ]):
            vals = mat_34[:, ch_idx, b_idx]  # (4,)
            ax = axes[row][col]
            colors = [b_color if v > 0 else "lightsteelblue" for v in vals]
            ax.bar(range(4), vals, color=colors, edgecolor="k", linewidth=0.5)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(range(4))
            ax.set_xticklabels(CLASS_SHORT, fontsize=6)
            ax.set_title(f"{ch} — {b_name}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("β", fontsize=8)
            # Add value labels
            for i, v in enumerate(vals):
                ax.text(i, v + np.sign(v) * 0.02, f"{v:.2f}",
                        ha="center", va="bottom" if v > 0 else "top",
                        fontsize=6)

    plt.tight_layout()
    out_b = out_dir / f"sub-{sub}_39d_roi_alpha_beta.png"
    fig.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_b.name}")

    # -----------------------------------------------------------------------
    # Figure C: Mini heatmap banda × clase para cada canal ROI
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, n_roi, figsize=(n_roi * 2.5, 4.5))
    fig.suptitle(
        f"sub-{sub} — Heatmap (banda × clase) por canal ROI\n"
        "Rojo=β positivo (predice esa clase), Azul=β negativo",
        fontsize=11,
    )
    for col, ch in enumerate(avail_roi):
        ch_idx = ch_names.index(ch)
        mat_ch = mat_34[:, ch_idx, :].T  # (n_bands, 4) — bands on y, classes on x
        vmax = max(np.abs(mat_ch).max(), 1e-6)
        ax = axes[col]
        im = ax.imshow(mat_ch, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["BL", "CU", "LU", "CD"], fontsize=7)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(BAND_NAMES, fontsize=8)
        ax.set_title(ch, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.06, pad=0.03, label="β")
        # Add numerical values
        for r in range(n_bands):
            for c in range(4):
                val = mat_ch[r, c]
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)

    plt.tight_layout()
    out_c = out_dir / f"sub-{sub}_39d_roi_heatmap.png"
    fig.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_c.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
