#!/usr/bin/env python
"""Test de hipótesis: contaminación ERP en β de delta/theta (tarea 27b).

Hipótesis: los canales con β positivo en delta/theta en el clasificador 27b
tienen mayor amplitud de ERP en la ventana POST (+50→+550ms) respecto al PRE
(-550→-50ms), porque el VEP/AEP aparece como aumento de potencia de baja
frecuencia al aplicar Welch sobre la ventana POST.

Por cada canal:
  1. Carga épocas CHANGE_PHOTO y calcula el ERP promedio
  2. Computa la amplitud media absoluta en POST vs PRE
  3. Coteja con β_delta y β_theta del JSON generado por script 39

Outputs → results/validation/interpretability/sub-{sub}/
  39e_erp_vs_delta_beta.png    scatter por canal: ERP amplitude vs β_delta
  39e_erp_vs_theta_beta.png    idem para theta
  39e_erp_waveforms.png        ERP de los 6 canales con mayor δ β, comparado
                               con los 6 de menor δ β (para validación visual)

Usage
-----
    micromamba run -n campeones python scripts/validation/39e_erp_vs_delta_theta.py --subject 27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "interpretability"
SESSION = "vr"

RUNS_CONFIG = {
    "27": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "006", "acq": "a", "task": "04"},
        {"run": "007", "acq": "b", "task": "01"},
        {"run": "009", "acq": "b", "task": "03"},
        {"run": "010", "acq": "b", "task": "04"},
    ],
}
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]

# Windows matching 27b definition
PRE_START,  PRE_END  = -0.55, -0.05   # s
POST_START, POST_END =  0.05,  0.55   # s

# Anatomical region labels for coloring
REGION = {
    "Fp1": "frontal", "Fp2": "frontal",
    "F7": "frontal", "F3": "frontal", "Fz": "frontal", "F4": "frontal", "F8": "frontal",
    "FC5": "fronto-central", "FC1": "fronto-central", "FCz": "fronto-central",
    "FC2": "fronto-central", "FC6": "fronto-central",
    "FT9": "temporal", "T7": "temporal", "T8": "temporal", "FT10": "temporal",
    "C3": "central", "Cz": "central", "C4": "central",
    "TP9": "temporal", "TP10": "temporal",
    "CP5": "parietal", "CP1": "parietal", "CP2": "parietal", "CP6": "parietal",
    "P7": "parietal", "P3": "parietal", "Pz": "parietal", "P4": "parietal", "P8": "parietal",
    "O1": "occipital", "O2": "occipital",
}
REGION_COLORS = {
    "frontal": "#4C72B0",
    "fronto-central": "#7B9FC4",
    "central": "#55A868",
    "temporal": "#DD8452",
    "parietal": "#8172B2",
    "occipital": "#C44E52",
}


def load_change_epochs(subject: str) -> mne.Epochs:
    """Load all CHANGE_PHOTO epochs, tmin=-1.5, tmax=1.5, baseline=(-1.5,-1.0)."""
    all_ep = []
    for rc in RUNS_CONFIG[subject]:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
        vhdr = eeg_dir / (
            f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
            f"_run-{run_id}_desc-preproc_eeg.vhdr"
        )
        tsv = (
            PHOTO_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
            / f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
              f"_run-{run_id}_desc-photo_events.tsv"
        )
        if not vhdr.exists() or not tsv.exists():
            continue
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"] == "CHANGE_PHOTO"]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
                  + raw.first_samp)
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets = onsets[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int),
                                   np.ones(len(onsets), int)])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        try:
            ep = mne.Epochs(raw, mne_ev, event_id={"CHANGE_PHOTO": 1},
                            tmin=-1.5, tmax=1.5, picks=avail,
                            baseline=(-1.5, -1.0), preload=True, verbose=False)
            all_ep.append(ep)
        except Exception as e:
            print(f"  run-{run_id}: {e}")
    if not all_ep:
        raise RuntimeError("No epochs loaded")
    return mne.concatenate_epochs(all_ep, verbose=False)


def erp_amplitude_increase(epochs: mne.Epochs) -> dict[str, float]:
    """
    Por cada canal: media absoluta del ERP en POST - media absoluta en PRE.
    Retorna dict canal → amplitud_increase (µV).
    """
    erp = epochs.average()   # EvokedArray: mean over trials
    times = erp.times
    data_uv = erp.data * 1e6  # (n_ch, n_times), µV

    pre_mask  = (times >= PRE_START)  & (times <= PRE_END)
    post_mask = (times >= POST_START) & (times <= POST_END)

    result = {}
    for ch_idx, ch_name in enumerate(erp.ch_names):
        pre_amp  = np.mean(np.abs(data_uv[ch_idx, pre_mask]))
        post_amp = np.mean(np.abs(data_uv[ch_idx, post_mask]))
        result[ch_name] = post_amp - pre_amp   # positive = ERP increased in POST
    return result


def scatter_erp_vs_coef(ax, erp_increase: dict, coef_per_ch: dict,
                         band: str, ch_names: list[str]) -> float:
    """Scatter ERP amplitude increase vs β coefficient. Returns Pearson r."""
    x, y, labels, colors = [], [], [], []
    for ch in ch_names:
        if ch in erp_increase and ch in coef_per_ch:
            x.append(erp_increase[ch])
            y.append(coef_per_ch[ch])
            labels.append(ch)
            reg = REGION.get(ch, "central")
            colors.append(REGION_COLORS.get(reg, "gray"))

    x, y = np.array(x), np.array(y)
    r, p = pearsonr(x, y)

    for xi, yi, lbl, c in zip(x, y, labels, colors):
        ax.scatter(xi, yi, color=c, s=50, alpha=0.8, zorder=3)
        # label outliers (top/bottom 25% by |β|)
        if abs(yi) >= np.percentile(np.abs(y), 75):
            ax.annotate(lbl, (xi, yi), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")

    # Regression line
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, "k--", lw=1.2, alpha=0.6)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("ERP amplitude increase: POST − PRE (µV)", fontsize=9)
    ax.set_ylabel(f"β_{band} coefficient", fontsize=9)
    ax.set_title(f"{band.capitalize()} β vs ERP amplitude\nr = {r:.3f}, p = {p:.3f}",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend for regions
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=reg)
        for reg, c in REGION_COLORS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left", framealpha=0.7)
    return r


def plot_erp_waveforms(ax_arr, epochs, coef_per_ch, ch_names, band, n_top=5):
    """
    ERP de los n_top canales con mayor β_delta (rojo) vs n_top con menor β (azul).
    Muestra si los de mayor β son los que tienen VEP grande.
    """
    sorted_chs = sorted(
        [(ch, coef_per_ch[ch]) for ch in ch_names if ch in coef_per_ch],
        key=lambda x: x[1],
    )
    bottom_chs = [ch for ch, _ in sorted_chs[:n_top]]   # más negativos (β bajo)
    top_chs    = [ch for ch, _ in sorted_chs[-n_top:]]  # más positivos (β alto)

    erp_data = epochs.average().data * 1e6  # (n_ch, n_t) µV
    times_ms = epochs.average().times * 1000
    all_ch = epochs.ch_names

    for ax, chs, color, label in [
        (ax_arr[0], top_chs,    "C3", f"Top-{n_top} canales β_{band} positivo"),
        (ax_arr[1], bottom_chs, "C0", f"Top-{n_top} canales β_{band} negativo"),
    ]:
        for ch in chs:
            if ch in all_ch:
                idx = all_ch.index(ch)
                ax.plot(times_ms, erp_data[idx], lw=1.2, alpha=0.7,
                        color=color, label=ch)
        ax.axvspan(PRE_START * 1000,  PRE_END * 1000,
                   alpha=0.12, color="steelblue", zorder=0)
        ax.axvspan(POST_START * 1000, POST_END * 1000,
                   alpha=0.12, color="tomato", zorder=0)
        ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Amplitude (µV)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-600, 700)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    json_path = out_dir / f"sub-{sub}_39_coefs.json"

    print("Loading coefs...")
    with open(json_path) as f:
        data = json.load(f)
    coef_27b = np.array(data["task_27b"]["coef"])   # (n_feat,)
    ch_names = data["task_27b"]["ch_names"]
    band_names = data["task_27b"]["band_names"]
    n_ch = len(ch_names)
    n_bands = len(band_names)
    mat_27b = coef_27b.reshape(n_ch, n_bands)   # (n_ch, n_bands)

    # Per-channel coefficient for delta and theta
    delta_idx = band_names.index("delta")
    theta_idx = band_names.index("theta")
    coef_delta = {ch: mat_27b[i, delta_idx] for i, ch in enumerate(ch_names)}
    coef_theta = {ch: mat_27b[i, theta_idx] for i, ch in enumerate(ch_names)}

    print("Loading epochs and computing ERP...")
    epochs = load_change_epochs(sub)
    print(f"  {len(epochs)} CHANGE_PHOTO epochs loaded, {len(epochs.ch_names)} channels")
    erp_inc = erp_amplitude_increase(epochs)

    # Print top channels by delta β
    sorted_delta = sorted(coef_delta.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Top-10 canales por β_delta (positivo = predice POST) ---")
    for ch, val in sorted_delta[:10]:
        print(f"  {ch:<6} β_delta={val:+.4f}  ERP_increase={erp_inc.get(ch, 0):.3f} µV")

    # -----------------------------------------------------------------------
    # Figure: 2×2 scatter + waveforms
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"sub-{sub} — Hipótesis contaminación ERP en β de baja frecuencia (27b)\n"
        "¿Los canales con β_delta/θ alto son los que tienen mayor VEP en POST?",
        fontsize=11,
    )

    # Scatter delta (top-left)
    ax_sc_d = fig.add_subplot(2, 2, 1)
    r_delta = scatter_erp_vs_coef(ax_sc_d, erp_inc, coef_delta, "delta", ch_names)

    # Scatter theta (top-right)
    ax_sc_t = fig.add_subplot(2, 2, 2)
    r_theta = scatter_erp_vs_coef(ax_sc_t, erp_inc, coef_theta, "theta", ch_names)

    # ERP waveforms — top vs bottom beta_delta channels (bottom row)
    ax_w1 = fig.add_subplot(2, 2, 3)
    ax_w2 = fig.add_subplot(2, 2, 4)
    plot_erp_waveforms([ax_w1, ax_w2], epochs, coef_delta, ch_names, "delta", n_top=5)

    plt.tight_layout()
    out = out_dir / f"sub-{sub}_39e_erp_vs_delta_theta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out.name}")
    print(f"\nCorrelación ERP_increase vs β_delta: r = {r_delta:.3f}")
    print(f"Correlación ERP_increase vs β_theta: r = {r_theta:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
