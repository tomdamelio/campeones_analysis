#!/usr/bin/env python
"""Raw EEG continuo con eventos marcados — Sanity check 3.1b.

Plotea la señal cruda continua de un run completo (por defecto run-002,
que contiene tanto CHANGE_PHOTO como NO_CHANGE_PHOTO), con líneas
verticales de colores distintos para cada tipo de evento.

Canales mostrados: O1, O2 (occipital) + T7, T8 (temporal) + Fz (referencia).

Si el efecto es visible "a ojo desnudo" (criterio Diego), debería verse
una deflexión consistente en O1/O2 justo después de cada línea roja
(CHANGE_PHOTO) que no aparece en las verdes (NO_CHANGE_PHOTO).

Usage
-----
    micromamba run -n campeones python scripts/validation/38e_raw_continuous.py --subject 27 --run 002
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "sanity_checks"
SESSION = "vr"

RUNS_CONFIG = {
    "27": {
        "002": {"acq": "a", "task": "01"},
        "003": {"acq": "a", "task": "02"},
        "004": {"acq": "a", "task": "03"},
        "006": {"acq": "a", "task": "04"},
        "007": {"acq": "b", "task": "01"},
        "009": {"acq": "b", "task": "03"},
        "010": {"acq": "b", "task": "04"},
    }
}

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}
EVENT_COLORS = {"CHANGE_PHOTO": "C3", "NO_CHANGE_PHOTO": "C2"}

DISPLAY_CHANNELS = ["O1", "O2", "T7", "T8", "Fz"]
SCALE_UV = 100.0   # µV — rango visual por canal (±SCALE_UV)

# Tiempo máximo a mostrar en segundos (None = run completo)
MAX_DURATION_S = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    parser.add_argument("--run", default="002",
                        help="Run ID a visualizar (default: 002, tiene ambos tipos de evento)")
    args = parser.parse_args()
    sub, run_id = args.subject, args.run

    rc = RUNS_CONFIG.get(sub, {}).get(run_id)
    if rc is None:
        raise ValueError(f"No config for sub-{sub} run-{run_id}")
    task, acq = rc["task"], rc["acq"]

    eeg_dir = PREPROC_ROOT / f"sub-{sub}" / f"ses-{SESSION}" / "eeg"
    vhdr = eeg_dir / (
        f"sub-{sub}_ses-{SESSION}_task-{task}_acq-{acq}"
        f"_run-{run_id}_desc-preproc_eeg.vhdr"
    )
    tsv = (
        PHOTO_EVENTS_ROOT / f"sub-{sub}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{sub}_ses-{SESSION}_task-{task}_acq-{acq}"
          f"_run-{run_id}_desc-photo_events.tsv"
    )

    print(f"Loading sub-{sub} run-{run_id}...")
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # --- Events ---
    df = pd.read_csv(tsv, sep="\t")
    rows = df[df["trial_type"].isin(EVENT_ID)]
    onsets_s = rows["onset"].astype(float).values
    enames  = rows["trial_type"].values
    n_change    = (enames == "CHANGE_PHOTO").sum()
    n_no_change = (enames == "NO_CHANGE_PHOTO").sum()
    print(f"  Events: {n_change} CHANGE_PHOTO, {n_no_change} NO_CHANGE_PHOTO")

    # --- Channels ---
    avail_ch = [ch for ch in DISPLAY_CHANNELS if ch in raw.ch_names]
    if not avail_ch:
        raise RuntimeError("None of the display channels found in raw")
    print(f"  Channels: {avail_ch}")

    raw_pick = raw.copy().pick(avail_ch)
    data, times = raw_pick.get_data(return_times=True)  # (n_ch, n_times)
    data_uv = data * 1e6  # → µV

    # Trim to MAX_DURATION_S if set
    if MAX_DURATION_S is not None:
        mask = times <= MAX_DURATION_S
        data_uv = data_uv[:, mask]
        times = times[mask]

    n_ch = len(avail_ch)
    fig, axes = plt.subplots(n_ch, 1, figsize=(22, n_ch * 2.2),
                              sharex=True, sharey=False)
    if n_ch == 1:
        axes = [axes]

    fig.suptitle(
        f"sub-{sub}  run-{run_id} — Raw EEG continuo\n"
        f"CHANGE_PHOTO (rojo, n={n_change})  |  "
        f"NO_CHANGE_PHOTO (verde, n={n_no_change})\n"
        "Sanity check 3.1b: ¿efecto visible en señal continua?",
        fontsize=11,
    )

    for ax, ch_name, ch_data in zip(axes, avail_ch, data_uv):
        ax.plot(times, ch_data, color="0.3", lw=0.4, rasterized=True)
        ax.set_ylim(-SCALE_UV, SCALE_UV)
        ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=8, rotation=0,
                      labelpad=40, va="center")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

        # Draw event lines
        first_change = True
        first_nc = True
        for t, ename in zip(onsets_s, enames):
            if t > times[-1]:
                continue
            color = EVENT_COLORS[ename]
            lbl = None
            if ename == "CHANGE_PHOTO" and first_change:
                lbl = "CHANGE_PHOTO"
                first_change = False
            elif ename == "NO_CHANGE_PHOTO" and first_nc:
                lbl = "NO_CHANGE_PHOTO"
                first_nc = False
            ax.axvline(t, color=color, lw=0.9, alpha=0.75, label=lbl)

        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout(rect=[0.04, 0, 1, 1])

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{sub}_38e_raw_continuous_run{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
