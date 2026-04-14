#!/usr/bin/env python
"""Raw EEG interactivo con eventos marcados.

Abre raw.plot() de MNE con ventana interactiva. Navegar con:
  - Flechas ← → : avanzar/retroceder en el tiempo
  - + / -       : aumentar/reducir la escala de amplitud
  - Page Up/Down: zoom temporal

CHANGE_PHOTO  → línea roja
NO_CHANGE_PHOTO → línea verde

Usage
-----
    micromamba run -n campeones python scripts/validation/38f_raw_interactive.py --subject 27 --run 002
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
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
EVENT_COLOR = {1: "red", 2: "green"}

DISPLAY_CHANNELS = ["O1", "O2", "T7", "T8", "Fz"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    parser.add_argument("--run", default="002")
    args = parser.parse_args()
    sub, run_id = args.subject, args.run

    rc = RUNS_CONFIG[sub][run_id]
    task, acq = rc["task"], rc["acq"]

    vhdr = (
        PREPROC_ROOT / f"sub-{sub}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{sub}_ses-{SESSION}_task-{task}_acq-{acq}"
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

    df = pd.read_csv(tsv, sep="\t")
    rows = df[df["trial_type"].isin(EVENT_ID)]
    onsets = (np.round(rows["onset"].astype(float).values * sfreq).astype(int)
              + raw.first_samp)
    eids = rows["trial_type"].map(EVENT_ID).values
    valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
    onsets, eids = onsets[valid], eids[valid]

    mne_events = np.column_stack([onsets, np.zeros(len(onsets), int), eids])
    mne_events = mne_events[np.argsort(mne_events[:, 0])]

    n_c  = (eids == 1).sum()
    n_nc = (eids == 2).sum()
    print(f"  CHANGE_PHOTO: {n_c}  |  NO_CHANGE_PHOTO: {n_nc}")
    print()
    print("=== QUÉ BUSCAR ===")
    print("1. O1/O2 — ¿ves una deflexión (~5-15 µV) en los 200-400ms DESPUÉS de")
    print("   cada línea ROJA (CHANGE) que NO aparece después de las VERDES?")
    print("   → Si sí: señal visual presente a nivel single-trial")
    print("   → Si no: señal solo visible en promedio (SNR baja, normal)")
    print()
    print("2. Inicio del recording (~primeros 60-80s):")
    print("   ¿La amplitud es mayor al principio? ¿Los NO_CHANGE verdes están")
    print("   agrupados ahí? → posible confound temporal en el clasificador")
    print()
    print("3. ¿Hay diferencia sistemática entre O1/O2 (visual) y T7/T8 (auditivo)?")
    print("   T7/T8 deberían responder más rápido (~100ms) si el tono auditivo")
    print("   domina; O1/O2 más tarde (~200ms) si es visual.")
    print()
    print("Navegación: ← → (tiempo)  |  + - (amplitud)  |  cierra ventana para salir")
    print("=" * 40)

    avail = [ch for ch in DISPLAY_CHANNELS if ch in raw.ch_names]
    raw.plot(
        events=mne_events,
        event_id=EVENT_ID,
        event_color=EVENT_COLOR,
        picks=avail,
        scalings={"eeg": 50e-6},   # ±50 µV por canal
        duration=30.0,             # 30s por ventana
        show=True,
        block=True,
        title=f"sub-{sub} run-{run_id} — CHANGE=rojo | NO_CHANGE=verde",
    )


if __name__ == "__main__":
    main()
