#!/usr/bin/env python
"""Visualize CHANGE_PHOTO / NO_CHANGE_PHOTO annotations on raw EEG.

Implements Task 1.5 from the research diary.

For each run of a subject, loads the preprocessed EEG together with the
photo_events TSV and opens the MNE interactive viewer showing:
- AUDIO and PHOTO channels (to confirm flicker alignment)
- A few EEG channels (O1, O2, Pz)
- Colour-coded annotations for CHANGE_PHOTO and NO_CHANGE_PHOTO

Usage (must be run interactively — needs a display)
-----
    python scripts/validation/24_visualize_photo_epochs.py --subject 27
    python scripts/validation/24_visualize_photo_epochs.py --subject 27 --run 002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"

SESSION = "vr"

RUNS_CONFIG: dict[str, list[dict]] = {
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

# Channels to display
VIS_CHANNELS = ["AUDIO", "PHOTO", "O1", "O2", "Pz"]


def resolve_eeg_path(subject: str, task: str, acq: str, run: str) -> Path | None:
    eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
    vhdr = eeg_dir / (
        f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
        f"_run-{run}_desc-preproc_eeg.vhdr"
    )
    return vhdr if vhdr.exists() else None


def resolve_photo_events_path(subject: str, task: str, acq: str, run: str) -> Path | None:
    tsv = (
        PHOTO_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
          f"_run-{run}_desc-photo_events.tsv"
    )
    return tsv if tsv.exists() else None


def events_to_annotations(events_df: pd.DataFrame, raw: mne.io.Raw) -> mne.Annotations:
    """Convert CHANGE_PHOTO / NO_CHANGE_PHOTO rows to MNE Annotations."""
    photo_rows = events_df[
        events_df["trial_type"].isin(["CHANGE_PHOTO", "NO_CHANGE_PHOTO"])
    ]
    onsets = photo_rows["onset"].astype(float).values
    durations = photo_rows["duration"].astype(float).values
    descriptions = photo_rows["trial_type"].values
    return mne.Annotations(onset=onsets, duration=durations,
                           description=descriptions,
                           orig_time=raw.info["meas_date"])


def apply_zscore(raw: mne.io.Raw) -> mne.io.Raw:
    """Z-score all channels for better visualisation."""
    data = raw.get_data()
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    data = (data - means) / stds
    raw._data = data
    return raw


def visualize_run(subject: str, rc: dict) -> None:
    """Open the MNE interactive viewer for one run."""
    run_id, task, acq = rc["run"], rc["task"], rc["acq"]
    label = f"task-{task}_acq-{acq}_run-{run_id}"

    vhdr = resolve_eeg_path(subject, task, acq, run_id)
    if vhdr is None:
        print(f"  SKIP {label} — EEG not found")
        return

    tsv_path = resolve_photo_events_path(subject, task, acq, run_id)
    if tsv_path is None:
        print(f"  SKIP {label} — photo_events TSV not found")
        return

    print(f"\n  Loading {label} ...")
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    events_df = pd.read_csv(tsv_path, sep="\t")

    # Pick only the channels we want to display
    available = [ch for ch in VIS_CHANNELS if ch in raw.ch_names]
    if not available:
        print(f"  SKIP {label} — none of {VIS_CHANNELS} found")
        return

    raw_vis = raw.copy().pick_channels(available)
    raw_vis = apply_zscore(raw_vis)

    # Downsample for smoother UI if sfreq is high
    if raw_vis.info["sfreq"] > 1000:
        raw_vis.resample(1000.0)

    # Set annotations
    annots = events_to_annotations(events_df, raw)
    raw_vis.set_annotations(annots)

    n_change = (events_df["trial_type"] == "CHANGE_PHOTO").sum()
    n_nc = (events_df["trial_type"] == "NO_CHANGE_PHOTO").sum()
    title = (f"sub-{subject} {label}  |  "
             f"{n_change} CHANGE_PHOTO  {n_nc} NO_CHANGE_PHOTO")

    print(f"  Opening viewer: {title}")
    print("  (close the window to continue to the next run)\n")

    scalings = {ch_type: 2.0 for ch_type in set(raw_vis.get_channel_types())}
    raw_vis.plot(
        title=title,
        scalings=scalings,
        duration=30,
        start=0,
        show=True,
        block=True,
    )


def run_pipeline(subject: str, single_run: str | None = None) -> None:
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    # Try to use Qt backend for interactive plots
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        print("Qt5Agg not available, using default backend")

    print("=" * 60)
    print(f"24 — Visualize photo epochs — sub-{subject}")
    print("=" * 60)

    for rc in runs:
        if single_run and rc["run"] != single_run:
            continue
        visualize_run(subject, rc)

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CHANGE_PHOTO / NO_CHANGE_PHOTO on raw EEG",
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--run", type=str, default=None,
                        help="Visualize only this run (e.g. '002')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject, single_run=args.run)
