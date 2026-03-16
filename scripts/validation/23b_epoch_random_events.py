#!/usr/bin/env python
"""Create MNE Epochs at RANDOM time points (control condition).

Implements Tarea 7 from the research diary.

For each run of a subject, generates random onsets uniformly distributed
across the recording (with minimum inter-onset distance = EPOCH_SPAN to
avoid overlap), then creates epochs with the same parameters as
23_epoch_photo_events.py.  This serves as a null-distribution control:
if the oscillatory patterns seen in NO_CHANGE ERPs also appear in these
random epochs, the effect is an artifact of averaging / alpha asymmetry
rather than a stimulus-related response.

Usage
-----
    micromamba run -n campeones python scripts/validation/23b_epoch_random_events.py --subject 27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_epochs"

SESSION = "vr"

# Match 23_epoch_photo_events.py exactly
TMIN = -4.5
TMAX = 3.0
BASELINE = (-4.5, -4.0)
EPOCH_SPAN = abs(TMIN) + TMAX  # 7.5 s

# How many random epochs to generate per run (will be capped by recording length)
TARGET_EPOCHS_PER_RUN = 40

RANDOM_SEED = 42
EVENT_ID = {"RANDOM": 99}

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

EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_eeg_path(subject: str, task: str, acq: str, run: str) -> Path | None:
    eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
    vhdr = eeg_dir / (
        f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
        f"_run-{run}_desc-preproc_eeg.vhdr"
    )
    return vhdr if vhdr.exists() else None


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_random_onsets(
    duration_s: float,
    n_target: int,
    epoch_span: float,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate non-overlapping random onsets within [margin, duration_s - margin].

    Parameters
    ----------
    duration_s : float
        Total recording duration in seconds.
    n_target : int
        Desired number of onsets.
    epoch_span : float
        Minimum distance between consecutive onsets (seconds).
    margin : float
        Safety margin from recording edges (seconds).  Should be >= |TMIN|
        so that the epoch window fits within the recording.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Sorted array of onset times (seconds).
    """
    lo = margin
    hi = duration_s - margin
    usable = hi - lo
    if usable <= 0:
        return np.array([])

    # Max epochs that fit without overlap
    max_epochs = int(usable // epoch_span)
    n = min(n_target, max_epochs)
    if n <= 0:
        return np.array([])

    # Strategy: sample n points in [0, usable - (n-1)*epoch_span],
    # sort them, then spread by adding i*epoch_span to the i-th point.
    # This guarantees minimum spacing = epoch_span.
    slack = usable - (n - 1) * epoch_span
    raw_pts = np.sort(rng.uniform(0, slack, size=n))
    onsets = lo + raw_pts + np.arange(n) * epoch_span

    return onsets


def create_random_epochs_for_run(
    eeg_raw: mne.io.Raw,
    n_target: int,
    rng: np.random.Generator,
) -> mne.Epochs | None:
    """Create epochs at random time points for one run."""
    sfreq = eeg_raw.info["sfreq"]
    duration_s = eeg_raw.n_times / sfreq

    available_chs = [ch for ch in EEG_CHANNELS if ch in eeg_raw.ch_names]
    if not available_chs:
        return None

    # Margin: need |TMIN| before onset and TMAX after onset
    margin = max(abs(TMIN), TMAX) + 0.5  # extra 0.5s safety

    onsets = generate_random_onsets(duration_s, n_target, EPOCH_SPAN, margin, rng)
    if len(onsets) == 0:
        return None

    # Convert to sample indices
    samples = np.round(onsets * sfreq).astype(int) + eeg_raw.first_samp

    mne_events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.full(len(samples), EVENT_ID["RANDOM"], dtype=int),
    ])

    try:
        epochs = mne.Epochs(
            eeg_raw,
            events=mne_events,
            event_id=EVENT_ID,
            tmin=TMIN,
            tmax=TMAX,
            picks=available_chs,
            baseline=BASELINE,
            preload=True,
            verbose=False,
        )
        return epochs
    except Exception as exc:
        print(f"    Could not create epochs: {exc}")
        return None


def run_pipeline(subject: str, seed: int = RANDOM_SEED) -> None:
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG for subject {subject}")
        sys.exit(1)

    print("=" * 60)
    print(f"23b — Random epochs (control) — sub-{subject}")
    print(f"      seed={seed}, target={TARGET_EPOCHS_PER_RUN}/run")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    all_epochs: list[mne.Epochs] = []

    for rc in runs:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        label = f"task-{task}_acq-{acq}_run-{run_id}"

        vhdr = resolve_eeg_path(subject, task, acq, run_id)
        if vhdr is None:
            print(f"  SKIP {label} — EEG not found")
            continue

        print(f"  Loading {label} ...")
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)

        epochs = create_random_epochs_for_run(raw, TARGET_EPOCHS_PER_RUN, rng)
        if epochs is None:
            print(f"    No random epochs for {label}")
            continue

        n_ep = len(epochs)
        print(f"    {label}: {n_ep} RANDOM epochs")
        all_epochs.append(epochs)

    if not all_epochs:
        print("No random epochs created.")
        return

    grand = mne.concatenate_epochs(all_epochs)
    n_total = len(grand)
    print(f"\n  Grand total: {n_total} RANDOM epochs")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ROOT / f"sub-{subject}_random-epo.fif"
    grand.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random control epochs",
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject, seed=args.seed)
