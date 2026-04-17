#!/usr/bin/env python
"""Generate CHANGE_PHOTO and NO_CHANGE_PHOTO event TSVs.

Implements Task 1 (sub-tasks 1.1–1.3) from the research diary:

- CHANGE_PHOTO: 1-second photodiode flicker marks that sandwich every event
  in the merged_events TSVs.  For each event row the script emits:
    * PRE  mark: onset = event.onset - 1.0,  duration = 1.0
    * POST mark: onset = event.onset + event.duration, duration = 1.0

- NO_CHANGE_PHOTO: 1-second windows equidistantly sampled inside the
  fixation baseline (~300 s) of task-01 blocks only (acq-a and acq-b).
  The first 5 seconds of fixation are discarded to avoid novelty effects.
  The total count of NO_CHANGE_PHOTO equals the total count of CHANGE_PHOTO.

Output
------
One TSV per run in ``data/derivatives/photo_events/sub-{ID}/ses-vr/eeg/``.
Each TSV keeps the original merged_events rows **plus** the new
CHANGE_PHOTO / NO_CHANGE_PHOTO rows, sorted by onset.

Usage
-----
    python scripts/validation/22_generate_photo_events.py --subject 27
    python scripts/validation/22_generate_photo_events.py --subject 27 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MERGED_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "merged_events"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"

SESSION = "vr"

# Runs config — mirrors config_luminance.py but kept local so this script
# has no dependency on the modeling config.
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
    "23": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "005", "acq": "a", "task": "04"},
        {"run": "006", "acq": "b", "task": "01"},
        {"run": "007", "acq": "b", "task": "02"},
        {"run": "008", "acq": "b", "task": "03"},
        {"run": "009", "acq": "b", "task": "04"},
    ],
    "24": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "005", "acq": "a", "task": "04"},
        {"run": "006", "acq": "b", "task": "01"},
        {"run": "007", "acq": "b", "task": "02"},
        {"run": "008", "acq": "b", "task": "03"},
        {"run": "009", "acq": "b", "task": "04"},
    ],
    "33": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "005", "acq": "a", "task": "04"},
        {"run": "006", "acq": "b", "task": "01"},
        {"run": "007", "acq": "b", "task": "02"},
        {"run": "008", "acq": "b", "task": "03"},
        {"run": "009", "acq": "b", "task": "04"},
    ],
}

# How many seconds to skip at the start of fixation
FIXATION_SKIP_S: float = 5.0

# Epoch span for overlap check: |TMIN| + TMAX from 23_epoch_photo_events.py
# Must match the epoch window used in epoching to prevent overlap.
EPOCH_SPAN_S: float = 7.5  # abs(-4.5) + 3.0

# Jitter parameters
JITTER_FRACTION: float = 0.3  # ±30% of spacing
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_merged_events_path(subject: str, task: str, acq: str, run: str) -> Path:
    """Return the path to a merged_events TSV for a given run."""
    filename = (
        f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
        f"_run-{run}_desc-merged_events.tsv"
    )
    return (
        MERGED_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg" / filename
    )


def resolve_photo_events_path(subject: str, task: str, acq: str, run: str) -> Path:
    """Return the output path for a photo_events TSV."""
    filename = (
        f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
        f"_run-{run}_desc-photo_events.tsv"
    )
    return (
        PHOTO_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg" / filename
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_change_photo_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Create CHANGE_PHOTO rows for every event in a merged_events DataFrame.

    For each event row, two CHANGE_PHOTO marks are generated:
    - PRE:  onset = event.onset - 1.0,  duration = 1.0
    - POST: onset = event.onset + event.duration, duration = 1.0

    Parameters
    ----------
    events_df : pd.DataFrame
        A merged_events DataFrame with at least ``onset`` and ``duration``.

    Returns
    -------
    pd.DataFrame
        DataFrame with CHANGE_PHOTO rows only.
    """
    rows: list[dict] = []
    for _, ev in events_df.iterrows():
        onset = float(ev["onset"])
        duration = float(ev["duration"])
        base = {
            "trial_type": "CHANGE_PHOTO",
            "stim_id": "n/a",
            "condition": "change_photo",
            "stim_file": "n/a",
            "duration": 1.0,
        }
        # PRE mark
        rows.append({**base, "onset": onset - 1.0})
        # POST mark
        rows.append({**base, "onset": onset + duration})
    return pd.DataFrame(rows)


def generate_no_change_photo_events(
    fixation_onset: float,
    fixation_duration: float,
    n_events: int,
    skip_s: float = FIXATION_SKIP_S,
    epoch_span_s: float = EPOCH_SPAN_S,
    jitter_fraction: float = JITTER_FRACTION,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample NO_CHANGE_PHOTO windows inside a fixation period.

    Events are placed at nominally equidistant positions and then jittered
    by a random offset to break phase synchrony.  The number of events is
    capped so that full epochs (of width ``epoch_span_s``) do not overlap.

    Parameters
    ----------
    fixation_onset : float
        Onset of the fixation event (seconds).
    fixation_duration : float
        Duration of the fixation event (seconds).
    n_events : int
        Requested number of NO_CHANGE_PHOTO events.  Will be reduced if
        they would cause epoch overlap.
    skip_s : float
        Seconds to skip at the beginning of fixation.
    epoch_span_s : float
        Total epoch duration (|TMIN| + TMAX) used to check for overlap.
    jitter_fraction : float
        Fraction of the nominal spacing used as maximum jitter range
        (±jitter_fraction * spacing).
    rng : numpy.random.Generator or None
        Random number generator for jitter.  If None, no jitter is applied
        (deterministic equidistant placement).

    Returns
    -------
    pd.DataFrame
        DataFrame with NO_CHANGE_PHOTO rows.
    """
    empty = pd.DataFrame(columns=["onset", "duration", "trial_type",
                                   "stim_id", "condition", "stim_file"])
    if n_events <= 0:
        return empty

    effective_onset = fixation_onset + skip_s
    available = fixation_duration - skip_s

    # Cap n_events so that epochs don't overlap
    max_no_overlap = int(available // epoch_span_s)
    if max_no_overlap <= 0:
        print(f"    WARNING: fixation too short for any epoch "
              f"(available={available:.1f}s, epoch_span={epoch_span_s:.1f}s)")
        return empty

    if n_events > max_no_overlap:
        print(f"    Capping NO_CHANGE from {n_events} to {max_no_overlap} "
              f"to avoid epoch overlap (epoch_span={epoch_span_s:.1f}s)")
        n_events = max_no_overlap

    spacing = available / n_events

    # Nominal equidistant onsets
    nominal_onsets = np.array([effective_onset + i * spacing
                               for i in range(n_events)])

    # Apply jitter if rng is provided
    if rng is not None and jitter_fraction > 0:
        max_jitter = spacing * jitter_fraction
        jitter = rng.uniform(-max_jitter, max_jitter, size=n_events)
        jittered = nominal_onsets + jitter

        # Enforce boundaries
        jittered = np.clip(jittered, effective_onset,
                           fixation_onset + fixation_duration - 1.0)

        # Enforce minimum inter-onset distance = epoch_span_s
        # Greedy forward pass: if an onset is too close to the previous,
        # push it forward to the minimum allowed position.
        for i in range(1, len(jittered)):
            min_allowed = jittered[i - 1] + epoch_span_s
            if jittered[i] < min_allowed:
                jittered[i] = min_allowed

        # Drop any events pushed beyond the fixation boundary
        end_limit = fixation_onset + fixation_duration - 1.0
        jittered = jittered[jittered <= end_limit]
        onsets = jittered
    else:
        onsets = nominal_onsets

    rows: list[dict] = []
    for onset in onsets:
        rows.append({
            "onset": float(onset),
            "duration": 1.0,
            "trial_type": "NO_CHANGE_PHOTO",
            "stim_id": "n/a",
            "condition": "no_change_photo",
            "stim_file": "n/a",
        })
    return pd.DataFrame(rows)


def run_pipeline(subject: str, dry_run: bool = False, seed: int = RANDOM_SEED) -> None:
    """Generate photo_events TSVs for all runs of a subject."""
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG defined for subject {subject}. "
              "Add an entry to RUNS_CONFIG in this script.")
        sys.exit(1)

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print(f"22 — Generate photo events — sub-{subject}")
    print(f"     seed={seed}  jitter_fraction={JITTER_FRACTION}  "
          f"epoch_span={EPOCH_SPAN_S}s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Pass 1: collect CHANGE_PHOTO across ALL runs and count them
    # ------------------------------------------------------------------
    all_change_dfs: dict[str, pd.DataFrame] = {}   # key = run id
    all_merged_dfs: dict[str, pd.DataFrame] = {}
    total_change_count = 0

    for rc in runs:
        merged_path = resolve_merged_events_path(
            subject, rc["task"], rc["acq"], rc["run"],
        )
        if not merged_path.exists():
            print(f"  SKIP {merged_path.name} — file not found")
            continue

        merged_df = pd.read_csv(merged_path, sep="\t")
        change_df = generate_change_photo_events(merged_df)

        all_merged_dfs[rc["run"]] = merged_df
        all_change_dfs[rc["run"]] = change_df
        total_change_count += len(change_df)

        print(f"  {merged_path.name}: {len(merged_df)} events "
              f"-> {len(change_df)} CHANGE_PHOTO")

    print(f"\n  Total CHANGE_PHOTO: {total_change_count}")

    # ------------------------------------------------------------------
    # Pass 2: generate NO_CHANGE_PHOTO from task-01 fixation blocks
    # ------------------------------------------------------------------
    task01_runs = [rc for rc in runs if rc["task"] == "01"]
    n_task01_blocks = len(task01_runs)

    if n_task01_blocks == 0:
        print("  WARNING: no task-01 runs found — cannot generate NO_CHANGE_PHOTO")
        no_change_per_block = 0
    else:
        # Split evenly across the available fixation blocks
        no_change_per_block = total_change_count // n_task01_blocks
        remainder = total_change_count % n_task01_blocks

    no_change_dfs: dict[str, pd.DataFrame] = {}

    for idx, rc in enumerate(task01_runs):
        run_id = rc["run"]
        merged_df = all_merged_dfs.get(run_id)
        if merged_df is None:
            continue

        fix_rows = merged_df[merged_df["trial_type"] == "fixation"]
        if fix_rows.empty:
            print(f"  WARNING: no fixation event in run {run_id}")
            continue

        fix_row = fix_rows.iloc[0]
        # Give the remainder to the first block
        n_here = no_change_per_block + (remainder if idx == 0 else 0)

        nc_df = generate_no_change_photo_events(
            fixation_onset=float(fix_row["onset"]),
            fixation_duration=float(fix_row["duration"]),
            n_events=n_here,
            rng=rng,
        )
        no_change_dfs[run_id] = nc_df
        actual_count = len(nc_df)
        print(f"  Run {run_id} (task-01 fixation): {actual_count} NO_CHANGE_PHOTO "
              f"(requested {n_here}, "
              f"available {float(fix_row['duration']) - FIXATION_SKIP_S:.1f}s, "
              f"jitter=±{JITTER_FRACTION * 100:.0f}%)")

    total_no_change = sum(len(df) for df in no_change_dfs.values())
    print(f"\n  Total NO_CHANGE_PHOTO: {total_no_change}")
    print(f"  Balance: {total_change_count} CHANGE vs {total_no_change} NO_CHANGE")

    # ------------------------------------------------------------------
    # Pass 3: merge and write output TSVs
    # ------------------------------------------------------------------
    if dry_run:
        print("\n  [DRY RUN] — no files written.")
        return

    columns = ["onset", "duration", "trial_type", "stim_id",
               "condition", "stim_file"]

    for rc in runs:
        run_id = rc["run"]
        merged_df = all_merged_dfs.get(run_id)
        if merged_df is None:
            continue

        parts = [merged_df]
        if run_id in all_change_dfs:
            parts.append(all_change_dfs[run_id])
        if run_id in no_change_dfs:
            parts.append(no_change_dfs[run_id])

        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values("onset").reset_index(drop=True)
        # Ensure column order
        combined = combined[columns]

        out_path = resolve_photo_events_path(
            subject, rc["task"], rc["acq"], rc["run"],
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, sep="\t", index=False)
        n_change = len(all_change_dfs.get(run_id, []))
        n_nc = len(no_change_dfs.get(run_id, pd.DataFrame()))
        print(f"  Wrote {out_path.name}  "
              f"({len(merged_df)} orig + {n_change} CHANGE + {n_nc} NO_CHANGE "
              f"= {len(combined)} total)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CHANGE_PHOTO / NO_CHANGE_PHOTO event TSVs",
    )
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject ID, e.g. '27'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for jitter (default: {RANDOM_SEED})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject, dry_run=args.dry_run, seed=args.seed)
