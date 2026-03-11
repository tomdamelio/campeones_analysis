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
}

# How many seconds to skip at the start of fixation
FIXATION_SKIP_S: float = 5.0


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
) -> pd.DataFrame:
    """Sample NO_CHANGE_PHOTO windows inside a fixation period.

    Parameters
    ----------
    fixation_onset : float
        Onset of the fixation event (seconds).
    fixation_duration : float
        Duration of the fixation event (seconds).
    n_events : int
        How many NO_CHANGE_PHOTO events to place.
    skip_s : float
        Seconds to skip at the beginning of fixation.

    Returns
    -------
    pd.DataFrame
        DataFrame with NO_CHANGE_PHOTO rows.
    """
    if n_events <= 0:
        return pd.DataFrame(columns=["onset", "duration", "trial_type",
                                      "stim_id", "condition", "stim_file"])

    effective_onset = fixation_onset + skip_s
    available = fixation_duration - skip_s
    spacing = available / n_events

    rows: list[dict] = []
    for i in range(n_events):
        rows.append({
            "onset": effective_onset + i * spacing,
            "duration": 1.0,
            "trial_type": "NO_CHANGE_PHOTO",
            "stim_id": "n/a",
            "condition": "no_change_photo",
            "stim_file": "n/a",
        })
    return pd.DataFrame(rows)


def run_pipeline(subject: str, dry_run: bool = False) -> None:
    """Generate photo_events TSVs for all runs of a subject."""
    runs = RUNS_CONFIG.get(subject)
    if runs is None:
        print(f"No RUNS_CONFIG defined for subject {subject}. "
              "Add an entry to RUNS_CONFIG in this script.")
        sys.exit(1)

    print("=" * 60)
    print(f"22 — Generate photo events — sub-{subject}")
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
        )
        no_change_dfs[run_id] = nc_df
        print(f"  Run {run_id} (task-01 fixation): {n_here} NO_CHANGE_PHOTO "
              f"(spacing {float(fix_row['duration']) - FIXATION_SKIP_S:.1f}s / "
              f"{n_here} = "
              f"{(float(fix_row['duration']) - FIXATION_SKIP_S) / max(n_here, 1):.2f}s)")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject, dry_run=args.dry_run)
