"""One-off fix: align sub-27 merged_events run numbers to the regenerated raw/preproc.

Context (2026-05-27): sub-27 was reprocessed from scratch. read_xdf names runs from the
XDF source files, so sub-27's acq-b runs are 007-010 and task-04 acq-a kept run-006
(run-005 was the aborted take, dropped). But sub-27's merged_events were produced under an
OLDER run numbering (acq-b = 006-009; task-04 acq-a = 005). Paso 4 (04_preprocessing_eeg.py)
builds the merged_events path with the SAME run number as the raw, so 5 sub-27 runs failed
with FileNotFoundError on the merged_events .tsv.

The run number is just a sequential label; (task, acq) uniquely identifies the recording
(qa_viz.py already matches merged_events by task+acq, ignoring run, for exactly this reason).
Event onsets are in seconds, so the content is valid regardless of the run label.

Fix = COPY (non-destructive; originals kept) each affected merged_events .tsv/.json to the
run number the raw/preproc uses. Idempotent: skips if the destination already exists.

Run:
    micromamba run -n campeones python scripts/preprocessing/_fix_sub27_merged_events_runnums.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MERGED = REPO / "data" / "derivatives" / "merged_events" / "sub-27" / "ses-vr" / "eeg"

# (task, acq, old_run, new_run) -- only the 5 mismatches; acq-a tasks 01-03 already match.
MAPPING = [
    ("04", "a", "005", "006"),
    ("01", "b", "006", "007"),
    ("02", "b", "007", "008"),
    ("03", "b", "008", "009"),
    ("04", "b", "009", "010"),
]


def main() -> int:
    print(f"merged_events dir: {MERGED}")
    n_copied = 0
    n_skipped = 0
    n_missing = 0
    for task, acq, old_run, new_run in MAPPING:
        for ext in (".tsv", ".json"):
            src = MERGED / f"sub-27_ses-vr_task-{task}_acq-{acq}_run-{old_run}_desc-merged_events{ext}"
            dst = MERGED / f"sub-27_ses-vr_task-{task}_acq-{acq}_run-{new_run}_desc-merged_events{ext}"
            if not src.exists():
                print(f"  MISSING src (skip): {src.name}")
                n_missing += 1
                continue
            if dst.exists():
                print(f"  EXISTS  dst (skip): {dst.name}")
                n_skipped += 1
                continue
            shutil.copy2(src, dst)
            print(f"  COPIED  {src.name}  ->  {dst.name}")
            n_copied += 1
    print(f"\nDone. copied={n_copied} skipped={n_skipped} missing={n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
