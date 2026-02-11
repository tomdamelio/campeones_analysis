"""Verify luminance stimulus markers in EEG events for sub-27.

For each run (Acq A and B), loads the events TSV and Order Matrix,
filters events with trial_type ``video_luminance``, cross-references
with the Order Matrix to determine the video_id mapping, and detects
discrepancies.

Generates a consolidated CSV report in
``results/modeling/luminance/verification/``.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# Ensure scripts/modeling is on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_luminance import (
    DERIVATIVES_PATH,
    LUMINANCE_CSV_MAP,
    PROJECT_ROOT,
    RESULTS_PATH,
    RUNS_CONFIG,
    SESSION,
    SUBJECT,
    XDF_PATH,
)


# ---------------------------------------------------------------------------
# Pure helper functions (testable, no I/O)
# ---------------------------------------------------------------------------


def filter_video_luminance_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter events DataFrame to keep only video_luminance trial_type rows.

    Args:
        events_df: DataFrame with at least a ``trial_type`` column.

    Returns:
        Filtered DataFrame containing only rows where
        ``trial_type == 'video_luminance'``.

    Requirements: 2.1
    """
    return events_df[events_df["trial_type"] == "video_luminance"].copy()


def extract_video_id_from_stim_id(stim_id: int) -> int:
    """Derive the luminance video_id from the encoded stim_id.

    The encoding convention is ``stim_id = 100 + video_id``
    (e.g. 112 → video 12, 103 → video 3).

    Args:
        stim_id: Integer stimulus identifier from the events TSV.

    Returns:
        The decoded video_id.
    """
    return stim_id - 100


def extract_video_id_from_stim_file(stim_file: str) -> int | None:
    """Extract the video_id from the stim_file path string.

    Expects a pattern like ``green_intensity_video_<N>.mp4``.

    Args:
        stim_file: Stimulus file path string from the events TSV.

    Returns:
        The extracted video_id, or ``None`` if the pattern is not found.
    """
    match = re.search(r"green_intensity_video_(\d+)", stim_file)
    if match:
        return int(match.group(1))
    return None



def build_expected_csv_filename(video_id: int) -> str | None:
    """Return the expected luminance CSV filename for a given video_id.

    Args:
        video_id: Integer video identifier (3, 7, 9, or 12).

    Returns:
        The CSV filename string, or ``None`` if the video_id is not in
        the luminance map.
    """
    return LUMINANCE_CSV_MAP.get(video_id)


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def load_events_tsv(events_path: Path) -> pd.DataFrame:
    """Load a BIDS events TSV file.

    Args:
        events_path: Path to the ``.tsv`` events file.

    Returns:
        DataFrame with event columns (onset, duration, trial_type, etc.).
    """
    return pd.read_csv(events_path, sep="\t")


def load_order_matrix(order_matrix_path: Path) -> pd.DataFrame:
    """Load an Order Matrix Excel file.

    Args:
        order_matrix_path: Path to the ``.xlsx`` Order Matrix file.

    Returns:
        DataFrame with columns: participant, session, modality,
        order_presentation, video_id, dimension, order_emojis_sn.
    """
    return pd.read_excel(order_matrix_path)


def _resolve_events_path(
    run_config: dict,
    events_base_dir: Path,
) -> Path | None:
    """Build the events TSV path for a run, trying merged first then regular.

    Args:
        run_config: Dictionary with keys ``id``, ``acq``, ``task``.
        events_base_dir: Base directory for events
            (e.g. ``data/derivatives/``).

    Returns:
        Path to the events TSV file, or ``None`` if not found.
    """
    run_id = run_config["id"]
    acq = run_config["acq"]
    task = run_config["task"]

    # Try merged events first (they have real onset times)
    merged_dir = (
        events_base_dir
        / "merged_events"
        / f"sub-{SUBJECT}"
        / f"ses-{SESSION}"
        / "eeg"
    )
    merged_name = (
        f"sub-{SUBJECT}_ses-{SESSION}_task-{task}"
        f"_acq-{acq}_run-{run_id}_desc-merged_events.tsv"
    )
    merged_path = merged_dir / merged_name
    if merged_path.exists():
        return merged_path

    # Fall back to regular events
    events_dir = (
        events_base_dir
        / "events"
        / f"sub-{SUBJECT}"
        / f"ses-{SESSION}"
        / "eeg"
    )
    events_name = (
        f"sub-{SUBJECT}_ses-{SESSION}_task-{task}"
        f"_acq-{acq}_run-{run_id}_events.tsv"
    )
    events_path = events_dir / events_name
    if events_path.exists():
        return events_path

    return None


def _resolve_order_matrix_path(
    run_config: dict,
    xdf_base_dir: Path,
) -> Path | None:
    """Build the Order Matrix path for a run.

    Args:
        run_config: Dictionary with keys ``acq``, ``block``.
        xdf_base_dir: Base directory for XDF/sourcedata.

    Returns:
        Path to the Order Matrix Excel file, or ``None`` if not found.
    """
    acq = run_config["acq"].upper()
    block = run_config["block"]
    subject_dir = xdf_base_dir / f"sub-{SUBJECT}"
    order_matrix_name = (
        f"order_matrix_{SUBJECT}_{acq}_{block}_VR.xlsx"
    )
    order_matrix_path = subject_dir / order_matrix_name
    if order_matrix_path.exists():
        return order_matrix_path
    return None



def verify_run(
    run_config: dict,
    events_df: pd.DataFrame,
    order_matrix_df: pd.DataFrame,
) -> list[dict]:
    """Verify luminance markers for a single run.

    Cross-references video_luminance events from the events TSV with the
    Order Matrix to determine the mapping and detect discrepancies.

    Args:
        run_config: Dictionary with run metadata (id, acq, task, block).
        events_df: Events DataFrame for this run.
        order_matrix_df: Order Matrix DataFrame for this block.

    Returns:
        List of report row dicts, one per video_luminance event found.

    Requirements: 2.1, 2.2, 2.3
    """
    luminance_events = filter_video_luminance_events(events_df)

    # Check if Order Matrix has a luminance dimension entry
    order_luminance = order_matrix_df[
        order_matrix_df["dimension"] == "luminance"
    ]
    order_has_luminance = not order_luminance.empty

    report_rows: list[dict] = []

    for _, event_row in luminance_events.iterrows():
        stim_id = int(event_row["stim_id"])
        video_id_from_stim_id = extract_video_id_from_stim_id(stim_id)
        video_id_from_stim_file = extract_video_id_from_stim_file(
            str(event_row.get("stim_file", ""))
        )

        # Determine the video_id (prefer stim_file, cross-check with stim_id)
        resolved_video_id = video_id_from_stim_id
        stim_id_stim_file_match = (
            video_id_from_stim_file is not None
            and video_id_from_stim_id == video_id_from_stim_file
        )

        # Check if this video_id is a known luminance video
        expected_csv = build_expected_csv_filename(resolved_video_id)
        is_known_luminance_video = expected_csv is not None

        # Determine match status
        if not order_has_luminance:
            match_status = "mismatch_no_luminance_in_order"
        elif not stim_id_stim_file_match:
            match_status = "mismatch_stim_id_vs_stim_file"
        elif not is_known_luminance_video:
            match_status = "mismatch_unknown_video_id"
        else:
            match_status = "ok"

        report_rows.append(
            {
                "run_id": run_config["id"],
                "acq": run_config["acq"],
                "task": run_config["task"],
                "block": run_config["block"],
                "event_onset": float(event_row["onset"]),
                "event_duration": float(event_row["duration"]),
                "event_stim_id": stim_id,
                "resolved_video_id": resolved_video_id,
                "video_id_from_stim_file": video_id_from_stim_file,
                "order_matrix_has_luminance": order_has_luminance,
                "csv_filename": expected_csv if expected_csv else "N/A",
                "match_status": match_status,
            }
        )

    return report_rows


def run_pipeline() -> None:
    """Execute the luminance marker verification pipeline.

    Iterates over all configured runs for sub-27, loads events and Order
    Matrix, verifies video_luminance markers, and saves a consolidated
    CSV report.

    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    print("=" * 60)
    print("09 — Verify Luminance Markers (sub-27)")
    print("=" * 60)

    output_dir = RESULTS_PATH / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    derivatives_base = PROJECT_ROOT / "data" / "derivatives"
    xdf_base = XDF_PATH

    all_report_rows: list[dict] = []

    for run_config in RUNS_CONFIG:
        run_label = (
            f"run-{run_config['id']} acq-{run_config['acq']} "
            f"task-{run_config['task']} ({run_config['block']})"
        )
        print(f"\nProcessing {run_label}")

        # Resolve events TSV
        events_path = _resolve_events_path(run_config, derivatives_base)
        if events_path is None:
            print(f"  WARNING: Events TSV not found, skipping.")
            continue
        print(f"  Events: {events_path.name}")
        events_df = load_events_tsv(events_path)

        # Resolve Order Matrix
        order_matrix_path = _resolve_order_matrix_path(run_config, xdf_base)
        if order_matrix_path is None:
            print(f"  WARNING: Order Matrix not found, skipping.")
            continue
        print(f"  Order Matrix: {order_matrix_path.name}")
        order_matrix_df = load_order_matrix(order_matrix_path)

        # Verify
        run_rows = verify_run(run_config, events_df, order_matrix_df)
        all_report_rows.extend(run_rows)

        for row in run_rows:
            status_icon = "✓" if row["match_status"] == "ok" else "✗"
            print(
                f"  {status_icon} stim_id={row['event_stim_id']} "
                f"→ video_id={row['resolved_video_id']} "
                f"[{row['match_status']}]"
            )

    # Save consolidated report
    if all_report_rows:
        report_df = pd.DataFrame(all_report_rows)
        report_path = output_dir / f"sub-{SUBJECT}_luminance_marker_verification.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\nReport saved: {report_path}")
        print(f"Total events verified: {len(report_df)}")
        n_ok = (report_df["match_status"] == "ok").sum()
        n_mismatch = len(report_df) - n_ok
        print(f"  OK: {n_ok}, Mismatches: {n_mismatch}")
    else:
        print("\nNo video_luminance events found across any run.")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
