"""EEG Quality Assurance with Autoreject (Script 16).

Loads preprocessed EEG for all subjects by run, applies AutoReject to quantify
epoch rejection rates per run and per video, generates rejection heatmaps,
and saves a CSV report + JSON sidecar to results/qa/eeg/.

Usage:
    micromamba run -n campeones python scripts/qa/16_eeg_qa_autoreject.py

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR.parent / "modeling"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from campeones_analysis.luminance.qa import plot_rejection_heatmap, run_autoreject_qa
from config_luminance import (
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    EXPERIMENTAL_VIDEOS,
    RANDOM_SEED,
    SESSION,
    SUBJECT,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eeg_qa_autoreject")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
QA_OUTPUT_DIR: Path = _PROJECT_ROOT / "results" / "qa" / "eeg"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_events_path(vhdr_path: Path, subject: str, task: str, acq: str, run_id: str) -> Path | None:
    merged_dir = _PROJECT_ROOT / "data" / "derivatives" / "merged_events" / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
    merged_name = f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}_run-{run_id}_desc-merged_events.tsv"
    merged_path = merged_dir / merged_name
    if merged_path.exists():
        return merged_path

    events_dir = DERIVATIVES_PATH / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
    events_name = f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}_run-{run_id}_desc-preproc_events.tsv"
    events_path = events_dir / events_name
    return events_path if events_path.exists() else None

def load_eeg_raw(vhdr_path: Path) -> mne.io.Raw:
    return mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

def load_events_df(events_path: Path) -> pd.DataFrame:
    return pd.read_csv(events_path, sep="\t")


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def extract_epochs_for_video_segment(
    eeg_raw: mne.io.Raw,
    onset_s: float,
    duration_s: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> mne.Epochs | None:
    try:
        video_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)
    except ValueError as exc:
        logger.warning(
            "Could not crop EEG [%.2f, %.2f]: %s", onset_s, onset_s + duration_s, exc
        )
        return None

    sfreq = video_raw.info["sfreq"]
    n_samples_segment = video_raw.n_times
    n_samples_epoch = int(round(epoch_duration_s * sfreq))
    n_samples_step = int(round(epoch_step_s * sfreq))

    if n_samples_segment < n_samples_epoch:
        return None

    epoch_onset_samples = np.arange(
        0, n_samples_segment - n_samples_epoch + 1, n_samples_step
    )
    if len(epoch_onset_samples) == 0:
        return None

    events_array = np.column_stack(
        [
            epoch_onset_samples + video_raw.first_samp,
            np.zeros(len(epoch_onset_samples), dtype=int),
            np.ones(len(epoch_onset_samples), dtype=int),
        ]
    )
    
    from config import EEG_CHANNELS
    eeg_picks = mne.pick_channels(video_raw.info["ch_names"], include=EEG_CHANNELS)
    epochs = mne.Epochs(
        video_raw,
        events=events_array,
        event_id=1,
        tmin=0.0,
        tmax=epoch_duration_s - 1.0 / sfreq,
        picks=eeg_picks,
        baseline=None,
        preload=True,
        verbose=False,
    )
    bvef_path = _PROJECT_ROOT / "scripts" / "preprocessing" / "BC-32_FCz_modified.bvef"
    montage = mne.channels.read_custom_montage(str(bvef_path))
    epochs.set_montage(montage, on_missing="ignore")
    return epochs if len(epochs) > 0 else None


# ---------------------------------------------------------------------------
# Per-run QA logic
# ---------------------------------------------------------------------------

def run_qa_for_run(vhdr_path: Path, subject: str, task: str, acq: str, run_id: str) -> list[dict]:
    events_path = resolve_events_path(vhdr_path, subject, task, acq, run_id)
    if events_path is None:
        logger.warning("Run %s: events TSV not found, skipping.", run_id)
        return []

    logger.info("Run %s: loading EEG from %s", run_id, vhdr_path.name)
    try:
        eeg_raw = load_eeg_raw(vhdr_path)
    except Exception as exc:
        logger.warning("Run %s: failed to load EEG: %s", run_id, exc)
        return []

    try:
        events_df = load_events_df(events_path)
    except Exception as exc:
        logger.warning("Run %s: failed to load events: %s", run_id, exc)
        return []

    luminance_events = events_df[
        events_df["trial_type"] == "video_luminance"
    ].reset_index(drop=True)

    if luminance_events.empty:
        logger.warning("Run %s: no video_luminance events found.", run_id)
        return []

    run_results: list[dict] = []

    for _, event_row in luminance_events.iterrows():
        stim_id = int(event_row["stim_id"])
        video_id = stim_id - 100
        onset_s = float(event_row["onset"])
        duration_s = float(event_row["duration"])

        if video_id not in EXPERIMENTAL_VIDEOS:
            continue

        logger.info(
            "Run %s: video_id=%d (onset=%.1fs, dur=%.1fs) — creating epochs...",
            run_id, video_id, onset_s, duration_s,
        )

        epochs = extract_epochs_for_video_segment(
            eeg_raw=eeg_raw,
            onset_s=onset_s,
            duration_s=duration_s,
            epoch_duration_s=EPOCH_DURATION_S,
            epoch_step_s=EPOCH_STEP_S,
        )

        if epochs is None:
            continue

        logger.info(
            "Run %s: applying AutoReject on %d epochs (video_id=%d)...",
            run_id, len(epochs), video_id,
        )
        try:
            qa_stats = run_autoreject_qa(epochs=epochs, random_seed=RANDOM_SEED)
        except Exception as exc:
            logger.warning(
                "Run %s: AutoReject failed for video_id=%d: %s", run_id, video_id, exc
            )
            continue

        logger.info(
            "Run %s: video_id=%d → %d/%d epochs rejected (%.1f%%) | n_interp=%.1f | cons=%.3f",
            run_id, video_id, qa_stats["n_epochs_rejected"], qa_stats["n_epochs_total"], qa_stats["rejection_pct"],
            qa_stats["ar_n_interpolate"], qa_stats["ar_consensus"]
        )

        run_results.append(
            {
                "Subject": f"sub-{subject}",
                "RunID": run_id,
                "Acq": acq,
                "VideoID": video_id,
                "TotalEpochs": qa_stats["n_epochs_total"],
                "RejectedEpochs": qa_stats["n_epochs_rejected"],
                "RejectionPct": round(qa_stats["rejection_pct"], 2),
                "ArNInterpolate": qa_stats["ar_n_interpolate"],
                "ArConsensus": round(qa_stats["ar_consensus"], 3),
                "BadEpochsIdx": str(qa_stats["bad_epochs_idx"]),
                "InterpCounts": str(qa_stats["interp_counts"]),
                "InterpChannels": str(qa_stats["interp_channels"]),
                "_reject_log": qa_stats["reject_log"],
                "_channel_names": qa_stats["channel_names"],
            }
        )

    return run_results


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmaps(all_run_results: list[dict], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for result_row in all_run_results:
        subject = result_row["Subject"]
        run_id = result_row["RunID"]
        video_id = result_row["VideoID"]
        reject_log = result_row["_reject_log"]
        channel_names = result_row["_channel_names"]

        heatmap_filename = f"{subject}_run-{run_id}_video-{video_id}_rejection_heatmap.png"
        heatmap_path = figures_dir / heatmap_filename

        plot_rejection_heatmap(
            reject_log=reject_log,
            channel_names=channel_names,
            output_path=heatmap_path,
        )


# ---------------------------------------------------------------------------
# CSV + JSON sidecar
# ---------------------------------------------------------------------------

def build_qa_dataframe(all_run_results: list[dict]) -> pd.DataFrame:
    csv_columns = [
        "Subject",
        "RunID",
        "Acq",
        "VideoID",
        "TotalEpochs",
        "RejectedEpochs",
        "RejectionPct",
        "ArNInterpolate",
        "ArConsensus",
        "BadEpochsIdx",
        "InterpCounts",
        "InterpChannels",
    ]
    rows = [{col: row[col] for col in csv_columns} for row in all_run_results]
    return pd.DataFrame(rows, columns=csv_columns)

def save_qa_csv(qa_df: pd.DataFrame, subject: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"sub-{subject}_eeg_qa_autoreject.tsv"
    qa_df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("QA results saved for %s to: %s", subject, tsv_path)
    return tsv_path


def write_qa_json_sidecar(json_path: Path) -> None:
    data_dictionary: dict[str, dict[str, str]] = {
        "Subject": {"Description": "Subject identifier (BIDS entity)", "DataType": "string"},
        "RunID": {"Description": "Run identifier", "DataType": "string"},
        "Acq": {"Description": "Acquisition label", "DataType": "string"},
        "VideoID": {"Description": "Luminance video identifier", "DataType": "integer"},
        "TotalEpochs": {"Description": "Total number of extracted epochs", "DataType": "integer"},
        "RejectedEpochs": {"Description": "Number of rejected epochs", "DataType": "integer"},
        "RejectionPct": {"Description": "Percentage of rejected epochs", "DataType": "float"},
        "ArNInterpolate": {"Description": "AutoReject learned max interpolated channels limit", "DataType": "float"},
        "ArConsensus": {"Description": "AutoReject learned consensus threshold", "DataType": "float"},
        "BadEpochsIdx": {"Description": "Indices of rejected epochs (0-indexed)", "DataType": "string"},
        "InterpCounts": {"Description": "Number of channels interpolated per epoch", "DataType": "string"},
        "InterpChannels": {"Description": "List of interpolated channels per epoch", "DataType": "string"},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data_dictionary, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_qa_summary(qa_df: pd.DataFrame, subject: str) -> None:
    logger.info("=" * 60)
    logger.info("EEG QA SUMMARY — sub-%s", subject)
    logger.info("=" * 60)

    for run_id, run_group in qa_df.groupby("RunID"):
        acq = run_group["Acq"].iloc[0]
        total_epochs = run_group["TotalEpochs"].sum()
        rejected_epochs = run_group["RejectedEpochs"].sum()
        overall_pct = (rejected_epochs / total_epochs * 100) if total_epochs > 0 else 0.0
        logger.info(
            "Run %s (acq-%s): %d/%d epochs rejected (%.1f%% overall)",
            run_id, acq, rejected_epochs, total_epochs, overall_pct,
        )

    grand_total = qa_df["TotalEpochs"].sum()
    grand_rejected = qa_df["RejectedEpochs"].sum()
    grand_pct = (grand_rejected / grand_total * 100) if grand_total > 0 else 0.0
    logger.info("-" * 60)
    logger.info("GRAND TOTAL %s: %d/%d epochs rejected (%.1f%%)", subject, grand_rejected, grand_total, grand_pct)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_eeg_qa_pipeline() -> None:
    logger.info("Starting EEG QA pipeline across all subjects. random_seed=%d", RANDOM_SEED)
    
    # regex to match: sub-XX_ses-vr_task-XX_acq-X_run-XXX_desc-preproc_eeg.vhdr
    vhdr_pattern = re.compile(r"sub-(?P<sub>\w+)_ses-vr_task-(?P<task>\w+)_acq-(?P<acq>\w+)_run-(?P<run>\w+)_desc-preproc_eeg\.vhdr")

    vhdr_files = list(DERIVATIVES_PATH.rglob("*_desc-preproc_eeg.vhdr"))
    subjects_map = {}
    
    for vhdr_file in vhdr_files:
        match = vhdr_pattern.match(vhdr_file.name)
        if match:
            sub = match.group("sub")
            if sub not in subjects_map:
                subjects_map[sub] = []
            subjects_map[sub].append({
                "path": vhdr_file,
                "task": match.group("task"),
                "acq": match.group("acq"),
                "run": match.group("run"),
            })

    if not subjects_map:
        logger.error("No valid .vhdr files found.")
        sys.exit(1)

    for sub, runs in subjects_map.items():
        if sub != SUBJECT:
            continue
            
        logger.info("\n>>> Processing Subject %s", sub)
        sub_results = []
        for r_info in sorted(runs, key=lambda x: x["run"]):
            res = run_qa_for_run(r_info["path"], sub, r_info["task"], r_info["acq"], r_info["run"])
            sub_results.extend(res)
            
        if not sub_results:
            continue
            
        generate_heatmaps(sub_results, QA_OUTPUT_DIR)
        
        qa_df = build_qa_dataframe(sub_results)
        tsv_path = save_qa_csv(qa_df, sub, QA_OUTPUT_DIR)
        
        json_path = tsv_path.with_suffix(".json")
        write_qa_json_sidecar(json_path)
        
        print_qa_summary(qa_df, sub)

    logger.info("\n>>> All subjects processed.")


if __name__ == "__main__":
    run_eeg_qa_pipeline()
