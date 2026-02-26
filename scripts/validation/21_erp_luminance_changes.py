"""ERP Analysis: event-related potentials locked to luminance change events.

Detects the top N moments of largest absolute luminance change per video,
creates MNE epochs centred on those moments, and computes the mean ERP
across posterior/occipital channels (ROI_Posterior).

Pipeline:
    1. Load preprocessed EEG and events for each run.
    2. For each video_luminance segment, load the luminance CSV and compute
       the first-difference time-series to identify change magnitudes.
    3. Detect the top ERP_N_CHANGES moments of largest |ΔL| per video.
    4. Create MNE Epochs centred on those moments (window: ERP_TMIN–ERP_TMAX).
    5. Average epochs to obtain the ERP per channel.
    6. Generate temporal waveform plots (O1, O2, Pz) and topographic maps
       at key latencies (100 ms, 200 ms, 300 ms post-change).
    7. Save figures and a CSV with per-channel ERP amplitudes to
       ``results/validation/erp/``.

Results saved to ``results/validation/erp/``.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import EEG_CHANNELS
from config_luminance import (
    DERIVATIVES_PATH,
    ERP_N_CHANGES,
    ERP_TMAX,
    ERP_TMIN,
    LUMINANCE_CSV_MAP,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
)
from campeones_analysis.luminance.sync import load_luminance_csv

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
ERP_RESULTS_PATH: Path = PROJECT_ROOT / "results" / "validation" / "erp"

# Channels to plot individual waveforms for
ERP_WAVEFORM_CHANNELS: list[str] = ["O1", "O2", "Pz"]

# Latencies (seconds post-change) for topographic maps
ERP_TOPOMAP_LATENCIES_S: list[float] = [0.1, 0.2, 0.3]


# =========================================================================
# Pure computation functions
# =========================================================================


def detect_top_n_luminance_changes(
    luminance_values: np.ndarray,
    n_changes: int,
) -> np.ndarray:
    """Detect the top N moments of largest absolute luminance change.

    Computes the first-difference of the luminance time-series and returns
    the indices (into the diff array) of the N largest absolute changes,
    sorted by magnitude descending.

    This is a pure function with no I/O — suitable for property-based testing.

    Args:
        luminance_values: 1-D array of luminance values over time.
        n_changes: Number of top changes to detect.

    Returns:
        1-D array of indices into the *diff* array (length
        ``len(luminance_values) - 1``), sorted by ``|diff|`` descending.
        If ``len(luminance_values) <= n_changes``, returns all valid diff
        indices sorted by magnitude descending.

    Requirements: 10.1
    """
    if len(luminance_values) < 2:
        return np.array([], dtype=np.intp)

    luminance_diff = np.diff(luminance_values)
    abs_diff = np.abs(luminance_diff)

    n_available = len(abs_diff)
    n_top = min(n_changes, n_available)

    # argpartition is O(n) but we need full sort by magnitude anyway
    top_indices = np.argsort(abs_diff)[::-1][:n_top]

    return top_indices


def select_roi_channels(
    available_channels: list[str],
    roi_channels: list[str],
) -> list[str]:
    """Select ROI channels that exist in the available EEG channels.

    Args:
        available_channels: Channel names present in the EEG recording.
        roi_channels: Desired ROI channel names.

    Returns:
        List of channel names present in both lists, preserving
        *roi_channels* order.
    """
    available_set = set(available_channels)
    selected: list[str] = []
    for channel_name in roi_channels:
        if channel_name in available_set:
            selected.append(channel_name)
        else:
            logger.warning("ROI channel %s not found in EEG, skipping.", channel_name)
    return selected


# =========================================================================
# Path resolution helpers
# =========================================================================


def _resolve_eeg_path(run_config: dict) -> Path | None:
    """Build the preprocessed EEG .vhdr path for a run.

    Args:
        run_config: Dictionary with keys ``id``, ``acq``, ``task``.

    Returns:
        Path to the .vhdr file, or ``None`` if not found.
    """
    run_id = run_config["id"]
    acq = run_config["acq"]
    task = run_config["task"]
    eeg_dir = DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
    vhdr_name = (
        f"sub-{SUBJECT}_ses-{SESSION}_task-{task}"
        f"_acq-{acq}_run-{run_id}_desc-preproc_eeg.vhdr"
    )
    vhdr_path = eeg_dir / vhdr_name
    return vhdr_path if vhdr_path.exists() else None


def _resolve_events_path(run_config: dict) -> Path | None:
    """Build the merged-events TSV path for a run, falling back to regular.

    Args:
        run_config: Dictionary with keys ``id``, ``acq``, ``task``.

    Returns:
        Path to the events TSV, or ``None`` if not found.
    """
    run_id = run_config["id"]
    acq = run_config["acq"]
    task = run_config["task"]
    derivatives_base = PROJECT_ROOT / "data" / "derivatives"

    merged_dir = (
        derivatives_base
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

    events_dir = DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
    events_name = (
        f"sub-{SUBJECT}_ses-{SESSION}_task-{task}"
        f"_acq-{acq}_run-{run_id}_desc-preproc_events.tsv"
    )
    events_path = events_dir / events_name
    return events_path if events_path.exists() else None


# =========================================================================
# ERP processing
# =========================================================================


def process_video_segment_erp(
    eeg_raw: mne.io.Raw,
    onset_s: float,
    duration_s: float,
    video_id: int,
    luminance_df: pd.DataFrame,
    sfreq: float,
    roi_channels: list[str],
) -> mne.Epochs | None:
    """Create ERP epochs centred on top luminance-change moments in a video segment.

    Steps:
        1. Crop EEG to the video segment.
        2. Compute first-difference of luminance and detect top N changes.
        3. Map luminance change indices to EEG sample positions using
           luminance timestamps and the segment sampling frequency.
        4. Build an MNE events array and create ``mne.Epochs``.

    Args:
        eeg_raw: Full-run MNE Raw object (preprocessed EEG).
        onset_s: Onset of the video segment in seconds (relative to run start).
        duration_s: Duration of the video segment in seconds.
        video_id: Numeric video identifier (e.g. 3, 7, 9, 12).
        luminance_df: DataFrame with ``timestamp`` and ``luminance`` columns
            as returned by :func:`load_luminance_csv`.
        sfreq: EEG sampling frequency in Hz.
        roi_channels: Channel names to pick for epoching.

    Returns:
        ``mne.Epochs`` object centred on luminance-change events, or ``None``
        if no valid epochs could be created.

    Requirements: 10.1, 10.2, 10.3
    """
    # 1. Crop EEG to video segment
    t_stop = onset_s + duration_s
    try:
        segment_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=t_stop)
    except ValueError as exc:
        logger.warning(
            "Could not crop EEG [%.2f, %.2f] for video %d: %s",
            onset_s,
            t_stop,
            video_id,
            exc,
        )
        return None

    # 2. Detect top N luminance changes
    luminance_values = luminance_df["luminance"].values
    luminance_timestamps = luminance_df["timestamp"].values

    change_indices = detect_top_n_luminance_changes(luminance_values, ERP_N_CHANGES)

    if len(change_indices) == 0:
        logger.warning("Video %d: no luminance changes detected.", video_id)
        return None

    # 3. Convert luminance diff indices to EEG sample positions.
    #    Each diff index *i* corresponds to the transition between
    #    luminance_timestamps[i] and luminance_timestamps[i+1].
    #    We use the timestamp at index i+1 (the "after" moment) as the
    #    event time, then convert to EEG samples relative to segment onset.
    change_times_s = luminance_timestamps[change_indices + 1]

    # Convert absolute luminance times to EEG sample indices within segment
    segment_start_time = segment_raw.times[0]
    change_samples = np.round(
        (change_times_s - segment_start_time) * sfreq
    ).astype(int)

    # Filter out samples that fall outside the segment
    n_segment_samples = len(segment_raw.times)
    valid_mask = (change_samples >= 0) & (change_samples < n_segment_samples)
    change_samples = change_samples[valid_mask]

    # Convert to MNE absolute sample indices mapping
    change_samples = change_samples + segment_raw.first_samp

    if len(change_samples) == 0:
        logger.warning(
            "Video %d: all change events fall outside EEG segment.", video_id
        )
        return None

    # 4. Build MNE events array: (sample, 0, event_id)
    event_id = 1
    mne_events = np.column_stack(
        [
            change_samples,
            np.zeros(len(change_samples), dtype=int),
            np.full(len(change_samples), event_id, dtype=int),
        ]
    )
    
    # Sort events chronologically to satisfy MNE Epochs requirement
    sort_idx = np.argsort(mne_events[:, 0])
    mne_events = mne_events[sort_idx]

    # 5. Create epochs
    try:
        epochs = mne.Epochs(
            segment_raw,
            events=mne_events,
            event_id={"luminance_change": event_id},
            tmin=ERP_TMIN,
            tmax=ERP_TMAX,
            picks=roi_channels,
            baseline=None,
            preload=True,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "Video %d: failed to create MNE Epochs: %s", video_id, exc
        )
        return None

    if len(epochs) == 0:
        logger.warning("Video %d: no valid ERP epochs after rejection.", video_id)
        return None

    logger.info(
        "Video %d: created %d ERP epochs (tmin=%.2f, tmax=%.2f).",
        video_id,
        len(epochs),
        ERP_TMIN,
        ERP_TMAX,
    )
    return epochs


# =========================================================================
# Plotting functions
# =========================================================================


def plot_erp_waveforms(
    evoked: mne.Evoked,
    output_dir: Path,
    video_label: str,
) -> None:
    """Plot ERP waveforms for key occipital channels and save to disk.

    Generates one figure with overlaid waveforms for O1, O2, and Pz
    (or whichever of those are available in the Evoked object).

    Args:
        evoked: MNE Evoked object with the averaged ERP.
        output_dir: Directory where the figure will be saved.
        video_label: Label used in the filename and title (e.g.
            ``"grand_average"`` or ``"video_3"``).

    Requirements: 10.4
    """
    available_channels = evoked.ch_names
    channels_to_plot = [
        ch for ch in ERP_WAVEFORM_CHANNELS if ch in available_channels
    ]

    if not channels_to_plot:
        logger.warning(
            "No waveform channels (%s) found in evoked for %s.",
            ERP_WAVEFORM_CHANNELS,
            video_label,
        )
        return

    times_ms = evoked.times * 1000.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for channel_name in channels_to_plot:
        channel_idx = evoked.ch_names.index(channel_name)
        signal_uv = evoked.data[channel_idx] * 1e6  # V → µV
        ax.plot(times_ms, signal_uv, label=channel_name)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, label="Change onset")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"ERP Waveforms — {video_label} (sub-{SUBJECT})")
    ax.legend(loc="upper right")
    ax.invert_yaxis()  # EEG convention: negative up

    fig.tight_layout()
    fig_path = output_dir / f"sub-{SUBJECT}_erp_waveforms_{video_label}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Waveform plot saved: %s", fig_path)


def plot_erp_topomaps(
    evoked: mne.Evoked,
    output_dir: Path,
    video_label: str,
) -> None:
    """Plot topographic maps at key post-change latencies and save to disk.

    Generates topomaps at 100 ms, 200 ms, and 300 ms post-change using
    MNE's built-in topomap plotting.

    Args:
        evoked: MNE Evoked object with the averaged ERP.
        output_dir: Directory where the figure will be saved.
        video_label: Label used in the filename and title.

    Requirements: 10.6
    """
    try:
        fig = evoked.plot_topomap(
            times=ERP_TOPOMAP_LATENCIES_S,
            ch_type="eeg",
            show=False,
            time_unit="s",
        )
        fig_path = output_dir / f"sub-{SUBJECT}_erp_topomaps_{video_label}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Topomap plot saved: %s", fig_path)
    except Exception as exc:
        logger.warning(
            "Could not generate topomap for %s: %s", video_label, exc
        )


# =========================================================================
# JSON sidecar (BIDS-compliant data dictionary)
# =========================================================================


def _write_erp_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the ERP results CSV.

    Args:
        json_path: Destination path for the JSON sidecar file.
    """
    sidecar = {
        "Description": (
            "Peak ERP amplitudes per channel at key latencies, derived from "
            "epochs centred on the top luminance-change moments in each video."
        ),
        "Subject": {
            "Description": "Subject identifier",
            "Type": "string",
        },
        "VideoLabel": {
            "Description": (
                "Video label (e.g. 'video_3', 'grand_average')"
            ),
            "Type": "string",
        },
        "Channel": {
            "Description": "EEG channel name",
            "Type": "string",
        },
        "PeakLatency_ms": {
            "Description": "Latency of the peak amplitude in milliseconds post-change",
            "Units": "ms",
            "Type": "float",
        },
        "PeakAmplitude_uV": {
            "Description": "Peak amplitude in microvolts",
            "Units": "µV",
            "Type": "float",
        },
        "MeanAmplitude_100_300_uV": {
            "Description": (
                "Mean amplitude in the 100–300 ms post-change window (µV)"
            ),
            "Units": "µV",
            "Type": "float",
        },
        "NumEpochs": {
            "Description": "Number of epochs averaged",
            "Type": "int",
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
    logger.info("JSON sidecar saved: %s", json_path)


# =========================================================================
# Pipeline orchestration
# =========================================================================


def _extract_peak_metrics(
    evoked: mne.Evoked,
    video_label: str,
    n_epochs: int,
) -> list[dict]:
    """Extract per-channel peak amplitude and latency from an Evoked object.

    Args:
        evoked: MNE Evoked object with the averaged ERP.
        video_label: Label for the video or condition.
        n_epochs: Number of epochs that were averaged.

    Returns:
        List of dicts, one per channel, with peak metrics.
    """
    rows: list[dict] = []
    times_ms = evoked.times * 1000.0

    # Window for mean amplitude: 100–300 ms post-change
    window_mask = (evoked.times >= 0.1) & (evoked.times <= 0.3)

    for channel_idx, channel_name in enumerate(evoked.ch_names):
        signal_uv = evoked.data[channel_idx] * 1e6  # V → µV

        peak_idx = np.argmax(np.abs(signal_uv))
        peak_latency_ms = float(times_ms[peak_idx])
        peak_amplitude_uv = float(signal_uv[peak_idx])

        if window_mask.any():
            mean_amplitude_uv = float(np.mean(signal_uv[window_mask]))
        else:
            mean_amplitude_uv = float("nan")

        rows.append(
            {
                "Subject": SUBJECT,
                "VideoLabel": video_label,
                "Channel": channel_name,
                "PeakLatency_ms": peak_latency_ms,
                "PeakAmplitude_uV": peak_amplitude_uv,
                "MeanAmplitude_100_300_uV": mean_amplitude_uv,
                "NumEpochs": n_epochs,
            }
        )
    return rows


def run_pipeline() -> None:
    """Execute the ERP analysis pipeline for luminance-change events.

    Steps:
        1. For each run: load EEG and events, iterate over video_luminance
           segments, create ERP epochs centred on top luminance changes.
        2. Concatenate all epochs across runs and videos.
        3. Compute grand-average ERP and per-video ERPs.
        4. Generate waveform plots and topographic maps.
        5. Save CSV with peak amplitudes per channel + JSON sidecar.

    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """
    print("=" * 60)
    print(f"21 — ERP Analysis: Luminance Changes — sub-{SUBJECT}")
    print(f"     Top N changes: {ERP_N_CHANGES}")
    print(f"     Window: [{ERP_TMIN}, {ERP_TMAX}] s")
    print("=" * 60)

    output_dir = ERP_RESULTS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")

    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Collect ERP epochs across all runs and videos
    # ------------------------------------------------------------------
    all_epochs: list[mne.Epochs] = []
    per_video_epochs: dict[int, list[mne.Epochs]] = {}

    for run_config in RUNS_CONFIG:
        run_label = (
            f"run-{run_config['id']} acq-{run_config['acq']} "
            f"task-{run_config['task']} ({run_config['block']})"
        )
        print(f"\nProcessing {run_label}")

        vhdr_path = _resolve_eeg_path(run_config)
        if vhdr_path is None:
            print("  WARNING: EEG file not found, skipping.")
            continue

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            print("  WARNING: Events file not found, skipping.")
            continue

        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(
            str(vhdr_path), preload=True, verbose=False
        )
        sfreq = eeg_raw.info["sfreq"]

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        recording_roi = select_roi_channels(eeg_raw.ch_names, POSTERIOR_CHANNELS)

        luminance_events = events_df[
            events_df["trial_type"] == "video_luminance"
        ].reset_index(drop=True)

        if luminance_events.empty:
            print(f"  WARNING: No video_luminance events in run {run_config['id']}.")
            continue

        for _, event_row in luminance_events.iterrows():
            stim_id = int(event_row["stim_id"])
            video_id = stim_id - 100
            onset_s = float(event_row["onset"])
            duration_s = float(event_row["duration"])

            csv_filename = LUMINANCE_CSV_MAP.get(video_id)
            if csv_filename is None:
                logger.warning(
                    "Run %s: video_id %d not in LUMINANCE_CSV_MAP, skipping.",
                    run_config["id"],
                    video_id,
                )
                continue

            csv_path = STIMULI_PATH / csv_filename
            try:
                luminance_df = load_luminance_csv(csv_path)
            except FileNotFoundError:
                logger.warning(
                    "Run %s: luminance CSV not found: %s, skipping.",
                    run_config["id"],
                    csv_path,
                )
                continue

            epochs = process_video_segment_erp(
                eeg_raw=eeg_raw,
                onset_s=onset_s,
                duration_s=duration_s,
                video_id=video_id,
                luminance_df=luminance_df,
                sfreq=sfreq,
                roi_channels=recording_roi,
            )

            if epochs is not None:
                all_epochs.append(epochs)
                per_video_epochs.setdefault(video_id, []).append(epochs)
                print(f"  Video {video_id}: {len(epochs)} ERP epochs")

    # ------------------------------------------------------------------
    # 3. Check we have data
    # ------------------------------------------------------------------
    if not all_epochs:
        print("ERROR: No ERP epochs generated across all runs. Exiting.")
        return

    total_epoch_count = sum(len(ep) for ep in all_epochs)
    print(f"\nTotal ERP epochs collected: {total_epoch_count}")

    # ------------------------------------------------------------------
    # 4. Grand-average ERP
    # ------------------------------------------------------------------
    grand_epochs = mne.concatenate_epochs(all_epochs)
    grand_evoked = grand_epochs.average()

    print(f"Grand-average ERP: {len(grand_epochs)} epochs")

    plot_erp_waveforms(grand_evoked, output_dir, "grand_average")
    plot_erp_topomaps(grand_evoked, output_dir, "grand_average")

    peak_rows: list[dict] = _extract_peak_metrics(
        grand_evoked, "grand_average", len(grand_epochs)
    )

    # ------------------------------------------------------------------
    # 5. Per-video ERPs
    # ------------------------------------------------------------------
    for video_id in sorted(per_video_epochs.keys()):
        video_epoch_list = per_video_epochs[video_id]
        video_epochs = mne.concatenate_epochs(video_epoch_list)
        video_evoked = video_epochs.average()
        video_label = f"video_{video_id}"

        print(f"  {video_label}: {len(video_epochs)} epochs")

        plot_erp_waveforms(video_evoked, output_dir, video_label)
        plot_erp_topomaps(video_evoked, output_dir, video_label)

        video_rows = _extract_peak_metrics(
            video_evoked, video_label, len(video_epochs)
        )
        peak_rows.extend(video_rows)

    # ------------------------------------------------------------------
    # 6. Save CSV + JSON sidecar
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(peak_rows)
    csv_path = output_dir / f"sub-{SUBJECT}_erp_peak_amplitudes.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_path = csv_path.with_suffix(".json")
    _write_erp_json_sidecar(json_path)
    print(f"JSON data dictionary saved: {json_path}")

    print("\n" + "=" * 60)
    print("ERP analysis pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )
    run_pipeline()
