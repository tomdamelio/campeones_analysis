"""Cross-correlation between real and perceived (joystick) luminance.

Measures the temporal lag between the physical luminance of each video
stimulus and the luminance reported by sub-27 via joystick, quantifying
the latency of the perceptual response.

Pipeline:
    1. For each run, load preprocessed EEG (which contains the joystick_x
       channel) and the merged-events TSV.
    2. For each ``video_luminance`` segment, load the stimulus luminance CSV
       and extract the joystick_x signal from the EEG.
    3. Resample the joystick signal to match the luminance CSV sampling rate.
    4. Compute the normalised cross-correlation between both signals.
    5. Identify the lag (in seconds) that maximises the correlation.
    6. Generate per-video plots of cross-correlation vs lag.
    7. Save a summary CSV + JSON sidecar to
       ``results/validation/cross_correlation/``.

Results saved to ``results/validation/cross_correlation/``.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
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
from scipy.interpolate import interp1d
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_luminance import (
    DERIVATIVES_PATH,
    EXPERIMENTAL_VIDEOS,
    LUMINANCE_CSV_MAP,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    XDF_PATH,
)
from campeones_analysis.luminance.sync import load_luminance_csv

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
XCORR_RESULTS_PATH: Path = (
    PROJECT_ROOT / "results" / "validation" / "cross_correlation"
)

# Maximum lag to consider (seconds). Perceptual response lags are typically
# well under 10 s; a generous window avoids missing slow responses.
MAX_LAG_S: float = 10.0


# =========================================================================
# Pure computation functions (tested by Property 9)
# =========================================================================


def compute_normalized_cross_correlation(
    signal_real: np.ndarray,
    signal_reported: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalised cross-correlation between two signals.

    Uses ``scipy.signal.correlate`` in "full" mode and normalises by the
    geometric mean of the zero-lag autocorrelations so that the output
    values lie in [-1, 1].

    Args:
        signal_real: 1-D array of the physical (stimulus) luminance signal.
        signal_reported: 1-D array of the reported (joystick) luminance
            signal.  Must have the same length as *signal_real*.
        max_lag_samples: Maximum lag (in samples) to retain on each side
            of zero lag.  The returned arrays span
            ``[-max_lag_samples, +max_lag_samples]``.

    Returns:
        Tuple of ``(lags_array, xcorr_values)`` where *lags_array* contains
        integer lag values in samples and *xcorr_values* contains the
        corresponding normalised cross-correlation coefficients in [-1, 1].

    Requirements: 11.1
    """
    signal_real = np.asarray(signal_real, dtype=np.float64)
    signal_reported = np.asarray(signal_reported, dtype=np.float64)

    # Zero-mean the signals to remove DC offset
    signal_real_zm = signal_real - np.mean(signal_real)
    signal_reported_zm = signal_reported - np.mean(signal_reported)

    # Full cross-correlation.
    # Convention: correlate(reported, real) so that a positive lag in the
    # output means the reported signal is delayed relative to the real
    # signal (i.e., the perceptual response lags the stimulus).
    full_xcorr = correlate(signal_reported_zm, signal_real_zm, mode="full")

    # Normalisation factor: sqrt(autocorr_real(0) * autocorr_reported(0))
    norm_factor = np.sqrt(
        np.sum(signal_real_zm**2) * np.sum(signal_reported_zm**2)
    )
    if norm_factor == 0.0:
        # One or both signals are constant → correlation undefined
        n_lags = 2 * max_lag_samples + 1
        return np.arange(-max_lag_samples, max_lag_samples + 1), np.zeros(n_lags)

    normalised_xcorr = full_xcorr / norm_factor

    # Build full lags array: -(N-1) … 0 … +(N-1)
    n_samples = len(signal_reported)
    full_lags = np.arange(-(n_samples - 1), n_samples)

    # Trim to [-max_lag_samples, +max_lag_samples]
    lag_mask = (full_lags >= -max_lag_samples) & (full_lags <= max_lag_samples)
    lags_array = full_lags[lag_mask]
    xcorr_values = normalised_xcorr[lag_mask]

    return lags_array, xcorr_values


def find_optimal_lag(
    lags_array: np.ndarray,
    xcorr_values: np.ndarray,
    sfreq: float,
) -> tuple[float, float]:
    """Find the lag that maximises the cross-correlation.

    Args:
        lags_array: 1-D array of lag values in samples.
        xcorr_values: 1-D array of normalised cross-correlation values.
        sfreq: Sampling frequency (Hz) used to convert samples → seconds.

    Returns:
        Tuple of ``(optimal_lag_seconds, max_correlation)`` where
        *optimal_lag_seconds* is the lag converted to seconds and
        *max_correlation* is the peak normalised cross-correlation value.

    Requirements: 11.2
    """
    best_idx = int(np.argmax(xcorr_values))
    optimal_lag_samples = int(lags_array[best_idx])
    optimal_lag_seconds = optimal_lag_samples / sfreq
    max_correlation = float(xcorr_values[best_idx])
    return optimal_lag_seconds, max_correlation


# =========================================================================
# Path resolution helpers (follow script 21 pattern)
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


def _resolve_order_matrix_path(run_config: dict) -> Path | None:
    """Build the Order Matrix Excel path for a run.

    Args:
        run_config: Dictionary with keys ``acq``, ``block``.

    Returns:
        Path to the .xlsx file, or ``None`` if not found.
    """
    acq = run_config["acq"]
    block = run_config["block"]
    order_matrix_path = (
        XDF_PATH
        / f"sub-{SUBJECT}"
        / f"order_matrix_{SUBJECT}_{acq}_{block}_VR.xlsx"
    )
    return order_matrix_path if order_matrix_path.exists() else None


# =========================================================================
# Signal extraction and resampling
# =========================================================================


def _extract_joystick_signal(
    eeg_raw: mne.io.Raw,
    onset_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Extract the joystick_x signal for a video segment from the EEG."""
    if "joystick_x" not in eeg_raw.ch_names:
        return None

    t_stop = onset_s + duration_s
    try:
        segment_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=t_stop)
    except ValueError as exc:
        logger.warning(
            "Could not crop EEG [%.2f, %.2f]: %s", onset_s, t_stop, exc
        )
        return None

    sfreq = segment_raw.info["sfreq"]
    joystick_signal = segment_raw.get_data(picks=["joystick_x"])[0]
    # Keep times relative to segment start (0 to duration_s) to match luminance CSV
    timestamps = segment_raw.times

    return joystick_signal, timestamps, sfreq


def _resample_joystick_to_luminance(
    joystick_signal: np.ndarray,
    joystick_timestamps: np.ndarray,
    luminance_timestamps: np.ndarray,
) -> np.ndarray:
    """Resample the joystick signal to match luminance CSV timestamps.

    Uses linear interpolation to align the joystick signal (typically at
    EEG sampling rate, e.g. 500 Hz) to the luminance CSV time-base
    (typically ~30 Hz video frame rate).

    Args:
        joystick_signal: 1-D array of joystick values at EEG sampling rate.
        joystick_timestamps: 1-D array of absolute timestamps for the
            joystick signal.
        luminance_timestamps: 1-D array of target timestamps from the
            luminance CSV.

    Returns:
        1-D array of joystick values interpolated at *luminance_timestamps*.
    """
    interpolator = interp1d(
        joystick_timestamps,
        joystick_signal,
        kind="linear",
        bounds_error=False,
        fill_value=(joystick_signal[0], joystick_signal[-1]),
    )
    return interpolator(luminance_timestamps)


# =========================================================================
# Plotting
# =========================================================================


def plot_cross_correlation(
    lags_seconds: np.ndarray,
    xcorr_values: np.ndarray,
    optimal_lag_s: float,
    max_correlation: float,
    video_id: int,
    run_id: str,
    output_dir: Path,
) -> None:
    """Plot the cross-correlation function vs lag for a single video.

    Args:
        lags_seconds: 1-D array of lag values in seconds.
        xcorr_values: 1-D array of normalised cross-correlation values.
        optimal_lag_s: Optimal lag in seconds (vertical marker).
        max_correlation: Maximum correlation value (annotation).
        video_id: Video identifier for the title and filename.
        run_id: Run identifier to differentiate repeated videos.
        output_dir: Directory where the figure will be saved.

    Requirements: 11.3
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags_seconds, xcorr_values, linewidth=0.8)
    ax.axvline(
        optimal_lag_s,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Optimal lag = {optimal_lag_s:.3f} s",
    )
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Normalised cross-correlation")
    ax.set_title(
        f"Cross-Correlation — Video {video_id} (sub-{SUBJECT}, run-{run_id})\n"
        f"Max r = {max_correlation:.4f} at lag = {optimal_lag_s:.3f} s"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig_path = output_dir / f"sub-{SUBJECT}_run-{run_id}_xcorr_video_{video_id}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Cross-correlation plot saved: %s", fig_path)


# =========================================================================
# JSON sidecar (BIDS-compliant data dictionary)
# =========================================================================


def _write_xcorr_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the xcorr CSV.

    Args:
        json_path: Destination path for the JSON sidecar file.
    """
    sidecar = {
        "Description": (
            "Cross-correlation analysis between real (stimulus) luminance "
            "and perceived (joystick-reported) luminance per video for "
            f"sub-{SUBJECT}. The optimal lag indicates the temporal delay "
            "of the perceptual response relative to the physical stimulus."
        ),
        "Subject": {
            "Description": "Subject identifier",
            "Type": "string",
        },
        "RunID": {
            "Description": "Acquisition run identifier",
            "Type": "string",
        },
        "VideoID": {
            "Description": "Experimental video identifier",
            "Type": "int",
        },
        "OptimalLag_s": {
            "Description": (
                "Lag in seconds that maximises the normalised "
                "cross-correlation. Positive values indicate the reported "
                "signal lags behind the real signal."
            ),
            "Units": "s",
            "Type": "float",
        },
        "MaxCorrelation": {
            "Description": (
                "Peak normalised cross-correlation value at the optimal lag, "
                "bounded in [-1, 1]."
            ),
            "Type": "float",
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
    logger.info("JSON sidecar saved: %s", json_path)


# =========================================================================
# Pipeline orchestration
# =========================================================================


def run_pipeline() -> None:
    """Execute the cross-correlation analysis pipeline.

    Steps:
        1. For each run: load EEG (with joystick_x) and events.
        2. For each video_luminance segment where the dimension is
           "luminance": load the stimulus CSV and extract the joystick
           signal.
        3. Resample joystick to luminance time-base.
        4. Compute normalised cross-correlation and find optimal lag.
        5. Generate per-video plots.
        6. Save summary CSV + JSON sidecar.

    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
    """
    print("=" * 60)
    print(f"22 — Cross-Correlation: Real vs Perceived Luminance — sub-{SUBJECT}")
    print(f"     Max lag window: ±{MAX_LAG_S} s")
    print("=" * 60)

    output_dir = XCORR_RESULTS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []

    for run_config in RUNS_CONFIG:
        run_label = (
            f"run-{run_config['id']} acq-{run_config['acq']} "
            f"task-{run_config['task']} ({run_config['block']})"
        )
        print(f"\nProcessing {run_label}")

        # --- Resolve paths ---
        vhdr_path = _resolve_eeg_path(run_config)
        if vhdr_path is None:
            print("  WARNING: EEG file not found, skipping.")
            continue

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            print("  WARNING: Events file not found, skipping.")
            continue

        order_matrix_path = _resolve_order_matrix_path(run_config)
        if order_matrix_path is None:
            print("  WARNING: Order Matrix not found, skipping.")
            continue

        # --- Load data ---
        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(
            str(vhdr_path), preload=True, verbose=False
        )

        if "joystick_x" not in eeg_raw.ch_names:
            print("  WARNING: joystick_x channel not found in EEG, skipping run.")
            continue

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        print(f"  Order Matrix: {order_matrix_path.name}")
        order_matrix_df = pd.read_excel(order_matrix_path)

        # --- Match video events to Order Matrix by position ---
        luminance_events = events_df[
            events_df["trial_type"] == "video_luminance"
        ].reset_index(drop=True)

        if luminance_events.empty:
            print(f"  WARNING: No video_luminance events in run {run_config['id']}.")
            continue

        for idx, event_row in luminance_events.iterrows():
            raw_video_id = event_row.get("video_id")
            if pd.isna(raw_video_id):
                stim_file = str(event_row.get("stim_file", ""))
                if "video_" in stim_file:
                    video_id = int(stim_file.split("video_")[1].split(".")[0])
                else:
                    print("DEBUG: Could not parse video_id")
                    continue
            else:
                video_id = int(raw_video_id)

            if video_id not in EXPERIMENTAL_VIDEOS:
                continue

            onset_s = float(event_row["onset"])
            duration_s = float(event_row["duration"])

            # --- Load stimulus luminance CSV ---
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

            # --- Extract joystick signal (Req 11.5: warn if unavailable) ---
            joystick_result = _extract_joystick_signal(
                eeg_raw, onset_s, duration_s
            )
            if joystick_result is None:
                logger.warning(
                    "Run %s: joystick signal not available for video %d, "
                    "skipping. (Req 11.5)",
                    run_config["id"],
                    video_id,
                )
                continue

            joystick_signal, joystick_timestamps, eeg_sfreq = joystick_result

            # --- Resample joystick to luminance time-base ---
            luminance_timestamps = luminance_df["timestamp"].values
            luminance_values = luminance_df["luminance"].values

            joystick_resampled = _resample_joystick_to_luminance(
                joystick_signal, joystick_timestamps, luminance_timestamps
            )

            # --- Compute cross-correlation ---
            # Estimate luminance sampling rate from timestamps
            if len(luminance_timestamps) < 2:
                logger.warning(
                    "Run %s: luminance CSV for video %d has < 2 samples.",
                    run_config["id"],
                    video_id,
                )
                continue

            luminance_sfreq = 1.0 / np.median(np.diff(luminance_timestamps))
            max_lag_samples = int(MAX_LAG_S * luminance_sfreq)

            lags_array, xcorr_values = compute_normalized_cross_correlation(
                luminance_values, joystick_resampled, max_lag_samples
            )

            optimal_lag_s, max_correlation = find_optimal_lag(
                lags_array, xcorr_values, luminance_sfreq
            )

            print(
                f"  Video {video_id}: optimal lag = {optimal_lag_s:.3f} s, "
                f"max r = {max_correlation:.4f}"
            )

            # --- Plot ---
            lags_seconds = lags_array / luminance_sfreq
            plot_cross_correlation(
                lags_seconds=lags_seconds,
                xcorr_values=xcorr_values,
                optimal_lag_s=optimal_lag_s,
                max_correlation=max_correlation,
                video_id=video_id,
                run_id=run_config["id"],
                output_dir=output_dir,
            )

            result_rows.append(
                {
                    "Subject": SUBJECT,
                    "RunID": run_config["id"],
                    "VideoID": video_id,
                    "OptimalLag_s": optimal_lag_s,
                    "MaxCorrelation": max_correlation,
                }
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not result_rows:
        print(
            "\nWARNING: No cross-correlation results generated. "
            "Check that joystick data is available for luminance videos."
        )
        return

    results_df = pd.DataFrame(result_rows)
    csv_path = output_dir / f"sub-{SUBJECT}_cross_correlation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_path = csv_path.with_suffix(".json")
    _write_xcorr_json_sidecar(json_path)
    print(f"JSON data dictionary saved: {json_path}")

    print("\n" + "=" * 60)
    print("Cross-correlation analysis pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )
    run_pipeline()
