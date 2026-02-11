"""Synchronisation between EEG epochs and physical luminance time-series.

Pure functions for loading luminance CSVs, generating epoch onsets, and
interpolating luminance values to match EEG epoch windows.

Requirements: 3.1, 3.2, 3.3
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def load_luminance_csv(csv_path: Path) -> pd.DataFrame:
    """Load a luminance CSV and return a validated DataFrame.

    The CSV is expected to have two columns: ``timestamp`` (seconds) and
    ``luminance`` (green-channel intensity, 0–255).

    Args:
        csv_path: Path to the luminance CSV file.

    Returns:
        DataFrame with columns ``timestamp`` (float) and ``luminance`` (float),
        sorted by timestamp in ascending order.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Luminance CSV not found: {csv_path}")

    luminance_df = pd.read_csv(csv_path)

    required_columns = {"timestamp", "luminance"}
    missing = required_columns - set(luminance_df.columns)
    if missing:
        raise ValueError(
            f"Luminance CSV is missing columns: {missing}. "
            f"Found: {list(luminance_df.columns)}"
        )

    luminance_df = luminance_df[["timestamp", "luminance"]].copy()
    luminance_df = luminance_df.sort_values("timestamp").reset_index(drop=True)
    luminance_df["timestamp"] = luminance_df["timestamp"].astype(float)
    luminance_df["luminance"] = luminance_df["luminance"].astype(float)

    return luminance_df


def create_epoch_onsets(
    n_samples_total: int,
    sfreq: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> np.ndarray:
    """Generate an array of epoch onset times relative to segment start.

    Epochs are placed so that the last epoch fits entirely within the segment.
    If the segment is too short for even one full epoch, an empty array is
    returned and a warning is logged.

    Args:
        n_samples_total: Total number of samples in the EEG segment.
        sfreq: Sampling frequency of the EEG (Hz).
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step (stride) between consecutive epoch onsets in seconds.

    Returns:
        1-D array of epoch onset times in seconds.
    """
    total_duration_s = n_samples_total / sfreq

    if total_duration_s < epoch_duration_s:
        logger.warning(
            "Segment duration (%.3f s) is shorter than epoch duration "
            "(%.3f s). No epochs generated.",
            total_duration_s,
            epoch_duration_s,
        )
        return np.array([], dtype=np.float64)

    last_valid_onset = total_duration_s - epoch_duration_s
    onsets = np.arange(0.0, last_valid_onset + epoch_step_s / 2.0, epoch_step_s)
    # Clip to ensure floating-point rounding does not exceed the limit
    onsets = onsets[onsets <= last_valid_onset + 1e-12]

    return onsets


def interpolate_luminance_to_epochs(
    luminance_df: pd.DataFrame,
    epoch_onsets_s: np.ndarray,
    epoch_duration_s: float,
) -> np.ndarray:
    """Interpolate average luminance for each EEG epoch window.

    For each epoch defined by ``[onset, onset + epoch_duration_s]``, the
    function computes the mean luminance over that interval using linear
    interpolation of the luminance time-series.

    Args:
        luminance_df: DataFrame with ``timestamp`` and ``luminance`` columns
            (as returned by :func:`load_luminance_csv`).
        epoch_onsets_s: 1-D array of epoch onset times in seconds, relative
            to the start of the video.
        epoch_duration_s: Duration of each epoch in seconds.

    Returns:
        1-D array of mean luminance values, one per epoch.  Length equals
        ``len(epoch_onsets_s)``.
    """
    if len(epoch_onsets_s) == 0:
        return np.array([], dtype=np.float64)

    timestamps = luminance_df["timestamp"].values
    luminance_values = luminance_df["luminance"].values

    interpolator = interp1d(
        timestamps,
        luminance_values,
        kind="linear",
        bounds_error=False,
        fill_value=(luminance_values[0], luminance_values[-1]),
    )

    epoch_luminance = np.empty(len(epoch_onsets_s), dtype=np.float64)

    for idx, onset in enumerate(epoch_onsets_s):
        window_start = onset
        window_end = onset + epoch_duration_s

        # Sample ~100 points within the window for a smooth average
        n_interp_points = max(10, int(epoch_duration_s * 100))
        sample_times = np.linspace(window_start, window_end, n_interp_points)
        interpolated_values = interpolator(sample_times)
        epoch_luminance[idx] = np.mean(interpolated_values)

    return epoch_luminance
