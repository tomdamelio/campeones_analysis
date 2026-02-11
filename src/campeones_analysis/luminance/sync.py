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
