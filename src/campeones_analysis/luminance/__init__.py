"""
Luminance prediction sub-package.

Provides synchronisation between EEG epochs and physical luminance time-series
extracted from stimulus videos, plus spectral feature extraction and Time Delay
Embedding utilities.

Public API:
    sync   – load_luminance_csv, create_epoch_onsets, interpolate_luminance_to_epochs
    features – extract_bandpower, apply_time_delay_embedding (pending)
"""

from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
