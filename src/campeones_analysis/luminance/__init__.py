"""
Luminance prediction sub-package.

Provides synchronisation between EEG epochs and physical luminance time-series
extracted from stimulus videos, plus spectral feature extraction and Time Delay
Embedding utilities.

Public API:
    sync          – load_luminance_csv, create_epoch_onsets, interpolate_luminance_to_epochs
    features      – extract_bandpower, apply_time_delay_embedding
    normalization – zscore_per_video
    permutation   – shuffle_targets_within_videos, compute_p_value,
                    run_permutation_test, plot_permutation_histogram
"""

from campeones_analysis.luminance.features import (
    apply_time_delay_embedding,
    extract_bandpower,
)
from campeones_analysis.luminance.normalization import zscore_per_video
from campeones_analysis.luminance.permutation import (
    compute_p_value,
    plot_permutation_histogram,
    run_permutation_test,
    shuffle_targets_within_videos,
)
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
