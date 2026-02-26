"""
Luminance prediction sub-package.

Provides synchronisation between EEG epochs and physical luminance time-series
extracted from stimulus videos, plus spectral feature extraction, Time Delay
Embedding utilities, evaluation metrics, and target computation functions.

Public API:
    sync          – load_luminance_csv, create_epoch_onsets, interpolate_luminance_to_epochs
    features      – extract_bandpower, apply_time_delay_embedding, compute_epoch_covariance
    normalization – zscore_per_video
    permutation   – shuffle_targets_within_videos, compute_p_value,
                    run_permutation_test, plot_permutation_histogram
    evaluation    – compute_r2_score
    targets       – compute_delta_luminance, compute_change_labels
    tde_glhmm     – apply_glhmm_tde_pipeline
"""

from campeones_analysis.luminance.evaluation import compute_r2_score
from campeones_analysis.luminance.features import (
    apply_time_delay_embedding,
    compute_epoch_covariance,
    extract_bandpower,
)
from campeones_analysis.luminance.targets import (
    compute_change_labels,
    compute_delta_luminance,
)
from campeones_analysis.luminance.tde_glhmm import apply_glhmm_tde_pipeline
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
