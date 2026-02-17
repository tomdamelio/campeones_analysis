"""Spectral TDE model: TDE on continuous spectral power → luminance.

Pipeline identical to Script 13 (raw TDE) except the input representation:
instead of raw EEG amplitude, we use multitaper time-frequency band-power
in 5 canonical frequency bands.  This isolates the effect of the input
space (raw waveform vs spectral power) while keeping every other pipeline
step identical.

Pipeline per video segment:
    1. Crop preprocessed EEG to video_luminance segment (ROI channels only)
    2. Compute multitaper time-frequency representation → average power in
       each of 5 bands (delta, theta, alpha, beta, gamma) per channel,
       yielding a continuous spectral time-series of shape
       ``(n_samples, n_channels × n_bands)``
    3. Apply TDE on the continuous spectral time-series
    4. PCA on TDE-expanded matrix → N component time-series
    5. Summarise each epoch: mean + variance per PCA component → 2N features
    6. Pair epochs with interpolated luminance targets

Evaluation: Leave-One-Video-Out CV with StandardScaler → Ridge
(GridSearchCV + LeaveOneGroupOut for alpha selection).
Results saved to ``results/modeling/luminance/tde/``.
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
from mne.time_frequency import tfr_array_multitaper
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure scripts/modeling is on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EEG_CHANNELS
from config_luminance import (
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    EXPERIMENTAL_VIDEOS,
    LUMINANCE_CSV_MAP,
    N_PERMUTATIONS,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA_GRID,
    RUNS_CONFIG,
    SESSION,
    SPECTRAL_BANDS,
    STIMULI_PATH,
    SUBJECT,
    TDE_PCA_COMPONENTS,
    TDE_WINDOW_HALF,
    XDF_PATH,
)

from campeones_analysis.luminance.features import apply_time_delay_embedding
from campeones_analysis.luminance.normalization import zscore_per_video
from campeones_analysis.luminance.permutation import (
    plot_permutation_histogram,
    run_permutation_test,
)
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Spearman scorer for GridSearchCV alpha selection
# ---------------------------------------------------------------------------


def _spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation for use as a sklearn scorer.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Spearman ρ between y_true and y_pred.
    """
    return float(spearmanr(y_true, y_pred).correlation)


spearman_scoring = make_scorer(_spearman_scorer)


# ---------------------------------------------------------------------------
# Path resolution helpers (shared with scripts 10–13)
# ---------------------------------------------------------------------------


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

    events_dir = (
        DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
    )
    events_name = (
        f"sub-{SUBJECT}_ses-{SESSION}_task-{task}"
        f"_acq-{acq}_run-{run_id}_desc-preproc_events.tsv"
    )
    events_path = events_dir / events_name
    if events_path.exists():
        return events_path

    return None


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


# ---------------------------------------------------------------------------
# ROI channel selection
# ---------------------------------------------------------------------------


def select_roi_channels(
    available_channels: list[str],
    roi_channels: list[str],
) -> list[str]:
    """Select ROI channels that exist in the available EEG channels.

    Returns the intersection preserving the order defined in *roi_channels*.

    Args:
        available_channels: Channel names present in the EEG recording.
        roi_channels: Desired ROI channel names.

    Returns:
        List of channel names present in both lists.
    """
    available_set = set(available_channels)
    selected: list[str] = []
    for channel in roi_channels:
        if channel in available_set:
            selected.append(channel)
        else:
            logger.warning("ROI channel %s not found in EEG, skipping.", channel)
    return selected


# ---------------------------------------------------------------------------
# Pure computation helpers (identical to Script 13)
# ---------------------------------------------------------------------------


def leave_one_video_out_split(
    epoch_entries: list[dict],
) -> list[tuple[list[dict], list[dict], str]]:
    """Generate Leave-One-Video-Out CV folds.

    Args:
        epoch_entries: List of epoch dicts with ``video_identifier`` key.

    Returns:
        List of ``(train_entries, test_entries, test_video_id)`` tuples.
    """
    unique_videos = sorted(set(entry["video_identifier"] for entry in epoch_entries))
    folds: list[tuple[list[dict], list[dict], str]] = []
    for test_video in unique_videos:
        train = [e for e in epoch_entries if e["video_identifier"] != test_video]
        test = [e for e in epoch_entries if e["video_identifier"] == test_video]
        folds.append((train, test, test_video))
    return folds


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for a single CV fold.

    Args:
        y_true: Ground-truth luminance values.
        y_pred: Predicted luminance values.

    Returns:
        Dictionary with ``PearsonR``, ``SpearmanRho``, and ``RMSE``.
    """
    pearson_r = (
        float(pearsonr(y_true, y_pred)[0])
        if np.std(y_pred) > 1e-9
        else 0.0
    )
    spearman_rho = (
        float(spearmanr(y_true, y_pred).correlation)
        if np.std(y_pred) > 1e-9
        else 0.0
    )
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"PearsonR": pearson_r, "SpearmanRho": spearman_rho, "RMSE": rmse}


# ---------------------------------------------------------------------------
# Spectral power extraction (THE key difference vs Script 13)
# ---------------------------------------------------------------------------


def compute_continuous_bandpower(
    eeg_data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    n_cycles_min: float = 2.0,
) -> np.ndarray:
    """Compute continuous band-power via multitaper time-frequency analysis.

    For each channel, computes the multitaper TFR at the centre frequency
    of each band, then averages power within each band.  This produces a
    time-resolved spectral representation at the same temporal resolution
    as the original EEG signal.

    The number of cycles per frequency is set to ``max(n_cycles_min,
    freq / 2)`` to balance temporal and spectral resolution, following
    MNE defaults for multitaper TFR.

    Args:
        eeg_data: Raw EEG array of shape ``(n_channels, n_samples)``.
        sfreq: Sampling frequency in Hz.
        bands: Mapping of band name to ``(freq_min, freq_max)`` in Hz.
        n_cycles_min: Minimum number of cycles for the lowest frequency.

    Returns:
        2-D array of shape ``(n_samples, n_channels * n_bands)`` with
        band-power values.  Column ordering is channel-major: all bands
        for channel 0, then all bands for channel 1, etc.  This matches
        the convention used by ``extract_bandpower`` in Script 11.
    """
    n_channels, n_samples = eeg_data.shape
    band_names = list(bands.keys())
    n_bands = len(band_names)

    # Build frequency grid: sample each band at ~1 Hz resolution
    freqs_list: list[float] = []
    band_indices: list[tuple[int, int]] = []  # (start_idx, end_idx) per band
    for band_name in band_names:
        freq_min, freq_max = bands[band_name]
        band_freqs = np.arange(freq_min, freq_max + 0.5, 1.0)
        band_freqs = band_freqs[(band_freqs >= freq_min) & (band_freqs <= freq_max)]
        start_idx = len(freqs_list)
        freqs_list.extend(band_freqs.tolist())
        end_idx = len(freqs_list)
        band_indices.append((start_idx, end_idx))

    freqs = np.array(freqs_list)
    n_cycles = np.maximum(n_cycles_min, freqs / 2.0)

    # tfr_array_multitaper expects (n_epochs, n_channels, n_samples)
    data_4d = eeg_data[np.newaxis, :, :]  # (1, n_channels, n_samples)

    # Compute TFR: output shape (1, n_channels, n_freqs, n_samples)
    tfr_power = tfr_array_multitaper(
        data_4d,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        verbose=False,
    )
    # Remove epoch dimension → (n_channels, n_freqs, n_samples)
    tfr_power = tfr_power[0]

    # Average power within each band → (n_channels, n_bands, n_samples)
    bandpower_continuous = np.empty(
        (n_channels, n_bands, n_samples), dtype=np.float64
    )
    for band_idx, (start, end) in enumerate(band_indices):
        bandpower_continuous[:, band_idx, :] = tfr_power[:, start:end, :].mean(
            axis=1
        )

    # Reshape to (n_samples, n_channels * n_bands) — channel-major ordering
    # For channel c, band b: column index = c * n_bands + b
    result = bandpower_continuous.transpose(2, 0, 1)  # (n_samples, n_ch, n_bands)
    result = result.reshape(n_samples, n_channels * n_bands)

    return result


# ---------------------------------------------------------------------------
# TDE on continuous spectral time-series
# ---------------------------------------------------------------------------


def apply_tde_on_continuous_spectral(
    eeg_data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    window_half: int,
) -> np.ndarray:
    """Apply TDE on continuous spectral band-power representation.

    This is the spectral counterpart of Script 13's
    ``apply_tde_on_continuous_signal``.  Instead of operating on raw EEG
    amplitudes, it first transforms the signal into a continuous
    band-power representation via multitaper TFR, then applies TDE.

    Args:
        eeg_data: Raw EEG array of shape ``(n_channels, n_samples)``.
        sfreq: Sampling frequency in Hz.
        bands: Mapping of band name to ``(freq_min, freq_max)`` in Hz.
        window_half: Half-width of the TDE embedding window.

    Returns:
        2-D array of shape
        ``(n_samples - 2 * window_half,
          n_channels * n_bands * (2 * window_half + 1))``
        containing the TDE-expanded spectral feature matrix.
    """
    spectral_timeseries = compute_continuous_bandpower(eeg_data, sfreq, bands)
    tde_expanded: np.ndarray = apply_time_delay_embedding(
        spectral_timeseries, window_half
    )
    return tde_expanded


# ---------------------------------------------------------------------------
# PCA reduction and epoching helpers (identical to Script 13)
# ---------------------------------------------------------------------------


def _apply_pca_to_tde_matrix(
    tde_matrix: np.ndarray,
    n_components: int,
    random_seed: int,
) -> np.ndarray:
    """Reduce TDE-expanded features to principal component time-series.

    Fits PCA on the TDE-expanded matrix and returns the transformed data.
    If the requested number of components exceeds the matrix dimensions,
    it is automatically reduced to ``min(n_rows, n_cols)``.

    Args:
        tde_matrix: 2-D array of shape ``(n_valid_timepoints, n_tde_features)``.
        n_components: Desired number of principal components.
        random_seed: Random seed for PCA reproducibility.

    Returns:
        2-D array of shape ``(n_valid_timepoints, actual_n_components)``.
    """
    n_rows, n_cols = tde_matrix.shape
    actual_components = min(n_components, n_rows, n_cols)
    pca = PCA(n_components=actual_components, random_state=random_seed)
    pca_timeseries: np.ndarray = pca.fit_transform(tde_matrix)
    return pca_timeseries


def _epoch_pca_timeseries(
    pca_timeseries: np.ndarray,
    sfreq: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Epoch PCA time-series and summarise each epoch with mean + variance.

    Identical to Script 13's ``_epoch_pca_timeseries``.

    Args:
        pca_timeseries: 2-D array ``(n_valid_timepoints, n_components)``.
        sfreq: Sampling frequency in Hz.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        Tuple of ``(epoch_features, epoch_onsets_s)`` where:
        - ``epoch_features``: 2-D array ``(n_epochs, 2 * n_components)``.
        - ``epoch_onsets_s``: 1-D array of epoch onset times in seconds.
    """
    n_valid_timepoints = pca_timeseries.shape[0]

    epoch_onsets_s = create_epoch_onsets(
        n_samples_total=n_valid_timepoints,
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )

    if len(epoch_onsets_s) == 0:
        return np.empty((0, 0), dtype=np.float64), epoch_onsets_s

    n_samples_per_epoch = int(epoch_duration_s * sfreq)
    epoch_features_list: list[np.ndarray] = []

    for onset_s in epoch_onsets_s:
        sample_start = int(round(onset_s * sfreq))
        sample_end = sample_start + n_samples_per_epoch
        if sample_end > n_valid_timepoints:
            break
        epoch_window = pca_timeseries[sample_start:sample_end, :]
        epoch_mean = epoch_window.mean(axis=0)
        epoch_var = epoch_window.var(axis=0)
        epoch_features_list.append(np.concatenate([epoch_mean, epoch_var]))

    if not epoch_features_list:
        return np.empty((0, 0), dtype=np.float64), epoch_onsets_s

    epoch_features = np.stack(epoch_features_list, axis=0)
    epoch_onsets_s = epoch_onsets_s[: len(epoch_features_list)]
    return epoch_features, epoch_onsets_s


# ---------------------------------------------------------------------------
# Spectral TDE epoch extraction (main feature extraction function)
# ---------------------------------------------------------------------------


def extract_spectral_tde_epochs_for_run(
    run_config: dict,
    eeg_raw: mne.io.Raw,
    events_df: pd.DataFrame,
    roi_channels: list[str],
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Extract spectral-TDE + PCA epochs for all luminance video segments.

    Identical pipeline to Script 13's ``extract_raw_tde_epochs_for_run``
    except step 2 computes multitaper band-power before TDE instead of
    using raw EEG amplitudes.

    Args:
        run_config: Run metadata dict with keys ``id``, ``acq``, ``task``,
            ``block``.
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts with keys: ``X``, ``y``, ``video_id``,
        ``video_identifier``, ``run_id``, ``acq``.
    """
    run_id: str = run_config["id"]
    acq: str = run_config["acq"]
    sfreq: float = eeg_raw.info["sfreq"]

    luminance_events = events_df[
        events_df["trial_type"] == "video_luminance"
    ].reset_index(drop=True)

    if luminance_events.empty:
        logger.warning("Run %s: no video_luminance events found.", run_id)
        return []

    epoch_entries: list[dict] = []

    for _, event_row in luminance_events.iterrows():
        segment_epochs = _process_single_video_segment(
            event_row=event_row,
            run_id=run_id,
            acq=acq,
            eeg_raw=eeg_raw,
            roi_channels=roi_channels,
            sfreq=sfreq,
            epoch_duration_s=epoch_duration_s,
            epoch_step_s=epoch_step_s,
        )
        epoch_entries.extend(segment_epochs)

    return epoch_entries


def _process_single_video_segment(
    event_row: pd.Series,
    run_id: str,
    acq: str,
    eeg_raw: mne.io.Raw,
    roi_channels: list[str],
    sfreq: float,
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Process a single video_luminance segment through the spectral-TDE pipeline.

    Steps mirror Script 13's ``_process_single_video_segment`` exactly,
    with the only difference being step 2: multitaper band-power instead
    of raw signal.

    Args:
        event_row: A single row from the events DataFrame.
        run_id: Run identifier string.
        acq: Acquisition label.
        eeg_raw: Loaded MNE Raw object.
        roi_channels: ROI channel names present in the EEG.
        sfreq: Sampling frequency in Hz.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts for this segment (may be empty).
    """
    stim_id = int(event_row["stim_id"])
    video_id = stim_id - 100
    onset_s = float(event_row["onset"])
    duration_s = float(event_row["duration"])

    csv_filename = LUMINANCE_CSV_MAP.get(video_id)
    if csv_filename is None:
        logger.warning(
            "Run %s: video_id %d not in LUMINANCE_CSV_MAP, skipping.",
            run_id,
            video_id,
        )
        return []

    csv_path = STIMULI_PATH / csv_filename
    try:
        luminance_df = load_luminance_csv(csv_path)
    except FileNotFoundError:
        logger.warning(
            "Run %s: luminance CSV not found: %s, skipping segment.",
            run_id,
            csv_path,
        )
        return []

    # --- Step 1: Crop EEG to segment, ROI channels only ---
    t_start = onset_s
    t_stop = onset_s + duration_s
    try:
        video_eeg = eeg_raw.copy().crop(tmin=t_start, tmax=t_stop)
    except ValueError as exc:
        logger.warning(
            "Run %s: could not crop EEG [%.2f, %.2f]: %s",
            run_id,
            t_start,
            t_stop,
            exc,
        )
        return []

    eeg_data = video_eeg.get_data(picks=roi_channels)

    # --- Step 2: Spectral TDE (band-power + TDE on continuous signal) ---
    window_half = TDE_WINDOW_HALF
    min_samples_for_tde = 2 * window_half + 1
    if eeg_data.shape[1] < min_samples_for_tde:
        logger.warning(
            "Run %s: segment too short for TDE (video_id=%d, "
            "n_samples=%d, need>=%d).",
            run_id,
            video_id,
            eeg_data.shape[1],
            min_samples_for_tde,
        )
        return []

    tde_matrix = apply_tde_on_continuous_spectral(
        eeg_data, sfreq, SPECTRAL_BANDS, window_half
    )

    # --- Step 3: Apply PCA ---
    pca_timeseries = _apply_pca_to_tde_matrix(
        tde_matrix, n_components=TDE_PCA_COMPONENTS, random_seed=RANDOM_SEED
    )

    # --- Step 4: Epoch PCA time-series ---
    epoch_features, epoch_onsets_s = _epoch_pca_timeseries(
        pca_timeseries,
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )

    if epoch_features.shape[0] == 0:
        logger.warning(
            "Run %s: segment too short for epochs after TDE border "
            "removal (video_id=%d). Skipping.",
            run_id,
            video_id,
        )
        return []

    # --- Step 5: Pair with luminance targets ---
    border_offset_s = window_half / sfreq
    luminance_epoch_onsets_s = epoch_onsets_s + border_offset_s

    luminance_targets = interpolate_luminance_to_epochs(
        luminance_df=luminance_df,
        epoch_onsets_s=luminance_epoch_onsets_s,
        epoch_duration_s=epoch_duration_s,
    )

    # --- Build epoch entries ---
    video_identifier = f"{video_id}_{acq}"
    segment_entries: list[dict] = []
    for idx in range(epoch_features.shape[0]):
        segment_entries.append(
            {
                "X": epoch_features[idx],
                "y": float(luminance_targets[idx]),
                "video_id": video_id,
                "video_identifier": video_identifier,
                "run_id": run_id,
                "acq": acq,
            }
        )

    return segment_entries


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_cv_results(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate bar plots of CV metrics per fold and save to disk.

    Args:
        results_df: DataFrame with columns TestVideo, PearsonR,
            SpearmanRho, RMSE.
        output_dir: Directory to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["PearsonR", "SpearmanRho", "RMSE"]
    titles = ["Pearson r", "Spearman ρ", "RMSE"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(results_df["TestVideo"], results_df[metric], color="darkorange")
        mean_val = results_df[metric].mean()
        ax.axhline(
            mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.4f}"
        )
        ax.set_xlabel("Test Video")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Fold")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Spectral TDE Model (Multitaper + TDE + PCA → Luminance) — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_tde_model_cv_results.png", dpi=150
    )
    plt.close("all")


def plot_predictions_per_fold(
    fold_predictions: list[dict],
    output_dir: Path,
) -> None:
    """Plot true vs predicted luminance for each CV fold.

    Args:
        fold_predictions: List of dicts with keys ``test_video``,
            ``y_true``, ``y_pred``.
        output_dir: Directory to save the figure.
    """
    n_folds = len(fold_predictions)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), squeeze=False)

    for idx, fold_data in enumerate(fold_predictions):
        ax = axes[0, idx]
        n_points = len(fold_data["y_true"])
        ax.plot(range(n_points), fold_data["y_true"], label="True", alpha=0.7)
        ax.plot(range(n_points), fold_data["y_pred"], label="Pred", alpha=0.7)
        ax.set_title(f"Test: {fold_data['test_video']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Luminance")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Predictions — Spectral TDE Model — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_tde_model_predictions.png", dpi=150
    )
    plt.close("all")


def plot_comparison_with_raw_tde(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot comparison of spectral TDE model vs raw TDE model metrics.

    Args:
        results_df: Spectral TDE model results DataFrame.
        output_dir: Directory to save the figure.
    """
    raw_tde_csv = (
        RESULTS_PATH / "raw_tde" / f"sub-{SUBJECT}_raw_tde_model_results.csv"
    )
    if not raw_tde_csv.exists():
        logger.warning(
            "Raw TDE model results not found at %s, skipping comparison.",
            raw_tde_csv,
        )
        return

    raw_tde_df = pd.read_csv(raw_tde_csv)

    metrics = ["PearsonR", "SpearmanRho", "RMSE"]
    labels = ["Pearson r", "Spearman ρ", "RMSE"]

    spectral_means = [results_df[metric].mean() for metric in metrics]
    raw_means = [raw_tde_df[metric].mean() for metric in metrics]

    x_positions = np.arange(len(metrics))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x_positions - bar_width / 2,
        spectral_means,
        bar_width,
        label="Spectral TDE (script 12)",
        color="darkorange",
    )
    ax.bar(
        x_positions + bar_width / 2,
        raw_means,
        bar_width,
        label="Raw TDE (script 13)",
        color="steelblue",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Spectral TDE vs Raw TDE Model — sub-{SUBJECT}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_spectral_tde_vs_raw_tde_comparison.png",
        dpi=150,
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# JSON data dictionary sidecar (BIDS compliance)
# ---------------------------------------------------------------------------


def _write_results_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the results CSV.

    Args:
        json_path: Destination path for the JSON sidecar file.
    """
    data_dictionary: dict[str, dict[str, str]] = {
        "Subject": {
            "Description": "Subject identifier (BIDS entity, zero-padded)",
            "DataType": "string",
        },
        "Acq": {
            "Description": "Acquisition label (a or b) for the test fold video",
            "DataType": "string",
        },
        "Model": {
            "Description": "Model identifier (always 'spectral_tde' for this pipeline)",
            "DataType": "string",
        },
        "TestVideo": {
            "Description": (
                "Video identifier held out as the test set in LOVO-CV "
                "(format: videoID_acq)"
            ),
            "DataType": "string",
        },
        "TrainSize": {
            "Description": "Number of training epochs in this fold",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TestSize": {
            "Description": "Number of test epochs in this fold",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TrainPearsonR": {
            "Description": (
                "Pearson correlation coefficient between predicted and "
                "actual z-scored luminance on the training set (for "
                "overfitting diagnostics)"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "PearsonR": {
            "Description": (
                "Pearson correlation coefficient between predicted and "
                "actual z-scored luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "SpearmanRho": {
            "Description": (
                "Spearman rank correlation coefficient between predicted "
                "and actual z-scored luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "RMSE": {
            "Description": (
                "Root mean squared error between predicted and actual "
                "z-scored luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "z-score units",
        },
        "BestAlpha": {
            "Description": (
                "Best Ridge regularization alpha selected by "
                "GridSearchCV (Spearman ρ scoring) with "
                "LeaveOneGroupOut for this fold"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(epoch_duration_override: float | None = None) -> None:
    """Execute the spectral-TDE luminance prediction pipeline.

    Args:
        epoch_duration_override: If provided, overrides the configured
            EPOCH_DURATION_S.  EPOCH_STEP_S is kept at 100 ms
            (overlap = duration − 100 ms).

    Identical to Script 13's ``run_pipeline`` except the feature extraction
    uses multitaper band-power instead of raw EEG amplitudes.

    Steps:
        1. Set random seed for reproducibility and log it.
        2. Determine ROI channels from the configured posterior set.
        3. For each run, load EEG + events, extract spectral-TDE + PCA epochs.
        4. Z-score normalize luminance targets per video group.
        5. Run Leave-One-Video-Out CV with StandardScaler → Ridge
           (GridSearchCV + LeaveOneGroupOut for alpha selection).
        6. Run permutation test if configured.
        7. Save results CSV, plots, and comparison with raw TDE model.
    """
    if epoch_duration_override is not None:
        active_epoch_duration = epoch_duration_override
        active_epoch_step = 0.1  # fixed 100 ms step (overlap = duration − 100 ms)
    else:
        active_epoch_duration = EPOCH_DURATION_S
        active_epoch_step = EPOCH_STEP_S

    epoch_ms_tag = f"{int(active_epoch_duration * 1000)}ms"

    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"12 — Spectral TDE Luminance Model (sub-{SUBJECT}) — epoch={epoch_ms_tag}")
    print("=" * 60)

    output_dir = RESULTS_PATH / "tde" / epoch_ms_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    n_bands = len(SPECTRAL_BANDS)
    n_spectral_features = len(roi_channels) * n_bands
    window_size = 2 * TDE_WINDOW_HALF + 1
    n_tde_features = n_spectral_features * window_size
    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")
    print(f"Spectral bands ({n_bands}): {list(SPECTRAL_BANDS.keys())}")
    print(f"Spectral features per time-point: {n_spectral_features}")
    print(f"TDE window: ±{TDE_WINDOW_HALF} → {window_size} time-points")
    print(f"TDE-expanded features per time-point: {n_tde_features}")
    print(f"PCA components: {TDE_PCA_COMPONENTS}")

    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Collect spectral-TDE + PCA epochs across all runs
    # ------------------------------------------------------------------
    all_epoch_entries: list[dict] = []

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
            print("  WARNING: Events TSV not found, skipping.")
            continue

        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(
            str(vhdr_path), preload=True, verbose=False
        )

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        recording_roi = select_roi_channels(eeg_raw.ch_names, POSTERIOR_CHANNELS)

        run_epochs = extract_spectral_tde_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            roi_channels=recording_roi,
            epoch_duration_s=active_epoch_duration,
            epoch_step_s=active_epoch_step,
        )
        print(f"  Spectral-TDE epochs extracted: {len(run_epochs)}")
        all_epoch_entries.extend(run_epochs)

    print(f"\nTotal spectral-TDE epochs collected: {len(all_epoch_entries)}")
    if not all_epoch_entries:
        print("ERROR: No epochs generated across all runs. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Z-score normalization per video
    # ------------------------------------------------------------------
    all_epoch_entries = zscore_per_video(all_epoch_entries)
    print("Z-score normalization applied per video.")

    # ------------------------------------------------------------------
    # 4. Leave-One-Video-Out CV
    # ------------------------------------------------------------------
    folds = leave_one_video_out_split(all_epoch_entries)
    print(f"Number of CV folds: {len(folds)}")

    results_list: list[dict] = []
    fold_predictions: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        X_train = np.array([entry["X"] for entry in train_entries])
        y_train = np.array([entry["y"] for entry in train_entries])
        X_test = np.array([entry["X"] for entry in test_entries])
        y_test = np.array([entry["y"] for entry in test_entries])

        print(
            f"\n  Fold: test={test_video} | "
            f"train={X_train.shape[0]} ({X_train.shape[1]} features) | "
            f"test={X_test.shape[0]}"
        )

        # Pipeline: StandardScaler → Ridge
        # Alpha selected via GridSearchCV (Spearman ρ scoring) with
        # LeaveOneGroupOut on training videos to prevent data leakage.
        groups_train = np.array([e["video_identifier"] for e in train_entries])
        pipeline = make_pipeline(
            StandardScaler(),
            Ridge(),
        )
        grid_search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
            cv=LeaveOneGroupOut(),
            scoring=spearman_scoring,
            refit=True,
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        y_pred = grid_search.predict(X_test)
        y_pred_train = grid_search.predict(X_train)
        best_alpha = grid_search.best_params_["ridge__alpha"]

        metrics = evaluate_fold(y_test, y_pred)
        train_metrics = evaluate_fold(y_train, y_pred_train)
        print(
            f"    Train r={train_metrics['PearsonR']:.4f} | "
            f"Test r={metrics['PearsonR']:.4f} | "
            f"Spearman ρ={metrics['SpearmanRho']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"BestAlpha={best_alpha}"
        )

        results_list.append(
            {
                "Subject": SUBJECT,
                "Acq": test_entries[0]["acq"],
                "Model": "spectral_tde",
                "TestVideo": test_video,
                "TrainSize": len(y_train),
                "TestSize": len(y_test),
                "TrainPearsonR": train_metrics["PearsonR"],
                **metrics,
                "BestAlpha": best_alpha,
            }
        )

        fold_predictions.append(
            {
                "test_video": test_video,
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )

    # ------------------------------------------------------------------
    # 5. Permutation test
    # ------------------------------------------------------------------
    if N_PERMUTATIONS > 0:

        def _build_and_evaluate(epoch_entries: list[dict]) -> float:
            """Evaluate mean Pearson r across LOVO-CV folds for permutation test."""
            perm_folds = leave_one_video_out_split(epoch_entries)
            pearson_values: list[float] = []
            for perm_train, perm_test, _ in perm_folds:
                perm_X_train = np.array([e["X"] for e in perm_train])
                perm_y_train = np.array([e["y"] for e in perm_train])
                perm_X_test = np.array([e["X"] for e in perm_test])
                perm_y_test = np.array([e["y"] for e in perm_test])

                perm_groups_train = np.array(
                    [e["video_identifier"] for e in perm_train]
                )
                perm_pipeline = make_pipeline(
                    StandardScaler(),
                    Ridge(),
                )
                perm_grid = GridSearchCV(
                    perm_pipeline,
                    param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
                    cv=LeaveOneGroupOut(),
                    scoring=spearman_scoring,
                    refit=True,
                )
                perm_grid.fit(perm_X_train, perm_y_train, groups=perm_groups_train)
                perm_y_pred = perm_grid.predict(perm_X_test)
                fold_metrics = evaluate_fold(perm_y_test, perm_y_pred)
                pearson_values.append(fold_metrics["PearsonR"])
            return float(np.mean(pearson_values))

        print(f"\nRunning permutation test ({N_PERMUTATIONS} iterations)...")
        perm_results = run_permutation_test(
            epoch_entries=all_epoch_entries,
            build_and_evaluate_fn=_build_and_evaluate,
            n_permutations=N_PERMUTATIONS,
            random_seed=RANDOM_SEED,
        )
        print(
            f"  Observed r: {perm_results['observed_r']:.4f} | "
            f"p-value: {perm_results['p_value']:.4f}"
        )

        np.savez(
            output_dir / f"sub-{SUBJECT}_tde_model_permutation.npz",
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
        )
        plot_permutation_histogram(
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
            output_path=output_dir
            / f"sub-{SUBJECT}_tde_model_permutation_hist.png",
        )
        print(f"  Permutation results saved to: {output_dir}")
    else:
        print("\nPermutation test skipped (N_PERMUTATIONS=0).")

    # ------------------------------------------------------------------
    # 6. Summary and save
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_list)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(
        f"\nMean Train r: {results_df['TrainPearsonR'].mean():.4f}"
        f" | Mean Test r: {results_df['PearsonR'].mean():.4f}"
        f" | Mean Spearman ρ: {results_df['SpearmanRho'].mean():.4f}"
        f" | Mean RMSE: {results_df['RMSE'].mean():.4f}"
    )

    csv_path = output_dir / f"sub-{SUBJECT}_tde_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_sidecar_path = csv_path.with_suffix(".json")
    _write_results_json_sidecar(json_sidecar_path)
    print(f"JSON data dictionary saved: {json_sidecar_path}")

    plot_cv_results(results_df, output_dir)
    plot_predictions_per_fold(fold_predictions, output_dir)
    plot_comparison_with_raw_tde(results_df, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Spectral TDE model pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectral TDE luminance model")
    parser.add_argument(
        "--epoch-duration",
        type=float,
        default=None,
        help="Epoch duration in seconds (default: use config value)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_pipeline(epoch_duration_override=args.epoch_duration)
