"""Raw TDE model: GLHMM Time Delay Embedding on continuous raw EEG → luminance.

Pipeline that bypasses spectral feature extraction. Instead, it applies the
GLHMM TDE preprocessing protocol (Vidaurre et al., 2025, Nature Protocols)
directly on the continuous preprocessed raw EEG signal from ROI channels,
then epochs the PCA-reduced time-series and extracts covariance features per
epoch, fitting Ridge regression via LOVO-CV.

Pipeline per video segment:
    1. Crop preprocessed EEG to video_luminance segment (ROI channels only)
    2. Apply GLHMM TDE pipeline on continuous raw signal:
       build_data_tde() → preprocess_data() (standardise + PCA)
       → (n_valid_timepoints, pca_components) array
    3. Epoch PCA time-series (EPOCH_DURATION_S, EPOCH_STEP_S)
    4. Compute covariance matrix per epoch → upper triangle (1-D feature vector)
       Length: n_components * (n_components + 1) // 2
    5. Pair epochs with interpolated luminance targets

Evaluation: Leave-One-Video-Out CV with StandardScaler → Ridge
(GridSearchCV + LeaveOneGroupOut for alpha selection).
Results saved to ``results/modeling/luminance/raw_tde/``.

References:
    Vidaurre et al. (2025). A protocol for time-delay embedded hidden
    Markov modelling of brain data. Nature Protocols.
    https://doi.org/10.1038/s41596-025-01300-2

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.2, 7b.1
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure scripts/modeling is on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure src is on sys.path so internal module imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
    STIMULI_PATH,
    SUBJECT,
    TARGET_ZSCORE,
    TDE_PCA_COMPONENTS,
    TDE_WINDOW_HALF,
    XDF_PATH,
)

from campeones_analysis.luminance.evaluation import compute_r2_score
from campeones_analysis.luminance.features import compute_epoch_covariance
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
from campeones_analysis.luminance.tde_glhmm import apply_glhmm_tde_pipeline

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
# Path resolution helpers (shared with scripts 10–12)
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


def _resolve_order_matrix_path(run_config: dict) -> Path | None:
    """Build the Order Matrix .xlsx path for a run.

    Args:
        run_config: Dictionary with keys ``acq``, ``block``.

    Returns:
        Path to the Order Matrix Excel file, or ``None`` if not found.
    """
    acq = run_config["acq"].upper()
    block = run_config["block"]
    order_matrix_path = (
        XDF_PATH
        / f"sub-{SUBJECT}"
        / f"order_matrix_{SUBJECT}_{acq}_{block}_VR.xlsx"
    )
    return order_matrix_path if order_matrix_path.exists() else None


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
# Pure computation helpers
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
        Dictionary with ``R2``, ``PearsonR``, ``SpearmanRho``, and ``RMSE``.
    """
    has_variance = np.std(y_pred) > 1e-9
    pearson_r = float(pearsonr(y_true, y_pred)[0]) if has_variance else 0.0
    spearman_rho = (
        float(spearmanr(y_true, y_pred).correlation) if has_variance else 0.0
    )
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = compute_r2_score(y_true, y_pred)
    return {"R2": r2, "PearsonR": pearson_r, "SpearmanRho": spearman_rho, "RMSE": rmse}


# ---------------------------------------------------------------------------
# Raw TDE epoch extraction (main feature extraction function)
# ---------------------------------------------------------------------------


def extract_raw_tde_epochs_for_run(
    run_config: dict,
    eeg_raw: mne.io.Raw,
    events_df: pd.DataFrame,
    roi_channels: list[str],
    bad_epochs_map: dict[int, list[int]] | None = None,
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Extract GLHMM-TDE + covariance epochs for all luminance video segments in a run.

    For each ``video_luminance`` segment in the events:

    1. Crop the preprocessed EEG to that segment using ROI channels only.
    2. Apply GLHMM TDE pipeline on the continuous raw signal:
       ``build_data_tde()`` + ``preprocess_data()`` (standardise + PCA).
    3. Epoch the PCA time-series into windows of ``epoch_duration_s`` with
       ``epoch_step_s`` step.
    4. Compute covariance matrix per epoch and extract upper triangle as
       1-D feature vector.
    5. Pair each epoch with the interpolated physical luminance target.
    6. Skip segments too short for at least one epoch after TDE border
       removal and log a warning.

    Args:
        run_config: Run metadata dict with keys ``id``, ``acq``, ``task``,
            ``block``.
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run (must contain columns
            ``trial_type``, ``stim_id``, ``onset``, ``duration``).
        roi_channels: List of ROI channel names present in the EEG.
        bad_epochs_map: Dictionary mapping video_id -> list of bad epoch indices.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts with keys: ``X`` (1-D feature vector),
        ``y`` (float), ``video_id`` (int), ``video_identifier`` (str),
        ``run_id`` (str), ``acq`` (str).

    Requirements: 4.1, 4.2, 4.3, 4.4
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
        stim_id = int(event_row["stim_id"])
        video_id = stim_id - 100
        bad_epochs = bad_epochs_map.get(video_id, []) if bad_epochs_map else []
        
        segment_epochs = _process_single_video_segment(
            event_row=event_row,
            run_id=run_id,
            acq=acq,
            eeg_raw=eeg_raw,
            roi_channels=roi_channels,
            bad_epochs=bad_epochs,
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
    bad_epochs: list[int],
    sfreq: float,
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Process a single video_luminance segment through the GLHMM TDE pipeline.

    Handles cropping, GLHMM TDE (build_data_tde + preprocess_data), epoching,
    covariance feature extraction, and luminance pairing for one video segment.

    Args:
        event_row: A single row from the events DataFrame.
        run_id: Run identifier string (e.g. ``"002"``).
        acq: Acquisition label (``"a"`` or ``"b"``).
        eeg_raw: Loaded MNE Raw object.
        roi_channels: ROI channel names present in the EEG.
        bad_epochs: List of bad epoch indices according to autoreject.
        sfreq: Sampling frequency in Hz.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts for this segment (may be empty).

    Requirements: 4.1, 4.2, 4.3, 4.4
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

    # eeg_data shape: (n_channels, n_samples) → transpose to (n_samples, n_channels)
    eeg_data_channels_first: np.ndarray = video_eeg.get_data(picks=roi_channels)
    eeg_data_time_major: np.ndarray = eeg_data_channels_first.T

    n_timepoints, n_channels = eeg_data_time_major.shape
    min_samples_for_tde = 2 * TDE_WINDOW_HALF + 1
    if n_timepoints < min_samples_for_tde:
        logger.warning(
            "Run %s: segment too short for TDE (video_id=%d, "
            "n_samples=%d, need>=%d).",
            run_id,
            video_id,
            n_timepoints,
            min_samples_for_tde,
        )
        return []

    # --- Step 2: Apply GLHMM TDE pipeline (Req 4.1, 4.2) ---
    # indices: (n_sessions, 2) array with [start, end] sample indices
    segment_indices = np.array([[0, n_timepoints]])
    pca_timeseries = apply_glhmm_tde_pipeline(
        eeg_data=eeg_data_time_major,
        indices=segment_indices,
        tde_lags=TDE_WINDOW_HALF,
        pca_components=TDE_PCA_COMPONENTS,
    )

    n_valid_timepoints = pca_timeseries.shape[0]

    # --- Step 3: Epoch PCA time-series ---
    epoch_onsets_s = create_epoch_onsets(
        n_samples_total=n_valid_timepoints,
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )

    if len(epoch_onsets_s) == 0:
        logger.warning(
            "Run %s: segment too short for epochs after TDE border "
            "removal (video_id=%d). Skipping.",
            run_id,
            video_id,
        )
        return []

    n_samples_per_epoch = int(epoch_duration_s * sfreq)
    epoch_features_list: list[np.ndarray] = []
    valid_onsets: list[float] = []

    for onset_epoch_s in epoch_onsets_s:
        sample_start = int(round(onset_epoch_s * sfreq))
        sample_end = sample_start + n_samples_per_epoch
        if sample_end > n_valid_timepoints:
            break
        pca_epoch = pca_timeseries[sample_start:sample_end, :]
        # --- Step 4: Covariance feature extraction (Req 4.3, 4.4) ---
        covariance_features = compute_epoch_covariance(pca_epoch)
        epoch_features_list.append(covariance_features)
        valid_onsets.append(onset_epoch_s)

    if not epoch_features_list:
        logger.warning(
            "Run %s: no valid epochs extracted (video_id=%d). Skipping.",
            run_id,
            video_id,
        )
        return []

    epoch_features = np.stack(epoch_features_list, axis=0)
    epoch_onsets_trimmed = np.array(valid_onsets)

    # --- Step 5: Pair with luminance targets ---
    # TDE border removal shifts the valid region by TDE_WINDOW_HALF samples.
    # Add the border offset so epoch onsets are relative to the original
    # video start for correct luminance interpolation.
    border_offset_s = TDE_WINDOW_HALF / sfreq
    luminance_epoch_onsets_s = epoch_onsets_trimmed + border_offset_s

    luminance_targets = interpolate_luminance_to_epochs(
        luminance_df=luminance_df,
        epoch_onsets_s=luminance_epoch_onsets_s,
        epoch_duration_s=epoch_duration_s,
    )

    # --- Build epoch entries ---
    video_identifier = f"{video_id}_{acq}"
    segment_entries: list[dict] = []
    
    total_epochs_in_seg = epoch_features.shape[0]
    dropped_count = 0

    for idx in range(total_epochs_in_seg):
        if idx in bad_epochs:
            dropped_count += 1
            continue
            
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

    if dropped_count > 0:
        logger.info(
            "Run %s (video %d): Dropped %d/%d raw_tde epochs according to AutoReject QA", 
            run_id, video_id, dropped_count, total_epochs_in_seg
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

    Creates a 1×4 subplot figure with per-fold bars for R², Pearson r,
    Spearman ρ, and RMSE, each annotated with a dashed mean line.

    Args:
        results_df: DataFrame with columns TestVideo, R2, PearsonR,
            SpearmanRho, RMSE.
        output_dir: Directory to save the figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    metrics = ["R2", "PearsonR", "SpearmanRho", "RMSE"]
    titles = ["R²", "Pearson r", "Spearman ρ", "RMSE"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(results_df["TestVideo"], results_df[metric], color="steelblue")
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
        f"Raw TDE Model (GLHMM TDE + Covariance → Luminance) — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_raw_tde_model_cv_results.png", dpi=150
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
        f"Predictions — Raw TDE Model — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_raw_tde_model_predictions.png", dpi=150
    )
    plt.close("all")


def plot_comparison_with_spectral_tde(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot comparison of raw TDE model vs spectral TDE model metrics.

    Loads the spectral TDE model results CSV from script 12 (if available)
    and generates a grouped bar chart comparing R², Pearson r, Spearman ρ,
    and RMSE.

    Args:
        results_df: Raw TDE model results DataFrame.
        output_dir: Directory to save the figure.
    """
    spectral_tde_csv = (
        RESULTS_PATH / "tde" / f"sub-{SUBJECT}_tde_model_results.csv"
    )
    if not spectral_tde_csv.exists():
        logger.warning(
            "Spectral TDE model results not found at %s, skipping comparison.",
            spectral_tde_csv,
        )
        return

    spectral_tde_df = pd.read_csv(spectral_tde_csv)

    metrics = ["PearsonR", "SpearmanRho", "RMSE"]
    labels = ["Pearson r", "Spearman ρ", "RMSE"]

    spectral_tde_means = [spectral_tde_df[metric].mean() for metric in metrics]
    raw_tde_means = [results_df[metric].mean() for metric in metrics]

    x_positions = np.arange(len(metrics))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x_positions - bar_width / 2,
        spectral_tde_means,
        bar_width,
        label="Spectral TDE (script 12)",
        color="darkorange",
    )
    ax.bar(
        x_positions + bar_width / 2,
        raw_tde_means,
        bar_width,
        label="Raw TDE + Covariance (script 13)",
        color="steelblue",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Spectral TDE vs Raw TDE Model — sub-{SUBJECT}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_raw_tde_vs_spectral_tde_comparison.png",
        dpi=150,
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# JSON data dictionary sidecar (BIDS compliance)
# ---------------------------------------------------------------------------


def _write_results_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the results CSV.

    Describes each column in the per-fold results CSV, including data type,
    description, and units where applicable.

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
            "Description": "Model identifier (always 'raw_tde' for this pipeline)",
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
                "actual luminance on the training set (for overfitting diagnostics)"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "R2": {
            "Description": (
                "Coefficient of determination (R²) between predicted and "
                "actual luminance on the test fold. Quantifies the proportion "
                "of luminance variability explained by the EEG model."
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "PearsonR": {
            "Description": (
                "Pearson correlation coefficient between predicted and "
                "actual luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "SpearmanRho": {
            "Description": (
                "Spearman rank correlation coefficient between predicted "
                "and actual luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "RMSE": {
            "Description": (
                "Root mean squared error between predicted and actual "
                "luminance on the test fold"
            ),
            "DataType": "float",
            "Units": "luminance units (0-255 raw or z-score if TARGET_ZSCORE=True)",
        },
        "BestAlpha": {
            "Description": (
                "Best Ridge regularization alpha selected by "
                "GridSearchCV (Spearman ρ scoring) with "
                "LeaveOneGroupOut for this fold"
            ),
            "DataType": "float",
        },
        "TargetZScore": {
            "Description": (
                "Whether luminance targets were z-score normalized per video "
                "(True) or kept as raw values 0-255 (False)"
            ),
            "DataType": "boolean",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(epoch_duration_override: float | None = None) -> None:
    """Execute the raw-TDE luminance prediction pipeline.

    Uses the GLHMM TDE preprocessing protocol (Vidaurre et al., 2025) with
    covariance feature extraction per epoch. Luminance targets are raw (0-255)
    by default; set TARGET_ZSCORE=True in config to apply z-score per video.

    Args:
        epoch_duration_override: If provided, overrides the configured
            EPOCH_DURATION_S.  EPOCH_STEP_S is kept at 100 ms
            (overlap = duration − 100 ms).

    Steps:
        1. Set random seed for reproducibility and log it.
        2. Determine ROI channels from the configured posterior set.
        3. For each run, load EEG + events, extract GLHMM-TDE + covariance epochs.
        4. Optionally z-score normalize luminance targets per video (TARGET_ZSCORE).
        5. Run Leave-One-Video-Out CV with StandardScaler → Ridge
           (GridSearchCV + LeaveOneGroupOut for alpha selection).
        6. Run permutation test if configured.
        7. Save results CSV, plots, and comparison with spectral TDE model.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.2, 7b.1
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
    print(
        f"13 — Raw TDE Luminance Model (GLHMM + Covariance) "
        f"(sub-{SUBJECT}) — epoch={epoch_ms_tag}"
    )
    print(f"     TARGET_ZSCORE={TARGET_ZSCORE}")
    print("=" * 60)

    output_dir = RESULTS_PATH / "raw_tde" / epoch_ms_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    n_pca_features = TDE_PCA_COMPONENTS * (TDE_PCA_COMPONENTS + 1) // 2
    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")
    print(f"TDE lags: ±{TDE_WINDOW_HALF}")
    print(f"PCA components: {TDE_PCA_COMPONENTS}")
    print(f"Covariance features per epoch: {n_pca_features}")

    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Collect GLHMM-TDE + covariance epochs across all runs
    # ------------------------------------------------------------------
    qa_tsv_path = PROJECT_ROOT / "results" / "qa" / "eeg" / f"sub-{SUBJECT}_eeg_qa_autoreject.tsv"
    qa_df = None
    if qa_tsv_path.exists():
        print(f"Loading QA AutoReject parameters from: {qa_tsv_path}")
        qa_df = pd.read_csv(qa_tsv_path, sep="\t")
    else:
        print(f"WARNING: No QA TSV found at {qa_tsv_path}. Running WITHOUT epoch rejection.")

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

        run_bad_epochs: dict[int, list[int]] = {}
        if qa_df is not None:
            run_str_match = str(run_config["id"])
            run_acq_match = str(run_config["acq"])
            run_qa = qa_df[
                (qa_df["RunID"].astype(str).str.zfill(3) == run_str_match.zfill(3)) &
                (qa_df["Acq"].astype(str) == run_acq_match)
            ]
            for _, qa_row in run_qa.iterrows():
                v_id = int(qa_row["VideoID"])
                bad_idx_list = ast.literal_eval(qa_row["BadEpochsIdx"])
                run_bad_epochs[v_id] = bad_idx_list

        run_epochs = extract_raw_tde_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            roi_channels=recording_roi,
            bad_epochs_map=run_bad_epochs,
            epoch_duration_s=active_epoch_duration,
            epoch_step_s=active_epoch_step,
        )
        print(f"  GLHMM-TDE covariance epochs extracted: {len(run_epochs)}")
        all_epoch_entries.extend(run_epochs)

    print(f"\nTotal epochs collected: {len(all_epoch_entries)}")
    if not all_epoch_entries:
        print("ERROR: No epochs generated across all runs. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Luminance target normalization (Req 7b.1)
    # Default: raw luminance (0-255). Z-score only if TARGET_ZSCORE=True.
    # ------------------------------------------------------------------
    if TARGET_ZSCORE:
        all_epoch_entries = zscore_per_video(all_epoch_entries)
        print("Z-score normalization applied per video (TARGET_ZSCORE=True).")
    else:
        print("Using raw luminance targets (0-255) — TARGET_ZSCORE=False.")

    # ------------------------------------------------------------------
    # 4. Leave-One-Video-Out CV (Req 4.5, 4.6)
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
            f"Test R²={metrics['R2']:.4f} | "
            f"Test r={metrics['PearsonR']:.4f} | "
            f"Spearman ρ={metrics['SpearmanRho']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"BestAlpha={best_alpha}"
        )

        results_list.append(
            {
                "Subject": SUBJECT,
                "Acq": test_entries[0]["acq"],
                "Model": "raw_tde",
                "TestVideo": test_video,
                "TrainSize": len(y_train),
                "TestSize": len(y_test),
                "TrainPearsonR": train_metrics["PearsonR"],
                **metrics,
                "BestAlpha": best_alpha,
                "TargetZScore": TARGET_ZSCORE,
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
            output_dir / f"sub-{SUBJECT}_raw_tde_model_permutation.npz",
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
        )
        plot_permutation_histogram(
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
            output_path=output_dir
            / f"sub-{SUBJECT}_raw_tde_model_permutation_hist.png",
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
        f" | Mean R²: {results_df['R2'].mean():.4f}"
        f" | Mean Test r: {results_df['PearsonR'].mean():.4f}"
        f" | Mean Spearman ρ: {results_df['SpearmanRho'].mean():.4f}"
        f" | Mean RMSE: {results_df['RMSE'].mean():.4f}"
    )

    csv_path = output_dir / f"sub-{SUBJECT}_raw_tde_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_sidecar_path = csv_path.with_suffix(".json")
    _write_results_json_sidecar(json_sidecar_path)
    print(f"JSON data dictionary saved: {json_sidecar_path}")

    plot_cv_results(results_df, output_dir)
    plot_predictions_per_fold(fold_predictions, output_dir)
    plot_comparison_with_spectral_tde(results_df, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Raw TDE model pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Raw TDE luminance model (GLHMM TDE + covariance features)"
    )
    parser.add_argument(
        "--epoch-duration",
        type=float,
        default=None,
        help="Epoch duration in seconds (default: use config value)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_pipeline(epoch_duration_override=args.epoch_duration)
