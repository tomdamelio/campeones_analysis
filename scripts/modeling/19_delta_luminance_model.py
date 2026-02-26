"""Delta Luminance Model: predict ΔL = L_i − L_{i-1} from EEG.

Evaluates whether EEG encodes luminance *transitions* rather than absolute
luminance levels. Uses the same GLHMM TDE + covariance feature pipeline as
script 13, but replaces the raw luminance target with delta luminance.

Two variants are evaluated:
    - ``delta_raw``: raw delta values (no normalization).
    - ``delta_zscore``: z-score normalized delta per video.

Pipeline per variant:
    1. Extract GLHMM TDE + covariance epochs (same as script 13).
    2. Apply ``compute_delta_luminance`` → discard first epoch per video.
    3. Optionally z-score normalize delta targets per video.
    4. Run LOVO_CV: StandardScaler → Ridge (GridSearchCV + LeaveOneGroupOut,
       Spearman ρ scoring).
    5. Report R², Pearson r, Spearman ρ, RMSE per fold and average.

Results saved to ``results/modeling/luminance/delta_luminance/``.

References:
    Vidaurre et al. (2025). A protocol for time-delay embedded hidden
    Markov modelling of brain data. Nature Protocols.
    https://doi.org/10.1038/s41596-025-01300-2

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import json
import logging
import sys
import ast
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import EEG_CHANNELS
from config_luminance import (
    DELTA_ZSCORE,
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    LUMINANCE_CSV_MAP,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA_GRID,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    TDE_PCA_COMPONENTS,
    TDE_WINDOW_HALF,
    XDF_PATH,
)
from campeones_analysis.luminance.evaluation import compute_r2_score
from campeones_analysis.luminance.features import compute_epoch_covariance
from campeones_analysis.luminance.normalization import zscore_per_video
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
from campeones_analysis.luminance.targets import compute_delta_luminance
from campeones_analysis.luminance.tde_glhmm import (
    apply_tde_only,
    fit_global_pca,
    apply_global_pca,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
DELTA_RESULTS_PATH: Path = RESULTS_PATH / "delta_luminance"

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
# Path resolution helpers (mirrors script 13)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ROI channel selection
# ---------------------------------------------------------------------------


def select_roi_channels(
    available_channels: list[str],
    roi_channels: list[str],
) -> list[str]:
    """Select ROI channels that exist in the available EEG channels.

    Args:
        available_channels: Channel names present in the EEG recording.
        roi_channels: Desired ROI channel names.

    Returns:
        List of channel names present in both lists, preserving roi_channels order.
    """
    available_set = set(available_channels)
    selected: list[str] = []
    for channel_name in roi_channels:
        if channel_name in available_set:
            selected.append(channel_name)
        else:
            logger.warning("ROI channel %s not found in EEG, skipping.", channel_name)
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
        y_true: Ground-truth delta luminance values.
        y_pred: Predicted delta luminance values.

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
# Epoch extraction (mirrors script 13's extract_raw_tde_epochs_for_run)
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

    Follows the same pipeline as script 13:
    1. Crop EEG to video_luminance segment (ROI channels only).
    2. Apply GLHMM TDE pipeline: build_data_tde() + preprocess_data() (PCA).
    3. Epoch PCA time-series.
    4. Compute covariance matrix per epoch → upper triangle feature vector.
    5. Pair epochs with interpolated luminance targets (raw, pre-delta).

    Note: Delta computation is applied *after* all epochs are collected,
    so raw luminance targets are stored here.

    Args:
        run_config: Run metadata dict with keys ``id``, ``acq``, ``task``, ``block``.
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.
        bad_epochs_map: Dictionary mapping video_id to bad epoch indices.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts with keys: ``X``, ``y``, ``video_id``,
        ``video_identifier``, ``run_id``, ``acq``.

    Requirements: 4.1, 4.2, 4.3, 4.4, 8.1
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
    pca_model=None,
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Process a single video_luminance segment through the GLHMM TDE pipeline.

    Args:
        event_row: A single row from the events DataFrame.
        run_id: Run identifier string (e.g. ``"002"``).
        acq: Acquisition label (``"a"`` or ``"b"``).
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

    # Step 1: Crop EEG to segment, ROI channels only
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

    eeg_data_channels_first: np.ndarray = video_eeg.get_data(picks=roi_channels)
    eeg_data_time_major: np.ndarray = eeg_data_channels_first.T

    n_timepoints, _ = eeg_data_time_major.shape
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

    # Step 2: Apply TDE only (no PCA yet)
    segment_indices = np.array([[0, n_timepoints]])
    tde_data, _ = apply_tde_only(
        eeg_data=eeg_data_time_major,
        indices=segment_indices,
        tde_lags=TDE_WINDOW_HALF,
    )

    # Step 2b: Apply global PCA (if provided)
    if pca_model is not None:
        pca_timeseries = apply_global_pca(tde_data, pca_model)
    else:
        # Pass 1: return TDE data for global PCA fitting
        return tde_data, eeg_data_time_major, luminance_df, video_id, acq, run_id, bad_epochs

    n_valid_timepoints = pca_timeseries.shape[0]

    # Step 3: Epoch PCA time-series
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
        # Step 4: Covariance feature extraction (Req 4.3, 4.4)
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

    # Step 5: Pair with raw luminance targets (delta applied later)
    border_offset_s = TDE_WINDOW_HALF / sfreq
    luminance_epoch_onsets_s = epoch_onsets_trimmed + border_offset_s

    luminance_targets = interpolate_luminance_to_epochs(
        luminance_df=luminance_df,
        epoch_onsets_s=luminance_epoch_onsets_s,
        epoch_duration_s=epoch_duration_s,
    )

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
# Plotting
# ---------------------------------------------------------------------------


def plot_delta_cv_results(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate bar plots of CV metrics per fold for both delta variants.

    Creates a 2×4 subplot figure (one row per variant) with per-fold bars
    for R², Pearson r, Spearman ρ, and RMSE, each annotated with a dashed
    mean line.

    Args:
        results_df: DataFrame with columns Subject, Model, TestVideo,
            R2, PearsonR, SpearmanRho, RMSE.
        output_dir: Directory to save the figure.
    """
    variant_names = results_df["Model"].unique()
    n_variants = len(variant_names)
    metrics = ["R2", "PearsonR", "SpearmanRho", "RMSE"]
    metric_labels = ["R²", "Pearson r", "Spearman ρ", "RMSE"]
    variant_colors = {"delta_raw": "steelblue", "delta_zscore": "darkorange"}

    fig, axes = plt.subplots(
        n_variants, 4, figsize=(20, 5 * n_variants), squeeze=False
    )

    for row_idx, variant in enumerate(variant_names):
        variant_df = results_df[results_df["Model"] == variant]
        color = variant_colors.get(variant, "steelblue")
        for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]
            ax.bar(variant_df["TestVideo"], variant_df[metric], color=color)
            mean_val = variant_df[metric].mean()
            ax.axhline(
                mean_val,
                color="red",
                linestyle="--",
                label=f"Mean={mean_val:.4f}",
            )
            ax.set_xlabel("Test Video")
            ax.set_ylabel(label)
            ax.set_title(f"{label} — {variant}")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Delta Luminance Model (GLHMM TDE + Covariance → ΔL) — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_delta_luminance_cv_results.png", dpi=150
    )
    plt.close("all")


def plot_delta_predictions_per_fold(
    fold_predictions: list[dict],
    output_dir: Path,
) -> None:
    """Plot true vs predicted delta luminance for each CV fold and variant.

    Args:
        fold_predictions: List of dicts with keys ``variant``, ``test_video``,
            ``y_true``, ``y_pred``.
        output_dir: Directory to save the figure.
    """
    variants = sorted(set(fp["variant"] for fp in fold_predictions))
    test_videos = sorted(set(fp["test_video"] for fp in fold_predictions))
    n_variants = len(variants)
    n_videos = len(test_videos)

    fig, axes = plt.subplots(
        n_variants, n_videos, figsize=(5 * n_videos, 4 * n_variants), squeeze=False
    )

    for row_idx, variant in enumerate(variants):
        for col_idx, test_video in enumerate(test_videos):
            ax = axes[row_idx, col_idx]
            matching = [
                fp
                for fp in fold_predictions
                if fp["variant"] == variant and fp["test_video"] == test_video
            ]
            if matching:
                fold_data = matching[0]
                n_points = len(fold_data["y_true"])
                ax.plot(range(n_points), fold_data["y_true"], label="True ΔL", alpha=0.7)
                ax.plot(range(n_points), fold_data["y_pred"], label="Pred ΔL", alpha=0.7)
                ax.legend(fontsize=7)
            ax.set_title(f"{variant} | Test: {test_video}", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("ΔL")

    fig.suptitle(
        f"Delta Luminance Predictions — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_delta_luminance_predictions.png", dpi=150
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# JSON data dictionary sidecar (BIDS compliance)
# ---------------------------------------------------------------------------


def _write_results_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the delta results CSV.

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
        "Model": {
            "Description": (
                "Delta luminance variant: 'delta_raw' (raw ΔL values) or "
                "'delta_zscore' (z-score normalized ΔL per video)"
            ),
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
            "Description": "Number of training epochs in this fold (after delta discard)",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TestSize": {
            "Description": "Number of test epochs in this fold (after delta discard)",
            "DataType": "integer",
            "Units": "epochs",
        },
        "R2": {
            "Description": (
                "Coefficient of determination (R²) between predicted and actual "
                "delta luminance on the test fold."
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "PearsonR": {
            "Description": (
                "Pearson correlation coefficient between predicted and actual "
                "delta luminance on the test fold."
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "SpearmanRho": {
            "Description": (
                "Spearman rank correlation coefficient between predicted and actual "
                "delta luminance on the test fold."
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "RMSE": {
            "Description": (
                "Root mean squared error between predicted and actual delta "
                "luminance on the test fold."
            ),
            "DataType": "float",
            "Units": "luminance units (raw delta) or z-score units (delta_zscore)",
        },
        "BestAlpha": {
            "Description": (
                "Best Ridge regularization alpha selected by GridSearchCV "
                "(Spearman ρ scoring) with LeaveOneGroupOut for this fold."
            ),
            "DataType": "float",
        },
        "DeltaZScore": {
            "Description": (
                "Whether delta luminance targets were z-score normalized per video "
                "(True for delta_zscore variant, False for delta_raw variant)."
            ),
            "DataType": "boolean",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core CV runner for a single delta variant
# ---------------------------------------------------------------------------


def run_lovo_cv_for_variant(
    delta_epoch_entries: list[dict],
    variant_name: str,
) -> tuple[list[dict], list[dict]]:
    """Run LOVO_CV for a single delta luminance variant.

    Executes Leave-One-Video-Out cross-validation with
    StandardScaler → Ridge (GridSearchCV + LeaveOneGroupOut, Spearman ρ
    scoring) on the provided delta epoch entries.

    Args:
        delta_epoch_entries: Epoch entries with delta luminance targets
            (output of ``compute_delta_luminance``, optionally z-scored).
        variant_name: Label for this variant (``"delta_raw"`` or
            ``"delta_zscore"``).

    Returns:
        Tuple of (results_list, fold_predictions_list) where:
            - results_list: List of per-fold metric dicts.
            - fold_predictions_list: List of dicts with ``variant``,
              ``test_video``, ``y_true``, ``y_pred``.
    """
    folds = leave_one_video_out_split(delta_epoch_entries)
    results_list: list[dict] = []
    fold_predictions: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        X_train = np.array([entry["X"] for entry in train_entries])
        y_train = np.array([entry["y"] for entry in train_entries])
        X_test = np.array([entry["X"] for entry in test_entries])
        y_test = np.array([entry["y"] for entry in test_entries])

        print(
            f"    [{variant_name}] Fold: test={test_video} | "
            f"train={X_train.shape[0]} | test={X_test.shape[0]}"
        )

        groups_train = np.array([e["video_identifier"] for e in train_entries])
        pipeline = make_pipeline(StandardScaler(), Ridge())
        grid_search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
            cv=LeaveOneGroupOut(),
            scoring=spearman_scoring,
            refit=True,
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        y_pred = grid_search.predict(X_test)
        best_alpha = grid_search.best_params_["ridge__alpha"]

        metrics = evaluate_fold(y_test, y_pred)
        is_zscore = variant_name == "delta_zscore"

        print(
            f"      R²={metrics['R2']:.4f} | "
            f"r={metrics['PearsonR']:.4f} | "
            f"ρ={metrics['SpearmanRho']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"α={best_alpha}"
        )

        results_list.append(
            {
                "Subject": SUBJECT,
                "Model": variant_name,
                "TestVideo": test_video,
                "TrainSize": len(y_train),
                "TestSize": len(y_test),
                **metrics,
                "BestAlpha": best_alpha,
                "DeltaZScore": is_zscore,
            }
        )
        fold_predictions.append(
            {
                "variant": variant_name,
                "test_video": test_video,
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )

    return results_list, fold_predictions


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the delta luminance prediction pipeline.

    Evaluates two variants of the delta luminance target:
        - ``delta_raw``: raw ΔL = L_i − L_{i-1} values.
        - ``delta_zscore``: z-score normalized ΔL per video.

    Both variants use the same GLHMM TDE + covariance feature pipeline as
    script 13, with LOVO_CV (StandardScaler → Ridge, Spearman ρ scoring).

    Steps:
        1. Log random seed for reproducibility.
        2. Collect GLHMM-TDE + covariance epochs across all runs (raw targets).
        3. Apply ``compute_delta_luminance`` → discard first epoch per video.
        4. For each variant: optionally z-score normalize, run LOVO_CV.
        5. Save combined CSV + JSON sidecar and plots.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"19 — Delta Luminance Model (GLHMM TDE + Covariance → ΔL) — sub-{SUBJECT}")
    print(f"     DELTA_ZSCORE config default: {DELTA_ZSCORE}")
    print(f"     Evaluating BOTH variants: delta_raw and delta_zscore")
    print("=" * 60)

    output_dir = DELTA_RESULTS_PATH
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
    # 2. PASS 1: Collect TDE-embedded data from all segments
    # ------------------------------------------------------------------
    qa_tsv_path = PROJECT_ROOT / "results" / "qa" / "eeg" / f"sub-{SUBJECT}_eeg_qa_autoreject.tsv"
    qa_df = None
    if qa_tsv_path.exists():
        print(f"Loading QA AutoReject parameters from: {qa_tsv_path}")
        qa_df = pd.read_csv(qa_tsv_path, sep="\t")
    else:
        print(f"WARNING: No QA TSV found at {qa_tsv_path}. Running WITHOUT epoch rejection.")

    tde_segments: list[np.ndarray] = []
    segment_metadata: list[dict] = []

    print("\n--- PASS 1: Collecting TDE-embedded segments ---")
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

        luminance_events = events_df[
            events_df["trial_type"] == "video_luminance"
        ].reset_index(drop=True)

        sfreq = eeg_raw.info["sfreq"]
        for _, event_row in luminance_events.iterrows():
            stim_id = int(event_row["stim_id"])
            video_id = stim_id - 100
            bad_epochs = run_bad_epochs.get(video_id, [])

            result = _process_single_video_segment(
                event_row=event_row,
                run_id=run_config["id"],
                acq=run_config["acq"],
                eeg_raw=eeg_raw,
                roi_channels=recording_roi,
                bad_epochs=bad_epochs,
                sfreq=sfreq,
                pca_model=None,
            )
            if isinstance(result, tuple):
                tde_data, _, _, vid_id, _, _, _ = result
                tde_segments.append(tde_data)
                segment_metadata.append({
                    "eeg_raw_path": vhdr_path,
                    "event_row": event_row,
                    "run_config": run_config,
                    "roi_channels": recording_roi,
                    "bad_epochs": bad_epochs,
                    "sfreq": sfreq,
                })
                print(f"  Video {vid_id}: TDE shape = {tde_data.shape}")

    if not tde_segments:
        print("ERROR: No TDE segments collected. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Fit GLOBAL PCA on concatenated TDE data
    # ------------------------------------------------------------------
    print("\n--- Fitting GLOBAL PCA ---")
    global_pca = fit_global_pca(tde_segments, TDE_PCA_COMPONENTS)

    # ------------------------------------------------------------------
    # 4. PASS 2: Project with global PCA → epoch → covariance
    # ------------------------------------------------------------------
    print("\n--- PASS 2: Projecting with global PCA + extracting epochs ---")
    all_epoch_entries: list[dict] = []

    for meta in segment_metadata:
        eeg_raw = mne.io.read_raw_brainvision(
            str(meta["eeg_raw_path"]), preload=True, verbose=False
        )
        run_epochs = _process_single_video_segment(
            event_row=meta["event_row"],
            run_id=meta["run_config"]["id"],
            acq=meta["run_config"]["acq"],
            eeg_raw=eeg_raw,
            roi_channels=meta["roi_channels"],
            bad_epochs=meta["bad_epochs"],
            sfreq=meta["sfreq"],
            pca_model=global_pca,
        )
        if isinstance(run_epochs, list):
            print(f"  Epochs extracted: {len(run_epochs)}")
            all_epoch_entries.extend(run_epochs)

    print(f"\nTotal raw epochs collected: {len(all_epoch_entries)}")
    if not all_epoch_entries:
        print("ERROR: No epochs generated across all runs. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Compute delta luminance — discard first epoch per video (Req 8.1, 8.2)
    # ------------------------------------------------------------------
    delta_entries_raw = compute_delta_luminance(all_epoch_entries)
    print(f"Delta epochs after discarding first per video: {len(delta_entries_raw)}")

    if not delta_entries_raw:
        print("ERROR: No delta epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 4. Evaluate both variants (Req 8.3)
    # ------------------------------------------------------------------
    all_results: list[dict] = []
    all_fold_predictions: list[dict] = []

    # Variant A: delta_raw (no normalization)
    print("\n--- Variant: delta_raw ---")
    raw_results, raw_preds = run_lovo_cv_for_variant(
        delta_epoch_entries=delta_entries_raw,
        variant_name="delta_raw",
    )
    all_results.extend(raw_results)
    all_fold_predictions.extend(raw_preds)

    # Variant B: delta_zscore (z-score per video)
    print("\n--- Variant: delta_zscore ---")
    delta_entries_zscore = zscore_per_video(delta_entries_raw)
    zscore_results, zscore_preds = run_lovo_cv_for_variant(
        delta_epoch_entries=delta_entries_zscore,
        variant_name="delta_zscore",
    )
    all_results.extend(zscore_results)
    all_fold_predictions.extend(zscore_preds)

    # ------------------------------------------------------------------
    # 5. Summary and save (Req 8.5, 8.6)
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for variant_name in ["delta_raw", "delta_zscore"]:
        variant_df = results_df[results_df["Model"] == variant_name]
        print(f"\n  {variant_name}:")
        print(
            f"    Mean R²={variant_df['R2'].mean():.4f} | "
            f"Mean r={variant_df['PearsonR'].mean():.4f} | "
            f"Mean ρ={variant_df['SpearmanRho'].mean():.4f} | "
            f"Mean RMSE={variant_df['RMSE'].mean():.4f}"
        )

    csv_path = output_dir / f"sub-{SUBJECT}_delta_luminance_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_sidecar_path = csv_path.with_suffix(".json")
    _write_results_json_sidecar(json_sidecar_path)
    print(f"JSON data dictionary saved: {json_sidecar_path}")

    plot_delta_cv_results(results_df, output_dir)
    plot_delta_predictions_per_fold(all_fold_predictions, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Delta luminance model pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
