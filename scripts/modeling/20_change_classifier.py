"""Change Classifier: binary classification of luminance change vs stability.

Evaluates whether EEG can discriminate epochs of luminance *transition*
(|ΔL| > CHANGE_THRESHOLD) from *stable* epochs (|ΔL| ≤ CHANGE_THRESHOLD).

Pipeline:
    1. Extract GLHMM TDE + covariance epochs (same as scripts 13 and 19).
    2. Apply ``compute_delta_luminance`` → discard first epoch per video.
    3. Apply ``compute_change_labels`` with CHANGE_THRESHOLD → binary targets.
    4. Run LOVO_CV: undersample majority class in training set only, then
       StandardScaler → LogisticRegression (solver='lbfgs', max_iter=1000).
    5. Report accuracy, precision, recall, F1-score, AUC-ROC per fold.
    6. Handle edge case: if only one class in a fold, log warning and report
       NaN metrics for that fold.

Results saved to ``results/modeling/luminance/change_classification/``.

References:
    Vidaurre et al. (2025). A protocol for time-delay embedded hidden
    Markov modelling of brain data. Nature Protocols.
    https://doi.org/10.1038/s41596-025-01300-2

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import EEG_CHANNELS
from config_luminance import (
    CHANGE_THRESHOLD,
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    LUMINANCE_CSV_MAP,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    TDE_PCA_COMPONENTS,
    TDE_WINDOW_HALF,
)
from campeones_analysis.luminance.features import compute_epoch_covariance
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
from campeones_analysis.luminance.targets import (
    compute_change_labels,
    compute_delta_luminance,
)
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
CHANGE_RESULTS_PATH: Path = RESULTS_PATH / "change_classification"


# ---------------------------------------------------------------------------
# Path resolution helpers (mirrors script 19)
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


def undersample_majority_class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample the majority class to balance binary training data.

    Finds the minority class count and randomly samples that many examples
    from the majority class, producing a balanced training set. The test
    set is never modified.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary label array of shape (n_samples,) with values 0 or 1.
        random_seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (X_balanced, y_balanced) with equal class counts matching
        the minority class size.

    Requirements: 9.2
    """
    rng = np.random.default_rng(random_seed)
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    minority_count = min(len(class_0_indices), len(class_1_indices))

    if len(class_0_indices) > len(class_1_indices):
        majority_indices = class_0_indices
        minority_indices = class_1_indices
    else:
        majority_indices = class_1_indices
        minority_indices = class_0_indices

    sampled_majority = rng.choice(majority_indices, size=minority_count, replace=False)
    balanced_indices = np.concatenate([minority_indices, sampled_majority])
    balanced_indices = np.sort(balanced_indices)

    return X_train[balanced_indices], y_train[balanced_indices]


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


def evaluate_classification_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
) -> dict[str, float]:
    """Compute classification metrics for a single CV fold.

    Handles the edge case where only one class is present in the test set
    by returning NaN for metrics that require both classes.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1, shape (n_samples,).
            If ``None``, AUC-ROC is reported as NaN.

    Returns:
        Dictionary with ``Accuracy``, ``Precision``, ``Recall``, ``F1``,
        and ``AUC_ROC``.

    Requirements: 9.4
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "Only one class (%s) present in test fold — reporting NaN metrics.",
            unique_classes[0],
        )
        return {
            "Accuracy": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "AUC_ROC": float("nan"),
        }

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if y_pred_proba is not None:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
    else:
        auc_roc = float("nan")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc_roc,
    }


# ---------------------------------------------------------------------------
# Epoch extraction (mirrors script 19's extract_raw_tde_epochs_for_run)
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

    Follows the same pipeline as scripts 13 and 19:
    1. Crop EEG to video_luminance segment (ROI channels only).
    2. Apply GLHMM TDE pipeline: build_data_tde() + preprocess_data() (PCA).
    3. Epoch PCA time-series.
    4. Compute covariance matrix per epoch → upper triangle feature vector.
    5. Pair epochs with interpolated luminance targets (raw, pre-delta).

    Note: Delta and change-label computation is applied *after* all epochs
    are collected, so raw luminance targets are stored here.

    Args:
        run_config: Run metadata dict with keys ``id``, ``acq``, ``task``, ``block``.
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        List of epoch entry dicts with keys: ``X``, ``y``, ``video_id``,
        ``video_identifier``, ``run_id``, ``acq``.

    Requirements: 4.1, 4.2, 4.3, 4.4, 9.1
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

    # Step 5: Pair with raw luminance targets (delta + labels applied later)
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


def plot_classification_cv_results(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate bar plots of classification metrics per CV fold.

    Creates a 1×5 subplot figure with per-fold bars for Accuracy, Precision,
    Recall, F1, and AUC-ROC, each annotated with a dashed mean line.
    NaN values (single-class folds) are shown as zero-height bars.

    Args:
        results_df: DataFrame with columns Subject, TestVideo, Accuracy,
            Precision, Recall, F1, AUC_ROC.
        output_dir: Directory to save the figure.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    bar_color = "steelblue"

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for col_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col_idx]
        metric_values = results_df[metric].fillna(0)
        ax.bar(results_df["TestVideo"], metric_values, color=bar_color)
        mean_val = results_df[metric].mean(skipna=True)
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            label=f"Mean={mean_val:.4f}",
        )
        ax.set_xlabel("Test Video")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Change Classifier (GLHMM TDE + Covariance → Binary) — sub-{SUBJECT}",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_change_classifier_cv_results.png", dpi=150
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# JSON data dictionary sidecar (BIDS compliance)
# ---------------------------------------------------------------------------


def _write_results_json_sidecar(json_path: Path) -> None:
    """Write a BIDS-compliant JSON data dictionary for the classification results CSV.

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
        "TestVideo": {
            "Description": (
                "Video identifier held out as the test set in LOVO-CV "
                "(format: videoID_acq)"
            ),
            "DataType": "string",
        },
        "TrainSize": {
            "Description": "Number of training epochs in this fold (after delta discard and undersampling)",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TestSize": {
            "Description": "Number of test epochs in this fold (after delta discard)",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TrainClass0": {
            "Description": "Number of class-0 (stable) epochs in balanced training set",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TrainClass1": {
            "Description": "Number of class-1 (change) epochs in balanced training set",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TestClass0": {
            "Description": "Number of class-0 (stable) epochs in test set",
            "DataType": "integer",
            "Units": "epochs",
        },
        "TestClass1": {
            "Description": "Number of class-1 (change) epochs in test set",
            "DataType": "integer",
            "Units": "epochs",
        },
        "Accuracy": {
            "Description": (
                "Classification accuracy on the test fold. NaN if only one "
                "class present in test set."
            ),
            "DataType": "float",
            "Units": "dimensionless [0, 1]",
        },
        "Precision": {
            "Description": (
                "Precision (positive predictive value) for class 1 on the test fold. "
                "NaN if only one class present in test set."
            ),
            "DataType": "float",
            "Units": "dimensionless [0, 1]",
        },
        "Recall": {
            "Description": (
                "Recall (sensitivity) for class 1 on the test fold. "
                "NaN if only one class present in test set."
            ),
            "DataType": "float",
            "Units": "dimensionless [0, 1]",
        },
        "F1": {
            "Description": (
                "F1-score (harmonic mean of precision and recall) for class 1 "
                "on the test fold. NaN if only one class present in test set."
            ),
            "DataType": "float",
            "Units": "dimensionless [0, 1]",
        },
        "AUC_ROC": {
            "Description": (
                "Area under the ROC curve on the test fold. "
                "NaN if only one class present in test set."
            ),
            "DataType": "float",
            "Units": "dimensionless [0, 1]",
        },
        "ChangeThreshold": {
            "Description": (
                "Absolute delta luminance threshold used to define change epochs "
                "(|ΔL| > threshold → class 1)."
            ),
            "DataType": "float",
            "Units": "luminance units (0–255 scale)",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core CV runner
# ---------------------------------------------------------------------------


def run_lovo_cv_classification(
    change_epoch_entries: list[dict],
) -> list[dict]:
    """Run LOVO_CV for the change classifier.

    Executes Leave-One-Video-Out cross-validation with:
    - Undersampling of majority class in training set only (Req 9.2).
    - StandardScaler → LogisticRegression (solver='lbfgs', max_iter=1000).
    - Metrics: accuracy, precision, recall, F1, AUC-ROC per fold (Req 9.4).
    - Edge case: if only one class in test fold, log warning and report NaN.

    Args:
        change_epoch_entries: Epoch entries with binary change labels in ``y``
            (output of ``compute_change_labels``).

    Returns:
        List of per-fold result dicts with classification metrics.

    Requirements: 9.2, 9.3, 9.4
    """
    folds = leave_one_video_out_split(change_epoch_entries)
    results_list: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        X_train_full = np.array([entry["X"] for entry in train_entries])
        y_train_full = np.array([entry["y"] for entry in train_entries], dtype=int)
        X_test = np.array([entry["X"] for entry in test_entries])
        y_test = np.array([entry["y"] for entry in test_entries], dtype=int)

        # Undersample majority class in training set only (Req 9.2)
        X_train_balanced, y_train_balanced = undersample_majority_class(
            X_train_full, y_train_full, random_seed=RANDOM_SEED
        )

        train_class0 = int(np.sum(y_train_balanced == 0))
        train_class1 = int(np.sum(y_train_balanced == 1))
        test_class0 = int(np.sum(y_test == 0))
        test_class1 = int(np.sum(y_test == 1))

        print(
            f"  Fold: test={test_video} | "
            f"train_balanced={X_train_balanced.shape[0]} "
            f"(0:{train_class0}, 1:{train_class1}) | "
            f"test={X_test.shape[0]} (0:{test_class0}, 1:{test_class1})"
        )

        # Check for single-class training set (degenerate fold)
        if len(np.unique(y_train_balanced)) < 2:
            logger.warning(
                "Fold test=%s: only one class in training set after undersampling. "
                "Reporting NaN metrics.",
                test_video,
            )
            metrics = {
                "Accuracy": float("nan"),
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "AUC_ROC": float("nan"),
            }
            results_list.append(
                {
                    "Subject": SUBJECT,
                    "TestVideo": test_video,
                    "TrainSize": len(y_train_balanced),
                    "TestSize": len(y_test),
                    "TrainClass0": train_class0,
                    "TrainClass1": train_class1,
                    "TestClass0": test_class0,
                    "TestClass1": test_class1,
                    **metrics,
                    "ChangeThreshold": CHANGE_THRESHOLD,
                }
            )
            continue

        # Fit pipeline: StandardScaler → LogisticRegression (Req 9.3)
        classifier_pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        )
        classifier_pipeline.fit(X_train_balanced, y_train_balanced)

        y_pred = classifier_pipeline.predict(X_test)
        # Get probability for class 1 for AUC-ROC
        y_pred_proba: np.ndarray | None = None
        if hasattr(classifier_pipeline, "predict_proba"):
            proba_matrix = classifier_pipeline.predict_proba(X_test)
            y_pred_proba = proba_matrix[:, 1]

        metrics = evaluate_classification_fold(y_test, y_pred, y_pred_proba)

        print(
            f"    Acc={metrics['Accuracy']:.4f} | "
            f"Prec={metrics['Precision']:.4f} | "
            f"Rec={metrics['Recall']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"AUC={metrics['AUC_ROC']:.4f}"
        )

        results_list.append(
            {
                "Subject": SUBJECT,
                "TestVideo": test_video,
                "TrainSize": len(y_train_balanced),
                "TestSize": len(y_test),
                "TrainClass0": train_class0,
                "TrainClass1": train_class1,
                "TestClass0": test_class0,
                "TestClass1": test_class1,
                **metrics,
                "ChangeThreshold": CHANGE_THRESHOLD,
            }
        )

    return results_list


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the change classifier pipeline.

    Classifies epochs as 'change' (|ΔL| > CHANGE_THRESHOLD) or 'stable'
    using the same GLHMM TDE + covariance feature pipeline as scripts 13
    and 19, with LOVO_CV and majority-class undersampling.

    Steps:
        1. Log random seed for reproducibility.
        2. Collect GLHMM-TDE + covariance epochs across all runs (raw targets).
        3. Apply ``compute_delta_luminance`` → discard first epoch per video.
        4. Apply ``compute_change_labels`` with CHANGE_THRESHOLD → binary labels.
        5. Run LOVO_CV with undersampling + LogisticRegression.
        6. Save CSV + JSON sidecar and bar plot.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(
        f"20 — Change Classifier (GLHMM TDE + Covariance → Binary) — sub-{SUBJECT}"
    )
    print(f"     CHANGE_THRESHOLD: {CHANGE_THRESHOLD}")
    print("=" * 60)

    output_dir = CHANGE_RESULTS_PATH
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
            print("  WARNING: Events file not found, skipping.")
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
                tde_data, _, _, vid_id, _, _, bad_ep = result
                tde_segments.append(tde_data)
                segment_metadata.append({
                    "eeg_raw_path": vhdr_path,
                    "event_row": event_row,
                    "run_config": run_config,
                    "roi_channels": recording_roi,
                    "bad_epochs": bad_ep,
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
    # 3. Compute delta luminance — discard first epoch per video (Req 9.1)
    # ------------------------------------------------------------------
    delta_entries = compute_delta_luminance(all_epoch_entries)
    print(f"Delta epochs after discarding first per video: {len(delta_entries)}")

    if not delta_entries:
        print("ERROR: No delta epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 4. Generate binary change labels (Req 9.1)
    # ------------------------------------------------------------------
    change_entries = compute_change_labels(delta_entries, threshold=CHANGE_THRESHOLD)
    n_change = sum(1 for e in change_entries if e["y"] == 1)
    n_stable = sum(1 for e in change_entries if e["y"] == 0)
    print(
        f"Binary labels: {len(change_entries)} total | "
        f"change(1)={n_change} | stable(0)={n_stable} | "
        f"threshold={CHANGE_THRESHOLD}"
    )

    # ------------------------------------------------------------------
    # 5. Run LOVO_CV with undersampling + LogisticRegression (Req 9.2, 9.3)
    # ------------------------------------------------------------------
    print("\nRunning LOVO_CV with undersampling + LogisticRegression...")
    results_list = run_lovo_cv_classification(change_entries)

    # ------------------------------------------------------------------
    # 6. Summary and save (Req 9.5)
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_list)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]:
        mean_val = results_df[metric].mean(skipna=True)
        print(f"  Mean {metric}: {mean_val:.4f}")

    csv_path = output_dir / f"sub-{SUBJECT}_change_classifier_cv_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_sidecar_path = csv_path.with_suffix(".json")
    _write_results_json_sidecar(json_sidecar_path)
    print(f"JSON data dictionary saved: {json_sidecar_path}")

    plot_classification_cv_results(results_df, output_dir)
    print(f"Plot saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Change classifier pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
