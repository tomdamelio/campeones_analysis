"""Baseline models and z-score vs raw target evaluation for luminance prediction.

Three analyses are implemented:

1. **Shuffle Baseline** (Req 2.1–2.5): Permutes luminance labels within each
   video group, trains StandardScaler → Ridge with GridSearchCV + LOVO_CV,
   and repeats N_SHUFFLE_ITERATIONS times to build a null distribution of
   R², Pearson r, Spearman ρ, and RMSE.

2. **Mean Baseline** (Req 3.1–3.4): In each LOVO fold, predicts the training
   mean for all test epochs. Reports the same four metrics.

3. **Z-score vs Raw evaluation** (Req 7b.1–7b.5): Trains the raw TDE model
   with both target representations (raw 0–255 and z-score per video) and
   compares metrics.

All results are saved to:
    - ``results/modeling/luminance/baselines/``   (shuffle + mean)
    - ``results/modeling/luminance/zscore_evaluation/``  (z-score vs raw)

Each CSV output is accompanied by a BIDS-compliant JSON sidecar.

CSV schema: Subject, Model, TestVideo, R2, PearsonR, SpearmanRho, RMSE
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

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
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    EXPERIMENTAL_VIDEOS,
    LUMINANCE_CSV_MAP,
    N_SHUFFLE_ITERATIONS,
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

import ast

from campeones_analysis.luminance.evaluation import compute_r2_score
from campeones_analysis.luminance.features import apply_time_delay_embedding
from campeones_analysis.luminance.normalization import zscore_per_video
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spearman scorer for GridSearchCV
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
# LOVO-CV helpers
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


def evaluate_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R², Pearson r, Spearman ρ, and RMSE for a single CV fold.

    Args:
        y_true: Ground-truth luminance values.
        y_pred: Predicted luminance values.

    Returns:
        Dictionary with keys ``R2``, ``PearsonR``, ``SpearmanRho``, ``RMSE``.
    """
    pred_std = float(np.std(y_pred))
    r2 = compute_r2_score(y_true, y_pred)
    pearson_r = float(pearsonr(y_true, y_pred)[0]) if pred_std > 1e-9 else 0.0
    spearman_rho = (
        float(spearmanr(y_true, y_pred).correlation) if pred_std > 1e-9 else 0.0
    )
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": r2,
        "PearsonR": pearson_r,
        "SpearmanRho": spearman_rho,
        "RMSE": rmse,
    }


# ---------------------------------------------------------------------------
# TDE epoch extraction (mirrors script 13, I/O-isolated)
# ---------------------------------------------------------------------------


def _apply_pca_to_tde_matrix(
    tde_matrix: np.ndarray,
    n_components: int,
    random_seed: int,
) -> np.ndarray:
    """Reduce TDE-expanded matrix to N PCA components.

    Args:
        tde_matrix: 2-D array of shape (n_valid_timepoints, n_tde_features).
        n_components: Number of PCA components to retain.
        random_seed: Random seed for PCA reproducibility.

    Returns:
        2-D array of shape (n_valid_timepoints, n_components).
    """
    from sklearn.decomposition import PCA

    effective_components = min(n_components, tde_matrix.shape[0], tde_matrix.shape[1])
    pca = PCA(n_components=effective_components, random_state=random_seed)
    return pca.fit_transform(tde_matrix)


def _epoch_pca_timeseries(
    pca_timeseries: np.ndarray,
    sfreq: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Epoch PCA time-series and summarise each epoch as mean + variance.

    Args:
        pca_timeseries: 2-D array (n_timepoints, n_components).
        sfreq: Sampling frequency in Hz.
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        Tuple of:
            - epoch_features: 2-D array (n_epochs, 2 * n_components).
            - epoch_onsets_s: 1-D array of epoch onset times in seconds.
    """
    epoch_duration_samples = int(round(epoch_duration_s * sfreq))
    n_timepoints, n_components = pca_timeseries.shape

    epoch_onsets_s = create_epoch_onsets(
        n_samples_total=n_timepoints,
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )

    if len(epoch_onsets_s) == 0:
        return np.empty((0, 2 * n_components)), np.empty(0)

    feature_rows: list[np.ndarray] = []
    for onset_s in epoch_onsets_s:
        onset_sample = int(round(onset_s * sfreq))
        epoch_slice = pca_timeseries[
            onset_sample : onset_sample + epoch_duration_samples
        ]
        epoch_mean = epoch_slice.mean(axis=0)
        epoch_var = epoch_slice.var(axis=0)
        feature_rows.append(np.concatenate([epoch_mean, epoch_var]))

    epoch_features = np.array(feature_rows)
    return epoch_features, epoch_onsets_s


def extract_tde_epochs_for_run(
    run_config: dict,
    eeg_raw: mne.io.Raw,
    events_df: pd.DataFrame,
    roi_channels: list[str],
    bad_epochs_map: dict[int, list[int]] | None = None,
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Extract raw-TDE + PCA epochs for all luminance video segments in a run.

    Mirrors the extraction logic from script 13 to provide the same feature
    representation for baseline comparison.

    Args:
        run_config: Run metadata dict with keys ``id``, ``acq``, ``task``.
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.
        bad_epochs_map: Dictionary mapping `video_id` -> list of bad epoch indices.
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
        stim_id = int(event_row["stim_id"])
        video_id = stim_id - 100
        bad_epochs = bad_epochs_map.get(video_id, []) if bad_epochs_map else []
        
        segment_epochs = _process_video_segment(
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


def _process_video_segment(
    event_row: pd.Series,
    run_id: str,
    acq: str,
    eeg_raw: mne.io.Raw,
    roi_channels: list[str],
    bad_epochs: list[int],
    sfreq: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> list[dict]:
    """Process a single video_luminance segment through the raw-TDE pipeline.

    Args:
        event_row: A single row from the events DataFrame.
        run_id: Run identifier string.
        acq: Acquisition label (``"a"`` or ``"b"``).
        eeg_raw: Loaded MNE Raw object.
        roi_channels: ROI channel names present in the EEG.
        bad_epochs: List of epoch indices to reject for this video.
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

    min_samples_for_tde = 2 * TDE_WINDOW_HALF + 1
    if eeg_data.shape[1] < min_samples_for_tde:
        logger.warning(
            "Run %s: segment too short for TDE (video_id=%d, n_samples=%d).",
            run_id,
            video_id,
            eeg_data.shape[1],
        )
        return []

    tde_matrix = apply_time_delay_embedding(eeg_data.T, window_half=TDE_WINDOW_HALF)
    pca_timeseries = _apply_pca_to_tde_matrix(
        tde_matrix, n_components=TDE_PCA_COMPONENTS, random_seed=RANDOM_SEED
    )

    epoch_features, epoch_onsets_s = _epoch_pca_timeseries(
        pca_timeseries,
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )

    if epoch_features.shape[0] == 0:
        logger.warning(
            "Run %s: no epochs after TDE border removal (video_id=%d).",
            run_id,
            video_id,
        )
        return []

    border_offset_s = TDE_WINDOW_HALF / sfreq
    luminance_epoch_onsets_s = epoch_onsets_s + border_offset_s

    luminance_targets = interpolate_luminance_to_epochs(
        luminance_df=luminance_df,
        epoch_onsets_s=luminance_epoch_onsets_s,
        epoch_duration_s=epoch_duration_s,
    )

    video_identifier = f"{video_id}_{acq}"
    segment_entries: list[dict] = []
    
    # Track metrics for dropping
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
            "Run %s (video %d): Dropped %d/%d epochs according to AutoReject QA", 
            run_id, video_id, dropped_count, total_epochs_in_seg
        )

    return segment_entries


# ---------------------------------------------------------------------------
# Pure computation: Shuffle Baseline (Req 2.1–2.5)
# ---------------------------------------------------------------------------


def permute_labels_within_video_groups(
    epoch_entries: list[dict],
    rng: np.random.Generator,
    video_key: str = "video_identifier",
    target_key: str = "y",
) -> list[dict]:
    """Permute luminance labels within each video group, keeping features intact.

    Implements Req 2.1: labels are shuffled independently per video group so
    that the marginal distribution of y within each video is preserved, but
    the feature–label correspondence is destroyed.

    Args:
        epoch_entries: List of epoch dicts with ``video_identifier`` and ``y``.
        rng: NumPy random Generator for reproducible permutation.
        video_key: Key used to group epochs by video.
        target_key: Key containing the luminance target.

    Returns:
        New list of epoch dicts with permuted targets; all other fields intact.
    """
    from collections import defaultdict

    # Group indices by video
    video_index_groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(epoch_entries):
        video_index_groups[entry[video_key]].append(idx)

    permuted = [copy.copy(entry) for entry in epoch_entries]

    for video_id, indices in video_index_groups.items():
        original_targets = [epoch_entries[i][target_key] for i in indices]
        shuffled_targets = rng.permutation(original_targets).tolist()
        for idx, shuffled_y in zip(indices, shuffled_targets):
            permuted[idx][target_key] = float(shuffled_y)

    return permuted


def run_shuffle_baseline_single_iteration(
    epoch_entries: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Run one shuffle baseline iteration across all LOVO folds.

    Permutes labels, then trains StandardScaler → Ridge with GridSearchCV
    (Spearman ρ scoring, LeaveOneGroupOut inner CV) on each LOVO fold.

    Args:
        epoch_entries: Epoch entries with raw (non-z-scored) targets.
        rng: NumPy random Generator for this iteration.

    Returns:
        List of per-fold result dicts with keys:
        ``Subject``, ``Model``, ``TestVideo``, ``R2``, ``PearsonR``,
        ``SpearmanRho``, ``RMSE``.
    """
    permuted_entries = permute_labels_within_video_groups(epoch_entries, rng)
    # Z-score the permuted targets per video (same normalisation as real model)
    permuted_entries = zscore_per_video(permuted_entries)

    folds = leave_one_video_out_split(permuted_entries)
    fold_results: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        x_train = np.array([e["X"] for e in train_entries])
        y_train = np.array([e["y"] for e in train_entries])
        x_test = np.array([e["X"] for e in test_entries])
        y_test = np.array([e["y"] for e in test_entries])

        groups_train = np.array([e["video_identifier"] for e in train_entries])
        pipeline = make_pipeline(StandardScaler(), Ridge())
        grid_search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
            cv=LeaveOneGroupOut(),
            scoring=spearman_scoring,
            refit=True,
        )
        grid_search.fit(x_train, y_train, groups=groups_train)
        y_pred = grid_search.predict(x_test)

        metrics = evaluate_fold_metrics(y_test, y_pred)
        fold_results.append(
            {
                "Subject": SUBJECT,
                "Model": "shuffle_baseline",
                "TestVideo": test_video,
                **metrics,
            }
        )

    return fold_results


def run_shuffle_baseline(
    epoch_entries: list[dict],
    n_iterations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build null distribution by repeating shuffle baseline N times.

    Implements Req 2.5: repeats the full LOVO-CV pipeline with permuted
    labels N_SHUFFLE_ITERATIONS times to construct a null distribution.

    Args:
        epoch_entries: All epoch entries (raw targets, not yet z-scored).
        n_iterations: Number of shuffle repetitions (Req 2.5).
        random_seed: Base seed; each iteration uses seed + iteration index.

    Returns:
        DataFrame with columns Subject, Model, TestVideo, R2, PearsonR,
        SpearmanRho, RMSE, plus an ``Iteration`` column.
    """
    all_rows: list[dict] = []

    for iteration_idx in range(n_iterations):
        iter_seed = random_seed + iteration_idx
        rng = np.random.default_rng(iter_seed)
        fold_results = run_shuffle_baseline_single_iteration(epoch_entries, rng)
        for row in fold_results:
            row["Iteration"] = iteration_idx
        all_rows.extend(fold_results)

        if (iteration_idx + 1) % 10 == 0 or iteration_idx == 0:
            print(f"  Shuffle iteration {iteration_idx + 1}/{n_iterations} done.")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Pure computation: Mean Baseline (Req 3.1–3.4)
# ---------------------------------------------------------------------------


def run_mean_baseline(
    epoch_entries: list[dict],
) -> pd.DataFrame:
    """Evaluate mean baseline across all LOVO folds.

    In each fold, predicts the arithmetic mean of training targets for every
    test epoch. Implements Req 3.1 and 3.2.

    Args:
        epoch_entries: Epoch entries with z-scored targets.

    Returns:
        DataFrame with columns Subject, Model, TestVideo, R2, PearsonR,
        SpearmanRho, RMSE.
    """
    folds = leave_one_video_out_split(epoch_entries)
    rows: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        y_train = np.array([e["y"] for e in train_entries])
        y_test = np.array([e["y"] for e in test_entries])

        training_mean = float(y_train.mean())
        y_pred = np.full_like(y_test, fill_value=training_mean)

        metrics = evaluate_fold_metrics(y_test, y_pred)
        rows.append(
            {
                "Subject": SUBJECT,
                "Model": "mean_baseline",
                "TestVideo": test_video,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure computation: Z-score vs Raw evaluation (Req 7b.1–7b.5)
# ---------------------------------------------------------------------------


def run_zscore_vs_raw_evaluation(
    epoch_entries_raw: list[dict],
) -> pd.DataFrame:
    """Compare raw TDE model performance with raw vs z-scored luminance targets.

    Trains StandardScaler → Ridge with LOVO_CV twice: once with raw luminance
    targets (0–255) and once with z-score-normalised targets per video.
    Implements Req 7b.2 and 7b.3.

    Args:
        epoch_entries_raw: Epoch entries with raw (0–255) luminance targets.

    Returns:
        DataFrame with columns Subject, Model, TestVideo, R2, PearsonR,
        SpearmanRho, RMSE. Model column is ``raw_tde_raw_target`` or
        ``raw_tde_zscore_target``.
    """
    rows: list[dict] = []

    target_variants: list[tuple[str, list[dict]]] = [
        ("raw_tde_raw_target", epoch_entries_raw),
        ("raw_tde_zscore_target", zscore_per_video(epoch_entries_raw)),
    ]

    for model_label, entries in target_variants:
        folds = leave_one_video_out_split(entries)

        for train_entries, test_entries, test_video in folds:
            x_train = np.array([e["X"] for e in train_entries])
            y_train = np.array([e["y"] for e in train_entries])
            x_test = np.array([e["X"] for e in test_entries])
            y_test = np.array([e["y"] for e in test_entries])

            groups_train = np.array([e["video_identifier"] for e in train_entries])
            pipeline = make_pipeline(StandardScaler(), Ridge())
            grid_search = GridSearchCV(
                pipeline,
                param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
                cv=LeaveOneGroupOut(),
                scoring=spearman_scoring,
                refit=True,
            )
            grid_search.fit(x_train, y_train, groups=groups_train)
            y_pred = grid_search.predict(x_test)

            metrics = evaluate_fold_metrics(y_test, y_pred)
            rows.append(
                {
                    "Subject": SUBJECT,
                    "Model": model_label,
                    "TestVideo": test_video,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I/O: JSON sidecar writers
# ---------------------------------------------------------------------------


def _write_baselines_json_sidecar(json_path: Path) -> None:
    """Write BIDS-compliant JSON data dictionary for baseline results CSV.

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
                "Baseline model identifier: 'shuffle_baseline' or 'mean_baseline'"
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
        "R2": {
            "Description": (
                "Coefficient of determination (R²) between predicted and "
                "actual luminance on the test fold"
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
            "Units": "luminance units (z-score)",
        },
        "Iteration": {
            "Description": (
                "Shuffle iteration index (0-based). Only present in the "
                "shuffle baseline CSV."
            ),
            "DataType": "integer",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


def _write_zscore_eval_json_sidecar(json_path: Path) -> None:
    """Write BIDS-compliant JSON data dictionary for z-score evaluation CSV.

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
                "Model variant: 'raw_tde_raw_target' (luminance 0–255) or "
                "'raw_tde_zscore_target' (z-score normalised per video)"
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
        "R2": {
            "Description": "Coefficient of determination (R²) on the test fold",
            "DataType": "float",
            "Units": "dimensionless",
        },
        "PearsonR": {
            "Description": "Pearson correlation coefficient on the test fold",
            "DataType": "float",
            "Units": "dimensionless",
        },
        "SpearmanRho": {
            "Description": "Spearman rank correlation coefficient on the test fold",
            "DataType": "float",
            "Units": "dimensionless",
        },
        "RMSE": {
            "Description": "Root mean squared error on the test fold",
            "DataType": "float",
            "Units": "luminance units (raw or z-score depending on Model)",
        },
    }
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(data_dictionary, file_handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute all three baseline analyses for sub-27 luminance prediction.

    Steps:
        1. Log random seed for reproducibility.
        2. Load preprocessed EEG + events for each run and extract raw-TDE
           epochs (same feature representation as script 13).
        3. Run Shuffle Baseline (N_SHUFFLE_ITERATIONS) → save CSV + JSON.
        4. Run Mean Baseline → save CSV + JSON.
        5. Run Z-score vs Raw evaluation → save CSV + JSON.

    Outputs:
        - ``results/modeling/luminance/baselines/sub-{SUBJECT}_shuffle_baseline.csv``
        - ``results/modeling/luminance/baselines/sub-{SUBJECT}_shuffle_baseline.json``
        - ``results/modeling/luminance/baselines/sub-{SUBJECT}_mean_baseline.csv``
        - ``results/modeling/luminance/baselines/sub-{SUBJECT}_mean_baseline.json``
        - ``results/modeling/luminance/zscore_evaluation/sub-{SUBJECT}_zscore_vs_raw.csv``
        - ``results/modeling/luminance/zscore_evaluation/sub-{SUBJECT}_zscore_vs_raw.json``
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"17 — Baseline Models & Z-score Evaluation (sub-{SUBJECT})")
    print("=" * 60)

    baselines_output_dir = RESULTS_PATH / "baselines"
    zscore_output_dir = RESULTS_PATH / "zscore_evaluation"
    baselines_output_dir.mkdir(parents=True, exist_ok=True)
    zscore_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    if not roi_channels:
        print("ERROR: No ROI channels found. Exiting.")
        return

    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")
    print(f"TDE window: ±{TDE_WINDOW_HALF} | PCA components: {TDE_PCA_COMPONENTS}")
    
    # ------------------------------------------------------------------
    # 2. Load AutoReject QA TSV to drop bad epochs
    # ------------------------------------------------------------------
    qa_tsv_path = PROJECT_ROOT / "results" / "qa" / "eeg" / f"sub-{SUBJECT}_eeg_qa_autoreject.tsv"
    qa_df = None
    if qa_tsv_path.exists():
        print(f"Loading QA AutoReject parameters from: {qa_tsv_path}")
        qa_df = pd.read_csv(qa_tsv_path, sep="\t")
    else:
        print(f"WARNING: No QA TSV found at {qa_tsv_path}. Running WITHOUT epoch rejection.")

    # ------------------------------------------------------------------
    # 3. Collect raw-TDE epochs across all runs
    # ------------------------------------------------------------------
    all_epoch_entries: list[dict] = []

    for run_config in RUNS_CONFIG:
        run_label = (
            f"run-{run_config['id']} acq-{run_config['acq']} "
            f"task-{run_config['task']}"
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

        eeg_raw = mne.io.read_raw_brainvision(
            str(vhdr_path), preload=True, verbose=False
        )
        events_df = pd.read_csv(events_path, sep="\t")
        recording_roi = select_roi_channels(eeg_raw.ch_names, POSTERIOR_CHANNELS)
        
        # Build bad epochs map for this run
        run_bad_epochs: dict[int, list[int]] = {}
        if qa_df is not None:
            # Drop zero-padding from run id to match QA dataframe optionally
            run_str_match = str(run_config["id"])
            run_acq_match = str(run_config["acq"])
            run_qa = qa_df[
                (qa_df["RunID"].astype(str).str.zfill(3) == run_str_match.zfill(3)) &
                (qa_df["Acq"].astype(str) == run_acq_match)
            ]
            for _, qa_row in run_qa.iterrows():
                v_id = int(qa_row["VideoID"])
                # Safely evaluate string representation of bad epochs array
                bad_idx_list = ast.literal_eval(qa_row["BadEpochsIdx"])
                run_bad_epochs[v_id] = bad_idx_list

        run_epochs = extract_tde_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            roi_channels=recording_roi,
            bad_epochs_map=run_bad_epochs,
        )
        print(f"  Epochs extracted: {len(run_epochs)}")
        all_epoch_entries.extend(run_epochs)

    print(f"\nTotal epochs collected: {len(all_epoch_entries)}")
    if not all_epoch_entries:
        print("ERROR: No epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 4. Shuffle Baseline (Req 2.1–2.5)
    # ------------------------------------------------------------------
    print(f"\n--- Shuffle Baseline ({N_SHUFFLE_ITERATIONS} iterations) ---")
    shuffle_df = run_shuffle_baseline(
        epoch_entries=all_epoch_entries,
        n_iterations=N_SHUFFLE_ITERATIONS,
        random_seed=RANDOM_SEED,
    )

    shuffle_csv_path = (
        baselines_output_dir / f"sub-{SUBJECT}_shuffle_baseline.csv"
    )
    shuffle_df.to_csv(shuffle_csv_path, index=False)
    _write_baselines_json_sidecar(
        baselines_output_dir / f"sub-{SUBJECT}_shuffle_baseline.json"
    )
    print(f"Shuffle baseline saved → {shuffle_csv_path}")

    shuffle_summary = shuffle_df.groupby("TestVideo")[
        ["R2", "PearsonR", "SpearmanRho", "RMSE"]
    ].mean()
    print("\nShuffle baseline mean metrics per video (across iterations):")
    print(shuffle_summary.to_string())
    overall_shuffle = shuffle_df[["R2", "PearsonR", "SpearmanRho", "RMSE"]].mean()
    print(f"\nOverall shuffle mean: {overall_shuffle.to_dict()}")

    # ------------------------------------------------------------------
    # 5. Mean Baseline (Req 3.1–3.4)
    # ------------------------------------------------------------------
    print("\n--- Mean Baseline ---")
    # Z-score targets before LOVO (same normalisation as real models)
    zscore_entries = zscore_per_video(all_epoch_entries)
    mean_baseline_df = run_mean_baseline(zscore_entries)

    mean_csv_path = baselines_output_dir / f"sub-{SUBJECT}_mean_baseline.csv"
    mean_baseline_df.to_csv(mean_csv_path, index=False)
    _write_baselines_json_sidecar(
        baselines_output_dir / f"sub-{SUBJECT}_mean_baseline.json"
    )
    print(f"Mean baseline saved → {mean_csv_path}")
    print(mean_baseline_df[["TestVideo", "R2", "PearsonR", "SpearmanRho", "RMSE"]])
    overall_mean = mean_baseline_df[
        ["R2", "PearsonR", "SpearmanRho", "RMSE"]
    ].mean()
    print(f"\nOverall mean baseline: {overall_mean.to_dict()}")

    # ------------------------------------------------------------------
    # 6. Z-score vs Raw evaluation (Req 7b.1–7b.5)
    # ------------------------------------------------------------------
    print("\n--- Z-score vs Raw Target Evaluation ---")
    zscore_eval_df = run_zscore_vs_raw_evaluation(all_epoch_entries)

    zscore_csv_path = zscore_output_dir / f"sub-{SUBJECT}_zscore_vs_raw.csv"
    zscore_eval_df.to_csv(zscore_csv_path, index=False)
    _write_zscore_eval_json_sidecar(
        zscore_output_dir / f"sub-{SUBJECT}_zscore_vs_raw.json"
    )
    print(f"Z-score evaluation saved → {zscore_csv_path}")

    for model_label in zscore_eval_df["Model"].unique():
        model_rows = zscore_eval_df[zscore_eval_df["Model"] == model_label]
        model_mean = model_rows[["R2", "PearsonR", "SpearmanRho", "RMSE"]].mean()
        print(f"\n  {model_label}: {model_mean.to_dict()}")

    print("\n✓ Script 17 complete.")


if __name__ == "__main__":
    run_pipeline()
