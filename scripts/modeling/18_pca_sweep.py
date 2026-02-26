"""PCA Sweep: variance analysis and Ridge performance curve for luminance prediction.

Applies PCA with 100 components on the concatenated TDE matrix from all video
segments (Req 5.1–5.4), then sweeps n_components in [10, 20, ..., 100] to
evaluate Ridge regression performance via LOVO_CV (Req 5b.1–5b.3).

Pipeline:
    1. Load preprocessed EEG + events for each run (same as script 13).
    2. Build continuous TDE-expanded data (build_data_tde) for all segments.
    3. Fit PCA with 100 components on the full concatenated TDE matrix.
    4. Record individual and cumulative explained variance → plot + CSV.
    5. For each n_components in [10, 20, ..., 100]:
       a. Apply full GLHMM TDE pipeline (build_data_tde + preprocess_data)
          with that n_components.
       b. Epoch PCA time-series, extract covariance features per epoch.
       c. Run LOVO_CV: StandardScaler → Ridge (GridSearchCV, Spearman ρ).
       d. Record mean Pearson r, R², Spearman ρ, RMSE.
    6. Generate performance curve plot → save + CSV.

Output files (all in ``results/modeling/luminance/pca_sweep/``):
    - ``sub-27_pca_variance.tsv`` + JSON sidecar
    - ``sub-27_pca_sweep_results.tsv`` + JSON sidecar
    - ``sub-27_pca_cumulative_variance.png``
    - ``sub-27_pca_sweep_performance.png``

References:
    Vidaurre et al. (2025). A protocol for time-delay embedded hidden
    Markov modelling of brain data. Nature Protocols.
    https://doi.org/10.1038/s41596-025-01300-2

Requirements: 5.1, 5.2, 5.3, 5.4, 5b.1, 5b.2, 5b.3
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
from glhmm import preproc as glhmm_preproc
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.decomposition import PCA
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
    LUMINANCE_CSV_MAP,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA_GRID,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    TARGET_ZSCORE,
    TDE_WINDOW_HALF,
    XDF_PATH,
)
from campeones_analysis.luminance.evaluation import compute_r2_score
from campeones_analysis.luminance.features import compute_epoch_covariance
from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
from campeones_analysis.luminance.tde_glhmm import (
    fit_global_pca,
    apply_global_pca,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PCA_MAX_COMPONENTS: int = 100
PCA_SWEEP_STEPS: list[int] = list(range(10, 110, 10))  # [10, 20, ..., 100]
OUTPUT_SUBDIR: str = "pca_sweep"


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

    Args:
        available_channels: Channel names present in the EEG recording.
        roi_channels: Desired ROI channel names.

    Returns:
        List of channel names present in both lists, preserving roi_channels order.
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
# TDE data collection (raw TDE matrix, before PCA)
# ---------------------------------------------------------------------------


def collect_tde_matrix_for_segment(
    eeg_data_time_major: np.ndarray,
    tde_lags: int,
) -> np.ndarray | None:
    """Build the raw TDE-expanded matrix for a single EEG segment (no PCA).

    Applies only ``glhmm.preproc.build_data_tde()`` without PCA reduction,
    returning the full TDE-expanded matrix for PCA variance analysis.

    Args:
        eeg_data_time_major: 2-D array of shape ``(n_timepoints, n_channels)``
            with continuous EEG signal from ROI channels.
        tde_lags: Number of lags for TDE embedding (symmetric ±tde_lags).

    Returns:
        2-D array of shape ``(n_valid_timepoints, n_channels * (2*tde_lags+1))``
        with TDE-expanded data, or ``None`` if the segment is too short.
    """
    n_timepoints = eeg_data_time_major.shape[0]
    min_samples = 2 * tde_lags + 1
    if n_timepoints < min_samples:
        return None

    lags = list(range(-tde_lags, tde_lags + 1))
    segment_indices = np.array([[0, n_timepoints]])

    tde_result = glhmm_preproc.build_data_tde(
        data=eeg_data_time_major,
        indices=segment_indices,
        lags=lags,
    )
    tde_data: np.ndarray = tde_result[0]
    return tde_data


def collect_all_tde_segments(
    run_configs: list[dict],
    roi_channels: list[str],
    tde_lags: int,
) -> list[dict]:
    """Collect raw TDE-expanded data for all video segments across all runs.

    For each run, loads EEG + events, crops to each video_luminance segment,
    and applies TDE embedding (without PCA). Returns a list of segment dicts
    containing the raw TDE matrix and metadata needed for epoching.

    Args:
        run_configs: List of run configuration dicts from RUNS_CONFIG.
        roi_channels: ROI channel names to use.
        tde_lags: Number of TDE lags (symmetric ±tde_lags).

    Returns:
        List of dicts with keys:
            - ``tde_data``: 2-D array (n_valid_timepoints, n_tde_features)
            - ``video_id``: int
            - ``video_identifier``: str (e.g. "3_a")
            - ``run_id``: str
            - ``acq``: str
            - ``sfreq``: float
            - ``luminance_df``: pd.DataFrame with luminance time-series
    """
    all_segments: list[dict] = []

    for run_config in run_configs:
        run_id = run_config["id"]
        acq = run_config["acq"]

        vhdr_path = _resolve_eeg_path(run_config)
        if vhdr_path is None:
            logger.warning("Run %s: EEG file not found, skipping.", run_id)
            continue

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            logger.warning("Run %s: events TSV not found, skipping.", run_id)
            continue

        try:
            eeg_raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        except Exception as exc:
            logger.warning("Run %s: failed to load EEG: %s", run_id, exc)
            continue

        try:
            events_df = pd.read_csv(events_path, sep="\t")
        except Exception as exc:
            logger.warning("Run %s: failed to load events: %s", run_id, exc)
            continue

        sfreq = eeg_raw.info["sfreq"]
        luminance_events = events_df[
            events_df["trial_type"] == "video_luminance"
        ].reset_index(drop=True)

        if luminance_events.empty:
            logger.warning("Run %s: no video_luminance events found.", run_id)
            continue

        for _, event_row in luminance_events.iterrows():
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
                continue

            csv_path = STIMULI_PATH / csv_filename
            try:
                luminance_df = load_luminance_csv(csv_path)
            except FileNotFoundError:
                logger.warning(
                    "Run %s: luminance CSV not found: %s, skipping.", run_id, csv_path
                )
                continue

            try:
                video_eeg = eeg_raw.copy().crop(
                    tmin=onset_s, tmax=onset_s + duration_s
                )
            except ValueError as exc:
                logger.warning(
                    "Run %s: could not crop EEG [%.2f, %.2f]: %s",
                    run_id,
                    onset_s,
                    onset_s + duration_s,
                    exc,
                )
                continue

            eeg_data_channels_first: np.ndarray = video_eeg.get_data(
                picks=roi_channels
            )
            eeg_data_time_major: np.ndarray = eeg_data_channels_first.T

            tde_data = collect_tde_matrix_for_segment(
                eeg_data_time_major=eeg_data_time_major,
                tde_lags=tde_lags,
            )
            if tde_data is None:
                logger.warning(
                    "Run %s: segment too short for TDE (video_id=%d), skipping.",
                    run_id,
                    video_id,
                )
                continue

            all_segments.append(
                {
                    "tde_data": tde_data,
                    "video_id": video_id,
                    "video_identifier": f"{video_id}_{acq}",
                    "run_id": run_id,
                    "acq": acq,
                    "sfreq": sfreq,
                    "luminance_df": luminance_df,
                }
            )

    return all_segments


# ---------------------------------------------------------------------------
# PCA variance analysis (Req 5.1–5.4)
# ---------------------------------------------------------------------------


def compute_pca_variance_analysis(
    all_segments: list[dict],
    n_components: int = PCA_MAX_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on the full concatenated TDE matrix and return explained variance.

    Concatenates the raw TDE-expanded data from all segments, fits PCA with
    ``n_components`` components, and returns individual and cumulative
    explained variance ratios.

    Args:
        all_segments: List of segment dicts from ``collect_all_tde_segments``,
            each containing a ``tde_data`` key with the raw TDE matrix.
        n_components: Number of PCA components to fit (default 100).

    Returns:
        Tuple of:
            - ``explained_variance_ratio``: 1-D array of shape (n_components,)
              with individual explained variance per component.
            - ``cumulative_variance``: 1-D array of shape (n_components,)
              with cumulative explained variance.

    Requirements: 5.1, 5.2
    """
    tde_matrices = [seg["tde_data"] for seg in all_segments]
    concatenated_tde = np.concatenate(tde_matrices, axis=0)

    # Cap n_components to the maximum feasible value
    max_feasible = min(concatenated_tde.shape[0], concatenated_tde.shape[1])
    actual_components = min(n_components, max_feasible)
    if actual_components < n_components:
        logger.warning(
            "Requested %d PCA components but data allows only %d. "
            "Reducing automatically.",
            n_components,
            actual_components,
        )

    pca_model = PCA(n_components=actual_components, random_state=RANDOM_SEED)
    pca_model.fit(concatenated_tde)

    explained_variance_ratio: np.ndarray = pca_model.explained_variance_ratio_
    cumulative_variance: np.ndarray = np.cumsum(explained_variance_ratio)

    return explained_variance_ratio, cumulative_variance


# ---------------------------------------------------------------------------
# Epoch extraction for a given n_components (Req 5b.1)
# ---------------------------------------------------------------------------


def extract_epochs_for_n_components(
    all_segments: list[dict],
    n_components: int,
    tde_lags: int,
    epoch_duration_s: float,
    epoch_step_s: float,
    bad_epochs_map: dict[tuple[str, int], list[int]] | None = None,
) -> list[dict]:
    """Extract covariance-feature epochs using GLHMM TDE with a given n_components.

    For each segment, applies the full GLHMM TDE pipeline (build_data_tde +
    preprocess_data with PCA) using ``n_components``, then epochs the PCA
    time-series and extracts covariance features per epoch.

    Args:
        all_segments: List of segment dicts from ``collect_all_tde_segments``.
            Each dict must contain ``video_id``, ``video_identifier``,
            ``run_id``, ``acq``, ``sfreq``, and ``luminance_df``.
        n_components: Number of PCA components for this sweep step.
        tde_lags: Number of TDE lags (symmetric ±tde_lags).
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.
        bad_epochs_map: Dictionary mapping `(run_id, video_id)` -> list of bad epoch indices.

    Returns:
        List of epoch entry dicts with keys: ``X`` (1-D covariance feature
        vector), ``y`` (float luminance target), ``video_id`` (int),
        ``video_identifier`` (str), ``run_id`` (str), ``acq`` (str).
    """
    epoch_entries: list[dict] = []

    # --- Fit GLOBAL PCA on concatenated TDE from all segments ---
    tde_matrices = [seg["tde_data"] for seg in all_segments]
    global_pca = fit_global_pca(tde_matrices, n_components)

    for segment in all_segments:
        video_id: int = segment["video_id"]
        video_identifier: str = segment["video_identifier"]
        run_id: str = segment["run_id"]
        acq: str = segment["acq"]
        sfreq: float = segment["sfreq"]
        luminance_df: pd.DataFrame = segment["luminance_df"]
        tde_data: np.ndarray = segment["tde_data"]

        n_timepoints = tde_data.shape[0]

        # Project into the global PCA subspace
        try:
            pca_timeseries = apply_global_pca(tde_data, global_pca)
        except Exception as exc:
            logger.warning(
                "Segment video_id=%d run=%s: PCA projection failed: %s. Skipping.",
                video_id,
                run_id,
                exc,
            )
            continue

        n_valid_timepoints = pca_timeseries.shape[0]

        epoch_onsets_s = create_epoch_onsets(
            n_samples_total=n_valid_timepoints,
            sfreq=sfreq,
            epoch_duration_s=epoch_duration_s,
            epoch_step_s=epoch_step_s,
        )

        if len(epoch_onsets_s) == 0:
            logger.warning(
                "Segment video_id=%d run=%s: no epochs after TDE border removal "
                "(n_components=%d). Skipping.",
                video_id,
                run_id,
                n_components,
            )
            continue

        n_samples_per_epoch = int(epoch_duration_s * sfreq)
        epoch_features_list: list[np.ndarray] = []
        valid_onsets: list[float] = []

        for onset_epoch_s in epoch_onsets_s:
            sample_start = int(round(onset_epoch_s * sfreq))
            sample_end = sample_start + n_samples_per_epoch
            if sample_end > n_valid_timepoints:
                break
            pca_epoch = pca_timeseries[sample_start:sample_end, :]
            covariance_features = compute_epoch_covariance(pca_epoch)
            epoch_features_list.append(covariance_features)
            valid_onsets.append(onset_epoch_s)

        if not epoch_features_list:
            logger.warning(
                "Segment video_id=%d run=%s: no valid epochs (n_components=%d).",
                video_id,
                run_id,
                n_components,
            )
            continue

        epoch_features = np.stack(epoch_features_list, axis=0)
        epoch_onsets_trimmed = np.array(valid_onsets)

        # Offset epoch onsets by TDE border to align with luminance CSV
        border_offset_s = tde_lags / sfreq
        luminance_epoch_onsets_s = epoch_onsets_trimmed + border_offset_s

        luminance_targets = interpolate_luminance_to_epochs(
            luminance_df=luminance_df,
            epoch_onsets_s=luminance_epoch_onsets_s,
            epoch_duration_s=epoch_duration_s,
        )

        if TARGET_ZSCORE:
            luminance_targets = zscore(luminance_targets)

        bad_epochs = []
        if bad_epochs_map is not None:
            bad_epochs = bad_epochs_map.get((run_id, video_id), [])

        total_epochs = epoch_features.shape[0]
        dropped_count = 0

        for idx in range(total_epochs):
            if idx in bad_epochs:
                dropped_count += 1
                continue

            epoch_entries.append(
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
            logger.info("Run %s (video %d, n_comp=%d): Dropped %d/%d epochs per QA", run_id, video_id, n_components, dropped_count, total_epochs)

    return epoch_entries


# ---------------------------------------------------------------------------
# LOVO_CV evaluation
# ---------------------------------------------------------------------------


def run_lovo_cv(
    epoch_entries: list[dict],
    ridge_alpha_grid: list[float],
    random_seed: int,
) -> list[dict]:
    """Run Leave-One-Video-Out CV with StandardScaler → Ridge (GridSearchCV).

    For each fold, holds out all epochs from one video as the test set,
    trains on the remaining videos, and evaluates with R², Pearson r,
    Spearman ρ, and RMSE.

    Args:
        epoch_entries: List of epoch dicts with keys ``X``, ``y``,
            ``video_identifier``.
        ridge_alpha_grid: List of Ridge alpha values for GridSearchCV.
        random_seed: Random seed for reproducibility.

    Returns:
        List of fold result dicts with keys: ``TestVideo``, ``TrainSize``,
        ``TestSize``, ``R2``, ``PearsonR``, ``SpearmanRho``, ``RMSE``,
        ``BestAlpha``.
    """
    unique_videos = sorted(set(e["video_identifier"] for e in epoch_entries))
    fold_results: list[dict] = []

    for test_video in unique_videos:
        train_entries = [e for e in epoch_entries if e["video_identifier"] != test_video]
        test_entries = [e for e in epoch_entries if e["video_identifier"] == test_video]

        if not train_entries or not test_entries:
            logger.warning("LOVO fold %s: empty train or test set, skipping.", test_video)
            continue

        x_train = np.stack([e["X"] for e in train_entries])
        y_train = np.array([e["y"] for e in train_entries])
        x_test = np.stack([e["X"] for e in test_entries])
        y_test = np.array([e["y"] for e in test_entries])

        # Groups for inner LeaveOneGroupOut (by video_identifier within train)
        train_groups = np.array([e["video_identifier"] for e in train_entries])
        unique_train_videos = np.unique(train_groups)

        # Need at least 2 groups for inner LOGO CV
        if len(unique_train_videos) < 2:
            logger.warning(
                "LOVO fold %s: only %d training video(s), using simple Ridge.",
                test_video,
                len(unique_train_videos),
            )
            best_alpha = ridge_alpha_grid[len(ridge_alpha_grid) // 2]
            pipeline = make_pipeline(
                StandardScaler(), Ridge(alpha=best_alpha, random_state=random_seed)
            )
            pipeline.fit(x_train, y_train)
        else:
            inner_logo = LeaveOneGroupOut()
            ridge_cv = GridSearchCV(
                Ridge(random_state=random_seed),
                param_grid={"alpha": ridge_alpha_grid},
                scoring=spearman_scoring,
                cv=inner_logo,
                refit=True,
            )
            pipeline = make_pipeline(StandardScaler(), ridge_cv)
            pipeline.fit(x_train, y_train, gridsearchcv__groups=train_groups)
            best_alpha = pipeline.named_steps["gridsearchcv"].best_params_["alpha"]

        y_pred = pipeline.predict(x_test)

        has_variance = np.std(y_pred) > 1e-9
        pearson_r = float(pearsonr(y_test, y_pred)[0]) if has_variance else 0.0
        spearman_rho = (
            float(spearmanr(y_test, y_pred).correlation) if has_variance else 0.0
        )
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = compute_r2_score(y_test, y_pred)

        fold_results.append(
            {
                "TestVideo": test_video,
                "TrainSize": len(train_entries),
                "TestSize": len(test_entries),
                "R2": r2,
                "PearsonR": pearson_r,
                "SpearmanRho": spearman_rho,
                "RMSE": rmse,
                "BestAlpha": best_alpha,
            }
        )

    return fold_results


# ---------------------------------------------------------------------------
# Plotting functions (Req 5.3, 5b.2)
# ---------------------------------------------------------------------------


def plot_cumulative_variance(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray,
    output_path: Path,
    subject: str,
) -> None:
    """Generate and save the cumulative PCA explained variance plot.

    Plots individual explained variance as a bar chart and cumulative
    explained variance as a line, with a reference line at 90%.

    Args:
        explained_variance_ratio: 1-D array of individual explained variance
            per component (values in [0, 1]).
        cumulative_variance: 1-D array of cumulative explained variance
            (monotonically non-decreasing, values in [0, 1]).
        output_path: Full path to save the PNG figure.
        subject: Subject identifier for the plot title.

    Requirements: 5.3
    """
    n_components = len(explained_variance_ratio)
    component_indices = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        component_indices,
        explained_variance_ratio * 100,
        color="steelblue",
        alpha=0.6,
        label="Individual variance (%)",
    )
    ax1.set_xlabel("PCA Component")
    ax1.set_ylabel("Individual Explained Variance (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        component_indices,
        cumulative_variance * 100,
        color="darkorange",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Cumulative variance (%)",
    )
    ax2.axhline(90, color="red", linestyle="--", linewidth=1, label="90% threshold")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle(
        f"PCA Cumulative Explained Variance (TDE matrix) — sub-{subject}",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close("all")


def plot_sweep_performance(
    sweep_results_df: pd.DataFrame,
    output_path: Path,
    subject: str,
) -> None:
    """Generate and save the Ridge performance curve vs n_components.

    Plots mean Pearson r and mean R² as a function of the number of PCA
    components used in the GLHMM TDE pipeline.

    Args:
        sweep_results_df: DataFrame with columns ``NComponents``,
            ``MeanPearsonR``, ``MeanR2``.
        output_path: Full path to save the PNG figure.
        subject: Subject identifier for the plot title.

    Requirements: 5b.2
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_pearson = "steelblue"
    color_r2 = "darkorange"

    ax1.plot(
        sweep_results_df["NComponents"],
        sweep_results_df["MeanPearsonR"],
        color=color_pearson,
        linewidth=2,
        marker="o",
        label="Mean Pearson r",
    )
    ax1.set_xlabel("Number of PCA Components")
    ax1.set_ylabel("Mean Pearson r", color=color_pearson)
    ax1.tick_params(axis="y", labelcolor=color_pearson)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(
        sweep_results_df["NComponents"],
        sweep_results_df["MeanR2"],
        color=color_r2,
        linewidth=2,
        marker="s",
        label="Mean R²",
    )
    ax2.set_ylabel("Mean R²", color=color_r2)
    ax2.tick_params(axis="y", labelcolor=color_r2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_xticks(sweep_results_df["NComponents"].tolist())
    fig.suptitle(
        f"Ridge Performance vs PCA Components (LOVO-CV) — sub-{subject}",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close("all")


# ---------------------------------------------------------------------------
# JSON sidecar writers (BIDS compliance)
# ---------------------------------------------------------------------------


def _write_variance_json_sidecar(json_path: Path) -> None:
    """Write BIDS-compliant JSON data dictionary for the PCA variance TSV.

    Args:
        json_path: Destination path for the JSON sidecar file.
    """
    data_dictionary: dict[str, dict[str, str]] = {
        "Component": {
            "Description": "PCA component index (1-indexed)",
            "DataType": "integer",
        },
        "ExplainedVariance": {
            "Description": (
                "Individual explained variance ratio for this PCA component "
                "(proportion of total variance, in [0, 1])"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "CumulativeVariance": {
            "Description": (
                "Cumulative explained variance ratio up to and including "
                "this component (monotonically non-decreasing, in [0, 1])"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data_dictionary, fh, indent=2, ensure_ascii=False)


def _write_sweep_json_sidecar(json_path: Path) -> None:
    """Write BIDS-compliant JSON data dictionary for the PCA sweep results TSV.

    Args:
        json_path: Destination path for the JSON sidecar file.
    """
    data_dictionary: dict[str, dict[str, str]] = {
        "NComponents": {
            "Description": (
                "Number of PCA components used in the GLHMM TDE pipeline "
                "for this sweep step"
            ),
            "DataType": "integer",
        },
        "MeanPearsonR": {
            "Description": (
                "Mean Pearson correlation coefficient across LOVO-CV folds "
                "between predicted and actual luminance"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "MeanR2": {
            "Description": (
                "Mean coefficient of determination (R²) across LOVO-CV folds. "
                "Quantifies the proportion of luminance variability explained "
                "by the EEG model."
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "MeanSpearmanRho": {
            "Description": (
                "Mean Spearman rank correlation coefficient across LOVO-CV folds"
            ),
            "DataType": "float",
            "Units": "dimensionless",
        },
        "MeanRMSE": {
            "Description": (
                "Mean root mean squared error across LOVO-CV folds between "
                "predicted and actual luminance"
            ),
            "DataType": "float",
            "Units": "luminance units (0-255)",
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data_dictionary, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pca_sweep_pipeline() -> None:
    """Execute the PCA sweep pipeline for luminance prediction.

    Orchestrates the full PCA variance analysis (Req 5.1–5.4) and Ridge
    performance sweep (Req 5b.1–5b.3):

    1. Set random seed and log it.
    2. Collect raw TDE-expanded data for all video segments across all runs.
    3. Fit PCA with 100 components on the concatenated TDE matrix.
    4. Save variance TSV + JSON sidecar and cumulative variance plot.
    5. For each n_components in [10, 20, ..., 100]:
       a. Extract covariance-feature epochs using GLHMM TDE pipeline.
       b. Run LOVO_CV with StandardScaler → Ridge (GridSearchCV, Spearman ρ).
       c. Record mean metrics across folds.
    6. Save sweep results TSV + JSON sidecar and performance curve plot.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5b.1, 5b.2, 5b.3
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"18 — PCA Sweep (sub-{SUBJECT})")
    print(f"     PCA_MAX_COMPONENTS={PCA_MAX_COMPONENTS}")
    print(f"     PCA_SWEEP_STEPS={PCA_SWEEP_STEPS}")
    print("=" * 60)

    output_dir = RESULTS_PATH / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    from config_luminance import POSTERIOR_CHANNELS

    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")
    print(f"TDE lags: ±{TDE_WINDOW_HALF}")

    # ------------------------------------------------------------------
    # 2. Collect raw TDE-expanded segments (no PCA yet)
    # ------------------------------------------------------------------
    print("\nCollecting TDE-expanded segments across all runs...")
    all_segments = collect_all_tde_segments(
        run_configs=RUNS_CONFIG,
        roi_channels=roi_channels,
        tde_lags=TDE_WINDOW_HALF,
    )

    if not all_segments:
        print("ERROR: No valid segments collected. Exiting.")
        return

    print(f"Collected {len(all_segments)} segments.")

    qa_tsv_path = PROJECT_ROOT / "results" / "qa" / "eeg" / f"sub-{SUBJECT}_eeg_qa_autoreject.tsv"
    qa_df = None
    if qa_tsv_path.exists():
        print(f"Loading QA AutoReject parameters from: {qa_tsv_path}")
        qa_df = pd.read_csv(qa_tsv_path, sep="\t")
    else:
        print(f"WARNING: No QA TSV found at {qa_tsv_path}. Running WITHOUT epoch rejection.")

    run_bad_epochs_map: dict[tuple[str, int], list[int]] = {}
    if qa_df is not None:
        for _, qa_row in qa_df.iterrows():
            vid = int(qa_row["VideoID"])
            runid_str = str(qa_row["RunID"]).zfill(3)
            bad_idx_list = ast.literal_eval(qa_row["BadEpochsIdx"])
            run_bad_epochs_map[(runid_str, vid)] = bad_idx_list

    # ------------------------------------------------------------------
    # 3. PCA variance analysis on full concatenated TDE matrix (Req 5.1, 5.2)
    # ------------------------------------------------------------------
    print(f"\nFitting PCA with {PCA_MAX_COMPONENTS} components on concatenated TDE matrix...")
    explained_variance_ratio, cumulative_variance = compute_pca_variance_analysis(
        all_segments=all_segments,
        n_components=PCA_MAX_COMPONENTS,
    )

    n_actual_components = len(explained_variance_ratio)
    print(
        f"PCA fitted: {n_actual_components} components, "
        f"total variance explained: {cumulative_variance[-1] * 100:.1f}%"
    )

    # ------------------------------------------------------------------
    # 4. Save variance TSV + JSON sidecar + plot (Req 5.3, 5.4)
    # ------------------------------------------------------------------
    variance_df = pd.DataFrame(
        {
            "Component": np.arange(1, n_actual_components + 1),
            "ExplainedVariance": explained_variance_ratio,
            "CumulativeVariance": cumulative_variance,
        }
    )
    variance_tsv_path = output_dir / f"sub-{SUBJECT}_pca_variance.tsv"
    variance_df.to_csv(variance_tsv_path, sep="\t", index=False)
    _write_variance_json_sidecar(variance_tsv_path.with_suffix(".json"))
    print(f"Saved: {variance_tsv_path}")

    variance_plot_path = output_dir / f"sub-{SUBJECT}_pca_cumulative_variance.png"
    plot_cumulative_variance(
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        output_path=variance_plot_path,
        subject=SUBJECT,
    )
    print(f"Saved: {variance_plot_path}")

    # ------------------------------------------------------------------
    # 5. Ridge performance sweep (Req 5b.1)
    # ------------------------------------------------------------------
    sweep_records: list[dict] = []

    for n_components in PCA_SWEEP_STEPS:
        print(f"\n--- n_components={n_components} ---")

        epoch_entries = extract_epochs_for_n_components(
            all_segments=all_segments,
            n_components=n_components,
            tde_lags=TDE_WINDOW_HALF,
            epoch_duration_s=EPOCH_DURATION_S,
            epoch_step_s=EPOCH_STEP_S,
            bad_epochs_map=run_bad_epochs_map,
        )

        if not epoch_entries:
            logger.warning(
                "n_components=%d: no epochs extracted, skipping.", n_components
            )
            continue

        print(f"  Epochs: {len(epoch_entries)}, feature dim: {epoch_entries[0]['X'].shape[0]}")

        fold_results = run_lovo_cv(
            epoch_entries=epoch_entries,
            ridge_alpha_grid=RIDGE_ALPHA_GRID,
            random_seed=RANDOM_SEED,
        )

        if not fold_results:
            logger.warning(
                "n_components=%d: no fold results, skipping.", n_components
            )
            continue

        folds_df = pd.DataFrame(fold_results)
        mean_pearson_r = float(folds_df["PearsonR"].mean())
        mean_r2 = float(folds_df["R2"].mean())
        mean_spearman_rho = float(folds_df["SpearmanRho"].mean())
        mean_rmse = float(folds_df["RMSE"].mean())

        print(
            f"  Mean Pearson r={mean_pearson_r:.4f}, "
            f"R²={mean_r2:.4f}, "
            f"Spearman ρ={mean_spearman_rho:.4f}, "
            f"RMSE={mean_rmse:.4f}"
        )

        sweep_records.append(
            {
                "NComponents": n_components,
                "MeanPearsonR": mean_pearson_r,
                "MeanR2": mean_r2,
                "MeanSpearmanRho": mean_spearman_rho,
                "MeanRMSE": mean_rmse,
            }
        )

    # ------------------------------------------------------------------
    # 6. Save sweep results TSV + JSON sidecar + plot (Req 5b.3)
    # ------------------------------------------------------------------
    if not sweep_records:
        print("ERROR: No sweep results generated. Exiting.")
        return

    sweep_df = pd.DataFrame(sweep_records)
    sweep_tsv_path = output_dir / f"sub-{SUBJECT}_pca_sweep_results.tsv"
    sweep_df.to_csv(sweep_tsv_path, sep="\t", index=False)
    _write_sweep_json_sidecar(sweep_tsv_path.with_suffix(".json"))
    print(f"\nSaved: {sweep_tsv_path}")

    sweep_plot_path = output_dir / f"sub-{SUBJECT}_pca_sweep_performance.png"
    plot_sweep_performance(
        sweep_results_df=sweep_df,
        output_path=sweep_plot_path,
        subject=SUBJECT,
    )
    print(f"Saved: {sweep_plot_path}")

    print("\n" + "=" * 60)
    print("PCA sweep complete.")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pca_sweep_pipeline()
