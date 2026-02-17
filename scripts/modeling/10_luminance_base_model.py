"""Base predictive model: raw EEG (posterior ROI) → luminance.

Loads preprocessed EEG, identifies video_luminance segments via merged events,
crops EEG to each segment, generates overlapping epochs (500 ms / 400 ms
overlap), vectorises raw EEG from the posterior ROI as features (X), and uses
epoch-average physical luminance as target (y).

Pipeline: Vectorizer → StandardScaler → PCA(100) → Ridge (GridSearchCV + LeaveOneGroupOut)
Evaluation: Leave-One-Video-Out CV with Pearson r, Spearman ρ, RMSE.

Results are saved to ``results/modeling/luminance/base/``.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4
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
from mne.decoding import Vectorizer
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
    PCA_COMPONENTS,
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA_GRID,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
)

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

    # Prefer merged events (contain real onset times)
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

    # Fall back to preprocessed events
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

    Returns the intersection of *roi_channels* and *available_channels*,
    preserving the order defined in *roi_channels*.  Logs a warning for
    any ROI channel not found.

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
        epoch_entries: List of epoch dicts, each with a ``video_identifier``
            key used for grouping.

    Returns:
        List of ``(train_entries, test_entries, test_video_id)`` tuples,
        one per unique video identifier.
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
# Epoch extraction
# ---------------------------------------------------------------------------


def extract_luminance_epochs_for_run(
    run_config: dict,
    eeg_raw: mne.io.Raw,
    events_df: pd.DataFrame,
    roi_channels: list[str],
    epoch_duration_s: float = EPOCH_DURATION_S,
    epoch_step_s: float = EPOCH_STEP_S,
) -> list[dict]:
    """Extract synchronised EEG–luminance epochs for one run.

    Identifies the ``video_luminance`` event, resolves the corresponding
    video_id via stim_id encoding (stim_id = 100 + video_id), loads the
    luminance CSV, crops EEG to the video segment, and generates overlapping
    epochs with epoch-average luminance targets.

    Args:
        run_config: Run metadata dict (id, acq, task, block).
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.

    Returns:
        List of epoch entry dicts with keys: X (2-D array, channels × samples),
        y, video_id, video_identifier, run_id, acq.

    Requirements: 3.1, 3.2, 3.3, 3.7
    """
    run_id = run_config["id"]
    acq = run_config["acq"]

    # Filter video_luminance events
    luminance_events = events_df[
        events_df["trial_type"] == "video_luminance"
    ].reset_index(drop=True)

    if luminance_events.empty:
        logger.warning("Run %s: no video_luminance events found.", run_id)
        return []

    sfreq = eeg_raw.info["sfreq"]

    epoch_entries: list[dict] = []

    for _, event_row in luminance_events.iterrows():
        stim_id = int(event_row["stim_id"])
        video_id = stim_id - 100  # Encoding: stim_id = 100 + video_id
        onset_s = float(event_row["onset"])
        duration_s = float(event_row["duration"])

        # Check if this is a known experimental luminance video
        csv_filename = LUMINANCE_CSV_MAP.get(video_id)
        if csv_filename is None:
            logger.warning(
                "Run %s: video_id %d not in LUMINANCE_CSV_MAP, skipping.",
                run_id,
                video_id,
            )
            continue

        # Load luminance CSV
        csv_path = STIMULI_PATH / csv_filename
        try:
            luminance_df = load_luminance_csv(csv_path)
        except FileNotFoundError:
            logger.warning(
                "Run %s: luminance CSV not found: %s, skipping segment.",
                run_id,
                csv_path,
            )
            continue

        # Crop EEG to the video segment
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
            continue

        eeg_data = video_eeg.get_data(picks=roi_channels)
        n_samples_segment = eeg_data.shape[1]

        # Generate epoch onsets
        epoch_onsets = create_epoch_onsets(
            n_samples_total=n_samples_segment,
            sfreq=sfreq,
            epoch_duration_s=epoch_duration_s,
            epoch_step_s=epoch_step_s,
        )

        if len(epoch_onsets) == 0:
            logger.warning(
                "Run %s: segment too short for epochs (video_id=%d).",
                run_id,
                video_id,
            )
            continue

        # Interpolate luminance targets
        luminance_targets = interpolate_luminance_to_epochs(
            luminance_df=luminance_df,
            epoch_onsets_s=epoch_onsets,
            epoch_duration_s=epoch_duration_s,
        )

        # Build epoch entries
        n_samples_epoch = int(epoch_duration_s * sfreq)
        video_identifier = f"{video_id}_{acq}"

        for idx, onset in enumerate(epoch_onsets):
            sample_start = int(round(onset * sfreq))
            sample_end = sample_start + n_samples_epoch
            if sample_end > n_samples_segment:
                break

            eeg_window = eeg_data[:, sample_start:sample_end]
            epoch_entries.append(
                {
                    "X": eeg_window,
                    "y": float(luminance_targets[idx]),
                    "video_id": video_id,
                    "video_identifier": video_identifier,
                    "run_id": run_id,
                    "acq": acq,
                }
            )

    return epoch_entries


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
            "Description": "Model identifier (always 'base' for this pipeline)",
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
# Plotting
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
        ax.bar(results_df["TestVideo"], results_df[metric], color="steelblue")
        mean_val = results_df[metric].mean()
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.4f}")
        ax.set_xlabel("Test Video")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Fold")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Base Model (Raw EEG → Luminance) — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(output_dir / f"sub-{SUBJECT}_base_model_cv_results.png", dpi=150)
    plt.close(fig)


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
        f"Predictions — Base Model — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_base_model_predictions.png", dpi=150
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(epoch_duration_override: float | None = None) -> None:
    """Execute the base luminance prediction pipeline.

    Args:
        epoch_duration_override: If provided, overrides the configured
            EPOCH_DURATION_S.  EPOCH_STEP_S is kept at 100 ms
            (overlap = duration − 100 ms).

    Steps:
        1. Set random seed for reproducibility.
        2. Determine ROI channels (posterior / occipital).
        3. For each run, load EEG + events, extract epochs from ROI.
        4. Run Leave-One-Video-Out CV with Vectorizer → Scaler → PCA → Ridge.
        5. Save results CSV, JSON sidecar, and plots.

    Requirements: 3.4, 3.5, 3.6, 6.1, 6.2, 6.3, 6.4
    """
    # Allow epoch duration override from CLI
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
    print(f"10 — Base Luminance Model (sub-{SUBJECT}) — epoch={epoch_ms_tag}")
    print("=" * 60)

    output_dir = RESULTS_PATH / "base" / epoch_ms_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")

    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Collect epochs across all runs
    # ------------------------------------------------------------------
    all_epochs: list[dict] = []

    for run_config in RUNS_CONFIG:
        run_label = (
            f"run-{run_config['id']} acq-{run_config['acq']} "
            f"task-{run_config['task']} ({run_config['block']})"
        )
        print(f"\nProcessing {run_label}")

        # Resolve paths
        vhdr_path = _resolve_eeg_path(run_config)
        if vhdr_path is None:
            print("  WARNING: EEG file not found, skipping.")
            continue

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            print("  WARNING: Events TSV not found, skipping.")
            continue

        # Load data
        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        # Resolve actual ROI channels for this recording
        recording_roi = select_roi_channels(eeg_raw.ch_names, POSTERIOR_CHANNELS)

        # Extract epochs
        run_epochs = extract_luminance_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            roi_channels=recording_roi,
            epoch_duration_s=active_epoch_duration,
            epoch_step_s=active_epoch_step,
        )
        print(f"  Epochs extracted: {len(run_epochs)}")
        all_epochs.extend(run_epochs)

    print(f"\nTotal epochs collected: {len(all_epochs)}")
    if not all_epochs:
        print("No epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Z-score normalization per video
    # ------------------------------------------------------------------
    all_epochs = zscore_per_video(all_epochs)
    print("Z-score normalization applied per video.")

    # ------------------------------------------------------------------
    # 4. Leave-One-Video-Out CV
    # ------------------------------------------------------------------
    folds = leave_one_video_out_split(all_epochs)
    print(f"Number of CV folds: {len(folds)}")

    results_list: list[dict] = []
    fold_predictions: list[dict] = []

    for train_entries, test_entries, test_video in folds:
        X_train = np.array([e["X"] for e in train_entries])
        y_train = np.array([e["y"] for e in train_entries])
        X_test = np.array([e["X"] for e in test_entries])
        y_test = np.array([e["y"] for e in test_entries])

        print(
            f"\n  Fold: test={test_video} | "
            f"train={X_train.shape[0]} | test={X_test.shape[0]}"
        )

        # Pipeline: Vectorizer → StandardScaler → PCA → Ridge
        # Alpha selected via GridSearchCV (Spearman ρ scoring) with
        # LeaveOneGroupOut on training videos to prevent data leakage.
        groups_train = np.array([e["video_identifier"] for e in train_entries])
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED),
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
                "Model": "base",
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
            """Evaluate mean Pearson r across LOVO_CV folds for permutation test."""
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
                    Vectorizer(),
                    StandardScaler(),
                    PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED),
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
            epoch_entries=all_epochs,
            build_and_evaluate_fn=_build_and_evaluate,
            n_permutations=N_PERMUTATIONS,
            random_seed=RANDOM_SEED,
        )
        print(
            f"  Observed r: {perm_results['observed_r']:.4f} | "
            f"p-value: {perm_results['p_value']:.4f}"
        )

        np.savez(
            output_dir / f"sub-{SUBJECT}_base_model_permutation.npz",
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
        )
        plot_permutation_histogram(
            null_distribution=perm_results["null_distribution"],
            observed_r=perm_results["observed_r"],
            p_value=perm_results["p_value"],
            output_path=output_dir
            / f"sub-{SUBJECT}_base_model_permutation_hist.png",
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

    # Save CSV
    csv_path = output_dir / f"sub-{SUBJECT}_base_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    json_sidecar_path = csv_path.with_suffix(".json")
    _write_results_json_sidecar(json_sidecar_path)
    print(f"JSON data dictionary saved: {json_sidecar_path}")

    # Save plots
    plot_cv_results(results_df, output_dir)
    plot_predictions_per_fold(fold_predictions, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Base model pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Base luminance model")
    parser.add_argument(
        "--epoch-duration",
        type=float,
        default=None,
        help="Epoch duration in seconds (default: use config value)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_pipeline(epoch_duration_override=args.epoch_duration)
