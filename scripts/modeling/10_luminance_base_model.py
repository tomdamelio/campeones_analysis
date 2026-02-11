"""Base predictive model: raw EEG → luminance.

Loads preprocessed EEG, identifies video_luminance segments via merged events
and Order Matrix, crops EEG to each segment, generates overlapping epochs
(500 ms / 400 ms overlap), vectorises raw EEG as features (X), and uses
interpolated physical luminance as target (y).

Pipeline: Vectorizer → StandardScaler → PCA(100) → Ridge(α=1.0)
Evaluation: Leave-One-Video-Out CV with Pearson r, Spearman ρ, RMSE.

Results are saved to ``results/modeling/luminance/base/``.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA,
    RIDGE_ALPHA_GRID,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    XDF_PATH,
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
# Path resolution helpers (reused from 09_verify_luminance_markers.py)
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
    order_matrix_df: pd.DataFrame,
) -> list[dict]:
    """Extract synchronised EEG–luminance epochs for one run.

    Identifies the ``video_luminance`` event, resolves the corresponding
    video_id via stim_id encoding (stim_id = 100 + video_id), loads the
    luminance CSV, crops EEG to the video segment, and generates overlapping
    epochs with interpolated luminance targets.

    Args:
        run_config: Run metadata dict (id, acq, task, block).
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        order_matrix_df: Order Matrix DataFrame for this block.

    Returns:
        List of epoch entry dicts with keys: X, y, video_id,
        video_identifier, run_id, acq.

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
    eeg_channels_present = [
        ch for ch in EEG_CHANNELS if ch in eeg_raw.ch_names
    ]

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

        eeg_data = video_eeg.get_data(picks=eeg_channels_present)
        n_samples_segment = eeg_data.shape[1]

        # Generate epoch onsets
        epoch_onsets = create_epoch_onsets(
            n_samples_total=n_samples_segment,
            sfreq=sfreq,
            epoch_duration_s=EPOCH_DURATION_S,
            epoch_step_s=EPOCH_STEP_S,
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
            epoch_duration_s=EPOCH_DURATION_S,
        )

        # Build epoch entries
        n_samples_epoch = int(EPOCH_DURATION_S * sfreq)
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


def run_pipeline() -> None:
    """Execute the base luminance prediction pipeline.

    Steps:
        1. Set random seed for reproducibility.
        2. For each run, load EEG + events + Order Matrix, extract epochs.
        3. Run Leave-One-Video-Out CV with Vectorizer → Scaler → PCA → Ridge.
        4. Save results CSV and plots.

    Requirements: 3.4, 3.5, 3.6, 6.1, 6.2, 6.3, 6.4
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"10 — Base Luminance Model (sub-{SUBJECT})")
    print("=" * 60)

    output_dir = RESULTS_PATH / "base"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect epochs across all runs
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
            print(f"  WARNING: EEG file not found, skipping.")
            continue

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            print(f"  WARNING: Events TSV not found, skipping.")
            continue

        order_matrix_path = _resolve_order_matrix_path(run_config)
        if order_matrix_path is None:
            print(f"  WARNING: Order Matrix not found, skipping.")
            continue

        # Load data
        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        print(f"  Order Matrix: {order_matrix_path.name}")
        order_matrix_df = pd.read_excel(order_matrix_path)

        # Extract epochs
        run_epochs = extract_luminance_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            order_matrix_df=order_matrix_df,
        )
        print(f"  Epochs extracted: {len(run_epochs)}")
        all_epochs.extend(run_epochs)

    print(f"\nTotal epochs collected: {len(all_epochs)}")
    if not all_epochs:
        print("No epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Z-score normalization per video
    # ------------------------------------------------------------------
    all_epochs = zscore_per_video(all_epochs)
    print("Z-score normalization applied per video.")

    # ------------------------------------------------------------------
    # 3. Leave-One-Video-Out CV
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

        # Pipeline: Vectorizer → StandardScaler → PCA → Ridge (GridSearchCV)
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED),
            Ridge(random_state=RANDOM_SEED),
        )

        grid_search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
            cv=3,
            scoring="neg_mean_squared_error",
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        best_alpha = grid_search.best_params_["ridge__alpha"]

        metrics = evaluate_fold(y_test, y_pred)
        print(
            f"    Pearson r={metrics['PearsonR']:.4f} | "
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
    # 4. Permutation test
    # ------------------------------------------------------------------
    def _build_and_evaluate(epoch_entries: list[dict]) -> float:
        """Evaluate mean Pearson r across LOVO_CV folds for permutation test."""
        perm_folds = leave_one_video_out_split(epoch_entries)
        pearson_values: list[float] = []
        for perm_train, perm_test, _ in perm_folds:
            perm_X_train = np.array([e["X"] for e in perm_train])
            perm_y_train = np.array([e["y"] for e in perm_train])
            perm_X_test = np.array([e["X"] for e in perm_test])
            perm_y_test = np.array([e["y"] for e in perm_test])

            perm_pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED),
                Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
            )
            perm_pipeline.fit(perm_X_train, perm_y_train)
            perm_y_pred = perm_pipeline.predict(perm_X_test)
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

    # Save permutation results
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
        output_path=output_dir / f"sub-{SUBJECT}_base_model_permutation_hist.png",
    )
    print(f"  Permutation results saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Summary and save
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_list)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(
        f"\nMean Pearson r: {results_df['PearsonR'].mean():.4f}"
        f" | Mean Spearman ρ: {results_df['SpearmanRho'].mean():.4f}"
        f" | Mean RMSE: {results_df['RMSE'].mean():.4f}"
    )

    # Save CSV
    csv_path = output_dir / f"sub-{SUBJECT}_base_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    # Save plots
    plot_cv_results(results_df, output_dir)
    plot_predictions_per_fold(fold_predictions, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Base model pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
