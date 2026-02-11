"""TDE model: Time Delay Embedding over spectral features → luminance.

Same pipeline flow as 11_luminance_spectral_model.py but applies Time Delay
Embedding (±10 time-points) to the sequential spectral feature matrix before
PCA dimensionality reduction.

Pipeline: StandardScaler → PCA(100) → Ridge(α=1.0)
Evaluation: Leave-One-Video-Out CV with Pearson r, Spearman ρ, RMSE.

Results are saved to ``results/modeling/luminance/tde/``.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
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
    POSTERIOR_CHANNELS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RIDGE_ALPHA,
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

from campeones_analysis.luminance.features import (
    apply_time_delay_embedding,
    extract_bandpower,
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
# Path resolution helpers (shared with 10/11 scripts)
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
# Epoch extraction with spectral features (per-video sequences for TDE)
# ---------------------------------------------------------------------------


def extract_spectral_epochs_for_run(
    run_config: dict,
    eeg_raw: mne.io.Raw,
    events_df: pd.DataFrame,
    roi_channels: list[str],
) -> list[dict]:
    """Extract spectral epochs grouped by video for later TDE application.

    Same logic as script 11 but returns epochs in temporal order per video
    so that TDE can be applied to each video's feature sequence.

    Args:
        run_config: Run metadata dict (id, acq, task, block).
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        events_df: Events DataFrame for this run.
        roi_channels: List of ROI channel names present in the EEG.

    Returns:
        List of epoch entry dicts with keys: X (1-D band-power vector),
        y, video_id, video_identifier, run_id, acq.

    Requirements: 3.1, 3.2, 3.3, 3.7, 4.1, 4.2, 4.3
    """
    run_id = run_config["id"]
    acq = run_config["acq"]
    sfreq = eeg_raw.info["sfreq"]

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
                "Run %s: luminance CSV not found: %s, skipping segment.",
                run_id,
                csv_path,
            )
            continue

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

        luminance_targets = interpolate_luminance_to_epochs(
            luminance_df=luminance_df,
            epoch_onsets_s=epoch_onsets,
            epoch_duration_s=EPOCH_DURATION_S,
        )

        n_samples_epoch = int(EPOCH_DURATION_S * sfreq)
        video_identifier = f"{video_id}_{acq}"

        for idx, onset in enumerate(epoch_onsets):
            sample_start = int(round(onset * sfreq))
            sample_end = sample_start + n_samples_epoch
            if sample_end > n_samples_segment:
                break

            eeg_window = eeg_data[:, sample_start:sample_end]
            bandpower_vector = extract_bandpower(
                eeg_window, sfreq, SPECTRAL_BANDS
            )

            epoch_entries.append(
                {
                    "X": bandpower_vector,
                    "y": float(luminance_targets[idx]),
                    "video_id": video_id,
                    "video_identifier": video_identifier,
                    "run_id": run_id,
                    "acq": acq,
                }
            )

    return epoch_entries


# ---------------------------------------------------------------------------
# TDE application per video group
# ---------------------------------------------------------------------------


def apply_tde_per_video(
    epoch_entries: list[dict],
    window_half: int,
) -> list[dict]:
    """Apply Time Delay Embedding to spectral features grouped by video.

    TDE must be applied within each video's temporal sequence (not across
    videos), because the temporal context is only meaningful within a
    continuous recording segment.  Epochs at the borders of each video
    (where the full ±window_half context is unavailable) are discarded.

    Args:
        epoch_entries: List of epoch dicts with ``X`` (1-D band-power),
            ``video_identifier``, and other metadata keys.
        window_half: Half-width of the TDE window.

    Returns:
        New list of epoch dicts where ``X`` contains the TDE-expanded
        feature vector.  The number of epochs per video is reduced by
        ``2 * window_half``.

    Requirements: 5.1, 5.2
    """
    video_groups: dict[str, list[dict]] = {}
    for entry in epoch_entries:
        vid = entry["video_identifier"]
        video_groups.setdefault(vid, []).append(entry)

    tde_entries: list[dict] = []

    for video_id_key in sorted(video_groups.keys()):
        group = video_groups[video_id_key]

        if len(group) < 2 * window_half + 1:
            logger.warning(
                "Video %s has %d epochs, need at least %d for TDE. Skipping.",
                video_id_key,
                len(group),
                2 * window_half + 1,
            )
            continue

        feature_matrix = np.array([entry["X"] for entry in group])
        tde_features = apply_time_delay_embedding(feature_matrix, window_half)

        valid_start = window_half
        valid_end = len(group) - window_half

        for idx, tde_idx in enumerate(range(valid_start, valid_end)):
            original_entry = group[tde_idx]
            tde_entries.append(
                {
                    "X": tde_features[idx],
                    "y": original_entry["y"],
                    "video_id": original_entry["video_id"],
                    "video_identifier": original_entry["video_identifier"],
                    "run_id": original_entry["run_id"],
                    "acq": original_entry["acq"],
                }
            )

    return tde_entries


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
        ax.bar(results_df["TestVideo"], results_df[metric], color="forestgreen")
        mean_val = results_df[metric].mean()
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.4f}")
        ax.set_xlabel("Test Video")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Fold")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"TDE Model (Spectral + TDE → Luminance) — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_tde_model_cv_results.png", dpi=150
    )
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
        f"Predictions — TDE Model — sub-{SUBJECT}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_tde_model_predictions.png", dpi=150
    )
    plt.close(fig)


def plot_comparison_with_spectral(
    tde_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot comparison of TDE model vs spectral model metrics.

    Loads the spectral model results CSV (if available) and generates a
    grouped bar chart comparing Pearson r, Spearman ρ, and RMSE.

    Args:
        tde_df: TDE model results DataFrame.
        output_dir: Directory to save the figure.
    """
    spectral_csv = (
        RESULTS_PATH / "spectral" / f"sub-{SUBJECT}_spectral_model_results.csv"
    )
    if not spectral_csv.exists():
        logger.warning(
            "Spectral model results not found at %s, skipping comparison.",
            spectral_csv,
        )
        return

    spectral_df = pd.read_csv(spectral_csv)

    metrics = ["PearsonR", "SpearmanRho", "RMSE"]
    labels = ["Pearson r", "Spearman ρ", "RMSE"]

    spectral_means = [spectral_df[metric].mean() for metric in metrics]
    tde_means = [tde_df[metric].mean() for metric in metrics]

    x_positions = np.arange(len(metrics))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x_positions - bar_width / 2,
        spectral_means,
        bar_width,
        label="Spectral (no TDE)",
        color="darkorange",
    )
    ax.bar(
        x_positions + bar_width / 2,
        tde_means,
        bar_width,
        label="TDE + PCA",
        color="forestgreen",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Spectral vs TDE Model — sub-{SUBJECT}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_spectral_vs_tde_comparison.png", dpi=150
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the TDE luminance prediction pipeline.

    Steps:
        1. Set random seed for reproducibility.
        2. Determine ROI channels.
        3. For each run, load EEG + events, extract spectral epochs.
        4. Apply TDE per video group to expand temporal context.
        5. Run Leave-One-Video-Out CV with Scaler → PCA → Ridge.
        6. Save results CSV, plots, and comparison with spectral model.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4
    """
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    print("=" * 60)
    print(f"12 — TDE Luminance Model (sub-{SUBJECT})")
    print("=" * 60)

    output_dir = RESULTS_PATH / "tde"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine ROI channels
    # ------------------------------------------------------------------
    roi_channels = select_roi_channels(EEG_CHANNELS, POSTERIOR_CHANNELS)
    n_spectral_features = len(roi_channels) * len(SPECTRAL_BANDS)
    window_size = 2 * TDE_WINDOW_HALF + 1
    n_tde_features = n_spectral_features * window_size
    print(f"ROI channels ({len(roi_channels)}): {roi_channels}")
    print(f"Spectral bands: {list(SPECTRAL_BANDS.keys())}")
    print(f"Spectral features per epoch: {n_spectral_features}")
    print(f"TDE window: ±{TDE_WINDOW_HALF} → {window_size} time-points")
    print(f"TDE-expanded features per epoch: {n_tde_features}")

    if not roi_channels:
        print("ERROR: No ROI channels found in EEG. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Collect spectral epochs across all runs
    # ------------------------------------------------------------------
    all_spectral_epochs: list[dict] = []

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

        order_matrix_path = _resolve_order_matrix_path(run_config)
        if order_matrix_path is None:
            print("  WARNING: Order Matrix not found, skipping.")
            continue

        print(f"  EEG: {vhdr_path.name}")
        eeg_raw = mne.io.read_raw_brainvision(
            str(vhdr_path), preload=True, verbose=False
        )

        print(f"  Events: {events_path.name}")
        events_df = pd.read_csv(events_path, sep="\t")

        recording_roi = select_roi_channels(eeg_raw.ch_names, POSTERIOR_CHANNELS)

        run_epochs = extract_spectral_epochs_for_run(
            run_config=run_config,
            eeg_raw=eeg_raw,
            events_df=events_df,
            roi_channels=recording_roi,
        )
        print(f"  Spectral epochs extracted: {len(run_epochs)}")
        all_spectral_epochs.extend(run_epochs)

    print(f"\nTotal spectral epochs collected: {len(all_spectral_epochs)}")
    if not all_spectral_epochs:
        print("No epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Z-score normalization per video (before TDE)
    # ------------------------------------------------------------------
    all_spectral_epochs = zscore_per_video(all_spectral_epochs)
    print("Z-score normalization applied per video (before TDE).")

    # ------------------------------------------------------------------
    # 4. Apply TDE per video group
    # ------------------------------------------------------------------
    print(f"\nApplying TDE (window_half={TDE_WINDOW_HALF})...")
    all_tde_epochs = apply_tde_per_video(all_spectral_epochs, TDE_WINDOW_HALF)
    print(
        f"Epochs after TDE: {len(all_tde_epochs)} "
        f"(dropped {len(all_spectral_epochs) - len(all_tde_epochs)} border epochs)"
    )

    if not all_tde_epochs:
        print("No TDE epochs generated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 5. Leave-One-Video-Out CV
    # ------------------------------------------------------------------
    folds = leave_one_video_out_split(all_tde_epochs)
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
            f"train={X_train.shape[0]} ({X_train.shape[1]} features) | "
            f"test={X_test.shape[0]}"
        )

        pca_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=pca_components, random_state=RANDOM_SEED),
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
                "Model": "tde",
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
    # 6. Permutation test
    # ------------------------------------------------------------------
    def _build_and_evaluate(epoch_entries: list[dict]) -> float:
        """Evaluate mean Pearson r across LOVO_CV folds for permutation test.

        Re-applies TDE internally so that shuffled targets are correctly
        aligned with the TDE-expanded features.
        """
        tde_epochs = apply_tde_per_video(epoch_entries, TDE_WINDOW_HALF)
        if not tde_epochs:
            return 0.0
        perm_folds = leave_one_video_out_split(tde_epochs)
        pearson_values: list[float] = []
        for perm_train, perm_test, _ in perm_folds:
            perm_X_train = np.array([e["X"] for e in perm_train])
            perm_y_train = np.array([e["y"] for e in perm_train])
            perm_X_test = np.array([e["X"] for e in perm_test])
            perm_y_test = np.array([e["y"] for e in perm_test])

            pca_comp = min(
                PCA_COMPONENTS, perm_X_train.shape[0], perm_X_train.shape[1]
            )
            perm_pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=pca_comp, random_state=RANDOM_SEED),
                Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
            )
            perm_pipeline.fit(perm_X_train, perm_y_train)
            perm_y_pred = perm_pipeline.predict(perm_X_test)
            fold_metrics = evaluate_fold(perm_y_test, perm_y_pred)
            pearson_values.append(fold_metrics["PearsonR"])
        return float(np.mean(pearson_values))

    print(f"\nRunning permutation test ({N_PERMUTATIONS} iterations)...")
    perm_results = run_permutation_test(
        epoch_entries=all_spectral_epochs,
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
        output_dir / f"sub-{SUBJECT}_tde_model_permutation.npz",
        null_distribution=perm_results["null_distribution"],
        observed_r=perm_results["observed_r"],
        p_value=perm_results["p_value"],
    )
    plot_permutation_histogram(
        null_distribution=perm_results["null_distribution"],
        observed_r=perm_results["observed_r"],
        p_value=perm_results["p_value"],
        output_path=output_dir / f"sub-{SUBJECT}_tde_model_permutation_hist.png",
    )
    print(f"  Permutation results saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 7. Summary and save
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

    csv_path = output_dir / f"sub-{SUBJECT}_tde_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")

    plot_cv_results(results_df, output_dir)
    plot_predictions_per_fold(fold_predictions, output_dir)
    plot_comparison_with_spectral(results_df, output_dir)
    print(f"Plots saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("TDE model pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
