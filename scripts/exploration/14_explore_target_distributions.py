"""Distribution Explorer: target variable histograms for luminance pipeline.

Exploratory script that loads joystick and luminance data across all runs,
groups them by dimension (valence, arousal, luminance), and generates
distribution histograms for both raw and z-score-normalized-per-video values.

Target variables explored:
    - Real luminance (physical stimulus from CSV)
    - Reported valence (joystick_x, polarity-corrected)
    - Reported arousal (joystick_x, polarity-corrected)
    - Reported luminance (joystick_x, polarity-corrected)

Results saved to ``results/modeling/luminance/exploration/distributions/``.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
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

# Ensure scripts/modeling is on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)
from config_luminance import (
    DERIVATIVES_PATH,
    EPOCH_DURATION_S,
    EPOCH_STEP_S,
    EXPERIMENTAL_VIDEOS,
    LUMINANCE_CSV_MAP,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_PATH,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
    XDF_PATH,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
EXPLORATION_OUTPUT_DIR: Path = (
    RESULTS_PATH / "exploration" / "distributions"
)


# ---------------------------------------------------------------------------
# Path resolution helpers (shared pattern with scripts 10–13)
# ---------------------------------------------------------------------------


def _resolve_events_path(run_config: dict) -> Path | None:
    """Build the merged-events TSV path for a run, falling back to regular.

    Looks first in ``data/derivatives/merged_events/``, then falls back to
    the preprocessed events in ``data/derivatives/campeones_preproc/``.

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
# Data loading helpers
# ---------------------------------------------------------------------------


def load_order_matrix(order_matrix_path: Path) -> pd.DataFrame:
    """Load an Order Matrix Excel file for a single run/block.

    The Order Matrix maps each video presentation within a block to its
    experimental dimension (valence, arousal, luminance), video_id, and
    joystick polarity (``order_emojis_slider``).

    Args:
        order_matrix_path: Path to the ``.xlsx`` Order Matrix file.

    Returns:
        DataFrame with columns including: participant, session, modality,
        order_presentation, video_id, dimension, order_emojis_slider
        (polarity). Rows with missing ``dimension`` are dropped.
    """
    order_df = pd.read_excel(order_matrix_path)
    order_df = order_df.dropna(subset=["dimension"]).reset_index(drop=True)
    logger.debug(
        "Loaded Order Matrix from %s — %d rows with dimension",
        order_matrix_path.name,
        len(order_df),
    )
    return order_df


def load_merged_events(events_path: Path) -> pd.DataFrame:
    """Load a Merged Events TSV for a single run.

    Reads the tab-separated events file and filters to video presentation
    events (``trial_type`` in {``video``, ``video_luminance``}).

    Args:
        events_path: Path to the merged-events ``.tsv`` file.

    Returns:
        DataFrame filtered to video events, with columns including:
        onset, duration, trial_type, stim_id, condition, stim_file.
        Index is reset.
    """
    events_df = pd.read_csv(events_path, sep="\t")
    video_events = events_df[
        events_df["trial_type"].isin(["video", "video_luminance"])
    ].reset_index(drop=True)
    logger.debug(
        "Loaded events from %s — %d video events",
        events_path.name,
        len(video_events),
    )
    return video_events


def load_all_runs_data() -> list[dict]:
    """Load Merged Events and Order Matrix for every configured run.

    Iterates ``RUNS_CONFIG``, resolves paths for the events TSV and Order
    Matrix, loads both, and returns a list of per-run data bundles. Runs
    whose files are missing are skipped with a warning.

    Returns:
        List of dicts, each containing:
            - ``run_config``: the original run config dict
            - ``events_df``: filtered video events DataFrame
            - ``order_matrix_df``: Order Matrix DataFrame (dimension rows)
            - ``events_path``: resolved Path to the events file
            - ``order_matrix_path``: resolved Path to the Order Matrix
    """
    runs_data: list[dict] = []

    for run_config in RUNS_CONFIG:
        run_id = run_config["id"]
        acq = run_config["acq"]

        events_path = _resolve_events_path(run_config)
        if events_path is None:
            logger.warning(
                "Events TSV not found for run %s acq-%s — skipping",
                run_id,
                acq,
            )
            continue

        order_matrix_path = _resolve_order_matrix_path(run_config)
        if order_matrix_path is None:
            logger.warning(
                "Order Matrix not found for run %s acq-%s — skipping",
                run_id,
                acq,
            )
            continue

        events_df = load_merged_events(events_path)
        order_matrix_df = load_order_matrix(order_matrix_path)

        runs_data.append(
            {
                "run_config": run_config,
                "events_df": events_df,
                "order_matrix_df": order_matrix_df,
                "events_path": events_path,
                "order_matrix_path": order_matrix_path,
            }
        )
        logger.info(
            "Run %s acq-%s: %d video events, %d Order Matrix rows",
            run_id,
            acq,
            len(events_df),
            len(order_matrix_df),
        )

    logger.info(
        "Loaded data for %d / %d configured runs",
        len(runs_data),
        len(RUNS_CONFIG),
    )
    return runs_data


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def apply_polarity_correction(
    signal: np.ndarray, polarity: str
) -> np.ndarray:
    """Apply polarity correction to a joystick signal.

    When the Order Matrix indicates "inverse" polarity, the joystick scale
    is flipped — higher physical deflection maps to the *low* end of the
    dimension.  Negating the signal restores the canonical direction so that
    positive values always mean "more" of the dimension.

    This is a pure function extracted for testability (Property 4).

    Args:
        signal: 1-D array of joystick values.
        polarity: Polarity string from the Order Matrix
            ``order_emojis_slider`` column.  ``"inverse"`` triggers
            negation; any other value (e.g. ``"direct"``) leaves the
            signal unchanged.

    Returns:
        Polarity-corrected copy of *signal* (never mutates the input).

    Requirements: 4.3
    """
    if polarity == "inverse":
        return -signal.copy()
    return signal.copy()


def _compute_epoch_means(
    signal: np.ndarray,
    sfreq: float,
    epoch_duration_s: float,
    epoch_step_s: float,
) -> np.ndarray:
    """Compute epoch-level mean values from a 1-D signal.

    Splits *signal* into overlapping windows of *epoch_duration_s* seconds
    with *epoch_step_s* step and returns the mean of each window.

    Args:
        signal: 1-D array of sample values.
        sfreq: Sampling frequency (Hz).
        epoch_duration_s: Duration of each epoch in seconds.
        epoch_step_s: Step between consecutive epoch onsets in seconds.

    Returns:
        1-D array of per-epoch mean values.  Empty if the signal is too
        short for even one full epoch.
    """
    epoch_onsets = create_epoch_onsets(
        n_samples_total=len(signal),
        sfreq=sfreq,
        epoch_duration_s=epoch_duration_s,
        epoch_step_s=epoch_step_s,
    )
    if len(epoch_onsets) == 0:
        return np.array([], dtype=np.float64)

    n_samples_epoch = int(epoch_duration_s * sfreq)
    means = np.empty(len(epoch_onsets), dtype=np.float64)
    for idx, onset_s in enumerate(epoch_onsets):
        start_sample = int(round(onset_s * sfreq))
        end_sample = start_sample + n_samples_epoch
        means[idx] = np.mean(signal[start_sample:end_sample])
    return means


# ---------------------------------------------------------------------------
# Dimension value extraction helpers (I/O-bound)
# ---------------------------------------------------------------------------


def _extract_luminance_values_for_video(
    video_id: int,
    onset_s: float,
    duration_s: float,
    acq: str,
    run_id: str,
) -> list[dict]:
    """Load physical luminance CSV and compute epoch-level mean values.

    Args:
        video_id: Experimental video identifier (3, 7, 9, or 12).
        onset_s: Video onset time in the EEG recording (seconds).
        duration_s: Video duration (seconds).
        acq: Acquisition label (``"a"`` or ``"b"``).
        run_id: Run identifier string.

    Returns:
        List of dimension-value dicts for the ``"real_luminance"``
        dimension.  Empty if the luminance CSV is missing.

    Requirements: 4.2
    """
    csv_filename = LUMINANCE_CSV_MAP.get(video_id)
    if csv_filename is None:
        logger.warning(
            "Run %s: video_id %d not in LUMINANCE_CSV_MAP — skipping "
            "luminance extraction.",
            run_id,
            video_id,
        )
        return []

    csv_path = STIMULI_PATH / csv_filename
    try:
        luminance_df = load_luminance_csv(csv_path)
    except FileNotFoundError:
        logger.warning(
            "Run %s: luminance CSV not found: %s — skipping.",
            run_id,
            csv_path,
        )
        return []

    # Compute epoch-level mean luminance from the CSV time-series.
    # The CSV timestamps are relative to video start (0-based), so we
    # generate epoch onsets for the video duration directly.
    # Number of virtual samples at 1 kHz (fine enough for onset generation)
    virtual_sfreq = 1000.0
    n_virtual_samples = int(duration_s * virtual_sfreq)
    epoch_onsets = create_epoch_onsets(
        n_samples_total=n_virtual_samples,
        sfreq=virtual_sfreq,
        epoch_duration_s=EPOCH_DURATION_S,
        epoch_step_s=EPOCH_STEP_S,
    )
    if len(epoch_onsets) == 0:
        return []

    epoch_means = interpolate_luminance_to_epochs(
        luminance_df=luminance_df,
        epoch_onsets_s=epoch_onsets,
        epoch_duration_s=EPOCH_DURATION_S,
    )

    video_identifier = f"{video_id}_{acq}"
    return [
        {
            "value": float(epoch_means[idx]),
            "video_id": video_id,
            "video_identifier": video_identifier,
            "dimension": "real_luminance",
            "run_id": run_id,
            "acq": acq,
        }
        for idx in range(len(epoch_means))
    ]


def _extract_joystick_values_for_video(
    eeg_raw: mne.io.Raw,
    onset_s: float,
    duration_s: float,
    video_id: int,
    dimension: str,
    polarity: str,
    acq: str,
    run_id: str,
) -> list[dict]:
    """Extract polarity-corrected joystick epoch means for one video.

    Crops the EEG to the video segment, extracts the ``joystick_x``
    channel, applies polarity correction, and computes epoch-level means.

    Args:
        eeg_raw: Loaded MNE Raw object (preprocessed EEG).
        onset_s: Video onset time in the recording (seconds).
        duration_s: Video duration (seconds).
        video_id: Experimental video identifier.
        dimension: Dimension label from Order Matrix (``"valence"``,
            ``"arousal"``, or ``"luminance"``).
        polarity: Polarity string from ``order_emojis_slider``.
        acq: Acquisition label.
        run_id: Run identifier string.

    Returns:
        List of dimension-value dicts.  Empty if ``joystick_x`` is
        absent or the segment is too short.

    Requirements: 4.3
    """
    if "joystick_x" not in eeg_raw.ch_names:
        logger.warning(
            "Run %s: joystick_x channel not found in EEG — skipping "
            "joystick extraction for video_id %d.",
            run_id,
            video_id,
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

    joystick_signal = video_eeg.get_data(picks=["joystick_x"])[0]
    corrected_signal = apply_polarity_correction(joystick_signal, polarity)

    sfreq = eeg_raw.info["sfreq"]
    epoch_means = _compute_epoch_means(
        corrected_signal,
        sfreq=sfreq,
        epoch_duration_s=EPOCH_DURATION_S,
        epoch_step_s=EPOCH_STEP_S,
    )
    if len(epoch_means) == 0:
        return []

    video_identifier = f"{video_id}_{acq}"
    return [
        {
            "value": float(epoch_means[idx]),
            "video_id": video_id,
            "video_identifier": video_identifier,
            "dimension": dimension,
            "run_id": run_id,
            "acq": acq,
        }
        for idx in range(len(epoch_means))
    ]


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------


def collect_dimension_values(
    runs_data: list[dict],
) -> dict[str, list[dict]]:
    """Collect epoch-level values for every dimension across all runs.

    Iterates the pre-loaded run bundles, matches each video event to its
    Order Matrix row by position, and dispatches to the appropriate
    extraction helper depending on the dimension:

    * **real_luminance** — physical luminance from the stimulus CSV
      (Req 4.2).
    * **valence / arousal / luminance** (joystick) — ``joystick_x``
      channel from the EEG, polarity-corrected per Order Matrix
      (Req 4.3).

    The positional matching follows the same convention used in scripts
    01–07: the *i*-th video event in the merged-events TSV corresponds
    to the *i*-th row (with non-null ``dimension``) in the Order Matrix.

    Args:
        runs_data: List of per-run data bundles as returned by
            :func:`load_all_runs_data`.  Each dict must contain
            ``run_config``, ``events_df``, ``order_matrix_df``.

    Returns:
        Mapping from dimension name to a list of value dicts.  Keys may
        include ``"real_luminance"``, ``"valence"``, ``"arousal"``, and
        ``"luminance"`` (reported).

    Requirements: 4.1, 4.2, 4.3
    """
    dimension_values: dict[str, list[dict]] = {}

    for run_bundle in runs_data:
        run_config = run_bundle["run_config"]
        events_df = run_bundle["events_df"]
        order_matrix_df = run_bundle["order_matrix_df"]
        run_id: str = run_config["id"]
        acq: str = run_config["acq"]

        # --- Load EEG for joystick extraction (lazy, once per run) ---
        eeg_raw: mne.io.Raw | None = None
        eeg_path = _resolve_eeg_path(run_config)
        if eeg_path is not None:
            try:
                eeg_raw = mne.io.read_raw_brainvision(
                    str(eeg_path), preload=True, verbose="WARNING"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Run %s acq-%s: failed to load EEG: %s — joystick "
                    "extraction will be skipped.",
                    run_id,
                    acq,
                    exc,
                )
        else:
            logger.warning(
                "Run %s acq-%s: EEG file not found — joystick "
                "extraction will be skipped.",
                run_id,
                acq,
            )

        # --- Match video events to Order Matrix rows by position ---
        n_pairs = min(len(events_df), len(order_matrix_df))
        for idx in range(n_pairs):
            event_row = events_df.iloc[idx]
            order_row = order_matrix_df.iloc[idx]

            dimension = str(order_row["dimension"]).strip().lower()
            polarity = str(order_row.get("order_emojis_slider", "direct"))

            raw_video_id = order_row.get("video_id")
            if pd.isna(raw_video_id):
                logger.warning(
                    "Run %s acq-%s: video_id is NaN at position %d — "
                    "skipping.",
                    run_id,
                    acq,
                    idx,
                )
                continue
            video_id = int(raw_video_id)
            onset_s = float(event_row["onset"])
            duration_s = float(event_row["duration"])

            # --- Real luminance (from CSV) for luminance videos ---
            if dimension == "luminance":
                lum_entries = _extract_luminance_values_for_video(
                    video_id=video_id,
                    onset_s=onset_s,
                    duration_s=duration_s,
                    acq=acq,
                    run_id=run_id,
                )
                dimension_values.setdefault("real_luminance", []).extend(
                    lum_entries
                )

            # --- Joystick values for all dimensions ---
            if eeg_raw is not None:
                joy_entries = _extract_joystick_values_for_video(
                    eeg_raw=eeg_raw,
                    onset_s=onset_s,
                    duration_s=duration_s,
                    video_id=video_id,
                    dimension=dimension,
                    polarity=polarity,
                    acq=acq,
                    run_id=run_id,
                )
                dimension_values.setdefault(dimension, []).extend(joy_entries)

        logger.info(
            "Run %s acq-%s: processed %d video–Order Matrix pairs.",
            run_id,
            acq,
            n_pairs,
        )

    # Log summary
    for dim_name, entries in dimension_values.items():
        logger.info("Dimension '%s': %d epoch values collected.", dim_name, len(entries))

    return dimension_values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distribution(
    values: np.ndarray,
    dimension_name: str,
    version: str,
    output_dir: Path,
) -> None:
    """Generate a histogram with descriptive statistics annotation.

    Creates a single histogram of *values*, annotates it with mean, std,
    min, max, and N, and saves the figure to *output_dir*.

    Args:
        values: 1-D array of numeric values to plot.
        dimension_name: Human-readable dimension label used in the title
            and filename (e.g. ``"real_luminance"``).
        version: Either ``"raw"`` or ``"normalized"`` — used in the title
            and filename.
        output_dir: Directory where the PNG will be saved.

    Requirements: 4.4, 4.5, 4.6
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(f"{dimension_name} — {version}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    stats_text = (
        f"mean = {np.mean(values):.4f}\n"
        f"std  = {np.std(values):.4f}\n"
        f"min  = {np.min(values):.4f}\n"
        f"max  = {np.max(values):.4f}\n"
        f"N    = {len(values)}"
    )
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()
    filename = f"sub-{SUBJECT}_{dimension_name}_{version}_distribution.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150)
    plt.close("all")
    logger.info("Saved %s", output_dir / filename)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Orchestrate the full distribution exploration pipeline.

    Steps:
        1. Load all runs data (events + Order Matrix).
        2. Collect epoch-level dimension values.
        3. For each dimension with data, generate raw and z-score-
           normalized-per-video histograms.
        4. Skip dimensions with no data (log warning).

    The z-score normalization is computed per ``video_identifier`` group:
    for each group the mean and std are computed, then each value is
    transformed as ``(value - group_mean) / group_std``.  Groups with
    zero std are skipped to avoid division by zero.

    Requirements: 4.4, 4.5, 4.6, 4.7
    """
    runs_data = load_all_runs_data()
    if not runs_data:
        logger.error("No run data loaded — nothing to explore. Exiting.")
        return

    dimension_values = collect_dimension_values(runs_data)
    if not dimension_values:
        logger.error("No dimension values collected. Exiting.")
        return

    output_dir = EXPLORATION_OUTPUT_DIR

    for dimension_name, entries in dimension_values.items():
        if not entries:
            logger.warning(
                "Dimension '%s' has no data — skipping histograms.",
                dimension_name,
            )
            continue

        raw_values = np.array([entry["value"] for entry in entries])

        # --- Raw histogram ---
        plot_distribution(raw_values, dimension_name, "raw", output_dir)

        # --- Z-score normalize per video_identifier ---
        groups: dict[str, list[int]] = {}
        for idx, entry in enumerate(entries):
            vid_id = entry["video_identifier"]
            groups.setdefault(vid_id, []).append(idx)

        normalized_values = np.full_like(raw_values, np.nan)
        for vid_id, indices in groups.items():
            group_vals = raw_values[indices]
            group_mean = np.mean(group_vals)
            group_std = np.std(group_vals)
            if group_std == 0.0:
                logger.warning(
                    "Dimension '%s', video '%s': std=0 — skipping "
                    "normalization for this group.",
                    dimension_name,
                    vid_id,
                )
                continue
            for idx_val in indices:
                normalized_values[idx_val] = (
                    raw_values[idx_val] - group_mean
                ) / group_std

        # Drop NaN entries (groups with zero std)
        valid_mask = ~np.isnan(normalized_values)
        normalized_clean = normalized_values[valid_mask]

        if len(normalized_clean) == 0:
            logger.warning(
                "Dimension '%s': all groups had zero std — skipping "
                "normalized histogram.",
                dimension_name,
            )
            continue

        plot_distribution(
            normalized_clean, dimension_name, "normalized", output_dir
        )

    logger.info("Distribution exploration complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("Script 14 — Distribution Explorer")
    logger.info("Random seed: %d", RANDOM_SEED)
    run_pipeline()
