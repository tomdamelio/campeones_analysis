"""Explore luminance time-series from stimulus videos.

Loads the 4 experimental luminance CSVs (videos 3, 7, 9, 12), generates
raw time-series plots, temporal difference plots, and prints descriptive
statistics for each video.

Figures are saved to ``results/modeling/luminance/exploration/``.

Requirements: 1.1, 1.2, 1.3, 1.4
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure scripts/modeling is on sys.path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_luminance import (
    EXPERIMENTAL_VIDEOS,
    LUMINANCE_CSV_MAP,
    RESULTS_PATH,
    STIMULI_PATH,
)

from campeones_analysis.luminance.sync import load_luminance_csv


def compute_temporal_diff(luminance_series: np.ndarray) -> np.ndarray:
    """Compute consecutive temporal differences of a luminance series.

    Args:
        luminance_series: 1-D array of luminance values.

    Returns:
        1-D array of length ``len(luminance_series) - 1`` where each element
        is ``luminance[i+1] - luminance[i]``.
    """
    return np.diff(luminance_series)


def compute_descriptive_stats(luminance_df: pd.DataFrame) -> dict[str, float]:
    """Compute descriptive statistics for a luminance time-series.

    Args:
        luminance_df: DataFrame with ``timestamp`` and ``luminance`` columns.

    Returns:
        Dictionary with keys: mean, std, min, max, duration_s.
    """
    luminance_values = luminance_df["luminance"].values
    timestamps = luminance_df["timestamp"].values
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    return {
        "mean": float(np.mean(luminance_values)),
        "std": float(np.std(luminance_values)),
        "min": float(np.min(luminance_values)),
        "max": float(np.max(luminance_values)),
        "duration_s": duration_s,
    }


def plot_raw_luminance(
    video_data: dict[int, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a figure with raw luminance time-series (one subplot per video).

    Args:
        video_data: Mapping of video_id → luminance DataFrame.
        output_dir: Directory where the figure will be saved.

    Requirements: 1.1, 1.3
    """
    video_ids = sorted(video_data.keys())
    n_videos = len(video_ids)

    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 3 * n_videos), sharex=False)
    if n_videos == 1:
        axes = [axes]

    for ax, video_id in zip(axes, video_ids):
        luminance_df = video_data[video_id]
        ax.plot(
            luminance_df["timestamp"].values,
            luminance_df["luminance"].values,
            linewidth=0.5,
            color="green",
            alpha=0.8,
        )
        ax.set_title(f"Video {video_id} — Raw Luminance")
        ax.set_ylabel("Luminance (0–255)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-5, 260)

    fig.tight_layout()
    output_path = output_dir / "raw_luminance_timeseries.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_temporal_diffs(
    video_data: dict[int, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate a figure with temporal differences (one subplot per video).

    Args:
        video_data: Mapping of video_id → luminance DataFrame.
        output_dir: Directory where the figure will be saved.

    Requirements: 1.2, 1.3
    """
    video_ids = sorted(video_data.keys())
    n_videos = len(video_ids)

    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 3 * n_videos), sharex=False)
    if n_videos == 1:
        axes = [axes]

    for ax, video_id in zip(axes, video_ids):
        luminance_df = video_data[video_id]
        luminance_values = luminance_df["luminance"].values
        timestamps = luminance_df["timestamp"].values

        diffs = compute_temporal_diff(luminance_values)
        # Use midpoint timestamps for diff series
        diff_timestamps = (timestamps[:-1] + timestamps[1:]) / 2.0

        ax.plot(
            diff_timestamps,
            diffs,
            linewidth=0.5,
            color="darkorange",
            alpha=0.8,
        )
        ax.set_title(f"Video {video_id} — Temporal Differences (Δluminance)")
        ax.set_ylabel("Δ Luminance")
        ax.set_xlabel("Time (s)")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    output_path = output_dir / "luminance_temporal_diffs.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_descriptive_stats(
    video_data: dict[int, pd.DataFrame],
) -> None:
    """Print descriptive statistics for each video luminance series.

    Args:
        video_data: Mapping of video_id → luminance DataFrame.

    Requirements: 1.4
    """
    print("\n" + "=" * 60)
    print("LUMINANCE DESCRIPTIVE STATISTICS")
    print("=" * 60)

    for video_id in sorted(video_data.keys()):
        stats = compute_descriptive_stats(video_data[video_id])
        n_frames = len(video_data[video_id])
        print(f"\n  Video {video_id} ({n_frames} frames):")
        print(f"    Mean:     {stats['mean']:.2f}")
        print(f"    Std:      {stats['std']:.2f}")
        print(f"    Min:      {stats['min']:.2f}")
        print(f"    Max:      {stats['max']:.2f}")
        print(f"    Duration: {stats['duration_s']:.2f} s")


def run_pipeline() -> None:
    """Execute the luminance exploration pipeline.

    Loads all 4 experimental luminance CSVs, generates plots, and prints
    descriptive statistics.
    """
    print("=" * 60)
    print("08 — Luminance Exploration")
    print("=" * 60)

    output_dir = RESULTS_PATH / "exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load luminance CSVs
    video_data: dict[int, pd.DataFrame] = {}
    for video_id in EXPERIMENTAL_VIDEOS:
        csv_filename = LUMINANCE_CSV_MAP[video_id]
        csv_path = STIMULI_PATH / csv_filename
        print(f"\nLoading video {video_id}: {csv_path}")
        try:
            video_data[video_id] = load_luminance_csv(csv_path)
            print(f"  → {len(video_data[video_id])} frames loaded")
        except FileNotFoundError:
            print(f"  WARNING: CSV not found, skipping video {video_id}")

    if not video_data:
        print("\nNo luminance data loaded. Exiting.")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_raw_luminance(video_data, output_dir)
    plot_temporal_diffs(video_data, output_dir)

    # Print statistics
    print_descriptive_stats(video_data)

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
