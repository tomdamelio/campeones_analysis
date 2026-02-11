"""Permutation testing for luminance prediction pipeline.

Provides non-parametric statistical significance testing by shuffling
target labels within video groups and comparing the observed Pearson r
against a null distribution.  The within-video shuffle preserves the
LOVO_CV structure so that the null hypothesis is correctly formulated.

Public API:
    shuffle_targets_within_videos – pure shuffle, no I/O
    compute_p_value              – proportion of null >= observed
    run_permutation_test         – full permutation loop
    plot_permutation_histogram   – save histogram to disk
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def shuffle_targets_within_videos(
    epoch_entries: list[dict],
    rng: np.random.Generator,
    video_key: str = "video_identifier",
    target_key: str = "y",
) -> list[dict]:
    """Shuffle target values within each video group, keeping features intact.

    Creates a shallow copy of each epoch dict and replaces the target value
    with a shuffled version drawn from the same video group.  Feature vectors
    (``X``) and all other keys are preserved unchanged.

    Args:
        epoch_entries: List of epoch dicts, each containing at minimum
            the keys specified by *video_key* and *target_key*.
        rng: NumPy random Generator used for shuffling.
        video_key: Key used to group epochs by video.
        target_key: Key containing the target value to shuffle.

    Returns:
        New list of epoch dicts with *target_key* shuffled within each
        video group.  All other fields are identical to the originals.
    """
    # Group indices by video
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(epoch_entries):
        groups[entry[video_key]].append(idx)

    shuffled: list[dict] = [copy.copy(entry) for entry in epoch_entries]

    for _video_id, indices in groups.items():
        targets = [epoch_entries[i][target_key] for i in indices]
        rng.shuffle(targets)
        for i, target_value in zip(indices, targets):
            shuffled[i][target_key] = target_value

    return shuffled


def compute_p_value(
    null_distribution: np.ndarray,
    observed_r: float,
) -> float:
    """Compute p-value as proportion of null values >= observed.

    Args:
        null_distribution: 1-D array of test-statistic values under the
            null hypothesis.
        observed_r: The observed test statistic (mean Pearson r).

    Returns:
        Proportion of null distribution values that are greater than or
        equal to *observed_r*.
    """
    if len(null_distribution) == 0:
        logger.warning("Empty null distribution; returning NaN p-value.")
        return float("nan")

    return float(np.sum(null_distribution >= observed_r) / len(null_distribution))


def run_permutation_test(
    epoch_entries: list[dict],
    build_and_evaluate_fn: Callable[[list[dict]], float],
    n_permutations: int,
    random_seed: int,
    video_key: str = "video_identifier",
    target_key: str = "y",
) -> dict:
    """Run a permutation test by shuffling targets within each video group.

    For each permutation iteration, shuffles the target values within video
    groups, then calls *build_and_evaluate_fn* on the shuffled data to obtain
    a test statistic (mean Pearson r across LOVO_CV folds).  The collection
    of these values forms the null distribution.

    Args:
        epoch_entries: The full list of epoch dicts.
        build_and_evaluate_fn: A callable that takes epoch_entries and returns
            the mean Pearson r across LOVO_CV folds.
        n_permutations: Number of permutation iterations.
        random_seed: Base seed for reproducibility.
        video_key: Key to group epochs by video.
        target_key: Key containing the target value to shuffle.

    Returns:
        Dict with keys:
            - ``null_distribution``: 1-D array of length *n_permutations*
            - ``observed_r``: float, the unshuffled test statistic
            - ``p_value``: float, proportion of null >= observed
    """
    if n_permutations <= 0:
        logger.warning(
            "n_permutations=%d; returning empty null distribution.", n_permutations
        )
        return {
            "null_distribution": np.array([]),
            "observed_r": build_and_evaluate_fn(epoch_entries),
            "p_value": float("nan"),
        }

    observed_r = build_and_evaluate_fn(epoch_entries)
    null_distribution = np.empty(n_permutations, dtype=np.float64)

    for perm_idx in range(n_permutations):
        rng = np.random.default_rng(random_seed + perm_idx)
        shuffled_entries = shuffle_targets_within_videos(
            epoch_entries, rng, video_key=video_key, target_key=target_key
        )
        null_distribution[perm_idx] = build_and_evaluate_fn(shuffled_entries)

    p_value = compute_p_value(null_distribution, observed_r)

    return {
        "null_distribution": null_distribution,
        "observed_r": observed_r,
        "p_value": p_value,
    }


def plot_permutation_histogram(
    null_distribution: np.ndarray,
    observed_r: float,
    p_value: float,
    output_path: Path,
) -> None:
    """Save a histogram of the null distribution with observed r marked.

    Args:
        null_distribution: 1-D array of null test-statistic values.
        observed_r: The observed (unshuffled) test statistic.
        p_value: The computed p-value.
        output_path: File path where the figure will be saved.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_distribution, bins=30, edgecolor="black", alpha=0.7, label="Null")
    ax.axvline(
        observed_r, color="red", linestyle="--", linewidth=2, label=f"Observed r={observed_r:.3f}"
    )
    ax.set_xlabel("Mean Pearson r")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test  (p = {p_value:.4f})")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
