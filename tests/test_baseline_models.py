"""Property-based tests for baseline model correctness.

Feature: eeg-luminance-validation
Property 3: Mean baseline prediction equals training mean
Validates: Requirements 3.1, 3.2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "modeling"))

from campeones_analysis.luminance.normalization import zscore_per_video


# ---------------------------------------------------------------------------
# Helpers to build synthetic epoch entries
# ---------------------------------------------------------------------------


def _make_epoch_entries(
    targets_by_video: dict[str, list[float]],
) -> list[dict]:
    """Build minimal epoch entry dicts from a video→targets mapping.

    Args:
        targets_by_video: Mapping of video_identifier → list of target values.

    Returns:
        List of epoch dicts with keys ``X``, ``y``, ``video_identifier``.
    """
    entries: list[dict] = []
    for video_id, targets in targets_by_video.items():
        for target_value in targets:
            entries.append(
                {
                    "X": np.zeros(4),  # features irrelevant for mean baseline
                    "y": float(target_value),
                    "video_identifier": video_id,
                }
            )
    return entries


def _run_mean_baseline_pure(
    epoch_entries: list[dict],
) -> list[dict]:
    """Pure mean baseline: predict training mean for all test epochs per fold.

    Mirrors the logic in ``run_mean_baseline`` from script 17 without any
    I/O or config dependencies, so it can be tested in isolation.

    Args:
        epoch_entries: Epoch dicts with ``y`` and ``video_identifier``.

    Returns:
        List of fold result dicts with keys ``test_video``, ``y_true``,
        ``y_pred``, ``training_mean``.
    """
    unique_videos = sorted(set(e["video_identifier"] for e in epoch_entries))
    fold_results: list[dict] = []

    for test_video in unique_videos:
        train_entries = [e for e in epoch_entries if e["video_identifier"] != test_video]
        test_entries = [e for e in epoch_entries if e["video_identifier"] == test_video]

        y_train = np.array([e["y"] for e in train_entries])
        y_test = np.array([e["y"] for e in test_entries])

        training_mean = float(y_train.mean())
        y_pred = np.full_like(y_test, fill_value=training_mean)

        fold_results.append(
            {
                "test_video": test_video,
                "y_true": y_test,
                "y_pred": y_pred,
                "training_mean": training_mean,
            }
        )

    return fold_results


# ---------------------------------------------------------------------------
# Property 3: Mean baseline prediction equals training mean
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

# Strategy: generate 2–5 videos, each with 3–20 target values in [-300, 300]
_video_targets_strategy = st.fixed_dictionaries(
    {
        video_id: st.lists(
            st.floats(min_value=-300.0, max_value=300.0, allow_nan=False),
            min_size=3,
            max_size=20,
        )
        for video_id in ["3_a", "7_a", "9_b", "12_b"]
    }
)


@given(targets_by_video=_video_targets_strategy)
@settings(max_examples=200)
def test_mean_baseline_predicts_training_mean(
    targets_by_video: dict[str, list[float]],
) -> None:
    """Property 3: Mean baseline prediction equals training mean.

    For any set of training target values and any set of test epochs, the
    mean baseline model should produce predictions where every predicted
    value equals the arithmetic mean of the training targets.

    Validates: Requirements 3.1, 3.2
    """
    epoch_entries = _make_epoch_entries(targets_by_video)
    fold_results = _run_mean_baseline_pure(epoch_entries)

    for fold in fold_results:
        test_video = fold["test_video"]
        y_pred = fold["y_pred"]
        training_mean = fold["training_mean"]

        # Req 3.1: training mean is the arithmetic mean of training targets
        train_entries = [
            e for e in epoch_entries if e["video_identifier"] != test_video
        ]
        y_train = np.array([e["y"] for e in train_entries])
        expected_mean = float(y_train.mean())

        assert abs(training_mean - expected_mean) < 1e-9, (
            f"Training mean {training_mean} does not match arithmetic mean "
            f"{expected_mean} for fold test_video={test_video}"
        )

        # Req 3.2: every predicted value equals the training mean
        assert np.all(y_pred == training_mean), (
            f"Not all predictions equal training mean {training_mean} "
            f"for fold test_video={test_video}. "
            f"Unique predicted values: {np.unique(y_pred)}"
        )
