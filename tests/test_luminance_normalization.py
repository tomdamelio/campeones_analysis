"""Property-based and unit tests for z-score normalization per video.

Tests correctness properties of zscore_per_video using Hypothesis,
plus edge-case unit tests.

Feature: luminance-model-improvements
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from campeones_analysis.luminance.normalization import zscore_per_video


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VIDEO_IDS = ["3_a", "7_a", "9_a", "12_a", "3_b", "7_b"]


@st.composite
def epoch_entries_with_variance(draw: st.DrawFn) -> list[dict]:
    """Generate epoch entries where each video group has >= 2 distinct targets.

    This ensures every group has non-zero standard deviation, which is the
    main-path scenario for Property 2.
    """
    n_videos = draw(st.integers(min_value=1, max_value=4))
    videos = draw(
        st.lists(
            st.sampled_from(VIDEO_IDS),
            min_size=n_videos,
            max_size=n_videos,
            unique=True,
        )
    )

    entries: list[dict] = []
    for video_id in videos:
        group_size = draw(st.integers(min_value=2, max_value=20))
        targets = draw(
            st.lists(
                st.floats(
                    min_value=-1e4,
                    max_value=1e4,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=group_size,
                max_size=group_size,
            )
        )
        # Ensure at least two values with meaningful spread so std > 0
        unique_targets = set(targets)
        assume(len(unique_targets) >= 2)
        spread = max(targets) - min(targets)
        assume(spread > 1e-6)
        for target in targets:
            entries.append({"video_identifier": video_id, "y": target, "X": np.zeros(5)})

    return entries


# ---------------------------------------------------------------------------
# Property 2: Z-score normalization produces zero mean and unit variance
# ---------------------------------------------------------------------------


@given(entries=epoch_entries_with_variance())
@settings(max_examples=100, deadline=None)
def test_property2_zscore_zero_mean_unit_variance(entries: list[dict]) -> None:
    """Property 2: Z-score normalization produces zero mean and unit variance per video.

    **Validates: Requirements 2.2, 2.3, 2.4**

    For any list of epoch entries where each video group has at least two
    distinct target values, after zscore_per_video the targets within each
    group should have mean ≈ 0.0 and std ≈ 1.0.  The total number of
    entries must be preserved.
    """
    result = zscore_per_video(entries)

    # Length preserved
    assert len(result) == len(entries)

    # Group by video and check statistics
    groups: dict[str, list[float]] = {}
    for entry in result:
        vid = entry["video_identifier"]
        groups.setdefault(vid, []).append(entry["y"])

    for video_id, values in groups.items():
        arr = np.array(values)
        assert math.isclose(arr.mean(), 0.0, abs_tol=1e-7), (
            f"Video {video_id}: mean={arr.mean()}, expected ≈ 0.0"
        )
        assert math.isclose(arr.std(), 1.0, abs_tol=1e-7), (
            f"Video {video_id}: std={arr.std()}, expected ≈ 1.0"
        )


# ---------------------------------------------------------------------------
# Unit tests – edge cases (Task 2.3)
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty() -> None:
    """Empty input returns empty output."""
    assert zscore_per_video([]) == []


def test_single_value_video_returns_zero() -> None:
    """A video group with a single entry (zero std) returns target 0.0."""
    entries = [{"video_identifier": "3_a", "y": 42.0}]
    result = zscore_per_video(entries)
    assert len(result) == 1
    assert result[0]["y"] == 0.0
