"""Property-based tests for the permutation test module.

Tests correctness properties of shuffle_targets_within_videos,
run_permutation_test, and compute_p_value using Hypothesis.

Feature: luminance-model-improvements
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from campeones_analysis.luminance.permutation import (
    compute_p_value,
    run_permutation_test,
    shuffle_targets_within_videos,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VIDEO_IDS = ["3_a", "7_a", "9_a", "12_a", "3_b", "7_b"]


@st.composite
def epoch_entries_multi_video(draw: st.DrawFn) -> list[dict]:
    """Generate epoch entries with multiple video groups for shuffle testing.

    Each video group has >= 2 entries so shuffling is meaningful.
    Feature vectors are small random arrays to verify they stay unchanged.
    """
    n_videos = draw(st.integers(min_value=2, max_value=4))
    videos = draw(
        st.lists(
            st.sampled_from(VIDEO_IDS),
            min_size=n_videos,
            max_size=n_videos,
            unique=True,
        )
    )

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    entries: list[dict] = []
    for video_id in videos:
        group_size = draw(st.integers(min_value=2, max_value=15))
        for _ in range(group_size):
            target = draw(
                st.floats(
                    min_value=-1e4,
                    max_value=1e4,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                )
            )
            feature_vec = rng.standard_normal(5)
            entries.append(
                {"video_identifier": video_id, "y": target, "X": feature_vec}
            )

    return entries


# ---------------------------------------------------------------------------
# Property 4: Permutation shuffle preserves within-video target multisets
#              and features
# Validates: Requirements 4.2, 4.3
# ---------------------------------------------------------------------------


@given(entries=epoch_entries_multi_video())
@settings(max_examples=100, deadline=None)
def test_property4_shuffle_preserves_multisets_and_features(
    entries: list[dict],
) -> None:
    """Property 4: Permutation shuffle preserves within-video target multisets and features.

    For any list of epoch entries with multiple video groups, after shuffling
    targets within each video group, the multiset of target values within each
    video group should be identical to the original, and all feature vectors
    (X) should be unchanged (element-wise equal).

    **Validates: Requirements 4.2, 4.3**
    """
    # Feature: luminance-model-improvements, Property 4: Permutation shuffle preserves within-video target multisets and features
    rng = np.random.default_rng(42)
    shuffled = shuffle_targets_within_videos(entries, rng)

    # Same length
    assert len(shuffled) == len(entries)

    # Group originals and shuffled by video
    orig_groups: dict[str, list[float]] = {}
    shuf_groups: dict[str, list[float]] = {}
    for orig, shuf in zip(entries, shuffled):
        vid = orig["video_identifier"]
        orig_groups.setdefault(vid, []).append(orig["y"])
        shuf_groups.setdefault(vid, []).append(shuf["y"])

    # Same video groups exist
    assert set(orig_groups.keys()) == set(shuf_groups.keys())

    # Within each video: multiset of targets is preserved
    for vid in orig_groups:
        orig_counter = Counter(orig_groups[vid])
        shuf_counter = Counter(shuf_groups[vid])
        assert orig_counter == shuf_counter, (
            f"Video {vid}: target multiset changed after shuffle"
        )

    # Feature vectors (X) are unchanged
    for orig, shuf in zip(entries, shuffled):
        np.testing.assert_array_equal(
            orig["X"],
            shuf["X"],
            err_msg="Feature vector X was modified by shuffle",
        )

    # Video identifiers are unchanged
    for orig, shuf in zip(entries, shuffled):
        assert orig["video_identifier"] == shuf["video_identifier"]


# ---------------------------------------------------------------------------
# Property 5: Null distribution has correct length
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


@given(
    n_permutations=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property5_null_distribution_length(
    n_permutations: int,
    seed: int,
) -> None:
    """Property 5: Null distribution has correct length.

    For any positive integer n_permutations, the permutation test should
    produce a null distribution array of exactly n_permutations elements.

    **Validates: Requirements 4.4**
    """
    # Feature: luminance-model-improvements, Property 5: Null distribution has correct length

    # Minimal epoch entries — two video groups with 3 entries each
    entries = [
        {"video_identifier": "3_a", "y": 1.0, "X": np.array([0.1])},
        {"video_identifier": "3_a", "y": 2.0, "X": np.array([0.2])},
        {"video_identifier": "3_a", "y": 3.0, "X": np.array([0.3])},
        {"video_identifier": "7_a", "y": 4.0, "X": np.array([0.4])},
        {"video_identifier": "7_a", "y": 5.0, "X": np.array([0.5])},
        {"video_identifier": "7_a", "y": 6.0, "X": np.array([0.6])},
    ]

    # Dummy evaluation function — returns mean of targets as "r"
    def dummy_evaluate(epoch_entries: list[dict]) -> float:
        targets = [entry["y"] for entry in epoch_entries]
        return float(np.mean(targets))

    result = run_permutation_test(
        epoch_entries=entries,
        build_and_evaluate_fn=dummy_evaluate,
        n_permutations=n_permutations,
        random_seed=seed,
    )

    assert result["null_distribution"].shape == (n_permutations,), (
        f"Expected null_distribution length {n_permutations}, "
        f"got {result['null_distribution'].shape}"
    )


# ---------------------------------------------------------------------------
# Property 6: P-value equals proportion of null values >= observed
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------


@given(
    null_values=st.lists(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
        min_size=1,
        max_size=200,
    ),
    observed_r=st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
@settings(max_examples=100, deadline=None)
def test_property6_p_value_proportion(
    null_values: list[float],
    observed_r: float,
) -> None:
    """Property 6: P-value equals proportion of null values >= observed.

    For any 1-D array of null distribution values and any observed value,
    the p-value should equal count(null >= observed_r) / len(null).
    When observed_r > all null values, p-value should be 0.0.
    When observed_r <= all null values, p-value should be 1.0.

    **Validates: Requirements 4.5**
    """
    # Feature: luminance-model-improvements, Property 6: P-value equals proportion of null values >= observed
    null_distribution = np.array(null_values)

    p_value = compute_p_value(null_distribution, observed_r)

    expected_p = float(np.sum(null_distribution >= observed_r) / len(null_distribution))
    assert p_value == expected_p, (
        f"p_value={p_value}, expected={expected_p} "
        f"(observed_r={observed_r}, null has {len(null_values)} values)"
    )

    # Boundary checks
    if observed_r > max(null_values):
        assert p_value == 0.0, "p-value should be 0.0 when observed > all null"
    if observed_r <= min(null_values):
        assert p_value == 1.0, "p-value should be 1.0 when observed <= all null"
