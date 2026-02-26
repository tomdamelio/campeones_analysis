"""Property-based tests for delta luminance and change label computation.

Feature: eeg-luminance-validation
Property 5: Delta luminance computation and first-epoch discard
Property 7: Binary change labels from threshold

Validates: Requirements 8.1, 8.2, 9.1
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from campeones_analysis.luminance.targets import (
    compute_change_labels,
    compute_delta_luminance,
)

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_luminance_value = st.floats(
    min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False
)

_video_id = st.sampled_from(["3_a", "7_a", "9_a", "12_a", "3_b", "7_b"])


@st.composite
def epoch_list_for_video(
    draw: st.DrawFn,
    video_id: str,
    min_epochs: int = 2,
    max_epochs: int = 30,
) -> list[dict]:
    """Draw a list of epoch dicts for a single video."""
    n_epochs = draw(st.integers(min_value=min_epochs, max_value=max_epochs))
    luminance_values = draw(
        st.lists(_luminance_value, min_size=n_epochs, max_size=n_epochs)
    )
    return [
        {"video_identifier": video_id, "y": lum, "run_id": "002"}
        for lum in luminance_values
    ]


@st.composite
def multi_video_epoch_list(draw: st.DrawFn) -> list[dict]:
    """Draw epoch lists for 1–4 distinct videos, concatenated."""
    n_videos = draw(st.integers(min_value=1, max_value=4))
    video_ids = draw(
        st.lists(
            _video_id,
            min_size=n_videos,
            max_size=n_videos,
            unique=True,
        )
    )
    all_epochs: list[dict] = []
    for vid in video_ids:
        epochs = draw(epoch_list_for_video(vid))
        all_epochs.extend(epochs)
    return all_epochs


# ---------------------------------------------------------------------------
# Property 5: Delta luminance computation and first-epoch discard
# ---------------------------------------------------------------------------


@given(epoch_entries=multi_video_epoch_list())
@settings(max_examples=200)
def test_delta_luminance_count_and_values(epoch_entries: list[dict]) -> None:
    """Property 5: compute_delta_luminance returns exactly one fewer epoch per video.

    For any list of epoch entries grouped by video, the result should have
    exactly (n_epochs_per_video - 1) entries per video, and each delta value
    should equal L_i - L_{i-1} within floating-point tolerance.

    Validates: Requirements 8.1, 8.2
    """
    delta_entries = compute_delta_luminance(epoch_entries)

    # Group original entries by video
    from collections import defaultdict

    video_groups: dict[str, list[dict]] = defaultdict(list)
    for entry in epoch_entries:
        video_groups[entry["video_identifier"]].append(entry)

    # Group delta entries by video
    delta_groups: dict[str, list[dict]] = defaultdict(list)
    for entry in delta_entries:
        delta_groups[entry["video_identifier"]].append(entry)

    for video_id, original in video_groups.items():
        n_original = len(original)
        delta_for_video = delta_groups[video_id]

        # Exactly one fewer epoch per video
        assert len(delta_for_video) == n_original - 1, (
            f"Video {video_id}: expected {n_original - 1} delta epochs, "
            f"got {len(delta_for_video)}"
        )

        # Each delta value equals L_i - L_{i-1}
        for position, delta_entry in enumerate(delta_for_video):
            expected_delta = original[position + 1]["y"] - original[position]["y"]
            assert abs(delta_entry["y"] - expected_delta) < 1e-10, (
                f"Video {video_id}, epoch {position + 1}: "
                f"delta={delta_entry['y']}, expected={expected_delta}"
            )


@given(epoch_entries=multi_video_epoch_list())
@settings(max_examples=100)
def test_delta_luminance_preserves_other_fields(epoch_entries: list[dict]) -> None:
    """Non-target fields are preserved unchanged after delta computation."""
    delta_entries = compute_delta_luminance(epoch_entries)
    for entry in delta_entries:
        assert "video_identifier" in entry
        assert "run_id" in entry


# ---------------------------------------------------------------------------
# Property 7: Binary change labels from threshold
# ---------------------------------------------------------------------------


@given(
    epoch_entries=multi_video_epoch_list(),
    threshold=st.floats(
        min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=200)
def test_change_labels_threshold_assignment(
    epoch_entries: list[dict], threshold: float
) -> None:
    """Property 7: compute_change_labels assigns 1 iff |delta| > threshold.

    For any list of epoch entries with delta luminance targets and any
    threshold > 0, labels should be 1 when |delta| > threshold and 0
    otherwise. All other fields must be preserved.

    Validates: Requirements 9.1
    """
    # First compute deltas so we have realistic delta values
    delta_entries = compute_delta_luminance(epoch_entries)
    if not delta_entries:
        return  # Nothing to test for single-epoch videos

    labeled_entries = compute_change_labels(delta_entries, threshold=threshold)

    assert len(labeled_entries) == len(delta_entries)

    for original, labeled in zip(delta_entries, labeled_entries):
        delta_value = original["y"]
        expected_label = int(abs(delta_value) > threshold)
        assert labeled["y"] == expected_label, (
            f"|delta|={abs(delta_value)}, threshold={threshold}, "
            f"expected={expected_label}, got={labeled['y']}"
        )
        # Other fields preserved
        assert labeled["video_identifier"] == original["video_identifier"]
        assert labeled["run_id"] == original["run_id"]


# ---------------------------------------------------------------------------
# Unit tests – specific examples
# ---------------------------------------------------------------------------


def test_delta_single_video_two_epochs() -> None:
    """Two epochs in one video → one delta entry."""
    entries = [
        {"video_identifier": "3_a", "y": 100.0, "run_id": "002"},
        {"video_identifier": "3_a", "y": 120.0, "run_id": "002"},
    ]
    result = compute_delta_luminance(entries)
    assert len(result) == 1
    assert result[0]["y"] == pytest.approx(20.0)


def test_delta_discards_first_epoch_per_video() -> None:
    """First epoch of each video is discarded."""
    entries = [
        {"video_identifier": "3_a", "y": 50.0, "run_id": "002"},
        {"video_identifier": "3_a", "y": 60.0, "run_id": "002"},
        {"video_identifier": "7_a", "y": 80.0, "run_id": "003"},
        {"video_identifier": "7_a", "y": 70.0, "run_id": "003"},
    ]
    result = compute_delta_luminance(entries)
    assert len(result) == 2
    deltas = {e["video_identifier"]: e["y"] for e in result}
    assert deltas["3_a"] == pytest.approx(10.0)
    assert deltas["7_a"] == pytest.approx(-10.0)


def test_change_labels_boundary() -> None:
    """Exactly at threshold → label 0 (not strictly greater)."""
    entries = [{"video_identifier": "3_a", "y": 5.0, "run_id": "002"}]
    result = compute_change_labels(entries, threshold=5.0)
    assert result[0]["y"] == 0  # |5.0| > 5.0 is False


def test_change_labels_above_threshold() -> None:
    """Above threshold → label 1."""
    entries = [{"video_identifier": "3_a", "y": 5.1, "run_id": "002"}]
    result = compute_change_labels(entries, threshold=5.0)
    assert result[0]["y"] == 1
