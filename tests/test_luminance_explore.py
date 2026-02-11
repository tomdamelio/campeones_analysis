"""Property-based tests for luminance exploration functions.

Tests correctness properties of temporal differencing and descriptive
statistics using Hypothesis.

Feature: eeg-luminance-prediction
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

# Make scripts/modeling importable
_scripts_modeling = str(Path(__file__).resolve().parents[1] / "scripts" / "modeling")
if _scripts_modeling not in sys.path:
    sys.path.insert(0, _scripts_modeling)

# Import the module with numeric prefix via importlib
_explore_mod = importlib.import_module("08_explore_luminance")
compute_temporal_diff = _explore_mod.compute_temporal_diff
compute_descriptive_stats = _explore_mod.compute_descriptive_stats


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

luminance_array_strategy = st.lists(
    st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=500,
).map(np.array)


@st.composite
def luminance_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a valid luminance DataFrame with monotonic timestamps."""
    n_frames = draw(st.integers(min_value=2, max_value=500))
    fps = draw(st.sampled_from([30.0, 60.0]))
    timestamps = np.arange(n_frames) / fps

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    luminance_values = rng.uniform(0.0, 255.0, size=n_frames)

    return pd.DataFrame({"timestamp": timestamps, "luminance": luminance_values})


# ---------------------------------------------------------------------------
# Property 1: La serie de diferencias temporales tiene longitud N-1
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


@given(luminance_series=luminance_array_strategy)
@settings(max_examples=100)
def test_property1_temporal_diff_length(luminance_series: np.ndarray) -> None:
    """Property 1: Temporal difference series has length N-1.

    For any luminance series of length N (N >= 2), the temporal difference
    series must have length N-1, and each diff[i] must equal
    luminance[i+1] - luminance[i].

    **Validates: Requirements 1.2**
    """
    # Feature: eeg-luminance-prediction, Property 1
    diffs = compute_temporal_diff(luminance_series)

    n_values = len(luminance_series)
    assert len(diffs) == n_values - 1, (
        f"Expected {n_values - 1} diffs, got {len(diffs)}"
    )

    # Each diff must be luminance[i+1] - luminance[i]
    for idx in range(len(diffs)):
        expected = luminance_series[idx + 1] - luminance_series[idx]
        np.testing.assert_allclose(
            diffs[idx],
            expected,
            atol=1e-10,
            err_msg=f"diff[{idx}] mismatch",
        )


# ---------------------------------------------------------------------------
# Property 2: Invariantes de estadísticas descriptivas de luminancia
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@given(luminance_df=luminance_dataframe())
@settings(max_examples=100)
def test_property2_descriptive_stats_invariants(
    luminance_df: pd.DataFrame,
) -> None:
    """Property 2: Descriptive statistics invariants.

    For any non-empty luminance series, the descriptive statistics must
    satisfy: min <= mean <= max, std >= 0, and all luminance values must
    be in [0, 255].

    **Validates: Requirements 1.4**
    """
    # Feature: eeg-luminance-prediction, Property 2
    stats = compute_descriptive_stats(luminance_df)

    assert stats["min"] <= stats["mean"], (
        f"min ({stats['min']}) > mean ({stats['mean']})"
    )
    assert stats["mean"] <= stats["max"], (
        f"mean ({stats['mean']}) > max ({stats['max']})"
    )
    assert stats["std"] >= 0.0, f"std ({stats['std']}) is negative"

    # All luminance values in [0, 255]
    luminance_values = luminance_df["luminance"].values
    assert np.all(luminance_values >= 0.0), "Luminance values below 0"
    assert np.all(luminance_values <= 255.0), "Luminance values above 255"

    # Duration must be non-negative
    assert stats["duration_s"] >= 0.0, (
        f"Duration ({stats['duration_s']}) is negative"
    )


# Import the verification module
_verify_mod = importlib.import_module("09_verify_luminance_markers")
filter_video_luminance_events = _verify_mod.filter_video_luminance_events


# ---------------------------------------------------------------------------
# Strategy for events DataFrames
# ---------------------------------------------------------------------------

TRIAL_TYPES = [
    "fixation",
    "calm",
    "video",
    "video_luminance",
    "rest",
    "instruction",
]


@st.composite
def events_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a synthetic EEG events DataFrame with mixed trial_types."""
    n_events = draw(st.integers(min_value=0, max_value=50))
    trial_types = draw(
        st.lists(
            st.sampled_from(TRIAL_TYPES),
            min_size=n_events,
            max_size=n_events,
        )
    )
    onsets = sorted(draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
            min_size=n_events,
            max_size=n_events,
        )
    ))
    durations = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=300.0, allow_nan=False, allow_infinity=False),
            min_size=n_events,
            max_size=n_events,
        )
    )
    return pd.DataFrame({
        "onset": onsets,
        "duration": durations,
        "trial_type": trial_types,
    })


# ---------------------------------------------------------------------------
# Property 3: Filtrado de eventos video_luminance
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


@given(events_df=events_dataframe())
@settings(max_examples=100)
def test_property3_filter_video_luminance(events_df: pd.DataFrame) -> None:
    """Property 3: Filtering by video_luminance returns only matching rows.

    For any events DataFrame, filtering by trial_type == 'video_luminance'
    must return exclusively rows with that trial_type, and the number of
    returned rows must be <= the total number of rows.

    **Validates: Requirements 2.1**
    """
    # Feature: eeg-luminance-prediction, Property 3
    filtered = filter_video_luminance_events(events_df)

    # All rows in filtered must have trial_type == 'video_luminance'
    if not filtered.empty:
        assert (filtered["trial_type"] == "video_luminance").all(), (
            "Filtered DataFrame contains non-video_luminance rows"
        )

    # Filtered count must be <= total count
    assert len(filtered) <= len(events_df), (
        f"Filtered ({len(filtered)}) > total ({len(events_df)})"
    )

    # Filtered count must equal the actual count of video_luminance rows
    expected_count = (events_df["trial_type"] == "video_luminance").sum()
    assert len(filtered) == expected_count, (
        f"Expected {expected_count} video_luminance rows, got {len(filtered)}"
    )
