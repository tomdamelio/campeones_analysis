"""Property-based tests for ERP luminance change detection.

Feature: eeg-luminance-validation
Property 8: Top N luminance changes detection
Validates: Requirements 10.1
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Pure function extracted for testing (mirrors script 21 logic).
# ---------------------------------------------------------------------------


def _detect_top_n_luminance_changes(
    luminance_values: np.ndarray,
    n_changes: int,
) -> np.ndarray:
    """Detect the top N moments of largest absolute luminance change.

    Mirrors ``detect_top_n_luminance_changes`` from script 21 without
    any config or I/O dependencies.

    Args:
        luminance_values: 1-D array of luminance values over time.
        n_changes: Number of top changes to detect.

    Returns:
        1-D array of indices into the diff array, sorted by |diff| descending.
    """
    if len(luminance_values) < 2:
        return np.array([], dtype=np.intp)

    luminance_diff = np.diff(luminance_values)
    abs_diff = np.abs(luminance_diff)

    n_available = len(abs_diff)
    n_top = min(n_changes, n_available)

    top_indices = np.argsort(abs_diff)[::-1][:n_top]

    return top_indices


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Luminance time-series: 3–100 values in [0, 255] range
_luminance_strategy = st.lists(
    st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=100,
).map(np.array)


# ---------------------------------------------------------------------------
# Property 8: Top N luminance changes detection
# Validates: Requirements 10.1
# ---------------------------------------------------------------------------


@given(
    luminance_values=_luminance_strategy,
    n_changes=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_top_n_luminance_changes_detection(
    luminance_values: np.ndarray,
    n_changes: int,
) -> None:
    """Property 8: Top N luminance changes detection.

    For any luminance time series of length L (L > N) and any N >= 1,
    the detected change indices should correspond to the N time-points
    with the largest absolute first-difference values, sorted by
    magnitude descending.

    Validates: Requirements 10.1
    """
    result_indices = _detect_top_n_luminance_changes(luminance_values, n_changes)

    luminance_diff = np.diff(luminance_values)
    abs_diff = np.abs(luminance_diff)
    n_available = len(abs_diff)
    expected_count = min(n_changes, n_available)

    # Property: correct number of indices returned
    assert len(result_indices) == expected_count, (
        f"Expected {expected_count} indices, got {len(result_indices)}"
    )

    # Property: all indices are valid (within diff array bounds)
    for idx in result_indices:
        assert 0 <= idx < n_available, (
            f"Index {idx} out of bounds for diff array of length {n_available}"
        )

    # Property: no duplicate indices
    assert len(set(result_indices)) == len(result_indices), (
        f"Duplicate indices found: {result_indices}"
    )

    # Property: sorted by magnitude descending
    magnitudes = abs_diff[result_indices]
    for pos in range(len(magnitudes) - 1):
        assert magnitudes[pos] >= magnitudes[pos + 1], (
            f"Not sorted by magnitude descending at position {pos}: "
            f"{magnitudes[pos]} < {magnitudes[pos + 1]}"
        )

    # Property: these are the actual top N (no larger value was missed)
    if expected_count < n_available:
        min_selected_magnitude = magnitudes[-1]
        all_sorted = np.sort(abs_diff)[::-1]
        nth_largest = all_sorted[expected_count - 1]
        assert min_selected_magnitude >= nth_largest - 1e-10, (
            f"Smallest selected magnitude {min_selected_magnitude} is less than "
            f"the {expected_count}-th largest value {nth_largest}"
        )
