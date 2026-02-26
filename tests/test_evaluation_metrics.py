"""Property-based tests for evaluation metrics.

Feature: eeg-luminance-validation
Property 6: R² calculation matches sklearn

Validates: Requirements 7.1
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import r2_score

from campeones_analysis.luminance.evaluation import compute_r2_score

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_float_array = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@st.composite
def non_constant_array(draw: st.DrawFn) -> np.ndarray:
    """Draw a float array that is not constant (std > 0)."""
    arr = draw(_float_array)
    # Ensure at least two distinct values so y_true is non-constant
    arr[0] = arr[0] + 1.0
    return arr


# ---------------------------------------------------------------------------
# Property 6: R² calculation matches sklearn
# ---------------------------------------------------------------------------


@given(
    y_true=non_constant_array(),
    y_pred=_float_array,
)
@settings(max_examples=200)
def test_r2_matches_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Property 6: compute_r2_score must equal sklearn.metrics.r2_score.

    For any pair of y_true (non-constant) and y_pred arrays of equal length
    (≥ 2), compute_r2_score(y_true, y_pred) should equal
    sklearn.metrics.r2_score(y_true, y_pred) within floating-point tolerance.

    Validates: Requirements 7.1
    """
    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    expected = r2_score(y_true, y_pred)
    actual = compute_r2_score(y_true, y_pred)

    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (
        f"compute_r2_score={actual} != sklearn r2_score={expected}"
    )


# ---------------------------------------------------------------------------
# Unit tests – specific examples
# ---------------------------------------------------------------------------


def test_r2_perfect_prediction() -> None:
    """Perfect predictions yield R² = 1.0."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_r2_score(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction() -> None:
    """Predicting the mean yields R² = 0.0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert compute_r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_r2_negative_for_bad_predictions() -> None:
    """Predictions worse than the mean yield negative R²."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert compute_r2_score(y_true, y_pred) < 0.0
