"""Property-based tests for EEG QA rejection percentage calculation.

Feature: eeg-luminance-validation
Property 1: Rejection percentage calculation

Validates: Requirements 1.2
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from campeones_analysis.luminance.qa import compute_rejection_percentage

# ---------------------------------------------------------------------------
# Property 1: Rejection percentage calculation
# ---------------------------------------------------------------------------


@given(
    n_total=st.integers(min_value=1, max_value=10_000),
    n_rejected=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=500)
def test_rejection_percentage_formula(n_total: int, n_rejected: int) -> None:
    """Property 1: rejection_pct == n_rejected / n_total * 100.

    For any reject log with N total epochs (N > 0) and M rejected epochs
    (0 <= M <= N), the computed rejection percentage should equal
    M / N × 100 exactly.

    Validates: Requirements 1.2
    """
    # Clamp n_rejected to valid range
    n_rejected = min(n_rejected, n_total)

    result = compute_rejection_percentage(n_total=n_total, n_rejected=n_rejected)
    expected = (n_rejected / n_total) * 100.0

    assert abs(result - expected) < 1e-10, (
        f"n_total={n_total}, n_rejected={n_rejected}: "
        f"got {result}, expected {expected}"
    )


@given(n_total=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=200)
def test_rejection_percentage_bounds(n_total: int) -> None:
    """Rejection percentage is always in [0.0, 100.0]."""
    pct_zero = compute_rejection_percentage(n_total=n_total, n_rejected=0)
    pct_all = compute_rejection_percentage(n_total=n_total, n_rejected=n_total)

    assert pct_zero == pytest.approx(0.0)
    assert pct_all == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Unit tests – specific examples
# ---------------------------------------------------------------------------


def test_rejection_percentage_half() -> None:
    """50% rejection for equal total and rejected."""
    assert compute_rejection_percentage(n_total=10, n_rejected=5) == pytest.approx(50.0)


def test_rejection_percentage_none() -> None:
    """0% rejection when no epochs are rejected."""
    assert compute_rejection_percentage(n_total=100, n_rejected=0) == pytest.approx(0.0)


def test_rejection_percentage_all() -> None:
    """100% rejection when all epochs are rejected."""
    assert compute_rejection_percentage(n_total=42, n_rejected=42) == pytest.approx(100.0)


def test_rejection_percentage_single_epoch() -> None:
    """Single epoch rejected out of one total → 100%."""
    assert compute_rejection_percentage(n_total=1, n_rejected=1) == pytest.approx(100.0)
