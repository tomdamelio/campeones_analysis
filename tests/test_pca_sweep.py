"""Property-based tests for PCA sweep cumulative variance analysis.

Feature: eeg-luminance-validation
Property 4: Cumulative PCA explained variance is monotonically non-decreasing

**Validates: Requirements 5.2**
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helper: compute cumulative variance the same way the pipeline does
# ---------------------------------------------------------------------------


def _compute_cumulative_variance(
    data_matrix: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA and return (explained_variance_ratio, cumulative_variance).

    Mirrors the logic in ``compute_pca_variance_analysis`` from script 18,
    isolated here for pure testability without I/O dependencies.

    Args:
        data_matrix: 2-D array of shape (n_samples, n_features).
        n_components: Number of PCA components to fit.

    Returns:
        Tuple of (explained_variance_ratio, cumulative_variance), each a
        1-D array of length n_components.
    """
    max_feasible = min(data_matrix.shape[0], data_matrix.shape[1])
    actual_components = min(n_components, max_feasible)
    pca_model = PCA(n_components=actual_components, random_state=42)
    pca_model.fit(data_matrix)
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    return explained_variance_ratio, cumulative_variance


# ---------------------------------------------------------------------------
# Property 4: Cumulative PCA explained variance is monotonically non-decreasing
# ---------------------------------------------------------------------------


@given(
    n_samples=st.integers(min_value=10, max_value=200),
    n_features=st.integers(min_value=2, max_value=50),
    n_components=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_cumulative_pca_variance_is_monotonically_nondecreasing(
    n_samples: int,
    n_features: int,
    n_components: int,
    seed: int,
) -> None:
    """Property 4: Cumulative PCA explained variance is monotonically non-decreasing.

    For any data matrix with more than one sample, the cumulative explained
    variance ratio from PCA should be monotonically non-decreasing, with each
    value in [0, 1], and the final value should equal 1.0 (within
    floating-point tolerance) when all components are retained.

    **Validates: Requirements 5.2**
    """
    rng = np.random.default_rng(seed)
    data_matrix = rng.standard_normal((n_samples, n_features))

    # Cap n_components to what PCA can actually produce
    max_feasible = min(n_samples, n_features)
    actual_n_components = min(n_components, max_feasible)

    explained_variance_ratio, cumulative_variance = _compute_cumulative_variance(
        data_matrix=data_matrix,
        n_components=actual_n_components,
    )

    # Each individual variance ratio must be in [0, 1]
    assert np.all(explained_variance_ratio >= 0.0), (
        f"Negative explained variance ratio found: {explained_variance_ratio}"
    )
    assert np.all(explained_variance_ratio <= 1.0 + 1e-9), (
        f"Explained variance ratio > 1 found: {explained_variance_ratio}"
    )

    # Cumulative variance must be monotonically non-decreasing
    diffs = np.diff(cumulative_variance)
    assert np.all(diffs >= -1e-10), (
        f"Cumulative variance is not monotonically non-decreasing: diffs={diffs}"
    )

    # Each cumulative value must be in [0, 1]
    assert np.all(cumulative_variance >= 0.0), (
        f"Negative cumulative variance found: {cumulative_variance}"
    )
    assert np.all(cumulative_variance <= 1.0 + 1e-9), (
        f"Cumulative variance > 1 found: {cumulative_variance}"
    )

    # When all components are retained, cumulative variance should reach 1.0
    if actual_n_components == max_feasible:
        assert abs(cumulative_variance[-1] - 1.0) < 1e-6, (
            f"Final cumulative variance should be ~1.0 when all components "
            f"are retained, got {cumulative_variance[-1]}"
        )


# ---------------------------------------------------------------------------
# Unit tests: specific examples and edge cases
# ---------------------------------------------------------------------------


def test_cumulative_variance_single_component() -> None:
    """Single PCA component: cumulative variance equals individual variance."""
    rng = np.random.default_rng(0)
    data_matrix = rng.standard_normal((50, 10))
    explained, cumulative = _compute_cumulative_variance(data_matrix, n_components=1)
    assert len(explained) == 1
    assert len(cumulative) == 1
    np.testing.assert_allclose(explained[0], cumulative[0], rtol=1e-10)


def test_cumulative_variance_all_components_sums_to_one() -> None:
    """Retaining all components: cumulative variance reaches 1.0."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 100, 15
    data_matrix = rng.standard_normal((n_samples, n_features))
    max_components = min(n_samples, n_features)
    _, cumulative = _compute_cumulative_variance(data_matrix, n_components=max_components)
    np.testing.assert_allclose(cumulative[-1], 1.0, atol=1e-6)


def test_cumulative_variance_strictly_increasing_for_nondegenerate_data() -> None:
    """For non-degenerate data, cumulative variance is strictly increasing."""
    rng = np.random.default_rng(7)
    data_matrix = rng.standard_normal((80, 20))
    _, cumulative = _compute_cumulative_variance(data_matrix, n_components=10)
    diffs = np.diff(cumulative)
    assert np.all(diffs > 0), f"Expected strictly increasing cumulative variance, got diffs={diffs}"


def test_cumulative_variance_values_bounded() -> None:
    """All cumulative variance values are in [0, 1]."""
    rng = np.random.default_rng(99)
    data_matrix = rng.standard_normal((60, 30))
    _, cumulative = _compute_cumulative_variance(data_matrix, n_components=15)
    assert np.all(cumulative >= 0.0)
    assert np.all(cumulative <= 1.0 + 1e-9)


def test_cumulative_variance_caps_to_feasible_components() -> None:
    """Requesting more components than feasible is handled gracefully."""
    rng = np.random.default_rng(5)
    n_samples, n_features = 10, 5
    data_matrix = rng.standard_normal((n_samples, n_features))
    # Request more components than min(n_samples, n_features)
    explained, cumulative = _compute_cumulative_variance(
        data_matrix, n_components=100
    )
    max_feasible = min(n_samples, n_features)
    assert len(explained) == max_feasible
    np.testing.assert_allclose(cumulative[-1], 1.0, atol=1e-6)
