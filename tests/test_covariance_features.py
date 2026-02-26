"""Property-based tests for covariance feature extraction.

Feature: eeg-luminance-validation
Property 2: Covariance feature extraction shape and content

Validates: Requirements 4.3, 4.4
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from campeones_analysis.luminance.features import compute_epoch_covariance

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_float_elements = st.floats(
    min_value=-1e4,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def pca_epoch_array(draw: st.DrawFn) -> np.ndarray:
    """Draw a valid PCA epoch matrix (n_samples >= 2, n_components >= 1)."""
    n_samples = draw(st.integers(min_value=2, max_value=100))
    n_components = draw(st.integers(min_value=1, max_value=50))
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_components),
            elements=_float_elements,
        )
    )
    return data


# ---------------------------------------------------------------------------
# Property 2: Covariance feature extraction shape and content
# ---------------------------------------------------------------------------


@given(pca_epoch=pca_epoch_array())
@settings(max_examples=300)
def test_covariance_output_shape(pca_epoch: np.ndarray) -> None:
    """Property 2a: Output length equals n_components * (n_components + 1) / 2.

    For any PCA epoch matrix of shape (n_samples, n_components) where
    n_samples >= 2 and n_components >= 1, compute_epoch_covariance should
    return a 1-D vector of the correct length.

    Validates: Requirements 4.3, 4.4
    """
    n_components = pca_epoch.shape[1]
    expected_length = n_components * (n_components + 1) // 2

    result = compute_epoch_covariance(pca_epoch)

    assert result.ndim == 1, f"Expected 1-D output, got shape {result.shape}"
    assert len(result) == expected_length, (
        f"n_components={n_components}: expected length {expected_length}, "
        f"got {len(result)}"
    )


@given(pca_epoch=pca_epoch_array())
@settings(max_examples=300)
def test_covariance_values_match_upper_triangle(pca_epoch: np.ndarray) -> None:
    """Property 2b: Values match the upper triangle of np.cov(epoch.T).

    The returned vector should contain exactly the upper triangle (including
    diagonal) of the full covariance matrix, in row-major order.

    Validates: Requirements 4.3, 4.4
    """
    result = compute_epoch_covariance(pca_epoch)

    cov_matrix = np.atleast_2d(np.cov(pca_epoch.T))
    upper_indices = np.triu_indices(cov_matrix.shape[0])
    expected = cov_matrix[upper_indices]

    assert np.allclose(result, expected, rtol=1e-10, atol=1e-12), (
        "Covariance feature values do not match upper triangle of np.cov(epoch.T)"
    )


# ---------------------------------------------------------------------------
# Unit tests – specific examples
# ---------------------------------------------------------------------------


def test_covariance_single_component() -> None:
    """Single PCA component → output length 1 (variance only)."""
    epoch = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = compute_epoch_covariance(epoch)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.var(epoch[:, 0], ddof=1))


def test_covariance_two_components_length() -> None:
    """Two PCA components → output length 3 (var1, cov12, var2)."""
    epoch = np.random.default_rng(42).standard_normal((20, 2))
    result = compute_epoch_covariance(epoch)
    assert result.shape == (3,)


def test_covariance_diagonal_equals_variance() -> None:
    """Diagonal elements of covariance matrix equal per-component variance."""
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((50, 5))
    result = compute_epoch_covariance(epoch)
    cov_matrix = np.cov(epoch.T)
    # Diagonal elements are at positions 0, 2, 5, 9, 14 for n=5
    diag_indices = [i * (i + 1) // 2 + i for i in range(5)]
    # Simpler: just check via triu_indices
    upper_idx = np.triu_indices(5)
    expected = cov_matrix[upper_idx]
    assert np.allclose(result, expected)


def test_covariance_fifty_components_length() -> None:
    """50 PCA components → output length 1275."""
    epoch = np.random.default_rng(7).standard_normal((100, 50))
    result = compute_epoch_covariance(epoch)
    assert result.shape == (1275,)
