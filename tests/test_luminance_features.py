"""Property-based tests for spectral feature extraction and TDE.

Tests correctness properties of band-power extraction, Time Delay Embedding,
and PCA dimensionality reduction using Hypothesis.

Feature: eeg-luminance-prediction
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sklearn.decomposition import PCA

from campeones_analysis.luminance.features import (
    apply_time_delay_embedding,
    extract_bandpower,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

BAND_CONFIGS = [
    {"delta": (1.0, 4.0)},
    {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
    {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    },
]


@st.composite
def bandpower_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for extract_bandpower.

    Produces a 2-D EEG epoch array with realistic dimensions and a
    sampling frequency high enough to resolve the chosen bands.
    """
    n_channels = draw(st.integers(min_value=1, max_value=8))
    sfreq = draw(st.sampled_from([250.0, 256.0, 500.0]))
    # Need enough samples for Welch to produce meaningful PSD
    n_samples = draw(st.integers(min_value=64, max_value=512))
    bands = draw(st.sampled_from(BAND_CONFIGS))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    eeg_epoch = rng.standard_normal((n_channels, n_samples))

    return {
        "eeg_epoch": eeg_epoch,
        "sfreq": sfreq,
        "bands": bands,
        "n_channels": n_channels,
        "n_bands": len(bands),
    }


@st.composite
def tde_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for apply_time_delay_embedding.

    Ensures the feature matrix has enough rows for the chosen window_half.
    """
    window_half = draw(st.integers(min_value=1, max_value=10))
    n_features = draw(st.integers(min_value=1, max_value=20))
    min_rows = 2 * window_half + 1
    n_epochs = draw(st.integers(min_value=min_rows, max_value=min_rows + 50))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    feature_matrix = rng.standard_normal((n_epochs, n_features))

    return {
        "feature_matrix": feature_matrix,
        "window_half": window_half,
        "n_epochs": n_epochs,
        "n_features": n_features,
    }


@st.composite
def pca_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for PCA dimensionality reduction testing.

    Produces a feature matrix and a target number of PCA components.
    """
    n_samples = draw(st.integers(min_value=10, max_value=100))
    n_features = draw(st.integers(min_value=5, max_value=80))
    n_components = draw(st.integers(min_value=1, max_value=50))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    feature_matrix = rng.standard_normal((n_samples, n_features))

    return {
        "feature_matrix": feature_matrix,
        "n_components": n_components,
        "n_samples": n_samples,
        "n_features": n_features,
    }


# ---------------------------------------------------------------------------
# Property 1: Spectral resolution matches epoch length
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@given(data=bandpower_inputs())
@settings(max_examples=100, deadline=5000)
def test_property1_spectral_resolution_matches_epoch_length(data: dict) -> None:
    """Property 1: Spectral resolution matches epoch length.

    For any EEG epoch of shape (n_channels, n_samples) with n_samples >= 2
    and any sampling frequency, calling extract_bandpower should use
    nperseg = n_samples, resulting in a frequency resolution of
    Δf = sfreq / n_samples. The returned band-power vector should have
    shape (n_channels * n_bands,) with all values >= 0.

    **Validates: Requirements 1.4**
    """
    # Feature: luminance-model-improvements, Property 1: Spectral resolution matches epoch length
    eeg_epoch = data["eeg_epoch"]
    sfreq = data["sfreq"]
    bands = data["bands"]
    n_channels = data["n_channels"]
    n_bands = data["n_bands"]
    n_samples = eeg_epoch.shape[1]

    result = extract_bandpower(
        eeg_epoch=eeg_epoch,
        sfreq=sfreq,
        bands=bands,
    )

    # Shape invariant: output has n_channels * n_bands elements
    expected_dim = n_channels * n_bands
    assert result.shape == (expected_dim,), (
        f"Expected shape ({expected_dim},), got {result.shape}"
    )

    # Non-negativity: spectral power is always >= 0
    assert np.all(result >= 0.0), (
        f"Negative band-power values found: min={result.min():.6e}"
    )

    # Spectral resolution: verify extract_bandpower uses nperseg = n_samples
    # by comparing its output against a direct Welch call with nperseg = n_samples.
    from scipy.signal import welch as welch_reference

    for ch_idx in range(n_channels):
        freqs_expected, psd_expected = welch_reference(
            eeg_epoch[ch_idx], fs=sfreq, nperseg=n_samples
        )

        # Verify frequency resolution matches Δf = sfreq / n_samples
        expected_delta_f = sfreq / n_samples
        if len(freqs_expected) > 1:
            actual_delta_f = freqs_expected[1] - freqs_expected[0]
            np.testing.assert_allclose(
                actual_delta_f,
                expected_delta_f,
                rtol=1e-10,
                err_msg=(
                    f"Frequency resolution mismatch: expected Δf={expected_delta_f}, "
                    f"got {actual_delta_f}"
                ),
            )

        # Verify band-power values match a direct computation with full nperseg
        for band_idx, band_name in enumerate(bands.keys()):
            freq_min, freq_max = bands[band_name]
            band_mask = (freqs_expected >= freq_min) & (freqs_expected <= freq_max)
            if np.any(band_mask):
                expected_power = np.trapz(
                    psd_expected[band_mask], freqs_expected[band_mask]
                )
            else:
                expected_power = 0.0

            np.testing.assert_allclose(
                result[ch_idx * n_bands + band_idx],
                expected_power,
                rtol=1e-10,
                err_msg=(
                    f"Band-power mismatch for ch={ch_idx}, band={band_name}: "
                    f"extract_bandpower uses different nperseg than n_samples"
                ),
            )


# ---------------------------------------------------------------------------
# Property 9: Forma y contenido del Time Delay Embedding
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------


@given(data=tde_inputs())
@settings(max_examples=100)
def test_property9_tde_shape_and_content(data: dict) -> None:
    """Property 9: TDE output shape and content.

    For any feature matrix of shape (N, F) with N > 2*W, the TDE must
    produce shape (N - 2*W, F * (2*W + 1)), and row i of the result
    must equal the concatenation of rows [i, i+1, ..., i+2*W] of the
    original matrix.

    **Validates: Requirements 5.1, 5.2**
    """
    # Feature: eeg-luminance-prediction, Property 9: Forma y contenido del Time Delay Embedding
    feature_matrix = data["feature_matrix"]
    window_half = data["window_half"]
    n_epochs = data["n_epochs"]
    n_features = data["n_features"]
    window_size = 2 * window_half + 1

    result = apply_time_delay_embedding(feature_matrix, window_half)

    expected_rows = n_epochs - 2 * window_half
    expected_cols = n_features * window_size
    assert result.shape == (expected_rows, expected_cols), (
        f"Expected shape ({expected_rows}, {expected_cols}), got {result.shape}"
    )

    # Verify content: row i should be concat of original rows [i..i+window_size)
    for idx in range(expected_rows):
        expected_row = feature_matrix[idx : idx + window_size].ravel()
        np.testing.assert_array_equal(
            result[idx],
            expected_row,
            err_msg=f"TDE row {idx} content mismatch",
        )


# ---------------------------------------------------------------------------
# Property 10: Reducción de dimensionalidad por PCA
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


@given(data=pca_inputs())
@settings(max_examples=100, deadline=5000)
def test_property10_pca_dimensionality_reduction(data: dict) -> None:
    """Property 10: PCA dimensionality reduction.

    For any feature matrix of shape (N, F_expanded) and a configured
    number of components C, the PCA output must have shape
    (N, min(C, F_expanded, N)), guaranteeing dimensionality is reduced
    or maintained.

    **Validates: Requirements 5.3**
    """
    # Feature: eeg-luminance-prediction, Property 10: Reducción de dimensionalidad por PCA
    feature_matrix = data["feature_matrix"]
    n_components = data["n_components"]
    n_samples = data["n_samples"]
    n_features = data["n_features"]

    effective_components = min(n_components, n_features, n_samples)
    pca = PCA(n_components=effective_components)
    result = pca.fit_transform(feature_matrix)

    assert result.shape == (n_samples, effective_components), (
        f"Expected shape ({n_samples}, {effective_components}), "
        f"got {result.shape}"
    )

    # Output dimensionality must be <= input dimensionality
    assert result.shape[1] <= n_features, (
        f"PCA output dim ({result.shape[1]}) > input dim ({n_features})"
    )
