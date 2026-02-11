"""Spectral feature extraction and Time Delay Embedding for EEG epochs.

Pure functions for extracting band-power features via Welch's method and
applying Time Delay Embedding (TDE) to sequential feature matrices.

Requirements: 4.1, 4.3, 5.1, 5.2
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def extract_bandpower(
    eeg_epoch: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Extract spectral band-power for each channel of an EEG epoch.

    Uses Welch's method (``scipy.signal.welch``) to estimate the power
    spectral density, then integrates power within each frequency band
    using the trapezoidal rule.

    Args:
        eeg_epoch: 2-D array of shape ``(n_channels, n_samples)``
            containing one EEG epoch.
        sfreq: Sampling frequency in Hz.
        bands: Mapping of band name to ``(freq_min, freq_max)`` in Hz.
            Example: ``{"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}``.

    Returns:
        1-D array of shape ``(n_channels * n_bands,)`` with the absolute
        band-power for each channel–band combination.  The ordering is
        channel-major: all bands for channel 0, then all bands for
        channel 1, etc.
    """
    n_channels = eeg_epoch.shape[0]
    n_bands = len(bands)
    band_names = list(bands.keys())

    bandpower_values = np.empty(n_channels * n_bands, dtype=np.float64)

    for ch_idx in range(n_channels):
        freqs, psd = welch(eeg_epoch[ch_idx], fs=sfreq, nperseg=eeg_epoch.shape[1])

        for band_idx, band_name in enumerate(band_names):
            freq_min, freq_max = bands[band_name]
            band_mask = (freqs >= freq_min) & (freqs <= freq_max)

            if np.any(band_mask):
                bandpower_values[ch_idx * n_bands + band_idx] = np.trapz(
                    psd[band_mask], freqs[band_mask]
                )
            else:
                bandpower_values[ch_idx * n_bands + band_idx] = 0.0

    return bandpower_values



def apply_time_delay_embedding(
    feature_matrix: np.ndarray,
    window_half: int,
) -> np.ndarray:
    """Apply Time Delay Embedding to a sequential feature matrix.

    For each sample *i* in the valid range, concatenates the feature vectors
    from samples ``[i - window_half, ..., i, ..., i + window_half]`` into a
    single expanded vector.  Samples at the borders (where the full window
    does not fit) are discarded.

    Args:
        feature_matrix: 2-D array of shape ``(n_epochs, n_features)``
            containing sequential feature vectors.
        window_half: Half-width of the embedding window (±window_half
            time-points).  The full window size is ``2 * window_half + 1``.

    Returns:
        2-D array of shape
        ``(n_epochs - 2 * window_half, n_features * (2 * window_half + 1))``
        with the time-delay-expanded features.

    Raises:
        ValueError: If *feature_matrix* has fewer rows than
            ``2 * window_half + 1``.
    """
    n_epochs, n_features = feature_matrix.shape
    window_size = 2 * window_half + 1

    if n_epochs < window_size:
        raise ValueError(
            f"feature_matrix has {n_epochs} rows but needs at least "
            f"{window_size} for window_half={window_half}."
        )

    n_valid = n_epochs - 2 * window_half
    embedded = np.empty(
        (n_valid, n_features * window_size), dtype=feature_matrix.dtype
    )

    for idx in range(n_valid):
        embedded[idx] = feature_matrix[idx : idx + window_size].ravel()

    return embedded
