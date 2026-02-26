"""Property-based tests for cross-correlation analysis.

Feature: eeg-luminance-validation
Property 9: Cross-correlation bounds and lag detection
Validates: Requirements 11.1, 11.2
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from scipy.signal import correlate


# ---------------------------------------------------------------------------
# Pure functions inlined for testing (mirrors script 22 logic).
# ---------------------------------------------------------------------------


def compute_normalized_cross_correlation(
    signal_real: np.ndarray,
    signal_reported: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalised cross-correlation between two signals.

    Args:
        signal_real: 1-D array of the physical (stimulus) luminance signal.
        signal_reported: 1-D array of the reported (joystick) luminance signal.
        max_lag_samples: Maximum lag (in samples) to retain on each side.

    Returns:
        Tuple of (lags_array, xcorr_values).
    """
    signal_real = np.asarray(signal_real, dtype=np.float64)
    signal_reported = np.asarray(signal_reported, dtype=np.float64)

    signal_real_zm = signal_real - np.mean(signal_real)
    signal_reported_zm = signal_reported - np.mean(signal_reported)

    full_xcorr = correlate(signal_reported_zm, signal_real_zm, mode="full")

    norm_factor = np.sqrt(
        np.sum(signal_real_zm**2) * np.sum(signal_reported_zm**2)
    )
    if norm_factor == 0.0:
        n_lags = 2 * max_lag_samples + 1
        return np.arange(-max_lag_samples, max_lag_samples + 1), np.zeros(n_lags)

    normalised_xcorr = full_xcorr / norm_factor

    n_samples = len(signal_reported)
    full_lags = np.arange(-(n_samples - 1), n_samples)

    lag_mask = (full_lags >= -max_lag_samples) & (full_lags <= max_lag_samples)
    lags_array = full_lags[lag_mask]
    xcorr_values = normalised_xcorr[lag_mask]

    return lags_array, xcorr_values


def find_optimal_lag(
    lags_array: np.ndarray,
    xcorr_values: np.ndarray,
    sfreq: float,
) -> tuple[float, float]:
    """Find the lag that maximises the cross-correlation.

    Args:
        lags_array: 1-D array of lag values in samples.
        xcorr_values: 1-D array of normalised cross-correlation values.
        sfreq: Sampling frequency (Hz).

    Returns:
        Tuple of (optimal_lag_seconds, max_correlation).
    """
    best_idx = int(np.argmax(xcorr_values))
    optimal_lag_samples = int(lags_array[best_idx])
    optimal_lag_seconds = optimal_lag_samples / sfreq
    max_correlation = float(xcorr_values[best_idx])
    return optimal_lag_seconds, max_correlation


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-constant signal: 10–200 samples, values in reasonable luminance range
_signal_strategy = st.lists(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    min_size=10,
    max_size=200,
).map(np.array)


# ---------------------------------------------------------------------------
# Property 9a: Cross-correlation values bounded in [-1, 1]
# Validates: Requirements 11.1
# ---------------------------------------------------------------------------


@given(
    signal_real=_signal_strategy,
    signal_reported=_signal_strategy,
)
@settings(max_examples=200)
def test_cross_correlation_bounds(
    signal_real: np.ndarray,
    signal_reported: np.ndarray,
) -> None:
    """Property 9a: Normalised cross-correlation values lie in [-1, 1].

    For any two non-zero signals of equal length, the normalised
    cross-correlation values should be bounded in [-1, 1].

    Validates: Requirements 11.1
    """
    # Make signals equal length (use the shorter)
    min_len = min(len(signal_real), len(signal_reported))
    signal_real = signal_real[:min_len]
    signal_reported = signal_reported[:min_len]

    # Skip constant signals (norm_factor would be 0)
    assume(np.std(signal_real) > 1e-10)
    assume(np.std(signal_reported) > 1e-10)

    max_lag_samples = min_len - 1

    lags_array, xcorr_values = compute_normalized_cross_correlation(
        signal_real, signal_reported, max_lag_samples
    )

    # All values must be in [-1, 1] (with small tolerance for float precision)
    assert np.all(xcorr_values >= -1.0 - 1e-9), (
        f"Cross-correlation below -1: min={np.min(xcorr_values)}"
    )
    assert np.all(xcorr_values <= 1.0 + 1e-9), (
        f"Cross-correlation above 1: max={np.max(xcorr_values)}"
    )

    # Lags array should be symmetric around 0
    assert lags_array[0] == -max_lag_samples
    assert lags_array[-1] == max_lag_samples


# ---------------------------------------------------------------------------
# Property 9b: Lag detection for known shifted signal
# Validates: Requirements 11.2
# ---------------------------------------------------------------------------


@given(
    base_signal=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=150,
    ).map(np.array),
    shift_samples=st.integers(min_value=1, max_value=10),
    sfreq=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_lag_detection_for_known_shift(
    base_signal: np.ndarray,
    shift_samples: int,
    sfreq: float,
) -> None:
    """Property 9b: Detected lag equals known shift for a shifted copy.

    For any signal and a copy shifted by k samples, the detected optimal
    lag should equal k.

    Validates: Requirements 11.2
    """
    # Skip constant signals
    assume(np.std(base_signal) > 1e-10)
    assume(shift_samples < len(base_signal) // 2)

    # Create shifted copy: reported = delayed version of real
    # If real is [a, b, c, d, e] and shift=2, reported is [0, 0, a, b, c]
    # The reported signal lags behind the real signal by shift_samples
    signal_real = base_signal.copy()
    signal_reported = np.zeros_like(base_signal)
    signal_reported[shift_samples:] = base_signal[:-shift_samples]

    # Ensure reported is not constant after shifting
    assume(np.std(signal_reported) > 1e-10)

    max_lag_samples = len(base_signal) - 1

    lags_array, xcorr_values = compute_normalized_cross_correlation(
        signal_real, signal_reported, max_lag_samples
    )

    optimal_lag_s, max_corr = find_optimal_lag(lags_array, xcorr_values, sfreq)
    detected_lag_samples = int(round(optimal_lag_s * sfreq))

    # The detected lag should equal the known shift
    assert detected_lag_samples == shift_samples, (
        f"Expected lag={shift_samples} samples, got {detected_lag_samples} "
        f"(optimal_lag_s={optimal_lag_s:.4f}, sfreq={sfreq:.1f})"
    )
