"""Shared test fixtures for the test suite."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_eeg_data():
    """Create a small sample of synthetic EEG data for testing."""
    # Create 10 seconds of synthetic EEG data at 100 Hz
    n_samples = 1000
    n_channels = 4
    sampling_rate = 100

    # Generate some synthetic data with different frequencies
    t = np.linspace(0, 10, n_samples)
    data = np.zeros((n_channels, n_samples))

    # Add some alpha (10 Hz) and beta (20 Hz) activity
    for ch in range(n_channels):
        data[ch] = (
            np.sin(2 * np.pi * 10 * t) * 0.5  # alpha
            + np.sin(2 * np.pi * 20 * t) * 0.3  # beta
            + np.random.randn(n_samples) * 0.1  # noise
        )

    return {
        "data": data,
        "sfreq": sampling_rate,
        "ch_names": [f"EEG{i + 1}" for i in range(n_channels)],
    }
