"""Property-based tests for the raw TDE model pipeline.

Tests correctness properties of Time Delay Embedding applied on continuous
raw EEG signals, validating shape invariants and border discard behaviour.

Feature: raw-tde-and-exploratory-analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# The function under test lives in scripts/modeling/, so we add it to
# sys.path to allow a direct import without installing as a package.
_SCRIPTS_MODELING_DIR = str(
    Path(__file__).resolve().parent.parent / "scripts" / "modeling"
)
if _SCRIPTS_MODELING_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_MODELING_DIR)

_SCRIPTS_EXPLORATION_DIR = str(
    Path(__file__).resolve().parent.parent / "scripts" / "exploration"
)
if _SCRIPTS_EXPLORATION_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_EXPLORATION_DIR)

from importlib import import_module

_script13 = import_module("13_luminance_raw_tde_model")
apply_tde_on_continuous_signal = _script13.apply_tde_on_continuous_signal


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

MIN_WINDOW_HALF = 1
MAX_WINDOW_HALF = 20
MIN_CHANNELS = 1
MAX_CHANNELS = 15
MIN_SAMPLES = 50
MAX_SAMPLES = 2000


@st.composite
def tde_continuous_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for apply_tde_on_continuous_signal.

    Produces a random EEG matrix of shape ``(n_channels, n_samples)`` and a
    ``window_half`` value, ensuring ``n_samples >= 2 * window_half + 1`` so
    that TDE can produce at least one valid output row.
    """
    n_channels = draw(st.integers(min_value=MIN_CHANNELS, max_value=MAX_CHANNELS))
    window_half = draw(st.integers(min_value=MIN_WINDOW_HALF, max_value=MAX_WINDOW_HALF))

    min_samples = 2 * window_half + 1
    clamped_min = max(min_samples, MIN_SAMPLES)
    n_samples = draw(st.integers(min_value=clamped_min, max_value=MAX_SAMPLES))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    eeg_data = rng.standard_normal((n_channels, n_samples))

    return {
        "eeg_data": eeg_data,
        "window_half": window_half,
        "n_channels": n_channels,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Property 1: TDE output shape invariant
# Validates: Requirements 1.2, 1.3
# ---------------------------------------------------------------------------


@given(data=tde_continuous_inputs())
@settings(max_examples=100, deadline=5000)
def test_property1_tde_output_shape_invariant(data: dict) -> None:
    """Property 1: TDE output shape invariant.

    For any EEG matrix of shape ``(n_channels, n_samples)`` where
    ``n_samples >= 2 * window_half + 1``, and any ``window_half >= 1``,
    applying TDE on the transposed matrix should produce an output of shape
    ``(n_samples - 2 * window_half, n_channels * (2 * window_half + 1))``.

    This validates both the correct temporal expansion (each valid
    time-point is concatenated with its ±window_half neighbours across all
    channels) and the border discard (time-points lacking full context are
    removed).

    **Validates: Requirements 1.2, 1.3**
    """
    # Feature: raw-tde-and-exploratory-analysis, Property 1: TDE output shape invariant
    eeg_data = data["eeg_data"]
    window_half = data["window_half"]
    n_channels = data["n_channels"]
    n_samples = data["n_samples"]

    result = apply_tde_on_continuous_signal(eeg_data, window_half)

    expected_rows = n_samples - 2 * window_half
    window_size = 2 * window_half + 1
    expected_cols = n_channels * window_size

    assert result.shape == (expected_rows, expected_cols), (
        f"Expected shape ({expected_rows}, {expected_cols}), got {result.shape}. "
        f"n_channels={n_channels}, n_samples={n_samples}, window_half={window_half}"
    )

# ---------------------------------------------------------------------------
# Strategies for Property 2
# ---------------------------------------------------------------------------

MIN_TIMEPOINTS = 20
MAX_TIMEPOINTS = 200
MIN_FEATURES = 10
MAX_FEATURES = 100


@st.composite
def pca_tde_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for _apply_pca_to_tde_matrix.

    Produces a random TDE-expanded matrix of shape
    ``(n_timepoints, n_features)`` and a ``n_components`` value constrained
    to ``[1, min(n_timepoints, n_features)]`` so that PCA can always fit.
    """
    n_timepoints = draw(
        st.integers(min_value=MIN_TIMEPOINTS, max_value=MAX_TIMEPOINTS)
    )
    n_features = draw(
        st.integers(min_value=MIN_FEATURES, max_value=MAX_FEATURES)
    )
    max_components = min(n_timepoints, n_features)
    n_components = draw(st.integers(min_value=1, max_value=max_components))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    tde_matrix = rng.standard_normal((n_timepoints, n_features))

    return {
        "tde_matrix": tde_matrix,
        "n_components": n_components,
        "n_timepoints": n_timepoints,
        "n_features": n_features,
        "random_seed": seed,
    }


import logging

import mne
import pandas as pd

_apply_pca_to_tde_matrix = _script13._apply_pca_to_tde_matrix
_epoch_pca_timeseries = _script13._epoch_pca_timeseries
extract_raw_tde_epochs_for_run = _script13.extract_raw_tde_epochs_for_run


# ---------------------------------------------------------------------------
# Property 2: PCA preserves time-points and reduces features
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@given(data=pca_tde_inputs())
@settings(max_examples=100, deadline=10000)
def test_property2_pca_preserves_timepoints(data: dict) -> None:
    """Property 2: PCA preserves time-points and reduces features.

    For any TDE-expanded matrix of shape ``(n_timepoints, n_features)`` and
    any ``n_components <= min(n_timepoints, n_features)``, applying PCA
    should produce a matrix of shape ``(n_timepoints, n_components)``.

    This validates that PCA preserves the number of time-points (rows)
    while reducing the feature dimension (columns) to the requested number
    of principal components.

    **Validates: Requirements 1.4**
    """
    # Feature: raw-tde-and-exploratory-analysis, Property 2: PCA preserves time-points and reduces features
    tde_matrix = data["tde_matrix"]
    n_components = data["n_components"]
    n_timepoints = data["n_timepoints"]
    random_seed = data["random_seed"]

    result = _apply_pca_to_tde_matrix(tde_matrix, n_components, random_seed)

    assert result.shape == (n_timepoints, n_components), (
        f"Expected shape ({n_timepoints}, {n_components}), got {result.shape}. "
        f"tde_matrix.shape={tde_matrix.shape}"
    )


# ---------------------------------------------------------------------------
# Strategies for Property 3
# ---------------------------------------------------------------------------

FIXED_SFREQ = 500.0
FIXED_EPOCH_DURATION_S = 1.0
FIXED_EPOCH_STEP_S = 0.1
MIN_COMPONENTS_P3 = 2
MAX_COMPONENTS_P3 = 20


@st.composite
def epoch_count_inputs(draw: st.DrawFn) -> dict:
    """Generate valid inputs for _epoch_pca_timeseries epoch-count property.

    Uses fixed sfreq (500 Hz), epoch_duration_s (1.0 s), and epoch_step_s
    (0.1 s) matching the project's EEG configuration.  The number of valid
    samples is drawn so that at least one full epoch fits (>= sfreq *
    epoch_duration_s).  n_components is drawn from a small range because it
    does not affect the epoch count.
    """
    min_samples_for_one_epoch = int(FIXED_SFREQ * FIXED_EPOCH_DURATION_S)
    n_valid_samples = draw(
        st.integers(min_value=min_samples_for_one_epoch, max_value=5000)
    )
    n_components = draw(
        st.integers(min_value=MIN_COMPONENTS_P3, max_value=MAX_COMPONENTS_P3)
    )

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    pca_timeseries = rng.standard_normal((n_valid_samples, n_components))

    return {
        "pca_timeseries": pca_timeseries,
        "n_valid_samples": n_valid_samples,
        "n_components": n_components,
        "sfreq": FIXED_SFREQ,
        "epoch_duration_s": FIXED_EPOCH_DURATION_S,
        "epoch_step_s": FIXED_EPOCH_STEP_S,
    }


# ---------------------------------------------------------------------------
# Property 3: Epoch count from PCA time-series
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


@given(data=epoch_count_inputs())
@settings(max_examples=100, deadline=5000)
def test_property3_epoch_count_from_pca_timeseries(data: dict) -> None:
    """Property 3: Epoch count from PCA time-series.

    For any PCA time-series of length ``n_valid_samples`` at sampling
    frequency ``sfreq``, with epoch duration ``epoch_duration_s`` and step
    ``epoch_step_s``, the number of epochs should equal
    ``floor((n_valid_samples / sfreq - epoch_duration_s) / epoch_step_s) + 1``
    (when the segment is long enough for at least one epoch).

    **Validates: Requirements 1.5**
    """
    # Feature: raw-tde-and-exploratory-analysis, Property 3: Epoch count from PCA time-series
    pca_timeseries = data["pca_timeseries"]
    n_valid_samples = data["n_valid_samples"]
    sfreq = data["sfreq"]
    epoch_duration_s = data["epoch_duration_s"]
    epoch_step_s = data["epoch_step_s"]

    features, onsets = _epoch_pca_timeseries(
        pca_timeseries, sfreq, epoch_duration_s, epoch_step_s
    )

    total_duration_s = n_valid_samples / sfreq
    last_valid_onset = total_duration_s - epoch_duration_s
    # Mirror the exact logic from create_epoch_onsets:
    # np.arange(0.0, last_valid_onset + epoch_step_s / 2.0, epoch_step_s)
    # then clip to <= last_valid_onset + 1e-12
    onsets = np.arange(0.0, last_valid_onset + epoch_step_s / 2.0, epoch_step_s)
    onsets = onsets[onsets <= last_valid_onset + 1e-12]
    expected_epoch_count = len(onsets)

    actual_epoch_count = features.shape[0]

    assert actual_epoch_count == expected_epoch_count, (
        f"Expected {expected_epoch_count} epochs, got {actual_epoch_count}. "
        f"n_valid_samples={n_valid_samples}, sfreq={sfreq}, "
        f"epoch_duration_s={epoch_duration_s}, epoch_step_s={epoch_step_s}, "
        f"total_duration_s={total_duration_s:.4f}"
    )
    assert len(onsets) == actual_epoch_count, (
        f"Onsets length ({len(onsets)}) does not match feature rows "
        f"({actual_epoch_count})."
    )


# ---------------------------------------------------------------------------
# Unit tests for _apply_pca_to_tde_matrix
# ---------------------------------------------------------------------------


class TestApplyPcaToTdeMatrix:
    """Unit tests for PCA reduction of TDE-expanded matrices."""

    def test_reduces_features_to_requested_components(self) -> None:
        """PCA should reduce columns to n_components while preserving rows."""
        rng = np.random.default_rng(42)
        tde_matrix = rng.standard_normal((200, 231))
        result = _apply_pca_to_tde_matrix(tde_matrix, n_components=100, random_seed=42)
        assert result.shape == (200, 100)

    def test_clamps_components_when_fewer_rows(self) -> None:
        """When n_rows < n_components, PCA should clamp to min(n_rows, n_cols)."""
        rng = np.random.default_rng(42)
        tde_matrix = rng.standard_normal((30, 231))
        result = _apply_pca_to_tde_matrix(tde_matrix, n_components=100, random_seed=42)
        assert result.shape == (30, 30)

    def test_clamps_components_when_fewer_cols(self) -> None:
        """When n_cols < n_components, PCA should clamp to min(n_rows, n_cols)."""
        rng = np.random.default_rng(42)
        tde_matrix = rng.standard_normal((200, 50))
        result = _apply_pca_to_tde_matrix(tde_matrix, n_components=100, random_seed=42)
        assert result.shape == (200, 50)


# ---------------------------------------------------------------------------
# Unit tests for _epoch_pca_timeseries
# ---------------------------------------------------------------------------


class TestEpochPcaTimeseries:
    """Unit tests for epoching PCA time-series into fixed-length windows."""

    def test_correct_epoch_count_and_feature_size(self) -> None:
        """Verify epoch count and feature vector size for a known input."""
        sfreq = 500.0
        epoch_duration_s = 1.0
        epoch_step_s = 0.1
        n_components = 10
        # 3.0 s of data → (3.0 - 1.0) / 0.1 + 1 = 21 epochs
        n_timepoints = int(3.0 * sfreq)
        rng = np.random.default_rng(42)
        pca_ts = rng.standard_normal((n_timepoints, n_components))

        features, onsets = _epoch_pca_timeseries(
            pca_ts, sfreq, epoch_duration_s, epoch_step_s
        )

        expected_epochs = 21
        # mean + variance per component → 2 × n_components
        expected_feature_dim = 2 * n_components
        assert features.shape == (expected_epochs, expected_feature_dim)
        assert len(onsets) == expected_epochs

    def test_returns_empty_when_segment_too_short(self) -> None:
        """Segment shorter than one epoch should return empty arrays."""
        sfreq = 500.0
        n_timepoints = int(0.5 * sfreq)  # 0.5 s < 1.0 s epoch
        rng = np.random.default_rng(42)
        pca_ts = rng.standard_normal((n_timepoints, 10))

        features, onsets = _epoch_pca_timeseries(pca_ts, sfreq, 1.0, 0.1)

        assert features.shape[0] == 0

    def test_exactly_one_epoch(self) -> None:
        """Segment exactly 1.0 s should produce exactly 1 epoch."""
        sfreq = 500.0
        n_timepoints = int(1.0 * sfreq)
        rng = np.random.default_rng(42)
        pca_ts = rng.standard_normal((n_timepoints, 5))

        features, onsets = _epoch_pca_timeseries(pca_ts, sfreq, 1.0, 0.1)

        assert features.shape[0] == 1
        assert features.shape[1] == 2 * 5  # mean + var for 5 components


# ---------------------------------------------------------------------------
# Unit tests for extract_raw_tde_epochs_for_run
# ---------------------------------------------------------------------------


def _make_synthetic_raw(
    n_channels: int,
    channel_names: list[str],
    sfreq: float,
    duration_s: float,
) -> mne.io.RawArray:
    """Create a synthetic MNE RawArray for testing.

    Args:
        n_channels: Number of EEG channels.
        channel_names: Channel name list.
        sfreq: Sampling frequency in Hz.
        duration_s: Total duration in seconds.

    Returns:
        MNE RawArray with random data.
    """
    rng = np.random.default_rng(42)
    n_samples = int(duration_s * sfreq)
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _make_luminance_csv(tmp_path: Path, video_id: int, duration_s: float) -> Path:
    """Create a minimal luminance CSV for testing.

    Args:
        tmp_path: Temporary directory path.
        video_id: Video identifier.
        duration_s: Duration of the luminance recording in seconds.

    Returns:
        Path to the created CSV file.
    """
    timestamps = np.linspace(0, duration_s, num=100)
    luminance_values = np.sin(timestamps) * 50 + 128
    luminance_df = pd.DataFrame(
        {"timestamp": timestamps, "luminance": luminance_values}
    )
    csv_path = tmp_path / f"green_intensity_video_{video_id}.csv"
    luminance_df.to_csv(csv_path, index=False)
    return csv_path


class TestExtractRawTdeEpochsForRun:
    """Unit tests for the full raw-TDE epoch extraction pipeline."""

    def test_returns_empty_when_no_luminance_events(self) -> None:
        """No video_luminance events should return an empty list."""
        channels = ["O1", "O2", "Pz"]
        raw = _make_synthetic_raw(3, channels, 500.0, 10.0)
        events_df = pd.DataFrame(
            {"trial_type": ["fixation"], "stim_id": [0], "onset": [0.0], "duration": [5.0]}
        )
        run_config = {"id": "002", "acq": "a", "task": "01", "block": "block1"}

        result = extract_raw_tde_epochs_for_run(run_config, raw, events_df, channels)

        assert result == []

    def test_produces_epochs_with_correct_keys(self, tmp_path: Path, monkeypatch) -> None:
        """Each epoch entry should have all required keys with correct types."""
        sfreq = 500.0
        duration_s = 60.0
        channels = ["O1", "O2", "Pz"]
        raw = _make_synthetic_raw(3, channels, sfreq, duration_s)

        video_id = 3
        _make_luminance_csv(tmp_path, video_id, 30.0)

        # Monkeypatch the config values used inside the function
        monkeypatch.setattr(_script13, "LUMINANCE_CSV_MAP", {video_id: f"green_intensity_video_{video_id}.csv"})
        monkeypatch.setattr(_script13, "STIMULI_PATH", tmp_path)
        monkeypatch.setattr(_script13, "TDE_WINDOW_HALF", 5)
        monkeypatch.setattr(_script13, "PCA_COMPONENTS", 10)
        monkeypatch.setattr(_script13, "RANDOM_SEED", 42)
        monkeypatch.setattr(_script13, "EPOCH_DURATION_S", 1.0)
        monkeypatch.setattr(_script13, "EPOCH_STEP_S", 0.1)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video_luminance"],
                "stim_id": [100 + video_id],
                "onset": [1.0],
                "duration": [30.0],
            }
        )
        run_config = {"id": "002", "acq": "a", "task": "01", "block": "block1"}

        result = extract_raw_tde_epochs_for_run(run_config, raw, events_df, channels)

        assert len(result) > 0
        required_keys = {"X", "y", "video_id", "video_identifier", "run_id", "acq"}
        for entry in result:
            assert set(entry.keys()) == required_keys
            assert isinstance(entry["X"], np.ndarray)
            assert entry["X"].ndim == 1
            assert isinstance(entry["y"], float)
            assert entry["video_id"] == video_id
            assert entry["video_identifier"] == f"{video_id}_a"
            assert entry["run_id"] == "002"
            assert entry["acq"] == "a"

    def test_skips_segment_too_short_for_tde(self, caplog) -> None:
        """Segment shorter than TDE window should be skipped with a warning."""
        sfreq = 500.0
        channels = ["O1", "O2"]
        # Very short recording: 0.1 s = 50 samples, TDE needs 2*10+1=21
        # but after crop the segment will be tiny
        raw = _make_synthetic_raw(2, channels, sfreq, 5.0)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video_luminance"],
                "stim_id": [103],
                "onset": [0.0],
                "duration": [0.02],  # 10 samples at 500 Hz — too short
            }
        )
        run_config = {"id": "002", "acq": "a", "task": "01", "block": "block1"}

        with caplog.at_level(logging.WARNING):
            result = extract_raw_tde_epochs_for_run(run_config, raw, events_df, channels)

        assert result == []

    def test_feature_vector_dimension(self, tmp_path: Path, monkeypatch) -> None:
        """Feature vector should be PCA_COMPONENTS × n_samples_per_epoch."""
        sfreq = 500.0
        channels = ["O1", "O2", "Pz"]
        raw = _make_synthetic_raw(3, channels, sfreq, 60.0)

        video_id = 7
        _make_luminance_csv(tmp_path, video_id, 30.0)

        n_pca = 10
        monkeypatch.setattr(_script13, "LUMINANCE_CSV_MAP", {video_id: f"green_intensity_video_{video_id}.csv"})
        monkeypatch.setattr(_script13, "STIMULI_PATH", tmp_path)
        monkeypatch.setattr(_script13, "TDE_WINDOW_HALF", 5)
        monkeypatch.setattr(_script13, "PCA_COMPONENTS", n_pca)
        monkeypatch.setattr(_script13, "RANDOM_SEED", 42)
        monkeypatch.setattr(_script13, "EPOCH_DURATION_S", 1.0)
        monkeypatch.setattr(_script13, "EPOCH_STEP_S", 0.1)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video_luminance"],
                "stim_id": [100 + video_id],
                "onset": [1.0],
                "duration": [20.0],
            }
        )
        run_config = {"id": "002", "acq": "a", "task": "01", "block": "block1"}

        result = extract_raw_tde_epochs_for_run(run_config, raw, events_df, channels)

        expected_dim = 2 * n_pca  # mean + var per PCA component
        assert len(result) > 0
        assert result[0]["X"].shape == (expected_dim,)


# ---------------------------------------------------------------------------
# Import script 14 functions for distribution explorer tests
# ---------------------------------------------------------------------------

_script14 = import_module("14_explore_target_distributions")
apply_polarity_correction = _script14.apply_polarity_correction
_compute_epoch_means = _script14._compute_epoch_means
collect_dimension_values = _script14.collect_dimension_values
plot_distribution = _script14.plot_distribution
run_pipeline = _script14.run_pipeline


# ---------------------------------------------------------------------------
# Strategies for Property 4
# ---------------------------------------------------------------------------

MIN_SIGNAL_LENGTH = 1
MAX_SIGNAL_LENGTH = 5000


@st.composite
def joystick_signal_inputs(draw: st.DrawFn) -> dict:
    """Generate random 1-D joystick signal arrays for polarity testing.

    Produces float64 arrays of varying lengths (1–5000) with values drawn
    from a standard normal distribution, simulating realistic joystick
    deflection ranges.
    """
    signal_length = draw(
        st.integers(min_value=MIN_SIGNAL_LENGTH, max_value=MAX_SIGNAL_LENGTH)
    )
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(signal_length)

    return {"signal": signal, "signal_length": signal_length}


# ---------------------------------------------------------------------------
# Property 4: Polarity correction is negation
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------


@given(data=joystick_signal_inputs())
@settings(max_examples=100, deadline=5000)
def test_property4_polarity_correction_is_negation(data: dict) -> None:
    """Property 4: Polarity correction is negation.

    For any joystick signal array and polarity ``"inverse"``, the corrected
    signal should equal the negation of the original.  Applying correction
    with ``"inverse"`` polarity twice should be an identity operation
    (round-trip / involution).

    Sub-property 1 — Negation:
        ``apply_polarity_correction(signal, "inverse") == -signal``

    Sub-property 2 — Round-trip (involution):
        ``apply_polarity_correction(
            apply_polarity_correction(signal, "inverse"), "inverse"
        ) == signal``

    **Validates: Requirements 4.3**
    """
    # Feature: raw-tde-and-exploratory-analysis, Property 4: Polarity correction is negation
    signal = data["signal"]

    # Sub-property 1: inverse polarity equals negation
    corrected = apply_polarity_correction(signal, "inverse")
    np.testing.assert_array_equal(
        corrected,
        -signal,
        err_msg="Inverse polarity correction should negate the signal.",
    )

    # Sub-property 2: applying inverse twice recovers the original (involution)
    round_trip = apply_polarity_correction(corrected, "inverse")
    np.testing.assert_array_equal(
        round_trip,
        signal,
        err_msg="Double inverse correction should be an identity operation.",
    )


# ---------------------------------------------------------------------------
# Unit tests for apply_polarity_correction
# ---------------------------------------------------------------------------


class TestApplyPolarityCorrection:
    """Unit tests for the polarity correction pure function."""

    def test_inverse_negates_signal(self) -> None:
        """Inverse polarity should negate every sample."""
        signal = np.array([1.0, -2.0, 3.0, 0.0])
        result = apply_polarity_correction(signal, "inverse")
        np.testing.assert_array_equal(result, np.array([-1.0, 2.0, -3.0, 0.0]))

    def test_direct_preserves_signal(self) -> None:
        """Direct polarity should leave the signal unchanged."""
        signal = np.array([1.0, -2.0, 3.0])
        result = apply_polarity_correction(signal, "direct")
        np.testing.assert_array_equal(result, signal)

    def test_does_not_mutate_input(self) -> None:
        """The original array must not be modified."""
        signal = np.array([5.0, 10.0])
        original_copy = signal.copy()
        apply_polarity_correction(signal, "inverse")
        np.testing.assert_array_equal(signal, original_copy)

    def test_unknown_polarity_preserves_signal(self) -> None:
        """Any polarity string other than 'inverse' should be a no-op."""
        signal = np.array([1.0, 2.0])
        result = apply_polarity_correction(signal, "normal")
        np.testing.assert_array_equal(result, signal)

    def test_empty_signal(self) -> None:
        """Empty array should return empty array for any polarity."""
        signal = np.array([], dtype=np.float64)
        result = apply_polarity_correction(signal, "inverse")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Unit tests for _compute_epoch_means
# ---------------------------------------------------------------------------


class TestComputeEpochMeans:
    """Unit tests for epoch-level mean computation."""

    def test_known_constant_signal(self) -> None:
        """Constant signal should produce constant epoch means."""
        sfreq = 500.0
        duration_s = 3.0
        signal = np.ones(int(duration_s * sfreq)) * 42.0
        means = _compute_epoch_means(signal, sfreq, 1.0, 0.1)
        assert len(means) > 0
        np.testing.assert_allclose(means, 42.0)

    def test_too_short_signal_returns_empty(self) -> None:
        """Signal shorter than one epoch should return empty."""
        sfreq = 500.0
        signal = np.ones(100)  # 0.2 s < 1.0 s epoch
        means = _compute_epoch_means(signal, sfreq, 1.0, 0.1)
        assert len(means) == 0

    def test_epoch_count_matches_expected(self) -> None:
        """Epoch count should match the formula from create_epoch_onsets."""
        sfreq = 500.0
        duration_s = 5.0
        signal = np.zeros(int(duration_s * sfreq))
        means = _compute_epoch_means(signal, sfreq, 1.0, 0.1)
        # (5.0 - 1.0) / 0.1 + 1 = 41 epochs
        assert len(means) == 41


# ---------------------------------------------------------------------------
# Unit tests for collect_dimension_values
# ---------------------------------------------------------------------------


class TestCollectDimensionValues:
    """Unit tests for the main dimension value collection function."""

    def test_returns_empty_dict_for_empty_runs(self) -> None:
        """No runs should produce an empty dict."""
        result = collect_dimension_values([])
        assert result == {}

    def test_collects_joystick_values_for_dimension(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should extract joystick epoch means for a valence video."""
        sfreq = 500.0
        duration_s = 30.0
        # Create EEG with joystick_x channel
        channel_names = ["O1", "joystick_x"]
        n_samples = int(duration_s * sfreq)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_samples)) * 1e-6
        info = mne.create_info(
            ch_names=channel_names, sfreq=sfreq, ch_types=["eeg", "misc"]
        )
        eeg_raw = mne.io.RawArray(data, info, verbose=False)

        # Save as BrainVision so _resolve_eeg_path can find it
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        vhdr_path = eeg_dir / "test.vhdr"
        mne.export.export_raw(str(vhdr_path), eeg_raw, overwrite=True)

        # Monkeypatch _resolve_eeg_path to return our test file
        monkeypatch.setattr(_script14, "_resolve_eeg_path", lambda rc: vhdr_path)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video"],
                "stim_id": [103],
                "onset": [1.0],
                "duration": [10.0],
            }
        )
        order_matrix_df = pd.DataFrame(
            {
                "dimension": ["valence"],
                "order_emojis_slider": ["direct"],
                "video_id": [3],
            }
        )
        runs_data = [
            {
                "run_config": {
                    "id": "002",
                    "acq": "a",
                    "task": "01",
                    "block": "block1",
                },
                "events_df": events_df,
                "order_matrix_df": order_matrix_df,
                "events_path": tmp_path / "events.tsv",
                "order_matrix_path": tmp_path / "order.xlsx",
            }
        ]

        result = collect_dimension_values(runs_data)

        assert "valence" in result
        assert len(result["valence"]) > 0
        entry = result["valence"][0]
        assert set(entry.keys()) == {
            "value",
            "video_id",
            "video_identifier",
            "dimension",
            "run_id",
            "acq",
        }
        assert entry["dimension"] == "valence"
        assert entry["video_id"] == 3
        assert entry["run_id"] == "002"
        assert entry["acq"] == "a"

    def test_luminance_dimension_extracts_real_luminance(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Luminance videos should produce real_luminance entries from CSV."""
        sfreq = 500.0
        duration_s = 30.0
        channel_names = ["O1", "joystick_x"]
        n_samples = int(duration_s * sfreq)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_samples)) * 1e-6
        info = mne.create_info(
            ch_names=channel_names, sfreq=sfreq, ch_types=["eeg", "misc"]
        )
        eeg_raw = mne.io.RawArray(data, info, verbose=False)

        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        vhdr_path = eeg_dir / "test.vhdr"
        mne.export.export_raw(str(vhdr_path), eeg_raw, overwrite=True)

        monkeypatch.setattr(_script14, "_resolve_eeg_path", lambda rc: vhdr_path)

        # Create luminance CSV
        video_id = 3
        csv_path = _make_luminance_csv(tmp_path, video_id, 10.0)
        monkeypatch.setattr(
            _script14,
            "LUMINANCE_CSV_MAP",
            {video_id: f"green_intensity_video_{video_id}.csv"},
        )
        monkeypatch.setattr(_script14, "STIMULI_PATH", tmp_path)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video_luminance"],
                "stim_id": [103],
                "onset": [1.0],
                "duration": [10.0],
            }
        )
        order_matrix_df = pd.DataFrame(
            {
                "dimension": ["luminance"],
                "order_emojis_slider": ["direct"],
                "video_id": [video_id],
            }
        )
        runs_data = [
            {
                "run_config": {
                    "id": "002",
                    "acq": "a",
                    "task": "01",
                    "block": "block1",
                },
                "events_df": events_df,
                "order_matrix_df": order_matrix_df,
                "events_path": tmp_path / "events.tsv",
                "order_matrix_path": tmp_path / "order.xlsx",
            }
        ]

        result = collect_dimension_values(runs_data)

        # Should have both real_luminance (from CSV) and luminance (joystick)
        assert "real_luminance" in result
        assert len(result["real_luminance"]) > 0
        assert result["real_luminance"][0]["dimension"] == "real_luminance"

    def test_skips_joystick_when_channel_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should skip joystick extraction when joystick_x is absent."""
        sfreq = 500.0
        duration_s = 30.0
        # EEG without joystick_x
        channel_names = ["O1", "O2"]
        n_samples = int(duration_s * sfreq)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, n_samples)) * 1e-6
        info = mne.create_info(
            ch_names=channel_names, sfreq=sfreq, ch_types="eeg"
        )
        eeg_raw = mne.io.RawArray(data, info, verbose=False)

        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        vhdr_path = eeg_dir / "test.vhdr"
        mne.export.export_raw(str(vhdr_path), eeg_raw, overwrite=True)

        monkeypatch.setattr(_script14, "_resolve_eeg_path", lambda rc: vhdr_path)

        events_df = pd.DataFrame(
            {
                "trial_type": ["video"],
                "stim_id": [103],
                "onset": [1.0],
                "duration": [10.0],
            }
        )
        order_matrix_df = pd.DataFrame(
            {
                "dimension": ["arousal"],
                "order_emojis_slider": ["direct"],
                "video_id": [3],
            }
        )
        runs_data = [
            {
                "run_config": {
                    "id": "002",
                    "acq": "a",
                    "task": "01",
                    "block": "block1",
                },
                "events_df": events_df,
                "order_matrix_df": order_matrix_df,
                "events_path": tmp_path / "events.tsv",
                "order_matrix_path": tmp_path / "order.xlsx",
            }
        ]

        result = collect_dimension_values(runs_data)

        # No joystick channel → no arousal entries
        assert "arousal" not in result or len(result.get("arousal", [])) == 0


# ---------------------------------------------------------------------------
# Tests for plot_distribution (Req 4.4, 4.5, 4.6)
# ---------------------------------------------------------------------------


class TestPlotDistribution:
    """Unit tests for the plot_distribution function."""

    def test_saves_png_with_correct_filename(self, tmp_path: Path) -> None:
        """Should save a PNG file with the BIDS-style naming convention."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(200)
        plot_distribution(values, "real_luminance", "raw", tmp_path)

        expected_name = "sub-27_real_luminance_raw_distribution.png"
        assert (tmp_path / expected_name).exists()

    def test_saves_normalized_version(self, tmp_path: Path) -> None:
        """Should save a normalized histogram with 'normalized' in name."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(100)
        plot_distribution(values, "valence", "normalized", tmp_path)

        expected_name = "sub-27_valence_normalized_distribution.png"
        assert (tmp_path / expected_name).exists()

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Should create the output directory if it doesn't exist."""
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        rng = np.random.default_rng(42)
        values = rng.standard_normal(50)
        plot_distribution(values, "arousal", "raw", nested_dir)

        assert nested_dir.exists()
        assert (nested_dir / "sub-27_arousal_raw_distribution.png").exists()

    def test_closes_all_figures(self, tmp_path: Path) -> None:
        """Should close all matplotlib figures after saving."""
        import matplotlib.pyplot as _plt

        rng = np.random.default_rng(42)
        values = rng.standard_normal(50)
        plot_distribution(values, "luminance", "raw", tmp_path)

        assert len(_plt.get_fignums()) == 0


# ---------------------------------------------------------------------------
# Tests for run_pipeline (Req 4.4, 4.5, 4.6, 4.7)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Unit tests for the run_pipeline orchestration function."""

    def test_skips_empty_dimension(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """Should log warning and skip when a dimension has no entries."""
        monkeypatch.setattr(
            _script14,
            "load_all_runs_data",
            lambda: [{"run_config": {}}],
        )
        monkeypatch.setattr(
            _script14,
            "collect_dimension_values",
            lambda _runs: {"arousal": []},
        )
        monkeypatch.setattr(
            _script14, "EXPLORATION_OUTPUT_DIR", tmp_path
        )

        with caplog.at_level(logging.WARNING):
            run_pipeline()

        assert "has no data" in caplog.text

    def test_generates_raw_and_normalized_histograms(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should produce both raw and normalized PNGs per dimension."""
        rng = np.random.default_rng(42)
        entries = [
            {
                "value": float(val),
                "video_identifier": f"{vid}_a",
                "video_id": vid,
                "dimension": "valence",
                "run_id": "002",
                "acq": "a",
            }
            for vid in [3, 7]
            for val in rng.standard_normal(50)
        ]

        monkeypatch.setattr(
            _script14,
            "load_all_runs_data",
            lambda: [{"run_config": {}}],
        )
        monkeypatch.setattr(
            _script14,
            "collect_dimension_values",
            lambda _runs: {"valence": entries},
        )
        monkeypatch.setattr(
            _script14, "EXPLORATION_OUTPUT_DIR", tmp_path
        )

        run_pipeline()

        assert (
            tmp_path / "sub-27_valence_raw_distribution.png"
        ).exists()
        assert (
            tmp_path / "sub-27_valence_normalized_distribution.png"
        ).exists()

    def test_handles_no_runs_data(
        self, monkeypatch, caplog
    ) -> None:
        """Should log error and return when no run data is loaded."""
        monkeypatch.setattr(
            _script14, "load_all_runs_data", lambda: []
        )

        with caplog.at_level(logging.ERROR):
            run_pipeline()

        assert "No run data loaded" in caplog.text

    def test_handles_no_dimension_values(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """Should log error and return when no values are collected."""
        monkeypatch.setattr(
            _script14,
            "load_all_runs_data",
            lambda: [{"run_config": {}}],
        )
        monkeypatch.setattr(
            _script14,
            "collect_dimension_values",
            lambda _runs: {},
        )

        with caplog.at_level(logging.ERROR):
            run_pipeline()

        assert "No dimension values collected" in caplog.text

    def test_skips_normalized_when_all_groups_zero_std(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """Should skip normalized histogram when all groups have std=0."""
        entries = [
            {
                "value": 5.0,
                "video_identifier": "3_a",
                "video_id": 3,
                "dimension": "luminance",
                "run_id": "002",
                "acq": "a",
            }
            for _ in range(10)
        ]

        monkeypatch.setattr(
            _script14,
            "load_all_runs_data",
            lambda: [{"run_config": {}}],
        )
        monkeypatch.setattr(
            _script14,
            "collect_dimension_values",
            lambda _runs: {"luminance": entries},
        )
        monkeypatch.setattr(
            _script14, "EXPLORATION_OUTPUT_DIR", tmp_path
        )

        with caplog.at_level(logging.WARNING):
            run_pipeline()

        # Raw should still be generated
        assert (
            tmp_path / "sub-27_luminance_raw_distribution.png"
        ).exists()
        # Normalized should be skipped
        assert not (
            tmp_path / "sub-27_luminance_normalized_distribution.png"
        ).exists()
        assert "std=0" in caplog.text
