"""Property-based and unit tests for EEG-luminance synchronisation module.

Tests correctness properties of epoch generation and luminance interpolation
using Hypothesis. Each property maps to a formal correctness property from
the design document.  Unit tests cover specific examples and edge cases.

Feature: eeg-luminance-prediction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
    load_luminance_csv,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Realistic EEG parameters
sfreq_strategy = st.sampled_from([250.0, 256.0, 500.0, 512.0, 1000.0])
epoch_duration_strategy = st.floats(min_value=0.1, max_value=2.0, allow_nan=False)
epoch_step_strategy = st.floats(min_value=0.01, max_value=1.0, allow_nan=False)


@st.composite
def epoch_params(draw: st.DrawFn) -> dict:
    """Generate valid epoch-generation parameters.

    Ensures the segment is long enough for at least one epoch and that
    step <= duration (overlap constraint).
    """
    sfreq = draw(sfreq_strategy)
    epoch_duration_s = draw(epoch_duration_strategy)
    epoch_step_s = draw(epoch_step_strategy)
    assume(epoch_step_s <= epoch_duration_s)

    # Segment must hold at least one full epoch
    min_samples = int(np.ceil(epoch_duration_s * sfreq)) + 1
    n_samples_total = draw(
        st.integers(min_value=min_samples, max_value=min_samples + 50000)
    )

    return {
        "n_samples_total": n_samples_total,
        "sfreq": sfreq,
        "epoch_duration_s": epoch_duration_s,
        "epoch_step_s": epoch_step_s,
    }


@st.composite
def luminance_and_epochs(draw: st.DrawFn) -> dict:
    """Generate a luminance DataFrame and matching epoch onsets.

    Produces a monotonically increasing timestamp series with luminance
    values in [0, 255], plus epoch onsets that fall within the time range.
    """
    n_frames = draw(st.integers(min_value=20, max_value=500))
    fps = draw(st.sampled_from([30.0, 60.0]))
    timestamps = np.arange(n_frames) / fps

    # Use a seed-based approach for fast array generation
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    luminance_values = rng.uniform(0.0, 255.0, size=n_frames)

    luminance_df = pd.DataFrame(
        {"timestamp": timestamps, "luminance": luminance_values}
    )

    total_duration = timestamps[-1]
    epoch_duration_s = draw(
        st.floats(min_value=0.05, max_value=min(0.5, total_duration * 0.5), allow_nan=False)
    )
    epoch_step_s = draw(
        st.floats(min_value=0.01, max_value=epoch_duration_s, allow_nan=False)
    )

    max_onset = total_duration - epoch_duration_s
    assume(max_onset >= 0.0)

    epoch_onsets_s = np.arange(0.0, max_onset + epoch_step_s / 2.0, epoch_step_s)
    epoch_onsets_s = epoch_onsets_s[epoch_onsets_s <= max_onset + 1e-12]
    assume(len(epoch_onsets_s) > 0)

    return {
        "luminance_df": luminance_df,
        "epoch_onsets_s": epoch_onsets_s,
        "epoch_duration_s": epoch_duration_s,
        "luminance_values": luminance_values,
    }


# ---------------------------------------------------------------------------
# Property 4: Conteo y espaciado de épocas generadas
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@given(params=epoch_params())
@settings(max_examples=100)
def test_property4_epoch_count_and_spacing(params: dict) -> None:
    """Property 4: Epoch count and spacing.

    For any EEG segment of total duration T seconds, with epoch duration D
    and step S, the number of epochs must be floor((T - D) / S) + 1, and
    consecutive onsets must be separated by exactly S seconds.

    **Validates: Requirements 3.2**
    """
    # Feature: eeg-luminance-prediction, Property 4: Conteo y espaciado de épocas generadas
    onsets = create_epoch_onsets(**params)

    total_duration_s = params["n_samples_total"] / params["sfreq"]
    epoch_duration_s = params["epoch_duration_s"]
    epoch_step_s = params["epoch_step_s"]

    # Expected count: floor((T - D) / S) + 1
    # Use round() to mitigate floating-point drift in the division before floor
    last_valid_onset = total_duration_s - epoch_duration_s
    expected_count = int(np.floor(round(last_valid_onset / epoch_step_s, 10))) + 1
    assert len(onsets) == expected_count, (
        f"Expected {expected_count} epochs, got {len(onsets)}. "
        f"T={total_duration_s:.6f}, D={epoch_duration_s}, S={epoch_step_s}"
    )

    # Consecutive onsets separated by exactly S seconds (within float tolerance)
    if len(onsets) > 1:
        diffs = np.diff(onsets)
        np.testing.assert_allclose(
            diffs,
            epoch_step_s,
            atol=1e-10,
            err_msg="Consecutive epoch onsets are not evenly spaced by epoch_step_s",
        )

    # All epochs must fit within the segment
    if len(onsets) > 0:
        assert onsets[-1] + epoch_duration_s <= total_duration_s + 1e-10, (
            "Last epoch extends beyond segment duration"
        )


# ---------------------------------------------------------------------------
# Property 5: Luminancia interpolada dentro de rango válido
# Validates: Requirements 3.1, 3.3
# ---------------------------------------------------------------------------


@given(data=luminance_and_epochs())
@settings(max_examples=100, deadline=1000)
def test_property5_interpolated_luminance_within_range(data: dict) -> None:
    """Property 5: Interpolated luminance within valid range.

    For any luminance CSV and any set of aligned epochs, each epoch-averaged
    luminance value must lie within [min(original), max(original)].

    **Validates: Requirements 3.1, 3.3**
    """
    # Feature: eeg-luminance-prediction, Property 5: Luminancia interpolada dentro de rango válido
    interpolated = interpolate_luminance_to_epochs(
        luminance_df=data["luminance_df"],
        epoch_onsets_s=data["epoch_onsets_s"],
        epoch_duration_s=data["epoch_duration_s"],
    )

    original_min = data["luminance_values"].min()
    original_max = data["luminance_values"].max()

    assert len(interpolated) == len(data["epoch_onsets_s"]), (
        f"Output length {len(interpolated)} != epoch count {len(data['epoch_onsets_s'])}"
    )

    # Each interpolated value must be within the original range (with tolerance
    # for floating-point interpolation)
    tolerance = 1e-9
    assert np.all(interpolated >= original_min - tolerance), (
        f"Interpolated min {interpolated.min():.6f} < original min {original_min:.6f}"
    )
    assert np.all(interpolated <= original_max + tolerance), (
        f"Interpolated max {interpolated.max():.6f} > original max {original_max:.6f}"
    )


# ===========================================================================
# Unit Tests – Sincronización EEG-Luminancia
# Requirements: 3.1, 3.2, 3.3, 3.7
# ===========================================================================


class TestLoadLuminanceCsv:
    """Unit tests for load_luminance_csv."""

    def test_load_valid_csv(self, tmp_path: Path) -> None:
        """Loading a well-formed CSV returns a sorted DataFrame with correct dtypes.

        Requirements: 3.1
        """
        csv_file = tmp_path / "luminance.csv"
        csv_file.write_text(
            "timestamp,luminance\n0.0,120.5\n0.016,130.0\n0.033,125.3\n"
        )

        result = load_luminance_csv(csv_file)

        assert list(result.columns) == ["timestamp", "luminance"]
        assert len(result) == 3
        assert result["timestamp"].dtype == np.float64
        assert result["luminance"].dtype == np.float64
        # Sorted by timestamp
        assert result["timestamp"].is_monotonic_increasing

    def test_csv_not_found_raises(self, tmp_path: Path) -> None:
        """A missing CSV must raise FileNotFoundError (Req 3.7)."""
        missing_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_luminance_csv(missing_path)

    def test_csv_missing_columns_raises(self, tmp_path: Path) -> None:
        """A CSV without required columns must raise ValueError."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("time,brightness\n0.0,100\n")

        with pytest.raises(ValueError, match="missing columns"):
            load_luminance_csv(csv_file)


class TestCreateEpochOnsets:
    """Unit tests for create_epoch_onsets."""

    def test_segment_too_short_returns_empty(self) -> None:
        """When the segment is shorter than one epoch, return an empty array.

        Requirements: 3.2 (edge case)
        """
        # 10 samples at 500 Hz = 0.02 s, epoch = 0.5 s → too short
        onsets = create_epoch_onsets(
            n_samples_total=10,
            sfreq=500.0,
            epoch_duration_s=0.5,
            epoch_step_s=0.1,
        )

        assert len(onsets) == 0
        assert onsets.dtype == np.float64

    def test_known_epoch_count(self) -> None:
        """Verify epoch count for a concrete example.

        1 s segment, 0.5 s epoch, 0.1 s step → floor((1.0 - 0.5) / 0.1) + 1 = 6
        Requirements: 3.2
        """
        onsets = create_epoch_onsets(
            n_samples_total=500,
            sfreq=500.0,
            epoch_duration_s=0.5,
            epoch_step_s=0.1,
        )

        assert len(onsets) == 6
        np.testing.assert_allclose(onsets[0], 0.0)
        np.testing.assert_allclose(np.diff(onsets), 0.1, atol=1e-12)


class TestInterpolateLuminanceToEpochs:
    """Unit tests for interpolate_luminance_to_epochs."""

    def test_empty_onsets_returns_empty(self) -> None:
        """No epochs → empty output array.

        Requirements: 3.3
        """
        luminance_df = pd.DataFrame(
            {"timestamp": [0.0, 1.0], "luminance": [100.0, 200.0]}
        )
        result = interpolate_luminance_to_epochs(
            luminance_df, epoch_onsets_s=np.array([]), epoch_duration_s=0.5
        )

        assert len(result) == 0

    def test_constant_luminance_returns_constant(self) -> None:
        """When luminance is constant, every epoch average equals that constant.

        Requirements: 3.1, 3.3
        """
        constant_value = 128.0
        luminance_df = pd.DataFrame(
            {
                "timestamp": np.linspace(0, 5, 300),
                "luminance": np.full(300, constant_value),
            }
        )
        onsets = np.array([0.0, 0.5, 1.0, 1.5])

        result = interpolate_luminance_to_epochs(
            luminance_df, epoch_onsets_s=onsets, epoch_duration_s=0.5
        )

        np.testing.assert_allclose(result, constant_value, atol=1e-6)
