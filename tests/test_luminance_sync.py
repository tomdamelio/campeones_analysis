"""Property-based tests for EEG-luminance synchronisation module.

Tests correctness properties of epoch generation and luminance interpolation
using Hypothesis. Each property maps to a formal correctness property from
the design document.

Feature: eeg-luminance-prediction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from campeones_analysis.luminance.sync import (
    create_epoch_onsets,
    interpolate_luminance_to_epochs,
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
    n_frames = draw(st.integers(min_value=20, max_value=2000))
    fps = draw(st.sampled_from([30.0, 60.0]))
    timestamps = np.arange(n_frames) / fps

    luminance_values = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False),
            min_size=n_frames,
            max_size=n_frames,
        )
    )

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
        "luminance_values": np.array(luminance_values),
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
    expected_count = int(np.floor((total_duration_s - epoch_duration_s) / epoch_step_s)) + 1
    assert len(onsets) == expected_count, (
        f"Expected {expected_count} epochs, got {len(onsets)}. "
        f"T={total_duration_s:.4f}, D={epoch_duration_s}, S={epoch_step_s}"
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
@settings(max_examples=100)
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
