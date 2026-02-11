"""Property-based tests for the luminance prediction pipeline.

Tests correctness properties of Leave-One-Video-Out CV splitting and
pipeline determinism using Hypothesis.

Feature: eeg-luminance-prediction
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import pure helpers from the base model script
# ---------------------------------------------------------------------------
_MODELING_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "modeling")
if _MODELING_DIR not in sys.path:
    sys.path.insert(0, _MODELING_DIR)

# Import the module; all its dependencies (mne, sklearn, etc.) are available
# in the campeones environment.
_base_model = importlib.import_module("10_luminance_base_model")
leave_one_video_out_split = _base_model.leave_one_video_out_split
evaluate_fold = _base_model.evaluate_fold


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def epoch_entries_with_videos(draw: st.DrawFn) -> list[dict]:
    """Generate a list of epoch entry dicts with multiple video identifiers.

    Each entry has a ``video_identifier`` key and minimal data.
    Guarantees at least 2 distinct video identifiers.
    """
    n_videos = draw(st.integers(min_value=2, max_value=6))
    acq_choices = ["a", "b"]
    video_ids: list[str] = []
    for idx in range(n_videos):
        acq = draw(st.sampled_from(acq_choices))
        video_ids.append(f"{idx}_{acq}")
    # Deduplicate
    video_ids = sorted(set(video_ids))
    assume(len(video_ids) >= 2)

    entries: list[dict] = []
    for video_id in video_ids:
        n_epochs = draw(st.integers(min_value=1, max_value=20))
        for _ in range(n_epochs):
            entries.append(
                {
                    "video_identifier": video_id,
                    "X": np.zeros((2, 10)),
                    "y": 0.0,
                    "acq": video_id.split("_")[-1],
                }
            )

    return entries


# ---------------------------------------------------------------------------
# Property 6: Correctitud del split Leave-One-Video-Out
# Validates: Requirements 3.5, 4.4, 5.4
# ---------------------------------------------------------------------------


@given(entries=epoch_entries_with_videos())
@settings(max_examples=100)
def test_property6_leave_one_video_out_correctness(entries: list[dict]) -> None:
    """Property 6: Leave-One-Video-Out split correctness.

    For any set of epochs with N distinct videos, each fold must have:
    (a) the test set containing exclusively epochs of a single video,
    (b) the training set containing epochs of all other videos,
    (c) the union of train and test equal to the total set,
    (d) N folds in total.

    **Validates: Requirements 3.5, 4.4, 5.4**
    """
    # Feature: eeg-luminance-prediction, Property 6: Correctitud del split Leave-One-Video-Out
    folds = leave_one_video_out_split(entries)
    unique_videos = sorted(set(e["video_identifier"] for e in entries))
    total_count = len(entries)

    # (d) N folds
    assert len(folds) == len(unique_videos), (
        f"Expected {len(unique_videos)} folds, got {len(folds)}"
    )

    test_videos_seen: list[str] = []

    for train_entries, test_entries, test_video in folds:
        # (a) Test set contains only the test video
        test_video_ids = set(e["video_identifier"] for e in test_entries)
        assert test_video_ids == {test_video}, (
            f"Test set should only contain '{test_video}', "
            f"found {test_video_ids}"
        )

        # (b) Training set contains all other videos
        train_video_ids = set(e["video_identifier"] for e in train_entries)
        expected_train_videos = set(unique_videos) - {test_video}
        assert train_video_ids == expected_train_videos, (
            f"Train set videos {train_video_ids} != "
            f"expected {expected_train_videos}"
        )

        # (c) Union of train + test equals total
        assert len(train_entries) + len(test_entries) == total_count, (
            f"train({len(train_entries)}) + test({len(test_entries)}) "
            f"!= total({total_count})"
        )

        test_videos_seen.append(test_video)

    # Each video appears exactly once as test
    assert sorted(test_videos_seen) == unique_videos


# ---------------------------------------------------------------------------
# Property 11: Determinismo del pipeline con semilla fija
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Import spectral model helpers
# ---------------------------------------------------------------------------
_spectral_model = importlib.import_module("11_luminance_spectral_model")
select_roi_channels = _spectral_model.select_roi_channels


# ---------------------------------------------------------------------------
# Property 8: Selección de canales ROI es subconjunto válido
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------


@st.composite
def channel_lists(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """Generate available EEG channels and a ROI channel list.

    Guarantees at least one channel in the ROI exists in the available list,
    plus some ROI channels that may not exist.
    """
    all_possible = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
        "Fz", "Cz", "Pz", "FC1", "FC2", "CP1", "CP2",
        "CP5", "CP6", "FC5", "FC6", "FT9", "FT10", "TP9", "TP10", "FCz",
    ]
    n_available = draw(st.integers(min_value=1, max_value=len(all_possible)))
    available = draw(
        st.lists(
            st.sampled_from(all_possible),
            min_size=n_available,
            max_size=n_available,
            unique=True,
        )
    )

    roi_pool = [
        "O1", "O2", "P3", "P4", "P7", "P8", "Pz",
        "CP1", "CP2", "CP5", "CP6",
    ]
    n_roi = draw(st.integers(min_value=1, max_value=len(roi_pool)))
    roi = draw(
        st.lists(
            st.sampled_from(roi_pool),
            min_size=n_roi,
            max_size=n_roi,
            unique=True,
        )
    )

    return available, roi


@given(data=channel_lists())
@settings(max_examples=100)
def test_property8_roi_channel_selection(data: tuple[list[str], list[str]]) -> None:
    """Property 8: ROI channel selection is a valid subset.

    For any list of available EEG channels and the ROI definition, the
    selected channels must be a subset of ROI_Posterior AND a subset of
    the available channels (intersection).

    **Validates: Requirements 4.2**
    """
    # Feature: eeg-luminance-prediction, Property 8: Selección de canales ROI es subconjunto válido
    available_channels, roi_channels = data
    selected = select_roi_channels(available_channels, roi_channels)

    available_set = set(available_channels)
    roi_set = set(roi_channels)

    # Selected must be a subset of both available and ROI
    assert set(selected) <= available_set, (
        f"Selected channels {selected} not subset of available {available_channels}"
    )
    assert set(selected) <= roi_set, (
        f"Selected channels {selected} not subset of ROI {roi_channels}"
    )

    # Selected must equal the intersection
    expected_intersection = roi_set & available_set
    assert set(selected) == expected_intersection, (
        f"Selected {set(selected)} != intersection {expected_intersection}"
    )

    # Order must follow ROI definition order
    roi_order = [ch for ch in roi_channels if ch in available_set]
    assert selected == roi_order, (
        f"Selected order {selected} != expected ROI order {roi_order}"
    )


# ---------------------------------------------------------------------------
# Import spectral model helpers
# ---------------------------------------------------------------------------
_spectral_model = importlib.import_module("11_luminance_spectral_model")
select_roi_channels = _spectral_model.select_roi_channels


# ---------------------------------------------------------------------------
# Property 8: Selección de canales ROI es subconjunto válido
# Validates: Requirements 4.2
# ---------------------------------------------------------------------------


@st.composite
def channel_lists(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """Generate available EEG channels and a ROI channel list.

    Guarantees at least one channel in the ROI exists in the available list,
    plus some ROI channels that may not exist.
    """
    all_possible = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
        "Fz", "Cz", "Pz", "FC1", "FC2", "CP1", "CP2",
        "CP5", "CP6", "FC5", "FC6", "FT9", "FT10", "TP9", "TP10", "FCz",
    ]
    n_available = draw(st.integers(min_value=1, max_value=len(all_possible)))
    available = draw(
        st.lists(
            st.sampled_from(all_possible),
            min_size=n_available,
            max_size=n_available,
            unique=True,
        )
    )

    roi_pool = [
        "O1", "O2", "P3", "P4", "P7", "P8", "Pz",
        "CP1", "CP2", "CP5", "CP6",
    ]
    n_roi = draw(st.integers(min_value=1, max_value=len(roi_pool)))
    roi = draw(
        st.lists(
            st.sampled_from(roi_pool),
            min_size=n_roi,
            max_size=n_roi,
            unique=True,
        )
    )

    return available, roi


@given(data=channel_lists())
@settings(max_examples=100)
def test_property8_roi_channel_selection(data: tuple[list[str], list[str]]) -> None:
    """Property 8: ROI channel selection is a valid subset.

    For any list of available EEG channels and the ROI definition, the
    selected channels must be a subset of ROI_Posterior AND a subset of
    the available channels (intersection).

    **Validates: Requirements 4.2**
    """
    # Feature: eeg-luminance-prediction, Property 8: Selección de canales ROI es subconjunto válido
    available_channels, roi_channels = data
    selected = select_roi_channels(available_channels, roi_channels)

    available_set = set(available_channels)
    roi_set = set(roi_channels)

    # Selected must be a subset of both available and ROI
    assert set(selected) <= available_set, (
        f"Selected channels {selected} not subset of available {available_channels}"
    )
    assert set(selected) <= roi_set, (
        f"Selected channels {selected} not subset of ROI {roi_channels}"
    )

    # Selected must equal the intersection
    expected_intersection = roi_set & available_set
    assert set(selected) == expected_intersection, (
        f"Selected {set(selected)} != intersection {expected_intersection}"
    )

    # Order must follow ROI definition order
    roi_order = [ch for ch in roi_channels if ch in available_set]
    assert selected == roi_order, (
        f"Selected order {selected} != expected ROI order {roi_order}"
    )


# ---------------------------------------------------------------------------
# Property 11: Determinismo del pipeline con semilla fija
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import config for the alpha grid
from config_luminance import RIDGE_ALPHA_GRID


# ---------------------------------------------------------------------------
# Property 3: Grid search selects alpha from the configured grid
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@st.composite
def grid_search_data(draw: st.DrawFn) -> dict:
    """Generate synthetic training data for grid search alpha selection testing.

    Produces 2-D feature matrices and continuous targets with enough samples
    for 3-fold inner CV across the alpha grid.
    """
    n_features = draw(st.integers(min_value=3, max_value=20))
    # Need at least 3 samples per fold × 3 folds = 9 minimum for inner CV
    n_train = draw(st.integers(min_value=15, max_value=80))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.standard_normal(n_train)

    pca_components = min(10, n_features, n_train)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "seed": seed,
        "pca_components": pca_components,
    }


@given(data=grid_search_data())
@settings(max_examples=100, deadline=10000)
def test_property3_grid_search_alpha_selection(data: dict) -> None:
    """Property 3: Grid search selects alpha from the configured grid.

    For any training feature matrix and target vector with sufficient
    samples, after running GridSearchCV with RIDGE_ALPHA_GRID, the
    selected best_params_["ridge__alpha"] must be a member of
    RIDGE_ALPHA_GRID.

    **Validates: Requirements 3.2**
    """
    # Feature: luminance-model-improvements, Property 3: Grid search selects alpha from the configured grid
    seed = data["seed"]
    pca_components = data["pca_components"]

    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=pca_components, random_state=seed),
        Ridge(random_state=seed),
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
        cv=3,
        scoring="neg_mean_squared_error",
        refit=True,
    )
    grid_search.fit(data["X_train"], data["y_train"])

    best_alpha = grid_search.best_params_["ridge__alpha"]
    assert best_alpha in RIDGE_ALPHA_GRID, (
        f"Selected alpha {best_alpha} not in RIDGE_ALPHA_GRID {RIDGE_ALPHA_GRID}"
    )


@st.composite
def determinism_data(draw: st.DrawFn) -> dict:
    """Generate synthetic train/test data for pipeline determinism testing.

    Produces small 3-D arrays (n_epochs, n_channels, n_samples) with
    continuous targets, plus a random seed.
    """
    n_channels = draw(st.integers(min_value=2, max_value=8))
    n_samples = draw(st.integers(min_value=20, max_value=100))
    n_train = draw(st.integers(min_value=30, max_value=100))
    n_test = draw(st.integers(min_value=5, max_value=30))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_channels, n_samples))
    y_train = rng.standard_normal(n_train)
    X_test = rng.standard_normal((n_test, n_channels, n_samples))

    # PCA components must be <= min(n_train, n_features)
    n_features = n_channels * n_samples
    pca_components = min(10, n_features, n_train)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "seed": seed,
        "pca_components": pca_components,
    }


@given(data=determinism_data())
@settings(max_examples=100, deadline=5000)
def test_property11_pipeline_determinism(data: dict) -> None:
    """Property 11: Pipeline determinism with fixed seed.

    For any execution of the pipeline with the same input data and the
    same random seed, the predictions must be identical.

    **Validates: Requirements 6.2**
    """
    # Feature: eeg-luminance-prediction, Property 11: Determinismo del pipeline con semilla fija
    seed = data["seed"]
    pca_components = data["pca_components"]

    predictions: list[np.ndarray] = []

    for _ in range(2):
        np.random.seed(seed)
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=pca_components, random_state=seed),
            Ridge(alpha=1.0, random_state=seed),
        )
        pipeline.fit(data["X_train"], data["y_train"])
        y_pred = pipeline.predict(data["X_test"])
        predictions.append(y_pred)

    np.testing.assert_array_equal(
        predictions[0],
        predictions[1],
        err_msg="Pipeline predictions differ between two runs with the same seed",
    )
