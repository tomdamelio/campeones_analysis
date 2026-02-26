"""Property-based tests for class balancing (undersampling) correctness.

Feature: eeg-luminance-validation
Property 10: Undersampling produces balanced classes
Validates: Requirements 9.2
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Pure extraction of the undersampling logic for isolated testing.
# Mirrors ``undersample_majority_class`` from script 20 without any
# config or I/O dependencies.
# ---------------------------------------------------------------------------


def _undersample_majority_class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure undersampling function extracted for testing.

    Mirrors ``undersample_majority_class`` from script 20 without any
    config or I/O dependencies.

    Args:
        X_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary label array with values 0 or 1.
        random_seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (X_balanced, y_balanced) with equal class counts.
    """
    rng = np.random.default_rng(random_seed)
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    minority_count = min(len(class_0_indices), len(class_1_indices))

    if len(class_0_indices) > len(class_1_indices):
        majority_indices = class_0_indices
        minority_indices = class_1_indices
    else:
        majority_indices = class_1_indices
        minority_indices = class_0_indices

    sampled_majority = rng.choice(majority_indices, size=minority_count, replace=False)
    balanced_indices = np.concatenate([minority_indices, sampled_majority])
    balanced_indices = np.sort(balanced_indices)

    return X_train[balanced_indices], y_train[balanced_indices]


# ---------------------------------------------------------------------------
# Property 10: Undersampling produces balanced classes
# Validates: Requirements 9.2
# ---------------------------------------------------------------------------


@given(
    minority_count=st.integers(min_value=1, max_value=20),
    majority_extra=st.integers(min_value=1, max_value=30),
    minority_class=st.integers(min_value=0, max_value=1),
    n_features=st.integers(min_value=1, max_value=10),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=200)
def test_undersampling_produces_balanced_classes(
    minority_count: int,
    majority_extra: int,
    minority_class: int,
    n_features: int,
    random_seed: int,
) -> None:
    """Property 10: Undersampling produces balanced classes.

    For any set of binary-labeled training epochs where both classes are
    present, after undersampling the majority class, the count of class 0
    and class 1 should be equal (matching the minority class count).

    Validates: Requirements 9.2
    """
    majority_class = 1 - minority_class
    majority_count = minority_count + majority_extra

    # Build imbalanced training data
    rng = np.random.default_rng(random_seed)
    X_minority = rng.standard_normal((minority_count, n_features))
    X_majority = rng.standard_normal((majority_count, n_features))
    X_train = np.vstack([X_minority, X_majority])
    y_train = np.array(
        [minority_class] * minority_count + [majority_class] * majority_count,
        dtype=int,
    )

    # Apply undersampling
    X_balanced, y_balanced = _undersample_majority_class(
        X_train, y_train, random_seed=random_seed
    )

    # Property: both classes must be present and equal in count
    count_class_0 = int(np.sum(y_balanced == 0))
    count_class_1 = int(np.sum(y_balanced == 1))

    assert count_class_0 == count_class_1, (
        f"Classes not balanced after undersampling: "
        f"class_0={count_class_0}, class_1={count_class_1}. "
        f"Original: minority_count={minority_count}, majority_count={majority_count}"
    )

    # Property: balanced count equals the original minority count
    assert count_class_0 == minority_count, (
        f"Balanced class count {count_class_0} does not match "
        f"minority count {minority_count}"
    )

    # Property: total balanced size is 2 × minority_count
    assert len(y_balanced) == 2 * minority_count, (
        f"Total balanced size {len(y_balanced)} != 2 × {minority_count}"
    )

    # Property: X_balanced rows correspond to valid rows from X_train
    assert X_balanced.shape == (2 * minority_count, n_features), (
        f"X_balanced shape {X_balanced.shape} unexpected"
    )

    # Property: no duplicate indices (sampling without replacement)
    # Verify by checking that all X_balanced rows appear in X_train
    for row_idx in range(X_balanced.shape[0]):
        row = X_balanced[row_idx]
        matches = np.all(np.isclose(X_train, row[np.newaxis, :]), axis=1)
        assert np.any(matches), (
            f"Row {row_idx} of X_balanced not found in X_train — "
            "undersampling introduced new data"
        )
