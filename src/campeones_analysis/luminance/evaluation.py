"""Evaluation metrics for luminance prediction models.

Pure functions for computing regression metrics used across all
luminance prediction scripts (10–19). Wraps sklearn to provide a
consistent, testable interface.
"""

import numpy as np
from sklearn.metrics import r2_score


def compute_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the coefficient of determination (R²) for luminance predictions.

    Quantifies the proportion of luminance variability explained by the EEG
    model. R² = 1 indicates perfect prediction; R² = 0 means the model
    performs no better than predicting the mean; negative values indicate
    worse-than-mean predictions.

    Args:
        y_true: 1-D array of true luminance target values.
        y_pred: 1-D array of predicted luminance values, same length as y_true.

    Returns:
        R² score as a float. Delegates to sklearn.metrics.r2_score.

    References:
        Requirements 7.1, 7.2.
    """
    return float(r2_score(y_true, y_pred))
