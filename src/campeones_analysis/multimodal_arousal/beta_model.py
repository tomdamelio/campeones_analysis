"""Beta-model for adaptive window selection (Etapas 2-3 of the framework).

Three operations, one per function:

  1. ``compute_b1_informativeness`` — for each window ``j``, score how
     well the KNN's local prediction matches the actual target trajectory:
        a_{m,j} = Pearson( y_traj_pred[j, :], Y_traj[j, :] )  in [-1, 1]

  2. ``fit_beta_ridge`` — ridge regression mapping window features to
     expected informativeness:
        β_m, b_m = argmin_{β,b}  Σ_j (a_{m,j} − D_m[j,:]·β − b)^2
                                + α_ridge ‖β‖^2

     The implementation defaults to ``fit_intercept=True``. This is a
     deliberate deviation from sec 4.5 of notas_ada_v2, which writes the
     formula without intercept and justifies it by a "gauge invariance"
     argument (``β → cβ ⟹ ŵ → cŵ``). The reconstruction-level invariance
     that actually matters — ``ŷ_m(t)`` is unchanged under ``ŵ → cŵ``
     because Stage A normalises by ``Σ ŵ`` — holds for ANY non-negative
     ``ŵ``, with or without intercept. Empirically, no-intercept ridge
     cannot fit the constant component of the informativeness signal and
     becomes pathologically sensitive to feature-mean shifts in ``D``.
     Pass ``fit_intercept=False`` to recover the spec-literal behaviour.

  3. ``predict_window_weights`` — per-window weight estimate at test time:
        ŵ_{m,j*} = max( D_m[j*, :] · β_m + b_m, 0 )

     The clipping discards windows that the model predicts as
     anti-informative; its output feeds Stage A reconstruction as the
     ``weights`` argument.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def compute_b1_informativeness(
    y_traj_pred: np.ndarray,
    y_traj_true: np.ndarray,
) -> np.ndarray:
    """Pearson correlation between predicted and true trajectory per window.

    Implements the "B1" definition of informativeness (sec 3.2): the
    vector generalisation of ADA's signed scalar product.

    Parameters
    ----------
    y_traj_pred
        Predicted trajectories per window, shape ``(W, L)``. Rows that
        are entirely NaN (e.g. KNN queries with all-zero weights) yield
        NaN informativeness.
    y_traj_true
        Actual target trajectories per window, shape ``(W, L)``.

    Returns
    -------
    a
        Informativeness per window, shape ``(W,)``. NaN where the
        prediction or the target had zero variance, or where the
        prediction row was NaN.
    """
    if y_traj_pred.shape != y_traj_true.shape:
        raise ValueError(
            f"Shape mismatch: y_traj_pred={y_traj_pred.shape}, "
            f"y_traj_true={y_traj_true.shape}"
        )
    pred_c = y_traj_pred - y_traj_pred.mean(axis=1, keepdims=True)
    true_c = y_traj_true - y_traj_true.mean(axis=1, keepdims=True)
    pred_norm = np.linalg.norm(pred_c, axis=1)
    true_norm = np.linalg.norm(true_c, axis=1)

    a = np.full(y_traj_pred.shape[0], np.nan, dtype=float)
    valid = (
        np.isfinite(pred_norm)
        & np.isfinite(true_norm)
        & (pred_norm > 1e-12)
        & (true_norm > 1e-12)
    )
    if valid.any():
        num = (pred_c[valid] * true_c[valid]).sum(axis=1)
        a[valid] = num / (pred_norm[valid] * true_norm[valid])
    return a


def fit_beta_ridge(
    D: np.ndarray,
    a: np.ndarray,
    alpha_ridge: float = 0.1,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """Fit ridge regression D -> a.

    Rows with NaN ``a`` are dropped before fitting.

    Parameters
    ----------
    D
        Window features, shape ``(W, L)``.
    a
        Informativeness per window, shape ``(W,)``. NaN entries are
        dropped (windows where the KNN failed to produce a valid
        prediction or where the target was constant).
    alpha_ridge
        Ridge regularisation strength. Default 0.1 (V1 baseline).
    fit_intercept
        Whether to fit a free intercept term. Default ``True`` — see the
        module docstring for the justification of deviating from the
        spec-literal no-intercept formula. Pass ``False`` to recover the
        original behaviour.

    Returns
    -------
    beta
        Coefficient vector, shape ``(L,)``.
    intercept
        Scalar intercept (0.0 when ``fit_intercept`` is ``False``).
    """
    if D.ndim != 2:
        raise ValueError(f"D must be 2D, got shape {D.shape}")
    if a.ndim != 1 or a.shape[0] != D.shape[0]:
        raise ValueError(
            f"a must be 1D with shape ({D.shape[0]},), got {a.shape}"
        )
    if alpha_ridge < 0:
        raise ValueError(f"alpha_ridge must be >= 0, got {alpha_ridge}")

    valid = np.isfinite(a)
    if valid.sum() < 2:
        raise ValueError(
            f"Need at least 2 valid (finite) targets to fit ridge, got "
            f"{int(valid.sum())}."
        )

    model = Ridge(alpha=alpha_ridge, fit_intercept=fit_intercept)
    model.fit(D[valid], a[valid])
    coef = np.asarray(model.coef_, dtype=float).copy()
    intercept = float(model.intercept_) if fit_intercept else 0.0
    return coef, intercept


def predict_window_weights(
    D: np.ndarray,
    beta: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Per-window weight predictions via ``ŵ = max(D @ β + b, 0)``.

    The clipping enforces non-negative weights so that the soft-weighted
    reconstruction in Stage A is well-defined (denominator non-negative).
    The reconstruction normalises by ``Σ ŵ``, so the absolute scale of
    the predictions never reaches the final ``ŷ_m(t)`` — it is the SHAPE
    (relative weighting between windows) that matters.
    """
    if D.ndim != 2:
        raise ValueError(f"D must be 2D, got shape {D.shape}")
    if beta.ndim != 1 or beta.shape[0] != D.shape[1]:
        raise ValueError(
            f"beta must be 1D with shape ({D.shape[1]},), got {beta.shape}"
        )
    return np.clip(D @ beta + intercept, 0.0, None)
