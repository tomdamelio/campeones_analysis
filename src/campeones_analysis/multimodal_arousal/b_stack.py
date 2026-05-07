"""Stage B — multimodal stacking via meta-weights ``alpha_m``.

Given per-modality predictions ``y_m_hat(t)`` produced out-of-fold (the
re-OOF set of sec 4.7), Stage B fits scalar meta-weights ``alpha_m`` so
that the stacked prediction is

    y_hat(t) = alpha_0 + sum_m  alpha_m * y_m_hat(t)

Three variants are supported:

  - ``"ols"``         : free ``alpha_m``, OLS with intercept.
  - ``"ridge"``       : free ``alpha_m`` with L2 penalty
                        ``alpha_ridge_stack``, intercept fitted.
  - ``"constrained"`` : ``alpha_m >= 0`` and ``sum alpha_m == 1`` (convex
                        combination, Breiman 1996); no intercept.

Diagnostics on the design matrix ``[y_1_hat, ..., y_M_hat]``:

  - condition number ``kappa`` (svd-based);
  - per-modality VIF (variance inflation factor).

If ``kappa > 30`` or ``VIF > 5`` the OLS interpretation of ``alpha_m``
becomes unstable — the spec recommends switching to ridge or
constrained variants in that regime.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge


def fit_b_stack(
    y_oof_pred: np.ndarray,
    y_oof_true: np.ndarray,
    variant: str = "ols",
    alpha_ridge_stack: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Fit Stage B meta-weights.

    Parameters
    ----------
    y_oof_pred
        Out-of-fold per-modality predictions, shape ``(T, M)``.
    y_oof_true
        Ground-truth target, shape ``(T,)``.
    variant
        ``"ols"``, ``"ridge"``, or ``"constrained"``.
    alpha_ridge_stack
        Ridge penalty when ``variant == "ridge"``. Ignored otherwise.

    Returns
    -------
    alpha_m
        ``(M,)`` array of meta-weights.
    alpha_0
        Scalar intercept (always 0.0 for the constrained variant).
    """
    if variant not in ("ols", "ridge", "constrained"):
        raise ValueError(
            f"Unknown variant: {variant!r}; expected ols/ridge/constrained."
        )
    if y_oof_pred.ndim != 2:
        raise ValueError(f"y_oof_pred must be 2D, got shape {y_oof_pred.shape}")
    if y_oof_true.ndim != 1 or y_oof_true.shape[0] != y_oof_pred.shape[0]:
        raise ValueError(
            f"y_oof_true must be 1D with shape ({y_oof_pred.shape[0]},), "
            f"got {y_oof_true.shape}."
        )

    valid = np.isfinite(y_oof_pred).all(axis=1) & np.isfinite(y_oof_true)
    if int(valid.sum()) < y_oof_pred.shape[1] + 1:
        raise ValueError(
            f"Not enough finite samples to fit Stage B: {int(valid.sum())} "
            f"valid rows for {y_oof_pred.shape[1]} modalities."
        )
    X = y_oof_pred[valid]
    y = y_oof_true[valid]

    if variant == "ols":
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        return np.asarray(model.coef_, dtype=float), float(model.intercept_)

    if variant == "ridge":
        if alpha_ridge_stack < 0:
            raise ValueError(
                f"alpha_ridge_stack must be >= 0, got {alpha_ridge_stack}"
            )
        model = Ridge(alpha=alpha_ridge_stack, fit_intercept=True)
        model.fit(X, y)
        return np.asarray(model.coef_, dtype=float), float(model.intercept_)

    M = X.shape[1]

    def loss(alpha: np.ndarray) -> float:
        residuals = y - X @ alpha
        return float(np.dot(residuals, residuals))

    def grad(alpha: np.ndarray) -> np.ndarray:
        residuals = y - X @ alpha
        return -2.0 * (X.T @ residuals)

    constraints = [{"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0)}]
    bounds = [(0.0, 1.0)] * M
    x0 = np.full(M, 1.0 / M)
    result = minimize(
        loss,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    alpha_m = np.asarray(result.x, dtype=float) if result.success else x0.copy()
    return alpha_m, 0.0


def predict_b_stack(
    y_test_pred: np.ndarray,
    alpha_m: np.ndarray,
    alpha_0: float = 0.0,
) -> np.ndarray:
    """Combine modality predictions: ``y_hat = alpha_0 + y_test_pred @ alpha_m``.

    NaN samples in ``y_test_pred`` propagate to NaN in the output (rows
    with any NaN modality become NaN).
    """
    if y_test_pred.ndim != 2 or y_test_pred.shape[1] != alpha_m.shape[0]:
        raise ValueError(
            f"y_test_pred shape {y_test_pred.shape} incompatible with "
            f"alpha_m shape {alpha_m.shape}."
        )
    return alpha_0 + y_test_pred @ alpha_m


def diagnostics_b_stack(
    y_oof_pred: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Condition number and per-modality VIF of the stacking design matrix.

    Both are computed on the column-centered OOF predictions (so they
    capture the geometry of the predictors, ignoring shifts in mean).

    Parameters
    ----------
    y_oof_pred
        Out-of-fold per-modality predictions, shape ``(T, M)``.

    Returns
    -------
    kappa
        Condition number ``s_max / s_min`` of the centered design matrix
        (``inf`` if degenerate). Spec threshold for "OLS interpretable":
        ``kappa <= 30``.
    vif
        ``(M,)`` array. ``vif[m] = 1 / (1 - R^2_m)`` where ``R^2_m``
        is the coefficient of determination from regressing modality
        ``m`` on the others. Spec threshold: ``VIF <= 5``.
    """
    if y_oof_pred.ndim != 2:
        raise ValueError(f"y_oof_pred must be 2D, got shape {y_oof_pred.shape}")

    valid = np.isfinite(y_oof_pred).all(axis=1)
    X = y_oof_pred[valid]
    M = X.shape[1]

    if X.shape[0] < M + 1:
        return float("inf"), np.full(M, float("inf"))

    X_centered = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X_centered, compute_uv=False)
    if s.min() <= 0:
        kappa = float("inf")
    else:
        kappa = float(s.max() / s.min())

    vif = np.zeros(M, dtype=float)
    for j in range(M):
        y_j = X_centered[:, j]
        X_others = np.delete(X_centered, j, axis=1)
        if X_others.shape[1] == 0:
            vif[j] = 1.0
            continue
        if float(np.var(y_j)) <= 0:
            vif[j] = float("inf")
            continue
        model = LinearRegression(fit_intercept=False)
        model.fit(X_others, y_j)
        ss_res = float(np.sum((y_j - model.predict(X_others)) ** 2))
        ss_tot = float(np.sum(y_j**2))
        if ss_tot <= 0:
            vif[j] = float("inf")
            continue
        r2 = 1.0 - ss_res / ss_tot
        if r2 >= 1.0 - 1e-12:
            vif[j] = float("inf")
        else:
            vif[j] = 1.0 / (1.0 - r2)

    return kappa, vif
