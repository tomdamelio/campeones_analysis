"""Soft-weighted reconstruction (Stage A) of the continuous prediction.

Given the per-window trajectory predictions ``ŷ^(traj)[j, :]`` and the
per-window informativeness weights ``ŵ_j``, build a continuous estimate
``ŷ(t)`` over the test video by averaging — at each sample ``t`` — the
predictions from every window whose post-lag support covers ``t``.

For Test 1 the weights are uniform (`weights=None` -> all ones).
"""

from __future__ import annotations

import numpy as np


def reconstruct_stage_a(
    y_traj_pred: np.ndarray,
    t_start: np.ndarray,
    L: int,
    n_samples: int,
    weights: np.ndarray | None = None,
    tau: int = 0,
) -> np.ndarray:
    """Stage A reconstruction of a continuous prediction from window predictions.

    For each sample ``t`` in ``[0, n_samples)``:
        cover(t) = { j : t_start[j] + tau <= t < t_start[j] + tau + L }
        ŷ_j(t) = y_traj_pred[j, t - t_start[j] - tau]
        ŷ(t) = sum_j(weights[j] * ŷ_j(t)) / sum_j(weights[j])

    Samples for which ``cover(t)`` is empty, the active weights sum to 0,
    or every covering prediction is NaN are returned as NaN.

    Parameters
    ----------
    y_traj_pred
        Predicted trajectories per window, shape ``(W_test, L)``. NaN rows
        (e.g. produced by ``knn_trajectory_predict`` when all neighbours
        had non-positive similarity) are skipped at every sample they
        would otherwise contribute to.
    t_start
        Start sample of each window in the test video, shape ``(W_test,)``.
    L
        Window length in samples (consistent with ``y_traj_pred.shape[1]``).
    n_samples
        Number of samples in the test video (length of the output).
    weights
        Per-window non-negative weights, shape ``(W_test,)``. ``None``
        falls back to uniform weights (1.0 each), as required for Test 1.
    tau
        Lag in samples applied to the target side (0 in Test 1).

    Returns
    -------
    y_hat
        Reconstructed continuous prediction, shape ``(n_samples,)``.
    """
    y_traj_pred = np.asarray(y_traj_pred, dtype=float)
    t_start = np.asarray(t_start, dtype=np.int64).ravel()
    w_test = y_traj_pred.shape[0]

    if y_traj_pred.shape[1] != L:
        raise ValueError(
            f"y_traj_pred has L={y_traj_pred.shape[1]} but reconstruct_stage_a "
            f"was called with L={L}."
        )
    if t_start.shape[0] != w_test:
        raise ValueError(
            f"t_start has length {t_start.shape[0]} but y_traj_pred has "
            f"{w_test} rows."
        )

    if weights is None:
        weights = np.ones(w_test, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape[0] != w_test:
            raise ValueError(
                f"weights has length {weights.shape[0]} but y_traj_pred has "
                f"{w_test} rows."
            )

    numerator = np.zeros(n_samples, dtype=float)
    denominator = np.zeros(n_samples, dtype=float)

    for j in range(w_test):
        traj = y_traj_pred[j]
        if not np.all(np.isfinite(traj)):
            continue
        w_j = weights[j]
        if not np.isfinite(w_j) or w_j <= 0:
            continue
        start = int(t_start[j]) + tau
        stop = start + L
        lo = max(start, 0)
        hi = min(stop, n_samples)
        if hi <= lo:
            continue
        local_lo = lo - start
        local_hi = hi - start
        numerator[lo:hi] += w_j * traj[local_lo:local_hi]
        denominator[lo:hi] += w_j

    y_hat = np.full(n_samples, np.nan, dtype=float)
    covered = denominator > 0
    y_hat[covered] = numerator[covered] / denominator[covered]
    return y_hat
