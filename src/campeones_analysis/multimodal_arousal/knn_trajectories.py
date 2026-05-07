"""KNN-of-trajectories: per-window prediction of the target trajectory.

For each query window ``j*``, find the ``K`` most-similar windows from
OTHER videos (cosine similarity over the feature window) and return the
weighted average of their target trajectories. Weights are
``max(cos_sim, 0)``, dropping anti-correlated neighbours.
"""

from __future__ import annotations

import numpy as np

from .data_structures import WindowMatrices


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm.

    Zero rows are left unchanged (the safe-divide replaces a zero norm
    with 1, so the row stays zero rather than producing NaNs).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return matrix / safe


# Backwards-compatible alias for the previous private name.
_row_normalize = row_normalize


def compute_full_similarity(wm: WindowMatrices) -> np.ndarray:
    """Compute the full ``(W, W)`` cosine-similarity matrix of ``wm.D``.

    Useful as a cache when many queries are made against the same ``wm``
    (e.g. inside an inner LOVO loop), since the cost of the dense matmul
    is paid once instead of once per query batch.
    """
    d_norm = row_normalize(wm.D)
    return d_norm @ d_norm.T


def knn_trajectory_predict(
    wm: WindowMatrices,
    query_indices: np.ndarray,
    K: int,
    *,
    sims_full: np.ndarray | None = None,
) -> np.ndarray:
    """Predict target trajectories for a set of query windows via KNN.

    Steps for each query index ``j*``:
      1. Compute cosine similarity between ``D[j*]`` and every ``D[k]``.
      2. Mask out neighbours from the same video as ``j*`` (set sim to -inf).
      3. Take the top-K neighbours by descending similarity.
      4. Build weights ``w_k = max(cos_sim, 0)``.
      5. If ``sum(w_k) == 0``, return a NaN row (handled downstream by
         the reconstruction step). Otherwise, return the weighted average
         of ``Y_traj`` over the top-K neighbours.

    The implementation row-normalises ``D`` once and uses the dense
    similarity matrix; brute-force is fine at the scales of Tests 1-2.

    Parameters
    ----------
    wm
        Window matrices for one modality.
    query_indices
        Indices into ``wm.D`` for which to produce predictions.
    K
        Number of neighbours.
    sims_full
        Optional precomputed cosine-similarity matrix of shape ``(W, W)``
        (e.g. from ``compute_full_similarity``). If provided, the function
        slices it instead of recomputing — useful when many query batches
        are issued against the same ``wm``.

    Returns
    -------
    y_traj_pred
        Array of shape ``(len(query_indices), L)`` with the predicted
        trajectories. Rows for queries that ended up with all-zero weights
        are filled with NaN.
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    query_indices = np.asarray(query_indices, dtype=np.int64).ravel()
    if query_indices.size == 0:
        return np.empty((0, wm.L), dtype=float)

    n_total = wm.D.shape[0]
    if K > n_total:
        raise ValueError(f"K={K} exceeds number of windows W={n_total}")

    if sims_full is None:
        d_norm = row_normalize(wm.D)
        sims = d_norm[query_indices] @ d_norm.T  # (Q, W)
    else:
        if sims_full.shape != (n_total, n_total):
            raise ValueError(
                f"sims_full has shape {sims_full.shape} but expected "
                f"({n_total}, {n_total})."
            )
        sims = sims_full[query_indices].copy()  # (Q, W); copy because we mask

    same_video = wm.video_id[query_indices][:, None] == wm.video_id[None, :]
    sims = np.where(same_video, -np.inf, sims)

    top_k_idx = np.argpartition(-sims, kth=K - 1, axis=1)[:, :K]
    rows = np.arange(query_indices.size)[:, None]
    top_k_sims = sims[rows, top_k_idx]

    weights = np.clip(top_k_sims, 0.0, None)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weight_sum = weights.sum(axis=1, keepdims=True)

    neighbor_traj = wm.Y_traj[top_k_idx]  # (Q, K, L)
    weighted = (weights[:, :, None] * neighbor_traj).sum(axis=1)
    y_traj_pred = np.full_like(weighted, np.nan, dtype=float)
    valid_query = (weight_sum.ravel() > 0)
    y_traj_pred[valid_query] = weighted[valid_query] / weight_sum[valid_query]

    return y_traj_pred
