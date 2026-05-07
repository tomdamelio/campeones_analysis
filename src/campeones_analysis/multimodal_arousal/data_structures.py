"""Window-matrix data structures for the KNN-of-trajectories pipeline.

A ``WindowMatrices`` object holds the unfolded representation of all windows
across all videos for a single modality: feature windows ``D``, target
trajectories ``Y_traj`` (with the per-modality lag already applied),
plus per-window metadata (origin video, start sample, lag, hyperparams).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowMatrices:
    """Unfolded windows + target trajectories for one modality.

    Attributes
    ----------
    D
        Feature windows, shape ``(W, L)``. Row ``j`` is the chunk of the
        modality signal covering samples ``[t_start[j], t_start[j] + L)``
        in video ``video_id[j]``.
    Y_traj
        Target trajectories, shape ``(W, L)``. Row ``j`` is the chunk of
        the target signal covering samples
        ``[t_start[j] + tau, t_start[j] + tau + L)``.
    video_id
        Index of the video each window comes from, shape ``(W,)``.
    t_start
        Start sample of each feature window in its source video, shape ``(W,)``.
    L
        Window length in samples.
    stride
        Step between consecutive windows in samples.
    tau
        Lag in samples applied to the target side (0 in Test 1).
    """

    D: np.ndarray
    Y_traj: np.ndarray
    video_id: np.ndarray
    t_start: np.ndarray
    L: int
    stride: int
    tau: int


def build_window_matrices(
    videos: list[tuple[np.ndarray, np.ndarray]],
    L: int,
    stride: int,
    tau: int = 0,
) -> WindowMatrices:
    """Slide windows across all videos and stack them into ``WindowMatrices``.

    For each video ``v`` with signals ``(X_v, y_v)`` of length ``T_v``,
    enumerate window starts ``t_j = 0, stride, 2*stride, ...`` such that
    BOTH ``t_j + L <= T_v`` (feature window fits) AND
    ``t_j + tau + L <= T_v`` (target trajectory fits with lag).

    Parameters
    ----------
    videos
        List of ``(X_v, y_v)`` pairs from the simulator. Each entry is a
        pair of 1D arrays of equal length ``T_v``.
    L
        Window length in samples.
    stride
        Step between consecutive windows in samples.
    tau
        Lag in samples applied to the target side. Use 0 for Test 1.

    Returns
    -------
    WindowMatrices
        Stacked windows across all videos.
    """
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")

    d_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    vid_ids: list[int] = []
    starts: list[int] = []

    for v_idx, (x_v, y_v) in enumerate(videos):
        if x_v.shape != y_v.shape or x_v.ndim != 1:
            raise ValueError(
                f"video {v_idx}: X_v and y_v must be matching 1D arrays, "
                f"got X_v.shape={x_v.shape}, y_v.shape={y_v.shape}"
            )
        t_v = x_v.shape[0]
        max_start = t_v - max(L, L + tau)
        if max_start < 0:
            continue
        for t_j in range(0, max_start + 1, stride):
            d_rows.append(x_v[t_j : t_j + L])
            y_rows.append(y_v[t_j + tau : t_j + tau + L])
            vid_ids.append(v_idx)
            starts.append(t_j)

    if not d_rows:
        raise ValueError(
            "No windows could be built from the provided videos with "
            f"L={L}, stride={stride}, tau={tau}."
        )

    return WindowMatrices(
        D=np.asarray(d_rows, dtype=float),
        Y_traj=np.asarray(y_rows, dtype=float),
        video_id=np.asarray(vid_ids, dtype=np.int64),
        t_start=np.asarray(starts, dtype=np.int64),
        L=L,
        stride=stride,
        tau=tau,
    )
