"""Test 1 — smoke test of the KNN-of-trajectories + Stage A pipeline.

Trivial scenario: X(t) = y(t) + white noise (no lag, no convolution),
1 modality, 10 synthetic videos. Pass criterion:
    Pearson(ŷ, y) > 0.7 averaged over 10 LOVO folds.

Runs as a standard pytest test or as a stand-alone script
(``python -m tests.multimodal_arousal.test_smoke_knn``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from src.campeones_analysis.multimodal_arousal.data_structures import (
    build_window_matrices,
)
from src.campeones_analysis.multimodal_arousal.knn_trajectories import (
    knn_trajectory_predict,
)
from src.campeones_analysis.multimodal_arousal.reconstruction import (
    reconstruct_stage_a,
)
from src.campeones_analysis.multimodal_arousal.simulator import simulate_smoke_test


def _run_smoke() -> tuple[float, list[float]]:
    fs = 25
    L = 5 * fs       # 125 samples
    stride = L // 2  # 62 samples

    videos = simulate_smoke_test(n_videos=10, fs=fs, snr=1.0, seed=42)
    wm = build_window_matrices(videos, L=L, stride=stride, tau=0)
    K = int(np.sqrt(wm.D.shape[0]))

    pearsons: list[float] = []
    for v_test in range(10):
        is_test = wm.video_id == v_test
        query_indices = np.where(is_test)[0]

        y_traj_pred = knn_trajectory_predict(wm, query_indices, K=K)

        y_true = videos[v_test][1]
        y_hat = reconstruct_stage_a(
            y_traj_pred=y_traj_pred,
            t_start=wm.t_start[is_test],
            L=L,
            n_samples=len(y_true),
            weights=None,
            tau=0,
        )

        valid = ~np.isnan(y_hat)
        r, _ = pearsonr(y_hat[valid], y_true[valid])
        pearsons.append(float(r))

    mean_p = float(np.mean(pearsons))
    return mean_p, pearsons


def test_smoke_knn_recovers_trivial_target():
    """KNN-of-trajectories + Stage A recover y(t) when X = y + noise."""
    mean_p, pearsons = _run_smoke()
    print(f"Pearson por fold: {[round(p, 3) for p in pearsons]}")
    print(f"Pearson promedio: {mean_p:.3f}")
    assert mean_p > 0.7, f"Pearson promedio = {mean_p:.3f}, esperado > 0.7"


if __name__ == "__main__":
    mean_p, pearsons = _run_smoke()
    print(f"Pearson por fold: {[round(p, 3) for p in pearsons]}")
    print(f"Pearson promedio: {mean_p:.3f}")
    assert mean_p > 0.7, f"Pearson promedio = {mean_p:.3f}, esperado > 0.7"
    print("Test 1 PASSED.")
