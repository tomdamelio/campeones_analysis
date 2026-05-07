"""Sanity test for the beta-model (Etapas 2-3 of the framework).

Setup
-----
20 videos at 25 Hz, 120 s each. The first 60 s of every video have
SNR_high = 4; the last 60 s have SNR_low = 0.5. No lag, no kernel
(``X = y + noise`` with the noise std switching at the segment border).

Pipeline
--------
1. Build window matrices over all 20 videos.
2. Run KNN-of-trajectories with same-video exclusion to obtain
   ``y_traj_pred`` for every window.
3. Compute B1 informativeness ``a`` (Pearson per window).
4. Fit ridge ``β`` mapping ``D`` to ``a``, no intercept.
5. Predict per-window weights ``ŵ = max(D @ β, 0)``.

Checks (asserted)
-----------------
A. ``mean(a[high-SNR]) > mean(a[low-SNR])`` by at least 0.05 absolute —
   the KNN works better in the high-SNR segment. Threshold deliberately
   modest because AR(1) phi=0.95 is so smooth that K~sqrt(W) neighbours
   amortise even SNR=0.1 noise; the framework's β model is meant to
   exploit weaker informativeness gaps than this.
B. ``mean(ŵ[high-SNR]) > mean(ŵ[low-SNR])`` strictly — the beta-model
   recovers the SNR structure linearly from ``D``.
C. ``Pearson(ŵ, a) > 0.3`` on training data — ridge actually learned a
   non-trivial mapping from features to informativeness.
D. Reconstruction with ``weights=ŵ`` on a held-out video keeps Pearson
   above the random baseline (>0.5) — the weighted aggregation does
   not destroy the signal.

Runs as standard pytest test or as stand-alone script
(``python -m tests.multimodal_arousal.test_beta_model``).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import pearsonr

from src.campeones_analysis.multimodal_arousal.beta_model import (
    compute_b1_informativeness,
    fit_beta_ridge,
    predict_window_weights,
)
from src.campeones_analysis.multimodal_arousal.data_structures import (
    build_window_matrices,
)
from src.campeones_analysis.multimodal_arousal.knn_trajectories import (
    compute_full_similarity,
    knn_trajectory_predict,
)
from src.campeones_analysis.multimodal_arousal.reconstruction import (
    reconstruct_stage_a,
)
from src.campeones_analysis.multimodal_arousal.simulator import (
    simulate_heterogeneous_snr_test,
)


FS = 25
L = 5 * FS
STRIDE = L // 2
ALPHA_RIDGE = 0.1
N_VIDEOS = 20


def _classify_windows(t_start: np.ndarray, L: int, high_n: int):
    """Return boolean masks for windows fully inside high-SNR / low-SNR."""
    t_end = t_start + L
    is_high = t_end <= high_n
    is_low = t_start >= high_n
    return is_high, is_low


def _run_pipeline():
    videos, high_n = simulate_heterogeneous_snr_test(
        n_videos=N_VIDEOS,
        fs=FS,
        duration_s=120.0,
        snr_high=4.0,
        snr_low=0.5,
        high_snr_duration_s=60.0,
        seed=42,
    )

    wm = build_window_matrices(videos, L=L, stride=STRIDE, tau=0)
    n_total = wm.D.shape[0]
    K = max(1, int(np.sqrt(n_total)))

    sims_full = compute_full_similarity(wm)
    all_indices = np.arange(n_total)
    y_traj_pred = knn_trajectory_predict(
        wm, all_indices, K=K, sims_full=sims_full
    )

    a = compute_b1_informativeness(y_traj_pred, wm.Y_traj)
    beta, intercept = fit_beta_ridge(wm.D, a, alpha_ridge=ALPHA_RIDGE)
    w_hat = predict_window_weights(wm.D, beta, intercept)

    return videos, wm, sims_full, y_traj_pred, a, beta, intercept, w_hat, high_n


def _print_diagnostics(a, w_hat, wm, high_n):
    is_high, is_low = _classify_windows(wm.t_start, L, high_n)
    a_high = a[is_high & np.isfinite(a)]
    a_low = a[is_low & np.isfinite(a)]
    w_high = w_hat[is_high]
    w_low = w_hat[is_low]

    print(f"Total windows: {wm.D.shape[0]}  "
          f"(high-SNR: {is_high.sum()}, low-SNR: {is_low.sum()}, "
          f"boundary: {(~is_high & ~is_low).sum()})")
    print(f"a  | mean: high = {a_high.mean():+.3f}  "
          f"low = {a_low.mean():+.3f}  Δ = {a_high.mean() - a_low.mean():+.3f}")
    print(f"a  | std : high = {a_high.std():.3f}   low = {a_low.std():.3f}")
    print(f"ŵ  | mean: high = {w_high.mean():.3f}  "
          f"low = {w_low.mean():.3f}  Δ = {w_high.mean() - w_low.mean():+.3f}")
    print(f"ŵ  | frac>0: high = {(w_high > 0).mean():.2f}  "
          f"low = {(w_low > 0).mean():.2f}")

    valid = np.isfinite(a) & np.isfinite(w_hat)
    r_train, _ = pearsonr(w_hat[valid], a[valid])
    print(f"Pearson(ŵ, a) on training: {r_train:.3f}")

    return a_high, a_low, w_high, w_low, r_train


def _reconstruct_one_video(
    wm,
    sims_full,
    videos,
    beta,
    intercept: float,
    v_test: int,
):
    """Reconstruct ŷ for video v_test using KNN + beta-derived weights."""
    is_test = wm.video_id == v_test
    query_indices = np.where(is_test)[0]
    n_train = wm.D.shape[0] - int(is_test.sum())
    K = max(1, int(np.sqrt(n_train)))

    y_traj_pred = knn_trajectory_predict(
        wm, query_indices, K=K, sims_full=sims_full
    )
    w_hat_test = predict_window_weights(wm.D[is_test], beta, intercept)
    y_v = videos[v_test][1]
    y_hat = reconstruct_stage_a(
        y_traj_pred=y_traj_pred,
        t_start=wm.t_start[is_test],
        L=L,
        n_samples=len(y_v),
        weights=w_hat_test,
        tau=0,
    )
    valid = ~np.isnan(y_hat)
    if valid.sum() < 3:
        return None
    r, _ = pearsonr(y_hat[valid], y_v[valid])
    return float(r)


def test_beta_model_recovers_snr_structure():
    """β-model recovers high-vs-low-SNR structure from D and Pearson(ŵ,a)>0."""
    t0 = time.perf_counter()
    (
        videos,
        wm,
        sims_full,
        y_traj_pred,
        a,
        beta,
        intercept,
        w_hat,
        high_n,
    ) = _run_pipeline()
    elapsed = time.perf_counter() - t0
    print(f"\n=== Beta-model sanity test ===")
    print(f"(elapsed: {elapsed:.1f} s)")
    print(f"intercept = {intercept:+.3f}")

    a_high, a_low, w_high, w_low, r_train = _print_diagnostics(
        a, w_hat, wm, high_n
    )

    print("\nReconstruction Pearson per fold (β-weighted):")
    rs_weighted = []
    for v_test in range(N_VIDEOS):
        r = _reconstruct_one_video(wm, sims_full, videos, beta, intercept, v_test)
        if r is not None:
            rs_weighted.append(r)
    print(f"  per-fold: {[round(r, 3) for r in rs_weighted]}")
    print(f"  mean = {np.mean(rs_weighted):.3f}")

    # Check A: a is higher in high-SNR segment
    assert a_high.mean() - a_low.mean() > 0.05, (
        f"a Δ = {a_high.mean() - a_low.mean():.3f}, "
        f"expected >0.05 (high-SNR should beat low-SNR by clear margin)."
    )
    # Check B: ŵ recovers the structure
    assert w_high.mean() > w_low.mean(), (
        f"ŵ mean high = {w_high.mean():.3f}, low = {w_low.mean():.3f}; "
        f"the β-model failed to recover the SNR structure."
    )
    # Check C: training-set learning sanity
    assert r_train > 0.3, (
        f"Pearson(ŵ, a) on training = {r_train:.3f}, expected > 0.3."
    )
    # Check D: reconstruction with β-weights stays useful
    mean_r = float(np.mean(rs_weighted))
    assert mean_r > 0.5, (
        f"Reconstruction Pearson with β-weights = {mean_r:.3f}, "
        f"expected > 0.5."
    )


if __name__ == "__main__":
    test_beta_model_recovers_snr_structure()
    print("\nBeta-model sanity test PASSED.")
