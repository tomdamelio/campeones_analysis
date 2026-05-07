"""Test 3 — multimodal stacking calibrates alpha_m correctly.

Setup (sec 4.3 of the framework explainer)
------------------------------------------
3 modalities, 20 videos. ``y(t) = AR(1)``.
``X_m(t) = (gauss_kernel(sigma=1 s) * y(t + tau_m))(t) + eps_m`` with
``(tau_1, tau_2, tau_3) = (2, 3, 4) s`` and ``SNR_m = (2, 1, 0.5)`` —
modality 1 is structurally the most informative, modality 3 the least.

Pipeline (sec 4.7)
------------------
Outer LOVO + inner LOVO grid search over tau per modality
+ re-OOF post-selection with K=5 distinct folds
+ B-stack-OLS over the re-OOF predictions
+ refit + predict on the held-out outer fold.

Pass criteria
-------------
A. ``alpha_1 > alpha_2 > alpha_3`` in >= 75 % of outer folds (the
   stacking respects the SNR ordering).
B. ``|alpha_3| < 0.5 * |alpha_1|`` in median across outer folds (the
   weakest modality is meaningfully down-weighted).
C. Outer-fold Pearson(y_hat, y) is significantly above zero (the
   pipeline does NOT degrade to noise).

Diagnostics (reported, not asserted)
------------------------------------
- ``tau_m*`` recovery: how often per-modality grid search picks the
  true lag (within +/- 1 grid step).
- ``kappa`` and per-modality VIF on the OOF design matrix; sec 4.7
  recommends switching to ridge / constrained when ``kappa > 30`` or
  ``VIF > 5``.

Runs as standard pytest test or stand-alone script:
``micromamba run -n campeones python -m tests.multimodal_arousal.test_multimodal_stacking``
"""

from __future__ import annotations

import time

import numpy as np

from src.campeones_analysis.multimodal_arousal.nested_cv import (
    run_test3_outer_loop,
)
from src.campeones_analysis.multimodal_arousal.simulator import (
    simulate_multimodal_test,
)


FS = 25
L = 5 * FS
STRIDE = L // 2
TAU_GRID_S = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
TAU_GRID_SAMPLES = [int(round(t * FS)) for t in TAU_GRID_S]
TAU_TRUE_S = (2.0, 3.0, 4.0)
SNR_TRUE = (2.0, 1.0, 0.5)
N_VIDEOS = 20
N_KFOLD = 5
ALPHA_RIDGE = 0.1


def _grid_neighbour_set(tau_grid: list[int], tau_true: int) -> set[int]:
    """Tau values within +/- 1 grid step of tau_true."""
    arr = np.asarray(tau_grid)
    idx = int(np.where(arr == tau_true)[0][0])
    lo = max(0, idx - 1)
    hi = min(len(arr), idx + 2)
    return set(int(t) for t in arr[lo:hi])


def _print_report(results: dict, tau_true_samples: np.ndarray) -> tuple[float, float]:
    """Print Test 3 diagnostics. Returns (frac_correct_order, alpha_ratio_median)."""
    tau_stars = results["tau_stars"]
    alpha_m = results["alpha_m"]
    alpha_0 = results["alpha_0"]
    kappa = results["kappa"]
    vif = results["vif"]
    outer_scores = results["outer_scores"]
    per_mod_scores = results["per_mod_scores"]
    M = alpha_m.shape[1]

    print("\n=== Per-fold tau* recovery (within +/- 1 grid step) ===")
    for m in range(M):
        accepted = _grid_neighbour_set(TAU_GRID_SAMPLES, int(tau_true_samples[m]))
        n_correct = sum(1 for t in tau_stars[:, m] if int(t) in accepted)
        pct = n_correct / len(tau_stars)
        print(
            f"  modality {m + 1}: tau_true = {tau_true_samples[m]} samples; "
            f"recovered in {n_correct}/{len(tau_stars)} folds ({pct:.0%})"
        )

    print("\n=== Stacking diagnostics across outer folds ===")
    print(
        f"  kappa : median = {np.nanmedian(kappa):.2f},  "
        f"q25 = {np.nanquantile(kappa, 0.25):.2f},  "
        f"q75 = {np.nanquantile(kappa, 0.75):.2f}"
    )
    for m in range(M):
        print(
            f"  VIF[{m + 1}] : median = {np.nanmedian(vif[:, m]):.2f},  "
            f"q25 = {np.nanquantile(vif[:, m], 0.25):.2f},  "
            f"q75 = {np.nanquantile(vif[:, m], 0.75):.2f}"
        )

    print("\n=== Stacking weights alpha_m across outer folds ===")
    for m in range(M):
        valid = np.isfinite(alpha_m[:, m])
        if not valid.any():
            continue
        print(
            f"  alpha[{m + 1}] : mean = {alpha_m[valid, m].mean():+.3f},  "
            f"median = {np.median(alpha_m[valid, m]):+.3f},  "
            f"std = {alpha_m[valid, m].std():.3f}"
        )
    valid_int = np.isfinite(alpha_0)
    if valid_int.any():
        print(
            f"  alpha_0  : mean = {alpha_0[valid_int].mean():+.3f},  "
            f"median = {np.median(alpha_0[valid_int]):+.3f}"
        )

    valid_alpha = np.isfinite(alpha_m).all(axis=1)
    if valid_alpha.any():
        ordered = (
            (alpha_m[valid_alpha, 0] > alpha_m[valid_alpha, 1])
            & (alpha_m[valid_alpha, 1] > alpha_m[valid_alpha, 2])
        )
        frac_ordered = float(ordered.mean())
        ratio = np.abs(alpha_m[valid_alpha, 2]) / np.abs(alpha_m[valid_alpha, 0])
        ratio = ratio[np.isfinite(ratio)]
        ratio_median = float(np.median(ratio)) if ratio.size > 0 else float("inf")
    else:
        frac_ordered = 0.0
        ratio_median = float("inf")

    print(
        f"\n  alpha_1 > alpha_2 > alpha_3 in "
        f"{int(frac_ordered * valid_alpha.sum())}/{int(valid_alpha.sum())} "
        f"folds ({frac_ordered:.0%})"
    )
    print(f"  median |alpha_3 / alpha_1| = {ratio_median:.3f}")

    print("\n=== Outer-fold scores ===")
    for m in range(M):
        valid = np.isfinite(per_mod_scores[:, m])
        if valid.any():
            print(
                f"  per-modality {m + 1}  Pearson: "
                f"mean = {per_mod_scores[valid, m].mean():.3f},  "
                f"median = {np.median(per_mod_scores[valid, m]):.3f}"
            )
    valid = np.isfinite(outer_scores)
    if valid.any():
        print(
            f"  stacked        Pearson: "
            f"mean = {outer_scores[valid].mean():.3f},  "
            f"median = {np.median(outer_scores[valid]):.3f},  "
            f"min = {outer_scores[valid].min():.3f}"
        )

    return frac_ordered, ratio_median


def test_multimodal_stacking_calibrates_alpha():
    """B-stack-OLS recovers SNR ordering of alpha_m on multimodal sim."""
    t0 = time.perf_counter()
    videos_per_mod, tau_true_samples = simulate_multimodal_test(
        n_videos=N_VIDEOS,
        fs=FS,
        tau_true_s=TAU_TRUE_S,
        snr=SNR_TRUE,
        seed=42,
    )
    print(
        f"\nMultimodal sim: {N_VIDEOS} videos, M=3 modalities; "
        f"tau_true = {tau_true_samples.tolist()} samples; SNR = {SNR_TRUE}."
    )

    results = run_test3_outer_loop(
        videos_per_mod,
        L=L,
        stride=STRIDE,
        tau_grid=TAU_GRID_SAMPLES,
        K="sqrt",
        alpha_ridge=ALPHA_RIDGE,
        n_kfold=N_KFOLD,
        stack_variant="ols",
        seed=42,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n(elapsed: {elapsed:.1f} s)")

    frac_ordered, ratio_median = _print_report(results, tau_true_samples)

    # Pass A: ordering in >= 75% of outer folds
    assert frac_ordered >= 0.75, (
        f"alpha_1 > alpha_2 > alpha_3 in only {frac_ordered:.0%} of folds, "
        f"expected >= 75%."
    )
    # Pass B: |alpha_3| < 0.5 * |alpha_1| in median
    assert ratio_median < 0.5, (
        f"median |alpha_3 / alpha_1| = {ratio_median:.3f}, expected < 0.5."
    )
    # Pass C: stacked Pearson is positive on average (sanity)
    valid = np.isfinite(results["outer_scores"])
    mean_pearson = float(results["outer_scores"][valid].mean()) if valid.any() else 0.0
    assert mean_pearson > 0.5, (
        f"Stacked outer-fold Pearson = {mean_pearson:.3f}, expected > 0.5."
    )


if __name__ == "__main__":
    test_multimodal_stacking_calibrates_alpha()
    print("\nTest 3 PASSED.")
