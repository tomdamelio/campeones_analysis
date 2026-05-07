"""Test 2 — grid search over tau recovers the true lag.

Setup (from sec 4.2 of the framework explainer):
  20 synthetic videos, 1 modality, X is a Gaussian-smoothed copy of y
  shifted so that X(t) reflects y(t + tau_true), plus white noise.
  L_m fixed to 5 s, three SNR levels (high=4, medium=2, low=1).

Pipeline:
  outer LOVO + inner LOVO grid search over tau, no beta model
  (uniform window weights), no stacking.

Pass criterion (only required at SNR=high):
  >=80% of outer folds pick tau* within +/-1 grid step of tau_true.

The medium and low SNRs are run for diagnostic reporting only — the
validation surface is expected to flatten as SNR drops.

Runs as a standard pytest test or as a stand-alone script
(``python -m tests.multimodal_arousal.test_lag_recovery``).
"""

from __future__ import annotations

import time

import numpy as np

from src.campeones_analysis.multimodal_arousal.nested_cv import nested_lovo_grid_tau
from src.campeones_analysis.multimodal_arousal.simulator import simulate_lag_test


FS = 25
L = 5 * FS                   # 125 samples
STRIDE = L // 2               # 62 samples
TAU_GRID_S = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
TAU_GRID_SAMPLES = [int(round(t * FS)) for t in TAU_GRID_S]
TAU_TRUE_S = 3.0
SNR_LEVELS = [("HIGH", 4.0), ("MEDIUM", 2.0), ("LOW", 1.0)]
PASS_THRESHOLD_HIGH = 0.80


def _run_lag_recovery(snr: float):
    videos, tau_true_samples = simulate_lag_test(
        n_videos=20, fs=FS, snr=snr, tau_true_s=TAU_TRUE_S, seed=42,
    )
    tau_star, val_surface = nested_lovo_grid_tau(
        videos, L=L, stride=STRIDE, tau_grid=TAU_GRID_SAMPLES, K="sqrt",
    )
    return tau_star, val_surface, tau_true_samples


def _accepted_tau_set(tau_grid: list[int], tau_true: int) -> set[int]:
    arr = np.asarray(tau_grid)
    true_idx = int(np.where(arr == tau_true)[0][0])
    lo = max(0, true_idx - 1)
    hi = min(len(arr), true_idx + 2)
    return set(int(t) for t in arr[lo:hi])


def _recovery_pct(tau_star: np.ndarray, accepted: set[int]) -> tuple[int, int, float]:
    n_correct = sum(1 for t in tau_star if int(t) in accepted)
    n_total = len(tau_star)
    return n_correct, n_total, n_correct / n_total


def _print_report(label: str, snr: float, tau_star, val_surface, tau_true) -> float:
    accepted = _accepted_tau_set(TAU_GRID_SAMPLES, tau_true)
    n_correct, n_total, pct = _recovery_pct(tau_star, accepted)
    mean_curve = np.nanmean(val_surface, axis=0)
    std_curve = np.nanstd(val_surface, axis=0)

    print(f"\n=== {label}  (SNR = {snr})  ===")
    print(
        f"tau_true = {tau_true} samples = {tau_true / FS:.2f} s ; "
        f"accepted set (±1 grid) = {sorted(accepted)} samples"
    )
    print(f"tau* per outer fold: {[int(t) for t in tau_star]}")
    print(f"recovery: {n_correct}/{n_total} = {pct:.0%}")
    print("validation surface (mean ± std across outer folds):")
    for tau, mean_s, std_s in zip(TAU_GRID_SAMPLES, mean_curve, std_curve):
        marker = "  <-- tau_true" if tau == tau_true else ""
        print(
            f"  tau = {tau:>3d} samples ({tau / FS:>4.2f} s) :  "
            f"{mean_s:>+.3f} ± {std_s:.3f}{marker}"
        )
    return pct


def _run_all_snrs() -> dict[str, float]:
    results: dict[str, float] = {}
    for label, snr in SNR_LEVELS:
        t0 = time.perf_counter()
        tau_star, val_surface, tau_true = _run_lag_recovery(snr=snr)
        elapsed = time.perf_counter() - t0
        pct = _print_report(label, snr, tau_star, val_surface, tau_true)
        print(f"(elapsed: {elapsed:.1f} s)")
        results[label] = pct
    return results


def test_grid_search_recovers_true_lag_high_snr():
    """Inner-CV grid search over tau recovers tau_true at SNR=high.

    Pass: >=80% of outer folds pick tau* within +/-1 grid step of tau_true
    when SNR=4. The medium/low SNRs are only reported for diagnostic
    purposes — the validation surface is expected to flatten there.
    """
    results = _run_all_snrs()
    pct_high = results["HIGH"]
    assert pct_high >= PASS_THRESHOLD_HIGH, (
        f"SNR=high recovery {pct_high:.0%}, expected "
        f">= {PASS_THRESHOLD_HIGH:.0%}"
    )


if __name__ == "__main__":
    test_grid_search_recovers_true_lag_high_snr()
    print("\nTest 2 PASSED.")
