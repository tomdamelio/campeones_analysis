#!/usr/bin/env python
"""Simulate alpha residual in trial-averaged ERP.

Implements Task 8.1 from the research diary.

Demonstrates that averaging N trials of ongoing alpha (8-12 Hz) with
random phase does NOT fully cancel the oscillation when N is small,
producing a residual that looks like an evoked response.

Simulations
-----------
1. **Basic replication** of Fede Poncio's simulation, scaled to our
   recording parameters (fs=500 Hz, duration=7.5 s, N=74 trials).
2. **N sweep**: residual amplitude as a function of N (20–300).
3. **Amplitude asymmetry** (Mazaheri & Jensen 2008): peaks > troughs
   produce a DC-shift in the average that does not cancel even at large N.
4. **Comparison**: symmetric vs asymmetric alpha residual across N.

Usage
-----
    python scripts/validation/28_alpha_residual_simulation.py [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "alpha_simulation"

# ---------------------------------------------------------------------------
# Signal parameters (matched to real data)
# ---------------------------------------------------------------------------
FS = 500          # Hz — same as preprocessed EEG
DURATION = 7.5    # seconds — epoch length (TMIN=-4.5, TMAX=3.0)
N_SAMPLES = int(FS * DURATION)
TIMES = np.arange(N_SAMPLES) / FS  # 0 .. 7.5 s
TIMES_MS = (TIMES - 4.5) * 1000    # -4500 .. 3000 ms (epoch-relative)

# Alpha band components (Hz)
ALPHA_FREQS = [8, 9, 10, 11, 12]
ALPHA_AMPS  = [3.0, 4.0, 10.0, 5.0, 2.0]  # relative amplitudes

# High-frequency noise band (Hz) — mimics beta/gamma
HF_RANGE = (20, 45)
HF_N_COMPONENTS = 4
HF_AMP = 1.5

# White noise
NOISE_SCALE = 0.5

# N values to sweep
N_TRIALS_LIST = [20, 50, 74, 150, 300]

# Asymmetry parameter: positive shifts peaks up relative to troughs
ASYMMETRY_FACTOR = 0.35  # fraction of alpha amplitude added as rectification


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_alpha(
    n_trials: int,
    rng: np.random.Generator,
    asymmetric: bool = False,
) -> NDArray[np.float64]:
    """Generate n_trials of alpha-dominated signals with random phase.

    Parameters
    ----------
    n_trials : int
    rng : numpy Generator
    asymmetric : bool
        If True, apply half-wave rectification bias so peaks > |troughs|.

    Returns
    -------
    signals : (n_trials, N_SAMPLES)
    """
    t = np.arange(N_SAMPLES) / FS  # (N_SAMPLES,)
    freqs_arr = np.array(ALPHA_FREQS)
    amps_arr = np.array(ALPHA_AMPS)

    # Build alpha by accumulating per-frequency (avoids 3D array)
    phases = rng.uniform(0, 2 * np.pi, size=(n_trials, len(ALPHA_FREQS)))
    alpha = np.zeros((n_trials, N_SAMPLES))
    for i, (freq, amp) in enumerate(zip(freqs_arr, amps_arr)):
        alpha += amp * np.sin(2 * np.pi * freq * t[None, :] + phases[:, i, None])

    if asymmetric:
        alpha += ASYMMETRY_FACTOR * np.clip(alpha, 0, None)

    # High-frequency components (accumulate per-component)
    hf_freqs = rng.uniform(HF_RANGE[0], HF_RANGE[1],
                           size=(n_trials, HF_N_COMPONENTS))
    hf_phases = rng.uniform(0, 2 * np.pi,
                            size=(n_trials, HF_N_COMPONENTS))
    hf = np.zeros((n_trials, N_SAMPLES))
    for j in range(HF_N_COMPONENTS):
        hf += (HF_AMP / HF_N_COMPONENTS) * np.sin(
            2 * np.pi * hf_freqs[:, j, None] * t[None, :] + hf_phases[:, j, None]
        )

    # White noise
    noise = rng.normal(scale=NOISE_SCALE, size=(n_trials, N_SAMPLES))

    return alpha + hf + noise



def compute_residual_amplitude(signals: NDArray[np.float64]) -> float:
    """RMS of the trial-average (proxy for residual oscillation strength)."""
    avg = signals.mean(axis=0)
    return float(np.sqrt(np.mean(avg ** 2)))


def compute_fft(signal_1d: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Return (freqs, magnitude) for a 1-D signal."""
    n = len(signal_1d)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    mag = np.abs(np.fft.rfft(signal_1d)) / n
    return freqs, mag


# ---------------------------------------------------------------------------
# Figure 1: Basic replication (Fede's simulation scaled to our params)
# ---------------------------------------------------------------------------

def fig1_basic_replication(rng: np.random.Generator, out: Path) -> None:
    """Single trials + average + FFTs for N=74 symmetric alpha."""
    print("  Fig 1: Basic replication (N=74, symmetric) ...")
    signals = generate_alpha(74, rng, asymmetric=False)
    avg = signals.mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) 10 single trials
    ax = axes[0, 0]
    for i in range(10):
        ax.plot(TIMES_MS, signals[i], alpha=0.4, lw=0.5)
    ax.set_title("10 single trials (symmetric alpha)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.axvline(0, color="k", ls="--", lw=1)

    # (b) Trial average
    ax = axes[0, 1]
    ax.plot(TIMES_MS, avg, color="C3", lw=1.5)
    ax.set_title(f"Trial average (N=74) — RMS={compute_residual_amplitude(signals):.3f}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.axvline(0, color="k", ls="--", lw=1)

    # (c) FFT of average
    freqs, mag_avg = compute_fft(avg)
    ax = axes[1, 0]
    ax.plot(freqs, mag_avg, color="C3")
    ax.set_xlim(0, 50)
    ax.set_title("FFT of trial average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    # (d) FFT of single trial
    freqs, mag_single = compute_fft(signals[0])
    ax = axes[1, 1]
    ax.plot(freqs, mag_single, color="C0", alpha=0.7)
    ax.set_xlim(0, 50)
    ax.set_title("FFT of single trial #0")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    fig.suptitle(
        "Simulation 1: Alpha residual in trial average (symmetric, N=74, fs=500 Hz)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_basic_replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: N sweep — residual amplitude vs number of trials
# ---------------------------------------------------------------------------

def fig2_n_sweep(rng: np.random.Generator, out: Path) -> None:
    """Residual RMS as a function of N, with theoretical 1/sqrt(N) curve."""
    print("  Fig 2: N sweep ...")
    n_repeats = 10

    results = {n: [] for n in N_TRIALS_LIST}
    for n in N_TRIALS_LIST:
        for _ in range(n_repeats):
            sig = generate_alpha(n, rng, asymmetric=False)
            results[n].append(compute_residual_amplitude(sig))

    means = [np.mean(results[n]) for n in N_TRIALS_LIST]
    stds = [np.std(results[n]) for n in N_TRIALS_LIST]

    # Theoretical: residual ~ A / sqrt(N)
    A_fit = means[0] * np.sqrt(N_TRIALS_LIST[0])
    theoretical = [A_fit / np.sqrt(n) for n in N_TRIALS_LIST]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(N_TRIALS_LIST, means, yerr=stds, fmt="o-", color="C3",
                capsize=4, label="Simulated (symmetric)")
    ax.plot(N_TRIALS_LIST, theoretical, "k--", alpha=0.6,
            label=r"Theoretical $\propto 1/\sqrt{N}$")
    ax.set_xlabel("Number of trials (N)")
    ax.set_ylabel("Residual RMS (a.u.)")
    ax.set_title("Alpha residual amplitude vs N (symmetric)")
    ax.legend()
    ax.set_xticks(N_TRIALS_LIST)
    fig.tight_layout()
    fig.savefig(out / "fig2_n_sweep_symmetric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Asymmetric alpha — single example at N=74
# ---------------------------------------------------------------------------

def fig3_asymmetric_example(rng: np.random.Generator, out: Path) -> None:
    """Compare symmetric vs asymmetric alpha average at N=74."""
    print("  Fig 3: Asymmetric vs symmetric (N=74) ...")
    sig_sym = generate_alpha(74, rng, asymmetric=False)
    sig_asym = generate_alpha(74, rng, asymmetric=True)

    avg_sym = sig_sym.mean(axis=0)
    avg_asym = sig_asym.mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Single trial comparison
    ax = axes[0, 0]
    ax.plot(TIMES_MS, sig_sym[0], alpha=0.7, label="Symmetric", color="C0")
    ax.plot(TIMES_MS, sig_asym[0], alpha=0.7, label="Asymmetric", color="C1")
    ax.set_title("Single trial: symmetric vs asymmetric")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", ls="--", lw=1)

    # (b) Trial averages
    ax = axes[0, 1]
    rms_s = compute_residual_amplitude(sig_sym)
    rms_a = compute_residual_amplitude(sig_asym)
    ax.plot(TIMES_MS, avg_sym, label=f"Symmetric (RMS={rms_s:.3f})", color="C0")
    ax.plot(TIMES_MS, avg_asym, label=f"Asymmetric (RMS={rms_a:.3f})", color="C1")
    ax.set_title("Trial average (N=74)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", ls="--", lw=1)

    # (c) FFT of symmetric average
    freqs, mag = compute_fft(avg_sym)
    ax = axes[1, 0]
    ax.plot(freqs, mag, color="C0")
    ax.set_xlim(0, 50)
    ax.set_title("FFT of symmetric average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    # (d) FFT of asymmetric average
    freqs, mag = compute_fft(avg_asym)
    ax = axes[1, 1]
    ax.plot(freqs, mag, color="C1")
    ax.set_xlim(0, 50)
    ax.set_title("FFT of asymmetric average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    fig.suptitle(
        "Simulation 3: Symmetric vs asymmetric alpha (Mazaheri & Jensen 2008)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "fig3_asymmetric_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: N sweep — symmetric vs asymmetric comparison
# ---------------------------------------------------------------------------

def fig4_comparison_n_sweep(rng: np.random.Generator, out: Path) -> None:
    """Residual RMS vs N for both symmetric and asymmetric alpha."""
    print("  Fig 4: N sweep comparison (symmetric vs asymmetric) ...")
    n_repeats = 10

    res_sym = {n: [] for n in N_TRIALS_LIST}
    res_asym = {n: [] for n in N_TRIALS_LIST}

    for n in N_TRIALS_LIST:
        for _ in range(n_repeats):
            s = generate_alpha(n, rng, asymmetric=False)
            res_sym[n].append(compute_residual_amplitude(s))
            a = generate_alpha(n, rng, asymmetric=True)
            res_asym[n].append(compute_residual_amplitude(a))

    mean_sym = [np.mean(res_sym[n]) for n in N_TRIALS_LIST]
    std_sym = [np.std(res_sym[n]) for n in N_TRIALS_LIST]
    mean_asym = [np.mean(res_asym[n]) for n in N_TRIALS_LIST]
    std_asym = [np.std(res_asym[n]) for n in N_TRIALS_LIST]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(N_TRIALS_LIST, mean_sym, yerr=std_sym, fmt="o-", color="C0",
                capsize=4, label="Symmetric")
    ax.errorbar(N_TRIALS_LIST, mean_asym, yerr=std_asym, fmt="s-", color="C1",
                capsize=4, label="Asymmetric")

    # Theoretical 1/sqrt(N) for symmetric
    A_fit = mean_sym[0] * np.sqrt(N_TRIALS_LIST[0])
    theoretical = [A_fit / np.sqrt(n) for n in N_TRIALS_LIST]
    ax.plot(N_TRIALS_LIST, theoretical, "k--", alpha=0.5,
            label=r"$\propto 1/\sqrt{N}$ (symmetric)")

    ax.set_xlabel("Number of trials (N)")
    ax.set_ylabel("Residual RMS (a.u.)")
    ax.set_title("Alpha residual: symmetric vs asymmetric across N")
    ax.legend()
    ax.set_xticks(N_TRIALS_LIST)
    fig.tight_layout()
    fig.savefig(out / "fig4_comparison_n_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Detailed asymmetry mechanism — peak/trough histogram
# ---------------------------------------------------------------------------

def fig5_asymmetry_mechanism(rng: np.random.Generator, out: Path) -> None:
    """Show how amplitude asymmetry creates a non-zero mean after averaging.

    Plots:
    (a) Histogram of peak and trough amplitudes for symmetric vs asymmetric.
    (b) Running mean across trials (convergence plot) for both conditions.
    """
    print("  Fig 5: Asymmetry mechanism detail ...")
    n_big = 300
    sig_sym = generate_alpha(n_big, rng, asymmetric=False)
    sig_asym = generate_alpha(n_big, rng, asymmetric=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Peak/trough distributions
    ax = axes[0]
    for label, sig, colors in [
        ("Symmetric", sig_sym, ("C0", "C9")),
        ("Asymmetric", sig_asym, ("C1", "C4")),
    ]:
        peaks = sig.max(axis=1)
        troughs = sig.min(axis=1)
        ax.hist(peaks, bins=30, alpha=0.4, color=colors[0],
                label=f"{label} peaks")
        ax.hist(np.abs(troughs), bins=30, alpha=0.4, color=colors[1],
                label=f"{label} |troughs|")
    ax.set_xlabel("Amplitude (a.u.)")
    ax.set_ylabel("Count")
    ax.set_title("Peak vs |trough| amplitude distribution")
    ax.legend(fontsize=7)

    # (b) Running mean convergence — sample at intervals to avoid slow loop
    ax = axes[1]
    sample_points = list(range(2, 51)) + list(range(55, n_big + 1, 5))
    running_rms_sym = [
        compute_residual_amplitude(sig_sym[:k])
        for k in sample_points
    ]
    running_rms_asym = [
        compute_residual_amplitude(sig_asym[:k])
        for k in sample_points
    ]
    ax.plot(sample_points, running_rms_sym, color="C0", label="Symmetric", alpha=0.8)
    ax.plot(sample_points, running_rms_asym, color="C1", label="Asymmetric", alpha=0.8)
    ax.set_xlabel("Number of trials averaged")
    ax.set_ylabel("Residual RMS")
    ax.set_title("Convergence of residual with increasing N")
    ax.legend()

    fig.suptitle(
        "Asymmetry mechanism (Mazaheri & Jensen 2008)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "fig5_asymmetry_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha residual simulation (Task 8.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("28 — Alpha residual simulation")
    print(f"     seed={args.seed}, fs={FS} Hz, duration={DURATION} s")
    print(f"     output: {RESULTS_ROOT}")
    print("=" * 60)

    fig1_basic_replication(rng, RESULTS_ROOT)
    fig2_n_sweep(rng, RESULTS_ROOT)
    fig3_asymmetric_example(rng, RESULTS_ROOT)
    fig4_comparison_n_sweep(rng, RESULTS_ROOT)
    fig5_asymmetry_mechanism(rng, RESULTS_ROOT)

    print(f"\nAll figures saved to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
