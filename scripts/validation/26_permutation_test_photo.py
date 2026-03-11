#!/usr/bin/env python
"""Cluster-based permutation test: CHANGE_PHOTO vs NO_CHANGE_PHOTO.

Implements Task 2 from the research diary.

Uses MNE's spatio-temporal cluster permutation test to compare ERP and TFR
between conditions, correcting for multiple comparisons across channels and
time points (and frequencies for TFR).

Usage
-----
    python scripts/validation/26_permutation_test_photo.py --subject 27
    python scripts/validation/26_permutation_test_photo.py --subject 27 --n-perm 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats as scipy_stats
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
from mne.time_frequency import tfr_morlet

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EPOCHS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_epochs"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_stats"

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

# TFR parameters (same as 25_erp_tfr_photo_contrast.py)
TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)
TFR_N_CYCLES = TFR_FREQS / 2.0

# Epoch parameters
BASELINE = (-1.0, -0.5)

# Default permutation settings
DEFAULT_N_PERM = 1000
RANDOM_SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
# ERP cluster permutation per ROI (Task 2.2 / 2.3 / 2.4)
# ---------------------------------------------------------------------------

def run_erp_permutation(
    epochs: mne.Epochs,
    output_dir: Path,
    subject: str,
    n_perm: int,
    seed: int,
) -> dict:
    """Run cluster-based permutation test on ERP per ROI.

    For each ROI, averages channels within the ROI, then runs a 1D
    temporal cluster permutation test (time only).

    Returns dict of {roi_name: {t_obs, clusters, cluster_pv, H0, sig_clusters}}.
    """
    results = {}

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs:
            continue

        # Shape: (n_epochs, n_times) — averaged across ROI channels, in uV
        data_change = (
            epochs["CHANGE_PHOTO"].copy().pick(valid_chs)
            .get_data().mean(axis=1) * 1e6
        )
        data_nochange = (
            epochs["NO_CHANGE_PHOTO"].copy().pick(valid_chs)
            .get_data().mean(axis=1) * 1e6
        )

        # Cluster-forming threshold: t-value for p < 0.05 (two-tailed)
        df = data_change.shape[0] + data_nochange.shape[0] - 2
        t_thresh = scipy_stats.t.ppf(1 - ALPHA / 2, df)

        # permutation_cluster_test expects list of (n_obs, n_times)
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
            [data_change, data_nochange],
            n_permutations=n_perm,
            threshold=t_thresh,
            tail=0,
            seed=seed,
            verbose=False,
        )

        sig_mask = cluster_pv < ALPHA
        n_sig = sig_mask.sum()
        print(f"  ERP {roi_name}: {len(clusters)} clusters, "
              f"{n_sig} significant (p < {ALPHA})")

        results[roi_name] = {
            "t_obs": t_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "H0": H0,
            "n_sig": n_sig,
        }

        # --- Plot ---
        times_ms = epochs.times * 1000
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        # Top: ERP waveforms
        ax = axes[0]
        for data, cond, color in [
            (data_change, "CHANGE_PHOTO", "C3"),
            (data_nochange, "NO_CHANGE_PHOTO", "C2"),
        ]:
            mean = data.mean(axis=0)
            sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            ax.plot(times_ms, mean, label=f"{cond} (n={data.shape[0]})",
                    color=color)
            ax.fill_between(times_ms, mean - sem, mean + sem,
                            color=color, alpha=0.25)
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.invert_yaxis()
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title(f"ERP permutation test: {roi_name} (sub-{subject})")
        ax.legend(fontsize=8)

        # Bottom: significant clusters highlighted
        ax2 = axes[1]
        ax2.plot(times_ms, t_obs, color="k", linewidth=0.8)
        ax2.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax2.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax2.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)

        for i_c, (cl, pv) in enumerate(zip(clusters, cluster_pv)):
            if pv < ALPHA:
                cl_times = times_ms[cl[0]]  # cl is a tuple of arrays
                ax2.axvspan(cl_times[0], cl_times[-1],
                            color="salmon", alpha=0.4,
                            label=f"cluster p={pv:.4f}" if i_c == 0 else None)

        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("t-statistic")
        if n_sig > 0:
            ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_erp_perm_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# TFR cluster permutation per ROI (Task 2.2 / 2.3 / 2.4)
# ---------------------------------------------------------------------------

def run_tfr_permutation(
    epochs: mne.Epochs,
    output_dir: Path,
    subject: str,
    n_perm: int,
    seed: int,
) -> dict:
    """Run cluster-based permutation test on TFR power per ROI.

    For each ROI, averages TFR across channels, then runs a 2D
    (freq x time) cluster permutation test.

    Returns dict of {roi_name: {t_obs, clusters, cluster_pv, n_sig}}.
    """
    # Compute single-trial TFR power for each condition
    print("  Computing single-trial TFR for CHANGE_PHOTO ...")
    tfr_change = tfr_morlet(
        epochs["CHANGE_PHOTO"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
        return_itc=False, average=False, verbose=False,
    )
    print("  Computing single-trial TFR for NO_CHANGE_PHOTO ...")
    tfr_nochange = tfr_morlet(
        epochs["NO_CHANGE_PHOTO"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
        return_itc=False, average=False, verbose=False,
    )

    # Apply baseline (percent change)
    tfr_change.apply_baseline(baseline=BASELINE, mode="percent")
    tfr_nochange.apply_baseline(baseline=BASELINE, mode="percent")

    # Scale to percent
    tfr_change.data *= 100
    tfr_nochange.data *= 100

    results = {}

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in tfr_change.ch_names]
        if not valid_chs:
            continue
        ch_idx = [tfr_change.ch_names.index(c) for c in valid_chs]

        # Average across ROI channels -> (n_epochs, n_freqs, n_times)
        data_a = tfr_change.data[:, ch_idx, :, :].mean(axis=1)
        data_b = tfr_nochange.data[:, ch_idx, :, :].mean(axis=1)

        print(f"  TFR {roi_name}: running {n_perm} permutations "
              f"on {data_a.shape[1]}x{data_a.shape[2]} (freq x time) ...")

        # Cluster-forming threshold (TFR): t-value for p < 0.05 (two-tailed)
        df = data_a.shape[0] + data_b.shape[0] - 2
        t_thresh = scipy_stats.t.ppf(1 - ALPHA / 2, df)

        # permutation_cluster_test expects list of (n_obs, p, q) for 2D
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
            [data_a, data_b],
            n_permutations=n_perm,
            threshold=t_thresh,
            tail=0,
            seed=seed,
            verbose=False,
        )

        sig_mask = cluster_pv < ALPHA
        n_sig = sig_mask.sum()
        print(f"  TFR {roi_name}: {len(clusters)} clusters, "
              f"{n_sig} significant (p < {ALPHA})")

        results[roi_name] = {
            "t_obs": t_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "n_sig": n_sig,
        }

        # --- Plot: t-statistic map with significant clusters outlined ---
        times_ms = tfr_change.times * 1000
        freqs = TFR_FREQS

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left: t-statistic map
        ax = axes[0]
        vmax = float(np.percentile(np.abs(t_obs), 98))
        im = ax.pcolormesh(times_ms, freqs, t_obs,
                           cmap="RdBu_r", shading="gouraud",
                           vmin=-vmax, vmax=vmax)
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_title(f"t-statistic (CHANGE - NO_CHANGE)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="t-value")

        # Right: significant clusters mask
        ax2 = axes[1]
        sig_map = np.zeros_like(t_obs, dtype=float)
        for cl, pv in zip(clusters, cluster_pv):
            if pv < ALPHA:
                sig_map[cl] = 1.0

        ax2.pcolormesh(times_ms, freqs, sig_map,
                       cmap="Reds", shading="gouraud",
                       vmin=0, vmax=1)
        ax2.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax2.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax2.set_title(f"Significant clusters (p < {ALPHA})")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Frequency (Hz)")

        fig.suptitle(f"TFR permutation test: {roi_name} (sub-{subject}), "
                     f"{n_perm} permutations", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_tfr_perm_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    erp_results: dict,
    tfr_results: dict,
    output_dir: Path,
    subject: str,
    n_perm: int,
) -> None:
    """Write a plain-text summary of the permutation test results."""
    lines = [
        f"Permutation test summary — sub-{subject}",
        f"N permutations: {n_perm}",
        f"Alpha: {ALPHA}",
        f"Seed: {RANDOM_SEED}",
        "",
        "=" * 50,
        "ERP results (temporal cluster test per ROI)",
        "=" * 50,
    ]
    for roi, res in erp_results.items():
        lines.append(f"\n  {roi}:")
        lines.append(f"    Total clusters: {len(res['clusters'])}")
        lines.append(f"    Significant clusters: {res['n_sig']}")
        for i, pv in enumerate(res["cluster_pv"]):
            marker = " ***" if pv < ALPHA else ""
            lines.append(f"      cluster {i}: p = {pv:.4f}{marker}")

    lines.extend([
        "",
        "=" * 50,
        "TFR results (freq x time cluster test per ROI)",
        "=" * 50,
    ])
    for roi, res in tfr_results.items():
        lines.append(f"\n  {roi}:")
        lines.append(f"    Total clusters: {len(res['clusters'])}")
        lines.append(f"    Significant clusters: {res['n_sig']}")
        for i, pv in enumerate(res["cluster_pv"]):
            marker = " ***" if pv < ALPHA else ""
            lines.append(f"      cluster {i}: p = {pv:.4f}{marker}")

    txt = "\n".join(lines) + "\n"
    out_path = output_dir / f"sub-{subject}_permutation_summary.txt"
    out_path.write_text(txt, encoding="utf-8")
    print(f"\n  Summary saved: {out_path}")
    print(txt)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str, n_perm: int) -> None:
    epochs_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    if not epochs_path.exists():
        print(f"Epochs file not found: {epochs_path}")
        print("Run 23_epoch_photo_events.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"26 — Permutation test photo contrast — sub-{subject}")
    print(f"     {n_perm} permutations, seed={RANDOM_SEED}, alpha={ALPHA}")
    print("=" * 60)

    epochs = mne.read_epochs(str(epochs_path), verbose=False)
    for cond in EVENT_ID:
        try:
            n = len(epochs[cond])
        except KeyError:
            n = 0
        print(f"  {cond}: {n} epochs")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- ERP permutation tests ---")
    erp_results = run_erp_permutation(epochs, output_dir, subject, n_perm,
                                      seed=RANDOM_SEED)

    print("\n--- TFR permutation tests ---")
    tfr_results = run_tfr_permutation(epochs, output_dir, subject, n_perm,
                                      seed=RANDOM_SEED)

    write_summary(erp_results, tfr_results, output_dir, subject, n_perm)

    print(f"\nAll results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-based permutation test for photo events",
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM,
                        help=f"Number of permutations (default: {DEFAULT_N_PERM})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject, n_perm=args.n_perm)
