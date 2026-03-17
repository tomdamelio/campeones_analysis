#!/usr/bin/env python
"""Check magnitude ratio between single trials and averaged ERP.

Implements Task 8.4 from the research diary.

For each condition (CHANGE, NO_CHANGE, RANDOM), compares the amplitude
(RMS) of individual trials vs the ERP average. The expected ratio is
~√N (≈8.6 for N=74). If the ratio is much smaller, it means the
averaging is not cancelling background activity as expected (possible
coherent component across trials).

Usage
-----
    micromamba run -n campeones python scripts/validation/30_magnitude_check.py --subject 27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

EPOCHS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_epochs"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "magnitude_check"

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

VIS_TMIN = -3.0  # crop for visualization (exclude baseline)

COND_COLORS = {
    "CHANGE": "C3",
    "NO_CHANGE": "C2",
    "RANDOM": "C0",
}


def compute_trial_rms(data_3d: np.ndarray, ch_indices: list[int]) -> np.ndarray:
    """Compute RMS per trial, averaged across ROI channels.

    Args:
        data_3d: (n_epochs, n_channels, n_times) in Volts
        ch_indices: channel indices for the ROI

    Returns:
        (n_epochs,) array of RMS values in µV
    """
    roi_data = data_3d[:, ch_indices, :]  # (n_ep, n_roi_ch, n_times)
    roi_mean = roi_data.mean(axis=1)  # (n_ep, n_times) — average across ROI channels
    rms_per_trial = np.sqrt(np.mean(roi_mean ** 2, axis=1))  # (n_ep,)
    return rms_per_trial * 1e6  # convert to µV


def compute_erp_rms(data_3d: np.ndarray, ch_indices: list[int]) -> float:
    """Compute RMS of the ERP (trial-averaged signal) for ROI channels.

    Returns: RMS in µV
    """
    erp = data_3d.mean(axis=0)  # (n_channels, n_times)
    roi_erp = erp[ch_indices].mean(axis=0)  # (n_times,)
    return float(np.sqrt(np.mean(roi_erp ** 2))) * 1e6


def analyze_condition(
    data_3d: np.ndarray, ch_names: list[str], cond_name: str,
) -> dict:
    """Compute magnitude stats for one condition across all ROIs."""
    n_epochs = data_3d.shape[0]
    expected_ratio = np.sqrt(n_epochs)
    results = {"condition": cond_name, "n_epochs": n_epochs,
               "expected_ratio_sqrt_n": float(expected_ratio), "rois": {}}

    for roi_name, roi_chs in ROIS.items():
        ch_idx = [ch_names.index(ch) for ch in roi_chs if ch in ch_names]
        if not ch_idx:
            continue

        trial_rms = compute_trial_rms(data_3d, ch_idx)
        erp_rms = compute_erp_rms(data_3d, ch_idx)
        mean_trial_rms = float(np.mean(trial_rms))
        actual_ratio = mean_trial_rms / erp_rms if erp_rms > 0 else float('inf')

        results["rois"][roi_name] = {
            "mean_trial_rms_uv": mean_trial_rms,
            "median_trial_rms_uv": float(np.median(trial_rms)),
            "std_trial_rms_uv": float(np.std(trial_rms)),
            "erp_rms_uv": erp_rms,
            "ratio_trial_over_erp": actual_ratio,
            "trial_rms_all": trial_rms.tolist(),
        }

        print(f"    {roi_name:12s}: trial RMS = {mean_trial_rms:.2f} ± "
              f"{np.std(trial_rms):.2f} µV | ERP RMS = {erp_rms:.2f} µV | "
              f"ratio = {actual_ratio:.1f} (expected √{n_epochs} = {expected_ratio:.1f})")

    return results


def plot_magnitude_comparison(
    all_results: list[dict], output_dir: Path, subject: str,
) -> None:
    """Bar chart: trial RMS vs ERP RMS for each condition × ROI."""
    conditions = [r["condition"] for r in all_results]
    roi_names = list(ROIS.keys())
    n_cond = len(conditions)
    n_rois = len(roi_names)

    fig, axes = plt.subplots(1, n_rois, figsize=(5 * n_rois, 5), sharey=True)
    if n_rois == 1:
        axes = [axes]

    for ax, roi_name in zip(axes, roi_names):
        x = np.arange(n_cond)
        width = 0.35

        trial_vals = []
        erp_vals = []
        trial_stds = []
        for r in all_results:
            roi_data = r["rois"].get(roi_name, {})
            trial_vals.append(roi_data.get("mean_trial_rms_uv", 0))
            trial_stds.append(roi_data.get("std_trial_rms_uv", 0))
            erp_vals.append(roi_data.get("erp_rms_uv", 0))

        bars1 = ax.bar(x - width/2, trial_vals, width, yerr=trial_stds,
                        capsize=3, label='Trial individual (media ± SD)',
                        color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, erp_vals, width,
                        label='ERP promediado', color='coral', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_title(roi_name)
        if ax == axes[0]:
            ax.set_ylabel('RMS (µV)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        f'Magnitud: trial individual vs ERP promediado — sub-{subject}',
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_magnitude_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar plot saved: {out}")


def plot_ratio_summary(
    all_results: list[dict], output_dir: Path, subject: str,
) -> None:
    """Plot actual ratio vs expected √N for each condition × ROI."""
    conditions = [r["condition"] for r in all_results]
    roi_names = list(ROIS.keys())
    n_cond = len(conditions)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(roi_names))
    width = 0.8 / n_cond

    for i, r in enumerate(all_results):
        ratios = [r["rois"].get(roi, {}).get("ratio_trial_over_erp", 0)
                  for roi in roi_names]
        color = COND_COLORS.get(r["condition"], f"C{i}")
        ax.bar(x + i * width, ratios, width, label=r["condition"],
               color=color, alpha=0.8)

    # Expected √N line (use first condition's N as reference)
    expected = all_results[0]["expected_ratio_sqrt_n"]
    ax.axhline(expected, color='black', ls='--', lw=1.5,
               label=f'Esperado √N = {expected:.1f}')

    ax.set_xticks(x + width * (n_cond - 1) / 2)
    ax.set_xticklabels(roi_names)
    ax.set_ylabel('Ratio (trial RMS / ERP RMS)')
    ax.set_title(f'Ratio de cancelación por promediado — sub-{subject}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    out = output_dir / f"sub-{subject}_ratio_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ratio plot saved: {out}")


def plot_trial_distributions(
    all_results: list[dict], output_dir: Path, subject: str,
) -> None:
    """Histogram of trial RMS per condition for Occipital and Temporal."""
    plot_rois = ["Occipital", "Temporal"]
    conditions = [r["condition"] for r in all_results]

    fig, axes = plt.subplots(1, len(plot_rois), figsize=(6 * len(plot_rois), 5))
    if len(plot_rois) == 1:
        axes = [axes]

    for ax, roi_name in zip(axes, plot_rois):
        for r in all_results:
            roi_data = r["rois"].get(roi_name, {})
            trial_rms = roi_data.get("trial_rms_all", [])
            erp_rms = roi_data.get("erp_rms_uv", 0)
            color = COND_COLORS.get(r["condition"], "gray")

            if trial_rms:
                ax.hist(trial_rms, bins=20, alpha=0.4, color=color,
                        label=f'{r["condition"]} trials (n={len(trial_rms)})')
                ax.axvline(erp_rms, color=color, ls='--', lw=2,
                           label=f'{r["condition"]} ERP RMS')

        ax.set_xlabel('RMS (µV)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(roi_name)
        ax.legend(fontsize=7)

    fig.suptitle(
        f'Distribución de RMS por trial vs ERP — sub-{subject}',
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / f"sub-{subject}_trial_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution plot saved: {out}")


def run_pipeline(subject: str) -> None:
    print("=" * 60)
    print(f"30 — Magnitude check: trial vs ERP — sub-{subject}")
    print("=" * 60)

    # Load epochs
    photo_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    random_path = EPOCHS_ROOT / f"sub-{subject}_random-epo.fif"

    if not photo_path.exists():
        print(f"ERROR: {photo_path} not found")
        return
    if not random_path.exists():
        print(f"ERROR: {random_path} not found")
        return

    print("\nLoading epochs...")
    photo_epochs = mne.read_epochs(str(photo_path), verbose=False)
    random_epochs = mne.read_epochs(str(random_path), verbose=False)

    ch_names = photo_epochs.ch_names

    # Crop to visualization window to exclude baseline region
    photo_epochs_vis = photo_epochs.copy().crop(tmin=VIS_TMIN)
    random_epochs_vis = random_epochs.copy().crop(tmin=VIS_TMIN)

    # Split CHANGE and NO_CHANGE
    change_data = photo_epochs_vis["CHANGE_PHOTO"].get_data()
    no_change_data = photo_epochs_vis["NO_CHANGE_PHOTO"].get_data()

    # Subsample RANDOM to match N=74
    random_data_all = random_epochs_vis.get_data()
    n_target = min(change_data.shape[0], random_data_all.shape[0])
    rng = np.random.default_rng(42)
    random_idx = rng.choice(random_data_all.shape[0], size=n_target, replace=False)
    random_data = random_data_all[random_idx]

    print(f"  CHANGE: {change_data.shape[0]} epochs")
    print(f"  NO_CHANGE: {no_change_data.shape[0]} epochs")
    print(f"  RANDOM: {random_data.shape[0]} epochs (subsampled from {random_data_all.shape[0]})")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each condition
    all_results = []

    print("\n  CHANGE:")
    res = analyze_condition(change_data, ch_names, "CHANGE")
    all_results.append(res)

    print("\n  NO_CHANGE:")
    res = analyze_condition(no_change_data, ch_names, "NO_CHANGE")
    all_results.append(res)

    print("\n  RANDOM:")
    res = analyze_condition(random_data, ch_names, "RANDOM")
    all_results.append(res)

    # Plots
    plot_magnitude_comparison(all_results, output_dir, subject)
    plot_ratio_summary(all_results, output_dir, subject)
    plot_trial_distributions(all_results, output_dir, subject)

    # Save JSON (without the large trial_rms_all arrays)
    json_results = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items() if k != "rois"}
        r_clean["rois"] = {}
        for roi_name, roi_data in r["rois"].items():
            r_clean["rois"][roi_name] = {
                k: v for k, v in roi_data.items() if k != "trial_rms_all"
            }
        json_results.append(r_clean)

    json_path = output_dir / f"sub-{subject}_magnitude_check.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results JSON saved: {json_path}")

    print(f"\nDone. Output in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Magnitude check: single trial vs averaged ERP",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
