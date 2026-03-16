#!/usr/bin/env python
"""Compare ERP and raw overlay of CHANGE / NO_CHANGE / RANDOM epochs.

Loads the photo epochs (script 23) and random epochs (script 23b),
then generates:
  1. ERP waveforms: 3 conditions overlaid per ROI (mean + SEM)
  2. Raw overlay: single-trial transparent lines per condition (3 panels)

If the oscillatory pattern in NO_CHANGE also appears in RANDOM, the effect
is an artifact of alpha asymmetry / averaging.  If RANDOM is flat, the
NO_CHANGE pattern is stimulus-related.

Usage
-----
    micromamba run -n campeones python scripts/validation/25c_erp_random_comparison.py --subject 27
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EPOCHS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_epochs"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_erp_random"

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

VIS_TMIN = -3.0
SINGLE_TRIAL_ALPHA = 0.08

COND_STYLE = {
    "CHANGE_PHOTO":    {"color": "C3", "label": "CHANGE"},
    "NO_CHANGE_PHOTO": {"color": "C2", "label": "NO_CHANGE"},
    "RANDOM":          {"color": "C0", "label": "RANDOM"},
}


def _get_roi_data(epochs: mne.Epochs, roi_chs: list[str]) -> np.ndarray | None:
    """Return (n_epochs, n_times) in µV, averaged across ROI channels."""
    valid = [c for c in roi_chs if c in epochs.ch_names]
    if not valid:
        return None
    return epochs.copy().pick(valid).get_data().mean(axis=1) * 1e6


# ---------------------------------------------------------------------------
# 1. ERP overlay (mean + SEM) — 3 conditions on same axes
# ---------------------------------------------------------------------------

def plot_erp_comparison(
    datasets: dict[str, mne.Epochs],
    output_dir: Path,
    subject: str,
) -> None:
    for roi_name, roi_chs in ROIS.items():
        fig, ax = plt.subplots(figsize=(14, 5))

        for cond, style in COND_STYLE.items():
            if cond not in datasets:
                continue
            data = _get_roi_data(datasets[cond], roi_chs)
            if data is None or data.shape[0] == 0:
                continue
            time_ms = datasets[cond].times * 1000
            mean = data.mean(axis=0)
            sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            ax.plot(time_ms, mean,
                    label=f"{style['label']} (n={data.shape[0]})",
                    color=style["color"])
            ax.fill_between(time_ms, mean - sem, mean + sem,
                            color=style["color"], alpha=0.2)

        ax.axvline(-1000, color="k", ls=":", lw=1, alpha=0.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(1000, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_title(f"ERP comparison: {roi_name} ROI (sub-{subject})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_erp_cmp_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  ERP comparison plots saved.")


# ---------------------------------------------------------------------------
# 2. Raw overlay — 3 panels side by side
# ---------------------------------------------------------------------------

def plot_raw_overlay_comparison(
    datasets: dict[str, mne.Epochs],
    output_dir: Path,
    subject: str,
) -> None:
    """Plot percentile bands (p25-p75) + median per condition, side by side."""
    conds = [c for c in COND_STYLE if c in datasets]
    n_conds = len(conds)
    if n_conds == 0:
        return

    for roi_name, roi_chs in ROIS.items():
        fig, axes = plt.subplots(1, n_conds, figsize=(10 * n_conds, 6),
                                 sharey=True)
        if n_conds == 1:
            axes = [axes]

        for ax, cond in zip(axes, conds):
            style = COND_STYLE[cond]
            data = _get_roi_data(datasets[cond], roi_chs)
            if data is None or data.shape[0] == 0:
                continue
            time_ms = datasets[cond].times * 1000
            n_ep = data.shape[0]

            p25 = np.percentile(data, 25, axis=0)
            p75 = np.percentile(data, 75, axis=0)
            p10 = np.percentile(data, 10, axis=0)
            p90 = np.percentile(data, 90, axis=0)
            mean = data.mean(axis=0)
            median = np.median(data, axis=0)

            ax.fill_between(time_ms, p10, p90,
                            color=style["color"], alpha=0.1, label="p10-p90")
            ax.fill_between(time_ms, p25, p75,
                            color=style["color"], alpha=0.25, label="p25-p75")
            ax.plot(time_ms, mean, color=style["color"], lw=2,
                    label=f"Mean (n={n_ep})")
            ax.plot(time_ms, median, color=style["color"], lw=2,
                    ls="--", label=f"Median")

            ax.axvline(-1000, color="k", ls=":", lw=1, alpha=0.5)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.axvline(1000, color="k", ls="--", lw=1, alpha=0.5)
            ax.axhline(0, color="gray", ls=":", lw=0.5)
            ax.set_title(f"{style['label']} (n={n_ep})")
            ax.set_xlabel("Time (ms)")
            ax.legend(fontsize=7)
            ax.invert_yaxis()

        axes[0].set_ylabel("Amplitude (µV)")
        fig.suptitle(
            f"Epoch distribution: {roi_name} ROI (sub-{subject})",
            fontsize=13)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"sub-{subject}_dist_cmp_{roi_name}.png",
            dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Distribution comparison plots saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    photo_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    random_path = EPOCHS_ROOT / f"sub-{subject}_random-epo.fif"

    if not photo_path.exists():
        print(f"Photo epochs not found: {photo_path}")
        print("Run 23_epoch_photo_events.py first.")
        sys.exit(1)
    if not random_path.exists():
        print(f"Random epochs not found: {random_path}")
        print("Run 23b_epoch_random_events.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"25c — ERP random comparison — sub-{subject}")
    print("=" * 60)

    photo_epochs = mne.read_epochs(str(photo_path), verbose=False)
    random_epochs = mne.read_epochs(str(random_path), verbose=False)

    # Crop for visualization
    photo_vis = photo_epochs.copy().crop(tmin=VIS_TMIN)
    random_vis = random_epochs.copy().crop(tmin=VIS_TMIN)

    # Build per-condition datasets
    datasets: dict[str, mne.Epochs] = {}
    for cond in ["CHANGE_PHOTO", "NO_CHANGE_PHOTO"]:
        try:
            datasets[cond] = photo_vis[cond]
            print(f"  {cond}: {len(datasets[cond])} epochs")
        except KeyError:
            pass
    # Subsample random epochs to match photo epoch count for fair comparison
    n_photo = max((len(datasets[c]) for c in datasets), default=74)
    if len(random_vis) > n_photo:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(random_vis), size=n_photo, replace=False)
        idx.sort()
        random_vis = random_vis[idx]
    datasets["RANDOM"] = random_vis
    print(f"  RANDOM: {len(random_vis)} epochs (subsampled to match)")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_erp_comparison(datasets, output_dir, subject)
    plot_raw_overlay_comparison(datasets, output_dir, subject)

    print(f"\nAll plots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ERP: CHANGE vs NO_CHANGE vs RANDOM",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
