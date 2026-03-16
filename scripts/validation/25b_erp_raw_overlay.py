#!/usr/bin/env python
"""ERP overlay plot: individual epochs as semi-transparent lines.

For each ROI, plots every single epoch as a thin transparent line
(green for NO_CHANGE, red for CHANGE) with the grand average on top
in a thick opaque line.  This reveals whether phase-locked structure
exists in the raw single-trial data or only emerges from averaging.

Usage
-----
    micromamba run -n campeones python scripts/validation/25b_erp_raw_overlay.py --subject 27
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
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_erp_raw_overlay"

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

COND_STYLE = {
    "CHANGE_PHOTO": {"color": "C3", "label_prefix": "CHANGE"},
    "NO_CHANGE_PHOTO": {"color": "C2", "label_prefix": "NO_CHANGE"},
}

SINGLE_TRIAL_ALPHA = 0.08  # transparency for individual epochs

# Visualization crop (exclude baseline window)
VIS_TMIN = -3.0


def plot_raw_overlay(epochs: mne.Epochs, output_dir: Path, subject: str) -> None:
    """Plot individual epochs as transparent lines + bold average per ROI."""
    epochs = epochs.copy().crop(tmin=VIS_TMIN)
    time_ms = epochs.times * 1000

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)

        for ax, (cond, style) in zip(axes, COND_STYLE.items()):
            try:
                ep = epochs[cond].copy().pick(valid_chs)
            except KeyError:
                continue

            # (n_epochs, n_channels, n_times) -> avg channels -> (n_epochs, n_times)
            data = ep.get_data().mean(axis=1) * 1e6  # to µV
            n_ep = data.shape[0]

            # Plot each epoch as a thin transparent line
            for i in range(n_ep):
                ax.plot(time_ms, data[i], color=style["color"],
                        alpha=SINGLE_TRIAL_ALPHA, linewidth=0.5)

            # Bold average + median on top
            mean = data.mean(axis=0)
            median = np.median(data, axis=0)
            ax.plot(time_ms, mean, color=style["color"], linewidth=2.0,
                    label=f"Mean (n={n_ep})")
            ax.plot(time_ms, median, color=style["color"], linewidth=2.0,
                    linestyle="--", label=f"Median (n={n_ep})")

            ax.axvline(-1000, color="k", linestyle=":", linewidth=1.0, alpha=0.5)
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
            ax.set_title(f"{style['label_prefix']} (n={n_ep})")
            ax.set_xlabel("Time (ms)")
            ax.legend(fontsize=8)
            ax.invert_yaxis()

        axes[0].set_ylabel("Amplitude (µV)")
        fig.suptitle(f"Single-trial overlay: {roi_name} ROI (sub-{subject})",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_raw_overlay_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Raw overlay plots saved.")


def run_pipeline(subject: str) -> None:
    epochs_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    if not epochs_path.exists():
        print(f"Epochs file not found: {epochs_path}")
        print("Run 23_epoch_photo_events.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"25b — ERP raw overlay — sub-{subject}")
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

    plot_raw_overlay(epochs, output_dir, subject)
    print(f"\nPlots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ERP raw overlay: individual epochs as transparent lines",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
