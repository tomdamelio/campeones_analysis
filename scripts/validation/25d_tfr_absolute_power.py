#!/usr/bin/env python
"""TFR with absolute power (no baseline normalization) for CHANGE / NO_CHANGE / RANDOM.

Implements Task 8.2 from the research diary.

Shows the raw spectral power across time and frequency without any
baseline correction.  This reveals the ongoing alpha power that is
present in all conditions and hidden by percent-change normalization.

Plots per ROI:
  1. Absolute TFR heatmaps: 3 conditions side by side (same color scale).
  2. Alpha-band time course: mean absolute power in 8-12 Hz per condition.

Usage
-----
    micromamba run -n campeones python scripts/validation/25d_tfr_absolute_power.py --subject 27
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
from mne.time_frequency import tfr_morlet

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EPOCHS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_epochs"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_tfr_absolute"

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

COND_STYLE = {
    "CHANGE_PHOTO":    {"color": "C3", "label": "CHANGE"},
    "NO_CHANGE_PHOTO": {"color": "C2", "label": "NO_CHANGE"},
    "RANDOM":          {"color": "C0", "label": "RANDOM"},
}

VIS_TMIN = -3.0

# TFR parameters (same as script 25)
TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)
TFR_N_CYCLES = TFR_FREQS / 2.0

# Alpha band indices (for time-course extraction)
ALPHA_LOW = 8.0
ALPHA_HIGH = 12.0


# ---------------------------------------------------------------------------
# 1. Absolute TFR heatmaps — 3 conditions per ROI, shared color scale
# ---------------------------------------------------------------------------

def plot_absolute_tfr(
    tfrs: dict[str, mne.time_frequency.AverageTFR],
    epochs_counts: dict[str, int],
    output_dir: Path,
    subject: str,
    global_vmin: float,
    global_vmax: float,
) -> None:
    """Plot absolute-power TFR heatmaps per ROI, all conditions side by side.

    Uses a global color scale (global_vmin, global_vmax) shared across
    all ROIs so that plots are directly comparable.
    """
    conds = [c for c in COND_STYLE if c in tfrs]
    n_conds = len(conds)
    if n_conds == 0:
        return

    for roi_name, roi_chs in ROIS.items():
        ref_tfr = list(tfrs.values())[0]
        valid_chs = [c for c in roi_chs if c in ref_tfr.ch_names]
        if not valid_chs:
            continue
        ch_idx = [ref_tfr.ch_names.index(c) for c in valid_chs]

        roi_data = []
        for cond in conds:
            roi_data.append(tfrs[cond].data[ch_idx].mean(axis=0))

        fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 5),
                                 squeeze=False)
        for ax, cond, data in zip(axes[0], conds, roi_data):
            style = COND_STYLE[cond]
            n_ep = epochs_counts.get(cond, 0)
            im = ax.pcolormesh(
                ref_tfr.times * 1000, ref_tfr.freqs, data,
                cmap="inferno", shading="gouraud",
                vmin=global_vmin, vmax=global_vmax,
            )
            ax.axvline(-1000, color="w", ls=":", lw=1, alpha=0.7)
            ax.axvline(0, color="w", ls="--", lw=1)
            ax.axvline(1000, color="w", ls="--", lw=1, alpha=0.7)
            ax.set_title(f"{style['label']} (n={n_ep})")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power (uV^2/Hz)")

        fig.suptitle(
            f"Absolute TFR: {roi_name} ROI (sub-{subject})",
            fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"sub-{subject}_tfr_abs_{roi_name}.png",
            dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Absolute TFR heatmaps saved.")


# ---------------------------------------------------------------------------
# 2. Alpha-band time course — mean power in 8-12 Hz per condition
# ---------------------------------------------------------------------------

def plot_alpha_timecourse(
    tfrs: dict[str, mne.time_frequency.AverageTFR],
    epochs_counts: dict[str, int],
    output_dir: Path,
    subject: str,
    global_alpha_ylim: tuple[float, float],
) -> None:
    """Plot alpha-band absolute power time course, all conditions overlaid.

    Uses a shared Y-axis range (global_alpha_ylim) across all ROIs.
    """
    conds = [c for c in COND_STYLE if c in tfrs]
    if not conds:
        return

    ref_tfr = list(tfrs.values())[0]
    alpha_mask = (ref_tfr.freqs >= ALPHA_LOW) & (ref_tfr.freqs <= ALPHA_HIGH)

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in ref_tfr.ch_names]
        if not valid_chs:
            continue
        ch_idx = [ref_tfr.ch_names.index(c) for c in valid_chs]

        fig, ax = plt.subplots(figsize=(14, 5))
        for cond in conds:
            style = COND_STYLE[cond]
            n_ep = epochs_counts.get(cond, 0)
            data = tfrs[cond].data[ch_idx][:, alpha_mask, :].mean(axis=(0, 1))
            ax.plot(ref_tfr.times * 1000, data,
                    color=style["color"], lw=1.5,
                    label=f"{style['label']} (n={n_ep})")

        ax.axvline(-1000, color="k", ls=":", lw=1, alpha=0.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(1000, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_ylim(global_alpha_ylim)
        ax.set_title(f"Alpha power (8-12 Hz): {roi_name} ROI (sub-{subject})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Power (uV^2/Hz)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"sub-{subject}_alpha_abs_{roi_name}.png",
            dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Alpha time-course plots saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    photo_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    random_path = EPOCHS_ROOT / f"sub-{subject}_random-epo.fif"

    if not photo_path.exists():
        print(f"Photo epochs not found: {photo_path}")
        sys.exit(1)
    if not random_path.exists():
        print(f"Random epochs not found: {random_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"25d — TFR absolute power — sub-{subject}")
    print("=" * 60)

    photo_epochs = mne.read_epochs(str(photo_path), verbose=False)
    random_epochs = mne.read_epochs(str(random_path), verbose=False)

    # Build per-condition epoch sets and crop for visualization
    datasets: dict[str, mne.Epochs] = {}
    for cond in ["CHANGE_PHOTO", "NO_CHANGE_PHOTO"]:
        try:
            datasets[cond] = photo_epochs[cond].copy().crop(tmin=VIS_TMIN)
            print(f"  {cond}: {len(datasets[cond])} epochs")
        except KeyError:
            pass

    # Subsample random to match photo count
    n_photo = max((len(datasets[c]) for c in datasets), default=74)
    random_vis = random_epochs.copy().crop(tmin=VIS_TMIN)
    if len(random_vis) > n_photo:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(random_vis), size=n_photo, replace=False)
        idx.sort()
        random_vis = random_vis[idx]
    datasets["RANDOM"] = random_vis
    print(f"  RANDOM: {len(random_vis)} epochs")

    # Compute TFRs — NO baseline normalization
    print("\n  Computing TFRs (absolute power, no baseline) ...")
    tfrs = {}
    epochs_counts = {}
    for cond, ep in datasets.items():
        tfr = tfr_morlet(
            ep, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
            return_itc=False, average=True, verbose=False,
        )
        # Convert to uV^2/Hz for readability (data comes in V^2/Hz)
        tfr.data *= 1e12
        tfrs[cond] = tfr
        epochs_counts[cond] = len(ep)
        print(f"    {cond}: done")

    output_dir = RESULTS_ROOT / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute global scales across all ROIs and conditions
    print("  Computing global scales ...")
    conds = [c for c in COND_STYLE if c in tfrs]
    ref_tfr = list(tfrs.values())[0]
    alpha_mask = (ref_tfr.freqs >= ALPHA_LOW) & (ref_tfr.freqs <= ALPHA_HIGH)

    global_tfr_min = float("inf")
    global_tfr_max = float("-inf")
    global_alpha_min = float("inf")
    global_alpha_max = float("-inf")

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in ref_tfr.ch_names]
        if not valid_chs:
            continue
        ch_idx = [ref_tfr.ch_names.index(c) for c in valid_chs]
        for cond in conds:
            roi_data = tfrs[cond].data[ch_idx].mean(axis=0)
            global_tfr_min = min(global_tfr_min, roi_data.min())
            global_tfr_max = max(global_tfr_max, roi_data.max())
            alpha_data = tfrs[cond].data[ch_idx][:, alpha_mask, :].mean(axis=(0, 1))
            global_alpha_min = min(global_alpha_min, alpha_data.min())
            global_alpha_max = max(global_alpha_max, alpha_data.max())

    # Add 5% padding to alpha Y range
    alpha_range = global_alpha_max - global_alpha_min
    global_alpha_ylim = (
        global_alpha_min - 0.05 * alpha_range,
        global_alpha_max + 0.05 * alpha_range,
    )
    print(f"    TFR color range: [{global_tfr_min:.2f}, {global_tfr_max:.2f}]")
    print(f"    Alpha Y range:   [{global_alpha_ylim[0]:.2f}, {global_alpha_ylim[1]:.2f}]")

    plot_absolute_tfr(tfrs, epochs_counts, output_dir, subject,
                      global_vmin=global_tfr_min, global_vmax=global_tfr_max)
    plot_alpha_timecourse(tfrs, epochs_counts, output_dir, subject,
                          global_alpha_ylim=global_alpha_ylim)

    print(f"\nAll plots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TFR absolute power: CHANGE vs NO_CHANGE vs RANDOM",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
