#!/usr/bin/env python
"""ERP and TFR analysis for CHANGE_PHOTO vs NO_CHANGE_PHOTO.

Implements Tasks 1.6, 1.7, 1.8 from the research diary.

Loads the concatenated photo epochs (from script 23), then:
- Plots ERP waveforms per ROI (CHANGE vs NO_CHANGE)
- Computes and plots TFR maps per ROI
- Plots the contrast (CHANGE - NO_CHANGE) for both ERP and TFR

Usage
-----
    python scripts/validation/25_erp_tfr_photo_contrast.py --subject 27
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
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "photo_erp_tfr"

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "FC1", "FC2", "FC5", "FC6", "FCz"],
    "Temporal": ["T7", "T8", "FT9", "FT10", "TP9", "TP10"],
    "Parietal": ["P3", "P4", "P7", "P8", "Pz", "CP1", "CP2", "CP5", "CP6"],
    "Occipital": ["O1", "O2"],
}

COND_COLORS = {"CHANGE_PHOTO": "C3", "NO_CHANGE_PHOTO": "C2"}

# TFR parameters
TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)
TFR_N_CYCLES = TFR_FREQS / 2.0


# ---------------------------------------------------------------------------
# ERP plots (Task 1.6)
# ---------------------------------------------------------------------------

def plot_erp(epochs: mne.Epochs, output_dir: Path, subject: str) -> None:
    """Plot ERP waveforms per ROI: CHANGE vs NO_CHANGE with SEM bands."""
    time_pts = epochs.times * 1000  # ms

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))

        for cond, color in COND_COLORS.items():
            try:
                ep = epochs[cond].copy().pick(valid_chs)
            except KeyError:
                continue
            # (n_epochs, n_channels, n_times) -> average over channels -> (n_epochs, n_times)
            data = ep.get_data().mean(axis=1) * 1e6  # to uV
            mean = data.mean(axis=0)
            sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
            ax.plot(time_pts, mean, label=f"{cond} (n={data.shape[0]})", color=color)
            ax.fill_between(time_pts, mean - sem, mean + sem,
                            color=color, alpha=0.25)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
                   label="flicker offset (1000 ms)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(f"ERP: {roi_name} ROI (sub-{subject})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_erp_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  ERP plots saved.")


# ---------------------------------------------------------------------------
# TFR plots (Task 1.7)
# ---------------------------------------------------------------------------

def plot_tfr(epochs: mne.Epochs, output_dir: Path, subject: str) -> float:
    """Plot TFR maps per ROI for each condition. Returns shared vmax."""
    tfrs = {}
    for cond in COND_COLORS:
        try:
            ep = epochs[cond]
        except KeyError:
            continue
        tfr = tfr_morlet(ep, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
                         return_itc=False, average=True, verbose=False)
        tfr.apply_baseline(baseline=(-1.0, -0.5), mode="percent")
        tfr.data *= 100  # to percent
        tfrs[cond] = tfr

    if not tfrs:
        return 0.0

    # Compute shared vmax across all conditions and ROIs
    all_data = np.concatenate([t.data.ravel() for t in tfrs.values()])
    vmax = float(np.percentile(np.abs(all_data), 98))

    for roi_name, roi_chs in ROIS.items():
        ref_tfr = list(tfrs.values())[0]
        valid_chs = [c for c in roi_chs if c in ref_tfr.ch_names]
        if not valid_chs:
            continue
        ch_idx = [ref_tfr.ch_names.index(c) for c in valid_chs]

        n_conds = len(tfrs)
        fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 5),
                                 squeeze=False)
        for ax, (cond, tfr) in zip(axes[0], tfrs.items()):
            data = tfr.data[ch_idx].mean(axis=0)
            im = ax.pcolormesh(tfr.times * 1000, tfr.freqs, data,
                               cmap="RdBu_r", shading="gouraud",
                               vmin=-vmax, vmax=vmax)
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
            n_ep = len(epochs[cond])
            ax.set_title(f"{cond} (n={n_ep})")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power change (%)")

        fig.suptitle(f"TFR: {roi_name} ROI (sub-{subject})", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_tfr_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  TFR plots saved.")
    return vmax


# ---------------------------------------------------------------------------
# Contrast plots (Task 1.8)
# ---------------------------------------------------------------------------

def plot_contrast(epochs: mne.Epochs, output_dir: Path, subject: str,
                  shared_vmax: float) -> None:
    """Plot CHANGE - NO_CHANGE contrast for ERP and TFR."""
    conds = list(COND_COLORS.keys())
    if len(conds) < 2:
        return

    cond_a, cond_b = conds[0], conds[1]  # CHANGE, NO_CHANGE

    # --- ERP contrast ---
    time_pts = epochs.times * 1000
    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        data_a = epochs[cond_a].copy().pick(valid_chs).get_data().mean(axis=1) * 1e6
        data_b = epochs[cond_b].copy().pick(valid_chs).get_data().mean(axis=1) * 1e6
        mean_a = data_a.mean(axis=0)
        mean_b = data_b.mean(axis=0)
        diff = mean_a - mean_b

        ax.plot(time_pts, diff, color="k", linewidth=1.5,
                label=f"{cond_a} - {cond_b}")
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(f"ERP contrast: {roi_name} (sub-{subject})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude diff (uV)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_erp_contrast_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- TFR contrast ---
    tfr_a = tfr_morlet(epochs[cond_a], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
                       return_itc=False, average=True, verbose=False)
    tfr_b = tfr_morlet(epochs[cond_b], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
                       return_itc=False, average=True, verbose=False)
    tfr_a.apply_baseline(baseline=(-1.0, -0.5), mode="percent")
    tfr_b.apply_baseline(baseline=(-1.0, -0.5), mode="percent")
    tfr_a.data *= 100
    tfr_b.data *= 100

    tfr_diff = tfr_a.copy()
    tfr_diff.data = tfr_a.data - tfr_b.data

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in tfr_diff.ch_names]
        if not valid_chs:
            continue
        ch_idx = [tfr_diff.ch_names.index(c) for c in valid_chs]
        data = tfr_diff.data[ch_idx].mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.pcolormesh(tfr_diff.times * 1000, tfr_diff.freqs, data,
                           cmap="RdBu_r", shading="gouraud",
                           vmin=-shared_vmax, vmax=shared_vmax)
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(1000, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_title(f"TFR contrast ({cond_a} - {cond_b}): {roi_name} (sub-{subject})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power diff (%)")
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{subject}_tfr_contrast_{roi_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  Contrast plots saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(subject: str) -> None:
    epochs_path = EPOCHS_ROOT / f"sub-{subject}_photo-epo.fif"
    if not epochs_path.exists():
        print(f"Epochs file not found: {epochs_path}")
        print("Run 23_epoch_photo_events.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"25 — ERP & TFR photo contrast — sub-{subject}")
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

    plot_erp(epochs, output_dir, subject)
    vmax = plot_tfr(epochs, output_dir, subject)
    plot_contrast(epochs, output_dir, subject, shared_vmax=vmax)

    print(f"\nAll plots saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ERP & TFR analysis for photo events",
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(subject=args.subject)
