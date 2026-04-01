#!/usr/bin/env python
"""Timeline visualization of 3-class luminance epochs.

Same format as results/modeling/validation/threshold_exploration plots:
  - Top panel   : ΔLuminance / frame (derivative) with threshold lines
  - Bottom panel: Luminance value (green channel, 0–255)
  - Colored spans: ChangeUp (red), ChangeDown (blue), NoChange (green)

Uses the exact same event detection as 36_decoding_luminance_3class.py
(threshold=2.0, NC_prior=1s, greedy non-overlapping selection).

One output plot per run (= per video presentation).

Outputs:
  results/validation/luminance_3class/sub-{sub}/timeline_plots/
    sub-{sub}_timeline_{run_label}.png

Usage
-----
    micromamba run -n campeones python scripts/validation/37_visualize_luminance_epochs.py --subject 27
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Import from script 36 (event detection + data loading)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "script36",
    SCRIPT_DIR / "36_decoding_luminance_3class.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_run_data           = _mod.load_run_data
extract_windows_for_run = _mod.extract_windows_for_run
DERIV_THRESHOLD         = _mod.DERIV_THRESHOLD
NC_THRESHOLD            = _mod.NC_THRESHOLD
WIN_SIZE_S              = _mod.WIN_SIZE_S
WIN_OFFSETS_S           = _mod.WIN_OFFSETS_S
GUARD_POST_S            = _mod.GUARD_POST_S
LABEL_NC                = _mod.LABEL_NC
LABEL_UP                = _mod.LABEL_UP
LABEL_DOWN              = _mod.LABEL_DOWN
CLASS_NAMES             = _mod.CLASS_NAMES
RESULTS_ROOT            = _mod.RESULTS_ROOT

# ---------------------------------------------------------------------------
# Plot one run
# ---------------------------------------------------------------------------

def plot_run_timeline(run: dict, output_dir: Path, subject: str) -> None:
    lum_times = run["lum_times"]      # (n_frames,) seconds
    lum_vals  = run["lum_vals"]       # (n_frames,) 0–255
    lum_deriv = run["lum_deriv"]      # (n_frames-1,) frame-level derivative
    run_label = run["run_label"]
    events    = run.get("_events", {})
    nc_frames = run.get("_nc_frames", np.array([], dtype=int))

    t_deriv = lum_times[1:]           # derivative is one shorter than lum_times

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax_d, ax_l) = plt.subplots(2, 1, figsize=(20, 6), sharex=True)

    # Top: derivative
    ax_d.plot(t_deriv, lum_deriv, color="0.35", lw=0.6, zorder=2)
    ax_d.axhline( DERIV_THRESHOLD, color="#d62728", ls=":", lw=0.9, alpha=0.8)
    ax_d.axhline(-DERIV_THRESHOLD, color="#1f77b4", ls=":", lw=0.9, alpha=0.8)
    ax_d.axhline( NC_THRESHOLD,    color="#2ca02c", ls=":", lw=0.6, alpha=0.5)
    ax_d.axhline(-NC_THRESHOLD,    color="#2ca02c", ls=":", lw=0.6, alpha=0.5)
    ax_d.axhline(0, color="k", lw=0.4)
    ax_d.set_ylabel("ΔLuminance / frame")

    # Bottom: luminance
    ax_l.plot(lum_times, lum_vals, color="0.35", lw=0.7, zorder=2)
    ax_l.set_ylabel("Luminance (green ch.)")
    ax_l.set_xlabel("Time (s)")

    # ── colored spans — 6 overlapping windows per event (all 3 classes) ─────
    plotted_labels: set[str] = set()

    for frames, color, lbl in [
        (events.get("ChangeUp",   np.array([], int)), "#d62728", "ChangeUp"),
        (events.get("ChangeDown", np.array([], int)), "#1f77b4", "ChangeDown"),
        (nc_frames,                                   "#2ca02c", "NoChange"),
    ]:
        for f in frames:
            t_onset = lum_times[f]
            legend_lbl = lbl if lbl not in plotted_labels else None
            for win_i, offset_s in enumerate(WIN_OFFSETS_S):
                t0 = t_onset + offset_s
                t1 = t0 + WIN_SIZE_S
                for ax in (ax_d, ax_l):
                    ax.axvspan(
                        t0, t1,
                        alpha=0.14,
                        color=color,
                        label=legend_lbl if (ax is ax_d and win_i == 0) else None,
                    )
            plotted_labels.add(lbl)

    # ── counts & title ───────────────────────────────────────────────────────
    n_up   = len(events.get("ChangeUp",   []))
    n_down = len(events.get("ChangeDown", []))
    n_nc   = len(nc_frames)

    ax_d.set_title(
        f"{run_label.replace('_', ' ')} | thr={DERIV_THRESHOLD} | "
        f"ChangeUp: {n_up} ({n_up * len(WIN_OFFSETS_S)} wins),  "
        f"ChangeDown: {n_down} ({n_down * len(WIN_OFFSETS_S)} wins),  "
        f"NoChange: {n_nc} wins"
    )
    ax_d.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"sub-{subject}_timeline_{run_label}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(subject: str) -> None:
    output_dir = RESULTS_ROOT / f"sub-{subject}" / "timeline_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data for sub-{subject}...")
    runs = load_run_data(subject)
    if not runs:
        print("No runs loaded. Aborting.")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs. Generating timeline plots...\n")

    for run in runs:
        # Populate run["_events"] and run["_nc_frames"]
        extract_windows_for_run(run)
        plot_run_timeline(run, output_dir, subject)

    print(f"\nDone. Plots in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timeline visualization of luminance 3-class epochs"
    )
    parser.add_argument("--subject", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.subject)
