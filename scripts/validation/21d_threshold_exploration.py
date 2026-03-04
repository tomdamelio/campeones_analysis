"""Threshold exploration for derivative-based epoch selection.

New approach (per supervisor feedback):
- Compute luminance derivative (ΔL per frame)
- Detect threshold crossings: ChangeUp (derivative > +thr), ChangeDown (derivative < -thr)
- Sample NoChange from stable regions (|derivative| < thr/2, no overlap with Change epochs)
- No overlapping epochs allowed

Generates per-video-presentation plots of the derivative with shaded epochs,
for multiple threshold values.

Output: results/modeling/validation/threshold_exploration/thr_{value}/
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_luminance import (
    ERP_TMAX,
    LUMINANCE_CSV_MAP,
    PROJECT_ROOT,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
)
from campeones_analysis.luminance.sync import load_luminance_csv

matplotlib.use("Agg")

RESULTS_BASE = PROJECT_ROOT / "results" / "modeling" / "validation" / "threshold_exploration"

# Epoch duration in seconds (no overlap)
EPOCH_TMIN = -1.0   # 1 s pre
EPOCH_TMAX = 0.8    # 0.8 s post
EPOCH_DUR = EPOCH_TMAX - EPOCH_TMIN  # 1.8 s total

# Thresholds to explore (in luminance units per frame)
THRESHOLDS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

COND_COLORS = {"ChangeUp": "C3", "ChangeDown": "C0", "NoChange": "C2"}


def detect_threshold_events(
    lum_derivative: np.ndarray, timestamps: np.ndarray, threshold: float, fps: float,
) -> dict[str, np.ndarray]:
    """Detect non-overlapping epoch onsets from derivative threshold crossings.

    Returns dict with keys ChangeUp, ChangeDown, NoChange, each an array of
    frame indices where the epoch is centred (the crossing frame).
    """
    epoch_frames = int(EPOCH_DUR * fps)
    pre_frames = int(abs(EPOCH_TMIN) * fps)

    # --- ChangeUp: derivative crosses above +threshold ---
    above = lum_derivative > threshold
    # Find onset of each crossing (transition from below to above)
    crossings_up = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    # --- ChangeDown: derivative crosses below -threshold ---
    below = lum_derivative < -threshold
    crossings_down = np.where(np.diff(below.astype(int)) == 1)[0] + 1

    # Merge all change events and greedily select non-overlapping epochs
    all_change = []
    for idx in crossings_up:
        all_change.append((idx, "ChangeUp"))
    for idx in crossings_down:
        all_change.append((idx, "ChangeDown"))
    all_change.sort(key=lambda x: x[0])

    selected_up = []
    selected_down = []
    occupied = set()  # frame indices already used by an epoch

    for idx, cond in all_change:
        epoch_start = idx - pre_frames
        epoch_end = epoch_start + epoch_frames
        if epoch_start < 0 or epoch_end > len(lum_derivative) + 1:
            continue
        # Check overlap with already selected epochs
        epoch_range = set(range(epoch_start, epoch_end))
        if epoch_range & occupied:
            continue
        occupied |= epoch_range
        if cond == "ChangeUp":
            selected_up.append(idx)
        else:
            selected_down.append(idx)

    # --- NoChange: sample from stable regions ---
    n_change = len(selected_up) + len(selected_down)
    stable_mask = np.abs(lum_derivative) < (threshold / 2.0)

    # Find candidate frames in stable regions
    stable_candidates = np.where(stable_mask)[0]
    np.random.seed(42)
    np.random.shuffle(stable_candidates)

    selected_nc = []
    for idx in stable_candidates:
        epoch_start = idx - pre_frames
        epoch_end = epoch_start + epoch_frames
        if epoch_start < 0 or epoch_end > len(lum_derivative) + 1:
            continue
        epoch_range = set(range(epoch_start, epoch_end))
        if epoch_range & occupied:
            continue
        occupied |= epoch_range
        selected_nc.append(idx)
        if len(selected_nc) >= n_change:
            break

    return {
        "ChangeUp": np.array(selected_up, dtype=int),
        "ChangeDown": np.array(selected_down, dtype=int),
        "NoChange": np.array(sorted(selected_nc), dtype=int),
    }


def plot_derivative_timeline(
    timestamps: np.ndarray, lum_values: np.ndarray, lum_derivative: np.ndarray,
    events: dict[str, np.ndarray], threshold: float,
    video_id: int, presentation: int, fps: float, output_dir: Path,
):
    """Plot derivative (top) and raw luminance (bottom) with shaded epochs."""
    pre_s = abs(EPOCH_TMIN)
    post_s = EPOCH_TMAX

    fig, (ax_deriv, ax_lum) = plt.subplots(2, 1, figsize=(20, 7), sharex=True)

    # --- Top: derivative ---
    ax_deriv.plot(timestamps[1:], lum_derivative, color="0.4", linewidth=0.6)
    ax_deriv.axhline(threshold, color="C3", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_deriv.axhline(-threshold, color="C0", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_deriv.axhline(0, color="0.7", linewidth=0.5)

    # --- Bottom: raw luminance ---
    ax_lum.plot(timestamps, lum_values, color="0.4", linewidth=0.6)

    # Shade epochs on both subplots
    plotted_labels: set[str] = set()
    counts: dict[str, int] = {}
    for cond in ("ChangeUp", "ChangeDown", "NoChange"):
        indices = events.get(cond, np.array([], dtype=int))
        counts[cond] = len(indices)
        color = COND_COLORS[cond]
        for idx in indices:
            t_evt = timestamps[idx]
            lbl = cond if cond not in plotted_labels else None
            for ax in (ax_deriv, ax_lum):
                ax.axvspan(t_evt - pre_s, t_evt + post_s, alpha=0.18, color=color,
                           label=lbl if ax is ax_deriv else None)
                ax.axvline(t_evt, color=color, linewidth=0.5, alpha=0.5)
            if lbl:
                plotted_labels.add(cond)

    count_str = ", ".join(f"{c}: {counts.get(c, 0)}" for c in ("ChangeUp", "ChangeDown", "NoChange"))
    ax_deriv.set_ylabel("ΔLuminance / frame")
    ax_deriv.set_title(
        f"video {video_id}, pres {presentation} | thr={threshold} | {count_str}"
    )
    ax_deriv.legend(loc="upper right", fontsize=8)

    ax_lum.set_xlabel("Time (s)")
    ax_lum.set_ylabel("Luminance (green ch.)")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"sub-{SUBJECT}_deriv_vid{video_id}_pres{presentation}.png",
        dpi=150,
    )
    plt.close(fig)


def _resolve_events_path(run_config: dict) -> Path | None:
    deriv = PROJECT_ROOT / "data" / "derivatives"
    merged = (
        deriv / "merged_events" / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-merged_events.tsv"
    )
    if merged.exists():
        return merged
    regular = (
        deriv / "campeones_preproc" / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-preproc_events.tsv"
    )
    return regular if regular.exists() else None


def run():
    print("=" * 60)
    print(f"21d — Threshold exploration for derivative-based epochs — sub-{SUBJECT}")
    print("=" * 60)

    for thr in THRESHOLDS:
        thr_dir = RESULTS_BASE / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Threshold = {thr} ---")

        vid_count: dict[int, int] = {}

        for run_config in RUNS_CONFIG:
            events_path = _resolve_events_path(run_config)
            if not events_path:
                continue

            events_df = pd.read_csv(events_path, sep="\t")
            lum_evs = events_df[events_df["trial_type"] == "video_luminance"]

            for _, row in lum_evs.iterrows():
                vid = int(row["stim_id"]) - 100
                csv_name = LUMINANCE_CSV_MAP.get(vid)
                if not csv_name:
                    continue
                try:
                    lum_df = load_luminance_csv(STIMULI_PATH / csv_name)
                except FileNotFoundError:
                    continue

                vid_count[vid] = vid_count.get(vid, 0) + 1
                pres = vid_count[vid]

                lum_vals = lum_df["luminance"].values
                lum_times = lum_df["timestamp"].values
                fps = 1.0 / np.median(np.diff(lum_times))

                lum_deriv = np.diff(lum_vals)

                evts = detect_threshold_events(lum_deriv, lum_times, thr, fps)
                print(
                    f"  vid {vid} pres {pres}: "
                    f"Up={len(evts['ChangeUp'])}, Down={len(evts['ChangeDown'])}, "
                    f"NC={len(evts['NoChange'])}"
                )

                plot_derivative_timeline(
                    lum_times, lum_vals, lum_deriv, evts, thr, vid, pres, fps, thr_dir,
                )

    print(f"\nResults saved to {RESULTS_BASE}")
    print("Done.")


if __name__ == "__main__":
    run()
