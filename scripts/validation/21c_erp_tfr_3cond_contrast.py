"""3-Condition ERP & TFR Contrast: NoChange vs ChangeUp vs ChangeDown.

Splits the original 'Change' condition by luminance direction:
- ChangeUp   (+ΔL): luminance increases (subida)
- ChangeDown (−ΔL): luminance decreases (bajada)
- NoChange:  stable moments (same as 21b)

Generates per-ROI ERP waveforms (3 lines) and TFR contrast heatmaps
(ChangeUp − NoChange, ChangeDown − NoChange).

Output: results/modeling/validation/erp_tfr_3cond_contrast/
"""

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config import EEG_CHANNELS
from config_luminance import (
    DERIVATIVES_PATH,
    ERP_N_CHANGES,
    ERP_TMAX,
    ERP_TMIN,
    LUMINANCE_CSV_MAP,
    PROJECT_ROOT,
    RUNS_CONFIG,
    SESSION,
    STIMULI_PATH,
    SUBJECT,
)
from campeones_analysis.luminance.sync import load_luminance_csv

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

ERP_TMIN = -1.0  # override for -1000 ms baseline

RESULTS_PATH = PROJECT_ROOT / "results" / "modeling" / "validation" / "erp_tfr_3cond_contrast"

ROIS = {
    "Frontal": ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'FCz'],
    "Temporal": ['T7', 'T8', 'FT9', 'FT10', 'TP9', 'TP10'],
    "Parietal": ['P3', 'P4', 'P7', 'P8', 'Pz', 'CP1', 'CP2', 'CP5', 'CP6'],
    "Occipital": ['O1', 'O2'],
}

TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)
TFR_N_CYCLES = TFR_FREQS / 2.0


# Event IDs: 1 = ChangeUp, 2 = ChangeDown, 3 = NoChange
EVENT_ID = {"ChangeUp": 1, "ChangeDown": 2, "NoChange": 3}
COND_COLORS = {"ChangeUp": "C3", "ChangeDown": "C0", "NoChange": "C2"}  # red-ish, blue-ish, green-ish


def detect_3cond_events(
    luminance_values: np.ndarray, n_events: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (up_idx, down_idx, nochange_idx) from luminance series.

    - up_idx:   indices where luminance *increased* the most  (+ΔL)
    - down_idx: indices where luminance *decreased* the most  (−ΔL)
    - nochange_idx: most stable moments, temporally separated from changes
    """
    if len(luminance_values) < 2:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    lum_diff = np.diff(luminance_values)
    abs_diff = np.abs(lum_diff)

    # Top N overall changes (same pool as 21b)
    n_top = min(n_events, len(abs_diff))
    top_indices = np.argsort(abs_diff)[::-1][:n_top]

    # Split top by sign of the diff
    up_idx = top_indices[lum_diff[top_indices] > 0]
    down_idx = top_indices[lum_diff[top_indices] < 0]
    # (zero-diff events are extremely unlikely in top-N but drop them)

    # Bottom N (NoChange) — same logic as 21b
    bottom_candidates = np.argsort(abs_diff)
    all_change = set(top_indices)
    selected_bottom: list[int] = []

    for idx in bottom_candidates:
        dist_to_top = np.min(np.abs(top_indices - idx)) if len(top_indices) > 0 else 1000
        if dist_to_top > 30:
            dist_to_bottom = (
                np.min(np.abs(np.array(selected_bottom) - idx))
                if selected_bottom
                else 1000
            )
            if dist_to_bottom > 15:
                selected_bottom.append(idx)
        if len(selected_bottom) == n_events:
            break

    return up_idx, down_idx, np.array(selected_bottom, dtype=int)


def _resolve_eeg_path(run_config: dict) -> Path | None:
    path = (
        DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-preproc_eeg.vhdr"
    )
    return path if path.exists() else None


def _resolve_events_path(run_config: dict) -> Path | None:
    merged = (
        PROJECT_ROOT / "data" / "derivatives" / "merged_events"
        / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-merged_events.tsv"
    )
    if merged.exists():
        return merged
    regular = (
        DERIVATIVES_PATH / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{run_config['task']}_acq-{run_config['acq']}_run-{run_config['id']}_desc-preproc_events.tsv"
    )
    return regular if regular.exists() else None


def select_roi_channels(available: list[str], roi: list[str]) -> list[str]:
    return [ch for ch in roi if ch in available]


def process_video_segment_3cond(
    eeg_raw: mne.io.Raw,
    onset_s: float,
    duration_s: float,
    video_id: int,
    luminance_df: pd.DataFrame,
    sfreq: float,
    roi_channels: list[str],
) -> mne.Epochs | None:
    """Create epochs with 3 conditions: ChangeUp / ChangeDown / NoChange."""
    try:
        segment_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)
    except ValueError:
        return None

    lum_vals = luminance_df["luminance"].values
    lum_times = luminance_df["timestamp"].values

    up_idx, down_idx, nc_idx = detect_3cond_events(lum_vals, ERP_N_CHANGES)
    if len(up_idx) == 0 and len(down_idx) == 0:
        return None
    if len(nc_idx) == 0:
        return None

    # Build MNE events array
    all_indices = np.concatenate([up_idx, down_idx, nc_idx])
    event_ids = np.concatenate([
        np.full(len(up_idx), EVENT_ID["ChangeUp"], dtype=int),
        np.full(len(down_idx), EVENT_ID["ChangeDown"], dtype=int),
        np.full(len(nc_idx), EVENT_ID["NoChange"], dtype=int),
    ])

    change_times_s = lum_times[all_indices + 1]
    change_samples = np.round((change_times_s - segment_raw.times[0]) * sfreq).astype(int)

    valid = (change_samples >= 0) & (change_samples < len(segment_raw.times))
    change_samples = change_samples[valid]
    event_ids = event_ids[valid]
    change_samples = change_samples + segment_raw.first_samp

    mne_events = np.column_stack([change_samples, np.zeros(len(change_samples), dtype=int), event_ids])
    mne_events = mne_events[np.argsort(mne_events[:, 0])]

    try:
        return mne.Epochs(
            segment_raw,
            events=mne_events,
            event_id=EVENT_ID,
            tmin=ERP_TMIN,
            tmax=ERP_TMAX,
            picks=roi_channels,
            baseline=None,
            preload=True,
            verbose=False,
        )
    except Exception:
        return None


# ── Luminance event-locked average ────────────────────────────────────

LUM_WINDOW_PRE = 30   # ~1 s before  (frames at ~30 fps)
LUM_WINDOW_POST = 25  # ~0.8 s after


def collect_luminance_snippets(
    luminance_df: pd.DataFrame, up_idx: np.ndarray,
    down_idx: np.ndarray, nc_idx: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Extract luminance time-courses around each event index."""
    lum = luminance_df["luminance"].values
    snippets: dict[str, list[np.ndarray]] = {
        "ChangeUp": [], "ChangeDown": [], "NoChange": [],
    }
    for label, indices in [("ChangeUp", up_idx), ("ChangeDown", down_idx), ("NoChange", nc_idx)]:
        for idx in indices:
            start = idx - LUM_WINDOW_PRE
            stop = idx + LUM_WINDOW_POST + 1
            if start < 0 or stop > len(lum):
                continue
            snippet = lum[start:stop].copy()
            snippet = snippet - snippet[:LUM_WINDOW_PRE].mean()
            snippets[label].append(snippet)
    return snippets


def plot_luminance_around_events(
    all_snippets: dict[str, list[np.ndarray]], output_dir: Path, fps: float = 30.0,
):
    """Plot mean ± SEM luminance time-locked to each condition."""
    n_frames = LUM_WINDOW_PRE + LUM_WINDOW_POST + 1
    time_ms = (np.arange(n_frames) - LUM_WINDOW_PRE) / fps * 1000

    fig, ax = plt.subplots(figsize=(10, 4))
    for cond, color in COND_COLORS.items():
        arrs = all_snippets.get(cond, [])
        if not arrs:
            continue
        mat = np.stack(arrs)
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(time_ms, mean, color=color, label=f"{cond} (n={mat.shape[0]})")
        ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.25)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time relative to event (ms)")
    ax.set_ylabel("Δ Luminance (baseline-subtracted)")
    ax.set_title(f"Luminance around detected events (sub-{SUBJECT})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{SUBJECT}_luminance_around_events.png", dpi=150)
    plt.close(fig)


def plot_luminance_timeline(
    luminance_df: pd.DataFrame, up_idx: np.ndarray, down_idx: np.ndarray,
    nc_idx: np.ndarray, video_id: int, presentation: int, output_dir: Path,
):
    """Plot full luminance time-series with shaded background per event condition."""
    lum = luminance_df["luminance"].values
    times_s = luminance_df["timestamp"].values

    fig, ax = plt.subplots(figsize=(20, 3.5))
    ax.plot(times_s, lum, color="0.3", linewidth=0.7)

    shade_half = 0.5  # ±0.5 s around each event
    event_map = [
        (up_idx, COND_COLORS["ChangeUp"], "ChangeUp"),
        (down_idx, COND_COLORS["ChangeDown"], "ChangeDown"),
        (nc_idx, COND_COLORS["NoChange"], "NoChange"),
    ]
    plotted_labels: set[str] = set()
    for indices, color, label in event_map:
        for idx in indices:
            if idx >= len(times_s):
                continue
            t_evt = times_s[idx]
            lbl = label if label not in plotted_labels else None
            ax.axvspan(t_evt - shade_half, t_evt + shade_half, alpha=0.20, color=color, label=lbl)
            ax.axvline(t_evt, color=color, linewidth=0.5, alpha=0.5)
            if lbl:
                plotted_labels.add(label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Luminance (green ch.)")
    ax.set_title(f"Luminance timeline — video {video_id}, presentation {presentation} (sub-{SUBJECT})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{SUBJECT}_lum_timeline_vid{video_id}_pres{presentation}.png", dpi=150)
    plt.close(fig)


# ── Plotting ──────────────────────────────────────────────────────────


def plot_erp_3cond(epochs: mne.Epochs, output_dir: Path):
    """ERP waveforms with 3 conditions per ROI."""
    time_pts = epochs.times * 1000

    roi_plot_data: dict = {}
    global_min, global_max = float("inf"), float("-inf")

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in epochs.ch_names]
        if not valid_chs:
            continue
        roi_plot_data[roi_name] = {}
        for cond in EVENT_ID:
            try:
                cond_data = epochs[cond].copy().pick(valid_chs).get_data()
            except KeyError:
                continue
            roi_avg = cond_data.mean(axis=1) * 1e6  # µV
            mean_erp = roi_avg.mean(axis=0)
            sem_erp = roi_avg.std(axis=0, ddof=1) / np.sqrt(roi_avg.shape[0])
            roi_plot_data[roi_name][cond] = {
                "n": roi_avg.shape[0],
                "mean": mean_erp,
                "sem": sem_erp,
            }
            global_min = min(global_min, np.min(mean_erp - sem_erp))
            global_max = max(global_max, np.max(mean_erp + sem_erp))

    margin = (global_max - global_min) * 0.05
    ylim = (global_min - margin, global_max + margin)

    for roi_name, conds in roi_plot_data.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        for cond, cp in conds.items():
            color = COND_COLORS[cond]
            ax.plot(time_pts, cp["mean"], label=f"{cond} (n={cp['n']})", color=color)
            ax.fill_between(
                time_pts, cp["mean"] - cp["sem"], cp["mean"] + cp["sem"],
                color=color, alpha=0.25,
            )
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"ERP 3-Cond Contrast: {roi_name} ROI (sub-{SUBJECT})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_erp_3cond_{roi_name}.png", dpi=150)
        plt.close(fig)

# ── Luminance event-locked average ────────────────────────────────────

# Window around each event (in frames at ~30 fps)
LUM_WINDOW_PRE = 30   # ~1 s before
LUM_WINDOW_POST = 25  # ~0.8 s after


def collect_luminance_snippets(
    luminance_df: pd.DataFrame, up_idx: np.ndarray,
    down_idx: np.ndarray, nc_idx: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Extract luminance time-courses around each event index."""
    lum = luminance_df["luminance"].values
    snippets: dict[str, list[np.ndarray]] = {
        "ChangeUp": [], "ChangeDown": [], "NoChange": [],
    }
    for label, indices in [("ChangeUp", up_idx), ("ChangeDown", down_idx), ("NoChange", nc_idx)]:
        for idx in indices:
            start = idx - LUM_WINDOW_PRE
            stop = idx + LUM_WINDOW_POST + 1
            if start < 0 or stop > len(lum):
                continue
            snippet = lum[start:stop].copy()
            # Baseline-subtract: remove mean of pre-event window
            snippet = snippet - snippet[:LUM_WINDOW_PRE].mean()
            snippets[label].append(snippet)
    return snippets


def plot_luminance_around_events(
    all_snippets: dict[str, list[np.ndarray]], output_dir: Path, fps: float = 30.0,
):
    """Plot mean ± SEM luminance time-locked to each condition."""
    n_frames = LUM_WINDOW_PRE + LUM_WINDOW_POST + 1
    time_ms = (np.arange(n_frames) - LUM_WINDOW_PRE) / fps * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 4))
    for cond, color in COND_COLORS.items():
        arrs = all_snippets.get(cond, [])
        if not arrs:
            continue
        mat = np.stack(arrs)  # (n_events, n_frames)
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(time_ms, mean, color=color, label=f"{cond} (n={mat.shape[0]})")
        ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.25)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time relative to event (ms)")
    ax.set_ylabel("Δ Luminance (baseline-subtracted)")
    ax.set_title(f"Luminance around detected events (sub-{SUBJECT})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{SUBJECT}_luminance_around_events.png", dpi=150)
    plt.close(fig)




def plot_tfr_3cond(epochs: mne.Epochs, output_dir: Path):
    """TFR contrast heatmaps: ChangeUp−NoChange and ChangeDown−NoChange per ROI."""
    conds_present = {c for c in EVENT_ID if len(epochs[c]) > 0}
    if "NoChange" not in conds_present:
        logger.warning("No NoChange epochs — skipping TFR plots.")
        return

    tfrs: dict = {}
    for cond in conds_present:
        tfr = tfr_morlet(
            epochs[cond], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
            return_itc=False, average=True,
        )
        tfr.apply_baseline(baseline=(-1.0, -0.2), mode="percent")
        tfrs[cond] = tfr

    contrasts = {}
    for change_cond in ("ChangeUp", "ChangeDown"):
        if change_cond not in tfrs:
            continue
        diff = tfrs[change_cond].copy()
        diff.data = tfrs[change_cond].data - tfrs["NoChange"].data
        contrasts[f"{change_cond} − NoChange"] = diff

    # 3rd contrast: ChangeUp − ChangeDown
    if "ChangeUp" in tfrs and "ChangeDown" in tfrs:
        diff_updown = tfrs["ChangeUp"].copy()
        diff_updown.data = tfrs["ChangeUp"].data - tfrs["ChangeDown"].data
        contrasts["ChangeUp − ChangeDown"] = diff_updown

    if not contrasts:
        return

    # Compute global vmax across all contrasts and ROIs
    global_vmax = 0
    roi_data_map: dict = {}  # (contrast_name, roi_name) -> roi_data
    for cname, diff in contrasts.items():
        for roi_name, roi_chs in ROIS.items():
            valid_chs = [c for c in roi_chs if c in diff.ch_names]
            if not valid_chs:
                continue
            ch_idx = [diff.ch_names.index(c) for c in valid_chs]
            roi_data = diff.data[ch_idx, :, :].mean(axis=0)
            roi_data_map[(cname, roi_name)] = roi_data
            global_vmax = max(global_vmax, np.percentile(np.abs(roi_data), 98))

    ref_diff = next(iter(contrasts.values()))
    times = ref_diff.times * 1000
    freqs = ref_diff.freqs

    for roi_name in ROIS:
        panels = [(cn, roi_data_map[(cn, roi_name)])
                  for cn in contrasts if (cn, roi_name) in roi_data_map]
        if not panels:
            continue

        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
        for ax, (cname, roi_data) in zip(axes[0], panels):
            im = ax.pcolormesh(
                times, freqs, roi_data, cmap="RdBu_r", shading="gouraud",
                vmin=-global_vmax, vmax=global_vmax,
            )
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.set_title(cname)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power diff (% points)")

        fig.suptitle(f"TFR — {roi_name} (ROI)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_tfr_3cond_{roi_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Pipeline ──────────────────────────────────────────────────────────


def run_pipeline():
    print("=" * 60)
    print(f"21c — 3-Cond ERP & TFR (NoChange vs ChangeUp vs ChangeDown) — sub-{SUBJECT}")
    print("=" * 60)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    roi_channels = select_roi_channels(EEG_CHANNELS, EEG_CHANNELS)
    if not roi_channels:
        return

    all_epochs: list[mne.Epochs] = []
    all_lum_snippets: dict[str, list[np.ndarray]] = {
        "ChangeUp": [], "ChangeDown": [], "NoChange": [],
    }

    for run_config in RUNS_CONFIG:
        vhdr = _resolve_eeg_path(run_config)
        events_path = _resolve_events_path(run_config)
        if not vhdr or not events_path:
            continue

        eeg_raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        events_df = pd.read_csv(events_path, sep="\t")
        rec_roi = select_roi_channels(eeg_raw.ch_names, EEG_CHANNELS)

        lum_evs = events_df[events_df["trial_type"] == "video_luminance"]
        vid_presentation_count: dict[int, int] = {}

        for _, row in lum_evs.iterrows():
            vid = int(row["stim_id"]) - 100
            csv_name = LUMINANCE_CSV_MAP.get(vid)
            if not csv_name:
                continue
            try:
                lum_df = load_luminance_csv(STIMULI_PATH / csv_name)
            except FileNotFoundError:
                continue

            vid_presentation_count[vid] = vid_presentation_count.get(vid, 0) + 1

            # Collect luminance snippets around events
            up_idx, down_idx, nc_idx = detect_3cond_events(lum_df["luminance"].values, ERP_N_CHANGES)
            snips = collect_luminance_snippets(lum_df, up_idx, down_idx, nc_idx)
            for k in all_lum_snippets:
                all_lum_snippets[k].extend(snips.get(k, []))

            # Timeline plot per video presentation
            plot_luminance_timeline(lum_df, up_idx, down_idx, nc_idx, vid, vid_presentation_count[vid], RESULTS_PATH)

            eps = process_video_segment_3cond(
                eeg_raw, float(row["onset"]), float(row["duration"]),
                vid, lum_df, eeg_raw.info["sfreq"], rec_roi,
            )
            if eps is not None:
                all_epochs.append(eps)

    if not all_epochs:
        print("No epochs created.")
        return

    grand = mne.concatenate_epochs(all_epochs)
    for cond in EVENT_ID:
        try:
            n = len(grand[cond])
        except KeyError:
            n = 0
        print(f"  {cond}: {n} epochs")

    plot_erp_3cond(grand, RESULTS_PATH)
    plot_tfr_3cond(grand, RESULTS_PATH)
    plot_luminance_around_events(all_lum_snippets, RESULTS_PATH)

    print(f"Results saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
