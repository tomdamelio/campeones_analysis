"""ERP & TFR 3-condition contrast using derivative threshold-based epoching.

Uses the approach proposed by supervisor:
- Compute luminance derivative per video
- Threshold crossings define ChangeUp (+thr) and ChangeDown (-thr) events
- NoChange sampled from stable regions (|deriv| < thr/2)
- No overlapping epochs

Threshold = 1.5 luminance units/frame.

Output: results/modeling/validation/erp_tfr_deriv_thr1.5/
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

THRESHOLD = 1.0
ERP_TMIN = -1.8
ERP_TMAX = 0.8
EPOCH_DUR = ERP_TMAX - ERP_TMIN  # 2.6 s
PLOT_TMIN_MS = -1000  # only plot from -1000 ms onwards

RESULTS_PATH = PROJECT_ROOT / "results" / "modeling" / "validation" / "erp_tfr_deriv_thr1.0"

EVENT_ID = {"ChangeUp": 1, "ChangeDown": 2, "NoChange": 3}
COND_COLORS = {"ChangeUp": "C3", "ChangeDown": "C0", "NoChange": "C2"}

ROIS = {
    "Frontal": ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'FCz'],
    "Temporal": ['T7', 'T8', 'FT9', 'FT10', 'TP9', 'TP10'],
    "Parietal": ['P3', 'P4', 'P7', 'P8', 'Pz', 'CP1', 'CP2', 'CP5', 'CP6'],
    "Occipital": ['O1', 'O2'],
}

TFR_FREQS = np.logspace(*np.log10([3, 40]), num=20)
TFR_N_CYCLES = TFR_FREQS / 2.0


# ── Event detection (from 21d) ────────────────────────────────────────


def detect_threshold_events(
    lum_derivative: np.ndarray, fps: float,
) -> dict[str, np.ndarray]:
    """Detect non-overlapping epochs from derivative threshold crossings."""
    epoch_frames = int(EPOCH_DUR * fps)
    pre_frames = int(abs(ERP_TMIN) * fps)

    above = lum_derivative > THRESHOLD
    crossings_up = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    below = lum_derivative < -THRESHOLD
    crossings_down = np.where(np.diff(below.astype(int)) == 1)[0] + 1

    all_change = sorted(
        [(idx, "ChangeUp") for idx in crossings_up]
        + [(idx, "ChangeDown") for idx in crossings_down],
        key=lambda x: x[0],
    )

    selected_up, selected_down = [], []
    occupied: set[int] = set()

    for idx, cond in all_change:
        epoch_start = idx - pre_frames
        epoch_end = epoch_start + epoch_frames
        if epoch_start < 0 or epoch_end > len(lum_derivative) + 1:
            continue
        epoch_range = set(range(epoch_start, epoch_end))
        if epoch_range & occupied:
            continue
        occupied |= epoch_range
        (selected_up if cond == "ChangeUp" else selected_down).append(idx)

    # NoChange from stable regions
    n_change = len(selected_up) + len(selected_down)
    stable_mask = np.abs(lum_derivative) < (THRESHOLD / 2.0)
    stable_candidates = np.where(stable_mask)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(stable_candidates)

    selected_nc: list[int] = []
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


# ── EEG epoch creation ───────────────────────────────────────────────


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


def create_epochs_from_threshold(
    eeg_raw: mne.io.Raw, onset_s: float, duration_s: float,
    luminance_df: pd.DataFrame, sfreq: float, roi_channels: list[str],
) -> mne.Epochs | None:
    """Create MNE Epochs using derivative threshold crossings."""
    try:
        segment_raw = eeg_raw.copy().crop(tmin=onset_s, tmax=onset_s + duration_s)
    except ValueError:
        return None

    lum_vals = luminance_df["luminance"].values
    lum_times = luminance_df["timestamp"].values
    fps = 1.0 / np.median(np.diff(lum_times))
    lum_deriv = np.diff(lum_vals)

    evts = detect_threshold_events(lum_deriv, fps)

    # Build MNE events array
    all_indices, all_ids = [], []
    for cond, eid in EVENT_ID.items():
        for idx in evts[cond]:
            all_indices.append(idx)
            all_ids.append(eid)

    if not all_indices:
        return None

    all_indices = np.array(all_indices)
    all_ids = np.array(all_ids)

    # Map luminance frame indices to EEG sample indices
    event_times_s = lum_times[all_indices]
    eeg_samples = np.round((event_times_s - segment_raw.times[0]) * sfreq).astype(int)

    valid = (eeg_samples >= 0) & (eeg_samples < len(segment_raw.times))
    eeg_samples = eeg_samples[valid]
    all_ids = all_ids[valid]
    eeg_samples = eeg_samples + segment_raw.first_samp

    mne_events = np.column_stack([eeg_samples, np.zeros(len(eeg_samples), dtype=int), all_ids])
    mne_events = mne_events[np.argsort(mne_events[:, 0])]

    try:
        return mne.Epochs(
            segment_raw, events=mne_events, event_id=EVENT_ID,
            tmin=ERP_TMIN, tmax=ERP_TMAX, picks=roi_channels,
            baseline=(-1.8, -1.0), preload=True, verbose=False,
        )
    except Exception:
        return None


# ── Plotting (same as 21c) ────────────────────────────────────────────


def plot_erp_3cond(epochs: mne.Epochs, output_dir: Path, ylim: tuple | None = None):
    time_pts = epochs.times * 1000
    # Only plot from -1000 ms onwards (baseline is hidden)
    plot_mask = time_pts >= PLOT_TMIN_MS
    time_vis = time_pts[plot_mask]

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
            if cond_data.shape[0] == 0:
                continue
            roi_avg = cond_data.mean(axis=1) * 1e6
            mean_erp = roi_avg.mean(axis=0)[plot_mask]
            sem_erp = (roi_avg.std(axis=0, ddof=1) / np.sqrt(roi_avg.shape[0]))[plot_mask]
            roi_plot_data[roi_name][cond] = {
                "n": roi_avg.shape[0], "mean": mean_erp, "sem": sem_erp,
            }
            global_min = min(global_min, np.min(mean_erp - sem_erp))
            global_max = max(global_max, np.max(mean_erp + sem_erp))

    if global_min == float("inf"):
        return (0, 0)

    if ylim is None:
        margin = (global_max - global_min) * 0.05
        ylim = (global_min - margin, global_max + margin)

    for roi_name, conds in roi_plot_data.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        for cond, cp in conds.items():
            color = COND_COLORS[cond]
            ax.plot(time_vis, cp["mean"], label=f"{cond} (n={cp['n']})", color=color)
            ax.fill_between(
                time_vis, cp["mean"] - cp["sem"], cp["mean"] + cp["sem"],
                color=color, alpha=0.25,
            )
        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"ERP 3-Cond (deriv thr={THRESHOLD}): {roi_name} ROI (sub-{SUBJECT})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_erp_3cond_{roi_name}.png", dpi=150)
        plt.close(fig)

    return ylim


def plot_tfr_3cond(epochs: mne.Epochs, output_dir: Path, shared_vmax: float | None = None):
    conds_present = {c for c in EVENT_ID if len(epochs[c]) > 0}
    if "NoChange" not in conds_present:
        logger.warning("No NoChange epochs — skipping TFR.")
        return 0

    tfrs: dict = {}
    for cond in conds_present:
        tfr = tfr_morlet(
            epochs[cond], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
            return_itc=False, average=True,
        )
        tfr.apply_baseline(baseline=(-1.8, -1.0), mode="percent")
        tfr.data *= 100  # convert fraction to percent
        tfrs[cond] = tfr

    contrasts: dict = {}
    for change_cond in ("ChangeUp", "ChangeDown"):
        if change_cond not in tfrs:
            continue
        diff = tfrs[change_cond].copy()
        diff.data = tfrs[change_cond].data - tfrs["NoChange"].data
        contrasts[f"{change_cond} − NoChange"] = diff

    if "ChangeUp" in tfrs and "ChangeDown" in tfrs:
        diff_ud = tfrs["ChangeUp"].copy()
        diff_ud.data = tfrs["ChangeUp"].data - tfrs["ChangeDown"].data
        contrasts["ChangeUp − ChangeDown"] = diff_ud

    if not contrasts:
        return 0

    global_vmax = 0
    roi_data_map: dict = {}
    for cname, diff in contrasts.items():
        for roi_name, roi_chs in ROIS.items():
            valid_chs = [c for c in roi_chs if c in diff.ch_names]
            if not valid_chs:
                continue
            ch_idx = [diff.ch_names.index(c) for c in valid_chs]
            roi_data = diff.data[ch_idx, :, :].mean(axis=0)
            roi_data_map[(cname, roi_name)] = roi_data

    ref_diff = next(iter(contrasts.values()))
    times = ref_diff.times * 1000
    freqs = ref_diff.freqs

    # Crop to visible window (hide baseline period)
    t_mask = times >= PLOT_TMIN_MS
    times_vis = times[t_mask]

    for key in list(roi_data_map.keys()):
        roi_data_map[key] = roi_data_map[key][:, t_mask]

    # Compute vmax on visible data only
    for roi_data in roi_data_map.values():
        global_vmax = max(global_vmax, np.percentile(np.abs(roi_data), 98))

    # Use shared scale if provided
    plot_vmax = shared_vmax if shared_vmax is not None else global_vmax

    for roi_name in ROIS:
        panels = [(cn, roi_data_map[(cn, roi_name)])
                  for cn in contrasts if (cn, roi_name) in roi_data_map]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
        for ax, (cname, roi_data) in zip(axes[0], panels):
            im = ax.pcolormesh(
                times_vis, freqs, roi_data, cmap="RdBu_r", shading="gouraud",
                vmin=-plot_vmax, vmax=plot_vmax,
            )
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.set_title(cname)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power diff (%)")
        fig.suptitle(f"TFR (deriv thr={THRESHOLD}) — {roi_name} (ROI)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_tfr_3cond_{roi_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return global_vmax


def plot_null_contrast(epochs: mne.Epochs, output_dir: Path,
                      shared_vmax: float | None = None, shared_ylim: tuple | None = None):
    """Split NoChange in half (interleaved) and plot TFR + ERP null contrasts."""
    nc_epochs = epochs["NoChange"]
    n_nc = len(nc_epochs)
    if n_nc < 4:
        logger.warning("Too few NoChange epochs for null contrast.")
        return

    # Interleaved split: odd-indexed epochs vs even-indexed epochs.
    # This ensures both halves sample evenly across all videos/runs,
    # since epochs are concatenated in temporal order.
    idx_a = np.arange(0, n_nc, 2)  # even: 0, 2, 4, ...
    idx_b = np.arange(1, n_nc, 2)  # odd:  1, 3, 5, ...
    nc_a = nc_epochs[idx_a]
    nc_b = nc_epochs[idx_b]
    half = len(idx_a)

    # ── TFR null contrast ──
    tfr_a = tfr_morlet(nc_a, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)
    tfr_b = tfr_morlet(nc_b, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)
    tfr_a.apply_baseline(baseline=(-1.8, -1.0), mode="percent")
    tfr_b.apply_baseline(baseline=(-1.8, -1.0), mode="percent")
    tfr_a.data *= 100
    tfr_b.data *= 100

    tfr_null = tfr_a.copy()
    tfr_null.data = tfr_a.data - tfr_b.data

    # Also compute ChangeUp - NoChange for side-by-side comparison
    conds_present = {c for c in EVENT_ID if len(epochs[c]) > 0}
    tfr_contrast = None
    if "ChangeUp" in conds_present:
        tfr_cu = tfr_morlet(epochs["ChangeUp"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)
        tfr_nc = tfr_morlet(epochs["NoChange"], freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES, return_itc=False, average=True)
        tfr_cu.apply_baseline(baseline=(-1.8, -1.0), mode="percent")
        tfr_nc.apply_baseline(baseline=(-1.8, -1.0), mode="percent")
        tfr_cu.data *= 100
        tfr_nc.data *= 100
        tfr_contrast = tfr_cu.copy()
        tfr_contrast.data = tfr_cu.data - tfr_nc.data

    times = tfr_null.times * 1000
    freqs = tfr_null.freqs
    t_mask = times >= PLOT_TMIN_MS
    times_vis = times[t_mask]

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in tfr_null.ch_names]
        if not valid_chs:
            continue
        ch_idx = [tfr_null.ch_names.index(c) for c in valid_chs]

        null_data = tfr_null.data[ch_idx, :, :].mean(axis=0)[:, t_mask]

        panels = [("NoChange_A − NoChange_B\n(null)", null_data)]
        if tfr_contrast is not None:
            contrast_data = tfr_contrast.data[ch_idx, :, :].mean(axis=0)[:, t_mask]
            panels.append(("ChangeUp − NoChange\n(real)", contrast_data))

        # Use shared vmax if provided, otherwise compute from these panels
        if shared_vmax is not None:
            vmax = shared_vmax
        else:
            all_data = np.concatenate([p[1].ravel() for p in panels])
            vmax = np.percentile(np.abs(all_data), 98)

        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
        for ax, (title, data) in zip(axes[0], panels):
            im = ax.pcolormesh(times_vis, freqs, data, cmap="RdBu_r", shading="gouraud", vmin=-vmax, vmax=vmax)
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power diff (%)")
        fig.suptitle(f"Null contrast sanity check — {roi_name} (ROI)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_null_contrast_tfr_{roi_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── ERP null contrast ──
    time_pts = nc_a.times * 1000
    plot_mask = time_pts >= PLOT_TMIN_MS
    time_vis = time_pts[plot_mask]

    for roi_name, roi_chs in ROIS.items():
        valid_chs = [c for c in roi_chs if c in nc_a.ch_names]
        if not valid_chs:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        for label, ep, color in [
            (f"NoChange_A (n={len(nc_a)})", nc_a, "C4"),
            (f"NoChange_B (n={len(nc_b)})", nc_b, "C5"),
        ]:
            data = ep.copy().pick(valid_chs).get_data().mean(axis=1) * 1e6
            mean = data.mean(axis=0)[plot_mask]
            sem = (data.std(axis=0, ddof=1) / np.sqrt(data.shape[0]))[plot_mask]
            ax.plot(time_vis, mean, label=label, color=color)
            ax.fill_between(time_vis, mean - sem, mean + sem, color=color, alpha=0.25)

        # Also plot ChangeUp for reference
        if "ChangeUp" in conds_present:
            cu_data = epochs["ChangeUp"].copy().pick(valid_chs).get_data().mean(axis=1) * 1e6
            cu_mean = cu_data.mean(axis=0)[plot_mask]
            cu_sem = (cu_data.std(axis=0, ddof=1) / np.sqrt(cu_data.shape[0]))[plot_mask]
            ax.plot(time_vis, cu_mean, label=f"ChangeUp (n={cu_data.shape[0]})", color=COND_COLORS["ChangeUp"])
            ax.fill_between(time_vis, cu_mean - cu_sem, cu_mean + cu_sem, color=COND_COLORS["ChangeUp"], alpha=0.15)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"Null contrast ERP: {roi_name} ROI (sub-{SUBJECT})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        if shared_ylim is not None:
            ax.set_ylim(shared_ylim)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sub-{SUBJECT}_null_contrast_erp_{roi_name}.png", dpi=150)
        plt.close(fig)

    print(f"  Null contrast plots saved (NoChange split: {half} vs {n_nc - half})")


# ── Luminance around events ──────────────────────────────────────────


def plot_luminance_around_events(
    all_snippets: dict[str, list[np.ndarray]], output_dir: Path, fps: float = 60.0,
):
    n_frames = int(abs(ERP_TMIN) * fps) + int(ERP_TMAX * fps) + 1
    pre_frames = int(abs(ERP_TMIN) * fps)
    time_ms = (np.arange(n_frames) - pre_frames) / fps * 1000

    fig, ax = plt.subplots(figsize=(10, 4))
    for cond, color in COND_COLORS.items():
        arrs = all_snippets.get(cond, [])
        if not arrs:
            continue
        # Trim/pad to same length
        arrs = [a[:n_frames] for a in arrs if len(a) >= n_frames]
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
    ax.set_title(f"Luminance around events (deriv thr={THRESHOLD}, sub-{SUBJECT})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"sub-{SUBJECT}_luminance_around_events.png", dpi=150)
    plt.close(fig)


# ── Pipeline ──────────────────────────────────────────────────────────


def run_pipeline():
    print("=" * 60)
    print(f"21e — ERP & TFR (derivative threshold={THRESHOLD}) — sub-{SUBJECT}")
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

        for _, row in lum_evs.iterrows():
            vid = int(row["stim_id"]) - 100
            csv_name = LUMINANCE_CSV_MAP.get(vid)
            if not csv_name:
                continue
            try:
                lum_df = load_luminance_csv(STIMULI_PATH / csv_name)
            except FileNotFoundError:
                continue

            # Collect luminance snippets for the event-locked average plot
            lum_vals = lum_df["luminance"].values
            lum_times = lum_df["timestamp"].values
            fps = 1.0 / np.median(np.diff(lum_times))
            lum_deriv = np.diff(lum_vals)
            evts = detect_threshold_events(lum_deriv, fps)

            pre_frames = int(abs(ERP_TMIN) * fps)
            post_frames = int(ERP_TMAX * fps)
            for cond in EVENT_ID:
                for idx in evts[cond]:
                    start = idx - pre_frames
                    stop = idx + post_frames + 1
                    if start < 0 or stop > len(lum_vals):
                        continue
                    snippet = lum_vals[start:stop].copy()
                    snippet = snippet - snippet[:pre_frames].mean()
                    all_lum_snippets[cond].append(snippet)

            # Create EEG epochs
            eps = create_epochs_from_threshold(
                eeg_raw, float(row["onset"]), float(row["duration"]),
                lum_df, eeg_raw.info["sfreq"], rec_roi,
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

    erp_ylim = plot_erp_3cond(grand, RESULTS_PATH)
    tfr_vmax = plot_tfr_3cond(grand, RESULTS_PATH)
    plot_null_contrast(grand, RESULTS_PATH, shared_vmax=tfr_vmax, shared_ylim=erp_ylim)
    plot_luminance_around_events(all_lum_snippets, RESULTS_PATH)

    print(f"Results saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
