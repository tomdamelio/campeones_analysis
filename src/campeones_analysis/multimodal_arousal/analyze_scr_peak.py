"""3rd condition (SCR peak-aligned) repeated for time / freq / time-frequency domains.

Conditions evaluated:
  1. real_onset -- SCR-onset-aligned, window (-5, +3) s, baseline (-5, -4.5)
                   (same as erp_scr.py / tfr_psd_scr.py)
  2. silent     -- silent-EDA matched control, window (-5, +3) s, baseline (-5, -4.5)
                   (same silent set as in erp_scr.py)
  3. real_peak  -- SCR-peak-aligned, window (-4, +4) s, baseline (-4, -3.5)  (NEW)
                   Uses the SAME underlying clean SCR events as condition 1; only the
                   alignment point (peak vs onset) and the window (symmetric vs asymmetric)
                   differ. Adds a cleanliness check that the peak-centered window also
                   contains no OTHER SCR onsets beyond the current one.

Outputs (research_diary/context/05_02/figures/peak_aligned/):
  Y3_epoch_audit_3cond_<sub>.png       -- 8 runs stacked: onsets + peaks + silent
  Y3_erp_3cond_<sub>.png               -- per-subject ERP, 4 ROIs, 3 lines
  Y3_erp_3cond_grandaverage.png        -- grand-average ERP
  Y3_psd_3cond_<sub>.png               -- per-subject PSD with SEM band
  Y3_psd_3cond_grandaverage.png        -- grand-average PSD with SD band
  Y3_tfr_3cond_<sub>.png               -- per-subject TFR (3 cols x 4 ROIs)
  Y3_tfr_3cond_grandaverage.png        -- grand-average TFR

CSVs (research_diary/context/05_02/y_candidates/):
  erp_3cond_roi_amplitudes.csv
  psd_3cond_band_summary.csv

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.analyze_scr_peak
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet

# Reuse infrastructure from erp_scr.py (epoching, cleanliness, silent sampling)
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    BASELINE,
    EDA_FS,
    NPZ_DIR,
    OUT,
    PRE_PHASIC_THRESH,
    SUBJECTS,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_and_peaks_s,
    epoch_one_run,
    real_scr_is_clean,
    run_label,
    runs_for,
    sample_silent_controls,
    silent_window_is_clean,
    EPOCH_SPAN_S,
    REQUIRE_CLEAN_SCR,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- output folder (separate so we don't overwrite the existing analysis) ---
FIG_DIR = OUT / "figures" / "peak_aligned"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- peak-aligned condition parameters ---
TMIN_PEAK, TMAX_PEAK = -4.0, 4.0
BASELINE_PEAK = (-4.0, -3.5)
PRE_S_PEAK = abs(TMIN_PEAK)
POST_S_PEAK = TMAX_PEAK

# --- ROIs (same as the rest of the pipeline) ---
ROIS: dict[str, list[str]] = {
    "Frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC1", "FCz", "FC2"],
    "Temporal":  ["T7", "T8", "FT9", "FT10", "TP9", "TP10", "FC5", "FC6", "CP5", "CP6"],
    "Parietal":  ["C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4"],
    "Occipital": ["O1", "O2", "P7", "P8"],
}
ROI_NAMES = list(ROIS.keys())

# --- TFR parameters (same as tfr_psd_scr.py for consistency) ---
TFR_FREQS = np.logspace(np.log10(1.0), np.log10(40.0), 30)
TFR_N_CYCLES = TFR_FREQS / 2.0
TFR_DECIM = 4

# --- PSD parameters ---
PSD_FMIN = 1.0
PSD_FMAX = 40.0
PSD_NFFT = 512

# --- colors per condition ---
COND_COLOR = {
    "real_onset": "C3",        # red
    "silent":     "0.4",       # gray
    "real_peak":  "C0",        # blue
}
COND_LS = {
    "real_onset": "-",
    "silent":     "--",
    "real_peak":  "-",
}
COND_LABEL = {
    "real_onset": "real onset-aligned (-5, +3)",
    "silent":     "silent-EDA control (-5, +3)",
    "real_peak":  "real peak-aligned (-4, +4)",
}
from src.campeones_analysis.multimodal_arousal.cohort import SUBJ_COLORS  # noqa: E402

# Fresh RNG with the same seed as erp_scr.py so silent controls match.
RNG = np.random.default_rng(20260513)


# -----------------------------------------------------------------------------
# Cleanliness rule for the peak-aligned window
# -----------------------------------------------------------------------------
def peak_window_is_clean(onset_t: float, peak_t: float, all_onsets: np.ndarray,
                          phasic: np.ndarray, fs: float = EDA_FS,
                          pre_s: float = PRE_S_PEAK, post_s: float = POST_S_PEAK) -> bool:
    """Check the peak-centered window [peak-pre_s, peak+post_s] is dominated by the
    current SCR only -- i.e. no OTHER SCR onset in the window.

    The current SCR's rise lives in PRE-peak (so we cannot require phasic-silent there).
    The current SCR's recovery lives in POST-peak (we also cannot require silent there).
    So we only check that the NEXT (or PREVIOUS) SCR onset doesn't fall inside the window.
    """
    eps = 1.5 / fs
    lo = peak_t - pre_s
    hi = peak_t + post_s
    others = all_onsets[(all_onsets >= lo) & (all_onsets <= hi)]
    others = others[np.abs(others - onset_t) > eps]  # exclude current
    if len(others) > 0:
        return False
    # also need the epoch to fit inside the run -- caller checks via epoch_one_run, but
    # we also guard here:
    n = len(phasic)
    if int(np.floor(lo * fs)) < 0 or int(np.ceil(hi * fs)) >= n:
        return False
    return True


def filter_clean_onsets_and_peaks(onsets_s: np.ndarray, peaks_s: np.ndarray,
                                    phasic: np.ndarray, fs: float = EDA_FS) -> tuple[np.ndarray, np.ndarray]:
    """Apply BOTH the onset-window cleanliness (from erp_scr.py) AND the peak-window
    no-other-onset rule. Returns (kept_onsets, kept_peaks) paired arrays.
    """
    if onsets_s.size == 0:
        return onsets_s, peaks_s
    keep_idx = []
    for i, (ot, pt) in enumerate(zip(onsets_s, peaks_s)):
        # cleanliness gates are skipped when REQUIRE_CLEAN_SCR is False (2026-05-27 v2)
        if REQUIRE_CLEAN_SCR:
            if not real_scr_is_clean(float(ot), phasic, fs, onsets_s):
                continue
            if not peak_window_is_clean(float(ot), float(pt), onsets_s, phasic, fs):
                continue
        keep_idx.append(i)
    onsets_k = np.asarray(onsets_s, dtype=float)[keep_idx]
    peaks_k = np.asarray(peaks_s, dtype=float)[keep_idx]
    # enforce non-overlapping onset-centered windows (carry paired peaks); greedy by time
    if onsets_k.size:
        order = np.argsort(onsets_k)
        onsets_k, peaks_k = onsets_k[order], peaks_k[order]
        sel: list[int] = []
        last = -np.inf
        for j, ot in enumerate(onsets_k):
            if float(ot) - last >= EPOCH_SPAN_S:
                sel.append(j)
                last = float(ot)
        onsets_k, peaks_k = onsets_k[sel], peaks_k[sel]
    return onsets_k, peaks_k


# -----------------------------------------------------------------------------
# Build 3 condition's epochs per subject
# -----------------------------------------------------------------------------
def build_subject_3cond_epochs(sub: str) -> dict | None:
    """Return dict with keys: real_onset, silent, real_peak, each holding a concatenated
    mne.Epochs, plus per-run metadata for the audit figure.
    """
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])

    real_onset_list: list[mne.Epochs] = []
    silent_list: list[mne.Epochs] = []
    real_peak_list: list[mne.Epochs] = []
    audit_per_run: dict = {}

    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.resample(250.0, verbose="ERROR")
            duration = float(raw.times[-1])

            phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets_all, peaks_all = detect_scr_onsets_and_peaks_s(phasic, EDA_FS)
            onsets_all = onsets_all[onsets_all < duration]
            peaks_all = peaks_all[: len(onsets_all)]

            # apply BOTH cleanliness rules (onset-window AND peak-window)
            onsets_clean, peaks_clean = filter_clean_onsets_and_peaks(onsets_all, peaks_all, phasic, EDA_FS)

            # silent control matched to onset window
            silent_t = sample_silent_controls(
                n_target=len(onsets_clean), duration_s=duration,
                phasic=phasic, fs=EDA_FS, rng=RNG, avoid_onsets_s=onsets_clean,
            )

            # build the 3 sets of epochs
            ep_real_onset = epoch_one_run(raw, onsets_clean, code=1,
                                            tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            ep_silent = epoch_one_run(raw, silent_t, code=2,
                                       tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            ep_real_peak = epoch_one_run(raw, peaks_clean, code=3,
                                          tmin=TMIN_PEAK, tmax=TMAX_PEAK, baseline=BASELINE_PEAK)
            if ep_real_onset is not None: real_onset_list.append(ep_real_onset)
            if ep_silent is not None: silent_list.append(ep_silent)
            if ep_real_peak is not None: real_peak_list.append(ep_real_peak)

            audit_per_run[label] = dict(
                t=np.asarray(cont[f"{label}__eda_t"], float),
                phasic=phasic,
                duration_s=duration,
                onsets_all=onsets_all,
                peaks_all=peaks_all,
                onsets_clean=onsets_clean,
                peaks_clean=peaks_clean,
                silent_t=silent_t,
            )
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    if not (real_onset_list and silent_list and real_peak_list):
        return None
    out = {
        "real_onset": mne.concatenate_epochs(real_onset_list, verbose="ERROR"),
        "silent":     mne.concatenate_epochs(silent_list, verbose="ERROR"),
        "real_peak":  mne.concatenate_epochs(real_peak_list, verbose="ERROR"),
        "audit":      audit_per_run,
    }
    return out


# -----------------------------------------------------------------------------
# ROI helpers
# -----------------------------------------------------------------------------
def roi_channel_indices(ch_names: list[str]) -> dict[str, list[int]]:
    return {roi: [ch_names.index(c) for c in chs if c in ch_names] for roi, chs in ROIS.items()}


def evoked_roi_mean(evoked: mne.Evoked, idxs: list[int]) -> np.ndarray:
    if not idxs:
        return np.full(len(evoked.times), np.nan)
    return evoked.data[idxs].mean(axis=0)


def _roi_psd_db(psd_per_epoch: np.ndarray, roi_idxs: list[int]) -> np.ndarray:
    """psd_per_epoch shape: (n_epochs, n_channels, n_freqs). Returns (n_epochs, n_freqs) in dB."""
    if not roi_idxs:
        return np.full((psd_per_epoch.shape[0], psd_per_epoch.shape[2]), np.nan)
    roi_pow = psd_per_epoch[:, roi_idxs, :].mean(axis=1)
    return 10.0 * np.log10(roi_pow + 1e-30)


def tfr_roi_mean(tfr: mne.time_frequency.AverageTFR, idxs: list[int]) -> np.ndarray:
    if not idxs:
        return np.full((tfr.data.shape[1], tfr.data.shape[2]), np.nan)
    return tfr.data[idxs].mean(axis=0)


# -----------------------------------------------------------------------------
# Epoch-audit figure (per subject, all runs stacked)
# -----------------------------------------------------------------------------
def plot_epoch_audit_3cond(sub: str, audit: dict, out_png: Path) -> None:
    run_labels = list(audit.keys())
    n_runs = len(run_labels)
    fig, axes = plt.subplots(n_runs, 1, figsize=(20, 1.5 * n_runs + 1.5))
    if n_runs == 1:
        axes = [axes]
    for i, label in enumerate(run_labels):
        ax = axes[i]
        info = audit[label]
        t = info["t"]
        phasic = info["phasic"]
        ax.plot(t, phasic, color="C2", lw=0.5, alpha=0.7, zorder=2)
        y_high = max(0.01, float(phasic.max()) * 1.30)
        ax.set_ylim(-0.0015, y_high)
        y_clean = y_high * 0.78
        y_peak = y_high * 0.86
        y_drop = y_high * 0.94
        y_silent = -0.0010

        dropped_onsets = np.setdiff1d(info["onsets_all"], info["onsets_clean"])
        # epoch window shading (very light)
        for tt in info["onsets_clean"]:
            ax.axvspan(tt + TMIN, tt + TMAX, color="C3", alpha=0.035, zorder=0)
        for tt in info["peaks_clean"]:
            ax.axvspan(tt + TMIN_PEAK, tt + TMAX_PEAK, color="C0", alpha=0.025, zorder=0)
        for tt in info["silent_t"]:
            ax.axvspan(tt + TMIN, tt + TMAX, color="0.5", alpha=0.025, zorder=0)

        if len(info["onsets_clean"]) > 0:
            ax.scatter(info["onsets_clean"], np.full(len(info["onsets_clean"]), y_clean),
                       marker="v", color="C3", s=24, edgecolors="black", linewidths=0.3, zorder=4,
                       label=f"clean SCR onset (N={len(info['onsets_clean'])})")
        if len(info["peaks_clean"]) > 0:
            ax.scatter(info["peaks_clean"], np.full(len(info["peaks_clean"]), y_peak),
                       marker="*", color="purple", s=40, edgecolors="black", linewidths=0.3, zorder=4,
                       label=f"clean SCR peak (N={len(info['peaks_clean'])})")
        if len(dropped_onsets) > 0:
            ax.scatter(dropped_onsets, np.full(len(dropped_onsets), y_drop),
                       marker="x", color="0.5", s=22, linewidths=0.8, zorder=3,
                       label=f"dropped (N={len(dropped_onsets)})")
        if len(info["silent_t"]) > 0:
            ax.scatter(info["silent_t"], np.full(len(info["silent_t"]), y_silent),
                       marker="o", color="0.3", s=14, edgecolors="black", linewidths=0.3, alpha=0.85, zorder=4,
                       label=f"silent ctrl (N={len(info['silent_t'])})")
        ax.set_title(f"{label}   (duration {info['duration_s']:.0f} s)", fontsize=8, loc="left")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel("phasic", fontsize=7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=4)
        if i == n_runs - 1:
            ax.set_xlabel("time in run (s)", fontsize=8)

    fig.suptitle(
        f"{sub}   --   Epoch audit (3 conditions: real_onset, silent, real_peak)\n"
        f"Clean SCR onsets (red ▼) feed real_onset condition; clean SCR peaks (purple ★) feed real_peak; "
        f"silent control points (gray ●) feed silent. Light shading shows each marker's epoch window "
        f"(real_onset: red, real_peak: blue, silent: gray).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# ERP plots (per subject + grand average)
# -----------------------------------------------------------------------------
def plot_erp_subject_3cond(sub: str, evokeds: dict[str, mne.Evoked], out_png: Path) -> pd.DataFrame:
    """4 ROIs in cols, 3 lines per panel (real_onset, silent, real_peak).

    Each line spans its own window; the x-axis is the union of all condition times.
    """
    ch_names = evokeds["real_onset"].ch_names
    roi_idx = roi_channel_indices(ch_names)

    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(22, 5), sharey=True)
    fig.suptitle(
        f"{sub}  --  ERP, 3 conditions: real_onset / silent / real_peak  "
        f"(n_real={len(evokeds['real_onset'].info['ch_names']) and -1})",
        fontsize=10,
    )
    rows = []
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            ax.text(0.5, 0.5, f"{roi}: no chans", ha="center", va="center", transform=ax.transAxes)
            continue
        for cond, ev in evokeds.items():
            t_ms = ev.times * 1000.0
            d = evoked_roi_mean(ev, idxs) * 1e6
            ax.plot(t_ms, d, color=COND_COLOR[cond], lw=1.5,
                     ls=COND_LS[cond], label=COND_LABEL[cond])
            # peak amplitudes per condition for CSV
            t_axis = ev.times
            pre_mask = (t_axis >= -3.0) & (t_axis <= 0.0)
            post_mask = (t_axis >= 0.0) & (t_axis <= 3.0)
            rows.append(dict(
                subject=sub, condition=cond, roi=roi,
                pre_peak_abs_uV=float(np.max(np.abs(d[pre_mask]))) if pre_mask.any() else np.nan,
                post_peak_abs_uV=float(np.max(np.abs(d[post_mask]))) if post_mask.any() else np.nan,
            ))
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from alignment point (ms)")
        if col == 0:
            ax.set_ylabel("uV")
        ax.set_title(roi)
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_erp_grand_average_3cond(per_sub: dict, out_png: Path) -> None:
    """3 conditions averaged across subjects."""
    any_ev = next(iter(per_sub.values()))["real_onset"]
    ch_names = any_ev.ch_names
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(22, 5), sharey=True)
    fig.suptitle(f"Grand-average ERP, 3 conditions  (N={len(per_sub)} subjects)", fontsize=10)
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            continue
        for cond in ("real_onset", "silent", "real_peak"):
            per_sub_curves = [evoked_roi_mean(per_sub[sub][cond], idxs) * 1e6 for sub in per_sub]
            mean_curve = np.mean(per_sub_curves, axis=0)
            sd_curve = np.std(per_sub_curves, axis=0, ddof=1) if len(per_sub_curves) > 1 else np.zeros_like(mean_curve)
            times_ms = per_sub[next(iter(per_sub))][cond].times * 1000.0
            ax.fill_between(times_ms, mean_curve - sd_curve, mean_curve + sd_curve,
                             color=COND_COLOR[cond], alpha=0.18, lw=0)
            ax.plot(times_ms, mean_curve, color=COND_COLOR[cond], lw=1.8,
                     ls=COND_LS[cond], label=COND_LABEL[cond])
            # per-subject thin lines
            for sub, curve in zip(per_sub.keys(), per_sub_curves):
                ax.plot(times_ms, curve, color=COND_COLOR[cond], lw=0.5, alpha=0.35,
                         ls=COND_LS[cond])
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from alignment point (ms)")
        if col == 0:
            ax.set_ylabel("uV  (mean ± SD across subjects)")
        ax.set_title(roi)
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# PSD
# -----------------------------------------------------------------------------
def compute_psd(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spectrum = epochs.compute_psd(method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX,
                                    n_fft=PSD_NFFT, verbose="ERROR")
    return spectrum.get_data(), spectrum.freqs, spectrum.ch_names


def plot_psd_subject_3cond(sub: str, psds: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
                            out_png: Path) -> pd.DataFrame:
    """4 ROIs in cols, 3 lines per panel with shaded SEM across epochs."""
    freqs = psds["real_onset"][1]
    ch_names = psds["real_onset"][2]
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(22, 5), sharex=True)
    fig.suptitle(
        f"{sub}  --  PSD, 3 conditions: real_onset / silent / real_peak   (mean ± SEM across epochs)",
        fontsize=10,
    )
    rows = []
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            ax.text(0.5, 0.5, f"{roi}: no chans", ha="center", va="center", transform=ax.transAxes)
            continue
        for cond, (psd_per_epoch, _, _) in psds.items():
            d_db = _roi_psd_db(psd_per_epoch, idxs)
            if d_db.size == 0 or d_db.shape[0] < 1:
                continue
            mean = d_db.mean(axis=0)
            sem = d_db.std(axis=0, ddof=1) / np.sqrt(d_db.shape[0]) if d_db.shape[0] > 1 else np.zeros_like(mean)
            ax.fill_between(freqs, mean - sem, mean + sem, color=COND_COLOR[cond], alpha=0.18, lw=0)
            ax.plot(freqs, mean, color=COND_COLOR[cond], lw=1.6,
                     ls=COND_LS[cond], label=COND_LABEL[cond])

            bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
            for bname, (lo, hi) in bands.items():
                m = (freqs >= lo) & (freqs < hi)
                if m.any():
                    rows.append(dict(
                        subject=sub, condition=cond, roi=roi, band=bname, fmin=lo, fmax=hi,
                        mean_db=float(mean[m].mean()),
                    ))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("Power (dB)")
        ax.set_title(roi)
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_psd_grand_average_3cond(per_sub_psds: dict, freqs: np.ndarray, ch_names: list[str],
                                   out_png: Path) -> None:
    """3 conditions averaged across subjects, shaded SD across subjects."""
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(22, 5), sharex=True)
    fig.suptitle(f"Grand-average PSD, 3 conditions  (N={len(per_sub_psds)} subjects, mean ± SD across subjects)",
                 fontsize=10)
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            continue
        for cond in ("real_onset", "silent", "real_peak"):
            curves = []
            for sub, psds in per_sub_psds.items():
                psd_per_epoch = psds[cond][0]
                d_db = _roi_psd_db(psd_per_epoch, idxs)
                curves.append(d_db.mean(axis=0))
            arr = np.array(curves)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.fill_between(freqs, mean - sd, mean + sd, color=COND_COLOR[cond], alpha=0.18, lw=0)
            ax.plot(freqs, mean, color=COND_COLOR[cond], lw=1.8,
                     ls=COND_LS[cond], label=COND_LABEL[cond])
            for i, sub in enumerate(per_sub_psds.keys()):
                ax.plot(freqs, arr[i], color=COND_COLOR[cond], lw=0.5, alpha=0.30, ls=COND_LS[cond])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("Power (dB)")
        ax.set_title(roi)
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# TFR
# -----------------------------------------------------------------------------
def compute_tfr(epochs: mne.Epochs) -> mne.time_frequency.AverageTFR:
    return tfr_morlet(epochs, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
                       use_fft=True, return_itc=False, decim=TFR_DECIM, n_jobs=1,
                       average=True, verbose="ERROR")


def baseline_window_for(cond: str) -> tuple[float, float]:
    return (-5.0, -4.0) if cond != "real_peak" else (-4.0, -3.0)


def plot_tfr_subject_3cond(sub: str, tfrs: dict[str, mne.time_frequency.AverageTFR], out_png: Path) -> None:
    """4 ROI rows × 3 condition cols. Each subplot is a logratio-baselined TFR heatmap."""
    ch_names = tfrs["real_onset"].ch_names
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(len(ROI_NAMES), 3, figsize=(18, 13))
    fig.suptitle(
        f"{sub}  --  TFR (Morlet 1-40 Hz, logratio baseline)  3 conditions",
        fontsize=10,
    )
    for row, roi in enumerate(ROI_NAMES):
        idxs = roi_idx[roi]
        for col, cond in enumerate(("real_onset", "silent", "real_peak")):
            ax = axes[row, col]
            tfr = tfrs[cond].copy().apply_baseline(baseline_window_for(cond), mode="logratio")
            mat = tfr_roi_mean(tfr, idxs)
            vmax = float(np.nanmax(np.abs(mat))) if np.any(np.isfinite(mat)) else 1.0
            im = ax.pcolormesh(tfr.times, tfr.freqs, mat, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_yscale("log")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{roi}: {cond}", fontsize=8)
            if col == 0:
                ax.set_ylabel("Hz")
            if row == len(ROI_NAMES) - 1:
                ax.set_xlabel("time from alignment (s)")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_tfr_grand_average_3cond(per_sub_tfrs: dict, out_png: Path) -> None:
    """4 ROI rows × 3 condition cols, averaged across subjects."""
    any_tfr = next(iter(per_sub_tfrs.values()))["real_onset"]
    ch_names = any_tfr.ch_names
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(len(ROI_NAMES), 3, figsize=(18, 13))
    fig.suptitle(f"Grand-average TFR  (N={len(per_sub_tfrs)} subjects, logratio baseline)",
                 fontsize=10)
    for row, roi in enumerate(ROI_NAMES):
        idxs = roi_idx[roi]
        for col, cond in enumerate(("real_onset", "silent", "real_peak")):
            ax = axes[row, col]
            mats = []
            for sub, tfrs in per_sub_tfrs.items():
                tfr = tfrs[cond].copy().apply_baseline(baseline_window_for(cond), mode="logratio")
                mats.append(tfr_roi_mean(tfr, idxs))
            ga = np.mean(mats, axis=0)
            tfr_ref = per_sub_tfrs[list(per_sub_tfrs.keys())[0]][cond].copy().apply_baseline(baseline_window_for(cond), mode="logratio")
            vmax = float(np.nanmax(np.abs(ga))) if np.any(np.isfinite(ga)) else 1.0
            im = ax.pcolormesh(tfr_ref.times, tfr_ref.freqs, ga, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_yscale("log")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{roi}: {cond}", fontsize=8)
            if col == 0:
                ax.set_ylabel("Hz")
            if row == len(ROI_NAMES) - 1:
                ax.set_xlabel("time from alignment (s)")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("analyze_scr_peak  ::  3-condition analysis (real_onset / silent / real_peak)")
    print(f"  output -> {FIG_DIR}")
    print("=" * 78)

    per_sub_evokeds: dict = {}
    per_sub_psds: dict = {}
    per_sub_tfrs: dict = {}
    erp_rows: list[pd.DataFrame] = []
    psd_rows: list[pd.DataFrame] = []
    common_freqs_psd = None
    common_ch = None

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        bundle = build_subject_3cond_epochs(sub)
        if bundle is None:
            print("  no epochs")
            continue
        ep_real_onset = bundle["real_onset"]
        ep_silent = bundle["silent"]
        ep_real_peak = bundle["real_peak"]
        print(f"  epochs: real_onset={len(ep_real_onset)}  silent={len(ep_silent)}  real_peak={len(ep_real_peak)}")

        # --- epoch audit ---
        out_audit = FIG_DIR / f"Y3_epoch_audit_3cond_{sub}.png"
        plot_epoch_audit_3cond(sub, bundle["audit"], out_audit)
        print(f"  audit -> {out_audit.name}")

        # --- ERP ---
        ev_real_onset = ep_real_onset.average()
        ev_silent = ep_silent.average()
        ev_real_peak = ep_real_peak.average()
        evokeds = {"real_onset": ev_real_onset, "silent": ev_silent, "real_peak": ev_real_peak}
        out_erp = FIG_DIR / f"Y3_erp_3cond_{sub}.png"
        df_erp = plot_erp_subject_3cond(sub, evokeds, out_erp)
        erp_rows.append(df_erp)
        print(f"  ERP -> {out_erp.name}")
        per_sub_evokeds[sub] = evokeds

        # --- PSD ---
        print("  computing PSDs ...")
        psd_real_onset = compute_psd(ep_real_onset)
        psd_silent = compute_psd(ep_silent)
        psd_real_peak = compute_psd(ep_real_peak)
        psds = {"real_onset": psd_real_onset, "silent": psd_silent, "real_peak": psd_real_peak}
        out_psd = FIG_DIR / f"Y3_psd_3cond_{sub}.png"
        df_psd = plot_psd_subject_3cond(sub, psds, out_psd)
        psd_rows.append(df_psd)
        print(f"  PSD -> {out_psd.name}")
        per_sub_psds[sub] = psds
        if common_freqs_psd is None:
            common_freqs_psd = psd_real_onset[1]
            common_ch = psd_real_onset[2]

        # --- TFR ---
        print("  computing TFRs ...")
        tfr_real_onset = compute_tfr(ep_real_onset)
        tfr_silent = compute_tfr(ep_silent)
        tfr_real_peak = compute_tfr(ep_real_peak)
        tfrs = {"real_onset": tfr_real_onset, "silent": tfr_silent, "real_peak": tfr_real_peak}
        out_tfr = FIG_DIR / f"Y3_tfr_3cond_{sub}.png"
        plot_tfr_subject_3cond(sub, tfrs, out_tfr)
        print(f"  TFR -> {out_tfr.name}")
        per_sub_tfrs[sub] = tfrs

    # --- grand averages ---
    if per_sub_evokeds:
        plot_erp_grand_average_3cond(per_sub_evokeds, FIG_DIR / "Y3_erp_3cond_grandaverage.png")
        print(f"\nGrand-avg ERP -> {(FIG_DIR / 'Y3_erp_3cond_grandaverage.png').name}")
    if per_sub_psds:
        plot_psd_grand_average_3cond(per_sub_psds, common_freqs_psd, common_ch,
                                       FIG_DIR / "Y3_psd_3cond_grandaverage.png")
        print(f"Grand-avg PSD -> {(FIG_DIR / 'Y3_psd_3cond_grandaverage.png').name}")
    if per_sub_tfrs:
        plot_tfr_grand_average_3cond(per_sub_tfrs, FIG_DIR / "Y3_tfr_3cond_grandaverage.png")
        print(f"Grand-avg TFR -> {(FIG_DIR / 'Y3_tfr_3cond_grandaverage.png').name}")

    # --- CSVs ---
    if erp_rows:
        out_csv = NPZ_DIR / "erp_3cond_roi_amplitudes.csv"
        pd.concat(erp_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"ERP CSV -> {out_csv.name}")
    if psd_rows:
        out_csv = NPZ_DIR / "psd_3cond_band_summary.csv"
        pd.concat(psd_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"PSD CSV -> {out_csv.name}")


if __name__ == "__main__":
    main()
