"""Time-frequency (Morlet) + PSD (Welch) analysis comparing SCR vs silent-EDA epochs.

Closes the third comparison domain (after the time-domain ERP in `erp_scr.py` +
`erp_scr_grandaverage.py`): frequency-domain modulations of EEG locked to SCR onsets
versus matched silent-EDA control windows.

Reuses epoching logic from `erp_scr.py` via imports (cleanliness rules, control sampling,
ROIs, channel selection, etc.) so the comparison is fully consistent with the ERP analysis.

Outputs (under research_diary/context/05_02/figures/):
  Per subject:
    Y3_tfr_scr_<sub>.png   -- TFR (real / control / diff with logratio baseline) per ROI
    Y3_psd_scr_<sub>.png   -- PSD (real vs control + diff) per ROI

  Grand average (across the 3-subject cohort):
    Y3_tfr_scr_grandaverage.png
    Y3_psd_scr_grandaverage.png

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.tfr_psd_scr
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

# Reuse epoching + control infrastructure from erp_scr.py
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    BASELINE,
    EDA_FS,
    NPZ_DIR,
    OUT,
    PREP,
    RNG,
    SUBJECTS,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ROIs (same as erp_scr_grandaverage.py)
ROIS: dict[str, list[str]] = {
    "Frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC1", "FCz", "FC2"],
    "Temporal":  ["T7", "T8", "FT9", "FT10", "TP9", "TP10", "FC5", "FC6", "CP5", "CP6"],
    "Parietal":  ["C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4"],
    "Occipital": ["O1", "O2", "P7", "P8"],
}
ROI_NAMES = list(ROIS.keys())

# TFR parameters
TFR_FREQS = np.logspace(np.log10(1.0), np.log10(40.0), 30)
TFR_N_CYCLES = TFR_FREQS / 2.0  # standard: half a cycle per freq, longer at low freqs
TFR_BASELINE = (-5.0, -4.0)  # 1 s baseline at far pre-window (consistent with ERP convention)
TFR_BASELINE_MODE = "logratio"  # log10(power / baseline_mean)
TFR_DECIM = 4  # decimate EEG before TFR; 250 Hz / 4 = 62.5 Hz still oversamples 40 Hz

# PSD parameters (computed on whole epoch via Welch)
PSD_FMIN = 1.0
PSD_FMAX = 40.0
PSD_NFFT = 512  # 512 samples / 250 Hz = ~2 s windows


# -----------------------------------------------------------------------------
# Epoch construction (mirrors erp_scr.py main() per-subject loop)
# -----------------------------------------------------------------------------
def build_subject_epochs(sub: str) -> tuple[mne.Epochs | None, mne.Epochs | None]:
    """Return (real_epochs_concat, silent_epochs_concat) for one subject, or (None, None)."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None, None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])

    real_epochs_list: list[mne.Epochs] = []
    silent_epochs_list: list[mne.Epochs] = []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")  # bandpass up to TFR max
            raw.resample(250.0, verbose="ERROR")
            duration = float(raw.times[-1])

            eda_phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets_all = detect_scr_onsets_s(eda_phasic, EDA_FS)
            onsets_all = onsets_all[onsets_all < duration]
            onsets_clean = filter_clean_onsets(onsets_all, eda_phasic, EDA_FS)
            silent_t = sample_silent_controls(
                n_target=len(onsets_clean), duration_s=duration,
                phasic=eda_phasic, fs=EDA_FS, rng=RNG, avoid_onsets_s=onsets_clean,
            )
            ep_real = epoch_one_run(raw, onsets_clean, code=1)
            ep_silent = epoch_one_run(raw, silent_t, code=2)
            if ep_real is not None: real_epochs_list.append(ep_real)
            if ep_silent is not None: silent_epochs_list.append(ep_silent)
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    if not real_epochs_list or not silent_epochs_list:
        return None, None
    real_all = mne.concatenate_epochs(real_epochs_list, verbose="ERROR")
    silent_all = mne.concatenate_epochs(silent_epochs_list, verbose="ERROR")
    return real_all, silent_all


# -----------------------------------------------------------------------------
# ROI averaging
# -----------------------------------------------------------------------------
def roi_channel_indices(ch_names: list[str]) -> dict[str, list[int]]:
    return {roi: [ch_names.index(c) for c in chs if c in ch_names] for roi, chs in ROIS.items()}


def tfr_roi_mean(tfr: mne.time_frequency.AverageTFR, idxs: list[int]) -> np.ndarray:
    """Return TFR data averaged across channels in `idxs`. Shape: (n_freqs, n_times)."""
    if not idxs:
        return np.full((tfr.data.shape[1], tfr.data.shape[2]), np.nan)
    return tfr.data[idxs].mean(axis=0)


# -----------------------------------------------------------------------------
# TFR
# -----------------------------------------------------------------------------
def compute_tfr(epochs: mne.Epochs) -> mne.time_frequency.AverageTFR:
    """Per-epoch Morlet TFR then average across epochs. Returns AverageTFR."""
    return tfr_morlet(
        epochs, freqs=TFR_FREQS, n_cycles=TFR_N_CYCLES,
        use_fft=True, return_itc=False, decim=TFR_DECIM, n_jobs=1, average=True, verbose="ERROR",
    )


def plot_tfr_per_subject(sub: str, tfr_real: mne.time_frequency.AverageTFR,
                          tfr_silent: mne.time_frequency.AverageTFR, out_png: Path) -> None:
    # apply baseline (logratio) to copies for plotting
    tfr_real_b = tfr_real.copy().apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)
    tfr_silent_b = tfr_silent.copy().apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)

    ch_names = tfr_real.ch_names
    roi_idx = roi_channel_indices(ch_names)
    times = tfr_real_b.times
    freqs = tfr_real_b.freqs

    # diff in raw (non-baselined) power, then express as logratio of silent
    # equivalent: subtract baselined logratios
    fig, axes = plt.subplots(len(ROI_NAMES), 3, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f"{sub} -- TFR (Morlet, {TFR_FREQS[0]:.0f}-{TFR_FREQS[-1]:.0f} Hz, "
                 f"baseline={TFR_BASELINE}s {TFR_BASELINE_MODE})  n_real={len(tfr_real.info['ch_names'])}", fontsize=10)

    for row, roi in enumerate(ROI_NAMES):
        idxs = roi_idx[roi]
        d_real = tfr_roi_mean(tfr_real_b, idxs)
        d_silent = tfr_roi_mean(tfr_silent_b, idxs)
        d_diff = d_real - d_silent

        for col, (mat, title, cmap) in enumerate(zip(
            [d_real, d_silent, d_diff],
            [f"{roi}: real (SCR)", f"{roi}: silent EDA", f"{roi}: real - silent"],
            ["RdBu_r", "RdBu_r", "RdBu_r"],
        )):
            ax = axes[row, col]
            vmax = float(np.nanmax(np.abs(mat))) if np.any(np.isfinite(mat)) else 1.0
            im = ax.pcolormesh(times, freqs, mat, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_yscale("log")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel("Freq (Hz)")
            if row == len(ROI_NAMES) - 1:
                ax.set_xlabel("time from SCR onset (s)")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_tfr_grand_average(per_sub_tfrs: dict, out_png: Path) -> None:
    """per_sub_tfrs: {sub: (tfr_real, tfr_silent)} after subject loop."""
    # average baselined diff across subjects per ROI
    fig, axes = plt.subplots(len(ROI_NAMES), 3, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f"Grand-average TFR  (N={len(per_sub_tfrs)} subjects, baseline={TFR_BASELINE}s {TFR_BASELINE_MODE})",
                 fontsize=10)

    any_tfr = next(iter(per_sub_tfrs.values()))[0]
    times = any_tfr.copy().apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE).times
    freqs = any_tfr.freqs
    ch_names = any_tfr.ch_names
    roi_idx = roi_channel_indices(ch_names)

    for row, roi in enumerate(ROI_NAMES):
        idxs = roi_idx[roi]
        real_stack, silent_stack, diff_stack = [], [], []
        for sub, (tr, ts) in per_sub_tfrs.items():
            tr_b = tr.copy().apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)
            ts_b = ts.copy().apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)
            real_stack.append(tfr_roi_mean(tr_b, idxs))
            silent_stack.append(tfr_roi_mean(ts_b, idxs))
            diff_stack.append(real_stack[-1] - silent_stack[-1])
        ga_real = np.mean(np.array(real_stack), axis=0)
        ga_silent = np.mean(np.array(silent_stack), axis=0)
        ga_diff = np.mean(np.array(diff_stack), axis=0)

        for col, (mat, title) in enumerate(zip(
            [ga_real, ga_silent, ga_diff],
            [f"{roi}: real (SCR)", f"{roi}: silent EDA", f"{roi}: real - silent"],
        )):
            ax = axes[row, col]
            vmax = float(np.nanmax(np.abs(mat))) if np.any(np.isfinite(mat)) else 1.0
            im = ax.pcolormesh(times, freqs, mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_yscale("log")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel("Freq (Hz)")
            if row == len(ROI_NAMES) - 1:
                ax.set_xlabel("time from SCR onset (s)")
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# PSD
# -----------------------------------------------------------------------------
def compute_psd(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute Welch PSD, preserving per-epoch data.

    Returns (psd_per_epoch, freqs, ch_names). Shape: (n_epochs, n_channels, n_freqs).
    """
    spectrum = epochs.compute_psd(method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX,
                                   n_fft=PSD_NFFT, verbose="ERROR")
    return spectrum.get_data(), spectrum.freqs, spectrum.ch_names


def _roi_psd_db(psd_per_epoch: np.ndarray, roi_idxs: list[int]) -> np.ndarray:
    """Average raw power across ROI channels per epoch, then convert to dB.

    Returns shape (n_epochs, n_freqs) in dB.
    """
    if not roi_idxs:
        return np.full((psd_per_epoch.shape[0], psd_per_epoch.shape[2]), np.nan)
    roi_pow = psd_per_epoch[:, roi_idxs, :].mean(axis=1)  # (n_epochs, n_freqs)
    return 10.0 * np.log10(roi_pow + 1e-30)


def plot_psd_per_subject(sub: str, psd_real: np.ndarray, psd_silent: np.ndarray,
                          freqs: np.ndarray, ch_names: list[str], n_real: int, n_silent: int,
                          out_png: Path) -> pd.DataFrame:
    """Per-subject PSD: solid line = mean across epochs, shaded = SEM across epochs.

    psd_real / psd_silent have shape (n_epochs, n_channels, n_freqs).
    """
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(20, 5), sharex=True)
    fig.suptitle(f"{sub} -- PSD (Welch, {PSD_FMIN}-{PSD_FMAX} Hz, n_fft={PSD_NFFT})  "
                 f"n_real={n_real}, n_silent={n_silent}  --  mean +/- SEM across epochs", fontsize=10)
    rows = []
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            ax.text(0.5, 0.5, f"{roi}: no chans", ha="center", va="center", transform=ax.transAxes)
            continue
        r_db = _roi_psd_db(psd_real, idxs)      # (n_epochs_real, n_freqs)
        s_db = _roi_psd_db(psd_silent, idxs)    # (n_epochs_silent, n_freqs)

        r_mean = r_db.mean(axis=0)
        r_sem = r_db.std(axis=0, ddof=1) / np.sqrt(r_db.shape[0]) if r_db.shape[0] > 1 else np.zeros_like(r_mean)
        s_mean = s_db.mean(axis=0)
        s_sem = s_db.std(axis=0, ddof=1) / np.sqrt(s_db.shape[0]) if s_db.shape[0] > 1 else np.zeros_like(s_mean)

        ax.fill_between(freqs, r_mean - r_sem, r_mean + r_sem, color="C3", alpha=0.22, lw=0)
        ax.fill_between(freqs, s_mean - s_sem, s_mean + s_sem, color="0.4", alpha=0.22, lw=0)
        ax.plot(freqs, r_mean, color="C3", lw=1.6, label="real (SCR)")
        ax.plot(freqs, s_mean, color="0.4", lw=1.4, ls="--", label="silent EDA")
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("Power (dB)")
        ax.set_xscale("log")
        ax.set_title(roi)
        ax.legend(fontsize=8)

        # per-band mean of (real - silent) in dB
        bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
        for bname, (lo, hi) in bands.items():
            m = (freqs >= lo) & (freqs < hi)
            if m.any():
                rows.append(dict(
                    subject=sub, roi=roi, band=bname, fmin=lo, fmax=hi,
                    mean_db_real=float(r_mean[m].mean()),
                    mean_db_silent=float(s_mean[m].mean()),
                    diff_db=float((r_mean[m] - s_mean[m]).mean()),
                ))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_psd_grand_average(per_sub_psds: dict, freqs: np.ndarray, ch_names: list[str],
                            out_png: Path) -> None:
    """Grand-average PSD: solid line = mean across subjects, shaded = SD across subjects.

    per_sub_psds: {sub: (psd_real_per_epoch, psd_silent_per_epoch)} with shapes (n_epochs, n_ch, n_freqs).
    """
    roi_idx = roi_channel_indices(ch_names)
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(20, 5), sharex=True)
    fig.suptitle(
        f"Grand-average PSD  (N={len(per_sub_psds)} subjects, {PSD_FMIN}-{PSD_FMAX} Hz)  --  "
        f"mean +/- SD across subjects",
        fontsize=10,
    )
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        idxs = roi_idx[roi]
        if not idxs:
            continue
        # per subject: mean dB across epochs in this ROI -> one curve per subject
        real_per_sub = []
        silent_per_sub = []
        for sub, (pr, ps) in per_sub_psds.items():
            real_per_sub.append(_roi_psd_db(pr, idxs).mean(axis=0))
            silent_per_sub.append(_roi_psd_db(ps, idxs).mean(axis=0))
        real_arr = np.array(real_per_sub)     # (n_sub, n_freqs)
        silent_arr = np.array(silent_per_sub)
        # mean and SD across subjects
        r_mean = real_arr.mean(axis=0)
        r_sd = real_arr.std(axis=0, ddof=1) if real_arr.shape[0] > 1 else np.zeros_like(r_mean)
        s_mean = silent_arr.mean(axis=0)
        s_sd = silent_arr.std(axis=0, ddof=1) if silent_arr.shape[0] > 1 else np.zeros_like(s_mean)
        ax.fill_between(freqs, r_mean - r_sd, r_mean + r_sd, color="C3", alpha=0.22, lw=0)
        ax.fill_between(freqs, s_mean - s_sd, s_mean + s_sd, color="0.4", alpha=0.22, lw=0)
        ax.plot(freqs, r_mean, color="C3", lw=1.8, label="grand-avg real (SCR)")
        ax.plot(freqs, s_mean, color="0.4", lw=1.5, ls="--", label="grand-avg silent EDA")
        # per-subject thin lines, dashed for silent
        for i, sub in enumerate(per_sub_psds.keys()):
            ax.plot(freqs, real_arr[i], color=f"C{i}", lw=0.6, alpha=0.55)
            ax.plot(freqs, silent_arr[i], color=f"C{i}", lw=0.6, alpha=0.40, ls=":")
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("Power (dB)")
        ax.set_xscale("log")
        ax.set_title(roi)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("tfr_psd_scr  ::  TFR (Morlet) + PSD (Welch)  real (SCR) vs silent-EDA control")
    print("=" * 78)

    per_sub_tfrs: dict = {}
    per_sub_psds: dict = {}
    psd_rows = []
    common_freqs_psd = None
    common_ch = None

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print(f"  no epochs")
            continue
        print(f"  epochs real={len(real_ep)} silent={len(silent_ep)}")

        # TFR (heavy)
        print("  computing TFR (real) ...")
        tfr_real = compute_tfr(real_ep)
        print("  computing TFR (silent) ...")
        tfr_silent = compute_tfr(silent_ep)
        out_png_tfr = FIG_DIR / f"Y3_tfr_scr_{sub}.png"
        plot_tfr_per_subject(sub, tfr_real, tfr_silent, out_png_tfr)
        print(f"  -> {out_png_tfr.name}")
        per_sub_tfrs[sub] = (tfr_real, tfr_silent)

        # PSD
        print("  computing PSD ...")
        psd_real, freqs_psd, ch_names = compute_psd(real_ep)
        psd_silent, _, _ = compute_psd(silent_ep)
        out_png_psd = FIG_DIR / f"Y3_psd_scr_{sub}.png"
        df_psd_sub = plot_psd_per_subject(sub, psd_real, psd_silent, freqs_psd, ch_names,
                                          n_real=len(real_ep), n_silent=len(silent_ep),
                                          out_png=out_png_psd)
        psd_rows.append(df_psd_sub)
        print(f"  -> {out_png_psd.name}")
        per_sub_psds[sub] = (psd_real, psd_silent)
        if common_freqs_psd is None:
            common_freqs_psd = freqs_psd
            common_ch = ch_names

    # ---- grand averages ----
    if len(per_sub_tfrs) > 0:
        ga_tfr_png = FIG_DIR / "Y3_tfr_scr_grandaverage.png"
        plot_tfr_grand_average(per_sub_tfrs, ga_tfr_png)
        print(f"\nGrand-avg TFR -> {ga_tfr_png.name}")

    if len(per_sub_psds) > 0:
        ga_psd_png = FIG_DIR / "Y3_psd_scr_grandaverage.png"
        plot_psd_grand_average(per_sub_psds, common_freqs_psd, common_ch, ga_psd_png)
        print(f"Grand-avg PSD -> {ga_psd_png.name}")

    if psd_rows:
        all_psd = pd.concat(psd_rows, ignore_index=True)
        csv_path = NPZ_DIR / "psd_scr_band_summary.csv"
        all_psd.to_csv(csv_path, index=False)
        print(f"PSD band summary -> {csv_path.name}  ({len(all_psd)} rows)")


if __name__ == "__main__":
    main()
