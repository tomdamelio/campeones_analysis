"""Focused test of the parieto-occipital alpha desynchronization hypothesis.

Classical claim: increased autonomic arousal causes alpha (8-13 Hz) desynchronization
(power DECREASES) over parieto-occipital cortex. Tests this specifically by:

  - Morlet TFR restricted to alpha band (8-13 Hz, 0.5 Hz spacing)
  - ROI restricted to strict parieto-occipital channels: P3, Pz, P4, P7, P8, O1, O2
  - Baseline correction: percent change vs (-5, -4.5) s
  - Compare REAL (SCR) vs SILENT (EDA-quiet matched control)
  - Window of interest: [0, +1] s post-onset (classical ERD window)
  - Direction: real < silent in % change -> supports the hypothesis (more ERD with SCR)

Outputs (figures/alpha_hypothesis/):
  Y3_alpha_timecourse_<sub>.png      -- alpha power(t) per sub, real vs silent with SEM
  Y3_alpha_summary.png                -- bar chart per sub: mean alpha [0,1] s + diff
  Y3_alpha_topomap_<sub>.png          -- topomap of (real - silent) alpha at +0.5 s

CSVs:
  alpha_hypothesis_summary.csv       -- subject, mean_real_pct, mean_silent_pct, diff_pct,
                                         supports_hypothesis (bool)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_hypothesis_scr
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
from scipy import stats

from src.campeones_analysis.multimodal_arousal.erp_scr import (
    NPZ_DIR,
    OUT,
    SUBJECTS,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "alpha_hypothesis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Hypothesis parameters ---
ALPHA_FREQS = np.arange(8.0, 13.5, 0.5)  # 11 frequencies covering 8-13 Hz
N_CYCLES = ALPHA_FREQS / 2.0  # standard, longer wavelet at lower freq
TFR_DECIM = 4
PARIETOOCCIPITAL = ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"]
BASELINE = (-5.0, -4.5)
BASELINE_MODE = "percent"  # (x - mean) / mean * 100, classical ERD/ERS metric
WIN_OF_INTEREST = (0.0, 1.0)  # post-onset; classical ERD window

SUBJ_COLORS = {"sub-23": "C0", "sub-24": "C1", "sub-33": "C2"}


def compute_alpha_tfr(epochs: mne.Epochs) -> mne.time_frequency.AverageTFR:
    """Per-epoch alpha-band Morlet TFR averaged across epochs."""
    return tfr_morlet(
        epochs, freqs=ALPHA_FREQS, n_cycles=N_CYCLES, use_fft=True,
        return_itc=False, decim=TFR_DECIM, n_jobs=1, average=True, verbose="ERROR",
    )


def alpha_roi_timecourse(tfr: mne.time_frequency.AverageTFR, roi: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Apply baseline, restrict to ROI channels, average across alpha freqs and channels.

    Returns (times, mean_pct_change_timecourse).
    """
    tfr_b = tfr.copy().apply_baseline(BASELINE, mode=BASELINE_MODE)
    ch_idx = [tfr_b.ch_names.index(ch) for ch in roi if ch in tfr_b.ch_names]
    if not ch_idx:
        return tfr_b.times, np.full(len(tfr_b.times), np.nan)
    data = tfr_b.data[ch_idx]  # (n_ch_roi, n_freqs, n_times)
    # average across channels and freqs
    mean_alpha = data.mean(axis=(0, 1))  # (n_times,)
    return tfr_b.times, mean_alpha


def alpha_roi_per_epoch(epochs: mne.Epochs, roi: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute % change alpha power per epoch (not averaged across epochs).

    Used to get SEM across epochs in the time-course plot.

    Returns (times, alpha_pct[n_epochs, n_times]).
    """
    tfr = tfr_morlet(
        epochs, freqs=ALPHA_FREQS, n_cycles=N_CYCLES, use_fft=True,
        return_itc=False, decim=TFR_DECIM, n_jobs=1, average=False, verbose="ERROR",
    )
    tfr_b = tfr.copy().apply_baseline(BASELINE, mode=BASELINE_MODE)
    ch_idx = [tfr_b.ch_names.index(ch) for ch in roi if ch in tfr_b.ch_names]
    if not ch_idx:
        return tfr_b.times, np.full((len(tfr_b), len(tfr_b.times)), np.nan)
    data = tfr_b.data[:, ch_idx]  # (n_epochs, n_ch_roi, n_freqs, n_times)
    mean_alpha_per_epoch = data.mean(axis=(1, 2))  # (n_epochs, n_times)
    return tfr_b.times, mean_alpha_per_epoch


def plot_alpha_timecourse_subject(
    sub: str,
    times: np.ndarray,
    real_per_epoch: np.ndarray,
    silent_per_epoch: np.ndarray,
    out_png: Path,
) -> dict:
    """Per-subject alpha power vs time with SEM across epochs."""
    real_mean = real_per_epoch.mean(axis=0)
    real_sem = real_per_epoch.std(axis=0, ddof=1) / np.sqrt(real_per_epoch.shape[0])
    silent_mean = silent_per_epoch.mean(axis=0)
    silent_sem = silent_per_epoch.std(axis=0, ddof=1) / np.sqrt(silent_per_epoch.shape[0])

    # window of interest mean per epoch
    woi_mask = (times >= WIN_OF_INTEREST[0]) & (times <= WIN_OF_INTEREST[1])
    real_woi = real_per_epoch[:, woi_mask].mean(axis=1)
    silent_woi = silent_per_epoch[:, woi_mask].mean(axis=1)

    # paired-like test (not actually paired since epochs differ but groups still comparable)
    t_stat, p_val = stats.ttest_ind(real_woi, silent_woi, equal_var=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(times, real_mean - real_sem, real_mean + real_sem, color="C3", alpha=0.25, lw=0)
    ax.fill_between(times, silent_mean - silent_sem, silent_mean + silent_sem, color="0.4", alpha=0.25, lw=0)
    ax.plot(times, real_mean, color="C3", lw=1.8, label=f"real SCR (N={real_per_epoch.shape[0]})")
    ax.plot(times, silent_mean, color="0.4", lw=1.6, ls="--", label=f"silent EDA (N={silent_per_epoch.shape[0]})")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(WIN_OF_INTEREST[0], WIN_OF_INTEREST[1], color="yellow", alpha=0.15, zorder=0,
                label=f"window of interest [{WIN_OF_INTEREST[0]}, {WIN_OF_INTEREST[1]}] s")
    ax.set_xlabel("time from SCR onset (s)")
    ax.set_ylabel("alpha (8-13 Hz) power, % change vs baseline (-5,-4.5)s")
    ax.set_title(
        f"{sub}  --  parieto-occipital alpha modulation  "
        f"(ROI={','.join(PARIETOOCCIPITAL)})\n"
        f"WOI [0,1]s: real={real_woi.mean():+.2f}%  silent={silent_woi.mean():+.2f}%  "
        f"diff={real_woi.mean() - silent_woi.mean():+.2f}%  "
        f"t({real_woi.shape[0]+silent_woi.shape[0]-2})={t_stat:.2f}  p={p_val:.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    return dict(
        subject=sub,
        n_real=int(real_per_epoch.shape[0]),
        n_silent=int(silent_per_epoch.shape[0]),
        mean_real_pct=float(real_woi.mean()),
        sd_real_pct=float(real_woi.std(ddof=1)),
        mean_silent_pct=float(silent_woi.mean()),
        sd_silent_pct=float(silent_woi.std(ddof=1)),
        diff_pct=float(real_woi.mean() - silent_woi.mean()),
        t_stat=float(t_stat),
        p_value=float(p_val),
        # supports hypothesis if real shows MORE ERD (more negative) than silent
        supports_classical_hypothesis=bool(real_woi.mean() < silent_woi.mean()),
    )


def plot_alpha_topomap_subject(
    sub: str,
    tfr_real: mne.time_frequency.AverageTFR,
    tfr_silent: mne.time_frequency.AverageTFR,
    out_png: Path,
) -> None:
    """Topomap of (real - silent) alpha % change at +0.5 s post-onset."""
    tfr_real_b = tfr_real.copy().apply_baseline(BASELINE, mode=BASELINE_MODE)
    tfr_silent_b = tfr_silent.copy().apply_baseline(BASELINE, mode=BASELINE_MODE)
    diff = tfr_real_b.copy()
    diff.data = tfr_real_b.data - tfr_silent_b.data

    t_target = 0.5
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, tfr_obj, title in zip(axes, [tfr_real_b, tfr_silent_b, diff],
                                    ["real SCR (% change)", "silent EDA (% change)", "diff (real - silent)"]):
        try:
            # average over alpha freqs first; plot_topomap with fmin/fmax
            tfr_obj.plot_topomap(
                tmin=t_target - 0.1, tmax=t_target + 0.1,
                fmin=ALPHA_FREQS[0], fmax=ALPHA_FREQS[-1],
                axes=ax, colorbar=True, show=False, contours=4,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"topomap fail\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{sub}  --  alpha (8-13 Hz) topomap at t = {t_target:.1f} s post-onset", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_summary(rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: per-subject bar chart of WOI means
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["mean_real_pct"], width, yerr=df["sd_real_pct"] / np.sqrt(df["n_real"]),
            color="C3", alpha=0.85, label="real SCR", capsize=4)
    ax.bar(x + width / 2, df["mean_silent_pct"], width, yerr=df["sd_silent_pct"] / np.sqrt(df["n_silent"]),
            color="0.4", alpha=0.85, label="silent EDA", capsize=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"])
    ax.set_ylabel("alpha power (% change vs baseline)")
    ax.set_title(f"alpha modulation in WOI [{WIN_OF_INTEREST[0]}, {WIN_OF_INTEREST[1]}] s post-onset")
    ax.legend(fontsize=9)

    # right: bar chart of difference (real - silent)
    ax = axes[1]
    colors = ["C2" if d < 0 else "C3" for d in df["diff_pct"]]  # green = ERD direction (supports), red = ERS
    ax.bar(x, df["diff_pct"], color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(0, color="k", lw=0.6)
    for xi, d, p in zip(x, df["diff_pct"], df["p_value"]):
        marker = "*" if p < 0.05 else ""
        ax.text(xi, d + (0.5 if d > 0 else -1.0), f"{d:+.2f}%\np={p:.3f}{marker}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"])
    ax.set_ylabel("real − silent (% change)")
    ax.set_title("Difference (negative → supports ERD hypothesis)")

    fig.suptitle(
        "Parieto-occipital alpha hypothesis test\n"
        "Classical claim: real (SCR) should show MORE desynchronization (ERD, more negative % change) than silent",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print(f"alpha_hypothesis_scr  ::  output -> {FIG_DIR}")
    print("=" * 78)
    print(f"  ROI: {PARIETOOCCIPITAL}")
    print(f"  Alpha band: {ALPHA_FREQS[0]:.1f} - {ALPHA_FREQS[-1]:.1f} Hz "
          f"({len(ALPHA_FREQS)} freqs at {ALPHA_FREQS[1]-ALPHA_FREQS[0]:.1f} Hz spacing)")
    print(f"  Baseline: {BASELINE} s ({BASELINE_MODE} change)")
    print(f"  WOI: {WIN_OF_INTEREST} s post-onset")
    print()

    rows = []
    for sub in SUBJECTS:
        print(f"=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print("  no epochs")
            continue
        print(f"  epochs real={len(real_ep)}  silent={len(silent_ep)}")

        # per-epoch alpha % change time-courses for SEM
        print("  computing per-epoch alpha TFR (real) ...")
        times, real_per_epoch = alpha_roi_per_epoch(real_ep, PARIETOOCCIPITAL)
        print("  computing per-epoch alpha TFR (silent) ...")
        _, silent_per_epoch = alpha_roi_per_epoch(silent_ep, PARIETOOCCIPITAL)

        out_tc = FIG_DIR / f"Y3_alpha_timecourse_{sub}.png"
        row = plot_alpha_timecourse_subject(sub, times, real_per_epoch, silent_per_epoch, out_tc)
        rows.append(row)
        print(f"  WOI alpha: real={row['mean_real_pct']:+.2f}%  silent={row['mean_silent_pct']:+.2f}%  "
              f"diff={row['diff_pct']:+.2f}%  p={row['p_value']:.3f}  "
              f"hypothesis_supported={row['supports_classical_hypothesis']}")
        print(f"  -> {out_tc.name}")

        # topomap (uses averaged TFR for plotting convenience)
        print("  computing topomap TFRs ...")
        tfr_real = compute_alpha_tfr(real_ep)
        tfr_silent = compute_alpha_tfr(silent_ep)
        out_topo = FIG_DIR / f"Y3_alpha_topomap_{sub}.png"
        plot_alpha_topomap_subject(sub, tfr_real, tfr_silent, out_topo)
        print(f"  -> {out_topo.name}")

    if rows:
        out_csv = NPZ_DIR / "alpha_hypothesis_summary.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSummary CSV -> {out_csv.name}  ({len(rows)} subjects)")
        plot_summary(rows, FIG_DIR / "Y3_alpha_summary.png")
        print(f"Summary plot -> Y3_alpha_summary.png")

        # final verdict
        n_support = sum(r["supports_classical_hypothesis"] for r in rows)
        print(f"\nVerdict: {n_support}/{len(rows)} subjects show alpha desynchronization (real < silent in WOI)")
        n_sig = sum(r["p_value"] < 0.05 for r in rows)
        print(f"        {n_sig}/{len(rows)} subjects have p < 0.05 for the real-vs-silent contrast")


if __name__ == "__main__":
    main()
