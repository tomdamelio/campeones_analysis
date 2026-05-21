"""Topomap of delta+theta PSD difference (real SCR - NO_SCR matched) per channel.

Closes a question left open by `tfr_psd_scr.py` which aggregates to 4 ROIs:
when we say "Temporal/Parietal delta+theta increase in real SCR", is the
underlying topography the same across subjects, or are ROI averages hiding
distinct spatial patterns?

Reuses epoch construction from tfr_psd_scr.build_subject_epochs() so the
analysis is fully consistent with the ERP/PSD/TFR pipelines.

Pipeline per subject:
  1. Build real + NO_SCR matched epochs via build_subject_epochs() (same
     cleanliness rules + matched controls as the rest of the multimodal_arousal
     scripts).
  2. Crop to [0, +3] s post-onset window.
  3. Welch PSD per epoch x channel (1-40 Hz, n_fft=512).
  4. For each band (delta 1-4 Hz, theta 4-8 Hz):
       channel_value_real = mean over epochs and frequencies of psd_real
       channel_value_NO_SCR = mean over epochs and frequencies of psd_NO_SCR
       diff_db = 10*log10(real) - 10*log10(NO_SCR)
  5. Plot topomap (one panel per band).

Grand average: per-channel mean of the diff_db across subjects.

Outputs (under research_diary/context/05_02/):
  figures/topomap_delta_theta/Y3_topomap_delta_theta_<sub>.png   (per subject, delta + theta)
  figures/topomap_delta_theta/Y3_topomap_delta_theta_grandaverage.png
  figures/topomap_delta_theta/Y3_topomap_delta_theta_overview.png   (all subs + GA in one PNG)
  y_candidates/topomap_delta_theta_summary.csv

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.topomap_delta_theta_scr
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

from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    build_subject_epochs,
    compute_psd,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "topomap_delta_theta"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT / "y_candidates" / "topomap_delta_theta_summary.csv"

BANDS: dict[str, tuple[float, float]] = {"delta": (1.0, 4.0), "theta": (4.0, 8.0)}
POST_TMIN = 0.0
POST_TMAX = 3.0


def channel_band_diff_db(
    psd_real: np.ndarray, psd_NO_SCR: np.ndarray, freqs: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    """Returns (n_channels,) array of per-channel diff in dB for the given band."""
    lo, hi = band
    fmask = (freqs >= lo) & (freqs < hi)
    real_band = psd_real[:, :, fmask].mean(axis=(0, 2))
    nosc_band = psd_NO_SCR[:, :, fmask].mean(axis=(0, 2))
    return 10.0 * np.log10(real_band + 1e-30) - 10.0 * np.log10(nosc_band + 1e-30)


def plot_subject_topomap(
    sub: str, info: mne.Info, diffs: dict[str, np.ndarray], out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4 * len(BANDS), 4.5))
    if len(BANDS) == 1:
        axes = [axes]
    fig.suptitle(f"{sub}  --  PSD diff real - NO_SCR (dB) per channel", fontsize=11)
    for ax, (band, data) in zip(axes, diffs.items()):
        vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False, vlim=(-vmax, vmax),
            cmap="RdBu_r", sensors=True, contours=4,
        )
        ax.set_title(f"{band} ({BANDS[band][0]:.0f}-{BANDS[band][1]:.0f} Hz)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_overview(
    info: mne.Info, per_sub: dict[str, dict[str, np.ndarray]],
    ga: dict[str, np.ndarray], out_png: Path,
) -> None:
    n_rows = len(per_sub) + 1  # subs + GA
    n_cols = len(BANDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle("PSD diff real - NO_SCR (dB) per channel  --  per subject + grand average",
                 fontsize=11)

    # Per-subject rows
    row_items = list(per_sub.items()) + [("Grand average (N=3)", ga)]
    for row, (label, band_dict) in enumerate(row_items):
        for col, (band, data) in enumerate(band_dict.items()):
            ax = axes[row, col]
            vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
            im, _ = mne.viz.plot_topomap(
                data, info, axes=ax, show=False, vlim=(-vmax, vmax),
                cmap="RdBu_r", sensors=True, contours=4,
            )
            title = f"{label} -- {band} ({BANDS[band][0]:.0f}-{BANDS[band][1]:.0f} Hz)" \
                    if row == 0 else f"{label} -- {band}"
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    rows: list[dict] = []
    per_sub_diffs: dict[str, dict[str, np.ndarray]] = {}
    info_ref: mne.Info | None = None

    for sub in SUBJECTS:
        print(f"[{sub}] building epochs...")
        real, nosc = build_subject_epochs(sub)
        if real is None or nosc is None:
            print(f"  {sub}: skipped (no epochs)")
            continue

        real_post = real.copy().crop(tmin=POST_TMIN, tmax=POST_TMAX)
        nosc_post = nosc.copy().crop(tmin=POST_TMIN, tmax=POST_TMAX)
        print(f"  n_real={len(real_post)}  n_NO_SCR={len(nosc_post)}")

        psd_real, freqs, ch_names = compute_psd(real_post)
        psd_nosc, _, _ = compute_psd(nosc_post)
        info = real_post.info

        diffs: dict[str, np.ndarray] = {}
        for band, rng in BANDS.items():
            d = channel_band_diff_db(psd_real, psd_nosc, freqs, rng)
            diffs[band] = d
            for i, ch in enumerate(ch_names):
                rows.append(dict(
                    subject=sub, channel=ch, band=band,
                    fmin=rng[0], fmax=rng[1], diff_db=float(d[i]),
                ))
        per_sub_diffs[sub] = diffs
        if info_ref is None:
            info_ref = info

        out_png = FIG_DIR / f"Y3_topomap_delta_theta_{sub}.png"
        plot_subject_topomap(sub, info, diffs, out_png)
        print(f"  saved {out_png}")

    if not per_sub_diffs:
        print("No subjects processed.")
        return

    # Grand average per band: mean of per-channel diff across subjects
    ga_diffs: dict[str, np.ndarray] = {}
    for band in BANDS:
        stack = np.array([d[band] for d in per_sub_diffs.values()])
        ga_diffs[band] = stack.mean(axis=0)

    ga_png = FIG_DIR / "Y3_topomap_delta_theta_grandaverage.png"
    plot_subject_topomap("Grand average (N=3)", info_ref, ga_diffs, ga_png)
    print(f"saved {ga_png}")

    # Overview (all subjects + GA in one PNG)
    overview_png = FIG_DIR / "Y3_topomap_delta_theta_overview.png"
    plot_overview(info_ref, per_sub_diffs, ga_diffs, overview_png)
    print(f"saved {overview_png}")

    # CSV with per-channel diff
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"saved {CSV_OUT}")


if __name__ == "__main__":
    main()
