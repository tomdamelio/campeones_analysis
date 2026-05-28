"""FOOOF (specparam) parametrization of SCR vs silent-EDA PSD: aperiodic vs periodic.

Tarea 2 QA artefacto-vs-señal, Deliverable 1. The cohort-6 spectral effect is
"broadband-excepto-alfa" and decodable (LORO AUC ~0.73) but absent in the time-domain
average. This script asks WHERE that spectral difference lives:

  - APERIODIC component (1/f offset + exponent) -> broadband shift, compatible with a
    global arousal / artifact change, OR
  - PERIODIC peaks (genuine oscillations) -> and in particular, is the "respeto" of alpha
    a preserved periodic alpha peak while the aperiodic floor shifts?

Reuses the exact epoching + PSD infrastructure of the rest of multimodal_arousal so the
parametrization is computed on the same Welch spectra that produced the effect:
  - build_subject_epochs() / compute_psd() / ROIS  from tfr_psd_scr.py
  - SUBJECTS / OUT                                  from erp_scr.py (-> cohort.py)

FOOOF is fit per CHANNEL (FOOOFGroup, 32 ch) and per ROI, per condition, in BOTH
aperiodic modes ("fixed" and "knee"); the primary contrasts use "fixed". Per-channel
aperiodic params feed Δoffset / Δexponente topomaps (first input to the Branković
dissociation test in topo_variance_scr.py).

Outputs (under research_diary/context/05_04/cohort6/qa_artifact_vs_signal/):
  tables/fooof_scr_aperiodic.csv   subject,channel|roi,level,condition,mode,offset,knee,exponent,r_squared,error
  tables/fooof_scr_periodic.csv    subject,channel|roi,level,condition,band,cf,pw,bw
  tables/fooof_scr_contrasts.csv   subject,level,key,d_offset,d_exponent,d_periodic_<band>
  figures/fooof_scr_<sub>.png            per-ROI FOOOF fit, real vs silent
  figures/fooof_scr_flattened_<sub>.png  per-ROI aperiodic-removed spectrum, real vs silent
  figures/fooof_scr_grandaverage.png     per-ROI GA spectra + fits
  figures/fooof_scr_topo_aperiodic.png   GA topomaps of Δoffset and Δexponent (real-silent)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.fooof_scr
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.fooof_scr --subjects sub-27
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from fooof import FOOOF, FOOOFGroup
from fooof.sim.gen import gen_aperiodic

from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    ROI_NAMES,
    build_subject_epochs,
    compute_psd,
    roi_channel_indices,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

QA_DIR = OUT / "qa_artifact_vs_signal"
FIG_DIR = QA_DIR / "figures"
TBL_DIR = QA_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
}
BAND_NAMES = list(BANDS.keys())
# Fit from 1.5 Hz: the ~1 Hz Welch bin is unstable, and periodic peaks in pure delta are
# unreliable -> the delta question is answered via Δexponent/Δoffset, not via a delta peak.
FIT_RANGE = (1.5, 40.0)
FOOOF_KW = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.10)
PRIMARY_MODE = "fixed"   # contrasts + topomaps use the fixed-knee model
MODES = ("fixed", "knee")
COND_COLORS = {"real": "C3", "silent": "0.4"}


# -----------------------------------------------------------------------------
# FOOOF helpers
# -----------------------------------------------------------------------------
def _periodic_band_power(peaks: list[tuple[float, float, float]]) -> dict[str, float]:
    """Sum periodic peak power (PW) into bands by center frequency."""
    out = {b: 0.0 for b in BANDS}
    for cf, pw, _bw in peaks:
        for b, (lo, hi) in BANDS.items():
            if lo <= cf < hi:
                out[b] += float(pw)
    return out


def fit_fooof(spectrum: np.ndarray, freqs: np.ndarray, mode: str) -> FOOOF:
    """Fit a single FOOOF model on one linear-power spectrum."""
    fm = FOOOF(aperiodic_mode=mode, verbose=False, **FOOOF_KW)
    fm.fit(np.asarray(freqs, float), np.asarray(spectrum, float), freq_range=list(FIT_RANGE))
    return fm


def fit_group(spectra: np.ndarray, freqs: np.ndarray, mode: str) -> FOOOFGroup:
    """Fit FOOOFGroup on (n_channels, n_freqs) linear power."""
    fg = FOOOFGroup(aperiodic_mode=mode, verbose=False, **FOOOF_KW)
    fg.fit(np.asarray(freqs, float), np.asarray(spectra, float), freq_range=list(FIT_RANGE))
    return fg


def group_aperiodic(fg: FOOOFGroup, mode: str) -> dict[str, np.ndarray]:
    off = fg.get_params("aperiodic_params", "offset")
    exp = fg.get_params("aperiodic_params", "exponent")
    knee = fg.get_params("aperiodic_params", "knee") if mode == "knee" else np.full_like(off, np.nan)
    return dict(offset=off, knee=knee, exponent=exp,
                r_squared=fg.get_params("r_squared"), error=fg.get_params("error"))


def group_peaks_by_channel(fg: FOOOFGroup, n_ch: int) -> list[list[tuple[float, float, float]]]:
    """Return per-channel list of (CF, PW, BW) peaks from a FOOOFGroup fit."""
    out: list[list[tuple[float, float, float]]] = [[] for _ in range(n_ch)]
    pk = fg.get_params("peak_params")
    pk = np.atleast_2d(np.asarray(pk, float))
    if pk.size == 0 or pk.shape[1] < 4:
        return out
    for cf, pw, bw, idx in pk:
        if np.isfinite(idx):
            out[int(idx)].append((float(cf), float(pw), float(bw)))
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _plot_roi_fits(title: str, roi_models: dict[tuple[str, str], FOOOF], out_png) -> None:
    """Per-ROI panels: log-power data + aperiodic fit, real vs silent."""
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(20, 5), sharex=True)
    fig.suptitle(title, fontsize=11)
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        for cond in ("real", "silent"):
            fm = roi_models.get((roi, cond))
            if fm is None or not np.any(np.isfinite(fm.power_spectrum)):
                continue
            c = COND_COLORS[cond]
            ax.plot(fm.freqs, fm.power_spectrum, color=c, lw=1.4,
                    label=f"{cond} (data)")
            ap = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
            ax.plot(fm.freqs, ap, color=c, lw=1.2, ls="--", alpha=0.8,
                    label=f"{cond} (1/f)")
        ax.set_xscale("log")
        ax.set_title(roi)
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("log10 power")
        ax.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_roi_flattened(sub: str, roi_models: dict[tuple[str, str], FOOOF], out_png) -> None:
    """Per-ROI aperiodic-removed (flattened) spectrum, real vs silent.

    Shows whether a periodic alpha peak is preserved across conditions while the 1/f
    floor shifts.
    """
    fig, axes = plt.subplots(1, len(ROI_NAMES), figsize=(20, 5), sharex=True)
    fig.suptitle(f"{sub} -- aperiodic-removed (flattened) spectrum, real vs silent "
                 f"[{PRIMARY_MODE}]", fontsize=11)
    for col, roi in enumerate(ROI_NAMES):
        ax = axes[col]
        for cond in ("real", "silent"):
            fm = roi_models.get((roi, cond))
            if fm is None or not np.any(np.isfinite(fm.power_spectrum)):
                continue
            ap = gen_aperiodic(fm.freqs, fm.aperiodic_params_)
            flat = fm.power_spectrum - ap
            ax.plot(fm.freqs, flat, color=COND_COLORS[cond], lw=1.5,
                    ls="-" if cond == "real" else "--", label=cond)
        for lo, hi in (BANDS["alpha"],):
            ax.axvspan(lo, hi, color="gold", alpha=0.12)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_title(roi)
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("flattened power (log10)")
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_topo_aperiodic(info: mne.Info, d_offset: np.ndarray, d_exponent: np.ndarray,
                         n_sub: int, out_png) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    fig.suptitle(f"Grand-average aperiodic Δ (real - silent), N={n_sub}  [{PRIMARY_MODE}]",
                 fontsize=11)
    for ax, data, name in zip(axes, (d_offset, d_exponent), ("Δ offset", "Δ exponent")):
        vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
        im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", sensors=True, contours=4)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_grandaverage(roi_lin: dict[tuple[str, str], list[np.ndarray]], freqs: np.ndarray,
                       out_png) -> None:
    """Refit FOOOF on across-subject-averaged ROI spectra for a clean GA model plot."""
    roi_models: dict[tuple[str, str], FOOOF] = {}
    for roi in ROI_NAMES:
        for cond in ("real", "silent"):
            specs = roi_lin.get((roi, cond))
            if not specs:
                continue
            ga_spec = np.mean(np.array(specs), axis=0)
            roi_models[(roi, cond)] = fit_fooof(ga_spec, freqs, PRIMARY_MODE)
    _plot_roi_fits(f"Grand-average FOOOF per ROI, real vs silent  [{PRIMARY_MODE}]",
                   roi_models, out_png)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(subjects: list[str]) -> None:
    print("=" * 78)
    print(f"fooof_scr :: FOOOF aperiodic/periodic, real (SCR) vs silent-EDA  [{FIT_RANGE} Hz]")
    print("=" * 78)

    aperiodic_rows: list[dict] = []
    periodic_rows: list[dict] = []
    # per-subject contrast accumulators (PRIMARY_MODE), keyed by channel/roi name
    ch_contrasts: dict[str, list[dict]] = {}
    roi_contrasts: dict[str, list[dict]] = {}
    roi_lin_ga: dict[tuple[str, str], list[np.ndarray]] = {}
    contrast_rows: list[dict] = []

    ref_ch_names: list[str] | None = None
    ref_info: mne.Info | None = None
    common_freqs: np.ndarray | None = None
    mode_r2: dict[str, list[float]] = {m: [] for m in MODES}

    for sub in subjects:
        print(f"\n=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print("  no epochs -- skipped")
            continue
        psd_r, freqs, ch_names = compute_psd(real_ep)
        psd_s, _, _ = compute_psd(silent_ep)
        spec = {"real": psd_r.mean(axis=0), "silent": psd_s.mean(axis=0)}  # (n_ch, n_freq) linear
        n_ch = len(ch_names)
        print(f"  n_real={len(real_ep)} n_silent={len(silent_ep)} n_ch={n_ch}")

        if ref_ch_names is None:
            ref_ch_names = list(ch_names)
            ref_info = real_ep.info.copy()
            common_freqs = freqs

        roi_idx = roi_channel_indices(ch_names)

        # --- per-channel group fits, both modes ---
        ch_ap: dict[str, dict[str, np.ndarray]] = {}   # cond -> aperiodic dict (PRIMARY_MODE)
        ch_peaks: dict[str, list[list[tuple]]] = {}     # cond -> per-channel peaks (PRIMARY_MODE)
        for cond in ("real", "silent"):
            for mode in MODES:
                fg = fit_group(spec[cond], freqs, mode)
                ap = group_aperiodic(fg, mode)
                mode_r2[mode].extend([v for v in ap["r_squared"] if np.isfinite(v)])
                for i, ch in enumerate(ch_names):
                    aperiodic_rows.append(dict(
                        subject=sub, level="channel", key=ch, condition=cond, mode=mode,
                        offset=float(ap["offset"][i]), knee=float(ap["knee"][i]),
                        exponent=float(ap["exponent"][i]),
                        r_squared=float(ap["r_squared"][i]), error=float(ap["error"][i]),
                    ))
                if mode == PRIMARY_MODE:
                    ch_ap[cond] = ap
                    ch_peaks[cond] = group_peaks_by_channel(fg, n_ch)
                    for i, ch in enumerate(ch_names):
                        for cf, pw, bw in ch_peaks[cond][i]:
                            periodic_rows.append(dict(
                                subject=sub, level="channel", key=ch, condition=cond,
                                band=_band_of(cf), cf=cf, pw=pw, bw=bw,
                            ))

        # per-channel contrasts (PRIMARY_MODE)
        for i, ch in enumerate(ch_names):
            pp_r = _periodic_band_power(ch_peaks["real"][i])
            pp_s = _periodic_band_power(ch_peaks["silent"][i])
            row = dict(
                subject=sub, level="channel", key=ch,
                d_offset=float(ch_ap["real"]["offset"][i] - ch_ap["silent"]["offset"][i]),
                d_exponent=float(ch_ap["real"]["exponent"][i] - ch_ap["silent"]["exponent"][i]),
            )
            for b in BAND_NAMES:
                row[f"d_periodic_{b}"] = pp_r[b] - pp_s[b]
            contrast_rows.append(row)
            ch_contrasts.setdefault(ch, []).append(row)

        # --- per-ROI single fits (PRIMARY_MODE) for figures + roi-level params ---
        roi_models: dict[tuple[str, str], FOOOF] = {}
        roi_ap: dict[tuple[str, str], FOOOF] = {}
        for roi in ROI_NAMES:
            idxs = roi_idx[roi]
            if not idxs:
                continue
            for cond in ("real", "silent"):
                roi_spec = spec[cond][idxs].mean(axis=0)
                fm = fit_fooof(roi_spec, freqs, PRIMARY_MODE)
                roi_models[(roi, cond)] = fm
                roi_ap[(roi, cond)] = fm
                roi_lin_ga.setdefault((roi, cond), []).append(
                    roi_spec.astype(float))
                aperiodic_rows.append(dict(
                    subject=sub, level="roi", key=roi, condition=cond, mode=PRIMARY_MODE,
                    offset=float(fm.aperiodic_params_[0]), knee=np.nan,
                    exponent=float(fm.aperiodic_params_[-1]),
                    r_squared=float(fm.r_squared_), error=float(fm.error_),
                ))
                peaks = [(p[0], p[1], p[2]) for p in np.atleast_2d(fm.peak_params_)] \
                    if fm.peak_params_.size else []
                for cf, pw, bw in peaks:
                    periodic_rows.append(dict(
                        subject=sub, level="roi", key=roi, condition=cond,
                        band=_band_of(cf), cf=cf, pw=pw, bw=bw,
                    ))
        # roi-level contrasts
        for roi in ROI_NAMES:
            if (roi, "real") not in roi_ap or (roi, "silent") not in roi_ap:
                continue
            fm_r, fm_s = roi_ap[(roi, "real")], roi_ap[(roi, "silent")]
            pk_r = [(p[0], p[1], p[2]) for p in np.atleast_2d(fm_r.peak_params_)] if fm_r.peak_params_.size else []
            pk_s = [(p[0], p[1], p[2]) for p in np.atleast_2d(fm_s.peak_params_)] if fm_s.peak_params_.size else []
            pp_r, pp_s = _periodic_band_power(pk_r), _periodic_band_power(pk_s)
            row = dict(
                subject=sub, level="roi", key=roi,
                d_offset=float(fm_r.aperiodic_params_[0] - fm_s.aperiodic_params_[0]),
                d_exponent=float(fm_r.aperiodic_params_[-1] - fm_s.aperiodic_params_[-1]),
            )
            for b in BAND_NAMES:
                row[f"d_periodic_{b}"] = pp_r[b] - pp_s[b]
            contrast_rows.append(row)
            roi_contrasts.setdefault(roi, []).append(row)

        # per-subject figures
        _plot_roi_fits(f"{sub} -- FOOOF per ROI, real vs silent  [{PRIMARY_MODE}]",
                       roi_models, FIG_DIR / f"fooof_scr_{sub}.png")
        _plot_roi_flattened(sub, roi_models, FIG_DIR / f"fooof_scr_flattened_{sub}.png")
        print(f"  -> fooof_scr_{sub}.png / fooof_scr_flattened_{sub}.png")

    if ref_ch_names is None:
        print("No subjects processed.")
        return

    n_sub = len({r["subject"] for r in contrast_rows})

    # --- grand-average channel contrasts -> rows + topomap ---
    d_off_ga = np.array([np.nanmean([r["d_offset"] for r in ch_contrasts.get(ch, [])])
                         if ch in ch_contrasts else np.nan for ch in ref_ch_names])
    d_exp_ga = np.array([np.nanmean([r["d_exponent"] for r in ch_contrasts.get(ch, [])])
                         if ch in ch_contrasts else np.nan for ch in ref_ch_names])
    for i, ch in enumerate(ref_ch_names):
        rows = ch_contrasts.get(ch, [])
        if not rows:
            continue
        ga = dict(subject="GA", level="channel", key=ch,
                  d_offset=float(d_off_ga[i]), d_exponent=float(d_exp_ga[i]))
        for b in BAND_NAMES:
            ga[f"d_periodic_{b}"] = float(np.nanmean([r[f"d_periodic_{b}"] for r in rows]))
        contrast_rows.append(ga)
    for roi in ROI_NAMES:
        rows = roi_contrasts.get(roi, [])
        if not rows:
            continue
        ga = dict(subject="GA", level="roi", key=roi,
                  d_offset=float(np.nanmean([r["d_offset"] for r in rows])),
                  d_exponent=float(np.nanmean([r["d_exponent"] for r in rows])))
        for b in BAND_NAMES:
            ga[f"d_periodic_{b}"] = float(np.nanmean([r[f"d_periodic_{b}"] for r in rows]))
        contrast_rows.append(ga)

    _plot_topo_aperiodic(ref_info, d_off_ga, d_exp_ga, n_sub,
                         FIG_DIR / "fooof_scr_topo_aperiodic.png")
    _plot_grandaverage(roi_lin_ga, common_freqs, FIG_DIR / "fooof_scr_grandaverage.png")
    print("\nGrand-average figures saved.")

    # --- write tables ---
    pd.DataFrame(aperiodic_rows).to_csv(TBL_DIR / "fooof_scr_aperiodic.csv", index=False)
    pd.DataFrame(periodic_rows).to_csv(TBL_DIR / "fooof_scr_periodic.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(TBL_DIR / "fooof_scr_contrasts.csv", index=False)
    print(f"Tables -> {TBL_DIR}")

    # --- model-choice diagnostic: which aperiodic mode fits better? ---
    print("\nAperiodic mode fit (mean r_squared across channels x subjects x conditions):")
    for mode in MODES:
        vals = mode_r2[mode]
        if vals:
            print(f"  {mode:6s}: {np.mean(vals):.4f}  (n={len(vals)})")
    print(f"  -> contrasts/topomaps use PRIMARY_MODE='{PRIMARY_MODE}'.")


def _band_of(cf: float) -> str:
    for b, (lo, hi) in BANDS.items():
        if lo <= cf < hi:
            return b
    return "out"


def _parse_args() -> list[str]:
    ap = argparse.ArgumentParser(description="FOOOF SCR vs silent-EDA parametrization.")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Subset of subjects (e.g. sub-27). Default: full cohort.")
    args = ap.parse_args()
    return args.subjects if args.subjects else list(SUBJECTS)


if __name__ == "__main__":
    main(_parse_args())
