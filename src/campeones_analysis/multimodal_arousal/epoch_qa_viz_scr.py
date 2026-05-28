"""Epoch-level QA visualization of SCR vs silent-EDA epochs (artifact inspection).

Goal (2026-05-27, Tarea A del prompt de QA): let the researcher SEE whether the
SCR-vs-no-SCR comparisons (time / frequency / time-frequency) are driven by
artifacts. Reuses the EXACT epochs that feed erp_scr / tfr_psd_scr / decoding via
`build_subject_epochs(sub)` (imported from tfr_psd_scr) so what you inspect here is
what the analyses actually used. This module ONLY reads the shared modules; it never
edits cohort.py / erp_scr.py / tfr_psd_scr.py.

Two modes (incompatible matplotlib backends -> kept strictly separate):

  A1  BATCH (default, Agg backend, non-interactive)
      One consolidated multi-panel PNG per subject with, for real (SCR) vs silent:
        - epochs x time GFP heatmap per condition (spot outlier epochs / bursts)
        - butterfly + GFP of the evoked, real vs silent
        - Welch PSD real vs silent (1-40 Hz, dB, log-x) -> broadband / gamma artifact
        - Morlet TFR difference (real - silent), channel-averaged, logratio baseline
        - per-epoch peak-to-peak distribution (autoreject's metric), real vs silent
        - per-channel temporal-variance topomap (real / silent / diff) in [0, 3] s
          -> localize artifactual channels (temporal/edge = EMG)
      Saves to research_diary/context/05_04/cohort6/figures/epoch_qa/ (new folder).
      Also writes a per-epoch peak-to-peak table to .../tables/epoch_qa_ptp_<sub>.csv.

  A2  INTERACTIVE (--interactive, Qt backend, BLOCKING)
      Opens MNE's navigable epochs browser for ONE subject + ONE condition so you can
      scroll epochs/channels, zoom, and CLICK epochs to mark them bad. On close it
      prints which epochs you marked (from the drop_log) so they can be reused later
      as a manual-rejection list. Run one subject/condition at a time, e.g.:
        ... epoch_qa_viz_scr --interactive --subject 27 --cond real
        ... epoch_qa_viz_scr --interactive --subject 27 --cond silent

Run (BATCH, all cohort):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr

Run (BATCH, subset):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr --subjects 27 31

Run (INTERACTIVE browser, foreground only -- NOT background, NOT Agg):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr --interactive --subject 27 --cond real
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# NOTE on backends: importing build_subject_epochs pulls in tfr_psd_scr -> erp_scr, both
# of which call matplotlib.use("Agg") at import time. For BATCH that is exactly what we
# want. For INTERACTIVE we override to QtAgg AFTER these imports (see run_interactive).
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    PSD_FMAX,
    PSD_FMIN,
    TFR_BASELINE,
    TFR_BASELINE_MODE,
    build_subject_epochs,
    compute_psd,
    compute_tfr,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, TMAX, TMIN

import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "epoch_qa"
TBL_DIR = OUT / "figures" / "epoch_qa" / "tables"

# Post-onset window used for the variance topomap + PSD (where SCR-related EEG / EMG
# would sit). Kept inside [TMIN, TMAX].
POST_WIN = (0.0, min(3.0, TMAX))

# Frequency bands for the per-band PSD diff annotation (matches tfr_psd_scr / decoding).
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def _norm_subject(s: str) -> str:
    """Accept '27' or 'sub-27' -> 'sub-27'."""
    s = str(s).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


# -----------------------------------------------------------------------------
# Per-epoch metrics
# -----------------------------------------------------------------------------
def epoch_ptp(epochs: mne.Epochs, win: tuple[float, float]) -> np.ndarray:
    """Per-epoch, per-channel peak-to-peak (uV) within `win`. Shape (n_epochs, n_ch)."""
    sl = epochs.copy().crop(tmin=win[0], tmax=win[1])
    data = sl.get_data(copy=True) * 1e6  # -> uV
    return data.max(axis=2) - data.min(axis=2)  # (n_epochs, n_ch)


def channel_var_topo(epochs: mne.Epochs, win: tuple[float, float]) -> np.ndarray:
    """Per-channel temporal variance (uV^2) in `win`, averaged across epochs. Shape (n_ch,)."""
    sl = epochs.copy().crop(tmin=win[0], tmax=win[1])
    data = sl.get_data(copy=True) * 1e6  # uV
    return data.var(axis=2).mean(axis=0)  # var over time, mean over epochs


# -----------------------------------------------------------------------------
# A1 -- BATCH consolidated QA figure
# -----------------------------------------------------------------------------
def make_subject_figure(sub: str, real_ep: mne.Epochs, silent_ep: mne.Epochs,
                        out_png: Path) -> "pd.DataFrame":  # noqa: F821
    import matplotlib.pyplot as plt
    import pandas as pd

    ch_names = real_ep.ch_names
    times = real_ep.times
    sfreq = real_ep.info["sfreq"]

    # --- evoked (time domain) ---
    ev_real = real_ep.average()
    ev_silent = silent_ep.average()

    # --- GFP per epoch (epochs x time) for the heatmaps ---
    def gfp_img(ep: mne.Epochs) -> np.ndarray:
        d = ep.get_data(copy=True) * 1e6  # (n_ep, n_ch, n_t)
        return d.std(axis=1)              # GFP across channels -> (n_ep, n_t)

    gfp_real = gfp_img(real_ep)
    gfp_silent = gfp_img(silent_ep)

    # --- PSD (reuse tfr_psd_scr.compute_psd) ---
    psd_real, freqs_psd, _ = compute_psd(real_ep)      # (n_ep, n_ch, n_freq)
    psd_silent, _, _ = compute_psd(silent_ep)
    # channel-averaged dB per epoch
    pr_db = 10.0 * np.log10(psd_real.mean(axis=1) + 1e-30)   # (n_ep, n_freq)
    ps_db = 10.0 * np.log10(psd_silent.mean(axis=1) + 1e-30)
    pr_m, pr_s = pr_db.mean(0), pr_db.std(0, ddof=1) / np.sqrt(max(1, pr_db.shape[0]))
    ps_m, ps_s = ps_db.mean(0), ps_db.std(0, ddof=1) / np.sqrt(max(1, ps_db.shape[0]))

    # --- TFR diff (reuse tfr_psd_scr.compute_tfr; channel-averaged) ---
    tfr_real = compute_tfr(real_ep).apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)
    tfr_silent = compute_tfr(silent_ep).apply_baseline(TFR_BASELINE, mode=TFR_BASELINE_MODE)
    tfr_diff = tfr_real.data.mean(axis=0) - tfr_silent.data.mean(axis=0)  # (n_freq, n_t)
    tfr_t, tfr_f = tfr_real.times, tfr_real.freqs

    # --- per-epoch peak-to-peak (autoreject metric) ---
    ptp_real = epoch_ptp(real_ep, POST_WIN)      # (n_ep, n_ch)
    ptp_silent = epoch_ptp(silent_ep, POST_WIN)
    ptp_real_max = ptp_real.max(axis=1)          # worst channel per epoch
    ptp_silent_max = ptp_silent.max(axis=1)

    # --- per-channel variance topomaps ---
    var_real = channel_var_topo(real_ep, POST_WIN)
    var_silent = channel_var_topo(silent_ep, POST_WIN)
    info = real_ep.info

    # ============================ FIGURE ============================
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.1, 1.0, 1.0, 1.1], hspace=0.38, wspace=0.28)
    fig.suptitle(
        f"{sub}  --  Epoch QA  (SCR 'real' vs silent-EDA control)   "
        f"n_real={len(real_ep)}  n_silent={len(silent_ep)}   "
        f"window=[{TMIN:g},{TMAX:g}]s  band=1-40Hz  {sfreq:g}Hz  {len(ch_names)}ch",
        fontsize=13, y=0.995,
    )

    # Row 0: GFP heatmaps (epochs x time) + colorbars
    vmax_gfp = float(np.percentile(np.concatenate([gfp_real.ravel(), gfp_silent.ravel()]), 99))
    for col, (img, name, n) in enumerate([(gfp_real, "real (SCR)", len(real_ep)),
                                          (gfp_silent, "silent EDA", len(silent_ep))]):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax_gfp,
                       extent=[times[0], times[-1], 0, img.shape[0]])
        ax.axvline(0, color="cyan", lw=0.8)
        ax.set_title(f"GFP per epoch -- {name} (N={n})", fontsize=10)
        ax.set_xlabel("time from onset (s)")
        ax.set_ylabel("epoch #")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="GFP (uV)")

    # Row 1, col 0-1: butterfly + GFP evoked overlay
    ax = fig.add_subplot(gs[1, 0:2])
    er = ev_real.data * 1e6  # (n_ch, n_t)
    es = ev_silent.data * 1e6
    ax.plot(times, er.T, color="C3", lw=0.4, alpha=0.35)
    ax.plot(times, es.T, color="0.5", lw=0.4, alpha=0.30)
    ax.plot(times, er.std(0), color="C3", lw=2.0, label="real GFP")
    ax.plot(times, es.std(0), color="k", lw=1.6, ls="--", label="silent GFP")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title("Butterfly (thin) + GFP (thick): real vs silent", fontsize=10)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("uV")
    ax.legend(fontsize=8)

    # Row 1, col 2-3: PSD real vs silent (channel-averaged, dB)
    ax = fig.add_subplot(gs[1, 2:4])
    ax.fill_between(freqs_psd, pr_m - pr_s, pr_m + pr_s, color="C3", alpha=0.22, lw=0)
    ax.fill_between(freqs_psd, ps_m - ps_s, ps_m + ps_s, color="0.4", alpha=0.22, lw=0)
    ax.plot(freqs_psd, pr_m, color="C3", lw=1.8, label="real (SCR)")
    ax.plot(freqs_psd, ps_m, color="0.4", lw=1.5, ls="--", label="silent EDA")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"PSD (Welch, {PSD_FMIN:g}-{PSD_FMAX:g} Hz, all-chan mean +/- SEM)", fontsize=10)
    ax.legend(fontsize=8)
    # per-band diff annotation
    band_txt = []
    for b, (lo, hi) in BANDS.items():
        m = (freqs_psd >= lo) & (freqs_psd < hi)
        if m.any():
            band_txt.append(f"{b}:{(pr_m[m] - ps_m[m]).mean():+.2f}")
    ax.text(0.02, 0.03, "dB diff (real-silent)  " + "  ".join(band_txt),
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # Row 2, col 0-1: TFR diff (channel-averaged)
    ax = fig.add_subplot(gs[2, 0:2])
    vmax_tfr = float(np.nanpercentile(np.abs(tfr_diff), 99)) or 1.0
    im = ax.pcolormesh(tfr_t, tfr_f, tfr_diff, cmap="RdBu_r", vmin=-vmax_tfr, vmax=vmax_tfr, shading="auto")
    ax.set_yscale("log")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"TFR diff real-silent (Morlet, all-chan mean, {TFR_BASELINE_MODE} baseline)", fontsize=10)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("Freq (Hz)")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="logratio")

    # Row 2, col 2-3: per-epoch peak-to-peak histograms
    ax = fig.add_subplot(gs[2, 2:4])
    allmax = np.concatenate([ptp_real_max, ptp_silent_max])
    bins = np.linspace(0, float(np.percentile(allmax, 99)), 40)
    ax.hist(ptp_real_max, bins=bins, color="C3", alpha=0.55, label=f"real (med {np.median(ptp_real_max):.0f}uV)")
    ax.hist(ptp_silent_max, bins=bins, color="0.5", alpha=0.55, label=f"silent (med {np.median(ptp_silent_max):.0f}uV)")
    ax.set_xlabel("max-channel peak-to-peak per epoch (uV)  [window 0-3 s]")
    ax.set_ylabel("# epochs")
    ax.set_title("Per-epoch peak-to-peak (autoreject metric)", fontsize=10)
    ax.legend(fontsize=8)

    # Row 3: variance topomaps real / silent / diff
    vmax_var = float(np.percentile(np.concatenate([var_real, var_silent]), 98))
    for col, (vals, name, cmap, vlim) in enumerate([
        (var_real, "var real (SCR)", "viridis", (0, vmax_var)),
        (var_silent, "var silent EDA", "viridis", (0, vmax_var)),
        (var_real - var_silent, "var diff (real-silent)", "RdBu_r", None),
    ]):
        ax = fig.add_subplot(gs[3, col])
        if vlim is None:
            vlim = (-float(np.percentile(np.abs(var_real - var_silent), 98)),
                    float(np.percentile(np.abs(var_real - var_silent), 98)))
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap,
                                     vlim=vlim, contours=4)
        ax.set_title(f"{name}\n[{POST_WIN[0]:g}-{POST_WIN[1]:g}s]", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="uV^2")

    # Row 3, col 3: worst channels by variance-diff (text)
    ax = fig.add_subplot(gs[3, 3])
    ax.axis("off")
    vdiff = var_real - var_silent
    order = np.argsort(vdiff)[::-1]
    lines = ["Top channels by var(real)-var(silent):", ""]
    for k in order[:10]:
        lines.append(f"  {ch_names[k]:>5s}   {vdiff[k]:+8.1f} uV^2")
    lines += ["", "(temporal/edge chans -> EMG-like;", " central/parietal -> more cortical)"]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="top", family="monospace")

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- per-epoch ptp table (for Task B autoreject comparison later) ----
    rows = []
    for cond, ptp_max in [("real", ptp_real_max), ("silent", ptp_silent_max)]:
        for i, v in enumerate(ptp_max):
            rows.append(dict(subject=sub, cond=cond, epoch=i, ptp_max_uV=float(v)))
    return pd.DataFrame(rows)


def run_batch(subjects: list[str]) -> None:
    import pandas as pd

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("epoch_qa_viz_scr :: A1 BATCH (Agg) -- one consolidated QA PNG per subject")
    print("=" * 78)
    all_tbl = []
    for sub in subjects:
        print(f"\n=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print("  no epochs -- skipping")
            continue
        print(f"  epochs real={len(real_ep)}  silent={len(silent_ep)}  ch={len(real_ep.ch_names)}")
        out_png = FIG_DIR / f"epoch_qa_{sub}.png"
        df = make_subject_figure(sub, real_ep, silent_ep, out_png)
        df.to_csv(TBL_DIR / f"epoch_qa_ptp_{sub}.csv", index=False)
        all_tbl.append(df)
        print(f"  -> {out_png.relative_to(OUT)}")
    if all_tbl:
        allc = pd.concat(all_tbl, ignore_index=True)
        allc.to_csv(TBL_DIR / "epoch_qa_ptp_all.csv", index=False)
        # quick per-subject/condition ptp summary to stdout
        print("\nPer-epoch peak-to-peak summary (max-channel, uV, window 0-3 s):")
        summ = allc.groupby(["subject", "cond"])["ptp_max_uV"].agg(["count", "median", "max"])
        print(summ.to_string())
        print(f"\nFigures -> {FIG_DIR}")
        print(f"Tables  -> {TBL_DIR}")


# -----------------------------------------------------------------------------
# A2 -- INTERACTIVE browser
# -----------------------------------------------------------------------------
def run_interactive(subject: str, cond: str, n_epochs: int, n_channels: int) -> None:
    # Override the Agg backend forced by the tfr_psd_scr/erp_scr imports.
    import matplotlib
    matplotlib.use("QtAgg", force=True)
    mne.viz.set_browser_backend("qt")

    print("=" * 78)
    print(f"epoch_qa_viz_scr :: A2 INTERACTIVE (Qt) -- {subject} / {cond}")
    print("  Scroll epochs (Page Up/Down) and channels; click an epoch to mark it BAD.")
    print("  Close the window to finish; marked epochs are reported below.")
    print("=" * 78)

    real_ep, silent_ep = build_subject_epochs(subject)
    if real_ep is None or silent_ep is None:
        print(f"  {subject}: no epochs -- nothing to show")
        return
    ep = real_ep if cond == "real" else silent_ep
    n_before = len(ep)
    print(f"  loaded {n_before} '{cond}' epochs ({len(ep.ch_names)} channels)")

    title = f"{subject} -- {cond} (SCR) epochs -- click to mark bad, close when done"
    ep.plot(block=True, n_epochs=n_epochs, n_channels=n_channels,
            scalings="auto", title=title, picks="eeg")

    # After close: report manually-marked epochs from the drop_log.
    bad_idx = [i for i, reason in enumerate(ep.drop_log)
               if reason and reason != ("IGNORED",)]
    print("\n" + "-" * 60)
    print(f"  Epochs marked bad (manual): {len(bad_idx)} / {n_before}")
    if bad_idx:
        print(f"  indices: {bad_idx}")
        print(f"  reasons: {[ep.drop_log[i] for i in bad_idx]}")
    else:
        print("  (none marked)")
    print(f"  Remaining good epochs: {len(ep)}")
    print("-" * 60)


# -----------------------------------------------------------------------------
def main() -> None:
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT

    p = argparse.ArgumentParser(description="Epoch QA visualization for SCR vs silent epochs.")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="subjects for BATCH (e.g. 27 31 or sub-27). Default = full cohort.")
    p.add_argument("--interactive", action="store_true",
                   help="open the Qt epochs browser for ONE subject/cond (blocking).")
    p.add_argument("--subject", default=None, help="subject for --interactive (e.g. 27).")
    p.add_argument("--cond", choices=["real", "silent"], default="real",
                   help="condition for --interactive.")
    p.add_argument("--n-epochs", type=int, default=10, help="epochs shown per page (interactive).")
    p.add_argument("--n-channels", type=int, default=32, help="channels shown per page (interactive).")
    args = p.parse_args()

    if args.interactive:
        if not args.subject:
            print("ERROR: --interactive requires --subject (e.g. --subject 27)", file=sys.stderr)
            sys.exit(2)
        run_interactive(_norm_subject(args.subject), args.cond, args.n_epochs, args.n_channels)
    else:
        subs = [_norm_subject(s) for s in args.subjects] if args.subjects else list(COHORT)
        run_batch(subs)


if __name__ == "__main__":
    main()
