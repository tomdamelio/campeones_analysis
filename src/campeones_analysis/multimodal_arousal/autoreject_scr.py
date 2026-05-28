"""Tarea B -- autoreject (Jas et al. 2017) test on the SCR vs silent-EDA epochs.

Phase 1 (standalone test; does NOT edit the shared modules). Motivation: the epoch QA
(epoch_qa_viz_scr.py) showed the SCR-vs-no-SCR difference is artifact-like (broadband PSD
elevation + variance concentrated at temporal/edge channels FT9/TP9/T7 = EMG/movement),
surviving the continuous preprocessing (PyPREP + ICA/ICLabel + average reference). Those
are stereotyped-spatial-component and global-bad-channel removers; autoreject targets the
COMPLEMENTARY residual class: high-amplitude transient artifacts localized per-epoch and
per-sensor. This is a TEST: does the artifact signature shrink after autoreject (confirms
artifact) or does a central/delta residual survive (more interesting)?

Key design for an UNBIASED comparison:
  Fit autoreject ONCE on the COMBINED real+silent epochs (same rejection criterion for
  both conditions), then split back by event code. Fitting per condition would bias the
  real-vs-silent contrast with different thresholds.

Compatibility with the existing preproc:
  - Complementary, not duplicative (see above).
  - autoreject interpolates bad sensors per epoch via spherical splines -> needs the
    montage, already attached by build_subject_epochs/attach_montage_and_drop_no_pos.
  - Average reference + prior PyPREP interpolation: re-interpolating is harmless; no
    re-referencing needed. Rank may already be reduced; that is fine.
  - Reproducible: fixed random_state (autoreject CV has randomness).

Outputs (cohort6/figures/autoreject/):
  reject_log_<sub>.png      epochs x channels grid (good / interpolated / bad)
  before_after_<sub>.png    evoked GFP + PSD + variance-topomap, before vs after, per cond
  <sub>_reject_log.npz      labels + bad_epochs + condition boundary (Phase-2 traceability)
  autoreject_summary.csv     per-subject drop/flag stats (real vs silent)

Run (the two target subjects):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.autoreject_scr --subjects 23 24

Run (full cohort):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.autoreject_scr
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

# Importing build_subject_epochs pulls tfr_psd_scr -> erp_scr, both matplotlib.use("Agg").
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    PSD_FMAX,
    PSD_FMIN,
    build_subject_epochs,
    compute_psd,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, TMAX, TMIN
# reuse the QA metric helpers (same package, my own new module)
from src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr import (
    BANDS,
    POST_WIN,
    channel_var_topo,
)

import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "autoreject"


def _norm_subject(s: str) -> str:
    s = str(s).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def combine_real_silent(real_ep: mne.Epochs, silent_ep: mne.Epochs) -> mne.Epochs:
    """Concatenate real+silent into one Epochs with distinct event ids (real=1, silent=2).

    build_subject_epochs returns epochs whose events carry code 1 (real) / 2 (silent) but
    both use the event_id key "x". Rename keys so concatenate_epochs keeps them distinct.
    """
    r = real_ep.copy()
    s = silent_ep.copy()
    r.event_id = {"real": 1}
    s.event_id = {"silent": 2}
    return mne.concatenate_epochs([r, s], verbose="ERROR")


def _psd_db_mean_sem(epochs: mne.Epochs):
    """All-channel-mean PSD in dB -> (freqs, mean_db, sem_db). Empty-safe."""
    if len(epochs) == 0:
        return None, None, None
    psd, freqs, _ = compute_psd(epochs)              # (n_ep, n_ch, n_freq)
    db = 10.0 * np.log10(psd.mean(axis=1) + 1e-30)   # (n_ep, n_freq)
    m = db.mean(0)
    sem = db.std(0, ddof=1) / np.sqrt(max(1, db.shape[0])) if db.shape[0] > 1 else np.zeros_like(m)
    return freqs, m, sem


def before_after_figure(sub: str, real_before, silent_before, real_after, silent_after,
                        out_png: Path) -> None:
    import matplotlib.pyplot as plt

    info = real_before.info
    times = real_before.times
    # shared variance color scale across all four states for honest comparison
    var_all = []
    for ep in (real_before, silent_before, real_after, silent_after):
        if len(ep):
            var_all.append(channel_var_topo(ep, POST_WIN))
    vmax_var = float(np.percentile(np.concatenate(var_all), 98)) if var_all else 1.0

    states = [
        ("real before", real_before, "C3"),
        ("real after", real_after, "C3"),
        ("silent before", silent_before, "0.4"),
        ("silent after", silent_after, "0.4"),
    ]

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.30)
    fig.suptitle(f"{sub}  --  autoreject before/after  (variance color scale shared across rows)",
                 fontsize=13, y=0.995)

    for row, (name, ep, color) in enumerate(states):
        n = len(ep)
        # col 0: evoked butterfly + GFP
        ax = fig.add_subplot(gs[row, 0])
        if n:
            d = ep.average().data * 1e6
            ax.plot(times, d.T, color=color, lw=0.35, alpha=0.3)
            ax.plot(times, d.std(0), color="k", lw=1.8)
            ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{name}  (N={n})  butterfly+GFP", fontsize=9)
        ax.set_xlabel("time (s)"); ax.set_ylabel("uV")

        # col 1: PSD all-chan mean
        ax = fig.add_subplot(gs[row, 1])
        freqs, m, sem = _psd_db_mean_sem(ep)
        if freqs is not None:
            ax.fill_between(freqs, m - sem, m + sem, color=color, alpha=0.22, lw=0)
            ax.plot(freqs, m, color=color, lw=1.6)
            ax.set_xscale("log")
            band_txt = " ".join(
                f"{b}:{m[(freqs>=lo)&(freqs<hi)].mean():.1f}" for b, (lo, hi) in BANDS.items()
                if ((freqs >= lo) & (freqs < hi)).any()
            )
            ax.text(0.02, 0.03, band_txt, transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.set_title(f"{name}  PSD (dB, {PSD_FMIN:g}-{PSD_FMAX:g}Hz)", fontsize=9)
        ax.set_xlabel("Hz"); ax.set_ylabel("dB")

        # col 2: variance topomap
        ax = fig.add_subplot(gs[row, 2])
        if n:
            vals = channel_var_topo(ep, POST_WIN)
            im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="viridis",
                                         vlim=(0, vmax_var), contours=4)
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="uV^2")
        ax.set_title(f"{name}  var [{POST_WIN[0]:g}-{POST_WIN[1]:g}s]", fontsize=9)

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_subject(sub: str, thresh_method: str, decim: int) -> dict | None:
    from autoreject import AutoReject

    t0 = time.time()
    print(f"\n=== {sub} ===", flush=True)
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        print("  no epochs -- skipping", flush=True)
        return None
    n_real_in, n_silent_in = len(real_ep), len(silent_ep)
    print(f"  epochs real={n_real_in} silent={n_silent_in} ch={len(real_ep.ch_names)}", flush=True)

    combined = combine_real_silent(real_ep, silent_ep)
    t_build = time.time()
    print(f"  built+combined in {t_build - t0:.0f}s; fitting AutoReject "
          f"(thresh={thresh_method}, decim={decim}, n_jobs=1)...", flush=True)

    ar = AutoReject(thresh_method=thresh_method, n_jobs=1, random_state=97,
                    verbose=False)
    # decim only affects the threshold computation (faster), not the returned epochs
    ar.fit(combined[::1] if decim == 1 else combined.copy().decimate(decim))
    reject_log = ar.get_reject_log(combined)
    clean = ar.transform(combined)
    t_fit = time.time()
    print(f"  AutoReject fit+transform in {t_fit - t_build:.0f}s", flush=True)

    # split back by condition
    real_after = clean["real"]
    silent_after = clean["silent"]

    # ---- stats from reject_log (combined order = real first, then silent) ----
    bad = np.asarray(reject_log.bad_epochs, dtype=bool)  # dropped epochs
    labels = np.asarray(reject_log.labels, dtype=float)  # (n_comb, n_ch): 0 good, !=0 flagged
    bad_real, bad_silent = bad[:n_real_in], bad[n_real_in:]
    flagged = np.nan_to_num(labels, nan=0.0) != 0.0      # any non-good cell
    badch_per_epoch = flagged.sum(axis=1)
    row = dict(
        subject=sub,
        n_real_in=n_real_in, n_real_kept=len(real_after),
        n_silent_in=n_silent_in, n_silent_kept=len(silent_after),
        pct_drop_real=round(100.0 * bad_real.mean(), 2),
        pct_drop_silent=round(100.0 * bad_silent.mean(), 2),
        mean_badch_real=round(float(badch_per_epoch[:n_real_in].mean()), 2),
        mean_badch_silent=round(float(badch_per_epoch[n_real_in:].mean()), 2),
        secs=round(t_fit - t0, 1),
    )
    print(f"  drop: real {row['pct_drop_real']}%  silent {row['pct_drop_silent']}%  "
          f"| mean bad-ch/epoch: real {row['mean_badch_real']} silent {row['mean_badch_silent']}",
          flush=True)

    # ---- figures ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig = reject_log.plot(orientation="horizontal", show=False)
        if fig is not None:
            fig.savefig(FIG_DIR / f"reject_log_{sub}.png", dpi=120, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
    except Exception as e:
        print(f"  reject_log plot failed: {e}", flush=True)

    before_after_figure(sub, real_ep, silent_ep, real_after, silent_after,
                        FIG_DIR / f"before_after_{sub}.png")

    # ---- persist reject_log for Phase-2 traceability / reuse ----
    np.savez_compressed(
        FIG_DIR / f"{sub}_reject_log.npz",
        labels=labels, bad_epochs=bad, n_real_in=n_real_in, n_silent_in=n_silent_in,
        ch_names=np.array(combined.ch_names),
    )
    print(f"  -> reject_log_{sub}.png, before_after_{sub}.png, {sub}_reject_log.npz "
          f"({row['secs']:.0f}s total)", flush=True)
    return row


def main() -> None:
    import pandas as pd
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT

    p = argparse.ArgumentParser(description="autoreject test on SCR vs silent epochs (Tarea B).")
    p.add_argument("--subjects", nargs="*", default=None, help="e.g. 23 24. Default=full cohort.")
    p.add_argument("--thresh-method", default="bayesian_optimization",
                   choices=["bayesian_optimization", "random_search"],
                   help="random_search is ~2-3x faster (coarser thresholds).")
    p.add_argument("--decim", type=int, default=1,
                   help="decimation for threshold computation only (decim=2 ~2x faster).")
    args = p.parse_args()

    subs = [_norm_subject(s) for s in args.subjects] if args.subjects else list(COHORT)
    print("=" * 78)
    print(f"autoreject_scr :: Tarea B test  subjects={subs}  "
          f"thresh={args.thresh_method} decim={args.decim}")
    print("=" * 78, flush=True)

    rows = []
    t_all = time.time()
    for sub in subs:
        r = run_subject(sub, args.thresh_method, args.decim)
        if r is not None:
            rows.append(r)
    if rows:
        df = pd.DataFrame(rows)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        csv = FIG_DIR / "autoreject_summary.csv"
        # append-safe: merge with any existing summary, dedup by subject (last wins)
        if csv.exists():
            old = pd.read_csv(csv)
            df = pd.concat([old[~old["subject"].isin(df["subject"])], df], ignore_index=True)
        df.sort_values("subject").to_csv(csv, index=False)
        print(f"\nSummary -> {csv}", flush=True)
        print(df.sort_values("subject").to_string(index=False), flush=True)
    print(f"\nTOTAL wall time: {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
