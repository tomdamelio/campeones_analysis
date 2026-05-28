"""Tarea B (tuning) -- compare autoreject aggressiveness configs on the SCR epochs.

Phase-1 follow-up. The default AutoReject interpolated ~12-13/32 channels per epoch
(the whole temporal belt on nearly every epoch) -- very aggressive. Before committing a
config for the full cohort / Phase 2, compare three options on the REAL (SCR) condition,
which is where the artifact lives:

  1. "default"        AutoReject()                         -> n_interpolate grid {1,4,32}
  2. "interp<=8"      AutoReject(n_interpolate=[1,4,8])     -> caps interpolation; epochs
                      that would need >8 bad channels get DROPPED instead of interpolated
  3. "drop-only"      get_rejection_threshold() + drop_bad  -> global ptp threshold, NO
                      interpolation (pure rejection)

All fit on real+silent COMBINED (same criterion both conditions), then we read the REAL
side. Output: one comparison figure per subject (variance topomap + PSD per config, with
the raw 'before' as reference) and a CSV of drop/interp stats per config.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.autoreject_tune_scr --subjects 23 24
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    PSD_FMAX,
    PSD_FMIN,
    build_subject_epochs,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import OUT
from src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr import BANDS, POST_WIN, channel_var_topo
from src.campeones_analysis.multimodal_arousal.autoreject_scr import combine_real_silent, _norm_subject

import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "autoreject"
RANDOM_STATE = 97


def _interp_per_epoch(reject_log, n_real_in: int) -> tuple[float, float]:
    """Mean #channels flagged (!=good) per epoch, for real and silent halves."""
    labels = np.nan_to_num(np.asarray(reject_log.labels, dtype=float), nan=0.0) != 0.0
    per_ep = labels.sum(axis=1)
    return float(per_ep[:n_real_in].mean()), float(per_ep[n_real_in:].mean())


def run_subject(sub: str) -> list[dict]:
    from autoreject import AutoReject, get_rejection_threshold

    print(f"\n=== {sub} ===", flush=True)
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        print("  no epochs -- skipping", flush=True)
        return []
    n_real_in, n_silent_in = len(real_ep), len(silent_ep)
    combined = combine_real_silent(real_ep, silent_ep)
    print(f"  real={n_real_in} silent={n_silent_in}; running 3 configs...", flush=True)

    results = []  # (label, real_after_epochs, stats)
    rows = []

    # --- raw reference ---
    results.append(("raw (before)", real_ep, dict(n_real_kept=n_real_in)))

    # --- config 1: default AutoReject ---
    t0 = time.time()
    ar = AutoReject(n_jobs=1, random_state=RANDOM_STATE, verbose=False)
    ar.fit(combined)
    rl = ar.get_reject_log(combined)
    clean = ar.transform(combined)
    bad = np.asarray(rl.bad_epochs, dtype=bool)
    ir, isi = _interp_per_epoch(rl, n_real_in)
    stats = dict(subject=sub, config="default", n_real_kept=len(clean["real"]),
                 n_silent_kept=len(clean["silent"]),
                 pct_drop_real=round(100 * bad[:n_real_in].mean(), 2),
                 pct_drop_silent=round(100 * bad[n_real_in:].mean(), 2),
                 mean_flagch_real=round(ir, 2), mean_flagch_silent=round(isi, 2),
                 secs=round(time.time() - t0, 1))
    results.append(("AutoReject default", clean["real"], stats)); rows.append(stats)
    print(f"  default: drop real {stats['pct_drop_real']}% flagch {stats['mean_flagch_real']} ({stats['secs']:.0f}s)", flush=True)

    # --- config 2: interp<=8 ---
    t0 = time.time()
    ar2 = AutoReject(n_interpolate=np.array([1, 4, 8]), n_jobs=1, random_state=RANDOM_STATE, verbose=False)
    ar2.fit(combined)
    rl2 = ar2.get_reject_log(combined)
    clean2 = ar2.transform(combined)
    bad2 = np.asarray(rl2.bad_epochs, dtype=bool)
    ir2, isi2 = _interp_per_epoch(rl2, n_real_in)
    stats2 = dict(subject=sub, config="interp<=8", n_real_kept=len(clean2["real"]),
                  n_silent_kept=len(clean2["silent"]),
                  pct_drop_real=round(100 * bad2[:n_real_in].mean(), 2),
                  pct_drop_silent=round(100 * bad2[n_real_in:].mean(), 2),
                  mean_flagch_real=round(ir2, 2), mean_flagch_silent=round(isi2, 2),
                  secs=round(time.time() - t0, 1))
    results.append(("AutoReject interp<=8", clean2["real"], stats2)); rows.append(stats2)
    print(f"  interp<=8: drop real {stats2['pct_drop_real']}% flagch {stats2['mean_flagch_real']} ({stats2['secs']:.0f}s)", flush=True)

    # --- config 3: drop-only (global threshold, no interpolation) ---
    t0 = time.time()
    thresh = get_rejection_threshold(combined, random_state=RANDOM_STATE, ch_types="eeg")
    dropped = combined.copy().drop_bad(reject=thresh, verbose="ERROR")
    stats3 = dict(subject=sub, config="drop-only", n_real_kept=len(dropped["real"]),
                  n_silent_kept=len(dropped["silent"]),
                  pct_drop_real=round(100 * (1 - len(dropped["real"]) / n_real_in), 2),
                  pct_drop_silent=round(100 * (1 - len(dropped["silent"]) / n_silent_in), 2),
                  mean_flagch_real=0.0, mean_flagch_silent=0.0,
                  secs=round(time.time() - t0, 1))
    results.append(("drop-only (threshold)", dropped["real"], stats3)); rows.append(stats3)
    print(f"  drop-only: drop real {stats3['pct_drop_real']}% (thresh ptp={thresh.get('eeg', float('nan')) * 1e6:.0f}uV) ({stats3['secs']:.0f}s)", flush=True)

    _comparison_figure(sub, results)
    return rows


def _comparison_figure(sub: str, results: list) -> None:
    import matplotlib.pyplot as plt

    info = results[0][1].info
    raw_real = results[0][1]
    # shared variance scale from the raw 'before'
    vmax_var = float(np.percentile(channel_var_topo(raw_real, POST_WIN), 98))
    # raw PSD reference
    from src.campeones_analysis.multimodal_arousal.autoreject_scr import _psd_db_mean_sem
    f0, m0, _ = _psd_db_mean_sem(raw_real)

    n = len(results)
    fig = plt.figure(figsize=(11, 3.4 * n))
    gs = fig.add_gridspec(n, 2, hspace=0.45, wspace=0.28)
    fig.suptitle(f"{sub} -- autoreject config comparison (REAL/SCR condition)\n"
                 f"variance scale shared; gray dashed = raw PSD reference", fontsize=12, y=0.997)

    for row, (label, ep, st) in enumerate(results):
        # col 0: variance topomap
        ax = fig.add_subplot(gs[row, 0])
        vals = channel_var_topo(ep, POST_WIN)
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="viridis",
                                     vlim=(0, vmax_var), contours=4)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="uV^2")
        sub_t = f"N={len(ep)}"
        if "pct_drop_real" in st:
            sub_t += f"  drop={st['pct_drop_real']}%  flagch/ep={st['mean_flagch_real']}"
        ax.set_title(f"{label}\n{sub_t}", fontsize=9)

        # col 1: PSD
        ax = fig.add_subplot(gs[row, 1])
        f, m, sem = _psd_db_mean_sem(ep)
        if f0 is not None:
            ax.plot(f0, m0, color="0.6", lw=1.0, ls="--", label="raw ref")
        if f is not None:
            ax.fill_between(f, m - sem, m + sem, color="C3", alpha=0.2, lw=0)
            ax.plot(f, m, color="C3", lw=1.6, label=label)
            band_txt = " ".join(f"{b}:{m[(f>=lo)&(f<hi)].mean():.1f}" for b, (lo, hi) in BANDS.items()
                                 if ((f >= lo) & (f < hi)).any())
            ax.text(0.02, 0.03, band_txt, transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.set_xscale("log"); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
        ax.set_title(f"PSD ({PSD_FMIN:g}-{PSD_FMAX:g}Hz)", fontsize=9)
        ax.legend(fontsize=7)

    out = FIG_DIR / f"tune_compare_{sub}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}", flush=True)


def main() -> None:
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="*", default=["23", "24"])
    args = p.parse_args()
    subs = [_norm_subject(s) for s in args.subjects]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"autoreject_tune_scr :: config comparison  subjects={subs}")
    print("=" * 78, flush=True)
    t_all = time.time()
    rows = []
    for sub in subs:
        rows.extend(run_subject(sub))
    if rows:
        df = pd.DataFrame(rows)
        csv = FIG_DIR / "autoreject_tune_summary.csv"
        df.to_csv(csv, index=False)
        print(f"\nSummary -> {csv}", flush=True)
        print(df.to_string(index=False), flush=True)
    print(f"\nTOTAL wall time: {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
