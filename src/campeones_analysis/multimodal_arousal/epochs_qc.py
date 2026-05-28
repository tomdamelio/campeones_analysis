"""Epoch-level QC = drop-only rejection (reject-not-repair) for the SCR epochs.

Decision (2026-05-27, after the autoreject synthesis + lit-review): the local AutoReject
default interpolated ~40% of channels per epoch (indefensible vs PREP 25% / MADE 10%, and the
contamination is a uniform/global broadband inflation, not a surgical channel issue). The
defensible policy is REJECT-NOT-REPAIR: a global peak-to-peak threshold (autoreject's
`get_rejection_threshold`) drops the worst epochs WITHOUT interpolating -> no fabricated
channels, rank preserved.

Unbiased comparison: fit ONE threshold on the COMBINED real+silent epochs, then split back by
condition. Session gating per lit-review: MADE (>10% globally bad channels -> exclude) read
from the preproc log, plus a minimum-epochs-per-condition floor.

Reusable: `apply_drop_only` is importable so build_subject_epochs can call it opt-in later
(Phase 2) and the whole SCR suite inherits a consistent rejection (re-run to cohort6_ar/).

NOTE the SCR-vs-silent contrast itself is artifact-dominated (CSD focal tests null for delta +
alpha); this QC does not change that conclusion. It establishes the project's rejection policy
for future analyses with real signal.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epochs_qc
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epochs_qc --subjects 27 31
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs, compute_psd
from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR, OUT, PREP
from src.campeones_analysis.multimodal_arousal.epoch_qa_viz_scr import BANDS, POST_WIN, channel_var_topo
from src.campeones_analysis.multimodal_arousal.autoreject_scr import combine_real_silent, _norm_subject

import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "epochs_qc"
N_CHAN = 32
MADE_FRAC = 0.10   # >10% globally bad channels -> exclude session (MADE)
PREP_FRAC = 0.25   # >25% -> flagged dubious (PREP)
PREP_LOG = PREP / "logs_preprocessing_details_all_subjects_eeg.json"


def apply_drop_only(real_ep: mne.Epochs, silent_ep: mne.Epochs, random_state: int = 97):
    """Drop-only rejection with a single global ptp threshold fit on combined real+silent.

    Returns (real_clean, silent_clean, thresh_volts).
    """
    from autoreject import get_rejection_threshold

    combined = combine_real_silent(real_ep, silent_ep)
    thresh = get_rejection_threshold(combined, ch_types="eeg", random_state=random_state,
                                     verbose=False)
    clean = combined.copy().drop_bad(reject=thresh, verbose="ERROR")
    return clean["real"], clean["silent"], thresh


def made_bad_channels(sub: str) -> set[str]:
    """Channels consistently bad in the preproc (>=50% of the subject's runs).

    'Globally bad' (MADE/PREP) means consistently bad across the recording, NOT the union
    across runs (a channel bad in 1 of ~7 runs is not a globally bad channel; the union
    overcounts and would flag every subject). Returns the consistent-bad set; empty if the
    log/subject is missing.
    """
    if not PREP_LOG.exists():
        return set()
    try:
        with open(PREP_LOG, encoding="utf-8") as fh:
            log = json.load(fh)
    except Exception:
        return set()
    num = sub.replace("sub-", "")
    ch_run_count: dict[str, int] = {}
    n_runs = 0
    for ses, runs in log.get(num, {}).items():
        if not isinstance(runs, dict):
            continue
        for run, info in runs.items():
            if isinstance(info, dict) and "bad_channels" in info:
                n_runs += 1
                for c in info.get("bad_channels", []) or []:
                    ch_run_count[c] = ch_run_count.get(c, 0) + 1
    if n_runs == 0:
        return set()
    return {c for c, k in ch_run_count.items() if k >= n_runs / 2}


def before_after_figure(sub: str, real_before, real_after, silent_before, silent_after,
                        out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = real_before.info
    var_all = []
    for ep in (real_before, real_after):
        if len(ep):
            var_all.append(channel_var_topo(ep, POST_WIN))
    vmax = float(np.percentile(np.concatenate(var_all), 98)) if var_all else 1.0

    def psd_db(ep):
        if not len(ep):
            return None, None
        p, f, _ = compute_psd(ep)
        return f, 10.0 * np.log10(p.mean(axis=1).mean(axis=0) + 1e-30)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    fig.suptitle(f"{sub} -- drop-only QC before/after (REAL/SCR). var scale shared.", fontsize=12)

    # PSD real before vs after
    ax = fig.add_subplot(gs[0, 0])
    for ep, c, l in [(real_before, "0.5", "before"), (real_after, "C3", "after")]:
        f, m = psd_db(ep)
        if f is not None:
            ax.plot(f, m, color=c, lw=1.6, label=f"{l} (N={len(ep)})")
    ax.set_xscale("log"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_title("real PSD"); ax.legend(fontsize=8)

    # variance topomaps real before / after
    for col, (ep, name) in enumerate([(real_before, "real before"), (real_after, "real after")], start=1):
        ax = fig.add_subplot(gs[0, col])
        if len(ep):
            im, _ = mne.viz.plot_topomap(channel_var_topo(ep, POST_WIN), info, axes=ax, show=False,
                                         cmap="viridis", vlim=(0, vmax), contours=4)
            fig.colorbar(im, ax=ax, shrink=0.7, label="uV^2")
        ax.set_title(f"{name} var [{POST_WIN[0]:g}-{POST_WIN[1]:g}s]", fontsize=9)

    # silent PSD + silent var before/after (row 1)
    ax = fig.add_subplot(gs[1, 0])
    for ep, c, l in [(silent_before, "0.5", "before"), (silent_after, "C0", "after")]:
        f, m = psd_db(ep)
        if f is not None:
            ax.plot(f, m, color=c, lw=1.6, label=f"{l} (N={len(ep)})")
    ax.set_xscale("log"); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_title("silent PSD"); ax.legend(fontsize=8)
    for col, (ep, name) in enumerate([(silent_before, "silent before"), (silent_after, "silent after")], start=1):
        ax = fig.add_subplot(gs[1, col])
        if len(ep):
            im, _ = mne.viz.plot_topomap(channel_var_topo(ep, POST_WIN), info, axes=ax, show=False,
                                         cmap="viridis", vlim=(0, vmax), contours=4)
            fig.colorbar(im, ax=ax, shrink=0.7, label="uV^2")
        ax.set_title(f"{name}", fontsize=9)

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_subject(sub: str, min_epochs: int, random_state: int) -> dict | None:
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        print(f"  {sub}: no epochs -> skip", flush=True)
        return None
    nr_in, ns_in = len(real_ep), len(silent_ep)
    real_after, silent_after, thresh = apply_drop_only(real_ep, silent_ep, random_state)
    nr_k, ns_k = len(real_after), len(silent_after)
    thr_uV = float(thresh.get("eeg", np.nan)) * 1e6

    bad = made_bad_channels(sub)
    n_bad = len(bad)
    pct_drop_real = round(100 * (1 - nr_k / nr_in), 2) if nr_in else 0.0
    pct_drop_silent = round(100 * (1 - ns_k / ns_in), 2) if ns_in else 0.0
    min_kept = min(nr_k, ns_k)
    row = dict(
        subject=sub, thresh_uV=round(thr_uV, 1),
        n_real_in=nr_in, n_real_kept=nr_k, n_silent_in=ns_in, n_silent_kept=ns_k,
        pct_drop_real=pct_drop_real, pct_drop_silent=pct_drop_silent,
        asym_real_minus_silent=round(pct_drop_real - pct_drop_silent, 2),
        min_kept=min_kept, n_bad_channels=n_bad, bad_channels=",".join(sorted(bad)),
        gate_made_excl=bool(n_bad > MADE_FRAC * N_CHAN),
        gate_prep_flag=bool(n_bad > PREP_FRAC * N_CHAN),
        gate_minep_fail=bool(min_kept < min_epochs),
    )
    print(f"  {sub}: thr={thr_uV:.0f}uV  drop real {pct_drop_real}% silent {pct_drop_silent}% "
          f"(asym {row['asym_real_minus_silent']:+})  min_kept={min_kept}  "
          f"bad_ch={n_bad}{' MADE-EXCL' if row['gate_made_excl'] else ''}"
          f"{' MINEP-FAIL' if row['gate_minep_fail'] else ''}", flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # persist kept-epoch info (reusable). store kept event sample indices per condition.
    np.savez_compressed(
        FIG_DIR / f"{sub}_dropmask.npz",
        thresh_eeg_V=float(thresh.get("eeg", np.nan)),
        real_kept_samples=real_after.events[:, 0], silent_kept_samples=silent_after.events[:, 0],
        n_real_in=nr_in, n_silent_in=ns_in,
    )
    try:
        before_after_figure(sub, real_ep, real_after, silent_ep, silent_after,
                            FIG_DIR / f"before_after_{sub}.png")
    except Exception as e:
        print(f"    before/after figure failed: {e}", flush=True)
    return row


def main() -> None:
    import pandas as pd
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT

    p = argparse.ArgumentParser(description="drop-only epoch QC for SCR epochs.")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--random-state", type=int, default=97)
    args = p.parse_args()
    subs = [_norm_subject(s) for s in args.subjects] if args.subjects else list(COHORT)

    print("=" * 78)
    print(f"epochs_qc :: drop-only (get_rejection_threshold)  subjects={subs}  "
          f"min_epochs={args.min_epochs}")
    print("=" * 78, flush=True)
    rows = []
    for sub in subs:
        r = run_subject(sub, args.min_epochs, args.random_state)
        if r is not None:
            rows.append(r)
    if rows:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        csv = FIG_DIR / "epochs_qc_droponly_summary.csv"
        df.to_csv(csv, index=False)
        print(f"\nSummary -> {csv}", flush=True)
        print(df.drop(columns=["bad_channels"]).to_string(index=False), flush=True)
        print(f"\nCaveat: drop_real > drop_silent in {(df['asym_real_minus_silent'] > 0).sum()}/{len(df)} "
              f"subjects -> rejection preferentially removes the (more contaminated) high-arousal "
              f"real epochs; do not attribute downstream changes to the rejection.", flush=True)


if __name__ == "__main__":
    main()
