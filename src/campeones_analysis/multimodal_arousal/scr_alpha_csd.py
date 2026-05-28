"""CSD version of the parieto-occipital alpha desynchronization (ERD) test.

Parallel to scr_ro_delta_csd.py, but for ALPHA. Key methodological difference: the delta
SCR-RO is a phase-locked slow potential -> a time-domain ERP is the right readout. Alpha is
NOT phase-locked to SCR onset -> a time-domain ERP would cancel it. So the alpha test must be
on alpha POWER (Morlet TFR, % change vs baseline = ERD/ERS), exactly like alpha_hypothesis_scr,
but computed in CSD space (surface Laplacian suppresses the global component; if a FOCAL
parieto-occipital alpha ERD is masked by global activity, CSD should reveal it).

Hypothesis (from alpha_hypothesis_scr): real (SCR) shows MORE alpha desynchronization
(more negative % change) than silent, over parieto-occipital cortex, in [0,1] s post-onset.

Pipeline per subject (reuses build_subject_epochs + compute_alpha_tfr):
  build_subject_epochs -> CSD the epochs -> alpha Morlet TFR (8-13 Hz) -> % change baseline
  (-5,-4.5) -> per-channel alpha power time course (real, silent).
Cross-subject: parieto-occipital ROI cluster-permutation over time + all-channel
spatiotemporal cluster + WOI [0,1] s summary + grand-average figure + diff topomap.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_alpha_csd
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import compute_current_source_density
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test

from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR, OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs
from src.campeones_analysis.multimodal_arousal.alpha_hypothesis_scr import (
    ALPHA_FREQS,
    BASELINE,
    BASELINE_MODE,
    PARIETOOCCIPITAL,
    WIN_OF_INTEREST,
    compute_alpha_tfr,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "scr_alpha_csd"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260513
# Cluster window covers PRE-onset too: the SCR onset lags the central/autonomic event by
# ~1-3 s (phasic rise time), so a cortical correlate would be EXPECTED pre-onset. Start at
# -4.0 s (after the (-5,-4.5) baseline + margin) through +3.0 s. The cluster test corrects
# over time, so testing this broad theory-motivated window is principled (not window-picking).
CLUSTER_WIN = (-4.0, 3.0)
WOI_PRE = (-3.0, -1.0)  # pre-onset descriptive window (where a SCR-driving cortical signal would sit)


def subject_alpha_csd(sub: str):
    """Return (tfr_real_pct, tfr_silent_pct, n_real, n_silent) in CSD-alpha space, or None."""
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        return None
    real_csd = compute_current_source_density(real_ep)
    silent_csd = compute_current_source_density(silent_ep)
    tr = compute_alpha_tfr(real_csd).apply_baseline(BASELINE, mode=BASELINE_MODE)
    ts = compute_alpha_tfr(silent_csd).apply_baseline(BASELINE, mode=BASELINE_MODE)
    return tr, ts, len(real_ep), len(silent_ep)


def main() -> int:
    print("=" * 78)
    print("CSD alpha ERD test: parieto-occipital alpha power (% change) real vs silent")
    print("=" * 78)
    tr_all, ts_all, counts = {}, {}, {}
    for sub in SUBJECTS:
        print(f"  {sub} ...", flush=True)
        out = subject_alpha_csd(sub)
        if out is None:
            print("    no epochs -> skip")
            continue
        tr, ts, nr, ns = out
        tr_all[sub], ts_all[sub], counts[sub] = tr, ts, (nr, ns)
        print(f"    real={nr} silent={ns}")
    subs = list(tr_all.keys())
    if len(subs) < 3:
        print("[ERROR] need >=3 subjects")
        return 1

    common = [c for c in tr_all[subs[0]].ch_names
              if all(c in tr_all[s].ch_names for s in subs)]
    times = tr_all[subs[0]].times
    roi = [c for c in PARIETOOCCIPITAL if c in common]
    print(f"\nCommon channels: {len(common)}  | parieto-occipital ROI present: {roi}")

    # per-channel alpha %change time course (mean over alpha freqs)
    def chan_tc(tfr):
        idx = [tfr.ch_names.index(c) for c in common]
        return tfr.data[idx].mean(axis=1)  # (n_ch, n_times)
    def roi_tc(tfr):
        idx = [tfr.ch_names.index(c) for c in roi]
        return tfr.data[idx].mean(axis=(0, 1))  # (n_times,)

    real_roi = np.array([roi_tc(tr_all[s]) for s in subs])      # (n_sub, n_t)
    silent_roi = np.array([roi_tc(ts_all[s]) for s in subs])
    diffs_roi = real_roi - silent_roi

    # ---- WOI summaries (direction of effect): classical post [0,1] AND pre-onset [-3,-1] ----
    woi_post = (times >= WIN_OF_INTEREST[0]) & (times <= WIN_OF_INTEREST[1])
    woi_pre = (times >= WOI_PRE[0]) & (times <= WOI_PRE[1])
    rows = []
    for i, s in enumerate(subs):
        rp, sp = real_roi[i, woi_post].mean(), silent_roi[i, woi_post].mean()
        rpre, spre = real_roi[i, woi_pre].mean(), silent_roi[i, woi_pre].mean()
        rows.append(dict(subject=s, n_real=counts[s][0], n_silent=counts[s][1],
                         pre_real=round(float(rpre), 2), pre_silent=round(float(spre), 2),
                         pre_diff=round(float(rpre - spre), 2), pre_ERD=bool(rpre < spre),
                         post_real=round(float(rp), 2), post_silent=round(float(sp), 2),
                         post_diff=round(float(rp - sp), 2), post_ERD=bool(rp < sp)))
    df = pd.DataFrame(rows)
    df.to_csv(NPZ_DIR / "scr_alpha_csd_woi_summary.csv", index=False)
    print(f"\nWOI pre [{WOI_PRE[0]},{WOI_PRE[1]}]s: {int(df['pre_ERD'].sum())}/{len(subs)} ERD (real<silent)  | "
          f"WOI post [{WIN_OF_INTEREST[0]},{WIN_OF_INTEREST[1]}]s: {int(df['post_ERD'].sum())}/{len(subs)} ERD")
    print(df.to_string(index=False))

    # ---- ROI temporal cluster test (within CLUSTER_WIN) ----
    cwin = (times >= CLUSTER_WIN[0]) & (times <= CLUSTER_WIN[1])
    tt = times[cwin]
    n_perm = min(2 ** len(subs), 2048)
    sig_mask = np.zeros(len(tt), bool)
    T_obs, clusters, cl_pv, _ = permutation_cluster_1samp_test(
        diffs_roi[:, cwin], n_permutations=n_perm, tail=0, seed=RNG_SEED, verbose="ERROR")
    minp_roi = float(cl_pv.min()) if len(cl_pv) else float("nan")
    print(f"\nROI cluster test (parieto-occipital, {CLUSTER_WIN}s): {len(cl_pv)} clusters, "
          f"min p={minp_roi:.4f}  ({n_perm} perms)")
    for cl, p in zip(clusters, cl_pv):
        idx = cl[0] if isinstance(cl, tuple) else cl
        m = np.zeros(len(tt), bool); m[idx] = True
        if p < 0.05:
            sig_mask |= m
        print(f"    cluster t=[{tt[m].min():.2f},{tt[m].max():.2f}]s  p={p:.4f}")

    # ---- all-channel spatiotemporal cluster ----
    X = np.array([(chan_tc(tr_all[s]) - chan_tc(ts_all[s])).T[cwin] for s in subs])  # (n_sub, n_t_win, n_ch)
    sf = float(tr_all[subs[0]].info["sfreq"])
    info_eeg = mne.create_info(common, sfreq=sf, ch_types="eeg")
    info_eeg.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         match_case=False, on_missing="ignore", verbose="ERROR")
    adjacency, _ = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    To, cls, pv, _ = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_perm, tail=0, seed=RNG_SEED, n_jobs=1, verbose="ERROR")
    minp_all = float(pv.min()) if len(pv) else float("nan")
    print(f"\nAll-channel spatiotemporal cluster ({CLUSTER_WIN}s): {len(pv)} clusters, min p={minp_all:.4f}")
    if len(pv):
        for k in np.argsort(pv)[:5]:
            tmask, cmask = cls[k][0], cls[k][1]
            ch_in = sorted({common[i] for i in np.unique(cmask)})
            print(f"    p={pv[k]:.4f}  t=[{tt[np.unique(tmask).min()]:.2f},{tt[np.unique(tmask).max()]:.2f}]s "
                  f"chans({len(ch_in)})={ch_in}")
    print(f"\nSignificant (p<0.05): ROI={'YES' if sig_mask.any() else 'NO'}  "
          f"all-channel={'YES' if (len(pv) and (pv<0.05).any()) else 'NO'}")

    # ---- fig: grand-average ROI alpha power time course ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for arr, col, lab in [(real_roi, "C3", "real SCR"), (silent_roi, "0.5", "silent")]:
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(subs))
        ax.plot(times, m, color=col, lw=2, label=lab)
        ax.fill_between(times, m - sem, m + sem, color=col, alpha=0.2)
    if sig_mask.any():
        ax.fill_between(tt, *ax.get_ylim(), where=sig_mask, color="orange", alpha=0.25, label="cluster p<0.05")
    ax.axvspan(*WOI_PRE, color="dodgerblue", alpha=0.10, label=f"pre WOI [{WOI_PRE[0]:g},{WOI_PRE[1]:g}]s")
    ax.axvspan(*WIN_OF_INTEREST, color="yellow", alpha=0.12, label="post WOI [0,1]s")
    ax.axvline(0, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time from SCR onset (s)"); ax.set_ylabel("alpha power (% change vs baseline), CSD")
    ax.set_title(f"CSD alpha ERD: parieto-occipital ROI {roi}  (N={len(subs)})  "
                 f"ROI min p={minp_roi:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "grandavg_roi_alpha.png", dpi=130); plt.close(fig)
    print(f"\nFigure -> {FIG_DIR / 'grandavg_roi_alpha.png'}")
    print(f"Counts: " + "  ".join(f"{s}:{counts[s][0]}/{counts[s][1]}" for s in subs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
