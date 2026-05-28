"""Recover the SCR-RO cluster test from SAVED delta-CSD evokeds (no CSD recompute).

The main script scr_ro_delta_csd.py ran the permutation cluster test successfully but
crashed parsing cluster indices (it assumed slice objects; this MNE version returns index
arrays). The per-subject evokeds were saved, so this reloads them, recomputes the ROI-mean
real-silent difference, re-runs the cluster test with robust parsing, prints all clusters +
min p, writes the clusters CSV, and redraws grandavg_roi.png with significance shading.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._csd_cluster_recover
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test

from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR, OUT, SUBJECTS

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "scr_ro_delta_csd"
ROI = ["Fz", "FCz", "Cz", "CP1", "CP2", "Pz", "C3", "C4"]
RNG_SEED = 20260513


def main() -> int:
    ev_real, ev_silent = {}, {}
    for sub in SUBJECTS:
        fr = NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_real-ave.fif"
        fs = NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_silent-ave.fif"
        if not (fr.exists() and fs.exists()):
            print(f"  {sub}: missing evoked -> skip")
            continue
        ev_real[sub] = mne.read_evokeds(fr, verbose="ERROR")[0]
        ev_silent[sub] = mne.read_evokeds(fs, verbose="ERROR")[0]
    subs = list(ev_real.keys())
    print(f"Loaded {len(subs)} subjects: {subs}")

    common = set(ev_real[subs[0]].ch_names)
    for s in subs:
        common &= set(ev_real[s].ch_names)
    roi = [c for c in ROI if c in common]
    times = ev_real[subs[0]].times
    print(f"ROI present: {roi}")

    def roi_mean(ev):
        idx = [ev.ch_names.index(c) for c in roi]
        return ev.data[idx].mean(axis=0)

    diffs = np.array([roi_mean(ev_real[s]) - roi_mean(ev_silent[s]) for s in subs])
    real_roi = np.array([roi_mean(ev_real[s]) for s in subs])
    silent_roi = np.array([roi_mean(ev_silent[s]) for s in subs])

    n_perm = min(2 ** len(subs), 2048)
    T_obs, clusters, cl_pv, _ = permutation_cluster_1samp_test(
        diffs, n_permutations=n_perm, tail=0, seed=RNG_SEED, verbose="ERROR")

    sig_mask = np.zeros(len(times), bool)
    rows = []
    for cl, p in zip(clusters, cl_pv):
        idx = cl[0] if isinstance(cl, tuple) else cl
        m = np.zeros(len(times), bool)
        m[idx] = True  # works for index-array or boolean-mask
        tt = times[m]
        rows.append(dict(t_start=float(tt.min()), t_end=float(tt.max()),
                         n_samples=int(m.sum()), p_value=float(p)))
        if p < 0.05:
            sig_mask |= m
    rows.sort(key=lambda r: r["p_value"])
    df = pd.DataFrame(rows)
    df.to_csv(NPZ_DIR / "scr_ro_delta_csd_clusters.csv", index=False)

    minp = min((r["p_value"] for r in rows), default=float("nan"))
    print(f"\n=== CLUSTER TEST ({n_perm} perms, N={len(subs)}) ===")
    print(f"clusters found: {len(rows)}   min p = {minp:.4f}")
    if rows:
        print(df.to_string(index=False))
    print(f"significant (p<0.05): {'YES' if sig_mask.any() else 'NO'}")

    # redraw grandavg with sig shading
    fig, ax = plt.subplots(figsize=(11, 5))
    for arr, col, lab in [(real_roi, "C3", "real SCR"), (silent_roi, "0.5", "silent")]:
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(subs))
        ax.plot(times, m, color=col, lw=2, label=lab)
        ax.fill_between(times, m - sem, m + sem, color=col, alpha=0.2)
    if sig_mask.any():
        ax.fill_between(times, *ax.get_ylim(), where=sig_mask, color="orange", alpha=0.25,
                        label="cluster p<0.05")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("time from SCR onset (s)"); ax.set_ylabel("CSD (delta) ROI mean")
    ax.set_title(f"SCR-RO: delta+CSD central ROI (N={len(subs)})  min p={minp:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "grandavg_roi.png", dpi=130); plt.close(fig)
    print(f"\nRedrawn -> {FIG_DIR / 'grandavg_roi.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
