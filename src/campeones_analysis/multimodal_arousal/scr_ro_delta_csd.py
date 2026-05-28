"""Branković SCR-RO test: delta (0.5-4 Hz) + CSD ERP locked to SCR onset, central-parietal ROI.

Question: is there a FOCAL central-delta deflection locked to SCR onset (Branković 2026 SCR-RO),
DISTINCT from the global broadband-gamma artifact found in the LOSO/Haufe analysis? CSD (surface
Laplacian) suppresses spatially-diffuse activity (global drift, EMG) and enhances focal cortical
generators; restricting to delta + a central-parietal ROI targets the SCR-RO directly.

Pipeline per subject (cohort6, relaxed criterion + non-overlap, reusing erp_scr helpers):
  preproc EEG -> pick eeg + montage -> filter 0.5-4 Hz (delta) -> resample 250 -> CSD
  -> epochs locked to SCR onsets (real) and EDA-silent controls -> baseline -> average.
Cross-subject: grand-average real vs silent at the central ROI + temporal cluster-permutation
(1-samp on the real-silent difference, N=6 -> 64 sign-flips), and difference topomaps.

Outputs (under cohort6/):
  y_candidates/scr_ro_delta_csd_<sub>_evoked_{real,silent}-ave.fif, scr_ro_delta_csd_clusters.csv
  figures/scr_ro_delta_csd/{grandavg_roi,diff_topomaps,per_subject_roi}.png

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_ro_delta_csd
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
from mne.stats import permutation_cluster_1samp_test

from src.campeones_analysis.multimodal_arousal.erp_scr import (
    BASELINE,
    EDA_FS,
    NPZ_DIR,
    OUT,
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

FIG_DIR = OUT / "figures" / "scr_ro_delta_csd"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260513
DELTA = (0.5, 4.0)
SR = 250.0
ROI = ["Fz", "FCz", "Cz", "CP1", "CP2", "Pz", "C3", "C4"]  # central-parietal (SCR-RO target)


def build_delta_csd_evokeds(sub: str, rng: np.random.Generator):
    """Return (evoked_real, evoked_silent) for one subject in delta-CSD space, or None."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    real_eps, silent_eps = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(DELTA[0], DELTA[1], verbose="ERROR")
            raw.resample(SR, verbose="ERROR")
            dur = float(raw.times[-1])
            phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = detect_scr_onsets_s(phasic, EDA_FS)
            onsets = onsets[onsets < dur]
            onsets = filter_clean_onsets(onsets, phasic, EDA_FS)  # all SCRs, non-overlapping
            silent = sample_silent_controls(len(onsets), dur, phasic, EDA_FS, rng,
                                            avoid_onsets_s=onsets)
            # epoch in sensor (eeg) space (epoch_one_run uses picks="eeg"), then CSD the epochs
            er = epoch_one_run(raw, onsets, code=1, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            es = epoch_one_run(raw, silent, code=2, tmin=TMIN, tmax=TMAX, baseline=BASELINE)
            if er is not None and len(er):
                real_eps.append(compute_current_source_density(er))  # surface Laplacian
            if es is not None and len(es):
                silent_eps.append(compute_current_source_density(es))
        except Exception as exc:
            print(f"    {label}: skip ({exc})")
            continue
    if not real_eps or not silent_eps:
        return None
    real = mne.concatenate_epochs(real_eps, verbose="ERROR")
    silent = mne.concatenate_epochs(silent_eps, verbose="ERROR")
    return real.average(), silent.average(), len(real), len(silent)


def main() -> int:
    print("=" * 78)
    print("SCR-RO test: delta (0.5-4 Hz) + CSD ERP locked to SCR onset, central-parietal ROI")
    print("=" * 78)
    rng = np.random.default_rng(RNG_SEED)
    ev_real, ev_silent, counts = {}, {}, {}
    for sub in SUBJECTS:
        print(f"  {sub} ...", flush=True)
        out = build_delta_csd_evokeds(sub, rng)
        if out is None:
            print(f"    no evokeds -> skip")
            continue
        er, es, nr, ns = out
        er.save(NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_real-ave.fif", overwrite=True)
        es.save(NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_silent-ave.fif", overwrite=True)
        ev_real[sub], ev_silent[sub], counts[sub] = er, es, (nr, ns)
        print(f"    real={nr} silent={ns}")
    subs = list(ev_real.keys())
    if len(subs) < 3:
        print("[ERROR] need >=3 subjects")
        return 1

    # common channels + ROI
    common = set(ev_real[subs[0]].ch_names)
    for s in subs:
        common &= set(ev_real[s].ch_names)
    common = [c for c in ev_real[subs[0]].ch_names if c in common]
    roi = [c for c in ROI if c in common]
    times = ev_real[subs[0]].times
    print(f"\nCommon channels: {len(common)}  | ROI present: {roi}")

    # per-subject ROI-mean difference (real - silent)
    def roi_mean(ev):
        idx = [ev.ch_names.index(c) for c in roi]
        return ev.data[idx].mean(axis=0)  # (n_times,)

    diffs = np.array([roi_mean(ev_real[s]) - roi_mean(ev_silent[s]) for s in subs])  # (n_sub, n_times)
    real_roi = np.array([roi_mean(ev_real[s]) for s in subs])
    silent_roi = np.array([roi_mean(ev_silent[s]) for s in subs])

    # temporal cluster-permutation (1-samp on diff)
    sig_mask = np.zeros(len(times), bool)
    cluster_rows = []
    try:
        n_perm = min(2 ** len(subs), 2048)
        T_obs, clusters, cl_pv, _ = permutation_cluster_1samp_test(
            diffs, n_permutations=n_perm, tail=0, seed=RNG_SEED, verbose="ERROR")
        for cl, p in zip(clusters, cl_pv):
            idx = cl[0] if isinstance(cl, tuple) else cl
            m = np.zeros(len(times), bool)
            m[idx] = True  # robust: handles index-array OR boolean-mask (MNE version-dependent)
            tt = times[m]
            cluster_rows.append(dict(t_start=float(tt.min()), t_end=float(tt.max()), p_value=float(p)))
            if p < 0.05:
                sig_mask |= m
        mins = min([r["p_value"] for r in cluster_rows], default=np.nan)
        print(f"\nROI temporal cluster test: {len(cluster_rows)} clusters, min p={mins:.3f} "
              f"({n_perm} perms)")
    except Exception as exc:
        print(f"  cluster test failed: {exc}")
    pd.DataFrame(cluster_rows).to_csv(NPZ_DIR / "scr_ro_delta_csd_clusters.csv", index=False)

    # --- fig 1: grand-average ROI waveform real vs silent +/- SEM, sig shaded ---
    try:
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
        ax.set_title(f"SCR-RO test: delta+CSD central ROI {roi}  (N={len(subs)})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "grandavg_roi.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig1 failed: {exc}")

    # --- fig 2: difference (real-silent) topomaps in delta-CSD ---
    try:
        diff_evs = [mne.combine_evoked([ev_real[s], ev_silent[s]], weights=[1, -1]) for s in subs]
        ga = mne.grand_average(diff_evs)
        topo_times = [-2.0, -1.0, 0.0, 0.5, 1.0]
        fig = ga.plot_topomap(times=topo_times, ch_type="csd", show=False,
                              colorbar=True, time_unit="s")
        fig.suptitle("Grand-avg difference (real - silent), delta+CSD", y=1.02)
        fig.savefig(FIG_DIR / "diff_topomaps.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    except Exception as exc:
        print(f"  fig2 (topomaps) failed: {exc}")

    # --- fig 3: per-subject ROI diff (consistency) ---
    try:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, s in enumerate(subs):
            ax.plot(times, diffs[i], lw=1, alpha=0.8, label=s)
        ax.plot(times, diffs.mean(0), color="k", lw=2.5, label="mean")
        ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="k", ls=":", lw=1)
        if sig_mask.any():
            ax.fill_between(times, *ax.get_ylim(), where=sig_mask, color="orange", alpha=0.2)
        ax.set_xlabel("time from SCR onset (s)"); ax.set_ylabel("real - silent (CSD delta ROI)")
        ax.set_title("Per-subject SCR-RO difference (central ROI)")
        ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "per_subject_roi.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig3 failed: {exc}")

    print(f"\nFigures -> {FIG_DIR}")
    print(f"Counts: " + "  ".join(f"{s}:{counts[s][0]}/{counts[s][1]}" for s in subs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
