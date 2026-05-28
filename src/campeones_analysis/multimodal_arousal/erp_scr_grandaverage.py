"""Grand-average ERP_SCR across the 3-subject cohort + cluster-based permutation test.

Loads the per-subject evokeds produced by `erp_scr.py` and:
  1. Computes the grand average for `real` and `random` conditions
  2. Computes the per-subject difference (real - random) and its grand average
  3. Runs a 1-sample spatio-temporal cluster permutation test on the 3 difference
     waves (sign-flip permutations). Caveat: with N=3 subjects the test has only
     2^3 = 8 possible sign-flips, so minimum cluster p ~= 0.125. The shape and
     extent of clusters is more informative than significance at p<0.05 here.
  4. Plots:
       - Top row: grand-average real vs random at Fz / Cz / Pz / Oz-proxy + per-subject
         curves overlaid as thin lines (lets you SEE inter-subject variability)
       - Middle row: difference wave (real - random) at the same channels with cluster
         highlights (any time-channel cluster found by the permutation test)
       - Bottom row: topomaps of the difference at four time points spanning the window

Inputs (from erp_scr.py outputs):
  research_diary/context/05_02/y_candidates/<sub>_evoked_scr_real-ave.fif    x3
  research_diary/context/05_02/y_candidates/<sub>_evoked_scr_random-ave.fif  x3

Outputs:
  research_diary/context/05_02/figures/Y3_erp_scr_grandaverage.png
  research_diary/context/05_02/y_candidates/erp_scr_grandaverage_clusters.csv

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.erp_scr_grandaverage
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
from mne.stats import spatio_temporal_cluster_1samp_test

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
from src.campeones_analysis.multimodal_arousal.cohort import (  # noqa: E402
    COHORT as SUBJECTS,
    NPZ_DIR,
    OUT,
    SUBJ_COLORS,
)

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 4-ROI groupings (32 EEG channels split anteriorly-posteriorly + lateral chain).
# Total: 10 + 10 + 8 + 4 = 32.
ROIS: dict[str, list[str]] = {
    "Frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC1", "FCz", "FC2"],
    "Temporal":  ["T7", "T8", "FT9", "FT10", "TP9", "TP10", "FC5", "FC6", "CP5", "CP6"],
    "Parietal":  ["C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4"],
    "Occipital": ["O1", "O2", "P7", "P8"],
}
ROI_COLOR = {"Frontal": "C3", "Temporal": "C2", "Parietal": "C0", "Occipital": "C4"}

# Cluster test parameters
CLUSTER_THRESH = None  # let MNE pick via t-distribution; pass float to fix
CLUSTER_TAIL = 0
N_PERMUTATIONS = 2 ** len(SUBJECTS)  # 8 -- max for N=3 sign-flips


def load_subject_evokeds() -> tuple[list[mne.Evoked], list[mne.Evoked]]:
    real, rand = [], []
    for sub in SUBJECTS:
        r = mne.read_evokeds(NPZ_DIR / f"{sub}_evoked_scr_real-ave.fif", verbose="ERROR")[0]
        n = mne.read_evokeds(NPZ_DIR / f"{sub}_evoked_scr_random-ave.fif", verbose="ERROR")[0]
        real.append(r)
        rand.append(n)
    return real, rand


def align_channels(*evoked_lists: list[mne.Evoked]) -> list[str]:
    """Intersect channel names across all evoked objects and pick that set on each."""
    common = None
    for evs in evoked_lists:
        for ev in evs:
            s = set(ev.ch_names)
            common = s if common is None else (common & s)
    common_sorted = sorted(common)
    for evs in evoked_lists:
        for ev in evs:
            ev.pick(common_sorted)
    return common_sorted


def per_subject_diff(real: list[mne.Evoked], rand: list[mne.Evoked]) -> list[mne.Evoked]:
    return [mne.combine_evoked([r, n], weights=[1, -1]) for r, n in zip(real, rand)]


def stack_for_cluster(diff_evokeds: list[mne.Evoked]) -> np.ndarray:
    # mne expects (n_observations, n_times, n_channels) for spatio-temporal
    X = np.array([d.data.T for d in diff_evokeds])  # (n_subj, n_times, n_chans)
    return X


def run_cluster_test(X: np.ndarray, info: mne.Info) -> dict:
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    t_obs, clusters, p_values, H0 = spatio_temporal_cluster_1samp_test(
        X, threshold=CLUSTER_THRESH, tail=CLUSTER_TAIL,
        adjacency=adjacency, n_permutations=N_PERMUTATIONS, out_type="indices", verbose="ERROR",
    )
    return {"t_obs": t_obs, "clusters": clusters, "p_values": p_values, "H0": H0}


def roi_indices(ch_names: list[str]) -> dict[str, list[int]]:
    """Map each ROI name to the list of channel indices (within ch_names) that belong to it.

    Channels in the ROI definition that aren't in `ch_names` are silently skipped.
    """
    out: dict[str, list[int]] = {}
    for roi, chs in ROIS.items():
        out[roi] = [ch_names.index(c) for c in chs if c in ch_names]
    return out


def evoked_roi_mean(evoked: mne.Evoked, idxs: list[int]) -> np.ndarray:
    """Mean across an ROI's channels for an evoked. Returns shape (n_times,) in volts."""
    if not idxs:
        return np.full(len(evoked.times), np.nan)
    return evoked.data[idxs].mean(axis=0)


def plot_grandaverage(real: list[mne.Evoked], rand: list[mne.Evoked], diff_subj: list[mne.Evoked],
                       ga_real: mne.Evoked, ga_random: mne.Evoked, ga_diff: mne.Evoked,
                       cluster_result: dict, out_png: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = ga_real.info
    times = ga_real.times
    times_ms = times * 1000.0
    ch_names = ga_real.ch_names

    roi_idx = roi_indices(ch_names)
    roi_names = list(ROIS.keys())

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.0])
    fig.suptitle(
        f"Grand average ERP_SCR by ROI  (N_subjects={len(SUBJECTS)}, "
        f"window={ga_real.times[0]:.1f}..{ga_real.times[-1]:.1f}s, "
        f"baseline applied per subject, cluster test n_perm={N_PERMUTATIONS} -> min p={1/N_PERMUTATIONS:.3f})",
        fontsize=11,
    )

    # build channel-level mask of any cluster passing a generous threshold
    t_obs = cluster_result["t_obs"]
    clusters = cluster_result["clusters"]
    p_values = cluster_result["p_values"]
    mask_time_ch = np.zeros_like(t_obs, dtype=bool)
    for cl, p in zip(clusters, p_values):
        if p <= 0.20:
            t_arr, ch_arr = cl
            mask_time_ch[t_arr, ch_arr] = True

    # --- ROW 0: ROI-averaged real vs random + per-subject thin lines ---
    for col, roi in enumerate(roi_names):
        ax = fig.add_subplot(gs[0, col])
        idxs = roi_idx[roi]
        n_ch_in = len(idxs)
        if n_ch_in == 0:
            ax.text(0.5, 0.5, f"{roi}: no channels", ha="center", va="center", transform=ax.transAxes)
            continue
        ga_r_data = evoked_roi_mean(ga_real, idxs) * 1e6
        ga_n_data = evoked_roi_mean(ga_random, idxs) * 1e6
        ax.plot(times_ms, ga_r_data, color="C3", lw=1.8, label="grand-avg real")
        ax.plot(times_ms, ga_n_data, color="0.4", lw=1.4, ls="--", label="grand-avg random")
        for ev, sub in zip(real, SUBJECTS):
            ax.plot(times_ms, evoked_roi_mean(ev, idxs) * 1e6, color=SUBJ_COLORS[sub], lw=0.7, alpha=0.55, label=f"{sub} real")
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from SCR onset (ms)")
        ax.set_ylabel("uV")
        ax.set_title(f"{roi} ROI (n={n_ch_in} ch)")
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85, loc="lower left")

    # --- ROW 1: ROI-averaged difference + cluster highlight + per-subject diffs ---
    for col, roi in enumerate(roi_names):
        ax = fig.add_subplot(gs[1, col])
        idxs = roi_idx[roi]
        if not idxs:
            continue
        ga_d = evoked_roi_mean(ga_diff, idxs) * 1e6
        ax.plot(times_ms, ga_d, color="C4", lw=1.8, label="grand-avg (real - random)")
        for d, sub in zip(diff_subj, SUBJECTS):
            ax.plot(times_ms, evoked_roi_mean(d, idxs) * 1e6, color=SUBJ_COLORS[sub], lw=0.7, alpha=0.55, label=f"{sub} diff")
        # cluster highlight: time-bins where ANY channel within this ROI is in a kept cluster
        sig = mask_time_ch[:, idxs].any(axis=1)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        if sig.any():
            ylim = ax.get_ylim()
            ax.fill_between(times_ms, ylim[0], ylim[1], where=sig,
                            color="yellow", alpha=0.18, zorder=0, label="cluster (p <= 0.20)")
            ax.set_ylim(ylim)
        ax.set_xlabel("time from SCR onset (ms)")
        ax.set_ylabel("uV")
        ax.set_title(f"diff: {roi}")
        if col == 0:
            ax.legend(fontsize=7, framealpha=0.85, loc="lower left")

    # --- ROW 2: topomaps of difference at 4 representative time points ---
    # pick 4 time points: 2 in pre, 2 in post
    pre_pts = [-3.0, -1.0]
    post_pts = [0.5, 2.0]
    topo_times = pre_pts + post_pts
    for col, tt in enumerate(topo_times):
        ax = fig.add_subplot(gs[2, col])
        try:
            ga_diff.plot_topomap(times=tt, axes=ax, colorbar=False, show=False, time_format="%.2f s")
        except Exception as e:
            ax.text(0.5, 0.5, f"topomap fail @ {tt}s\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"diff @ {tt:+.2f} s")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # --- summary CSV 1: per cluster, p, time span, channels involved ---
    rows = []
    for cl_idx, (cl, p) in enumerate(zip(clusters, p_values)):
        t_arr, ch_arr = cl
        t_start, t_end = times[t_arr.min()], times[t_arr.max()]
        unique_chs = sorted({ch_names[i] for i in np.unique(ch_arr)})
        rows.append(dict(
            cluster=cl_idx,
            p_value=float(p),
            t_start_s=float(t_start),
            t_end_s=float(t_end),
            n_timepoints=int(len(np.unique(t_arr))),
            n_channels=int(len(unique_chs)),
            channels=";".join(unique_chs),
            mean_t_stat=float(t_obs[t_arr, ch_arr].mean()),
        ))

    # --- summary CSV 2: per-subject per-ROI peak amplitudes in pre / post windows ---
    roi_rows = []
    pre_mask = (times >= -3.0) & (times <= 0.0)
    post_mask = (times >= 0.0) & (times <= 3.0)
    for roi, idxs in roi_idx.items():
        if not idxs:
            continue
        for ev_r, ev_n, sub in zip(real, rand, SUBJECTS):
            r = evoked_roi_mean(ev_r, idxs) * 1e6
            n = evoked_roi_mean(ev_n, idxs) * 1e6
            d = r - n
            roi_rows.append(dict(
                subject=sub, roi=roi, n_channels_in_roi=len(idxs),
                pre_peak_abs_uV_real=float(np.max(np.abs(r[pre_mask]))),
                pre_peak_abs_uV_random=float(np.max(np.abs(n[pre_mask]))),
                post_peak_abs_uV_real=float(np.max(np.abs(r[post_mask]))),
                post_peak_abs_uV_random=float(np.max(np.abs(n[post_mask]))),
                pre_diff_peak_uV=float(d[pre_mask][np.argmax(np.abs(d[pre_mask]))]),
                post_diff_peak_uV=float(d[post_mask][np.argmax(np.abs(d[post_mask]))]),
            ))
    return pd.DataFrame(rows), pd.DataFrame(roi_rows)


def main() -> None:
    print("=" * 78)
    print("erp_scr_grandaverage")
    print("=" * 78)

    real, rand = load_subject_evokeds()
    common_chs = align_channels(real, rand)
    print(f"Common channels across {len(SUBJECTS)} subjects: {len(common_chs)} -> {common_chs}")

    diff_subj = per_subject_diff(real, rand)
    ga_real = mne.grand_average(real)
    ga_random = mne.grand_average(rand)
    ga_diff = mne.grand_average(diff_subj)

    # cluster test
    X = stack_for_cluster(diff_subj)
    print(f"Stacked diff shape (n_subj, n_times, n_chans): {X.shape}")
    cluster_result = run_cluster_test(X, diff_subj[0].info)
    p_values = cluster_result["p_values"]
    print(f"Cluster test: {len(p_values)} clusters found")
    if len(p_values):
        print(f"  cluster p-values (sorted): {sorted(p_values)[:10]}")
        print(f"  N permutations = {N_PERMUTATIONS}  ->  minimum achievable p = {1/N_PERMUTATIONS:.3f}")

    out_png = FIG_DIR / "Y3_erp_scr_grandaverage.png"
    df_clusters, df_roi = plot_grandaverage(real, rand, diff_subj, ga_real, ga_random, ga_diff, cluster_result, out_png)
    out_csv_cl = NPZ_DIR / "erp_scr_grandaverage_clusters.csv"
    out_csv_roi = NPZ_DIR / "erp_scr_grandaverage_roi_amplitudes.csv"
    df_clusters.to_csv(out_csv_cl, index=False)
    df_roi.to_csv(out_csv_roi, index=False)

    print(f"\nFigure -> {out_png}")
    print(f"Cluster CSV -> {out_csv_cl} ({len(df_clusters)} rows)")
    print(f"ROI amplitudes CSV -> {out_csv_roi} ({len(df_roi)} rows)")


if __name__ == "__main__":
    main()
