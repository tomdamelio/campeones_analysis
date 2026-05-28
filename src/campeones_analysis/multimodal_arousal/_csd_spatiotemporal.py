"""All-channel spatiotemporal cluster test on saved delta-CSD evokeds.

Upgrades the central-ROI SCR-RO test to an omnibus: 'is there ANY spatially- and
temporally-contiguous focal cluster anywhere on the scalp in delta-CSD?' -- ONE corrected
p-value (channel adjacency from the montage), avoiding the multiple-comparison trap of
testing many ROIs separately. Reloads saved evokeds (no CSD recompute).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal._csd_spatiotemporal
"""

from __future__ import annotations

import warnings

import mne
import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test

from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR, SUBJECTS

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
RNG_SEED = 20260513


def main() -> int:
    ev_real, ev_silent = {}, {}
    for sub in SUBJECTS:
        fr = NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_real-ave.fif"
        fs = NPZ_DIR / f"scr_ro_delta_csd_{sub}_evoked_silent-ave.fif"
        if fr.exists() and fs.exists():
            ev_real[sub] = mne.read_evokeds(fr, verbose="ERROR")[0]
            ev_silent[sub] = mne.read_evokeds(fs, verbose="ERROR")[0]
    subs = list(ev_real.keys())
    print(f"Loaded {len(subs)} subjects: {subs}")

    common = [c for c in ev_real[subs[0]].ch_names
              if all(c in ev_real[s].ch_names for s in subs)]
    times = ev_real[subs[0]].times
    n_ch = len(common)
    print(f"All-channel test: {n_ch} channels, {len(times)} timepoints")

    # diff array: (n_subjects, n_times, n_channels)
    def pick(ev):
        idx = [ev.ch_names.index(c) for c in common]
        return ev.data[idx].T  # (n_times, n_channels)
    X = np.array([pick(ev_real[s]) - pick(ev_silent[s]) for s in subs])
    print(f"X shape (subj, time, chan) = {X.shape}")

    # channel adjacency depends only on sensor POSITIONS -> build a fresh eeg info with the
    # same montage (the csd ch_type blocks find_ch_adjacency / set_channel_types).
    sf = float(ev_real[subs[0]].info["sfreq"])
    info_eeg = mne.create_info(common, sfreq=sf, ch_types="eeg")
    info_eeg.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         match_case=False, on_missing="ignore", verbose="ERROR")
    adjacency, names = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    print(f"adjacency: {adjacency.shape}")

    n_perm = min(2 ** len(subs), 2048)
    T_obs, clusters, cl_pv, _ = spatio_temporal_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_perm, tail=0, seed=RNG_SEED,
        n_jobs=1, verbose="ERROR")
    print(f"\n=== ALL-CHANNEL SPATIOTEMPORAL CLUSTER ({n_perm} perms, N={len(subs)}) ===")
    if len(cl_pv) == 0:
        print("no clusters formed at all.")
    else:
        order = np.argsort(cl_pv)
        print(f"clusters: {len(cl_pv)}   min p = {cl_pv.min():.4f}")
        for k in order[:5]:
            tmask = clusters[k][0]  # time idx
            cmask = clusters[k][1]  # channel idx
            t_in = np.unique(tmask)
            ch_in = sorted({common[i] for i in np.unique(cmask)})
            print(f"  cluster p={cl_pv[k]:.4f}  t=[{times[t_in.min()]:.2f},{times[t_in.max()]:.2f}]s "
                  f"chans({len(ch_in)})={ch_in}")
        print(f"\nsignificant (p<0.05): {'YES' if (cl_pv < 0.05).any() else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
