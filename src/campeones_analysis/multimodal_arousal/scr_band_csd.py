"""CSD focal SCR test across bands (theta/beta/gamma) -- power-based, parametrized.

Generalizes scr_alpha_csd.py to arbitrary bands to close the question 'is there ANY focal
cortical SCR signal in any canonical band?' (delta SCR-RO and alpha ERD already null). Like
alpha, the readout is BAND POWER (Morlet, % change vs baseline) in CSD space (surface
Laplacian suppresses the global component, enhances focal). Primary test = all-channel
spatiotemporal cluster (no ROI -> no fishing); a generic central ROI is reported descriptively.
Window (-4, 3) s (pre-onset included: the SCR onset lags the central event ~1-3 s).

Caveat: post-CSD, beta/gamma are dominated by LOCAL high-spatial-frequency noise (EMG) which
CSD amplifies -> a 'hit' in beta/gamma at edge/temporal channels is suspect of artifact, not
cortical signal. Bands capped at 40 Hz (epochs are 1-40 Hz band-passed upstream).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_band_csd --bands theta beta gamma
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
from mne.preprocessing import compute_current_source_density
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test

from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR, OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs
from src.campeones_analysis.multimodal_arousal.alpha_hypothesis_scr import BASELINE, BASELINE_MODE
from src.campeones_analysis.multimodal_arousal.autoreject_scr import _norm_subject

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "scr_band_csd"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260513
CLUSTER_WIN = (-4.0, 3.0)
CENTRAL_ROI = ["Fz", "FCz", "Cz", "CP1", "CP2", "Pz", "C3", "C4"]  # descriptive only
TFR_DECIM = 4

# bands capped at 40 Hz (epochs are band-passed 1-40 Hz upstream)
BAND_FREQS = {
    "theta": np.arange(4.0, 8.5, 0.5),
    "beta": np.arange(13.0, 30.5, 1.0),
    "gamma": np.arange(30.0, 40.5, 1.0),
}


def subject_band_csd(sub: str, freqs: np.ndarray):
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        return None
    rc = compute_current_source_density(real_ep)
    sc = compute_current_source_density(silent_ep)
    n_cycles = freqs / 2.0
    tr = tfr_morlet(rc, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False,
                    decim=TFR_DECIM, n_jobs=1, average=True, verbose="ERROR").apply_baseline(
        BASELINE, mode=BASELINE_MODE)
    ts = tfr_morlet(sc, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False,
                    decim=TFR_DECIM, n_jobs=1, average=True, verbose="ERROR").apply_baseline(
        BASELINE, mode=BASELINE_MODE)
    return tr, ts, len(real_ep), len(silent_ep)


def run_band(band: str, freqs: np.ndarray, subs_arg) -> dict:
    print(f"\n{'='*70}\nBAND {band} ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)\n{'='*70}", flush=True)
    tr_all, ts_all = {}, {}
    for sub in subs_arg:
        out = subject_band_csd(sub, freqs)
        if out is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        tr, ts, nr, ns = out
        tr_all[sub], ts_all[sub] = tr, ts
        print(f"  {sub}: real={nr} silent={ns}", flush=True)
    subs = list(tr_all.keys())
    if len(subs) < 3:
        print(f"  [skip {band}] <3 subjects", flush=True)
        return dict(band=band, status="too_few_subjects")

    common = [c for c in tr_all[subs[0]].ch_names if all(c in tr_all[s].ch_names for s in subs)]
    times = tr_all[subs[0]].times
    roi = [c for c in CENTRAL_ROI if c in common]
    cwin = (times >= CLUSTER_WIN[0]) & (times <= CLUSTER_WIN[1])
    tt = times[cwin]
    n_perm = min(2 ** len(subs), 2048)

    def chan_tc(tfr):
        idx = [tfr.ch_names.index(c) for c in common]
        return tfr.data[idx].mean(axis=1)  # (n_ch, n_t)
    def roi_tc(tfr):
        idx = [tfr.ch_names.index(c) for c in roi]
        return tfr.data[idx].mean(axis=(0, 1))

    real_roi = np.array([roi_tc(tr_all[s]) for s in subs])
    silent_roi = np.array([roi_tc(ts_all[s]) for s in subs])
    diffs_roi = (real_roi - silent_roi)[:, cwin]

    # ROI cluster
    _, cl, pv, _ = permutation_cluster_1samp_test(diffs_roi, n_permutations=n_perm, tail=0,
                                                  seed=RNG_SEED, verbose="ERROR")
    minp_roi = float(pv.min()) if len(pv) else float("nan")

    # all-channel spatiotemporal
    X = np.array([(chan_tc(tr_all[s]) - chan_tc(ts_all[s])).T[cwin] for s in subs])
    info_eeg = mne.create_info(common, sfreq=float(tr_all[subs[0]].info["sfreq"]), ch_types="eeg")
    info_eeg.set_montage(mne.channels.make_standard_montage("standard_1020"),
                         match_case=False, on_missing="ignore", verbose="ERROR")
    adjacency, _ = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    _, cls, pv2, _ = spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_permutations=n_perm,
                                                        tail=0, seed=RNG_SEED, n_jobs=1, verbose="ERROR")
    minp_all = float(pv2.min()) if len(pv2) else float("nan")
    # channels in the best all-channel cluster (for the EMG-edge caveat)
    best_ch = []
    if len(pv2):
        k = int(np.argmin(pv2))
        best_ch = sorted({common[i] for i in np.unique(cls[k][1])})

    print(f"  ROI cluster min p={minp_roi:.4f} | all-channel min p={minp_all:.4f} "
          f"({n_perm} perms)  sig: ROI={'Y' if (len(pv) and (pv<0.05).any()) else 'N'} "
          f"all={'Y' if (len(pv2) and (pv2<0.05).any()) else 'N'}", flush=True)
    if best_ch:
        print(f"  best all-channel cluster chans: {best_ch}", flush=True)

    # figure: grand-average ROI
    fig, ax = plt.subplots(figsize=(11, 5))
    for arr, col, lab in [(real_roi, "C3", "real SCR"), (silent_roi, "0.5", "silent")]:
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(subs))
        ax.plot(times, m, color=col, lw=2, label=lab); ax.fill_between(times, m-sem, m+sem, color=col, alpha=0.2)
    ax.axvline(0, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time from SCR onset (s)"); ax.set_ylabel(f"{band} power (% change), CSD")
    ax.set_title(f"CSD {band} central ROI {roi} (N={len(subs)})  ROI minp={minp_roi:.3f} all minp={minp_all:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"grandavg_roi_{band}.png", dpi=130); plt.close(fig)

    return dict(band=band, fmin=float(freqs[0]), fmax=float(freqs[-1]), n_sub=len(subs),
                roi_min_p=round(minp_roi, 4), allch_min_p=round(minp_all, 4),
                roi_sig=bool(len(pv) and (pv < 0.05).any()),
                allch_sig=bool(len(pv2) and (pv2 < 0.05).any()),
                best_allch_chans=",".join(best_ch))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bands", nargs="*", default=["theta", "beta", "gamma"])
    p.add_argument("--subjects", nargs="*", default=None)
    args = p.parse_args()
    subs_arg = [_norm_subject(s) for s in args.subjects] if args.subjects else list(SUBJECTS)
    print("=" * 78)
    print(f"scr_band_csd :: focal CSD power sweep  bands={args.bands}  subjects={subs_arg}")
    print("=" * 78, flush=True)
    rows = []
    for band in args.bands:
        if band not in BAND_FREQS:
            print(f"  unknown band {band}, skip", flush=True)
            continue
        try:
            rows.append(run_band(band, BAND_FREQS[band], subs_arg))
        except Exception as e:
            import traceback
            print(f"  [BAND {band} FAILED] {e}", flush=True)
            traceback.print_exc()
            rows.append(dict(band=band, status=f"failed: {e}"))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(NPZ_DIR / "scr_band_csd_summary.csv", index=False)
        print(f"\nSummary -> {NPZ_DIR / 'scr_band_csd_summary.csv'}", flush=True)
        print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
