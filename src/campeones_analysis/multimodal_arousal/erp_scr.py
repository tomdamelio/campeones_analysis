"""Y3-bis -- ERP of preprocessed EEG locked to SCR onsets (diary 05_2).

Parallel to `erp_smna.py` (which locks to SMNA driver peaks). This script uses SCR
onsets detected by `neurokit2.eda_peaks` on the phasic component (EDR), instead of
peaks of the SMNA sparse driver. Motivation (decision 2026-05-13):
  - SCRs are macroscopic, fewer, and visually well-defined; no threshold tuning required.
  - SMNA driver requires choosing a height threshold (A/B/C explored in
    `smna_run_explorer.py`); SCR-based ERP avoids that decision.
  - SCR onsets lag the underlying SMNA impulse by ~1-2 s (phasic rise time); for a
    first pass this is acceptable, and we can compare ERP_SCR vs ERP_SMNA directly.

Anti-overlap rules (stricter than erp_smna.py, two-sided):
  REAL SCR epoch (event at t=0):
    PRE  [-PRE_S, 0]:  max(EDA phasic) <= PRE_PHASIC_THRESH (clean baseline, no prior SCR tail)
    POST (0, +POST_S]: no ADDITIONAL SCR onset (the current SCR's rise is allowed)

  SILENT-EDA control epoch (matched within-subject control, t = no event):
    PRE AND POST  [-PRE_S, +POST_S]:  max(EDA phasic) <= PRE_PHASIC_THRESH
    The entire window must be EDA-silent. This is stronger than "no new SCR onset"
    because it also rejects subthreshold phasic bumps that nk.eda_peaks doesn't flag.
    The control isolates "EEG during autonomic quiet" so that real-vs-control captures
    SCR-specific cortical activity.

Inputs (per subject, all 8 non-practice runs):
  data/derivatives/campeones_preproc/<sub>/ses-vr/eeg/<sub>_*_desc-preproc_eeg.vhdr
  research_diary/context/05_02/y_candidates/<sub>_continuous.npz   (eda_phasic @ 50 Hz)

Pipeline per (subject, run):
  - Load preproc EEG (sr ~500 Hz). Pick EEG channels, attach standard_1020 montage.
    Bandpass 1-20 Hz, resample to 250 Hz (consistency with erp_smna.py).
  - Load eda_phasic from npz; run nk.eda_peaks(method="neurokit") -> SCR onset indices.
  - Apply anti-overlap rule: keep only SCRs whose pre-window has no other SCR onset.
  - Epoch EEG (-5.0 .. +3.0 s) around each clean SCR onset; baseline (-5.0, -4.5).
    Asymmetric: longer pre to ensure clean baseline (cleanliness check is more
    informative looking back); post just covers the SCR rise (~1-3 s).
  - Silent-EDA matched control: sample within the same run such that the ENTIRE
    [-PRE_S, +POST_S] window has max(phasic) <= PRE_PHASIC_THRESH. This is a
    within-subject control of "EEG during autonomic quiet". Real - control isolates
    SCR-specific cortical activity better than uniform-random sampling because the
    control is matched in its EDA state.

Across runs of one subject, concatenate all real and random epochs separately, average.

Outputs (under research_diary/context/05_02/):
  y_candidates/erp_scr_summary.csv          per-subject N (real/random) + per-channel amplitudes
  y_candidates/<sub>_evoked_scr_real-ave.fif
  y_candidates/<sub>_evoked_scr_random-ave.fif
  figures/Y3_erp_scr_<sub>.png              4-panel: phasic + SCR onsets; ERP Fz/Cz/Pz/Oz; topomaps; real vs random

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.erp_scr
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
DATA = REPO / "data"
PREP = DATA / "derivatives" / "campeones_preproc"
NPZ_DIR = REPO / "research_diary" / "context" / "05_02" / "y_candidates"
OUT = REPO / "research_diary" / "context" / "05_02"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "y_candidates").mkdir(parents=True, exist_ok=True)

SUBJECTS = ["sub-23", "sub-24", "sub-33"]

EDA_FS = 50.0
# Asymmetric window: longer pre to ensure a clean baseline (the cleanliness check matters
# more here than post). Post just needs to cover the SCR response (~1-3 s rise).
TMIN, TMAX = -5.0, 3.0
BASELINE = (-5.0, -4.5)
PRE_S = abs(TMIN)
POST_S = TMAX
PRE_PHASIC_THRESH = 1e-4  # max allowed EDA phasic in PRE-window for a "clean" epoch
PLOT_CH = ["Fz", "Cz", "Pz"]
TOPO_TIMES = (0.5, 1.5, 2.5)  # s post-SCR-onset; pushed later vs erp_smna because SCR onset lags impulse

RNG = np.random.default_rng(20260513)


def runs_for(sub: str) -> list[Path]:
    eeg_dir = PREP / sub / "ses-vr" / "eeg"
    return sorted(eeg_dir.glob(f"{sub}_ses-vr_task-*_acq-*_run-*_desc-preproc_eeg.vhdr"))


def run_label(vhdr: Path) -> str:
    parts = vhdr.stem.split("_")
    return "_".join(p for p in parts if p.startswith(("task-", "acq-", "run-")))


def detect_scr_onsets_s(eda_phasic: np.ndarray, fs: float = EDA_FS) -> np.ndarray:
    """Run nk.eda_peaks on the phasic; return SCR onset times in seconds.

    Returns empty array if the detection fails or no SCRs are found.
    """
    try:
        _, info = nk.eda_peaks(eda_phasic, sampling_rate=fs, method="neurokit")
    except Exception:
        return np.array([], dtype=float)
    onsets = np.asarray(info.get("SCR_Onsets", []), dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0:
        return np.array([], dtype=float)
    return np.sort(onsets / fs)


def detect_scr_onsets_and_peaks_s(eda_phasic: np.ndarray, fs: float = EDA_FS) -> tuple[np.ndarray, np.ndarray]:
    """Run nk.eda_peaks and return paired (onsets_s, peaks_s) arrays, sorted by onset time.

    Each onset has its corresponding peak (1:1 pairing per the nk.eda_peaks contract).
    Pairs with any NaN or missing index are dropped.
    """
    try:
        _, info = nk.eda_peaks(eda_phasic, sampling_rate=fs, method="neurokit")
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float)
    onsets = np.asarray(info.get("SCR_Onsets", []), dtype=float)
    peaks = np.asarray(info.get("SCR_Peaks", []), dtype=float)
    n = min(len(onsets), len(peaks))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    onsets = onsets[:n]
    peaks = peaks[:n]
    valid = np.isfinite(onsets) & np.isfinite(peaks) & (peaks > onsets)
    onsets_t = onsets[valid] / fs
    peaks_t = peaks[valid] / fs
    order = np.argsort(onsets_t)
    return onsets_t[order], peaks_t[order]


def real_scr_is_clean(t_s: float, phasic: np.ndarray, fs: float,
                       all_onsets_s: np.ndarray, *,
                       pre_s: float = PRE_S, post_s: float = POST_S,
                       pre_thresh: float = PRE_PHASIC_THRESH) -> bool:
    """Cleanliness check for a REAL SCR epoch centered at SCR onset `t_s`.

    PRE  [t-pre_s, t]: max(phasic) <= pre_thresh  (clean baseline)
    POST (t, t+post_s]: no ADDITIONAL SCR onset (current SCR's rise is allowed)
    """
    n = len(phasic)
    lo_idx = max(0, int(np.floor((t_s - pre_s) * fs)))
    hi_idx = min(n - 1, int(np.ceil((t_s + post_s) * fs)))
    if lo_idx < 0 or hi_idx >= n or hi_idx <= lo_idx:
        return False
    t_idx = int(np.round(t_s * fs))
    pre_segment = phasic[lo_idx:t_idx + 1]
    if pre_segment.size == 0 or float(np.nanmax(pre_segment)) > pre_thresh:
        return False
    eps = 1.5 / fs
    post_onsets = all_onsets_s[(all_onsets_s > t_s) & (all_onsets_s <= t_s + post_s)]
    post_onsets = post_onsets[np.abs(post_onsets - t_s) > eps]
    if len(post_onsets) > 0:
        return False
    return True


def silent_window_is_clean(t_s: float, phasic: np.ndarray, fs: float, *,
                            pre_s: float = PRE_S, post_s: float = POST_S,
                            thresh: float = PRE_PHASIC_THRESH) -> bool:
    """Cleanliness check for a SILENT-EDA control epoch centered at `t_s`.

    Both PRE and POST must have max(phasic) <= thresh. The whole [-pre_s, +post_s]
    window is required to be EDA-silent (no SCR happening anywhere, including
    subthreshold bumps).
    """
    n = len(phasic)
    lo_idx = max(0, int(np.floor((t_s - pre_s) * fs)))
    hi_idx = min(n - 1, int(np.ceil((t_s + post_s) * fs)))
    if lo_idx < 0 or hi_idx >= n or hi_idx <= lo_idx + 1:
        return False
    segment = phasic[lo_idx:hi_idx + 1]
    if segment.size == 0 or float(np.nanmax(segment)) > thresh:
        return False
    return True


def filter_clean_onsets(onsets_s: np.ndarray, phasic: np.ndarray, fs: float) -> np.ndarray:
    """Keep only SCR onsets whose epoch passes the REAL-SCR cleanliness check."""
    if onsets_s.size == 0:
        return onsets_s
    keep = [t for t in onsets_s if real_scr_is_clean(float(t), phasic, fs, onsets_s)]
    return np.asarray(keep, dtype=float)


def sample_silent_controls(n_target: int, duration_s: float, phasic: np.ndarray, fs: float,
                            rng: np.random.Generator,
                            pre_s: float = PRE_S, post_s: float = POST_S,
                            min_separation_s: float = 1.0,
                            max_attempts_factor: int = 500) -> np.ndarray:
    """Sample `n_target` times where the entire [-pre_s, +post_s] window is EDA-silent.

    Candidates are also required to be at least `min_separation_s` apart from
    previously-picked controls (avoid duplicating epochs around the same silent stretch).
    Returns up to n_target picks; may return fewer if budget exhausted.
    """
    if n_target <= 0 or duration_s <= pre_s + post_s:
        return np.array([], dtype=float)
    lo, hi = pre_s, duration_s - post_s
    out: list[float] = []
    attempts = 0
    budget = max_attempts_factor * max(1, n_target)
    while len(out) < n_target and attempts < budget:
        attempts += 1
        t = float(rng.uniform(lo, hi))
        if not silent_window_is_clean(t, phasic, fs, pre_s=pre_s, post_s=post_s):
            continue
        # spacing rule: avoid picking near already-chosen control
        if out:
            arr = np.asarray(out, dtype=float)
            if np.min(np.abs(arr - t)) < min_separation_s:
                continue
        out.append(t)
    return np.asarray(out, dtype=float)


def attach_montage_and_drop_no_pos(raw: mne.io.BaseRaw) -> None:
    """Attach standard_1020 montage and drop channels still lacking positions.

    `raw.pick("eeg")` keeps peripherals incorrectly typed as EEG (ECG, R_EYE,
    L_EYE, RESP, triggerStream in this dataset). Those break plot_topomap.
    """
    try:
        mont = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(mont, match_case=False, on_missing="ignore", verbose="ERROR")
    except Exception:
        return
    no_pos = []
    for ch_info in raw.info["chs"]:
        loc = ch_info["loc"][:3]
        if np.isnan(loc).any() or np.allclose(loc, 0.0):
            no_pos.append(ch_info["ch_name"])
    if no_pos:
        raw.drop_channels(no_pos)


def build_events(times_s: np.ndarray, eeg_sfreq: float, code: int) -> np.ndarray:
    if len(times_s) == 0:
        return np.empty((0, 3), dtype=int)
    samples = np.round(np.asarray(times_s) * eeg_sfreq).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), np.full_like(samples, code)])
    return events


def epoch_one_run(raw: mne.io.BaseRaw, events_t_s: np.ndarray, code: int,
                  tmin: float | None = None, tmax: float | None = None,
                  baseline: tuple | None = None) -> mne.Epochs | None:
    """Build mne.Epochs locked to `events_t_s` (in seconds).

    tmin/tmax/baseline default to the module-level TMIN/TMAX/BASELINE so existing
    callers (erp_scr.main, tfr_psd_scr) keep working unchanged. Pass explicit values
    to use a different window (e.g. peak-aligned [-4, +4] in analyze_scr_peak.py).
    """
    if tmin is None: tmin = TMIN
    if tmax is None: tmax = TMAX
    if baseline is None: baseline = BASELINE
    sfreq = float(raw.info["sfreq"])
    duration = raw.times[-1]
    valid = (events_t_s + tmin > 0) & (events_t_s + tmax < duration)
    events_t_s = events_t_s[valid]
    if len(events_t_s) == 0:
        return None
    events = build_events(events_t_s, sfreq, code=code)
    epochs = mne.Epochs(
        raw, events=events, event_id={"x": code},
        tmin=tmin, tmax=tmax, baseline=baseline,
        preload=True, picks="eeg", reject_by_annotation=False, verbose="ERROR",
    )
    return epochs


def panel_figure(sub: str, scr_per_run: dict, evoked_real: mne.Evoked, evoked_rand: mne.Evoked,
                 n_real: int, n_rand: int, path: Path) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.2, 1.2])
    fig.suptitle(
        f"{sub}  --  Y3-SCR: ERP del EEG locked a SCR onsets  "
        f"(n_real={n_real}, n_silent_control={n_rand}, window={TMIN}..{TMAX}s, baseline={BASELINE})",
        fontsize=11,
    )

    # row 0: phasic + SCR onsets for the longest run
    ax0 = fig.add_subplot(gs[0, :])
    longest_label = max(scr_per_run.keys(), key=lambda k: len(scr_per_run[k]["phasic"]))
    sr = scr_per_run[longest_label]
    t = np.arange(len(sr["phasic"])) / EDA_FS
    ax0.plot(t, sr["phasic"], lw=0.7, color="C2", label="EDA phasic (EDR)")
    onsets_all = sr["onsets_s_all"]
    onsets_kept = sr["onsets_s_kept"]
    if len(onsets_all) > 0:
        # plot dropped SCRs in gray
        dropped = np.setdiff1d(onsets_all, onsets_kept, assume_unique=False)
        if len(dropped) > 0:
            yvals_d = np.interp(dropped, t, sr["phasic"])
            ax0.plot(dropped, yvals_d, "x", ms=5, color="0.55",
                     label=f"SCR onsets dropped (overlap, N={len(dropped)})")
    if len(onsets_kept) > 0:
        yvals = np.interp(onsets_kept, t, sr["phasic"])
        ax0.plot(onsets_kept, yvals, "v", ms=5, color="C3",
                 label=f"SCR onsets kept (N={len(onsets_kept)})")
    ax0.set_xlabel("time in run (s)")
    ax0.set_ylabel("EDA phasic")
    ax0.set_title(f"sanity: {longest_label}")
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # row 1: ERP at Fz, Cz, Pz, and Oz-proxy = mean(O1,O2)
    times_ms = evoked_real.times * 1000.0
    for j, ch in enumerate(PLOT_CH):
        ax = fig.add_subplot(gs[1, j])
        if ch in evoked_real.ch_names:
            i = evoked_real.ch_names.index(ch)
            ax.plot(times_ms, evoked_real.data[i] * 1e6, color="C3", lw=1.2, label="real (SCR onsets)")
            ax.plot(times_ms, evoked_rand.data[i] * 1e6, color="0.5", lw=1.0, ls="--", label="silent-EDA control")
        else:
            ax.text(0.5, 0.5, f"{ch} no presente", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from SCR onset (ms)")
        ax.set_ylabel("uV")
        ax.set_title(ch)
        ax.legend(fontsize=7)
    axO = fig.add_subplot(gs[1, 3])
    ozs = [c for c in ("O1", "O2") if c in evoked_real.ch_names]
    if ozs:
        idxs = [evoked_real.ch_names.index(c) for c in ozs]
        oz_real = evoked_real.data[idxs].mean(axis=0) * 1e6
        oz_rand = evoked_rand.data[idxs].mean(axis=0) * 1e6
        axO.plot(times_ms, oz_real, color="C3", lw=1.2, label="real")
        axO.plot(times_ms, oz_rand, color="0.5", lw=1.0, ls="--", label="random")
    axO.axvline(0, color="k", lw=0.5)
    axO.axhline(0, color="k", lw=0.5)
    axO.set_xlabel("time from SCR onset (ms)")
    axO.set_ylabel("uV")
    axO.set_title("mean(O1, O2)")
    axO.legend(fontsize=7)

    # row 2: topomaps
    for j, tt in enumerate(TOPO_TIMES):
        ax = fig.add_subplot(gs[2, j])
        try:
            evoked_real.plot_topomap(times=tt, axes=ax, colorbar=False, show=False, time_format="real %.2fs")
        except Exception as e:
            ax.text(0.5, 0.5, f"topomap fail @ {tt}s\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"real @ {tt:.2f}s")
    ax = fig.add_subplot(gs[2, 3])
    try:
        diff = mne.combine_evoked([evoked_real, evoked_rand], weights=[1, -1])
        diff.plot_topomap(times=TOPO_TIMES[1], axes=ax, colorbar=False, show=False, time_format=f"diff @ %.2fs")
    except Exception as e:
        ax.text(0.5, 0.5, f"diff topomap fail\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    rows = []
    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        cont_path = NPZ_DIR / f"{sub}_continuous.npz"
        if not cont_path.exists():
            print(f"  missing {cont_path.name}; skipping subject")
            continue
        cont = np.load(cont_path, allow_pickle=True)
        runs_in_npz = list(cont["runs"])
        all_real_epochs: list[mne.Epochs] = []
        all_rand_epochs: list[mne.Epochs] = []
        scr_per_run: dict = {}

        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                print(f"  {label}: no EDA in npz, skipping")
                continue
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
                raw.pick("eeg")
                attach_montage_and_drop_no_pos(raw)
                raw.filter(l_freq=1.0, h_freq=20.0, verbose="ERROR")
                raw.resample(250.0, verbose="ERROR")
                duration = float(raw.times[-1])

                eda_phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
                onsets_s_all = detect_scr_onsets_s(eda_phasic, EDA_FS)
                # cleanliness filter: PRE phasic at baseline AND no extra SCR onset in POST
                onsets_s_all = onsets_s_all[onsets_s_all < duration]
                onsets_s_kept = filter_clean_onsets(onsets_s_all, eda_phasic, EDA_FS)

                # silent-EDA matched control: same N, full-window phasic-silent
                rand_t_s = sample_silent_controls(
                    n_target=len(onsets_s_kept), duration_s=duration,
                    phasic=eda_phasic, fs=EDA_FS, rng=RNG,
                )

                ep_real = epoch_one_run(raw, onsets_s_kept, code=1)
                ep_rand = epoch_one_run(raw, rand_t_s, code=2)

                scr_per_run[label] = {
                    "phasic": eda_phasic,
                    "onsets_s_all": onsets_s_all,
                    "onsets_s_kept": onsets_s_kept,
                }

                n_real_kept = len(ep_real) if ep_real is not None else 0
                n_rand_kept = len(ep_rand) if ep_rand is not None else 0
                if ep_real is not None: all_real_epochs.append(ep_real)
                if ep_rand is not None: all_rand_epochs.append(ep_rand)

                print(
                    f"  {label}: SCRs total={len(onsets_s_all)}  clean={len(onsets_s_kept)} "
                    f"(dropped {len(onsets_s_all) - len(onsets_s_kept)} by cleanliness)  "
                    f"real epochs={n_real_kept}  rand={n_rand_kept}/{len(onsets_s_kept)}"
                )
            except Exception as e:
                print(f"  {label}: FAILED -- {e}")

        if not all_real_epochs:
            print(f"  {sub}: no epochs")
            continue
        ep_real_all = mne.concatenate_epochs(all_real_epochs, verbose="ERROR")
        ep_rand_all = mne.concatenate_epochs(all_rand_epochs, verbose="ERROR")
        ev_real = ep_real_all.average()
        ev_rand = ep_rand_all.average()

        # per-channel summary metric: max |amplitude| 0..2.5s post SCR onset
        for ch in PLOT_CH:
            if ch in ev_real.ch_names:
                i = ev_real.ch_names.index(ch)
                t_ax = ev_real.times
                m = (t_ax >= 0) & (t_ax <= 2.5)
                amp_real = float(np.max(np.abs(ev_real.data[i, m])) * 1e6)
                amp_rand = float(np.max(np.abs(ev_rand.data[i, m])) * 1e6)
                rows.append(dict(
                    subject=sub, channel=ch,
                    n_real=len(ep_real_all), n_rand=len(ep_rand_all),
                    peak_abs_uV_real=amp_real, peak_abs_uV_random=amp_rand,
                    ratio_real_over_random=amp_real / amp_rand if amp_rand > 0 else np.nan,
                ))

        panel_figure(sub, scr_per_run, ev_real, ev_rand,
                     n_real=len(ep_real_all), n_rand=len(ep_rand_all),
                     path=OUT / "figures" / f"Y3_erp_scr_{sub}.png")

        ev_real.save(NPZ_DIR / f"{sub}_evoked_scr_real-ave.fif", overwrite=True)
        ev_rand.save(NPZ_DIR / f"{sub}_evoked_scr_random-ave.fif", overwrite=True)
        print(f"  {sub}: total real epochs={len(ep_real_all)}  rand={len(ep_rand_all)}")

    pd.DataFrame(rows).to_csv(NPZ_DIR / "erp_scr_summary.csv", index=False)
    print(f"\nDone. Outputs: {OUT/'figures'/'Y3_erp_scr_*.png'} + {NPZ_DIR/'erp_scr_summary.csv'}")


if __name__ == "__main__":
    main()
