"""Y3 -- ERP of preprocessed EEG locked to SMNA peaks (Tarea 1.C, diario 05_2).

Forward sanity (NOT decoding): if the average EEG locked to SMNA peaks shows a clean
component (e.g. P200-P300 or SCR-locked slow wave), it is evidence of EEG <-> SMNA coupling
and supports inverting the pipeline (decoding SMNA from EEG). If nothing emerges, surface
that to the meeting.

Inputs (per subject, all 8 non-practice runs concatenated as epochs):
  data/derivatives/campeones_preproc/<sub>/ses-vr/eeg/<sub>_*_desc-preproc_eeg.vhdr
  research_diary/context/05_02/y_candidates/<sub>_continuous.npz   (SMNA @ 50 Hz from build_y_candidates v2)

Pipeline per (subject, run):
  - Load preproc EEG (sr ~500 Hz). Pick the 31 EEG channels and apply standard_1020 montage.
  - Load continuous SMNA from the npz. Detect peaks via scipy.signal.find_peaks
      height = 5% of run-max-SMNA, distance = 1.5 s.
  - Convert SMNA peak times -> EEG sample indices.
  - Epoch EEG (-1.5 .. +3.0 s) around each peak; baseline (-1.5 .. -0.5).
  - Random control: same number of "peaks" placed uniformly at random within the run.
Across runs of one subject, concatenate all real-peak and random-peak epochs separately,
then average -> evoked. Plot per subject.

Outputs (in research_diary/context/05_02/):
  y_candidates/erp_smna_summary.csv         per-subject N peaks (real / random) + per-channel peak amplitudes
  figures/Y3_erp_smna_<sub>.png             4-panel: SMNA + peaks; ERP Fz/Cz/Pz/Oz-proxy; topomaps; real vs random

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.erp_smna
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
from scipy.signal import find_peaks

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

EDA_FS = 50.0  # SMNA in continuous npz is at 50 Hz (matches build_y_candidates v2)
PEAK_HEIGHT_FRAC = 0.05  # min peak height = 5% of per-run max SMNA
PEAK_MIN_SEP_S = 1.5  # min separation between consecutive peaks (s)
TMIN, TMAX = -1.5, 3.0  # epoch window (s) relative to peak
BASELINE = (-1.5, -0.5)
PLOT_CH = ["Fz", "Cz", "Pz"]  # central midline; Oz proxy = mean(O1, O2) added later
TOPO_TIMES = (0.3, 0.8, 1.5)  # s post-peak for topomaps

RNG = np.random.default_rng(20260513)


def runs_for(sub: str) -> list[Path]:
    eeg_dir = PREP / sub / "ses-vr" / "eeg"
    return sorted(eeg_dir.glob(f"{sub}_ses-vr_task-*_acq-*_run-*_desc-preproc_eeg.vhdr"))


def run_label(vhdr: Path) -> str:
    parts = vhdr.stem.split("_")
    return "_".join(p for p in parts if p.startswith(("task-", "acq-", "run-")))


def detect_smna_peaks(smna: np.ndarray, fs: float) -> np.ndarray:
    if not np.any(smna > 0):
        return np.array([], dtype=int)
    height = PEAK_HEIGHT_FRAC * float(np.nanmax(smna))
    distance = max(1, int(round(PEAK_MIN_SEP_S * fs)))
    peaks, _ = find_peaks(smna, height=height, distance=distance)
    return peaks


def build_events(times_s: np.ndarray, eeg_sfreq: float, code: int) -> np.ndarray:
    if len(times_s) == 0:
        return np.empty((0, 3), dtype=int)
    samples = np.round(np.asarray(times_s) * eeg_sfreq).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), np.full_like(samples, code)])
    return events


def epoch_one_run(raw: mne.io.BaseRaw, peak_t_s: np.ndarray, code: int) -> mne.Epochs | None:
    sfreq = float(raw.info["sfreq"])
    duration = raw.times[-1]
    # keep events whose epoch fits inside the run
    valid = (peak_t_s + TMIN > 0) & (peak_t_s + TMAX < duration)
    peak_t_s = peak_t_s[valid]
    if len(peak_t_s) == 0:
        return None
    events = build_events(peak_t_s, sfreq, code=code)
    epochs = mne.Epochs(
        raw, events=events, event_id={"x": code},
        tmin=TMIN, tmax=TMAX, baseline=BASELINE,
        preload=True, picks="eeg", reject_by_annotation=False, verbose="ERROR",
    )
    return epochs


def attach_montage(raw: mne.io.BaseRaw) -> None:
    try:
        mont = mne.channels.make_standard_montage("standard_1020")
        # only set for channels that exist
        raw.set_montage(mont, match_case=False, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass


def panel_figure(sub: str, smna_concat: dict, evoked_real: mne.Evoked, evoked_rand: mne.Evoked,
                 n_real: int, n_rand: int, peak_max_per_run: list, path: Path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.2, 1.2])
    fig.suptitle(
        f"{sub}  -- Y3: ERP del EEG locked a peaks de SMNA  (n_peaks_real={n_real}, n_random_control={n_rand})",
        fontsize=12,
    )

    # row 0 (full width): SMNA + peaks for the longest run, eventos overlaid
    ax0 = fig.add_subplot(gs[0, :])
    longest_label = max(smna_concat.keys(), key=lambda k: len(smna_concat[k]["smna"]))
    sm = smna_concat[longest_label]
    t = np.arange(len(sm["smna"])) / EDA_FS
    ax0.plot(t, sm["smna"], lw=0.6, color="C2", label="SMNA continuous")
    if len(sm["peaks_idx"]):
        ax0.plot(t[sm["peaks_idx"]], sm["smna"][sm["peaks_idx"]], "rv", ms=5, label=f"peaks ({len(sm['peaks_idx'])})")
    ax0.set_xlabel("time in run (s)")
    ax0.set_ylabel("SMNA driver")
    ax0.set_title(f"sanity: {longest_label}")
    ax0.legend(loc="upper right", fontsize=8)

    # row 1 (4 axes): ERP at Fz, Cz, Pz, and Oz-proxy = mean(O1,O2). Real vs random.
    ch_targets = list(PLOT_CH)
    times_ms = evoked_real.times * 1000.0
    for j, ch in enumerate(ch_targets):
        ax = fig.add_subplot(gs[1, j])
        if ch in evoked_real.ch_names:
            i = evoked_real.ch_names.index(ch)
            ax.plot(times_ms, evoked_real.data[i] * 1e6, color="C3", lw=1.2, label="real (peaks SMNA)")
            ax.plot(times_ms, evoked_rand.data[i] * 1e6, color="0.5", lw=1.0, ls="--", label="random control")
        else:
            ax.text(0.5, 0.5, f"{ch} no presente", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from SMNA peak (ms)")
        ax.set_ylabel("uV")
        ax.set_title(ch)
        ax.legend(fontsize=7)
    # 4th panel: Oz-proxy = mean(O1, O2)
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
    axO.set_xlabel("time from SMNA peak (ms)")
    axO.set_ylabel("uV")
    axO.set_title("mean(O1, O2)")
    axO.legend(fontsize=7)

    # row 2 (4 axes): topomaps real at TOPO_TIMES + topomap real-minus-random at one time
    has_montage = any(c is not None for c in evoked_real.info.get("chs", [{}])[0].get("loc", [0]) if c is not None)
    for j, t in enumerate(TOPO_TIMES):
        ax = fig.add_subplot(gs[2, j])
        try:
            evoked_real.plot_topomap(times=t, axes=ax, colorbar=False, show=False, time_format="real %.2fs")
        except Exception as e:
            ax.text(0.5, 0.5, f"topomap fail @ {t}s\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"real @ {t:.2f}s")
    # 4th panel: real - random difference
    ax = fig.add_subplot(gs[2, 3])
    try:
        diff = mne.combine_evoked([evoked_real, evoked_rand], weights=[1, -1])
        diff.plot_topomap(times=TOPO_TIMES[1], axes=ax, colorbar=False, show=False, time_format=f"diff @ %.2fs")
    except Exception as e:
        ax.text(0.5, 0.5, f"diff topomap fail\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    rows = []
    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = list(cont["runs"])
        all_real_epochs: list[mne.Epochs] = []
        all_rand_epochs: list[mne.Epochs] = []
        smna_per_run: dict = {}

        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                print(f"  {label}: no SMNA in npz, skipping")
                continue
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
                raw.pick("eeg")
                attach_montage(raw)
                # apply a 1-20 Hz bandpass to keep slow components but cut LF drift, since this is ERP-ish averaging
                raw.filter(l_freq=1.0, h_freq=20.0, verbose="ERROR")
                # resample to a common sr -- raw header sr varies slightly between runs (~499.997-499.999 Hz)
                # which makes mne.concatenate_epochs reject across runs; 250 Hz is plenty for 1-20 Hz signals
                raw.resample(250.0, verbose="ERROR")
                duration = float(raw.times[-1])

                smna = np.asarray(cont[f"{label}__eda_smna"], float)
                peaks_idx = detect_smna_peaks(smna, EDA_FS)
                # cap at duration of EEG (avoid overrun)
                peak_t_s = peaks_idx / EDA_FS
                peak_t_s = peak_t_s[peak_t_s < duration]

                # random control: same n peaks, uniform random in [0, duration]
                n = len(peak_t_s)
                rand_t_s = RNG.uniform(0, duration, size=n) if n > 0 else np.array([])

                ep_real = epoch_one_run(raw, peak_t_s, code=1)
                ep_rand = epoch_one_run(raw, rand_t_s, code=2)

                smna_per_run[label] = {"smna": smna, "peaks_idx": peaks_idx}

                n_real_kept = len(ep_real) if ep_real is not None else 0
                n_rand_kept = len(ep_rand) if ep_rand is not None else 0
                if ep_real is not None: all_real_epochs.append(ep_real)
                if ep_rand is not None: all_rand_epochs.append(ep_rand)

                print(f"  {label}: SMNA peaks={len(peaks_idx)}  real epochs kept={n_real_kept}  rand kept={n_rand_kept}")
            except Exception as e:
                print(f"  {label}: FAILED -- {e}")

        if not all_real_epochs:
            print(f"  {sub}: no epochs")
            continue
        ep_real_all = mne.concatenate_epochs(all_real_epochs, verbose="ERROR")
        ep_rand_all = mne.concatenate_epochs(all_rand_epochs, verbose="ERROR")
        ev_real = ep_real_all.average()
        ev_rand = ep_rand_all.average()

        # peak amplitude (uV) at Cz @ 0..1.5s post-peak as a summary metric
        for ch in PLOT_CH:
            if ch in ev_real.ch_names:
                i = ev_real.ch_names.index(ch)
                t = ev_real.times
                m = (t >= 0) & (t <= 1.5)
                amp_real = float(np.max(np.abs(ev_real.data[i, m])) * 1e6)
                amp_rand = float(np.max(np.abs(ev_rand.data[i, m])) * 1e6)
                rows.append(dict(
                    subject=sub, channel=ch,
                    n_real=len(ep_real_all), n_rand=len(ep_rand_all),
                    peak_abs_uV_real=amp_real, peak_abs_uV_random=amp_rand,
                    ratio_real_over_random=amp_real / amp_rand if amp_rand > 0 else np.nan,
                ))

        peak_max_per_run = [smna_per_run[k]["smna"].max() for k in smna_per_run]
        panel_figure(sub, smna_per_run, ev_real, ev_rand,
                     n_real=len(ep_real_all), n_rand=len(ep_rand_all),
                     peak_max_per_run=peak_max_per_run,
                     path=OUT / "figures" / f"Y3_erp_smna_{sub}.png")

        # also save evoked for downstream use
        ev_real.save(NPZ_DIR / f"{sub}_evoked_smna_real-ave.fif", overwrite=True)
        ev_rand.save(NPZ_DIR / f"{sub}_evoked_smna_random-ave.fif", overwrite=True)
        print(f"  {sub}: total real epochs={len(ep_real_all)}  rand={len(ep_rand_all)}")

    pd.DataFrame(rows).to_csv(NPZ_DIR / "erp_smna_summary.csv", index=False)
    print(f"\nDone. Outputs: {OUT/'figures'/'Y3_erp_smna_*.png'} + {NPZ_DIR/'erp_smna_summary.csv'}")


if __name__ == "__main__":
    main()
