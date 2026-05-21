"""Minimal QA visualization for peripheral physiology (EDA/GSR, ECG, RESP) in CAMPEONES.

Replicates the ad-hoc NeuroKit-based QC plotting used in ``dmt-emotions`` (which had no
dedicated QA viewer -- just ``nk.<modality>_plot()`` on the processed signals), adapted to:

* the CAMPEONES BIDS layout (1 subject = ~8 runs in ``ses-vr/eeg/``),
* the recommended cleaning per modality from the lit consolidation
  (``research_diary/context/05_02/physio_lit/0A_04_consolidacion.md``):
  EDA -> ``nk.eda_clean`` (LP 3 Hz), ECG -> ``nk.ecg_clean`` (HP 0.5 Hz + 50 Hz notch),
  RESP -> ``nk.rsp_clean`` (khodadad2018) + RVT ``harrison2021``,
* event onsets overlaid from the ``merged_events`` derivatives,
* sub-27's run anomaly (the aborted take ``run-005`` of ``task-04_acq-a`` is dropped;
  ``run-006`` is kept -- see ``research_diary/05_2_diario_tareas.md`` 0.C).

This is deliberately minimal: one multi-panel PNG per (subject, run) + one summary TSV.
It is a *visual sanity* tool, not the analysis pipeline -- EDA/RESP are decimated to 50 Hz
for speed (well above their bandwidth); ECG is kept at native ~500 Hz. cvxEDA is NOT run
here (that is Task 1.0); the default NeuroKit tonic/phasic split is enough for QA.

Run::

    micromamba run -n campeones python -m src.campeones_analysis.physio.qa_viz
    micromamba run -n campeones python -m src.campeones_analysis.physio.qa_viz --subjects 23 --limit 1

Outputs::

    reports/physio_qa/sub-XX/sub-XX_ses-vr_task-NN_acq-X_run-YYY_desc-qc_physio.png
    reports/physio_qa/physio_qa_summary.tsv
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import decimate

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MERGED_EVENTS_DIR = PROJECT_ROOT / "data" / "derivatives" / "merged_events"
OUT_DIR = PROJECT_ROOT / "reports" / "physio_qa"

SESSION = "vr"
TIER1_SUBJECTS = ["23", "24", "27", "33"]

# Channel names in the BrainVision multimodal recording (same as dmt-emotions' CANALES).
PHYSIO_CHANNELS = {"EDA": "GSR", "ECG": "ECG", "RESP": "RESP"}

# sub-27: task-04_acq-a was recorded twice -- run-005 is an aborted take (~2.4 MB, ~27 s),
# run-006 is the full recording. Keep run-006, drop run-005 (matches config_luminance.py).
SUBJECT_RUNS_TO_DROP = {"27": {"005"}}

# Decimation targets for the QA viz (processing speed only; not the analysis pipeline).
EDA_VIZ_SR = 50.0
RESP_VIZ_SR = 50.0
# ECG kept at native sr.

# Plausible resting ranges for sanity bands on the plots (from 0A_04 sec 2.3).
HR_PLAUSIBLE = (40.0, 180.0)
RESP_RATE_PLAUSIBLE = (6.0, 30.0)

_FNAME_RE = re.compile(
    r"sub-(?P<sub>\d+)_ses-(?P<ses>\w+)_task-(?P<task>\w+)_acq-(?P<acq>\w+)_run-(?P<run>\d+)_eeg\.vhdr$"
)


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def discover_runs(subjects, tasks=None, acqs=None):
    """Yield dicts {sub, ses, task, acq, run, path} for each non-practice raw run."""
    out = []
    for sub in subjects:
        eeg_dir = RAW_DIR / f"sub-{sub}" / f"ses-{SESSION}" / "eeg"
        if not eeg_dir.is_dir():
            warnings.warn(f"no raw eeg dir for sub-{sub}: {eeg_dir}")
            continue
        for vhdr in sorted(eeg_dir.glob(f"sub-{sub}_ses-{SESSION}_task-*_acq-*_run-*_eeg.vhdr")):
            m = _FNAME_RE.search(vhdr.name)
            if not m:
                continue
            d = m.groupdict()
            if d["task"] == "practice":
                continue
            if d["run"] in SUBJECT_RUNS_TO_DROP.get(sub, set()):
                continue
            if tasks and d["task"] not in tasks:
                continue
            if acqs and d["acq"] not in acqs:
                continue
            out.append(
                {"sub": sub, "ses": d["ses"], "task": d["task"], "acq": d["acq"],
                 "run": d["run"], "path": vhdr}
            )
    return out


def find_merged_events(sub, task, acq):
    """Return the merged_events DataFrame for (sub, task, acq), matched by task+acq.

    The run number in merged_events filenames is *not* trusted (sub-27 is renumbered);
    matching is done on task+acq, which is unambiguous.
    """
    eeg_dir = MERGED_EVENTS_DIR / f"sub-{sub}" / f"ses-{SESSION}" / "eeg"
    if not eeg_dir.is_dir():
        return None
    hits = list(eeg_dir.glob(f"sub-{sub}_ses-{SESSION}_task-{task}_acq-{acq}_run-*_desc-merged_events.tsv"))
    if not hits:
        return None
    try:
        df = pd.read_csv(hits[0], sep="\t")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"failed reading {hits[0]}: {exc}")
        return None
    return df


# --------------------------------------------------------------------------------------
# Loading / processing
# --------------------------------------------------------------------------------------
def _load_channel(raw, ch_name):
    if ch_name not in raw.ch_names:
        return None, None
    data = raw.get_data(picks=[ch_name])[0].astype(float)
    return data, raw.info["sfreq"]


def _decimate_to(sig, sr, target_sr):
    if target_sr >= sr:
        return sig, sr
    q = int(round(sr / target_sr))
    q = max(q, 2)
    return decimate(sig, q, ftype="iir", zero_phase=True), sr / q


def _pct_flat(clean, eps_rel=1e-4):
    """Fraction of samples whose first difference is ~0 (electrode off / clipped)."""
    if clean is None or len(clean) < 2:
        return np.nan
    d = np.abs(np.diff(clean))
    scale = np.nanstd(clean)
    if not np.isfinite(scale) or scale == 0:
        return 1.0
    return float(np.mean(d < eps_rel * scale))


def process_run(vhdr_path):
    """Run NeuroKit processing on the 3 physio channels of one raw .vhdr.

    Returns a dict with per-modality {signals, info, sr, summary} (or that modality absent
    on failure), plus run-level metadata.
    """
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    sr_native = raw.info["sfreq"]
    dur_s = raw.n_times / sr_native
    res = {"sr_native": sr_native, "dur_s": dur_s, "modalities": {}}

    # ---- EDA -------------------------------------------------------------------------
    eda_raw, sr = _load_channel(raw, PHYSIO_CHANNELS["EDA"])
    if eda_raw is not None:
        try:
            sig, sr_e = _decimate_to(eda_raw, sr, EDA_VIZ_SR)
            signals, info = nk.eda_process(sig, sampling_rate=sr_e, method="neurokit")
            scl = signals["EDA_Tonic"].to_numpy()
            n_scr = int(np.nansum(signals["SCR_Peaks"].to_numpy()))
            res["modalities"]["EDA"] = {
                "signals": signals, "info": info, "sr": sr_e,
                "summary": {
                    "eda_scl_mean": float(np.nanmean(scl)),
                    "eda_scl_min": float(np.nanmin(scl)),
                    "eda_scl_max": float(np.nanmax(scl)),
                    "eda_n_scr": n_scr,
                    "eda_scr_per_min": n_scr / (dur_s / 60.0) if dur_s else np.nan,
                    "eda_pct_flat": _pct_flat(signals["EDA_Clean"].to_numpy()),
                },
            }
        except Exception as exc:
            warnings.warn(f"EDA processing failed for {vhdr_path.name}: {exc}")

    # ---- ECG -------------------------------------------------------------------------
    ecg_raw, sr = _load_channel(raw, PHYSIO_CHANNELS["ECG"])
    if ecg_raw is not None:
        try:
            # Polarity check: Brain Products ECG can come inverted (electrode swap).
            # R-up convention is needed for ecg_quality (template matching), for
            # cleaner figures, and to keep the lit-rec rule mean(R)/mean(T) > 1
            # interpretable (0A_04 sec 2.3). nk.ecg_invert uses a skewness/concavity
            # heuristic; it is a no-op when polarity is already correct.
            ecg_fixed, ecg_inverted = nk.ecg_invert(ecg_raw, sampling_rate=sr)
            signals, info = nk.ecg_process(ecg_fixed, sampling_rate=sr, method="neurokit")
            hr = signals["ECG_Rate"].to_numpy()
            qual = signals["ECG_Quality"].to_numpy() if "ECG_Quality" in signals else np.array([np.nan])
            rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            res["modalities"]["ECG"] = {
                "signals": signals, "info": info, "sr": sr,
                "summary": {
                    "ecg_inverted": bool(ecg_inverted),
                    "ecg_hr_mean": float(np.nanmean(hr)),
                    "ecg_hr_min": float(np.nanmin(hr)),
                    "ecg_hr_max": float(np.nanmax(hr)),
                    "ecg_quality_mean": float(np.nanmean(qual)),
                    "ecg_n_rpeaks": int(rpeaks.size),
                    "ecg_pct_flat": _pct_flat(signals["ECG_Clean"].to_numpy()),
                },
            }
        except Exception as exc:
            warnings.warn(f"ECG processing failed for {vhdr_path.name}: {exc}")

    # ---- RESP ------------------------------------------------------------------------
    resp_raw, sr = _load_channel(raw, PHYSIO_CHANNELS["RESP"])
    if resp_raw is not None:
        try:
            sig, sr_r = _decimate_to(resp_raw, sr, RESP_VIZ_SR)
            signals, info = nk.rsp_process(sig, sampling_rate=sr_r, method="khodadad2018",
                                           method_rvt="harrison2021")
            rate = signals["RSP_Rate"].to_numpy()
            rvt = signals["RSP_RVT"].to_numpy() if "RSP_RVT" in signals else np.array([np.nan])
            n_peaks = int(np.nansum(signals["RSP_Peaks"].to_numpy()))
            finite_rvt = rvt[np.isfinite(rvt) & (rvt > 0)]
            res["modalities"]["RESP"] = {
                "signals": signals, "info": info, "sr": sr_r,
                "summary": {
                    "resp_rate_mean": float(np.nanmean(rate)),
                    "resp_rate_min": float(np.nanmin(rate)),
                    "resp_rate_max": float(np.nanmax(rate)),
                    "resp_rvt_mean": float(np.nanmean(finite_rvt)) if finite_rvt.size else np.nan,
                    "resp_n_peaks": n_peaks,
                    "resp_pct_flat": _pct_flat(signals["RSP_Clean"].to_numpy()),
                },
            }
        except Exception as exc:
            warnings.warn(f"RESP processing failed for {vhdr_path.name}: {exc}")

    return res


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def _add_event_lines(ax, onsets, labels=None):
    for i, on in enumerate(onsets):
        ax.axvline(on, color="0.4", ls="--", lw=0.7, alpha=0.7, zorder=1)
        if labels is not None:
            ax.text(on, ax.get_ylim()[1], f" {labels[i]}", fontsize=6, color="0.3",
                    va="top", ha="left", rotation=90)


def plot_run_qc(run_meta, res, events_df, out_png):
    sub, task, acq, run = run_meta["sub"], run_meta["task"], run_meta["acq"], run_meta["run"]
    dur_s = res["dur_s"]
    mods = res["modalities"]

    onsets, labels = [], None
    if events_df is not None and "onset" in events_df.columns:
        onsets = events_df["onset"].to_numpy()
        if "stim_id" in events_df.columns:
            labels = [str(s) for s in events_df["stim_id"].to_numpy()]
        elif "trial_type" in events_df.columns:
            labels = [str(s) for s in events_df["trial_type"].to_numpy()]

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), constrained_layout=True)

    # --- EDA ---
    ax = axes[0]
    if "EDA" in mods:
        s, sr = mods["EDA"]["signals"], mods["EDA"]["sr"]
        t = np.arange(len(s)) / sr
        ax.plot(t, s["EDA_Clean"], lw=0.6, color="C0", label="EDA clean (uS?)")
        ax.plot(t, s["EDA_Tonic"], lw=1.0, color="C1", label="EDA tonic (SCL)")
        scr_idx = np.where(s["SCR_Peaks"].to_numpy() == 1)[0]
        ax.scatter(t[scr_idx], s["EDA_Clean"].to_numpy()[scr_idx], s=12, color="red",
                   zorder=3, label=f"SCR peaks (n={len(scr_idx)})")
        sm = mods["EDA"]["summary"]
        ax.set_title(f"EDA / GSR  |  SCL mu={sm['eda_scl_mean']:.2f}  range[{sm['eda_scl_min']:.2f},"
                     f"{sm['eda_scl_max']:.2f}]  nSCR={sm['eda_n_scr']} ({sm['eda_scr_per_min']:.1f}/min)"
                     f"  flat={sm['eda_pct_flat']*100:.1f}%  [decim {sr:.0f} Hz]", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    else:
        ax.set_title("EDA / GSR  |  (channel missing or processing failed)", fontsize=9)
    ax.set_ylabel("EDA")
    _add_event_lines(ax, onsets, labels)
    ax.set_xlim(0, dur_s)

    # --- ECG: HR over full run ---
    ax = axes[1]
    if "ECG" in mods:
        s, sr = mods["ECG"]["signals"], mods["ECG"]["sr"]
        t = np.arange(len(s)) / sr
        ax.plot(t, s["ECG_Rate"], lw=0.7, color="C3", label="HR (bpm)")
        ax.axhspan(*HR_PLAUSIBLE, color="green", alpha=0.06)
        ax.axhline(HR_PLAUSIBLE[0], color="green", lw=0.5, ls=":")
        ax.axhline(HR_PLAUSIBLE[1], color="green", lw=0.5, ls=":")
        if "ECG_Quality" in s:
            ax2 = ax.twinx()
            ax2.plot(t, s["ECG_Quality"], lw=0.6, color="0.6", alpha=0.7)
            ax2.set_ylabel("ECG quality", color="0.5", fontsize=8)
            ax2.set_ylim(0, 1)
        sm = mods["ECG"]["summary"]
        inv_tag = "  [ECG auto-inverted]" if sm.get("ecg_inverted") else ""
        ax.set_title(f"ECG HR{inv_tag}  |  HR mu={sm['ecg_hr_mean']:.1f} range[{sm['ecg_hr_min']:.0f},"
                     f"{sm['ecg_hr_max']:.0f}]  q mu={sm['ecg_quality_mean']:.2f}  nR={sm['ecg_n_rpeaks']}"
                     f"  flat={sm['ecg_pct_flat']*100:.1f}%  [{sr:.0f} Hz native]", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    else:
        ax.set_title("ECG HR  |  (channel missing or processing failed)", fontsize=9)
    ax.set_ylabel("HR (bpm)")
    _add_event_lines(ax, onsets, labels=None)
    ax.set_xlim(0, dur_s)

    # --- ECG zoom: ~20 s window with R-peaks (sanity that detection works) ---
    ax = axes[2]
    if "ECG" in mods:
        s, sr = mods["ECG"]["signals"], mods["ECG"]["sr"]
        rpeaks = np.asarray(mods["ECG"]["info"].get("ECG_R_Peaks", []), dtype=int)
        t0 = float(onsets[0]) + 5.0 if len(onsets) else dur_s / 3.0
        t0 = min(max(t0, 0.0), max(dur_s - 20.0, 0.0))
        win = (t0, t0 + 20.0)
        i0, i1 = int(win[0] * sr), int(win[1] * sr)
        i1 = min(i1, len(s))
        clean = s["ECG_Clean"].to_numpy()
        t = np.arange(i0, i1) / sr
        ax.plot(t, clean[i0:i1], lw=0.7, color="C3")
        rp_in = rpeaks[(rpeaks >= i0) & (rpeaks < i1)]
        ax.scatter(rp_in / sr, clean[rp_in], s=18, color="k", zorder=3,
                   label=f"R-peaks (n={len(rp_in)})")
        inv_tag = "  [ECG auto-inverted]" if mods["ECG"]["summary"].get("ecg_inverted") else ""
        ax.set_title(f"ECG zoom @ t=[{win[0]:.0f},{win[1]:.0f}] s{inv_tag}  (R-peak detection sanity)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(*win)
    else:
        ax.set_title("ECG zoom  |  (unavailable)", fontsize=9)
    ax.set_ylabel("ECG clean")

    # --- RESP ---
    ax = axes[3]
    if "RESP" in mods:
        s, sr = mods["RESP"]["signals"], mods["RESP"]["sr"]
        t = np.arange(len(s)) / sr
        ax.plot(t, s["RSP_Clean"], lw=0.7, color="C2", label="RESP clean")
        pk = np.where(s["RSP_Peaks"].to_numpy() == 1)[0]
        tr = np.where(s["RSP_Troughs"].to_numpy() == 1)[0]
        ax.scatter(t[pk], s["RSP_Clean"].to_numpy()[pk], s=12, color="darkgreen", marker="^",
                   zorder=3, label=f"peaks (n={len(pk)})")
        ax.scatter(t[tr], s["RSP_Clean"].to_numpy()[tr], s=12, color="purple", marker="v",
                   zorder=3, label=f"troughs (n={len(tr)})")
        if "RSP_RVT" in s:
            ax2 = ax.twinx()
            ax2.plot(t, s["RSP_RVT"], lw=0.8, color="orange", alpha=0.8)
            ax2.set_ylabel("RVT (harrison2021)", color="orange", fontsize=8)
        sm = mods["RESP"]["summary"]
        ax.set_title(f"RESP  |  rate mu={sm['resp_rate_mean']:.1f} cpm range[{sm['resp_rate_min']:.0f},"
                     f"{sm['resp_rate_max']:.0f}]  RVT mu={sm['resp_rvt_mean']:.2f}  nPeaks={sm['resp_n_peaks']}"
                     f"  flat={sm['resp_pct_flat']*100:.1f}%  [decim {sr:.0f} Hz]", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    else:
        ax.set_title("RESP  |  (channel missing or processing failed)", fontsize=9)
    ax.set_ylabel("RESP")
    ax.set_xlabel("time (s)")
    _add_event_lines(ax, onsets, labels)
    ax.set_xlim(0, dur_s)

    n_ev = len(onsets)
    fig.suptitle(
        f"sub-{sub}  ses-{run_meta['ses']}  task-{task} acq-{acq} run-{run}   |   "
        f"sr={res['sr_native']:.1f} Hz   dur={dur_s:.0f} s   |   {n_ev} events   |   "
        f"physio QA (NeuroKit; cvxEDA NOT run)",
        fontsize=11,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=TIER1_SUBJECTS, help="subject ids without 'sub-' prefix")
    ap.add_argument("--tasks", nargs="*", default=None, help="task ids to keep, e.g. 01 02")
    ap.add_argument("--acqs", nargs="*", default=None, help="acquisitions to keep, e.g. a b")
    ap.add_argument("--limit", type=int, default=None, help="process at most N runs (smoke test)")
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    runs = discover_runs(args.subjects, tasks=args.tasks, acqs=args.acqs)
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        print("No runs found for the requested filters.")
        return

    print(f"Processing {len(runs)} run(s) -> {args.outdir}")
    summary_rows = []
    for k, rm in enumerate(runs, 1):
        tag = f"sub-{rm['sub']}_ses-{rm['ses']}_task-{rm['task']}_acq-{rm['acq']}_run-{rm['run']}"
        print(f"[{k}/{len(runs)}] {tag} ...", flush=True)
        try:
            res = process_run(rm["path"])
        except Exception as exc:
            print(f"   FAILED to load/process: {exc}")
            summary_rows.append({**{c: rm[c] for c in ("sub", "ses", "task", "acq", "run")},
                                 "status": f"error: {exc}"})
            continue
        events_df = find_merged_events(rm["sub"], rm["task"], rm["acq"])
        out_png = args.outdir / f"sub-{rm['sub']}" / f"{tag}_desc-qc_physio.png"
        plot_run_qc(rm, res, events_df, out_png)
        row = {**{c: rm[c] for c in ("sub", "ses", "task", "acq", "run")},
               "duration_s": round(res["dur_s"], 1), "sfreq": round(res["sr_native"], 3),
               "n_events": 0 if events_df is None else len(events_df), "status": "ok"}
        for mod in res["modalities"].values():
            row.update(mod["summary"])
        for mname in ("EDA", "ECG", "RESP"):
            if mname not in res["modalities"]:
                row[f"{mname.lower()}_missing"] = True
        summary_rows.append(row)
        print(f"   -> {out_png.relative_to(PROJECT_ROOT)}")

    summary = pd.DataFrame(summary_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "physio_qa_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary -> {summary_path.relative_to(PROJECT_ROOT)}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
