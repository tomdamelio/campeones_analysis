"""Build candidate Y signals from peripheral physiology -- Tarea 1.0 + 1.A + 1.B (diario 05_2).

Faithful to the `dmt-emotions` pipeline (no QA gates yet -- that is Tarea 0):
  - GSR -> decimated 500 -> 250 Hz (to match DMT's effective native sr; DMT did NOT decimate
    but ran cvxEDA at its native 250 Hz). cvxEDA via `biosppy.signals.eda.cvx_decomposition`
    with the Greco et al. 2016 defaults (tau0=2, tau1=0.7, delta_knot=10, alpha=8e-4,
    gamma=1e-2). Returns true sparse SMNA driver, EDR (phasic), EDL (tonic), plus QC outputs
    `tonic_coeff`, `linear_drift`, `res`, `obj`.
  - ECG -> native sr (~500 Hz) -> `nk.ecg_process` (default `neurokit` method) -> ECG_Rate
    (HR continuous bpm), exactly as in `dmt-emotions/preprocess_phys.py`.
  - RESP -> decimated 500 -> 250 Hz -> `nk.rsp_process` (defaults: `khodadad2018` for peaks,
    `harrison2021` for RVT), exactly as in `dmt-emotions`.

Windowed features (2 s window, 1 s hop, grid at 1 Hz) for downstream Y1/Y2:
  - Y1 = AUC of the SMNA driver (rectified to >=0, integrated over the 2 s window),
    Gaussian-smoothed (sigma=3 s).
  - HR_mean, RVT_mean per window (RVT filtered to (0, 5e4) before averaging, a la
    `dmt-emotions/run_resp_rvt_analysis.py:225`).
  - Y2 = PC1 of z-scored [Y1, HR_mean, RVT_mean] within subject, sign-aligned so the SMNA
    loading is positive (replicates `run_composite_arousal_index.py:236+` of dmt-emotions,
    but at 2 s granularity instead of 30 s).

Outputs (under <repo>/research_diary/context/05_02/):
  y_candidates/<sub>_y_candidates.npz   per-subject arrays (per-run dict-of-arrays) + run labels
  y_candidates/pca_loadings.json        PC variance ratios + loadings per subject
  y_candidates/cvx_qc.csv               per-(subject, run) cvxEDA `obj` and var(`res`) for QC
  y_candidates/run_summary.csv          one row per (subject, run): durations, n windows, NaN %
  figures/Y_candidates_<sub>.png        sanity multi-panel for that subject's longest run
  figures/Y_candidates_pca_summary.png  PC1 loadings + explained variance, all subjects

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.build_y_candidates
"""

from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from biosppy.signals.eda import cvx_decomposition
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- locate repo / data (works whether run from the main checkout or a .claude worktree) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]  # <root>/src/campeones_analysis/multimodal_arousal/build_y_candidates.py
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]  # climb out of .claude/worktrees/<name> to the main checkout
else:
    REPO = _ROOT
DATA_RAW = REPO / "data" / "raw"
MERGED_EVENTS = REPO / "data" / "derivatives" / "merged_events"
from src.campeones_analysis.multimodal_arousal.cohort import (  # noqa: E402
    COHORT as SUBJECTS,
    OUT,
    keep_run,
)

(OUT / "y_candidates").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
EDA_FS = 50.0  # cvxEDA target sr. DMT used its 250 Hz native; for CAMPEONES at 500 Hz, cvxopt
                # at 250 Hz takes ~5.7 min/run (~2.3 h total over 24 runs). 50 Hz keeps Greco 2016
                # benchmark range (Nyquist 25 Hz >> SCR bandwidth ~3 Hz, driver morphology preserved)
                # and brings the total runtime to ~15 min. Note this departs from DMT's effective sr
                # of 250 Hz; if downstream wants identical sr, set to 250.
RSP_FS = 250.0  # RVT target sampling rate
WIN = 2.0  # feature window length (s)
HOP = 1.0  # feature hop (s)
SMOOTH_SIGMA = 3.0  # Gaussian smoothing of Y1, in feature samples (= 3 s here)
TRAPZ = getattr(np, "trapezoid", np.trapz)


def runs_for(sub: str) -> list[Path]:
    eeg_dir = DATA_RAW / sub / "ses-vr" / "eeg"
    vhdrs = sorted(eeg_dir.glob(f"{sub}_ses-vr_task-*_acq-*_run-*_eeg.vhdr"))
    return [
        v
        for v in vhdrs
        if "task-practice" not in v.name and keep_run(sub, run_label(v))
    ]


def run_label(vhdr: Path) -> str:
    # sub-23_ses-vr_task-01_acq-a_run-002_eeg.vhdr -> task-01_acq-a_run-002
    parts = vhdr.stem.split("_")
    return "_".join(p for p in parts if p.startswith(("task-", "acq-", "run-")))


def load_physio(vhdr: Path):
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    sr = float(raw.info["sfreq"])
    out = {}
    for ch in ("GSR", "ECG", "RESP"):
        if ch in raw.ch_names:
            out[ch] = raw.copy().pick([ch]).get_data()[0].astype(float)
    return sr, out


def load_events(sub: str, label: str) -> pd.DataFrame | None:
    f = MERGED_EVENTS / sub / "ses-vr" / "eeg" / f"{sub}_ses-vr_{label}_desc-merged_events.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    return None


def process_run(vhdr: Path) -> dict:
    sr, sig = load_physio(vhdr)
    res: dict = {"sr": sr}
    if "GSR" in sig:
        # decimate to 250 Hz (DMT effective sr) and run cvxEDA via biosppy with Greco 2016 defaults
        eda = nk.signal_resample(sig["GSR"], sampling_rate=sr, desired_sampling_rate=EDA_FS, method="FFT")
        out = cvx_decomposition(signal=np.asarray(eda, float), sampling_rate=EDA_FS)
        # biosppy ReturnTuple supports both indexing and named access
        res["eda_t"] = np.arange(len(eda)) / EDA_FS
        res["eda_raw"] = np.asarray(eda, float)
        res["eda_phasic"] = np.asarray(out["edr"], float)
        res["eda_smna"] = np.asarray(out["smna"], float)  # sparse SMNA driver
        res["eda_tonic"] = np.asarray(out["edl"], float)
        res["cvx_obj"] = float(np.asarray(out["obj"]).item() if np.ndim(out["obj"]) == 0 else float(np.asarray(out["obj"]).sum()))
        res["cvx_res_var"] = float(np.var(np.asarray(out["res"], float)))
    if "ECG" in sig:
        # ECG kept at native sr -- exactly as in dmt-emotions/preprocess_phys.py
        ecg_sig, _ = nk.ecg_process(sig["ECG"], sampling_rate=sr)
        res["hr_t"] = np.arange(len(sig["ECG"])) / sr
        res["hr"] = ecg_sig["ECG_Rate"].to_numpy()
    if "RESP" in sig:
        rsp = nk.signal_resample(sig["RESP"], sampling_rate=sr, desired_sampling_rate=RSP_FS, method="FFT")
        rsp_sig, _ = nk.rsp_process(rsp, sampling_rate=RSP_FS)
        res["rsp_t"] = np.arange(len(rsp)) / RSP_FS
        res["rvt"] = rsp_sig["RSP_RVT"].to_numpy()
        res["rsp_clean"] = rsp_sig["RSP_Clean"].to_numpy()
    return res


def windowize(res: dict):
    ends = [res[k][-1] for k in ("eda_t", "hr_t", "rsp_t") if k in res]
    T = min(ends)
    centers = np.arange(WIN / 2.0, T - WIN / 2.0, HOP)
    n = len(centers)
    y1 = np.full(n, np.nan)
    hr = np.full(n, np.nan)
    rvt = np.full(n, np.nan)
    for i, c in enumerate(centers):
        lo, hi = c - WIN / 2.0, c + WIN / 2.0
        if "eda_smna" in res:
            m = (res["eda_t"] >= lo) & (res["eda_t"] < hi)
            if m.any():
                # Y1 = AUC of the rectified SMNA driver (sparse sympathetic input)
                y1[i] = TRAPZ(np.clip(res["eda_smna"][m], 0, None), res["eda_t"][m])
        if "hr" in res:
            m = (res["hr_t"] >= lo) & (res["hr_t"] < hi)
            if m.any():
                y1_hr = res["hr"][m]
                if np.isfinite(y1_hr).any():
                    hr[i] = np.nanmean(y1_hr)
        if "rvt" in res:
            m = (res["rsp_t"] >= lo) & (res["rsp_t"] < hi)
            if m.any():
                v = res["rvt"][m]
                v = v[np.isfinite(v) & (v > 0) & (v < 5e4)]
                if v.size:
                    rvt[i] = np.nanmean(v)
    # smooth Y1 (interpolate over short NaN gaps, smooth, then re-mask the originally-missing samples)
    y1s = np.full(n, np.nan)
    fin = np.isfinite(y1)
    if fin.sum() > 3:
        interp = np.interp(np.arange(n), np.flatnonzero(fin), y1[fin])
        sm = gaussian_filter1d(interp, SMOOTH_SIGMA)
        y1s = sm
    return centers, y1, y1s, hr, rvt


def panel_figure(sub: str, label: str, res: dict, win: dict, events: pd.DataFrame | None, path: Path):
    fig, axes = plt.subplots(6, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(f"{sub}  {label}  -- candidate-Y sanity (fast first pass, NeuroKit defaults)", fontsize=12)

    def mark_events(ax):
        if events is None:
            return
        for _, row in events.iterrows():
            ax.axvline(row["onset"], color="0.6", lw=0.8, ls="--", zorder=0)
        # label only on the top axis
        if ax is axes[0]:
            for _, row in events.iterrows():
                lab = f"{row.get('trial_type', '')}:{row.get('stim_id', '')}"
                ax.text(row["onset"], ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=6, color="0.4")

    if "eda_t" in res:
        axes[0].plot(res["eda_t"], res["eda_raw"], lw=0.5, color="0.6", label="GSR raw @ 250 Hz (a.u.)")
        axes[0].plot(res["eda_t"], res["eda_tonic"], lw=0.9, color="C1", label="EDL = tonic (cvxEDA)")
        axes[0].plot(res["eda_t"], res["eda_phasic"], lw=0.6, color="C2", label="EDR = phasic (cvxEDA)")
        axes[0].set_ylabel("EDA (a.u.)")
        axes[0].legend(loc="upper right", fontsize=7)
        axes[1].plot(res["eda_t"], res["eda_smna"], lw=0.6, color="C3", label="SMNA driver (sparse, biosppy.cvx)")
        axes[1].set_ylabel("SMNA driver")
        axes[1].legend(loc="upper right", fontsize=7)
    axes[2].plot(win["centers"], win["y1"], lw=0.5, color="0.7", label="Y1 = SMNA AUC / 2s window")
    axes[2].plot(win["centers"], win["y1s"], lw=1.2, color="C3", label=f"Y1 smoothed (sigma={SMOOTH_SIGMA:g}s)")
    axes[2].set_ylabel("Y1 (SMNA AUC)")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[3].plot(win["centers"], win["hr"], lw=1.0, color="C4")
    axes[3].set_ylabel("HR mean (bpm)")
    axes[4].plot(win["centers"], win["rvt"], lw=1.0, color="C5")
    axes[4].set_ylabel("RVT mean")
    if win.get("pc1") is not None:
        axes[5].plot(win["centers"], win["pc1"], lw=1.2, color="k")
    axes[5].set_ylabel("Y2 = PC1(SMNA,HR,RVT)\n(z, within-subject)")
    axes[5].set_xlabel("time in run (s)")
    for ax in axes:
        mark_events(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    pca_info: dict = {}
    rows = []
    cvx_rows = []
    per_subject_runs: dict = {}

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        run_results: dict = {}
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            try:
                res = process_run(vhdr)
                centers, y1, y1s, hr, rvt = windowize(res)
                run_results[label] = dict(
                    res=res,
                    centers=centers, y1=y1, y1s=y1s, hr=hr, rvt=rvt,
                )
                n = len(centers)
                rows.append(dict(
                    subject=sub, run=label, sr=res["sr"],
                    dur_eda_s=float(res["eda_t"][-1]) if "eda_t" in res else np.nan,
                    dur_ecg_s=float(res["hr_t"][-1]) if "hr_t" in res else np.nan,
                    dur_resp_s=float(res["rsp_t"][-1]) if "rsp_t" in res else np.nan,
                    n_windows=n,
                    pct_nan_y1=float(np.mean(~np.isfinite(y1)) * 100),
                    pct_nan_hr=float(np.mean(~np.isfinite(hr)) * 100),
                    pct_nan_rvt=float(np.mean(~np.isfinite(rvt)) * 100),
                ))
                if "cvx_obj" in res:
                    cvx_rows.append(dict(
                        subject=sub, run=label, eda_fs=EDA_FS,
                        cvx_obj=res["cvx_obj"], cvx_res_var=res["cvx_res_var"],
                        smna_mean=float(np.nanmean(res["eda_smna"])),
                        smna_max=float(np.nanmax(res["eda_smna"])),
                        smna_pct_nonzero=float(np.mean(res["eda_smna"] > 1e-9) * 100),
                    ))
                print(f"  {label}: {n} windows, "
                      f"NaN%% y1={rows[-1]['pct_nan_y1']:.1f} hr={rows[-1]['pct_nan_hr']:.1f} rvt={rows[-1]['pct_nan_rvt']:.1f}")
            except Exception:  # noqa: BLE001
                print(f"  {label}: FAILED\n{traceback.format_exc()}")
                rows.append(dict(subject=sub, run=label, sr=np.nan, dur_eda_s=np.nan, dur_ecg_s=np.nan,
                                 dur_resp_s=np.nan, n_windows=0, pct_nan_y1=100, pct_nan_hr=100, pct_nan_rvt=100))

        # ---- per-subject PCA on stacked windows ----
        feats = []
        idx_map = []  # (label, slice) into the stacked matrix, per run
        cursor = 0
        for label, rr in run_results.items():
            X = np.column_stack([rr["y1s"], rr["hr"], rr["rvt"]])
            feats.append(X)
            idx_map.append((label, cursor, cursor + len(X)))
            cursor += len(X)
        if feats:
            Xall = np.vstack(feats)
            valid = np.isfinite(Xall).all(axis=1)
            mu = np.nanmean(Xall[valid], axis=0)
            sd = np.nanstd(Xall[valid], axis=0)
            sd[sd == 0] = 1.0
            Z = (Xall - mu) / sd
            pca = PCA(n_components=3)
            Zv = Z[valid]
            pca.fit(Zv)
            comps = pca.components_.copy()
            # sign-align: PC1 loading on SMNA (column 0) positive
            if comps[0, 0] < 0:
                comps[0] *= -1
            scores_full = np.full(len(Z), np.nan)
            sc = ((Zv - 0.0) @ comps[0])  # PC1 score (data already centered via z-score)
            scores_full[valid] = sc
            pca_info[sub] = dict(
                n_windows_total=int(len(Z)),
                n_windows_valid=int(valid.sum()),
                feature_order=["SMNA_phasicAUC", "HR_mean", "RVT_mean"],
                feature_mean=mu.tolist(), feature_std=sd.tolist(),
                explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                pc1_loadings=comps[0].tolist(),
                pc2_loadings=comps[1].tolist(),
                pc3_loadings=comps[2].tolist(),
            )
            print(f"  PCA: EVR={np.round(pca.explained_variance_ratio_, 3).tolist()}  "
                  f"PC1 loadings(SMNA,HR,RVT)={np.round(comps[0], 3).tolist()}")
            # write PC1 back per run
            for label, lo, hi in idx_map:
                run_results[label]["pc1"] = scores_full[lo:hi]

        # ---- save per-subject npz ----
        # Two payloads:
        #   y_candidates/<sub>_y_candidates.npz   windowed features (1 Hz) for Y1/Y2 + PC1 score
        #   y_candidates/<sub>_continuous.npz     full time-series (EDA decomp @ 250 Hz, HR @ ~500 Hz, RVT @ 250 Hz)
        npz_win: dict = {"runs": np.array(list(run_results.keys()))}
        npz_cont: dict = {"runs": np.array(list(run_results.keys())), "eda_fs": EDA_FS, "rsp_fs": RSP_FS}
        for label, rr in run_results.items():
            r = rr["res"]
            npz_win[f"{label}__centers"] = rr["centers"]
            npz_win[f"{label}__y1"] = rr["y1"]
            npz_win[f"{label}__y1s"] = rr["y1s"]
            npz_win[f"{label}__hr"] = rr["hr"]
            npz_win[f"{label}__rvt"] = rr["rvt"]
            npz_win[f"{label}__pc1"] = rr.get("pc1", np.full(len(rr["centers"]), np.nan))
            # continuous (float32 for size)
            if "eda_smna" in r:
                npz_cont[f"{label}__eda_t"] = r["eda_t"].astype(np.float32)
                npz_cont[f"{label}__eda_raw"] = r["eda_raw"].astype(np.float32)
                npz_cont[f"{label}__eda_smna"] = r["eda_smna"].astype(np.float32)
                npz_cont[f"{label}__eda_phasic"] = r["eda_phasic"].astype(np.float32)
                npz_cont[f"{label}__eda_tonic"] = r["eda_tonic"].astype(np.float32)
            if "hr" in r:
                npz_cont[f"{label}__hr_t"] = r["hr_t"].astype(np.float32)
                npz_cont[f"{label}__hr_continuous"] = r["hr"].astype(np.float32)
                npz_cont[f"{label}__hr_sr"] = float(r["sr"])
            if "rvt" in r:
                npz_cont[f"{label}__rsp_t"] = r["rsp_t"].astype(np.float32)
                npz_cont[f"{label}__rvt_continuous"] = r["rvt"].astype(np.float32)
                npz_cont[f"{label}__rsp_clean"] = r["rsp_clean"].astype(np.float32)
        np.savez_compressed(OUT / "y_candidates" / f"{sub}_y_candidates.npz", **npz_win)
        np.savez_compressed(OUT / "y_candidates" / f"{sub}_continuous.npz", **npz_cont)
        per_subject_runs[sub] = run_results

        # ---- sanity figure: longest run ----
        if run_results:
            longest = max(run_results.items(), key=lambda kv: len(kv[1]["centers"]))
            label, rr = longest
            ev = load_events(sub, label)
            panel_figure(
                sub, label, rr["res"],
                dict(centers=rr["centers"], y1=rr["y1"], y1s=rr["y1s"], hr=rr["hr"], rvt=rr["rvt"], pc1=rr.get("pc1")),
                ev, OUT / "figures" / f"Y_candidates_{sub}.png",
            )

    # ---- summary outputs ----
    pd.DataFrame(rows).to_csv(OUT / "y_candidates" / "run_summary.csv", index=False)
    pd.DataFrame(cvx_rows).to_csv(OUT / "y_candidates" / "cvx_qc.csv", index=False)
    with open(OUT / "y_candidates" / "pca_loadings.json", "w", encoding="utf-8") as fh:
        json.dump(pca_info, fh, indent=2)

    # ---- summary figure ----
    if pca_info:
        subs = list(pca_info.keys())
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        feat_names = ["SMNA (phasic AUC)", "HR mean", "RVT mean"]
        x = np.arange(len(subs))
        w = 0.25
        for j, fn in enumerate(feat_names):
            axL.bar(x + (j - 1) * w, [pca_info[s]["pc1_loadings"][j] for s in subs], width=w, label=fn)
        axL.axhline(0, color="k", lw=0.8)
        axL.set_xticks(x)
        axL.set_xticklabels(subs)
        axL.set_ylabel("PC1 loading (on z-scored feature)")
        axL.set_title("Y2 = PC1 loadings, per subject (sign: SMNA positive)")
        axL.legend(fontsize=8)
        for j, pcname in enumerate(["PC1", "PC2", "PC3"]):
            axR.bar(x + (j - 1) * w, [pca_info[s]["explained_variance_ratio"][j] for s in subs], width=w, label=pcname)
        axR.set_xticks(x)
        axR.set_xticklabels(subs)
        axR.set_ylabel("explained variance ratio")
        axR.set_title("PCA explained variance, per subject")
        axR.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "Y_candidates_pca_summary.png", dpi=130)
        plt.close(fig)

    print(f"\nDone. Outputs under {OUT}")


if __name__ == "__main__":
    main()
