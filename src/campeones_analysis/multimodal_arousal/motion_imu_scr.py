"""2.2 — Métrica de movimiento desde el acelerómetro (X/Y/Z), por época.

Frente-DELTA (+ refinamiento usuario: también GAMMA y OFFSET). Covariado de movimiento de
cabeza/casco por época para 2.5. Enzo: "sacar una métrica de cuánto se movieron en base al
giroscopio... la derivada y a eso la varianza" (el sensor es acelerómetro lineal 3D, no giro).

Vías por las que el movimiento toca el EEG:
  - mecánica lenta (0.5-8 Hz) -> toca DELTA (la pro-señal) = el control de Q-delta.
  - sub-banda 0.5-2 Hz -> solapa el scalp-sweat (Kalevo 2020) = control de 2.6/Q-delta.
  - magnitud de movimiento (var_jerk banda ancha) -> micro-movimiento de electrodo + co-EMG ->
    toca GAMMA y OFFSET (refinamiento usuario 2026-06-07). NO se band-limita el IMU a 30-40
    (el acelerómetro no tiene energía ahí); se usa el escalar de magnitud.

Covariados por época: var_jerk / rms_jerk (banda ancha), var_jerk_0p5_8, var_jerk_0p5_2,
y var de jerk por eje (x/y/z, el vertical suele ser el más predictivo, Kline 2015).
jerk = derivada de la aceleración a(t)=sqrt(x²+y²+z²).

Canales X/Y/Z (MISC, BrainProducts 3D Acceleration Sensor) -> NO pasan por CAR/ICA/drop:
GENERACIÓN inmune a Track B; la correlación contra la VD-EEG final se re-corre tras Track B.
Alineamiento 1:1 al cache panel_psd (mismo RNG/orden/boundary-drop que 2.1, validado por tnorm).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.motion_imu_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.stats import mannwhitneyu, spearmanr

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED, load_cache
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    TMAX,
    TMIN,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import attach_montage_and_drop_no_pos

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_2_motion_imu"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

ACC_CHS = ["X", "Y", "Z"]
COV_COLS = ["var_jerk", "rms_jerk", "var_jerk_0p5_8", "var_jerk_0p5_2",
            "var_jerk_x", "var_jerk_y", "var_jerk_z"]


def _wvar(sig: np.ndarray, sf: float, t0: float) -> float:
    a, b = int((t0 + TMIN) * sf), int((t0 + TMAX) * sf)
    a, b = max(0, a), min(len(sig), b)
    return float(np.var(sig[a:b])) if b - a > 2 else np.nan


def _jerk(a: np.ndarray, sf: float) -> np.ndarray:
    return np.gradient(a) * sf


def _run_signals(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    if not set(ACC_CHS).issubset(set(raw.ch_names)):
        return None
    g = lambda c: raw.get_data(picks=[c])[0]
    x, y, z = g("X"), g("Y"), g("Z")
    a = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return dict(
        sf=sf,
        jerk=_jerk(a, sf),
        jerk_0p5_8=_jerk(filter_data(a[None, :].astype(float), sf, 0.5, 8.0, verbose="ERROR")[0], sf),
        jerk_0p5_2=_jerk(filter_data(a[None, :].astype(float), sf, 0.5, 2.0, verbose="ERROR")[0], sf),
        jerk_x=_jerk(x, sf), jerk_y=_jerk(y, sf), jerk_z=_jerk(z, sf),
    )


def _rows(sig, onsets, sub, label, cond):
    sf = sig["sf"]
    out = []
    for t0 in onsets:
        out.append(dict(
            subject=sub, run=label, condition=cond, tnorm=np.nan,
            var_jerk=_wvar(sig["jerk"], sf, t0),
            rms_jerk=np.sqrt(_wvar(sig["jerk"], sf, t0)),
            var_jerk_0p5_8=_wvar(sig["jerk_0p5_8"], sf, t0),
            var_jerk_0p5_2=_wvar(sig["jerk_0p5_2"], sf, t0),
            var_jerk_x=_wvar(sig["jerk_x"], sf, t0),
            var_jerk_y=_wvar(sig["jerk_y"], sf, t0),
            var_jerk_z=_wvar(sig["jerk_z"], sf, t0),
        ))
    return out


def main() -> None:
    print("=" * 78)
    print("motion_imu_scr :: 2.2 covariado de movimiento (acelerómetro X/Y/Z) por época (N=6)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    cache, _, _ = load_cache("uniform")
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)
    all_rows, align = [], []

    for sub in COHORT:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_rows, sil_rows, real_tn, sil_tn = [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw_eeg = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw_eeg.pick("eeg"); attach_montage_and_drop_no_pos(raw_eeg)
            raw_eeg.filter(1.0, 40.0, verbose="ERROR"); raw_eeg.resample(250.0, verbose="ERROR")
            dur = float(raw_eeg.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = filter_clean_onsets(detect_scr_onsets_s(eda, EDA_FS)[detect_scr_onsets_s(eda, EDA_FS) < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            real_kept = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sil_kept = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            sig = _run_signals(vhdr)
            if sig is None:
                print(f"  {sub} {label}: faltan X/Y/Z -> skip", flush=True)
                continue
            real_rows += _rows(sig, real_kept, sub, label, 1)
            sil_rows += _rows(sig, sil_kept, sub, label, 0)
            real_tn += list(real_kept / dur); sil_tn += list(sil_kept / dur)
            print(f"  {sub} {label}: SCR={len(real_kept)} no-SCR={len(sil_kept)} (sf={sig['sf']:.0f})", flush=True)

        for r, tn in zip(real_rows, real_tn): r["tnorm"] = tn
        for r, tn in zip(sil_rows, sil_tn): r["tnorm"] = tn
        all_rows += real_rows + sil_rows
        my_tn = np.array(real_tn + sil_tn)
        if sub in cache:
            ctn = cache[sub][2]; ok = len(my_tn) == len(ctn)
            align.append(dict(subject=sub, n_mine=len(my_tn), n_cache=len(ctn), match=ok,
                              max_dev=round(float(np.max(np.abs(my_tn - ctn))), 6) if ok else np.nan))

    df = pd.DataFrame(all_rows)
    df.to_csv(TBL_DIR / "motion_covariate.csv", index=False)
    pd.DataFrame(align).to_csv(TBL_DIR / "alignment_check.csv", index=False)

    # within-subject SCR vs no-SCR + corr con tiempo (sanity colinealidad, F8)
    summ = []
    for col in COV_COLS:
        rels, npos, n, tnc = [], 0, 0, []
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna().to_numpy()
            b = sd[sd.condition == 0][col].dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            ma, mb = np.median(a), np.median(b)
            rels.append((ma - mb) / mb if abs(mb) > 1e-12 else np.nan); npos += int(ma > mb); n += 1
            if col == "var_jerk":
                bsd = sd[sd.condition == 0]
                tnc.append(spearmanr(bsd[col].to_numpy(), bsd["tnorm"].to_numpy()).correlation)
        summ.append(dict(covariate=col, n_subj=n,
                         mean_rel_pct=round(100 * np.nanmean(rels), 1) if rels else np.nan,
                         n_pos=f"{npos}/{n}",
                         corr_tnorm=round(float(np.nanmean(tnc)), 3) if tnc else ""))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(TBL_DIR / "motion_within_subject.csv", index=False)

    print("\n=== Alineamiento ===")
    print(pd.DataFrame(align).to_string(index=False))
    print("\n=== Within-subject SCR vs no-SCR (movimiento) ===")
    print(summ_df.to_string(index=False))

    _plot(df)
    print(f"\nCSV -> {TBL_DIR / 'motion_covariate.csv'}\n[2.2] done", flush=True)


def _plot(df):
    cols = [("var_jerk", "var(jerk) banda ancha"),
            ("var_jerk_0p5_8", "var(jerk) 0.5-8 Hz (delta/movimiento)"),
            ("var_jerk_0p5_2", "var(jerk) 0.5-2 Hz (scalp-sweat band)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, cols):
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna(); b = sd[sd.condition == 0][col].dropna()
            if not len(a) or not len(b):
                continue
            ax.plot([0, 1], [b.median(), a.median()], "-o", color=SUBJ_COLORS[sub], lw=1.5, ms=5, label=sub)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no-SCR", "SCR"]); ax.set_ylabel(col)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("2.2 Movimiento (acelerómetro): SCR vs no-SCR por sujeto\n"
                 "(sube en SCR = más movimiento en las épocas SCR = confound)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "motion_bysubject.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
