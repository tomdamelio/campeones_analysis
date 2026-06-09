"""R3.2 test 1 — ¿la desync alfa es específica del SMNA/EDA o arousal autonómico COMÚN?

R3.1: la desync alfa escala con el SMNA-AUC (6/6, sobrevive el control ocular/movimiento). Pero
SMNA, HR y RVT son índices del mismo arousal -> ¿la relación alfa-SMNA es específica (acoplamiento
EDA<->EEG directo) o solo refleja un arousal autonómico común? Test 1 = SOLO las 3 fisiológicas
(sin joystick todavía). Por época: alfa periódica posterior + SMNA-AUC + HR_mean + RVT_mean.

  - raw Spearman(alfa, {SMNA, HR, RVT}): ¿alfa se relaciona con cada una? RVT = control de
    especificidad (en el frente continuo dio null -> NO debería relacionar).
  - PARCIAL(alfa, SMNA | HR, RVT, tiempo): ¿el acoplamiento alfa-SMNA sobrevive controlando el
    arousal autonómico general (HR) y la respiración? Sobrevive = específico EDA; cae = común.
  - Commonality: R²(alfa ~ SMNA+HR+RVT) y único de cada una (único-SMNA = relación directa).

HR/RVT se re-derivan del RAW (nk.ecg_process/rsp_process), promediados sobre la ventana [0, POST_S]
de cada época. Alfa del cache; SMNA-AUC del driver `eda_smna`. Alineado 1:1 al cache. Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_physio_partial
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import stats

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.build_y_candidates import load_physio
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED, load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    POST_S,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_2_specificity"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_ROOT = REPO / "data" / "raw"
RSP_FS = 250.0
POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]


def _alpha_post(psd, freqs, post_idx):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-30"])
    m = (f >= 8.0) & (f < 13.0)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)[:, post_idx].mean(axis=1)


def _hr_rvt(label, sub):
    """HR y RVT continuos del RAW (mismo procesamiento que build_y_candidates)."""
    raw_vhdr = RAW_ROOT / sub / "ses-vr" / "eeg" / f"{sub}_ses-vr_{label}_eeg.vhdr"
    if not raw_vhdr.exists():
        return None
    sr, sig = load_physio(raw_vhdr)
    out = {}
    if "ECG" in sig:
        ecg, _ = nk.ecg_process(sig["ECG"], sampling_rate=sr)
        out["hr"], out["hr_t"] = ecg["ECG_Rate"].to_numpy(), np.arange(len(sig["ECG"])) / sr
    if "RESP" in sig:
        rsp = nk.signal_resample(sig["RESP"], sampling_rate=sr, desired_sampling_rate=RSP_FS, method="FFT")
        rsig, _ = nk.rsp_process(rsp, sampling_rate=RSP_FS)
        out["rvt"], out["rsp_t"] = rsig["RSP_RVT"].to_numpy(), np.arange(len(rsp)) / RSP_FS
    return out


def _smna_auc(smna, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(smna), i1)
    return float(np.trapz(np.clip(smna[i0:i1], 0, None), dx=1.0 / EDA_FS)) if i1 - i0 >= 2 else np.nan


def _win_mean(sig, t, t0):
    m = (t >= t0) & (t < t0 + POST_S)
    return float(np.nanmean(sig[m])) if m.any() else np.nan


def _resid(v, C):
    A = np.column_stack([C, np.ones(len(C))])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _partial(x, y, C):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    return float(stats.spearmanr(_resid(rx, C), _resid(ry, C)).correlation)


def _r2(y, X):
    A = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - np.var(resid) / (np.var(y) + 1e-30)


def main():
    print("=" * 78)
    print("alpha_physio_partial :: R3.2 test 1 — alfa vs {SMNA, HR, RVT} (específico vs común)")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    post_idx = [ch.index(c) for c in POSTERIOR if c in ch]
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)
    rows, comm = [], []

    for sub in COHORT:
        psd, y, tn = cache[sub]
        alpha = _alpha_post(psd, freqs, post_idx)
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real, sil = {"auc": [], "hr": [], "rvt": [], "tn": []}, {"auc": [], "hr": [], "rvt": [], "tn": []}
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(250.0, verbose="ERROR")
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)
            phys = _hr_rvt(label, sub) or {}
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            silt = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                          fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            for kept, store in [(onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)], real),
                                (silt[(silt + TMIN > 0) & (silt + TMAX < dur)], sil)]:
                for t in kept:
                    store["auc"].append(_smna_auc(smna, t))
                    store["hr"].append(_win_mean(phys["hr"], phys["hr_t"], t) if "hr" in phys else np.nan)
                    store["rvt"].append(_win_mean(phys["rvt"], phys["rsp_t"], t) if "rvt" in phys else np.nan)
                    store["tn"].append(t / dur)
        auc = np.array(real["auc"] + sil["auc"]); hr = np.array(real["hr"] + sil["hr"])
        rvt = np.array(real["rvt"] + sil["rvt"]); mytn = np.array(real["tn"] + sil["tn"])
        if len(auc) != len(tn) or np.max(np.abs(mytn - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN -> skip", flush=True); continue

        m = np.isfinite(auc) & np.isfinite(hr) & np.isfinite(rvt) & np.isfinite(alpha)
        a, S, H, R, T = alpha[m], auc[m], hr[m], rvt[m], tn[m]
        raw_s = stats.spearmanr(a, S).correlation
        raw_h = stats.spearmanr(a, H).correlation
        raw_r = stats.spearmanr(a, R).correlation
        par_s = _partial(a, S, np.column_stack([H, R, T]))   # alfa-SMNA controlando HR+RVT+tiempo
        rows.append(dict(subject=sub, n=int(m.sum()),
                         raw_SMNA=round(float(raw_s), 3), raw_HR=round(float(raw_h), 3),
                         raw_RVT=round(float(raw_r), 3),
                         partial_SMNA_given_HR_RVT=round(par_s, 3)))
        # commonality: único de cada fisiológica para predecir alfa (sobre rangos)
        ar = stats.rankdata(a); Z = np.column_stack([stats.rankdata(S), stats.rankdata(H), stats.rankdata(R)])
        full = _r2(ar, Z)
        uniq = {nm: full - _r2(ar, np.delete(Z, i, axis=1)) for i, nm in enumerate(["SMNA", "HR", "RVT"])}
        comm.append(dict(subject=sub, R2_full=round(full, 4),
                         uniq_SMNA=round(uniq["SMNA"], 4), uniq_HR=round(uniq["HR"], 4),
                         uniq_RVT=round(uniq["RVT"], 4)))
        print(f"  {sub}: raw SMNA={raw_s:+.3f} HR={raw_h:+.3f} RVT={raw_r:+.3f}  "
              f"partial SMNA|HR,RVT={par_s:+.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(TBL_DIR / "alpha_physio_partial.csv", index=False)
    cdf = pd.DataFrame(comm); cdf.to_csv(TBL_DIR / "alpha_physio_commonality.csv", index=False)

    def _grp(col):
        v = df[col].to_numpy(float); z = np.arctanh(np.clip(v, -0.999, 0.999))
        _, p = stats.ttest_1samp(z, 0.0)
        return v.mean(), int((v < 0).sum()), p

    print("\n=== R3.2 test 1: alfa vs fisiológicas (within-subject) ===")
    for col in ["raw_SMNA", "raw_HR", "raw_RVT", "partial_SMNA_given_HR_RVT"]:
        mn, nneg, p = _grp(col)
        print(f"  {col:28s}: mean={mn:+.3f}  neg={nneg}/6  p_group={p:.3f}  "
              f"por_suj={[f'{x:+.2f}' for x in df[col]]}", flush=True)
    print("\n=== Commonality (R² único por fisiológica para predecir alfa) ===")
    print(cdf.to_string(index=False), flush=True)
    print(f"  medias: único SMNA={cdf['uniq_SMNA'].mean():.4f}  HR={cdf['uniq_HR'].mean():.4f}  "
          f"RVT={cdf['uniq_RVT'].mean():.4f}", flush=True)

    _plot(df, cdf)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.2 test 1] done", flush=True)


def _plot(df, cdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cols = ["raw_SMNA", "raw_HR", "raw_RVT", "partial_SMNA_given_HR_RVT"]
    x = np.arange(len(cols))
    for k, sub in enumerate(df["subject"]):
        axes[0].plot(x + np.linspace(-0.2, 0.2, len(df))[k], df.iloc[k][cols].to_numpy(float),
                     "o", color=SUBJ_COLORS.get(sub, "C0"), ms=6, label=sub)
    axes[0].plot(x, [df[c].mean() for c in cols], "_", color="k", ms=30, mew=2.5)
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["raw\nSMNA", "raw\nHR", "raw\nRVT", "partial\nSMNA|HR,RVT"], fontsize=8)
    axes[0].set_ylabel("Spearman con alfa periódica"); axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("¿alfa específica del SMNA o arousal común?", fontsize=10)
    # commonality
    u = cdf[["uniq_SMNA", "uniq_HR", "uniq_RVT"]].mean()
    axes[1].bar(["único\nSMNA", "único\nHR", "único\nRVT"], u.to_numpy(), color=["C0", "C3", "C2"])
    axes[1].set_ylabel("R² único (medio) para predecir alfa")
    axes[1].set_title("Commonality: varianza única de alfa por fisiológica", fontsize=10)
    fig.suptitle("R3.2 test 1 — especificidad de la desync alfa (SMNA/HR/RVT, sin joystick)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "alpha_physio_specificity.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
