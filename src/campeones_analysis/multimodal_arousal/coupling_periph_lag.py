"""Perfil de lags del acoplamiento EEG-banda <-> periferia (SMNA/phasic/HR/RVT), N=6.

Generaliza coupling_periph_multitarget.py de lag-0 a un BARRIDO de lags, asumiendo que hay
diferencia temporal entre la señal central (EEG) y las periféricas autonómicas (efector lento).

Por sujeto, target, banda y ROI ({PO, edge}): perfil rho(lag) de Spearman (Pearson sobre rangos
estandarizados) de -LAG_MAX a +LAG_MAX. Convención: lag>0 = EEG ADELANTA a la periferia
(EEG(t) vs periferia(t+lag)). Reporta lag-del-pico, rho-del-pico, rho a lag 0, y la consistencia
del lag del pico cross-sujeto. Null circular-shift SOBRE EL MÁXIMO-sobre-lags (controla el sesgo
de selección por múltiples lags + la autocorrelación), a nivel grupo, para PO.

Reusa continuous_band.build_subject_continuous (envelopes Hilbert log por banda a 12.5 Hz).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.coupling_periph_lag
  ... --subjects sub-23   (smoke)
"""

from __future__ import annotations

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR, OUT
from src.campeones_analysis.multimodal_arousal.continuous_band import (
    BANDS,
    COMMON_FS,
    build_subject_continuous,
)

warnings.filterwarnings("ignore")

EDGE = ["FT9", "TP9", "T7", "T8", "P7", "P8"]
TARGETS = ["smna", "phasic", "hr", "rvt"]
USE_BANDS = list(BANDS)
LAG_MAX_S = 8.0
LAG_STEP_S = 0.5
LAGS_S = np.arange(-LAG_MAX_S, LAG_MAX_S + 1e-9, LAG_STEP_S)
LAGS_K = np.round(LAGS_S * COMMON_FS).astype(int)  # en samples (12.5 Hz)
N_PERM = 200
RNG = np.random.default_rng(20260605)


def _zrank(x):
    """Rangos estandarizados (media 0, std 1) -> Pearson sobre estos = Spearman."""
    r = rankdata(x).astype(float)
    s = r.std()
    return (r - r.mean()) / s if s > 0 else r - r.mean()


def _xcorr_profile(zr_eeg_runs, zr_tgt_runs):
    """rho(lag) combinando runs: para cada lag, sum de productos / n total sobre el solape.
    lag k>0 = EEG adelanta (eeg[t] vs tgt[t+k])."""
    out = np.zeros(len(LAGS_K))
    for j, k in enumerate(LAGS_K):
        num = 0.0
        cnt = 0
        for ze, zt in zip(zr_eeg_runs, zr_tgt_runs):
            n = len(ze)
            if k >= 0:
                a, b = ze[: n - k], zt[k:] if k > 0 else zt
            else:
                a, b = ze[-k:], zt[: n + k]
            m = min(len(a), len(b))
            if m > 10:
                num += float(np.dot(a[:m], b[:m]))
                cnt += m
        out[j] = num / cnt if cnt else np.nan
    return out


def _shift_runs(zr_tgt_runs):
    """Circular-shift independiente por run (preserva autocorrelación marginal)."""
    out = []
    for zt in zr_tgt_runs:
        s = int(RNG.integers(1, max(2, len(zt))))
        out.append(np.roll(zt, s))
    return out


def subject_profiles(sub):
    runs = build_subject_continuous(sub)
    if not runs:
        return None
    cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
    # series por run, rankeadas/estandarizadas
    eeg = {b: {"PO": [], "edge": []} for b in USE_BANDS}
    tgt = {t: [] for t in TARGETS}
    have = {t: True for t in TARGETS}
    for label, r in runs.items():
        times, ch = r["times"], r["ch"]
        edge_idx = [ch.index(c) for c in EDGE if c in ch]
        per_tgt = {"smna": r["eda"], "phasic": r["eda_phasic"]}
        try:
            per_tgt["hr"] = np.interp(times, np.asarray(cont[f"{label}__hr_t"], float),
                                      np.asarray(cont[f"{label}__hr_continuous"], float))
        except Exception:
            have["hr"] = False
        try:
            per_tgt["rvt"] = np.interp(times, np.asarray(cont[f"{label}__rsp_t"], float),
                                       np.asarray(cont[f"{label}__rvt_continuous"], float))
        except Exception:
            have["rvt"] = False
        for b in USE_BANDS:
            eeg[b]["PO"].append(_zrank(r["bands"][b]["po"]))
            eeg[b]["edge"].append(_zrank(r["bands"][b]["all"][:, edge_idx].mean(axis=1)))
        for t in TARGETS:
            if t in per_tgt:
                tgt[t].append(_zrank(per_tgt[t]))
    prof = {}
    for t in TARGETS:
        for b in USE_BANDS:
            for roi in ("PO", "edge"):
                if len(tgt[t]) == len(eeg[b][roi]) and len(tgt[t]) > 0:
                    prof[(t, b, roi)] = _xcorr_profile(eeg[b][roi], tgt[t])
    return dict(prof=prof, eeg=eeg, tgt=tgt, have=have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--nperm", type=int, default=N_PERM)
    args = ap.parse_args()
    subs = args.subjects if args.subjects else list(COHORT)
    print("=" * 78)
    print(f"coupling_periph_lag :: lags {-LAG_MAX_S}..{LAG_MAX_S}s (lag>0 = EEG adelanta)  subjects={subs}")
    print("=" * 78, flush=True)

    per_sub = {}
    for sub in subs:
        print(f"\n=== {sub} ===", flush=True)
        d = subject_profiles(sub)
        if d is None:
            print(f"  {sub}: no data", flush=True)
            continue
        per_sub[sub] = d
        print(f"  {sub}: perfiles OK ({len(d['prof'])} combinaciones)", flush=True)

    keys = sorted({k for d in per_sub.values() for k in d["prof"]})
    # perfil grupo = media sobre sujetos
    group = {}
    for k in keys:
        arrs = [per_sub[s]["prof"][k] for s in per_sub if k in per_sub[s]["prof"]]
        group[k] = dict(mean=np.nanmean(arrs, axis=0), n=len(arrs),
                        per_subj_peak_lag=[float(LAGS_S[int(np.nanargmax(np.abs(a)))]) for a in arrs])

    # null circular-shift sobre el MÁXIMO-sobre-lags, a nivel grupo, solo PO
    summary = {}
    for t in TARGETS:
        summary[t] = {}
        for b in USE_BANDS:
            for roi in ("PO", "edge"):
                k = (t, b, roi)
                if k not in group:
                    continue
                prof = group[k]["mean"]
                lag0 = float(prof[np.argmin(np.abs(LAGS_S))])
                jpk = int(np.nanargmax(np.abs(prof)))
                entry = dict(lag0_rho=round(lag0, 4),
                             peak_lag_s=float(LAGS_S[jpk]), peak_rho=round(float(prof[jpk]), 4),
                             n=group[k]["n"],
                             per_subj_peak_lag=group[k]["per_subj_peak_lag"])
                if roi == "PO" and args.nperm > 0:
                    obs_max = float(np.nanmax(np.abs(prof)))
                    null_max = np.empty(args.nperm)
                    subj_with = [s for s in per_sub if k in per_sub[s]["prof"]]
                    for p in range(args.nperm):
                        arrs = []
                        for s in subj_with:
                            zr_eeg = per_sub[s]["eeg"][b][roi]
                            zr_tgt = _shift_runs(per_sub[s]["tgt"][t])
                            arrs.append(_xcorr_profile(zr_eeg, zr_tgt))
                        null_max[p] = np.nanmax(np.abs(np.nanmean(arrs, axis=0)))
                    entry["p_maxlag_perm"] = float((np.sum(null_max >= obs_max) + 1) / (args.nperm + 1))
                summary[t][f"{b}__{roi}"] = entry

    sdir = OUT / "eeg_smna_coupling"
    (sdir / "tables").mkdir(parents=True, exist_ok=True)
    with open(sdir / "tables" / "periph_lag_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # figura: filas=targets, cols=bandas; PO sólido vs edge punteado
    fig, axes = plt.subplots(len(TARGETS), len(USE_BANDS), figsize=(13, 11), sharex=True)
    TLAB = {"smna": "SMNA", "phasic": "EDA-phasic", "hr": "HR", "rvt": "Respiración (RVT)"}
    for i, t in enumerate(TARGETS):
        for j, b in enumerate(USE_BANDS):
            ax = axes[i, j]
            for roi, ls, col in (("PO", "-", "crimson"), ("edge", "--", "0.45")):
                k = (t, b, roi)
                if k in group:
                    ax.plot(LAGS_S, group[k]["mean"], ls, color=col, lw=1.4, label=roi)
            ax.axvline(0, color="k", lw=0.5)
            ax.axhline(0, color="k", lw=0.5)
            po = summary.get(t, {}).get(f"{b}__PO", {})
            if po:
                ax.plot(po["peak_lag_s"], po["peak_rho"], "o", color="crimson", ms=5)
                pp = po.get("p_maxlag_perm")
                ax.set_title(f"{TLAB[t]} | {b}  (pico {po['peak_lag_s']:+.1f}s"
                             + (f", p={pp:.3f})" if pp is not None else ")"), fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{TLAB[t]}\nrho", fontsize=8)
            if i == len(TARGETS) - 1:
                ax.set_xlabel("lag (s)   [>0 = EEG adelanta]", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Perfil de lags EEG band-power <-> periferia (N=6) — PO (rojo) vs edge (gris). "
                 "Pico marcado; p = null circular-shift sobre max-sobre-lags (PO).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = sdir / "figures" / "periph_lag_profiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n=== RESUMEN (PO): lag0 -> pico@lag (p) por target x banda ===")
    for t in TARGETS:
        for b in USE_BANDS:
            e = summary.get(t, {}).get(f"{b}__PO")
            if e:
                pp = e.get("p_maxlag_perm")
                print(f"  {t:7s} {b:6s}: lag0={e['lag0_rho']:+.3f}  pico={e['peak_rho']:+.3f}@{e['peak_lag_s']:+.1f}s"
                      + (f"  p={pp:.3f}" if pp is not None else "")
                      + f"  | peak-lags suj={e['per_subj_peak_lag']}", flush=True)
    print(f"\nOutputs -> {sdir}", flush=True)


if __name__ == "__main__":
    main()
