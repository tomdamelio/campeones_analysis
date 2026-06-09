"""R3.3 (1) cross-check CONTINUO — cross-correlación envelope-alfa <-> SMNA continuo + Granger AR.

El per-época (`lag_sweep_alpha_smna.py`) dio brain->body (alfa anticipa el SMNA, asim 6/6, mediana
tau*=-1 s). Este es el cross-check con un método INDEPENDIENTE (continuo, no event-locked, resolución
fina): envelope Hilbert de alfa-PO vs SMNA continuo @12.5 Hz (reusa `continuous_band`), por run,
run-respecting. Un pico de lag concordante en las dos vías = robusto.

Convención de lag (continua): tau>0 = SMNA SIGUE a la alfa (alfa LIDERA). El acoplamiento es NEGATIVO
(desync: más alfa -> menos SMNA), así que se busca el lag del MÍNIMO de la cross-correlación:
  xcorr más negativa en tau>0 -> alfa lidera -> brain->body  (concordante con per-época tau*<0)
  xcorr más negativa en tau<0 -> alfa sigue  -> body->brain

Dos lentes:
  (A) Cross-correlación xcorr(tau), tau ∈ ±6 s, run-respecting, promediada cross-run (peso=n).
      Null subject-respecting + max-statistic: circular-shift del SMNA dentro del run (preserva
      autocorrelación), recomputa la curva, toma su MIN. 1000 perms. p por sujeto.
  (B) Granger-flavored AR (direccional, descuenta la autocorrelación del SMNA): dR2 forward
      (pasado de alfa -> SMNA, sobre AR del SMNA) vs dR2 reverse (pasado de SMNA -> alfa, sobre AR
      de alfa). forward>reverse -> alfa lidera -> brain->body. LORO R2.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.lag_xcorr_alpha_smna
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.continuous_band import (
    COMMON_FS,
    build_subject_continuous,
)

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_3_directionality"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

LAG_MAX_S = 6.0
LAG_GRID = np.arange(-int(round(LAG_MAX_S * COMMON_FS)), int(round(LAG_MAX_S * COMMON_FS)) + 1)
LAG_S = LAG_GRID / COMMON_FS
AR_LAGS_S = [0.5, 1.0, 1.5, 2.0, 3.0]
ALPHAS = np.logspace(-1, 4, 6)
N_PERM = 1000


def _xcorr_run(a, s, lags):
    """corr(a[t], s[t+L]) por lag L (en muestras). L>0 = s en el futuro de a (a lidera)."""
    out = np.full(len(lags), np.nan)
    for k, L in enumerate(lags):
        if L > 0:
            x, y = a[:-L], s[L:]
        elif L < 0:
            x, y = a[-L:], s[:L]
        else:
            x, y = a, s
        if len(x) > 10 and x.std() > 0 and y.std() > 0:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def _xcorr_subject(runs, lags, shift_smna=None, rng=None):
    """Cross-correlación promediada cross-run (peso = n). shift_smna: circular-shift por run (null)."""
    acc = np.zeros(len(lags)); wsum = np.zeros(len(lags))
    for r in runs.values():
        a = r["bands"]["alpha"]["po"]; s = r["eda"]
        if shift_smna:
            sh = int(rng.integers(1, max(2, len(s))))
            s = np.roll(s, sh)
        rr = _xcorr_run(a, s, lags)
        w = len(a)
        ok = np.isfinite(rr)
        acc[ok] += rr[ok] * w; wsum[ok] += w
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), np.nan)


def _lag_design(x, lags_samp):
    """Diseño con lags de PASADO (L muestras): col k = x[t-L_k]. NaN en el borde inicial del run."""
    T = len(x); X = np.full((T, len(lags_samp)), np.nan)
    for k, L in enumerate(lags_samp):
        X[L:, k] = x[:T - L]
    return X


def _loro_r2(Xparts_runs, y_runs):
    """LORO R2: cada run es test una vez. Xparts_runs/y_runs: listas por run (ya con lags+valid)."""
    r2s = []
    for te in range(len(y_runs)):
        Xtr = np.vstack([Xparts_runs[i] for i in range(len(y_runs)) if i != te])
        ytr = np.concatenate([y_runs[i] for i in range(len(y_runs)) if i != te])
        Xte, yte = Xparts_runs[te], y_runs[te]
        if len(ytr) < 20 or len(yte) < 20:
            continue
        best, bestr2 = None, -np.inf
        for al in ALPHAS:
            m = Ridge(alpha=al).fit(Xtr, ytr)
            ss = ((yte - yte.mean()) ** 2).sum()
            r2 = 1.0 - ((yte - m.predict(Xte)) ** 2).sum() / ss if ss > 0 else np.nan
            if np.isfinite(r2) and r2 > bestr2:
                bestr2, best = r2, r2
        if best is not None:
            r2s.append(best)
    return float(np.mean(r2s)) if r2s else np.nan


def _granger(runs):
    """dR2 forward (alfa pasado -> SMNA | AR-SMNA) vs reverse (SMNA pasado -> alfa | AR-alfa)."""
    lags = [int(round(l * COMMON_FS)) for l in AR_LAGS_S]
    A_runs, S_runs = [], []
    for r in runs.values():
        A_runs.append(r["bands"]["alpha"]["po"]); S_runs.append(r["eda"])

    def assemble(target_runs, ar_src, extra_src):
        ar_p, full_p, y_p = [], [], []
        for a_t, ar_s, ex_s in zip(target_runs, ar_src, extra_src):
            AR = _lag_design(ar_s, lags); EX = _lag_design(ex_s, lags)
            v = np.all(np.isfinite(AR), axis=1) & np.all(np.isfinite(EX), axis=1)
            ar_p.append(AR[v]); full_p.append(np.hstack([AR[v], EX[v]])); y_p.append(a_t[v])
        return ar_p, full_p, y_p

    ar_p, full_p, y_p = assemble(S_runs, S_runs, A_runs)   # forward: predecir SMNA
    r2_ar_f = _loro_r2(ar_p, y_p); r2_full_f = _loro_r2(full_p, y_p)
    ar_p, full_p, y_p = assemble(A_runs, A_runs, S_runs)   # reverse: predecir alfa
    r2_ar_r = _loro_r2(ar_p, y_p); r2_full_r = _loro_r2(full_p, y_p)
    return dict(dr2_fwd=r2_full_f - r2_ar_f, dr2_rev=r2_full_r - r2_ar_r,
                r2_ar_fwd=r2_ar_f, r2_full_fwd=r2_full_f)


def main():
    print("=" * 78)
    print("lag_xcorr_alpha_smna :: cross-check CONTINUO (envelope alfa-PO vs SMNA) + Granger AR")
    print("=" * 78, flush=True)

    rng = np.random.default_rng(20260609)
    rows, curves = [], []
    subj_runs = {}
    for sub in COHORT:
        runs = build_subject_continuous(sub)
        if not runs:
            print(f"  {sub}: sin runs -> skip", flush=True); continue
        subj_runs[sub] = runs
        xc = _xcorr_subject(runs, LAG_GRID)
        jstar = int(np.nanargmin(xc))                  # mínimo = acoplamiento desync más fuerte
        tau_star = float(LAG_S[jstar]); rho_star = float(xc[jstar])
        # null max-statistic: circular-shift del SMNA dentro del run
        null_min = np.empty(N_PERM)
        for p in range(N_PERM):
            null_min[p] = np.nanmin(_xcorr_subject(runs, LAG_GRID, shift_smna=True, rng=rng))
        pperm = (1 + int((null_min <= rho_star).sum())) / (1 + N_PERM)
        g = _granger(runs)
        rows.append(dict(subject=sub, n_runs=len(runs), tau_star_s=round(tau_star, 2),
                         rho_star=round(rho_star, 3), p_perm=round(pperm, 4),
                         dr2_fwd=round(g["dr2_fwd"], 4), dr2_rev=round(g["dr2_rev"], 4),
                         fwd_minus_rev=round(g["dr2_fwd"] - g["dr2_rev"], 4)))
        for k, tau in enumerate(LAG_S):
            curves.append(dict(subject=sub, tau_s=round(float(tau), 3), xcorr=round(float(xc[k]), 4)))
        print(f"  {sub}: tau*={tau_star:+.2f}s  rho*={rho_star:+.3f}  p_perm={pperm:.3f}  "
              f"dR2(fwd-rev)={g['dr2_fwd'] - g['dr2_rev']:+.4f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(TBL_DIR / "lag_xcorr_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(TBL_DIR / "lag_xcorr_curves.csv", index=False)

    tstars = df["tau_star_s"].to_numpy(float)
    fmr = df["fwd_minus_rev"].to_numpy(float)
    print("\n=== VEREDICTO CONTINUO (cross-check) ===")
    print(f"  tau* por sujeto (s): {dict(zip(df.subject, tstars))}")
    print(f"  tau* > 0 (alfa LIDERA = brain->body): {int((tstars > 0).sum())}/{len(df)}  | "
          f"mediana={np.median(tstars):+.2f}s")
    print(f"  Granger forward>reverse (alfa->SMNA domina): {int((fmr > 0).sum())}/{len(df)}  | "
          f"media dR2(fwd-rev)={fmr.mean():+.4f}")
    print(f"  xcorr sig. (circular-shift null p<0.05): {int((df.p_perm < 0.05).sum())}/{len(df)}", flush=True)

    _plot(subj_runs, pd.DataFrame(curves), df)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.3(1) cross-check continuo] done", flush=True)


def _plot(subj_runs, dc, df):
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = dc.pivot_table(index="tau_s", columns="subject", values="xcorr")
    for sub in piv.columns:
        ax.plot(piv.index, piv[sub], color=SUBJ_COLORS.get(sub, "0.7"), lw=1, alpha=0.5)
        ts = df[df.subject == sub]["tau_star_s"].values[0]
        ax.plot(ts, df[df.subject == sub]["rho_star"].values[0], "v",
                color=SUBJ_COLORS.get(sub, "0.7"), ms=7)
    ax.plot(piv.index, piv.mean(axis=1), "ko-", lw=2.2, ms=3, label="media")
    ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag tau (s)  [tau>0 = SMNA sigue a la alfa = alfa LIDERA]")
    ax.set_ylabel("cross-correlación envelope-alfa-PO vs SMNA")
    ax.set_title("R3.3(1) cross-check CONTINUO: xcorr alfa-PO <-> SMNA @12.5Hz\n"
                 "mínimo (acoplamiento desync) en tau>0 = alfa lidera = brain->body", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lag_xcorr_continuous.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
