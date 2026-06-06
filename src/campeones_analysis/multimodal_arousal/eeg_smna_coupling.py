"""Evaluación rigurosa del acoplamiento EEG-banda <-> SMNA-AUC (ventanas), para discernir
ruido vs acoplamiento real. Lee el cache de eeg_smna_windows ({sub}_eegsmna_win.npz).

Lentes (todas por banda {delta,alpha,gamma} x ROI {PO,central,frontal,edge,all}, cross-subject):
  L1 correlación + PARCIAL (control tiempo-en-sesión tnorm): rho0 vs rho_partial; descarta drift.
  L2 lag cross-correlación (banda-PO): a qué lag (ventanas de 2 s) pica EEG vs SMNA.
  L3 decode + null circular-shift + baseline AR (banda-PO, lags de ventana): ¿le gana al azar y al AR?
  L4 evento bidireccional: ventanas alto-SMNA vs bajo -> diff de banda; y al revés.
  L5 especificidad: delta vs alpha vs gamma, PO vs edge -> ¿cortical-específico o EMG/arousal-general?
Dumpea coupling_results.json + CSVs + figuras. Veredicto por banda.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eeg_smna_coupling
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr, ttest_1samp
from sklearn.linear_model import Ridge

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR, OUT

OUT_DIR = OUT / "eeg_smna_coupling"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["delta", "alpha", "gamma"]
ROIS = ["PO", "central", "frontal", "edge", "all"]
WIN_LAGS = np.arange(-3, 4)   # ventanas (2 s c/u) para lag-xcorr y decode
AR_LAGS = np.arange(1, 4)     # 3 ventanas de pasado de SMNA para el baseline AR
ALPHAS = np.logspace(-1, 4, 6)
N_PERM = 200
RNG = np.random.default_rng(20260605)


def load_all():
    data = {}
    for sub in COHORT:
        p = NPZ_DIR / f"{sub}_eegsmna_win.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        data[sub] = {k: d[k] for k in d.files}
    return data


def _zscore_perrun(x, run):
    """z-score dentro de cada run -> aísla covariación INTRA-run (saca offsets entre-runs)."""
    out = np.asarray(x, float).copy()
    for g in np.unique(run):
        idx = run == g
        s = out[idx].std()
        out[idx] = (out[idx] - out[idx].mean()) / s if s > 0 else out[idx] - out[idx].mean()
    return out


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]; ryz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


def _lag_matrix_perrun(x, run, lags):
    """Lagged design respecting run boundaries (no cross-run leakage)."""
    T = len(x); X = np.full((T, len(lags)), np.nan)
    for g in np.unique(run):
        idx = np.where(run == g)[0]
        xs = x[idx]
        for k, L in enumerate(lags):
            if L > 0:
                X[idx[L:], k] = xs[:len(xs) - L]
            elif L < 0:
                X[idx[:len(xs) + L], k] = xs[-L:]
            else:
                X[idx, k] = xs
    return X


def _r2(yt, yp):
    ss = ((yt - yt.mean()) ** 2).sum()
    return 1.0 - ((yt - yp) ** 2).sum() / ss if ss > 0 else np.nan


def _loro_r2(X, y, run, alpha):
    r2s = []
    for g in np.unique(run):
        te = run == g; tr = ~te
        if te.sum() < 5 or tr.sum() < 5:
            continue
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        r2s.append(_r2(y[te], m.predict(X[te])))
    return float(np.mean(r2s)) if r2s else np.nan


def _best_alpha(X, y, run):
    return ALPHAS[int(np.argmax([_loro_r2(X, y, run, a) for a in ALPHAS]))]


# ---------------- L1: correlación + parcial ----------------
def lens_corr(data):
    rows = []
    for sub, d in data.items():
        tn, run = d["tnorm"], d["run"]
        y = _zscore_perrun(d["y"], run)  # intra-run (saca confound entre-runs)
        for b in BANDS:
            for r in ROIS:
                bp = _zscore_perrun(d[f"bp__{b}__{r}"], run)
                m = np.isfinite(bp) & np.isfinite(y)
                rho0 = spearmanr(bp[m], y[m])[0]
                rhop = partial_spearman(bp[m], y[m], tn[m])
                rows.append(dict(subject=sub, band=b, roi=r, rho0=rho0, rho_partial=rhop))
    df = pd.DataFrame(rows)
    summ = []
    for b in BANDS:
        for r in ROIS:
            s = df[(df.band == b) & (df.roi == r)]
            for col in ("rho0", "rho_partial"):
                z = np.arctanh(np.clip(s[col].to_numpy(float), -0.999, 0.999))
                t, p = ttest_1samp(z, 0.0)
                summ.append(dict(band=b, roi=r, metric=col, mean_rho=float(s[col].mean()),
                                 n_pos=int((s[col] > 0).sum()), n=len(s),
                                 t=float(t), p=float(p)))
    return df, pd.DataFrame(summ)


# ---------------- L2: lag cross-correlation (banda-PO) ----------------
def lens_lag(data):
    rows = []
    for sub, d in data.items():
        y, run = d["y"], d["run"]
        for b in BANDS:
            bp = d[f"bp__{b}__PO"]
            rr = []
            for L in WIN_LAGS:
                Xl = _lag_matrix_perrun(bp, run, [L])[:, 0]
                m = np.isfinite(Xl) & np.isfinite(y)
                rr.append(spearmanr(Xl[m], y[m])[0])
            rr = np.array(rr)
            k = int(np.argmax(np.abs(rr)))
            rows.append(dict(subject=sub, band=b, peak_lag_win=int(WIN_LAGS[k]),
                             peak_lag_s=float(WIN_LAGS[k] * 2.0), peak_rho=float(rr[k])))
    return pd.DataFrame(rows)


# ---------------- L3: decode + null + AR (banda-PO) ----------------
def lens_decode(data):
    rows = []
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)  # intra-run -> R2 no penalizado por offsets entre-runs
        ar = _lag_matrix_perrun(y, run, AR_LAGS)
        valid = np.all(np.isfinite(ar), axis=1)
        for b in BANDS:
            bp = _zscore_perrun(d[f"bp__{b}__PO"], run)
            Xb = _lag_matrix_perrun(bp, run, WIN_LAGS)
            v = valid & np.all(np.isfinite(Xb), axis=1)
            Xb_, y_, run_, ar_ = Xb[v], y[v], run[v], ar[v]
            a = _best_alpha(Xb_, y_, run_)
            r2 = _loro_r2(Xb_, y_, run_, a)
            r2_ar = _loro_r2(ar_, y_, run_, _best_alpha(ar_, y_, run_))
            Xfull = np.hstack([ar_, Xb_])
            r2_full = _loro_r2(Xfull, y_, run_, _best_alpha(Xfull, y_, run_))
            # circular-shift null on R2 (shift bp within run)
            null = np.empty(N_PERM)
            for p in range(N_PERM):
                sh = bp.copy()
                for g in np.unique(run):
                    idx = np.where(run == g)[0]
                    s = int(RNG.integers(1, max(2, len(idx))))
                    sh[idx] = np.roll(bp[idx], s)
                Xs = _lag_matrix_perrun(sh, run, WIN_LAGS)[v]
                null[p] = _loro_r2(Xs, y_, run_, a)
            p_emp = float((np.sum(null >= r2) + 1) / (N_PERM + 1))
            rows.append(dict(subject=sub, band=b, r2=float(r2), r2_ar=float(r2_ar),
                             r2_full=float(r2_full), dr2_over_ar=float(r2_full - r2_ar),
                             p_perm=p_emp))
    return pd.DataFrame(rows)


# ---------------- L4: evento bidireccional ----------------
def lens_event(data):
    rows = []
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)
        for b in BANDS:
            for r in ROIS:
                bp = _zscore_perrun(d[f"bp__{b}__{r}"], run)
                m = np.isfinite(bp) & np.isfinite(y)
                bpv, yv = bp[m], y[m]
                qy = np.quantile(yv, [1 / 3, 2 / 3])
                hi, lo = yv >= qy[1], yv <= qy[0]
                d_bp = float(bpv[hi].mean() - bpv[lo].mean())  # forward: high SMNA -> more band?
                pf = mannwhitneyu(bpv[hi], bpv[lo])[1] if hi.sum() and lo.sum() else np.nan
                qb = np.quantile(bpv, [1 / 3, 2 / 3])
                hb, lb = bpv >= qb[1], bpv <= qb[0]
                d_y = float(yv[hb].mean() - yv[lb].mean())     # reverse: high band -> more SMNA?
                rows.append(dict(subject=sub, band=b, roi=r, d_bp_hiSMNA=d_bp, p_fwd=float(pf),
                                 d_y_hiBand=d_y))
    return pd.DataFrame(rows)


def verdict(corr_summ, decode, lag):
    """Por banda (PO): ruido vs acoplamiento vs inconcluso."""
    out = {}
    for b in BANDS:
        cp = corr_summ[(corr_summ.band == b) & (corr_summ.roi == "PO") & (corr_summ.metric == "rho_partial")].iloc[0]
        dc = decode[decode.band == b]
        n_sig_perm = int((dc["p_perm"] < 0.05).sum())
        dr2_pos = int((dc["dr2_over_ar"] > 0).sum())
        consistent = cp["n_pos"] >= 5 or cp["n_pos"] <= 1  # consistent sign (either direction)
        partial_holds = abs(cp["mean_rho"]) > 0.05 and cp["p"] < 0.1
        beats_null = n_sig_perm >= 4
        beats_ar = dr2_pos >= 4
        if partial_holds and beats_null and beats_ar:
            v = "ACOPLAMIENTO (sobrevive parcial+null+AR)"
        elif partial_holds and (beats_null or beats_ar):
            v = "SUGESTIVO (parcial ok; null/AR parcial)"
        else:
            v = "RUIDO / no robusto"
        out[b] = dict(mean_rho_partial_PO=round(float(cp["mean_rho"]), 3),
                      n_pos_PO=int(cp["n_pos"]), p_group_PO=round(float(cp["p"]), 4),
                      n_sig_perm=n_sig_perm, dr2_pos=dr2_pos, verdict=v)
    return out


def main():
    print("=" * 78); print("eeg_smna_coupling :: evaluación EEG-banda <-> SMNA-AUC"); print("=" * 78, flush=True)
    data = load_all()
    print(f"sujetos con cache: {list(data)}", flush=True)
    if len(data) < 3:
        print("FALTA CACHE (correr eeg_smna_windows primero)"); return

    corr_df, corr_summ = lens_corr(data)
    lag_df = lens_lag(data)
    decode_df = lens_decode(data)
    event_df = lens_event(data)
    corr_df.to_csv(TBL_DIR / "L1_corr_perSubject.csv", index=False)
    corr_summ.to_csv(TBL_DIR / "L1_corr_group.csv", index=False)
    lag_df.to_csv(TBL_DIR / "L2_lag.csv", index=False)
    decode_df.to_csv(TBL_DIR / "L3_decode.csv", index=False)
    event_df.to_csv(TBL_DIR / "L4_event.csv", index=False)

    vd = verdict(corr_summ, decode_df, lag_df)
    results = dict(
        bands=BANDS, rois=ROIS, n_subjects=len(data),
        L1_partial_PO={b: corr_summ[(corr_summ.band == b) & (corr_summ.roi == "PO") &
                                    (corr_summ.metric == "rho_partial")][["mean_rho", "n_pos", "p"]]
                       .to_dict("records")[0] for b in BANDS},
        L1_zeroorder_PO={b: corr_summ[(corr_summ.band == b) & (corr_summ.roi == "PO") &
                                      (corr_summ.metric == "rho0")][["mean_rho", "n_pos", "p"]]
                         .to_dict("records")[0] for b in BANDS},
        L3_decode={b: decode_df[decode_df.band == b][["r2", "dr2_over_ar", "p_perm"]]
                   .mean(numeric_only=True).round(4).to_dict() for b in BANDS},
        lag_peak_s={b: lag_df[lag_df.band == b]["peak_lag_s"].tolist() for b in BANDS},
        specificity_PO_vs_edge={b: dict(
            PO=float(corr_summ[(corr_summ.band == b) & (corr_summ.roi == "PO") & (corr_summ.metric == "rho_partial")]["mean_rho"].iloc[0]),
            edge=float(corr_summ[(corr_summ.band == b) & (corr_summ.roi == "edge") & (corr_summ.metric == "rho_partial")]["mean_rho"].iloc[0]),
        ) for b in BANDS},
        verdict=vd,
    )
    with open(TBL_DIR / "coupling_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # figura resumen: rho parcial por banda x ROI (heatmap) + decode dR2/p
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    piv = corr_summ[corr_summ.metric == "rho_partial"].pivot(index="band", columns="roi", values="mean_rho").reindex(BANDS)[ROIS]
    im = axes[0].imshow(piv.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    axes[0].set_xticks(range(len(ROIS))); axes[0].set_xticklabels(ROIS); axes[0].set_yticks(range(len(BANDS))); axes[0].set_yticklabels(BANDS)
    for i in range(len(BANDS)):
        for j in range(len(ROIS)):
            axes[0].text(j, i, f"{piv.values[i,j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="rho parcial (control tiempo)")
    axes[0].set_title("L1 rho parcial EEG-banda vs SMNA-AUC (banda x ROI)")
    x = np.arange(len(BANDS))
    axes[1].bar(x - 0.2, [results["L3_decode"][b]["dr2_over_ar"] for b in BANDS], 0.4, label="dR2 sobre AR")
    axes[1].bar(x + 0.2, [results["L3_decode"][b]["r2"] for b in BANDS], 0.4, label="R2 decode")
    axes[1].set_xticks(x); axes[1].set_xticklabels(BANDS); axes[1].axhline(0, color="k", lw=0.5); axes[1].legend()
    axes[1].set_title("L3 decode banda-PO -> SMNA (R2, dR2 sobre AR)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "coupling_summary.png", dpi=130); plt.close(fig)

    print("\n=== VEREDICTO por banda (PO) ===")
    print(json.dumps(vd, indent=2), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
