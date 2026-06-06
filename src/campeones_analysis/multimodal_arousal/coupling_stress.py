"""Stress-tests del acoplamiento EEG-banda <-> SMNA-AUC, para la revisión profunda del INFORME.

Todos sobre el cache {sub}_eegsmna_win.npz (rápido, sin re-procesar raw). Reusa la máquina de
eeg_smna_coupling. Cuatro tests que atacan las afirmaciones centrales del INFORME:

  T1 EMG-partial: ¿sobrevive el acoplamiento PO<->SMNA al CONTROLAR el band-power de borde (proxy EMG)?
     partial Spearman(PO, SMNA | tnorm, edge) y decode dR2 de agregar PO sobre [AR + edge].
     Si colapsa -> el "acoplamiento PO" es el componente EMG compartido (NO cortical).
  T2 Innovación de la SMNA: residualizar SMNA contra su propio pasado AR(1-3) (out-of-fold) y testear
     si el EEG predice la INNOVACIÓN (lo que el AR no puede). Si no -> redundante con autocorrelación.
  T3 Robustez del orden AR: r2_ar y dR2(full-ar) para órdenes AR 1..6. ¿dR2~0 es robusto al orden?
  T4 EMG-level vs acoplamiento: correlación cross-subject entre el nivel de EMG (HF 60-90 dB, de
     emg_highfreq_summary_postica.csv) y la fuerza de acoplamiento por sujeto (gamma-PO, delta-PO).
     Si más EMG -> más acoplamiento -> firma EMG.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.coupling_stress
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from src.campeones_analysis.multimodal_arousal.cohort import OUT
from src.campeones_analysis.multimodal_arousal.eeg_smna_coupling import (
    AR_LAGS,
    BANDS,
    TBL_DIR,
    WIN_LAGS,
    _best_alpha,
    _lag_matrix_perrun,
    _loro_r2,
    _r2,
    _zscore_perrun,
    load_all,
)

warnings.filterwarnings("ignore")
QA = OUT / "qa_artifact_vs_signal"


def _resid_on(ranks_y, ranks_covs):
    """Residual de rank(y) tras regresión lineal OLS sobre rank(covariables) (con intercepto)."""
    X = np.column_stack([np.ones(len(ranks_y))] + list(ranks_covs))
    beta, *_ = np.linalg.lstsq(X, ranks_y, rcond=None)
    return ranks_y - X @ beta


def partial_spearman_multi(x, y, covs):
    """Spearman parcial de x,y controlando >=1 covariables (rank-based)."""
    rx, ry = rankdata(x), rankdata(y)
    rc = [rankdata(c) for c in covs]
    ex, ey = _resid_on(rx, rc), _resid_on(ry, rc)
    return np.corrcoef(ex, ey)[0, 1]


def loro_oof(X, y, run, alpha):
    yp = np.full(len(y), np.nan)
    for g in np.unique(run):
        te = run == g; tr = ~te
        if te.sum() < 5 or tr.sum() < 5:
            continue
        yp[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    return yp


# ---------------- T1: EMG-partial ----------------
def t1_emg_partial(data):
    rows = []
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)
        tn = d["tnorm"]
        ar = _lag_matrix_perrun(y, run, AR_LAGS)
        for b in BANDS:
            po = _zscore_perrun(d[f"bp__{b}__PO"], run)
            edge = _zscore_perrun(d[f"bp__{b}__edge"], run)
            m = np.isfinite(po) & np.isfinite(edge) & np.isfinite(y) & np.isfinite(tn)
            # partial spearman PO~SMNA controlando tnorm (baseline) y luego + edge
            rho_t = partial_spearman_multi(po[m], y[m], [tn[m]])
            rho_te = partial_spearman_multi(po[m], y[m], [tn[m], edge[m]])
            # decode: ¿PO agrega sobre [AR + edge]?
            Xedge = _lag_matrix_perrun(edge, run, WIN_LAGS)
            Xpo = _lag_matrix_perrun(po, run, WIN_LAGS)
            v = (np.all(np.isfinite(ar), axis=1) & np.all(np.isfinite(Xedge), axis=1)
                 & np.all(np.isfinite(Xpo), axis=1))
            base = np.hstack([ar[v], Xedge[v]])
            full = np.hstack([ar[v], Xedge[v], Xpo[v]])
            yv, runv = y[v], run[v]
            r2_base = _loro_r2(base, yv, runv, _best_alpha(base, yv, runv))
            r2_full = _loro_r2(full, yv, runv, _best_alpha(full, yv, runv))
            rows.append(dict(subject=sub, band=b,
                             rho_partial_tnorm=float(rho_t),
                             rho_partial_tnorm_edge=float(rho_te),
                             shrink_pct=float(100 * (1 - rho_te / rho_t)) if rho_t != 0 else np.nan,
                             dr2_PO_over_AR_edge=float(r2_full - r2_base)))
    return pd.DataFrame(rows)


# ---------------- T2: SMNA innovation ----------------
def t2_innovation(data):
    rows = []
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)
        ar = _lag_matrix_perrun(y, run, AR_LAGS)
        v0 = np.all(np.isfinite(ar), axis=1)
        # innovación OOF = y - prediccion AR out-of-fold
        a = _best_alpha(ar[v0], y[v0], run[v0])
        yp_ar = np.full(len(y), np.nan)
        yp_ar[v0] = loro_oof(ar[v0], y[v0], run[v0], a)
        innov = y - yp_ar  # lo que el AR no explica
        for b in BANDS:
            for roi in ("PO", "edge"):
                bp = _zscore_perrun(d[f"bp__{b}__{roi}"], run)
                # mejor lag: corr(bp lag L, innov)
                best = (0, 0.0)
                for L in WIN_LAGS:
                    Xl = _lag_matrix_perrun(bp, run, [L])[:, 0]
                    m = np.isfinite(Xl) & np.isfinite(innov)
                    if m.sum() < 20:
                        continue
                    r = spearmanr(Xl[m], innov[m])[0]
                    if abs(r) > abs(best[1]):
                        best = (int(L), float(r))
                rows.append(dict(subject=sub, band=b, roi=roi,
                                 innov_best_lag_s=best[0] * 2.0, innov_rho=best[1]))
    return pd.DataFrame(rows)


# ---------------- T3: AR-order robustness ----------------
def t3_ar_order(data, orders=range(1, 7)):
    rows = []
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)
        po = _zscore_perrun(d["bp__gamma__PO"], run)  # gamma = el más fuerte
        Xpo = _lag_matrix_perrun(po, run, WIN_LAGS)
        for na in orders:
            ar = _lag_matrix_perrun(y, run, np.arange(1, na + 1))
            v = np.all(np.isfinite(ar), axis=1) & np.all(np.isfinite(Xpo), axis=1)
            yv, runv = y[v], run[v]
            r2_ar = _loro_r2(ar[v], yv, runv, _best_alpha(ar[v], yv, runv))
            full = np.hstack([ar[v], Xpo[v]])
            r2_full = _loro_r2(full, yv, runv, _best_alpha(full, yv, runv))
            rows.append(dict(subject=sub, ar_order=na, r2_ar=float(r2_ar),
                             dr2_gammaPO_over_ar=float(r2_full - r2_ar)))
    return pd.DataFrame(rows)


# ---------------- T4: EMG-level vs coupling ----------------
def t4_emg_level(data):
    # acoplamiento por sujeto (zero-order Spearman PO~SMNA, gamma y delta)
    coup = {}
    for sub, d in data.items():
        run = d["run"]
        y = _zscore_perrun(d["y"], run)
        row = {}
        for b in ("delta", "gamma"):
            po = _zscore_perrun(d[f"bp__{b}__PO"], run)
            m = np.isfinite(po) & np.isfinite(y)
            row[b] = spearmanr(po[m], y[m])[0]
        coup[sub] = row
    # nivel EMG por sujeto
    emg_path = QA / "highfreq" / "tables" / "emg_highfreq_summary_postica.csv"
    out = dict(available=False)
    if emg_path.exists():
        emg = pd.read_csv(emg_path)
        # nivel EMG = HF (60-90) real-silent dB, all-channel; fallbacks por preferencia
        prefer = ["hf_diff_db_all", "hf_ratio_all", "hf_diff_db_edge"]
        cand = [c for c in prefer if c in emg.columns] or \
               [c for c in emg.columns if "db" in c.lower() or "ratio" in c.lower()]
        subcol = [c for c in emg.columns if c.lower() == "subject"] or \
                 [c for c in emg.columns if "sub" in c.lower()]
        if subcol and cand:
            sc, vc = subcol[0], cand[0]
            emg_by_sub = {str(r[sc]): float(r[vc]) for _, r in emg.iterrows()}
            xs, gd, dd, labs = [], [], [], []
            for sub in coup:
                key = sub if sub in emg_by_sub else sub.replace("sub-", "")
                if key in emg_by_sub:
                    xs.append(emg_by_sub[key]); gd.append(coup[sub]["gamma"]); dd.append(coup[sub]["delta"]); labs.append(sub)
            if len(xs) >= 4:
                out = dict(available=True, emg_col=vc, n=len(xs), subjects=labs,
                           emg=xs, gamma_coup=gd, delta_coup=dd,
                           r_emg_gamma=float(np.corrcoef(xs, gd)[0, 1]),
                           r_emg_delta=float(np.corrcoef(xs, dd)[0, 1]))
    out["coupling_per_subject"] = coup
    return out


def main():
    print("=" * 78); print("coupling_stress :: stress-tests del acoplamiento EEG<->SMNA"); print("=" * 78, flush=True)
    data = load_all()
    print(f"sujetos: {list(data)}", flush=True)
    sdir = QA / "coupling_stress" / "tables"
    sdir.mkdir(parents=True, exist_ok=True)

    t1 = t1_emg_partial(data); t1.to_csv(sdir / "T1_emg_partial.csv", index=False)
    t2 = t2_innovation(data); t2.to_csv(sdir / "T2_innovation.csv", index=False)
    t3 = t3_ar_order(data); t3.to_csv(sdir / "T3_ar_order.csv", index=False)
    t4 = t4_emg_level(data)

    summary = dict(
        T1_emg_partial={b: dict(
            rho_tnorm=round(float(t1[t1.band == b]["rho_partial_tnorm"].mean()), 4),
            rho_tnorm_edge=round(float(t1[t1.band == b]["rho_partial_tnorm_edge"].mean()), 4),
            mean_shrink_pct=round(float(t1[t1.band == b]["shrink_pct"].mean()), 1),
            dr2_PO_over_AR_edge=round(float(t1[t1.band == b]["dr2_PO_over_AR_edge"].mean()), 5),
            n_pos_after_edge=int((t1[t1.band == b]["rho_partial_tnorm_edge"] > 0).sum()),
        ) for b in BANDS},
        T2_innovation={b: dict(
            PO_mean_rho=round(float(t2[(t2.band == b) & (t2.roi == "PO")]["innov_rho"].mean()), 4),
            PO_n_pos=int((t2[(t2.band == b) & (t2.roi == "PO")]["innov_rho"] > 0).sum()),
            edge_mean_rho=round(float(t2[(t2.band == b) & (t2.roi == "edge")]["innov_rho"].mean()), 4),
        ) for b in BANDS},
        T3_ar_order={int(na): dict(
            r2_ar=round(float(t3[t3.ar_order == na]["r2_ar"].mean()), 4),
            dr2_gammaPO=round(float(t3[t3.ar_order == na]["dr2_gammaPO_over_ar"].mean()), 5),
        ) for na in sorted(t3.ar_order.unique())},
        T4_emg_level={k: v for k, v in t4.items() if k != "coupling_per_subject"},
    )
    with open(sdir / "coupling_stress_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nOutputs -> {sdir}", flush=True)


if __name__ == "__main__":
    main()
