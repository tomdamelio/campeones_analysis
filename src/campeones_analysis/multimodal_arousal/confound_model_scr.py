"""2.5 (STOPGAP) — ¿cuánto del efecto SCR sobre las VDs sobrevive a los covariados + tiempo?

Test central de Enzo, operacionalizado como CVCR (deconfounding out-of-sample, Snoek 2019):
para cada VD periódica, ¿la separabilidad SCR-vs-no-SCR sobrevive cuando residualizamos la VD
contra los covariados de artefacto + tiempo DENTRO de cada fold de CV? Si el AUC deconfounded
cae a azar -> el efecto vivía en los confounds; si sobrevive -> señal SCR única.

STOPGAP (2026-06-08): usa los covariados disponibles HOY — 2.1 (gamma_EOG/sp/blink, raw pre-ICA)
+ 2.4 (edge/central gamma) + tiempo (tnorm). Faltan 2.2 (IMU) y 2.3-extracción (EOG-bipolar);
el veredicto FINAL del paper espera esos + Track B. Acá corre el grueso del test.

VDs (periódico, 1/f removido, per canal, 29 ch): gamma / delta / alfa + offset (aperiódico).
Adjudicaciones (cada VD contra SUS covariados):
  Q-gamma  : gamma  ~ SCR + {gamma_EOG, sp_hf, edge_central_gamma, time, time^2}
  Q-delta  : delta  ~ SCR + {blink_slow, blink_2hz_pre, time, time^2}
  Q-alfa   : alfa   ~ SCR + {blink_slow, time, time^2}
  Q-offset : offset ~ SCR + {gamma_EOG, blink_slow, edge_central_gamma, time, time^2}

Reusa load_cache (panel_psd, 29 ch) + _linear_aperiodic + _clf + _perm_within (probados).
Covariados alineados 1:1 al cache (validado en 2.1: max|Δtnorm|=0). Carta principal = intra
(within-subject k-fold); LOSO se reporta como robustez (F10).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.confound_model_scr --probe
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.confound_model_scr --nperm 2000
"""

from __future__ import annotations

import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO
from src.campeones_analysis.multimodal_arousal.decoding_panel import load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES,
    SEED,
    _clf,
    _linear_aperiodic,
    _perm_within,
)

warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_5_confound_model"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

_C = REPO / "research_diary" / "context" / "05_05"
COV21 = _C / "2_1_artifact_raw" / "tables" / "artifact_covariate.csv"
COV22 = _C / "2_2_motion_imu" / "tables" / "motion_covariate.csv"
COV23 = _C / "2_3_eog_covariate" / "tables" / "eog_covariate.csv"
COV24 = _C / "2_4_edge_central" / "tables" / "edge_central_covariate.csv"
RANGE = "1-40"


# ------------------------------- VD builders -------------------------------
def feat_band(psd, freqs, rng, band):
    _, _, resid, f = _linear_aperiodic(psd, freqs, rng)
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)


def feat_offset(psd, freqs, rng):
    off, _, _, _ = _linear_aperiodic(psd, freqs, rng)
    return off


VDS = {
    "gamma": lambda p, f: feat_band(p, f, RANGES[RANGE], (30, 40)),
    "delta": lambda p, f: feat_band(p, f, RANGES[RANGE], (1, 4)),
    "alpha": lambda p, f: feat_band(p, f, RANGES[RANGE], (8, 13)),
    "offset": lambda p, f: feat_offset(p, f, RANGES[RANGE]),
}
# Familia COMPLETA de covariados (2.1 radial+Fp · 2.2 IMU · 2.3 EOG directo · 2.4 edge · tiempo)
ADJ = {
    "Q-gamma": ("gamma", ["gamma_EOG", "sp_hf", "heog_gamma_30_40", "edge_gamma", "var_jerk", "tnorm", "tnorm2"]),
    "Q-delta": ("delta", ["blink_slow", "blink_2hz_pre", "veog_slow_0p5_8", "veog_slow_2hz_pre", "var_jerk_0p5_2", "tnorm", "tnorm2"]),
    "Q-alfa": ("alpha", ["blink_slow", "veog_slow_0p5_8", "tnorm", "tnorm2"]),
    "Q-offset": ("offset", ["gamma_EOG", "blink_slow", "edge_gamma", "var_jerk", "tnorm", "tnorm2"]),
}


# ------------------------------- data assembly -------------------------------
def _load_covariates(cache):
    """Por sujeto: dict col->array alineado 1:1 al cache (verifica tnorm)."""
    d21, d22, d23, d24 = (pd.read_csv(p) for p in (COV21, COV22, COV23, COV24))
    out = {}
    for sub in cache:
        psd, y, tn = cache[sub]
        a = d21[d21.subject == sub].reset_index(drop=True)
        m = d22[d22.subject == sub].reset_index(drop=True)
        e = d23[d23.subject == sub].reset_index(drop=True)
        b = d24[d24.subject == sub].reset_index(drop=True)
        for nm, t in [("2.1", a), ("2.2", m), ("2.3", e), ("2.4", b)]:
            assert len(t) == len(y), f"{sub}: len mismatch {nm} {len(t)} vs {len(y)}"
            assert np.max(np.abs(t["tnorm"].to_numpy() - tn)) < 1e-3, f"{sub}: {nm} tnorm misalign"
        cols = dict(
            gamma_EOG=a["gamma_EOG"].to_numpy(), sp_hf=a["sp_hf"].to_numpy(),
            blink_slow=a["blink_slow"].to_numpy(), blink_2hz_pre=a["blink_2hz_pre"].to_numpy(),
            var_jerk=m["var_jerk"].to_numpy(), var_jerk_0p5_2=m["var_jerk_0p5_2"].to_numpy(),
            heog_gamma_30_40=e["heog_gamma_30_40"].to_numpy(),
            veog_slow_0p5_8=e["veog_slow_0p5_8"].to_numpy(),
            veog_slow_2hz_pre=e["veog_slow_2hz_pre"].to_numpy(),
            edge_gamma=b["edge_central_db_gamma"].to_numpy(),
            tnorm=tn.astype(float), tnorm2=tn.astype(float) ** 2,
        )
        for k in cols:
            v = cols[k]
            cols[k] = np.nan_to_num(v, nan=float(np.nanmedian(v)))
        out[sub] = cols
    return out


def _build(adj, cache, freqs, covs):
    vd_name, cov_cols = ADJ[adj]
    builder = VDS[vd_name]
    Xs, ys, Cs = [], [], []
    for sub in cache:
        psd, y, tn = cache[sub]
        Xs.append(builder(psd, freqs).astype(float))
        ys.append(y.astype(int))
        Cs.append(np.column_stack([covs[sub][c] for c in cov_cols]))
    return Xs, ys, Cs


# ------------------------------- deconfound + CV -------------------------------
def _decon(Xtr, Xte, Ctr, Cte):
    """Residualiza X contra C (z-score con stats de train, betas de train). CVCR."""
    mu, sd = Ctr.mean(0), Ctr.std(0) + 1e-12
    Atr = np.hstack([(Ctr - mu) / sd, np.ones((len(Ctr), 1))])
    Ate = np.hstack([(Cte - mu) / sd, np.ones((len(Cte), 1))])
    beta, *_ = np.linalg.lstsq(Atr, Xtr, rcond=None)
    return Xtr - Atr @ beta, Xte - Ate @ beta


def evaluate(Xs, ys, Cs, deconf):
    # intra: within-subject 5-fold
    intra = []
    for X, y, C in zip(Xs, ys, Cs):
        if len(np.unique(y)) < 2:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        proba = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            Xtr, Xte = (_decon(X[tr], X[te], C[tr], C[te]) if deconf else (X[tr], X[te]))
            proba[te] = _clf().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        intra.append(roc_auc_score(y, proba))
    # loso
    Xall, yall = np.concatenate(Xs), np.concatenate(ys)
    Call = np.concatenate(Cs)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    loso = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        Xtr, Xte = (_decon(Xall[tr], Xall[te], Call[tr], Call[te]) if deconf else (Xall[tr], Xall[te]))
        loso.append(roc_auc_score(yall[te], _clf().fit(Xtr, yall[tr]).predict_proba(Xte)[:, 1]))
    return float(np.mean(intra)), float(np.mean(loso)), groups


def _perm_p(Xs, ys, Cs, deconf, obs_intra, obs_loso, n_perm, groups):
    if n_perm <= 0:
        return np.nan, np.nan
    yall = np.concatenate(ys)
    pr = np.random.default_rng(SEED + 7)
    ge_i = ge_l = 0
    for _ in range(n_perm):
        yp = _perm_within(yall, groups, pr)
        ys_p = [yp[groups == i] for i in range(len(ys))]
        pi, pl, _ = evaluate(Xs, ys_p, Cs, deconf)
        ge_i += (pi >= obs_intra); ge_l += (pl >= obs_loso)
    return (1 + ge_i) / (1 + n_perm), (1 + ge_l) / (1 + n_perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="smoke test: 1 VD, nperm=20")
    ap.add_argument("--replot", action="store_true", help="regenerar figura desde el CSV (sin perms)")
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()
    if args.replot:
        _plot(pd.read_csv(TBL_DIR / "confound_model_full.csv"))
        print(f"replot -> {FIG_DIR / 'confound_model_full.png'}")
        return
    n_perm = 20 if args.probe else args.nperm

    print("=" * 78)
    print("confound_model_scr :: 2.5 — CVCR deconfounding (familia COMPLETA 2.1+2.2+2.3+2.4+tiempo)")
    print(f"OUT -> {OUT_DIR}  nperm={n_perm}")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    print(f"  cache {len(cache)} suj, {len(ch)} ch; covariados alineados OK", flush=True)

    adjs = ["Q-gamma"] if args.probe else list(ADJ)
    rows = []
    t0 = time.time()
    for adj in adjs:
        Xs, ys, Cs = _build(adj, cache, freqs, covs)
        ri, rl, groups = evaluate(Xs, ys, Cs, deconf=False)
        di, dl, _ = evaluate(Xs, ys, Cs, deconf=True)
        pr_i, pr_l = _perm_p(Xs, ys, Cs, False, ri, rl, n_perm, groups)
        pd_i, pd_l = _perm_p(Xs, ys, Cs, True, di, dl, n_perm, groups)
        row = dict(adjudication=adj, vd=ADJ[adj][0],
                   raw_intra=round(ri, 4), deconf_intra=round(di, 4), d_intra=round(di - ri, 4),
                   p_deconf_intra=pd_i, raw_loso=round(rl, 4), deconf_loso=round(dl, 4),
                   d_loso=round(dl - rl, 4), p_deconf_loso=pd_l,
                   p_raw_intra=pr_i, p_raw_loso=pr_l)
        rows.append(row)
        print(f"  {adj:9s} ({ADJ[adj][0]:6s}): intra {ri:.3f}->{di:.3f} (Δ{di-ri:+.3f}, p_dec={pd_i})  "
              f"loso {rl:.3f}->{dl:.3f} (Δ{dl-rl:+.3f}, p_dec={pd_l})  [{(time.time()-t0)/60:.1f} min]",
              flush=True)

    df = pd.DataFrame(rows)
    tag = "probe" if args.probe else "full"
    df.to_csv(TBL_DIR / f"confound_model_{tag}.csv", index=False)
    if not args.probe:
        _plot(df)
    print(f"\nTabla -> {TBL_DIR / f'confound_model_{tag}.csv'}")
    print("[2.5 stopgap] done", flush=True)


def _plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, sch in zip(axes, ["intra", "loso"]):
        x = np.arange(len(df)); w = 0.38
        ax.bar(x - w / 2, df[f"raw_{sch}"], w, label="raw (sin deconfound)", color="C0", alpha=0.8)
        ax.bar(x + w / 2, df[f"deconf_{sch}"], w, label="deconfounded (CVCR)", color="C1", alpha=0.85)
        for xi, (r, d, p) in enumerate(zip(df[f"raw_{sch}"], df[f"deconf_{sch}"], df[f"p_deconf_{sch}"])):
            ax.annotate(f"p={p:.3f}", (xi + w / 2, d + 0.006), ha="center", fontsize=7,
                        color="darkred" if p < 0.05 else "0.4")
        ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
        ax.set_xticks(x); ax.set_xticklabels(df["adjudication"], rotation=15, fontsize=9)
        ax.set_ylabel(f"{sch.upper()} AUC"); ax.set_ylim(0.45, 0.75)
        ax.set_title(f"{sch.upper()}: ¿sobrevive el SCR al deconfound?", fontsize=10)
        if sch == "intra":
            ax.legend(fontsize=8)
    fig.suptitle("2.5 — fracción del efecto SCR que sobrevive a la familia COMPLETA de covariados\n"
                 "(2.1 microsacada/blink · 2.2 movimiento · 2.3 EOG · 2.4 edge · tiempo) — "
                 "deconfounded >azar = señal SCR única · N=6, 29 ch",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "confound_model_full.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
