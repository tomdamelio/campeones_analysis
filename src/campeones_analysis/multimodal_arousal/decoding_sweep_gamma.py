"""1.4 Gamma -- Decoding como REGRESION Tweedie (no clasificacion): el EEG predice la MAGNITUD
del arousal sudomotor (SMNA-AUC), modelada con la verosimilitud que caracteriza al EDA.

Por que esto y no "1.4 con link gamma" a secas. En 1.4 el target es BINARIO (SCR vs no-SCR) y el
modelo ya es un GLM (binomial/logit, = LogisticRegression). El link gamma describe una respuesta
CONTINUA y positiva -- la *amplitud* de la SCR, no el presente/ausente. Para aprovechar esa
distribucion hay que dar vuelta el problema a REGRESION: predecir SMNA-AUC (continua, cero exacto en
no-SCR + cola gamma en SCR) desde el EEG, con un GLM **Tweedie** (compound Poisson-Gamma, 1<p<2),
que es justo la familia del Bloque 3 (scr_glm_tweedie.py).

Diferencia de maquinaria vs el Bloque 3: alla el GLM es statsmodels sm.GLM (sin regularizacion),
valido para 1 predictor (alfa) + confounds. Aca el barrido tiene p>>n (hasta ~1131 features), donde
sm.GLM es singular. El analogo correcto al ridge-logistico de 1.4 es **sklearn.TweedieRegressor**
(GLM Tweedie con penalizacion L2). Mismo StandardScaler + ridge, misma grilla de feature-sets,
mismo intra/LOSO/permutaciones que 1.4.

Metricas (por celda del barrido):
  - **AUC (Tweedie-derivado)**: rankear epocas por SMNA predicho vs la etiqueta binaria SCR.
    Es DIRECTAMENTE comparable al AUC de 1.4 -> contesta "¿performance mejor que el logistico?".
  - **AUC (logistico)**: baseline en el MISMO cache/features -> head-to-head limpio (no se compara
    contra los numeros publicados de 1.4, que salen de otro cache).
  - **D2 de Tweedie** (devianza explicada / pseudo-R2): la metrica NATIVA de la regresion -- cuanto
    de la *magnitud* del arousal explica el EEG. 1.4 (clasificacion) no podia dar esto.

Target + features 1:1 alineados: SMNA-AUC per-epoca via scr_glm_tweedie.assemble() (que ya verifica
alineamiento por tnorm contra el cache 'uniform' del panel), y el psd del MISMO cache para construir
las familias de features con los builders de decoding_sweep.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_sweep_gamma --probe
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_sweep_gamma [--nperm 100]
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import d2_tweedie_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES,
    build_X,
    cells,
    _clf,
    _cell_order,
)
from src.campeones_analysis.multimodal_arousal.scr_glm_tweedie import assemble

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "decoding_sweep_gamma"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260607
POWER = 1.5          # var_power Tweedie (medio de P_GRID=[1.3..1.7] del Bloque 3; sensib. ya establecida alla)
RIDGE_ALPHA = 1.0    # L2 del TweedieRegressor; paralelo al C=1.0 del logistico de 1.4
N_PERM = 100
N_JOBS = -2          # joblib en las permutaciones (todas las cores menos una)
CACHE = TBL_DIR / "target_features.npz"   # psd + SMNA-AUC + y por sujeto (evita re-correr assemble())


# =============================================================================
# Step 1 -- target continuo (SMNA-AUC) + features, 1:1 alineados
# =============================================================================
def build_cache():
    """Corre assemble() (lee raws, ~lento) UNA vez y guarda psd + SMNA-AUC + y por sujeto alineados."""
    cache, freqs, ch = load_cache("uniform")
    tw = assemble()                       # {sub: {auc, amp, y, alpha, C}} ya alineado + MISALIGN guard
    store = {"freqs": freqs, "ch_names": np.array(ch), "subjects": []}
    for sub in COHORT:
        if sub not in tw or sub not in cache:
            continue
        psd, y_cache, _tn = cache[sub]
        auc = np.asarray(tw[sub]["auc"], float)
        y = np.asarray(tw[sub]["y"], int)
        if len(auc) != len(y_cache) or not np.array_equal(y, y_cache):
            print(f"  {sub}: target/feature MISALIGN -> skip", flush=True)
            continue
        m = np.isfinite(auc)              # SMNA-AUC finito (Tweedie admite ceros, no NaN)
        store[f"psd_{sub}"] = psd[m].astype(np.float32)
        store[f"auc_{sub}"] = np.clip(auc[m], 0, None).astype(np.float32)
        store[f"y_{sub}"] = y[m].astype(int)
        store["subjects"].append(sub)
        print(f"  {sub}: n={int(m.sum())} (SCR={int((y[m]==1).sum())})  target+features OK", flush=True)
    store["subjects"] = np.array(store["subjects"])
    np.savez_compressed(CACHE, **store)
    print(f"[1.4G] cache -> {CACHE}", flush=True)


def load_target_and_features():
    """Por sujeto: (psd, smna_auc, y_binario), desde el cache npz (build_cache si no existe)."""
    if not CACHE.exists():
        print("[1.4G] no cache -- corriendo assemble() (lento) ...", flush=True)
        build_cache()
    z = np.load(CACHE, allow_pickle=True)
    freqs = z["freqs"]; ch = list(z["ch_names"])
    data = {}
    for sub in [str(s) for s in z["subjects"]]:
        data[sub] = (z[f"psd_{sub}"].astype(float), z[f"auc_{sub}"].astype(float), z[f"y_{sub}"])
        print(f"  {sub}: n={len(data[sub][1])} (SCR={int((data[sub][2]==1).sum())})", flush=True)
    return data, freqs, ch


# =============================================================================
# Step 2 -- estimadores
# =============================================================================
def _reg():
    """GLM Tweedie con ridge L2, sobre features estandarizadas (analogo al ridge-logistico de 1.4)."""
    return make_pipeline(
        StandardScaler(),
        TweedieRegressor(power=POWER, alpha=RIDGE_ALPHA, link="log", max_iter=2000),
    )


def _d2(y_true, y_pred):
    """D2 de Tweedie (devianza explicada). Robusto a y_pred<=0 (clip minimo positivo)."""
    yp = np.clip(y_pred, 1e-12, None)
    try:
        return float(d2_tweedie_score(y_true, yp, power=POWER))
    except Exception:
        return np.nan


# =============================================================================
# Step 3 -- CV: intra (within-subject) + LOSO. Tweedie (D2 + AUC-derivado) y logistico (AUC).
# =============================================================================
def intra_eval(Xs, aucs, ys):
    """Within-subject 5-fold. OOF: SMNA predicho (Tweedie) y proba SCR (logistico). Devuelve listas
    por sujeto de (d2_tweedie, auc_tweedie, auc_logit)."""
    d2L, aucTW, aucLO = [], [], []
    for X, a, y in zip(Xs, aucs, ys):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pred_tw = np.zeros(len(y)); proba_lo = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            pred_tw[te] = _reg().fit(X[tr], a[tr]).predict(X[te])
            proba_lo[te] = _clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        d2L.append(_d2(a, pred_tw))
        aucTW.append(roc_auc_score(y, pred_tw))
        aucLO.append(roc_auc_score(y, proba_lo))
    return d2L, aucTW, aucLO


def loso_eval(Xs, aucs, ys):
    """Leave-one-subject-out. Por sujeto dejado afuera: D2 + AUC Tweedie + AUC logistico."""
    n = len(Xs)
    d2L, aucTW, aucLO = [], [], []
    for g in range(n):
        Xtr = np.concatenate([Xs[i] for i in range(n) if i != g], axis=0)
        atr = np.concatenate([aucs[i] for i in range(n) if i != g])
        ytr = np.concatenate([ys[i] for i in range(n) if i != g])
        Xte, ate, yte = Xs[g], aucs[g], ys[g]
        pred_tw = _reg().fit(Xtr, atr).predict(Xte)
        proba_lo = _clf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        d2L.append(_d2(ate, pred_tw))
        aucTW.append(roc_auc_score(yte, pred_tw))
        aucLO.append(roc_auc_score(yte, proba_lo))
    return d2L, aucTW, aucLO


def _intra_tw_auc(Xs, aucs, ys):
    """Solo el AUC-Tweedie intra medio (para permutaciones: sin logistico ni D2)."""
    out = []
    for X, a, y in zip(Xs, aucs, ys):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            pred[te] = _reg().fit(X[tr], a[tr]).predict(X[te])
        out.append(roc_auc_score(y, pred))
    return float(np.mean(out))


def _loso_tw_auc(Xs, aucs, ys):
    n = len(Xs); out = []
    for g in range(n):
        Xtr = np.concatenate([Xs[i] for i in range(n) if i != g], axis=0)
        atr = np.concatenate([aucs[i] for i in range(n) if i != g])
        pred = _reg().fit(Xtr, atr).predict(Xs[g])
        out.append(roc_auc_score(ys[g], pred))
    return float(np.mean(out))


def _perm_pairs(aucs, ys, rng):
    """Permuta el par (auc, y) DENTRO de cada sujeto (rompe feature<->target, mantiene auc<->y)."""
    ap, yp = [], []
    for a, y in zip(aucs, ys):
        idx = rng.permutation(len(y))
        ap.append(a[idx]); yp.append(y[idx])
    return ap, yp


def _one_perm(Xs, aucs, ys, seed):
    """Una permutacion: devuelve (auc_intra, auc_loso) Tweedie bajo labels permutadas."""
    rng = np.random.default_rng(seed)
    ap, yp = _perm_pairs(aucs, ys, rng)
    return _intra_tw_auc(Xs, ap, yp), _loso_tw_auc(Xs, ap, yp)


def run_cell(family, res, rng_name, data, freqs, n_perm):
    rng = RANGES[rng_name]
    Xs, aucs, ys = [], [], []
    for sub in data:
        psd, a, y = data[sub]
        Xs.append(build_X(family, psd, freqs, rng, res)); aucs.append(a); ys.append(y)
    n_feat = Xs[0].shape[1]

    d2_i, auctw_i, auclo_i = intra_eval(Xs, aucs, ys)
    d2_l, auctw_l, auclo_l = loso_eval(Xs, aucs, ys)
    intra_auctw, loso_auctw = float(np.mean(auctw_i)), float(np.mean(auctw_l))
    intra_d2, loso_d2 = float(np.nanmean(d2_i)), float(np.nanmean(d2_l))

    # permutacion subject-respecting sobre el AUC-Tweedie (intra y LOSO), paralelizada
    p_i = p_l = np.nan
    if n_perm:
        perms = Parallel(n_jobs=N_JOBS)(
            delayed(_one_perm)(Xs, aucs, ys, SEED + 1 + k) for k in range(n_perm))
        ge_i = sum(pi >= intra_auctw for pi, _ in perms)
        ge_l = sum(pl >= loso_auctw for _, pl in perms)
        p_i = (1 + ge_i) / (1 + n_perm)
        p_l = (1 + ge_l) / (1 + n_perm)

    return dict(
        family=family, resolution=res or "-", range=rng_name, n_features=n_feat,
        intra_auc_tw=round(intra_auctw, 4), loso_auc_tw=round(loso_auctw, 4),
        intra_auc_logit=round(float(np.mean(auclo_i)), 4), loso_auc_logit=round(float(np.mean(auclo_l)), 4),
        intra_d2=round(intra_d2, 4), loso_d2=round(loso_d2, 4),
        p_intra=round(p_i, 4), p_loso=round(p_l, 4),
        loso_auc_tw_min=round(float(np.min(auctw_l)), 3), loso_auc_tw_max=round(float(np.max(auctw_l)), 3),
    ), dict(auctw_i=auctw_i, auctw_l=auctw_l, d2_i=d2_i, d2_l=d2_l)


# =============================================================================
# Step 4 -- figuras (espejo de 1.4: heatmap feature-set x rango + por sujeto)
# =============================================================================
def _disp_labels(summ, order):
    nf = summ.pivot(index="cell", columns="range", values="n_features")
    return {c: f"{c}  ({int(nf.loc[c, '1-30'])}/{int(nf.loc[c, '1-40'])} feat)" for c in order}


def plot_scheme(summ, persub, scheme, out_name):
    """scheme in {'intra','loso'}. (izq) heatmap AUC-Tweedie + baseline logistico; (der) por sujeto."""
    auc_col, p_col = f"{scheme}_auc_tw", f"p_{scheme}"
    logit_col = f"{scheme}_auc_logit"
    summ = summ.copy()
    summ["cell"] = summ["family"] + " [" + summ["resolution"].astype(str) + "]"
    order = _cell_order()
    disp = _disp_labels(summ, order)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    ax = axes[0]
    m = summ.pivot(index="cell", columns="range", values=auc_col).loc[order]
    pm = summ.pivot(index="cell", columns="range", values=p_col).loc[order]
    lm = summ.pivot(index="cell", columns="range", values=logit_col).loc[order]
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.45, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([disp[c] for c in order], fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            star = "*" if pm.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{m.values[i, j]:.2f}{star}\n(LR {lm.values[i, j]:.2f})",
                    ha="center", va="center", fontsize=8)
    ax.set_title(f"{scheme.upper()} AUC Tweedie (* p_perm<0.05); (LR x.xx)=logistico", fontsize=10.5)
    fig.colorbar(im, ax=ax, shrink=0.6, label="AUC")
    # ---- der: por sujeto, AUC Tweedie en rango 1-40 ----
    ax = axes[1]
    p40 = persub[persub["range"] == "1-40"].copy()
    p40["cell"] = p40["family"] + " [" + p40["resolution"].astype(str) + "]"
    subs = [s for s in COHORT if s in set(p40["subject"])]
    val_col = f"{scheme}_auc_tw"
    for xi, lab in enumerate(order):
        sub_df = p40[p40["cell"] == lab]
        for s in subs:
            v = sub_df[sub_df["subject"] == s][val_col]
            if len(v):
                ax.scatter(xi + np.linspace(-0.22, 0.22, len(subs))[subs.index(s)],
                           float(v.iloc[0]), color=SUBJ_COLORS[s], s=45,
                           label=s if xi == 0 else None, zorder=3)
        ax.hlines(sub_df[val_col].mean(), xi - 0.3, xi + 0.3, color="k", lw=2, zorder=2)
    ax.axhline(0.5, color="0.5", lw=1.0, ls="--", label="chance")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}" for c in order], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(f"{scheme} AUC Tweedie (por sujeto)")
    sub_t = "un modelo por sujeto" if scheme == "intra" else "AUC del sujeto dejado afuera"
    ax.set_title(f"Per-subject {scheme.upper()} (1-40): {sub_t}; barra negra = media", fontsize=10.5)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(f"1.4 Gamma -- Decoding Tweedie {scheme.upper()} (target=SMNA-AUC, N=6)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[1.4G] saved {out}", flush=True)


def plot_d2(summ):
    """Metrica nativa de la regresion: D2 de Tweedie (devianza explicada) intra y LOSO."""
    summ = summ.copy()
    summ["cell"] = summ["family"] + " [" + summ["resolution"].astype(str) + "]"
    order = _cell_order()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for ax, scheme in zip(axes, ["intra_d2", "loso_d2"]):
        m = summ.pivot(index="cell", columns="range", values=scheme).loc[order]
        vmax = float(np.nanmax(np.abs(m.values))) or 0.1
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(scheme.replace("_", " ").upper())
        fig.colorbar(im, ax=ax, shrink=0.6, label="D2 (devianza explicada)")
    fig.suptitle("1.4 Gamma -- D2 de Tweedie: cuanto de la MAGNITUD del arousal (SMNA-AUC) "
                 "explica el EEG (N=6)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "decoding_d2_gamma.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[1.4G] saved {out}", flush=True)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="solo correr assemble() y cachear target+features")
    ap.add_argument("--probe", action="store_true", help="una sola celda (periodico 1-40) para medir tiempo")
    ap.add_argument("--nperm", type=int, default=N_PERM)
    args = ap.parse_args()
    if args.build:
        build_cache(); return
    print("=" * 78)
    print("decoding_sweep_gamma :: 1.4 Gamma -- Tweedie regression (target = SMNA-AUC)")
    print(f"  power={POWER}  ridge_alpha={RIDGE_ALPHA}  nperm={args.nperm}")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    data, freqs, ch = load_target_and_features()
    if not data:
        print("[1.4G] sin datos alineados -- abort", flush=True); return
    print(f"[1.4G] {len(data)} sujetos, {len(ch)} canales, {len(freqs)} freqs", flush=True)

    if args.probe:
        r, _ = run_cell("periodic", None, "1-40", data, freqs, n_perm=5)
        print("\n[PROBE]", {k: r[k] for k in
              ["family", "range", "n_features", "intra_auc_tw", "intra_auc_logit",
               "intra_d2", "loso_auc_tw", "loso_auc_logit", "loso_d2", "p_intra", "p_loso"]}, flush=True)
        return

    rows, persub_rows = [], []
    for fam, res, rng_name in cells():
        r, det = run_cell(fam, res, rng_name, data, freqs, args.nperm)
        rows.append(r)
        for i, sub in enumerate(data):
            persub_rows.append(dict(family=fam, resolution=res or "-", range=rng_name, subject=sub,
                                    intra_auc_tw=det["auctw_i"][i], loso_auc_tw=det["auctw_l"][i],
                                    intra_d2=det["d2_i"][i], loso_d2=det["d2_l"][i]))
        print(f"  {r['family']:>20} [{r['resolution']:>4}] {r['range']}: nfeat={r['n_features']:>4}  "
              f"AUC_tw intra={r['intra_auc_tw']:.3f}(LR {r['intra_auc_logit']:.3f}) "
              f"LOSO={r['loso_auc_tw']:.3f}(LR {r['loso_auc_logit']:.3f})  "
              f"D2 intra={r['intra_d2']:.3f} LOSO={r['loso_d2']:.3f}  "
              f"p={r['p_intra']:.3f}/{r['p_loso']:.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(TBL_DIR / "sweep_gamma.csv", index=False)
    persub = pd.DataFrame(persub_rows); persub.to_csv(TBL_DIR / "per_subject_gamma.csv", index=False)
    plot_scheme(df, persub, "intra", "decoding_intra_gamma.png")
    plot_scheme(df, persub, "loso", "decoding_loso_gamma.png")
    plot_d2(df)
    print("\n[1.4G] done.", flush=True)


if __name__ == "__main__":
    main()
