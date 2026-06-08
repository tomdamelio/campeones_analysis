"""1.6 — Qué banda/parámetro maneja la caída del decoding al controlar el confound temporal.

Bloque 1 del ciclo 05_05 -> GATE F. El 1.5 mostró que el feature-set PERIÓDICO (potencia por
banda, 1/f removido) se debilita al emparejar temporalmente las épocas no-SCR (LOSO 0.611→0.560)
PERO sigue significativo (no colapsa). Acá descomponemos esa caída POR BANDA: ¿cuál banda pierde
AUC al emparejar, y cuál aguanta?

Pedido de Diego (literal): "hay que ver qué parte del espectro es... no me sorprendería que esto
tenga que ver con alfa. Por cuestiones de atención, fatiga, eso varía a lo largo de la sesión.
Pero eso no significa que esté mal, es un componente más que va modelando la varianza".

Diseño (decisiones del usuario 2026-06-07):
  - Feature por banda = potencia periódica de banda × canal (1/f removido), NO CF/PW por época
    (peak-detection por época es ruidoso; el shift de CF de alfa ya se caracterizó en 1.3).
  - Familias: delta/theta/alfa/beta/gamma (5 bandas) + offset + exponent (los 2 aperiódicos como
    CONTROL: el aperiódico sobrevivió al emparejado en 05_04, no deberían caer).
  - Para cada familia: intra + LOSO bajo UNIFORME vs EMPAREJADO, CIs bootstrap inter-sujeto,
    p-perm (subject-respecting, re-corriendo CV). ΔAUC = emparejado − uniforme.
  - Correlación de cada banda con el tiempo-en-run (tnorm) sobre épocas NO-SCR (el "fondo" que
    varía solo por tiempo): el feature que (a) cae al emparejar Y (b) correlaciona con tnorm =
    estado tónico/atención-fatiga (Diego: modela varianza, no está "mal").
  - Haufe (forward) por banda (esquema uniforme): dónde se apoya cada banda (posterior/central =
    cortical vs borde = músculo/ocular).

Reusa el cache panel_psd.npz (29 ch, AMBOS esquemas con psd+y+tnorm, construido en 1.5) y la
maquinaria de decoding_panel/decoding_sweep. Corre SECUENCIAL (sin paralelismo).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.matched_banda --probe
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.matched_banda [--nperm 1000]
"""

from __future__ import annotations

import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES,
    BANDS_40,
    _linear_aperiodic,
    intra_auc,
    loso_auc,
    _perm_within,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import (
    load_cache,
    _clf,
    _boot_ci,
    _make_info,
    _topo,
    haufe_pattern,  # not used directly (band-specific Haufe below) but kept for parity
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "matched_banda"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260607
RANGE = "1-40"   # incluye gamma (la banda EMG-sospechosa)


# =============================================================================
# Feature builders por familia (cada uno -> (n_ep, n_ch))
# =============================================================================
def feat_band(psd, freqs, rng, band):
    """Potencia periódica de UNA banda (1/f removido) por canal."""
    _, _, resid, f = _linear_aperiodic(psd, freqs, rng)
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)


def feat_offset(psd, freqs, rng):
    off, _, _, _ = _linear_aperiodic(psd, freqs, rng)
    return off


def feat_exponent(psd, freqs, rng):
    _, exp, _, _ = _linear_aperiodic(psd, freqs, rng)
    return exp


def _families():
    fams = []
    for bn, band in BANDS_40.items():          # delta, theta, alpha, beta, gamma
        fams.append((bn, (lambda p, f, r, b=band: feat_band(p, f, r, b)), "banda"))
    fams.append(("offset", feat_offset, "control"))
    fams.append(("exponent", feat_exponent, "control"))
    return fams


# =============================================================================
# AUC + CI + perm para una familia, en un esquema (uniforme/emparejado)
# =============================================================================
def eval_family(builder, data, freqs, rng, n_perm):
    Xs = [builder(data[s][0], freqs, rng) for s in data]
    ys = [data[s][1] for s in data]
    Xall = np.concatenate(Xs, axis=0)
    yall = np.concatenate(ys, axis=0)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    intra, intra_list = intra_auc(Xs, ys)
    loso, loso_list = loso_auc(Xall, yall, groups)
    ci_i = _boot_ci(intra_list); ci_l = _boot_ci(loso_list)
    p_intra = p_loso = np.nan
    if n_perm > 0:
        pr = np.random.default_rng(SEED + 1)
        ge_i = ge_l = 0
        for _ in range(n_perm):
            yp = _perm_within(yall, groups, pr)
            ys_p = [yp[groups == i] for i in range(len(ys))]
            ip, _ = intra_auc(Xs, ys_p)
            lp, _ = loso_auc(Xall, yp, groups)
            ge_i += (ip >= intra); ge_l += (lp >= loso)
        p_intra = (1 + ge_i) / (1 + n_perm)
        p_loso = (1 + ge_l) / (1 + n_perm)
    n = [int(len(y)) for y in ys]
    return dict(intra=round(intra, 4), loso=round(loso, 4),
                intra_ci=(round(ci_i[0], 3), round(ci_i[1], 3)),
                loso_ci=(round(ci_l[0], 3), round(ci_l[1], 3)),
                p_intra=p_intra, p_loso=p_loso, n_total=int(sum(n)),
                loso_list=[round(a, 3) for a in loso_list])


# =============================================================================
# Correlación de cada banda con tnorm (sobre épocas NO-SCR del esquema uniforme)
# =============================================================================
def tnorm_corr(builder, data_u, freqs, rng):
    """Spearman(potencia-media-de-banda, tnorm) por sujeto, en épocas no-SCR. Agrega signo+Wilcoxon."""
    rhos = {}
    for s in data_u:
        psd, y, tn = data_u[s]
        X = builder(psd, freqs, rng)          # (n_ep, n_ch)
        scalar = X.mean(axis=1)               # potencia media por época
        mask = y == 0                         # no-SCR
        if mask.sum() < 5:
            rhos[s] = np.nan; continue
        r = spearmanr(scalar[mask], tn[mask]).correlation
        rhos[s] = float(r)
    vals = np.array([rhos[s] for s in data_u if not np.isnan(rhos[s])])
    n_pos = int(np.sum(vals > 0))
    try:
        wp = float(wilcoxon(vals).pvalue) if len(vals) >= 5 else np.nan
    except Exception:
        wp = np.nan
    return rhos, float(np.mean(vals)), n_pos, len(vals), wp


# =============================================================================
# Haufe por banda (esquema uniforme): A = cov(X_std)·w -> topomap por banda
# =============================================================================
def band_haufe(builder, data_u, freqs, rng):
    Xs = [builder(data_u[s][0], freqs, rng) for s in data_u]
    ys = [data_u[s][1] for s in data_u]
    Xall = np.concatenate(Xs, axis=0); yall = np.concatenate(ys, axis=0)
    pipe = _clf().fit(Xall, yall)
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    Xstd = scaler.transform(Xall)
    return np.cov(Xstd, rowvar=False) @ lr.coef_[0]   # (n_ch,)


# =============================================================================
# Plots
# =============================================================================
def plot_byband(df, data_u):
    """ΔAUC y AUC uniforme vs emparejado por familia (intra y LOSO)."""
    order = list(df["family"])
    subs = list(data_u.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, sch in zip(axes, ["intra", "loso"]):
        x = np.arange(len(order)); w = 0.38
        u = df[f"{sch}_unif"].values; m = df[f"{sch}_match"].values
        ax.bar(x - w / 2, u, w, label="uniforme (sin control temporal)", color="C0", alpha=0.8)
        ax.bar(x + w / 2, m, w, label="emparejado (con control temporal)", color="C1", alpha=0.8)
        for xi, (uu, mm) in enumerate(zip(u, m)):
            ax.annotate(f"Δ{mm-uu:+.02f}", (xi, max(uu, mm) + 0.005), ha="center", fontsize=7,
                        color="darkred" if mm - uu < -0.02 else "0.3")
        ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(f"{sch.upper()} AUC"); ax.set_ylim(0.45, 0.75)
        ax.set_title(f"{sch.upper()} por banda/parámetro: uniforme vs emparejado", fontsize=10)
        if sch == "intra":
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("1.6 ¿Qué banda maneja la caída al controlar el tiempo? (29 ch, N=6)\n"
                 "Controles offset/exponent (aperiódico) NO deberían caer.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "byband_auc.png", dpi=120); plt.close(fig)


def plot_drop_tnorm(df, corr_rows, data_u):
    """La(s) banda(s) que cae(n): ΔAUC LOSO + correlación con tnorm por sujeto."""
    bands = df[df["kind"] == "banda"].copy()
    subs = list(data_u.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: dAUC_loso por banda
    ax = axes[0]
    x = np.arange(len(bands))
    cols = ["darkred" if d < -0.02 else "0.5" for d in bands["dAUC_loso"]]
    ax.bar(x, bands["dAUC_loso"], color=cols)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bands["family"], fontsize=9)
    ax.set_ylabel("ΔAUC LOSO (emparejado − uniforme)")
    ax.set_title("Caída de AUC al controlar el tiempo (rojo = cae >0.02)", fontsize=10)
    # right: corr con tnorm por sujeto, por banda
    ax = axes[1]
    cmap = corr_rows.set_index("family")
    for xi, bn in enumerate(bands["family"]):
        for k, s in enumerate(subs):
            v = cmap.loc[bn, f"rho_{s}"]
            if not np.isnan(v):
                ax.scatter(xi + np.linspace(-0.2, 0.2, len(subs))[k], v, color=SUBJ_COLORS[s], s=40, zorder=3)
        ax.hlines(cmap.loc[bn, "mean_rho"], xi - 0.28, xi + 0.28, color="k", lw=2, zorder=2)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands["family"], fontsize=9)
    ax.set_ylabel("Spearman(potencia, tiempo-en-run)  [épocas no-SCR]")
    ax.set_title("Correlación con el tiempo (barra negra = media)", fontsize=10)
    fig.suptitle("1.6 La banda que CAE al emparejar Y correlaciona con el tiempo = estado tónico", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "drop_vs_tnorm.png", dpi=120); plt.close(fig)


def plot_band_haufe(haufes, ch):
    info = _make_info(ch)
    bnames = list(BANDS_40.keys())
    fig, axes = plt.subplots(1, len(bnames), figsize=(2.4 * len(bnames), 3.2))
    for j, bn in enumerate(bnames):
        _topo(axes[j], haufes[bn], info, bn)
    fig.suptitle("1.6 Haufe forward por banda (esquema uniforme): dónde se apoya el modelo\n"
                 "(posterior/central = cortical · borde = músculo/ocular)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "haufe_byband.png", dpi=120); plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def _probe(data_u, freqs):
    rng = RANGES[RANGE]
    b = (lambda p, f, r: feat_band(p, f, r, BANDS_40["alpha"]))
    Xs = [b(data_u[s][0], freqs, rng) for s in data_u]
    ys = [data_u[s][1] for s in data_u]
    Xall = np.concatenate(Xs); yall = np.concatenate(ys)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    pr = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(5):
        yp = _perm_within(yall, groups, pr)
        ys_p = [yp[groups == i] for i in range(len(ys))]
        intra_auc(Xs, ys_p); loso_auc(Xall, yp, groups)
    dt = (time.time() - t0) / 5
    print(f"[probe] {Xall.shape[1]} feats/familia, {Xall.shape[0]} épocas. ~{dt:.2f}s/perm.")
    print(f"[probe] 7 familias × 2 esquemas × 1000 perms ≈ {dt*7*2*1000/60:.0f} min.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78)
    print("matched_banda :: 1.6 qué banda maneja la caída al controlar el tiempo")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    data_u, freqs, ch = load_cache("uniform")
    data_m, _, _ = load_cache("matched")
    rng = RANGES[RANGE]
    print(f"[1.6] cache: uniforme={len(data_u)} suj, emparejado={len(data_m)} suj, {len(ch)} ch", flush=True)
    if args.probe:
        _probe(data_u, freqs); return

    fams = _families()
    rows, corr_rows, haufes = [], [], {}
    for name, builder, kind in fams:
        ru = eval_family(builder, data_u, freqs, rng, args.nperm)
        rm = eval_family(builder, data_m, freqs, rng, args.nperm)
        rhos, mrho, n_pos, n_val, wp = tnorm_corr(builder, data_u, freqs, rng)
        row = dict(family=name, kind=kind,
                   intra_unif=ru["intra"], intra_match=rm["intra"],
                   dAUC_intra=round(rm["intra"] - ru["intra"], 4),
                   loso_unif=ru["loso"], loso_match=rm["loso"],
                   dAUC_loso=round(rm["loso"] - ru["loso"], 4),
                   p_loso_unif=ru["p_loso"], p_loso_match=rm["p_loso"],
                   loso_ci_unif=str(ru["loso_ci"]), loso_ci_match=str(rm["loso_ci"]),
                   n_unif=ru["n_total"], n_match=rm["n_total"])
        rows.append(row)
        cr = dict(family=name, mean_rho=round(mrho, 3), n_pos=f"{n_pos}/{n_val}", wilcoxon_p=round(wp, 4))
        for s in data_u:
            cr[f"rho_{s}"] = round(rhos[s], 3) if not np.isnan(rhos[s]) else np.nan
        corr_rows.append(cr)
        if kind == "banda":
            haufes[name] = band_haufe(builder, data_u, freqs, rng)
        print(f"  {name:9s} ({kind}): LOSO {ru['loso']:.3f}→{rm['loso']:.3f} "
              f"(Δ{rm['loso']-ru['loso']:+.3f}, p_u={ru['p_loso']}, p_m={rm['p_loso']})  "
              f"corr_tnorm={mrho:+.2f} ({n_pos}/{n_val})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "byband_uniform_vs_matched.csv", index=False)
    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv(TBL_DIR / "feature_tnorm_corr.csv", index=False)

    plot_byband(df, data_u)
    plot_drop_tnorm(df, cdf, data_u)
    plot_band_haufe(haufes, ch)

    # veredicto P3 textual
    bands = df[df["kind"] == "banda"]
    worst = bands.loc[bands["dAUC_loso"].idxmin()]
    print("\n[1.6] === VEREDICTO P3 (preliminar) ===", flush=True)
    print(f"  Banda que MÁS cae (LOSO) al emparejar: {worst['family']} "
          f"(Δ{worst['dAUC_loso']:+.3f}: {worst['loso_unif']}→{worst['loso_match']})", flush=True)
    print(f"  Controles aperiódicos: offset Δ{df[df.family=='offset']['dAUC_loso'].iloc[0]:+.3f}, "
          f"exponent Δ{df[df.family=='exponent']['dAUC_loso'].iloc[0]:+.3f} (deberían ~0)", flush=True)
    print(f"\n[1.6] done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
