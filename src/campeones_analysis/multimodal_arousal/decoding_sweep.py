"""1.4 — Decoding con TODOS los bins + barrido de robustez al binning.

Bloque 1 del ciclo 05_05. Pedido de Pablo: decodificar SCR vs no-SCR con bins (no bandas)
y barrer granularidad x rango para mostrar que el efecto es robusto a como se particiona el
espectro. Diego: "sistematiza eso de alguna manera".

Familias de features (31 canales EEG), conteos EXPLICITOS (se reportan en sweep_auc.csv):
  - all-bins   : log-power por (bin x canal), re-bineado a la resolucion objetivo.
  - aperiodic  : offset + exponent por canal (fit lineal log-log por epoca). 2 x 31 = 62.
  - periodic   : potencia por banda con aperiodico REMOVIDO (flattened), por canal.
                 NO usa CF/PW/BW por epoca (peak-detection por epoca unica = ruidoso) ->
                 usa la potencia periodica por banda (robusta, interpretable, = band_decomp).
  - aperiodic+periodic : union de las dos.
  - bandpower  : log band-power CRUDO (sin remover 1/f), por canal (ref. del 05_04).
                 "periodic vs bandpower" mide si remover el 1/f ayuda o no al decoding.

Grilla: familia x resolucion(all-bins) x rango {1-30, 1-40}. NO >45 Hz (banda EMG).
Clasificador: LogisticRegression L2 (ridge) sobre features estandarizadas, SIN PCA
(mantiene interpretabilidad via Haufe en 1.5; ridge maneja p>>n sin truncar direcciones).
CV: intra (within-subject, repeated stratified k-fold) + LOSO (leave-one-subject-out).
Significancia por permutacion (subject-respecting), chance != 0.5 a N chico (Combrisson 2015).

Cache: el PSD por epoca se computa UNA vez (build_subject_epochs, ~lento) y se guarda en npz;
el barrido deriva todas las familias/resoluciones del cache sin reconstruir epocas.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_sweep --build
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_sweep --counts
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_sweep
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS, DROP_CHANNELS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs, compute_psd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "decoding_sweep"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
CACHE = TBL_DIR / "perepoch_psd.npz"

RANGES = {"1-30": (1.0, 30.0), "1-40": (1.0, 40.0)}
BANDS_30 = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BANDS_40 = {**BANDS_30, "gamma": (30, 40)}
RNG = np.random.default_rng(20260607)
N_PERM = 100
SEED = 20260607


# =============================================================================
# Step 1 -- build per-epoch PSD cache (the slow part: epoch rebuild)
# =============================================================================
def build_cache() -> None:
    print("[1.4] building per-epoch PSD cache ...", flush=True)
    store = {}
    ref_ch = None
    for sub in COHORT:
        print(f"[1.4] {sub}: building epochs ...", flush=True)
        real, silent = build_subject_epochs(sub)
        if real is None or silent is None:
            print(f"[1.4] {sub}: skipped", flush=True)
            continue
        psd_r, freqs, ch = compute_psd(real)
        psd_s, _, _ = compute_psd(silent)
        if ref_ch is None:
            ref_ch = ch
        if ch != ref_ch:
            # align to common channels in ref order
            idx = [ch.index(c) for c in ref_ch if c in ch]
            psd_r = psd_r[:, idx, :]; psd_s = psd_s[:, idx, :]
        psd = np.concatenate([psd_r, psd_s], axis=0)         # (n_ep, n_ch, n_freq)
        y = np.concatenate([np.ones(len(psd_r)), np.zeros(len(psd_s))]).astype(int)
        store[f"psd_{sub}"] = psd.astype(np.float32)
        store[f"y_{sub}"] = y
        print(f"[1.4] {sub}: {psd.shape[0]} epochs ({len(psd_r)} SCR / {len(psd_s)} no-SCR), "
              f"{psd.shape[1]} ch x {psd.shape[2]} freqs", flush=True)
    store["freqs"] = freqs
    store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(CACHE, **store)
    print(f"[1.4] cache saved -> {CACHE}", flush=True)


def load_cache():
    """Carga el cache y DESCARTA DROP_CHANNELS (Track A, 29 ch). El cache guarda los 32 ch crudos;
    el drop se aplica en load (los 29 restantes no cambian; no reconstruye épocas)."""
    z = np.load(CACHE, allow_pickle=True)
    freqs = z["freqs"]; ch = list(z["ch_names"])
    keep = [i for i, c in enumerate(ch) if c not in DROP_CHANNELS]
    ch_kept = [ch[i] for i in keep]
    data = {}
    for sub in COHORT:
        if f"psd_{sub}" in z:
            data[sub] = (z[f"psd_{sub}"].astype(float)[:, keep, :], z[f"y_{sub}"])
    return data, freqs, ch_kept


# =============================================================================
# Step 2 -- feature builders (from cached per-epoch PSD)
# =============================================================================
def _range_mask(freqs, rng):
    lo, hi = rng
    return (freqs >= lo) & (freqs <= hi)


def feat_allbins(psd, freqs, rng, res):
    """log-power per (rebinned freq x channel). res in {1,2,4} Hz or 'bands'."""
    lo, hi = rng
    bands = BANDS_40 if hi > 30 else BANDS_30
    if res == "bands":
        edges = [(b[0], b[1]) for b in bands.values()]
    else:
        e = np.arange(lo, hi + 1e-9, float(res))
        edges = list(zip(e[:-1], e[1:]))
    cols = []
    for (a, b) in edges:
        m = (freqs >= a) & (freqs < b)
        if m.sum() == 0:
            continue
        cols.append(np.log10(psd[:, :, m].mean(axis=2) + 1e-30))  # (n_ep, n_ch)
    X = np.concatenate(cols, axis=1)   # (n_ep, n_bins*n_ch)
    return X


def _linear_aperiodic(psd, freqs, rng):
    """Per epoch per channel: linear log-log fit -> (offset, exponent) and the residual."""
    m = _range_mask(freqs, rng)
    f = freqs[m]
    lf = np.log10(f)
    lp = np.log10(psd[:, :, m] + 1e-30)        # (n_ep, n_ch, n_f)
    A = np.vstack([lf, np.ones_like(lf)]).T    # (n_f, 2)
    ne, nc, nf = lp.shape
    Y = lp.reshape(-1, nf).T                    # (n_f, ne*nc)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)   # (2, ne*nc)
    slope = coef[0].reshape(ne, nc)
    intercept = coef[1].reshape(ne, nc)
    fit = (slope[:, :, None] * lf[None, None, :]) + intercept[:, :, None]
    resid = lp - fit                            # (n_ep, n_ch, n_f) aperiodic-removed
    offset = intercept                          # (n_ep, n_ch)
    exponent = -slope                           # (n_ep, n_ch)
    return offset, exponent, resid, f


def feat_aperiodic(psd, freqs, rng):
    off, exp, _, _ = _linear_aperiodic(psd, freqs, rng)
    return np.concatenate([off, exp], axis=1)   # (n_ep, 2*n_ch)


def feat_periodic(psd, freqs, rng):
    _, _, resid, f = _linear_aperiodic(psd, freqs, rng)
    hi = rng[1]
    bands = BANDS_40 if hi > 30 else BANDS_30
    cols = []
    for (a, b) in bands.values():
        m = (f >= a) & (f < b)
        if m.sum() == 0:
            continue
        cols.append(np.clip(resid[:, :, m], 0, None).mean(axis=2))   # (n_ep, n_ch)
    return np.concatenate(cols, axis=1)


def feat_aperiodic_periodic(psd, freqs, rng):
    return np.concatenate([feat_aperiodic(psd, freqs, rng), feat_periodic(psd, freqs, rng)], axis=1)


def feat_bandpower(psd, freqs, rng):
    """log band-power CRUDO (sin remover 1/f), por canal."""
    hi = rng[1]
    bands = BANDS_40 if hi > 30 else BANDS_30
    cols = []
    for (a, b) in bands.values():
        m = (freqs >= a) & (freqs < b)
        cols.append(np.log10(psd[:, :, m].mean(axis=2) + 1e-30))
    return np.concatenate(cols, axis=1)


def build_X(family, psd, freqs, rng, res=None):
    if family == "all-bins":
        return feat_allbins(psd, freqs, rng, res)
    if family == "aperiodic":
        return feat_aperiodic(psd, freqs, rng)
    if family == "periodic":
        return feat_periodic(psd, freqs, rng)
    if family == "aperiodic+periodic":
        return feat_aperiodic_periodic(psd, freqs, rng)
    if family in ("bandpower(raw)", "bandpower"):
        return feat_bandpower(psd, freqs, rng)
    raise ValueError(family)


# the cells of the sweep. NOTE: 'bandpower(raw)' = log mean power per canonical band per channel
# (= the old 'all-bins [bands]', identical computation); kept as the RAW comparator to 'periodic'
# (aperiodic-removed). The redundant 'all-bins [bands]' / separate 'bandpower' cells were removed.
def cells():
    out = []
    for rng in RANGES:
        for res in ["1", "2", "4"]:
            out.append(("all-bins", res, rng))
        for fam in ["bandpower(raw)", "aperiodic", "periodic", "aperiodic+periodic"]:
            out.append((fam, None, rng))
    return out


# =============================================================================
# Step 3 -- classifiers / CV / permutation
# =============================================================================
def _clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs"))


def intra_auc(Xs, ys):
    """Within-subject 5-fold stratified AUC, averaged across subjects."""
    aucs = []
    for X, y in zip(Xs, ys):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        proba = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            clf = _clf().fit(X[tr], y[tr])
            proba[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, proba))
    return float(np.mean(aucs)), aucs


def loso_auc(Xall, yall, groups):
    """Leave-one-subject-out AUC, averaged across held-out subjects."""
    aucs = []
    for g in np.unique(groups):
        tr = groups != g; te = groups == g
        clf = _clf().fit(Xall[tr], yall[tr])
        proba = clf.predict_proba(Xall[te])[:, 1]
        aucs.append(roc_auc_score(yall[te], proba))
    return float(np.mean(aucs)), aucs


def _perm_within(y, groups, rng):
    yp = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yp[idx] = rng.permutation(y[idx])
    return yp


def run_cell(family, res, rng_name, data, freqs):
    rng = RANGES[rng_name]
    Xs, ys = [], []
    for sub in data:
        psd, y = data[sub]
        Xs.append(build_X(family, psd, freqs, rng, res))
        ys.append(y)
    Xall = np.concatenate(Xs, axis=0)
    yall = np.concatenate(ys, axis=0)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    n_feat = Xall.shape[1]

    intra, _ = intra_auc(Xs, ys)
    loso, loso_subj = loso_auc(Xall, yall, groups)

    # permutation (subject-respecting) on BOTH schemes
    pr = np.random.default_rng(SEED + 1)
    ge_intra = ge_loso = 0
    for _ in range(N_PERM):
        yp = _perm_within(yall, groups, pr)
        ys_p = [yp[groups == i] for i in range(len(ys))]
        Xs_p = Xs  # features unchanged, only labels permuted
        ip, _ = intra_auc(Xs_p, ys_p)
        lp, _ = loso_auc(Xall, yp, groups)
        ge_intra += (ip >= intra)
        ge_loso += (lp >= loso)
    p_intra = (1 + ge_intra) / (1 + N_PERM)
    p_loso = (1 + ge_loso) / (1 + N_PERM)
    return dict(family=family, resolution=res or "-", range=rng_name, n_features=n_feat,
                intra_auc=round(intra, 4), loso_auc=round(loso, 4),
                p_intra=round(p_intra, 4), p_loso=round(p_loso, 4),
                loso_min=round(min(loso_subj), 3), loso_max=round(max(loso_subj), 3))


# =============================================================================
# counts table + heatmap
# =============================================================================
def print_counts(data, freqs):
    rows = []
    for fam, res, rng_name in cells():
        rng = RANGES[rng_name]
        sub0 = next(iter(data))
        X = build_X(fam, data[sub0][0], freqs, rng, res)
        rows.append({"family": fam, "resolution": res or "-", "range": rng_name,
                     "n_features": X.shape[1]})
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "feature_counts.csv", index=False)
    print("\n[1.4] FEATURE COUNTS per cell:")
    print(df.to_string(index=False), flush=True)


def compute_persubject(data, freqs):
    """Per-subject AUCs (NO permutations -> fast): intra = within-subject model per subject;
    LOSO = AUC on each held-out subject."""
    subs = list(data.keys())
    rows = []
    for fam, res, rng_name in cells():
        rng = RANGES[rng_name]
        Xs = [build_X(fam, data[s][0], freqs, rng, res) for s in subs]
        ys = [data[s][1] for s in subs]
        Xall = np.concatenate(Xs, axis=0)
        yall = np.concatenate(ys, axis=0)
        groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
        _, intra_list = intra_auc(Xs, ys)
        _, loso_list = loso_auc(Xall, yall, groups)   # order = sorted group id = subs order
        for i, s in enumerate(subs):
            rows.append({"family": fam, "resolution": res or "-", "range": rng_name,
                         "subject": s, "intra_auc": intra_list[i], "loso_auc": loso_list[i]})
    return pd.DataFrame(rows)


def _cell_order():
    seen, order = set(), []
    for fam, res, _ in cells():
        lab = f"{fam} [{res or '-'}]"
        if lab not in seen:
            seen.add(lab); order.append(lab)
    return order


def plot_scheme(summ, persub, scheme, out_name):
    """One figure, 2 panels: (left) AUC heatmap feature-set x range; (right) per-subject AUC."""
    auc_col, p_col = f"{scheme}_auc", f"p_{scheme}"
    summ = summ.copy()
    summ["cell"] = summ["family"] + " [" + summ["resolution"].astype(str) + "]"
    order = _cell_order()
    # display label with feature count (n_features at 1-30 / 1-40)
    nf = summ.pivot(index="cell", columns="range", values="n_features")
    disp = {c: f"{c}  ({int(nf.loc[c, '1-30'])}/{int(nf.loc[c, '1-40'])} feat)" for c in order}
    disp_labels = [disp[c] for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    # ---- left: heatmap (AUC + * if p<0.05) ----
    ax = axes[0]
    m = summ.pivot(index="cell", columns="range", values=auc_col).loc[order]
    pm = summ.pivot(index="cell", columns="range", values=p_col).loc[order]
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.45, vmax=0.75, aspect="auto")
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(disp_labels, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            star = "*" if pm.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{m.values[i, j]:.2f}{star}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{scheme.upper()} AUC: feature-set x range (* p_perm<0.05)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label="AUC")
    # ---- right: per-subject AUC at range 1-40 ----
    ax = axes[1]
    p40 = persub[persub["range"] == "1-40"].copy()
    p40["cell"] = p40["family"] + " [" + p40["resolution"].astype(str) + "]"
    subs = [s for s in COHORT if s in set(p40["subject"])]
    for xi, lab in enumerate(order):
        sub_df = p40[p40["cell"] == lab]
        for s in subs:
            v = sub_df[sub_df["subject"] == s][auc_col]
            if len(v):
                ax.scatter(xi + np.linspace(-0.22, 0.22, len(subs))[subs.index(s)],
                           float(v.iloc[0]), color=SUBJ_COLORS[s], s=45,
                           label=s if xi == 0 else None, zorder=3)
        ax.hlines(sub_df[auc_col].mean(), xi - 0.3, xi + 0.3, color="k", lw=2, zorder=2)
    ax.axhline(0.5, color="0.5", lw=1.0, ls="--", label="chance")
    # x-labels with the 1-40 feature count (right panel is range 1-40)
    xlabs = [f"{c}\n({int(nf.loc[c, '1-40'])} feat)" for c in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xlabs, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(f"{scheme} AUC (per subject)")
    sub_t = ("un modelo por sujeto" if scheme == "intra"
             else "AUC del sujeto dejado afuera")
    ax.set_title(f"Per-subject {scheme.upper()} (range 1-40): {sub_t}; black bar = mean", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(f"1.4 Decoding {scheme.upper()} -- aggregate + per-subject (N=6)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"[1.4] saved {out}", flush=True)


def plot_heatmap(df):
    piv = df.copy()
    piv["cell"] = piv["family"] + " [" + piv["resolution"] + "]"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, scheme in zip(axes, ["intra_auc", "loso_auc"]):
        pmetric = "p_intra" if scheme == "intra_auc" else "p_loso"
        m = piv.pivot(index="cell", columns="range", values=scheme)
        pm = piv.pivot(index="cell", columns="range", values=pmetric)
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.45, vmax=0.75, aspect="auto")
        ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                star = "*" if pm.values[i, j] < 0.05 else ""
                ax.text(j, i, f"{m.values[i, j]:.2f}{star}", ha="center", va="center", fontsize=8)
        ax.set_title(scheme.replace("_auc", "").upper() + "  (* p_perm<0.05)")
        fig.colorbar(im, ax=ax, shrink=0.6, label="AUC")
    fig.suptitle("1.4 Decoding AUC sweep: feature-set x resolution x range (N=6). "
                 "Robustness to binning if AUC>chance across cells.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "sweep_heatmap.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.4] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build per-epoch PSD cache (slow)")
    ap.add_argument("--counts", action="store_true", help="print feature counts only")
    ap.add_argument("--figs", action="store_true",
                    help="regenerate intra/loso per-subject figures from cache + sweep_auc.csv (no perms)")
    args = ap.parse_args()
    print("=" * 78)
    print("decoding_sweep :: 1.4 all-bins + binning sweep")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(); return
    if not CACHE.exists():
        print("[1.4] no cache -- run with --build first", flush=True); return
    data, freqs, ch = load_cache()
    print(f"[1.4] cache: {len(data)} subjects, {len(ch)} channels, {len(freqs)} freqs", flush=True)
    if args.counts:
        print_counts(data, freqs); return
    if args.figs:
        summ = pd.read_csv(TBL_DIR / "sweep_auc.csv")
        # bandpower(raw) == old 'all-bins [bands]' == old 'bandpower' (identical). Drop the
        # redundant all-bins[bands], relabel 'bandpower' -> 'bandpower(raw)'.
        summ = summ[~((summ.family == "all-bins") & (summ.resolution == "bands"))].copy()
        summ.loc[summ.family == "bandpower", "family"] = "bandpower(raw)"
        persub = compute_persubject(data, freqs)
        persub.to_csv(TBL_DIR / "per_subject_auc.csv", index=False)
        plot_scheme(summ, persub, "intra", "decoding_intra.png")
        plot_scheme(summ, persub, "loso", "decoding_loso.png")
        print("[1.4] per-subject figures regenerated.", flush=True); return
    print_counts(data, freqs)
    rows = []
    for fam, res, rng_name in cells():
        r = run_cell(fam, res, rng_name, data, freqs)
        rows.append(r)
        print(f"  {r['family']:>20} [{r['resolution']:>5}] {r['range']}: "
              f"nfeat={r['n_features']:>4}  intra={r['intra_auc']:.3f}(p={r['p_intra']:.3f})  "
              f"LOSO={r['loso_auc']:.3f}(p={r['p_loso']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "sweep_auc.csv", index=False)
    plot_heatmap(df)
    print("\n[1.4] done.", flush=True)


if __name__ == "__main__":
    main()
