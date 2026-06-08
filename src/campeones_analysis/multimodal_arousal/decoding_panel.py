"""1.5 — Panel intra + LOSO lado a lado, permutaciones(1000) y patrones de Haufe -> GATE E.

Bloque 1 del ciclo 05_05. Ensambla el panel comparativo que pidieron Diego (intra Y
cross-subject juntos, "creo que hay mucha variabilidad entre sujetos") y Tomas (las
permutaciones que faltaban sobre los feature-sets nuevos), con:

  - 4 feature-sets (nombres estandar): espectro completo (all-bins 1 Hz), aperiodico
    (offset+exp), periodico (band-power 1/f-removido), aperiodico+periodico.
  - columnas: intra AUC +- CI, LOSO AUC +- CI, p-perm intra, p-perm LOSO.
  - CIs por bootstrap sobre los 6 AUCs por-sujeto (dispersion inter-sujeto, Varoquaux 2018),
    NO el SE entre folds.
  - permutaciones (1000) subject-respecting, re-corriendo TODO el ciclo de CV con labels
    permutadas DENTRO de cada sujeto (Valente 2021).
  - patrones forward de Haufe (A = cov(X_std) . w) por feature-set: DONDE mira el decoder
    (banda/canal); pesos crudos NO interpretables (Haufe 2014).
  - framing honesto: el periodico (desync) bajo no-SCR UNIFORME vs EMPAREJADO (puente a 1.6).

Reusa TODO de decoding_sweep.py (feature builders, clasificador LR-L2 sin PCA, intra/loso/perm).
La unica maquinaria nueva es el cache `panel_psd.npz`, que -a diferencia del cache de 1.4-
guarda AMBOS esquemas de epocas {uniforme, emparejado} con psd + y + tnorm por epoca. El
tnorm (fraccion temporal del onset en el run) es la dependencia dura de 1.6; se construye una
sola vez aca y 1.6 lo reutiliza.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_panel --build
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_panel --probe
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_panel [--nperm 1000]
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

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS, DROP_CHANNELS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import compute_psd
import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
from src.campeones_analysis.multimodal_arousal.epoch_matched import sample_silent_matched
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES,
    BANDS_30,
    BANDS_40,
    feat_allbins,
    feat_aperiodic,
    feat_periodic,
    feat_aperiodic_periodic,
    intra_auc,
    loso_auc,
    _clf,
    _perm_within,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "decoding_panel"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
CACHE = TBL_DIR / "panel_psd.npz"

SEED = 20260607
# Permutaciones escalonadas: 1000 para los sets livianos (aperiódico/periódico/aper+per, donde
# Tomás señaló que faltaban); 300 para 'espectro completo' (1248 feats, ~11 s/perm -> 1000 serían
# ~3 h; ese set YA tenía perms en 1.4 y su AUC ~0.70 deja p en el piso de cualquier modo).
HEAVY_NPERM = 300
HEAVY_SETS = {"espectro completo"}
UNIFORM_SEED = 20260513   # erp_scr import-time seed (reproduce uniform silents)
MATCHED_SEED = 20260603   # epoch_matched per-subject seed
LOWPASS = 40.0
RESAMPLE = 250.0

# 4 feature-sets del panel (nombres estandar). all-bins a 1 Hz = espectro completo.
FEATURE_SETS = {
    "espectro completo": lambda psd, freqs, rng: feat_allbins(psd, freqs, rng, "1"),
    "aperiódico": feat_aperiodic,
    "periódico": feat_periodic,
    "aperiódico+periódico": feat_aperiodic_periodic,
}
PANEL_RANGES = ["1-30", "1-40"]
PRIMARY_RANGE = "1-40"


# =============================================================================
# Cache: build BOTH schemes (uniform / matched) with psd + y + tnorm per epoch
# =============================================================================
def _epoch_times_tnorm(ep, duration):
    """tnorm = onset-fraction-in-run per surviving epoch (events sample / sfreq / dur)."""
    if ep is None or len(ep) == 0:
        return np.empty(0)
    sf = float(ep.info["sfreq"])
    return (ep.events[:, 0].astype(float) / sf) / duration


def _build_one(sub, scheme):
    """Mirror tfr_psd_scr.build_subject_epochs (uniform) / epoch_matched (matched), but also
    capture per-epoch tnorm. Returns (psd, y, tnorm, ch) or None."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    if scheme == "matched":
        _erp.RNG = np.random.default_rng(MATCHED_SEED)  # epoch_matched resets per subject

    real_eps, sil_eps, real_tn, sil_tn = [], [], [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=LOWPASS, verbose="ERROR")
            raw.resample(RESAMPLE, verbose="ERROR")
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(onsets[onsets < dur], eda, EDA_FS)
            if scheme == "uniform":
                sil_t = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                               fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            else:
                sil_t = sample_silent_matched(onsets, dur, eda, EDA_FS, _erp.RNG)
            er = epoch_one_run(raw, onsets, code=1)
            si = epoch_one_run(raw, sil_t, code=2)
            if er is not None:
                real_eps.append(er); real_tn.append(_epoch_times_tnorm(er, dur))
            if si is not None:
                sil_eps.append(si); sil_tn.append(_epoch_times_tnorm(si, dur))
        except Exception as e:
            print(f"    {label}: FAILED -- {e}", flush=True)
    if not real_eps or not sil_eps:
        return None
    real = mne.concatenate_epochs(real_eps, verbose="ERROR")
    sil = mne.concatenate_epochs(sil_eps, verbose="ERROR")
    psd_r, freqs, ch = compute_psd(real)
    psd_s, _, _ = compute_psd(sil)
    psd = np.concatenate([psd_r, psd_s], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(psd_r)), np.zeros(len(psd_s))]).astype(int)
    tn = np.concatenate([np.concatenate(real_tn), np.concatenate(sil_tn)]).astype(np.float32)
    return psd, y, tn, ch, freqs


def build_cache():
    print("[1.5] building panel cache (uniform + matched, with tnorm) ...", flush=True)
    store = {}
    ref_ch = None
    freqs = None
    for scheme in ("uniform", "matched"):
        if scheme == "uniform":
            _erp.RNG = np.random.default_rng(UNIFORM_SEED)
        print(f"[1.5] scheme={scheme}", flush=True)
        for sub in COHORT:
            r = _build_one(sub, scheme)
            if r is None:
                print(f"  {scheme} {sub}: skipped", flush=True); continue
            psd, y, tn, ch, fr = r
            if ref_ch is None:
                ref_ch = ch; freqs = fr
            if ch != ref_ch:
                idx = [ch.index(c) for c in ref_ch if c in ch]
                psd = psd[:, idx, :]
            store[f"psd_{scheme}_{sub}"] = psd
            store[f"y_{scheme}_{sub}"] = y
            store[f"tnorm_{scheme}_{sub}"] = tn
            print(f"  {scheme} {sub}: {psd.shape[0]} ep ({int(y.sum())} SCR / {int((1-y).sum())} no-SCR)",
                  flush=True)
    store["freqs"] = freqs
    store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(CACHE, **store)
    print(f"[1.5] cache saved -> {CACHE}", flush=True)


def load_cache(scheme):
    """Carga el cache y DESCARTA DROP_CHANNELS (Track A, 29 ch). El cache guarda los 32 ch
    crudos; el drop se aplica en load para no reconstruir épocas (los 29 restantes no cambian)."""
    z = np.load(CACHE, allow_pickle=True)
    freqs = z["freqs"]; ch = list(z["ch_names"])
    keep = [i for i, c in enumerate(ch) if c not in DROP_CHANNELS]
    ch_kept = [ch[i] for i in keep]
    data = {}
    for sub in COHORT:
        k = f"psd_{scheme}_{sub}"
        if k in z:
            data[sub] = (z[k].astype(float)[:, keep, :], z[f"y_{scheme}_{sub}"], z[f"tnorm_{scheme}_{sub}"])
    return data, freqs, ch_kept


# =============================================================================
# AUC + bootstrap CI + permutation p
# =============================================================================
def _build_Xs(fset, data, freqs, rng):
    fn = FEATURE_SETS[fset]
    Xs = [fn(data[s][0], freqs, rng) for s in data]
    ys = [data[s][1] for s in data]
    return Xs, ys


def _boot_ci(vals, n_boot=2000, seed=SEED):
    """Percentile bootstrap CI over per-subject AUCs (between-subject dispersion)."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_fset(fset, data, freqs, rng_name, n_perm):
    rng = RANGES[rng_name]
    Xs, ys = _build_Xs(fset, data, freqs, rng)
    Xall = np.concatenate(Xs, axis=0)
    yall = np.concatenate(ys, axis=0)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    n_feat = Xall.shape[1]

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

    return dict(feature_set=fset, range=rng_name, n_features=n_feat,
                intra_auc=round(intra, 4), intra_ci_lo=round(ci_i[0], 4), intra_ci_hi=round(ci_i[1], 4),
                loso_auc=round(loso, 4), loso_ci_lo=round(ci_l[0], 4), loso_ci_hi=round(ci_l[1], 4),
                p_intra=p_intra, p_loso=p_loso,
                intra_subj=[round(a, 3) for a in intra_list],
                loso_subj=[round(a, 3) for a in loso_list])


# =============================================================================
# Haufe forward patterns: A = cov(X_std) . w
# =============================================================================
def haufe_pattern(fset, data, freqs, rng_name):
    rng = RANGES[rng_name]
    Xs, ys = _build_Xs(fset, data, freqs, rng)
    Xall = np.concatenate(Xs, axis=0)
    yall = np.concatenate(ys, axis=0)
    pipe = _clf().fit(Xall, yall)
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    Xs_std = scaler.transform(Xall)
    w = lr.coef_[0]
    cov = np.cov(Xs_std, rowvar=False)
    A = cov @ w  # forward activation per feature
    return A


def _make_info(ch_names):
    info = mne.create_info(list(ch_names), sfreq=RESAMPLE, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="ignore", verbose="ERROR")
    return info


def _topo(ax, vals, info, title):
    v = np.asarray(vals, float)
    lim = float(np.nanmax(np.abs(v))) or 1.0
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-lim, lim), contours=4)
    ax.set_title(title, fontsize=9)
    return im


def plot_haufe(data, freqs, ch, rng_name=PRIMARY_RANGE):
    info = _make_info(ch)
    nch = len(ch)
    bands = BANDS_40 if RANGES[rng_name][1] > 30 else BANDS_30

    # --- aperiodico: offset + exponent topos ---
    A = haufe_pattern("aperiódico", data, freqs, rng_name)
    off, exp = A[:nch], A[nch:2 * nch]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    _topo(axes[0], off, info, "Haufe: offset (aperiódico)")
    _topo(axes[1], exp, info, "Haufe: exponent (aperiódico)")
    fig.suptitle(f"1.5 Haufe forward — aperiódico ({rng_name} Hz)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "haufe_aperiodic.png", dpi=120); plt.close(fig)

    # --- periodico: una topo por banda ---
    A = haufe_pattern("periódico", data, freqs, rng_name)
    bnames = list(bands.keys())
    fig, axes = plt.subplots(1, len(bnames), figsize=(2.4 * len(bnames), 3.4))
    for j, bn in enumerate(bnames):
        _topo(axes[j], A[j * nch:(j + 1) * nch], info, f"{bn}")
    fig.suptitle(f"1.5 Haufe forward — periódico por banda ({rng_name} Hz): "
                 "rojo=el decoder se apoya en SCR>no-SCR", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "haufe_periodic.png", dpi=120); plt.close(fig)

    # --- espectro completo: heatmap freq-bin x canal + topos colapsadas por banda ---
    A = haufe_pattern("espectro completo", data, freqs, rng_name)
    lo, hi = RANGES[rng_name]
    edges = list(zip(np.arange(lo, hi, 1.0), np.arange(lo + 1, hi + 1, 1.0)))
    edges = [(a, b) for (a, b) in edges if ((freqs >= a) & (freqs < b)).sum() > 0]
    nb = len(edges)
    M = A.reshape(nb, nch)
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2.0])
    ax = fig.add_subplot(gs[0])
    lim = float(np.nanmax(np.abs(M))) or 1.0
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    ax.set_yticks(range(0, nb, 4)); ax.set_yticklabels([f"{int(edges[i][0])}" for i in range(0, nb, 4)])
    ax.set_ylabel("freq bin (Hz)"); ax.set_xlabel("canal")
    ax.set_xticks(range(nch)); ax.set_xticklabels(ch, rotation=90, fontsize=5)
    ax.set_title("Haufe A (freq-bin x canal)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    # band-collapsed topos in the right gridspec
    bcenters = [(a + b) / 2 for (a, b) in edges]
    gax = gs[1].subgridspec(1, len(bnames))
    for j, bn in enumerate(bnames):
        lo_b, hi_b = bands[bn]
        sel = [i for i, c in enumerate(bcenters) if lo_b <= c < hi_b]
        ax2 = fig.add_subplot(gax[j])
        vals = M[sel].mean(axis=0) if sel else np.zeros(nch)
        _topo(ax2, vals, info, bn)
    fig.suptitle(f"1.5 Haufe forward — espectro completo ({rng_name} Hz): "
                 "heatmap + colapso por banda", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "haufe_allbins.png", dpi=120); plt.close(fig)
    print("[1.5] Haufe figures saved", flush=True)


# =============================================================================
# Panel figure
# =============================================================================
def plot_panel(df, data_u):
    """2 paneles (intra / loso) al rango primario: barras AUC + CI + scatter por sujeto."""
    sub = df[df["range"] == PRIMARY_RANGE].set_index("feature_set")
    order = list(FEATURE_SETS.keys())
    subs = list(data_u.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, scheme in zip(axes, ["intra", "loso"]):
        auc = [sub.loc[f, f"{scheme}_auc"] for f in order]
        lo = [sub.loc[f, f"{scheme}_ci_lo"] for f in order]
        hi = [sub.loc[f, f"{scheme}_ci_hi"] for f in order]
        x = np.arange(len(order))
        ax.bar(x, auc, color="C0", alpha=0.55, width=0.6, zorder=1)
        ax.errorbar(x, auc, yerr=[np.array(auc) - np.array(lo), np.array(hi) - np.array(auc)],
                    fmt="none", ecolor="k", capsize=4, lw=1.4, zorder=3)
        for xi, f in enumerate(order):
            pts = sub.loc[f, f"{scheme}_subj"]
            for k, s in enumerate(subs):
                ax.scatter(xi + np.linspace(-0.22, 0.22, len(subs))[k], pts[k],
                           color=SUBJ_COLORS[s], s=42, zorder=4, label=s if xi == 0 else None)
            p = sub.loc[f, f"p_{scheme}"]
            ax.text(xi, hi[xi] + 0.01, f"p={p:.3f}", ha="center", fontsize=7)
        ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"{scheme.upper()} AUC"); ax.set_ylim(0.40, 0.82)
        ax.set_title(f"{scheme.upper()} ({PRIMARY_RANGE} Hz): barra=media, CI=bootstrap inter-sujeto, "
                     "puntos=6 sujetos", fontsize=9)
        if scheme == "intra":
            ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.suptitle("1.5 Panel decoding intra vs LOSO por feature-set (N=6, 1000 perms)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "panel_intra_loso.png", dpi=110); plt.close(fig)
    print("[1.5] panel figure saved", flush=True)


# =============================================================================
# Periodic: uniform vs matched (framing -> 1.6)
# =============================================================================
def periodic_uniform_vs_matched(data_u, data_m, freqs, n_perm):
    rows = []
    for scheme, data in [("uniforme", data_u), ("emparejado", data_m)]:
        r = eval_fset("periódico", data, freqs, PRIMARY_RANGE, n_perm)
        r["scheme"] = scheme
        rows.append(r)
        print(f"  periódico {scheme:11s}: intra={r['intra_auc']:.3f} (p={r['p_intra']})  "
              f"LOSO={r['loso_auc']:.3f} (p={r['p_loso']})", flush=True)
    df = pd.DataFrame(rows)
    df_save = df.drop(columns=["intra_subj", "loso_subj"])
    df_save.to_csv(TBL_DIR / "periodic_uniform_vs_matched.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2); w = 0.36
    for j, sch in enumerate(["intra", "loso"]):
        vals = [df[df.scheme == s].iloc[0][f"{sch}_auc"] for s in ("uniforme", "emparejado")]
        lo = [df[df.scheme == s].iloc[0][f"{sch}_ci_lo"] for s in ("uniforme", "emparejado")]
        hi = [df[df.scheme == s].iloc[0][f"{sch}_ci_hi"] for s in ("uniforme", "emparejado")]
        ax.bar(x + (j - 0.5) * w, vals, w, label=sch.upper(), color=["C0", "C1"][j], alpha=0.8)
        ax.errorbar(x + (j - 0.5) * w, vals,
                    yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                    fmt="none", ecolor="k", capsize=3, lw=1.2)
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(["no-SCR uniforme\n(sin control temporal)",
                                          "no-SCR emparejado\n(con control temporal)"])
    ax.set_ylabel("AUC"); ax.set_ylim(0.40, 0.75)
    ax.set_title("Periódico (desync, 1/f-removido): ¿colapsa al controlar el tiempo?\n"
                 "Puente a 1.6 (¿qué banda maneja la caída?)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "periodic_uniform_vs_matched.png", dpi=120); plt.close(fig)
    return df


# =============================================================================
# Main
# =============================================================================
def _probe(data_u, freqs):
    """Time one permutation of the heaviest feature-set to size N_PERM."""
    rng = RANGES[PRIMARY_RANGE]
    Xs, ys = _build_Xs("espectro completo", data_u, freqs, rng)
    Xall = np.concatenate(Xs, axis=0); yall = np.concatenate(ys, axis=0)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    print(f"[probe] espectro completo: {Xall.shape[1]} feats, {Xall.shape[0]} epochs", flush=True)
    pr = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(3):
        yp = _perm_within(yall, groups, pr)
        ys_p = [yp[groups == i] for i in range(len(ys))]
        intra_auc(Xs, ys_p); loso_auc(Xall, yp, groups)
    dt = (time.time() - t0) / 3
    print(f"[probe] ~{dt:.2f}s / perm (heaviest set). 1000 perms ~= {dt*1000/60:.1f} min for this set.",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78)
    print("decoding_panel :: 1.5 intra vs LOSO + perms + Haufe")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(); return
    if not CACHE.exists():
        print("[1.5] no cache -- run with --build first", flush=True); return
    data_u, freqs, ch = load_cache("uniform")
    data_m, _, _ = load_cache("matched")
    print(f"[1.5] cache: uniform={len(data_u)} subj, matched={len(data_m)} subj, "
          f"{len(ch)} ch, {len(freqs)} freqs", flush=True)
    if args.probe:
        _probe(data_u, freqs); return

    n_perm = args.nperm
    print(f"\n[1.5] PANEL (uniforme) — {n_perm} perms ----------------------------", flush=True)
    rows = []
    for fset in FEATURE_SETS:
        for rng_name in PANEL_RANGES:
            # perms solo en el rango primario (1-40); el otro rango va sin p (robustez descriptiva)
            if rng_name != PRIMARY_RANGE:
                np_use = 0
            elif fset in HEAVY_SETS:
                np_use = min(n_perm, HEAVY_NPERM)
            else:
                np_use = n_perm
            r = eval_fset(fset, data_u, freqs, rng_name, np_use)
            rows.append(r)
            print(f"  {fset:24s} {rng_name}: intra={r['intra_auc']:.3f} "
                  f"[{r['intra_ci_lo']:.2f},{r['intra_ci_hi']:.2f}] (p={r['p_intra']})  "
                  f"LOSO={r['loso_auc']:.3f} [{r['loso_ci_lo']:.2f},{r['loso_ci_hi']:.2f}] "
                  f"(p={r['p_loso']})  nfeat={r['n_features']}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["intra_subj", "loso_subj"]).to_csv(TBL_DIR / "decoding_intra_vs_loso.csv", index=False)
    plot_panel(df, data_u)

    print(f"\n[1.5] HAUFE patterns ({PRIMARY_RANGE}) -------------------------------", flush=True)
    plot_haufe(data_u, freqs, ch)

    print(f"\n[1.5] PERIODICO uniforme vs emparejado — {n_perm} perms --------------", flush=True)
    periodic_uniform_vs_matched(data_u, data_m, freqs, n_perm)

    print(f"\n[1.5] done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
