"""R3.3 (1) — Direccionalidad: ¿la desync alfa PRECEDE o SIGUE al SMNA? (barrido de lag per-época)

R3.1 mostró que la desync alfa posterior escala con el SMNA-AUC del evento (6/6 neg), pero con la
alfa de la ventana FIJA [-5,+3] -> dice *que* hay acoplamiento, no *cuándo*. Acá deslizamos una
ventana corta de alfa (w=2 s) a lo largo de la época y, para cada centro tau, correlacionamos la
alfa periódica posterior de esa ventana contra el SMNA-AUC del evento (FIJO en [0,+3], el de R3.1).
El tau que MAXIMIZA la correlación NEGATIVA dice el lead/lag:

  tau < 0 (ventana alfa PRE-onset) gana  -> alfa lidera -> brain->body (anticipatorio)
  tau > 0 (ventana alfa POST-onset) gana -> alfa sigue  -> body->brain (aferente)
  pico en tau≈0 / curva simétrica        -> co-ocurrencia / driver común (no resuelve dirección)

Diseño (validado con el usuario):
  - SELECCIÓN de épocas IDÉNTICA al cache uniform (bounds [-5,+3], mismo UNIFORM_SEED) -> 1:1 con
    R3.1 y con los covariados de 2.1-2.4. Solo la VENTANA DE EXTRACCIÓN de alfa se desliza dentro de
    un raw más ancho; NaN donde la ventana corrida se sale de la grabación (épocas de borde).
  - tau ∈ {-4..+4} s (paso 1 s), ventana w=2 s -> alfa sobre [tau-1, tau+1].
  - Métrica = alfa periódica posterior (1/f removido en 1-30, banda 8-13), idéntica a R3.1.
  - Veredicto sobre la curva PARCIAL (controla tnorm [drift compartido] + ojo + movimiento).
  - Null subject-respecting + max-statistic: permutar SMNA dentro del sujeto, recomputar la curva
    entera, tomar su MIN (corrige la selección de tau* = argmin sobre la grilla). 1000 perms.

Caveats (al diario): el video maneja alfa y SMNA a la vez -> el lag dice CUÁNDO, no causalidad pura
(eso es el VARX de R3.3 con exógeno). Per-época = primario; la cross-correlación continua es el
cross-check (envelope Hilbert vs SMNA continuo, null circular-shift + baseline AR). Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.lag_sweep_alpha_smna
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.confound_model_scr import _load_covariates
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

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_3_directionality"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

EEG_FS = 250.0                                  # raw.resample(250) como en R3.1
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]
CONTROL_COLS = ["veog_slow_0p5_8", "blink_slow", "var_jerk", "tnorm"]   # ojo + blink + mov + tiempo
TAU_GRID = np.arange(-4.0, 4.0 + 0.5, 1.0)      # {-4,-3,...,+4} s
WIN_S = 2.0                                      # ancho de la ventana de alfa
NPERSEG = 250                                    # Welch: 1 s @250Hz -> 1 Hz, 3 segs sobre 2 s
N_PERM = 1000


def _smna_auc(smna, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(smna), i1)
    return float(np.trapz(np.clip(smna[i0:i1], 0, None), dx=1.0 / EDA_FS)) if i1 - i0 >= 2 else np.nan


def _band_windows(D, onsets_s, post_idx):
    """Para cada onset (s) y cada tau, alfa/theta/beta periódica posterior sobre [tau-1,tau+1].

    D: (n_ch, n_times) @EEG_FS. Devuelve dict band -> (n_onsets, n_tau), NaN si la ventana se sale.
    """
    n, nt = len(onsets_s), len(TAU_GRID)
    out = {b: np.full((n, nt), np.nan) for b in BANDS}
    half = int(round(WIN_S / 2 * EEG_FS))
    c0 = np.round(np.asarray(onsets_s) * EEG_FS).astype(int)
    for j, tau in enumerate(TAU_GRID):
        c = c0 + int(round(tau * EEG_FS))
        s, e = c - half, c + half
        valid = (s >= 0) & (e <= D.shape[1])
        if not valid.any():
            continue
        segs = np.stack([D[post_idx, a:b] for a, b in zip(s[valid], e[valid])])  # (nv,nch,2*half)
        f, pxx = welch(segs, fs=EEG_FS, nperseg=NPERSEG, noverlap=NPERSEG // 2, axis=-1)
        _, _, resid, fr = _linear_aperiodic(pxx, f, RANGES["1-30"])               # (nv,nch,nf)
        for b, (lo, hi) in BANDS.items():
            m = (fr >= lo) & (fr < hi)
            out[b][valid, j] = np.clip(resid[:, :, m], 0, None).mean(axis=2).mean(axis=1)
    return out


def _resid(v, C):
    A = np.column_stack([C, np.ones(len(C))])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _partial_curve(smna, alpha_tau, C, mask):
    """Spearman parcial(smna, alfa(tau) | C) por tau, sobre épocas en mask & finitas en cada tau.

    Devuelve (rho_raw[tau], rho_par[tau]). Pre-residualiza la rama alfa por tau (fija bajo perm).
    """
    nt = alpha_tau.shape[1]
    rho_raw, rho_par = np.full(nt, np.nan), np.full(nt, np.nan)
    rs_full = stats.rankdata(smna)
    for j in range(nt):
        m = mask & np.isfinite(alpha_tau[:, j]) & np.isfinite(smna)
        if m.sum() < 8:
            continue
        x, a = smna[m], alpha_tau[m, j]
        rho_raw[j] = stats.spearmanr(x, a).correlation
        Cm = C[m]
        ra = _resid(stats.rankdata(a), Cm)
        rx = _resid(stats.rankdata(x), Cm)
        rho_par[j] = stats.spearmanr(rx, ra).correlation
    return rho_raw, rho_par


def _perm_pvalue(smna, alpha_tau, C, mask, obs_min, rng):
    """Null max-statistic: permuta SMNA dentro del sujeto, recomputa curva parcial, toma su min."""
    idx = np.where(mask)[0]
    null_min = np.empty(N_PERM)
    for p in range(N_PERM):
        sp = smna.copy()
        sp[idx] = smna[rng.permutation(idx)]
        _, par = _partial_curve(sp, alpha_tau, C, mask)
        null_min[p] = np.nanmin(par)
    return (1 + int((null_min <= obs_min).sum())) / (1 + N_PERM)


def main():
    print("=" * 78)
    print("lag_sweep_alpha_smna :: R3.3(1) — direccionalidad desync alfa vs SMNA (barrido de lag)")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)   # reproduce el set/orden del cache (real luego silent)

    data = {}
    for sub in COHORT:
        psd, y, tn = cache[sub]
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_rows, sil_rows = {b: [] for b in BANDS}, {b: [] for b in BANDS}
        real_auc, sil_auc, real_tn, sil_tn = [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(EEG_FS, verbose="ERROR")
            dur = float(raw.times[-1])
            D = raw.get_data()
            post_idx = [raw.ch_names.index(c) for c in POSTERIOR if c in raw.ch_names]
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            rk = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]   # bounds IDÉNTICOS al cache
            sk = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            wr = _band_windows(D, rk, post_idx)
            ws = _band_windows(D, sk, post_idx)
            for b in BANDS:
                real_rows[b].append(wr[b]); sil_rows[b].append(ws[b])
            real_auc += [_smna_auc(smna, t) for t in rk]; real_tn += list(rk / dur)
            sil_auc += [_smna_auc(smna, t) for t in sk]; sil_tn += list(sk / dur)

        auc = np.array(real_auc + sil_auc)
        my_tn = np.array(real_tn + sil_tn)
        if len(auc) != len(tn) or np.max(np.abs(my_tn - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN n={len(auc)} vs {len(tn)} -> skip", flush=True); continue
        bands = {b: np.vstack(real_rows[b] + sil_rows[b]) for b in BANDS}   # (n_ep, n_tau), orden cache
        C = np.column_stack([covs[sub][c] for c in CONTROL_COLS])
        data[sub] = dict(auc=auc, y=y, C=C, **bands)
        cov = float(np.mean(np.isfinite(bands["alpha"])))
        print(f"  {sub}: n={len(auc)} (SCR={int((y==1).sum())})  align OK  cobertura_tau={cov:.2f}", flush=True)

    # -------------------- curvas + tau* + null (alfa, primario) --------------------
    rows, curves = [], []
    rng = np.random.default_rng(20260609)
    for sub, d in data.items():
        for scope, mk in [("all", np.ones(len(d["y"]), bool)), ("scr_only", d["y"] == 1)]:
            raw_a, par_a = _partial_curve(d["auc"], d["alpha"], d["C"], mk)
            jstar = int(np.nanargmin(par_a))           # tau* = max corr negativa (curva parcial)
            tau_star, rho_star = float(TAU_GRID[jstar]), float(par_a[jstar])
            pre = np.nanmean(par_a[TAU_GRID < 0]); post = np.nanmean(par_a[TAU_GRID > 0])
            asym = float(pre - post)                   # <0 => más desync PRE => brain->body
            pperm = _perm_pvalue(d["auc"], d["alpha"], d["C"], mk, rho_star, rng) if scope == "all" else np.nan
            rows.append(dict(subject=sub, scope=scope, n=int(mk.sum()), tau_star=tau_star,
                             rho_star=round(rho_star, 3), asym_pre_minus_post=round(asym, 3),
                             p_perm=round(pperm, 4) if np.isfinite(pperm) else np.nan))
            for j, tau in enumerate(TAU_GRID):
                curves.append(dict(subject=sub, scope=scope, tau=float(tau),
                                   rho_raw=round(float(raw_a[j]), 3), rho_partial=round(float(par_a[j]), 3)))
            if scope == "all":
                print(f"  {sub} [all]: tau*={tau_star:+.0f}s  rho*={rho_star:+.3f}  "
                      f"asym(pre-post)={asym:+.3f}  p_perm={pperm:.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(TBL_DIR / "lag_sweep_summary.csv", index=False)
    dc = pd.DataFrame(curves); dc.to_csv(TBL_DIR / "lag_sweep_curves.csv", index=False)

    # secundario: theta/beta tau* (raw, contexto)
    for b in ("theta", "beta"):
        srows = []
        for sub, d in data.items():
            raw_b, _ = _partial_curve(d["auc"], d[b], d["C"], np.ones(len(d["y"]), bool))
            j = int(np.nanargmin(raw_b))
            srows.append(dict(subject=sub, band=b, tau_star=float(TAU_GRID[j]), rho_star=round(float(raw_b[j]), 3)))
        pd.DataFrame(srows).to_csv(TBL_DIR / f"lag_sweep_{b}_raw.csv", index=False)

    # -------------------- veredicto de grupo (alfa, scope=all) --------------------
    sa = df[df.scope == "all"]
    tstars = sa["tau_star"].to_numpy(float)
    asyms = sa["asym_pre_minus_post"].to_numpy(float)
    nsig = int((sa["p_perm"] < 0.05).sum())
    print("\n=== VEREDICTO DIRECCIONALIDAD (alfa, scope=all) ===")
    print(f"  tau* por sujeto: {dict(zip(sa.subject, tstars))}")
    print(f"  tau* < 0 (alfa PRECEDE = brain->body): {int((tstars < 0).sum())}/6  | "
          f"mediana tau*={np.median(tstars):+.1f}s")
    print(f"  asimetría pre-post < 0 (más desync PRE = brain->body): {int((asyms < 0).sum())}/6  | "
          f"media={asyms.mean():+.3f}")
    print(f"  acoplamiento sig. (p_perm<0.05): {nsig}/6", flush=True)

    _plot(data, dc, df)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.3(1) lag-sweep] done", flush=True)


def _plot(data, dc, df):
    # (A) curvas por sujeto (raw + parcial), scope=all
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (sub, d) in zip(axes.ravel(), data.items()):
        cs = dc[(dc.subject == sub) & (dc.scope == "all")].sort_values("tau")
        ax.plot(cs.tau, cs.rho_raw, "o-", color="0.6", lw=1.2, ms=4, label="raw")
        ax.plot(cs.tau, cs.rho_partial, "o-", color=SUBJ_COLORS.get(sub, "C0"), lw=1.8, ms=4,
                label="parcial (| ojo+mov+tiempo)")
        ts = df[(df.subject == sub) & (df.scope == "all")]["tau_star"].values[0]
        ax.axvline(ts, color="r", ls="--", lw=1)
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{sub}  tau*={ts:+.0f}s", fontsize=9)
        ax.set_xlabel("tau ventana alfa (s)  [<0 PRE-onset]"); ax.set_ylabel("Spearman(alfa, SMNA-AUC)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("R3.3(1): barrido de lag alfa posterior vs SMNA-AUC (per-época, scope=all)\n"
                 "rho más NEGATIVO = más desync acoplada · tau*<0 (PRE) = brain->body · tau*>0 (POST) = body->brain",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "lag_sweep_per_subject.png", dpi=120); plt.close(fig)

    # (B) curva media de grupo (parcial)
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = dc[dc.scope == "all"].pivot_table(index="tau", columns="subject", values="rho_partial")
    for sub in piv.columns:
        ax.plot(piv.index, piv[sub], color=SUBJ_COLORS.get(sub, "0.7"), lw=1, alpha=0.5)
    ax.plot(piv.index, piv.mean(axis=1), "ko-", lw=2.2, ms=5, label="media 6 sujetos")
    ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("tau ventana alfa (s)  [<0 PRE-onset]")
    ax.set_ylabel("parcial Spearman(alfa, SMNA-AUC | ojo+mov+tiempo)")
    ax.set_title("R3.3(1): curva de lag media (parcial)\n"
                 "mínimo en tau<0 => alfa anticipa el SMNA (brain->body); en tau>0 => sigue (body->brain)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lag_sweep_group_mean.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
