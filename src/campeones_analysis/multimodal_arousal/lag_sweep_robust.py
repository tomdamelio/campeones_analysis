"""R3.3 (1) robustez — blindar el brain->body: (A) control ocular/movimiento τ-MATCHEADO + (B) LOSO.

El barrido de lag (lag_sweep_alpha_smna) dio que la desync alfa ANTICIPA el SMNA (pico τ≈−1s, asim 6/6).
Dos preguntas obvias de la reunión que el resultado original NO blinda del todo:

(A) "¿No es MOVIMIENTO OCULAR anticipatorio en vez de cortical?" — el control parcial original usa
    covariados de la ventana FIJA [−5,+3] mientras la alfa se desliza. Si el pico pre-onset viene de un
    blink/sacada/movimiento que también ocurre pre-onset, un covariado de ventana fija NO lo descuenta en
    la ventana corrida. Acá se re-extrae el VEOG-lento (L_EYE 0.5-8) y el jerk (acelerómetro) en la MISMA
    ventana corrida [τ−1,τ+1] y se re-hace la parcial τ-matcheada. Si el pico pre-onset SOBREVIVE → no es
    ocular/movimiento anticipatorio, el brain->body queda mucho más sólido.
(B) "¿No lo maneja un sujeto?" — sub-23 es el disidente (τ*=+2). LOSO de la consistencia de grupo
    (asimetría pre<post 6/6, τ*<0) dejando fuera cada sujeto. Dropear sub-23 debería REFORZAR.

Tres curvas por sujeto: raw / parcial ventana-FIJA (= R3.3(1) original) / parcial τ-MATCHEADA.
Reusa la extracción de alfa de lag_sweep_alpha_smna; 1:1 al cache (verificado por tnorm). Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.lag_sweep_robust
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy import stats
from scipy.signal import hilbert

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.confound_model_scr import _load_covariates
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED, load_cache
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
from src.campeones_analysis.multimodal_arousal.lag_sweep_alpha_smna import (
    EEG_FS,
    POSTERIOR,
    TAU_GRID,
    WIN_S,
    _band_windows,
    _smna_auc,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_3_directionality"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
HALF = WIN_S / 2.0


def _env(x, sf, lo, hi):
    hi = min(hi, sf / 2.0 - 1.0)
    return np.abs(hilbert(filter_data(x[None, :].astype(float), sf, lo, hi, verbose="ERROR")[0]))


def _jerk(a, sf):
    return np.gradient(a) * sf


def _wmean(env, sf, t0, tau):
    a, b = int((t0 + tau - HALF) * sf), int((t0 + tau + HALF) * sf)
    a, b = max(0, a), min(len(env), b)
    return float(np.mean(env[a:b])) if b - a > 2 else np.nan


def _wvar(sig, sf, t0, tau):
    a, b = int((t0 + tau - HALF) * sf), int((t0 + tau + HALF) * sf)
    a, b = max(0, a), min(len(sig), b)
    return float(np.var(sig[a:b])) if b - a > 2 else np.nan


def _tau_cov(sig, sf, onsets, fn):
    """Covariado τ-matcheado: (n_onsets, n_tau) usando fn (mean env / var) en [τ−1,τ+1]."""
    out = np.full((len(onsets), len(TAU_GRID)), np.nan)
    for i, t0 in enumerate(onsets):
        for j, tau in enumerate(TAU_GRID):
            out[i, j] = fn(sig, sf, t0, tau)
    return out


def _resid(v, C):
    A = np.column_stack([C, np.ones(len(C))])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _partial(x, y, C, mask):
    m = mask & np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    if m.sum() < 8:
        return np.nan
    rx, ry = stats.rankdata(x[m]), stats.rankdata(y[m])
    return float(stats.spearmanr(_resid(rx, C[m]), _resid(ry, C[m])).correlation)


def _curve_fixed(smna, alpha_tau, Cfix, mask):
    return np.array([_partial(smna, alpha_tau[:, j], Cfix, mask) for j in range(len(TAU_GRID))])


def _curve_taumatched(smna, alpha_tau, veog_tau, jerk_tau, tnorm, mask):
    rho = np.full(len(TAU_GRID), np.nan)
    for j in range(len(TAU_GRID)):
        C = np.column_stack([veog_tau[:, j], jerk_tau[:, j], tnorm])
        rho[j] = _partial(smna, alpha_tau[:, j], C, mask)
    return rho


def main():
    print("=" * 78)
    print("lag_sweep_robust :: (A) control ocular/mov τ-MATCHEADO + (B) LOSO del brain->body")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)

    data = {}
    for sub in COHORT:
        psd, y, tn = cache[sub]
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        r_a = {b: [] for b in ("alpha",)}
        real_alpha, sil_alpha = [], []
        real_veog, sil_veog, real_jerk, sil_jerk = [], [], [], []
        real_auc, sil_auc, real_tn, sil_tn = [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw_full = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            sf = float(raw_full.info["sfreq"])
            veog_env = (_env(raw_full.get_data(picks=["L_EYE"])[0], sf, 0.5, 8.0)
                        if "L_EYE" in raw_full.ch_names else None)
            if {"X", "Y", "Z"}.issubset(raw_full.ch_names):
                xx, yy, zz = (raw_full.get_data(picks=[c])[0] for c in ("X", "Y", "Z"))
                jerk = _jerk(np.sqrt(xx ** 2 + yy ** 2 + zz ** 2), sf)
            else:
                jerk = None
            raw = raw_full.copy().pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(EEG_FS, verbose="ERROR")
            dur = float(raw.times[-1]); D = raw.get_data()
            post_idx = [raw.ch_names.index(c) for c in POSTERIOR if c in raw.ch_names]
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            rk = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sk = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            real_alpha.append(_band_windows(D, rk, post_idx)["alpha"])
            sil_alpha.append(_band_windows(D, sk, post_idx)["alpha"])
            nan_v = lambda n: np.full((n, len(TAU_GRID)), np.nan)
            real_veog.append(_tau_cov(veog_env, sf, rk, _wmean) if veog_env is not None else nan_v(len(rk)))
            sil_veog.append(_tau_cov(veog_env, sf, sk, _wmean) if veog_env is not None else nan_v(len(sk)))
            real_jerk.append(_tau_cov(jerk, sf, rk, _wvar) if jerk is not None else nan_v(len(rk)))
            sil_jerk.append(_tau_cov(jerk, sf, sk, _wvar) if jerk is not None else nan_v(len(sk)))
            real_auc += [_smna_auc(smna, t) for t in rk]; real_tn += list(rk / dur)
            sil_auc += [_smna_auc(smna, t) for t in sk]; sil_tn += list(sk / dur)

        auc = np.array(real_auc + sil_auc); my_tn = np.array(real_tn + sil_tn)
        if len(auc) != len(tn) or np.max(np.abs(my_tn - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN -> skip", flush=True); continue
        alpha_tau = np.vstack(real_alpha + sil_alpha)
        veog_tau = np.vstack(real_veog + sil_veog)
        jerk_tau = np.vstack(real_jerk + sil_jerk)
        Cfix = np.column_stack([covs[sub]["veog_slow_0p5_8"], covs[sub]["var_jerk"], covs[sub]["tnorm"]])
        veog_ok = np.isfinite(veog_tau).all()
        data[sub] = dict(auc=auc, y=y, tn=covs[sub]["tnorm"], alpha=alpha_tau,
                         veog=veog_tau, jerk=jerk_tau, Cfix=Cfix, veog_ok=veog_ok)
        print(f"  {sub}: n={len(auc)}  align OK  L_EYE={'sí' if veog_ok else 'NO(fallback jerk+tiempo)'}",
              flush=True)

    # ---- curvas + τ* + asimetría (raw / fija / τ-matcheada) ----
    rng = np.random.default_rng(20260610)
    rows, curves = [], []
    for sub, d in data.items():
        mask = np.ones(len(d["y"]), bool)
        raw_c = np.array([float(stats.spearmanr(d["auc"][np.isfinite(d["alpha"][:, j])],
                          d["alpha"][np.isfinite(d["alpha"][:, j]), j]).correlation)
                          for j in range(len(TAU_GRID))])
        fix_c = _curve_fixed(d["auc"], d["alpha"], d["Cfix"], mask)
        tm_c = _curve_taumatched(d["auc"], d["alpha"], d["veog"], d["jerk"], d["tn"], mask)
        for name, c in [("raw", raw_c), ("partial_fixed", fix_c), ("partial_taumatched", tm_c)]:
            jstar = int(np.nanargmin(c))
            pre = np.nanmean(c[TAU_GRID < 0]); post = np.nanmean(c[TAU_GRID > 0])
            rows.append(dict(subject=sub, curve=name, tau_star=float(TAU_GRID[jstar]),
                             rho_star=round(float(c[jstar]), 3), asym=round(float(pre - post), 3)))
            for j, tau in enumerate(TAU_GRID):
                curves.append(dict(subject=sub, curve=name, tau=float(tau), rho=round(float(c[j]), 3)))
        # null max-statistic sobre la curva τ-matcheada
        idx = np.where(mask)[0]; obs = float(np.nanmin(tm_c))
        nmin = np.empty(N_PERM)
        for p in range(N_PERM):
            sp = d["auc"].copy(); sp[idx] = d["auc"][rng.permutation(idx)]
            nmin[p] = np.nanmin(_curve_taumatched(sp, d["alpha"], d["veog"], d["jerk"], d["tn"], mask))
        pperm = (1 + int((nmin <= obs).sum())) / (1 + N_PERM)
        r = [x for x in rows if x["subject"] == sub and x["curve"] == "partial_taumatched"][0]
        r["p_perm"] = round(pperm, 4)
        print(f"  {sub}: τ* raw={[x['tau_star'] for x in rows if x['subject']==sub and x['curve']=='raw'][0]:+.0f} "
              f"fija={[x['tau_star'] for x in rows if x['subject']==sub and x['curve']=='partial_fixed'][0]:+.0f} "
              f"τmatch={r['tau_star']:+.0f}  asym(τmatch)={r['asym']:+.3f}  p_perm={pperm:.3f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(TBL_DIR / "lag_sweep_robust_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(TBL_DIR / "lag_sweep_robust_curves.csv", index=False)

    # ---- (A) ¿sobrevive el brain->body al control τ-matcheado? ----
    print("\n=== (A) Control ocular/movimiento τ-MATCHEADO (curva parcial τ-matcheada) ===")
    tm = df[df.curve == "partial_taumatched"]
    asym = tm["asym"].to_numpy(float); tstar = tm["tau_star"].to_numpy(float)
    print(f"  asimetría pre<post (brain->body): {int((asym < 0).sum())}/6  (media {asym.mean():+.3f})")
    print(f"  τ* < 0 (alfa precede):            {int((tstar < 0).sum())}/6  (mediana {np.median(tstar):+.1f}s)")
    print(f"  acoplamiento sig. (p_perm<0.05):  {int((tm['p_perm'] < 0.05).sum())}/6")
    for c in ("raw", "partial_fixed", "partial_taumatched"):
        a = df[df.curve == c]["asym"].to_numpy(float)
        print(f"    [{c:20s}] asim<0: {int((a<0).sum())}/6  media={a.mean():+.3f}")

    # ---- (B) LOSO de la consistencia de grupo (sobre la curva τ-matcheada) ----
    print("\n=== (B) LOSO de la direccionalidad (curva τ-matcheada) ===")
    subs = list(tm["subject"])
    for left in subs:
        sub_asym = tm[tm.subject != left]["asym"].to_numpy(float)
        sub_tstar = tm[tm.subject != left]["tau_star"].to_numpy(float)
        flag = "  <-- el disidente" if left == "sub-23" else ""
        print(f"  sin {left}: asim<0 {int((sub_asym<0).sum())}/5  τ*<0 {int((sub_tstar<0).sum())}/5  "
              f"media_asim={sub_asym.mean():+.3f}{flag}", flush=True)

    _plot(data, pd.DataFrame(curves), df)
    print(f"\nOutputs -> {OUT_DIR}\n[lag-sweep robustez] done", flush=True)


def _plot(data, dc, df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, sub in zip(axes.ravel(), data):
        for name, col, lw in [("raw", "0.6", 1.0), ("partial_fixed", "C0", 1.4),
                              ("partial_taumatched", "C3", 2.0)]:
            cs = dc[(dc.subject == sub) & (dc.curve == name)].sort_values("tau")
            ax.plot(cs.tau, cs.rho, "o-", color=col, lw=lw, ms=3, label=name)
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6)
        ts = df[(df.subject == sub) & (df.curve == "partial_taumatched")]["tau_star"].values[0]
        ax.set_title(f"{sub}  τ*(τmatch)={ts:+.0f}s", fontsize=9)
        ax.set_xlabel("τ ventana alfa (s) [<0 PRE]"); ax.set_ylabel("Spearman(alfa,SMNA)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("R3.3(1) robustez: control ocular/movimiento τ-MATCHEADO (rojo) vs ventana fija (azul)\n"
                 "si el mínimo en τ<0 sobrevive el rojo → el brain->body no es ocular/movimiento anticipatorio",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "lag_sweep_robust.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
