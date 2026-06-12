"""R3.1 / 3.1 — GLM TWEEDIE FORMAL (lo que pidió Diego): modelar la Y del arousal con un GLM
no-gaussiano prediciéndola desde la desync alfa + confounds + tiempo.

Por qué no OLS: la Y del arousal es censurada-inflada (masa en cero por no-respondedores + cola
continua positiva, sesgo +1.5..+2.0 ya visto en R3.1). OLS/Pearson sobre eso es inferencia inválida.
  - **Tweedie** (compound Poisson-Gamma, link log, var_power p∈[1.3,1.7]): genera ceros exactos + cola
    positiva en UN modelo, biológicamente plausible (SCR = suma de descargas sudomotoras Poisson×Gamma).
  - **Hurdle/two-part**: logística P(SCR>0) [= el binario del 05_04] + Gamma sobre la magnitud condicional.

Predictor de interés = **alfa periódica posterior** (la señal de desync validada). Predicción: β_alfa
NEGATIVO (más desync -> más arousal). Los confounds van como COLUMNAS de X (NO se residualiza la Y en
GLM no-gaussiano): β_alfa parcial. El número titular: ¿β_alfa sobrevive con los confounds adentro?
(debería, dado R3.1 paso 2). Carta principal a N=6: signo de β_alfa 6/6 within-subject.

Ys: (1) SMNA-AUC (drive sudomotor integrado, all épocas) = Tweedie primario; (2) amplitud SCR
(0 en no-SCR -> zero-inflada) = comparación Tweedie vs hurdle por AIC/BIC.

Features 1:1 al cache (reusa load_cache + _load_covariates + alfa posterior periódica, como R3.1).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_glm_tweedie
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_1_alpha_amplitude"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]
CONTROL_COLS = ["veog_slow_0p5_8", "blink_slow", "var_jerk", "tnorm"]   # ojo + blink + mov + tiempo
P_GRID = [1.3, 1.4, 1.5, 1.6, 1.7]


def _alpha_post(psd, freqs, post_idx):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-30"])
    m = (f >= 8.0) & (f < 13.0)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)[:, post_idx].mean(axis=1)


def _smna_auc(smna, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(smna), i1)
    return float(np.trapz(np.clip(smna[i0:i1], 0, None), dx=1.0 / EDA_FS)) if i1 - i0 >= 2 else np.nan


def _scr_amp(eda, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(eda), i1)
    return float(np.max(eda[i0:i1])) if i1 - i0 >= 2 else np.nan


def _z(v):
    s = np.std(v)
    return (v - np.mean(v)) / s if s > 0 else v - np.mean(v)


def assemble():
    """Por sujeto, 1:1 al cache: alfa-PO periódica, SMNA-AUC, amplitud SCR (0 en no-SCR), covariados, y."""
    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    post_idx = [ch.index(c) for c in POSTERIOR if c in ch]
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)
    data = {}
    for sub in COHORT:
        psd, y, tn = cache[sub]
        alpha = _alpha_post(psd, freqs, post_idx)
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        r_auc, s_auc, r_amp, s_amp, r_tn, s_tn = [], [], [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(250.0, verbose="ERROR")
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            rk = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sk = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            r_auc += [_smna_auc(smna, t) for t in rk]; r_amp += [_scr_amp(eda, t) for t in rk]; r_tn += list(rk / dur)
            s_auc += [_smna_auc(smna, t) for t in sk]; s_amp += [0.0 for _ in sk]; s_tn += list(sk / dur)
        auc = np.array(r_auc + s_auc); amp = np.array(r_amp + s_amp); my_tn = np.array(r_tn + s_tn)
        if len(auc) != len(tn) or np.max(np.abs(my_tn - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN -> skip", flush=True); continue
        C = {c: covs[sub][c] for c in CONTROL_COLS}
        C["tnorm2"] = covs[sub]["tnorm"] ** 2
        data[sub] = dict(alpha=alpha, auc=auc, amp=amp, y=y, C=C)
        print(f"  {sub}: n={len(auc)} (SCR={int((y==1).sum())})  align OK", flush=True)
    return data


def _design(d, mask, with_conf):
    """X estandarizado: [alfa, (confounds)], + const. Y NO se toca (va a la familia del GLM)."""
    cols = {"alpha": _z(d["alpha"][mask])}
    if with_conf:
        for c in CONTROL_COLS:
            cols[c] = _z(d["C"][c][mask]) if c != "tnorm" else d["C"][c][mask]
        cols["tnorm2"] = d["C"]["tnorm2"][mask]
    X = pd.DataFrame(cols)
    return sm.add_constant(X, has_constant="add")


def _fit_tweedie(y, X, p):
    fam = sm.families.Tweedie(var_power=p, link=sm.families.links.Log())
    return sm.GLM(y, X, family=fam).fit()


def main():
    print("=" * 78)
    print("scr_glm_tweedie :: GLM Tweedie/hurdle — Y del arousal ~ desync alfa + confounds + tiempo")
    print("=" * 78, flush=True)

    data = assemble()
    if not data:
        print("sin datos"); return

    # ---------- (1) TWEEDIE sobre SMNA-AUC (primario, titular) ----------
    rows_tw, prows = [], []
    for sub, d in data.items():
        m = np.isfinite(d["auc"]) & np.isfinite(d["alpha"])
        y = np.clip(d["auc"][m], 0, None)
        # perfil de var_power p por deviance/AIC (full model)
        Xf = _design(d, m, with_conf=True)
        best = None
        for p in P_GRID:
            try:
                r = _fit_tweedie(y, Xf, p)
                prows.append(dict(subject=sub, p=p, deviance=round(float(r.deviance), 2),
                                  aic=round(float(r.aic), 1) if np.isfinite(r.aic) else np.nan,
                                  beta_alpha=round(float(r.params["alpha"]), 4),
                                  p_alpha=round(float(r.pvalues["alpha"]), 4)))
                if best is None or r.deviance < best[1]:
                    best = (p, r.deviance)
            except Exception as e:
                prows.append(dict(subject=sub, p=p, deviance=np.nan, aic=np.nan,
                                  beta_alpha=np.nan, p_alpha=np.nan))
        p_star = best[0] if best else 1.5
        # full (confounds dentro) vs minimal (solo alfa) al p*
        for tag, wc in [("full", True), ("alpha_only", False)]:
            X = _design(d, m, with_conf=wc)
            r = _fit_tweedie(y, X, p_star)
            rows_tw.append(dict(subject=sub, model=tag, p_star=p_star, n=int(m.sum()),
                                beta_alpha=round(float(r.params["alpha"]), 4),
                                p_alpha=round(float(r.pvalues["alpha"]), 4),
                                aic=round(float(r.aic), 1) if np.isfinite(r.aic) else np.nan))
        rf = [x for x in rows_tw if x["subject"] == sub and x["model"] == "full"][0]
        print(f"  [Tweedie SMNA] {sub}: p*={p_star}  beta_alpha(full)={rf['beta_alpha']:+.4f} "
              f"(p={rf['p_alpha']:.3f})", flush=True)
    df_tw = pd.DataFrame(rows_tw); df_tw.to_csv(TBL_DIR / "glm_tweedie_smna.csv", index=False)
    dfp = pd.DataFrame(prows); dfp.to_csv(TBL_DIR / "glm_tweedie_pprofile.csv", index=False)

    # ---------- sensibilidad de β_alfa a var_power (full model, toda la grilla) ----------
    # p* se elige por min-deviance y cae pegado al borde 1.3 (la deviance NO es comparable entre p);
    # acá mostramos que el TITULAR (signo/magnitud de β_alfa) es estable en toda la grilla 1.3-1.7.
    sens_rows = []
    for p in P_GRID:
        bp = dfp[dfp.p == p]["beta_alpha"].to_numpy(float)
        bp = bp[np.isfinite(bp)]
        sens_rows.append(dict(p=p, n=int(len(bp)), n_neg=int((bp < 0).sum()),
                              mean_beta_alpha=round(float(np.mean(bp)), 4) if len(bp) else np.nan,
                              min_beta=round(float(np.min(bp)), 4) if len(bp) else np.nan,
                              max_beta=round(float(np.max(bp)), 4) if len(bp) else np.nan))
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(TBL_DIR / "glm_tweedie_beta_vs_p.csv", index=False)
    print("\n=== SENSIBILIDAD β_alfa a var_power (full model, 6 sujetos) ===")
    for _, rr in df_sens.iterrows():
        print(f"  p={rr['p']}: β_alfa {int(rr['n_neg'])}/{int(rr['n'])} neg, "
              f"media={rr['mean_beta_alpha']:+.4f} (rango {rr['min_beta']:+.4f}..{rr['max_beta']:+.4f})",
              flush=True)

    # ---------- (2) TWEEDIE vs HURDLE sobre amplitud SCR (zero-inflada) ----------
    rows_h = []
    for sub, d in data.items():
        m = np.isfinite(d["amp"]) & np.isfinite(d["alpha"])
        amp = np.clip(d["amp"][m], 0, None)
        X = _design(d, m, with_conf=True)
        pos = amp > 0
        # Tweedie sobre amplitud
        try:
            rt = _fit_tweedie(amp, X, 1.5)
            tw_aic, tw_b = float(rt.aic), float(rt.params["alpha"])
        except Exception:
            tw_aic, tw_b = np.nan, np.nan
        # Hurdle: logística P(amp>0) + Gamma(log) sobre positivos
        rl = sm.GLM(pos.astype(float), X, family=sm.families.Binomial()).fit()
        try:
            rg = sm.GLM(amp[pos], X.loc[pos], family=sm.families.Gamma(sm.families.links.Log())).fit()
            g_b, g_p, g_llf, g_k = float(rg.params["alpha"]), float(rg.pvalues["alpha"]), float(rg.llf), int(X.shape[1] + 1)
        except Exception:
            g_b, g_p, g_llf, g_k = np.nan, np.nan, np.nan, 0
        hurdle_aic = -2 * (float(rl.llf) + g_llf) + 2 * (int(X.shape[1]) + g_k)
        rows_h.append(dict(subject=sub, n=int(m.sum()), n_pos=int(pos.sum()),
                           tweedie_beta_alpha=round(tw_b, 4), tweedie_aic=round(tw_aic, 1) if np.isfinite(tw_aic) else np.nan,
                           hurdle_logit_beta=round(float(rl.params["alpha"]), 4),
                           hurdle_logit_p=round(float(rl.pvalues["alpha"]), 4),
                           hurdle_gamma_beta=round(g_b, 4), hurdle_gamma_p=round(g_p, 4),
                           hurdle_aic=round(hurdle_aic, 1) if np.isfinite(hurdle_aic) else np.nan))
        print(f"  [amp] {sub}: Tweedie β={tw_b:+.3f} (AIC {tw_aic:.0f}) | "
              f"hurdle logit β={float(rl.params['alpha']):+.3f} gamma β={g_b:+.3f} (AIC {hurdle_aic:.0f})", flush=True)
    df_h = pd.DataFrame(rows_h); df_h.to_csv(TBL_DIR / "glm_tweedie_vs_hurdle_amp.csv", index=False)

    # ---------- veredictos ----------
    print("\n=== VEREDICTO GLM (titular) ===")
    bf = df_tw[df_tw.model == "full"]["beta_alpha"].to_numpy(float)
    bm = df_tw[df_tw.model == "alpha_only"]["beta_alpha"].to_numpy(float)
    _, pf = stats.ttest_1samp(bf, 0.0)
    print(f"  Tweedie SMNA β_alfa FULL (confounds dentro): {int((bf<0).sum())}/6 neg, "
          f"media={bf.mean():+.4f}, p_group={pf:.3f}")
    print(f"  Tweedie SMNA β_alfa alpha-only:               {int((bm<0).sum())}/6 neg, media={bm.mean():+.4f}")
    print(f"  -> ¿β_alfa sobrevive con confounds adentro? {'SÍ' if (bf<0).sum()>=5 else 'NO'} "
          f"(signo {int((bf<0).sum())}/6)")
    hl = df_h["hurdle_logit_beta"].to_numpy(float); hg = df_h["hurdle_gamma_beta"].to_numpy(float)
    print(f"  Hurdle logit β_alfa (P(SCR>0)): {int((hl<0).sum())}/6 neg, media={hl.mean():+.3f}")
    print(f"  Hurdle gamma β_alfa (|amp>0):   {int((hg<0).sum())}/6 neg, media={np.nanmean(hg):+.3f}")
    awin = int((df_h["tweedie_aic"] < df_h["hurdle_aic"]).sum())
    print(f"  AIC: Tweedie < hurdle (amp) en {awin}/6 sujetos", flush=True)

    _plot(df_tw, df_h)
    print(f"\nOutputs -> {OUT_DIR}\n[GLM Tweedie] done", flush=True)


def _plot(df_tw, df_h):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sf = df_tw[df_tw.model == "full"].reset_index(drop=True)
    sm_ = df_tw[df_tw.model == "alpha_only"].reset_index(drop=True)
    x = np.arange(len(sf)); w = 0.38
    axes[0].bar(x - w / 2, sm_["beta_alpha"], w, label="alfa sola", color="C0")
    axes[0].bar(x + w / 2, sf["beta_alpha"], w, label="full (confounds dentro)", color="C1")
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xticks(x); axes[0].set_xticklabels(sf["subject"], fontsize=8)
    axes[0].set_ylabel("β_alfa (Tweedie, link log)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Tweedie sobre SMNA-AUC: β_alfa con/sin confounds\n(negativo = más desync -> más drive)", fontsize=10)
    axes[1].bar(x - w / 2, df_h["hurdle_logit_beta"], w, label="hurdle logit (P SCR>0)", color="C2")
    axes[1].bar(x + w / 2, df_h["hurdle_gamma_beta"], w, label="hurdle gamma (|amp)", color="C3")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xticks(x); axes[1].set_xticklabels(df_h["subject"], fontsize=8)
    axes[1].set_ylabel("β_alfa"); axes[1].legend(fontsize=8)
    axes[1].set_title("Hurdle (two-part) sobre amplitud SCR: β_alfa por parte", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "glm_tweedie_beta_alpha.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
