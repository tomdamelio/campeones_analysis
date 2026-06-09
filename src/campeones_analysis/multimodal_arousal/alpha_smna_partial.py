"""R3.1 paso 2 — ¿el graded alfa-SMNA sobrevive el control fuerte (mediador ojo/movimiento)?

R3.1 mostró que la desync alfa periódica escala con el SMNA-AUC (6/6, rho −0.08..−0.11). Pero el
confound clave es el **mediador cierre-de-ojos→alfa**: más arousal → cambia el estado ocular → cambia
alfa, por una vía que NO es acoplamiento cortical-SCR genuino. Acá se descuenta vía correlación PARCIAL
controlando: EOG-lento (VEOG slow = estado ocular), blink (Fp1/Fp2 slow), movimiento (var_jerk), tiempo.

Si la parcial(alfa, SMNA | ojo+movimiento+tiempo) se mantiene negativa 6/6 → acoplamiento genuino
(no mediado por el ojo); si cae a 0 → era mediado. Within-subject, todas las épocas y SCR-only.
Covariados alineados 1:1 al cache (reusa confound_model._load_covariates). Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_smna_partial
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
CONTROL_COLS = ["veog_slow_0p5_8", "blink_slow", "var_jerk", "tnorm"]   # ojo + blink + movimiento + tiempo


def _alpha_post(psd, freqs, post_idx):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-30"])
    m = (f >= 8.0) & (f < 13.0)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)[:, post_idx].mean(axis=1)


def _smna_auc(smna, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(smna), i1)
    return float(np.trapz(np.clip(smna[i0:i1], 0, None), dx=1.0 / EDA_FS)) if i1 - i0 >= 2 else np.nan


def _resid(v, C):
    A = np.column_stack([C, np.ones(len(C))])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _partial_spearman(x, y, C):
    """Spearman de los residuos de x e y tras quitar C (linealmente). x,y a rangos primero."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    return float(stats.spearmanr(_resid(rx, C), _resid(ry, C)).correlation)


def main():
    print("=" * 78)
    print("alpha_smna_partial :: R3.1 paso 2 — parcial(alfa, SMNA | ojo+movimiento+tiempo)")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    post_idx = [ch.index(c) for c in POSTERIOR if c in ch]
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)

    rows = []
    for sub in COHORT:
        psd, y, tn = cache[sub]
        alpha = _alpha_post(psd, freqs, post_idx)
        C = np.column_stack([covs[sub][c] for c in CONTROL_COLS])
        # SMNA-AUC por época (real luego silent, orden del cache)
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_auc, sil_auc, real_tn, sil_tn = [], [], [], []
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
            real_auc += [_smna_auc(smna, t) for t in rk]; real_tn += list(rk / dur)
            sil_auc += [_smna_auc(smna, t) for t in sk]; sil_tn += list(sk / dur)
        auc = np.array(real_auc + sil_auc)
        if len(auc) != len(tn) or np.max(np.abs(np.array(real_tn + sil_tn) - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN -> skip", flush=True); continue

        for scope, mk in [("all", np.ones(len(y), bool)), ("scr_only", y == 1)]:
            m = mk & np.isfinite(auc) & np.isfinite(alpha)
            raw_rho = float(stats.spearmanr(auc[m], alpha[m]).correlation)
            par_rho = _partial_spearman(auc[m], alpha[m], C[m])
            rows.append(dict(subject=sub, scope=scope, n=int(m.sum()),
                             raw_rho=round(raw_rho, 3), partial_rho=round(par_rho, 3)))
        r = [x for x in rows if x["subject"] == sub and x["scope"] == "all"][0]
        print(f"  {sub}: all raw={r['raw_rho']:+.3f} -> partial={r['partial_rho']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_smna_partial.csv", index=False)

    print("\n=== Parcial alfa-SMNA controlando ojo(VEOG/blink)+movimiento+tiempo ===")
    print("  (parcial negativa 6/6 = acoplamiento genuino, no mediado por el ojo)")
    for scope in ("all", "scr_only"):
        sd = df[df.scope == scope]
        raw = sd["raw_rho"].to_numpy(); par = sd["partial_rho"].to_numpy()
        zr = np.arctanh(np.clip(par, -0.999, 0.999))
        _, pt = stats.ttest_1samp(zr, 0.0)
        print(f"  [{scope:8s}] raw mean={raw.mean():+.3f} ({int((raw<0).sum())}/6 neg)  ->  "
              f"PARCIAL mean={par.mean():+.3f} ({int((par<0).sum())}/6 neg, p_group={pt:.3f})", flush=True)
        print(f"            parcial por sujeto: {[f'{x:+.2f}' for x in par]}", flush=True)

    # figura: raw vs partial por sujeto (scope=all)
    sd = df[df.scope == "all"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sd)); w = 0.38
    ax.bar(x - w / 2, sd["raw_rho"], w, label="raw rho", color="C0")
    ax.bar(x + w / 2, sd["partial_rho"], w, label="parcial (| ojo+mov+tiempo)", color="C1")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sd["subject"], fontsize=9)
    ax.set_ylabel("Spearman(alfa periódica, SMNA-AUC)")
    ax.set_title("R3.1 paso 2: ¿el graded alfa-SMNA sobrevive el control del ojo/movimiento?\n"
                 "(parcial sigue negativa = acoplamiento genuino · cae a 0 = mediado por el ojo)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "alpha_smna_partial.png", dpi=120); plt.close(fig)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.1 paso 2] done", flush=True)


if __name__ == "__main__":
    main()
