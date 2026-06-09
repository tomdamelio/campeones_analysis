"""R3.1 (v2, idea del usuario) — ¿la desync alfa escala con el SMNA-AUC de la época?

El pico del SCR dio un graded null (4/6, p=0.11). El usuario propone una Y mejor: el **SMNA-AUC**
(integral del driver sudomotor de cvxEDA sobre la ventana de la época) — integra el drive autonómico
(menos ruidoso que un solo pico), y usando TODAS las épocas (SCR + no-SCR) el eje X cubre el rango
completo (de ~0 en no-SCR a alto en SCRs grandes) = regresión continua de verdad.

Alfa/theta/beta periódico posterior (del cache panel_psd, 1/f sin gamma) vs SMNA-AUC por época
(re-derivado del driver `eda_smna`, alineado 1:1 al cache vía tnorm). Within-subject Spearman:
  - TODAS las épocas (SCR+no-SCR): la versión continua del contraste (rango completo de SMNA).
  - SCR-only: el dose-response puro (¿escala dentro del rango de SCRs?).
Predicción: NEGATIVO (más SMNA -> más desync). Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_smna_graded
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
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

DESYNC_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]


def _periodic_post(psd, freqs, band, post_idx):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-30"])
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2)[:, post_idx].mean(axis=1)


def _smna_auc(smna, t0):
    i0, i1 = int(round(t0 * EDA_FS)), int(round((t0 + POST_S) * EDA_FS))
    i0, i1 = max(0, i0), min(len(smna), i1)
    if i1 - i0 < 2:
        return np.nan
    return float(np.trapz(np.clip(smna[i0:i1], 0, None), dx=1.0 / EDA_FS))


def main():
    print("=" * 78)
    print("alpha_smna_graded :: R3.1 v2 — desync alfa periódica vs SMNA-AUC por época")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    post_idx = [ch.index(c) for c in POSTERIOR if c in ch]
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)

    data, align = {}, []
    for sub in COHORT:
        psd, y, tn = cache[sub]
        bands = {b: _periodic_post(psd, freqs, band, post_idx) for b, band in DESYNC_BANDS.items()}
        # re-derivar SMNA-AUC por época (real luego silent, orden del cache)
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
        my_tn = np.array(real_tn + sil_tn)
        ok = len(auc) == len(tn) and np.max(np.abs(my_tn - tn)) < 1e-3
        align.append(dict(subject=sub, n=len(auc), match=ok))
        if not ok:
            print(f"  {sub}: MISALIGN n={len(auc)} vs {len(tn)}", flush=True); continue
        data[sub] = dict(auc=auc, y=y, **bands)
        print(f"  {sub}: n={len(auc)} (SCR={int((y==1).sum())})  align OK", flush=True)

    # within-subject Spearman: todas las épocas y SCR-only
    rows = []
    for sub, d in data.items():
        for b in DESYNC_BANDS:
            for scope, mask in [("all", np.isfinite(d["auc"])),
                                ("scr_only", (d["y"] == 1) & np.isfinite(d["auc"]))]:
                m = mask & np.isfinite(d[b])
                rho, p = stats.spearmanr(d["auc"][m], d[b][m])
                rows.append(dict(subject=sub, band=b, scope=scope, n=int(m.sum()),
                                 rho=round(float(rho), 3), p=round(float(p), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_smna_graded.csv", index=False)

    print("\n=== Spearman(alfa/theta/beta periódico posterior, SMNA-AUC) within-subject ===")
    print("  (NEGATIVO = más desync con más drive sudomotor)")
    summ = []
    for scope in ("all", "scr_only"):
        for b in DESYNC_BANDS:
            rr = df[(df.band == b) & (df.scope == scope)]["rho"].to_numpy(float)
            z = np.arctanh(np.clip(rr, -0.999, 0.999))
            t, pt = stats.ttest_1samp(z, 0.0)
            summ.append(dict(scope=scope, band=b, mean_rho=round(float(rr.mean()), 3),
                             n_neg=f"{int((rr<0).sum())}/{len(rr)}", p_group=round(float(pt), 4)))
            print(f"  [{scope:8s}] {b:6s}: mean_rho={rr.mean():+.3f}  neg={int((rr<0).sum())}/{len(rr)}  "
                  f"p={pt:.3f}  por_suj={[f'{x:+.2f}' for x in rr]}", flush=True)
    pd.DataFrame(summ).to_csv(TBL_DIR / "alpha_smna_graded_group.csv", index=False)

    _plot(data, df)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.1 v2] done", flush=True)


def _plot(data, df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (sub, d) in zip(axes.ravel(), data.items()):
        x, yv = d["auc"], d["alpha"]
        scr = d["y"] == 1
        ax.scatter(x[~scr], yv[~scr], s=8, alpha=0.3, color="0.6", label="no-SCR")
        ax.scatter(x[scr], yv[scr], s=8, alpha=0.4, color=SUBJ_COLORS.get(sub, "C0"), label="SCR")
        m = np.isfinite(x) & np.isfinite(yv)
        if m.sum() > 3:
            b1, b0 = np.polyfit(x[m], yv[m], 1)
            xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5)
        rho = df[(df.subject == sub) & (df.band == "alpha") & (df.scope == "all")]["rho"].values[0]
        ax.set_title(f"{sub}  alpha rho(all)={rho:+.2f}", fontsize=9)
        ax.set_xlabel("SMNA-AUC época"); ax.set_ylabel("alfa periódica posterior")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("R3.1 v2: alfa periódica posterior vs SMNA-AUC por época (SCR+no-SCR)\n"
                 "(pendiente NEGATIVA = más desync con más drive sudomotor)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "alpha_smna_scatter.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
