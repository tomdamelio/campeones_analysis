"""2.4 — Topografía edge/central como covariado de confound muscular POR ÉPOCA.

Bloque 2 del ciclo 05_05, frente-gamma (Q-gamma, el crux). El 05_04 reportó el contraste
edge-vs-central de potencia alta como grand-average descriptivo (emg_highfreq_scr /
topo_variance_scr). Acá lo exponemos como **valor por época** -> el covariado `edge_central_db`
que 2.5 va a usar para adjudicar si el gamma↑ (6/6, EMG-sospechoso) es cortical o muscular.

Enzo (reunión 2026-06-05): "qué tanto en los electrodos más periféricos tenés gama, relativo
a los centrales... suele ser un indicador de musculares". No hay mastoides ni EMG puro -> el
proxy se infiere de la topografía EEG (borde temporal/posterolateral vs central-parietal).

Diseño (consistente con 1.5/1.6 y el RE-ENCUADRE post-B1):
  - Reusa el cache panel_psd.npz (29 ch, esquema uniforme) vía decoding_panel.load_cache:
    PSD por época, ya con DROP_CHANNELS aplicado (FC1/TP10/Fz fuera).
  - Sets edge/central desde cohort.EMG_EDGE / cohort.CENTRAL (única fuente de verdad, 29 ch).
  - Covariado primario = log-ratio (dB) de potencia de banda CRUDA edge/central por época, por
    banda (delta 1-4, delta 2-4, theta, alpha, beta, gamma 30-40). La potencia cruda es
    estrictamente positiva -> el log-ratio por época es estable (la firma muscular es potencia
    cruda edge-dominante; índice canónico de emg_highfreq_scr, acá por época).
  - Secundario = gamma PERIÓDICO (1/f removido, _linear_aperiodic) edge/central, alineado al
    crux F3 (30-40 Hz), como cross-check de la métrica continua del periódico.
  - Offset (aperiódico) regional edge/central por época (Q-offset descansa en covariados
    externos; el offset edge-central se reporta within-subject, F6).
  - Test within-subject SCR vs no-SCR (Mann-Whitney por sujeto) + consistencia de signo 6/6.

NO computa la disociación topográfica GA ni el CSD-survival (eso vive en topo_variance_scr /
emg_highfreq_scr, ya corregidos a 29 ch). Este script produce el COVARIADO por época para 2.5.

Estatus Track B: la métrica edge/central sobre 29 ch es robusta a FC1 (ya fuera) -> arranca
como stopgap; la CSD-survival definitiva y el cruce con la VD periódica final se re-corren tras
Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.edge_central_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.campeones_analysis.multimodal_arousal.cohort import (
    CENTRAL,
    EMG_EDGE,
    FRONTOPOLAR,
    REPO,
    SUBJ_COLORS,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic

warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_4_edge_central"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Bandas (incluye delta partido 1-4 / 2-4 para el cruce con 2.6 corte ≤2 Hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta_1_4": (1.0, 4.0),
    "delta_2_4": (2.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
RANGE = "1-40"   # para el ajuste 1/f (offset + gamma periódico)


def _idx(ch_names: list[str], region: list[str]) -> list[int]:
    return [ch_names.index(c) for c in region if c in ch_names]


def _raw_bandpower(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Potencia cruda media de banda por época y canal (lineal). psd=(n_ep,n_ch,n_freq)."""
    lo, hi = band
    m = (freqs >= lo) & (freqs < hi)
    return psd[:, :, m].mean(axis=2)


def _edge_central_db(power: np.ndarray, e_idx: list[int], c_idx: list[int]) -> np.ndarray:
    """log-ratio (dB) edge/central por época sobre potencia lineal estrictamente positiva."""
    e = power[:, e_idx].mean(axis=1)
    c = power[:, c_idx].mean(axis=1)
    return 10.0 * np.log10((e + 1e-30) / (c + 1e-30))


def _subject_covariates(psd, freqs, e_idx, c_idx) -> dict[str, np.ndarray]:
    """Todos los covariados edge/central por época para un sujeto."""
    covs: dict[str, np.ndarray] = {}
    # (1) covariado primario: edge/central dB de potencia CRUDA por banda
    for bn, band in BANDS.items():
        covs[f"edge_central_db_{bn}"] = _edge_central_db(_raw_bandpower(psd, freqs, band), e_idx, c_idx)
    # (2) cross-check: gamma PERIÓDICO (1/f removido) edge/central
    off, _exp, resid, f = _linear_aperiodic(psd, freqs, RANGES[RANGE])
    gmask = (f >= 30.0) & (f < 40.0)
    gper = np.clip(resid[:, :, gmask], 0, None).mean(axis=2)   # (n_ep, n_ch)
    ge = gper[:, e_idx].mean(axis=1)
    gc = gper[:, c_idx].mean(axis=1)
    eps = max(float(np.median(gper)) * 1e-3, 1e-30)            # piso robusto (periódico ~0)
    covs["gamma_periodic_edge_central"] = np.log10((ge + eps) / (gc + eps))
    # (3) offset (aperiódico) regional
    covs["offset_edge"] = off[:, e_idx].mean(axis=1)
    covs["offset_central"] = off[:, c_idx].mean(axis=1)
    covs["offset_edge_minus_central"] = covs["offset_edge"] - covs["offset_central"]
    return covs


def _within_subject_test(df: pd.DataFrame, cov_cols: list[str], subjects: list[str]) -> pd.DataFrame:
    """Por covariado: SCR vs no-SCR within-subject (Mann-Whitney) + consistencia de signo 6/6."""
    rows = []
    for col in cov_cols:
        diffs, pvals, n_pos, n = [], [], 0, 0
        for sub in subjects:
            sd = df[df["subject"] == sub]
            scr = sd[sd["condition"] == 1][col].to_numpy()
            nos = sd[sd["condition"] == 0][col].to_numpy()
            if len(scr) < 5 or len(nos) < 5:
                continue
            d = float(np.median(scr) - np.median(nos))
            try:
                p = float(mannwhitneyu(scr, nos, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            diffs.append(d)
            pvals.append(p)
            n += 1
            n_pos += int(d > 0)
        rows.append(dict(
            covariate=col, n_subj=n,
            mean_diff_scr_minus_nos=round(float(np.mean(diffs)), 4) if diffs else np.nan,
            n_pos=f"{n_pos}/{n}",
            median_p=round(float(np.median(pvals)), 4) if pvals else np.nan,
        ))
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, subjects: list[str]) -> None:
    band_cols = [f"edge_central_db_{bn}" for bn in BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # (izq) nivel basal edge/central por banda (todas las épocas): gamma debería ser el más
    # edge-dominante si la alta freq es muscular. Media por sujeto -> dispersión inter-sujeto.
    ax = axes[0]
    means = {bn: [] for bn in BANDS}
    for sub in subjects:
        sd = df[df["subject"] == sub]
        for bn in BANDS:
            means[bn].append(sd[f"edge_central_db_{bn}"].mean())
    x = np.arange(len(BANDS))
    gm = [float(np.mean(means[bn])) for bn in BANDS]
    ax.bar(x, gm, color="0.7", zorder=1)
    for k, sub in enumerate(subjects):
        ax.scatter(x + np.linspace(-0.22, 0.22, len(subjects))[k],
                   [means[bn][k] for bn in BANDS], color=SUBJ_COLORS[sub], s=28, zorder=3, label=sub)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(list(BANDS.keys()), rotation=20, fontsize=9)
    ax.set_ylabel("edge/central (dB, potencia cruda)")
    ax.set_title("Nivel edge/central por banda (todas las épocas)\n>0 = más potencia en borde",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # (der) gamma edge/central: SCR vs no-SCR por sujeto (¿más edge en SCR? = confound de evento)
    ax = axes[1]
    for k, sub in enumerate(subjects):
        sd = df[df["subject"] == sub]
        scr = sd[sd["condition"] == 1]["edge_central_db_gamma"].median()
        nos = sd[sd["condition"] == 0]["edge_central_db_gamma"].median()
        ax.plot([0, 1], [nos, scr], "-o", color=SUBJ_COLORS[sub], lw=1.5, ms=5, label=sub)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-SCR", "SCR"])
    ax.set_ylabel("gamma edge/central (dB), mediana por sujeto")
    ax.set_title("gamma edge/central: SCR vs no-SCR\n(sube en SCR = gamma más edge = más sospechoso)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("2.4 Covariado edge/central por época (29 ch, N=6, esquema uniforme)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "edge_central_byband.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("edge_central_scr :: 2.4 covariado edge/central por época (frente-gamma)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    data, freqs, ch = load_cache("uniform")
    subjects = list(data.keys())
    e_idx, c_idx, fp_idx = _idx(ch, EMG_EDGE), _idx(ch, CENTRAL), _idx(ch, FRONTOPOLAR)
    print(f"  cache: {len(subjects)} suj, {len(ch)} ch")
    print(f"  EMG_EDGE  ({len(e_idx)}/{len(EMG_EDGE)}): {[ch[i] for i in e_idx]}")
    print(f"  CENTRAL   ({len(c_idx)}/{len(CENTRAL)}): {[ch[i] for i in c_idx]}")
    print(f"  FRONTOPOLAR ({len(fp_idx)}/{len(FRONTOPOLAR)}): {[ch[i] for i in fp_idx]}", flush=True)
    missing = [c for c in EMG_EDGE + CENTRAL if c not in ch]
    if missing:
        print(f"  ⚠️  canales esperados ausentes del cache: {missing}", flush=True)

    rows = []
    for sub in subjects:
        psd, y, tn = data[sub]
        covs = _subject_covariates(psd, freqs, e_idx, c_idx)
        for i in range(len(y)):
            row = dict(subject=sub, epoch=int(i), condition=int(y[i]), tnorm=float(tn[i]))
            for k, v in covs.items():
                row[k] = float(v[i])
            rows.append(row)
        print(f"  {sub}: n_ep={len(y)} (SCR={int((y == 1).sum())}, no-SCR={int((y == 0).sum())})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "edge_central_covariate.csv", index=False)
    print(f"\nCovariado por época -> {TBL_DIR / 'edge_central_covariate.csv'} ({len(df)} épocas)")

    cov_cols = [c for c in df.columns if c not in ("subject", "epoch", "condition", "tnorm")]
    summ = _within_subject_test(df, cov_cols, subjects)
    summ.to_csv(TBL_DIR / "edge_central_within_subject.csv", index=False)
    print("\nWithin-subject SCR vs no-SCR (signo 6/6 = confound de evento consistente):")
    print(summ.to_string(index=False), flush=True)

    _plot(df, subjects)
    print(f"\nFigura -> {FIG_DIR / 'edge_central_byband.png'}")
    print(f"[2.4] done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
