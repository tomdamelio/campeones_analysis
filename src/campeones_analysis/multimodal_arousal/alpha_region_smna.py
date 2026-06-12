"""Test banda(alfa)×REGIÓN: ¿el acoplamiento desync-alfa <-> SMNA es POSTERIOR-DESPROPORCIONADO
(específico) o UNIFORME/global (à la Barry 2007 / Duda 2024)?

Barry et al. 2007: la modulación alfa-arousal es GLOBAL (reducción uniforme proporcional), NO
posterior-específica — el alfa es posterior solo porque ahí vive el generador. Duda, Clarke & Barry
2024 + Chien 2020: el desync alfa posterior se asocia a atención/saliencia, no necesariamente al
drive autonómico. Este test pregunta sobre NUESTRO acoplamiento graduado (R3.1: alfa-PO periódica
<-> SMNA-AUC, 6/6 neg): ¿es más fuerte en posterior que en central/frontal, o uniforme entre regiones?

Reusa la maquinaria per-época de `lag_sweep_alpha_smna` (mismo cache `uniform`, mismos covariados,
misma alfa periódica 1/f-removida 8-13 Hz, mismo barrido de tau). ÚNICO cambio: el SET DE CANALES
sobre el que se promedia la alfa -> tres ROIs DISJUNTOS (posterior / central / frontal).

Métrica = Spearman parcial(SMNA-AUC, alfa-región | ojo+mov+tiempo) por sujeto y región:
  - PRIMARIO: a tau=0 (contemporáneo, neutral; evita circularidad de elegir tau* por región).
  - SECUNDARIO: min-sobre-tau (mejor acoplamiento por región).
Comparación de grupo: Wilcoxon pareado posterior vs central / posterior vs frontal (n=6, 1-tailed:
posterior MÁS negativa = más desync acoplada).

CAVEAT (al diario): posterior tiene MÁS alfa (mejor SNR del estimador) -> una correlación más fuerte
podría reflejar SNR y no especificidad regional verdadera. Es evidencia SUGESTIVA, no definitiva.
La métrica es Spearman (rank, scale-free), lo que mitiga diferencias de amplitud pero no de SNR.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_region_smna
"""

from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import compute_current_source_density
from scipy import stats

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, DROP_CHANNELS, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.confound_model_scr import _load_covariates
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED, load_cache
from src.campeones_analysis.multimodal_arousal.erp_scr import (
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
    CONTROL_COLS,
    EEG_FS,
    TAU_GRID,
    _band_windows,
    _partial_curve,
    _smna_auc,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

USE_CSD = os.environ.get("ALPHA_REGION_CSD", "0") == "1"  # CSD/Laplaciano vs sensores (ref. promedio)
_TAG = "_csd" if USE_CSD else ""
OUT_DIR = REPO / "research_diary" / "context" / "05_05" / f"3_3_alpha_region{_TAG}"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ROIs disjuntos (se intersecan con los canales disponibles; se excluyen DROP_CHANNELS y el borde EMG)
REGIONS_RAW = {
    "posterior": ["P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
    "central": ["FCz", "FC2", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"],
    "frontal": ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "F7", "F3", "F4", "F8", "FC5", "FC6"],
}
TAU0_J = int(np.argmin(np.abs(TAU_GRID)))  # índice de tau=0 en la grilla


def _region_idx(ch_names: list[str]) -> dict[str, list[int]]:
    """Índices por región: intersección con canales presentes, sin DROP_CHANNELS, disjuntos."""
    out = {}
    for reg, names in REGIONS_RAW.items():
        idx = [ch_names.index(c) for c in names if c in ch_names and c not in DROP_CHANNELS]
        out[reg] = idx
    return out


def main():
    print("=" * 80)
    print("alpha_region_smna :: banda(alfa)×REGIÓN — ¿acoplamiento alfa<->SMNA posterior-específico o global?")
    print(f"  modo = {'CSD (surface Laplacian)' if USE_CSD else 'SENSORES (ref. promedio)'}")
    print("=" * 80, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)

    regions = list(REGIONS_RAW)
    data = {}
    for sub in COHORT:
        psd, y, tn = cache[sub]
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_rows = {reg: [] for reg in regions}
        sil_rows = {reg: [] for reg in regions}
        real_auc, sil_auc, real_tn, sil_tn = [], [], [], []
        ridx_print = None
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(EEG_FS, verbose="ERROR")
            if USE_CSD:  # surface Laplacian: suprime volume-conduction far-field, realza fuentes focales
                raw.drop_channels([c for c in DROP_CHANNELS if c in raw.ch_names])
                raw = compute_current_source_density(raw, verbose="ERROR")
            dur = float(raw.times[-1])
            D = raw.get_data()
            ridx = _region_idx(list(raw.ch_names))
            if ridx_print is None:
                ridx_print = {r: [raw.ch_names[i] for i in ix] for r, ix in ridx.items()}
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)
            ons = detect_scr_onsets_s(eda, EEG_FS if False else _erp.EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, _erp.EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=_erp.EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            rk = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sk = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            for reg, ix in ridx.items():
                if not ix:
                    continue
                real_rows[reg].append(_band_windows(D, rk, ix)["alpha"])
                sil_rows[reg].append(_band_windows(D, sk, ix)["alpha"])
            real_auc += [_smna_auc(smna, t) for t in rk]; real_tn += list(rk / dur)
            sil_auc += [_smna_auc(smna, t) for t in sk]; sil_tn += list(sk / dur)

        auc = np.array(real_auc + sil_auc)
        my_tn = np.array(real_tn + sil_tn)
        if len(auc) != len(tn) or np.max(np.abs(my_tn - tn)) > 1e-3:
            print(f"  {sub}: MISALIGN n={len(auc)} vs {len(tn)} -> skip", flush=True); continue
        reg_alpha = {reg: np.vstack(real_rows[reg] + sil_rows[reg]) for reg in regions if real_rows[reg]}
        C = np.column_stack([covs[sub][c] for c in CONTROL_COLS])
        data[sub] = dict(auc=auc, y=y, C=C, reg_alpha=reg_alpha)
        print(f"  {sub}: n={len(auc)} (SCR={int((y == 1).sum())}) align OK  "
              f"regs={ {r: len(v) for r, v in (ridx_print or {}).items()} }", flush=True)

    if ridx_print:
        print("\nCanales por región:")
        for r, cs in ridx_print.items():
            print(f"  {r:9s} ({len(cs)}): {cs}", flush=True)

    # --------- acoplamiento parcial por sujeto×región (tau=0 primario + min-sobre-tau) ---------
    rows = []
    for sub, d in data.items():
        mk = np.ones(len(d["y"]), bool)
        for reg in regions:
            if reg not in d["reg_alpha"]:
                continue
            _, par = _partial_curve(d["auc"], d["reg_alpha"][reg], d["C"], mk)
            rho0 = float(par[TAU0_J])
            rho_min = float(np.nanmin(par))
            tau_min = float(TAU_GRID[int(np.nanargmin(par))])
            rows.append(dict(subject=sub, region=reg, rho_tau0=round(rho0, 3),
                             rho_min=round(rho_min, 3), tau_min=tau_min))
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_region_coupling.csv", index=False)

    # --------- veredicto de grupo: posterior vs central / frontal ---------
    piv0 = df.pivot_table(index="subject", columns="region", values="rho_tau0")
    pivm = df.pivot_table(index="subject", columns="region", values="rho_min")
    print("\n=== ACOPLAMIENTO alfa<->SMNA POR REGIÓN (parcial Spearman, neg = más desync acoplada) ===")
    print("\n  rho a tau=0 (contemporáneo):")
    print(piv0.round(3).to_string())
    print(f"\n  media por región (tau=0): "
          f"{ {r: round(float(piv0[r].mean()), 3) for r in piv0.columns} }")
    print(f"  media por región (min-tau): "
          f"{ {r: round(float(pivm[r].mean()), 3) for r in pivm.columns} }")

    def _paired(a_col, b_col, piv, label):
        if a_col not in piv.columns or b_col not in piv.columns:
            return
        a, b = piv[a_col].to_numpy(float), piv[b_col].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        nneg = int((a < b).sum())  # posterior más negativa que la otra región
        try:
            w = stats.wilcoxon(a, b, alternative="less")  # H1: posterior < otra (más negativa)
            p = w.pvalue
        except ValueError:
            p = np.nan
        print(f"  [{label}] posterior MÁS negativa que {b_col}: {nneg}/{len(a)}  | "
              f"Wilcoxon 1-tailed p={p:.3f}  (Δ medio={np.mean(a - b):+.3f})", flush=True)

    print("\n  --- posterior-específico vs uniforme (Wilcoxon pareado, n=6) ---")
    print("  tau=0:")
    _paired("posterior", "central", piv0, "tau0 post-vs-central")
    _paired("posterior", "frontal", piv0, "tau0 post-vs-frontal")
    print("  min-sobre-tau:")
    _paired("posterior", "central", pivm, "min post-vs-central")
    _paired("posterior", "frontal", pivm, "min post-vs-frontal")

    print("\n  LECTURA: si posterior es consistentemente MÁS negativa que central/frontal -> el "
          "acoplamiento es posterior-DESPROPORCIONADO (especificidad > Barry/Duda). Si es uniforme "
          "entre regiones -> 'arousal global' à la Barry 2007 (no posterior-específico).")

    _plot(piv0, pivm)
    print(f"\nOutputs -> {OUT_DIR}\n[banda×región] done", flush=True)


def _plot(piv0, pivm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    order = [r for r in ["frontal", "central", "posterior"] if r in piv0.columns]
    for ax, piv, title in [(axes[0], piv0, "tau=0 (contemporáneo)"),
                           (axes[1], pivm, "min-sobre-tau (mejor)")]:
        x = np.arange(len(order))
        for sub in piv.index:
            ax.plot(x, [piv.loc[sub, r] for r in order], "o-",
                    color=SUBJ_COLORS.get(sub, "0.7"), lw=1, alpha=0.55, ms=4, label=sub)
        ax.plot(x, [piv[r].mean() for r in order], "ks-", lw=2.4, ms=8, label="media")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_xlabel("región"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("parcial Spearman(alfa, SMNA-AUC | ojo+mov+tiempo)\n(más negativo = más acoplamiento desync)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Test banda(alfa)×REGIÓN: ¿acoplamiento alfa<->SMNA posterior-específico o global?\n"
                 "posterior MÁS negativa que central/frontal => específico (> Barry/Duda); uniforme => arousal global",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "alpha_region_coupling.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
