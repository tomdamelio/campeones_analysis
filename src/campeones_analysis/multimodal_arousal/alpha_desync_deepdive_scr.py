"""Deep-dive del alfa-desync: ¿ERD cortical time-locked o un offset TÓNICO de estado?

Tarea A del 2026-06-03 (tras cerrar el frente HF = EMG residual). El alfa-desync quedó como
el único candidato a señal cortical, pero hay DOS medidas que aparentemente discrepan:
  - TÓNICO (FOOOF, whole-epoch, sin baseline): Δalfa periódico GA -0.128 (sub-33 -0.65).
  - EVENT-RELATED (alpha_hypothesis_scr, WOI [0,1] post-onset, % change vs baseline (-5,-4.5),
    parieto-occipital): diffs ±0.1-0.5%, 0/6 significativo.

Pregunta: ¿el alfa baja LOCKEADO al SCR (ERD genuino) o es un nivel tónico más bajo a lo largo
de toda la época (estado de mayor arousal, confound de estímulo)? Y: ¿el efecto tónico sobrevive
dropear al sujeto dominante (sub-33)?

Estrategia (reusa build_subject_epochs + alpha_roi_per_epoch + constantes de alpha_hypothesis;
sin reprocesar; mismas épocas no-AR que generaron el -0.128 del FOOOF):
  - Por sujeto: time course alfa (8-13 Hz, parieto-occipital) baseline-corregido, real vs silent.
    Ventanas: PRE (-3,-1) y POST (0,1). Si hay ERD genuino, real < silent localizado cerca del
    onset; si es tónico, el FOOOF (sin baseline) baja pero el baseline-corregido NO.
  - TÓNICO desde fooof_scr_contrasts.csv (d_periodic_alpha por canal, promediado parieto-occipital).
  - GA time course (real-silent) con los 6 y SIN sub-33 (robustez al sujeto dominante).
  - Tabla per-sujeto: tónico (FOOOF) vs event-related (WOI pre/post) + figura GA.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_desync_deepdive_scr
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.alpha_hypothesis_scr import (
    PARIETOOCCIPITAL,
    WIN_OF_INTEREST,
    alpha_roi_per_epoch,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "qa_artifact_vs_signal" / "alpha_desync"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

FOOOF_CONTRASTS = OUT / "qa_artifact_vs_signal" / "tables" / "fooof_scr_contrasts.csv"
WOI_PRE = (-3.0, -1.0)    # pre-onset (el SCR onset lagea al evento autonómico ~1-3 s)
WOI_POST = WIN_OF_INTEREST  # (0,1) post-onset clásico
DOMINANT = "sub-33"       # sujeto que domina el efecto tónico FOOOF


def _woi_mean(times: np.ndarray, tc: np.ndarray, woi: tuple[float, float]) -> float:
    m = (times >= woi[0]) & (times <= woi[1])
    return float(np.nanmean(tc[m]))


def _tonic_alpha_fooof() -> dict[str, float]:
    """Δalfa periódico (real-silent) tónico por sujeto, promediado sobre canales parieto-occipitales.

    Lee fooof_scr_contrasts.csv (level==channel). Devuelve {subject: d_periodic_alpha_PO}.
    """
    df = pd.read_csv(FOOOF_CONTRASTS)
    ch = df[(df["level"] == "channel") & (df["key"].isin(PARIETOOCCIPITAL))]
    out: dict[str, float] = {}
    for sub, g in ch.groupby("subject"):
        out[str(sub)] = float(g["d_periodic_alpha"].mean())
    return out


def main() -> None:
    print("=" * 78)
    print(f"alpha_desync_deepdive :: tónico (FOOOF) vs event-related (WOI)  -> {OUT_DIR}")
    print(f"  ROI parieto-occipital: {PARIETOOCCIPITAL}")
    print(f"  WOI pre={WOI_PRE}  post={WOI_POST}")
    print("=" * 78, flush=True)

    tonic = _tonic_alpha_fooof()
    rows: list[dict] = []
    tc_diff: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # sub -> (times, real-silent tc)

    for sub in COHORT:
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        times, real_pe = alpha_roi_per_epoch(real_ep, PARIETOOCCIPITAL)
        _, silent_pe = alpha_roi_per_epoch(silent_ep, PARIETOOCCIPITAL)
        real_tc, silent_tc = real_pe.mean(axis=0), silent_pe.mean(axis=0)
        diff_tc = real_tc - silent_tc
        tc_diff[sub] = (times, diff_tc)

        er_pre = _woi_mean(times, real_tc, WOI_PRE) - _woi_mean(times, silent_tc, WOI_PRE)
        er_post = _woi_mean(times, real_tc, WOI_POST) - _woi_mean(times, silent_tc, WOI_POST)
        row = dict(
            subject=sub, n_real=len(real_ep), n_silent=len(silent_ep),
            tonic_fooof_d_alpha=round(tonic.get(sub, np.nan), 4),     # whole-epoch, no baseline
            er_pre_woi=round(er_pre, 3), er_post_woi=round(er_post, 3),  # baseline-corrected
        )
        rows.append(row)
        print(f"  {sub}: tónico(FOOOF)={row['tonic_fooof_d_alpha']:+.3f}  "
              f"ER pre={er_pre:+.2f}%  ER post={er_post:+.2f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_desync_deepdive_summary.csv", index=False)

    # --- robustez: GA tónico con y sin el sujeto dominante ---
    tonic_all = df["tonic_fooof_d_alpha"].mean()
    tonic_drop = df.loc[df["subject"] != DOMINANT, "tonic_fooof_d_alpha"].mean()
    er_post_all = df["er_post_woi"].mean()
    n_er_neg = int((df["er_post_woi"] < 0).sum())  # ERD direction post-onset

    # --- GA time course (real-silent), all-6 vs drop-dominant ---
    common_t = next(iter(tc_diff.values()))[0]
    stack = np.array([tc for (_, tc) in tc_diff.values()])
    subs_order = list(tc_diff.keys())
    ga_all = np.nanmean(stack, axis=0)
    keep = [i for i, s in enumerate(subs_order) if s != DOMINANT]
    ga_drop = np.nanmean(stack[keep], axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s, (t, tc) in tc_diff.items():
        ax.plot(t, tc, color=SUBJ_COLORS.get(s, "0.6"), lw=0.9, alpha=0.55, label=s)
    ax.plot(common_t, ga_all, color="k", lw=2.4, label="GA (N=6)")
    ax.plot(common_t, ga_drop, color="C1", lw=2.0, ls="--", label=f"GA sin {DOMINANT} (N=5)")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(*WOI_POST, color="gold", alpha=0.12)
    ax.axvspan(*WOI_PRE, color="cyan", alpha=0.08)
    ax.set_xlabel("time from SCR onset (s)")
    ax.set_ylabel("alpha (8-13 Hz) % change, real − silent")
    ax.set_title("Alfa parieto-occipital: real − silent (negativo = más ERD en real). "
                 "Baseline (-5,-4.5)s. Si fuera ERD time-locked habría un valle cerca del onset.",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_desync_GA_timecourse.png", dpi=130)
    plt.close(fig)

    # --- veredicto ---
    # event-related casi null (0/6 sig en alpha_hypothesis; acá miramos magnitud/consistencia)
    er_locked = (n_er_neg >= 4) and (abs(er_post_all) > 0.5)
    tonic_robust = (tonic_all < -0.05) and (tonic_drop < -0.05)
    if er_locked:
        verdict = "ERD-TIME-LOCKED-PLAUSIBLE"
    elif tonic_robust:
        verdict = "TONIC-STATE-OFFSET (no ERD time-locked; sobrevive dropear dominante)"
    else:
        verdict = "TONIC-DRIVEN-BY-DOMINANT (efecto tónico se cae sin el sujeto dominante)"

    decision = dict(
        tonic_fooof_GA=round(float(tonic_all), 4),
        tonic_fooof_GA_drop_dominant=round(float(tonic_drop), 4),
        dominant_subject=DOMINANT,
        er_post_woi_GA=round(float(er_post_all), 3),
        n_er_post_negative=n_er_neg,
        verdict=verdict,
    )
    with open(TBL_DIR / "alpha_desync_decision.json", "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2)

    print("\n" + df.to_string(index=False), flush=True)
    print("\nDECISION:", json.dumps(decision, indent=2), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
