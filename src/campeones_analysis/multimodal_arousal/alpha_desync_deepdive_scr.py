"""Deep-dive del alfa-desync: ¿ERD cortical time-locked o un offset TÓNICO de estado?

Tarea A del 2026-06-03 (tras cerrar el frente HF = EMG residual). El alfa-desync quedó como
el único candidato a señal cortical, pero hay DOS medidas que aparentemente discrepan:
  - TÓNICO (FOOOF, whole-epoch, sin baseline): Δalfa periódico negativo (desync).
  - EVENT-RELATED (alpha_hypothesis_scr, WOI post-onset, % change vs baseline (-5,-4.5)): ~null.

Preguntas (incl. follow-up del usuario 2026-06-03):
  1) ¿El alfa baja LOCKEADO al SCR (ERD) o es un nivel TÓNICO más bajo en toda la época?
     Sutileza: el event-related está baseline-corregido a (-5,-4.5) -> un offset tónico presente
     YA en el baseline es invisible al event-related y solo lo ve el FOOOF (absoluto). Por eso
     se reportan ambas medidas + una ventana ANCHA event-related (-4,+3) que captaría un desync
     SOSTENIDO que se desarrolla durante la época (distinto de un ERD agudo y de un offset puro).
  2) ¿ROI parieto-occipital o GLOBAL? -> se computa todo para AMBOS ROIs + topomap del alfa tónico.
  3) ¿Sobrevive dropear al sujeto dominante (sub-33)? -> robustez para cada medida/ROI.

Reusa build_subject_epochs + constantes de alpha_hypothesis (ALPHA_FREQS/N_CYCLES/TFR_DECIM/
BASELINE/BASELINE_MODE/PARIETOOCCIPITAL); computa el Morlet por época UNA vez por condición y
selecciona ambos ROIs. Sin reprocesar; mismas épocas no-AR que generaron el FOOOF.

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
from mne.time_frequency import tfr_morlet

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.alpha_hypothesis_scr import (
    ALPHA_FREQS,
    BASELINE,
    BASELINE_MODE,
    N_CYCLES,
    PARIETOOCCIPITAL,
    TFR_DECIM,
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
WOI_POST = (0.0, 1.0)     # post-onset clásico
WOI_FULL = (-4.0, 3.0)    # ventana ANCHA (post-baseline): capta un desync sostenido
DOMINANT = "sub-33"
ROI_NAMES = ("parieto-occipital", "global")


def _alpha_tc_baselined(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-epoch alpha (8-13 Hz) Morlet TFR, baseline-corrected in dB, averaged over freqs.

    Usa LOGRATIO -> dB (10*log10(power/baseline)) en vez de 'percent'. Motivo (2026-06-03,
    catch del usuario): MNE 'percent' = (d-m)/m SIN ×100 = fracción, y explota cuando el
    baseline alfa es ~0 (blow-ups de +700%). dB/logratio es la métrica ERD/ERS estándar y
    robusta a esos blow-ups (la usa tfr_psd_scr). Devuelve (times, dB[n_epochs, n_ch, n_times]).
    """
    tfr = tfr_morlet(epochs, freqs=ALPHA_FREQS, n_cycles=N_CYCLES, use_fft=True,
                     return_itc=False, decim=TFR_DECIM, n_jobs=1, average=False, verbose="ERROR")
    tfr_b = tfr.copy().apply_baseline(BASELINE, mode="logratio")  # log10(power/baseline)
    data = 10.0 * tfr_b.data.mean(axis=2)  # dB, avg over alpha freqs -> (n_epochs, n_ch, n_times)
    return tfr_b.times, data, list(tfr_b.ch_names)


def _roi_tc(times, data, ch_names, roi_chs):
    """Mean over ROI channels and epochs -> time course (n_times,)."""
    idx = [ch_names.index(c) for c in roi_chs if c in ch_names]
    return data[:, idx].mean(axis=(0, 1))


def _woi(times, tc, woi):
    m = (times >= woi[0]) & (times <= woi[1])
    return float(np.nanmean(tc[m]))


def _tonic_fooof(channels: list[str] | None) -> dict[str, float]:
    """Δalfa periódico tónico (real-silent) por sujeto, promediado sobre `channels`
    (None = todos los canales = global). Lee fooof_scr_contrasts.csv (level==channel)."""
    df = pd.read_csv(FOOOF_CONTRASTS)
    ch = df[df["level"] == "channel"]
    if channels is not None:
        ch = ch[ch["key"].isin(channels)]
    return {str(s): float(g["d_periodic_alpha"].mean()) for s, g in ch.groupby("subject")}


def _tonic_topo_GA() -> tuple[list[str], np.ndarray]:
    """Per-channel GA Δalfa periódico tónico (subject==GA rows) for the topomap."""
    df = pd.read_csv(FOOOF_CONTRASTS)
    ga = df[(df["subject"] == "GA") & (df["level"] == "channel")]
    return ga["key"].astype(str).tolist(), ga["d_periodic_alpha"].to_numpy(float)


def main() -> None:
    print("=" * 78)
    print(f"alpha_desync_deepdive :: tónico vs event-related, PO vs global  -> {OUT_DIR}")
    print(f"  WOI pre={WOI_PRE} post={WOI_POST} full={WOI_FULL}  dominante={DOMINANT}")
    print("=" * 78, flush=True)

    tonic_po = _tonic_fooof(PARIETOOCCIPITAL)
    tonic_gl = _tonic_fooof(None)
    rows: list[dict] = []
    # tc_diff[roi][sub] = (times, real-silent tc)
    tc_diff: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {r: {} for r in ROI_NAMES}
    info_ref: mne.Info | None = None

    for sub in COHORT:
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        if info_ref is None:
            info_ref = real_ep.info.copy()
        t, dr, ch = _alpha_tc_baselined(real_ep)
        _, ds, _ = _alpha_tc_baselined(silent_ep)
        all_ch = ch
        roi_map = {"parieto-occipital": PARIETOOCCIPITAL, "global": all_ch}

        row = dict(subject=sub, n_real=len(real_ep), n_silent=len(silent_ep))
        for rname, rchs in roi_map.items():
            real_tc, silent_tc = _roi_tc(t, dr, ch, rchs), _roi_tc(t, ds, ch, rchs)
            diff = real_tc - silent_tc
            tc_diff[rname][sub] = (t, diff)
            tag = "po" if rname == "parieto-occipital" else "gl"
            row[f"er_pre_{tag}"] = round(_woi(t, real_tc, WOI_PRE) - _woi(t, silent_tc, WOI_PRE), 3)
            row[f"er_post_{tag}"] = round(_woi(t, real_tc, WOI_POST) - _woi(t, silent_tc, WOI_POST), 3)
            row[f"er_full_{tag}"] = round(_woi(t, real_tc, WOI_FULL) - _woi(t, silent_tc, WOI_FULL), 3)
        row["tonic_po"] = round(tonic_po.get(sub, np.nan), 4)
        row["tonic_gl"] = round(tonic_gl.get(sub, np.nan), 4)
        rows.append(row)
        print(f"  {sub}: tónico PO={row['tonic_po']:+.3f} GL={row['tonic_gl']:+.3f} | "
              f"ER full PO={row['er_full_po']:+.2f} GL={row['er_full_gl']:+.2f} | "
              f"ER post PO={row['er_post_po']:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_desync_deepdive_summary.csv", index=False)

    def _ga(col):
        return float(df[col].mean()), float(df.loc[df["subject"] != DOMINANT, col].mean())

    decision = {"dominant_subject": DOMINANT}
    for col in ("tonic_po", "tonic_gl", "er_full_po", "er_full_gl", "er_post_po", "er_post_gl",
                "er_pre_po", "er_pre_gl"):
        a, d = _ga(col)
        decision[col] = {"GA": round(a, 4), "GA_drop_dominant": round(d, 4)}

    # robustez del veredicto: ¿alguna medida da un desync negativo que SOBREVIVA dropear al
    # dominante con magnitud NO-trivial? OJO con las unidades: el tónico (FOOOF) está en
    # log-potencia (umbral -0.05 razonable); el event-related está en % de cambio, donde un
    # ERD real es de VARIOS % -> un -0.1% es ruido, no señal. Umbrales separados.
    TONIC_THR = -0.05   # FOOOF log-power units
    ERD_THR = -0.5      # dB; ERD alfa modesto ~-0.5 dB (≈ -11%), fuerte ~-1.5 dB (≈ -30%)
    tonic_robust = any(decision[c]["GA"] < TONIC_THR and decision[c]["GA_drop_dominant"] < TONIC_THR
                       for c in ("tonic_po", "tonic_gl"))
    erd_robust = any(decision[c]["GA"] < ERD_THR and decision[c]["GA_drop_dominant"] < ERD_THR
                     for c in ("er_full_po", "er_full_gl", "er_post_po", "er_post_gl",
                               "er_pre_po", "er_pre_gl"))
    decision["verdict"] = ("ALPHA-EFFECT-SURVIVES" if (tonic_robust or erd_robust)
                           else "NO-ROBUST-ALPHA (tónico depende de sub-33; event-related ~ruido en % cambio)")
    with open(TBL_DIR / "alpha_desync_decision.json", "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2)

    # --- figura: GA time-course por ROI (all-6 vs drop-dominant) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, rname in zip(axes, ROI_NAMES):
        d = tc_diff[rname]
        common_t = next(iter(d.values()))[0]
        stack = np.array([tc for (_, tc) in d.values()])
        order = list(d.keys())
        for s in order:
            ax.plot(d[s][0], d[s][1], color=SUBJ_COLORS.get(s, "0.6"), lw=0.8, alpha=0.5, label=s)
        ax.plot(common_t, np.nanmean(stack, axis=0), color="k", lw=2.4, label="GA (N=6)")
        keep = [i for i, s in enumerate(order) if s != DOMINANT]
        ax.plot(common_t, np.nanmean(stack[keep], axis=0), color="C1", lw=2.0, ls="--",
                label=f"GA sin {DOMINANT}")
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(*WOI_POST, color="gold", alpha=0.12)
        ax.set_xlabel("time from SCR onset (s)"); ax.set_title(f"ROI: {rname}", fontsize=10)
    axes[0].set_ylabel("alpha dB (logratio), real − silent")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Alfa real − silent (baseline-corregido). Plano sin valle en el onset = no ERD time-locked.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "alpha_desync_GA_timecourse.png", dpi=130)
    plt.close(fig)

    # --- topomap del alfa TÓNICO (FOOOF GA per-channel) ---
    if info_ref is not None:
        ga_ch, ga_val = _tonic_topo_GA()
        order = [info_ref.ch_names.index(c) for c in ga_ch if c in info_ref.ch_names]
        vals = np.array([ga_val[ga_ch.index(info_ref.ch_names[i])] for i in order])
        sub_info = mne.pick_info(info_ref, order)
        figt, axt = plt.subplots(figsize=(5, 4.5))
        vmax = float(np.nanmax(np.abs(vals))) if np.any(np.isfinite(vals)) else 1.0
        im, _ = mne.viz.plot_topomap(vals, sub_info, axes=axt, show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", contours=4, sensors=True)
        figt.colorbar(im, ax=axt, shrink=0.7, label="Δalfa periódico (real-silent)")
        axt.set_title("Alfa tónico (FOOOF) GA, real-silent\n(azul = menos alfa en real)", fontsize=9)
        figt.tight_layout()
        figt.savefig(FIG_DIR / "alpha_tonic_topomap_GA.png", dpi=130)
        plt.close(figt)

    print("\n" + df.to_string(index=False), flush=True)
    print("\nDECISION:", json.dumps(decision, indent=2), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
