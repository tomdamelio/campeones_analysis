"""Alfa-desync con baseline EDGE-SAFE + potencia absoluta: ¿el null del event-related era
real o un artefacto de un baseline corto/contaminado de borde? (follow-up usuario 2026-06-03).

Problema detectado: el deep-dive usaba BASELINE=(-5,-4.5) = 0.5 s PEGADO al borde de la época
(-5,+3). El wavelet Morlet a 8 Hz dura ~0.5 s, así que la estimación de alfa en ese baseline
está contaminada de borde (no hay datos antes de -5) -> baseline ruidoso divide cada punto del
ERD y puede ENTERRAR un efecto. Estándar (Pfurtscheller & Lopes da Silva 1999, Clin Neurophysiol
DOI 10.1016/s1388-2457(99)00141-8): referencia ~1 s, lejos del borde y del evento; épocas con
PADDING y recorte de bordes; si no hay rest limpio -> contraste de condiciones o potencia absoluta.

Este script re-epoquea con PADDING (-7,+4) para TFR limpio de borde en -5..+3, y compara TRES
lecturas del alfa parieto-occipital (real - silent, en dB):
  (1) baseline VIEJO (-5,-4.5)  -- la versión actual (0.5 s, borde)
  (2) baseline EDGE-SAFE (-4.5,-3.5)  -- 1 s, 2.5 s de datos antes = sin contaminación de borde
  (3) ABSOLUTO sin baseline: 10*log10(alfa_real / alfa_silent) por tiempo -- esquiva el baseline,
      muestra tónico+fásico (contraste de condiciones, el camino cuando no hay rest limpio).
GA + robustez drop-sub-33 + WOIs (pre/post/ancha).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.alpha_erd_baseline_scr
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
    N_CYCLES,
    PARIETOOCCIPITAL,
)
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

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "qa_artifact_vs_signal" / "alpha_desync"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

PAD = (-7.0, 4.0)          # padded epoch for edge-clean TFR over the display window
DISPLAY = (-5.0, 3.0)
BL_OLD = (-5.0, -4.5)      # current (short, at edge)
BL_SAFE = (-4.5, -3.5)     # 1 s, edge-safe (2.5 s of padded data before it)
WOI_PRE, WOI_POST, WOI_FULL = (-3.0, -1.0), (0.0, 1.0), (-4.0, 3.0)
DOMINANT = "sub-33"
RNG_SEED = 20260513


def _build_padded(sub: str):
    """(real, silent) epochs with PAD window. Mirrors tfr_psd_scr.build_subject_epochs but
    passes the wide tmin/tmax to epoch_one_run (baseline default is harmless for power)."""
    _erp.RNG = np.random.default_rng(RNG_SEED)
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None, None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    real_list, silent_list = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.resample(250.0, verbose="ERROR")
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(onsets[onsets < dur], eda, EDA_FS)
            silent_t = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                              fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            er = epoch_one_run(raw, onsets, code=1, tmin=PAD[0], tmax=PAD[1])
            si = epoch_one_run(raw, silent_t, code=2, tmin=PAD[0], tmax=PAD[1])
            if er is not None:
                real_list.append(er)
            if si is not None:
                silent_list.append(si)
        except Exception as e:
            print(f"  {label}: FAILED -- {e}", flush=True)
    if not real_list or not silent_list:
        return None, None
    return (mne.concatenate_epochs(real_list, verbose="ERROR"),
            mne.concatenate_epochs(silent_list, verbose="ERROR"))


def _alpha_power(epochs):
    """Per-epoch alpha Morlet power (NO baseline). Returns (times, power[n_ep,n_ch,n_t], ch)."""
    tfr = tfr_morlet(epochs, freqs=ALPHA_FREQS, n_cycles=N_CYCLES, use_fft=True,
                     return_itc=False, decim=4, n_jobs=1, average=False, verbose="ERROR")
    return tfr.times, tfr.data.mean(axis=2), list(tfr.ch_names)  # avg over alpha freqs


def _roi(power, ch, roi):
    idx = [ch.index(c) for c in roi if c in ch]
    return power[:, idx].mean(axis=1)  # (n_ep, n_t)


def _crop(times, arr):
    m = (times >= DISPLAY[0]) & (times <= DISPLAY[1])
    return times[m], arr[..., m]


def _bl_db(times, roi_pow, bl):
    """dB ERD vs baseline window bl: 10*log10(mean_ep(power) / mean_ep,bl(power))."""
    m = (times >= bl[0]) & (times <= bl[1])
    base = roi_pow[:, m].mean()  # scalar baseline (mean over epochs & baseline time)
    return 10.0 * np.log10(roi_pow.mean(axis=0) / base + 1e-30)


def _woi(times, tc, w):
    m = (times >= w[0]) & (times <= w[1])
    return float(np.nanmean(tc[m]))


def main():
    print("=" * 78)
    print(f"alpha_erd_baseline :: PAD={PAD} BL_OLD={BL_OLD} BL_SAFE={BL_SAFE}  -> {OUT_DIR}")
    print("=" * 78, flush=True)

    rows = []
    curves = {"old": {}, "safe": {}, "abs": {}}  # mode -> sub -> (times, real-silent dB)
    for sub in COHORT:
        real_ep, silent_ep = _build_padded(sub)
        if real_ep is None:
            print(f"  {sub}: no epochs", flush=True)
            continue
        t, pr, ch = _alpha_power(real_ep)
        _, ps, _ = _alpha_power(silent_ep)
        rr, rs = _roi(pr, ch, PARIETOOCCIPITAL), _roi(ps, ch, PARIETOOCCIPITAL)

        # crop to display AFTER baseline computed on padded data
        tc_old = _bl_db(t, rr, BL_OLD) - _bl_db(t, rs, BL_OLD)
        tc_safe = _bl_db(t, rr, BL_SAFE) - _bl_db(t, rs, BL_SAFE)
        tc_abs = 10.0 * np.log10(rr.mean(axis=0) / (rs.mean(axis=0) + 1e-30) + 1e-30)  # real/silent abs
        tcd, tc_old = _crop(t, tc_old)
        _, tc_safe = _crop(t, tc_safe)
        _, tc_abs = _crop(t, tc_abs)
        curves["old"][sub] = (tcd, tc_old)
        curves["safe"][sub] = (tcd, tc_safe)
        curves["abs"][sub] = (tcd, tc_abs)

        row = dict(subject=sub, n_real=len(real_ep), n_silent=len(silent_ep),
                   safe_post=round(_woi(tcd, tc_safe, WOI_POST), 3),
                   safe_full=round(_woi(tcd, tc_safe, WOI_FULL), 3),
                   old_full=round(_woi(tcd, tc_old, WOI_FULL), 3),
                   abs_full=round(_woi(tcd, tc_abs, WOI_FULL), 3))
        rows.append(row)
        print(f"  {sub}: safe_post={row['safe_post']:+.2f}dB safe_full={row['safe_full']:+.2f} "
              f"old_full={row['old_full']:+.2f} abs_full={row['abs_full']:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "alpha_erd_baseline_summary.csv", index=False)

    def _ga(col):
        return round(float(df[col].mean()), 3), round(float(df.loc[df.subject != DOMINANT, col].mean()), 3)

    dec = {"baseline_safe": list(BL_SAFE), "baseline_old": list(BL_OLD)}
    for col in ("safe_post", "safe_full", "old_full", "abs_full"):
        a, d = _ga(col)
        dec[col] = {"GA": a, "GA_drop_dominant": d}
    # ERD real => negativo y de magnitud no-trivial (>~0.5 dB) que sobreviva drop-dominant
    dec["verdict"] = ("ALPHA-ERD-WITH-SAFE-BASELINE"
                      if dec["safe_full"]["GA"] < -0.5 and dec["safe_full"]["GA_drop_dominant"] < -0.5
                      else "NO-ERD (edge-safe baseline no cambia el null; alfa absoluto idem)")
    with open(TBL_DIR / "alpha_erd_baseline_decision.json", "w", encoding="utf-8") as fh:
        json.dump(dec, fh, indent=2)

    # figure: 3 panels (old / safe / absolute), GA + drop-dominant
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = {"old": f"ERD baseline VIEJO {BL_OLD} (borde)",
              "safe": f"ERD baseline EDGE-SAFE {BL_SAFE}",
              "abs": "ABSOLUTO 10*log10(real/silent), sin baseline"}
    for ax, mode in zip(axes, ("old", "safe", "abs")):
        d = curves[mode]
        ct = next(iter(d.values()))[0]
        stack = np.array([c for (_, c) in d.values()])
        order = list(d.keys())
        for s in order:
            ax.plot(d[s][0], d[s][1], color=SUBJ_COLORS.get(s, "0.6"), lw=0.8, alpha=0.5, label=s)
        ax.plot(ct, np.nanmean(stack, axis=0), color="k", lw=2.4, label="GA")
        keep = [i for i, s in enumerate(order) if s != DOMINANT]
        ax.plot(ct, np.nanmean(stack[keep], axis=0), color="C1", lw=2.0, ls="--", label=f"GA sin {DOMINANT}")
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(*WOI_POST, color="gold", alpha=0.12)
        ax.set_xlabel("time from SCR onset (s)"); ax.set_title(titles[mode], fontsize=9)
    axes[0].set_ylabel("alpha dB, real − silent"); axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Alfa PO real−silent con baseline edge-safe + absoluto. Negativo = ERD.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "alpha_erd_baseline_compare.png", dpi=130)
    plt.close(fig)

    print("\n" + df.to_string(index=False), flush=True)
    print("\nDECISION:", json.dumps(dec, indent=2), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
