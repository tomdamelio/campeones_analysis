"""Epocheo con control temporal: silent TEMPORALMENTE EMPAREJADO al SCR (follow-up usuario
2026-06-03). Para que real (SCR) y silent (no-SCR) tengan el mismo tiempo-en-sesión y contexto
local, y así un efecto real-silent no arrastre drift temporal del run (fatiga, drift de
electrodos, tendencia de arousal). Versión principista del "interleaving" pedido por el usuario.

Hoy `sample_silent_controls` (erp_scr) muestrea silents UNIFORME al azar en el run, mientras los
SCR se agrupan donde el estímulo/arousal los dispara -> posible desbalance temporal. Acá, para
CADA SCR real, el control silent se muestrea CERCA en el tiempo (±window_s, mismo run, EDA-
silencioso, sin solape). Conserva todos los reales (no descarta como la alternancia estricta).

API:
  sample_silent_matched(onsets, duration, phasic, fs, rng, window_s=45) -> silent times
  build_subject_epochs_matched(sub, lowpass=40, tmin, tmax, resample=250, window_s=45)
                                                                      -> (real_ep, silent_ep)
Diagnóstico (main): balance temporal real-vs-silent (uniforme vs emparejado) + retención.

Run (diagnóstico):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epoch_matched
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    EPOCH_SPAN_S,
    NPZ_DIR,
    POST_S,
    PRE_S,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
    silent_window_is_clean,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

RNG_SEED = 20260603
DEFAULT_WINDOW_S = 45.0
FIG_DIR = OUT / "qa_artifact_vs_signal" / "temporality"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def sample_silent_matched(onsets_clean, duration_s, phasic, fs, rng, *,
                          window_s=DEFAULT_WINDOW_S, pre_s=PRE_S, post_s=POST_S,
                          min_sep=EPOCH_SPAN_S, tries_per=600):
    """Para cada SCR real, un silent EDA-silencioso a ±window_s, no-solapado. Conserva orden."""
    lo, hi = pre_s, duration_s - post_s
    reals = np.asarray(onsets_clean, float)
    if hi <= lo or reals.size == 0:
        return np.array([], float)
    chosen: list[float] = []
    for t in reals:
        wlo, whi = max(lo, t - window_s), min(hi, t + window_s)
        if whi <= wlo:
            continue
        for _ in range(tries_per):
            cand = float(rng.uniform(wlo, whi))
            if not silent_window_is_clean(cand, phasic, fs, pre_s=pre_s, post_s=post_s):
                continue
            if float(np.min(np.abs(reals - cand))) < min_sep:
                continue
            if chosen and float(np.min(np.abs(np.asarray(chosen) - cand))) < min_sep:
                continue
            chosen.append(cand)
            break
    return np.asarray(chosen, float)


def _load_runs(sub):
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None, None
    cont = np.load(cont_path, allow_pickle=True)
    return cont, list(cont["runs"])


def build_subject_epochs_matched(sub, *, lowpass=40.0, tmin=TMIN, tmax=TMAX, resample=250.0,
                                 window_s=DEFAULT_WINDOW_S, reset_rng=True):
    """(real, silent) con silent temporalmente emparejado. Mirror de tfr_psd_scr.build_subject_epochs."""
    if reset_rng:
        _erp.RNG = np.random.default_rng(RNG_SEED)
    cont, runs_in_npz = _load_runs(sub)
    if cont is None:
        return None, None
    real_list, silent_list = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=lowpass, verbose="ERROR")
            raw.resample(resample, verbose="ERROR")
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = filter_clean_onsets(detect_scr_onsets_s(eda, EDA_FS)[
                detect_scr_onsets_s(eda, EDA_FS) < dur], eda, EDA_FS)
            silent_t = sample_silent_matched(onsets, dur, eda, EDA_FS, _erp.RNG, window_s=window_s)
            er = epoch_one_run(raw, onsets, code=1, tmin=tmin, tmax=tmax)
            si = epoch_one_run(raw, silent_t, code=2, tmin=tmin, tmax=tmax)
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


def _onset_fractions(sub, window_s=DEFAULT_WINDOW_S):
    """Per-run: fracción temporal (t/dur) de onsets reales, silents UNIFORMES y silents EMPAREJADOS."""
    cont, runs_in_npz = _load_runs(sub)
    if cont is None:
        return None
    _erp.RNG = np.random.default_rng(RNG_SEED)
    rng_u = np.random.default_rng(RNG_SEED)
    real_f, unif_f, match_f = [], [], []
    n_real = n_unif = n_match = 0
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        eda = np.asarray(cont[f"{label}__eda_phasic"], float)
        dur = len(eda) / EDA_FS
        onsets = detect_scr_onsets_s(eda, EDA_FS)
        onsets = filter_clean_onsets(onsets[onsets < dur], eda, EDA_FS)
        if onsets.size == 0:
            continue
        unif = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda, fs=EDA_FS,
                                      rng=rng_u, avoid_onsets_s=onsets)
        matched = sample_silent_matched(onsets, dur, eda, EDA_FS, _erp.RNG, window_s=window_s)
        real_f += list(onsets / dur); unif_f += list(unif / dur); match_f += list(matched / dur)
        n_real += len(onsets); n_unif += len(unif); n_match += len(matched)
    return dict(sub=sub, real=np.array(real_f), unif=np.array(unif_f), match=np.array(match_f),
                n_real=n_real, n_unif=n_unif, n_match=n_match)


def main():
    import pandas as pd
    print("=" * 78)
    print(f"epoch_matched :: diagnóstico temporal  window=±{DEFAULT_WINDOW_S}s  -> {FIG_DIR}")
    print("=" * 78, flush=True)
    rows = []
    diag = {}
    for sub in COHORT:
        d = _onset_fractions(sub)
        if d is None:
            continue
        diag[sub] = d
        rows.append(dict(subject=sub, n_real=d["n_real"], n_silent_unif=d["n_unif"],
                         n_silent_matched=d["n_match"],
                         retention_matched=round(d["n_match"] / max(1, d["n_real"]), 3),
                         mean_t_real=round(float(np.mean(d["real"])), 3),
                         mean_t_unif=round(float(np.mean(d["unif"])), 3),
                         mean_t_matched=round(float(np.mean(d["match"])), 3),
                         dt_real_unif=round(float(np.mean(d["real"]) - np.mean(d["unif"])), 3),
                         dt_real_matched=round(float(np.mean(d["real"]) - np.mean(d["match"])), 3)))
        print(f"  {sub}: n_real={d['n_real']} matched={d['n_match']} "
              f"(ret {rows[-1]['retention_matched']:.2f})  "
              f"mean t: real={rows[-1]['mean_t_real']} unif={rows[-1]['mean_t_unif']} "
              f"matched={rows[-1]['mean_t_matched']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "temporality_diagnostic.csv", index=False)

    # figure: real vs unif vs matched onset-time fraction distributions per subject
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, sub in zip(axes.ravel(), [r["subject"] for r in rows]):
        d = diag[sub]
        ax.hist(d["real"], bins=15, range=(0, 1), alpha=0.5, color="C3", label="real (SCR)", density=True)
        ax.hist(d["unif"], bins=15, range=(0, 1), alpha=0.4, color="0.5", label="silent uniforme", density=True)
        ax.hist(d["match"], bins=15, range=(0, 1), alpha=0.4, color="C0", label="silent emparejado", density=True)
        ax.set_title(f"{sub}  Δt(real-unif)={np.mean(d['real'])-np.mean(d['unif']):+.2f} "
                     f"Δt(real-match)={np.mean(d['real'])-np.mean(d['match']):+.2f}", fontsize=9)
        ax.set_xlabel("posición temporal en el run (t/dur)"); ax.legend(fontsize=7)
    fig.suptitle("Distribución temporal de onsets en el run: real vs silent (uniforme vs emparejado). "
                 "El emparejado debe seguir al real.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "temporality_diagnostic.png", dpi=130)
    plt.close(fig)
    print("\n" + df.to_string(index=False), flush=True)
    print(f"\n-> {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
