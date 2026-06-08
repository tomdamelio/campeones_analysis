"""2.6-B — Contraste delta en ventana PRE-onset (especificidad temporal / anticipación).

Frente-DELTA. La topografía (2.6-A) no pudo decidir si el delta es cortical (difusa, leve edge).
Este es el discriminador que queda: si el delta↑ está presente en una ventana que TERMINA antes
del onset del SCR, no puede ser una respuesta electrodérmica posterior del scalp -> es
anticipatorio-neural (Branković: EEG lidera el SCR 1-4 s, brain->body) o un confound que también
precede al SCR.

Ventana pre-onset = [-5, -1] s (guard band 1 s; larga porque delta necesita ≥~2-4 s para estimarse).
GATE DE SIMETRÍA (F8/F9): los controles silent ya son EDA-silenciosos en toda la ventana, pero las
épocas SCR (REQUIRE_CLEAN_SCR=False) no -> se gatean a pre-onset limpio con silent_window_is_clean
(pre_s=5, post_s=-1) para que el contraste sea simétrico. Se reporta la retención.

Métrica: delta periódico channel-mean (1/f en 1-30, sin gamma), igual que 2.6-A. EEG preprocesado
(runs_for -> desc-preproc), 29 ch -> hereda Track B (re-correr para el paper).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.delta_preonset_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, DROP_CHANNELS, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
    silent_window_is_clean,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import attach_montage_and_drop_no_pos, compute_psd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_6_delta_robustness"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

PRE_TMIN, PRE_TMAX = -5.0, -1.0   # ventana pre-onset (termina 1 s antes del onset)
RANGE = "1-30"


def feat_delta(psd, freqs, band=(1, 4)):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES[RANGE])
    lo, hi = band
    m = (f >= lo) & (f < hi)
    return np.clip(resid[:, :, m], 0, None).mean(axis=2).mean(axis=1)   # channel-mean por época


def _subject_epochs(sub):
    """Épocas pre-onset [-5,-1] de SCR (gated a pre-onset limpio) y silent. 29 ch."""
    cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
    runs_in_npz = [str(r) for r in cont["runs"]]
    real_eps, sil_eps = [], []
    n_total, n_gated = 0, 0
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
        raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
        raw.filter(1.0, 40.0, verbose="ERROR"); raw.resample(250.0, verbose="ERROR")
        raw.drop_channels([c for c in DROP_CHANNELS if c in raw.ch_names])
        dur = float(raw.times[-1])
        eda = np.asarray(cont[f"{label}__eda_phasic"], float)
        ons = detect_scr_onsets_s(eda, EDA_FS)
        onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
        sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                     fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
        # GATE de simetría: SCR con pre-onset [t-5, t-1] EDA-silencioso
        gated = np.array([t for t in onsets
                          if silent_window_is_clean(t, eda, EDA_FS, pre_s=5.0, post_s=-1.0)], dtype=float)
        n_total += len(onsets); n_gated += len(gated)
        er = epoch_one_run(raw, gated, code=1, tmin=PRE_TMIN, tmax=PRE_TMAX, baseline=None)
        si = epoch_one_run(raw, sil, code=2, tmin=PRE_TMIN, tmax=PRE_TMAX, baseline=None)
        if er is not None: real_eps.append(er)
        if si is not None: sil_eps.append(si)
    if not real_eps or not sil_eps:
        return None
    real = mne.concatenate_epochs(real_eps, verbose="ERROR")
    sil = mne.concatenate_epochs(sil_eps, verbose="ERROR")
    return real, sil, n_total, n_gated


def main():
    print("=" * 78)
    print("delta_preonset_scr :: 2.6-B delta en ventana PRE-onset [-5,-1] (anticipación)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    _erp.RNG = np.random.default_rng(UNIFORM_SEED)   # mismos silents que el cache
    rows = []
    for sub in COHORT:
        out = _subject_epochs(sub)
        if out is None:
            print(f"  {sub}: sin épocas", flush=True); continue
        real, sil, n_total, n_gated = out
        psd_r, freqs, ch = compute_psd(real)
        psd_s, _, _ = compute_psd(sil)
        d14 = float(feat_delta(psd_r, freqs, (1, 4)).mean() - feat_delta(psd_s, freqs, (1, 4)).mean())
        d24 = float(feat_delta(psd_r, freqs, (2, 4)).mean() - feat_delta(psd_s, freqs, (2, 4)).mean())
        ret = n_gated / n_total if n_total else np.nan
        rows.append(dict(subject=sub, n_scr_total=n_total, n_scr_gated=n_gated,
                         retention=round(ret, 2), n_silent=len(sil),
                         d_delta_1_4_pre=round(d14, 4), d_delta_2_4_pre=round(d24, 4)))
        print(f"  {sub}: SCR {n_gated}/{n_total} (ret {ret:.0%}) silent={len(sil)}  "
              f"delta_pre 1-4={d14:+.4f}  2-4={d24:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "delta_preonset.csv", index=False)

    d14 = df["d_delta_1_4_pre"].to_numpy(); d24 = df["d_delta_2_4_pre"].to_numpy()
    print("\n=== VEREDICTO pre-onset (¿delta↑ presente ANTES del onset?) ===")
    print(f"  delta 1-4 pre-onset: {int((d14>0).sum())}/6 positivos (media {d14.mean():+.4f})")
    print(f"  delta 2-4 pre-onset: {int((d24>0).sum())}/6 positivos (media {d24.mean():+.4f})")
    print(f"  retención SCR media: {df['retention'].mean():.0%}")
    print("  6/6 + = anticipatorio-neural (brain->body, Branković) · ~0/mixto = respuesta posterior")

    _plot(df)
    print(f"\nCSV/figura -> {OUT_DIR}\n[2.6-B] done", flush=True)


def _plot(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["d_delta_1_4_pre"], 0.4, label="delta 1-4 (pre-onset)", color="C0")
    ax.bar(x + 0.2, df["d_delta_2_4_pre"], 0.4, label="delta 2-4 (pre-onset, ≤2Hz cut)", color="C1")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["subject"], fontsize=9)
    ax.set_ylabel("delta periódico SCR − silent, ventana [-5,-1] s")
    for xi, r in enumerate(df["retention"]):
        ax.annotate(f"ret {r:.0%}", (xi, ax.get_ylim()[1] * 0.9), ha="center", fontsize=7, color="0.4")
    ax.set_title("2.6-B Delta en ventana PRE-onset [-5,-1] s (gate de simetría)\n"
                 ">0 = delta↑ presente ANTES del SCR = anticipatorio-neural (brain→body)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "delta_preonset.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
