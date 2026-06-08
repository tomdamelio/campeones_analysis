"""2.3 extracción — covariados oculares por época desde R_EYE/L_EYE (referencia original).

Complementa el QA (eog_qa_scr.py, precondición) y los covariados de 2.1 (radial-EOG + Fp1/Fp2).
Acá: slow_EOG (componente lento, dipolo/movimiento ocular -> Q-delta) y gamma_EOG (microsacada/SP
-> Q-gamma) directamente de los canales EOG, en su referencia ORIGINAL (bipolar, pre-ICA) ->
INMUNE a Track B. Triangula con el radial-EOG de 2.1 (varios proxies imperfectos convergentes).

QA (eog_qa_scr): R_EYE (HEOG) vivo 6/6; L_EYE (VEOG) funcional/tracks-blinks 4/6 (sub-19/24/31/33),
ALIVE-NO-BLINKS sub-23/27 (fallback Fp1/Fp2 ya cubierto por 2.1). Acá se extrae igual de ambos.

Covariados por época (envelope, F11 continuo):
  - heog_gamma_30_40 : R_EYE 30-40 Hz (microsacada/SP, banda del crux)  [Q-gamma]
  - veog_gamma_30_40 : L_EYE 30-40 Hz                                    [Q-gamma]
  - heog_slow_0p5_8  : R_EYE 0.5-8 Hz (sacada/dipolo)
  - veog_slow_0p5_8  : L_EYE 0.5-8 Hz (blink/dipolo)                     [Q-delta]
  - veog_slow_2hz_pre: L_EYE 0.5-2 Hz, ventana PRE-onset (drift/sweat)  [Q-delta, scalp-sweat]

Alineamiento 1:1 al cache (mismo RNG/orden/boundary-drop que 2.1/2.2, validado por tnorm).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eog_covariate_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy.signal import hilbert

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_panel import UNIFORM_SEED, load_cache
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    TMAX,
    TMIN,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import attach_montage_and_drop_no_pos

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_3_eog_covariate"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

COV_COLS = ["heog_gamma_30_40", "veog_gamma_30_40", "heog_slow_0p5_8",
            "veog_slow_0p5_8", "veog_slow_2hz_pre"]


def _env(x, sf, lo, hi):
    hi = min(hi, sf / 2.0 - 1.0)
    return np.abs(hilbert(filter_data(x[None, :].astype(float), sf, lo, hi, verbose="ERROR")[0]))


def _wmean(env, sf, t0, lo_s, hi_s):
    a, b = int((t0 + lo_s) * sf), int((t0 + hi_s) * sf)
    a, b = max(0, a), min(len(env), b)
    return float(np.mean(env[a:b])) if b > a else np.nan


def _run_signals(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    if not {"R_EYE", "L_EYE"}.issubset(set(raw.ch_names)):
        return None
    r = raw.get_data(picks=["R_EYE"])[0]
    l = raw.get_data(picks=["L_EYE"])[0]
    return dict(
        sf=sf,
        heog_gamma=_env(r, sf, 30.0, 40.0), veog_gamma=_env(l, sf, 30.0, 40.0),
        heog_slow=_env(r, sf, 0.5, 8.0), veog_slow=_env(l, sf, 0.5, 8.0),
        veog_2hz=_env(l, sf, 0.5, 2.0),
    )


def _rows(sig, onsets, sub, label, cond):
    sf = sig["sf"]
    out = []
    for t0 in onsets:
        out.append(dict(
            subject=sub, run=label, condition=cond, tnorm=np.nan,
            heog_gamma_30_40=_wmean(sig["heog_gamma"], sf, t0, TMIN, TMAX),
            veog_gamma_30_40=_wmean(sig["veog_gamma"], sf, t0, TMIN, TMAX),
            heog_slow_0p5_8=_wmean(sig["heog_slow"], sf, t0, TMIN, TMAX),
            veog_slow_0p5_8=_wmean(sig["veog_slow"], sf, t0, TMIN, TMAX),
            veog_slow_2hz_pre=_wmean(sig["veog_2hz"], sf, t0, TMIN, 0.0),
        ))
    return out


def main() -> None:
    print("=" * 78)
    print("eog_covariate_scr :: 2.3 extracción — slow_EOG + gamma_EOG (R_EYE/L_EYE) por época")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    cache, _, _ = load_cache("uniform")
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)
    all_rows, align = [], []

    for sub in COHORT:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_rows, sil_rows, real_tn, sil_tn = [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw_eeg = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw_eeg.pick("eeg"); attach_montage_and_drop_no_pos(raw_eeg)
            raw_eeg.filter(1.0, 40.0, verbose="ERROR"); raw_eeg.resample(250.0, verbose="ERROR")
            dur = float(raw_eeg.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            real_kept = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sil_kept = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]
            sig = _run_signals(vhdr)
            if sig is None:
                print(f"  {sub} {label}: faltan R_EYE/L_EYE -> skip", flush=True)
                continue
            real_rows += _rows(sig, real_kept, sub, label, 1)
            sil_rows += _rows(sig, sil_kept, sub, label, 0)
            real_tn += list(real_kept / dur); sil_tn += list(sil_kept / dur)
            print(f"  {sub} {label}: SCR={len(real_kept)} no-SCR={len(sil_kept)} (sf={sig['sf']:.0f})", flush=True)

        for r, tn in zip(real_rows, real_tn): r["tnorm"] = tn
        for r, tn in zip(sil_rows, sil_tn): r["tnorm"] = tn
        all_rows += real_rows + sil_rows
        my_tn = np.array(real_tn + sil_tn)
        if sub in cache:
            ctn = cache[sub][2]; ok = len(my_tn) == len(ctn)
            align.append(dict(subject=sub, n_mine=len(my_tn), n_cache=len(ctn), match=ok,
                              max_dev=round(float(np.max(np.abs(my_tn - ctn))), 6) if ok else np.nan))

    df = pd.DataFrame(all_rows)
    df.to_csv(TBL_DIR / "eog_covariate.csv", index=False)
    pd.DataFrame(align).to_csv(TBL_DIR / "alignment_check.csv", index=False)

    summ = []
    for col in COV_COLS:
        rels, npos, n = [], 0, 0
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna().to_numpy()
            b = sd[sd.condition == 0][col].dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            ma, mb = np.median(a), np.median(b)
            rels.append((ma - mb) / mb if abs(mb) > 1e-12 else np.nan); npos += int(ma > mb); n += 1
        summ.append(dict(covariate=col, n_subj=n,
                         mean_rel_pct=round(100 * np.nanmean(rels), 1) if rels else np.nan,
                         n_pos=f"{npos}/{n}"))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(TBL_DIR / "eog_within_subject.csv", index=False)

    print("\n=== Alineamiento ===")
    print(pd.DataFrame(align).to_string(index=False))
    print("\n=== Within-subject SCR vs no-SCR (EOG directo) ===")
    print(summ_df.to_string(index=False))

    _plot(df)
    print(f"\nCSV -> {TBL_DIR / 'eog_covariate.csv'}\n[2.3 extracción] done", flush=True)


def _plot(df):
    cols = [("heog_gamma_30_40", "HEOG gamma 30-40 (microsacada) [Q-gamma]"),
            ("veog_slow_0p5_8", "VEOG slow 0.5-8 (blink/dipolo) [Q-delta]"),
            ("veog_slow_2hz_pre", "VEOG 0.5-2 pre-onset (sweat/drift)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, cols):
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna(); b = sd[sd.condition == 0][col].dropna()
            if not len(a) or not len(b):
                continue
            ax.plot([0, 1], [b.median(), a.median()], "-o", color=SUBJ_COLORS[sub], lw=1.5, ms=5, label=sub)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no-SCR", "SCR"]); ax.set_ylabel(col)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("2.3 Covariados oculares directos (R_EYE/L_EYE): SCR vs no-SCR por sujeto", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "eog_covariate_bysubject.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
