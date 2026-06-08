"""2.1 — Cuantificar artefactos en la data CRUDA (pre-ICA), por época, alineado por tiempo.

Frente-gamma (SP/microsacada -> Q-gamma) + frente-delta (blink lento -> Q-delta). Responde P2:
¿hay MÁS blinks/microsacadas/músculo en las épocas SCR que en las no-SCR, medido en el raw
pre-ICA? Produce los covariados por época que 2.5 va a usar.

Detección sobre RAW pre-ICA, sfreq nativo, referencia original -> INMUNE a Track B.

Covariados por época (continuo PRIMARIO, conteo secundario — F11):
  - gamma_EOG     : envelope del radial-EOG (mean(R_EYE,L_EYE)-Pz) en 30-40 Hz  [Q-gamma, crux]
  - sp_hf         : envelope del radial-EOG en 20-90 Hz (SP/microsacada banda ancha)
  - n_microsac    : conteo de SP (radial 20-90, z>3.5, refractario 50 ms)  [secundario]
  - blink_slow    : envelope de mean(Fp1,Fp2) en 1-10 Hz (blink, familia frontal — refinamiento)
  - blink_slow_pre: idem en ventana PRE-onset [TMIN,0]  [Q-delta anticipatorio]
  - blink_2hz_pre : envelope de mean(Fp1,Fp2) en 0.5-2 Hz, PRE-onset  [scalp-sweat / Q-delta, 2.6]
  - muscle_hf     : envelope de mean(EMG_EDGE) en 110-140 Hz (músculo de borde, 29 ch sin TP10)

ALINEAMIENTO (crítico): reproduce EXACTO el cache panel_psd (decoding_panel._build_one):
  - _erp.RNG reseteado con UNIFORM_SEED y consumido a lo largo de COHORT en orden (silents).
  - boundary-drop idéntico (epoch_one_run: onset+TMIN>0 & onset+TMAX<dur).
  - orden de filas: REAL (run-by-run) luego SILENT (run-by-run).
  Se VALIDA comparando el tnorm re-derivado contra cache['tnorm_uniform_<sub>'] (debe coincidir).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.artifact_raw_scr
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
from scipy.stats import mannwhitneyu

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, EMG_EDGE, REPO, SUBJ_COLORS
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

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_1_artifact_raw"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

SP_SD = 3.5
SP_REFRACT_S = 0.05
PERI_FS = 50.0                                   # grilla para el peri-SCR time-course
PERI_T = np.arange(TMIN, TMAX, 1.0 / PERI_FS)    # eje de tiempo relativo al onset
COV_COLS = ["gamma_EOG", "sp_hf", "n_microsac", "blink_slow",
            "blink_slow_pre", "blink_2hz_pre", "muscle_hf"]


def _env(x: np.ndarray, sf: float, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, sf / 2.0 - 1.0)
    y = filter_data(x[None, :].astype(float), sf, lo, hi, verbose="ERROR")[0]
    return np.abs(hilbert(y))


def _wmean(env: np.ndarray, sf: float, t0: float, lo_s: float, hi_s: float) -> float:
    a, b = int((t0 + lo_s) * sf), int((t0 + hi_s) * sf)
    a, b = max(0, a), min(len(env), b)
    return float(np.mean(env[a:b])) if b > a else np.nan


def _sp_count(zhf: np.ndarray, sf: float, t0: float) -> int:
    a, b = int((t0 + TMIN) * sf), int((t0 + TMAX) * sf)
    a, b = max(0, a), min(len(zhf), b)
    over = np.where(zhf[a:b] > SP_SD)[0]
    if len(over) == 0:
        return 0
    return 1 + int(np.sum(np.diff(over) > SP_REFRACT_S * sf))


def _peri(env: np.ndarray, sf: float, t0: float) -> np.ndarray:
    """envelope interpolado en onset+PERI_T (vector de largo fijo)."""
    tt = np.arange(len(env)) / sf
    return np.interp(t0 + PERI_T, tt, env, left=np.nan, right=np.nan)


def _run_covariates(vhdr) -> dict | None:
    """Envelopes continuos sobre el raw nativo pre-ICA. None si faltan canales clave."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    need = {"R_EYE", "L_EYE", "Pz"}
    if not need.issubset(set(raw.ch_names)):
        return None
    g = lambda c: raw.get_data(picks=[c])[0]
    radial = 0.5 * (g("R_EYE") + g("L_EYE")) - g("Pz")
    fp = [c for c in ("Fp1", "Fp2") if c in raw.ch_names]
    blink_src = np.mean([g(c) for c in fp], axis=0) if fp else g("R_EYE")
    edge = [c for c in EMG_EDGE if c in raw.ch_names]
    edge_mean = np.mean([g(c) for c in edge], axis=0) if edge else None

    hf = filter_data(radial[None, :].astype(float), sf, 20.0, min(90.0, sf / 2 - 1), verbose="ERROR")[0]
    zhf = np.abs(hf) / (np.std(hf) + 1e-30)
    return dict(
        sf=sf,
        gamma_EOG=_env(radial, sf, 30.0, 40.0),
        sp_hf=_env(radial, sf, 20.0, 90.0),
        zhf=zhf,
        blink_slow=_env(blink_src, sf, 1.0, 10.0),
        blink_2hz=_env(blink_src, sf, 0.5, 2.0),
        muscle_hf=_env(edge_mean, sf, 110.0, 140.0) if edge_mean is not None else None,
    )


def _epoch_rows(cov, onsets, sub, label, cond):
    sf = cov["sf"]
    rows, peris = [], []
    for t0 in onsets:
        rows.append(dict(
            subject=sub, run=label, condition=cond, tnorm=np.nan,  # tnorm se setea afuera
            gamma_EOG=_wmean(cov["gamma_EOG"], sf, t0, TMIN, TMAX),
            sp_hf=_wmean(cov["sp_hf"], sf, t0, TMIN, TMAX),
            n_microsac=_sp_count(cov["zhf"], sf, t0),
            blink_slow=_wmean(cov["blink_slow"], sf, t0, TMIN, TMAX),
            blink_slow_pre=_wmean(cov["blink_slow"], sf, t0, TMIN, 0.0),
            blink_2hz_pre=_wmean(cov["blink_2hz"], sf, t0, TMIN, 0.0),
            muscle_hf=_wmean(cov["muscle_hf"], sf, t0, TMIN, TMAX) if cov["muscle_hf"] is not None else np.nan,
        ))
        peris.append(_peri(cov["gamma_EOG"], sf, t0))
    return rows, peris


def main() -> None:
    print("=" * 78)
    print("artifact_raw_scr :: 2.1 covariados de artefacto en raw pre-ICA, por época (N=6)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    cache, _, _ = load_cache("uniform")   # para validar alineamiento (tnorm)
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)   # reproducir silents del cache

    all_rows = []
    peri = {1: {}, 0: {}}   # peri[cond][sub] = lista de vectores
    align = []

    for sub in COHORT:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = [str(r) for r in cont["runs"]]
        real_rows, real_peri, real_tn = [], [], []
        sil_rows, sil_peri, sil_tn = [], [], []

        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            # onsets: mirror EXACTO de _build_one (filtro 1-40, resample 250, duration)
            raw_eeg = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw_eeg.pick("eeg"); attach_montage_and_drop_no_pos(raw_eeg)
            raw_eeg.filter(1.0, 40.0, verbose="ERROR"); raw_eeg.resample(250.0, verbose="ERROR")
            dur = float(raw_eeg.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(onsets[onsets < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            # boundary-drop idéntico a epoch_one_run
            real_kept = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            sil_kept = sil[(sil + TMIN > 0) & (sil + TMAX < dur)]

            cov = _run_covariates(vhdr)
            if cov is None:
                print(f"  {sub} {label}: faltan canales radial-EOG -> skip", flush=True)
                continue
            rr, rp = _epoch_rows(cov, real_kept, sub, label, 1)
            sr, sp = _epoch_rows(cov, sil_kept, sub, label, 0)
            real_rows += rr; real_peri += rp; real_tn += list(real_kept / dur)
            sil_rows += sr; sil_peri += sp; sil_tn += list(sil_kept / dur)
            print(f"  {sub} {label}: SCR={len(real_kept)} no-SCR={len(sil_kept)} (sf={cov['sf']:.0f})", flush=True)

        # orden del cache: REAL luego SILENT
        for r, tn in zip(real_rows, real_tn): r["tnorm"] = tn
        for r, tn in zip(sil_rows, sil_tn): r["tnorm"] = tn
        sub_rows = real_rows + sil_rows
        all_rows += sub_rows
        peri[1][sub] = real_peri
        peri[0][sub] = sil_peri

        # --- VALIDACIÓN de alineamiento contra el cache ---
        my_tn = np.array(real_tn + sil_tn)
        if sub in cache:
            cache_tn = cache[sub][2]
            n_ok = len(my_tn) == len(cache_tn)
            max_dev = float(np.max(np.abs(my_tn - cache_tn))) if n_ok else np.nan
            align.append(dict(subject=sub, n_mine=len(my_tn), n_cache=len(cache_tn),
                              n_match=n_ok, max_tnorm_dev=round(max_dev, 5) if n_ok else np.nan))
            if n_ok:
                print(f"  -> ALINEAMIENTO {sub}: n={len(my_tn)} match=True max|Δtnorm|={max_dev:.2e}", flush=True)
            else:
                print(f"  -> ALINEAMIENTO {sub}: MISMATCH n_mine={len(my_tn)} n_cache={len(cache_tn)}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(TBL_DIR / "artifact_covariate.csv", index=False)
    pd.DataFrame(align).to_csv(TBL_DIR / "alignment_check.csv", index=False)

    # --- test within-subject SCR vs no-SCR ---
    summ = []
    for col in COV_COLS:
        rels, pvals, npos, n = [], [], 0, 0
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna().to_numpy()
            b = sd[sd.condition == 0][col].dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            ma, mb = float(np.median(a)), float(np.median(b))
            # cambio fraccional SCR vs no-SCR; NaN para conteos discretos con mediana~0
            # (n_microsac: la rel explota por div/0 -> es secundario, se reporta por n_pos)
            rel = (ma - mb) / mb if abs(mb) > 1e-12 else np.nan
            try:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            rels.append(rel); pvals.append(p); n += 1; npos += int(ma > mb)
        summ.append(dict(covariate=col, n_subj=n,
                         mean_rel_diff_pct=round(100 * float(np.mean(rels)), 1) if rels else np.nan,
                         n_pos=f"{npos}/{n}", median_p=round(float(np.median(pvals)), 4) if pvals else np.nan))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(TBL_DIR / "artifact_within_subject.csv", index=False)

    print("\n=== Alineamiento (debe ser match=True, Δtnorm≈0) ===")
    print(pd.DataFrame(align).to_string(index=False))
    print("\n=== Within-subject SCR vs no-SCR (signo 6/6 = confound de evento) ===")
    print(summ_df.to_string(index=False))

    _plot_bysubject(df)
    _plot_peri(peri)
    print(f"\nCSV   -> {TBL_DIR / 'artifact_covariate.csv'}")
    print(f"Figuras-> {FIG_DIR}")
    print("[2.1] done", flush=True)


def _plot_bysubject(df: pd.DataFrame) -> None:
    cols = [("gamma_EOG", "gamma-EOG (microsacada/SP) [Q-gamma]"),
            ("blink_slow", "blink lento Fp1/Fp2 [Q-delta]"),
            ("muscle_hf", "músculo 110-140 borde")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, cols):
        for sub in COHORT:
            sd = df[df.subject == sub]
            a = sd[sd.condition == 1][col].dropna()
            b = sd[sd.condition == 0][col].dropna()
            if not len(a) or not len(b):
                continue
            ax.plot([0, 1], [b.median(), a.median()], "-o", color=SUBJ_COLORS[sub], lw=1.5, ms=5, label=sub)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no-SCR", "SCR"])
        ax.set_ylabel(col); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("2.1 Covariados de artefacto (raw pre-ICA): SCR vs no-SCR por sujeto\n"
                 "(sube en SCR = más artefacto en las épocas SCR = confound del evento)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "artifact_bysubject.png", dpi=120)
    plt.close(fig)


def _plot_peri(peri: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cond, col, lab in [(1, "C3", "SCR"), (0, "0.5", "no-SCR")]:
        # media por sujeto, luego grand-average
        subj_means = []
        for sub in COHORT:
            if sub in peri[cond] and len(peri[cond][sub]):
                subj_means.append(np.nanmean(np.array(peri[cond][sub]), axis=0))
        if not subj_means:
            continue
        M = np.array(subj_means)
        m = np.nanmean(M, axis=0)
        sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(len(M))
        ax.plot(PERI_T, m, color=col, lw=2, label=f"{lab} (N={len(M)})")
        ax.fill_between(PERI_T, m - sem, m + sem, color=col, alpha=0.2)
    ax.axvline(0, color="k", ls=":", lw=1, label="SCR onset")
    ax.set_xlabel("tiempo relativo al onset SCR (s)")
    ax.set_ylabel("gamma-EOG envelope (radial 30-40 Hz)")
    ax.set_title("2.1 Peri-SCR: gamma-EOG (microsacada/SP) time-locked al onset\n"
                 "(pico cerca de 0 en SCR = microsacada time-locked al evento = confound de Q-gamma)",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "artifact_peri_scr.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
