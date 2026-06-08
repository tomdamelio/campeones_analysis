"""1.1 — Descriptivos por sujeto de X (espectro EEG) y de Y (respuesta electrodermica).

Bloque 1 del ciclo 05_05. Produce la "ficha descriptiva" de las dos variables ANTES de
que entren a cualquier modelo (pedido de Pablo: "volver atras en los descriptivos ...
mirar los espectros de cada participante de una manera mas descriptiva para que despues
... hagas una cuantificacion en la que se pueda confiar").

X (espectro EEG) -- 4 figuras (una por ROI), 6 paneles (sujetos) cada una:
  Por panel, PSD log-log de epocas SCR (rojo) vs no-SCR (gris), separando explicitamente
  la VENTANA-RESPUESTA (0..3 s, solido) de la VENTANA-BASELINE (-5..-0.5 s, punteado).
  El baseline-overlay es el control de Muthukumaraswamy 2013 (Fig 1): una diferencia que
  ya existe en el baseline puede imitar una diferencia en la respuesta -> si los punteados
  se superponen pero los solidos no, el efecto es una respuesta genuina, no un desajuste
  de baseline. Bandas canonicas como guias verticales.

Y (respuesta electrodermica) -- distribucion de amplitudes de pico SCR + timing-en-run:
  histograma de amplitudes (parte positiva) por sujeto + pooled; QQ-plot vs gamma y vs
  log-normal (Diego: "plotear la distribucion ... no vaya a ser que sea bimodal"; Tomas:
  "curva muy asimetrica ... mayoria cero ... cola larga"); conteo de SCR por sujeto/run;
  timing dentro del run (tnorm in [0,1]) que cuantifica el confound temporal de Enzo
  (las SCR "pueden estar todas juntas al inicio ... al final").

Reusa la infra de epoching/deteccion del resto de multimodal_arousal para que los
descriptivos se computen sobre exactamente las mismas epocas/onsets que el modelo:
  build_subject_epochs / compute_psd / ROIS / roi_channel_indices  (tfr_psd_scr)
  detect_scr_onsets_s / filter_clean_onsets / EDA_FS / POST_S / TMIN / TMAX  (erp_scr)
Y NO re-procesa raw EEG (sale de cache npz); X re-construye epocas (build_subject_epochs).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_descriptives
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_descriptives --only y
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_descriptives --only x
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, DROP_CHANNELS, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    POST_S,
    TMAX,
    TMIN,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    ROI_NAMES,
    ROIS,
    build_subject_epochs,
    compute_psd,
    roi_channel_indices,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- output root: NEW outputs go to 05_05/, leaving the 05_04 cache untouched ---
OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "descriptives"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Canonical bands (vertical guides + band labels)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
}

# PSD is computed over the FULL epoch [-5, +3] s for both conditions (user 2026-06-06):
# the pre-onset window is NOT a neutral baseline -- t=0 is the SCR onset, so [-5,0] is the
# EEG that PRECEDES the bodily response (potential brain->body antecedent; Brankovic 2026
# lead of 1-4 s). No baseline normalization here -> matches how the decoding/FOOOF effect
# was computed. ("real"=SCR epochs, "silent"=no-SCR EDA-silent control epochs.)
FULL_WIN = (float(TMIN), float(TMAX))   # -5..+3 s


# =============================================================================
# X -- per-subject EEG spectra (SCR vs no-SCR), response vs baseline window
# =============================================================================
def _roi_db_curve(psd: np.ndarray, freqs: np.ndarray, roi_idxs: list[int]):
    """ROI-averaged power per epoch -> dB; return (mean_db, sem_db) over epochs."""
    if not roi_idxs or psd.shape[0] == 0:
        n_f = psd.shape[2]
        return np.full(n_f, np.nan), np.full(n_f, np.nan)
    roi_pow = psd[:, roi_idxs, :].mean(axis=1)          # (n_ep, n_freq)
    db = 10.0 * np.log10(roi_pow + 1e-30)               # (n_ep, n_freq)
    mean = db.mean(axis=0)
    sem = db.std(axis=0, ddof=1) / np.sqrt(db.shape[0]) if db.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def collect_x() -> dict:
    """For each subject, ROI-averaged PSD (mean+sem, dB) over the full epoch, SCR vs no-SCR."""
    store: dict = {}
    freqs_ref = None
    for sub in COHORT:
        print(f"[X] {sub}: building epochs ...", flush=True)
        real, silent = build_subject_epochs(sub)
        if real is None or silent is None:
            print(f"[X] {sub}: no epochs, skipped", flush=True)
            continue
        # STOPGAP Track A (cohort.DROP_CHANNELS): drop chronically-bad channels so X is
        # computed on the common 29-ch set (32 - 3). roi_channel_indices already filters
        # by `c in ch_names`, so dropping here suffices -- ROIS need no change.
        for ep in (real, silent):
            ep.drop_channels([c for c in DROP_CHANNELS if c in ep.ch_names])
        print(f"[X] {sub}: EEG channels after drop = {len(real.ch_names)} "
              f"(dropped {[c for c in DROP_CHANNELS if c not in real.ch_names]})", flush=True)
        roi_idx = roi_channel_indices(real.ch_names)
        sub_store = {"n_real": len(real), "n_silent": len(silent), "rois": {}}
        for cond, ep in (("real", real), ("silent", silent)):
            psd, freqs, _ = compute_psd(ep)  # full epoch [-5,+3]
            if freqs_ref is None:
                freqs_ref = freqs
            for roi in ROI_NAMES:
                mean, sem = _roi_db_curve(psd, freqs, roi_idx[roi])
                sub_store["rois"].setdefault(roi, {})[cond] = (mean, sem)
        store[sub] = sub_store
        print(f"[X] {sub}: n_real={len(real)} n_silent={len(silent)}", flush=True)
    store["_freqs"] = freqs_ref
    return store


def plot_x_roi(roi: str, store: dict) -> None:
    freqs = store["_freqs"]
    subs = [s for s in COHORT if s in store]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    fig.suptitle(
        f"PSD per subject -- ROI {roi} -- full epoch {FULL_WIN}s (no baseline) -- "
        "SCR (red) vs no-SCR (grey)",
        fontsize=11,
    )
    for ax, sub in zip(axes.ravel(), subs):
        d = store[sub]["rois"][roi]
        for cond, color, lab in (("real", "C3", "SCR"), ("silent", "0.45", "no-SCR")):
            m, s = d[cond]
            ax.plot(freqs, m, color=color, lw=1.8, ls="-", label=lab)
            ax.fill_between(freqs, m - s, m + s, color=color, alpha=0.18)
        ax.set_xscale("log")
        for (lo, hi) in BANDS.values():
            ax.axvline(lo, color="0.85", lw=0.6, zorder=0)
        ax.set_title(f"{sub}  (n_SCR={store[sub]['n_real']}, n_noSCR={store[sub]['n_silent']})",
                     fontsize=9)
        ax.set_xticks([1, 2, 4, 8, 13, 30, 40])
        ax.set_xticklabels(["1", "2", "4", "8", "13", "30", "40"], fontsize=7)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Power (dB)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / f"psd_{roi}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[X] saved {out}", flush=True)


def main_x() -> None:
    store = collect_x()
    if store.get("_freqs") is None:
        print("[X] no data collected", flush=True)
        return
    for roi in ROI_NAMES:
        plot_x_roi(roi, store)


# =============================================================================
# Y -- electrodermal response distribution + timing-in-run
# =============================================================================
def collect_y() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-event SCR amplitude + tnorm (from cached eda_phasic), and per-run counts."""
    ev_rows = []   # one row per detected clean SCR event
    run_rows = []  # one row per (subject, run)
    for sub in COHORT:
        cont_path = NPZ_DIR / f"{sub}_continuous.npz"
        if not cont_path.exists():
            print(f"[Y] {sub}: no npz, skipped", flush=True)
            continue
        cont = np.load(cont_path, allow_pickle=True)
        runs_in_npz = list(cont["runs"])
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            dur = len(eda) / EDA_FS
            onsets = detect_scr_onsets_s(eda, EDA_FS)
            onsets = onsets[onsets < dur]
            onsets = filter_clean_onsets(onsets, eda, EDA_FS)
            valid = (onsets + TMIN > 0) & (onsets + TMAX < dur)
            onsets = onsets[valid]
            amps = []
            for t in onsets:
                i0 = int(round(t * EDA_FS))
                i1 = int(round((t + POST_S) * EDA_FS))
                a = float(np.max(eda[i0:i1])) if (i1 > i0 and i1 <= len(eda)) else np.nan
                amps.append(a)
                ev_rows.append({"subject": sub, "run": label, "onset_s": t,
                                "tnorm": t / dur if dur > 0 else np.nan, "amp": a})
            amps = np.asarray(amps, float)
            amps = amps[np.isfinite(amps)]
            run_rows.append({
                "subject": sub, "run": label, "dur_min": dur / 60.0,
                "n_scr": len(onsets),
                "scr_rate_per_min": len(onsets) / (dur / 60.0) if dur > 0 else np.nan,
                "amp_mean": float(np.mean(amps)) if amps.size else np.nan,
                "amp_median": float(np.median(amps)) if amps.size else np.nan,
            })
    return pd.DataFrame(ev_rows), pd.DataFrame(run_rows)


def _bysubject_summary(ev: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sub in COHORT:
        e = ev[ev.subject == sub]
        r = run[run.subject == sub]
        amp = e["amp"].dropna().to_numpy()
        tn = e["tnorm"].dropna().to_numpy()
        if amp.size == 0:
            continue
        q = np.array([np.mean((tn >= a) & (tn < b)) for a, b in
                      [(0, .25), (.25, .5), (.5, .75), (.75, 1.0001)]])
        rows.append({
            "subject": sub, "n_scr": int(amp.size), "n_runs": int(r.shape[0]),
            "scr_rate_per_min": float(r["scr_rate_per_min"].mean()),
            "amp_mean": float(np.mean(amp)), "amp_median": float(np.median(amp)),
            "amp_iqr": float(np.subtract(*np.percentile(amp, [75, 25]))),
            "amp_skew": float(stats.skew(amp)), "amp_kurtosis": float(stats.kurtosis(amp)),
            "mean_tnorm": float(np.mean(tn)),
            "pct_Q1": float(q[0]), "pct_Q2": float(q[1]),
            "pct_Q3": float(q[2]), "pct_Q4": float(q[3]),
        })
    return pd.DataFrame(rows)


def plot_y_amplitude(ev: pd.DataFrame) -> None:
    subs = [s for s in COHORT if s in set(ev.subject)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    fig.suptitle("SCR peak-amplitude distribution (detected SCRs, positive part) -- "
                 "per subject + pooled", fontsize=12)
    for ax, sub in zip(axes.ravel()[:6], subs):
        a = ev[ev.subject == sub]["amp"].dropna().to_numpy()
        ax.hist(a, bins=30, color=SUBJ_COLORS[sub], alpha=0.85)
        ax.set_title(f"{sub}  n={a.size}\nskew={stats.skew(a):.2f} "
                     f"kurt={stats.kurtosis(a):.2f}", fontsize=9)
        ax.set_xlabel("SCR peak amplitude (uS, phasic)")
        ax.set_ylabel("count")
    # pooled
    ax = axes.ravel()[6]
    allamp = ev["amp"].dropna().to_numpy()
    ax.hist(allamp, bins=50, color="0.3", alpha=0.85)
    ax.set_title(f"POOLED  n={allamp.size}\nskew={stats.skew(allamp):.2f} "
                 f"kurt={stats.kurtosis(allamp):.2f}", fontsize=9)
    ax.set_xlabel("SCR peak amplitude (uS, phasic)")
    # pooled log-x
    ax = axes.ravel()[7]
    ax.hist(np.log10(allamp + 1e-6), bins=50, color="0.3", alpha=0.85)
    ax.set_title("POOLED log10(amp)", fontsize=9)
    ax.set_xlabel("log10 SCR peak amplitude")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "scr_amplitude.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[Y] saved {out}", flush=True)


def plot_y_qq(ev: pd.DataFrame) -> None:
    a = ev["amp"].dropna().to_numpy()
    a = a[a > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # gamma (floc=0)
    g = stats.gamma.fit(a, floc=0)
    stats.probplot(a, dist=stats.gamma, sparams=(g[0],), plot=axes[0])
    axes[0].set_title(f"QQ vs Gamma (shape={g[0]:.2f})")
    # lognormal (floc=0)
    ln = stats.lognorm.fit(a, floc=0)
    stats.probplot(a, dist=stats.lognorm, sparams=(ln[0],), plot=axes[1])
    axes[1].set_title(f"QQ vs Log-normal (sigma={ln[0]:.2f})")
    fig.suptitle(f"SCR amplitude QQ-plots (pooled, n={a.size}) -- "
                 "decide gamma vs log-normal for 3.1", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "scr_qq.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[Y] saved {out}", flush=True)


def plot_y_timing(ev: pd.DataFrame) -> None:
    subs = [s for s in COHORT if s in set(ev.subject)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("SCR timing within run (tnorm in [0,1]) -- quantifies Enzo's temporal "
                 "confound (SCRs cluster later in session)", fontsize=11)
    for ax, sub in zip(axes.ravel(), subs):
        tn = ev[ev.subject == sub]["tnorm"].dropna().to_numpy()
        ax.hist(tn, bins=20, range=(0, 1), color=SUBJ_COLORS[sub], alpha=0.8)
        ax.axvline(np.mean(tn), color="k", lw=1.2, ls="--",
                   label=f"mean={np.mean(tn):.2f}")
        for qx in (0.25, 0.5, 0.75):
            ax.axvline(qx, color="0.8", lw=0.6, zorder=0)
        # rug
        ax.plot(tn, np.full_like(tn, -0.5), "|", color="k", alpha=0.3, ms=6)
        ax.set_title(f"{sub}  n={tn.size}", fontsize=9)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("normalized position in run (0=start, 1=end)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SCR count")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "scr_timing.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[Y] saved {out}", flush=True)


def plot_y_counts(run: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    subs = [s for s in COHORT if s in set(run.subject)]
    x = 0
    xticks, xlabels = [], []
    for sub in subs:
        r = run[run.subject == sub].sort_values("run")
        xs = np.arange(x, x + len(r))
        ax.bar(xs, r["n_scr"].to_numpy(), color=SUBJ_COLORS[sub], alpha=0.85)
        xticks.append(xs.mean())
        xlabels.append(f"{sub}\n({int(r['n_scr'].sum())})")
        x += len(r) + 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("n SCR per run")
    ax.set_title("SCR count per run, grouped by subject (total in parens)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "scr_counts.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[Y] saved {out}", flush=True)


def main_y() -> None:
    ev, run = collect_y()
    if ev.empty:
        print("[Y] no events collected", flush=True)
        return
    bysub = _bysubject_summary(ev, run)
    ev.to_csv(TBL_DIR / "scr_events.csv", index=False)
    run.to_csv(TBL_DIR / "scr_descriptives.csv", index=False)
    bysub.to_csv(TBL_DIR / "scr_descriptives_bysubject.csv", index=False)
    print("\n[Y] per-subject summary:")
    print(bysub.to_string(index=False), flush=True)
    plot_y_amplitude(ev)
    plot_y_qq(ev)
    plot_y_timing(ev)
    plot_y_counts(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["x", "y"], default=None,
                    help="run only the X (spectra) or Y (electrodermal) ficha")
    args = ap.parse_args()
    print("=" * 78)
    print("scr_descriptives :: 1.1 ficha descriptiva de X (espectro) e Y (EDA)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    if args.only in (None, "y"):
        main_y()
    if args.only in (None, "x"):
        main_x()


if __name__ == "__main__":
    main()
