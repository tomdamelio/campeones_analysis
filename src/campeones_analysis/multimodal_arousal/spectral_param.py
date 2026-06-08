"""1.3 — Parametrizacion sistematica del espectro: corrimiento de center-frequency de alfa.

Bloque 1 del ciclo 05_05. Responde el enfasis de Diego ("parametriza el espectro ... puede
que lo que cambie sea la FRECUENCIA del pico de alfa y no la amplitud") y, a la vez, la
hipotesis del usuario (2026-06-06): en SCR el pico periodico de alfa seria mas BAJO,
"comido" por el aperiodico que sube -> verificable sobre el espectro APLANADO (1/f removido).

Logica: describir el espectro con pocas "perillas" (offset, exponent + CF/PW/BW por pico) en
vez de promedios de banda (que mezclan frecuencia + amplitud + fondo en un solo numero y
"tienen arte"). El contraste central es ALFA: DeltaCF (corrimiento) vs DeltaPW (amplitud),
SCR vs no-SCR, por sujeto.

Doble via (concordancia = robustez):
  Via A -- FOOOF (Donoghue 2020): ajusta 1/f + picos; CF/PW/BW de alfa con el fondo separado.
  Via B -- PAF/CoG (Corcoran 2018): sobre el espectro APLANADO (aperiodico removido, critico
           para no confundir corrimiento con rotacion 1/f), PAF = freq del max en 7-13 Hz,
           CoG = centroide ponderado por potencia.

ROI = parieto-occipital (donde vive alfa). Se recomputa el espectro por sujeto/condicion
(reusando build_subject_epochs / compute_psd) para tener via A + via B + espectro aplanado
totalmente consistentes; los params se contrastan SCR vs no-SCR (paired, N=6).

Outputs (research_diary/context/05_05/parametrization/):
  tables/params_fooof.csv    subject,condition,offset,exponent,alpha_cf,alpha_pw,alpha_bw,r2
  tables/params_iaf.csv      subject,condition,paf,cog
  tables/contrast_alpha.csv  metric,mean_real,mean_silent,mean_delta,n_pos,n_neg,wilcoxon_p
  figures/alpha_cf_pw.png    paired CF, paired PW, DeltaCF-vs-DeltaPW scatter (money panel)
  figures/flattened_alpha.png  espectro aplanado per subject, SCR vs no-SCR (alpha-masking test)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.spectral_param
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fooof import FOOOF
from scipy import stats

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT,
    DROP_CHANNELS,
    REPO,
    SUBJ_COLORS,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    ROIS,
    build_subject_epochs,
    compute_psd,
)

warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "parametrization"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Posterior ROI where alpha lives (same set used in band_scr_amplitude_scr.py).
POSTERIOR = ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"]
# Run the alpha contrast on EVERY ROI (consistency check, user request 2026-06-06) plus a
# combined posterior ROI. ROIS = {Frontal, Temporal, Parietal, Occipital} from tfr_psd_scr.
ROIS_RUN: dict[str, list[str]] = {**ROIS, "Posterior": POSTERIOR}
ROI_ORDER = ["Frontal", "Temporal", "Parietal", "Occipital", "Posterior"]
ALPHA_BAND = (8.0, 13.0)
IAF_BAND = (7.0, 13.0)   # Corcoran search band (a bit wider than canonical alpha)
# Per-band PERIODIC power (mean of the aperiodic-removed spectrum in each band) -- to check
# whether bands OTHER than alpha (esp. beta/low-gamma) differ between conditions (user 2026-06-06).
# delta starts at the fit floor (1.5 Hz) and is UNRELIABLE there (Gerster 2022) -> caveat, not verdict.
BANDS_FULL: dict[str, tuple[float, float]] = {
    "delta": (1.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
}

# FOOOF params: mirror the cached fooof_scr.py fit so via A is comparable to the 05_04 cache.
FIT_RANGE = (1.5, 40.0)
FOOOF_KW = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.10)
COND_COLORS = {"real": "C3", "silent": "0.4"}
COND_LABEL = {"real": "SCR", "silent": "no-SCR"}


def roi_mean_psd(psd, ch, ch_list):
    """Mean linear PSD across epochs and the channels of one ROI. Returns power or None."""
    idx = [ch.index(c) for c in ch_list if c in ch]
    if not idx or psd.shape[0] == 0:
        return None
    return psd[:, idx, :].mean(axis=(0, 1))       # mean over epochs + channels -> (n_freq,)


def fit_and_params(freqs, power):
    """Fit FOOOF; return dict with aperiodic + dominant-alpha peak (via A) and PAF/CoG (via B)."""
    fm = FOOOF(aperiodic_mode="fixed", verbose=False, **FOOOF_KW)
    fm.fit(np.asarray(freqs, float), np.asarray(power, float), freq_range=list(FIT_RANGE))
    offset, exponent = fm.aperiodic_params_[0], fm.aperiodic_params_[1]
    # via A: dominant alpha peak (max PW with CF in [8,13])
    peaks = fm.peak_params_  # rows of [CF, PW, BW]
    a_cf = a_pw = a_bw = np.nan
    if peaks.size:
        in_alpha = peaks[(peaks[:, 0] >= ALPHA_BAND[0]) & (peaks[:, 0] < ALPHA_BAND[1])]
        if in_alpha.size:
            best = in_alpha[np.argmax(in_alpha[:, 1])]
            a_cf, a_pw, a_bw = float(best[0]), float(best[1]), float(best[2])
    # via B: PAF/CoG on the flattened (aperiodic-removed) spectrum, in 7-13 Hz
    f = fm.freqs
    flat = fm.power_spectrum - fm._ap_fit          # log10 power, aperiodic removed
    m = (f >= IAF_BAND[0]) & (f <= IAF_BAND[1])
    paf = cog = np.nan
    if m.sum() >= 3:
        ff, pp = f[m], flat[m]
        pp_pos = np.clip(pp, 0, None)              # only positive (above-aperiodic) power
        if pp_pos.sum() > 0:
            paf = float(ff[np.argmax(pp)])
            cog = float(np.sum(ff * pp_pos) / np.sum(pp_pos))
    # continuous alpha-bump height above 1/f in 8-13 Hz -- ALWAYS defined (no peak-detection
    # threshold), so it captures cases where the FOOOF discrete peak vanished (alpha collapsed
    # below min_peak_height in SCR). This is the robust, no-dropout amplitude metric.
    am = (f >= ALPHA_BAND[0]) & (f < ALPHA_BAND[1])
    alpha_flatpow = float(max(0.0, np.max(flat[am]))) if am.sum() else np.nan
    # per-band periodic power = mean positive elevation above the 1/f in each band
    out = dict(offset=float(offset), exponent=float(exponent),
               alpha_cf=a_cf, alpha_pw=a_pw, alpha_bw=a_bw, alpha_flatpow=alpha_flatpow,
               r2=float(fm.r_squared_), paf=paf, cog=cog, _freqs=f, _flat=flat)
    for b, (lo, hi) in BANDS_FULL.items():
        bm = (f >= lo) & (f < hi)
        out[f"pp_{b}"] = float(np.mean(np.clip(flat[bm], 0, None))) if bm.any() else np.nan
    return out


def collect() -> tuple[pd.DataFrame, dict]:
    rows = []
    flats: dict = {}   # {(sub, roi): {cond: (freqs, flat, alpha_cf, alpha_pw)}}
    for sub in COHORT:
        print(f"[1.3] {sub}: building epochs ...", flush=True)
        real, silent = build_subject_epochs(sub)
        if real is None or silent is None:
            print(f"[1.3] {sub}: no epochs, skipped", flush=True)
            continue
        # Exclude the 3 bad channels (cohort.DROP_CHANNELS = FC1, TP10, Fz) -> common 29-ch set.
        # ROI definitions still list these explicitly, but roi_mean_psd guards with `if c in ch`,
        # so dropping here removes them from Frontal (Fz, FC1) and Temporal (TP10); Posterior
        # ROI (P3,Pz,P4,P7,P8,O1,O2) is untouched.
        for ep in (real, silent):
            drop = [c for c in DROP_CHANNELS if c in ep.ch_names]
            if drop:
                ep.drop_channels(drop)
        print(f"[1.3] {sub}: dropped {[c for c in DROP_CHANNELS if c not in real.ch_names]} "
              f"-> {len(real.ch_names)} ch", flush=True)
        for cond, ep in (("real", real), ("silent", silent)):
            psd, freqs, ch = compute_psd(ep)   # ONE PSD per (sub, cond); slice ROIs from it
            for roi, ch_list in ROIS_RUN.items():
                power = roi_mean_psd(psd, ch, ch_list)
                if power is None:
                    continue
                p = fit_and_params(freqs, power)
                row = {"subject": sub, "roi": roi, "condition": cond,
                       "offset": p["offset"], "exponent": p["exponent"],
                       "alpha_cf": p["alpha_cf"], "alpha_pw": p["alpha_pw"],
                       "alpha_bw": p["alpha_bw"], "alpha_flatpow": p["alpha_flatpow"],
                       "r2": p["r2"], "paf": p["paf"], "cog": p["cog"]}
                row.update({f"pp_{b}": p[f"pp_{b}"] for b in BANDS_FULL})
                rows.append(row)
                flats.setdefault((sub, roi), {})[cond] = (
                    p["_freqs"], p["_flat"], p["alpha_cf"], p["alpha_pw"])
        # quick posterior log line
        post = [r for r in rows if r["subject"] == sub and r["roi"] == "Posterior"]
        for r in post:
            print(f"[1.3] {sub} Posterior {COND_LABEL[r['condition']]}: "
                  f"alpha_cf={r['alpha_cf']:.2f} pw={r['alpha_pw']:.3f} "
                  f"exp={r['exponent']:.3f} off={r['offset']:.3f}", flush=True)
    return pd.DataFrame(rows), flats


def contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Paired SCR vs no-SCR contrast across subjects, per ROI x metric."""
    metrics = ["offset", "exponent", "alpha_cf", "alpha_pw", "alpha_bw",
               "alpha_flatpow", "paf", "cog"] + [f"pp_{b}" for b in BANDS_FULL]
    out = []
    for roi in ROI_ORDER:
        sub_df = df[df.roi == roi]
        if sub_df.empty:
            continue
        piv = sub_df.pivot(index="subject", columns="condition")
        for m in metrics:
            if m not in piv.columns.get_level_values(0):
                continue
            r = piv[m]["real"] if "real" in piv[m].columns else pd.Series(dtype=float)
            s = piv[m]["silent"] if "silent" in piv[m].columns else pd.Series(dtype=float)
            d = (r - s).dropna()
            if d.empty:
                continue
            try:
                wp = stats.wilcoxon(d)[1] if (d != 0).any() else np.nan
            except Exception:
                wp = np.nan
            out.append({"roi": roi, "metric": m,
                        "mean_real": float(r.mean()), "mean_silent": float(s.mean()),
                        "mean_delta": float(d.mean()), "n": int(d.size),
                        "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                        "wilcoxon_p": float(wp) if wp == wp else np.nan})
    return pd.DataFrame(out)


def plot_alpha_cf_pw(df: pd.DataFrame) -> None:
    """Posterior-ROI paired CF + PW, plus panel 3 = DeltaPW of alpha across ALL ROIs."""
    post = df[df.roi == "Posterior"]
    piv = post.pivot(index="subject", columns="condition")
    subs = [s for s in COHORT if s in piv.index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # paired CF (posterior)
    for sub in subs:
        cf_r, cf_s = piv["alpha_cf"]["real"][sub], piv["alpha_cf"]["silent"][sub]
        axes[0].plot([0, 1], [cf_s, cf_r], "-o", color=SUBJ_COLORS[sub], label=sub)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["no-SCR", "SCR"])
    axes[0].set_ylabel("alpha CF (Hz)")
    axes[0].set_title("Posterior: alpha center frequency (shift?)")
    axes[0].legend(fontsize=7)
    # paired PW (posterior)
    for sub in subs:
        pw_r, pw_s = piv["alpha_pw"]["real"][sub], piv["alpha_pw"]["silent"][sub]
        axes[1].plot([0, 1], [pw_s, pw_r], "-o", color=SUBJ_COLORS[sub])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["no-SCR", "SCR"])
    axes[1].set_ylabel("alpha PW (log power, aperiodic-removed)")
    axes[1].set_title("Posterior: alpha peak power (amplitude?)")
    # panel 3: Delta alpha bump (continuous, no dropout) across ROIs -- consistency check
    ax = axes[2]
    for sub in subs:
        ys = []
        for roi in ROI_ORDER:
            sd = df[(df.subject == sub) & (df.roi == roi)].set_index("condition")
            if "real" in sd.index and "silent" in sd.index:
                ys.append(float(sd.loc["real", "alpha_flatpow"] - sd.loc["silent", "alpha_flatpow"]))
            else:
                ys.append(np.nan)
        ax.plot(range(len(ROI_ORDER)), ys, "-o", color=SUBJ_COLORS[sub], alpha=0.8)
    ax.axhline(0, color="0.5", lw=1.0)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=30, fontsize=8)
    ax.set_ylabel("Delta alpha bump above 1/f (SCR - no-SCR)")
    ax.set_title("Alpha amplitude drop across ROIs (continuous, no dropout)")
    fig.suptitle("1.3 Alpha parametrization -- SCR vs no-SCR (N=6); panels 1-2 posterior, "
                 "panel 3 all ROIs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "alpha_cf_pw.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.3] saved {out}", flush=True)


def plot_alpha_byroi(df: pd.DataFrame) -> None:
    """DeltaPW and DeltaCF of alpha across all ROIs, one line per subject + group mean."""
    subs = [s for s in COHORT if s in set(df.subject)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # left = continuous alpha bump height (always defined). right = PAF (argmax of flattened
    # spectrum = continuous CF; recovers the discrete-peak dropouts EXCEPT where alpha vanished
    # entirely, i.e. no positive bump at all -> frequency genuinely undefined, stays NaN).
    for ax, metric, ylab, down in (
        (axes[0], "alpha_flatpow", "Delta alpha bump above 1/f (SCR - no-SCR)", True),
        (axes[1], "paf", "Delta PAF alpha (Hz, SCR - no-SCR)", False)):
        group = np.full((len(subs), len(ROI_ORDER)), np.nan)
        for i, sub in enumerate(subs):
            for j, roi in enumerate(ROI_ORDER):
                sd = df[(df.subject == sub) & (df.roi == roi)].set_index("condition")
                if "real" in sd.index and "silent" in sd.index and \
                        np.isfinite(sd.loc["real", metric]) and np.isfinite(sd.loc["silent", metric]):
                    group[i, j] = float(sd.loc["real", metric] - sd.loc["silent", metric])
            ax.plot(range(len(ROI_ORDER)), group[i], "-o", color=SUBJ_COLORS[sub],
                    alpha=0.55, label=sub)
        mean = np.nanmean(group, axis=0)
        ax.plot(range(len(ROI_ORDER)), mean, "-s", color="k", lw=2.4, ms=8, label="mean")
        for j, roi in enumerate(ROI_ORDER):
            col = group[:, j][np.isfinite(group[:, j])]
            if col.size:
                sign = f"{int((col < 0).sum())}/{col.size}↓" if down else \
                       f"{int((col > 0).sum())}/{col.size}↑"
                ax.annotate(sign, (j, np.nanmax(group[:, j])),
                            fontsize=7, ha="center", va="bottom", color="0.3")
        ax.axhline(0, color="0.5", lw=1.0)
        ax.set_xticks(range(len(ROI_ORDER)))
        ax.set_xticklabels(ROI_ORDER, rotation=30, fontsize=9)
        ax.set_ylabel(ylab)
    # mark ROIs where some subject has alpha fully absent (PAF undefined) on the right panel
    for j, roi in enumerate(ROI_ORDER):
        n_absent = int(df[(df.roi == roi)].groupby("subject").apply(
            lambda g: g["paf"].isna().any()).sum())
        if n_absent:
            axes[1].annotate(f"{n_absent} α-absent", (j, axes[1].get_ylim()[0]),
                             fontsize=6, ha="center", va="bottom", color="firebrick")
    axes[0].set_title("Alpha amplitude (continuous, no dropout): SCR drop consistent across ROIs?")
    axes[1].set_title("Alpha frequency (PAF, continuous): shift across ROIs? "
                      "(NaN only where alpha absent)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("1.3 Alpha contrast across ALL ROIs (N=6) -- consistency check", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "alpha_byroi.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.3] saved {out}", flush=True)


def plot_band_periodic(df: pd.DataFrame) -> None:
    """Delta periodic power (aperiodic-removed) per band x ROI -- is beta/gamma differential,
    and is it edge-max (EMG hint) or distributed? SCR>noSCR shows as positive."""
    subs = [s for s in COHORT if s in set(df.subject)]
    bands = list(BANDS_FULL.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, band in zip(axes.ravel(), bands):
        metric = f"pp_{band}"
        group = np.full((len(subs), len(ROI_ORDER)), np.nan)
        for i, sub in enumerate(subs):
            for j, roi in enumerate(ROI_ORDER):
                sd = df[(df.subject == sub) & (df.roi == roi)].set_index("condition")
                if "real" in sd.index and "silent" in sd.index:
                    group[i, j] = float(sd.loc["real", metric] - sd.loc["silent", metric])
            ax.plot(range(len(ROI_ORDER)), group[i], "-o", color=SUBJ_COLORS[sub], alpha=0.5)
        ax.plot(range(len(ROI_ORDER)), np.nanmean(group, axis=0), "-s", color="k", lw=2.4, ms=8)
        for j in range(len(ROI_ORDER)):
            col = group[:, j][np.isfinite(group[:, j])]
            if col.size:
                ax.annotate(f"{int((col > 0).sum())}/{col.size}↑", (j, np.nanmax(group[:, j])),
                            fontsize=7, ha="center", va="bottom", color="0.3")
        ax.axhline(0, color="0.5", lw=1.0)
        ax.set_xticks(range(len(ROI_ORDER)))
        ax.set_xticklabels(ROI_ORDER, rotation=30, fontsize=8)
        ttl = f"{band} periodic power"
        if band == "delta":
            ttl += " (UNRELIABLE <2 Hz)"
        if band in ("beta", "gamma"):
            ttl += " -- EMG-suspect band"
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("Delta periodic power (SCR - no-SCR)")
    axes.ravel()[-1].axis("off")
    handles = [plt.Line2D([0], [0], color=SUBJ_COLORS[s], marker="o", label=s) for s in subs]
    handles.append(plt.Line2D([0], [0], color="k", marker="s", lw=2.4, label="mean"))
    axes.ravel()[-1].legend(handles=handles, loc="center", fontsize=10)
    fig.suptitle("1.3 Periodic power per band x ROI (aperiodic-removed) -- SCR vs no-SCR (N=6). "
                 "SCR>noSCR = positive; edge-max (Temporal/Frontal) hints EMG", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "band_periodic_byroi.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.3] saved {out}", flush=True)


def plot_flattened(flats: dict, roi: str = "Posterior") -> None:
    subs = [s for s in COHORT if (s, roi) in flats]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(f"1.3 Flattened spectra (aperiodic removed) -- {roi} ROI -- "
                 "alpha-masking test: is alpha PW lower in SCR AFTER removing 1/f?", fontsize=11)
    for ax, sub in zip(axes.ravel(), subs):
        for cond in ("real", "silent"):
            if cond not in flats[(sub, roi)]:
                continue
            f, flat, _, _ = flats[(sub, roi)][cond]
            ax.plot(f, flat, color=COND_COLORS[cond], lw=1.6, label=COND_LABEL[cond])
        ax.axvspan(ALPHA_BAND[0], ALPHA_BAND[1], color="gold", alpha=0.12)
        ax.set_xlim(2, 25)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(sub, fontsize=9)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("flattened log-power")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "flattened_alpha.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.3] saved {out}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="regenerate figures from cached params_fooof.csv (no epoch rebuild)")
    args = ap.parse_args()
    print("=" * 78)
    print("spectral_param :: 1.3 parametrizacion del espectro (DeltaCF vs DeltaPW de alfa)")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    if args.replot:
        df = pd.read_csv(TBL_DIR / "params_fooof.csv")
        con = contrast(df)
        con.to_csv(TBL_DIR / "contrast_alpha.csv", index=False)
        plot_alpha_cf_pw(df)
        plot_alpha_byroi(df)
        if all(f"pp_{b}" in df.columns for b in BANDS_FULL):
            plot_band_periodic(df)
        print("[1.3] replotted from cache (flattened figure unchanged)", flush=True)
        return
    df, flats = collect()
    if df.empty:
        print("[1.3] no data", flush=True)
        return
    df.to_csv(TBL_DIR / "params_fooof.csv", index=False)
    df[["subject", "roi", "condition", "paf", "cog"]].to_csv(TBL_DIR / "params_iaf.csv", index=False)
    con = contrast(df)
    con.to_csv(TBL_DIR / "contrast_alpha.csv", index=False)
    print("\n[1.3] paired contrast SCR vs no-SCR, per ROI (N=6) -- alpha PW/CF + aperiodic:")
    show = con[con.metric.isin(["alpha_pw", "alpha_cf", "alpha_bw", "exponent", "offset"])]
    print(show.to_string(index=False), flush=True)
    # concordance via A (alpha_cf) vs via B (paf), posterior
    d = df[df.roi == "Posterior"].dropna(subset=["alpha_cf", "paf"])
    if len(d) > 2:
        r = np.corrcoef(d["alpha_cf"], d["paf"])[0, 1]
        print(f"\n[1.3] concordance FOOOF-CF vs PAF (posterior): r={r:.3f} (n={len(d)})", flush=True)
    plot_alpha_cf_pw(df)
    plot_alpha_byroi(df)
    plot_band_periodic(df)
    plot_flattened(flats, "Posterior")


if __name__ == "__main__":
    main()
