"""1.2 — Caracterizar el aperiodico (1/f) + alfa, chequeo del bump temporal, y TEST PEAK-BLIND.

Bloque 1 del ciclo 05_05. Re-encuadra los dos efectos espectrales solidos del 05_04 (rotacion
del 1/f y alfa) como caracterizacion, y resuelve la tension que destapo 1.3: el 05_04 reporto
Delta-exponent<0 (flattening, per-canal) pero el ROI-mean de 1.3 dio exponent ~0.

Tres partes:
  A) Reconciliacion (cache fooof_scr_aperiodic.csv): Delta-exponent / Delta-offset per-canal y
     per-sujeto + topomaps. Cuantifica que el flattening es real pero HETEROGENEO (no 6/6).
  B) TEST PEAK-BLIND (recompute, el plato fuerte). El riesgo (Gerster 2022): cuando el alfa/theta
     CAEN en SCR (visto en 1.3), la joroba se achata y puede tirar de la recta del fondo haciendola
     parecer mas plana, aunque el 1/f real no cambie -> un cambio OSCILATORIO disfrazado de
     APERIODICO. Para distinguir, se estima el exponent de 3 maneras por sujeto (espectro
     channel-mean global) y se contrasta SCR vs no-SCR:
       exp_full    : FOOOF 1.5-40 Hz (el estandar/cache)
       exp_nogamma : FOOOF 1.5-30 Hz (excluye la banda gamma EMG-sospechosa que aplana artificial)
       exp_blind   : ajuste lineal log-log en 1.5-30 ENMASCARANDO theta(4-8)+alfa(8-13) -> fondo 1/f
                     ajustado solo sobre bins sin pico (Gerster 2022, control extra-robusto)
     Veredicto: si el flattening (Delta-exp<0) PERSISTE en exp_blind -> es genuinamente del fondo
     aperiodico; si DESAPARECE al enmascarar los picos -> era el alfa/theta cayendo, no el 1/f.
  C) Bump temporal ~4-6 Hz (cache fooof_scr_periodic.csv): en cuantos sujetos hay un pico theta
     en el ROI Temporal, por condicion. (En 1.3 el theta periodico CAE en SCR, no aparece un bump.)

Outputs (research_diary/context/05_05/spectral_effects/):
  tables/exponent_reconcile.csv   per-subject Delta-exp/offset per-channel summary (cache)
  tables/peakblind.csv            per-subject exp_full/nogamma/blind x condition + Deltas
  tables/temporal_bump.csv        theta-peak counts in Temporal ROI by condition (cache)
  figures/topo_aperiodic.png      GA topomaps Delta-exponent + Delta-offset (cache)
  figures/peakblind.png           per-subject Delta-exponent: full vs nogamma vs blind

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.aperiodic_char
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from fooof import FOOOF

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT, DROP_CHANNELS, OUT as OUT_0504, REPO, SUBJ_COLORS,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs, compute_psd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "spectral_effects"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

APERIODIC_CACHE = OUT_0504 / "qa_artifact_vs_signal" / "tables" / "fooof_scr_aperiodic.csv"
PERIODIC_CACHE = OUT_0504 / "qa_artifact_vs_signal" / "tables" / "fooof_scr_periodic.csv"

FOOOF_KW = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.10)
MASK_BANDS = ((4.0, 8.0), (8.0, 13.0))   # theta + alpha masked for the peak-blind fit
COND_LABEL = {"real": "SCR", "silent": "no-SCR"}
TEMPORAL = ["T7", "T8", "FT9", "FT10", "TP9", "TP10", "FC5", "FC6", "CP5", "CP6"]


# =============================================================================
# A) Reconciliation from cache: per-channel / per-subject Delta-exponent & offset
# =============================================================================
def reconcile_aperiodic() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(APERIODIC_CACHE)
    a = a[(a["level"] == "channel") & (a["mode"] == "fixed")]
    a = a[~a["key"].isin(DROP_CHANNELS)]   # drop bad channels -> 29-channel set
    perch = a.pivot_table(index=["subject", "key"], columns="condition", values=["exponent", "offset"])
    perch = perch.reset_index()
    perch["d_exp"] = perch[("exponent", "real")] - perch[("exponent", "silent")]
    perch["d_off"] = perch[("offset", "real")] - perch[("offset", "silent")]
    perch.columns = ["subject", "channel", "exp_real", "exp_silent", "off_real", "off_silent",
                     "d_exp", "d_off"]
    bysub = perch.groupby("subject").agg(
        n_ch=("d_exp", "size"), n_flat=("d_exp", lambda x: int((x < 0).sum())),
        mean_dexp=("d_exp", "mean"), mean_doff=("d_off", "mean")).reset_index()
    return perch, bysub


def plot_topo_aperiodic(perch: pd.DataFrame) -> None:
    """GA topomaps of Delta-exponent and Delta-offset (mean across subjects, per channel)."""
    ga = perch.groupby("channel")[["d_exp", "d_off"]].mean()
    info = None
    try:
        mont = mne.channels.make_standard_montage("standard_1020")
        chs = [c for c in ga.index if c in mont.ch_names]
        info = mne.create_info(chs, sfreq=250.0, ch_types="eeg")
        info.set_montage(mont, match_case=False, on_missing="ignore")
        ga = ga.loc[chs]
    except Exception as e:
        print(f"[1.2] topomap skipped (montage): {e}", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, ttl in ((axes[0], "d_exp", "Delta exponent (SCR - no-SCR)\n<0 = flattening"),
                          (axes[1], "d_off", "Delta offset (SCR - no-SCR)\n>0 = broadband power up")):
        v = ga[col].to_numpy()
        vlim = float(np.nanmax(np.abs(v)))
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-vlim, vlim), contours=4)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("1.2 Aperiodic topography (cache 05_04, fixed mode, per-channel GA, N=6)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "topo_aperiodic.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


# =============================================================================
# B) Peak-blind exponent test (recompute on global channel-mean spectra)
# =============================================================================
def linear_exponent(freqs, power, frange=(1.5, 30.0), mask=MASK_BANDS) -> float:
    f = np.asarray(freqs, float)
    lp = np.log10(np.asarray(power, float) + 1e-30)
    keep = (f >= frange[0]) & (f <= frange[1])
    for lo, hi in mask:
        keep &= ~((f >= lo) & (f < hi))
    if keep.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log10(f[keep]), lp[keep], 1)[0]
    return float(-slope)   # exponent = -slope of log-log


def fooof_exponent(freqs, power, frange) -> float:
    fm = FOOOF(aperiodic_mode="fixed", verbose=False, **FOOOF_KW)
    fm.fit(np.asarray(freqs, float), np.asarray(power, float), freq_range=list(frange))
    return float(fm.aperiodic_params_[1])


def peakblind_test() -> tuple[pd.DataFrame, dict]:
    rows = []
    spectra: dict = {}   # {sub: {cond: (freqs, db)}} channel-mean log-PSD in dB
    for sub in COHORT:
        print(f"[1.2] {sub}: building epochs for peak-blind ...", flush=True)
        real, silent = build_subject_epochs(sub)
        if real is None or silent is None:
            continue
        for ep in (real, silent):
            ep.drop_channels([c for c in DROP_CHANNELS if c in ep.ch_names])  # 29-ch set
        spectra[sub] = {}
        for cond, ep in (("real", real), ("silent", silent)):
            psd, freqs, ch = compute_psd(ep)
            power = psd.mean(axis=(0, 1))   # global channel-mean PSD (linear)
            spectra[sub][cond] = (freqs, 10.0 * np.log10(power + 1e-30))
            rows.append({
                "subject": sub, "condition": cond,
                "exp_full": fooof_exponent(freqs, power, (1.5, 40.0)),
                "exp_nogamma": fooof_exponent(freqs, power, (1.5, 30.0)),
                "exp_blind": linear_exponent(freqs, power, (1.5, 30.0), MASK_BANDS),
            })
    return pd.DataFrame(rows), spectra


BANDS_PLOT = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def _band_strip(ax, ymax):
    for b, (lo, hi) in BANDS_PLOT.items():
        if b in ("alpha",):
            ax.axvspan(lo, hi, alpha=0.10, color="gold")
        if b == "gamma":
            ax.axvspan(lo, hi, alpha=0.08, color="red")
        ax.axvline(lo, color="0.85", lw=0.6, zorder=0)
        ax.text(np.sqrt(lo * hi), ymax, b, ha="center", va="top", fontsize=8, color="0.45")


def plot_diff_spectrum(spectra: dict) -> None:
    """SCR - noSCR difference spectrum (dB), the INTERPRETIVE centerpiece of 1.2.
    Reading guide: positive everywhere = broadband OFFSET up; dips below the offset baseline at
    theta/alpha/beta = oscillatory DESYNC; delta ~ at baseline = nothing extra; gamma = TBD."""
    subs = [s for s in COHORT if s in spectra and "real" in spectra[s] and "silent" in spectra[s]]
    f = np.asarray(spectra[subs[0]]["real"][0], float)
    D = np.vstack([spectra[s]["real"][1] - spectra[s]["silent"][1] for s in subs])
    mean = D.mean(axis=0)
    sem = D.std(axis=0, ddof=1) / np.sqrt(len(subs))
    # offset baseline = median of the mean difference (a robust "broadband level" reference)
    off = float(np.median(mean))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # ---- left: mean +/- SEM, annotated ----
    ax = axes[0]
    ymax = float(np.nanmax(mean + sem)) + 0.5
    ax.fill_between(f, mean - sem, mean + sem, color="C0", alpha=0.25, label="+/- SEM (N=6)")
    ax.plot(f, mean, color="k", lw=2.6, label="mean SCR - no-SCR")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(off, color="green", ls="--", lw=1.3, label=f"broadband offset ~{off:.1f} dB")
    _band_strip(ax, ymax)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 13, 30, 40]); ax.set_xticklabels(["1", "2", "4", "8", "13", "30", "40"])
    ax.set_xlabel("Frequency (Hz, log)"); ax.set_ylabel("SCR - no-SCR  (dB)")
    ax.set_title("Mean difference +/- SEM -- the robust picture", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("positive everywhere\n= broadband OFFSET up", xy=(2, off), xytext=(1.2, off + 2.2),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))
    # dips below offset at theta-alpha-beta
    amask = (f >= 8) & (f < 13)
    ax.annotate("dips below offset at theta-alpha-beta\n= oscillatory DESYNC", xy=(10, mean[amask].mean()),
                xytext=(4.2, off - 3.0), fontsize=8, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))
    # ---- right: per-subject (heterogeneity) ----
    ax = axes[1]
    for s in subs:
        ax.plot(f, spectra[s]["real"][1] - spectra[s]["silent"][1],
                color=SUBJ_COLORS[s], lw=1.3, alpha=0.7, label=s)
    ax.axhline(0, color="0.6", lw=0.8)
    _band_strip(ax, ymax)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 13, 30, 40]); ax.set_xticklabels(["1", "2", "4", "8", "13", "30", "40"])
    ax.set_xlabel("Frequency (Hz, log)")
    ax.set_title("Per subject -- note the heterogeneity", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("1.2 Difference spectrum (channel-mean, N=6): broadband offset up + theta-alpha-beta "
                 "desync; delta ~flat; gamma TBD", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "diff_spectrum.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


def plot_exponent_fragile(df: pd.DataFrame, bysub: pd.DataFrame) -> None:
    """Show that the 1/f SLOPE change is NOT robust: it flips with method (left) and is
    heterogeneous across subjects (right). Replaces the misleading exponent topomap."""
    piv = df.pivot(index="subject", columns="condition")
    dfull = (piv["exp_full"]["real"] - piv["exp_full"]["silent"])
    dno = (piv["exp_nogamma"]["real"] - piv["exp_nogamma"]["silent"])
    dbl = (piv["exp_blind"]["real"] - piv["exp_blind"]["silent"])
    perch_mean = float(bysub["mean_dexp"].mean())   # per-channel cache mean (~-0.068)
    methods = ["per-channel\nFOOOF 1.5-40\n(05_04 method)", "channel-mean\nFOOOF 1.5-40",
               "channel-mean\nFOOOF 1.5-30\n(no gamma)", "channel-mean\npeak-blind"]
    vals = [perch_mean, float(dfull.mean()), float(dno.mean()), float(dbl.mean())]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    colors = ["firebrick" if v < 0 else "steelblue" for v in vals]
    ax.bar(range(len(methods)), vals, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("group-mean Delta exponent (SCR - no-SCR)")
    ax.set_title("The 1/f slope change FLIPS SIGN with method\n(<0 flattening = red; >0 = blue) "
                 "-> NOT robust", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:+.3f}", ha="center", fontsize=9)
    ax = axes[1]
    subs = list(dfull.index)
    ax.bar(range(len(subs)), bysub.set_index("subject").loc[subs, "mean_dexp"].to_numpy(),
           color=[SUBJ_COLORS[s] for s in subs], alpha=0.85)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(subs))); ax.set_xticklabels([s.replace("sub-", "") for s in subs])
    ax.set_ylabel("per-subject mean Delta exponent (per-channel)")
    ax.set_title("...and is HETEROGENEOUS across subjects\n(sub-24 reverses; effect driven by 19/27)",
                 fontsize=10)
    fig.suptitle("1.2 Why we do NOT claim a 1/f rotation: the aperiodic exponent is fragile at N=6",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "exponent_fragile.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


def plot_band_decomp(spectra: dict) -> None:
    """THE interpretive figure for 1.2, in the user's terms: decompose the SCR-noSCR difference
    into (a) OFFSET (broadband level) and (b) PERIODIC power per band (aperiodic-removed).
    Maps directly to: offset clear; theta/alpha/beta desync; delta ~nothing; gamma TBD."""
    bands = {"delta": (1.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
             "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}
    subs = [s for s in COHORT if s in spectra]
    rec: dict = {}
    for sub in subs:
        rec[sub] = {}
        for cond in ("real", "silent"):
            f, db = spectra[sub][cond]
            power = 10.0 ** (np.asarray(db, float) / 10.0)        # dB -> linear
            fm = FOOOF(aperiodic_mode="fixed", verbose=False, **FOOOF_KW)
            fm.fit(np.asarray(f, float), power, freq_range=[1.5, 40.0])
            flat = fm.power_spectrum - fm._ap_fit
            ff = fm.freqs
            d = {"offset": float(fm.aperiodic_params_[0])}
            for b, (lo, hi) in bands.items():
                bm = (ff >= lo) & (ff < hi)
                d[b] = float(np.mean(np.clip(flat[bm], 0, None))) if bm.any() else np.nan
            rec[sub][cond] = d

    # save per-subject deltas (offset + periodic per band) for inspection
    csv_rows = []
    for s in subs:
        row = {"subject": s, "d_offset": rec[s]["real"]["offset"] - rec[s]["silent"]["offset"]}
        for b in bands:
            row[f"d_pp_{b}"] = rec[s]["real"][b] - rec[s]["silent"][b]
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(TBL_DIR / "band_decomp.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # ---- A) offset ----
    ax = axes[0]
    doff = [rec[s]["real"]["offset"] - rec[s]["silent"]["offset"] for s in subs]
    ax.bar(range(len(subs)), doff, color=[SUBJ_COLORS[s] for s in subs], alpha=0.85)
    ax.axhline(0, color="k", lw=1.0)
    ax.axhline(float(np.mean(doff)), color="green", ls="--", lw=1.5,
               label=f"mean {np.mean(doff):+.2f}")
    ax.set_xticks(range(len(subs))); ax.set_xticklabels([s.replace("sub-", "") for s in subs])
    ax.set_ylabel("Delta offset (SCR - no-SCR)")
    ax.set_title(f"(a) Broadband OFFSET: {int(sum(d>0 for d in doff))}/{len(subs)} up", fontsize=11)
    ax.legend(fontsize=9)
    # ---- B) periodic per band ----
    ax = axes[1]
    bnames = list(bands.keys())
    for s in subs:
        ys = [rec[s]["real"][b] - rec[s]["silent"][b] for b in bnames]
        ax.plot(range(len(bnames)), ys, "-o", color=SUBJ_COLORS[s], alpha=0.55, label=s)
    grp = np.array([[rec[s]["real"][b] - rec[s]["silent"][b] for b in bnames] for s in subs])
    ax.plot(range(len(bnames)), np.nanmean(grp, axis=0), "-s", color="k", lw=2.5, ms=9, label="mean")
    for j, b in enumerate(bnames):
        col = grp[:, j][np.isfinite(grp[:, j])]
        updown = "↑" if np.nanmean(grp[:, j]) > 0 else "↓"
        n = int((col > 0).sum()) if updown == "↑" else int((col < 0).sum())
        ax.annotate(f"{n}/{col.size}{updown}", (j, np.nanmax(grp[:, j])), fontsize=8,
                    ha="center", va="bottom", color="0.3")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(bnames))); ax.set_xticklabels(bnames)
    ax.set_ylabel("Delta periodic power (SCR - no-SCR), aperiodic-removed")
    ax.set_title("(b) PERIODIC per band: delta UP (largest); theta/alpha/beta desync; gamma up",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("1.2 Decomposition of the SCR - no-SCR difference (channel-mean, N=6): "
                 "offset (broadband) + periodic per band", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "band_decomp.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


def plot_exponent_fits(spectra: dict) -> None:
    """Concrete view of WHY the 1/f slope is fragile: draw the fitted aperiodic LINES on the
    grand-average spectrum, for two fit ranges. With gamma the SCR line fits flatter; without
    gamma it fits steeper -> the SCR-noSCR slope difference flips sign with the fit range."""
    subs = [s for s in COHORT if s in spectra]
    f = np.asarray(spectra[subs[0]]["real"][0], float)
    ga = {c: np.mean([10.0 ** (np.asarray(spectra[s][c][1], float) / 10.0) for s in subs], axis=0)
          for c in ("real", "silent")}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, frange, sub_t in ((axes[0], (1.5, 40.0), "fit WITH gamma (1.5-40 Hz)"),
                              (axes[1], (1.5, 30.0), "fit WITHOUT gamma (1.5-30 Hz)")):
        exps = {}
        for cond, color, lab in (("real", "C3", "SCR"), ("silent", "0.45", "no-SCR")):
            db = 10.0 * np.log10(ga[cond] + 1e-30)
            ax.plot(f, db, color=color, lw=1.3, alpha=0.9, label=f"{lab} spectrum")
            fm = FOOOF(aperiodic_mode="fixed", verbose=False, **FOOOF_KW)
            fm.fit(f, ga[cond], freq_range=list(frange))
            ax.plot(fm.freqs, 10.0 * fm._ap_fit, color=color, ls="--", lw=2.6,
                    label=f"{lab} 1/f fit (exp={fm.aperiodic_params_[1]:.2f})")
            exps[cond] = fm.aperiodic_params_[1]
        dexp = exps["real"] - exps["silent"]
        ax.axvspan(frange[0], frange[1], color="0.5", alpha=0.05)
        ax.axvspan(30, 40, color="red", alpha=0.10 if frange[1] > 30 else 0.0)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 13, 30, 40]); ax.set_xticklabels(["1", "2", "4", "8", "13", "30", "40"])
        ax.set_xlabel("Frequency (Hz, log)")
        sign = "flatter in SCR (Δexp<0)" if dexp < 0 else "steeper in SCR (Δexp>0)"
        ax.set_title(f"{sub_t}\nΔexp(SCR-noSCR) = {dexp:+.3f}  →  {sign}", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Power (dB)")
    fig.suptitle("1.2 Why the 1/f slope is fragile: the fitted aperiodic lines (dashed) change with "
                 "the fit range.\nIncluding gamma (left, EMG-suspect) makes SCR fit flatter; excluding "
                 "it (right) flips the sign.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "exponent_fits.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


def _save_spectra(spectra: dict) -> None:
    subs = [s for s in COHORT if s in spectra]
    d = {"freqs": spectra[subs[0]]["real"][0]}
    for s in subs:
        d[f"{s}__real"] = spectra[s]["real"][1]
        d[f"{s}__silent"] = spectra[s]["silent"][1]
    np.savez(TBL_DIR / "channel_mean_spectra.npz", **d)


def _load_spectra() -> dict:
    z = np.load(TBL_DIR / "channel_mean_spectra.npz")
    freqs = z["freqs"]
    spectra: dict = {}
    for s in COHORT:
        if f"{s}__real" in z:
            spectra[s] = {"real": (freqs, z[f"{s}__real"]), "silent": (freqs, z[f"{s}__silent"])}
    return spectra


def peakblind_contrast(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot(index="subject", columns="condition")
    out = []
    for m in ["exp_full", "exp_nogamma", "exp_blind"]:
        d = (piv[m]["real"] - piv[m]["silent"]).dropna()
        out.append({"method": m, "mean_dexp": float(d.mean()),
                    "n_flat": int((d < 0).sum()), "n": int(d.size),
                    "subjects_flat": ",".join(sorted(d[d < 0].index.str.replace("sub-", "")))})
    return pd.DataFrame(out)


def plot_peakblind(df: pd.DataFrame) -> None:
    piv = df.pivot(index="subject", columns="condition")
    subs = [s for s in COHORT if s in piv.index]
    methods = ["exp_full", "exp_nogamma", "exp_blind"]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(methods))
    for sub in subs:
        ys = [piv[m]["real"][sub] - piv[m]["silent"][sub] for m in methods]
        ax.plot(x, ys, "-o", color=SUBJ_COLORS[sub], label=sub, alpha=0.8)
    means = [float((piv[m]["real"] - piv[m]["silent"]).mean()) for m in methods]
    ax.plot(x, means, "-s", color="k", lw=2.5, ms=10, label="mean")
    ax.axhline(0, color="0.5", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["FOOOF 1.5-40\n(full)", "FOOOF 1.5-30\n(no gamma)",
                        "linear blind\n(theta+alpha masked)"], fontsize=9)
    ax.set_ylabel("Delta exponent (SCR - no-SCR);  <0 = flattening")
    ax.set_title("1.2 PEAK-BLIND test: does the 1/f flattening survive masking the peaks?\n"
                 "if Delta stays <0 left-to-right -> genuinely aperiodic; if it rises to ~0 -> "
                 "driven by alpha/theta drop", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "peakblind.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[1.2] saved {out}", flush=True)


# =============================================================================
# C) Temporal bump check (cache periodic)
# =============================================================================
def temporal_bump() -> pd.DataFrame:
    p = pd.read_csv(PERIODIC_CACHE)
    temporal_keep = [c for c in TEMPORAL if c not in DROP_CHANNELS]   # drop bad channels
    p = p[(p["level"] == "channel") & (p["key"].isin(temporal_keep))]
    p = p[(p["cf"] >= 4.0) & (p["cf"] < 8.0)]   # theta-range peaks in Temporal channels
    rows = []
    for cond in ("real", "silent"):
        pc = p[p["condition"] == cond]
        n_sub = pc["subject"].nunique()
        rows.append({"condition": cond, "n_subjects_with_theta_peak_in_temporal": n_sub,
                     "n_peaks_total": len(pc),
                     "subjects": ",".join(sorted(pc["subject"].unique()))})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="regenerate figures from cached spectra npz + cache CSVs (no rebuild)")
    args = ap.parse_args()
    print("=" * 78)
    print("aperiodic_char :: 1.2 caracterizar 1/f + alfa + peak-blind + bump temporal")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)

    # A) reconciliation (cache)
    perch, bysub = reconcile_aperiodic()
    bysub.to_csv(TBL_DIR / "exponent_reconcile.csv", index=False)
    print("\n[A] Delta-exponent per-canal POR SUJETO (cache):")
    print(bysub.round(4).to_string(index=False), flush=True)
    print(f"[A] global: {int((perch['d_exp'] < 0).sum())}/{len(perch)} canales flatten, "
          f"mean Delta-exp={perch['d_exp'].mean():.4f}", flush=True)

    # C) temporal bump (cache)
    tb = temporal_bump()
    tb.to_csv(TBL_DIR / "temporal_bump.csv", index=False)
    print("\n[C] Bump temporal (picos theta 4-8 Hz en ROI Temporal, por condicion):")
    print(tb.to_string(index=False), flush=True)

    # B) peak-blind: recompute or load cached spectra
    if args.replot:
        df = pd.read_csv(TBL_DIR / "peakblind.csv")
        spectra = _load_spectra()
    else:
        df, spectra = peakblind_test()
        df.to_csv(TBL_DIR / "peakblind.csv", index=False)
        _save_spectra(spectra)
    con = peakblind_contrast(df)
    print("\n[B] PEAK-BLIND -- Delta exponent (SCR-noSCR) por metodo:")
    print(con.to_string(index=False), flush=True)

    # figures that actually interpret 1.2
    plot_band_decomp(spectra)            # CENTERPIECE: offset + periodic per band (user's terms)
    plot_diff_spectrum(spectra)          # raw evidence: where the difference lives
    plot_exponent_fits(spectra)          # concrete: fitted 1/f lines flip with fit range


if __name__ == "__main__":
    main()
