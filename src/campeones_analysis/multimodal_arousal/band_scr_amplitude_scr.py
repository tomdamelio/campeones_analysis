"""#2 — Acoplamiento GRADUADO: ¿la potencia por banda escala con la AMPLITUD del SCR? (2026-06-03).

En vez del binario SCR-vs-no-SCR, usa la amplitud continua del SCR. Por época-real: amplitud SCR
(pico del phasic post-onset) y potencia log por banda (5) por ROI (parieto-occipital y edge/
temporal). Within-subject: Spearman r (potencia vs amplitud). Cross-subject: 6 r por banda×ROI ->
mean r, conteo de signo, t de una muestra sobre Fisher-z.

Predicciones: si alfa-PO desincroniza en proporción al SCR -> r negativo consistente (rescataría
el alfa que el binario no halló). Si gamma/broadband-edge sube con la amplitud -> r positivo (EMG
dose-response). Test más sensible y mecanístico que el binario; puente a C.

Solo usa épocas REALES (los silents no entran) -> independiente del esquema de silent.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.band_scr_amplitude_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    POST_S,
    TMAX,
    TMIN,
    attach_montage_and_drop_no_pos,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import compute_psd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "qa_artifact_vs_signal" / "scr_amplitude"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}
ROIS = {"parieto-occipital": ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"],
        "edge/temporal": ["FT9", "TP9", "T7", "T8", "P7", "P8"]}


def subject_amp_power(sub):
    """Per real epoch: SCR amplitude + log band-power per (band, ROI). Returns dict or None."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    amp_all, bp_all = [], {(b, r): [] for b in BANDS for r in ROIS}
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
            onsets = filter_clean_onsets(detect_scr_onsets_s(eda, EDA_FS)[
                detect_scr_onsets_s(eda, EDA_FS) < dur], eda, EDA_FS)
            valid = (onsets + TMIN > 0) & (onsets + TMAX < dur)
            onsets = onsets[valid]
            if onsets.size == 0:
                continue
            # SCR amplitude = max phasic in [onset, onset+POST_S] (phasic ~0 at onset)
            amp = []
            for t in onsets:
                i0 = int(round(t * EDA_FS)); i1 = int(round((t + POST_S) * EDA_FS))
                amp.append(float(np.max(eda[i0:i1])) if i1 > i0 and i1 <= len(eda) else np.nan)
            ep = epoch_one_run(raw, onsets, code=1)
            if ep is None or len(ep) != len(onsets):
                # align: rebuild amplitude only for kept events (epoch_one_run re-filters identically)
                if ep is None:
                    continue
            psd, freqs, ch = compute_psd(ep)  # (n_ep, n_ch, n_freq)
            amp_all += amp[:len(ep)]
            for (b, (lo, hi)) in BANDS.items():
                fm = (freqs >= lo) & (freqs < hi)
                for r, chs in ROIS.items():
                    idx = [ch.index(c) for c in chs if c in ch]
                    bp = np.log10(psd[:, idx][:, :, fm].mean(axis=(1, 2)) + 1e-30)  # (n_ep,)
                    bp_all[(b, r)] += list(bp)
        except Exception as e:
            print(f"  {label}: FAILED -- {e}", flush=True)
    amp = np.asarray(amp_all, float)
    if amp.size < 10:
        return None
    out = {"amp": amp}
    for k, v in bp_all.items():
        out[k] = np.asarray(v, float)
    return out


def main():
    print("=" * 78)
    print("band_scr_amplitude :: Spearman(potencia banda/ROI, amplitud SCR) por sujeto")
    print("=" * 78, flush=True)
    data = {}
    rows = []
    for sub in COHORT:
        d = subject_amp_power(sub)
        if d is None:
            print(f"  {sub}: insuficiente", flush=True); continue
        data[sub] = d
        n = len(d["amp"])
        for b in BANDS:
            for r in ROIS:
                mask = np.isfinite(d["amp"]) & np.isfinite(d[(b, r)])
                rho, p = stats.spearmanr(d["amp"][mask], d[(b, r)][mask])
                rows.append(dict(subject=sub, n=int(mask.sum()), band=b, roi=r,
                                 rho=round(float(rho), 3), p=round(float(p), 4)))
        a = [x for x in rows if x["subject"] == sub and x["band"] == "alpha" and x["roi"] == "parieto-occipital"][0]
        g = [x for x in rows if x["subject"] == sub and x["band"] == "gamma" and x["roi"] == "edge/temporal"][0]
        print(f"  {sub}: n={n}  alpha-PO rho={a['rho']:+.3f}(p={a['p']:.3f})  "
              f"gamma-edge rho={g['rho']:+.3f}(p={g['p']:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "scr_amplitude_perSubject.csv", index=False)

    # cross-subject summary per band x ROI: mean rho, sign count, one-sample t on Fisher-z
    summ = []
    for b in BANDS:
        for r in ROIS:
            rr = df[(df.band == b) & (df.roi == r)]["rho"].to_numpy(float)
            z = np.arctanh(np.clip(rr, -0.999, 0.999))
            t, pt = stats.ttest_1samp(z, 0.0)
            summ.append(dict(band=b, roi=r, mean_rho=round(float(rr.mean()), 3),
                             n_neg=int((rr < 0).sum()), n_pos=int((rr > 0).sum()),
                             t=round(float(t), 2), p_group=round(float(pt), 4)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(TBL_DIR / "scr_amplitude_group.csv", index=False)

    # figure 1: scatter alpha-PO power vs SCR amplitude per subject
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, sub in zip(axes.ravel(), list(data)):
        d = data[sub]
        x, y = d["amp"], d[("alpha", "parieto-occipital")]
        ax.scatter(x, y, s=10, alpha=0.4, color=SUBJ_COLORS.get(sub, "C0"))
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 3:
            b1, b0 = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5)
            rho = df[(df.subject == sub) & (df.band == "alpha") & (df.roi == "parieto-occipital")]["rho"].values[0]
            ax.set_title(f"{sub}  alpha-PO  rho={rho:+.2f}", fontsize=9)
        ax.set_xlabel("amplitud SCR (phasic peak)"); ax.set_ylabel("log alpha PO power")
    fig.suptitle("Acoplamiento graduado: potencia alfa PO por época vs amplitud del SCR (real). "
                 "Negativo = más desync con SCR mayor.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "scr_amplitude_alphaPO_scatter.png", dpi=130)
    plt.close(fig)

    # figure 2: heatmap of mean rho (band x ROI) + per-subject strip for alpha-PO & gamma-edge
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    piv = sdf.pivot(index="band", columns="roi", values="mean_rho").reindex(list(BANDS))
    im = axes[0].imshow(piv.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    axes[0].set_xticks(range(len(piv.columns))); axes[0].set_xticklabels(piv.columns, fontsize=9)
    axes[0].set_yticks(range(len(piv.index))); axes[0].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, f"{piv.values[i,j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="mean Spearman rho")
    axes[0].set_title("mean rho (potencia vs amplitud SCR) por banda×ROI")
    # per-subject strip for the two key cells
    for k, (b, r, c) in enumerate([("alpha", "parieto-occipital", "C0"), ("gamma", "edge/temporal", "C3")]):
        vals = df[(df.band == b) & (df.roi == r)]["rho"].to_numpy(float)
        axes[1].scatter(np.full(len(vals), k) + np.linspace(-0.1, 0.1, len(vals)), vals, color=c, s=40)
        axes[1].plot([k - 0.2, k + 0.2], [vals.mean(), vals.mean()], color="k", lw=2)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["alpha\nPO", "gamma\nedge"])
    axes[1].set_ylabel("Spearman rho por sujeto"); axes[1].set_title("celdas clave: alfa-PO vs gamma-edge")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scr_amplitude_summary.png", dpi=130)
    plt.close(fig)

    print("\n--- group (band x ROI) ---")
    print(sdf.to_string(index=False), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
