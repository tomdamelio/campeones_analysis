"""R3.1 paso 1 — Relación graduada: ¿la desync alfa PERIÓDICA escala con la amplitud del SCR?

Bloque 3 reconstruido sobre la desync alfa (la única señal cortical que sobrevivió el strict-test;
las subidas gamma/delta son artefacto). El band_scr_amplitude_scr original usó potencia CRUDA y no
halló que alfa escalara (solo delta-PO, ahora artefacto). Acá se re-hace con la métrica correcta:
**alfa periódica (1/f removido), posterior** = la señal de desync validada.

Paso 1 (este script): (a) distribución de la amplitud SCR (lo que pidió Diego, antes de modelar);
(b) test graduado central — within-subject Spearman(alfa-PO periódica, amplitud SCR), signo 6/6.
Predicción: NEGATIVO (SCR más grande -> más desync -> menos alfa). Se reportan también theta/beta
(desync coherente). Paso 2 (luego): GLM Tweedie + control fuerte (mediador cierre-de-ojos→alfa).

Solo épocas REALES (cada una con su amplitud). 29 ch (drop FC1/TP10/Fz). Stopgap pre-Track B.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.scr_amplitude_glm
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

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, DROP_CHANNELS, REPO, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, _linear_aperiodic
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

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "3_1_alpha_amplitude"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

DESYNC_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
POSTERIOR = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]   # alfa posterior (set del strict-test)


def _periodic_posterior(psd, freqs, ch, band, post_idx):
    _, _, resid, f = _linear_aperiodic(psd, freqs, RANGES["1-30"])
    lo, hi = band
    m = (f >= lo) & (f < hi)
    a = np.clip(resid[:, :, m], 0, None).mean(axis=2)        # (n_ep, n_ch) periódico
    return a[:, post_idx].mean(axis=1)                        # (n_ep,) posterior channel-mean


def subject_data(sub):
    """Por época REAL: amplitud SCR + alfa/theta/beta periódico posterior. 29 ch."""
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    amp_all, band_all = [], {b: [] for b in DESYNC_BANDS}
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
        onsets = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
        if onsets.size == 0:
            continue
        amp = []
        for t in onsets:
            i0, i1 = int(round(t * EDA_FS)), int(round((t + POST_S) * EDA_FS))
            amp.append(float(np.max(eda[i0:i1])) if i1 > i0 and i1 <= len(eda) else np.nan)
        ep = epoch_one_run(raw, onsets, code=1)
        if ep is None or len(ep) == 0:
            continue
        psd, freqs, ch = compute_psd(ep)
        post_idx = [ch.index(c) for c in POSTERIOR if c in ch]
        amp_all += amp[:len(ep)]
        for b, band in DESYNC_BANDS.items():
            band_all[b] += list(_periodic_posterior(psd, freqs, ch, band, post_idx))
    amp = np.asarray(amp_all, float)
    if amp.size < 10:
        return None
    out = {"amp": amp}
    for b in DESYNC_BANDS:
        out[b] = np.asarray(band_all[b], float)
    return out


def main():
    print("=" * 78)
    print("scr_amplitude_glm :: R3.1 paso 1 — desync alfa periódica vs amplitud SCR (graded)")
    print("=" * 78, flush=True)

    data = {}
    for sub in COHORT:
        d = subject_data(sub)
        if d is None:
            print(f"  {sub}: insuficiente", flush=True); continue
        data[sub] = d
        print(f"  {sub}: n_real={len(d['amp'])}", flush=True)

    # --- (a) distribución de la amplitud SCR ---
    rows_amp = []
    for sub, d in data.items():
        a = d["amp"][np.isfinite(d["amp"])]
        rows_amp.append(dict(subject=sub, n=len(a), mean=round(float(a.mean()), 4),
                             median=round(float(np.median(a)), 4),
                             skew=round(float(stats.skew(a)), 2), kurt=round(float(stats.kurtosis(a)), 2)))
    pd.DataFrame(rows_amp).to_csv(TBL_DIR / "scr_amplitude_dist.csv", index=False)

    # --- (b) test graduado: within-subject Spearman(periódico posterior, amplitud) ---
    rows = []
    for sub, d in data.items():
        for b in DESYNC_BANDS:
            mask = np.isfinite(d["amp"]) & np.isfinite(d[b])
            rho, p = stats.spearmanr(d["amp"][mask], d[b][mask])
            rows.append(dict(subject=sub, band=b, n=int(mask.sum()),
                             rho=round(float(rho), 3), p=round(float(p), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "graded_alpha_perSubject.csv", index=False)

    print("\n=== Test graduado (Spearman periódico-posterior vs amplitud SCR) ===")
    print("  (NEGATIVO 6/6 = más desync con SCR mayor = readout graduado)")
    summ = []
    for b in DESYNC_BANDS:
        rr = df[df.band == b]["rho"].to_numpy(float)
        z = np.arctanh(np.clip(rr, -0.999, 0.999))
        t, pt = stats.ttest_1samp(z, 0.0)
        n_neg = int((rr < 0).sum())
        summ.append(dict(band=b, mean_rho=round(float(rr.mean()), 3), n_neg=f"{n_neg}/{len(rr)}",
                         p_group=round(float(pt), 4)))
        print(f"  {b:6s}: mean_rho={rr.mean():+.3f}  neg={n_neg}/{len(rr)}  p_group={pt:.3f}  "
              f"por_suj={[f'{x:+.2f}' for x in rr]}", flush=True)
    pd.DataFrame(summ).to_csv(TBL_DIR / "graded_alpha_group.csv", index=False)

    _plot_dist(data)
    _plot_graded(data, df)
    print(f"\nOutputs -> {OUT_DIR}\n[R3.1 paso 1] done", flush=True)


def _plot_dist(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (sub, d) in zip(axes.ravel(), data.items()):
        a = d["amp"][np.isfinite(d["amp"])]
        ax.hist(a, bins=30, color=SUBJ_COLORS.get(sub, "C0"), alpha=0.8)
        ax.set_title(f"{sub}  n={len(a)}  skew={stats.skew(a):+.1f}", fontsize=9)
        ax.set_xlabel("amplitud SCR (phasic peak)")
    fig.suptitle("R3.1 Distribución de la amplitud del SCR por sujeto (Y, antes de modelar)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "scr_amplitude_dist.png", dpi=120); plt.close(fig)


def _plot_graded(data, df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (sub, d) in zip(axes.ravel(), data.items()):
        x, y = d["amp"], d["alpha"]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, alpha=0.4, color=SUBJ_COLORS.get(sub, "C0"))
        if m.sum() > 3:
            b1, b0 = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.5)
        rho = df[(df.subject == sub) & (df.band == "alpha")]["rho"].values[0]
        ax.set_title(f"{sub}  alpha-PO periódico  rho={rho:+.2f}", fontsize=9)
        ax.set_xlabel("amplitud SCR"); ax.set_ylabel("alfa periódica posterior")
    fig.suptitle("R3.1 Graded: alfa PERIÓDICA posterior por época vs amplitud SCR\n"
                 "(pendiente NEGATIVA = más desync con SCR mayor = readout cortical graduado)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "graded_alpha_scatter.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
