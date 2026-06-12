"""Bloque 4.3 — Caracterización espectral evento-vs-calma del joystick (pre-shift).

Lee el cache de `joystick_panel.py` y contrasta, por sujeto / ROI / banda, la potencia PERIÓDICA
(1/f removido) entre épocas de evento (cambio afectivo abrupto) y de calma. Aplica la REGLA de
decisión heredada del cierre SCR:

  - Una CAÍDA de potencia periódica (desincronización, p.ej. alfa posterior) = candidato a señal
    cortical genuina: un artefacto ADITIVO no puede FABRICAR una caída.
  - Una SUBIDA consistente (gamma/delta/broadband-offset) = SOSPECHA DE ARTEFACTO (EMG/movimiento).
    Si una SUBIDA es consistente (>=5/6 sujetos), se marca y se FRENA para auditar antes de
    interpretarla como señal.

Métricas por banda: potencia periódica (delta/theta/alpha/beta/gamma) + aperiódico (offset =
nivel broadband; exponent = pendiente 1/f). offset-UP = firma aditiva; alpha-DOWN = desync.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_spectral --dim arousal
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.campeones_analysis.multimodal_arousal.joystick_panel import OUT_DIR, load_cache
from src.campeones_analysis.multimodal_arousal.decoding_sweep import _linear_aperiodic, BANDS_40
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import ROIS as _ROIS

warnings.filterwarnings("ignore")

FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

FULL_RANGE = (1.0, 40.0)
# ROIs de tfr_psd_scr + Posterior (parietal+occipital) donde vive el desync alfa.
ROIS = dict(_ROIS)
ROIS["Posterior"] = sorted(set(_ROIS["Parietal"]) | set(_ROIS["Occipital"]))


def _periodic_byband(psd, freqs):
    """Potencia periódica (1/f removido) por banda, conservando canales: {band: (n_ep, n_ch)}."""
    _, _, resid, f = _linear_aperiodic(psd, freqs, FULL_RANGE)
    out = {}
    for bn, (a, b) in BANDS_40.items():
        m = (f >= a) & (f < b)
        out[bn] = np.clip(resid[:, :, m], 0, None).mean(axis=2)
    return out


def _aperiodic(psd, freqs):
    off, exp, _, _ = _linear_aperiodic(psd, freqs, FULL_RANGE)
    return {"offset": off, "exponent": exp}


def _roi_idx(ch):
    return {roi: [ch.index(c) for c in chs if c in ch] for roi, chs in ROIS.items()}


def characterize(dim_tag, lag=None):
    """lag=None -> ventana pre-shift del cache de joystick_panel. lag=<int> -> ventana del lag i del
    cache de joystick_lag_sweep (p.ej. 4 = post-shift [-0.5,3.0]) para caracterizar la señal motor."""
    dim = None if dim_tag == "combined" else dim_tag
    if lag is None:
        data, freqs, ch, _ = load_cache("matched", dim=dim)
        suffix = dim_tag
    else:
        import numpy as _np
        from src.campeones_analysis.multimodal_arousal.joystick_lag_sweep import CACHE as LAG_CACHE, load_lag
        z = _np.load(LAG_CACHE, allow_pickle=True)
        data, freqs, ch = load_lag(z, lag, dim=dim, min_per_class=1)
        suffix = f"{dim_tag}_lag{lag}"
    dim_tag = suffix
    roi_idx = _roi_idx(ch)
    metrics = list(BANDS_40.keys()) + ["offset", "exponent"]

    rows = []
    diff_spec = {roi: [] for roi in ROIS}  # (event-calma) mean spectrum per subject per ROI
    ev_spec = {roi: [] for roi in ROIS}
    ca_spec = {roi: [] for roi in ROIS}
    for sub, (psd, y, tn) in data.items():
        pb = _periodic_byband(psd, freqs)
        ap = _aperiodic(psd, freqs)
        feats = {**pb, **ap}
        ev = y == 1
        ca = y == 0
        for roi, idx in roi_idx.items():
            if not idx:
                continue
            for met in metrics:
                v = feats[met][:, idx].mean(axis=1)  # per-epoch ROI mean
                de = float(v[ev].mean()); dc = float(v[ca].mean())
                rows.append(dict(subject=sub, roi=roi, metric=met,
                                 mean_event=de, mean_calma=dc, delta=de - dc))
            # raw log-power diff spectrum per ROI
            lp = np.log10(psd[:, idx, :].mean(axis=1) + 1e-30)  # (n_ep, n_freq)
            ev_spec[roi].append(lp[ev].mean(axis=0))
            ca_spec[roi].append(lp[ca].mean(axis=0))
            diff_spec[roi].append(lp[ev].mean(axis=0) - lp[ca].mean(axis=0))

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / f"spectral_{dim_tag}.csv", index=False)

    # group-level: sign across subjects + wilcoxon on delta
    grp = []
    for roi in ROIS:
        for met in metrics:
            d = df[(df.roi == roi) & (df.metric == met)]["delta"].values
            if len(d) < 2:
                continue
            n_neg = int((d < 0).sum()); n_pos = int((d > 0).sum())
            try:
                p = float(wilcoxon(d)[1])
            except Exception:
                p = np.nan
            md = float(np.mean(d))
            if met in ("offset",):
                verdict = "broadband-UP (aditivo?)" if n_pos >= 5 else ("broadband-DOWN" if n_neg >= 5 else "mixto")
            else:
                verdict = ("DESYNC (caída -> candidato)" if n_neg >= 5
                           else ("RISE (SOSPECHA ARTEFACTO)" if n_pos >= 5 else "mixto"))
            grp.append(dict(roi=roi, metric=met, n=len(d), n_neg=n_neg, n_pos=n_pos,
                            mean_delta=round(md, 4), wilcoxon_p=round(p, 4), verdict=verdict))
    gdf = pd.DataFrame(grp)
    gdf.to_csv(TBL_DIR / f"spectral_group_{dim_tag}.csv", index=False)

    # ---- figure: per-ROI per-band group delta + subject dots ----
    n_roi = len(ROIS)
    fig, axes = plt.subplots(1, n_roi, figsize=(3.4 * n_roi, 4.6), sharey=True)
    for ax, roi in zip(np.atleast_1d(axes), ROIS):
        x = np.arange(len(metrics))
        md = [gdf[(gdf.roi == roi) & (gdf.metric == m)]["mean_delta"].values for m in metrics]
        md = [float(v[0]) if len(v) else np.nan for v in md]
        colors = ["C3" if (gdf[(gdf.roi == roi) & (gdf.metric == m)]["verdict"].astype(str)
                           .str.contains("ARTEFACTO").any()) else "C0" for m in metrics]
        ax.bar(x, md, color=colors, alpha=0.6)
        for xi, m in enumerate(metrics):
            d = df[(df.roi == roi) & (df.metric == m)]["delta"].values
            ax.scatter(np.full(len(d), xi) + np.linspace(-0.18, 0.18, len(d)), d,
                       s=22, color="k", alpha=0.6, zorder=3)
        ax.axhline(0, color="0.4", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=55, ha="right", fontsize=7)
        ax.set_title(roi, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Δ (evento − calma), log-power periódico")
    fig.suptitle(f"4.3 Joystick {dim_tag}: caída=desync (candidato), subida=sospecha artefacto "
                 "(rojo); puntos=6 sujetos", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / f"spectral_delta_{dim_tag}.png", dpi=120); plt.close(fig)

    # ---- figure: diff spectrum per ROI ----
    fig, axes = plt.subplots(1, n_roi, figsize=(3.4 * n_roi, 4.0), sharey=True)
    for ax, roi in zip(np.atleast_1d(axes), ROIS):
        if not diff_spec[roi]:
            continue
        D = np.array(diff_spec[roi])  # (n_sub, n_freq)
        m = D.mean(axis=0); sd = D.std(axis=0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.zeros_like(m)
        ax.fill_between(freqs, m - sd, m + sd, color="C0", alpha=0.25, lw=0)
        ax.plot(freqs, m, color="C0", lw=1.6)
        ax.axhline(0, color="0.4", lw=1)
        for a, b in [(8, 13), (30, 40)]:
            ax.axvspan(a, b, color="0.85", alpha=0.5, zorder=0)
        ax.set_xlabel("Hz"); ax.set_title(roi, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("log-power evento − calma (dB-ish)")
    fig.suptitle(f"4.3 Espectro diferencia evento−calma {dim_tag} (gris=alfa 8-13 / gamma 30-40)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / f"diff_spectrum_{dim_tag}.png", dpi=120); plt.close(fig)

    # ---- console verdict ----
    print("=" * 78)
    print(f"4.3 caracterización espectral — {dim_tag}  ({len(data)} sujetos)")
    print("=" * 78)
    print(gdf.to_string(index=False))
    rises = gdf[gdf.verdict.str.contains("ARTEFACTO")]
    desyncs = gdf[gdf.verdict.str.contains("DESYNC")]
    print("\n--- LECTURA ---")
    if len(desyncs):
        for _, r in desyncs.iterrows():
            print(f"  CANDIDATO desync: {r['roi']}/{r['metric']} "
                  f"({r['n_neg']}/{r['n']} caída, p={r['wilcoxon_p']})")
    if len(rises):
        print("  *** SUBIDAS CONSISTENTES (FRENAR / auditar artefacto): ***")
        for _, r in rises.iterrows():
            print(f"  RISE: {r['roi']}/{r['metric']} ({r['n_pos']}/{r['n']} subida, p={r['wilcoxon_p']})")
    else:
        print("  Sin subidas consistentes (ninguna banda con >=5/6 RISE).")
    print(f"\n-> {FIG_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", choices=["arousal", "valence", "combined"], default="arousal")
    ap.add_argument("--lag", type=int, default=None,
                    help="índice de lag del cache joystick_lag_sweep (4 = post-shift [-0.5,3])")
    args = ap.parse_args()
    characterize(args.dim, lag=args.lag)


if __name__ == "__main__":
    main()
