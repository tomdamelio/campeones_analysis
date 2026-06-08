"""2.4 — CSD-survival del contraste edge/central (control de confound muscular).

Bloque 2, frente-gamma. Complementa edge_central_scr.py (covariado por época) con el control
de surface Laplacian (CSD) pedido por el diario: ¿el aumento de gamma en SCR SOBREVIVE el CSD
en los canales CENTRALES (= candidato neuronal focal) o CAE (= componente global/volume-
conducido = EMG del borde)?

Lógica (Fitzgibbon 2013): el Laplacian de superficie suprime el componente global/far-field
(volume-conduction) y realza fuentes focales. El EMG es far-field desde el rim muscular -> en
canales centrales el CSD lo atenúa. Entonces:
  - gamma central SOBREVIVE CSD (sigue +, consistente 6/6) -> fuente central focal = neuronal.
  - gamma central CAE con CSD (->0 o negativo) -> era volume-conducción del borde = EMG.
Caveat (scr_band_csd): post-CSD el CSD AMPLIFICA el ruido local de alta freq espacial (EMG) en
canales de BORDE -> un 'hit' de gamma en edge post-CSD es esperable bajo EMG, no informativo;
lo decisivo es el comportamiento en CENTRAL.

Controles: delta (1-4, la pro-señal: debería sobrevivir central, link con 2.6) y broadband
(1-40, contexto). Ventana POST 0-3 s, consistente con topo_variance_scr. 29 ch (drop FC1/TP10/Fz).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.edge_central_csd
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import compute_current_source_density

from src.campeones_analysis.multimodal_arousal.cohort import CENTRAL, DROP_CHANNELS, EMG_EDGE, SUBJ_COLORS
from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import (
    PSD_FMAX,
    PSD_FMIN,
    PSD_NFFT,
    build_subject_epochs,
)
from src.campeones_analysis.multimodal_arousal.topomap_delta_theta_scr import channel_band_diff_db

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT.parent.parent / "05_05" / "2_4_edge_central"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS: dict[str, tuple[float, float]] = {
    "gamma": (30.0, 40.0),     # el crux
    "delta": (1.0, 4.0),       # pro-señal (control positivo, link 2.6)
    "broadband": (1.0, 40.0),  # contexto
}
POST_TMIN, POST_TMAX = 0.0, 3.0


def _region_mean(diff_map: np.ndarray, ch_names: list[str], region: list[str]) -> float:
    idx = [ch_names.index(c) for c in region if c in ch_names]
    return float(np.mean(diff_map[idx])) if idx else float("nan")


def _psd(epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Welch PSD con picks='data' (sirve para canales eeg Y csd, que usan unidad V/m²)."""
    sp = epochs.compute_psd(method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX, n_fft=PSD_NFFT,
                            picks="data", verbose="ERROR")
    return sp.get_data(), sp.freqs, sp.ch_names


def _diffs(real_ep, silent_ep) -> tuple[dict[str, np.ndarray], list[str]]:
    """SCR-noSCR dB diff por banda y canal sobre las épocas dadas (raw o CSD)."""
    psd_r, freqs, ch = _psd(real_ep)
    psd_s, _, _ = _psd(silent_ep)
    return {bn: channel_band_diff_db(psd_r, psd_s, freqs, rng) for bn, rng in BANDS.items()}, ch


def _subject(sub: str) -> dict | None:
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None:
        return None
    for ep in (real_ep, silent_ep):
        ep.drop_channels([c for c in DROP_CHANNELS if c in ep.ch_names])
    real_post = real_ep.copy().crop(POST_TMIN, POST_TMAX)
    silent_post = silent_ep.copy().crop(POST_TMIN, POST_TMAX)

    # --- RAW (referencia promedio) ---
    diff_raw, ch = _diffs(real_post, silent_post)

    # --- CSD (surface Laplacian) ---
    real_csd = compute_current_source_density(real_post.copy())
    silent_csd = compute_current_source_density(silent_post.copy())
    diff_csd, _ = _diffs(real_csd, silent_csd)

    row: dict = dict(subject=sub, n_real=len(real_post), n_silent=len(silent_post))
    for bn in BANDS:
        row[f"{bn}_central_raw"] = _region_mean(diff_raw[bn], ch, CENTRAL)
        row[f"{bn}_edge_raw"] = _region_mean(diff_raw[bn], ch, EMG_EDGE)
        row[f"{bn}_central_csd"] = _region_mean(diff_csd[bn], ch, CENTRAL)
        row[f"{bn}_edge_csd"] = _region_mean(diff_csd[bn], ch, EMG_EDGE)
    print(f"  {sub}: n_real={len(real_post)} n_silent={len(silent_post)}  "
          f"gamma central raw={row['gamma_central_raw']:+.2f}->csd={row['gamma_central_csd']:+.2f}  "
          f"edge raw={row['gamma_edge_raw']:+.2f}->csd={row['gamma_edge_csd']:+.2f}", flush=True)
    return row


def _plot(df: pd.DataFrame, subjects: list[str]) -> None:
    fig, axes = plt.subplots(1, len(BANDS), figsize=(5.2 * len(BANDS), 5.0))
    for ax, bn in zip(np.atleast_1d(axes), BANDS):
        conds = [f"{bn}_central_raw", f"{bn}_central_csd", f"{bn}_edge_raw", f"{bn}_edge_csd"]
        labels = ["central\nraw", "central\nCSD", "edge\nraw", "edge\nCSD"]
        x = np.arange(len(conds))
        # central raw/CSD = azul claro/oscuro; edge raw/CSD = rojo claro/oscuro
        ax.bar(x, [df[c].mean() for c in conds],
               color=["#9ecae1", "#3182bd", "#fc9272", "#de2d26"], zorder=1)
        for k, sub in enumerate(subjects):
            r = df[df["subject"] == sub]
            if not len(r):
                continue
            ax.plot(x, [r[c].iloc[0] for c in conds], "-o", color=SUBJ_COLORS[sub], lw=1, ms=4,
                    alpha=0.8, zorder=3)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("SCR-noSCR (dB)")
        ax.set_title(bn, fontsize=11)
    fig.suptitle("2.4 CSD-survival: ¿el contraste central sobrevive el surface Laplacian?\n"
                 "(central CSD sigue + = fuente focal neuronal · cae = volume-conducción/EMG)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "edge_central_csd.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("edge_central_csd :: 2.4 CSD-survival del contraste edge/central")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    print(f"  CENTRAL ({len(CENTRAL)}): {CENTRAL}")
    print(f"  EMG_EDGE ({len(EMG_EDGE)}): {EMG_EDGE}", flush=True)

    rows = [r for r in (_subject(s) for s in SUBJECTS) if r is not None]
    if not rows:
        print("No subjects.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "edge_central_csd.csv", index=False)

    subjects = list(df["subject"])
    print("\n=== CSD-survival GA (N={}) ===".format(len(df)))
    for bn in BANDS:
        cr, cc = df[f"{bn}_central_raw"], df[f"{bn}_central_csd"]
        er, ec = df[f"{bn}_edge_raw"], df[f"{bn}_edge_csd"]
        n_central_survive = int((cc > 0).sum())
        print(f"  {bn:9s}: CENTRAL raw {cr.mean():+.2f} -> CSD {cc.mean():+.2f} "
              f"(>0 en {n_central_survive}/{len(df)})   "
              f"EDGE raw {er.mean():+.2f} -> CSD {ec.mean():+.2f}", flush=True)

    _plot(df, subjects)
    print(f"\nTabla -> {TBL_DIR / 'edge_central_csd.csv'}")
    print(f"Figura -> {FIG_DIR / 'edge_central_csd.png'}")
    print("[2.4 CSD] done", flush=True)


if __name__ == "__main__":
    main()
