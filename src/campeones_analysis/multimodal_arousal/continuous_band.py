"""Fundación para el TRF continuo (C1): construye envelopes de EEG por banda (delta/alfa/gamma)
parieto-occipital, continuas, alineadas a la EDA phasic continua. No existía EEG continua en el repo.

Pipeline por run (reusa erp_scr): preproc vhdr -> pick eeg -> montage -> por banda: filtro banda ->
Hilbert envelope a sr NATIVO (evita aliasing) -> log10 -> resample a 25 Hz. EDA phasic (50 Hz, del
{sub}_continuous.npz) -> 25 Hz. Ambos parten de t=0 del run; se truncan a min(dur) y se recortan 1 s
de borde (transitorios). Devuelve por run: env por banda (PO y all-channel) + eda, en 25 Hz alineados.

Run (sanity de alineación):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.continuous_band
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    attach_montage_and_drop_no_pos,
    run_label,
    runs_for,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

COMMON_FS = 12.5  # señales lentas (EDA<2Hz, envelope delta lento) -> 12.5 Hz sobra y es rápido
SMNA_SMOOTH_S = 0.25  # suavizado gaussiano del SMNA esparso antes de decimar (rate-like)
BANDS = {"delta": (1.0, 4.0), "alpha": (8.0, 13.0), "gamma": (30.0, 40.0)}
PARIETOOCCIPITAL = ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"]
EDGE_TRIM_S = 1.0
FIG_DIR = OUT / "continuous_bands" / "trf" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def band_envelopes(raw) -> tuple[dict, list[str], float]:
    """Per band: log Hilbert envelope per channel at native sr. Returns (env_by_band, ch_names, sfreq)."""
    raw = raw.copy().pick("eeg")
    attach_montage_and_drop_no_pos(raw)
    sfreq = float(raw.info["sfreq"])
    ch = list(raw.ch_names)
    out = {}
    for b, (lo, hi) in BANDS.items():
        r = raw.copy().filter(l_freq=lo, h_freq=hi, verbose="ERROR")
        r.apply_hilbert(envelope=True)
        out[b] = np.log10(r.get_data() + 1e-30)  # (n_ch, n_t) at native sr
    return out, ch, sfreq


def build_subject_continuous(sub: str) -> dict | None:
    """Per-run aligned 25 Hz: env_by_band {band: {'po':(T,), 'all':(T,n_ch)}}, eda (T,), times, ch.

    Returns dict {label: {...}} or None.
    """
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    eda_fs = float(cont["eda_fs"]) if "eda_fs" in cont else EDA_FS
    runs = {}
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            env_by_band, ch, sfreq = band_envelopes(raw)
            smna = np.asarray(cont[f"{label}__eda_smna"], float)      # primary target (sparse driver)
            phasic = np.asarray(cont[f"{label}__eda_phasic"], float)  # secondary
            d_eeg = int(round(sfreq / COMMON_FS))   # decim factor EEG (~500->12.5 = 40)
            d_eda = int(round(eda_fs / COMMON_FS))  # decim factor EDA (50->12.5 = 4)
            trim = int(round(EDGE_TRIM_S * COMMON_FS))
            po_idx = [ch.index(c) for c in PARIETOOCCIPITAL if c in ch]
            # SMNA: leve suavizado gaussiano (driver esparso -> rate-like) y decimación polifásica (AA)
            smna_s = gaussian_filter1d(smna, sigma=SMNA_SMOOTH_S * eda_fs)
            smna25 = resample_poly(smna_s, 1, d_eda)
            phasic25 = resample_poly(phasic, 1, d_eda)
            env25_by_band = {b: np.vstack([resample_poly(env[i], 1, d_eeg) for i in range(env.shape[0])])
                             for b, env in env_by_band.items()}
            n = min([len(smna25), len(phasic25)] + [v.shape[1] for v in env25_by_band.values()])
            if n <= 2 * trim + 10:
                continue
            sl = slice(trim, n - trim)
            eda_t = _zscore(smna25[:n])[sl]            # target primario = SMNA
            eda_phasic_t = _zscore(phasic25[:n])[sl]   # secundario
            band_out = {}
            for b, env25 in env25_by_band.items():
                env25 = env25[:, :n]
                po = _zscore(env25[po_idx].mean(axis=0))[sl]
                allch = np.vstack([_zscore(env25[i])[sl] for i in range(env25.shape[0])]).T  # (T, n_ch)
                band_out[b] = {"po": po, "all": allch}
            times = (np.arange(n) / COMMON_FS)[sl]
            runs[label] = {"bands": band_out, "eda": eda_t, "eda_phasic": eda_phasic_t,
                           "times": times, "ch": ch, "dur": n / COMMON_FS, "n25": len(eda_t)}
            print(f"  {sub} {label}: dur={n / COMMON_FS:.0f}s T25={len(eda_t)} ch={len(ch)}", flush=True)
        except Exception as e:
            print(f"  {sub} {label}: FAILED -- {e}", flush=True)
    return runs or None


def main():
    """Sanity: alignment check on sub-19 + an overlay figure (delta-PO env vs eda)."""
    sub = "sub-19"
    print(f"continuous_band sanity :: {sub}")
    runs = build_subject_continuous(sub)
    if not runs:
        print("no runs"); return
    label = max(runs, key=lambda k: runs[k]["n25"])
    r = runs[label]
    t, po, eda = r["times"], r["bands"]["delta"]["po"], r["eda"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, _zscore(po), color="C0", lw=0.8, label="delta-PO env (z)")
    ax.plot(t, _zscore(eda), color="C3", lw=0.8, alpha=0.8, label="SMNA target (z)")
    ax.set_xlabel("t (s, run)"); ax.set_ylabel("z"); ax.legend(fontsize=9)
    ax.set_title(f"{sub} {label} :: alineación EEG-delta-PO vs EDA @25Hz (sanity)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "sanity_alignment_sub19.png", dpi=120); plt.close(fig)
    print(f"  runs={len(runs)}  -> sanity_alignment_sub19.png")
    for lab, rr in runs.items():
        print(f"    {lab}: T25={rr['n25']} bands={list(rr['bands'])}")


if __name__ == "__main__":
    main()
