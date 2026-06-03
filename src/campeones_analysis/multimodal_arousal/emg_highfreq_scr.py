"""Diagnóstico EMG-vs-neural en altas frecuencias (60-90 Hz) sobre el stream broadband
re-derivado (recon_wide). Tarea QA artefacto-vs-señal, deliverable highfreq (2026-06-03).

Pregunta: el aplanamiento del 1/f en el contraste SCR-vs-silent (exponent ↓) es ambiguo
entre EMG (potencia broadband que pico ~70-150 Hz, fuerte en canales temporal/edge) y
excitación cortical (Gao-Voytek 2017). Las dos divergen >40 Hz. Acá medimos `real`-`silent`
en 60-90 Hz (limpio entre los notch de 50/100) y miramos su TOPOGRAFÍA.

Regla de decisión (pre-especificada, sobre el stream post-ICA):
  EMG-POSITIVO   si HF real>silent consistente (≥4/6 mismo signo) Y concentrado en edge
                 (edge-central > 0 en ≥4/6 y en el topo GA) Y joroba/knee FOOOF >40 Hz.
  E/I-FAVORABLE  si HF real>silent ausente/difuso (edge-central ≈0/neg) Y sin joroba >40 Hz.
  INCONCLUSO     en otro caso -> futura corrida del control pre-ICA.

Reusa: build_subject_epochs_wide/compute_psd_wide (recon_wide), apply_drop_only (epochs_qc),
fit_group/group_aperiodic (fooof_scr), roi_channel_indices (tfr_psd_scr).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.emg_highfreq_scr
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.emg_highfreq_scr --subjects sub-27
"""

from __future__ import annotations

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT
from src.campeones_analysis.multimodal_arousal.erp_scr import NPZ_DIR
from src.campeones_analysis.multimodal_arousal.epochs_qc import apply_drop_only
from src.campeones_analysis.multimodal_arousal.fooof_scr import fit_group, group_aperiodic
from src.campeones_analysis.multimodal_arousal.recon_wide import (
    build_subject_epochs_wide,
    compute_psd_wide,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

HF_DIR = NPZ_DIR.parent / "qa_artifact_vs_signal" / "highfreq"
FIG_DIR = HF_DIR / "figures"
TBL_DIR = HF_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

HF_BAND = (60.0, 90.0)
PSD_FMAX = 95.0
NOTCH_GUARD = 2.0          # excluir bins a ±2 Hz de 50 y 100 (faldas del notch)
NOTCH_LINES = (50.0, 100.0)
EMG_CHANNELS = ["FT9", "TP9", "T7", "T8", "P7", "P8"]   # temporal/edge (firma EMG)
CENTRAL_CHANNELS = ["Cz", "Pz", "Fz", "FCz", "CP1", "CP2"]
FOOOF_WIDE_RANGE = (1.5, 90.0)


def hf_bin_mask(freqs: np.ndarray) -> np.ndarray:
    """Bins en [HF_BAND] excluyendo ±NOTCH_GUARD alrededor de las líneas de notch."""
    m = (freqs >= HF_BAND[0]) & (freqs <= HF_BAND[1])
    for nl in NOTCH_LINES:
        m &= np.abs(freqs - nl) > NOTCH_GUARD
    return m


def _idx(ch_names: list[str], wanted: list[str]) -> list[int]:
    return [ch_names.index(c) for c in wanted if c in ch_names]


def _subject_hf(sub: str, apply_ica: bool = True) -> dict | None:
    real_ep, silent_ep = build_subject_epochs_wide(sub, apply_ica=apply_ica)
    if real_ep is None or silent_ep is None:
        print(f"  {sub}: no epochs -> skip", flush=True)
        return None
    real_c, silent_c, thresh = apply_drop_only(real_ep, silent_ep)
    psd_r, freqs, ch = compute_psd_wide(real_c, fmax=PSD_FMAX)
    psd_s, _, _ = compute_psd_wide(silent_c, fmax=PSD_FMAX)
    mr = psd_r.mean(axis=0)   # (n_ch, n_freq) linear power
    ms = psd_s.mean(axis=0)

    hf = hf_bin_mask(freqs)
    hf_r = mr[:, hf].mean(axis=1)      # mean HF power per channel (linear)
    hf_s = ms[:, hf].mean(axis=1)
    hf_diff_db = 10.0 * np.log10(hf_r + 1e-30) - 10.0 * np.log10(hf_s + 1e-30)
    hf_ratio = hf_r / (hf_s + 1e-30)

    e_idx, c_idx = _idx(ch, EMG_CHANNELS), _idx(ch, CENTRAL_CHANNELS)
    edge_diff = float(np.mean(hf_diff_db[e_idx])) if e_idx else np.nan
    central_diff = float(np.mean(hf_diff_db[c_idx])) if c_idx else np.nan

    # extended FOOOF (1.5-90, knee) on mean spectra -> exponent/knee/offset per condition
    fg_r = fit_group(mr, freqs, "knee")
    fg_s = fit_group(ms, freqs, "knee")
    ap_r, ap_s = group_aperiodic(fg_r, "knee"), group_aperiodic(fg_s, "knee")
    d_exp_wide = float(np.nanmean(ap_r["exponent"] - ap_s["exponent"]))
    d_off_wide = float(np.nanmean(ap_r["offset"] - ap_s["offset"]))

    print(f"  {sub}: n_real={len(real_c)} n_silent={len(silent_c)}  "
          f"HF diff all={np.mean(hf_diff_db):+.2f}dB  edge={edge_diff:+.2f}  "
          f"central={central_diff:+.2f}  edge-central={edge_diff - central_diff:+.2f}  "
          f"d_exp_wide={d_exp_wide:+.3f}", flush=True)

    _plot_subject(sub, freqs, mr, ms, ch, hf, hf_diff_db, real_c.info, len(real_c), len(silent_c))

    return dict(
        subject=sub, n_real=len(real_c), n_silent=len(silent_c),
        hf_diff_db_all=float(np.mean(hf_diff_db)),
        hf_ratio_all=float(np.mean(hf_ratio)),
        hf_diff_db_edge=edge_diff, hf_diff_db_central=central_diff,
        edge_minus_central=float(edge_diff - central_diff),
        d_exponent_wide=d_exp_wide, d_offset_wide=d_off_wide,
        ch_names=ch, hf_diff_db_per_ch=hf_diff_db.tolist(),
    )


def _plot_subject(sub, freqs, mr, ms, ch, hf_mask, hf_diff_db, info, n_r, n_s) -> None:
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    fig.suptitle(f"{sub} -- HF (60-90 Hz) EMG diagnostic [post-ICA]  "
                 f"n_real={n_r} n_silent={n_s}", fontsize=11)

    # PSD real vs silent (mean over all channels), log-x, HF band shaded
    ax = fig.add_subplot(gs[0, 0])
    db_r = 10.0 * np.log10(mr.mean(axis=0) + 1e-30)
    db_s = 10.0 * np.log10(ms.mean(axis=0) + 1e-30)
    ax.plot(freqs, db_r, color="C3", lw=1.4, label="real (SCR)")
    ax.plot(freqs, db_s, color="0.4", lw=1.2, ls="--", label="silent")
    ax.axvspan(HF_BAND[0], HF_BAND[1], color="gold", alpha=0.15, label="HF 60-90")
    for nl in NOTCH_LINES:
        ax.axvspan(nl - NOTCH_GUARD, nl + NOTCH_GUARD, color="0.85", alpha=0.6)
    ax.set_xscale("log"); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title("PSD (mean ch)"); ax.legend(fontsize=8)

    # HF diff topomap
    ax2 = fig.add_subplot(gs[0, 1])
    vmax = float(np.nanmax(np.abs(hf_diff_db))) if np.any(np.isfinite(hf_diff_db)) else 1.0
    try:
        im, _ = mne.viz.plot_topomap(hf_diff_db, info, axes=ax2, show=False,
                                     vlim=(-vmax, vmax), cmap="RdBu_r", contours=4, sensors=True)
        fig.colorbar(im, ax=ax2, shrink=0.7, label="dB (real-silent)")
    except Exception as e:
        ax2.text(0.5, 0.5, f"topo fail\n{e}", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("HF 60-90 diff (red=real>silent)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / f"emg_highfreq_{sub}.png", dpi=130)
    plt.close(fig)


def _decision(df: pd.DataFrame) -> dict:
    n = len(df)
    pos_sign = int((df["hf_diff_db_all"] > 0).sum())
    edge_pos = int((df["edge_minus_central"] > 0).sum())
    ga_edge_minus_central = float(df["edge_minus_central"].mean())
    ga_hf_diff = float(df["hf_diff_db_all"].mean())
    emg_positive = (pos_sign >= max(4, int(np.ceil(0.66 * n)))) and (edge_pos >= max(4, int(np.ceil(0.66 * n)))) \
        and (ga_edge_minus_central > 0)
    ei_favoring = (pos_sign <= n - 4 or abs(ga_hf_diff) < 0.3) and (ga_edge_minus_central <= 0)
    verdict = "EMG-POSITIVE" if emg_positive else ("E/I-FAVORING" if ei_favoring else "INCONCLUSIVE")
    return dict(
        n_subjects=n, hf_diff_db_all_GA=round(ga_hf_diff, 3),
        n_real_gt_silent=pos_sign, n_edge_gt_central=edge_pos,
        edge_minus_central_GA=round(ga_edge_minus_central, 3),
        d_exponent_wide_GA=round(float(df["d_exponent_wide"].mean()), 4),
        verdict=verdict,
    )


def _plot_ga(df: pd.DataFrame, ref_ch: list[str]) -> None:
    # GA HF-diff topomap (need an info; rebuild a montage-only info from ref channels)
    diffs = np.array(df["hf_diff_db_per_ch"].tolist())            # (n_sub, n_ch) aligned to ref_ch
    ga = np.nanmean(diffs, axis=0)
    info = mne.create_info(ref_ch, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="ignore")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle(f"GA HF (60-90 Hz) real-silent, N={len(df)} [post-ICA]", fontsize=11)
    vmax = float(np.nanmax(np.abs(ga))) if np.any(np.isfinite(ga)) else 1.0
    try:
        im, _ = mne.viz.plot_topomap(ga, info, axes=axes[0], show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", contours=4, sensors=True)
        fig.colorbar(im, ax=axes[0], shrink=0.7, label="dB")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"topo fail\n{e}", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("GA HF diff topomap", fontsize=10)

    # edge vs central per subject
    x = np.arange(len(df))
    axes[1].bar(x - 0.2, df["hf_diff_db_edge"], 0.4, color="C3", label="edge (EMG ch)")
    axes[1].bar(x + 0.2, df["hf_diff_db_central"], 0.4, color="C0", label="central")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["subject"], rotation=45, fontsize=7)
    axes[1].set_ylabel("HF diff (dB)"); axes[1].set_title("edge vs central per subject", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "emg_highfreq_GA.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="HF 60-90 Hz EMG-vs-neural diagnostic.")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--pre-ica", action="store_true", help="use the pre-ICA stream (positive control)")
    args = p.parse_args()
    subs = args.subjects if args.subjects else list(COHORT)
    apply_ica = not args.pre_ica
    tag = "preica" if args.pre_ica else "postica"

    print("=" * 78)
    print(f"emg_highfreq_scr :: HF {HF_BAND} Hz diagnostic  subjects={subs}  stream={tag}")
    print("=" * 78, flush=True)

    rows = [r for r in (_subject_hf(s, apply_ica=apply_ica) for s in subs) if r is not None]
    if not rows:
        print("No subjects processed.")
        return

    ref_ch = rows[0]["ch_names"]
    # align per-channel diffs to ref_ch order
    for r in rows:
        d = dict(zip(r["ch_names"], r["hf_diff_db_per_ch"]))
        r["hf_diff_db_per_ch"] = [d.get(c, np.nan) for c in ref_ch]

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ch_names"} for r in rows])
    df_flat = df.drop(columns=["hf_diff_db_per_ch"])
    df_flat.to_csv(TBL_DIR / f"emg_highfreq_summary_{tag}.csv", index=False)

    _plot_ga(df, ref_ch)
    decision = _decision(df)
    decision["stream"] = tag
    with open(TBL_DIR / f"emg_highfreq_decision_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(decision, fh, indent=2)

    print("\n" + df_flat.to_string(index=False), flush=True)
    print("\nDECISION:", json.dumps(decision, indent=2), flush=True)
    print(f"\nOutputs -> {HF_DIR}", flush=True)


if __name__ == "__main__":
    main()
