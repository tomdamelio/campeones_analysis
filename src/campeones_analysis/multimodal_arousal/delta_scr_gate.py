"""Gate mínimo del lead delta-SCR (antes de C): ¿el acoplamiento delta-PO vs amplitud del SCR
sobrevive (1) controlar el tiempo-en-sesión y (2) el CSD, y dónde acopla (cortical vs movimiento)?

Confounds a descartar: (1) drift temporal -> SCR grandes podrían caer más tarde en el run y el
delta driftar con el tiempo (correlación espuria); se testea con correlación PARCIAL controlando
tiempo. (2) movimiento lento/ocular difuso -> el CSD (Laplaciano) mata lo global/volumen-conducido
y realza lo focal-cortical; si el acoplamiento sobrevive en CSD y es central-parietal = SCR-RO de
Branković; si desaparece o es edge/frontal = artefacto.

Por época real: potencia delta (1-4 Hz) por canal (sensor y CSD), amplitud SCR (pico phasic post-
onset), y tiempo-en-sesión (onset/dur). Salidas: tabla por sujeto (rho cero-orden vs parcial, sensor
y CSD; rho amp-vs-tiempo) + topomaps GA del rho por canal (sensor vs CSD).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.delta_scr_gate
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from mne.preprocessing import compute_current_source_density

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
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

OUT_DIR = OUT / "qa_artifact_vs_signal" / "delta_gate"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

DELTA = (1.0, 4.0)
PARIETOOCCIPITAL = ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"]


def partial_spearman(x, y, z):
    """Partial Spearman of (x, y) controlling z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


def _delta_per_channel(epochs):
    psd, freqs, ch = compute_psd(epochs)  # (n_ep, n_ch, n_freq)
    m = (freqs >= DELTA[0]) & (freqs < DELTA[1])
    return np.log10(psd[:, :, m].mean(axis=2) + 1e-30), list(ch)  # (n_ep, n_ch)


def subject_delta(sub):
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    Dsens, Dcsd, amp, tf = [], [], [], []
    ref_info = None
    ch_names = None
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
            onsets = onsets[(onsets + TMIN > 0) & (onsets + TMAX < dur)]
            if onsets.size == 0:
                continue
            a = []
            for t in onsets:
                i0, i1 = int(round(t * EDA_FS)), int(round((t + POST_S) * EDA_FS))
                a.append(float(np.max(eda[i0:i1])) if i1 > i0 and i1 <= len(eda) else np.nan)
            ep = epoch_one_run(raw, onsets, code=1)
            if ep is None or len(ep) == 0:
                continue
            ds, ch = _delta_per_channel(ep)
            dc, _ = _delta_per_channel(compute_current_source_density(ep.copy()))
            n = min(len(ep), len(a))
            Dsens.append(ds[:n]); Dcsd.append(dc[:n]); amp += a[:n]; tf += list(onsets[:n] / dur)
            ch_names = ch
            if ref_info is None:
                ref_info = ep.info.copy()
        except Exception as e:
            print(f"  {label}: FAILED -- {e}", flush=True)
    if not Dsens:
        return None
    return dict(Dsens=np.vstack(Dsens), Dcsd=np.vstack(Dcsd), amp=np.asarray(amp, float),
                tf=np.asarray(tf, float), ch=ch_names, info=ref_info)


def main():
    print("=" * 78)
    print("delta_scr_gate :: ¿el lead delta-SCR sobrevive control temporal + CSD?")
    print("=" * 78, flush=True)
    rows = []
    rho_ch_sensor, rho_ch_csd = {}, {}
    ref_info = None
    ref_ch = None
    for sub in COHORT:
        d = subject_delta(sub)
        if d is None:
            print(f"  {sub}: no data", flush=True); continue
        if ref_info is None:
            ref_info, ref_ch = d["info"], d["ch"]
        po = [d["ch"].index(c) for c in PARIETOOCCIPITAL if c in d["ch"]]
        amp, tf = d["amp"], d["tf"]
        po_sens = d["Dsens"][:, po].mean(axis=1)
        po_csd = d["Dcsd"][:, po].mean(axis=1)
        mask = np.isfinite(amp) & np.isfinite(tf) & np.isfinite(po_sens) & np.isfinite(po_csd)
        rho0_s = spearmanr(amp[mask], po_sens[mask])[0]
        rhop_s = partial_spearman(po_sens[mask], amp[mask], tf[mask])
        rho0_c = spearmanr(amp[mask], po_csd[mask])[0]
        rhop_c = partial_spearman(po_csd[mask], amp[mask], tf[mask])
        amp_tf = spearmanr(amp[mask], tf[mask])[0]
        rows.append(dict(subject=sub, n=int(mask.sum()),
                         rho0_sensor=round(float(rho0_s), 3), rho_partial_sensor=round(float(rhop_s), 3),
                         rho0_csd=round(float(rho0_c), 3), rho_partial_csd=round(float(rhop_c), 3),
                         rho_amp_vs_time=round(float(amp_tf), 3)))
        print(f"  {sub}: sensor rho0={rho0_s:+.2f}->partial={rhop_s:+.2f} | "
              f"CSD rho0={rho0_c:+.2f}->partial={rhop_c:+.2f} | amp~time={amp_tf:+.2f}", flush=True)
        # per-channel rho (sensor + CSD), aligned by name
        for arr, store in ((d["Dsens"], rho_ch_sensor), (d["Dcsd"], rho_ch_csd)):
            for j, cn in enumerate(d["ch"]):
                r = spearmanr(amp[mask], arr[mask, j])[0]
                store.setdefault(cn, []).append(r)

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "delta_gate_summary.csv", index=False)

    # GA topomaps of per-channel rho (sensor vs CSD)
    ga_s = np.array([np.nanmean(rho_ch_sensor.get(c, [np.nan])) for c in ref_ch])
    ga_c = np.array([np.nanmean(rho_ch_csd.get(c, [np.nan])) for c in ref_ch])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, name in zip(axes, (ga_s, ga_c), ("sensor", "CSD (Laplaciano)")):
        vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.1
        im, _ = mne.viz.plot_topomap(data, ref_info, axes=ax, show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", contours=4, sensors=True)
        fig.colorbar(im, ax=ax, shrink=0.7, label="rho (delta vs amp SCR)")
        ax.set_title(name, fontsize=10)
    fig.suptitle("Topografía GA del acoplamiento delta-vs-amplitud-SCR: ¿central-parietal (SCR-RO) "
                 "o edge/frontal (movimiento)? CSD mata lo global.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "delta_gate_topomap.png", dpi=130)
    plt.close(fig)

    # bar: rho0 vs partial (sensor) per subject + GA
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, df["rho0_sensor"], w, color="C0", label="rho cero-orden (sensor)")
    ax.bar(x + w / 2, df["rho_partial_sensor"], w, color="C1", label="rho parcial | tiempo")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(df["subject"])
    ax.set_ylabel("Spearman rho (delta-PO vs amplitud SCR)")
    ax.set_title("¿Sobrevive el acoplamiento delta-PO al controlar tiempo-en-sesión?", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delta_gate_partial.png", dpi=130)
    plt.close(fig)

    # verdict
    n_pos_partial_s = int((df["rho_partial_sensor"] > 0).sum())
    n_pos_partial_c = int((df["rho_partial_csd"] > 0).sum())
    print("\n" + df.to_string(index=False), flush=True)
    print(f"\nGA rho0 sensor={df['rho0_sensor'].mean():+.3f}  partial sensor={df['rho_partial_sensor'].mean():+.3f} "
          f"({n_pos_partial_s}/{len(df)} pos)", flush=True)
    print(f"GA rho0 CSD={df['rho0_csd'].mean():+.3f}  partial CSD={df['rho_partial_csd'].mean():+.3f} "
          f"({n_pos_partial_c}/{len(df)} pos)", flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
