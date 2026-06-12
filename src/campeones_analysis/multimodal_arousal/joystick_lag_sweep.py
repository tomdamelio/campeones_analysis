"""Bloque 4.x — Barrido de LAG de la ventana de análisis joystick (evento-vs-calma).

El nulo en pre-shift `[-4,-0.5]` puede ser un artefacto de COLOCACIÓN de la ventana: en SCR la señal
vivía en la ventana-RESPUESTA (post-onset), que acá excluimos para evitar el motor. Este barrido
desliza una ventana de 3.5 s desde pre-shift, a través del shift (t=0), hasta post-shift, y decodifica
en cada offset, para LOCALIZAR dónde (si en algún lado) emerge señal.

  - lags con tmax <= 0  -> motor-LIMPIOS (terminan antes del movimiento).
  - lags con tmax > 0   -> incluyen el shift/post -> motor-CONTAMINADOS (se decodifican igual, pero un
    AUC>azar acá obliga a strict-test motor en 4.5 antes de interpretarlo como afecto).

Épocas anchas `[-4,+3]` construidas UNA vez por sujeto (pase EEG único); cada lag = crop + PSD.
Solo esquema EMPAREJADO (calma matched, el primario). Decoding intra+LOSO SIN perms (localización);
las permutaciones se corren después solo en el/los lag(s) donde aparezca señal.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_lag_sweep --build
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_lag_sweep --dim arousal
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_lag_sweep --dim combined
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

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, DROP_CHANNELS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import compute_psd
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    attach_montage_and_drop_no_pos, epoch_one_run, run_label, runs_for,
)
from src.campeones_analysis.multimodal_arousal.joystick_events import (
    C_EVENT, DIMS, load_joystick, run_dimension, subject_amplitude_scale,
    detect_shift_onsets_s, sample_calma_matched,
)
from src.campeones_analysis.multimodal_arousal.joystick_panel import (
    OUT_DIR, LOWPASS, RESAMPLE, MATCHED_SEED, DIM_CODE,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import eval_fset, PRIMARY_RANGE

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
CACHE = TBL_DIR / "joystick_lag_psd.npz"

# Ventana ancha que cubre todos los lags (debe caber en el segmento que la detección ya garantizó: [-4,+3]).
WIDE_TMIN, WIDE_TMAX = -4.0, 3.0
# Lags: ventana de 3.5 s deslizándose. tmax<=0 = motor-limpio; tmax>0 = motor-contaminado.
LAGS = [(-4.0, -0.5), (-3.0, 0.5), (-2.0, 1.5), (-1.0, 2.5), (-0.5, 3.0)]
LAG_LABELS = [f"[{a:.1f},{b:.1f}]" for a, b in LAGS]
FSETS = ["periódico", "espectro completo", "aperiódico+periódico"]


def _build_subject(sub):
    npz, runs = load_joystick(sub)
    if npz is None:
        return None
    scales = {d: subject_amplitude_scale(npz, runs, d) for d in DIMS}
    rng_m = np.random.default_rng(MATCHED_SEED)
    ev_eps, ca_eps = [], []          # mne.Epochs anchas
    ev_meta, ca_meta = [], []        # (tnorm, dim, mag) por época
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs:
            continue
        dim = run_dimension(npz, label)
        if dim is None:
            continue
        arr = np.asarray(npz[f"{label}__{dim}"], float)
        mode = "rise" if dim == "arousal" else "abs"
        scale = scales[dim]
        onsets, mags = detect_shift_onsets_s(arr, mode=mode, scale=scale, c=C_EVENT)
        if onsets.size == 0:
            continue
        calma = sample_calma_matched(onsets, arr, 50.0, rng_m, scale=scale)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=LOWPASS, verbose="ERROR")
            raw.resample(RESAMPLE, verbose="ERROR")
        except Exception as e:
            print(f"    {label}: FAILED -- {e}", flush=True); continue
        dur = float(raw.times[-1]); dcode = DIM_CODE[dim]
        # eventos
        kept = (onsets + WIDE_TMIN > 0) & (onsets + WIDE_TMAX < dur)
        ok, mg = onsets[kept], mags[kept]
        if ok.size:
            ep = epoch_one_run(raw, ok, code=1, tmin=WIDE_TMIN, tmax=WIDE_TMAX, baseline=(None, None))
            if ep is not None and len(ep):
                ev_eps.append(ep)
                for t, m in zip(ok, mg):
                    ev_meta.append((t / dur, dcode, m))
        # calma emparejada
        keptc = (calma + WIDE_TMIN > 0) & (calma + WIDE_TMAX < dur)
        ck = calma[keptc]
        if ck.size:
            ep = epoch_one_run(raw, ck, code=2, tmin=WIDE_TMIN, tmax=WIDE_TMAX, baseline=(None, None))
            if ep is not None and len(ep):
                ca_eps.append(ep)
                for t in ck:
                    ca_meta.append((t / dur, dcode, 0.0))
    if not ev_eps or not ca_eps:
        return None
    ev = mne.concatenate_epochs(ev_eps, verbose="ERROR")
    ca = mne.concatenate_epochs(ca_eps, verbose="ERROR")
    # PSD por lag (crop + welch); alinear canales al primer sujeto se hace en build_cache
    out = {"y": np.r_[np.ones(len(ev)), np.zeros(len(ca))].astype(int)}
    meta = np.array(ev_meta + ca_meta, float)  # (n_ep, 3): tnorm, dim, mag
    out["tnorm"] = meta[:, 0].astype(np.float32)
    out["dim"] = meta[:, 1].astype(int)
    out["mag"] = meta[:, 2].astype(np.float32)
    ch = ref = None; freqs = None
    for i, (a, b) in enumerate(LAGS):
        pe = ev.copy().crop(tmin=a, tmax=b)
        pc = ca.copy().crop(tmin=a, tmax=b)
        psd_e, freqs, ch = compute_psd(pe)
        psd_c, _, _ = compute_psd(pc)
        out[f"psd_l{i}"] = np.concatenate([psd_e, psd_c], 0).astype(np.float32)
    out["_ch"] = ch; out["_freqs"] = freqs
    return out


def build_cache():
    print("[lag] building wide-epoch per-lag PSD cache (matched) ...", flush=True)
    store = {}; ref_ch = freqs = None
    for sub in COHORT:
        r = _build_subject(sub)
        if r is None:
            print(f"  {sub}: skipped", flush=True); continue
        if ref_ch is None:
            ref_ch = r["_ch"]; freqs = r["_freqs"]
        ch = r["_ch"]
        idx = [ch.index(c) for c in ref_ch if c in ch] if ch != ref_ch else None
        for i in range(len(LAGS)):
            p = r[f"psd_l{i}"]
            store[f"psd_l{i}_{sub}"] = p[:, idx, :] if idx else p
        store[f"y_{sub}"] = r["y"]; store[f"tnorm_{sub}"] = r["tnorm"]
        store[f"dim_{sub}"] = r["dim"]; store[f"mag_{sub}"] = r["mag"]
        na = int((r["dim"][r["y"] == 1] == 0).sum()); nv = int((r["dim"][r["y"] == 1] == 1).sum())
        print(f"  {sub}: eventos arousal={na} valence={nv}  calma={int((r['y']==0).sum())}", flush=True)
    store["freqs"] = freqs; store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(CACHE, **store)
    print(f"[lag] cache -> {CACHE}", flush=True)


def load_lag(z, i, dim=None, min_per_class=5):
    ch = list(z["ch_names"]); keep = [k for k, c in enumerate(ch) if c not in DROP_CHANNELS]
    ch_kept = [ch[k] for k in keep]
    want = None if dim is None else DIM_CODE[dim]
    data = {}
    for sub in COHORT:
        k = f"psd_l{i}_{sub}"
        if k not in z:
            continue
        psd = z[k].astype(float)[:, keep, :]
        y = z[f"y_{sub}"]; tn = z[f"tnorm_{sub}"]; dm = z[f"dim_{sub}"]
        if want is not None:
            sel = dm == want
            psd, y, tn = psd[sel], y[sel], tn[sel]
        if int(y.sum()) < min_per_class or int((1 - y).sum()) < min_per_class:
            continue
        data[sub] = (psd, y, tn)
    return data, z["freqs"], ch_kept


def run_sweep(dim_tag):
    dim = None if dim_tag == "combined" else dim_tag
    z = np.load(CACHE, allow_pickle=True)
    rows = []
    for i, (a, b) in enumerate(LAGS):
        data, freqs, ch = load_lag(z, i, dim=dim)
        clean = "limpio" if b <= 0 else "MOTOR"
        for fset in FSETS:
            r = eval_fset(fset, data, freqs, PRIMARY_RANGE, 0)  # sin perms (localización)
            rows.append(dict(lag=LAG_LABELS[i], tmax=b, motor=clean, n_subj=len(data),
                             feature_set=fset, intra_auc=r["intra_auc"], loso_auc=r["loso_auc"],
                             intra_subj=r["intra_subj"], loso_subj=r["loso_subj"]))
            print(f"  lag {LAG_LABELS[i]:>11} ({clean:6}) {fset:22s}: "
                  f"intra={r['intra_auc']:.3f}  LOSO={r['loso_auc']:.3f}  "
                  f"intra/suj={r['intra_subj']}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["intra_subj", "loso_subj"]).to_csv(TBL_DIR / f"lag_sweep_{dim_tag}.csv", index=False)

    # plot AUC vs lag
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    centers = [(a + b) / 2 for a, b in LAGS]
    for ax, sch in zip(axes, ["intra_auc", "loso_auc"]):
        for fset in FSETS:
            sub = df[df.feature_set == fset]
            ax.plot(centers, sub[sch].values, marker="o", label=fset)
        ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
        ax.axvspan(0, max(centers) + 1, color="0.9", alpha=0.6, zorder=0)  # zona motor (tmax>0)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("centro de ventana (s rel. al shift; gris=incluye post-shift motor)")
        ax.set_ylabel(f"{sch.replace('_auc','').upper()} AUC"); ax.set_ylim(0.35, 0.75)
        ax.set_title(sch.replace("_auc", "").upper(), fontsize=10)
        if sch == "intra_auc":
            ax.legend(fontsize=7)
    fig.suptitle(f"4.x Barrido de lag — joystick {dim_tag} (¿dónde emerge señal al deslizar la ventana?)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / f"lag_sweep_{dim_tag}.png", dpi=120); plt.close(fig)
    print(f"\n-> {FIG_DIR}/lag_sweep_{dim_tag}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dim", choices=["arousal", "valence", "combined"], default="arousal")
    args = ap.parse_args()
    print("=" * 78)
    print(f"joystick_lag_sweep :: build={args.build} dim={args.dim}")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(); return
    if not CACHE.exists():
        print("[lag] no cache -- correr con --build primero", flush=True); return
    run_sweep(args.dim)


if __name__ == "__main__":
    main()
