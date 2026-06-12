"""Bloque 4.x (opción B-2) — Movimiento AFECTIVO vs movimiento de LUMINANCIA (control motor-emparejado).

Idea (usuario): cada run tiene un segmento (~60 s) donde el sujeto sigue la LUMINANCIA del video con el
joystick — MISMO acto motor (mover la palanca), pero perceptual, SIN cambio afectivo. Es el control
emparejado-en-cinemática / distinto-en-afecto que faltaba:

  - arousal-move = movimientos bruscos (|Δ|) en segmentos de arousal (reporte afectivo).
  - lum-move     = movimientos bruscos (|Δ|) en segmentos de luminancia (seguimiento perceptual).
  Ambos son movimientos del joystick; difieren en la TAREA/afecto. Si la señal post-shift es PURO MOTOR
  -> arousal-move ≈ lum-move (no separables). Si hay algo afectivo/de-tarea -> separan.

Clave de NPZ: el array de luminancia se guarda como `{label}__lum` (no `__luminance`). Existe en (casi)
todos los runs, independientemente de la dimensión afectiva del run. Detección con mode="abs" (cualquier
movimiento abrupto, no monótono). Ventanas pre `[-4,-0.5]` y post `[-0.5,3]`. CV intra adaptativo + LOSO.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_luminance --build [--c 0.33]
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_luminance [--c 0.33] [--nperm 1000]
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
    FLAT_STD_THRESH, load_joystick, run_dimension, subject_amplitude_scale,
    detect_shift_onsets_s, sample_calma_matched,
)
from src.campeones_analysis.multimodal_arousal.joystick_panel import (
    OUT_DIR, LOWPASS, RESAMPLE, MATCHED_SEED,
)
from src.campeones_analysis.multimodal_arousal.joystick_updown import (
    WIDE_TMIN, WIDE_TMAX, WINDOWS, RNG, SEED, MIN_PER_CLASS, _intra_adaptive,
)
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    loso_auc, _perm_within, feat_periodic, feat_allbins,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"


def _cache(c):
    return TBL_DIR / f"joystick_lum_psd_c{int(round(c * 100)):03d}.npz"


def _epoch_movements(raw, onsets, dur):
    on = np.asarray(onsets, float)
    kept = (on + WIDE_TMIN > 0) & (on + WIDE_TMAX < dur)
    ok = on[kept]
    if ok.size == 0:
        return None, kept
    ep = epoch_one_run(raw, ok, code=1, tmin=WIDE_TMIN, tmax=WIDE_TMAX, baseline=(None, None))
    return ep, kept


def _build_subject(sub, c):
    npz, runs = load_joystick(sub)
    if npz is None:
        return None
    scale_ar = subject_amplitude_scale(npz, runs, "arousal")
    scale_lum = subject_amplitude_scale(npz, runs, "lum")
    rng_m = np.random.default_rng(MATCHED_SEED)
    ar_eps, lum_eps, ca_eps = [], [], []
    ar_mag, lum_mag = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs:
            continue
        ar_on = lum_on = np.array([])
        ar_m = lum_m = np.array([])
        # arousal-move (|Δ|) si el run es de arousal
        if run_dimension(npz, label) == "arousal":
            arr = np.asarray(npz[f"{label}__arousal"], float)
            ar_on, ar_m = detect_shift_onsets_s(arr, mode="abs", scale=scale_ar, c=c)
        # lum-move (|Δ|) si el run tiene segmento de luminancia movido
        lkey = f"{label}__lum"
        if lkey in npz.files:
            larr = np.asarray(npz[lkey], float)
            if np.isfinite(larr).any() and float(np.nanstd(larr)) >= FLAT_STD_THRESH:
                lum_on, lum_m = detect_shift_onsets_s(larr, mode="abs", scale=scale_lum, c=c)
        if ar_on.size == 0 and lum_on.size == 0:
            continue
        # calma: emparejada a los eventos del run (sobre la señal de arousal si existe, si no lum)
        base_arr = (np.asarray(npz[f"{label}__arousal"], float)
                    if run_dimension(npz, label) == "arousal" else None)
        ev_for_calma = np.sort(np.concatenate([ar_on, lum_on])) if (ar_on.size or lum_on.size) else np.array([])
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=LOWPASS, verbose="ERROR")
            raw.resample(RESAMPLE, verbose="ERROR")
        except Exception as e:
            print(f"    {label}: FAILED -- {e}", flush=True); continue
        dur = float(raw.times[-1])
        if ar_on.size:
            ep, kept = _epoch_movements(raw, ar_on, dur)
            if ep is not None and len(ep):
                ar_eps.append(ep); ar_mag.append(ar_m[kept])
        if lum_on.size:
            ep, kept = _epoch_movements(raw, lum_on, dur)
            if ep is not None and len(ep):
                lum_eps.append(ep); lum_mag.append(lum_m[kept])
        if base_arr is not None and ev_for_calma.size:
            calma = sample_calma_matched(ev_for_calma, base_arr, 50.0, rng_m, scale=scale_ar)
            if calma.size:
                ep, _ = _epoch_movements(raw, calma, dur)
                if ep is not None and len(ep):
                    ca_eps.append(ep)
    if not ar_eps or not lum_eps:
        return None
    out = {}; ref_ch = None
    for win, (a, b) in WINDOWS.items():
        sets = {}
        for name, eps in [("arousal", ar_eps), ("lum", lum_eps), ("calma", ca_eps)]:
            if not eps:
                continue
            ep = mne.concatenate_epochs(eps, verbose="ERROR").copy().crop(tmin=a, tmax=b)
            psd, freqs, ch = compute_psd(ep)
            if ref_ch is None:
                ref_ch = ch; out["_freqs"] = freqs
            if ch != ref_ch:
                idx = [ch.index(cc) for cc in ref_ch if cc in ch]
                psd = psd[:, idx, :]
            sets[name] = psd.astype(np.float32)
        out[win] = sets
    out["_ch"] = ref_ch
    out["mag_arousal"] = np.concatenate(ar_mag) if ar_mag else np.array([])
    out["mag_lum"] = np.concatenate(lum_mag) if lum_mag else np.array([])
    return out


def build_cache(c):
    print(f"[B2] building arousal-move/lum-move/calma cache (c={c}) ...", flush=True)
    store = {}; ref_ch = freqs = None
    for sub in COHORT:
        r = _build_subject(sub, c)
        if r is None:
            print(f"  {sub}: skipped (sin arousal-move y lum-move)", flush=True); continue
        if ref_ch is None:
            ref_ch = r["_ch"]; freqs = r["_freqs"]
        for win in WINDOWS:
            for name, psd in r[win].items():
                store[f"psd_{win}_{name}_{sub}"] = psd
        na = r["post"].get("arousal", np.empty((0,))).shape[0]
        nl = r["post"].get("lum", np.empty((0,))).shape[0]
        nc = r["post"].get("calma", np.empty((0,))).shape[0]
        print(f"  {sub}: arousal-move={na} lum-move={nl} calma={nc}  "
              f"mag_ar p50={np.median(r['mag_arousal']) if r['mag_arousal'].size else float('nan'):.2f} "
              f"mag_lum p50={np.median(r['mag_lum']) if r['mag_lum'].size else float('nan'):.2f}", flush=True)
    store["freqs"] = freqs; store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(_cache(c), **store)
    print(f"[B2] cache -> {_cache(c)}", flush=True)


def _load(z, win, classes, min_per_class=MIN_PER_CLASS):
    ch = list(z["ch_names"]); keep = [i for i, cc in enumerate(ch) if cc not in DROP_CHANNELS]
    ch_kept = [ch[i] for i in keep]
    pos, neg = classes
    data = {}
    for sub in COHORT:
        kp, kn = f"psd_{win}_{pos}_{sub}", f"psd_{win}_{neg}_{sub}"
        if kp not in z or kn not in z:
            continue
        Pp = z[kp].astype(float)[:, keep, :]; Pn = z[kn].astype(float)[:, keep, :]
        if Pp.shape[0] < min_per_class or Pn.shape[0] < min_per_class:
            continue
        psd = np.concatenate([Pp, Pn], 0)
        y = np.concatenate([np.ones(len(Pp)), np.zeros(len(Pn))]).astype(int)
        data[sub] = (psd, y)
    return data, z["freqs"], ch_kept


def _eval(data, freqs, builder, n_perm=0):
    Xs = [builder(psd, freqs, RNG) for psd, y in data.values()]
    ys = [y for _, y in data.values()]
    Xall = np.concatenate(Xs, 0); yall = np.concatenate(ys)
    groups = np.concatenate([[i] * len(y) for i, y in enumerate(ys)])
    intra, intra_l = _intra_adaptive(Xs, ys)
    loso, _ = loso_auc(Xall, yall, groups)
    p_intra = p_loso = np.nan
    if n_perm > 0:
        pr = np.random.default_rng(SEED + 1); gi = gl = 0
        for _ in range(n_perm):
            yp = _perm_within(yall, groups, pr)
            ys_p = [yp[groups == i] for i in range(len(ys))]
            ip, _ = _intra_adaptive(Xs, ys_p); lp, _ = loso_auc(Xall, yp, groups)
            gi += ip >= intra; gl += lp >= loso
        p_intra = (1 + gi) / (1 + n_perm); p_loso = (1 + gl) / (1 + n_perm)
    return dict(intra=round(intra, 3), loso=round(loso, 3),
                p_intra=round(p_intra, 4) if n_perm else np.nan,
                p_loso=round(p_loso, 4) if n_perm else np.nan,
                intra_subj=[round(a, 3) for a in intra_l], n_feat=Xall.shape[1])


def run(c, nperm):
    z = np.load(_cache(c), allow_pickle=True)
    CONTRASTS = [("arousal", "lum"), ("arousal", "calma"), ("lum", "calma")]
    rows = []
    for win in WINDOWS:
        for classes in CONTRASTS:
            for fname, builder in [("periódico", feat_periodic),
                                   ("espectro", lambda p, f, r: feat_allbins(p, f, r, "1"))]:
                data, freqs, ch = _load(z, win, classes)
                if len(data) < 3:
                    print(f"  {win:4} {classes[0]:7}-{classes[1]:7} {fname:9}: "
                          f"solo {len(data)} suj (insuf.)", flush=True)
                    continue
                np_use = nperm if classes == ("arousal", "lum") else 0
                r = _eval(data, freqs, builder, n_perm=np_use)
                rows.append(dict(window=win, contrast=f"{classes[0]}-{classes[1]}", feature=fname,
                                 c=c, n_subj=len(data), **r))
                pstr = f" p_intra={r['p_intra']} p_loso={r['p_loso']}" if np_use else ""
                print(f"  {win:4} {classes[0]:7}-{classes[1]:7} {fname:9}: "
                      f"intra={r['intra']:.3f} LOSO={r['loso']:.3f} (N={len(data)}, "
                      f"nfeat={r['n_feat']}){pstr}  intra/suj={r['intra_subj']}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["intra_subj"]).to_csv(
        TBL_DIR / f"lum_decoding_c{int(round(c*100)):03d}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    per = df[df.feature == "periódico"]
    for ax, win in zip(axes, WINDOWS):
        sub = per[per.window == win]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["intra"], 0.4, label="intra", color="C0", alpha=0.7)
        ax.bar(x + 0.2, sub["loso"], 0.4, label="LOSO", color="C1", alpha=0.7)
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sub["contrast"], fontsize=8)
        ax.set_ylim(0.3, 0.85); ax.set_ylabel("AUC"); ax.set_title(f"{win}-shift {WINDOWS[win]}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"B2. Movimiento afectivo vs luminancia (periódico, c={c}): ¿arousal-vs-lum separa? (motor↔afecto)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"lum_decoding_c{int(round(c*100)):03d}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--c", type=float, default=0.33)
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()
    print("=" * 78); print(f"joystick_luminance :: afecto vs luminancia (c={args.c})")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(args.c); return
    if not _cache(args.c).exists():
        print(f"[B2] no cache para c={args.c} -- correr con --build --c {args.c}", flush=True); return
    run(args.c, args.nperm)


if __name__ == "__main__":
    main()
