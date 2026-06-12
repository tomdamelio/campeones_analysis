"""Bloque 4.x (opción B) — Arousal SUBE vs BAJA: ¿la señal post-shift es puro motor o codifica dirección?

Idea (usuario): no podemos re-diseñar para detectar en la dirección opuesta con datos nuevos, PERO la
palanca de arousal también BAJA (arousal decreciente). Subir y bajar son AMBOS movimientos del joystick
(motor, cinemática comparable: ambos ≥ c·rango), pero OPUESTOS en el eje afectivo. Entonces:

  - Si la señal post-shift es PURO MOVIMIENTO -> SUBE ≈ BAJA (AUC up-vs-down ~ 0.5) y ambas dan el mismo
    contraste vs calma. La dirección del afecto no aporta nada.
  - Si hay un componente DIRECCIONAL/AFECTIVO -> SUBE ≠ BAJA (AUC up-vs-down > 0.5) pese a ser ambos
    movimientos. Discrimina (parcialmente) motor de afecto.

Específico de AROUSAL (monótono: arriba=más, abajo=menos). Ventanas: pre-shift `[-4,-0.5]` (motor-limpia)
y post-shift `[-0.5,3]` (donde vive la señal). Tres contrastes: up-vs-down, up-vs-calma, down-vs-calma.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_updown --build
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_updown
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
    C_EVENT, load_joystick, run_dimension, subject_amplitude_scale,
    detect_shift_onsets_s, sample_calma_matched,
)
from src.campeones_analysis.multimodal_arousal.joystick_panel import (
    OUT_DIR, LOWPASS, RESAMPLE, MATCHED_SEED,
)
from src.campeones_analysis.multimodal_arousal.decoding_sweep import (
    RANGES, loso_auc, _perm_within, feat_periodic, feat_allbins, _clf,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"

WIDE_TMIN, WIDE_TMAX = -4.0, 3.0
WINDOWS = {"pre": (-4.0, -0.5), "post": (-0.5, 3.0)}
RNG = RANGES["1-40"]
SEED = 20260610
MIN_PER_CLASS = 4  # CV intra adaptativo permite k=min(5, min_clase); floor 2


def _cache(c):
    return TBL_DIR / f"joystick_updown_psd_c{int(round(c * 100)):03d}.npz"


def _build_subject(sub, c):
    npz, runs = load_joystick(sub)
    if npz is None:
        return None
    scale = subject_amplitude_scale(npz, runs, "arousal")
    rng_m = np.random.default_rng(MATCHED_SEED)
    up_eps, dn_eps, ca_eps = [], [], []
    up_mag, dn_mag = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs or run_dimension(npz, label) != "arousal":
            continue
        arr = np.asarray(npz[f"{label}__arousal"], float)
        up, upm = detect_shift_onsets_s(arr, mode="rise", scale=scale, c=c)
        dn, dnm = detect_shift_onsets_s(arr, mode="fall", scale=scale, c=c)
        all_ev = np.concatenate([up, dn]) if (up.size or dn.size) else np.array([])
        if all_ev.size == 0:
            continue
        calma = sample_calma_matched(np.sort(all_ev), arr, 50.0, rng_m, scale=scale)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg"); attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=LOWPASS, verbose="ERROR")
            raw.resample(RESAMPLE, verbose="ERROR")
        except Exception as e:
            print(f"    {label}: FAILED -- {e}", flush=True); continue
        dur = float(raw.times[-1])
        for onsets, mags, eps, mlist in [(up, upm, up_eps, up_mag), (dn, dnm, dn_eps, dn_mag),
                                         (calma, None, ca_eps, None)]:
            on = np.asarray(onsets, float)
            kept = (on + WIDE_TMIN > 0) & (on + WIDE_TMAX < dur)
            ok = on[kept]
            if ok.size == 0:
                continue
            ep = epoch_one_run(raw, ok, code=1, tmin=WIDE_TMIN, tmax=WIDE_TMAX, baseline=(None, None))
            if ep is not None and len(ep):
                eps.append(ep)
                if mlist is not None:
                    mlist.append(np.asarray(mags, float)[kept])
    if not up_eps or not dn_eps:
        return None
    out = {}
    ref_ch = None
    for win, (a, b) in WINDOWS.items():
        psd_sets = {}
        for name, eps in [("up", up_eps), ("down", dn_eps), ("calma", ca_eps)]:
            if not eps:
                continue
            ep = mne.concatenate_epochs(eps, verbose="ERROR").copy().crop(tmin=a, tmax=b)
            psd, freqs, ch = compute_psd(ep)
            if ref_ch is None:
                ref_ch = ch; out["_freqs"] = freqs
            if ch != ref_ch:
                idx = [ch.index(c) for c in ref_ch if c in ch]
                psd = psd[:, idx, :]
            psd_sets[name] = psd.astype(np.float32)
        out[win] = psd_sets
    out["_ch"] = ref_ch
    out["mag_up"] = np.concatenate(up_mag) if up_mag else np.array([])
    out["mag_down"] = np.concatenate(dn_mag) if dn_mag else np.array([])
    return out


def build_cache(c):
    print(f"[B] building up/down/calma cache (arousal; pre+post; c={c}) ...", flush=True)
    store = {}; ref_ch = freqs = None
    for sub in COHORT:
        r = _build_subject(sub, c)
        if r is None:
            print(f"  {sub}: skipped (sin up y down)", flush=True); continue
        if ref_ch is None:
            ref_ch = r["_ch"]; freqs = r["_freqs"]
        for win in WINDOWS:
            for name, psd in r[win].items():
                store[f"psd_{win}_{name}_{sub}"] = psd
        nu = r["post"]["up"].shape[0] if "up" in r["post"] else 0
        nd = r["post"]["down"].shape[0] if "down" in r["post"] else 0
        nc = r["post"]["calma"].shape[0] if "calma" in r["post"] else 0
        print(f"  {sub}: up={nu} down={nd} calma={nc}  "
              f"mag_up p50={np.median(r['mag_up']) if r['mag_up'].size else float('nan'):.2f} "
              f"mag_down p50={np.median(r['mag_down']) if r['mag_down'].size else float('nan'):.2f}",
              flush=True)
    store["freqs"] = freqs; store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(_cache(c), **store)
    print(f"[B] cache -> {_cache(c)}", flush=True)


def _load(z, win, classes, min_per_class=MIN_PER_CLASS):
    """classes = (pos_name, neg_name). Devuelve data[sub]=(psd, y, None), freqs, ch_kept."""
    ch = list(z["ch_names"]); keep = [i for i, c in enumerate(ch) if c not in DROP_CHANNELS]
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
        data[sub] = (psd, y, None)
    return data, z["freqs"], ch_kept


def _intra_adaptive(Xs, ys):
    """Intra-sujeto con k adaptativo = min(5, n de la clase minoritaria); floor 2. Necesario
    porque a c=0.50 algunos sujetos tienen 4 eventos por clase (el 5-fold fijo fallaría)."""
    aucs = []
    for X, y in zip(Xs, ys):
        n_min = int(np.bincount(y).min())
        k = max(2, min(5, n_min))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        proba = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            proba[te] = _clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, proba))
    return float(np.mean(aucs)), aucs


def _eval(data, freqs, builder, n_perm=0):
    Xs = [builder(psd, freqs, RNG) for psd, y, _ in data.values()]
    ys = [y for _, y, _ in data.values()]
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
    CONTRASTS = [("up", "down"), ("up", "calma"), ("down", "calma")]
    rows = []
    for win in WINDOWS:
        for classes in CONTRASTS:
            for fname, builder in [("periódico", feat_periodic),
                                   ("espectro", lambda p, f, r: feat_allbins(p, f, r, "1"))]:
                data, freqs, ch = _load(z, win, classes)
                if len(data) < 3:
                    print(f"  {win:4} {classes[0]:5}-{classes[1]:5} {fname:9}: "
                          f"solo {len(data)} suj (insuf.)", flush=True)
                    continue
                # perms en el contraste clave up-vs-down (ambas ventanas + ambos features)
                np_use = nperm if classes == ("up", "down") else 0
                r = _eval(data, freqs, builder, n_perm=np_use)
                rows.append(dict(window=win, contrast=f"{classes[0]}-{classes[1]}", feature=fname,
                                 c=c, n_subj=len(data), **r))
                pstr = f" p_intra={r['p_intra']} p_loso={r['p_loso']}" if np_use else ""
                print(f"  {win:4} {classes[0]:5}-{classes[1]:5} {fname:9}: "
                      f"intra={r['intra']:.3f} LOSO={r['loso']:.3f} (N={len(data)}, "
                      f"nfeat={r['n_feat']}){pstr}  intra/suj={r['intra_subj']}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["intra_subj"]).to_csv(TBL_DIR / f"updown_decoding_c{int(round(c*100)):03d}.csv",
                                           index=False)

    # figura: AUC por contraste x ventana (periódico)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    per = df[df.feature == "periódico"]
    for ax, win in zip(axes, WINDOWS):
        sub = per[per.window == win]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["intra"], 0.4, label="intra", color="C0", alpha=0.7)
        ax.bar(x + 0.2, sub["loso"], 0.4, label="LOSO", color="C1", alpha=0.7)
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sub["contrast"], fontsize=9)
        ax.set_ylim(0.3, 0.85); ax.set_ylabel("AUC")
        ax.set_title(f"{win}-shift {WINDOWS[win]}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"B. Arousal sube-vs-baja vs calma (periódico, c={c}): ¿up-vs-down separa? (motor↔afecto)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"updown_decoding_c{int(round(c*100)):03d}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--c", type=float, default=0.33, help="umbral de detección (fracción del rango)")
    ap.add_argument("--nperm", type=int, default=1000, help="permutaciones para up-vs-down")
    args = ap.parse_args()
    print("=" * 78); print(f"joystick_updown :: arousal SUBE vs BAJA (c={args.c})")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(args.c); return
    if not _cache(args.c).exists():
        print(f"[B] no cache para c={args.c} -- correr con --build --c {args.c} primero", flush=True); return
    run(args.c, args.nperm)


if __name__ == "__main__":
    main()
