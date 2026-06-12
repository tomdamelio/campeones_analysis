"""Bloque 4.1/4.2 — Cache PSD de épocas joystick (evento-vs-calma) + panel decoding intra/LOSO.

Análogo a `decoding_panel.py` (SCR), pero las épocas se anclan a EVENTOS de cambio afectivo
abrupto del joystick (ver `joystick_events.py`) y la ventana EEG es PRE-SHIFT [-4,-0.5] s
(termina ANTES del movimiento de la palanca -> limpia de artefacto motor). El cache guarda,
por sujeto y por esquema de control {uniforme, emparejado}, la PSD por época + label + tnorm +
DIMENSIÓN (arousal/valence) + MAGNITUD del cambio (|Δ|/rango). Las tres variantes del bloque se
arman filtrando por dimensión en load_cache:
  A (arousal)  : dim == arousal
  B (valencia) : dim == valence
  C (combinado): sin filtro (unión a nivel de evento)

Reusa la maquinaria de decoding SIN reescribirla: feature builders + clasificador LR-L2 +
intra/loso/perm de decoding_sweep, y eval_fset/haufe_pattern/_boot_ci de decoding_panel
(operan sobre `data` pasado por argumento, no sobre el cache de SCR).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_panel --build
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_panel --dim arousal [--nperm 100]
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_panel --dim valence
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_panel --dim combined
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

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT, REPO, SUBJ_COLORS, DROP_CHANNELS,
)
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import compute_psd
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    attach_montage_and_drop_no_pos,
    epoch_one_run,
    run_label,
    runs_for,
)
from src.campeones_analysis.multimodal_arousal.joystick_events import (
    ANALYSIS_TMIN, ANALYSIS_TMAX, C_EVENT, DIMS,
    load_joystick, run_dimension, subject_amplitude_scale,
    detect_shift_onsets_s, sample_calma_matched, sample_calma_uniform,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import (
    FEATURE_SETS, PANEL_RANGES, PRIMARY_RANGE, eval_fset, haufe_pattern, _make_info, _topo,
)
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES, BANDS_40, BANDS_30

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "joystick_panel"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
CACHE = TBL_DIR / "joystick_psd.npz"

LOWPASS = 40.0
RESAMPLE = 250.0
UNIFORM_SEED = 20260609
MATCHED_SEED = 20260610
DIM_CODE = {"arousal": 0, "valence": 1}
DIM_NAME = {0: "arousal", 1: "valence"}


# =============================================================================
# Cache build: pre-shift PSD per evento + calma {uniforme, emparejado}
# =============================================================================
def _psd_of(raw, times_s, code):
    """Epoca en la ventana pre-shift [-4,-0.5] (baseline=None) y devuelve (psd, freqs, ch, kept_mask)
    donde kept_mask alinea con `times_s` los que sobreviven el chequeo de ventana de epoch_one_run."""
    times_s = np.asarray(times_s, float)
    if times_s.size == 0:
        return None, None, None, None
    dur = float(raw.times[-1])
    kept = (times_s + ANALYSIS_TMIN > 0) & (times_s + ANALYSIS_TMAX < dur)
    ts = times_s[kept]
    if ts.size == 0:
        return None, None, None, kept
    # baseline=(None,None) = DC del epoch completo (neutral para PSD; Welch ya detrenda por
    # segmento). NO se puede pasar None: epoch_one_run lo interpreta como el baseline SCR por
    # defecto (-5,-4.5), fuera de la ventana pre-shift.
    ep = epoch_one_run(raw, ts, code=code, tmin=ANALYSIS_TMIN, tmax=ANALYSIS_TMAX,
                       baseline=(None, None))
    if ep is None or len(ep) == 0:
        return None, None, None, kept
    psd, freqs, ch = compute_psd(ep)
    return psd.astype(np.float32), freqs, ch, kept


def _build_subject(sub):
    """Devuelve dict por esquema con psd/y/tnorm/dim/mag apilados (evento luego calma), o None.

    Carga cada run UNA vez; los eventos son deterministas (idénticos en ambos esquemas), solo
    cambia el muestreo de calma (uniforme vs emparejado)."""
    npz, runs = load_joystick(sub)
    if npz is None:
        return None
    scales = {d: subject_amplitude_scale(npz, runs, d) for d in DIMS}
    rng_u = np.random.default_rng(UNIFORM_SEED)
    rng_m = np.random.default_rng(MATCHED_SEED)

    ev_psd, ev_tn, ev_dim, ev_mag = [], [], [], []
    cu_psd, cu_tn, cu_dim = [], [], []
    cm_psd, cm_tn, cm_dim = [], [], []
    ref_ch, freqs = None, None

    def _align(psd, ch):
        nonlocal ref_ch, freqs
        if ref_ch is None:
            ref_ch = ch
            return psd
        if ch != ref_ch:
            idx = [ch.index(c) for c in ref_ch if c in ch]
            return psd[:, idx, :]
        return psd

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
        calma_u = sample_calma_uniform(onsets, arr, 50.0, rng_u, scale=scale)
        calma_m = sample_calma_matched(onsets, arr, 50.0, rng_m, scale=scale)
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=LOWPASS, verbose="ERROR")
            raw.resample(RESAMPLE, verbose="ERROR")
        except Exception as e:
            print(f"    {label}: FAILED -- {e}", flush=True)
            continue
        dur = float(raw.times[-1])
        dcode = DIM_CODE[dim]

        # eventos
        psd, fr, ch, kept = _psd_of(raw, onsets, code=1)
        if psd is not None:
            freqs = fr
            psd = _align(psd, ch)
            ok = onsets[kept]; mg = mags[kept]
            ev_psd.append(psd)
            ev_tn.append(ok / dur)
            ev_dim.append(np.full(len(ok), dcode))
            ev_mag.append(mg)
        # calma uniforme
        psd, fr, ch, kept = _psd_of(raw, calma_u, code=2)
        if psd is not None:
            psd = _align(psd, ch)
            ck = calma_u[kept]
            cu_psd.append(psd); cu_tn.append(ck / dur); cu_dim.append(np.full(len(ck), dcode))
        # calma emparejada
        psd, fr, ch, kept = _psd_of(raw, calma_m, code=2)
        if psd is not None:
            psd = _align(psd, ch)
            ck = calma_m[kept]
            cm_psd.append(psd); cm_tn.append(ck / dur); cm_dim.append(np.full(len(ck), dcode))

    if not ev_psd:
        return None
    E = np.concatenate(ev_psd, 0)
    e_tn = np.concatenate(ev_tn); e_dim = np.concatenate(ev_dim); e_mag = np.concatenate(ev_mag)

    out = {}
    for scheme, cpsd, ctn, cdim in [("uniform", cu_psd, cu_tn, cu_dim),
                                    ("matched", cm_psd, cm_tn, cm_dim)]:
        if not cpsd:
            continue
        C = np.concatenate(cpsd, 0)
        c_tn = np.concatenate(ctn); c_dim = np.concatenate(cdim)
        psd = np.concatenate([E, C], 0).astype(np.float32)
        y = np.concatenate([np.ones(len(E)), np.zeros(len(C))]).astype(int)
        tn = np.concatenate([e_tn, c_tn]).astype(np.float32)
        dm = np.concatenate([e_dim, c_dim]).astype(int)
        mg = np.concatenate([e_mag, np.zeros(len(C))]).astype(np.float32)
        out[scheme] = (psd, y, tn, dm, mg)
    out["_ch"] = ref_ch
    out["_freqs"] = freqs
    return out


def build_cache():
    print("[4.1] building joystick cache (uniforme + emparejado; pre-shift PSD) ...", flush=True)
    store = {}
    ref_ch = freqs = None
    for sub in COHORT:
        r = _build_subject(sub)
        if r is None:
            print(f"  {sub}: skipped (sin eventos)", flush=True)
            continue
        if ref_ch is None:
            ref_ch = r["_ch"]; freqs = r["_freqs"]
        for scheme in ("uniform", "matched"):
            if scheme not in r:
                continue
            psd, y, tn, dm, mg = r[scheme]
            store[f"psd_{scheme}_{sub}"] = psd
            store[f"y_{scheme}_{sub}"] = y
            store[f"tnorm_{scheme}_{sub}"] = tn
            store[f"dim_{scheme}_{sub}"] = dm
            store[f"mag_{scheme}_{sub}"] = mg
        u = r.get("uniform"); m = r.get("matched")
        na = int((r["uniform"][3][r["uniform"][1] == 1] == 0).sum()) if u else 0
        nv = int((r["uniform"][3][r["uniform"][1] == 1] == 1).sum()) if u else 0
        print(f"  {sub}: eventos arousal={na} valence={nv}  "
              f"calma_unif={int((u[1]==0).sum()) if u else 0} "
              f"calma_emp={int((m[1]==0).sum()) if m else 0}", flush=True)
    store["freqs"] = freqs
    store["ch_names"] = np.array(ref_ch)
    np.savez_compressed(CACHE, **store)
    print(f"[4.1] cache -> {CACHE}", flush=True)


def load_cache(scheme, dim=None, min_per_class=1):
    """Carga el cache, descarta DROP_CHANNELS (29 ch) y opcionalmente filtra por dimensión.

    dim in {'arousal','valence', None}. None = combinado (sin filtro).
    min_per_class: descarta sujetos con < N épocas en cualquier clase (decoding usa 5 por el
    5-fold; spectral 1). Devuelve (data, freqs, ch_kept, extra) con data[sub]=(psd[:,keep,:],
    y, tnorm) y extra[sub]=(dim_arr, mag_arr) alineado por época (para 4.6 graded)."""
    z = np.load(CACHE, allow_pickle=True)
    freqs = z["freqs"]; ch = list(z["ch_names"])
    keep = [i for i, c in enumerate(ch) if c not in DROP_CHANNELS]
    ch_kept = [ch[i] for i in keep]
    want = None if dim is None else DIM_CODE[dim]
    data, extra = {}, {}
    for sub in COHORT:
        k = f"psd_{scheme}_{sub}"
        if k not in z:
            continue
        psd = z[k].astype(float)[:, keep, :]
        y = z[f"y_{scheme}_{sub}"]; tn = z[f"tnorm_{scheme}_{sub}"]
        dm = z[f"dim_{scheme}_{sub}"]; mg = z[f"mag_{scheme}_{sub}"]
        if want is not None:
            sel = dm == want
            psd, y, tn, dm, mg = psd[sel], y[sel], tn[sel], dm[sel], mg[sel]
        if int(y.sum()) < min_per_class or int((1 - y).sum()) < min_per_class:
            continue  # sujeto con muy pocas épocas por clase (no soporta el CV)
        data[sub] = (psd, y, tn)
        extra[sub] = (dm, mg)
    return data, freqs, ch_kept, extra


# =============================================================================
# Panel decoding (reusa eval_fset / haufe_pattern de decoding_panel)
# =============================================================================
def _plot_panel(df, data, dim_tag):
    sub = df[df["range"] == PRIMARY_RANGE].set_index("feature_set")
    order = [f for f in FEATURE_SETS if f in sub.index]
    subs = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, scheme in zip(axes, ["intra", "loso"]):
        auc = [sub.loc[f, f"{scheme}_auc"] for f in order]
        lo = [sub.loc[f, f"{scheme}_ci_lo"] for f in order]
        hi = [sub.loc[f, f"{scheme}_ci_hi"] for f in order]
        x = np.arange(len(order))
        ax.bar(x, auc, color="C0", alpha=0.55, width=0.6)
        ax.errorbar(x, auc, yerr=[np.array(auc) - np.array(lo), np.array(hi) - np.array(auc)],
                    fmt="none", ecolor="k", capsize=4, lw=1.4)
        for xi, f in enumerate(order):
            pts = sub.loc[f, f"{scheme}_subj"]
            for k, s in enumerate(subs):
                ax.scatter(xi + np.linspace(-0.22, 0.22, len(subs))[k], pts[k],
                           color=SUBJ_COLORS.get(s, "k"), s=42, zorder=4, label=s if xi == 0 else None)
            ax.text(xi, hi[xi] + 0.01, f"p={sub.loc[f, f'p_{scheme}']:.3f}", ha="center", fontsize=7)
        ax.axhline(0.5, color="0.5", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"{scheme.upper()} AUC"); ax.set_ylim(0.30, 0.85)
        ax.set_title(f"{scheme.upper()} ({PRIMARY_RANGE} Hz)", fontsize=9)
        if scheme == "intra":
            ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.suptitle(f"4.2 Joystick {dim_tag}: decoding evento-vs-calma intra vs LOSO (emparejado)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / f"panel_{dim_tag}.png", dpi=110); plt.close(fig)


def _plot_haufe(data, freqs, ch, dim_tag):
    info = _make_info(ch); nch = len(ch)
    bands = BANDS_40 if RANGES[PRIMARY_RANGE][1] > 30 else BANDS_30
    A = haufe_pattern("periódico", data, freqs, PRIMARY_RANGE)
    bnames = list(bands.keys())
    fig, axes = plt.subplots(1, len(bnames), figsize=(2.4 * len(bnames), 3.4))
    for j, bn in enumerate(bnames):
        _topo(axes[j], A[j * nch:(j + 1) * nch], info, bn)
    fig.suptitle(f"4.2 Haufe periódico por banda — {dim_tag} (rojo=decoder se apoya en evento>calma)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / f"haufe_periodic_{dim_tag}.png", dpi=120); plt.close(fig)


def run_panel(dim_tag, n_perm):
    dim = None if dim_tag == "combined" else dim_tag
    data, freqs, ch, _ = load_cache("matched", dim=dim, min_per_class=5)  # 5-fold intra
    n_ev = sum(int(d[1].sum()) for d in data.values())
    n_ca = sum(int((1 - d[1]).sum()) for d in data.values())
    print(f"[4.2] {dim_tag}: {len(data)} sujetos, {n_ev} eventos / {n_ca} calma (emparejado), "
          f"{len(ch)} ch, {len(freqs)} freqs", flush=True)
    rows = []
    for fset in FEATURE_SETS:
        for rng_name in PANEL_RANGES:
            np_use = n_perm if rng_name == PRIMARY_RANGE else 0
            r = eval_fset(fset, data, freqs, rng_name, np_use)
            rows.append(r)
            print(f"  {fset:22s} {rng_name}: intra={r['intra_auc']:.3f} "
                  f"[{r['intra_ci_lo']:.2f},{r['intra_ci_hi']:.2f}] (p={r['p_intra']})  "
                  f"LOSO={r['loso_auc']:.3f} (p={r['p_loso']})  "
                  f"intra/suj={r['intra_subj']}  nfeat={r['n_features']}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["intra_subj", "loso_subj"]).to_csv(
        TBL_DIR / f"decoding_{dim_tag}.csv", index=False)
    _plot_panel(df, data, dim_tag)
    _plot_haufe(data, freqs, ch, dim_tag)
    print(f"[4.2] {dim_tag} done -> {OUT_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dim", choices=["arousal", "valence", "combined"], default="arousal")
    ap.add_argument("--nperm", type=int, default=100)
    args = ap.parse_args()
    print("=" * 78)
    print(f"joystick_panel :: build={args.build} dim={args.dim}")
    print(f"OUT -> {OUT_DIR}")
    print("=" * 78, flush=True)
    if args.build:
        build_cache(); return
    if not CACHE.exists():
        print("[4.x] no cache -- correr con --build primero", flush=True); return
    run_panel(args.dim, args.nperm)


if __name__ == "__main__":
    main()
