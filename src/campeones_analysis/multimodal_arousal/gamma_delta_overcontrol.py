"""(3) Over-control test — ¿gamma/delta colapsan TAMBIÉN bajo un covariado muscular de banda INDEPENDIENTE?

El strict-test que tumbó gamma (0.635->0.511) y delta (0.600->0.528) usó como covariado la potencia
CRUDA de la MISMA banda de la VD (gamma 30-40 edge para Q-gamma; delta 1-4 edge para Q-delta) -> posible
**over-control / circularidad** (el covariado y la VD comparten banda y canales vecinos). Caveat explícito
del veredicto provisional del Bloque 2.

Test limpio (lo que pide la tarea 3): re-correr el strict-test con un covariado muscular de una banda
**DISTINTA E INDEPENDIENTE** = la potencia **60-90 Hz por época** (donde el córtex casi no tiene potencia
= EMG casi puro; ya establecido EMG-POSITIVE 6/6 en `emg_highfreq_scr`). Re-derivada del stream ancho
1-100 Hz post-ICA (`recon_wide`), per-época, alineada 1:1 al cache (verificado por tnorm).
  - Si gamma/delta COLAPSAN también controlando 60-90 (banda independiente) -> NO era over-control, es
    artefacto (el músculo broadband per-época explica la subida).
  - Si SOBREVIVEN -> el strict-test previo sobre-controlaba (el covariado de la misma banda comía señal).

Covariado 60-90 = stream post-ICA (residual EMG; ICA no remueve el EMG broadband); VD = cache (no-ICA).
Mezclar streams para un NUISANCE es conservador (banda + stream independientes). Hereda Track B.

Run (paso 1 cachea el covariado 60-90; paso 2 corre los tests):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.gamma_delta_overcontrol --nperm 1000
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

import src.campeones_analysis.multimodal_arousal.erp_scr as _erp
from src.campeones_analysis.multimodal_arousal.cohort import COHORT, EMG_EDGE, REPO
from src.campeones_analysis.multimodal_arousal.confound_model_scr import (
    COV22,
    OUT_DIR,
    _load_covariates,
    _perm_p,
    evaluate,
    feat_band,
    feat_offset,
)
from src.campeones_analysis.multimodal_arousal.decoding_panel import (
    UNIFORM_SEED,
    _epoch_times_tnorm,
    load_cache,
)
from src.campeones_analysis.multimodal_arousal.decoding_sweep import RANGES
from src.campeones_analysis.multimodal_arousal.emg_highfreq_scr import hf_bin_mask
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    detect_scr_onsets_s,
    epoch_one_run,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)
from src.campeones_analysis.multimodal_arousal.recon_wide import (
    compute_psd_wide,
    reconstruct_wide_run,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
HF_CACHE = NPZ_DIR / "hf6090_edge_percepoch.npz"   # covariado 60-90 edge per-época, cache-aligned

GBAND, DBAND = (30.0, 40.0), (1.0, 4.0)
GAMMA_LIGHT = ["gamma_EOG", "sp_hf", "heog_gamma_30_40", "edge_gamma", "var_jerk", "tnorm", "tnorm2"]
DELTA_LIGHT = ["blink_slow", "blink_2hz_pre", "veog_slow_0p5_8", "veog_slow_2hz_pre",
               "var_jerk_0p5_2", "tnorm", "tnorm2"]
DELTA_MOTION = ["var_jerk_x", "var_jerk_y", "var_jerk_z", "var_jerk_0p5_8", "var_jerk"]
POSTERIOR_CH = ["Pz", "P3", "P4", "O1", "O2", "CP1", "CP2"]


def _hf_edge_per_epoch(psd, freqs, e_idx):
    """Potencia 60-90 Hz (dB, sin faldas de notch) por canal edge, por época. (n_ep, n_edge)."""
    m = hf_bin_mask(freqs)
    hf = 10.0 * np.log10(psd[:, :, m].mean(axis=2) + 1e-30)
    return hf[:, e_idx]


def build_hf_covariate(cache, ch):
    """60-90 edge per-época del stream wide post-ICA, en orden del cache (real luego silent), 1:1."""
    e_idx_wide = None
    _erp.RNG = np.random.default_rng(UNIFORM_SEED)   # MISMO flujo RNG que el cache uniform
    store = {}
    for sub in COHORT:
        psd_c, y, tn = cache[sub]
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        runs_in_npz = list(cont["runs"])
        real_hf, sil_hf, real_tn, sil_tn = [], [], [], []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            if label not in runs_in_npz:
                continue
            raw = reconstruct_wide_run(sub, label, apply_ica=True, resample_hz=250.0)
            dur = float(raw.times[-1])
            eda = np.asarray(cont[f"{label}__eda_phasic"], float)
            ons = detect_scr_onsets_s(eda, EDA_FS)
            onsets = filter_clean_onsets(ons[ons < dur], eda, EDA_FS)
            sil = sample_silent_controls(n_target=len(onsets), duration_s=dur, phasic=eda,
                                         fs=EDA_FS, rng=_erp.RNG, avoid_onsets_s=onsets)
            er = epoch_one_run(raw, onsets, code=1)
            si = epoch_one_run(raw, sil, code=2)
            if er is None or si is None:
                print(f"    {label}: epochs None -> skip run", flush=True); continue
            pr, fr, chw = compute_psd_wide(er, fmax=95.0)
            ps, _, _ = compute_psd_wide(si, fmax=95.0)
            if e_idx_wide is None:
                e_idx_wide = [chw.index(c) for c in EMG_EDGE if c in chw]
            real_hf.append(_hf_edge_per_epoch(pr, fr, e_idx_wide)); real_tn.append(_epoch_times_tnorm(er, dur))
            sil_hf.append(_hf_edge_per_epoch(ps, fr, e_idx_wide)); sil_tn.append(_epoch_times_tnorm(si, dur))
        hf = np.vstack(real_hf + sil_hf)
        my_tn = np.concatenate(real_tn + sil_tn)
        ok = len(hf) == len(tn) and np.max(np.abs(my_tn - tn)) < 1e-3
        print(f"  {sub}: hf60-90 edge {hf.shape}  align={'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise RuntimeError(f"{sub}: HF covariate misaligned with cache (n={len(hf)} vs {len(tn)})")
        store[sub] = hf
    np.savez_compressed(HF_CACHE, **{s: store[s] for s in store},
                        edge_ch=np.array([c for c in EMG_EDGE]))
    print(f"  -> HF covariate cacheado en {HF_CACHE}", flush=True)
    return store


def _run_test(band_name, Xs, ys, C_light, C_strict, nperm):
    rows = []
    for name, C in [("light (=2.5)", C_light), ("strict-INDEP (+60-90 edge)", C_strict)]:
        ri, rl, groups = evaluate(Xs, ys, C, deconf=False)
        di, dl, _ = evaluate(Xs, ys, C, deconf=True)
        pi, pl = _perm_p(Xs, ys, C, True, di, dl, nperm, groups)
        rows.append(dict(band=band_name, covariados=name, raw_loso=round(rl, 4),
                         deconf_loso=round(dl, 4), p_deconf_loso=pl))
        print(f"  [{band_name}] {name:28s}: loso {rl:.3f}->{dl:.3f} (p={pl})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true", help="recomputar el covariado 60-90 (ignora cache)")
    args = ap.parse_args()
    print("=" * 78)
    print("gamma_delta_overcontrol :: ¿colapsan bajo 60-90 Hz (banda INDEPENDIENTE)?")
    print("=" * 78, flush=True)

    cache, freqs, ch = load_cache("uniform")
    covs = _load_covariates(cache)

    if HF_CACHE.exists() and not args.rebuild:
        z = np.load(HF_CACHE, allow_pickle=True)
        hf_cov = {s: z[s] for s in COHORT if s in z.files}
        print(f"  HF covariate cargado de cache ({HF_CACHE.name})", flush=True)
    else:
        print("  construyendo covariado 60-90 edge per-época (wide post-ICA, ~ICA por run)...", flush=True)
        hf_cov = build_hf_covariate(cache, ch)

    e_idx = [ch.index(c) for c in EMG_EDGE if c in ch]
    a_idx = [ch.index(c) for c in (["Fp1", "Fp2", "F7", "F8"] + EMG_EDGE) if c in ch]
    p_idx = [ch.index(c) for c in POSTERIOR_CH if c in ch]
    d22 = pd.read_csv(COV22)

    # VDs + covariados
    Xg, Xd_all, Xd_post, ys = [], [], [], []
    Cg_light, Cg_strict, Cd_light, Cd_strict = [], [], [], []
    for sub in cache:
        psd, y, tn = cache[sub]
        hf = hf_cov[sub]                                              # (n_ep, n_edge) 60-90 INDEP
        ys.append(y.astype(int))
        # ----- gamma -----
        Xg.append(feat_band(psd, freqs, RANGES["1-40"], GBAND))
        clg = np.column_stack([covs[sub][c] for c in GAMMA_LIGHT])
        Cg_light.append(clg)
        Cg_strict.append(np.column_stack([clg, hf]))                 # + 60-90 edge (independiente)
        # ----- delta -----
        dper = feat_band(psd, freqs, RANGES["1-30"], DBAND)
        Xd_all.append(dper); Xd_post.append(dper[:, p_idx])
        cld = np.column_stack([covs[sub][c] for c in DELTA_LIGHT])
        m = d22[d22.subject == sub].reset_index(drop=True)
        motion = np.column_stack([m[c].to_numpy() for c in DELTA_MOTION])
        Cd_light.append(cld)
        Cd_strict.append(np.column_stack([cld, motion, hf]))         # + movimiento + 60-90 edge

    print(f"  60-90 edge: {hf_cov[list(hf_cov)[0]].shape[1]} canales edge; nperm={args.nperm}\n", flush=True)

    rows = []
    rows += _run_test("gamma", Xg, ys, Cg_light, Cg_strict, args.nperm)
    rows += _run_test("delta all-ch", Xd_all, ys, Cd_light, Cd_strict, args.nperm)
    rows += _run_test("delta posterior", Xd_post, ys, Cd_light, Cd_strict, args.nperm)
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "gamma_delta_overcontrol.csv", index=False)

    print("\n=== VEREDICTO OVER-CONTROL (banda independiente 60-90) ===")
    for band in ("gamma", "delta all-ch", "delta posterior"):
        s = df[df.band == band]
        light = s[s.covariados == "light (=2.5)"]["deconf_loso"].values[0]
        strict = s[s.covariados.str.startswith("strict")]["deconf_loso"].values[0]
        ps = s[s.covariados.str.startswith("strict")]["p_deconf_loso"].values[0]
        col = "COLAPSA -> artefacto (no over-control)" if (strict < 0.54 or ps >= 0.05) else "SOBREVIVE -> el strict previo sobre-controlaba"
        print(f"  {band:16s}: light {light:.3f} -> strict-INDEP {strict:.3f} (p={ps}) => {col}", flush=True)

    _plot(df)
    print(f"\nOutputs -> {TBL_DIR}\n[over-control] done", flush=True)


def _plot(df):
    bands = ["gamma", "delta all-ch", "delta posterior"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bands)); w = 0.38
    light = [df[(df.band == b) & (df.covariados == "light (=2.5)")]["deconf_loso"].values[0] for b in bands]
    strict = [df[(df.band == b) & (df.covariados.str.startswith("strict"))]["deconf_loso"].values[0] for b in bands]
    ps = [df[(df.band == b) & (df.covariados.str.startswith("strict"))]["p_deconf_loso"].values[0] for b in bands]
    ax.bar(x - w / 2, light, w, label="light (=2.5)", color="C0")
    ax.bar(x + w / 2, strict, w, label="strict-INDEP (+60-90 edge)", color="C3")
    for xi, (d, p) in enumerate(zip(strict, ps)):
        ax.annotate(f"p={p:.3f}", (xi + w / 2, d + 0.005), ha="center", fontsize=8,
                    color="darkred" if p < 0.05 else "0.4")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_ylim(0.45, 0.7)
    ax.set_ylabel("LOSO AUC deconfounded")
    ax.set_title("Over-control test: ¿gamma/delta colapsan bajo 60-90 Hz (banda INDEPENDIENTE)?\n"
                 "colapso => artefacto confirmado (no era over-control)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gamma_delta_overcontrol.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
