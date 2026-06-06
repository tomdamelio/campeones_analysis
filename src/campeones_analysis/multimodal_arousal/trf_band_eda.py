"""C1 — TRF continuo EEG-banda (PO) <-> EDA phasic, R0 descriptivo, 3 bandas (delta/alfa/gamma).

Lagged-ridge manual (diseño por-run, sin fuga entre runs) para control total del baseline AR y del
null circular-shift. Por sujeto y banda:
  - FORWARD (EEG_po -> EDA): kernel(lag), R2 held-out (LORO), peak lag.
  - Baseline AR target-only (EDA desde su pasado) + dR2 = R2(AR+EEG) - R2(AR): el acoplamiento cuenta
    solo si EEG le gana al AR.
  - BACKWARD (EDA -> EEG_po): R2 (pista direccional fwd-vs-bwd, NO prueba).
  - Null circular-shift (roll del EEG dentro de cada run) sobre R2 forward, p empírico.
DELTA además: topomap GA del R2 forward por canal (central-parietal = SCR-RO vs edge/frontal).

DESCRIPTIVO: no descuenta el estímulo (eso sería VARX). Lags -2..+6 s (Branković: EEG adelanta EDA
0.7-3.7 s + 1-4 s de conducción periférica). LORO, nunca k-fold aleatorio.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.trf_band_eda
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
from src.campeones_analysis.multimodal_arousal.continuous_band import (
    BANDS,
    COMMON_FS,
    build_subject_continuous,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "continuous_bands" / "trf"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

TMIN, TMAX = -2.0, 6.0  # s; positive lag = EEG leads EDA
DMIN, DMAX = int(round(TMIN * COMMON_FS)), int(round(TMAX * COMMON_FS))  # -50..+150
LAGS = np.arange(DMIN, DMAX + 1)
LAGS_S = LAGS / COMMON_FS
AR_LAGS = np.arange(1, int(round(1.0 * COMMON_FS)) + 1)  # 1..25 = 1 s of EDA past
ALPHAS = np.logspace(-1, 4, 6)
N_PERM = 100
RUN_PERM = False  # sin permutaciones (rápido para la reunión); True para el null circular-shift
RNG = np.random.default_rng(20260603)


def lag_matrix(x, lags):
    """design[t,k] = x[t - lags[k]] (zeros at edges)."""
    T = len(x)
    X = np.zeros((T, len(lags)))
    for k, L in enumerate(lags):
        if L > 0:
            X[L:, k] = x[: T - L]
        elif L < 0:
            X[: T + L, k] = x[-L:]
        else:
            X[:, k] = x
    return X


def _valid(T):
    """Row mask dropping edge rows contaminated by lag zeros."""
    m = np.ones(T, bool)
    if DMAX > 0:
        m[:DMAX] = False
    if DMIN < 0:
        m[T + DMIN:] = False
    return m


def _r2(yt, yp):
    ss = ((yt - yt.mean()) ** 2).sum()
    return 1.0 - ((yt - yp) ** 2).sum() / ss if ss > 0 else np.nan


def _loro_r2(designs, ys, alpha):
    """LORO mean R2: designs/ys are per-run lists; returns mean held-out R2."""
    n = len(designs)
    r2s = []
    for i in range(n):
        Xtr = np.vstack([designs[j] for j in range(n) if j != i])
        ytr = np.concatenate([ys[j] for j in range(n) if j != i])
        mdl = Ridge(alpha=alpha).fit(Xtr, ytr)
        r2s.append(_r2(ys[i], mdl.predict(designs[i])))
    return float(np.mean(r2s))


def _best_alpha(designs, ys):
    return ALPHAS[int(np.argmax([_loro_r2(designs, ys, a) for a in ALPHAS]))]


def _per_run_designs(runs, feat_key, band, eeg_to_eda=True):
    """Build (design, y) per run. feat_key: 'po'. eeg_to_eda True=forward, False=backward."""
    designs, ys = [], []
    for lab, r in runs.items():
        eeg = r["bands"][band][feat_key]
        eda = r["eda"]
        T = len(eda)
        v = _valid(T)
        if eeg_to_eda:
            X = lag_matrix(eeg, LAGS)[v]
            y = eda[v]
        else:
            X = lag_matrix(eda, LAGS)[v]
            y = eeg[v]
        designs.append(X)
        ys.append(y)
    return designs, ys


def _ar_designs(runs):
    designs, ys = [], []
    for lab, r in runs.items():
        eda = r["eda"]; T = len(eda); v = _valid(T)
        designs.append(lag_matrix(eda, AR_LAGS)[v]); ys.append(eda[v])
    return designs, ys


def _full_designs(runs, band):
    """AR(eda past) + EEG_po lags, for dR2-over-AR."""
    designs, ys = [], []
    for lab, r in runs.items():
        eda = r["eda"]; eeg = r["bands"][band]["po"]; T = len(eda); v = _valid(T)
        X = np.hstack([lag_matrix(eda, AR_LAGS)[v], lag_matrix(eeg, LAGS)[v]])
        designs.append(X); ys.append(eda[v])
    return designs, ys


def _kernel(designs, ys, alpha):
    """Refit on all data, return forward coef vs lag."""
    X = np.vstack(designs); y = np.concatenate(ys)
    return Ridge(alpha=alpha).fit(X, y).coef_


def subject_band(runs, band):
    df_des, df_y = _per_run_designs(runs, "po", band, eeg_to_eda=True)
    alpha = _best_alpha(df_des, df_y)
    r2_fwd = _loro_r2(df_des, df_y, alpha)
    kernel = _kernel(df_des, df_y, alpha)
    peak_lag = float(LAGS_S[int(np.argmax(np.abs(kernel)))])
    # AR baseline + dR2
    ar_des, ar_y = _ar_designs(runs)
    a_ar = _best_alpha(ar_des, ar_y)
    r2_ar = _loro_r2(ar_des, ar_y, a_ar)
    full_des, full_y = _full_designs(runs, band)
    r2_full = _loro_r2(full_des, full_y, _best_alpha(full_des, full_y))  # alpha propio (fix del bug)
    dr2 = r2_full - r2_ar
    # backward
    bd, by = _per_run_designs(runs, "po", band, eeg_to_eda=False)
    r2_bwd = _loro_r2(bd, by, _best_alpha(bd, by))
    # circular-shift null on forward R2 (opcional; off por defecto para la reunión)
    p_emp = np.nan
    if RUN_PERM:
        labs = list(runs)
        null = np.empty(N_PERM)
        for p in range(N_PERM):
            sd = []
            for lab in labs:
                eeg = runs[lab]["bands"][band]["po"]; eda = runs[lab]["eda"]; T = len(eda); v = _valid(T)
                sh = int(RNG.integers(COMMON_FS, T - COMMON_FS))
                sd.append(lag_matrix(np.roll(eeg, sh), LAGS)[v])
            null[p] = _loro_r2(sd, df_y, alpha)
        p_emp = float((np.sum(null >= r2_fwd) + 1) / (N_PERM + 1))
    return dict(band=band, alpha=alpha, r2_fwd=round(r2_fwd, 4), r2_ar=round(r2_ar, 4),
                r2_full=round(r2_full, 4), dr2_over_ar=round(dr2, 4), r2_bwd=round(r2_bwd, 4),
                peak_lag_s=peak_lag, p_fwd=round(p_emp, 4), kernel=kernel.tolist())


def _per_channel_r2_delta(runs):
    """Forward R2 per channel (delta), LORO, for the topomap."""
    ch = next(iter(runs.values()))["ch"]
    n_ch = len(ch)
    r2 = np.full(n_ch, np.nan)
    for c in range(n_ch):
        des, ys = [], []
        for lab, r in runs.items():
            eeg = r["bands"]["delta"]["all"][:, c]; eda = r["eda"]; T = len(eda); v = _valid(T)
            des.append(lag_matrix(eeg, LAGS)[v]); ys.append(eda[v])
        r2[c] = _loro_r2(des, ys, 1.0)
    return ch, r2


def main():
    print("=" * 78)
    print(f"trf_band_eda :: TRF continuo EEG-PO<->EDA  bandas={list(BANDS)}  lags {TMIN}..{TMAX}s")
    print("=" * 78, flush=True)
    rows = []
    kernels = {b: {} for b in BANDS}
    ref_info = None
    ch_r2_delta = {}
    for sub in COHORT:
        runs = build_subject_continuous(sub)
        if not runs:
            print(f"  {sub}: no runs", flush=True); continue
        if ref_info is None:
            # build a montage info for topomap from preproc of this subject
            import mne as _mne
            from src.campeones_analysis.multimodal_arousal.erp_scr import runs_for
            raw0 = _mne.io.read_raw_brainvision(runs_for(sub)[0], preload=False, verbose="ERROR").pick("eeg")
            from src.campeones_analysis.multimodal_arousal.erp_scr import attach_montage_and_drop_no_pos
            raw0.load_data(); attach_montage_and_drop_no_pos(raw0)
            ref_info = raw0.info
        for b in BANDS:
            r = subject_band(runs, b)
            kernels[b][sub] = np.array(r.pop("kernel"))
            rows.append(dict(subject=sub, **r))
            print(f"  {sub} {b}: R2_fwd={r['r2_fwd']:+.3f} dR2_AR={r['dr2_over_ar']:+.4f} "
                  f"R2_bwd={r['r2_bwd']:+.3f} peak_lag={r['peak_lag_s']:+.1f}s p={r['p_fwd']:.3f}", flush=True)
        ch, r2 = _per_channel_r2_delta(runs)
        for cn, val in zip(ch, r2):
            ch_r2_delta.setdefault(cn, []).append(val)

    df = pd.DataFrame(rows)
    df.drop(columns=[]).to_csv(TBL_DIR / "trf_summary.csv", index=False)

    # kernels figure (per band, GA + per subject)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, b in zip(axes, BANDS):
        ks = np.array([kernels[b][s] for s in kernels[b]])
        for i, s in enumerate(kernels[b]):
            ax.plot(LAGS_S, kernels[b][s], lw=0.7, alpha=0.5)
        ax.plot(LAGS_S, ks.mean(0), color="k", lw=2.2, label="GA")
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(1, 4, color="gold", alpha=0.12)  # Branković EEG-leads window
        ax.set_title(f"{b}  forward kernel (EEG_po->EDA)", fontsize=10)
        ax.set_xlabel("lag (s)  [+ = EEG adelanta EDA]")
    axes[0].set_ylabel("peso TRF"); axes[0].legend(fontsize=8)
    fig.suptitle("TRF forward EEG-PO -> EDA por banda (zona dorada = ventana Brankovic EEG adelanta 1-4 s)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG_DIR / "trf_kernels.png", dpi=130); plt.close(fig)

    # dR2-over-AR bars (3 bands x subjects)
    fig, ax = plt.subplots(figsize=(12, 5))
    subs = df["subject"].unique(); x = np.arange(len(subs)); w = 0.25
    for j, b in enumerate(BANDS):
        vals = [df[(df.subject == s) & (df.band == b)]["dr2_over_ar"].values[0] for s in subs]
        ax.bar(x + (j - 1) * w, vals, w, label=b)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(subs)
    ax.set_ylabel("dR2 sobre AR (forward)"); ax.legend()
    ax.set_title("¿El EEG le gana al baseline AR? dR2 por banda y sujeto", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "trf_dr2_over_ar.png", dpi=130); plt.close(fig)

    # delta per-channel R2 topomap (GA)
    if ref_info is not None:
        ga = np.array([np.nanmean(ch_r2_delta.get(c, [np.nan])) for c in ref_info.ch_names])
        fig, axx = plt.subplots(figsize=(5, 4.5))
        vmax = float(np.nanmax(np.abs(ga))) if np.any(np.isfinite(ga)) else 0.01
        im, _ = mne.viz.plot_topomap(ga, ref_info, axes=axx, show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", contours=4, sensors=True)
        fig.colorbar(im, ax=axx, shrink=0.7, label="R2 forward (delta->EDA)")
        axx.set_title("GA topomap R2 forward delta->EDA\n(central-parietal=SCR-RO; edge/frontal=artefacto)",
                      fontsize=9)
        fig.tight_layout(); fig.savefig(FIG_DIR / "trf_delta_topomap.png", dpi=130); plt.close(fig)

    print("\n" + df.drop(columns=["alpha"]).to_string(index=False), flush=True)
    print("\nGA por banda (dR2 sobre AR, p forward):")
    for b in BANDS:
        sub = df[df.band == b]
        print(f"  {b:6s} dR2_AR GA={sub['dr2_over_ar'].mean():+.4f} ({(sub['dr2_over_ar']>0).sum()}/{len(sub)} pos)  "
              f"R2_fwd GA={sub['r2_fwd'].mean():+.3f}  p<0.05: {(sub['p_fwd']<0.05).sum()}/{len(sub)}", flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
