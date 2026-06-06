"""Plot indicativo de decoding: SMNA-AUC real vs predicha (out-of-fold, LORO).

Reusa EXACTAMENTE la máquina del lens L3 de eeg_smna_coupling (ridge + z-score intra-run +
lagged design por-run + LORO), pero en vez de reportar solo R2/p, genera las predicciones
out-of-fold (cada run predicho cuando es held-out) y las grafica contra la señal real.

Por banda en {delta, gamma} (las dos que le ganan al null), tres paneles:
  (A) serie temporal real vs predicha (EEG-solo) en un sujeto representativo (R2 = mediana),
      runs concatenados con líneas divisorias;
  (B) scatter real vs predicha agrupado cross-subject, con R2 LORO (media de sujetos);
  (C) barra honesta R2: AR-solo (pasado de la SMNA) vs EEG-solo vs Full (AR+EEG)
      -> muestra que el EEG casi no agrega sobre el AR (dR2 ~ 0).

Todo z-score INTRA-run (como el análisis). Out-of-fold = sin leakage. Descriptivo
(no descuenta estímulo).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decode_real_vs_pred
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from src.campeones_analysis.multimodal_arousal.eeg_smna_coupling import (
    AR_LAGS,
    FIG_DIR,
    WIN_LAGS,
    _best_alpha,
    _lag_matrix_perrun,
    _r2,
    _zscore_perrun,
    load_all,
)

warnings.filterwarnings("ignore")

PLOT_BANDS = ["delta", "gamma"]


def loro_oof(X, y, run, alpha):
    """Predicción out-of-fold: cada run se predice con un modelo entrenado en los demás."""
    yp = np.full(len(y), np.nan)
    for g in np.unique(run):
        te = run == g
        tr = ~te
        if te.sum() < 5 or tr.sum() < 5:
            continue
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
    return yp


def subject_designs(d, band):
    """Devuelve (y, run, mask_valido, X_ar, X_eeg, X_full) para un sujeto y banda, z-score intra-run."""
    run = d["run"]
    y = _zscore_perrun(d["y"], run)
    ar = _lag_matrix_perrun(y, run, AR_LAGS)
    bp = _zscore_perrun(d[f"bp__{band}__PO"], run)
    eeg = _lag_matrix_perrun(bp, run, WIN_LAGS)
    v = np.all(np.isfinite(ar), axis=1) & np.all(np.isfinite(eeg), axis=1)
    full = np.hstack([ar, eeg])
    return y, run, v, ar, eeg, full


def main():
    data = load_all()
    print(f"sujetos con cache: {list(data)}", flush=True)

    fig, axes = plt.subplots(len(PLOT_BANDS), 3, figsize=(18, 4.6 * len(PLOT_BANDS)))
    if len(PLOT_BANDS) == 1:
        axes = axes[None, :]

    for bi, band in enumerate(PLOT_BANDS):
        # --- por sujeto: OOF EEG-solo + R2 de los 3 modelos ---
        per_sub = {}
        r2_ar_list, r2_eeg_list, r2_full_list = [], [], []
        pooled_real, pooled_pred = [], []
        for sub, d in data.items():
            y, run, v, ar, eeg, full = subject_designs(d, band)
            yv, runv = y[v], run[v]
            a_eeg = _best_alpha(eeg[v], yv, runv)
            a_ar = _best_alpha(ar[v], yv, runv)
            a_full = _best_alpha(full[v], yv, runv)
            yp_eeg = loro_oof(eeg[v], yv, runv, a_eeg)
            yp_ar = loro_oof(ar[v], yv, runv, a_ar)
            yp_full = loro_oof(full[v], yv, runv, a_full)
            ok = np.isfinite(yp_eeg)
            r2_eeg = _r2(yv[ok], yp_eeg[ok])
            r2_ar = _r2(yv[ok & np.isfinite(yp_ar)], yp_ar[ok & np.isfinite(yp_ar)])
            r2_full = _r2(yv[ok & np.isfinite(yp_full)], yp_full[ok & np.isfinite(yp_full)])
            r2_eeg_list.append(r2_eeg)
            r2_ar_list.append(r2_ar)
            r2_full_list.append(r2_full)
            per_sub[sub] = dict(y=yv, run=runv, yp=yp_eeg, yp_ar=yp_ar, r2=r2_eeg)
            pooled_real.append(yv[ok])
            pooled_pred.append(yp_eeg[ok])

        r2_eeg_mean = float(np.nanmean(r2_eeg_list))
        r2_ar_mean = float(np.nanmean(r2_ar_list))
        r2_full_mean = float(np.nanmean(r2_full_list))

        # sujeto representativo = R2 mediana (ni mejor ni peor)
        subs_sorted = sorted(per_sub, key=lambda s: per_sub[s]["r2"])
        rep = subs_sorted[len(subs_sorted) // 2]
        rp = per_sub[rep]

        # --- Panel A: serie temporal real vs predicha (sujeto representativo) ---
        axA = axes[bi, 0]
        yv, runv, yp, yp_ar = rp["y"], rp["run"], rp["yp"], rp["yp_ar"]
        order = []
        boundaries = [0]
        for g in np.unique(runv):
            idx = np.where(runv == g)[0]
            order.extend(idx.tolist())
            boundaries.append(len(order))
        order = np.array(order)
        t = np.arange(len(order))
        axA.plot(t, yv[order], color="k", lw=1.0, label="SMNA real (z, intra-run)")
        axA.plot(t, yp_ar[order], color="0.55", lw=1.0, alpha=0.9,
                 label="predicha AR-solo (pasado SMNA)")
        axA.plot(t, yp[order], color="crimson", lw=1.2, alpha=0.85, label="predicha EEG-solo (OOF)")
        for b0 in boundaries[1:-1]:
            axA.axvline(b0, color="0.7", lw=0.6, ls="--")
        axA.set_title(f"{band}  |  {rep} (R2 mediana = {rp['r2']:+.3f})\nreal vs predicha out-of-fold, runs concatenados")
        axA.set_xlabel("ventana (~2 s) — runs separados por líneas")
        axA.set_ylabel("SMNA-AUC (z intra-run)")
        axA.legend(fontsize=8, loc="upper right")

        # --- Panel B: scatter agrupado real vs predicha ---
        axB = axes[bi, 1]
        pr = np.concatenate(pooled_real)
        pp = np.concatenate(pooled_pred)
        axB.scatter(pp, pr, s=6, alpha=0.18, color="crimson", edgecolors="none")
        lim = np.percentile(np.abs(np.concatenate([pr, pp])), 99)
        axB.plot([-lim, lim], [-lim, lim], color="k", lw=0.8, ls="--", label="identidad")
        # línea de regresión
        bcoef = np.polyfit(pp, pr, 1)
        xs = np.array([-lim, lim])
        axB.plot(xs, bcoef[0] * xs + bcoef[1], color="navy", lw=1.2, label="ajuste")
        axB.set_xlim(-lim, lim)
        axB.set_ylim(-lim, lim)
        axB.set_title(f"{band}  |  agrupado cross-subject\nR2 LORO (media sujetos) = {r2_eeg_mean:+.3f}")
        axB.set_xlabel("SMNA predicha (EEG-solo, OOF)")
        axB.set_ylabel("SMNA real (z)")
        axB.legend(fontsize=8, loc="upper left")

        # --- Panel C: barra honesta AR vs EEG vs Full ---
        axC = axes[bi, 2]
        vals = [r2_ar_mean, r2_eeg_mean, r2_full_mean]
        labels = ["AR-solo\n(pasado SMNA)", "EEG-solo\n(band-PO)", "Full\n(AR+EEG)"]
        colors = ["0.5", "crimson", "navy"]
        bars = axC.bar(range(3), vals, color=colors)
        for r, val in zip(bars, vals):
            axC.text(r.get_x() + r.get_width() / 2, val, f"{val:+.3f}",
                     ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
        axC.axhline(0, color="k", lw=0.5)
        axC.set_xticks(range(3))
        axC.set_xticklabels(labels, fontsize=8)
        axC.set_ylabel("R2 LORO (media sujetos)")
        dr2 = r2_full_mean - r2_ar_mean
        axC.set_title(f"{band}  |  ¿el EEG agrega sobre el AR?\ndR2 (Full - AR) = {dr2:+.4f}")

    fig.suptitle("Decoding SMNA-AUC desde EEG band-power parieto-occipital (out-of-fold, LORO, z intra-run)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = FIG_DIR / "decode_real_vs_pred.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado -> {out}", flush=True)
    print("\nR2 LORO (media sujetos) por banda:")
    print("  (ver título de paneles para AR/EEG/Full)", flush=True)


if __name__ == "__main__":
    main()
