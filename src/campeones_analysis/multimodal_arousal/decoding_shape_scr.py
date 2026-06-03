"""#1 — ¿La decodabilidad SCR-vs-no-SCR es OSCILATORIA o BROADBAND/EMG? (2026-06-03).

El LOSO decodifica SCR-vs-no-SCR (AUC ~0.63) con log-band-power (5 bandas × 32 ch). Descompone
cada época en:
  FULL      : log-power 5 bandas × 32 ch (160 feats) -> reproduce el AUC base.
  BROADBAND : por canal, media de log-power across-bandas (32 feats) = nivel/offset (sube con
              EMG/aperiódico).
  SHAPE     : log-power - media_bandas(log-power) por canal (160 feats) = forma espectral
              (offset removido) = énfasis relativo de bandas (oscilatorio).
LOSO LogisticRegression L2 (mismo esquema que decoding_loso_scr) + null por permutación de labels.
Si SHAPE colapsa a azar y BROADBAND mantiene el AUC -> la decodabilidad es broadband/EMG (cierra
"nada cortical"). Si SHAPE sobrevive -> info oscilatoria residual.

Se corre bajo DOS esquemas de épocas: silent UNIFORME (build_subject_epochs) y silent
TEMPORALMENTE EMPAREJADO (epoch_matched.build_subject_epochs_matched), para controlar temporalidad.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_shape_scr
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs, compute_psd
from src.campeones_analysis.multimodal_arousal.epoch_matched import build_subject_epochs_matched

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = OUT / "qa_artifact_vs_signal" / "shape_decoding"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}
RNG_SEED = 42
N_PERM = 200


def _pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                                            penalty="l2", class_weight="balanced"))


def _band_features(epochs):
    """X[n_ep, n_ch, n_band] = log10 mean band power. Returns (X, ch_names)."""
    psd, freqs, ch = compute_psd(epochs)  # (n_ep, n_ch, n_freq)
    feats = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log10(psd[:, :, m].mean(axis=2) + 1e-30))  # (n_ep, n_ch)
    return np.stack(feats, axis=2), list(ch)  # (n_ep, n_ch, n_band)


def _variants(X):
    """FULL / BROADBAND / SHAPE feature matrices from X[n_ep, n_ch, n_band]."""
    n = X.shape[0]
    bb = X.mean(axis=2)                              # (n_ep, n_ch)  offset/level
    shape = X - X.mean(axis=2, keepdims=True)        # (n_ep, n_ch, n_band) band-demeaned
    return {"FULL": X.reshape(n, -1), "BROADBAND": bb, "SHAPE": shape.reshape(n, -1)}


def _zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _loso(data, key):
    """LOSO mean AUC + permutation null for feature variant `key`. data[sub]=(variants, y)."""
    subs = list(data)
    aucs = []
    for ts in subs:
        tr = [s for s in subs if s != ts]
        Xtr = np.vstack([data[s][0][key] for s in tr]); ytr = np.concatenate([data[s][1] for s in tr])
        Xte, yte = data[ts][0][key], data[ts][1]
        pipe = _pipe(); pipe.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    mean_auc = float(np.mean(aucs))
    rng = np.random.default_rng(RNG_SEED)
    perm = np.empty(N_PERM)
    for p in range(N_PERM):
        fa = []
        for ts in subs:
            tr = [s for s in subs if s != ts]
            Xtr = np.vstack([data[s][0][key] for s in tr]); ytr = np.concatenate([data[s][1] for s in tr])
            pipe = _pipe(); pipe.fit(Xtr, rng.permutation(ytr))
            fa.append(roc_auc_score(data[ts][1], pipe.predict_proba(data[ts][0][key])[:, 1]))
        perm[p] = np.mean(fa)
    return dict(mean_auc=round(mean_auc, 4), perm95=round(float(np.percentile(perm, 95)), 4),
                p_emp=round(float((np.sum(perm >= mean_auc) + 1) / (N_PERM + 1)), 4),
                aucs=[round(a, 3) for a in aucs])


def _build(scheme):
    builder = build_subject_epochs if scheme == "uniform" else build_subject_epochs_matched
    data = {}
    for sub in COHORT:
        try:
            real_ep, silent_ep = builder(sub)
        except Exception as e:
            print(f"  {scheme} {sub}: FAILED {e}", flush=True); continue
        if real_ep is None or silent_ep is None or len(real_ep) == 0 or len(silent_ep) == 0:
            print(f"  {scheme} {sub}: no epochs", flush=True); continue
        Xr, _ = _band_features(real_ep); Xs, _ = _band_features(silent_ep)
        X = np.vstack([Xr, Xs])
        y = np.concatenate([np.ones(len(Xr), int), np.zeros(len(Xs), int)])
        var = _variants(X)
        var = {k: _zscore(v) for k, v in var.items()}   # per-subject z-score (label-free)
        data[sub] = (var, y)
        print(f"  {scheme} {sub}: real={len(Xr)} silent={len(Xs)}", flush=True)
    return data


def main():
    print("=" * 78)
    print("decoding_shape_scr :: FULL vs BROADBAND vs SHAPE, uniform vs matched epochs")
    print("=" * 78, flush=True)
    rows = []
    for scheme in ("uniform", "matched"):
        print(f"\n--- scheme: {scheme} ---", flush=True)
        data = _build(scheme)
        if len(data) < 3:
            print(f"  {scheme}: <3 subjects, skip", flush=True); continue
        for key in ("FULL", "BROADBAND", "SHAPE"):
            r = _loso(data, key)
            rows.append(dict(scheme=scheme, features=key, **{k: r[k] for k in ("mean_auc", "perm95", "p_emp")},
                             aucs=str(r["aucs"])))
            print(f"  {scheme:8s} {key:10s} AUC={r['mean_auc']:.3f}  perm95={r['perm95']:.3f}  p={r['p_emp']:.4f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "shape_decoding_summary.csv", index=False)

    # figure: grouped bars AUC by variant, uniform vs matched
    fig, ax = plt.subplots(figsize=(10, 6))
    variants = ["FULL", "BROADBAND", "SHAPE"]
    x = np.arange(len(variants)); w = 0.38
    for j, scheme in enumerate(("uniform", "matched")):
        sub = df[df.scheme == scheme].set_index("features")
        if sub.empty:
            continue
        vals = [sub.loc[v, "mean_auc"] if v in sub.index else np.nan for v in variants]
        p95 = [sub.loc[v, "perm95"] if v in sub.index else np.nan for v in variants]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, label=f"{scheme}",
                      color=["C0", "C1"][j], alpha=0.85)
        for xi, v, pp in zip(x + (j - 0.5) * w, vals, p95):
            ax.plot([xi - w / 2, xi + w / 2], [pp, pp], color="0.3", lw=1.4)  # perm95 marker
            ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="azar")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("LOSO AUC"); ax.set_ylim(0.4, 0.8)
    ax.set_title("Decodabilidad SCR-vs-no-SCR: forma espectral vs broadband.\n"
                 "Si SHAPE≈azar y BROADBAND alto -> la decodabilidad es broadband/EMG. "
                 "(línea gris=perm95)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shape_decoding.png", dpi=130)
    plt.close(fig)

    print("\n" + df.to_string(index=False), flush=True)
    print(f"\nOutputs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
