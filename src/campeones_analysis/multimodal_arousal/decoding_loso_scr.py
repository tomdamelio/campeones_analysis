"""Cross-subject Leave-One-Subject-Out (LOSO) decoding of SCR vs no-SCR + interpretability.

Train on 5 subjects, test on the held-out 6th, rotating over the cohort. Tests whether the
SCR-vs-no-SCR spectral pattern GENERALIZES across people (robustness with several subjects),
the key question for the cross-subject (Version II / LOSO) goal. Runs on the cohort6
(relaxed-criterion) epochs, so there are many epochs to pool.

Reuses (no reimplementation):
  - cohort.COHORT / OUT / NPZ_DIR
  - tfr_psd_scr.build_subject_epochs(sub) -> (real_ep, silent_ep)
  - decoding_scr.epochs_to_band_features(epochs) -> (X[n_epochs, 32ch*5bands], feat_names "ch|band")

Design:
  - Spectral band-power features (log10), identical feature order across subjects -> poolable.
  - Per-subject z-score (label-free) to remove inter-subject scale/offset, then a train-fit
    StandardScaler inside the pipeline. Model: LogisticRegression(L2, C=1.0, class_weight=balanced).
  - LOSO outer loop (6 folds). Metrics per held-out subject: AUC, accuracy, F1.
  - Permutation null of the mean LOSO AUC (shuffle train labels) -> empirical p.

Interpretability (complete):
  - Cross-subject generalizing features: mean LR coefficient across the 6 folds + sign
    consistency; topomaps per band (delta..gamma) -> central-parietal (signal) vs edges (artifact).
  - Per-subject heterogeneity: merge each subject's within-subject top features from
    decoding_spectral.csv.
  - What drives per-subject performance: held-out AUC vs n_epochs; and LOSO vs within-subject
    (LORO) AUC from decoding_3models_per_subject.csv.

Outputs (additive, under cohort6/):
  y_candidates/loso_per_subject.csv, y_candidates/loso_coefficients.csv
  figures/decoding_loso/{auc_by_subject,coef_topomaps,auc_vs_nepochs}.png

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.decoding_loso_scr
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR, OUT
from src.campeones_analysis.multimodal_arousal.decoding_scr import epochs_to_band_features
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures" / "decoding_loso"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42
N_PERM = 200


def _pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                           penalty="l2", class_weight="balanced"),
    )


def build_subject_Xy(sub: str) -> dict | None:
    """Per-subject z-scored spectral features + labels (1=real SCR, 0=silent)."""
    real_ep, silent_ep = build_subject_epochs(sub)
    if real_ep is None or silent_ep is None or len(real_ep) == 0 or len(silent_ep) == 0:
        return None
    Xr, feat_names = epochs_to_band_features(real_ep)
    Xs, _ = epochs_to_band_features(silent_ep)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(len(Xr), int), np.zeros(len(Xs), int)])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd  # per-subject z-score (label-free) removes inter-subject scale
    return dict(X=Xz, y=y, n_real=len(Xr), n_silent=len(Xs),
                feat_names=feat_names, info=real_ep.info, ch_names=list(real_ep.ch_names))


def band_order(feat_names: list[str]) -> list[str]:
    seen: list[str] = []
    for f in feat_names:
        b = f.split("|")[1]
        if b not in seen:
            seen.append(b)
    return seen


def main() -> int:
    print("=" * 78)
    print("LOSO cross-subject decoding (spectral + LR) -- cohort6")
    print("=" * 78)

    data: dict[str, dict] = {}
    for sub in COHORT:
        d = build_subject_Xy(sub)
        if d is None:
            print(f"  {sub}: no epochs -> skip")
            continue
        data[sub] = d
        print(f"  {sub}: X={d['X'].shape}  real={d['n_real']} silent={d['n_silent']}")
    subs = list(data.keys())
    if len(subs) < 3:
        print("[ERROR] need >=3 subjects for LOSO")
        return 1
    feat_names = data[subs[0]]["feat_names"]
    info = data[subs[0]]["info"]
    ch_names = data[subs[0]]["ch_names"]
    bands = band_order(feat_names)
    n_band, n_ch = len(bands), len(ch_names)

    # --- LOSO ---
    rows, coefs, acts, aucs = [], [], [], []
    for test_sub in subs:
        tr = [s for s in subs if s != test_sub]
        Xtr = np.vstack([data[s]["X"] for s in tr])
        ytr = np.concatenate([data[s]["y"] for s in tr])
        Xte, yte = data[test_sub]["X"], data[test_sub]["y"]
        pipe = _pipe()
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred)
        aucs.append(auc)
        w = pipe.named_steps["logisticregression"].coef_.ravel()
        coefs.append(w)
        # Haufe (2014) activation pattern A = Cov(X_scaled) @ w, on the features the LR saw
        # (StandardScaler-transformed training pool). Forward model -> interpretable topomap.
        Xtr_sc = pipe.named_steps["standardscaler"].transform(Xtr)
        acts.append(np.cov(Xtr_sc, rowvar=False) @ w)
        rows.append(dict(test_subject=test_sub, n_train=len(ytr), n_test=len(yte),
                         n_real_test=data[test_sub]["n_real"], n_silent_test=data[test_sub]["n_silent"],
                         auc=float(auc), acc=float(acc), f1=float(f1)))
        print(f"  holdout {test_sub}: AUC={auc:.3f}  acc={acc:.3f}  f1={f1:.3f}")
    mean_auc = float(np.mean(aucs))
    print(f"\nMean LOSO AUC = {mean_auc:.3f} (chance 0.5)")

    # --- permutation null of mean AUC (shuffle train labels) ---
    rng = np.random.default_rng(RNG_SEED)
    perm = np.zeros(N_PERM)
    for p in range(N_PERM):
        fa = []
        for test_sub in subs:
            tr = [s for s in subs if s != test_sub]
            Xtr = np.vstack([data[s]["X"] for s in tr])
            ytr = np.concatenate([data[s]["y"] for s in tr])
            pipe = _pipe()
            pipe.fit(Xtr, rng.permutation(ytr))
            fa.append(roc_auc_score(data[test_sub]["y"],
                                    pipe.predict_proba(data[test_sub]["X"])[:, 1]))
        perm[p] = np.mean(fa)
    p_emp = float((np.sum(perm >= mean_auc) + 1) / (N_PERM + 1))
    perm95 = float(np.percentile(perm, 95))
    print(f"Permutation: null mean AUC 95th pct={perm95:.3f}  p_emp={p_emp:.4f}")

    # --- merge within-subject (LORO) AUC + within-subject top feat for context ---
    loro_auc, top_feat = {}, {}
    try:
        d3 = pd.read_csv(NPZ_DIR / "decoding_3models_per_subject.csv")
        sp = d3[d3["model"] == "spectral"]
        loro_auc = dict(zip(sp["subject"], sp["mean_auc"]))
    except Exception as exc:
        print(f"  (no LORO comparison: {exc})")
    try:
        ds = pd.read_csv(NPZ_DIR / "decoding_spectral.csv")
        top_feat = dict(zip(ds["subject"], ds["top_feat"]))
    except Exception as exc:
        print(f"  (no within-subject top_feat: {exc})")
    for r in rows:
        r["loro_within_auc"] = float(loro_auc.get(r["test_subject"], np.nan))
        r["within_top_feat"] = top_feat.get(r["test_subject"], "")

    # --- save CSVs FIRST (persist before plotting) ---
    df = pd.DataFrame(rows)
    df.attrs["mean_loso_auc"] = mean_auc
    summary = dict(mean_loso_auc=mean_auc, mean_loso_acc=float(np.mean([r["acc"] for r in rows])),
                   perm_null_auc_95=perm95, perm_p=p_emp, n_subjects=len(subs), n_perm=N_PERM)
    df.to_csv(NPZ_DIR / "loso_per_subject.csv", index=False)
    pd.DataFrame([summary]).to_csv(NPZ_DIR / "loso_summary.csv", index=False)

    coef_arr = np.vstack(coefs)  # (n_folds, n_feat)
    mean_coef = coef_arr.mean(axis=0)
    sign_consistency = np.mean(np.sign(coef_arr) == np.sign(mean_coef)[None, :], axis=0)
    act_arr = np.vstack(acts)  # (n_folds, n_feat) -- Haufe activation patterns
    mean_act = act_arr.mean(axis=0)
    act_sign_consistency = np.mean(np.sign(act_arr) == np.sign(mean_act)[None, :], axis=0)
    pd.DataFrame(dict(feature=feat_names, mean_coef=mean_coef,
                      std_coef=coef_arr.std(axis=0), sign_consistency=sign_consistency,
                      mean_activation=mean_act, act_sign_consistency=act_sign_consistency)
                 ).to_csv(NPZ_DIR / "loso_coefficients.csv", index=False)
    print(f"CSVs -> {NPZ_DIR}/loso_per_subject.csv, loso_coefficients.csv, loso_summary.csv")

    # --- figure 1: AUC per held-out subject (LOSO vs LORO) ---
    try:
        x = np.arange(len(subs))
        loso_v = [df.loc[df.test_subject == s, "auc"].values[0] for s in subs]
        loro_v = [loro_auc.get(s, np.nan) for s in subs]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - 0.2, loso_v, 0.4, label="LOSO (cross-subject)", color="C0")
        ax.bar(x + 0.2, loro_v, 0.4, label="LORO (within-subject)", color="C1", alpha=0.8)
        ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
        ax.axhline(mean_auc, color="C0", ls="--", lw=1, label=f"LOSO mean={mean_auc:.2f}")
        ax.axhline(perm95, color="0.5", ls="--", lw=1, label=f"perm 95%={perm95:.2f}")
        ax.set_xticks(x); ax.set_xticklabels(subs); ax.set_ylabel("AUC"); ax.set_ylim(0, 1)
        ax.set_title(f"SCR-vs-no-SCR decoding AUC  |  LOSO mean={mean_auc:.3f} (p_perm={p_emp:.3f})")
        ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "auc_by_subject.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig1 failed: {exc}")

    # --- figure 2: coefficient topomaps per band ---
    try:
        cmat = mean_coef.reshape(n_band, n_ch)
        vmax = float(np.abs(cmat).max())
        fig, axes = plt.subplots(1, n_band, figsize=(3 * n_band, 3.4))
        for bi, bname in enumerate(bands):
            mne.viz.plot_topomap(cmat[bi], info, axes=axes[bi], show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), contours=0)
            axes[bi].set_title(bname, fontsize=10)
        fig.suptitle("LOSO mean LR coefficient per band (red=+ real, blue=+ silent)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIG_DIR / "coef_topomaps.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig2 (topomaps) failed: {exc}")

    # --- figure 2b: Haufe ACTIVATION pattern topomaps per band (the interpretable forward map) ---
    try:
        amat = mean_act.reshape(n_band, n_ch)
        vmaxa = float(np.abs(amat).max())
        fig, axes = plt.subplots(1, n_band, figsize=(3 * n_band, 3.4))
        for bi, bname in enumerate(bands):
            mne.viz.plot_topomap(amat[bi], info, axes=axes[bi], show=False,
                                 cmap="RdBu_r", vlim=(-vmaxa, vmaxa), contours=0)
            axes[bi].set_title(bname, fontsize=10)
        fig.suptitle("LOSO Haufe ACTIVATION pattern per band (forward model; red=+ real SCR, blue=+ silent)",
                     fontsize=11)
        fig.tight_layout(); fig.savefig(FIG_DIR / "haufe_activation_topomaps.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig2b (Haufe activation) failed: {exc}")

    # --- figure 3: held-out AUC vs n_epochs (what drives per-subject performance) ---
    try:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        n_te = [df.loc[df.test_subject == s, "n_test"].values[0] for s in subs]
        ax.scatter(n_te, loso_v, s=60)
        for s, nx, ay in zip(subs, n_te, loso_v):
            ax.annotate(s, (nx, ay), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.5, color="k", ls=":", lw=1)
        if len(subs) >= 3:
            r = np.corrcoef(n_te, loso_v)[0, 1]
            ax.set_title(f"Held-out AUC vs nº epochs (test)   r={r:.2f}")
        ax.set_xlabel("n_test epochs (held-out subject)"); ax.set_ylabel("LOSO AUC"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "auc_vs_nepochs.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print(f"  fig3 failed: {exc}")

    print(f"Figures -> {FIG_DIR}")
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nMean LOSO AUC={mean_auc:.3f}  acc={summary['mean_loso_acc']:.3f}  "
          f"perm95={perm95:.3f}  p={p_emp:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
