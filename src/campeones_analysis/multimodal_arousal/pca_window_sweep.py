"""Sanity check: re-do the PC1 of (SMNA, HR, RVT) at different effective window sizes,
loading the per-window features cached by ``build_y_candidates`` (no re-cvxEDA needed).

Motivation: ``dmt-emotions`` aligned the 3 physiological features at 30 s windows where
inter-modality lags are negligible. CAMPEONES first pass used 2 s windows where lags
(SCR ~1-3 s, HR <1 s, RVT ~3-6 s) might wash out the cross-modal correlation. This sweep
shows whether the PC1 cleans up at coarser windows (which would be empirical evidence in
favour of lag-per-modality / DAMN! config (ii) for the Diego meeting on 2026-05-14).

Approach: take the 1 Hz feature time-series from the cached .npz, **block-average** them
into target window sizes (1, 2, 5, 10, 30, 60 s), drop NaN rows, z-score within subject,
and refit PCA. Report EVR + PC1 loadings as a function of window size.

Run:  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.pca_window_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
OUT = REPO / "research_diary" / "context" / "05_02"
NPZ_DIR = OUT / "y_candidates"

SUBJECTS = ["sub-23", "sub-24", "sub-33"]
WINDOWS_S = [1, 2, 5, 10, 30, 60]  # the cache is at 1 Hz (hop=1 s, win=2 s); aggregations are integer factors


def block_mean(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    n = (len(x) // k) * k
    if n == 0:
        return np.array([])
    return x[:n].reshape(-1, k).mean(axis=1)


def main():
    rows = []
    per_subject_loadings: dict = {}

    for sub in SUBJECTS:
        npz = np.load(NPZ_DIR / f"{sub}_y_candidates.npz", allow_pickle=True)
        runs = list(npz["runs"])
        per_subject_loadings[sub] = {"win_s": [], "evr": [], "pc1_loadings": [], "n_valid": []}
        for win_s in WINDOWS_S:
            k = win_s  # 1 Hz cache -> aggregation factor = win_s
            feats = []
            for r in runs:
                # use the *smoothed* y1 (Y1 final from the first script)
                y1 = npz[f"{r}__y1s"]
                hr = npz[f"{r}__hr"]
                rvt = npz[f"{r}__rvt"]
                X = np.column_stack([block_mean(y1, k), block_mean(hr, k), block_mean(rvt, k)])
                feats.append(X)
            Xall = np.vstack(feats)
            valid = np.isfinite(Xall).all(axis=1)
            if valid.sum() < 10:
                continue
            mu = np.nanmean(Xall[valid], axis=0)
            sd = np.nanstd(Xall[valid], axis=0)
            sd[sd == 0] = 1.0
            Z = (Xall[valid] - mu) / sd
            pca = PCA(n_components=3)
            pca.fit(Z)
            comps = pca.components_.copy()
            if comps[0, 0] < 0:
                comps[0] *= -1
            evr = pca.explained_variance_ratio_.tolist()
            pc1 = comps[0].tolist()
            per_subject_loadings[sub]["win_s"].append(win_s)
            per_subject_loadings[sub]["evr"].append(evr)
            per_subject_loadings[sub]["pc1_loadings"].append(pc1)
            per_subject_loadings[sub]["n_valid"].append(int(valid.sum()))
            rows.append(dict(
                subject=sub, win_s=win_s, n_valid=int(valid.sum()),
                evr_pc1=evr[0], evr_pc2=evr[1], evr_pc3=evr[2],
                pc1_load_smna=pc1[0], pc1_load_hr=pc1[1], pc1_load_rvt=pc1[2],
            ))
            print(f"  {sub}  win={win_s:3d}s  n_valid={valid.sum():>5}  "
                  f"EVR={[round(e, 3) for e in evr]}  PC1=({pc1[0]:+.2f},{pc1[1]:+.2f},{pc1[2]:+.2f})")

    df = pd.DataFrame(rows)
    df.to_csv(NPZ_DIR / "pca_window_sweep.csv", index=False)
    with open(NPZ_DIR / "pca_window_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(per_subject_loadings, fh, indent=2)

    # Summary figure: 2x3 grid (rows = EVR/PC1-loadings, cols = subjects)
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(5 * len(SUBJECTS), 7), sharex=True)
    for j, sub in enumerate(SUBJECTS):
        sub_df = df[df["subject"] == sub]
        # Top: EVR PC1/PC2/PC3 vs window size
        ax = axes[0, j]
        ax.plot(sub_df["win_s"], sub_df["evr_pc1"], "o-", label="EVR PC1")
        ax.plot(sub_df["win_s"], sub_df["evr_pc2"], "s--", label="EVR PC2")
        ax.plot(sub_df["win_s"], sub_df["evr_pc3"], "^--", label="EVR PC3")
        ax.axhline(1 / 3, color="k", lw=0.5, ls=":", label="uniform (1/3)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(sub)
        ax.set_ylabel("explained variance ratio") if j == 0 else None
        ax.legend(fontsize=7)
        # Bottom: PC1 loadings vs window size
        ax = axes[1, j]
        ax.plot(sub_df["win_s"], sub_df["pc1_load_smna"], "o-", label="SMNA")
        ax.plot(sub_df["win_s"], sub_df["pc1_load_hr"], "s-", label="HR")
        ax.plot(sub_df["win_s"], sub_df["pc1_load_rvt"], "^-", label="RVT")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("window size (s, log scale)")
        ax.set_ylabel("PC1 loading") if j == 0 else None
        ax.legend(fontsize=7)
    fig.suptitle("PCA window-size sweep: ¿el PC1 'arousal común' emerge a granularidad gruesa?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "figures" / "Y_candidates_pca_window_sweep.png", dpi=130)
    plt.close(fig)

    print(f"\nDone. Outputs: {NPZ_DIR/'pca_window_sweep.csv'},"
          f" {NPZ_DIR/'pca_window_sweep.json'},"
          f" {OUT/'figures'/'Y_candidates_pca_window_sweep.png'}")


if __name__ == "__main__":
    main()
