"""Report<->EDA 1-Hz correlation summary, AROUSAL and VALENCE kept fully separate -- Tarea 4.

Headline, meeting-ready artifact. For each dimension SEPARATELY (never mixing arousal & valence)
and for every (report metric x EDA metric) pair, we compute the contemporaneous (lag-0) 1-Hz
Pearson correlation within each affective video segment, aggregate across that subject's segments
by length-weighted Fisher-z, then summarise across the 6 subjects (mean r + sign consistency).

Contemporaneous lag-0 is used deliberately: it avoids the selection inflation of "max over lags".
(The lead/lag question lives in the separate lag-profile module ``eda_joystick_xcorr``.)

report metrics : rep_mean, rep_var, rep_dmean, abs_rep_dmean, rep_dvar
EDA metrics    : eda_mean, tonic_mean, phasic_mean, smna_mean, smna_auc, scr_rate

Input:  eda_joystick/tables/features_1hz.csv
Outputs:
  eda_joystick/tables/corr_summary.csv             long: dim, report, eda, ga_r, sd, n_pos, n_sub
  eda_joystick/tables/corr_persubject.csv          per (sub, dim, report, eda) r
  eda_joystick/figures/corr_heatmap.png            arousal | valence GA heatmaps (presentable)
  eda_joystick/figures/corr_heatmap_persub_{dim}.png   6 per-subject heatmaps (transparency)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eda_joystick_corr_summary
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT as SUBJECTS, OUT

TABLES = OUT / "eda_joystick" / "tables"
FIGS = OUT / "eda_joystick" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DIMS = ("arousal", "valence")
REPORT = ["rep_mean", "rep_var", "rep_dmean", "abs_rep_dmean", "rep_dvar"]
EDA = ["eda_mean", "tonic_mean", "phasic_mean", "smna_mean", "smna_auc", "scr_rate"]
MIN_SEG = 25  # require >=25 s segments

REPORT_LAB = {
    "rep_mean": "report\nmean", "rep_var": "report\nvar", "rep_dmean": "report\nd/dt mean",
    "abs_rep_dmean": "|report\nd/dt mean|", "rep_dvar": "report\nd/dt var\n(motion)",
}
EDA_LAB = {
    "eda_mean": "EDA\nmean", "tonic_mean": "tonic\nmean", "phasic_mean": "phasic\nmean",
    "smna_mean": "SMNA\nmean", "smna_auc": "SMNA\nAUC", "scr_rate": "SCR\nrate",
}


def seg_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_SEG or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fisher_mean(rs: list[float], ws: list[int]) -> float:
    rs = np.asarray(rs, float)
    ws = np.asarray(ws, float)
    ok = np.isfinite(rs)
    if ok.sum() == 0:
        return np.nan
    z = np.arctanh(np.clip(rs[ok], -0.999, 0.999))
    return float(np.tanh(np.sum(z * ws[ok]) / np.sum(ws[ok])))


def main() -> None:
    df = pd.read_csv(TABLES / "features_1hz.csv")
    df["abs_rep_dmean"] = df["rep_dmean"].abs()

    persub_rows: list[dict] = []
    # cell[dim][(rep, eda)] = {sub: r}
    cell: dict[str, dict[tuple, dict[str, float]]] = {
        d: {(rp, ed): {} for rp in REPORT for ed in EDA} for d in DIMS
    }

    for dim in DIMS:
        for sub in SUBJECTS:
            seg_ids = df[(df["sub"] == sub) & (df["dim"] == dim)]["seg_uid"].unique()
            segs = [
                df[(df["sub"] == sub) & (df["seg_uid"] == uid)].sort_values("sec")
                for uid in seg_ids
            ]
            segs = [s for s in segs if len(s) >= MIN_SEG]
            for rp in REPORT:
                for ed in EDA:
                    rs = [seg_corr(s[rp].to_numpy(), s[ed].to_numpy()) for s in segs]
                    ws = [len(s) for s in segs]
                    r = fisher_mean(rs, ws)
                    cell[dim][(rp, ed)][sub] = r
                    persub_rows.append(
                        {"sub": sub, "dim": dim, "report": rp, "eda": ed,
                         "r": round(r, 4) if np.isfinite(r) else np.nan,
                         "n_seg": int(np.isfinite(rs).sum())}
                    )

    pd.DataFrame(persub_rows).to_csv(TABLES / "corr_persubject.csv", index=False)

    # --- GA summary + heatmap matrices ---
    summary_rows: list[dict] = []
    mats: dict[str, np.ndarray] = {}
    cons: dict[str, np.ndarray] = {}
    for dim in DIMS:
        M = np.full((len(REPORT), len(EDA)), np.nan)
        C = np.zeros((len(REPORT), len(EDA)), int)
        for i, rp in enumerate(REPORT):
            for j, ed in enumerate(EDA):
                vals = np.array([cell[dim][(rp, ed)][s] for s in SUBJECTS], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ga = float(np.mean(vals))
                    npos = int((vals > 0).sum())
                    consistency = max(npos, vals.size - npos)
                    M[i, j] = ga
                    C[i, j] = consistency
                    summary_rows.append(
                        {"dim": dim, "report": rp, "eda": ed, "ga_r": round(ga, 4),
                         "sd": round(float(np.std(vals)), 4), "n_pos": npos,
                         "n_sub": int(vals.size)}
                    )
        mats[dim], cons[dim] = M, C

    pd.DataFrame(summary_rows).to_csv(TABLES / "corr_summary.csv", index=False)
    print("corr_summary.csv written. Top |GA r| cells per dim:")
    sdf = pd.DataFrame(summary_rows)
    for dim in DIMS:
        top = sdf[sdf.dim == dim].reindex(
            sdf[sdf.dim == dim]["ga_r"].abs().sort_values(ascending=False).index
        ).head(5)
        print(f"\n[{dim}]")
        print(top.to_string(index=False))

    # --- presentable GA heatmaps ---
    vmax = np.nanmax([np.nanmax(np.abs(mats[d])) for d in DIMS])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, dim in zip(axes, DIMS):
        M, C = mats[dim], cons[dim]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(EDA)))
        ax.set_xticklabels([EDA_LAB[e] for e in EDA], fontsize=8)
        ax.set_yticks(range(len(REPORT)))
        ax.set_yticklabels([REPORT_LAB[r] for r in REPORT], fontsize=8)
        for i in range(len(REPORT)):
            for j in range(len(EDA)):
                if not np.isfinite(M[i, j]):
                    continue
                star = "*" if C[i, j] >= 5 else ""  # >=5/6 subjects agree in sign
                ax.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center",
                        fontsize=8, color="k" if abs(M[i, j]) < 0.6 * vmax else "w")
        ax.set_title(f"{dim}  (n={cons[dim].max() and len(SUBJECTS)} subj)", fontsize=12)
        ax.set_xlabel("EDA metric")
        ax.set_ylabel("report metric")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="cross-subject mean r (lag 0)")
    fig.suptitle(
        "Report <-> EDA 1-Hz correlation, by dimension (separate)  |  * = >=5/6 subjects agree in sign",
        fontsize=13,
    )
    fig.savefig(FIGS / "corr_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nfigure -> corr_heatmap.png")

    # --- per-subject heatmaps (transparency) ---
    for dim in DIMS:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, sub in zip(axes.ravel(), SUBJECTS):
            M = np.array([[cell[dim][(rp, ed)][sub] for ed in EDA] for rp in REPORT])
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(EDA)))
            ax.set_xticklabels([EDA_LAB[e].replace("\n", " ") for e in EDA], fontsize=6, rotation=45)
            ax.set_yticks(range(len(REPORT)))
            ax.set_yticklabels([REPORT_LAB[r].replace("\n", " ") for r in REPORT], fontsize=6)
            for i in range(len(REPORT)):
                for j in range(len(EDA)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
            ax.set_title(sub, fontsize=10)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="r (lag 0)")
        fig.suptitle(f"{dim}: per-subject report<->EDA 1-Hz correlation", fontsize=13)
        fig.savefig(FIGS / f"corr_heatmap_persub_{dim}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> corr_heatmap_persub_{dim}.png")


if __name__ == "__main__":
    main()
