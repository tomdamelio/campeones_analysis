"""Per-stimulus EDA <-> report exploration -- metric-grid layout -- Tarea 4.

For each participant and dimension (arousal / valence), one figure laid out as a metric grid:
rows = 1-Hz aggregated metrics, columns = the ~14 video stimuli, each cell a sparkline of that
metric over time within that video. y-limits are shared per row (across stimuli) so a metric is
comparable across the 14 videos. Report metrics are drawn in red, EDA metrics in green; SCR
onsets are shown as marks on the ``scr_rate`` row.

Input:
  eda_joystick/tables/features_1hz.csv
Output:
  eda_joystick/figures/stimgrid_{sub}_{dim}.png

Run (one subject):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_eda_stim_viz sub-33
Run (all subjects):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_eda_stim_viz
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT as SUBJECTS, OUT

TABLES = OUT / "eda_joystick" / "tables"
FIGS = OUT / "eda_joystick" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

REP_COLOR = "tab:red"
EDA_COLOR = "tab:green"
SIGNED = {"rep_mean", "rep_dmean"}  # metrics that straddle 0 -> draw a baseline

# (metric, family); order = top-to-bottom rows
METRIC_ROWS = [
    ("rep_mean", "report"),
    ("rep_var", "report"),
    ("rep_dmean", "report"),
    ("rep_dvar", "report"),
    ("eda_mean", "eda"),
    ("tonic_mean", "eda"),
    ("phasic_mean", "eda"),
    ("smna_mean", "eda"),
    ("smna_auc", "eda"),
    ("scr_rate", "eda"),
]


def _uid_sort(u: str) -> tuple[str, int]:
    run, idx = u.split("#")
    return (run, int(idx))


def plot_grid(df: pd.DataFrame, sub: str, dim: str) -> None:
    sub_df = df[(df["sub"] == sub) & (df["dim"] == dim)]
    uids = sorted(sub_df["seg_uid"].unique(), key=_uid_sort)
    if not uids:
        return
    nrows, ncols = len(METRIC_ROWS), len(uids)

    # per-row shared y-limits across stimuli
    ylims: dict[str, tuple[float, float]] = {}
    for metric, _ in METRIC_ROWS:
        v = sub_df[metric].to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            ylims[metric] = (0.0, 1.0)
            continue
        lo, hi = float(np.min(v)), float(np.max(v))
        if metric == "rep_mean":
            lo, hi = -1.05, 1.05
        elif lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        else:
            pad = 0.06 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        ylims[metric] = (lo, hi)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(12.0, 1.05 * ncols), 0.85 * nrows), squeeze=False
    )

    for i, (metric, fam) in enumerate(METRIC_ROWS):
        col = REP_COLOR if fam == "report" else EDA_COLOR
        lo, hi = ylims[metric]
        for j, uid in enumerate(uids):
            ax = axes[i][j]
            seg = sub_df[sub_df["seg_uid"] == uid].sort_values("sec")
            t = seg["t_center_s"].to_numpy()
            y = seg[metric].to_numpy()
            if metric == "scr_rate":
                tt = t[y > 0]
                ax.vlines(tt, 0, 1, color="darkgreen", lw=0.8)
                ax.set_ylim(0, 1.1)
            else:
                ax.plot(t, y, color=col, lw=0.8)
                ax.set_ylim(lo, hi)
                if metric in SIGNED:
                    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                vid = seg["video_id"].iloc[0]
                pol = seg["polarity"].iloc[0]
                vlab = "lum" if np.isnan(vid) else f"v{int(vid)}"
                run = uid.split("#")[0].replace("task-", "t").replace("_acq-", "/")
                ax.set_title(f"{vlab}\n{run}\n{pol}", fontsize=6.2)
            if j == 0:
                ax.set_ylabel(
                    metric, fontsize=7.5, color=col, rotation=0, ha="right", va="center",
                    labelpad=2,
                )

    fig.suptitle(
        f"{sub}  --  {dim}  (rows=1-Hz metric, cols=stimulus; red=report, green=EDA)",
        fontsize=11, y=0.997,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out = FIGS / f"stimgrid_{sub}_{dim}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[{sub}] {dim} -> {out.name}")


def main() -> None:
    df = pd.read_csv(TABLES / "features_1hz.csv")
    subs = sys.argv[1:] if len(sys.argv) > 1 else SUBJECTS
    for sub in subs:
        for dim in ("arousal", "valence"):
            plot_grid(df, sub, dim)


if __name__ == "__main__":
    main()
