"""Epoch-selection audit figure per subject -- one PNG showing all runs stacked.

For each subject, produces a vertically-stacked figure (1 row per run) showing:
  - EDA phasic continuous trace (background)
  - CLEAN SCR onsets (red triangles, ABOVE the trace) -- the ones used in ERP analysis
  - DROPPED SCRs (gray X, ABOVE the trace) -- detected by nk.eda_peaks but failed
    the cleanliness rule (PRE phasic >= threshold OR new SCR in POST window)
  - SILENT-EDA control timepoints (blue circles, BELOW the trace) -- matched control
    locations sampled to have phasic <= threshold across the entire epoch window

Light shading marks the [-PRE_S, +POST_S] epoch window for each marker so reviewers
can see exactly what slice of the run feeds the ERP/PSD/TFR averages.

Reproducibility: the silent-EDA control sampling uses the SAME RNG seed (20260513)
and SAME call order as `erp_scr.py`, so the silent controls shown here are exactly
the ones used in the actual ERP analysis.

Outputs:
  research_diary/context/05_02/figures/Y3_epoch_audit_<sub>.png

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.epoch_audit_scr
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reuse epoching / cleanliness machinery from erp_scr.py
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EDA_FS,
    NPZ_DIR,
    OUT,
    POST_S,
    PRE_S,
    SUBJECTS,
    detect_scr_onsets_s,
    filter_clean_onsets,
    run_label,
    runs_for,
    sample_silent_controls,
)

warnings.filterwarnings("ignore")

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fresh RNG with the same seed as erp_scr.py -- guarantees the silent controls drawn
# here match those used in the actual ERP analysis (assuming same call order).
RNG = np.random.default_rng(20260513)


def plot_subject(sub: str, out_png: Path) -> dict:
    cont_path = NPZ_DIR / f"{sub}_continuous.npz"
    if not cont_path.exists():
        print(f"  {sub}: missing {cont_path.name} -- skipping")
        return {}
    cont = np.load(cont_path, allow_pickle=True)
    run_labels = [str(r) for r in cont["runs"]]
    n_runs = len(run_labels)

    fig, axes = plt.subplots(n_runs, 1, figsize=(20, 1.5 * n_runs + 1.5))
    if n_runs == 1:
        axes = [axes]
    summary: dict = {}

    for i, label in enumerate(run_labels):
        ax = axes[i]
        t = np.asarray(cont[f"{label}__eda_t"], float)
        phasic = np.asarray(cont[f"{label}__eda_phasic"], float)

        onsets_all = detect_scr_onsets_s(phasic, EDA_FS)
        onsets_all = onsets_all[onsets_all < t[-1]]
        onsets_clean = filter_clean_onsets(onsets_all, phasic, EDA_FS)
        dropped = np.setdiff1d(onsets_all, onsets_clean)
        # silent controls: same RNG state progression as erp_scr.py main()
        silent_t = sample_silent_controls(
            n_target=len(onsets_clean), duration_s=float(t[-1]),
            phasic=phasic, fs=EDA_FS, rng=RNG, avoid_onsets_s=onsets_clean,
        )

        # background phasic
        ax.plot(t, phasic, color="C2", lw=0.5, alpha=0.7, zorder=2)
        ax.set_ylim(-0.0015, max(0.01, float(phasic.max()) * 1.20))

        # shade epoch windows lightly so reviewers see WHICH slices are taken
        y_low, y_high = ax.get_ylim()
        for t_evt in onsets_clean:
            ax.axvspan(t_evt + (-PRE_S), t_evt + POST_S, color="C3", alpha=0.04, zorder=0)
        for t_evt in silent_t:
            ax.axvspan(t_evt + (-PRE_S), t_evt + POST_S, color="C0", alpha=0.04, zorder=0)

        # marker rows: above-trace = SCRs, below-trace = silent controls
        y_clean = y_high * 0.85
        y_drop = y_high * 0.92
        y_silent = -0.0010
        if len(onsets_clean) > 0:
            ax.scatter(onsets_clean, np.full(len(onsets_clean), y_clean), marker="v",
                       color="C3", s=22, edgecolors="black", linewidths=0.3, zorder=4,
                       label=f"clean SCR (N={len(onsets_clean)})")
        if len(dropped) > 0:
            ax.scatter(dropped, np.full(len(dropped), y_drop), marker="x",
                       color="0.5", s=20, linewidths=0.8, zorder=3,
                       label=f"dropped SCR (N={len(dropped)})")
        if len(silent_t) > 0:
            ax.scatter(silent_t, np.full(len(silent_t), y_silent), marker="o",
                       color="C0", s=14, edgecolors="black", linewidths=0.3, alpha=0.85, zorder=4,
                       label=f"silent ctrl (N={len(silent_t)})")

        ax.set_title(f"{label}   (duration {t[-1]:.0f} s)", fontsize=8, loc="left")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel("phasic", fontsize=7)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=3)
        if i == n_runs - 1:
            ax.set_xlabel("time in run (s)", fontsize=8)

        summary[label] = dict(
            n_clean=int(len(onsets_clean)),
            n_dropped=int(len(dropped)),
            n_silent=int(len(silent_t)),
            duration_s=float(t[-1]),
        )

    fig.suptitle(
        f"{sub}   --   Epoch audit (which parts of the data feed the ERP/PSD/TFR analysis)\n"
        f"clean SCR onsets (red triangles, above) entered the ERP as 'real' events; "
        f"silent-EDA control points (blue circles, below) entered as the matched control. "
        f"Dropped SCRs (gray X) failed the cleanliness rule (PRE phasic baseline OR no new SCR in POST window). "
        f"Light red/blue shading shows the [-{PRE_S:.0f}, +{POST_S:.0f}] s epoch window of every marker.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return summary


def main() -> None:
    print("=" * 78)
    print("epoch_audit_scr  ::  one PNG per subject, all runs stacked")
    print("=" * 78)
    for sub in SUBJECTS:
        out_png = FIG_DIR / f"Y3_epoch_audit_{sub}.png"
        print(f"\n=== {sub} ===")
        summary = plot_subject(sub, out_png)
        for label, info in summary.items():
            print(f"  {label}: clean={info['n_clean']}  dropped={info['n_dropped']}  "
                  f"silent={info['n_silent']}  duration={info['duration_s']:.0f}s")
        print(f"  -> {out_png.name}")


if __name__ == "__main__":
    main()
