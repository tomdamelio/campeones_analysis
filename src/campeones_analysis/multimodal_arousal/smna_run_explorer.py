"""Per-run SMNA visualization + peak-detection threshold exploration -- diary 05_02.

Loads continuous SMNA from `<sub>_continuous.npz` (already produced by
`build_y_candidates.py`), then for every (subject, run) produces a 2-panel figure:

  Top panel (70%)
    - SMNA driver (50 Hz, signed; in practice >=0 by cvxopt's L1 formulation)
    - Three threshold criteria evaluated simultaneously:
        A = 0.05 * smna.max()          (matches erp_smna.py current threshold)
        B = smna.mean() + 1.5 * smna.std()
        C = smna.mean() + 2.5 * smna.std()
    - find_peaks per threshold, distance = 1.5 s * 50 Hz = 75 samples
    - Inset (top-right ~20%): N_peaks-vs-threshold sweep curve with A/B/C marked
    - Paradigm events overlaid:
        condition == "affective"   -> blue solid + stim_id label
        condition == "luminance"   -> green solid
        condition in {"baseline","calm"} -> gray dotted

  Bottom panel (30%)
    - Y1 dmt-style: AUC = np.trapezoid(signed SMNA, t) on 2 s windows / 1 s hop,
      centered alignment. NO clip, NO smoothing -- deliberately distinct from the
      campeones Y1 in `<sub>_y_candidates.npz` (which is clipped + Gaussian sigma=3s).
    - y=0 reference line.

Output:
  research_diary/context/05_02/figures/smna_per_run/<sub>_<label>_smna_exploration.png
  research_diary/context/05_02/y_candidates/smna_peak_summary.csv

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.smna_run_explorer
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks

# --- paths (worktree-safe, mirrors build_y_candidates.py:57-67) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
MERGED_EVENTS = REPO / "data" / "derivatives" / "merged_events"
NPZ_DIR = REPO / "research_diary" / "context" / "05_02" / "y_candidates"
FIG_DIR = REPO / "research_diary" / "context" / "05_02" / "figures" / "smna_per_run"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["sub-23", "sub-24", "sub-33"]
EDA_FS = 50.0
WIN_S = 2.0
HOP_S = 1.0
PEAK_DIST_S = 1.5
PEAK_DIST_SAMP = int(round(PEAK_DIST_S * EDA_FS))  # = 75

THRESH_COLOR = {"A": "darkorange", "B": "crimson", "C": "purple"}
COND_STYLE = {
    "affective": dict(color="C0", lw=1.0, ls="-", alpha=0.55),
    "luminance": dict(color="C2", lw=0.9, ls="-", alpha=0.45),
    "baseline":  dict(color="0.5", lw=0.6, ls=":", alpha=0.55),
    "calm":      dict(color="0.5", lw=0.6, ls=":", alpha=0.55),
}


def load_events(sub: str, label: str) -> pd.DataFrame | None:
    f = MERGED_EVENTS / sub / "ses-vr" / "eeg" / f"{sub}_ses-vr_{label}_desc-merged_events.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    return None


def compute_thresholds(smna: np.ndarray) -> dict[str, float]:
    mx = float(np.nanmax(smna))
    mu = float(np.nanmean(smna))
    sd = float(np.nanstd(smna))
    return {"A": 0.05 * mx, "B": mu + 1.5 * sd, "C": mu + 2.5 * sd}


def detect_peaks(smna: np.ndarray, height: float, distance: int) -> np.ndarray:
    if not np.any(smna > 0) or height <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(smna, height=height, distance=distance)
    return peaks


def threshold_sweep(smna: np.ndarray, n: int = 50, distance: int = PEAK_DIST_SAMP) -> tuple[np.ndarray, np.ndarray]:
    mx = float(np.nanmax(smna))
    if mx <= 0:
        return np.array([0.0]), np.array([0])
    sweep = np.linspace(0.0, 0.5 * mx, n)
    counts = np.array([len(detect_peaks(smna, h, distance)) for h in sweep])
    return sweep, counts


def compute_y1_dmt(smna: np.ndarray, t: np.ndarray, win_s: float = WIN_S, hop_s: float = HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """Y1 dmt-style: AUC = trapezoid(signed SMNA, t) over centered windows.

    Returns (centers, y1). No clip, no smoothing. Mirrors the formula in
    dmt-emotions/run_eda_smna_analysis.py (line ~353) but at finer granularity
    (2 s / 1 s vs 30 s / 30 s) so the visualization has usable resolution.
    """
    T = float(t[-1])
    centers = np.arange(win_s / 2.0, T - win_s / 2.0 + 1e-9, hop_s)
    y1 = np.full(len(centers), np.nan)
    half = win_s / 2.0
    for i, c in enumerate(centers):
        m = (t >= c - half) & (t < c + half)
        if m.sum() >= 2:
            y1[i] = float(np.trapezoid(smna[m], t[m]))
    return centers, y1


def draw_events(ax: plt.Axes, events: pd.DataFrame | None, *, top_panel: bool, smna_max: float | None = None) -> None:
    if events is None or len(events) == 0:
        return
    y_top = ax.get_ylim()[1] if top_panel else None
    for _, row in events.iterrows():
        cond = str(row.get("condition", "")).strip().lower()
        style = COND_STYLE.get(cond)
        if style is None:
            continue  # skip practice / unrecognized
        ax.axvline(float(row["onset"]), zorder=0, **style)
        if top_panel and cond == "affective":
            stim_id = row.get("stim_id", "")
            try:
                lab = f"{int(stim_id):02d}"
            except (TypeError, ValueError):
                lab = str(stim_id)
            ax.text(
                float(row["onset"]),
                y_top if y_top is not None else (smna_max or 1.0),
                lab,
                rotation=90, va="top", ha="right",
                fontsize=6, color="0.25", alpha=0.9,
            )


def plot_run(
    sub: str,
    label: str,
    t: np.ndarray,
    smna: np.ndarray,
    centers_y1: np.ndarray,
    y1: np.ndarray,
    events: pd.DataFrame | None,
    qc_row: dict,
    out_png: Path,
) -> dict:
    thresh = compute_thresholds(smna)
    peaks = {k: detect_peaks(smna, thresh[k], PEAK_DIST_SAMP) for k in ("A", "B", "C")}
    sweep_x, sweep_y = threshold_sweep(smna)

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 1, height_ratios=[7, 3], hspace=0.18, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # --- top panel: SMNA + peaks + threshold lines ---
    ax_top.plot(t, smna, lw=0.6, color="0.30", zorder=2, label="SMNA driver")
    smna_max = float(np.nanmax(smna)) if smna.size else 0.0
    smna_pad = 1.10 * smna_max if smna_max > 0 else 0.01

    # horizontal threshold lines
    for k in ("A", "B", "C"):
        ax_top.axhline(thresh[k], color=THRESH_COLOR[k], lw=0.7, ls="--", alpha=0.45, zorder=1)

    # peak scatter (stacked at three vertical positions above the trace)
    y_pos = {"A": 1.02 * smna_pad, "B": 1.08 * smna_pad, "C": 1.14 * smna_pad}
    legend_handles = []
    for k in ("A", "B", "C"):
        idx = peaks[k]
        n = len(idx)
        h = ax_top.scatter(
            t[idx] if n else [],
            np.full(n, y_pos[k]) if n else [],
            marker="o", s=14, color=THRESH_COLOR[k], edgecolors="none",
            zorder=3,
            label=f"Peaks {k} (thr={thresh[k]:.4f}, N={n})",
        )
        legend_handles.append(h)

    ax_top.set_ylim(top=1.20 * smna_pad)
    ax_top.set_ylabel("SMNA driver (a.u.)")
    ax_top.set_title(
        f"{sub} | {label} | SMNA @ 50 Hz (cvxEDA, biosppy) | "
        f"Y1 dmt-style (signed, no smoothing, 2 s / 1 s) | peak threshold exploration",
        fontsize=10,
    )
    ax_top.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.85)
    draw_events(ax_top, events, top_panel=True, smna_max=smna_pad)

    # --- threshold-sweep inset (top-right of top panel) ---
    ax_inset = inset_axes(ax_top, width="20%", height="35%", loc="upper right", borderpad=1.5)
    ax_inset.plot(sweep_x, sweep_y, color="0.20", lw=1.0)
    for k in ("A", "B", "C"):
        ax_inset.axvline(thresh[k], color=THRESH_COLOR[k], lw=1.0, alpha=0.85)
    ax_inset.set_xlabel("threshold", fontsize=6)
    ax_inset.set_ylabel("N peaks", fontsize=6)
    ax_inset.tick_params(labelsize=5)
    ax_inset.set_title("N peaks vs threshold", fontsize=6, pad=2)

    # --- bottom panel: Y1 dmt-style ---
    ax_bot.axhline(0.0, color="0.55", lw=0.4, alpha=0.6, zorder=1)
    ax_bot.plot(centers_y1, y1, lw=1.0, color="C0", zorder=2,
                label="Y1 = AUC(signed SMNA) / 2 s window (dmt-style)")
    ax_bot.set_xlabel("time in run (s)")
    ax_bot.set_ylabel("SMNA AUC (a.u.·s)")
    ax_bot.legend(loc="upper left", fontsize=7, framealpha=0.85)
    draw_events(ax_bot, events, top_panel=False)

    # sync xlim across panels
    xlim = (float(t[0]), float(t[-1]))
    ax_top.set_xlim(xlim)
    ax_bot.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # metrics for CSV
    duration_s = float(t[-1])
    duration_min = duration_s / 60.0
    label_parts = dict(p.split("-", 1) for p in label.split("_"))
    return {
        "task": label_parts.get("task", ""),
        "acq": label_parts.get("acq", ""),
        "smna_max": float(np.nanmax(smna)),
        "smna_mean": float(np.nanmean(smna)),
        "smna_std": float(np.nanstd(smna)),
        "threshold_A": thresh["A"],
        "threshold_B": thresh["B"],
        "threshold_C": thresh["C"],
        "n_peaks_A": int(len(peaks["A"])),
        "n_peaks_B": int(len(peaks["B"])),
        "n_peaks_C": int(len(peaks["C"])),
        "peaks_per_min_A": len(peaks["A"]) / duration_min if duration_min > 0 else np.nan,
        "peaks_per_min_B": len(peaks["B"]) / duration_min if duration_min > 0 else np.nan,
        "peaks_per_min_C": len(peaks["C"]) / duration_min if duration_min > 0 else np.nan,
        "y1_mean": float(np.nanmean(y1)),
        "y1_std": float(np.nanstd(y1)),
        "y1_min": float(np.nanmin(y1)),
        "y1_max": float(np.nanmax(y1)),
        "run_duration_s": duration_s,
        "cvx_obj": qc_row.get("cvx_obj", np.nan),
        "cvx_res_var": qc_row.get("cvx_res_var", np.nan),
        "smna_sr_hz": EDA_FS,
        "sr_note": "departs from dmt-emotions native 250 Hz; see build_y_candidates.py:71-75",
        "y1_formula": "trapezoid(signed SMNA, t) over 2 s window, 1 s hop, centered (dmt-style, no clip, no smoothing)",
    }


def main() -> None:
    cvx_qc = pd.read_csv(NPZ_DIR / "cvx_qc.csv")
    rows = []
    print("=" * 78)
    print(f"smna_run_explorer  ::  output -> {FIG_DIR}")
    print("=" * 78)

    for sub in SUBJECTS:
        npz_path = NPZ_DIR / f"{sub}_continuous.npz"
        if not npz_path.exists():
            print(f"[{sub}] MISSING {npz_path.name} -- skipping")
            continue
        npz = np.load(npz_path, allow_pickle=True)
        run_labels = [str(r) for r in npz["runs"]]
        print(f"\n[{sub}] {len(run_labels)} runs")

        for label in run_labels:
            t_key = f"{label}__eda_t"
            s_key = f"{label}__eda_smna"
            if t_key not in npz.files or s_key not in npz.files:
                print(f"  {label}: SMNA/time missing in npz -- skipped")
                continue
            t = np.asarray(npz[t_key], dtype=float)
            smna = np.asarray(npz[s_key], dtype=float)
            if smna.size < int(EDA_FS * WIN_S):  # less than one window
                print(f"  {label}: signal too short ({smna.size} samples) -- skipped")
                continue

            centers_y1, y1 = compute_y1_dmt(smna, t)
            events = load_events(sub, label)
            qc = cvx_qc[(cvx_qc["subject"] == sub) & (cvx_qc["run"] == label)]
            qc_row = qc.iloc[0].to_dict() if len(qc) else {}

            out_png = FIG_DIR / f"{sub}_{label}_smna_exploration.png"
            metrics = plot_run(sub, label, t, smna, centers_y1, y1, events, qc_row, out_png)
            metrics_full = {"subject": sub, "run": label, **metrics}
            rows.append(metrics_full)
            print(
                f"  {label}: peaks A={metrics['n_peaks_A']:3d} "
                f"B={metrics['n_peaks_B']:3d} C={metrics['n_peaks_C']:3d}  "
                f"thr A/B/C = {metrics['threshold_A']:.4f}/{metrics['threshold_B']:.4f}/{metrics['threshold_C']:.4f}"
            )

    summary = pd.DataFrame(rows)
    out_csv = NPZ_DIR / "smna_peak_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {len(summary)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
