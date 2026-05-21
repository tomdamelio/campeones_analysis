"""EDA Quality-Assurance visualization, NeuroKit-style, per (subject, run) -- diary 05_02.

For every run of every subject in the active cohort, produces a 4-panel figure
mimicking `nk.eda_plot` but using the cvxEDA decomposition already stored in
`<sub>_continuous.npz`:

  Panel 1  EDA raw (50 Hz, post-FFT-resample) + EDA clean (nk.eda_clean, computed
           on the fly for visualization only -- NOT used downstream)
  Panel 2  Phasic component (EDR from cvxEDA) with SCR onsets and peaks marked
           via nk.eda_peaks(method="neurokit") on the phasic signal
  Panel 3  Tonic component (EDL from cvxEDA)
  Panel 4  SMNA driver (sparse impulses from cvxEDA)

Paradigm events are overlaid on every panel using the same convention as
`smna_run_explorer.py` (affective blue / luminance green / baseline-calm gray).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eda_qa_explorer
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- paths (worktree-safe, mirrors smna_run_explorer.py) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
MERGED_EVENTS = REPO / "data" / "derivatives" / "merged_events"
NPZ_DIR = REPO / "research_diary" / "context" / "05_02" / "y_candidates"
FIG_DIR = REPO / "research_diary" / "context" / "05_02" / "figures" / "eda_qa_per_run"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["sub-23", "sub-24", "sub-33"]
EDA_FS = 50.0

COND_STYLE = {
    "affective": dict(color="C0", lw=1.0, ls="-", alpha=0.45),
    "luminance": dict(color="C2", lw=0.9, ls="-", alpha=0.40),
    "baseline":  dict(color="0.5", lw=0.6, ls=":", alpha=0.50),
    "calm":      dict(color="0.5", lw=0.6, ls=":", alpha=0.50),
}


def load_events(sub: str, label: str) -> pd.DataFrame | None:
    f = MERGED_EVENTS / sub / "ses-vr" / "eeg" / f"{sub}_ses-vr_{label}_desc-merged_events.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    return None


def draw_events(ax: plt.Axes, events: pd.DataFrame | None, *, label_top: bool = False) -> None:
    if events is None or len(events) == 0:
        return
    y_top = ax.get_ylim()[1]
    for _, row in events.iterrows():
        cond = str(row.get("condition", "")).strip().lower()
        style = COND_STYLE.get(cond)
        if style is None:
            continue
        ax.axvline(float(row["onset"]), zorder=0, **style)
        if label_top and cond == "affective":
            stim_id = row.get("stim_id", "")
            try:
                lab = f"{int(stim_id):02d}"
            except (TypeError, ValueError):
                lab = str(stim_id)
            ax.text(
                float(row["onset"]), y_top, lab,
                rotation=90, va="top", ha="right",
                fontsize=6, color="0.25", alpha=0.9,
            )


def detect_scrs(eda_phasic: np.ndarray, sampling_rate: float = EDA_FS) -> dict:
    """Run nk.eda_peaks on the phasic signal. Returns dict with onset/peak/amplitude arrays.

    Returns empty arrays if detection fails (e.g. very sparse / flat phasic).
    """
    try:
        signals, info = nk.eda_peaks(eda_phasic, sampling_rate=sampling_rate, method="neurokit")
        return {
            "onsets": np.asarray(info.get("SCR_Onsets", []), dtype=int),
            "peaks": np.asarray(info.get("SCR_Peaks", []), dtype=int),
            "amplitudes": np.asarray(info.get("SCR_Amplitude", []), dtype=float),
        }
    except Exception:
        return {"onsets": np.array([], dtype=int), "peaks": np.array([], dtype=int), "amplitudes": np.array([], dtype=float)}


def plot_run(
    sub: str,
    label: str,
    t: np.ndarray,
    eda_raw: np.ndarray,
    eda_phasic: np.ndarray,
    eda_tonic: np.ndarray,
    eda_smna: np.ndarray,
    events: pd.DataFrame | None,
    out_png: Path,
) -> dict:
    # compute clean on the fly (cosmetic, viz-only)
    try:
        eda_clean = np.asarray(nk.eda_clean(eda_raw, sampling_rate=EDA_FS), dtype=float)
    except Exception:
        eda_clean = eda_raw.copy()

    scrs = detect_scrs(eda_phasic, sampling_rate=EDA_FS)

    fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(
        f"{sub} | {label} | EDA QA (NeuroKit-style) | cvxEDA @ 50 Hz | "
        f"n_SCR={len(scrs['peaks'])}",
        fontsize=11,
    )

    # --- Panel 1: raw + clean ---
    ax0 = axes[0]
    ax0.plot(t, eda_raw, lw=0.6, color="0.55", label="EDA raw (50 Hz, post-resample)")
    ax0.plot(t, eda_clean, lw=0.9, color="C3", label="EDA clean (nk.eda_clean)")
    ax0.set_ylabel("EDA (a.u.)")
    ax0.legend(loc="upper right", fontsize=7, framealpha=0.85)
    draw_events(ax0, events, label_top=True)

    # --- Panel 2: phasic + SCR onsets/peaks ---
    ax1 = axes[1]
    ax1.plot(t, eda_phasic, lw=0.8, color="C2", label="EDA phasic = EDR (cvxEDA)")
    if len(scrs["peaks"]) > 0:
        peaks_idx = scrs["peaks"][(scrs["peaks"] >= 0) & (scrs["peaks"] < len(t))]
        ax1.plot(t[peaks_idx], eda_phasic[peaks_idx], "v", ms=5, color="C3",
                 mec="black", mew=0.4, label=f"SCR peaks (N={len(peaks_idx)})")
    if len(scrs["onsets"]) > 0:
        ons_idx = scrs["onsets"][(scrs["onsets"] >= 0) & (scrs["onsets"] < len(t))]
        ax1.plot(t[ons_idx], eda_phasic[ons_idx], "o", ms=3, color="C0",
                 mec="black", mew=0.3, alpha=0.7, label=f"SCR onsets (N={len(ons_idx)})")
    ax1.set_ylabel("EDA phasic (a.u.)")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.85)
    draw_events(ax1, events)

    # --- Panel 3: tonic ---
    ax2 = axes[2]
    ax2.plot(t, eda_tonic, lw=1.0, color="C1", label="EDA tonic = EDL (cvxEDA)")
    ax2.set_ylabel("EDA tonic (a.u.)")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.85)
    draw_events(ax2, events)

    # --- Panel 4: SMNA driver ---
    ax3 = axes[3]
    ax3.plot(t, eda_smna, lw=0.6, color="0.20", label="SMNA driver (sparse, cvxEDA)")
    ax3.set_ylabel("SMNA driver (a.u.)")
    ax3.set_xlabel("time in run (s)")
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.85)
    draw_events(ax3, events)

    ax0.set_xlim(float(t[0]), float(t[-1]))

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return {"n_scr": int(len(scrs["peaks"]))}


def main() -> None:
    print("=" * 78)
    print(f"eda_qa_explorer  ::  output -> {FIG_DIR}")
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
            needed_keys = [f"{label}__eda_t", f"{label}__eda_raw", f"{label}__eda_phasic",
                           f"{label}__eda_tonic", f"{label}__eda_smna"]
            if not all(k in npz.files for k in needed_keys):
                print(f"  {label}: missing EDA arrays -- skipped")
                continue

            t = np.asarray(npz[f"{label}__eda_t"], dtype=float)
            eda_raw = np.asarray(npz[f"{label}__eda_raw"], dtype=float)
            eda_phasic = np.asarray(npz[f"{label}__eda_phasic"], dtype=float)
            eda_tonic = np.asarray(npz[f"{label}__eda_tonic"], dtype=float)
            eda_smna = np.asarray(npz[f"{label}__eda_smna"], dtype=float)

            events = load_events(sub, label)
            out_png = FIG_DIR / f"{sub}_{label}_eda_qa.png"
            metrics = plot_run(sub, label, t, eda_raw, eda_phasic, eda_tonic, eda_smna, events, out_png)
            print(f"  {label}: n_SCR = {metrics['n_scr']}")

    print(f"\nDone. PNGs -> {FIG_DIR}")


if __name__ == "__main__":
    main()
