"""Per-run EDA <-> joystick visual exploration -- Tarea 4 (diario 06_xx).

For every (subject, run) draws EDA phasic / SMNA / tonic (from the cached continuous EDA)
together with the polarity-corrected arousal & valence joystick reports, all on the shared
50 Hz run timeline. Video segments are shaded by dimension (arousal=red, valence=blue,
luminance=grey) using the segment table cached by ``joystick_extract``.

The joystick row is z-scored per run so the report (arbitrary units) is comparable in scale
to the EDA; the EDA rows stay in native units.

Inputs:
  research_diary/context/05_04/cohort6/y_candidates/{sub}_continuous.npz   (EDA)
  research_diary/context/05_04/cohort6/y_candidates/{sub}_joystick.npz     (joystick)
Output:
  research_diary/context/05_04/cohort6/eda_joystick/figures/{sub}_{label}.png

Run (all subjects):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_eda_viz
Run (one subject):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_eda_viz sub-19
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.campeones_analysis.multimodal_arousal.cohort import COHORT as SUBJECTS, NPZ_DIR, OUT

FIG_DIR = OUT / "eda_joystick" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIM_COLOR = {"arousal": "tab:red", "valence": "tab:blue", "luminance": "0.6"}


def _z(x: np.ndarray) -> np.ndarray:
    """Z-score a signal ignoring NaNs (returns NaNs where input is NaN)."""
    m = np.isfinite(x)
    if m.sum() < 2:
        return x
    mu, sd = np.nanmean(x[m]), np.nanstd(x[m])
    if sd == 0:
        return x - mu
    return (x - mu) / sd


def plot_run(sub: str, label: str, eda: dict, joy, out_png) -> None:
    eda_t = eda[f"{label}__eda_t"]
    phasic = eda[f"{label}__eda_phasic"]
    smna = eda[f"{label}__eda_smna"]
    tonic = eda[f"{label}__eda_tonic"]
    t = joy[f"{label}__t"]
    arousal = joy[f"{label}__arousal"]
    valence = joy[f"{label}__valence"]

    n = min(len(eda_t), len(t), len(phasic), len(smna), len(tonic), len(arousal), len(valence))
    tt = eda_t[:n]
    phasic, smna, tonic = phasic[:n], smna[:n], tonic[:n]
    arousal, valence = arousal[:n], valence[:n]

    seg_on = joy[f"{label}__seg_onset"]
    seg_dur = joy[f"{label}__seg_dur"]
    seg_dim = [str(x) for x in joy[f"{label}__seg_dim"]]

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    panels = [
        (axes[0], phasic, "EDA phasic (uS)", "tab:green"),
        (axes[1], smna, "EDA SMNA driver", "tab:olive"),
        (axes[2], tonic, "EDA tonic (uS)", "tab:brown"),
    ]
    for ax, sig, ylab, col in panels:
        ax.plot(tt, sig, color=col, lw=0.7)
        ax.set_ylabel(ylab, fontsize=9)
        ax.margins(x=0)

    ax = axes[3]
    ax.plot(tt, _z(arousal), color="tab:red", lw=0.9, label="arousal (z)")
    ax.plot(tt, _z(valence), color="tab:blue", lw=0.9, label="valence (z)")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_ylabel("report (z, +=more)", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # shade segments across all panels
    for on, du, dm in zip(seg_on, seg_dur, seg_dim):
        col = DIM_COLOR.get(dm, "0.8")
        for a in axes:
            a.axvspan(on, on + du, color=col, alpha=0.10, lw=0)

    handles = [mpatches.Patch(color=c, alpha=0.3, label=d) for d, c in DIM_COLOR.items()]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, ncol=3, title="video segment")
    axes[0].set_title(f"{sub}  {label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    subs = sys.argv[1:] if len(sys.argv) > 1 else SUBJECTS
    for sub in subs:
        cont_p = NPZ_DIR / f"{sub}_continuous.npz"
        joy_p = NPZ_DIR / f"{sub}_joystick.npz"
        if not (cont_p.exists() and joy_p.exists()):
            print(f"[{sub}] missing npz -- skipping")
            continue
        eda = np.load(cont_p, allow_pickle=True)
        joy = np.load(joy_p, allow_pickle=True)
        runs = [str(r) for r in joy["runs"]]
        for label in runs:
            out_png = FIG_DIR / f"{sub}_{label}.png"
            try:
                plot_run(sub, label, eda, joy, out_png)
                print(f"[{sub}] {label} -> {out_png.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{sub}] {label}: FAILED {exc}")


if __name__ == "__main__":
    main()
