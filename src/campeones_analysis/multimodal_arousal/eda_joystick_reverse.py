"""Reverse event-based analysis: SCR -> report -- Tarea 4, Subtarea 3c.

Question: when the skin responds (a phasic SCR), what is the subject doing with the joystick?
Around each SCR onset we look at the corrected report LEVEL and the report |d/dt| (motor activity)
versus matched control moments without an SCR.

SCR onsets are detected on each run's phasic EDA (neurokit eda_peaks) and kept only when they fall
inside an arousal/valence video segment with enough margin for the [-5, +10] s window. Controls
are in-segment times >=10 s from any SCR. Report epochs are baseline-corrected to [-5, -3] s; the
summary metric is the post-window (0..+8 s) mean. AROUSAL and VALENCE are analysed SEPARATELY.

Inputs:
  y_candidates/{sub}_continuous.npz   ({label}__eda_phasic at 50 Hz)
  y_candidates/{sub}_joystick.npz     ({label}__arousal|valence at 50 Hz + {label}__seg_*)
Outputs:
  eda_joystick/tables/reverse_summary.csv      per (sub, dim): peri-SCR report level & motion diffs
  eda_joystick/figures/reverse_{dim}.png       peri-SCR report level + |d/dt| GA + per-subject

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eda_joystick_reverse
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT as SUBJECTS,
    NPZ_DIR,
    OUT,
    SUBJ_COLORS,
)
from src.campeones_analysis.multimodal_arousal.joystick_eda_features import scr_onsets_s

FS = 50.0
PRE_S, POST_S = 5.0, 10.0
PRE, POST = int(PRE_S * FS), int(POST_S * FS)
TVEC = np.arange(-PRE, POST) / FS
BASE = (TVEC >= -5.0) & (TVEC <= -3.0)
POSTWIN = (TVEC >= 0.0) & (TVEC <= 8.0)
SCR_CTRL_SEP_S = 10.0
REFRACTORY_S = 8.0
MIN_EVENTS = 5
DIMS = ("arousal", "valence")

TABLES = OUT / "eda_joystick" / "tables"
FIGS = OUT / "eda_joystick" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def pick_times(cand: np.ndarray, refractory_s: float) -> np.ndarray:
    kept: list[float] = []
    for t in cand:
        if all(abs(t - k) >= refractory_s for k in kept):
            kept.append(t)
    return np.asarray(kept, float)


def epoch_at(arr: np.ndarray, t_abs: float, baseline: bool = True) -> np.ndarray | None:
    idx = int(round(t_abs * FS))
    a, b = idx - PRE, idx + POST
    if a < 0 or b > len(arr):
        return None
    ep = arr[a:b].astype(float)
    if np.isnan(ep).any():
        return None
    return ep - np.mean(ep[BASE]) if baseline else ep


def main() -> None:
    summary: list[dict] = []
    # curves[dim]["level"/"motion"][sub] = (real_mean, ctrl_mean)
    curves: dict[str, dict[str, dict[str, tuple]]] = {
        d: {"level": {}, "motion": {}} for d in DIMS
    }

    for sub in SUBJECTS:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        joy = np.load(NPZ_DIR / f"{sub}_joystick.npz", allow_pickle=True)
        runs = [str(r) for r in joy["runs"]]
        for dim in DIMS:
            real_lvl, ctrl_lvl, real_mot, ctrl_mot = [], [], [], []
            for label in runs:
                pkey = f"{label}__eda_phasic"
                if pkey not in cont.files:
                    continue
                phasic = cont[pkey]
                report = joy[f"{label}__{dim}"]
                n = min(len(phasic), len(report))
                motion = np.abs(np.gradient(report[:n]) * FS)  # |d/dt| of report
                # this dim's segments in this run
                seg_on = joy[f"{label}__seg_onset"]
                seg_dur = joy[f"{label}__seg_dur"]
                seg_dim = [str(x) for x in joy[f"{label}__seg_dim"]]
                segs = [
                    (on, on + du)
                    for on, du, dm in zip(seg_on, seg_dur, seg_dim)
                    if dm == dim
                ]
                if not segs:
                    continue
                scr = scr_onsets_s(phasic[:n])  # seconds from run start
                # keep SCRs inside a segment with full-window margin
                def in_seg(t: float) -> bool:
                    return any(a + PRE_S <= t <= b - POST_S for a, b in segs)

                scr = np.array([t for t in scr if in_seg(t)])
                scr = pick_times(np.sort(scr), REFRACTORY_S)
                # control candidates: grid of in-segment times >=10 s from any SCR
                cand = []
                for a, b in segs:
                    cand.extend(np.arange(a + PRE_S, b - POST_S, 2.0))
                cand = np.array(cand)
                if scr.size:
                    cand = np.array(
                        [t for t in cand if np.all(np.abs(t - scr) >= SCR_CTRL_SEP_S)]
                    )
                ctrl = pick_times(cand, REFRACTORY_S)

                for t in scr:
                    el = epoch_at(report[:n], t)
                    em = epoch_at(motion, t, baseline=True)
                    if el is not None:
                        real_lvl.append(el)
                    if em is not None:
                        real_mot.append(em)
                for t in ctrl:
                    el = epoch_at(report[:n], t)
                    em = epoch_at(motion, t, baseline=True)
                    if el is not None:
                        ctrl_lvl.append(el)
                    if em is not None:
                        ctrl_mot.append(em)

            if len(real_lvl) < MIN_EVENTS or len(ctrl_lvl) < MIN_EVENTS:
                continue
            rl, cl = np.mean(real_lvl, axis=0), np.mean(ctrl_lvl, axis=0)
            rm, cm = np.mean(real_mot, axis=0), np.mean(ctrl_mot, axis=0)
            curves[dim]["level"][sub] = (rl, cl)
            curves[dim]["motion"][sub] = (rm, cm)
            summary.append(
                {
                    "sub": sub, "dim": dim,
                    "n_scr": len(real_lvl), "n_ctrl": len(ctrl_lvl),
                    "post_level_real": round(float(np.mean(rl[POSTWIN])), 5),
                    "post_level_ctrl": round(float(np.mean(cl[POSTWIN])), 5),
                    "diff_level": round(float(np.mean(rl[POSTWIN]) - np.mean(cl[POSTWIN])), 5),
                    "post_motion_real": round(float(np.mean(rm[POSTWIN])), 5),
                    "post_motion_ctrl": round(float(np.mean(cm[POSTWIN])), 5),
                    "diff_motion": round(float(np.mean(rm[POSTWIN]) - np.mean(cm[POSTWIN])), 5),
                }
            )

    df = pd.DataFrame(summary)
    df.to_csv(TABLES / "reverse_summary.csv", index=False)
    print("reverse_summary.csv (peri-SCR report, real vs control):")
    if not df.empty:
        print(df.to_string(index=False))
        for dim in DIMS:
            sd = df[df["dim"] == dim]
            if len(sd):
                print(
                    f"  [{dim}] level diff>0 in {(sd['diff_level'] > 0).sum()}/{len(sd)}; "
                    f"motion diff>0 in {(sd['diff_motion'] > 0).sum()}/{len(sd)}"
                )

    # --- figures: per dim, report level + motion peri-SCR ---
    for dim in DIMS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for ax, kind, ylab in zip(
            axes, ("level", "motion"),
            ("report level (baseline-corr)", "report |d/dt| (motion, baseline-corr)"),
        ):
            reals, ctrls = [], []
            for sub in SUBJECTS:
                if sub in curves[dim][kind]:
                    rr, cc = curves[dim][kind][sub]
                    ax.plot(TVEC, rr, color=SUBJ_COLORS[sub], lw=0.6, alpha=0.35)
                    reals.append(rr)
                    ctrls.append(cc)
            if reals:
                R, C = np.mean(reals, axis=0), np.mean(ctrls, axis=0)
                se = np.std(reals, axis=0) / np.sqrt(len(reals))
                ax.plot(TVEC, R, color="crimson", lw=2.2, label=f"peri-SCR (n={len(reals)} subj)")
                ax.fill_between(TVEC, R - se, R + se, color="crimson", alpha=0.2)
                ax.plot(TVEC, C, color="0.4", lw=2.0, ls="--", label="control")
            ax.axvline(0, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(ylab, fontsize=10)
            ax.set_xlabel("time from SCR onset (s)")
            ax.legend(fontsize=8)
        fig.suptitle(f"REVERSE  {dim}: SCR -> report  (peri-SCR)", fontsize=12)
        fig.tight_layout()
        out = FIGS / f"reverse_{dim}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"figure -> {out.name}")


if __name__ == "__main__":
    main()
