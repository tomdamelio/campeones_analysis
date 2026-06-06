"""Forward event-based analysis: report change -> EDA -- Tarea 4, Subtarea 3b.

Question: when the subject makes a large joystick move (a big net change in the reported
arousal/valence), does phasic skin sympathetic activity rise in the seconds that follow?

Events are 1-s moments of large net report change (|rep_dmean| above the subject's 90th
percentile within that dimension), with a 4 s refractory. Controls are no-movement seconds
(|rep_dmean| below the 10th percentile). Around each event/control absolute time we cut the
50 Hz EDA phasic and SMNA over [-5, +10] s, baseline-correct to the [-5, -3] s pre-window, and
average per subject. The summary metric is the post-event mean (0..+8 s). AROUSAL and VALENCE are
analysed SEPARATELY (never mixed).

Inputs:
  eda_joystick/tables/features_1hz.csv          (event detection: onset_s, t_center_s, rep_dmean)
  y_candidates/{sub}_continuous.npz             ({label}__eda_phasic|eda_smna at 50 Hz)
Outputs:
  eda_joystick/tables/forward_summary.csv       per (sub, dim): post real vs control + diff
  eda_joystick/figures/forward_{dim}.png        peri-event GA + per-subject curves (phasic, SMNA)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eda_joystick_forward
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

FS = 50.0
PRE_S, POST_S = 5.0, 10.0
PRE, POST = int(PRE_S * FS), int(POST_S * FS)
TVEC = np.arange(-PRE, POST) / FS
BASE = (TVEC >= -5.0) & (TVEC <= -3.0)
POSTWIN = (TVEC >= 0.0) & (TVEC <= 8.0)
REFRACTORY_S = 8.0
EV_CTRL_SEP_S = 10.0  # controls must be >=10 s from any event (no peri-event contamination)
P_EVENT, P_CTRL = 95.0, 40.0
MIN_EVENTS = 5
DIMS = ("arousal", "valence")
SIGNALS = ("eda_phasic", "eda_smna")

TABLES = OUT / "eda_joystick" / "tables"
FIGS = OUT / "eda_joystick" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def pick_times(cand: np.ndarray, refractory_s: float) -> np.ndarray:
    """Greedy refractory selection of candidate absolute times (already priority-sorted)."""
    kept: list[float] = []
    for t in cand:
        if all(abs(t - k) >= refractory_s for k in kept):
            kept.append(t)
    return np.asarray(kept, float)


def extract(arr: np.ndarray, t_abs: float) -> np.ndarray | None:
    idx = int(round(t_abs * FS))
    a, b = idx - PRE, idx + POST
    if a < 0 or b > len(arr):
        return None
    ep = arr[a:b].astype(float)
    return ep - np.mean(ep[BASE])


def main() -> None:
    feats = pd.read_csv(TABLES / "features_1hz.csv")
    feats["abs_dmean"] = feats["rep_dmean"].abs()
    feats["t_abs"] = feats["onset_s"] + feats["t_center_s"]

    summary: list[dict] = []
    # curves[dim][sig][sub] = (real_mean, ctrl_mean)
    curves: dict[str, dict[str, dict[str, tuple]]] = {
        d: {s: {} for s in SIGNALS} for d in DIMS
    }

    for sub in SUBJECTS:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        for dim in DIMS:
            fd = feats[(feats["sub"] == sub) & (feats["dim"] == dim)]
            if fd.empty:
                continue
            thr_ev = np.percentile(fd["abs_dmean"], P_EVENT)
            thr_ct = np.percentile(fd["abs_dmean"], P_CTRL)
            if thr_ev <= 0:
                pos = fd["abs_dmean"][fd["abs_dmean"] > 0]
                thr_ev = pos.min() if len(pos) else np.inf

            # select events/controls PER RUN (refractory is only meaningful within a recording)
            sel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for label, g in fd.groupby("run"):
                ev_cand = g[g["abs_dmean"] >= thr_ev].sort_values("abs_dmean", ascending=False)
                ct_cand = g[g["abs_dmean"] <= thr_ct].sort_values("abs_dmean", ascending=True)
                ev_t = pick_times(ev_cand["t_abs"].to_numpy(), REFRACTORY_S)
                ct_t = pick_times(ct_cand["t_abs"].to_numpy(), REFRACTORY_S)
                ct_t = np.array(
                    [t for t in ct_t if ev_t.size == 0 or np.all(np.abs(t - ev_t) >= EV_CTRL_SEP_S)]
                )
                sel[str(label)] = (ev_t, ct_t)

            for sig in SIGNALS:
                real_eps, ctrl_eps = [], []
                for label, (ev_t, ct_t) in sel.items():
                    key = f"{label}__{sig}"
                    if key not in cont.files:
                        continue
                    arr = cont[key]
                    for t in ev_t:
                        ep = extract(arr, t)
                        if ep is not None:
                            real_eps.append(ep)
                    for t in ct_t:
                        ep = extract(arr, t)
                        if ep is not None:
                            ctrl_eps.append(ep)
                if len(real_eps) < MIN_EVENTS or len(ctrl_eps) < MIN_EVENTS:
                    continue
                real_mean = np.mean(real_eps, axis=0)
                ctrl_mean = np.mean(ctrl_eps, axis=0)
                curves[dim][sig][sub] = (real_mean, ctrl_mean)
                if sig == "eda_phasic":
                    summary.append(
                        {
                            "sub": sub, "dim": dim,
                            "n_events": len(real_eps), "n_ctrl": len(ctrl_eps),
                            "post_phasic_real": round(float(np.mean(real_mean[POSTWIN])), 6),
                            "post_phasic_ctrl": round(float(np.mean(ctrl_mean[POSTWIN])), 6),
                            "diff_phasic": round(
                                float(np.mean(real_mean[POSTWIN]) - np.mean(ctrl_mean[POSTWIN])), 6
                            ),
                        }
                    )

    df = pd.DataFrame(summary)
    df.to_csv(TABLES / "forward_summary.csv", index=False)
    print("forward_summary.csv (phasic post 0..8 s, real vs control):")
    if not df.empty:
        print(df.to_string(index=False))
        for dim in DIMS:
            sub_d = df[df["dim"] == dim]
            if len(sub_d):
                npos = int((sub_d["diff_phasic"] > 0).sum())
                print(f"  [{dim}] diff>0 in {npos}/{len(sub_d)} subjects, "
                      f"GA diff={sub_d['diff_phasic'].mean():.6f}")

    # --- figures: per dimension, phasic + SMNA peri-event GA ---
    for dim in DIMS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for ax, sig in zip(axes, SIGNALS):
            reals, ctrls = [], []
            for sub in SUBJECTS:
                if sub in curves[dim][sig]:
                    rm, cm = curves[dim][sig][sub]
                    ax.plot(TVEC, rm, color=SUBJ_COLORS[sub], lw=0.6, alpha=0.35)
                    reals.append(rm)
                    ctrls.append(cm)
            if reals:
                R = np.mean(reals, axis=0)
                C = np.mean(ctrls, axis=0)
                se = np.std(reals, axis=0) / np.sqrt(len(reals))
                ax.plot(TVEC, R, color="crimson", lw=2.2, label=f"event (n={len(reals)} subj)")
                ax.fill_between(TVEC, R - se, R + se, color="crimson", alpha=0.2)
                ax.plot(TVEC, C, color="0.4", lw=2.0, ls="--", label="control")
            ax.axvline(0, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(f"{sig} (baseline-corrected)", fontsize=10)
            ax.set_xlabel("time from report-change event (s)")
            ax.set_ylabel("uS (event - baseline)")
            ax.legend(fontsize=8)
        fig.suptitle(f"FORWARD  {dim}: large report change -> EDA  (peri-event)", fontsize=12)
        fig.tight_layout()
        out = FIGS / f"forward_{dim}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"figure -> {out.name}")


if __name__ == "__main__":
    main()
