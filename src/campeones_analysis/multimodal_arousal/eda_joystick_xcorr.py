"""Cross-correlation of 1-Hz movement <-> phasic-EDA metrics -- Tarea 4, Subtarea 3a.

We do NOT correlate the raw 50 Hz signals (the joystick is step-like, intentional-motor). We use
the 1-Hz derived metrics from ``joystick_eda_features`` and ask whether joystick *movement* and
phasic skin sympathetic activity covary, and with what lead/lag.

Within each affective video segment, for each (movement, phasic) metric pair we z-score both
1-Hz series and compute the lag profile

    r(tau) = corr( movement(t), phasic(t + tau) ),   tau in [-10, +10] s

so tau>0 means the EDA *follows* the joystick movement. Per (subject, dim, pair) we aggregate the
per-segment profiles by length-weighted Fisher-z average, then take the peak (most positive) lag
and r, plus r at lag 0. Cross-subject consistency is reported as mean+-sd over subjects (N=6 ->
descriptive only, no group test).

Movement metrics:  rep_dvar (variance of the joystick 1st derivative), abs_rep_dmean (|mean d/dt|)
Phasic metrics:    smna_auc (integral of rectified SMNA), scr_rate (# SCR onsets / s)

Input:  eda_joystick/tables/features_1hz.csv
Output:
  eda_joystick/tables/xcorr_move_phasic.csv         per (sub, dim, pair) + GA rows
  eda_joystick/figures/xcorr_move_phasic_{dim}.png  GA + per-subject lag profiles, 2x2 metric grid

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eda_joystick_xcorr
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT as SUBJECTS,
    OUT,
    SUBJ_COLORS,
)

TABLES = OUT / "eda_joystick" / "tables"
FIGS = OUT / "eda_joystick" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MAX_LAG = 10  # seconds == samples at 1 Hz
LAGS = np.arange(-MAX_LAG, MAX_LAG + 1)
MIN_OVERLAP = 15  # samples (s) required at a given lag to accept a correlation
MIN_SEG = 25  # require >=25 s segments

MOVE_METRICS = ["rep_dvar", "abs_rep_dmean"]
PHASIC_METRICS = ["smna_auc", "scr_rate"]
DIMS = ("arousal", "valence")


def _z(x: np.ndarray) -> np.ndarray:
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 0 else x - np.mean(x)


def lag_profile(move: np.ndarray, phasic: np.ndarray) -> np.ndarray:
    """r(tau) = corr(move(t), phasic(t+tau)) over LAGS (1 Hz)."""
    R, E = _z(move), _z(phasic)
    n = len(R)
    out = np.full(len(LAGS), np.nan)
    for k, tau in enumerate(LAGS):
        if tau >= 0:
            r1, e1 = R[: n - tau], E[tau:]
        else:
            r1, e1 = R[-tau:], E[: n + tau]
        if len(r1) >= MIN_OVERLAP and np.std(r1) > 0 and np.std(e1) > 0:
            out[k] = np.corrcoef(r1, e1)[0, 1]
    return out


def aggregate(profiles: list[np.ndarray], weights: list[int]) -> np.ndarray:
    if not profiles:
        return np.full(len(LAGS), np.nan)
    P = np.vstack(profiles)
    w = np.asarray(weights, float)[:, None]
    z = np.arctanh(np.clip(P, -0.999, 0.999))
    num = np.nansum(z * w, axis=0)
    den = np.nansum(np.where(np.isfinite(z), w, 0.0), axis=0)
    den[den == 0] = np.nan
    return np.tanh(num / den)


def main() -> None:
    df = pd.read_csv(TABLES / "features_1hz.csv")
    df["abs_rep_dmean"] = df["rep_dmean"].abs()

    rows: list[dict] = []
    # per_prof[(dim, mv, ph)][sub] = mean lag profile
    per_prof: dict[tuple, dict[str, np.ndarray]] = {}

    for dim in DIMS:
        for mv in MOVE_METRICS:
            for ph in PHASIC_METRICS:
                per_prof[(dim, mv, ph)] = {}
                for sub in SUBJECTS:
                    seg_ids = df[(df["sub"] == sub) & (df["dim"] == dim)]["seg_uid"].unique()
                    profs, wts = [], []
                    for uid in seg_ids:
                        seg = df[(df["sub"] == sub) & (df["seg_uid"] == uid)].sort_values("sec")
                        if len(seg) < MIN_SEG:
                            continue
                        prof = lag_profile(seg[mv].to_numpy(), seg[ph].to_numpy())
                        if np.isfinite(prof).any():
                            profs.append(prof)
                            wts.append(len(seg))
                    mean_prof = aggregate(profs, wts)
                    per_prof[(dim, mv, ph)][sub] = mean_prof
                    if np.isfinite(mean_prof).any():
                        pk = int(np.nanargmax(mean_prof))
                        peak_lag, peak_r = int(LAGS[pk]), float(mean_prof[pk])
                        r0 = float(mean_prof[MAX_LAG])
                    else:
                        peak_lag, peak_r, r0 = np.nan, np.nan, np.nan
                    rows.append(
                        {
                            "sub": sub, "dim": dim, "move": mv, "phasic": ph,
                            "n_segments": len(profs),
                            "peak_lag_s": peak_lag, "peak_r": round(peak_r, 4)
                            if np.isfinite(peak_r) else np.nan,
                            "r_at_lag0": round(r0, 4) if np.isfinite(r0) else np.nan,
                        }
                    )

    df_out = pd.DataFrame(rows)
    # GA rows: mean +- sd of per-subject peaks
    ga_rows = []
    for (dim, mv, ph), _ in per_prof.items():
        sub_rows = df_out[
            (df_out.dim == dim) & (df_out.move == mv) & (df_out.phasic == ph)
        ].dropna(subset=["peak_r"])
        if len(sub_rows):
            ga_rows.append(
                {
                    "sub": "GA", "dim": dim, "move": mv, "phasic": ph,
                    "n_segments": int(sub_rows["n_segments"].sum()),
                    "peak_lag_s": round(float(sub_rows["peak_lag_s"].mean()), 2),
                    "peak_r": round(float(sub_rows["peak_r"].mean()), 4),
                    "peak_r_sd": round(float(sub_rows["peak_r"].std()), 4),
                    "r_at_lag0": round(float(sub_rows["r_at_lag0"].mean()), 4),
                    "n_sub_pos": int((sub_rows["peak_r"] > 0).sum()),
                }
            )
    df_out = pd.concat([df_out, pd.DataFrame(ga_rows)], ignore_index=True)
    csv = TABLES / "xcorr_move_phasic.csv"
    df_out.to_csv(csv, index=False)
    print(f"xcorr move<->phasic -> {csv}")
    print(df_out[df_out["sub"] == "GA"].to_string(index=False))

    # --- figures: one per dim, 2x2 grid of (move x phasic) ---
    for dim in DIMS:
        fig, axes = plt.subplots(len(MOVE_METRICS), len(PHASIC_METRICS),
                                 figsize=(11, 8), sharex=True)
        for r, mv in enumerate(MOVE_METRICS):
            for c, ph in enumerate(PHASIC_METRICS):
                ax = axes[r][c]
                stack = []
                for sub in SUBJECTS:
                    prof = per_prof[(dim, mv, ph)][sub]
                    if np.isfinite(prof).any():
                        ax.plot(LAGS, prof, color=SUBJ_COLORS[sub], lw=0.8, alpha=0.5, label=sub)
                        stack.append(prof)
                if stack:
                    ga = np.nanmean(np.vstack(stack), axis=0)
                    ax.plot(LAGS, ga, color="k", lw=2.2, label="GA")
                ax.axvline(0, color="k", lw=0.5, alpha=0.5)
                ax.axhline(0, color="k", lw=0.5, alpha=0.5)
                ax.set_title(f"{mv}  <->  {ph}", fontsize=9)
                if c == 0:
                    ax.set_ylabel("r")
                if r == len(MOVE_METRICS) - 1:
                    ax.set_xlabel("lag tau (s)  [tau>0: EDA follows movement]")
        axes[0][0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{dim}: joystick movement <-> phasic EDA (1-Hz cross-correlation)", fontsize=12)
        fig.tight_layout()
        out = FIGS / f"xcorr_move_phasic_{dim}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"figure -> {out.name}")


if __name__ == "__main__":
    main()
