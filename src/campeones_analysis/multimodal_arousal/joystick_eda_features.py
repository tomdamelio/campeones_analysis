"""1-Hz EDA & report features per affective video segment -- Tarea 4, Subtarea 3 prep.

Rather than correlating the raw 50 Hz signals point-to-point (the joystick is a step-like,
intentional-motor signal), we aggregate both modalities into derived metrics every 1 s within
each affective video segment, then (downstream) cross-correlate those 1-Hz series.

Per 1-s window (1 s long, 1 s hop -> one value per second) within each arousal/valence video:
  EDA      : eda_mean (uS), phasic_mean, smna_mean, smna_auc (integral of rectified SMNA),
             tonic_mean, scr_rate (# SCR onsets in the window)
  report   : rep_mean, rep_var, rep_dmean (mean 1st derivative), rep_dvar (variance of 1st
             derivative -- how much the joystick is being moved)

Inputs (per subject):
  y_candidates/{sub}_continuous.npz   {label}__eda_raw|eda_phasic|eda_smna|eda_tonic|eda_t
  y_candidates/{sub}_joystick.npz     {label}__arousal|valence + {label}__seg_*
Output:
  eda_joystick/tables/features_1hz.csv   long format, one row per (sub, dim, segment, second)

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_eda_features
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.campeones_analysis.multimodal_arousal.cohort import (
    COHORT as SUBJECTS,
    NPZ_DIR,
    OUT,
)

EDA_FS = 50.0
WIN_S = 1.0
HOP_S = 1.0
DIMS = ("arousal", "valence")
TRAPZ = getattr(np, "trapezoid", np.trapz)

TABLES = OUT / "eda_joystick" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def scr_onsets_s(phasic: np.ndarray) -> np.ndarray:
    """SCR onset times (s, relative to the segment start) via neurokit eda_peaks."""
    try:
        _, info = nk.eda_peaks(phasic, sampling_rate=EDA_FS, method="neurokit")
    except Exception:
        return np.array([], dtype=float)
    on = np.asarray(info.get("SCR_Onsets", []), dtype=float)
    on = on[np.isfinite(on)]
    return np.sort(on / EDA_FS)


def seg_features(
    eda_raw: np.ndarray,
    phasic: np.ndarray,
    smna: np.ndarray,
    tonic: np.ndarray,
    report: np.ndarray,
) -> list[dict]:
    """1-Hz feature rows for one segment (all inputs same length, 50 Hz)."""
    n = len(report)
    win = int(WIN_S * EDA_FS)
    hop = int(HOP_S * EDA_FS)
    deriv = np.gradient(report) * EDA_FS  # report units / s
    onsets = scr_onsets_s(phasic)
    rows: list[dict] = []
    start = 0
    while start + win <= n:
        sl = slice(start, start + win)
        t_local = np.arange(start, start + win) / EDA_FS
        smna_rect = np.clip(smna[sl], 0, None)
        scr_rate = float(
            np.sum((onsets >= start / EDA_FS) & (onsets < (start + win) / EDA_FS))
        )
        rows.append(
            {
                "t_center_s": (start + win / 2) / EDA_FS,
                "eda_mean": float(np.mean(eda_raw[sl])),
                "phasic_mean": float(np.mean(phasic[sl])),
                "smna_mean": float(np.mean(smna[sl])),
                "smna_auc": float(TRAPZ(smna_rect, t_local)),
                "tonic_mean": float(np.mean(tonic[sl])),
                "scr_rate": scr_rate,
                "rep_mean": float(np.mean(report[sl])),
                "rep_var": float(np.var(report[sl])),
                "rep_dmean": float(np.mean(deriv[sl])),
                "rep_dvar": float(np.var(deriv[sl])),
            }
        )
        start += hop
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for sub in SUBJECTS:
        cont = np.load(NPZ_DIR / f"{sub}_continuous.npz", allow_pickle=True)
        joy = np.load(NPZ_DIR / f"{sub}_joystick.npz", allow_pickle=True)
        runs = [str(r) for r in joy["runs"]]
        n_seg = 0
        for label in runs:
            if f"{label}__eda_raw" not in cont.files:
                continue
            eda_raw = cont[f"{label}__eda_raw"]
            phasic = cont[f"{label}__eda_phasic"]
            smna = cont[f"{label}__eda_smna"]
            tonic = cont[f"{label}__eda_tonic"]
            rep = {d: joy[f"{label}__{d}"] for d in DIMS}
            seg_on = joy[f"{label}__seg_onset"]
            seg_dur = joy[f"{label}__seg_dur"]
            seg_dim = [str(x) for x in joy[f"{label}__seg_dim"]]
            seg_pol = [str(x) for x in joy[f"{label}__seg_pol"]]
            seg_vid = joy[f"{label}__seg_vid"]
            nmin = min(len(eda_raw), len(rep["arousal"]))
            for si, (on, du, dm, pol, vid) in enumerate(
                zip(seg_on, seg_dur, seg_dim, seg_pol, seg_vid)
            ):
                if dm not in DIMS:
                    continue
                i0 = max(0, int(round(on * EDA_FS)))
                i1 = min(nmin, int(round((on + du) * EDA_FS)))
                r = rep[dm][i0:i1]
                m = np.isfinite(r)
                if m.sum() < int(WIN_S * EDA_FS):
                    continue
                # use the finite span (segments are contiguous, so m is a solid block)
                lo = int(np.argmax(m))
                hi = i1 - i0 - int(np.argmax(m[::-1]))
                a, b = i0 + lo, i0 + hi
                rows = seg_features(
                    eda_raw[a:b], phasic[a:b], smna[a:b], tonic[a:b], rep[dm][a:b]
                )
                for k, row in enumerate(rows):
                    row.update(
                        sub=sub,
                        dim=dm,
                        run=label,
                        seg_idx=si,
                        seg_uid=f"{label}#{si}",
                        video_id=float(vid),
                        polarity=pol,
                        onset_s=float(on),
                        sec=k,
                    )
                    all_rows.append(row)
                n_seg += 1
        print(f"[{sub}] {n_seg} affective segments featurized")

    df = pd.DataFrame(all_rows)
    cols = [
        "sub", "dim", "run", "seg_idx", "seg_uid", "video_id", "polarity",
        "onset_s", "sec", "t_center_s",
        "eda_mean", "phasic_mean", "smna_mean", "smna_auc", "tonic_mean", "scr_rate",
        "rep_mean", "rep_var", "rep_dmean", "rep_dvar",
    ]
    df = df[cols]
    out = TABLES / "features_1hz.csv"
    df.to_csv(out, index=False)
    print(f"\nfeatures -> {out}  ({len(df)} rows, {df['seg_uid'].nunique()} segments)")
    print(df.groupby(["sub", "dim"])["seg_uid"].nunique().to_string())


if __name__ == "__main__":
    main()
