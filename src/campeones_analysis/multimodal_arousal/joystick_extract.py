"""Extract continuous joystick arousal/valence reports per run -- Tarea 4 (diario 06_xx).

The subject continuously reports ONE affective dimension per video with a joystick
(channel ``joystick_x``). Which dimension (arousal / valence / luminance) and which
polarity (direct / inverse) each video uses lives in the per-block Order Matrix
(``data/sourcedata/xdf/sub-{S}/order_matrix_{S}_{A|B}_block{N}_VR.xlsx``, columns
``video_id``, ``dimension``, ``order_emojis_slider``). Balance across the N=6 cohort is
42 direct / 42 inverse for both arousal and valence, so polarity correction is essential.

Mechanism (mirrors scripts 01-14):
  - ``joystick_x`` (~500 Hz) lives in the SAME ``data/raw`` vhdr from which the cached EDA
    was extracted, so they share the run timeline (t=0 = run start). We resample it to 50 Hz
    (same FFT method as the EDA) and truncate to the cached EDA length.
  - run -> Order Matrix:  ``task-0N`` <-> ``blockN``,  ``acq-a/b`` <-> ``A/B``.
  - video events in merged_events (``trial_type in {video, video_luminance}``) align
    POSITIONALLY with the (dimension-non-null) rows of the matching block's Order Matrix.
  - per video segment: crop joystick_x, apply polarity correction (inverse -> negate, so
    positive = "more"), label with its dimension.

Because dimension+polarity are per video segment, ``arousal`` only exists during arousal
videos and ``valence`` only during valence videos: the per-dimension arrays are full-run
length but NaN outside their own segments.

Outputs (under research_diary/context/05_04/cohort6/):
  y_candidates/<sub>_joystick.npz   per-run: {label}__t, __joy_raw (uncorrected, 50 Hz),
                                    __arousal, __valence, __lum (corrected, NaN off-segment),
                                    and segment table __seg_onset/__seg_dur/__seg_dim/
                                    __seg_pol/__seg_vid. Plus globals ``runs``, ``eda_fs``.
  eda_joystick/tables/joystick_qc.csv  one row per (sub, run): segment counts, coverage,
                                       per-dim std/range, and a ``flag`` for any mismatch.

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_extract
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- locate repo / data (works from the main checkout or a .claude worktree) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT
XDF_DIR = REPO / "data" / "sourcedata" / "xdf"

from src.campeones_analysis.multimodal_arousal.build_y_candidates import (  # noqa: E402
    load_events,
    run_label,
    runs_for,
)
from src.campeones_analysis.multimodal_arousal.cohort import (  # noqa: E402
    COHORT as SUBJECTS,
    NPZ_DIR,
    OUT,
)

EDA_FS = 50.0
VIDEO_TYPES = ("video", "video_luminance")
DIMS = ("arousal", "valence", "luminance")

OUT_DIR = OUT / "eda_joystick"
TABLES = OUT_DIR / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_joystick_50hz(vhdr: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load ``joystick_x`` from a raw vhdr and resample to 50 Hz (EDA timeline)."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    if "joystick_x" not in raw.ch_names:
        return None, None
    sr = float(raw.info["sfreq"])
    x = raw.copy().pick(["joystick_x"]).get_data()[0].astype(float)
    joy = np.asarray(
        nk.signal_resample(x, sampling_rate=sr, desired_sampling_rate=EDA_FS, method="FFT"),
        float,
    )
    t = np.arange(len(joy)) / EDA_FS
    return t, joy


def order_matrix_for(sub: str, label: str) -> tuple[Path, int, str]:
    """Resolve the Order Matrix .xlsx path for a run from its label.

    ``sub-19`` + ``task-01_acq-a_run-002`` -> .../sub-19/order_matrix_19_A_block1_VR.xlsx
    Returns (path, task_number, acq_upper).
    """
    snum = sub.split("-")[1]
    toks = dict(p.split("-", 1) for p in label.split("_"))  # task/acq/run
    acq = toks["acq"].upper()
    task = int(toks["task"])
    path = XDF_DIR / sub / f"order_matrix_{snum}_{acq}_block{task}_VR.xlsx"
    return path, task, acq


def load_order_matrix(path: Path) -> pd.DataFrame:
    """Order Matrix rows that carry a dimension (one per presented video)."""
    return pd.read_excel(path).dropna(subset=["dimension"]).reset_index(drop=True)


def apply_polarity(sig: np.ndarray, polarity: str) -> np.ndarray:
    """Negate when polarity == 'inverse' so positive always means 'more'."""
    return -sig.copy() if str(polarity).strip().lower() == "inverse" else sig.copy()


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------
def process_run(sub: str, vhdr: Path, eda_len: int) -> tuple[dict | None, dict]:
    """Build per-dimension corrected joystick arrays + a segment table for one run."""
    label = run_label(vhdr)
    qc: dict = {"sub": sub, "run": label, "flag": ""}

    t, joy = load_joystick_50hz(vhdr)
    if joy is None:
        qc["flag"] = "no_joystick_x"
        return None, qc

    n = int(min(len(joy), eda_len))
    t, joy = t[:n], joy[:n]
    arr = {d: np.full(n, np.nan) for d in DIMS}
    seg = {k: [] for k in ("onset", "dur", "dim", "pol", "vid")}

    ev = load_events(sub, label)
    om_path, _, _ = order_matrix_for(sub, label)
    flags: list[str] = []

    if ev is None:
        flags.append("no_events")
    elif not om_path.exists():
        flags.append("no_order_matrix")
    else:
        vids = ev[ev["trial_type"].isin(VIDEO_TYPES)].reset_index(drop=True)
        om = load_order_matrix(om_path)

        # The filename block index (= task number = presentation order) is the right key;
        # the Order Matrix ``block_num`` column is a shuffled canonical id, so it is NOT
        # expected to equal ``task``. The meaningful check is positional video_id
        # concordance: the i-th affective video stim_id must equal the i-th Order Matrix
        # video_id (luminance rows carry NaN video_id and are skipped).
        if len(vids) != len(om):
            flags.append(f"count(ev={len(vids)},om={len(om)})")
        for i in range(min(len(vids), len(om))):
            sid = vids.iloc[i].get("stim_id", np.nan)
            vid = om.iloc[i].get("video_id", np.nan)
            dimi = str(om.iloc[i]["dimension"]).strip().lower()
            if dimi != "luminance" and not pd.isna(vid) and not pd.isna(sid):
                if int(sid) != int(vid):
                    flags.append(f"vid@{i}(ev{int(sid)}!=om{int(vid)})")

        for i in range(min(len(vids), len(om))):
            er, orr = vids.iloc[i], om.iloc[i]
            dim = str(orr["dimension"]).strip().lower()
            pol = str(orr.get("order_emojis_slider", "direct")).strip().lower()
            onset, dur = float(er["onset"]), float(er["duration"])
            i0 = max(0, int(round(onset * EDA_FS)))
            i1 = min(n, int(round((onset + dur) * EDA_FS)))
            if i1 <= i0:
                continue
            if dim in arr:
                arr[dim][i0:i1] = apply_polarity(joy[i0:i1], pol)
            seg["onset"].append(onset)
            seg["dur"].append(dur)
            seg["dim"].append(dim)
            seg["pol"].append(pol)
            seg["vid"].append(float(orr.get("video_id", np.nan)))

    qc["flag"] = ";".join(flags)
    qc["n_seg"] = len(seg["dim"])
    for d in DIMS:
        m = np.isfinite(arr[d])
        qc[f"n_{d}_seg"] = sum(1 for x in seg["dim"] if x == d)
        qc[f"cov_{d}_s"] = float(m.sum() / EDA_FS)
        qc[f"std_{d}"] = float(np.nanstd(arr[d])) if m.any() else np.nan
        qc[f"min_{d}"] = float(np.nanmin(arr[d])) if m.any() else np.nan
        qc[f"max_{d}"] = float(np.nanmax(arr[d])) if m.any() else np.nan
    qc["dur_s"] = float(n / EDA_FS)

    packed = {
        "t": t.astype(np.float32),
        "joy_raw": joy.astype(np.float32),
        "arousal": arr["arousal"].astype(np.float32),
        "valence": arr["valence"].astype(np.float32),
        "lum": arr["luminance"].astype(np.float32),
        "seg_onset": np.asarray(seg["onset"], np.float32),
        "seg_dur": np.asarray(seg["dur"], np.float32),
        "seg_dim": np.asarray(seg["dim"], dtype="<U12"),
        "seg_pol": np.asarray(seg["pol"], dtype="<U12"),
        "seg_vid": np.asarray(seg["vid"], np.float32),
    }
    return packed, qc


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    qc_rows: list[dict] = []
    for sub in SUBJECTS:
        cont_path = NPZ_DIR / f"{sub}_continuous.npz"
        if not cont_path.exists():
            print(f"[{sub}] no continuous EDA npz -- skipping")
            continue
        cont = np.load(cont_path, allow_pickle=True)

        npz: dict = {}
        labels_done: list[str] = []
        for vhdr in runs_for(sub):
            label = run_label(vhdr)
            eda_key = f"{label}__eda_t"
            if eda_key not in cont.files:
                print(f"[{sub}] {label}: no EDA in cache -- skipping run")
                continue
            eda_len = int(cont[eda_key].shape[0])
            packed, qc = process_run(sub, vhdr, eda_len)
            qc_rows.append(qc)
            if packed is None:
                print(f"[{sub}] {label}: {qc['flag']}")
                continue
            for k, v in packed.items():
                npz[f"{label}__{k}"] = v
            labels_done.append(label)
            print(
                f"[{sub}] {label}: n_seg={qc['n_seg']} "
                f"arousal={qc['n_arousal_seg']} valence={qc['n_valence_seg']} "
                f"lum={qc['n_luminance_seg']} flag='{qc['flag']}'"
            )

        npz["runs"] = np.asarray(labels_done, dtype="<U32")
        npz["eda_fs"] = np.float32(EDA_FS)
        out_path = NPZ_DIR / f"{sub}_joystick.npz"
        np.savez_compressed(out_path, **npz)
        print(f"[{sub}] saved {out_path.name} ({len(labels_done)} runs)")

    qc_df = pd.DataFrame(qc_rows)
    qc_csv = TABLES / "joystick_qc.csv"
    qc_df.to_csv(qc_csv, index=False)
    print(f"\nQC -> {qc_csv}  ({len(qc_df)} rows)")
    flagged = qc_df[qc_df["flag"].astype(str).str.len() > 0]
    if len(flagged):
        print(f"FLAGGED rows ({len(flagged)}):")
        print(flagged[["sub", "run", "flag"]].to_string(index=False))
    else:
        print("No flagged rows.")


if __name__ == "__main__":
    main()
