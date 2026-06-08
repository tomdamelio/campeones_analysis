"""2.3 paso 1 — QA de los canales EOG (R_EYE/L_EYE) sobre los 6 sujetos del cohort.

PRECONDICIÓN BLOQUEANTE de 2.3: antes de usar el EOG como covariado (slow_EOG/gamma_EOG)
hay que saber qué canal sirve por sujeto. Generaliza utils/diagnose_eog_channels.py (que
estaba hardcodeado a sub-27 run-002, solo PNGs) a N=6, con métricas CONTINUAS (F11) + log a
archivo + tabla.

Mapeo (hipótesis del usuario a testear):
  - H_EOG = R_EYE  = electrodo a la DERECHA del ojo derecho (cantal lateral, HEOG horizontal)
    -> tracks saccades; el canal correcto para microsacadas/gamma (2.3-gamma, Q-gamma).
    Hipótesis: FUNCIONAL en los 6.
  - L_EOG = L_EYE  = electrodo DEBAJO del ojo derecho (infraorbital, VEOG vertical)
    -> tracks blinks; alimenta slow_EOG/Q-delta. Hipótesis: puede estar MUERTO en todos
    (el canal sospechoso). Worst case: blinks se reconstruyen desde Fp1/Fp2 (frontal).

Métricas por (sujeto, sesión acq, canal):
  - std/ptp/MAD (µV): vivo vs flatline.
  - corr con Fp1/Fp2: un blink debería correlacionar fuerte con los frontales.
  - find_eog_events: tasa de eventos/min.
  - blink-locked peak (µV), trigger INDEPENDIENTE en Fp2: ¿el canal sigue los blinks?
  - ratio varianza banda-lenta(1-8)/ruido(20-40): un EOG funcional tiene más potencia lenta.
  - verdict por canal: DEAD / ALIVE-NO-BLINKS / FUNCTIONAL (+ nota HEOG vs VEOG).

Usa task-01 (acq-a y acq-b) por sujeto. RAW pre-ICA, referencia original (INMUNE a Track B).

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.eog_qa_scr
"""

from __future__ import annotations

import re
import warnings

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, REPO, keep_run

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

OUT_DIR = REPO / "research_diary" / "context" / "05_05" / "2_3_eog_qa"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_ROOT = REPO / "data" / "raw"

EOG_CHS = ["R_EYE", "L_EYE"]
FRONTAL = ["Fp1", "Fp2"]
SLOW_BAND = (1.0, 8.0)     # blink/sacada/drift
NOISE_BAND = (20.0, 40.0)  # ruido de alta freq
_LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _LOG.append(msg)


def load_raw(sub_num: str, acq: str, run: str) -> mne.io.BaseRaw | None:
    bp = BIDSPath(subject=sub_num, session="vr", task="01", acquisition=acq, run=run,
                  datatype="eeg", suffix="eeg", extension=".vhdr", root=str(RAW_ROOT))
    try:
        raw = read_raw_bids(bp, verbose="ERROR")
    except Exception as exc:
        log(f"    [load fail] {bp.basename}: {exc}")
        return None
    raw.load_data()
    raw.filter(l_freq=1.0, h_freq=40.0, picks=["eeg", "eog"], verbose="ERROR")
    return raw


def _corr(raw, ch_a, ch_b) -> float:
    if ch_a not in raw.ch_names or ch_b not in raw.ch_names:
        return float("nan")
    a = raw.get_data(picks=[ch_a])[0]
    b = raw.get_data(picks=[ch_b])[0]
    return float(np.corrcoef(a, b)[0, 1])


def _band_var(raw, ch, lo, hi) -> float:
    d = raw.copy().pick([ch]).filter(lo, hi, picks="all", verbose="ERROR").get_data()[0]
    return float(np.var(d))


def _blink_peak(raw, ch, trigger="Fp2") -> tuple[float, int]:
    """Blink-locked signed peak (µV) en [0,0.3] s, trigger independiente (Fp2)."""
    if trigger not in raw.ch_names or ch not in raw.ch_names:
        return float("nan"), 0
    try:
        events = mne.preprocessing.find_eog_events(raw, ch_name=trigger, verbose="ERROR")
    except Exception:
        return float("nan"), 0
    if events.shape[0] < 5:
        return float("nan"), int(events.shape[0])
    ep = mne.Epochs(raw, events, tmin=-0.3, tmax=0.5, baseline=(-0.3, -0.1), picks=[ch],
                    preload=True, reject_by_annotation=False, verbose="ERROR")
    ev = ep.average(picks="all").data[0] * 1e6
    t = ep.times
    m = (t >= 0.0) & (t <= 0.3)
    seg = ev[m]
    return float(seg[np.argmax(np.abs(seg))]), int(events.shape[0])


def channel_metrics(raw, ch: str, dur_min: float) -> dict:
    d = raw.get_data(picks=[ch])[0] * 1e6
    std, ptp = float(np.std(d)), float(np.ptp(d))
    mad = float(np.median(np.abs(d - np.median(d))))
    try:
        n_ev = int(mne.preprocessing.find_eog_events(raw, ch_name=ch, verbose="ERROR").shape[0])
    except Exception:
        n_ev = -1
    blink_peak, n_blink = _blink_peak(raw, ch, trigger="Fp2")
    slow, noise = _band_var(raw, ch, *SLOW_BAND), _band_var(raw, ch, *NOISE_BAND)
    return dict(
        channel=ch, std_uv=round(std, 2), ptp_uv=round(ptp, 1), mad_uv=round(mad, 2),
        corr_fp1=round(_corr(raw, ch, "Fp1"), 3), corr_fp2=round(_corr(raw, ch, "Fp2"), 3),
        n_eog_events=n_ev, rate_per_min=round(n_ev / dur_min, 1) if (n_ev >= 0 and dur_min) else np.nan,
        blink_peak_uv=round(blink_peak, 1) if np.isfinite(blink_peak) else np.nan,
        slow_noise_ratio=round(slow / noise, 2) if noise > 0 else np.nan,
    )


def channel_verdict(m: dict, other_std: float) -> str:
    # DEAD = flatline ABSOLUTO. NO se usa criterio relativo al otro ojo: el VEOG
    # (L_EYE, blinks ~300-1200 µV) es naturalmente >> que el HEOG (R_EYE, ~40-90 µV),
    # así que comparar std relativo marcaría un R_EYE sano como "muerto" (falso DEAD).
    if m["std_uv"] < 1.0:
        return "DEAD"
    tracks_blinks = (np.isfinite(m["corr_fp2"]) and abs(m["corr_fp2"]) > 0.4) or \
                    (np.isfinite(m["blink_peak_uv"]) and abs(m["blink_peak_uv"]) > 30)
    return "FUNCTIONAL" if tracks_blinks else "ALIVE-NO-BLINKS"


def runs_for_subject(sub: str) -> list[tuple[str, str]]:
    """(acq, run) de task-01 por sujeto, respetando keep_run."""
    num = sub.replace("sub-", "")
    pat = re.compile(rf"sub-{num}_ses-vr_task-01_acq-(?P<acq>[ab])_run-(?P<run>\d+)_eeg\.vhdr$")
    found = []
    eeg_dir = RAW_ROOT / sub / "ses-vr" / "eeg"
    for f in sorted(eeg_dir.glob(f"{sub}_ses-vr_task-01_acq-*_run-*_eeg.vhdr")):
        mt = pat.search(f.name)
        if not mt:
            continue
        acq, run = mt.group("acq"), mt.group("run")
        if keep_run(sub, f"task-01_acq-{acq}_run-{run}"):
            found.append((acq, run))
    return found


def main() -> None:
    log("=" * 78)
    log("eog_qa_scr :: 2.3 paso 1 — QA de R_EYE/L_EYE sobre el cohort (N=6)")
    log(f"OUT -> {OUT_DIR}")
    log("=" * 78)
    log("Hipótesis: R_EYE (HEOG lateral) FUNCIONAL 6/6 · L_EYE (VEOG infraorbital) sospechoso\n")

    rows = []
    for sub in COHORT:
        runs = runs_for_subject(sub)
        log(f"=== {sub}  (runs task-01: {runs}) ===")
        for acq, run in runs:
            raw = load_raw(sub.replace("sub-", ""), acq, run)
            if raw is None:
                continue
            dur_min = raw.times[-1] / 60.0
            present = {c: (c in raw.ch_names) for c in EOG_CHS + FRONTAL}
            mets = {ch: channel_metrics(raw, ch, dur_min) for ch in EOG_CHS if present[ch]}
            for ch in EOG_CHS:
                if ch not in mets:
                    log(f"  acq-{acq} run-{run}: {ch} AUSENTE")
                    continue
                other = "L_EYE" if ch == "R_EYE" else "R_EYE"
                other_std = mets[other]["std_uv"] if other in mets else 0.0
                v = channel_verdict(mets[ch], other_std)
                m = mets[ch]
                role = "HEOG/saccades" if ch == "R_EYE" else "VEOG/blinks"
                log(f"  acq-{acq} run-{run} {ch:6s} [{role:13s}]: std={m['std_uv']:.1f} "
                    f"ptp={m['ptp_uv']:.0f} corr_Fp2={m['corr_fp2']:+.2f} "
                    f"blink_peak={m['blink_peak_uv']} events={m['n_eog_events']} "
                    f"slow/noise={m['slow_noise_ratio']} -> {v}")
                rows.append(dict(subject=sub, acq=acq, run=run, role=role, verdict=v, **m))
        log("")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eog_qa_metrics.csv", index=False)

    # --- resumen por canal por sujeto (FUNCIONAL si lo es en alguna sesión) ---
    log("=" * 78)
    log("RESUMEN POR SUJETO (verdict por canal, sobre las sesiones disponibles)")
    log("=" * 78)
    for ch in EOG_CHS:
        log(f"\n{ch}:")
        for sub in COHORT:
            sd = df[(df.subject == sub) & (df.channel == ch)]
            if not len(sd):
                log(f"  {sub}: sin datos")
                continue
            verdicts = list(sd["verdict"])
            best = "FUNCTIONAL" if "FUNCTIONAL" in verdicts else (
                "ALIVE-NO-BLINKS" if "ALIVE-NO-BLINKS" in verdicts else "DEAD")
            log(f"  {sub}: {best:16s} (sesiones: {dict(zip(sd['acq'], sd['verdict']))})")

    # --- veredicto de cohort vs hipótesis ---
    def cohort_count(ch, status):
        return sum(
            1 for sub in COHORT
            if (lambda v: status in v if v else False)(
                list(df[(df.subject == sub) & (df.channel == ch)]["verdict"]))
        )
    r_alive = sum(1 for sub in COHORT if any(
        v != "DEAD" for v in df[(df.subject == sub) & (df.channel == "R_EYE")]["verdict"]))
    l_dead = sum(1 for sub in COHORT if len(df[(df.subject == sub) & (df.channel == "L_EYE")]) and all(
        v == "DEAD" for v in df[(df.subject == sub) & (df.channel == "L_EYE")]["verdict"]))
    l_func = sum(1 for sub in COHORT if "FUNCTIONAL" in list(
        df[(df.subject == sub) & (df.channel == "L_EYE")]["verdict"]))
    log("\n" + "=" * 78)
    log("VEREDICTO DE COHORT (vs hipótesis del usuario)")
    log("=" * 78)
    log(f"  R_EYE vivo (no-DEAD) en {r_alive}/6   [hipótesis: 6/6 funcional como HEOG]")
    log(f"  L_EYE FUNCTIONAL (tracks blinks) en {l_func}/6")
    log(f"  L_EYE DEAD en todas sus sesiones en {l_dead}/6   [hipótesis: muerto en todos]")
    log(f"\n  -> Para 2.3-gamma (microsacadas): usar R_EYE (HEOG) donde esté vivo.")
    log(f"  -> Para 2.3-lento (blinks/Q-delta): usar L_EYE donde FUNCTIONAL; "
        f"si DEAD, fallback Fp1/Fp2 (frontal).")

    with open(OUT_DIR / "eog_qa_log.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    log(f"\nLog  -> {OUT_DIR / 'eog_qa_log.txt'}")
    log(f"Tabla-> {OUT_DIR / 'eog_qa_metrics.csv'}")


if __name__ == "__main__":
    main()
