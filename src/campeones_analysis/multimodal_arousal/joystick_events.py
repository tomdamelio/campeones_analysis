"""Bloque 4 -- Detección de EVENTOS de cambio afectivo abrupto sobre el joystick + CONTROLES
de no-cambio ("calma"), análogo a la maquinaria SCR (erp_scr.py / epoch_matched.py).

El sujeto reporta UNA dimensión afectiva por sesión con el joystick (arousal O valencia, ver
`joystick_extract.py`). Acá detectamos, DENTRO de cada segmento de video, los momentos de cambio
brusco del reporte y los emparejamos con controles de calma en la misma sesión.

Tres definiciones de evento (todas vía velocidad de la palanca suavizada):
  (A) AROUSAL : pico de velocidad POSITIVA (subida brusca de arousal).        mode="rise"
  (B) VALENCIA: pico de |velocidad| (cambio brusco del módulo, bipolar).      mode="abs"
  (C) COMBINADO: unión a nivel de evento (A en runs arousal + B en runs valencia).

Decisiones de diseño (validadas con el usuario 2026-06-09):
  - Umbral = z ROBUSTO within-subject sobre la señal de detección:
        z = (s - mediana(s)) / (1.4826 * MAD(s)),  evento si z > K_Z.
    Robusto a que sub-27 tenga ~5x menos excursión que sub-33. K_Z se barre (2 y 3).
  - t0 del evento = CRUCE ASCENDENTE del umbral (inicio del shift, no el pico) -> la ventana
    de análisis pre-shift queda limpia del movimiento.
  - Ventana de análisis EEG = PRE-SHIFT [t0-4, t0-0.5] s (termina ANTES del movimiento; guard
    0.5 s contra jitter del onset). POST-shift [0,+3] s = contaminado por el motor (se muestra,
    NO se usa como señal). Lag sweep se aplica downstream desplazando la ventana.
  - Calma = |v| por debajo de umbral bajo en TODA la ventana [t0-4, t0+3] (análogo a
    silent_window_is_clean), misma sesión, no-solapada, emparejada en tiempo ±45 s al evento.
  - Simetría pre-shift: en pre-shift evento y calma están AMBOS quietos de movimiento.

Reusa de erp_scr.py: run_label, runs_for, EPOCH_SPAN_S, select_nonoverlapping_onsets.
Reusa el patrón de matching de epoch_matched.sample_silent_matched.

Run (self-test / conteos):
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_events
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, NPZ_DIR
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    EPOCH_SPAN_S,
    select_nonoverlapping_onsets,
)

warnings.filterwarnings("ignore")

JOY_FS = 50.0  # joystick cacheado a 50 Hz (timeline del run, t=0 = inicio del run)

# --- ventana de épocas (pre-shift) ---
ANALYSIS_TMIN, ANALYSIS_TMAX = -4.0, -0.5  # ventana de análisis EEG, relativa a t0 (pre-shift)
POST_S = 3.0                               # post-shift contaminado (se muestra, no se usa)
PRE_S = abs(ANALYSIS_TMIN)                 # 4.0 s
# Span total de la época (pre-análisis .. post-contaminado) para no-solape y chequeo de calma.
FULL_TMIN, FULL_TMAX = ANALYSIS_TMIN, POST_S  # [-4, +3] -> span 7.0
JOY_EPOCH_SPAN_S = FULL_TMAX - FULL_TMIN       # 7.0 s

# --- detección (cambio en ventana, normalizado por el rango robusto del sujeto) ---
# La velocidad INSTANTÁNEA robusto-z es numéricamente degenerada en esta señal (tramos planos/
# saturados en los rieles -> MAD(v)~0 -> z explota a ~1e8). En su lugar medimos el CAMBIO sobre
# una ventana de WIN_TAU_S (una "derivada promediada" estable) en unidades normalizadas de palanca,
# y lo estandarizamos por el RANGO ROBUSTO del sujeto (p95-p5 de los valores), que no colapsa.
SMOOTH_SIGMA_S = 0.5      # suavizado gaussiano de la palanca antes del cambio-en-ventana
WIN_TAU_S = 2.0           # ventana del cambio: Δ(t) = x(t+τ) - x(t)  (mira hacia adelante; t0 = inicio)
C_EVENT = 0.33            # evento si |Δ| >= C_EVENT * rango_robusto_sujeto (primario; robustez: 0.5)
REFRACTORY_S = 2.0        # separación mínima entre eventos (no partir una deflexión)
RANGE_PCTL = (5, 95)      # rango robusto del sujeto = p95 - p5 de los valores de la palanca

# --- calma ---
C_CALMA = 0.15            # calma: |Δ| < C_CALMA * rango en TODA la ventana (sin cambio brusco)
MATCH_WINDOW_S = 45.0     # emparejamiento temporal evento<->calma (mismo run)

# --- exclusión: runs con joystick degenerado (plano) ---
FLAT_STD_THRESH = 1e-3    # std de la señal de dimensión por debajo de esto = palanca no movida

DIMS = ("arousal", "valence")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_joystick(sub: str):
    """Carga {sub}_joystick.npz; devuelve (npz, lista_de_run_labels) o (None, None)."""
    path = NPZ_DIR / f"{sub}_joystick.npz"
    if not path.exists():
        return None, None
    npz = np.load(path, allow_pickle=True)
    runs = [str(r) for r in npz["runs"]]
    return npz, runs


def run_dimension(npz, label: str) -> str | None:
    """Devuelve 'arousal' o 'valence' según qué array tenga señal finita en el run, o None.

    Cada run es unimodal: solo una de las dos dimensiones tiene muestras finitas (no-NaN).
    Excluye runs planos (palanca no movida, std ~ 0).
    """
    best = None
    for d in DIMS:
        key = f"{label}__{d}"
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key], float)
        m = np.isfinite(arr)
        if not m.any():
            continue
        if float(np.nanstd(arr)) < FLAT_STD_THRESH:
            continue  # palanca plana -> sin eventos posibles
        best = d
    return best


# ---------------------------------------------------------------------------
# Segmentos finitos, suavizado, cambio-en-ventana, rango robusto del sujeto
# ---------------------------------------------------------------------------
def finite_segments(arr: np.ndarray) -> list[tuple[int, int]]:
    """Lista de (start, end) de tramos contiguos finitos (no-NaN) -> los segmentos de video."""
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    idx = np.flatnonzero(finite)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), ends.tolist()))


def smooth_signal(arr: np.ndarray, fs: float = JOY_FS) -> np.ndarray:
    """Suaviza la palanca (gaussiano) por SEGMENTO finito; preserva NaN fuera de segmento."""
    sm = np.full_like(arr, np.nan, dtype=float)
    sigma = SMOOTH_SIGMA_S * fs
    for i0, i1 in finite_segments(arr):
        if i1 - i0 < 5:
            continue
        sm[i0:i1] = gaussian_filter1d(arr[i0:i1], sigma=sigma, mode="nearest")
    return sm


def windowed_change(arr: np.ndarray, fs: float = JOY_FS, tau: float = WIN_TAU_S) -> np.ndarray:
    """Cambio hacia adelante Δ(t) = x_smooth(t+τ) - x_smooth(t), en unidades de palanca.

    Δ(t) mide el cambio que ARRANCA en t y se completa en t+τ -> t0 del evento = t (inicio del
    shift), de modo que la ventana pre-shift queda antes del movimiento. NaN cuando (t) o (t+τ)
    caen fuera del mismo segmento finito (no cruza gaps de NaN).
    """
    sm = smooth_signal(arr, fs)
    lag = int(round(tau * fs))
    d = np.full_like(arr, np.nan, dtype=float)
    if lag <= 0:
        return d
    fwd = np.full_like(arr, np.nan, dtype=float)
    fwd[:-lag] = sm[lag:]
    d = fwd - sm  # NaN se propaga si cualquiera de los dos es NaN (cruza segmento)
    return d


def subject_amplitude_scale(npz, runs: list[str], dim: str) -> float:
    """Rango robusto (p95 - p5) de los valores de la palanca del sujeto en esa dimensión.

    Captura 'cuánto usa el rango' este sujeto (sub-27 chico, sub-33 grande). Estable frente a
    saturación (un riel sostenido sigue teniendo dispersión entre rieles). Floor para evitar /0.
    """
    vals = []
    for label in runs:
        key = f"{label}__{dim}"
        if key in npz.files:
            a = np.asarray(npz[key], float)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)
    if not vals:
        return 1.0
    x = np.concatenate(vals)
    lo, hi = np.percentile(x, RANGE_PCTL)
    rng = float(hi - lo)
    return rng if rng > 1e-3 else 1.0


# ---------------------------------------------------------------------------
# Detección de eventos (shift onsets) y calma
# ---------------------------------------------------------------------------
def window_ptp(sm: np.ndarray, onset_idx: int, fs: float, tmin: float, tmax: float) -> float:
    """Peak-to-peak (max-min) de la señal SUAVIZADA en [t0+tmin, t0+tmax]. inf si la ventana
    se sale del array o toca un NaN (fuera de segmento finito). Es la métrica de 'planitud':
    ptp chico = sin movimiento; correcto para chequear quietud (no contaminada por el Δ-forward)."""
    lo = onset_idx + int(round(tmin * fs))
    hi = onset_idx + int(round(tmax * fs))
    if lo < 0 or hi > len(sm) or hi <= lo:
        return np.inf
    seg = sm[lo:hi]
    if seg.size == 0 or not np.all(np.isfinite(seg)):
        return np.inf
    return float(np.max(seg) - np.min(seg))


def detect_shift_onsets_s(arr: np.ndarray, fs: float = JOY_FS, *, mode: str = "rise",
                          scale: float = 1.0, c: float = C_EVENT, c_calma: float = C_CALMA,
                          tau: float = WIN_TAU_S) -> tuple[np.ndarray, np.ndarray]:
    """Detecta inicios de cambio brusco. Devuelve (onsets_s, magnitudes) con magnitud = |Δ|/scale
    (cambio estandarizado por el rango del sujeto; magnitud 1.0 = se movió un rango completo en τ).

    mode="rise": señal de detección = Δ (subidas de arousal).
    mode="fall": señal de detección = -Δ (BAJADAS de arousal; control direccional within-movement).
    mode="abs" : señal de detección = |Δ| (cambios bipolares de valencia).
    Evento si: (a) la señal de detección >= c (|Δ| >= c*scale) en el pico, Y (b) la ventana
    PRE-SHIFT [t0+ANALYSIS_TMIN, t0+ANALYSIS_TMAX] está QUIETA (ptp suavizado < c_calma*scale) ->
    simetría motor con la calma: el cambio debe estar PRECEDIDO por un período sin movimiento.
    t0 = pico del cambio-en-ventana (= inicio del shift, Δ mira hacia adelante). Ventana completa
    [t0+FULL_TMIN, t0+FULL_TMAX] dentro del mismo segmento finito. No-solape greedy.
    """
    sm = smooth_signal(arr, fs)
    d = windowed_change(arr, fs, tau)
    if mode == "rise":
        s = d           # subidas de arousal
    elif mode == "fall":
        s = -d          # BAJADAS de arousal (dirección opuesta; control within-movement)
    else:
        s = np.abs(d)   # "abs": cambios bipolares (valencia)
    sn = s / scale  # estandarizado

    onsets: list[float] = []
    mags: list[float] = []
    for i0, i1 in finite_segments(arr):
        sseg = sn[i0:i1].copy()
        sseg[~np.isfinite(sseg)] = -np.inf
        peaks, _ = find_peaks(sseg, height=c, distance=int(REFRACTORY_S * fs))
        for p in peaks:
            onset_idx = i0 + p
            lo = onset_idx + int(round(FULL_TMIN * fs))
            hi = onset_idx + int(round(FULL_TMAX * fs))
            if lo < i0 or hi > i1:
                continue  # ventana completa debe caber en el segmento
            # gate de simetría: pre-shift quieto (sin movimiento previo)
            if window_ptp(sm, onset_idx, fs, ANALYSIS_TMIN, ANALYSIS_TMAX) >= c_calma * scale:
                continue
            onsets.append(float(onset_idx / fs))
            mags.append(float(sseg[p]))
    if not onsets:
        return np.array([], float), np.array([], float)
    order = np.argsort(onsets)
    onsets = np.asarray(onsets, float)[order]
    mags = np.asarray(mags, float)[order]
    kept = select_nonoverlapping_onsets(onsets, JOY_EPOCH_SPAN_S)
    keep_mask = np.isin(onsets, kept)
    return onsets[keep_mask], mags[keep_mask]


def calma_window_is_clean(t0: float, arr: np.ndarray, fs: float = JOY_FS, *,
                          scale: float = 1.0, c: float = C_CALMA) -> bool:
    """La ventana [t0+FULL_TMIN, t0+FULL_TMAX] es 'calma': la palanca está PLANA (ptp suavizado
    < c*scale) en todo el tramo (ningún cambio brusco) Y cae dentro de un segmento finito."""
    sm = smooth_signal(arr, fs)
    onset_idx = int(round(t0 * fs))
    return window_ptp(sm, onset_idx, fs, FULL_TMIN, FULL_TMAX) < c * scale


def sample_calma_matched(onsets_s: np.ndarray, arr: np.ndarray, fs: float,
                         rng: np.random.Generator, *, scale: float = 1.0,
                         window_s: float = MATCH_WINDOW_S,
                         min_sep: float = JOY_EPOCH_SPAN_S, tries_per: int = 600) -> np.ndarray:
    """Para cada evento, un control de calma a ±window_s, no-solapado (vs eventos y vs otros
    controles). Mirror de epoch_matched.sample_silent_matched, con calma_window_is_clean."""
    reals = np.asarray(onsets_s, float)
    if reals.size == 0:
        return np.array([], float)
    dur = len(arr) / fs
    lo_b, hi_b = PRE_S, dur - POST_S
    chosen: list[float] = []
    for t in reals:
        wlo, whi = max(lo_b, t - window_s), min(hi_b, t + window_s)
        if whi <= wlo:
            continue
        for _ in range(tries_per):
            cand = float(rng.uniform(wlo, whi))
            if not calma_window_is_clean(cand, arr, fs, scale=scale):
                continue
            if float(np.min(np.abs(reals - cand))) < min_sep:
                continue
            if chosen and float(np.min(np.abs(np.asarray(chosen) - cand))) < min_sep:
                continue
            chosen.append(cand)
            break
    return np.asarray(chosen, float)


def sample_calma_uniform(onsets_s: np.ndarray, arr: np.ndarray, fs: float,
                         rng: np.random.Generator, *, scale: float = 1.0,
                         n_target: int | None = None, min_sep: float = JOY_EPOCH_SPAN_S,
                         tries_factor: int = 1000) -> np.ndarray:
    """Calma muestreada UNIFORME en el run (sin emparejar en tiempo), no-solapada vs eventos y
    vs otras calmas. Mirror de erp_scr.sample_silent_controls. Es el control SIN ajuste temporal:
    el contraste uniforme-vs-emparejado (4.4) mide cuánto del efecto es posición temporal en el run.
    """
    reals = np.asarray(onsets_s, float)
    n = len(reals) if n_target is None else int(n_target)
    if n <= 0:
        return np.array([], float)
    dur = len(arr) / fs
    lo_b, hi_b = PRE_S, dur - POST_S
    if hi_b <= lo_b:
        return np.array([], float)
    chosen: list[float] = []
    budget = tries_factor * n
    tries = 0
    while len(chosen) < n and tries < budget:
        tries += 1
        cand = float(rng.uniform(lo_b, hi_b))
        if not calma_window_is_clean(cand, arr, fs, scale=scale):
            continue
        if reals.size and float(np.min(np.abs(reals - cand))) < min_sep:
            continue
        if chosen and float(np.min(np.abs(np.asarray(chosen) - cand))) < min_sep:
            continue
        chosen.append(cand)
    return np.asarray(chosen, float)


# ---------------------------------------------------------------------------
# Self-test / conteos
# ---------------------------------------------------------------------------
def main() -> None:
    """Barrido del umbral K para calibrar: conteo de eventos, retención de calma y percentiles
    de magnitud por dimensión. El joystick es un reporte CONTINUO -> K bajo sobre-detecta; hay
    que encontrar el K donde los eventos son raros/grandes y la calma se vuelve abundante."""
    import pandas as pd

    C_GRID = [0.33, 0.5, 0.75, 1.0]
    print("=" * 78)
    print("joystick_events :: barrido de C  (cambio-en-ventana normalizado por rango del sujeto)")
    print(f"  τ={WIN_TAU_S}s  ventana pre-shift=[{ANALYSIS_TMIN},{ANALYSIS_TMAX}]s  "
          f"c_calma={C_CALMA}  match=±{MATCH_WINDOW_S}s  refractory={REFRACTORY_S}s")
    print("=" * 78)

    # unidades = (sub, label, dim, mode, arr, scale_sujeto)
    units = []
    for sub in COHORT:
        npz, runs = load_joystick(sub)
        if npz is None:
            continue
        scales = {d: subject_amplitude_scale(npz, runs, d) for d in DIMS}
        print(f"[{sub}] rango robusto (p95-p5): arousal={scales['arousal']:.2f} "
              f"valencia={scales['valence']:.2f}")
        for label in runs:
            dim = run_dimension(npz, label)
            if dim is None:
                continue
            arr = np.asarray(npz[f"{label}__{dim}"], float)
            units.append((sub, label, dim, "rise" if dim == "arousal" else "abs", arr, scales[dim]))

    summary = []
    for c in C_GRID:
        rng = np.random.default_rng(20260609)
        ev_ar = ev_va = ca = 0
        all_mags = []
        for (sub, label, dim, mode, arr, scale) in units:
            onsets, mags = detect_shift_onsets_s(arr, JOY_FS, mode=mode, scale=scale, c=c)
            calma = sample_calma_matched(onsets, arr, JOY_FS, rng, scale=scale)
            if dim == "arousal":
                ev_ar += len(onsets)
            else:
                ev_va += len(onsets)
            ca += len(calma)
            all_mags += list(mags)
        n_ev = ev_ar + ev_va
        ret = ca / max(1, n_ev)
        mp = np.percentile(all_mags, [50, 90]) if all_mags else [np.nan, np.nan]
        summary.append(dict(C=c, ev_arousal=ev_ar, ev_valence=ev_va, ev_total=n_ev,
                            calma=ca, retention=round(ret, 2),
                            mag_p50=round(float(mp[0]), 2), mag_p90=round(float(mp[1]), 2),
                            ev_per_subj=round(n_ev / len(COHORT), 1)))
        print(f"  C={c}: arousal={ev_ar} valencia={ev_va} total={n_ev} "
              f"(~{n_ev / len(COHORT):.0f}/suj)  calma={ca} ret={ret:.2f}  "
              f"mag p50={mp[0]:.2f} p90={mp[1]:.2f}")

    df = pd.DataFrame(summary)
    out = NPZ_DIR.parents[0] / "K_calibration.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
