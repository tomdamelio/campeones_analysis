"""Bloque 4.0 -- Audit visual de eventos/controles del joystick, análogo a epoch_audit_scr.py.

Una figura por sujeto (1 fila por run afectivo) mostrando:
  - traza continua del joystick de la dimensión del run (arousal o valencia), por segmento;
  - EVENTOS de cambio brusco (▲ rojos, ARRIBA de la traza) -- t0 = inicio del shift;
  - CONTROLES de calma (● azules, ABAJO) -- emparejados ±45 s, no-solapados;
  - sombreado VERDE = ventana de análisis PRE-SHIFT [t0-4, t0-0.5] s (lo que alimenta el EEG);
  - sombreado ROJO = ventana POST-SHIFT [0, +3] s (contaminada por el movimiento, NO se usa).

Permite ver exactamente qué tramos de la palanca alimentan el análisis y comprobar que la
ventana pre-shift cae ANTES del movimiento (limpia del confound motor), tanto en eventos como
en calma (simetría pre-shift).

Reproducibilidad: misma semilla y orden de llamadas que joystick_events.main() -> los controles
mostrados son los que entrarían al análisis real.

Outputs:
  research_diary/context/05_05/4_0_event_audit/figures/B4_event_audit_<sub>.png
  research_diary/context/05_05/4_0_event_audit/event_audit_summary.csv

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.joystick_epoch_audit
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.cohort import COHORT, OUT
from src.campeones_analysis.multimodal_arousal.joystick_events import (
    ANALYSIS_TMAX,
    ANALYSIS_TMIN,
    C_EVENT,
    DIMS,
    JOY_FS,
    POST_S,
    detect_shift_onsets_s,
    finite_segments,
    load_joystick,
    run_dimension,
    sample_calma_matched,
    subject_amplitude_scale,
)

warnings.filterwarnings("ignore")

CTX_05 = OUT.parents[1] / "05_05"            # research_diary/context/05_05
OUT_DIR = CTX_05 / "4_0_event_audit"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260609


def plot_subject(sub: str, out_png) -> list[dict]:
    npz, runs = load_joystick(sub)
    if npz is None:
        print(f"  {sub}: sin joystick npz -- skip")
        return []
    rng = np.random.default_rng(RNG_SEED)
    scales = {d: subject_amplitude_scale(npz, runs, d) for d in DIMS}

    # solo runs afectivos (con dimensión válida, no planos)
    run_dims = [(lab, run_dimension(npz, lab)) for lab in runs]
    run_dims = [(lab, d) for lab, d in run_dims if d is not None]
    n_runs = len(run_dims)
    if n_runs == 0:
        print(f"  {sub}: sin runs afectivos válidos")
        return []

    fig, axes = plt.subplots(n_runs, 1, figsize=(20, 1.7 * n_runs + 1.5))
    if n_runs == 1:
        axes = [axes]
    rows: list[dict] = []

    for ax, (label, dim) in zip(axes, run_dims):
        arr = np.asarray(npz[f"{label}__{dim}"], float)
        t = np.arange(len(arr)) / JOY_FS
        mode = "rise" if dim == "arousal" else "abs"
        scale = scales[dim]
        onsets, mags = detect_shift_onsets_s(arr, JOY_FS, mode=mode, scale=scale, c=C_EVENT)
        calma = sample_calma_matched(onsets, arr, JOY_FS, rng, scale=scale)

        # traza por segmento finito (gaps de NaN quedan en blanco)
        for i0, i1 in finite_segments(arr):
            ax.plot(t[i0:i1], arr[i0:i1], color="0.35", lw=0.6, zorder=2)
        ax.axhline(0, color="0.7", lw=0.4, zorder=1)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlim(0, t[-1] if len(t) else 1)

        # ventanas: pre-shift (verde) y post-shift contaminado (rojo) por evento
        for t0 in onsets:
            ax.axvspan(t0 + ANALYSIS_TMIN, t0 + ANALYSIS_TMAX, color="C2", alpha=0.18, zorder=0)
            ax.axvspan(t0, t0 + POST_S, color="C3", alpha=0.10, zorder=0)
        # ventana pre-shift de la calma (verde claro)
        for tc in calma:
            ax.axvspan(tc + ANALYSIS_TMIN, tc + ANALYSIS_TMAX, color="C0", alpha=0.12, zorder=0)

        # marcadores: eventos arriba, calma abajo
        if len(onsets):
            ax.scatter(onsets, np.full(len(onsets), 1.05), marker="v", color="C3", s=24,
                       edgecolors="black", linewidths=0.3, zorder=4,
                       label=f"evento (N={len(onsets)})")
        if len(calma):
            ax.scatter(calma, np.full(len(calma), -1.05), marker="o", color="C0", s=16,
                       edgecolors="black", linewidths=0.3, alpha=0.85, zorder=4,
                       label=f"calma (N={len(calma)})")

        ax.set_title(f"{label}  [{dim}]  (dur {t[-1]:.0f} s)", fontsize=8, loc="left")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel(dim[:4], fontsize=7)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.85, ncol=2)

        rows.append(dict(sub=sub, run=label, dim=dim, n_events=len(onsets), n_calma=len(calma),
                         mag_median=float(np.median(mags)) if len(mags) else np.nan))

    axes[-1].set_xlabel("tiempo en el run (s)", fontsize=8)
    fig.suptitle(
        f"{sub}  --  Audit de eventos/calma del joystick (Bloque 4.0)\n"
        f"▲ rojo (arriba) = inicio de cambio brusco (evento); ● azul (abajo) = calma emparejada ±45 s. "
        f"Verde = ventana de análisis PRE-SHIFT [{ANALYSIS_TMIN:.1f},{ANALYSIS_TMAX:.1f}]s (limpia de movimiento); "
        f"rojo = POST-SHIFT [0,{POST_S:.0f}]s (contaminado por el motor, NO se usa). "
        f"Evento = cambio >= {C_EVENT}x rango del sujeto en {2.0:.0f}s.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return rows


def main() -> None:
    print("=" * 78)
    print("joystick_epoch_audit :: una PNG por sujeto (runs apilados)")
    print("=" * 78)
    all_rows: list[dict] = []
    for sub in COHORT:
        print(f"\n=== {sub} ===")
        out_png = FIG_DIR / f"B4_event_audit_{sub}.png"
        rows = plot_subject(sub, out_png)
        for r in rows:
            print(f"  {r['run']} [{r['dim']}]: eventos={r['n_events']} calma={r['n_calma']}")
        if rows:
            print(f"  -> {out_png.name}")
        all_rows += rows
    df = pd.DataFrame(all_rows)
    csv = OUT_DIR / "event_audit_summary.csv"
    df.to_csv(csv, index=False)
    print(f"\nResumen -> {csv}")
    if len(df):
        print("\nEventos por dimensión:")
        print(df.groupby("dim")[["n_events", "n_calma"]].sum().to_string())


if __name__ == "__main__":
    main()
