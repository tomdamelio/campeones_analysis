"""Single source of truth for the multimodal_arousal analysis cohort (2026-05-27).

Extends the original 3 Tier-1 subjects (23, 24, 33) to the 8-subject cohort after
reprocessing 19/27/28/30/31 from scratch. Every analysis script in this package should
import COHORT / OUT / NPZ_DIR / SUBJ_COLORS / keep_run from here instead of hardcoding
``SUBJECTS = ["sub-23", "sub-24", "sub-33"]``.

QC exclusions (from the 2026-05-27 physio QA, reports/physio_qa/physio_qa_summary.tsv):
  - sub-28 acq-b: EDA quality bad (SCL ~0.01, spurious SCR counts). Since the whole suite
    is SCR/EDA-locked, those runs are dropped from EVERY analysis here (not just the
    EDA-target build). Decision confirmed by the user.
  - Aborted takes sub-19 task-04 acq-a run-005 (45 s) and sub-27 task-04 acq-a run-005
    (~27 s) were never preprocessed (excluded at Paso 4), so they are absent from
    derivatives and need no run-level filter here; listed for traceability.

Outputs go to research_diary/context/05_04/ to keep the N=3 results in 05_02 untouched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# --- locate repo (works from the main checkout or a .claude worktree) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]  # <root>/src/campeones_analysis/multimodal_arousal/cohort.py
if "worktrees" in _ROOT.parts and ".claude" in _ROOT.parts:
    REPO = _ROOT.parents[2]
else:
    REPO = _ROOT

# --- the 6-subject high-quality cohort (2026-05-27 v2) ---
# Dropped from the original 8: sub-28 (bad EDA: SCL ~0.01, spurious SCRs) and sub-30
# (weak/low EDA responder). User decision: keep only the 6 cleaner subjects.
COHORT = [
    "sub-19",
    "sub-23",
    "sub-24",
    "sub-27",
    "sub-31",
    "sub-33",
]

# --- canales descartados globalmente del Bloque 1 (STOPGAP Track A, auditoría 2026-06-07) ---
# FC1 = bad crónico (PyPREP lo marca en 76% de runs, 5/6 sujetos, 100% en sub-31/33; potencia
# anómala baja post-preproc, z-outlier en 2). TP10 = #2 (62%/50% en sub-31/33, z-outlier potencia
# alta = edge/EMG-like). Fz = marginal, fuera por prudencia (30% mean, sub-23 50%).
# PyPREP los interpola per-run de forma INCONSISTENTE (los deja crudos en parte de los runs ->
# contaminan CAR+ICA, que no se deshace downstream). Track A: descartarlos del set de análisis
# (set común de 29 ch para LOSO; no se fabrica data, a diferencia de interpolar post-CAR/ICA).
# Track B (deuda, antes de Bloque 2/paper): re-preprocesar con lista de bads curada per-sujeto
# (PyPREP auto + manual de los crónicos) interpolando ANTES de CAR+ICA. Ver diario 05_05.
DROP_CHANNELS: list[str] = ["FC1", "TP10", "Fz"]

# --- sets topográficos curados (29 ch, post-drop Track A) para índices edge/central ---
# Única fuente de verdad para el contraste muscular edge-vs-central (Bloque 2, tarea 2.4).
# EMG_EDGE = borde temporal/posterolateral (firma muscular craneal). TP10 sale (era el edge
#   más EMG-like, ahora dropeado); TP9 sobrevive (asimetría L/R documentada en el diario).
# CENTRAL  = central-parietal (región "señal" tipo Branković). Fz sale (dropeado); FCz queda.
# FRONTOPOLAR = Fp1/Fp2 = proxy ocular (frontopolar), se mantiene fuera del set muscular.
# Todos auto-filtrados contra DROP_CHANNELS para que NUNCA se desincronicen del set de 29 ch.
_EMG_EDGE_RAW = ["FT9", "FT10", "TP9", "TP10", "T7", "T8", "P7", "P8"]
_CENTRAL_RAW = ["Cz", "Pz", "Fz", "FCz", "CP1", "CP2", "C3", "C4", "P3", "P4"]
_FRONTOPOLAR_RAW = ["Fp1", "Fp2"]
EMG_EDGE: list[str] = [c for c in _EMG_EDGE_RAW if c not in DROP_CHANNELS]
CENTRAL: list[str] = [c for c in _CENTRAL_RAW if c not in DROP_CHANNELS]
FRONTOPOLAR: list[str] = [c for c in _FRONTOPOLAR_RAW if c not in DROP_CHANNELS]

# --- per-subject run exclusions for SCR/EDA analyses (acq tokens to drop) ---
# (sub-28's acq-b exclusion is now moot — sub-28 is dropped entirely from COHORT.)
RUN_EXCLUDE: dict[str, set[str]] = {}

# --- exact-label exclusions (aborted/short takes). Needed for scripts that read raw
#     (build_y_candidates), where these runs still exist; EEG-side scripts read preproc
#     where they are already absent (excluded at Paso 4), so this is harmless there. ---
RUN_EXCLUDE_LABELS: dict[str, set[str]] = {
    "sub-19": {"task-04_acq-a_run-005"},  # aborted take, 45 s
    "sub-27": {"task-04_acq-a_run-005"},  # aborted take, ~27 s (keep run-006)
}

# --- output root for the N=6 high-quality cohort. Kept separate from the N=8 run
#     (05_04/figures, 05_04/y_candidates) so the two can be compared. ---
OUT = REPO / "research_diary" / "context" / "05_04" / "cohort6"
NPZ_DIR = OUT / "y_candidates"
OUT.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

# --- plotting colors for up to 10 subjects ---
_cmap = matplotlib.colormaps["tab10"]
SUBJ_COLORS = {s: _cmap(i % 10) for i, s in enumerate(COHORT)}


def keep_run(sub: str, label: str) -> bool:
    """Return False if (sub, run) is QC-excluded.

    Parameters
    ----------
    sub : str
        e.g. "sub-28"
    label : str
        run label like "task-02_acq-b_run-008" (tokens joined by "_").
    """
    if label in RUN_EXCLUDE_LABELS.get(sub, set()):
        return False
    excl = RUN_EXCLUDE.get(sub, set())
    if not excl:
        return True
    tokens = set(label.split("_"))
    return tokens.isdisjoint(excl)
