"""Diagnostic audit of an ALREADY-FITTED ICA solution (no re-fitting).

Reloads the per-run ICA object saved by ``scripts/preprocessing/04_preprocessing_eeg.py``
(``*_desc-ica_ica.fif``) and reconstructs the exact wide-band CAR copy the pipeline fitted
it on (``raw_for_ica``: notch 50/100 -> band-pass 1-100 Hz -> +FCz -> montage -> interpolate
the logged bad channels -> common average reference; see 04_preprocessing_eeg.py:823-857).

Two modes share the SAME loaded ICA + SAME reconstructed ``raw_for_ica`` so the PNG dump
(read by the agent) and the interactive Qt viewer (run by the user) describe one decomposition:

  * default (Agg backend) -- "dump" mode. Writes per-component diagnostics to disk:
      - ica_components_grid*.png   (topographies of all components)
      - properties_ICAxxx.png      (topo + PSD + ERP-image + variance, annotated)
      - iclabel_heatmap.png + iclabel_probabilities.tsv   (full 7-class matrix)
      - ica_overlay.png            (signal with/without ica.exclude -> over-correction check)
      - recon_check.tsv            (recomputed vs logged ICLabel labels & exclude set)

  * --interactive (Qt5Agg backend) -- run BY THE USER (Qt windows block and the agent
    cannot see them). Opens ica.plot_sources + ica.plot_components on the same objects.

WHY reconstruct instead of reusing a saved derivative: muscle artifacts live in 48-100 Hz,
which the analysis copy (desc-filtered, 0.1-48 Hz) has already removed. Topographies come
straight from the ICA mixing matrix and are therefore EXACT regardless of the inst; only the
PSD / time-course panels depend on the reconstruction (faithful, not sample-identical because
bad-segment annotations are not reloaded by default -- see ``--bad-annotations``).

Run (sequentially, n_jobs=1 -- Windows WinError 1450 with aggressive parallelism):
    micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.ica_audit \
        --subject 23 --session vr --task 04 --acq b --run 009

Interactive (run by the user, e.g. with the ``! <cmd>`` prompt prefix):
    micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.ica_audit \
        --subject 23 --session vr --task 04 --acq b --run 009 --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# Backend must be selected BEFORE importing pyplot. The user-facing interactive viewer needs
# Qt5Agg (blocking windows); the agent-facing dump needs the headless Agg backend.
_INTERACTIVE = "--interactive" in sys.argv
matplotlib.use("Qt5Agg" if _INTERACTIVE else "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import mne  # noqa: E402
from mne.preprocessing import read_ica  # noqa: E402
from mne_bids import BIDSPath, read_raw_bids  # noqa: E402

from src.campeones_analysis.multimodal_arousal.cohort import OUT, REPO  # noqa: E402

# ICLabel class order, copied verbatim from 04_preprocessing_eeg.py:916-920 so the recomputed
# matrix lines up with what the pipeline used for the R-7 exclusion decision.
ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]
ARTIFACT_CLASSES = ["muscle artifact", "eye blink", "heart beat", "channel noise"]
BRAIN_IDX = ICLABEL_CLASSES.index("brain")
ICLABEL_THRESHOLD = 0.85  # R-7: min top-class prob to trust an ICLabel artifact call
BRAIN_FLOOR = 0.30  # R-7: never exclude if brain prob >= this

DERIV = REPO / "data" / "derivatives" / "campeones_preproc"
RAW_ROOT = REPO / "data" / "raw"
MONTAGE_FILE = REPO / "scripts" / "preprocessing" / "BC-32_FCz_modified.bvef"
LOG_JSON = DERIV / "logs_preprocessing_details_all_subjects_eeg.json"


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------
def _basename(subject: str, session: str, task: str, acq: str, run: str) -> str:
    return f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}"


def _eeg_dir(subject: str, session: str) -> Path:
    return DERIV / f"sub-{subject}" / f"ses-{session}" / "eeg"


def _ica_path(subject: str, session: str, task: str, acq: str, run: str) -> Path:
    base = _basename(subject, session, task, acq, run)
    return _eeg_dir(subject, session) / f"{base}_desc-ica_ica.fif"


def _read_log_entry(subject: str, session: str, task: str, run: str) -> dict:
    """Return the preprocessing-log dict for this run, or {} if absent.

    The log key is ``f"{task}_run-{run}"`` (LogPreprocessingDetails init,
    04_preprocessing_eeg.py:241); ``acq`` is not part of the key because the run number is
    already unique per (task, acq).
    """
    if not LOG_JSON.exists():
        return {}
    with open(LOG_JSON, encoding="utf-8") as fh:
        log = json.load(fh)
    return log.get(subject, {}).get(session, {}).get(f"{task}_run-{run}", {})


def get_bad_channels(subject: str, session: str, task: str, run: str) -> list[str]:
    """Bad channels the pipeline interpolated, read from the log (source of truth).

    Prefers ``interpolated_channels`` (04_preprocessing_eeg.py:742); falls back to
    ``bad_channels``. PyPREP's RANSAC detection is not deterministic, so re-detecting would
    NOT reproduce the same set -- we must reuse what the pipeline logged.
    """
    entry = _read_log_entry(subject, session, task, run)
    bads = entry.get("interpolated_channels") or entry.get("bad_channels") or []
    return list(bads)


# --------------------------------------------------------------------------------------
# Shared loaders (single source of truth for both dump and interactive modes)
# --------------------------------------------------------------------------------------
def reconstruct_raw_for_ica(
    subject: str,
    session: str,
    task: str,
    acq: str,
    run: str,
    load_bad_annotations: bool = False,
) -> mne.io.BaseRaw:
    """Rebuild the wide-band CAR copy ICA was fitted on (04_preprocessing_eeg.py:823-857).

    notch(50,100) -> band-pass(1,100) -> add FCz -> montage -> interpolate logged bads -> CAR.
    Topographies are unaffected by any inexactness here (they come from the mixing matrix);
    only PSD / time-course panels use this signal.
    """
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acq,
        run=run,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
        root=str(RAW_ROOT),
    )
    raw = read_raw_bids(bids_path, verbose=False)
    raw.info["bads"] = []
    raw.load_data()

    # 3A notch + wide-band 1-100 Hz copy (from the notched-only data, not the 0.1-48 copy)
    raw.notch_filter(freqs=[50, 100], picks="eeg", method="fir", phase="zero", verbose=False)
    raw.filter(l_freq=1.0, h_freq=100.0, picks="eeg", method="fir", phase="zero", verbose=False)

    # Section 7 spatial state: add FCz -> montage -> interpolate bads -> common average ref
    raw = mne.add_reference_channels(raw, ref_channels=["FCz"])
    if not MONTAGE_FILE.exists():
        raise FileNotFoundError(f"Montage file not found: {MONTAGE_FILE}")
    montage = mne.channels.read_custom_montage(str(MONTAGE_FILE))
    raw.set_montage(montage)

    bads = get_bad_channels(subject, session, task, run)
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=False)
    raw, _ = mne.set_eeg_reference(raw, ref_channels="average", copy=False)

    if load_bad_annotations:
        _maybe_load_bad_annotations(raw, subject, session, task, acq, run)

    return raw


def _maybe_load_bad_annotations(raw, subject, session, task, acq, run) -> None:
    """Best-effort: copy bad-segment annotations from the preproc events sidecar, if present.

    Off by default; only relevant if you want PSDs computed over good segments only.
    """
    base = _basename(subject, session, task, acq, run)
    events_tsv = _eeg_dir(subject, session) / f"{base}_desc-preproc_events.tsv"
    if not events_tsv.exists():
        print(f"  [bad-annotations] {events_tsv.name} not found; skipping.")
        return
    df = pd.read_csv(events_tsv, sep="\t")
    bad = df[df.get("trial_type", df.get("description", "")).astype(str).str.lower() == "bad"]
    if bad.empty:
        print("  [bad-annotations] no 'bad' rows in events sidecar; skipping.")
        return
    ann = mne.Annotations(
        onset=bad["onset"].to_numpy(),
        duration=bad["duration"].to_numpy(),
        description=["bad"] * len(bad),
    )
    raw.set_annotations(ann)
    print(f"  [bad-annotations] loaded {len(bad)} bad segments.")


def load_ica(subject, session, task, acq, run) -> mne.preprocessing.ICA:
    path = _ica_path(subject, session, task, acq, run)
    if not path.exists():
        raise FileNotFoundError(f"ICA file not found: {path}")
    return read_ica(str(path), verbose=False)


# --------------------------------------------------------------------------------------
# ICLabel: recompute the full 7-class matrix (the log only stores top/brain per component)
# --------------------------------------------------------------------------------------
def iclabel_table(raw, ica) -> pd.DataFrame:
    """Recompute the (n_components, 7) ICLabel matrix and apply the R-7 exclusion rule.

    Mirrors 04_preprocessing_eeg.py:910-991. ``excluded_in_pipeline`` comes from the loaded
    ica.exclude (i.e. what was actually applied), so any mismatch with ``action`` is itself
    diagnostic (e.g. a pattern-matching call that ICLabel alone would not flag).
    """
    from mne_icalabel.iclabel import iclabel_label_components

    proba = iclabel_label_components(raw, ica)  # (n_components, 7)
    argmax = proba.argmax(axis=1)
    top = proba.max(axis=1)
    brain = proba[:, BRAIN_IDX]
    labels = [ICLABEL_CLASSES[i] for i in argmax]

    rows = []
    excluded = set(int(i) for i in ica.exclude)
    for i in range(proba.shape[0]):
        is_artifact_call = labels[i] in ARTIFACT_CLASSES and top[i] >= ICLABEL_THRESHOLD
        if is_artifact_call and brain[i] < BRAIN_FLOOR:
            action = "EXCLUDE"
        elif labels[i] == "brain":
            action = "KEEP"
        else:
            action = "REVIEW"
        row = {
            "component": f"ICA{i:03d}",
            "idx": i,
            "iclabel_class": labels[i],
            "top_prob": round(float(top[i]), 4),
            "brain_prob": round(float(brain[i]), 4),
            "action_iclabel_only": action,
            "excluded_in_pipeline": i in excluded,
        }
        for c, cls in enumerate(ICLABEL_CLASSES):
            row[f"p_{cls.replace(' ', '_')}"] = round(float(proba[i, c]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def recon_check(df_iclabel: pd.DataFrame, ica, subject, session, task, run) -> pd.DataFrame:
    """Sanity-check the reconstruction against the pipeline log.

    Compares (a) recomputed argmax label vs logged ``component_labels`` and (b) the loaded
    ica.exclude set vs logged ``ica_components_excluded``. High label agreement + identical
    exclude set => the reconstructed raw_for_ica reproduces the pipeline's fit input.
    """
    entry = _read_log_entry(subject, session, task, run)
    logged_labels = entry.get("iclabel_classifications", {}).get("component_labels", {})
    logged_excluded = sorted(int(i) for i in entry.get("ica_components_excluded", []))
    loaded_excluded = sorted(int(i) for i in ica.exclude)

    rows = []
    for _, r in df_iclabel.iterrows():
        comp = r["component"]
        rows.append(
            {
                "component": comp,
                "recomputed_label": r["iclabel_class"],
                "logged_label": logged_labels.get(comp, "?"),
                "label_match": logged_labels.get(comp, "?") == r["iclabel_class"],
            }
        )
    df = pd.DataFrame(rows)
    n = len(df)
    agree = int(df["label_match"].sum()) if n else 0
    print("\n=== RECONSTRUCTION SANITY CHECK ===")
    print(f"  ICLabel label agreement (recomputed vs logged): {agree}/{n} "
          f"({100 * agree / n:.1f}%)" if n else "  (no components)")
    print(f"  Loaded ica.exclude:        {loaded_excluded}")
    print(f"  Logged ica_components_excluded: {logged_excluded}")
    print(f"  Exclude set identical: {loaded_excluded == logged_excluded}")
    return df


# --------------------------------------------------------------------------------------
# Dump mode (Agg): generate the PNGs the agent reads
# --------------------------------------------------------------------------------------
def _save_components_grid(ica, raw, fig_dir: Path) -> None:
    figs = ica.plot_components(inst=raw, show=False)
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    for k, fig in enumerate(figs):
        out = fig_dir / (f"ica_components_grid_p{k + 1}.png" if len(figs) > 1
                         else "ica_components_grid.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")


def _save_properties(ica, raw, df_iclabel: pd.DataFrame, fig_dir: Path) -> None:
    meta = df_iclabel.set_index("idx")
    n = ica.n_components_
    for i in range(n):
        figs = ica.plot_properties(raw, picks=[i], show=False)
        fig = figs[0] if isinstance(figs, (list, tuple)) else figs
        r = meta.loc[i]
        flag = "EXCLUDED" if r["excluded_in_pipeline"] else "kept"
        fig.suptitle(
            f"ICA{i:03d} | {r['iclabel_class']} | top={r['top_prob']:.2f} "
            f"brain={r['brain_prob']:.2f} | pipeline: {flag}",
            fontsize=10,
        )
        out = fig_dir / f"properties_ICA{i:03d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {n} properties_ICAxxx.png")


def _save_iclabel_heatmap(df_iclabel: pd.DataFrame, fig_dir: Path) -> None:
    pcols = [f"p_{c.replace(' ', '_')}" for c in ICLABEL_CLASSES]
    mat = df_iclabel[pcols].to_numpy()
    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * n + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(ICLABEL_CLASSES)))
    ax.set_xticklabels(ICLABEL_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ylabels = [
        f"{c}{'  *EXCL' if e else ''}"
        for c, e in zip(df_iclabel["component"], df_iclabel["excluded_in_pipeline"])
    ]
    ax.set_yticklabels(ylabels, fontsize=7)
    for i in range(n):
        for j in range(len(ICLABEL_CLASSES)):
            v = mat[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="ICLabel probability")
    ax.set_title("ICLabel 7-class probabilities (*EXCL = excluded by pipeline)", fontsize=9)
    out = fig_dir / "iclabel_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def _save_overlay(ica, raw, fig_dir: Path) -> None:
    if not ica.exclude:
        print("  ica.exclude is empty -- skipping overlay (nothing removed).")
        return
    fig = ica.plot_overlay(raw, exclude=ica.exclude, picks="eeg", show=False)
    out = fig_dir / "ica_overlay.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def run_dump(subject, session, task, acq, run, load_bad_annotations: bool) -> None:
    print(f"=== ICA AUDIT (dump) sub-{subject} ses-{session} task-{task} acq-{acq} run-{run} ===")
    ica = load_ica(subject, session, task, acq, run)
    raw = reconstruct_raw_for_ica(subject, session, task, acq, run, load_bad_annotations)
    print(f"  ICA: n_components={ica.n_components_}, exclude={sorted(int(i) for i in ica.exclude)}")

    out_root = OUT / "ica_audit" / f"sub-{subject}"
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df_iclabel = iclabel_table(raw, ica)
    df_iclabel.to_csv(tab_dir / "iclabel_probabilities.tsv", sep="\t", index=False)
    print(f"  wrote tables/iclabel_probabilities.tsv ({len(df_iclabel)} components)")

    df_recon = recon_check(df_iclabel, ica, subject, session, task, run)
    df_recon.to_csv(tab_dir / "recon_check.tsv", sep="\t", index=False)

    _save_components_grid(ica, raw, fig_dir)
    _save_iclabel_heatmap(df_iclabel, fig_dir)
    _save_properties(ica, raw, df_iclabel, fig_dir)
    _save_overlay(ica, raw, fig_dir)

    print(f"\n✓ Done. Outputs in: {out_root}")


# --------------------------------------------------------------------------------------
# Interactive mode (Qt5Agg): run BY THE USER
# --------------------------------------------------------------------------------------
def run_interactive(subject, session, task, acq, run, load_bad_annotations: bool) -> None:
    print(f"=== ICA AUDIT (interactive) sub-{subject} task-{task} acq-{acq} run-{run} ===")
    print(f"matplotlib backend: {matplotlib.get_backend()}")
    ica = load_ica(subject, session, task, acq, run)
    raw = reconstruct_raw_for_ica(subject, session, task, acq, run, load_bad_annotations)
    print(f"  ICA: n_components={ica.n_components_}, exclude={sorted(int(i) for i in ica.exclude)}")
    print("\nOpening ica.plot_sources (close the window to continue)...")
    ica.plot_sources(raw, block=True, show=True)
    print("Opening ica.plot_components topographies...")
    ica.plot_components(inst=raw, show=True)
    plt.show(block=True)


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostic audit of a saved ICA solution.")
    p.add_argument("--subject", required=True, help="Subject ID, e.g. '23'")
    p.add_argument("--session", default="vr", help="Session ID (default: vr)")
    p.add_argument("--task", default="04", help="Task ID (default: 04)")
    p.add_argument("--acq", default="b", help="Acquisition (default: b)")
    p.add_argument("--run", default="009", help="Run ID (default: 009)")
    p.add_argument("--interactive", action="store_true",
                   help="Open Qt viewer (run by the USER; blocks).")
    p.add_argument("--bad-annotations", action="store_true",
                   help="Reload bad-segment annotations into the reconstructed raw (optional).")
    args = p.parse_args()

    fn = run_interactive if args.interactive else run_dump
    fn(args.subject, args.session, args.task, args.acq, args.run, args.bad_annotations)


if __name__ == "__main__":
    main()
