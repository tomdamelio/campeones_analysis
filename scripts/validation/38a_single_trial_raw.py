#!/usr/bin/env python
"""Single-trial raw EEG — Sanity check 3.1.

Shows 3 individual CHANGE_PHOTO + 3 individual NO_CHANGE_PHOTO epochs
in O1/O2 (occipital mean), sin promediado.

Pre [-0.55, -0.05]s y post [+0.05, +0.55]s windows sombreadas para
coincidir con la definición del script 27b.

Usage
-----
    micromamba run -n campeones python scripts/validation/38a_single_trial_raw.py --subject 27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREPROC_ROOT = PROJECT_ROOT / "data" / "derivatives" / "campeones_preproc"
PHOTO_EVENTS_ROOT = PROJECT_ROOT / "data" / "derivatives" / "photo_events"
RESULTS_ROOT = PROJECT_ROOT / "results" / "validation" / "sanity_checks"
SESSION = "vr"

TMIN, TMAX = -1.5, 1.5
BASELINE = (-1.5, -1.0)
VIS_TMIN, VIS_TMAX = -0.6, 0.7

PRE_START, PRE_END = -0.55, -0.05
POST_START, POST_END = 0.05, 0.55

OCCIPITAL = ["O1", "O2"]
N_TRIALS = 6
YLIM = (-50, 50)   # µV — escala fija para comparación entre trials
SEED = 42

EVENT_ID = {"CHANGE_PHOTO": 1, "NO_CHANGE_PHOTO": 2}
RUNS_CONFIG = {
    "27": [
        {"run": "002", "acq": "a", "task": "01"},
        {"run": "003", "acq": "a", "task": "02"},
        {"run": "004", "acq": "a", "task": "03"},
        {"run": "006", "acq": "a", "task": "04"},
        {"run": "007", "acq": "b", "task": "01"},
        {"run": "009", "acq": "b", "task": "03"},
        {"run": "010", "acq": "b", "task": "04"},
    ],
}
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
    'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
]


def load_all_epochs(subject: str) -> mne.Epochs:
    all_list = []
    for rc in RUNS_CONFIG[subject]:
        run_id, task, acq = rc["run"], rc["task"], rc["acq"]
        eeg_dir = PREPROC_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
        vhdr = eeg_dir / (
            f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
            f"_run-{run_id}_desc-preproc_eeg.vhdr"
        )
        tsv = (
            PHOTO_EVENTS_ROOT / f"sub-{subject}" / f"ses-{SESSION}" / "eeg"
            / f"sub-{subject}_ses-{SESSION}_task-{task}_acq-{acq}"
              f"_run-{run_id}_desc-photo_events.tsv"
        )
        if not vhdr.exists() or not tsv.exists():
            continue
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        df = pd.read_csv(tsv, sep="\t")
        rows = df[df["trial_type"].isin(EVENT_ID)]
        if rows.empty:
            continue
        sfreq = raw.info["sfreq"]
        avail = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        onsets = np.round(rows["onset"].astype(float).values * sfreq).astype(int) + raw.first_samp
        eids = rows["trial_type"].map(EVENT_ID).values
        valid = (onsets >= raw.first_samp) & (onsets < raw.first_samp + raw.n_times)
        onsets, eids = onsets[valid], eids[valid]
        if len(onsets) == 0:
            continue
        mne_ev = np.column_stack([onsets, np.zeros(len(onsets), int), eids])
        mne_ev = mne_ev[np.argsort(mne_ev[:, 0])]
        run_eid = {k: v for k, v in EVENT_ID.items() if k in rows["trial_type"].unique()}
        try:
            ep = mne.Epochs(raw, mne_ev, event_id=run_eid,
                            tmin=TMIN, tmax=TMAX, picks=avail,
                            baseline=BASELINE, preload=True, verbose=False)
            all_list.append(ep)
            print(f"  run-{run_id}: {len(ep)} epochs")
        except Exception as e:
            print(f"  run-{run_id}: {e}")
    if not all_list:
        raise RuntimeError("No epochs loaded")
    return mne.concatenate_epochs(all_list, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="27")
    args = parser.parse_args()
    sub = args.subject

    out_dir = RESULTS_ROOT / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading epochs for sub-{sub}...")
    epochs = load_all_epochs(sub)
    epochs_vis = epochs.copy().crop(tmin=VIS_TMIN, tmax=VIS_TMAX)
    time_ms = epochs_vis.times * 1000

    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, N_TRIALS, figsize=(N_TRIALS * 3.2, 6), sharey=False)
    fig.suptitle(
        f"sub-{sub} — Single-trial raw EEG (O1/O2 mean)  |  y-axis fijo: {YLIM} µV\n"
        "Sanity check 3.1: ¿se ve el VEP en trials individuales?",
        fontsize=11,
    )

    cond_info = [
        ("CHANGE_PHOTO",    "C3", "CHANGE"),
        ("NO_CHANGE_PHOTO", "C2", "NO_CHANGE"),
    ]

    occ_avail = [ch for ch in OCCIPITAL if ch in epochs_vis.ch_names]
    if not occ_avail:
        raise RuntimeError("No occipital channels found")

    for row, (cond, color, label) in enumerate(cond_info):
        try:
            ep = epochs_vis[cond]
        except KeyError:
            print(f"  No epochs for {cond}")
            continue
        n_avail = len(ep)
        idx = rng.choice(n_avail, size=min(N_TRIALS, n_avail), replace=False)
        ch_idx = [ep.ch_names.index(ch) for ch in occ_avail]

        for col, trial_i in enumerate(idx):
            ax = axes[row][col]
            signal = ep.get_data()[trial_i][ch_idx].mean(axis=0) * 1e6  # µV

            ax.axvspan(PRE_START * 1000, PRE_END * 1000,
                       alpha=0.15, color="steelblue", zorder=0)
            ax.axvspan(POST_START * 1000, POST_END * 1000,
                       alpha=0.15, color="tomato", zorder=0)
            ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.6)
            ax.plot(time_ms, signal, color=color, lw=1.3)
            ax.set_ylim(YLIM)
            ax.set_title(f"{label} — trial #{trial_i + 1}", fontsize=9)
            ax.set_xlabel("Time (ms)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Amplitude (µV)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.4, label="pre  [−550, −50 ms]"),
        plt.Rectangle((0, 0), 1, 1, fc="tomato",    alpha=0.4, label="post [+50, +550 ms]"),
        plt.Line2D([0], [0], color="k", ls="--", lw=1, label="onset (t=0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = out_dir / f"sub-{sub}_38a_single_trial_raw.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
