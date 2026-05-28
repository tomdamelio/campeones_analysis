"""Fase 2 de Tarea B: re-correr la suite SCR con drop-only aplicado, salidas a cohort6_ar/.

Sin editar módulos protegidos (cohort/erp_scr/tfr_psd_scr). Estrategia: monkey-patch
per-módulo del `build_subject_epochs` local + redirección de OUT/NPZ_DIR/FIG_DIR/QA_DIR
hacia `cohort6_ar/`, luego call a cada main(). Para los scripts con loop propio (erp_scr
ERP cluster + scr_ro_delta_csd), implementación inline que mimetiza la lógica con
drop-only inyectado.

Secuencial, tolerante a fallos (try/except por análisis), log a stdout. ~2.5–3.5h.

Run:
  micromamba run -n campeones python -u -m src.campeones_analysis.multimodal_arousal.run_suite_ar
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ----- target output root: cohort6_ar/ -----
from src.campeones_analysis.multimodal_arousal.erp_scr import OUT as ORIG_OUT
from src.campeones_analysis.multimodal_arousal.epochs_qc import apply_drop_only

OUT_AR = ORIG_OUT.parent / "cohort6_ar"
NPZ_DIR_AR = OUT_AR / "y_candidates"
ORIG_NPZ_DIR = ORIG_OUT / "y_candidates"
(OUT_AR / "figures").mkdir(parents=True, exist_ok=True)
NPZ_DIR_AR.mkdir(parents=True, exist_ok=True)


def _copy_inputs_to_ar():
    """Copy INPUT files from cohort6/y_candidates -> cohort6_ar/y_candidates.

    The suite reads <sub>_continuous.npz (and sometimes <sub>_y_candidates.npz) from NPZ_DIR.
    After patching NPZ_DIR to cohort6_ar, the loaders need to find these files there.
    Outputs (evoked .fif, CSV summaries) are regenerated; we do NOT copy them so cohort6_ar
    is a clean re-generation. Idempotent (skips existing files)."""
    import shutil
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT
    INPUT_PATTERNS = [
        "{sub}_continuous.npz",
        "{sub}_y_candidates.npz",
    ]
    CONTEXT_FILES = ["pca_loadings.json", "run_summary.csv", "cvx_qc.csv"]
    copied = 0
    for sub in COHORT:
        for pat in INPUT_PATTERNS:
            src = ORIG_NPZ_DIR / pat.format(sub=sub)
            dst = NPZ_DIR_AR / pat.format(sub=sub)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    for f in CONTEXT_FILES:
        src = ORIG_NPZ_DIR / f
        dst = NPZ_DIR_AR / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  [inputs] copied {copied} files to {NPZ_DIR_AR}", flush=True)


# ----- core wrapper -----
def make_bse_ar(orig_bse):
    """Wrap a build_subject_epochs so it applies drop-only after building. Idempotent."""
    if getattr(orig_bse, "_ar_wrapped", False):
        return orig_bse

    def bse_ar(sub):
        r, s = orig_bse(sub)
        if r is None or s is None or len(r) == 0 or len(s) == 0:
            return r, s
        try:
            r2, s2, thresh = apply_drop_only(r, s, random_state=97)
            print(f"    [drop-only] {sub}: real {len(r)}->{len(r2)} ({100*(1-len(r2)/len(r)):.1f}%); "
                  f"silent {len(s)}->{len(s2)} ({100*(1-len(s2)/len(s)):.1f}%); thr={thresh.get('eeg', np.nan)*1e6:.0f}uV",
                  flush=True)
            return r2, s2
        except Exception as exc:
            print(f"    [drop-only FAILED] {sub}: {exc}; using uncleaned", flush=True)
            return r, s

    bse_ar._ar_wrapped = True
    return bse_ar


def run_step(label: str, fn):
    print("\n" + "=" * 78, flush=True)
    print(f"[{label}]   start", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"[{label}]   DONE in {time.time()-t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"[{label}]   FAILED: {exc}", flush=True)
        traceback.print_exc()


def _patch_paths(mod, subfolder=None, qa_dir=False):
    """Redirect mod's OUT/NPZ_DIR/FIG_DIR (and QA_DIR if present) to cohort6_ar."""
    mod.OUT = OUT_AR
    mod.NPZ_DIR = NPZ_DIR_AR
    if qa_dir:
        mod.QA_DIR = OUT_AR / "qa_artifact_vs_signal"
        mod.QA_DIR.mkdir(parents=True, exist_ok=True)
        mod.FIG_DIR = mod.QA_DIR / "figures"
    elif subfolder is None:
        mod.FIG_DIR = OUT_AR / "figures"
    else:
        mod.FIG_DIR = OUT_AR / "figures" / subfolder
    mod.FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Group A: scripts using build_subject_epochs (monkey-patch + call main)
# =============================================================================
def run_tfr_psd():
    from src.campeones_analysis.multimodal_arousal import tfr_psd_scr
    tfr_psd_scr.build_subject_epochs = make_bse_ar(tfr_psd_scr.build_subject_epochs)
    _patch_paths(tfr_psd_scr, subfolder=None)
    tfr_psd_scr.main()


def run_alpha_hypothesis():
    from src.campeones_analysis.multimodal_arousal import alpha_hypothesis_scr
    alpha_hypothesis_scr.build_subject_epochs = make_bse_ar(alpha_hypothesis_scr.build_subject_epochs)
    _patch_paths(alpha_hypothesis_scr, subfolder="alpha_hypothesis")
    alpha_hypothesis_scr.main()


def run_fooof():
    from src.campeones_analysis.multimodal_arousal import fooof_scr
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT
    fooof_scr.build_subject_epochs = make_bse_ar(fooof_scr.build_subject_epochs)
    _patch_paths(fooof_scr, qa_dir=True)
    fooof_scr.main(list(COHORT))


def run_topo_variance():
    from src.campeones_analysis.multimodal_arousal import topo_variance_scr
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT
    topo_variance_scr.build_subject_epochs = make_bse_ar(topo_variance_scr.build_subject_epochs)
    _patch_paths(topo_variance_scr, qa_dir=True)
    topo_variance_scr.main(list(COHORT))


def run_topomap_delta_theta():
    from src.campeones_analysis.multimodal_arousal import topomap_delta_theta_scr
    topomap_delta_theta_scr.build_subject_epochs = make_bse_ar(topomap_delta_theta_scr.build_subject_epochs)
    _patch_paths(topomap_delta_theta_scr, subfolder="topomap_delta_theta")
    topomap_delta_theta_scr.main()


def run_decoding_scr():
    from src.campeones_analysis.multimodal_arousal import decoding_scr
    decoding_scr.build_subject_epochs = make_bse_ar(decoding_scr.build_subject_epochs)
    _patch_paths(decoding_scr, subfolder="decoding")
    decoding_scr.main()


def run_decoding_loso():
    from src.campeones_analysis.multimodal_arousal import decoding_loso_scr
    decoding_loso_scr.build_subject_epochs = make_bse_ar(decoding_loso_scr.build_subject_epochs)
    _patch_paths(decoding_loso_scr, subfolder="decoding_loso")
    decoding_loso_scr.main()


def run_scr_alpha_csd():
    from src.campeones_analysis.multimodal_arousal import scr_alpha_csd
    scr_alpha_csd.build_subject_epochs = make_bse_ar(scr_alpha_csd.build_subject_epochs)
    _patch_paths(scr_alpha_csd, subfolder="scr_alpha_csd")
    scr_alpha_csd.main()


def run_scr_band_csd():
    from src.campeones_analysis.multimodal_arousal import scr_band_csd
    scr_band_csd.build_subject_epochs = make_bse_ar(scr_band_csd.build_subject_epochs)
    _patch_paths(scr_band_csd, subfolder="scr_band_csd")
    # simulate argparse: full band set, all subjects
    sys_argv_bak = sys.argv
    sys.argv = ["scr_band_csd", "--bands", "theta", "beta", "gamma"]
    try:
        scr_band_csd.main()
    finally:
        sys.argv = sys_argv_bak


# =============================================================================
# Group B: custom-loop scripts (inline implementation with drop-only)
# =============================================================================
def _build_sensor_epochs_per_subject(sub, l_freq=1.0, h_freq=20.0, sfreq=250.0):
    """Mimics the per-run epoching loop of erp_scr but returns concat real & silent
    sensor epochs ready for drop-only. Reuses erp_scr helpers (read-only)."""
    from src.campeones_analysis.multimodal_arousal.erp_scr import (
        EDA_FS, NPZ_DIR as ORIG_NPZ, RNG, attach_montage_and_drop_no_pos,
        detect_scr_onsets_s, epoch_one_run, filter_clean_onsets, run_label,
        runs_for, sample_silent_controls,
    )
    cont_path = ORIG_NPZ / f"{sub}_continuous.npz"
    if not cont_path.exists():
        return None, None
    cont = np.load(cont_path, allow_pickle=True)
    runs_in_npz = list(cont["runs"])
    real_eps, silent_eps = [], []
    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq, h_freq, verbose="ERROR")
            raw.resample(sfreq, verbose="ERROR")
            dur = float(raw.times[-1])
            phasic = np.asarray(cont[f"{label}__eda_phasic"], float)
            onsets = detect_scr_onsets_s(phasic, EDA_FS)
            onsets = onsets[onsets < dur]
            onsets = filter_clean_onsets(onsets, phasic, EDA_FS)
            silent = sample_silent_controls(len(onsets), dur, phasic, EDA_FS, RNG,
                                            avoid_onsets_s=onsets)
            er = epoch_one_run(raw, onsets, code=1)
            es = epoch_one_run(raw, silent, code=2)
            if er is not None and len(er): real_eps.append(er)
            if es is not None and len(es): silent_eps.append(es)
        except Exception as exc:
            print(f"    {label}: skip ({exc})", flush=True)
    if not real_eps or not silent_eps:
        return None, None
    real = mne.concatenate_epochs(real_eps, verbose="ERROR")
    silent = mne.concatenate_epochs(silent_eps, verbose="ERROR")
    return real, silent


def run_erp_grandaverage_ar():
    """Re-implement erp_scr per-subject + erp_scr_grandaverage cluster, with drop-only."""
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT
    from src.campeones_analysis.multimodal_arousal.erp_scr import PLOT_CH, TOPO_TIMES
    from mne.stats import permutation_cluster_1samp_test

    fig_dir = OUT_AR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_sub = {}
    for sub in COHORT:
        print(f"\n  === {sub} ===", flush=True)
        real_ep, silent_ep = _build_sensor_epochs_per_subject(sub, l_freq=1.0, h_freq=20.0)
        if real_ep is None or silent_ep is None:
            print(f"    no epochs"); continue
        try:
            real_clean, silent_clean, thr = apply_drop_only(real_ep, silent_ep, random_state=97)
            print(f"    drop-only: real {len(real_ep)}->{len(real_clean)} silent {len(silent_ep)}->{len(silent_clean)}", flush=True)
        except Exception as exc:
            print(f"    drop-only failed: {exc}; using uncleaned", flush=True)
            real_clean, silent_clean = real_ep, silent_ep
        ev_r = real_clean.average(); ev_s = silent_clean.average()
        ev_r.save(NPZ_DIR_AR / f"{sub}_evoked_scr_real-ave.fif", overwrite=True)
        ev_s.save(NPZ_DIR_AR / f"{sub}_evoked_scr_random-ave.fif", overwrite=True)
        per_sub[sub] = (ev_r, ev_s, len(real_clean), len(silent_clean))

    if len(per_sub) < 3:
        print("  [erp_grandavg_ar] <3 subjects, skip cluster"); return

    subs = list(per_sub.keys())
    # ROI cluster (Cz mean) — keep simple: cluster on Cz channel diff across subjects
    times = per_sub[subs[0]][0].times
    diffs_cz = []
    for s in subs:
        er, es, _, _ = per_sub[s]
        if "Cz" in er.ch_names:
            ci = er.ch_names.index("Cz")
            diffs_cz.append((er.data[ci] - es.data[ci]) * 1e6)
    if len(diffs_cz) >= 3:
        D = np.array(diffs_cz)
        n_perm = min(2 ** len(subs), 2048)
        T, cl, pv, _ = permutation_cluster_1samp_test(D, n_permutations=n_perm, tail=0, seed=20260513, verbose="ERROR")
        minp = float(pv.min()) if len(pv) else float("nan")
        print(f"  ERP cluster (Cz, N={len(subs)}): {len(cl)} clusters, min p={minp:.4f}", flush=True)
        rows = []
        for c, p in zip(cl, pv):
            idx = c[0] if isinstance(c, tuple) else c
            m = np.zeros(len(times), bool); m[idx] = True
            rows.append(dict(t_start=float(times[m].min()), t_end=float(times[m].max()), p_value=float(p)))
        pd.DataFrame(rows).to_csv(NPZ_DIR_AR / "erp_scr_grandaverage_ar_clusters_Cz.csv", index=False)
        # quick figure
        fig, ax = plt.subplots(figsize=(11, 5))
        m_real = np.mean([per_sub[s][0].data[per_sub[s][0].ch_names.index("Cz")] * 1e6 for s in subs if "Cz" in per_sub[s][0].ch_names], axis=0)
        m_silent = np.mean([per_sub[s][1].data[per_sub[s][1].ch_names.index("Cz")] * 1e6 for s in subs if "Cz" in per_sub[s][1].ch_names], axis=0)
        ax.plot(times, m_real, color="C3", lw=2, label="real (drop-only)")
        ax.plot(times, m_silent, color="0.5", lw=2, label="silent (drop-only)")
        ax.axvline(0, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("time (s)"); ax.set_ylabel("uV"); ax.set_title(f"ERP grand-avg Cz (drop-only)  min p={minp:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "Y3_erp_scr_grandaverage_ar_Cz.png", dpi=130); plt.close(fig)


def run_delta_csd_ar():
    """Re-implement scr_ro_delta_csd with drop-only on sensor epochs BEFORE CSD."""
    from src.campeones_analysis.multimodal_arousal.cohort import COHORT
    from mne.preprocessing import compute_current_source_density
    from mne.stats import permutation_cluster_1samp_test

    fig_dir = OUT_AR / "figures" / "scr_ro_delta_csd"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ROI = ["Fz", "FCz", "Cz", "CP1", "CP2", "Pz", "C3", "C4"]

    per_sub = {}
    for sub in COHORT:
        print(f"\n  === {sub} ===", flush=True)
        # sensor epochs in delta band, 250 Hz
        real_ep, silent_ep = _build_sensor_epochs_per_subject(sub, l_freq=0.5, h_freq=4.0)
        if real_ep is None or silent_ep is None:
            print("    no epochs"); continue
        try:
            real_c, silent_c, _ = apply_drop_only(real_ep, silent_ep, random_state=97)
            print(f"    drop-only: real {len(real_ep)}->{len(real_c)} silent {len(silent_ep)}->{len(silent_c)}", flush=True)
        except Exception as exc:
            print(f"    drop-only failed: {exc}", flush=True)
            real_c, silent_c = real_ep, silent_ep
        real_csd = compute_current_source_density(real_c)
        silent_csd = compute_current_source_density(silent_c)
        er = real_csd.average(); es = silent_csd.average()
        er.save(NPZ_DIR_AR / f"scr_ro_delta_csd_ar_{sub}_evoked_real-ave.fif", overwrite=True)
        es.save(NPZ_DIR_AR / f"scr_ro_delta_csd_ar_{sub}_evoked_silent-ave.fif", overwrite=True)
        per_sub[sub] = (er, es)

    if len(per_sub) < 3:
        print("  [delta_csd_ar] <3 subjects, skip cluster"); return
    subs = list(per_sub.keys())
    times = per_sub[subs[0]][0].times
    common = [c for c in per_sub[subs[0]][0].ch_names if all(c in per_sub[s][0].ch_names for s in subs)]
    roi = [c for c in ROI if c in common]
    def roi_mean(ev):
        idx = [ev.ch_names.index(c) for c in roi]
        return ev.data[idx].mean(axis=0)
    diffs = np.array([roi_mean(per_sub[s][0]) - roi_mean(per_sub[s][1]) for s in subs])
    n_perm = min(2 ** len(subs), 2048)
    T, cl, pv, _ = permutation_cluster_1samp_test(diffs, n_permutations=n_perm, tail=0, seed=20260513, verbose="ERROR")
    minp = float(pv.min()) if len(pv) else float("nan")
    print(f"  delta-CSD ROI cluster: {len(cl)} clusters, min p={minp:.4f}", flush=True)
    rows = []
    for c, p in zip(cl, pv):
        idx = c[0] if isinstance(c, tuple) else c
        m = np.zeros(len(times), bool); m[idx] = True
        rows.append(dict(t_start=float(times[m].min()), t_end=float(times[m].max()), p_value=float(p)))
    pd.DataFrame(rows).to_csv(NPZ_DIR_AR / "scr_ro_delta_csd_ar_clusters.csv", index=False)
    # quick figure
    fig, ax = plt.subplots(figsize=(11, 5))
    rs = np.array([roi_mean(per_sub[s][0]) for s in subs]); ss = np.array([roi_mean(per_sub[s][1]) for s in subs])
    for arr, c, l in [(rs, "C3", "real (drop-only)"), (ss, "0.5", "silent (drop-only)")]:
        m = arr.mean(0); sem = arr.std(0, ddof=1) / np.sqrt(len(subs))
        ax.plot(times, m, color=c, lw=2, label=l); ax.fill_between(times, m-sem, m+sem, color=c, alpha=0.2)
    ax.axvline(0, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("CSD delta ROI mean"); ax.set_title(f"Delta+CSD ROI (drop-only) min p={minp:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "grandavg_roi_ar.png", dpi=130); plt.close(fig)


# =============================================================================
def main():
    print(f"Phase 2: suite re-run with drop-only -> {OUT_AR}", flush=True)
    t0 = time.time()
    # 0) Copy input npz files so build_subject_epochs finds them in cohort6_ar/
    run_step("copy inputs", _copy_inputs_to_ar)
    # Group A
    run_step("tfr_psd_scr", run_tfr_psd)
    run_step("alpha_hypothesis_scr", run_alpha_hypothesis)
    run_step("fooof_scr", run_fooof)
    run_step("topo_variance_scr", run_topo_variance)
    run_step("topomap_delta_theta_scr", run_topomap_delta_theta)
    run_step("decoding_scr", run_decoding_scr)
    run_step("decoding_loso_scr", run_decoding_loso)
    run_step("scr_alpha_csd", run_scr_alpha_csd)
    run_step("scr_band_csd", run_scr_band_csd)
    # Group B (custom inline)
    run_step("erp_grandavg_ar (custom)", run_erp_grandaverage_ar)
    run_step("delta_csd_ar (custom)", run_delta_csd_ar)
    print(f"\nPHASE 2 SUITE COMPLETE in {(time.time()-t0)/60:.1f} min  -> {OUT_AR}", flush=True)


if __name__ == "__main__":
    main()
