"""Topography of power & variance differences (real SCR - silent EDA) + Branković test.

Tarea 2 QA artefacto-vs-señal, Deliverable 2. Maps, per channel, the SCR-vs-silent
difference of (A) band power (delta..gamma + broadband) and (B) time-domain variance,
then applies the Branković (2026) dissociation test to decide artifact vs signal:

  - If the effect is CENTRAL-PARIETAL (Cz/Pz, C/CP/P chain) -> compatible with a real,
    centrally-generated SCR-related oscillation (SCR-RO).
  - If it is FRONTOPOLAR / EDGE (Fp1/Fp2, temporal/mastoid rim) -> compatible with an
    ocular (frontopolar) or muscular (edge) artifact.

Branković had only 3 midline channels and could not see topography; our 32-ch montage
turns the QA into a concrete spatial dissociation. Ocular check uses Fp1/Fp2 as a
frontopolar proxy (the EOG channels R_EYE/L_EYE are dropped upstream by
attach_montage_and_drop_no_pos); a with-EOG epoch builder is a documented follow-up.

Complements topomap_delta_theta_scr.py (which covers delta+theta only) -- does not modify
it. Reuses build_subject_epochs / compute_psd / channel_band_diff_db.

Outputs (under research_diary/context/05_04/cohort6/qa_artifact_vs_signal/):
  tables/topo_variance_scr_summary.csv   subject,channel,map,diff_db
  tables/topo_dissociation_index.csv     subject,map,idx_central_parietal,idx_frontal_edge,
                                         fp_proxy,corr_fp_template,cp_minus_edge,verdict
  figures/topo_variance_scr_<sub>.png        per-subject band + broadband + variance maps
  figures/topo_variance_scr_grandaverage.png
  figures/topo_variance_scr_overview.png

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.topo_variance_scr
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.topo_variance_scr --subjects sub-27
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.campeones_analysis.multimodal_arousal.erp_scr import OUT, SUBJECTS
from src.campeones_analysis.multimodal_arousal.tfr_psd_scr import build_subject_epochs, compute_psd
from src.campeones_analysis.multimodal_arousal.topomap_delta_theta_scr import channel_band_diff_db

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

QA_DIR = OUT / "qa_artifact_vs_signal"
FIG_DIR = QA_DIR / "figures"
TBL_DIR = QA_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 40.0), "broadband": (1.0, 40.0),
}
MAP_NAMES = list(BANDS.keys()) + ["variance"]
POST_TMIN, POST_TMAX = 0.0, 3.0  # post-onset window (consistent with topomap_delta_theta_scr)

# Branković dissociation regions
CENTRAL_PARIETAL = ["Cz", "Pz", "CP1", "CP2", "C3", "C4", "P3", "P4"]
FRONTAL_EDGE = ["Fp1", "Fp2", "F7", "F8", "FT9", "FT10", "TP9", "TP10", "T7", "T8"]
FRONTOPOLAR = ["Fp1", "Fp2"]


def variance_diff_db(real_ep: mne.Epochs, silent_ep: mne.Epochs) -> np.ndarray:
    """Per-channel time-domain variance diff in dB: 10*log10(var_real / var_silent)."""
    vr = real_ep.get_data(copy=True).var(axis=2).mean(axis=0)     # (n_ch,)
    vs = silent_ep.get_data(copy=True).var(axis=2).mean(axis=0)
    return 10.0 * np.log10((vr + 1e-30) / (vs + 1e-30))


def _region_index(diff_map: np.ndarray, ch_names: list[str], region: list[str]) -> float:
    idxs = [ch_names.index(c) for c in region if c in ch_names]
    if not idxs:
        return float("nan")
    return float(np.nanmean(diff_map[idxs]))


def dissociation_indices(diff_map: np.ndarray, ch_names: list[str]) -> dict:
    """Branković central-parietal vs frontal/edge dissociation + frontopolar/ocular proxy."""
    idx_cp = _region_index(diff_map, ch_names, CENTRAL_PARIETAL)
    idx_edge = _region_index(diff_map, ch_names, FRONTAL_EDGE)
    fp_proxy = _region_index(diff_map, ch_names, FRONTOPOLAR)
    # correlation of the effect map with a frontopolar template (1 at Fp1/Fp2, else 0)
    template = np.array([1.0 if c in FRONTOPOLAR else 0.0 for c in ch_names])
    finite = np.isfinite(diff_map)
    corr = float(np.corrcoef(diff_map[finite], template[finite])[0, 1]) if finite.sum() > 2 else float("nan")
    cp_minus_edge = idx_cp - idx_edge
    # peak channel by absolute effect
    peak_ch = ch_names[int(np.nanargmax(np.abs(diff_map)))] if np.any(finite) else None

    if not np.isfinite(cp_minus_edge):
        verdict = "n/a"
    elif peak_ch in FRONTOPOLAR or (np.isfinite(corr) and corr > 0.5):
        verdict = "frontopolar/ocular (artifact-like)"
    elif cp_minus_edge > 0:
        verdict = "central-parietal (signal-like)"
    elif peak_ch in FRONTAL_EDGE:
        verdict = "frontal/edge (artifact-like)"
    else:
        verdict = "mixed"
    return dict(idx_central_parietal=idx_cp, idx_frontal_edge=idx_edge, fp_proxy=fp_proxy,
                corr_fp_template=corr, cp_minus_edge=cp_minus_edge,
                peak_channel=peak_ch, verdict=verdict)


def _plot_maps(title: str, info: mne.Info, maps: dict[str, np.ndarray], out_png) -> None:
    n = len(maps)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=11)
    for ax, name in zip(axes, maps.keys()):
        data = maps[name]
        vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
        im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False, vlim=(-vmax, vmax),
                                     cmap="RdBu_r", sensors=True, contours=4)
        sub = f" ({BANDS[name][0]:.0f}-{BANDS[name][1]:.0f} Hz)" if name in BANDS else ""
        ax.set_title(f"{name}{sub}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_overview(info: mne.Info, per_sub: dict[str, dict[str, np.ndarray]],
                   ga: dict[str, np.ndarray], out_png) -> None:
    rows = list(per_sub.items()) + [(f"GA (N={len(per_sub)})", ga)]
    ncols = len(MAP_NAMES)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    fig.suptitle("PSD/variance diff real - silent (dB) per channel -- per subject + GA",
                 fontsize=12)
    for r, (label, maps) in enumerate(rows):
        for c, name in enumerate(MAP_NAMES):
            ax = axes[r, c]
            data = maps[name]
            vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
            mne.viz.plot_topomap(data, info, axes=ax, show=False, vlim=(-vmax, vmax),
                                 cmap="RdBu_r", sensors=True, contours=4)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.text(-0.25, 0.5, label, transform=ax.transAxes, fontsize=9,
                        ha="right", va="center", rotation=90)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(subjects: list[str]) -> None:
    print("=" * 78)
    print("topo_variance_scr :: power + variance topography, real (SCR) vs silent-EDA")
    print("=" * 78)

    summary_rows: list[dict] = []
    diss_rows: list[dict] = []
    per_sub_maps: dict[str, dict[str, np.ndarray]] = {}
    info_ref: mne.Info | None = None
    ref_ch_names: list[str] | None = None

    for sub in subjects:
        print(f"\n=== {sub} ===")
        real_ep, silent_ep = build_subject_epochs(sub)
        if real_ep is None or silent_ep is None:
            print("  no epochs -- skipped")
            continue
        real_post = real_ep.copy().crop(POST_TMIN, POST_TMAX)
        silent_post = silent_ep.copy().crop(POST_TMIN, POST_TMAX)
        psd_r, freqs, ch_names = compute_psd(real_post)
        psd_s, _, _ = compute_psd(silent_post)
        print(f"  n_real={len(real_post)} n_silent={len(silent_post)} n_ch={len(ch_names)}")

        maps: dict[str, np.ndarray] = {}
        for band, rng in BANDS.items():
            maps[band] = channel_band_diff_db(psd_r, psd_s, freqs, rng)
        maps["variance"] = variance_diff_db(real_post, silent_post)
        per_sub_maps[sub] = maps

        if info_ref is None:
            info_ref = real_post.info.copy()
            ref_ch_names = list(ch_names)

        for name, data in maps.items():
            for i, ch in enumerate(ch_names):
                summary_rows.append(dict(subject=sub, channel=ch, map=name, diff_db=float(data[i])))
            d = dissociation_indices(data, ch_names)
            diss_rows.append(dict(subject=sub, map=name, **d))

        _plot_maps(f"{sub} -- diff real - silent (dB) per channel  [{POST_TMIN:.0f}-{POST_TMAX:.0f} s]",
                   info_ref, maps, FIG_DIR / f"topo_variance_scr_{sub}.png")
        print(f"  -> topo_variance_scr_{sub}.png")

    if not per_sub_maps:
        print("No subjects processed.")
        return

    # grand average per map: mean per-channel across subjects
    ga_maps: dict[str, np.ndarray] = {}
    for name in MAP_NAMES:
        ga_maps[name] = np.mean(np.array([m[name] for m in per_sub_maps.values()]), axis=0)
        d = dissociation_indices(ga_maps[name], ref_ch_names)
        diss_rows.append(dict(subject="GA", map=name, **d))

    _plot_maps(f"Grand average (N={len(per_sub_maps)}) -- diff real - silent (dB) per channel",
               info_ref, ga_maps, FIG_DIR / "topo_variance_scr_grandaverage.png")
    _plot_overview(info_ref, per_sub_maps, ga_maps, FIG_DIR / "topo_variance_scr_overview.png")
    print("\nGrand-average + overview figures saved.")

    pd.DataFrame(summary_rows).to_csv(TBL_DIR / "topo_variance_scr_summary.csv", index=False)
    pd.DataFrame(diss_rows).to_csv(TBL_DIR / "topo_dissociation_index.csv", index=False)
    print(f"Tables -> {TBL_DIR}")

    # quick console verdict for the broadband + delta GA maps (the QA crux)
    print("\nBranković dissociation (GA):")
    for name in ("broadband", "delta", "variance"):
        d = dissociation_indices(ga_maps[name], ref_ch_names)
        print(f"  {name:10s}: CP={d['idx_central_parietal']:+.2f} "
              f"edge={d['idx_frontal_edge']:+.2f} Fp={d['fp_proxy']:+.2f} "
              f"peak={d['peak_channel']} -> {d['verdict']}")


def _parse_args() -> list[str]:
    ap = argparse.ArgumentParser(description="Power/variance topography + dissociation test.")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Subset of subjects (e.g. sub-27). Default: full cohort.")
    args = ap.parse_args()
    return args.subjects if args.subjects else list(SUBJECTS)


if __name__ == "__main__":
    main(_parse_args())
