"""Tarea 4 (LOVO variant) -- SPoC decoding leave-one-affective-video-out.

Same SPoC + Ridge pipeline as ``spoc_decoding.py`` (delta band 1-4 Hz, K=4,
log=True, reg='oas', rank='full', transform_into='average_power'), but the
cross-validation scheme changes from Leave-One-Run-Out to **Leave-One-Affective-
Video-Out**:

  - 14 folds per subject (one per ``stim_id`` in 1..14)
  - When a video is held out, BOTH presentations (acq-a and acq-b) of that
    ``stim_id`` go to the test set; the other 13 videos (26 instances total)
    constitute the training set.
  - The inner alpha grid search uses LeaveOneGroupOut over ``run_label`` of the
    training pool (same outer-inner separation principle as the LORO script,
    keeping the inner CV semantically distinct from the outer CV).
  - Tests generalization to **novel affective content**, not just to a held-out
    temporal segment. Stricter than LORO.

Mapping of epochs to videos: each epoch is centered (in Y-coordinates) at
``t_y``. We load the ``merged_events.tsv`` of the run that produced the epoch
and assign the epoch to the affective video whose ``[onset, onset+duration)``
window contains ``t_y``. Epochs whose ``t_y`` falls in fixation / calm / gap /
luminance are dropped from LOVO -- they are not part of any affective video.

Scope: affective only (``condition == 'affective'``, stim_id 1..14). Luminance
videos and baselines are not part of the LOVO universe in this first pass.

Cohort: sub-23, sub-24, sub-33 (sub-27 excluded -- see diary 05_2 banner).

Outputs:
  figures/spoc/Y4_spoc_lovo_<sub>.png             per-subject (scatter / topomap / bars / time-series)
  figures/spoc/Y4_spoc_lovo_per_video_<sub>.png   per-subject Pearson r bar chart, 14 bars
  figures/spoc/Y4_spoc_lovo_summary.png           cross-subject summary
  y_candidates/decoding_spoc_lovo_per_fold.csv    per-fold metrics (one row per (sub, video))
  y_candidates/decoding_spoc_lovo_per_subject.csv per-subject means + SDs
  y_candidates/spoc_lovo_patterns.npz             per-subject patterns_[0,:] + ch_names

Run:
  micromamba run -n campeones python -m src.campeones_analysis.multimodal_arousal.spoc_decoding_lovo
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.campeones_analysis.multimodal_arousal.decoding_y1_3models import (
    LAG_S,
    SUBSAMPLE_S,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    NPZ_DIR,
    OUT,
    SUBJECTS,
)
from src.campeones_analysis.multimodal_arousal.spoc_decoding import (
    SPOC_BAND,
    SPOC_BAND_NAME,
    SPOC_N_COMPONENTS,
    _build_topo_info,
    build_subject_dataset_banded,
    fit_eval_spoc_fold,
    refit_for_patterns,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
MERGED_EVENTS_DIR = REPO / "data" / "derivatives" / "merged_events"
AFFECTIVE_STIM_IDS: list[int] = list(range(1, 15))  # 1..14

FIG_DIR: Path = OUT / "figures" / "spoc"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Epoch -> video mapping
# -----------------------------------------------------------------------------
def load_run_events(sub: str, run_label: str) -> pd.DataFrame | None:
    """Find and load the merged_events.tsv for (sub, run_label).

    ``run_label`` looks like ``"task-01_acq-a_run-002"``. The filename pattern
    is ``<sub>_ses-vr_<run_label>_desc-merged_events.tsv``.
    """
    pattern = f"{sub}_ses-vr_{run_label}_desc-merged_events.tsv"
    eeg_dir = MERGED_EVENTS_DIR / sub / "ses-vr" / "eeg"
    candidates = list(eeg_dir.glob(pattern))
    if not candidates:
        return None
    try:
        return pd.read_csv(candidates[0], sep="\t")
    except Exception:
        return None


def assign_video_to_epoch(t_y: float, events_df: pd.DataFrame) -> int | None:
    """Return the stim_id of the affective video active at t_y (or None)."""
    if events_df is None or events_df.empty:
        return None
    aff = events_df[events_df["condition"].astype(str) == "affective"]
    for _, row in aff.iterrows():
        try:
            onset = float(row["onset"])
            duration = float(row["duration"])
            stim_id = int(row["stim_id"])
        except (ValueError, TypeError):
            continue
        if onset <= t_y < onset + duration and stim_id in AFFECTIVE_STIM_IDS:
            return stim_id
    return None


def build_lovo_groups(
    sub: str, centers: np.ndarray, groups_run: np.ndarray
) -> np.ndarray:
    """For each epoch, map t_y -> stim_id of active affective video (or -1)."""
    video_ids = np.full(len(centers), -1, dtype=int)
    cache: dict[str, pd.DataFrame | None] = {}
    for run_label in np.unique(groups_run):
        cache[run_label] = load_run_events(sub, str(run_label))
    for i, (t_y, rl) in enumerate(zip(centers, groups_run)):
        ev = cache.get(str(rl))
        if ev is None:
            continue
        sid = assign_video_to_epoch(float(t_y), ev)
        if sid is not None:
            video_ids[i] = sid
    return video_ids


# -----------------------------------------------------------------------------
# Per-subject LOVO loop
# -----------------------------------------------------------------------------
def lovo_subject(
    sub: str, ds: dict
) -> tuple[list[dict], dict[int, dict], np.ndarray, np.ndarray]:
    """Run LOVO on a single subject. Return rows, per-video predictions, video_ids,
    and the boolean mask of which epochs in the original dataset survived."""
    X_all = ds["X"]
    y_all = ds["y"]
    groups_run = ds["groups"]
    centers = ds["centers"]

    video_ids_all = build_lovo_groups(sub, centers, groups_run)
    mask = video_ids_all >= 1
    n_kept = int(mask.sum())
    n_total = len(video_ids_all)
    print(
        f"  epoch -> video mapping: {n_kept}/{n_total} epochs assigned to an affective video "
        f"({100.0 * n_kept / max(n_total, 1):.1f}%)"
    )
    if n_kept == 0:
        return [], {}, video_ids_all, mask

    X = X_all[mask]
    y = y_all[mask]
    video_ids = video_ids_all[mask]
    runs_in_train_pool = groups_run[mask]

    unique_videos = sorted(np.unique(video_ids).tolist())
    print(f"  unique videos present: {len(unique_videos)} -> {unique_videos}")

    rows: list[dict] = []
    per_video_preds: dict[int, dict] = {}
    for vid_held in unique_videos:
        test_mask = video_ids == vid_held
        train_mask = ~test_mask
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_test < 5 or n_train < 20:
            print(f"  video {vid_held}: too few epochs ({n_test},{n_train}) -- skipping")
            continue
        # Inner alpha grid CV uses run_label of the training pool (LOGO over runs)
        groups_inner = runs_in_train_pool[train_mask]
        print(f"  fold video={vid_held:02d}  n_train={n_train}  n_test={n_test} ...")
        r = fit_eval_spoc_fold(
            X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
            groups_inner,
        )
        per_video_preds[vid_held] = {
            "y_test": y[test_mask],
            "y_pred": r.pop("y_pred"),
            "run_labels": runs_in_train_pool[test_mask],
        }
        rows.append(dict(
            subject=sub,
            model=f"spoc_lovo_{SPOC_BAND_NAME}",
            video_held_out=int(vid_held),
            best_alpha=r["best_alpha"],
            pearson_r=r["pearson_r"],
            spearman_rho=r["spearman_rho"],
            r2=r["r2"],
            rmse=r["rmse"],
            n_train=n_train,
            n_test=n_test,
        ))
        print(
            f"    r={r['pearson_r']:+.3f}  rho={r['spearman_rho']:+.3f}  "
            f"R2={r['r2']:+.3f}  rmse={r['rmse']:.4f}  alpha={r['best_alpha']:.1f}"
        )

    return rows, per_video_preds, video_ids_all, mask


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_subject(
    sub: str,
    rows: list[dict],
    per_video_preds: dict[int, dict],
    spoc_fitted,
    ch_names: list[str],
    sfreq: float,
    out_png: Path,
) -> None:
    df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35, height_ratios=[1.0, 0.85])

    # Panel 1: scatter y_true vs y_pred across all videos, color-coded by video
    ax1 = fig.add_subplot(gs[0, 0])
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    videos_sorted = sorted(per_video_preds.keys())
    cmap = plt.get_cmap("tab20")
    for i, vid in enumerate(videos_sorted):
        yt = per_video_preds[vid]["y_test"]
        yp = per_video_preds[vid]["y_pred"]
        ax1.scatter(yt, yp, s=10, alpha=0.45, color=cmap(i % 20), label=f"v{vid:02d}")
        y_true_all.append(yt)
        y_pred_all.append(yp)
    if y_true_all:
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        lo = float(min(y_true_cat.min(), y_pred_cat.min()))
        hi = float(max(y_true_cat.max(), y_pred_cat.max()))
        ax1.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--")
        r_cf = float(pearsonr(y_true_cat, y_pred_cat).statistic) if np.std(y_pred_cat) > 1e-12 else 0.0
        r2_cf = float(np.corrcoef(y_true_cat, y_pred_cat)[0, 1] ** 2) if np.std(y_pred_cat) > 1e-12 else 0.0
    else:
        r_cf, r2_cf = 0.0, 0.0
    ax1.set_xlabel("y_true (Y1s)")
    ax1.set_ylabel("y_pred")
    ax1.set_title(f"Cross-video scatter\nr = {r_cf:+.3f}   R²(approx) = {r2_cf:+.3f}", fontsize=10)
    ax1.legend(fontsize=6, loc="best", ncol=2)

    # Panel 2: topomap of pattern of component 0
    ax2 = fig.add_subplot(gs[0, 1])
    info = _build_topo_info(ch_names, sfreq)
    pattern0 = spoc_fitted.patterns_[0, :]
    has_all_pos = all(
        not (np.isnan(ch_info["loc"][:3]).any() or np.allclose(ch_info["loc"][:3], 0.0))
        for ch_info in info["chs"]
    )
    if has_all_pos and len(info.ch_names) >= 3:
        vmax = float(np.max(np.abs(pattern0)))
        im, _ = mne.viz.plot_topomap(
            pattern0, info, axes=ax2, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), sensors=True, contours=4,
        )
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f"Spatial pattern -- {SPOC_BAND_NAME} comp 1", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "topomap unavailable\n(montage mismatch)",
                 ha="center", va="center", fontsize=9, transform=ax2.transAxes)
        ax2.set_axis_off()

    # Panel 3: bars of [Pearson r, R^2, RMSE] mean +- SD across videos
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ["pearson_r", "r2", "rmse"]
    means = [float(df[m].mean()) for m in metrics]
    stds = [float(df[m].std(ddof=1)) for m in metrics]
    x = np.arange(len(metrics))
    ax3.bar(x, means, yerr=stds, color=["C0", "C2", "C3"],
            alpha=0.85, edgecolor="black", capsize=4)
    for xi, mn, sd, m in zip(x, means, stds, metrics):
        y_text = mn + (sd if sd > 0 else 0.0) + 0.02 * (1 if mn >= 0 else -1)
        ax3.text(xi, y_text, f"{mn:.3f}±{sd:.3f}", ha="center", fontsize=9)
        vals = df[m].to_numpy()
        ax3.scatter(np.full(len(vals), xi), vals, color="black", s=14, alpha=0.55, zorder=3)
    ax3.axhline(0, color="0.4", lw=0.8, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in metrics])
    ax3.set_title("Per-video metrics", fontsize=10)

    # Panel 4 (full-width row 1): time-series y_true vs y_pred for the best video
    ax_ts = fig.add_subplot(gs[1, :])
    if per_video_preds:
        best_vid = int(df.loc[df["pearson_r"].idxmax(), "video_held_out"])
        best_r = float(df.loc[df["pearson_r"].idxmax(), "pearson_r"])
        yt = per_video_preds[best_vid]["y_test"]
        yp = per_video_preds[best_vid]["y_pred"]
        run_labels = per_video_preds[best_vid]["run_labels"]
        # x-axis: epoch index (LOVO concatenates both acq-a and acq-b of the held-out video)
        idx = np.arange(len(yt))
        ax_ts.plot(idx, yt, color="0.35", lw=1.4, label="y_true (Y1s)")
        ax_ts.plot(idx, yp, color="C0", lw=1.6, label=f"y_pred (SPoC {SPOC_BAND_NAME})")
        # Vertical line between acq-a and acq-b (if both present)
        change_points = np.where(run_labels[:-1] != run_labels[1:])[0]
        for cp in change_points:
            ax_ts.axvline(cp + 0.5, color="0.6", lw=0.7, ls=":", alpha=0.7)
        ax_ts.set_xlabel(f"epoch index within held-out video {best_vid:02d} (step = {SUBSAMPLE_S:.0f} s)")
        ax_ts.set_ylabel("Y1s (smoothed SMNA-AUC)")
        ax_ts.set_title(
            f"Best held-out video = {best_vid:02d}   |   Pearson r = {best_r:+.3f}   "
            f"|   dashed verticals = acq-a / acq-b boundary",
            fontsize=10,
        )
        ax_ts.legend(fontsize=9, loc="upper right")
        ax_ts.grid(alpha=0.25)
    else:
        ax_ts.text(0.5, 0.5, "no folds available", ha="center", va="center",
                   transform=ax_ts.transAxes)
        ax_ts.set_axis_off()

    fig.suptitle(
        f"{sub}  --  SPoC LOVO ({SPOC_BAND_NAME} {SPOC_BAND[0]:.0f}-{SPOC_BAND[1]:.0f} Hz)  "
        f"K={SPOC_N_COMPONENTS}, 14 affective videos, EEG(t)->Y(t+{LAG_S:.0f}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_per_video_bars(sub: str, rows: list[dict], out_png: Path) -> None:
    """One bar per affective video held-out: Pearson r, color-coded by sign."""
    df = pd.DataFrame(rows).sort_values("video_held_out").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    vals = df["pearson_r"].to_numpy()
    colors = ["#2b7a3a" if v >= 0 else "#a83232" for v in vals]
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black")
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.3f}", ha="center", fontsize=8)
    mean_r = float(vals.mean())
    ax.axhline(mean_r, color="black", lw=1.5, ls="-")
    ax.text(len(df) - 0.5, mean_r, f" mean = {mean_r:+.3f}", va="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"v{int(v):02d}" for v in df["video_held_out"]])
    ax.set_xlabel("held-out affective video (stim_id)")
    ax.set_ylabel("Pearson r")
    ax.set_title(
        f"{sub}  --  SPoC LOVO Pearson r per held-out video  "
        f"({SPOC_BAND_NAME} {SPOC_BAND[0]:.0f}-{SPOC_BAND[1]:.0f} Hz)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.25)
    _ = bars
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_summary(all_rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(all_rows)
    subs = sorted(df["subject"].unique())
    metrics = ("pearson_r", "spearman_rho", "r2", "rmse")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        means: list[float] = []
        stds: list[float] = []
        for sub in subs:
            sub_df = df[df["subject"] == sub]
            means.append(float(sub_df[metric].mean()))
            stds.append(float(sub_df[metric].std(ddof=1)))
        x = np.arange(len(subs))
        colors = [f"C{i}" for i in range(len(subs))]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, edgecolor="black", capsize=4)
        for xi, mn, sd in zip(x, means, stds):
            y_text = mn + (sd if sd > 0 else 0.0) + 0.02 * (1 if mn >= 0 else -1)
            ax.text(xi, y_text, f"{mn:.3f}", ha="center", fontsize=9)
        ga = float(df[metric].mean())
        ax.axhline(ga, color="black", lw=1.5, ls="-")
        ax.text(len(subs) - 0.5, ga, f" GA={ga:.3f}", va="center", fontsize=9, fontweight="bold")
        if metric in ("pearson_r", "spearman_rho", "r2"):
            ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(subs)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
    fig.suptitle(
        f"SPoC LOVO ({SPOC_BAND_NAME} {SPOC_BAND[0]:.0f}-{SPOC_BAND[1]:.0f} Hz) -- "
        f"per-subject means ± SD across 14 video folds (grand average as black line)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(
        f"spoc_decoding_lovo :: band={SPOC_BAND} ({SPOC_BAND_NAME}), "
        f"K={SPOC_N_COMPONENTS}, leave-one-affective-video-out per subject"
    )
    print(f"  affective stim_ids: {AFFECTIVE_STIM_IDS}")
    print(f"  output -> {FIG_DIR}")
    print("=" * 78)

    all_rows: list[dict] = []
    patterns_per_sub: dict[str, np.ndarray] = {}
    ch_names_per_sub: dict[str, list[str]] = {}

    for sub in SUBJECTS:
        print(f"\n=== {sub} ===")
        ds = build_subject_dataset_banded(sub, SPOC_BAND, SPOC_BAND_NAME)
        if ds is None:
            continue
        print(f"  X shape={ds['X'].shape}  y mean={ds['y'].mean():.4f}")

        rows, per_video_preds, video_ids_all, mask = lovo_subject(sub, ds)
        all_rows.extend(rows)

        if rows:
            df_sub = pd.DataFrame(rows)
            alpha_for_refit = float(np.median(df_sub["best_alpha"].to_numpy()))
            print(f"  refit on full LOVO-eligible data with alpha={alpha_for_refit:.2f}")
            spoc_fitted = refit_for_patterns(ds["X"][mask], ds["y"][mask], alpha_for_refit)
            patterns_per_sub[sub] = spoc_fitted.patterns_[0, :].astype(float)
            ch_names_per_sub[sub] = list(ds["ch_names"])

            out_png = FIG_DIR / f"Y4_spoc_lovo_{sub}.png"
            plot_subject(sub, rows, per_video_preds, spoc_fitted,
                         ds["ch_names"], ds["sfreq"], out_png)
            print(f"  -> {out_png.name}")

            out_pv = FIG_DIR / f"Y4_spoc_lovo_per_video_{sub}.png"
            plot_per_video_bars(sub, rows, out_pv)
            print(f"  -> {out_pv.name}")

    if all_rows:
        out_summary = FIG_DIR / "Y4_spoc_lovo_summary.png"
        plot_summary(all_rows, out_summary)
        print(f"\nSummary figure -> {out_summary.name}")

        df_fold = pd.DataFrame(all_rows)
        out_csv_fold = NPZ_DIR / "decoding_spoc_lovo_per_fold.csv"
        df_fold.to_csv(out_csv_fold, index=False)
        print(f"Per-fold CSV   -> {out_csv_fold.name}  ({len(df_fold)} rows)")

        rows_subj: list[dict] = []
        for sub in df_fold["subject"].unique():
            sub_df = df_fold[df_fold["subject"] == sub]
            rows_subj.append(dict(
                subject=sub,
                model=f"spoc_lovo_{SPOC_BAND_NAME}",
                n_videos=int(len(sub_df)),
                mean_pearson_r=float(sub_df["pearson_r"].mean()),
                sd_pearson_r=float(sub_df["pearson_r"].std(ddof=1)),
                mean_spearman_rho=float(sub_df["spearman_rho"].mean()),
                mean_r2=float(sub_df["r2"].mean()),
                sd_r2=float(sub_df["r2"].std(ddof=1)),
                mean_rmse=float(sub_df["rmse"].mean()),
                median_alpha=float(np.median(sub_df["best_alpha"].to_numpy())),
            ))
        out_csv_subj = NPZ_DIR / "decoding_spoc_lovo_per_subject.csv"
        pd.DataFrame(rows_subj).to_csv(out_csv_subj, index=False)
        print(f"Per-subject CSV -> {out_csv_subj.name}")

        if patterns_per_sub:
            out_patt = NPZ_DIR / "spoc_lovo_patterns.npz"
            np.savez(
                out_patt,
                band=np.array(SPOC_BAND),
                band_name=SPOC_BAND_NAME,
                **{f"{sub}__pattern0": patterns_per_sub[sub] for sub in patterns_per_sub},
                **{f"{sub}__ch_names": np.array(ch_names_per_sub[sub], dtype=object)
                   for sub in patterns_per_sub},
            )
            print(f"Patterns NPZ    -> {out_patt.name}")


if __name__ == "__main__":
    main()
