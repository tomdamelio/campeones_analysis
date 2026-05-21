"""Tareas 1+2+3 — Lag sweep + permutation test + decoder topomap.

Tarea 1 — lag sweep sobre LAG_GRID = {-4, -2, 0, +2, +4, +6} s. Identifica lag*
          óptimo per sujeto (argmax de la media de Pearson r across LORO folds).
Tarea 2 — permutation test (N=100) en el lag*. Shuffle de y DENTRO de cada run
          (preserva autocorrelación temporal local, rompe asociación EEG↔Y).
Tarea 3 — topomap de pesos del decoder spectral (CONDICIONAL a Tarea 1 dando
          max mean-r > 0 para el sujeto). Refit del Pipeline (StandardScaler →
          PCA → Ridge) en TODO el dataset al lag*, back-projection de los pesos
          del Ridge al espacio (canales × bandas), 1 topomap por banda.

CV: Leave-One-Run-Out per sujeto. Inner LOGO sobre los 6 runs de training para
grid α (sólo en Tarea 1; en Tareas 2 y 3 se usa α = moda del best_alpha de los
folds en Tarea 1 para ahorrar cómputo).

Modelo único: spectral (PSD log-bandpower × canales → StandardScaler → PCA(50)
→ Ridge). Se reutiliza el pipeline byte-idéntico definido en
`decoding_y1_3models.fit_eval_spectral` para que los resultados sean comparables
con la corrida con lag fijo = 2 s.

TODO (futuro): Haufe transform (Haufe et al. 2014, NeuroImage) para convertir
los pesos del decoder (backward filter) en un patrón espacial interpretable
como forward model. Multiplicar `coef_features` por `cov(X_features)` antes del
reshape. No se implementa en este pase porque la interpretación de magnitudes
relativas en el espacio de features de decoder ya es informativa para detectar
los electrodos/bandas que el modelo está usando.

CLI:
    --task sweep         # Tarea 1
    --task permutation   # Tarea 2 (requiere sweep CSV)
    --task topomap       # Tarea 3 (requiere sweep CSV)
    --task all           # las 3 en orden (default)

Outputs:
    NPZ_DIR/lag_sweep_per_fold.csv
    NPZ_DIR/lag_sweep_permutation.csv
    OUT/figures/lag_sweep/Y3_lag_sweep_summary.png
    OUT/figures/lag_sweep/Y3_permutation_distributions.png
    OUT/figures/lag_sweep/Y3_topomap_decoder_weights_{sub}.png
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.campeones_analysis.multimodal_arousal.decoding_scr_3models import BANDS
from src.campeones_analysis.multimodal_arousal.decoding_y1_3models import (
    EEG_FS_TARGET,
    EEG_WIN_S,
    PCA_SPECTRAL,
    RNG_SEED,
    SUBSAMPLE_S,
    compute_band_power_features_local,
    fit_eval_spectral,
)
from src.campeones_analysis.multimodal_arousal.erp_scr import (
    NPZ_DIR,
    OUT,
    SUBJECTS,
    attach_montage_and_drop_no_pos,
    run_label,
    runs_for,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

LAG_GRID: list[float] = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
N_PERMUTATIONS: int = 100
FIG_DIR: Path = OUT / "figures" / "lag_sweep"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset construction (lag-parametric)
# -----------------------------------------------------------------------------
def build_subject_dataset(sub: str, lag_s: float) -> dict | None:
    """Build (X, y, groups, ch_names, sfreq) for a subject at an arbitrary lag.

    Adapted from `decoding_y1_3models.build_subject_dataset` with the hardcoded
    `LAG_S` replaced by `lag_s`. EEG window is centered at (t_y - lag_s):
    `lag_s > 0` → EEG precedes Y (causal); `lag_s < 0` → Y precedes EEG
    (anti-causal); `lag_s = 0` → synchronous.

    Per-run boundary handling:
      - `lag_s > 0`: drop the first subsampled sample (window center would fall
        before the run start).
      - `lag_s < 0`: drop the last subsampled sample (window center would fall
        after the run end).
      - `lag_s == 0`: keep all.

    The strict window-validity check downstream already guards both cases; the
    edge-sample drop just avoids guaranteed-skipped iterations.
    """
    cont_npz = NPZ_DIR / f"{sub}_y_candidates.npz"
    if not cont_npz.exists():
        print(f"  missing {cont_npz.name}; skipping")
        return None
    yc = np.load(cont_npz, allow_pickle=True)
    runs_in_npz = [str(r) for r in yc["runs"]]

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    g_list: list[str] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None

    eeg_win_samples = int(round(EEG_WIN_S * EEG_FS_TARGET))
    half_win_s = EEG_WIN_S / 2.0

    for vhdr in runs_for(sub):
        label = run_label(vhdr)
        if label not in runs_in_npz:
            continue
        if f"{label}__centers" not in yc.files or f"{label}__y1s" not in yc.files:
            continue
        centers = np.asarray(yc[f"{label}__centers"], dtype=float)
        y1s = np.asarray(yc[f"{label}__y1s"], dtype=float)
        valid = np.isfinite(y1s)
        centers = centers[valid]
        y1s = y1s[valid]

        if len(centers) < 2:
            continue
        step = int(round(SUBSAMPLE_S / (centers[1] - centers[0])))
        if step < 1:
            step = 1
        idx = np.arange(0, len(centers), step)
        if lag_s > 0:
            idx = idx[1:]
        elif lag_s < 0:
            idx = idx[:-1]
        if len(idx) == 0:
            continue
        centers_sel = centers[idx]
        y_sel = y1s[idx]

        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            raw.pick("eeg")
            attach_montage_and_drop_no_pos(raw)
            raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
            raw.resample(EEG_FS_TARGET, verbose="ERROR")
            if ch_names is None:
                ch_names = list(raw.ch_names)
                sfreq = float(raw.info["sfreq"])
            duration = float(raw.times[-1])

            data_full = raw.get_data()  # (n_channels, n_samples)
            n_total_samples = data_full.shape[1]
            n_kept = 0
            for t_y, y_val in zip(centers_sel, y_sel):
                t_center_eeg = t_y - lag_s
                t_start = t_center_eeg - half_win_s
                t_end = t_center_eeg + half_win_s
                if t_start < 0 or t_end > duration:
                    continue
                s_start = int(round(t_start * sfreq))
                s_end = s_start + eeg_win_samples
                if s_end > n_total_samples:
                    continue
                X_list.append(data_full[:, s_start:s_end])
                y_list.append(float(y_val))
                g_list.append(label)
                n_kept += 1
            print(f"  {label} (lag={lag_s:+.1f}s): kept {n_kept}/{len(centers_sel)} epochs")
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")

    if not X_list:
        return None
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=float)
    groups = np.array(g_list)
    return dict(X=X, y=y, groups=groups, ch_names=ch_names, sfreq=sfreq)


# -----------------------------------------------------------------------------
# Tarea 1 — lag sweep
# -----------------------------------------------------------------------------
def task_lag_sweep(out_csv: Path, out_png: Path) -> pd.DataFrame:
    """Sweep LAG_GRID × SUBJECTS × LORO folds. Spectral model only.

    Writes per-fold CSV and a 4-panel summary figure (3 subs + grand average).
    """
    rows: list[dict] = []
    for sub in SUBJECTS:
        print(f"\n[sweep] subject {sub}")
        for lag_s in LAG_GRID:
            print(f"  lag {lag_s:+.1f}s — building dataset")
            ds = build_subject_dataset(sub, lag_s)
            if ds is None:
                print("    no data; skipping")
                continue
            X_spec = compute_band_power_features_local(ds["X"], ds["sfreq"])
            unique_runs = np.unique(ds["groups"])
            for run_held in unique_runs:
                test_mask = ds["groups"] == run_held
                train_mask = ~test_mask
                if test_mask.sum() < 2 or train_mask.sum() < 2:
                    continue
                X_tr, X_te = X_spec[train_mask], X_spec[test_mask]
                y_tr, y_te = ds["y"][train_mask], ds["y"][test_mask]
                g_tr = ds["groups"][train_mask]
                res = fit_eval_spectral(X_tr, y_tr, X_te, y_te, g_tr)
                rows.append(
                    dict(
                        subject=sub,
                        lag_s=float(lag_s),
                        run_held_out=str(run_held),
                        n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()),
                        best_alpha=float(res["best_alpha"]),
                        pearson_r=float(res["pearson_r"]),
                        spearman_rho=float(res["spearman_rho"]),
                        r2=float(res["r2"]),
                        rmse=float(res["rmse"]),
                    )
                )
                print(
                    f"    fold {run_held}: r={res['pearson_r']:+.3f}, "
                    f"ρ={res['spearman_rho']:+.3f}, α={res['best_alpha']:g}"
                )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[sweep] wrote {out_csv} ({len(df)} rows)")
    if len(df) > 0:
        _plot_sweep_summary(df, out_png)
    return df


def _plot_sweep_summary(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for ax, sub in zip(flat_axes[:3], SUBJECTS):
        d = df[df["subject"] == sub]
        if len(d) == 0:
            ax.text(0.5, 0.5, f"{sub}: no data", ha="center", transform=ax.transAxes)
            ax.set_title(sub)
            continue
        agg = d.groupby("lag_s")["pearson_r"].agg(["mean", "std"]).reset_index().sort_values("lag_s")
        ax.errorbar(
            agg["lag_s"], agg["mean"], yerr=agg["std"], marker="o", capsize=3, color="C0"
        )
        lag_star = float(agg.loc[agg["mean"].idxmax(), "lag_s"])
        max_r = float(agg["mean"].max())
        ax.axvline(
            lag_star, color="red", linestyle=":", label=f"lag*={lag_star:+.1f}s, r={max_r:+.3f}"
        )
        ax.axhline(0, color="k", linestyle="--", alpha=0.4)
        ax.set_title(sub)
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=8, loc="best")

    # Grand average: mean across subjects of per-subject mean-r at each lag.
    sub_means = df.groupby(["subject", "lag_s"])["pearson_r"].mean().reset_index()
    ga = sub_means.groupby("lag_s")["pearson_r"].agg(["mean", "std"]).reset_index().sort_values("lag_s")
    ax = flat_axes[3]
    if len(ga) > 0:
        ax.errorbar(
            ga["lag_s"], ga["mean"], yerr=ga["std"], marker="s", capsize=3, color="purple"
        )
        lag_star_ga = float(ga.loc[ga["mean"].idxmax(), "lag_s"])
        ax.axvline(lag_star_ga, color="red", linestyle=":", label=f"lag*={lag_star_ga:+.1f}s")
        ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="k", linestyle="--", alpha=0.4)
    ax.set_title("Grand average (SD across subjects)")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("Pearson r")

    fig.suptitle("Y3 lag sweep — spectral decoder, LORO-CV")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[sweep] wrote {out_png}")


# -----------------------------------------------------------------------------
# Tarea 2 — permutation test at lag*
# -----------------------------------------------------------------------------
def task_permutation(sweep_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Permutation test (N=100) at lag* per subject.

    Shuffle de y DENTRO de cada run (preserva autocorrelación local, rompe
    asociación causal EEG↔Y). Para cada permutation: LORO completo con α fijo
    (moda del best_alpha de Tarea 1 en lag*) — no inner grid-search, lo que
    reduce drásticamente el costo computacional.

    Two-sided lectura de p (right-tail aquí porque la hipótesis es r > 0):
    p = (#perms con mean_r_perm >= mean_r_real + 1) / (N + 1)
    (Wood/Phipson correction; min p ≈ 1/(N+1) ≈ 0.0099 con N=100.)
    """
    df_sweep = pd.read_csv(sweep_csv)
    rows: list[dict] = []
    fig_panels: list[tuple] = []  # (sub, lag*, real_r, r_null, p)

    for sub in SUBJECTS:
        d_sub = df_sweep[df_sweep["subject"] == sub]
        if len(d_sub) == 0:
            print(f"  {sub}: no rows in sweep CSV; skipping")
            continue
        lag_means = d_sub.groupby("lag_s")["pearson_r"].mean()
        lag_optimal = float(lag_means.idxmax())
        real_r = float(lag_means.max())
        d_sub_lag = d_sub[d_sub["lag_s"] == lag_optimal]
        if d_sub_lag.empty:
            print(f"  {sub}: no folds at lag*={lag_optimal:+.1f}s; skipping")
            continue
        fixed_alpha = float(d_sub_lag["best_alpha"].mode().iloc[0])

        print(
            f"\n[perm] {sub}: lag*={lag_optimal:+.1f}s, real_r={real_r:+.4f}, "
            f"α={fixed_alpha:g}"
        )

        ds = build_subject_dataset(sub, lag_optimal)
        if ds is None:
            print(f"  {sub}: dataset rebuild failed at lag*; skipping")
            continue
        X_spec = compute_band_power_features_local(ds["X"], ds["sfreq"])
        unique_runs = np.unique(ds["groups"])

        rng = np.random.default_rng(RNG_SEED + SUBJECTS.index(sub))
        r_null: list[float] = []
        for perm_idx in range(N_PERMUTATIONS):
            y_shuf = ds["y"].copy()
            for run in unique_runs:
                mask = ds["groups"] == run
                y_shuf[mask] = rng.permutation(y_shuf[mask])

            fold_rs: list[float] = []
            for run_held in unique_runs:
                test_mask = ds["groups"] == run_held
                train_mask = ~test_mask
                if test_mask.sum() < 2 or train_mask.sum() < 2:
                    continue
                X_tr, X_te = X_spec[train_mask], X_spec[test_mask]
                y_tr, y_te = y_shuf[train_mask], y_shuf[test_mask]
                n_pca = min(PCA_SPECTRAL, X_tr.shape[0] - 1, X_tr.shape[1])
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=n_pca, random_state=RNG_SEED)),
                        ("ridge", Ridge(alpha=fixed_alpha)),
                    ]
                )
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                if np.std(y_pred) > 1e-12 and np.std(y_te) > 1e-12:
                    fold_rs.append(float(pearsonr(y_te, y_pred).statistic))
                else:
                    fold_rs.append(0.0)
            r_null.append(float(np.mean(fold_rs)) if fold_rs else 0.0)
            if (perm_idx + 1) % 10 == 0:
                print(
                    f"  perm {perm_idx + 1}/{N_PERMUTATIONS}: "
                    f"r_null mean so far = {np.mean(r_null):+.4f}"
                )

        r_null_arr = np.array(r_null)
        p_value = float((np.sum(r_null_arr >= real_r) + 1) / (N_PERMUTATIONS + 1))
        rows.append(
            dict(
                subject=sub,
                lag_optimal=lag_optimal,
                real_pearson_r=real_r,
                fixed_alpha=fixed_alpha,
                p_value=p_value,
                n_permutations=N_PERMUTATIONS,
                significant_05=bool(p_value < 0.05),
                trend_10=bool(0.05 <= p_value < 0.10),
                r_null_mean=float(r_null_arr.mean()),
                r_null_sd=float(r_null_arr.std(ddof=1)) if len(r_null_arr) > 1 else 0.0,
            )
        )
        fig_panels.append((sub, lag_optimal, real_r, r_null_arr, p_value))
        print(f"  {sub}: p={p_value:.4f}, null mean={r_null_arr.mean():+.4f}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[perm] wrote {out_csv}")

    if fig_panels:
        _plot_permutation_distributions(
            fig_panels, FIG_DIR / "Y3_permutation_distributions.png"
        )
    return df


def _plot_permutation_distributions(panels: list[tuple], out_png: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (sub, lag_opt, real_r, r_null, p_val) in zip(axes[0], panels):
        ax.hist(r_null, bins=20, color="gray", alpha=0.7, edgecolor="black")
        ax.axvline(
            real_r, color="red", linestyle="-", linewidth=2, label=f"real r={real_r:+.3f}"
        )
        ax.axvline(0, color="k", linestyle="--", alpha=0.4)
        sig_txt = (
            "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        )
        ax.set_title(f"{sub} — lag*={lag_opt:+.1f}s\np={p_val:.4f} {sig_txt}")
        ax.set_xlabel("mean Pearson r (LORO)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Y3 permutation test (N={N_PERMUTATIONS}, shuffle y within run) at lag*")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[perm] wrote {out_png}")


# -----------------------------------------------------------------------------
# Tarea 3 — decoder topomap (conditional)
# -----------------------------------------------------------------------------
def task_topomap(sweep_csv: Path, perm_csv: Path, out_dir: Path) -> None:
    """Conditional topomap of decoder weights for subjects with max mean-r > 0.

    Refit del Pipeline (StandardScaler → PCA → Ridge) en TODO el dataset al
    lag*, sin LORO. Back-projection del ridge.coef_ a través de PCA y división
    por scaler.scale_ → vector de pesos en el espacio de features originales
    (n_features = n_channels × n_bands). Reshape a (n_channels, n_bands)
    aprovechando que `compute_band_power_features_local` produce un layout
    channel-major: `feats[:, ch*n_bands + b]` = log10 bandpower (canal ch, banda b).

    Identidad de reshape verificada por `assert`:
        coef_matrix[:, b]  ==  coef_features[b::n_bands]    (para todo b)
    El reshape natural (n_channels, n_bands) es equivalente.

    Caveat: estos son pesos del **backward filter** (decoder). Para una
    interpretación tipo forward model (qué hace la fuente sobre los sensores),
    aplicar Haufe transform — TODO documentado en el docstring del módulo.
    """
    df_sweep = pd.read_csv(sweep_csv)
    df_perm = pd.read_csv(perm_csv) if perm_csv.exists() else None

    band_names = list(BANDS.keys())
    n_bands = len(band_names)

    for sub in SUBJECTS:
        d_sub = df_sweep[df_sweep["subject"] == sub]
        if len(d_sub) == 0:
            print(f"  {sub}: no rows in sweep CSV; skipping")
            continue
        lag_means = d_sub.groupby("lag_s")["pearson_r"].mean()
        max_r = float(lag_means.max())
        if max_r <= 0:
            print(f"  {sub}: skipping topomap (max mean r={max_r:+.3f} <= 0)")
            continue
        lag_optimal = float(lag_means.idxmax())
        d_sub_lag = d_sub[d_sub["lag_s"] == lag_optimal]
        best_alpha = float(d_sub_lag["best_alpha"].mode().iloc[0])

        ds = build_subject_dataset(sub, lag_optimal)
        if ds is None:
            print(f"  {sub}: dataset rebuild failed at lag*; skipping")
            continue
        X_spec = compute_band_power_features_local(ds["X"], ds["sfreq"])
        n_samples, n_features = X_spec.shape
        n_channels = ds["X"].shape[1]
        assert n_features == n_channels * n_bands, (
            f"feature count mismatch for {sub}: {n_features} != "
            f"{n_channels}×{n_bands}"
        )

        n_pca = min(PCA_SPECTRAL, n_samples - 1, n_features)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=RNG_SEED)),
                ("ridge", Ridge(alpha=best_alpha)),
            ]
        )
        pipe.fit(X_spec, ds["y"])

        ridge_coef = pipe.named_steps["ridge"].coef_  # (n_pca,)
        pca = pipe.named_steps["pca"]
        scaler = pipe.named_steps["scaler"]
        coef_scaled_feat = ridge_coef @ pca.components_  # (n_features,)
        coef_features = coef_scaled_feat / scaler.scale_  # (n_features,)

        coef_matrix = coef_features.reshape(n_channels, n_bands)
        for b_idx in range(n_bands):
            assert np.allclose(coef_matrix[:, b_idx], coef_features[b_idx::n_bands]), (
                f"feature reshape inconsistent with band-stride layout for {sub}"
            )

        info = mne.create_info(ch_names=ds["ch_names"], sfreq=ds["sfreq"], ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="ignore")

        vmax = float(np.max(np.abs(coef_matrix)))
        if vmax == 0:
            vmax = 1.0

        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
        if n_bands == 1:
            axes = np.array([axes])
        im = None
        for ax, b_idx, bname in zip(axes, range(n_bands), band_names):
            im, _ = mne.viz.plot_topomap(
                coef_matrix[:, b_idx],
                info,
                axes=ax,
                show=False,
                vlim=(-vmax, vmax),
                cmap="RdBu_r",
                contours=4,
            )
            ax.set_title(bname)
        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), shrink=0.7, pad=0.02)
            cbar.set_label("decoder weight (a.u.)")

        suptitle = (
            f"{sub} — decoder weights (spectral, refit on all data)\n"
            f"lag*={lag_optimal:+.1f}s, α={best_alpha:g}, "
            f"max mean r (LORO)={max_r:+.3f}"
        )
        if df_perm is not None:
            d_perm = df_perm[df_perm["subject"] == sub]
            if len(d_perm):
                p_val = float(d_perm["p_value"].iloc[0])
                suptitle += f", perm p={p_val:.4f}"
        fig.suptitle(suptitle)

        out_png = out_dir / f"Y3_topomap_decoder_weights_{sub}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {sub}: wrote {out_png}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["sweep", "permutation", "topomap", "all"],
        default="all",
        help="Subtask to run (default: all in order).",
    )
    args = parser.parse_args()

    sweep_csv = NPZ_DIR / "lag_sweep_per_fold.csv"
    perm_csv = NPZ_DIR / "lag_sweep_permutation.csv"
    sweep_png = FIG_DIR / "Y3_lag_sweep_summary.png"

    if args.task in ("sweep", "all"):
        print("=== TASK 1: LAG SWEEP ===")
        task_lag_sweep(sweep_csv, sweep_png)

    if args.task in ("permutation", "all"):
        print("\n=== TASK 2: PERMUTATION TEST ===")
        if not sweep_csv.exists():
            print(f"  Error: {sweep_csv.name} missing. Run --task sweep first.")
            return
        task_permutation(sweep_csv, perm_csv)

    if args.task in ("topomap", "all"):
        print("\n=== TASK 3: TOPOMAP ===")
        if not sweep_csv.exists():
            print(f"  Error: {sweep_csv.name} missing. Run --task sweep first.")
            return
        task_topomap(sweep_csv, perm_csv, FIG_DIR)


if __name__ == "__main__":
    main()
