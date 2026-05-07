"""Nested cross-validation loops for hyperparameter selection.

Implements:
  - ``nested_lovo_grid_tau``: outer LOVO + inner LOVO grid search over
    the per-modality lag ``tau`` (Test 2 of the framework spec).
  - ``fit_predict_one_modality``: full per-modality pipeline runner
    (window matrices + KNN + B1 informativeness + beta-ridge +
    soft-weighted reconstruction). Returns ``y_hat(t)`` over a single
    held-out test video.
  - ``run_test3_outer_loop``: full Test 3 pipeline (sec 4.7). For each
    outer fold ``v_outer``:
      1. Inner LOVO grid search per modality -> ``tau_m*`` using the
         full beta-model pipeline (spec-compliant, sec 4.7).
      2. Re-OOF post-selection with K=5 random splits (distinct from
         the inner LOVO) -> per-modality OOF predictions over the
         training pool, with the beta-model fitted inside each fit.
      3. Fit ``alpha_m`` via Stage B (OLS / ridge / constrained) on the
         re-OOF predictions; report ``kappa`` and per-modality VIF.
      4. Refit pipeline per modality on the full training pool with
         ``tau_m*`` and predict ``y_m_hat(v_outer)``; combine via
         Stage B to obtain ``y_hat(v_outer)``.
      5. Score ``Pearson(y_hat, y_v_outer)`` over non-NaN samples.

Deviations from sec 4.7 of notas_ada_v2:
  * The hyperparameter grid is ``tau`` only (``L`` is fixed). The spec
    grids ``(tau, L)``; we drop ``L`` for compute and because Test 3's
    pass criteria are about ``alpha_m`` calibration, not ``L``-recovery.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from .b_stack import diagnostics_b_stack, fit_b_stack, predict_b_stack
from .beta_model import (
    compute_b1_informativeness,
    fit_beta_ridge,
    predict_window_weights,
)
from .data_structures import build_window_matrices
from .knn_trajectories import compute_full_similarity, knn_trajectory_predict
from .reconstruction import reconstruct_stage_a


# -----------------------------------------------------------------------
# Inner LOVO grid search over tau (full beta-model pipeline)
# -----------------------------------------------------------------------


def _inner_lovo_grid_tau(
    pool_videos: list[tuple[np.ndarray, np.ndarray]],
    L: int,
    stride: int,
    tau_grid: np.ndarray,
    K: int | str = "sqrt",
    alpha_ridge: float = 0.1,
) -> tuple[int, np.ndarray]:
    """Inner LOVO grid search over ``tau`` on a single video pool.

    Each inner fold uses the full per-modality pipeline (KNN + B1 +
    beta-ridge + soft-weighted reconstruction), in compliance with
    sec 4.7 of the framework spec.

    Returns
    -------
    tau_star
        Lag value chosen by argmax of the mean inner-CV score.
        ``-1`` if no valid score could be computed.
    val_surface
        ``(len(tau_grid),)`` array with the mean inner-CV Pearson per
        tau value. ``NaN`` for tau values that produced no valid score.
    """
    n_pool = len(pool_videos)
    tau_grid = np.asarray(tau_grid, dtype=np.int64).ravel()
    val_surface = np.full(len(tau_grid), np.nan, dtype=float)

    for ti, tau in enumerate(tau_grid):
        inner_scores: list[float] = []
        for v_val_local in range(n_pool):
            train_pool = [pool_videos[i] for i in range(n_pool) if i != v_val_local]
            test_video = pool_videos[v_val_local]
            _, y_v_val = test_video
            try:
                y_hat = fit_predict_one_modality(
                    train_pool,
                    test_video,
                    L=L,
                    stride=stride,
                    tau=int(tau),
                    K=K,
                    alpha_ridge=alpha_ridge,
                )
            except Exception:
                continue
            valid = ~np.isnan(y_hat)
            if valid.sum() < 3:
                continue
            if np.std(y_hat[valid]) < 1e-12 or np.std(y_v_val[valid]) < 1e-12:
                continue
            r, _ = pearsonr(y_hat[valid], y_v_val[valid])
            if not np.isfinite(r):
                continue
            inner_scores.append(float(r))

        if inner_scores:
            val_surface[ti] = float(np.mean(inner_scores))

    if np.all(np.isnan(val_surface)):
        return -1, val_surface
    return int(tau_grid[int(np.nanargmax(val_surface))]), val_surface


def nested_lovo_grid_tau(
    videos: list[tuple[np.ndarray, np.ndarray]],
    L: int,
    stride: int,
    tau_grid,
    K: int | str = "sqrt",
    alpha_ridge: float = 0.1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Outer LOVO + inner LOVO grid search over the lag ``tau``.

    Returns
    -------
    tau_star
        ``(N,)`` int array. ``-1`` if no valid score was produced.
    val_surface
        ``(N, len(tau_grid))`` float array.
    """
    n_videos = len(videos)
    tau_grid = np.asarray(tau_grid, dtype=np.int64).ravel()
    val_surface = np.full((n_videos, len(tau_grid)), np.nan, dtype=float)
    tau_star = np.full(n_videos, -1, dtype=np.int64)

    for v_outer in range(n_videos):
        pool_videos = [videos[v] for v in range(n_videos) if v != v_outer]
        tau_v, surf_v = _inner_lovo_grid_tau(
            pool_videos, L=L, stride=stride, tau_grid=tau_grid, K=K,
            alpha_ridge=alpha_ridge,
        )
        tau_star[v_outer] = tau_v
        val_surface[v_outer] = surf_v

        if verbose:
            best = (
                np.nanmax(surf_v) if not np.all(np.isnan(surf_v)) else float("nan")
            )
            print(
                f"  outer fold {v_outer:>2d}: tau* = {tau_v} samples"
                f"  (best score = {best:.3f})"
            )

    return tau_star, val_surface


# -----------------------------------------------------------------------
# Per-modality pipeline runner (KNN + beta-model + reconstruction)
# -----------------------------------------------------------------------


def fit_predict_one_modality(
    pool_videos: list[tuple[np.ndarray, np.ndarray]],
    test_video: tuple[np.ndarray, np.ndarray],
    L: int,
    stride: int,
    tau: int,
    K: int | str = "sqrt",
    alpha_ridge: float = 0.1,
) -> np.ndarray:
    """Train pipeline on ``pool_videos`` and predict ``y_hat(t)`` on
    ``test_video``.

    Steps:
      1. Build window matrices over ``pool_videos + [test_video]`` with
         the given ``tau``.
      2. KNN-of-trajectories on the training windows (queries = train
         windows; neighbours = other-video training windows).
      3. Compute B1 informativeness on training windows.
      4. Fit beta-ridge on (D_train, a_train).
      5. KNN-of-trajectories on the test windows (queries = test
         windows; neighbours = training windows via same-video
         exclusion).
      6. Predict per-test-window weights via beta.
      7. Soft-weighted reconstruction of ``y_hat(t)`` on the test video.

    Returns
    -------
    y_hat
        ``(T_test,)`` array. NaN where no window covered the sample, or
        where the prediction was otherwise invalid.
    """
    all_videos = pool_videos + [test_video]
    test_local_id = len(pool_videos)

    wm = build_window_matrices(all_videos, L=L, stride=stride, tau=int(tau))
    sims_full = compute_full_similarity(wm)

    is_test = wm.video_id == test_local_id
    is_train = ~is_test

    if isinstance(K, str) and K == "sqrt":
        n_train = int(is_train.sum())
        K_eff = max(1, int(np.sqrt(n_train)))
    else:
        K_eff = int(K)
    K_eff = min(K_eff, max(1, wm.D.shape[0] - 1))

    train_indices = np.where(is_train)[0]
    test_indices = np.where(is_test)[0]
    _, y_v_test = test_video

    if test_indices.size == 0:
        return np.full(len(y_v_test), np.nan)

    y_traj_pred_train = knn_trajectory_predict(
        wm, train_indices, K=K_eff, sims_full=sims_full
    )
    a_train = compute_b1_informativeness(y_traj_pred_train, wm.Y_traj[is_train])
    beta, intercept = fit_beta_ridge(
        wm.D[is_train], a_train, alpha_ridge=alpha_ridge
    )

    y_traj_pred_test = knn_trajectory_predict(
        wm, test_indices, K=K_eff, sims_full=sims_full
    )
    w_hat_test = predict_window_weights(wm.D[is_test], beta, intercept)
    y_hat = reconstruct_stage_a(
        y_traj_pred=y_traj_pred_test,
        t_start=wm.t_start[is_test],
        L=L,
        n_samples=len(y_v_test),
        weights=w_hat_test,
        tau=int(tau),
    )
    return y_hat


# -----------------------------------------------------------------------
# Test 3: full nested CV with re-OOF and B-stack
# -----------------------------------------------------------------------


def _make_kfold_assignment(
    n_pool: int, n_kfold: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign each pool index to one of ``n_kfold`` folds, balanced."""
    base = np.arange(n_pool) % n_kfold
    rng.shuffle(base)
    return base


def run_test3_outer_loop(
    videos_per_mod: list[list[tuple[np.ndarray, np.ndarray]]],
    L: int,
    stride: int,
    tau_grid,
    K: int | str = "sqrt",
    alpha_ridge: float = 0.1,
    n_kfold: int = 5,
    stack_variant: str = "ols",
    alpha_ridge_stack: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Test 3 outer loop: nested LOVO + re-OOF post-selection + B-stack.

    Parameters
    ----------
    videos_per_mod
        ``M``-element list. ``videos_per_mod[m][v]`` is the
        ``(X_{m,v}, y_v)`` pair for modality ``m``, video ``v``. The
        ``y_v`` array must be shared across modalities for each video.
    L, stride
        Window hyperparameters (fixed).
    tau_grid
        Iterable of candidate lag values in samples. Searched
        independently per modality.
    K
        Neighbour count strategy passed to the per-modality pipeline.
    alpha_ridge
        Ridge penalty for the beta-model.
    n_kfold
        Number of folds for the re-OOF post-selection split (default 5).
    stack_variant
        Stage B variant: ``"ols"`` (default), ``"ridge"``, or
        ``"constrained"``.
    alpha_ridge_stack
        Ridge penalty for ``stack_variant="ridge"``.
    seed
        Random seed for the K-fold split.
    verbose
        If ``True``, print one line per outer fold with summary numbers.

    Returns
    -------
    dict with keys:
      ``tau_stars``       (N, M) int   selected tau per (outer fold, modality)
      ``val_surfaces``    (N, M, |grid|) float  inner-CV mean score
      ``alpha_m``         (N, M) float           stacking weights
      ``alpha_0``         (N,)   float           stacking intercept
      ``kappa``           (N,)   float           stacking design matrix kappa
      ``vif``             (N, M) float           per-modality VIF
      ``outer_scores``    (N,)   float           Pearson(y_hat, y) per outer fold
      ``per_mod_scores``  (N, M) float           Pearson(y_m_hat, y) per outer fold
    """
    M = len(videos_per_mod)
    if M < 2:
        raise ValueError(f"Test 3 expects M >= 2 modalities, got M={M}.")
    N = len(videos_per_mod[0])
    for m in range(M):
        if len(videos_per_mod[m]) != N:
            raise ValueError(
                f"All modalities must have the same number of videos. "
                f"videos_per_mod[0] has {N}, videos_per_mod[{m}] has "
                f"{len(videos_per_mod[m])}."
            )

    rng = np.random.default_rng(seed)
    tau_grid = np.asarray(tau_grid, dtype=np.int64).ravel()

    tau_stars = np.full((N, M), -1, dtype=np.int64)
    val_surfaces = np.full((N, M, len(tau_grid)), np.nan, dtype=float)
    alpha_m = np.full((N, M), np.nan, dtype=float)
    alpha_0 = np.full(N, np.nan, dtype=float)
    kappa_arr = np.full(N, np.nan, dtype=float)
    vif_arr = np.full((N, M), np.nan, dtype=float)
    outer_scores = np.full(N, np.nan, dtype=float)
    per_mod_scores = np.full((N, M), np.nan, dtype=float)

    for v_outer in range(N):
        pool_indices = [v for v in range(N) if v != v_outer]
        n_pool = len(pool_indices)

        # --- Step 1: per-modality inner LOVO grid search over tau --------
        for m in range(M):
            pool_videos_m = [videos_per_mod[m][v] for v in pool_indices]
            tau_m, surf_m = _inner_lovo_grid_tau(
                pool_videos_m, L=L, stride=stride, tau_grid=tau_grid, K=K,
                alpha_ridge=alpha_ridge,
            )
            tau_stars[v_outer, m] = tau_m
            val_surfaces[v_outer, m] = surf_m

        # --- Step 2: re-OOF post-selection (K-fold split of the pool) ----
        # Distinct from the inner LOVO above; this generates clean OOF
        # predictions for fitting the stacking weights.
        kfold_assignment = _make_kfold_assignment(n_pool, n_kfold, rng)

        # re_oof_pred[m][v_in_pool] = y_hat array (T_v,)
        re_oof_pred: list[dict[int, np.ndarray]] = [{} for _ in range(M)]
        for m in range(M):
            tau_m = int(tau_stars[v_outer, m])
            if tau_m < 0:
                continue
            pool_videos_m = [videos_per_mod[m][v] for v in pool_indices]

            for k in range(n_kfold):
                test_local = np.where(kfold_assignment == k)[0]
                train_local = np.where(kfold_assignment != k)[0]
                if test_local.size == 0 or train_local.size == 0:
                    continue
                train_pool_m = [pool_videos_m[i] for i in train_local]
                for tli in test_local:
                    test_video_m = pool_videos_m[tli]
                    y_hat = fit_predict_one_modality(
                        train_pool_m,
                        test_video_m,
                        L=L,
                        stride=stride,
                        tau=tau_m,
                        K=K,
                        alpha_ridge=alpha_ridge,
                    )
                    v_in_pool = pool_indices[tli]
                    re_oof_pred[m][v_in_pool] = y_hat

        # Concatenate OOF predictions across pool videos and modalities
        oof_X_blocks: list[np.ndarray] = []
        oof_y_blocks: list[np.ndarray] = []
        for v_in_pool in pool_indices:
            y_v = videos_per_mod[0][v_in_pool][1]
            preds = []
            ok = True
            for m in range(M):
                arr = re_oof_pred[m].get(v_in_pool)
                if arr is None or arr.shape[0] != y_v.shape[0]:
                    ok = False
                    break
                preds.append(arr)
            if not ok:
                continue
            oof_X_blocks.append(np.column_stack(preds))
            oof_y_blocks.append(y_v)

        if not oof_X_blocks:
            if verbose:
                print(f"  outer fold {v_outer:>2d}: no valid OOF data, skipping.")
            continue

        oof_X = np.concatenate(oof_X_blocks, axis=0)
        oof_y = np.concatenate(oof_y_blocks, axis=0)
        valid = np.isfinite(oof_X).all(axis=1) & np.isfinite(oof_y)
        if int(valid.sum()) < M + 5:
            if verbose:
                print(
                    f"  outer fold {v_outer:>2d}: only {int(valid.sum())} "
                    f"finite OOF rows; skipping."
                )
            continue
        oof_X = oof_X[valid]
        oof_y = oof_y[valid]

        # --- Step 3: diagnostics + Stage B fit ---------------------------
        kappa, vif = diagnostics_b_stack(oof_X)
        kappa_arr[v_outer] = kappa
        vif_arr[v_outer] = vif

        a_m, a_0 = fit_b_stack(
            oof_X, oof_y,
            variant=stack_variant,
            alpha_ridge_stack=alpha_ridge_stack,
        )
        alpha_m[v_outer] = a_m
        alpha_0[v_outer] = a_0

        # --- Step 4: refit per modality on full pool, predict on v_outer -
        per_mod_y_hat: list[np.ndarray] = []
        for m in range(M):
            tau_m = int(tau_stars[v_outer, m])
            pool_videos_m = [videos_per_mod[m][v] for v in pool_indices]
            test_video_m = videos_per_mod[m][v_outer]
            y_m_hat = fit_predict_one_modality(
                pool_videos_m,
                test_video_m,
                L=L,
                stride=stride,
                tau=tau_m,
                K=K,
                alpha_ridge=alpha_ridge,
            )
            per_mod_y_hat.append(y_m_hat)

            y_outer_true = videos_per_mod[m][v_outer][1]
            valid_m = np.isfinite(y_m_hat) & np.isfinite(y_outer_true)
            if (
                valid_m.sum() >= 3
                and np.std(y_m_hat[valid_m]) > 1e-12
                and np.std(y_outer_true[valid_m]) > 1e-12
            ):
                r_m, _ = pearsonr(y_m_hat[valid_m], y_outer_true[valid_m])
                if np.isfinite(r_m):
                    per_mod_scores[v_outer, m] = float(r_m)

        # --- Step 5: combine via Stage B + score -------------------------
        y_outer_true = videos_per_mod[0][v_outer][1]
        per_mod_arr = np.column_stack(per_mod_y_hat)
        y_hat_combined = predict_b_stack(per_mod_arr, a_m, a_0)
        valid_o = np.isfinite(y_hat_combined) & np.isfinite(y_outer_true)
        if (
            valid_o.sum() >= 3
            and np.std(y_hat_combined[valid_o]) > 1e-12
            and np.std(y_outer_true[valid_o]) > 1e-12
        ):
            r_o, _ = pearsonr(y_hat_combined[valid_o], y_outer_true[valid_o])
            if np.isfinite(r_o):
                outer_scores[v_outer] = float(r_o)

        if verbose:
            tau_str = ",".join(str(int(t)) for t in tau_stars[v_outer])
            alpha_str = ", ".join(f"{a:+.2f}" for a in a_m)
            print(
                f"  outer fold {v_outer:>2d}: tau*=({tau_str}) "
                f"alpha=({alpha_str}) intercept={a_0:+.2f} "
                f"kappa={kappa:.1f} pearson={outer_scores[v_outer]:.3f}"
            )

    return {
        "tau_stars": tau_stars,
        "val_surfaces": val_surfaces,
        "alpha_m": alpha_m,
        "alpha_0": alpha_0,
        "kappa": kappa_arr,
        "vif": vif_arr,
        "outer_scores": outer_scores,
        "per_mod_scores": per_mod_scores,
    }
