"""Synthetic data generators for the multimodal arousal framework.

Test 1 simulator: trivial scenario with a single modality where
X(t) = y(t) + white noise (no lag, no convolution). Used to smoke-test
the KNN-of-trajectories + soft-weighted reconstruction plumbing.

Test 2 simulator: 1 modality with a Gaussian-smoothed, time-shifted X
relative to y, plus tunable SNR. Used to validate that the inner-CV
grid search over the lag hyperparameter ``tau`` recovers the true lag.

Heterogeneous-SNR simulator: each video has a high-SNR segment followed
by a low-SNR segment. Used as the sanity-check setup for the beta-model
(Etapas 2-3): a window's informativeness should track which segment it
falls into, and the beta-model should learn this from D features.

Multimodal simulator: 3 modalities share a common ``y(t)``; each X_m has
its own lag ``tau_m`` and SNR_m, optionally its own kernel sigma_m. Used
for Test 3 (multimodal stacking calibration).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def simulate_smoke_test(
    n_videos: int = 10,
    fs: int = 25,
    duration_range: tuple[float, float] = (60.0, 180.0),
    snr: float = 1.0,
    ar1_phi: float = 0.95,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate the trivial-decoding dataset for Test 1.

    For each of ``n_videos`` videos, build:
      - ``y_v``: an AR(1) process with autoregressive coefficient ``ar1_phi``
        (high phi acts as a low-pass), normalized to unit variance and
        independent across videos.
      - ``X_v = y_v + N(0, sigma^2)`` with ``sigma`` chosen so that
        ``var(y) / var(noise) = snr``.

    Parameters
    ----------
    n_videos
        Number of independent video realisations.
    fs
        Sampling rate in Hz.
    duration_range
        (min, max) duration in seconds. Per-video durations are drawn
        uniformly from this range and rounded to integer samples.
    snr
        Signal-to-noise ratio defined as ``var(y) / var(noise)``.
    ar1_phi
        AR(1) coefficient. Stationary variance is fixed to 1 via the
        innovation variance ``1 - phi^2``.
    seed
        Random seed.

    Returns
    -------
    videos
        List of length ``n_videos`` of ``(X_v, y_v)`` pairs, each a 1D
        array of shape ``(T_v,)`` with ``T_v = int(round(duration_v * fs))``.
    """
    if not 0.0 < ar1_phi < 1.0:
        raise ValueError(f"ar1_phi must be in (0, 1), got {ar1_phi}")
    if snr <= 0:
        raise ValueError(f"snr must be > 0, got {snr}")

    rng = np.random.default_rng(seed)
    burn_in = max(200, int(10 / (1.0 - ar1_phi)))
    innov_std = float(np.sqrt(1.0 - ar1_phi**2))
    noise_std = float(np.sqrt(1.0 / snr))

    duration_lo, duration_hi = duration_range
    videos: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_videos):
        duration_s = rng.uniform(duration_lo, duration_hi)
        n_samples = int(round(duration_s * fs))

        innovations = rng.normal(0.0, innov_std, size=burn_in + n_samples)
        y_full = np.empty(burn_in + n_samples, dtype=float)
        y_full[0] = innovations[0]
        for t in range(1, burn_in + n_samples):
            y_full[t] = ar1_phi * y_full[t - 1] + innovations[t]
        y_v = y_full[burn_in:]

        noise = rng.normal(0.0, noise_std, size=n_samples)
        x_v = y_v + noise

        videos.append((x_v, y_v))

    return videos


def _ar1_realization(
    n_samples: int,
    ar1_phi: float,
    innov_std: float,
    rng: np.random.Generator,
    burn_in: int,
) -> np.ndarray:
    """Draw an AR(1) realization of length ``n_samples`` (post burn-in)."""
    n_total = burn_in + n_samples
    innovations = rng.normal(0.0, innov_std, size=n_total)
    y = np.empty(n_total, dtype=float)
    y[0] = innovations[0]
    for t in range(1, n_total):
        y[t] = ar1_phi * y[t - 1] + innovations[t]
    return y[burn_in:]


def simulate_lag_test(
    n_videos: int = 20,
    fs: int = 25,
    duration_range: tuple[float, float] = (60.0, 180.0),
    snr: float = 4.0,
    tau_true_s: float = 3.0,
    kernel_sigma_s: float = 1.0,
    ar1_phi: float = 0.95,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Generate the dataset for Test 2 (lag-recovery via grid search).

    For each video, ``X(t)`` is a Gaussian-smoothed copy of ``y`` shifted
    so that ``X(t)`` reflects ``y`` at time ``t + tau_true``, plus white
    noise. With this convention,
    ``build_window_matrices(..., tau=tau_true_samples)`` produces D and
    Y_traj that are aligned in y-time (D[j] is roughly a smoothed copy of
    Y_traj[j] plus noise), so the framework's grid search over ``tau``
    should pick ``tau ~= tau_true_samples``.

    Note on the sign convention. The framework's reconstruction defines
    ``cover(t) = {j : t_j + tau <= t < t_j + tau + L}``, i.e. positive
    ``tau`` means X *leads* y by ``tau`` samples. The simulator follows
    that operational convention rather than the literal causal one
    (where X would lag y).

    SNR is defined as ``var(X_clean) / var(noise)``, where ``X_clean``
    is the smoothed shifted ``y`` before adding noise. This isolates the
    "input SNR" from the smoothing-induced variance shrinkage.

    Parameters
    ----------
    n_videos
        Number of independent videos.
    fs
        Sampling rate in Hz.
    duration_range
        (min, max) duration in seconds. Drawn uniformly per video.
    snr
        Signal-to-noise ratio ``var(X_clean) / var(noise)``.
    tau_true_s
        True lead-lag of X relative to y, in seconds.
    kernel_sigma_s
        Standard deviation of the Gaussian convolution kernel applied
        to ``y`` before time-shifting, in seconds.
    ar1_phi
        AR(1) coefficient for ``y``.
    seed
        Random seed.

    Returns
    -------
    videos
        List of length ``n_videos`` of ``(X_v, y_v)`` pairs of equal length.
    tau_true_samples
        ``tau_true_s`` rounded to integer samples (= ``round(tau_true_s * fs)``).
    """
    if not 0.0 < ar1_phi < 1.0:
        raise ValueError(f"ar1_phi must be in (0, 1), got {ar1_phi}")
    if snr <= 0:
        raise ValueError(f"snr must be > 0, got {snr}")
    if tau_true_s < 0:
        raise ValueError(f"tau_true_s must be >= 0, got {tau_true_s}")
    if kernel_sigma_s <= 0:
        raise ValueError(f"kernel_sigma_s must be > 0, got {kernel_sigma_s}")

    rng = np.random.default_rng(seed)
    burn_in = max(200, int(10 / (1.0 - ar1_phi)))
    innov_std = float(np.sqrt(1.0 - ar1_phi**2))

    tau_true_samples = int(round(tau_true_s * fs))
    sigma_samples = float(kernel_sigma_s * fs)
    edge_pad = int(np.ceil(4 * sigma_samples)) + tau_true_samples

    duration_lo, duration_hi = duration_range
    videos: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_videos):
        duration_s = rng.uniform(duration_lo, duration_hi)
        n_samples = int(round(duration_s * fs))

        y_extended = _ar1_realization(
            n_samples=n_samples + edge_pad,
            ar1_phi=ar1_phi,
            innov_std=innov_std,
            rng=rng,
            burn_in=burn_in,
        )
        y_smooth = gaussian_filter1d(y_extended, sigma=sigma_samples, mode="reflect")

        y_v = y_extended[:n_samples]
        x_clean = y_smooth[tau_true_samples : tau_true_samples + n_samples]

        x_var = float(np.var(x_clean))
        noise_std = float(np.sqrt(x_var / snr))
        x_v = x_clean + rng.normal(0.0, noise_std, size=n_samples)

        videos.append((x_v, y_v))

    return videos, tau_true_samples


def simulate_heterogeneous_snr_test(
    n_videos: int = 20,
    fs: int = 25,
    duration_s: float = 120.0,
    snr_high: float = 8.0,
    snr_low: float = 0.1,
    high_snr_duration_s: float = 60.0,
    low_snr_offset: float = -2.0,
    ar1_phi: float = 0.95,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Generate the sanity-check dataset for the beta-model.

    Each video has fixed length ``duration_s``. The first
    ``high_snr_duration_s`` seconds are generated with ``snr_high`` and
    no offset; the remaining samples with ``snr_low`` plus
    ``low_snr_offset`` added to ``X`` (NOT to ``y``).

    Two design notes
    ----------------
    1. SNR gap is deliberately large (default 80x in variance terms)
       because AR(1) ``phi=0.95`` is so smooth that K~sqrt(W) neighbours
       average out moderate noise — without a wide gap the per-window
       informativeness barely differs between segments.
    2. The DC offset on the low-SNR segment is the linear feature the
       beta-model needs to discriminate. Linear ridge over a raw window
       cannot extract noise level (a quadratic statistic), but it can
       trivially extract ``mean(D[j])``, which under this simulator
       tracks SNR segment membership.

    With this setup:
      * windows fully inside the high-SNR segment have ``mean(D) ~ 0``
        and high B1 informativeness;
      * windows fully inside the low-SNR segment have ``mean(D) ~ offset``
        and low B1 informativeness;
      * ``β`` should learn ``mean(D)`` as a positive predictor of
        informativeness.

    SNR is defined per Test 1 convention: ``var(y) / var(noise)``.

    Parameters
    ----------
    n_videos
        Number of videos.
    fs
        Sampling rate in Hz.
    duration_s
        Per-video duration (fixed across videos for clean window
        classification at the SNR-segment boundary).
    snr_high, snr_low
        SNR in the high-SNR and low-SNR segments respectively.
    high_snr_duration_s
        Length of the high-SNR segment, in seconds, measured from the
        start of each video.
    low_snr_offset
        Constant added to ``X`` in the low-SNR segment. Set to 0 to
        recover a pure-SNR-gap simulator (where the beta-model should
        not be expected to recover the structure linearly).
    ar1_phi
        AR(1) coefficient for ``y``.
    seed
        Random seed.

    Returns
    -------
    videos
        List of length ``n_videos`` of ``(X_v, y_v)`` pairs of equal
        length ``int(round(duration_s * fs))``.
    high_n_samples
        Number of samples that belong to the high-SNR segment from the
        start of each video. Use this to classify windows as "fully
        inside high-SNR" (``t_start + L <= high_n_samples``) or "fully
        inside low-SNR" (``t_start >= high_n_samples``).
    """
    if not 0.0 < ar1_phi < 1.0:
        raise ValueError(f"ar1_phi must be in (0, 1), got {ar1_phi}")
    if snr_high <= 0 or snr_low <= 0:
        raise ValueError(
            f"SNRs must be > 0, got snr_high={snr_high}, snr_low={snr_low}"
        )
    if not 0 < high_snr_duration_s < duration_s:
        raise ValueError(
            f"high_snr_duration_s must be in (0, {duration_s}), got "
            f"{high_snr_duration_s}"
        )

    rng = np.random.default_rng(seed)
    burn_in = max(200, int(10 / (1.0 - ar1_phi)))
    innov_std = float(np.sqrt(1.0 - ar1_phi**2))

    n_samples = int(round(duration_s * fs))
    high_n_samples = int(round(high_snr_duration_s * fs))
    noise_std_high = float(np.sqrt(1.0 / snr_high))
    noise_std_low = float(np.sqrt(1.0 / snr_low))

    videos: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_videos):
        y_v = _ar1_realization(
            n_samples=n_samples,
            ar1_phi=ar1_phi,
            innov_std=innov_std,
            rng=rng,
            burn_in=burn_in,
        )
        x_v = y_v.copy()
        x_v[:high_n_samples] = (
            y_v[:high_n_samples]
            + rng.normal(0.0, noise_std_high, size=high_n_samples)
        )
        x_v[high_n_samples:] = (
            y_v[high_n_samples:]
            + rng.normal(0.0, noise_std_low, size=n_samples - high_n_samples)
            + low_snr_offset
        )
        videos.append((x_v, y_v))

    return videos, high_n_samples


def simulate_multimodal_test(
    n_videos: int = 20,
    fs: int = 25,
    duration_range: tuple[float, float] = (60.0, 180.0),
    tau_true_s: tuple[float, ...] = (2.0, 3.0, 4.0),
    snr: tuple[float, ...] = (2.0, 1.0, 0.5),
    kernel_sigma_s: float | tuple[float, ...] = 1.0,
    ar1_phi: float = 0.95,
    seed: int = 42,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Generate the dataset for Test 3 (multimodal stacking calibration).

    For each video, one shared AR(1) target ``y_v(t)`` is drawn, then
    each modality ``m`` produces its own observation
    ``X_m,v(t) = (g_{sigma_m} * y_shifted_m)(t) + noise_m(t)`` with its
    own lag, SNR, and (optionally) kernel sigma. The shift convention is
    the same as ``simulate_lag_test`` (positive ``tau_m`` means X_m
    leads y).

    Defaults match sec 4.3 of the framework explainer:
      - 3 modalities,
      - ``(tau_1, tau_2, tau_3) = (2, 3, 4) s``,
      - ``SNR = (2, 1, 0.5)`` — modality 1 structurally most informative,
      - common kernel ``sigma = 1 s``.

    SNR is defined as ``var(X_m_clean) / var(noise_m)``.

    Parameters
    ----------
    n_videos
        Number of videos.
    fs
        Sampling rate in Hz.
    duration_range
        (min, max) duration in seconds. Drawn uniformly per video.
    tau_true_s
        Tuple of per-modality true lags in seconds. Length defines the
        number of modalities ``M``.
    snr
        Per-modality SNR. Must have the same length as ``tau_true_s``.
    kernel_sigma_s
        Either a scalar (same kernel across modalities) or a tuple of
        the same length as ``tau_true_s`` (per-modality kernel sigma).
    ar1_phi
        AR(1) coefficient for ``y``.
    seed
        Random seed.

    Returns
    -------
    videos_per_mod
        List of length ``M`` of per-modality video lists. Element
        ``videos_per_mod[m][v]`` is the ``(X_{m,v}, y_v)`` pair for the
        modality ``m`` of video ``v``. ``y_v`` is the same array
        instance across modalities (shared target).
    tau_true_samples
        ``(M,)`` integer array with the true lags converted to samples.
    """
    M = len(tau_true_s)
    if len(snr) != M:
        raise ValueError(
            f"snr has length {len(snr)}, expected {M} (matching tau_true_s)."
        )
    if not 0.0 < ar1_phi < 1.0:
        raise ValueError(f"ar1_phi must be in (0, 1), got {ar1_phi}")
    if any(s <= 0 for s in snr):
        raise ValueError(f"All SNRs must be > 0, got {snr}")
    if any(t < 0 for t in tau_true_s):
        raise ValueError(f"All tau_true_s must be >= 0, got {tau_true_s}")

    if np.isscalar(kernel_sigma_s):
        kernel_sigma_s_per_mod = tuple([float(kernel_sigma_s)] * M)
    else:
        if len(kernel_sigma_s) != M:
            raise ValueError(
                f"kernel_sigma_s has length {len(kernel_sigma_s)}, "
                f"expected {M}."
            )
        kernel_sigma_s_per_mod = tuple(float(s) for s in kernel_sigma_s)
    if any(s <= 0 for s in kernel_sigma_s_per_mod):
        raise ValueError(
            f"All kernel sigmas must be > 0, got {kernel_sigma_s_per_mod}"
        )

    rng = np.random.default_rng(seed)
    burn_in = max(200, int(10 / (1.0 - ar1_phi)))
    innov_std = float(np.sqrt(1.0 - ar1_phi**2))

    tau_true_samples = np.array(
        [int(round(t * fs)) for t in tau_true_s], dtype=np.int64
    )
    sigma_samples_per_mod = np.array(
        [s * fs for s in kernel_sigma_s_per_mod], dtype=float
    )
    edge_pad = int(
        np.ceil(4 * sigma_samples_per_mod.max())
    ) + int(tau_true_samples.max())

    duration_lo, duration_hi = duration_range
    videos_per_mod: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(M)
    ]

    for _ in range(n_videos):
        duration_s = rng.uniform(duration_lo, duration_hi)
        n_samples = int(round(duration_s * fs))

        y_extended = _ar1_realization(
            n_samples=n_samples + edge_pad,
            ar1_phi=ar1_phi,
            innov_std=innov_std,
            rng=rng,
            burn_in=burn_in,
        )
        y_v = y_extended[:n_samples]

        for m in range(M):
            tau_m = int(tau_true_samples[m])
            sigma_m = float(sigma_samples_per_mod[m])
            y_smooth_m = gaussian_filter1d(
                y_extended, sigma=sigma_m, mode="reflect"
            )
            x_clean_m = y_smooth_m[tau_m : tau_m + n_samples]
            x_var = float(np.var(x_clean_m))
            noise_std = float(np.sqrt(x_var / snr[m]))
            x_m_v = x_clean_m + rng.normal(0.0, noise_std, size=n_samples)
            videos_per_mod[m].append((x_m_v, y_v))

    return videos_per_mod, tau_true_samples
