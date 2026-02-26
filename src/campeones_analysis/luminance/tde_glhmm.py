"""GLHMM-based Time-Delay Embedding preprocessing pipeline.

Wraps the TDE-GLHMM preprocessing protocol from Vidaurre et al. (2025,
Nature Protocols) for use in the luminance prediction pipeline. Only the
preprocessing stage is reused here — the GLHMM model itself is replaced
by Ridge regression.

References:
    Vidaurre et al. (2025). A protocol for time-delay embedded hidden
    Markov modelling of brain data. Nature Protocols.
    https://doi.org/10.1038/s41596-025-01300-2

Requirements: 4.1, 4.2
"""

import numpy as np
from sklearn.decomposition import PCA
from glhmm import preproc as glhmm_preproc


def apply_tde_only(
    eeg_data: np.ndarray,
    indices: np.ndarray,
    tde_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ONLY the Time-Delay Embedding step, without PCA.

    Uses ``glhmm.preproc.build_data_tde()`` to embed symmetric lags.
    The returned data is already standardised (zero-mean, unit-variance)
    by the library function.

    Args:
        eeg_data: 2-D array ``(n_timepoints, n_channels)``.
        indices: 2-D array ``(n_sessions, 2)`` with start/end indices.
        tde_lags: Number of lags for symmetric TDE embedding.

    Returns:
        Tuple of ``(tde_data, tde_indices)`` where ``tde_data`` has shape
        ``(n_valid_timepoints, n_channels * (2 * tde_lags + 1))``.
    """
    lags = list(range(-tde_lags, tde_lags + 1))
    tde_result = glhmm_preproc.build_data_tde(
        data=eeg_data,
        indices=indices,
        lags=lags,
    )
    return tde_result[0], tde_result[1]


def fit_global_pca(
    tde_segments: list[np.ndarray],
    pca_components: int,
) -> PCA:
    """Fit a single global PCA model on concatenated TDE data from all segments.

    This ensures all segments are projected into the **same** subspace,
    which is essential for cross-video generalization in downstream models.

    Args:
        tde_segments: List of 2-D TDE-embedded arrays, one per video segment.
        pca_components: Number of PCA components to retain.

    Returns:
        Fitted ``sklearn.decomposition.PCA`` model.
    """
    concatenated = np.vstack(tde_segments)
    pca = PCA(n_components=pca_components, svd_solver="full")
    pca.fit(concatenated)
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(
        f"  Global PCA fitted on {concatenated.shape[0]} timepoints × "
        f"{concatenated.shape[1]} features → {pca_components} components "
        f"({explained:.1f}% variance explained)"
    )
    return pca


def apply_global_pca(
    tde_data: np.ndarray,
    pca_model: PCA,
    standardise_pc: bool = True,
) -> np.ndarray:
    """Project TDE-embedded data into the global PCA subspace.

    Matches the canonical ``glhmm.preproc.build_data_tde()`` behavior:
    after PCA projection, each principal component is re-standardised
    to zero-mean, unit-variance (``standardise_pc=True`` by default).

    Args:
        tde_data: 2-D TDE-embedded array ``(n_timepoints, n_tde_features)``.
        pca_model: Pre-fitted PCA model from ``fit_global_pca()``.
        standardise_pc: If ``True``, zero-mean and unit-variance each PC
            column after projection (matches glhmm default).

    Returns:
        2-D array ``(n_timepoints, n_components)``.
    """
    projected = pca_model.transform(tde_data)
    if standardise_pc:
        projected -= np.mean(projected, axis=0)
        projected /= np.std(projected, axis=0)
    return projected


def apply_glhmm_tde_pipeline(
    eeg_data: np.ndarray,
    indices: np.ndarray,
    tde_lags: int,
    pca_components: int,
) -> np.ndarray:
    """Apply GLHMM TDE preprocessing pipeline on continuous EEG data.

    .. deprecated::
        This function fits PCA per-segment. For cross-video models, use
        the two-step ``apply_tde_only()`` + ``fit_global_pca()`` +
        ``apply_global_pca()`` instead.

    Args:
        eeg_data: 2-D array of shape ``(n_timepoints, n_channels)``.
        indices: 2-D array of shape ``(n_sessions, 2)``.
        tde_lags: Number of lags for TDE embedding.
        pca_components: Number of PCA components to retain.

    Returns:
        2-D array ``(n_valid_timepoints, pca_components)``.
    """
    lags = list(range(-tde_lags, tde_lags + 1))

    # Step 1: Time-Delay Embedding
    tde_result = glhmm_preproc.build_data_tde(
        data=eeg_data,
        indices=indices,
        lags=lags,
    )
    tde_data, tde_indices = tde_result[0], tde_result[1]

    # Step 2: Standardise + PCA
    preproc_result = glhmm_preproc.preprocess_data(
        data=tde_data,
        indices=tde_indices,
        standardise=True,
        pca=pca_components,
    )
    pca_data: np.ndarray = preproc_result[0]

    return pca_data

