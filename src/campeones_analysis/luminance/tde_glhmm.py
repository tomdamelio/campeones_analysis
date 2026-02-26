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
from glhmm import preproc as glhmm_preproc


def apply_glhmm_tde_pipeline(
    eeg_data: np.ndarray,
    indices: np.ndarray,
    tde_lags: int,
    pca_components: int,
) -> np.ndarray:
    """Apply GLHMM TDE preprocessing pipeline on continuous EEG data.

    Follows the TDE-GLHMM protocol from Vidaurre et al. (2025, Nature
    Protocols):

    1. Apply Time-Delay Embedding via ``glhmm.preproc.build_data_tde()``
       using symmetric lags ``[-tde_lags, ..., 0, ..., +tde_lags]``.
    2. Standardise and reduce dimensionality with PCA via
       ``glhmm.preproc.preprocess_data()``.

    This replaces the custom ``apply_tde_on_continuous_signal`` +
    sklearn PCA approach used in the original script 13.

    Args:
        eeg_data: 2-D array of shape ``(n_timepoints, n_channels)`` with
            continuous EEG signal from ROI channels.
        indices: 2-D array of shape ``(n_sessions, 2)`` with start/end
            sample indices for each session/segment, as required by
            ``glhmm.preproc``.
        tde_lags: Number of lags for TDE embedding. Symmetric lags
            ``range(-tde_lags, tde_lags + 1)`` are applied, producing
            ``n_channels * (2 * tde_lags + 1)`` features before PCA.
        pca_components: Number of PCA components to retain after
            standardisation.

    Returns:
        2-D array of shape ``(n_valid_timepoints, pca_components)`` with
        TDE-embedded, standardised, PCA-reduced time-series. The number
        of valid timepoints is reduced relative to the input because
        boundary samples are discarded during embedding.

    References:
        Vidaurre et al. (2025). Nature Protocols.
        https://doi.org/10.1038/s41596-025-01300-2
    """
    lags = list(range(-tde_lags, tde_lags + 1))

    # Step 1: Time-Delay Embedding
    tde_result = glhmm_preproc.build_data_tde(
        data=eeg_data,
        indices=indices,
        lags=lags,
    )
    # build_data_tde returns (X_emb, indices_emb) when no PCA is requested
    tde_data, tde_indices = tde_result[0], tde_result[1]

    # Step 2: Standardise + PCA
    preproc_result = glhmm_preproc.preprocess_data(
        data=tde_data,
        indices=tde_indices,
        standardise=True,
        pca=pca_components,
    )
    # preprocess_data returns (data_out, indices_out, pcamodel) when pca is set
    pca_data: np.ndarray = preproc_result[0]

    return pca_data
