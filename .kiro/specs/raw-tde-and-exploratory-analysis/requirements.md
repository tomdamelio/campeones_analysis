# Requirements Document

## Introduction

This feature adds two new scripts to the luminance prediction pipeline for sub-27 from the CAMPEONES dataset:

1. Script 13 (`13_luminance_raw_tde_model.py`): A new model that works directly on preprocessed raw EEG from occipital ROI channels, bypassing spectral band-power extraction. The approach is: (a) crop the preprocessed EEG to each video_luminance segment, (b) apply TDE on the continuous raw EEG signal (each time-point is expanded with ±window_half neighbors, producing a matrix of (2×window_half+1) × n_channels features per time-point), (c) apply PCA on the TDE-expanded features to keep the first N principal components (default N=100), yielding N time-series, (d) epoch those N PCA-reduced time-series into 1 s windows with 0.1 s step, (e) fit Ridge regression. This tests whether TDE on raw EEG + PCA captures temporal dynamics more effectively than spectral features + TDE (script 12).

2. An exploratory analysis script (`14_explore_target_distributions.py`): Generates distribution histograms for the variables being predicted or reported by the participant — real luminance, valence (joystick), arousal (joystick), and reported/perceived luminance (joystick). Histograms are produced both for the raw values and for the z-score-normalized-per-video values (which is what the models actually use). This helps understand the statistical properties of the target variables before and after normalization.

## Glossary

- **Raw_TDE_Pipeline**: The modeling pipeline that crops preprocessed raw EEG from ROI channels, applies Time-Delay Embedding on the continuous signal to expand temporal context, applies PCA to reduce dimensionality to N principal component time-series, epochs those time-series, and fits Ridge regression.
- **Spectral_TDE_Pipeline**: The existing script 12 pipeline that first computes spectral band-power features, then applies TDE on those features.
- **TDE**: Time-Delay Embedding — a technique that concatenates feature vectors from neighboring time-points (±window_half) into a single expanded vector, capturing local temporal context.
- **ROI_Channels**: The 11 posterior/occipital electrodes used for luminance prediction: O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6.
- **LOVO_CV**: Leave-One-Video-Out Cross-Validation — the CV strategy where each fold holds out all epochs from one video as the test set.
- **Order_Matrix**: Excel file per run/block that maps each video presentation to its dimension (valence, arousal, luminance) and joystick polarity.
- **Epoch_Entry**: A dictionary containing feature vector `X`, target `y`, `video_id`, `video_identifier`, `run_id`, and `acq` metadata.
- **Distribution_Explorer**: The script that loads event and joystick data across all runs and generates histograms of target variable distributions.
- **Merged_Events**: TSV files in `data/derivatives/merged_events/` containing onset, duration, trial_type, stim_id, condition, and stim_file for each event.

## Requirements

### Requirement 1: Raw EEG TDE and PCA Reduction

**User Story:** As a researcher, I want to apply TDE on the continuous raw preprocessed EEG from ROI channels for each video segment, then reduce dimensionality with PCA, so that I obtain compact time-series that capture temporal context from the raw signal.

#### Acceptance Criteria

1. WHEN a video_luminance segment is identified in the events, THE Raw_TDE_Pipeline SHALL crop the preprocessed EEG to that segment and extract data from ROI_Channels only.
2. WHEN a video segment is cropped, THE Raw_TDE_Pipeline SHALL apply TDE on the continuous raw EEG signal: for each time-point, concatenate the signal from [t − window_half, …, t, …, t + window_half] across all ROI channels, producing a feature matrix of shape (n_valid_timepoints × (2×window_half+1) × n_roi_channels).
3. WHEN TDE is applied on the continuous signal, THE Raw_TDE_Pipeline SHALL discard border time-points where the full ±window_half context is unavailable.
4. WHEN the TDE-expanded feature matrix is obtained, THE Raw_TDE_Pipeline SHALL apply PCA to reduce it to N principal components (default N=100, or fewer if the matrix has fewer rows or columns), yielding N time-series of length n_valid_timepoints.
5. WHEN PCA-reduced time-series are obtained, THE Raw_TDE_Pipeline SHALL epoch them into 1.0 s windows with 0.1 s step (0.9 s overlap), consistent with the existing pipeline.
6. WHEN an epoch's feature vector is created, THE Raw_TDE_Pipeline SHALL pair it with the interpolated physical luminance target from the corresponding luminance CSV.
7. IF a video segment is too short to produce at least one full epoch after TDE border removal, THEN THE Raw_TDE_Pipeline SHALL skip that segment and log a warning.

### Requirement 2: CV Pipeline for Raw TDE Model

**User Story:** As a researcher, I want the epoched PCA features to be fed into the same CV and regression pipeline as the existing models, so that results are directly comparable.

#### Acceptance Criteria

1. THE Raw_TDE_Pipeline SHALL apply z-score normalization of the luminance target per video group, using the existing `zscore_per_video` function.
2. THE Raw_TDE_Pipeline SHALL use Leave-One-Video-Out Cross-Validation with the same fold structure as scripts 10–12.
3. WHEN training each fold, THE Raw_TDE_Pipeline SHALL use a StandardScaler → Ridge pipeline with GridSearchCV over the configured alpha grid (PCA is already applied earlier on the full segment).
4. THE Raw_TDE_Pipeline SHALL set the random seed at the entry point using the configured RANDOM_SEED value and log the seed used.

### Requirement 3: Results Output and Comparison

**User Story:** As a researcher, I want the raw TDE model to produce results in the same format as existing models, so that I can compare performance across approaches.

#### Acceptance Criteria

1. THE Raw_TDE_Pipeline SHALL save a CSV with per-fold metrics (PearsonR, SpearmanRho, RMSE, BestAlpha) to `results/modeling/luminance/raw_tde/`.
2. THE Raw_TDE_Pipeline SHALL generate CV results bar plots and per-fold prediction scatter plots, consistent with scripts 10–12.
3. THE Raw_TDE_Pipeline SHALL generate a comparison plot against the spectral TDE model (script 12) results.
4. WHEN the permutation test is configured (N_PERMUTATIONS > 0), THE Raw_TDE_Pipeline SHALL run a permutation test and save the null distribution and p-value.

### Requirement 4: Exploratory Distribution Analysis

**User Story:** As a researcher, I want to visualize the distributions of the target variables (real luminance, valence, arousal, reported luminance) both raw and z-score-normalized per video, so that I can understand the statistical properties of what the models are predicting before and after normalization.

#### Acceptance Criteria

1. WHEN loading data for exploration, THE Distribution_Explorer SHALL read the Order_Matrix files and Merged_Events TSVs for all configured runs to identify which videos correspond to which dimension (valence, arousal, luminance).
2. WHEN a video segment has dimension "luminance" in the Order_Matrix, THE Distribution_Explorer SHALL load the corresponding physical luminance CSV and compute epoch-level mean luminance values as the real luminance distribution.
3. WHEN a video segment has a joystick recording, THE Distribution_Explorer SHALL extract the joystick_x signal, apply polarity correction from the Order_Matrix, and compute epoch-level mean joystick values as the reported rating distribution for that dimension.
4. THE Distribution_Explorer SHALL generate histograms for each variable (real luminance, reported valence, reported arousal, reported luminance) in two versions: raw values and z-score-normalized-per-video values.
5. THE Distribution_Explorer SHALL annotate each histogram with descriptive statistics (mean, std, min, max, N).
6. THE Distribution_Explorer SHALL save all plots to `results/modeling/luminance/exploration/distributions/`.
7. IF a dimension has no data across all runs (e.g., no arousal videos for this subject), THEN THE Distribution_Explorer SHALL log a warning and skip that histogram.
