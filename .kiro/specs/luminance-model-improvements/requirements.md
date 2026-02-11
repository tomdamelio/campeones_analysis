# Requirements Document

## Introduction

This feature improves the existing EEG luminance prediction pipeline (scripts 10–12) for sub-27. The current pipeline achieves poor performance (Base r=−0.01, Spectral r=0.06, TDE r=0.18). Four targeted improvements are introduced: longer epochs for better spectral resolution, z-score normalization of the luminance target per video, grid search over Ridge alpha, and a permutation test for statistical significance. All changes modify the existing codebase rather than creating new pipelines.

## Glossary

- **Pipeline**: The EEG-to-luminance prediction system comprising scripts 10 (base), 11 (spectral), and 12 (TDE).
- **Config**: The shared configuration module `scripts/modeling/config_luminance.py`.
- **Epoch**: A fixed-duration window of EEG data extracted from a continuous recording.
- **PSD**: Power Spectral Density, estimated via Welch's method.
- **Spectral_Resolution**: The frequency spacing Δf = sfreq / nperseg in the PSD estimate.
- **Z-Score_Normalization**: Transforming values to zero mean and unit variance within a group.
- **Ridge_Alpha**: The L2 regularization strength parameter of the Ridge regression model.
- **Grid_Search**: Systematic search over a predefined set of hyperparameter values using cross-validation.
- **Inner_CV**: Cross-validation performed within the training fold to select hyperparameters, distinct from the outer Leave-One-Video-Out CV.
- **Permutation_Test**: A non-parametric statistical test that shuffles target labels to build a null distribution.
- **Null_Distribution**: The distribution of a test statistic (Pearson r) under the null hypothesis of no relationship.
- **LOVO_CV**: Leave-One-Video-Out Cross-Validation, the outer evaluation strategy used by the Pipeline.

## Requirements

### Requirement 1: Longer Epochs for Improved Spectral Resolution

**User Story:** As a researcher, I want to use 1-second epochs instead of 0.5-second epochs, so that the PSD estimation has 1 Hz frequency resolution (Δf = 500/500 = 1 Hz) instead of 2 Hz, enabling better separation of delta, theta, and alpha bands.

#### Acceptance Criteria

1. THE Config SHALL define EPOCH_DURATION_S as 1.0 seconds
2. THE Config SHALL define EPOCH_OVERLAP_S as 0.9 seconds to maintain the existing 0.1-second step size
3. THE Config SHALL define EPOCH_STEP_S as 0.1 seconds (unchanged)
4. WHEN extracting band-power features, THE Pipeline SHALL use nperseg equal to the number of samples in the epoch (500 at 500 Hz sfreq) for Welch's PSD estimation
5. WHEN the epoch duration changes, THE Pipeline SHALL produce epochs with 500 samples per epoch at 500 Hz sampling frequency

### Requirement 2: Z-Score Normalization of Luminance Target Per Video

**User Story:** As a researcher, I want to normalize luminance values to zero mean and unit variance within each video, so that scale differences between videos (e.g., video 3 mean=86.5 vs video 12 mean=111.0) do not bias the model.

#### Acceptance Criteria

1. WHEN luminance targets are collected for a video segment, THE Pipeline SHALL compute the mean and standard deviation of luminance values within that video segment
2. WHEN the standard deviation of a video segment's luminance is greater than zero, THE Pipeline SHALL transform each luminance value to (value − mean) / std
3. IF the standard deviation of a video segment's luminance is zero, THEN THE Pipeline SHALL set all normalized values to 0.0 for that segment
4. WHEN z-score normalization is applied, THE Pipeline SHALL produce luminance targets with mean approximately equal to 0.0 and standard deviation approximately equal to 1.0 within each video segment
5. THE Pipeline SHALL apply z-score normalization before the train/test split in LOVO_CV, so that normalization statistics are computed per video independently

### Requirement 3: Grid Search Over Ridge Alpha

**User Story:** As a researcher, I want to search over multiple Ridge alpha values using inner cross-validation within each training fold, so that the regularization strength is optimized rather than fixed at 1.0.

#### Acceptance Criteria

1. THE Config SHALL define RIDGE_ALPHA_GRID as a list of candidate values [0.01, 0.1, 1.0, 10.0, 100.0]
2. WHEN training a model in each outer LOVO_CV fold, THE Pipeline SHALL perform an inner cross-validation grid search over RIDGE_ALPHA_GRID to select the optimal alpha
3. WHEN the grid search completes, THE Pipeline SHALL use the alpha value that maximizes the inner CV score for the final model fit on that fold
4. WHEN reporting results, THE Pipeline SHALL log the selected alpha for each outer fold
5. THE Config SHALL retain RIDGE_ALPHA as a fallback default value

### Requirement 4: Permutation Test for Statistical Significance

**User Story:** As a researcher, I want to run a permutation test that shuffles the target labels and re-runs the pipeline, so that I can determine whether the observed Pearson r is statistically significant compared to chance.

#### Acceptance Criteria

1. THE Config SHALL define N_PERMUTATIONS as 100 (the number of permutation iterations)
2. WHEN running a permutation iteration, THE Pipeline SHALL shuffle the luminance target labels randomly while keeping features unchanged
3. WHEN shuffling targets, THE Pipeline SHALL shuffle within each video group to preserve the video-level structure of the LOVO_CV
4. WHEN all permutation iterations complete, THE Pipeline SHALL compute a null distribution of mean Pearson r values across folds
5. WHEN the null distribution is computed, THE Pipeline SHALL calculate a p-value as the proportion of permutation r values greater than or equal to the observed r
6. WHEN the permutation test completes, THE Pipeline SHALL save the null distribution and p-value to the results directory
7. WHEN the permutation test completes, THE Pipeline SHALL generate a histogram plot of the null distribution with the observed r marked

### Requirement 5: Backward Compatibility and Integration

**User Story:** As a researcher, I want the improvements to integrate cleanly into the existing pipeline scripts, so that the overall structure and output format remain consistent.

#### Acceptance Criteria

1. WHEN the Pipeline runs with updated parameters, THE Pipeline SHALL produce results CSV files in the same format as the current output
2. WHEN the Pipeline runs, THE Pipeline SHALL save all outputs to the existing results directory structure under `results/modeling/luminance/`
3. THE Pipeline SHALL maintain the existing LOVO_CV evaluation strategy as the outer cross-validation loop
