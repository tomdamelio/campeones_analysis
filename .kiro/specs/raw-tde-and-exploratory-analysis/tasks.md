# Implementation Plan: Raw TDE and Exploratory Analysis

## Overview

Implement Script 13 (raw TDE model) and Script 14 (distribution explorer) following the existing patterns from scripts 10–12. Script 13 introduces a new `apply_tde_on_continuous_signal` function and a new epoch extraction flow. Script 14 reuses existing data loading patterns and adds joystick-based dimension extraction.

## Tasks

- [x] 1. Implement the raw TDE epoch extraction function
  - [x] 1.1 Create `scripts/modeling/13_luminance_raw_tde_model.py` with imports, logger setup, and helper functions (`_resolve_events_path`, `_resolve_eeg_path`, `_resolve_order_matrix_path`, `select_roi_channels`, `leave_one_video_out_split`, `evaluate_fold`) reused from script 12
    - _Requirements: 1.1, 2.2_
  - [x] 1.2 Implement `apply_tde_on_continuous_signal(eeg_data, window_half)` that transposes the EEG matrix to (n_samples, n_channels), calls `apply_time_delay_embedding`, and returns the TDE-expanded matrix
    - _Requirements: 1.2, 1.3_
  - [x] 1.3 Write property test for TDE output shape invariant
    - **Property 1: TDE output shape invariant**
    - **Validates: Requirements 1.2, 1.3**
  - [x] 1.4 Implement `extract_raw_tde_epochs_for_run(run_config, eeg_raw, events_df, roi_channels)` that: crops EEG to video_luminance segments, applies TDE on continuous signal, applies PCA, epochs the PCA time-series, pairs with luminance targets
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  - [x] 1.5 Write property test for PCA preserves time-points
    - **Property 2: PCA preserves time-points and reduces features**
    - **Validates: Requirements 1.4**
  - [x] 1.6 Write property test for epoch count from PCA time-series
    - **Property 3: Epoch count from PCA time-series**
    - **Validates: Requirements 1.5**

- [x] 2. Implement the CV pipeline and results output for Script 13
  - [x] 2.1 Implement `run_pipeline()` orchestrating: seed setup, ROI selection, epoch collection across runs, z-score normalization, LOVO-CV with StandardScaler → Ridge (GridSearchCV), permutation test
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.4_
  - [x] 2.2 Implement `plot_cv_results`, `plot_predictions_per_fold`, and `plot_comparison_with_spectral_tde` following the same patterns as script 12
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 2.3 Add CSV saving with the standard results schema (Subject, Acq, Model="raw_tde", TestVideo, TrainSize, TestSize, PearsonR, SpearmanRho, RMSE, BestAlpha)
    - _Requirements: 3.1_

- [x] 3. Checkpoint — Ensure Script 13 is complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement the distribution explorer (Script 14)
  - [x] 4.1 Create `scripts/modeling/14_explore_target_distributions.py` with imports, logger setup, and data loading helpers that read Merged Events TSVs and Order Matrix files for all configured runs
    - _Requirements: 4.1_
  - [x] 4.2 Implement `collect_dimension_values` that iterates runs, matches video events to Order Matrix rows by dimension, extracts epoch-level mean values (luminance from CSV, joystick from EEG) with polarity correction
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 4.3 Write property test for polarity correction negation
    - **Property 4: Polarity correction is negation**
    - **Validates: Requirements 4.3**
  - [x] 4.4 Implement `plot_distribution` and `run_pipeline` that generates raw and z-score-normalized histograms per dimension with descriptive stats annotations, saves to `results/modeling/luminance/exploration/distributions/`
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

- [x] 5. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using `hypothesis`
- Script 13 follows the same patterns as scripts 10–12 for consistency
- All functions use type hints and Google-style docstrings per project steering rules
