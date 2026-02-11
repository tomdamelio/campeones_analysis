# Implementation Plan: Luminance Model Improvements

## Overview

Incremental improvements to the existing EEG luminance prediction pipeline: longer epochs, z-score normalization, Ridge alpha grid search, and permutation testing. All changes modify existing files or add minimal new modules. Python with scikit-learn, scipy, hypothesis.

## Tasks

- [x] 1. Update config and epoch parameters
  - [x] 1.1 Update `scripts/modeling/config_luminance.py` with new epoch and ML parameters
    - Change `EPOCH_DURATION_S` from 0.5 to 1.0
    - Change `EPOCH_OVERLAP_S` from 0.4 to 0.9
    - Add `RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]`
    - Add `N_PERMUTATIONS = 100`
    - Keep `EPOCH_STEP_S = 0.1` and `RIDGE_ALPHA = 1.0` unchanged
    - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.5, 4.1_
  - [x] 1.2 Update `extract_bandpower` in `src/campeones_analysis/luminance/features.py`
    - Change `nperseg=min(eeg_epoch.shape[1], 256)` to `nperseg=eeg_epoch.shape[1]`
    - _Requirements: 1.4, 1.5_
  - [x] 1.3 Update Property 7 test in `tests/test_luminance_features.py` to verify full nperseg
    - **Property 1: Spectral resolution matches epoch length**
    - **Validates: Requirements 1.4**

- [x] 2. Implement z-score normalization per video
  - [x] 2.1 Create `src/campeones_analysis/luminance/normalization.py`
    - Implement `zscore_per_video(epoch_entries, video_key, target_key) -> list[dict]`
    - Pure function: groups by video_key, computes mean/std per group, transforms target_key
    - Handle zero-std edge case (set to 0.0)
    - Export in `src/campeones_analysis/luminance/__init__.py`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 2.2 Write property test for z-score normalization in `tests/test_luminance_normalization.py`
    - **Property 2: Z-score normalization produces zero mean and unit variance per video**
    - **Validates: Requirements 2.2, 2.3, 2.4**
  - [x] 2.3 Write unit tests for edge cases in `tests/test_luminance_normalization.py`
    - Test empty list input returns empty list
    - Test single-value video group returns 0.0
    - _Requirements: 2.3_

- [x] 3. Checkpoint — Verify normalization module
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement permutation test module
  - [x] 4.1 Create `src/campeones_analysis/luminance/permutation.py`
    - Implement `shuffle_targets_within_videos(epoch_entries, rng, video_key, target_key) -> list[dict]`
    - Implement `compute_p_value(null_distribution, observed_r) -> float`
    - Implement `run_permutation_test(epoch_entries, build_and_evaluate_fn, n_permutations, random_seed, video_key, target_key) -> dict`
    - Implement `plot_permutation_histogram(null_distribution, observed_r, p_value, output_path) -> None`
    - Export in `src/campeones_analysis/luminance/__init__.py`
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x] 4.2 Write property test for shuffle invariant in `tests/test_luminance_permutation.py`
    - **Property 4: Permutation shuffle preserves within-video target multisets and features**
    - **Validates: Requirements 4.2, 4.3**
  - [x] 4.3 Write property test for null distribution size in `tests/test_luminance_permutation.py`
    - **Property 5: Null distribution has correct length**
    - **Validates: Requirements 4.4**
  - [x] 4.4 Write property test for p-value computation in `tests/test_luminance_permutation.py`
    - **Property 6: P-value equals proportion of null values ≥ observed**
    - **Validates: Requirements 4.5**

- [x] 5. Checkpoint — Verify permutation module
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Update model scripts with all improvements
  - [x] 6.1 Update `scripts/modeling/10_luminance_base_model.py`
    - Import `zscore_per_video` and `run_permutation_test`, `plot_permutation_histogram`
    - Call `zscore_per_video()` on epoch entries after collection, before CV
    - Replace `Ridge(alpha=RIDGE_ALPHA)` with `GridSearchCV` over `RIDGE_ALPHA_GRID` (inner 3-fold CV)
    - Log `BestAlpha` in results DataFrame
    - Add permutation test call after CV loop, save null distribution + p-value + histogram
    - _Requirements: 2.5, 3.2, 3.3, 3.4, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3_
  - [x] 6.2 Update `scripts/modeling/11_luminance_spectral_model.py`
    - Same changes as 6.1: normalization, grid search, permutation test
    - _Requirements: 2.5, 3.2, 3.3, 3.4, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3_
  - [x] 6.3 Update `scripts/modeling/12_luminance_tde_model.py`
    - Same changes as 6.1: normalization, grid search, permutation test
    - Note: normalization applied before TDE (on spectral epoch entries)
    - _Requirements: 2.5, 3.2, 3.3, 3.4, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3_
  - [x] 6.4 Write property test for grid search alpha selection in `tests/test_luminance_pipeline.py`
    - **Property 3: Grid search selects alpha from the configured grid**
    - **Validates: Requirements 3.2**

- [x] 7. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All new modules follow pure-function design (no I/O in computation logic)
- The permutation test uses a callback pattern (`build_and_evaluate_fn`) so it works with all 3 model types
- Config changes in task 1.1 affect all three model scripts immediately (shared import)
- Hypothesis is already available in the environment for property-based testing
