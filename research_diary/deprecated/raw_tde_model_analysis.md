# Raw TDE Luminance Model Analysis

**Date:** 2026-02-25 (updated 2026-02-26)
**Subject:** sub-27
**Script:** `13_luminance_raw_tde_model.py`

## Objective
Evaluate the predictive performance of the GLHMM TDE protocol (Vidaurre et al., 2025) applied directly to the continuous preprocessed EEG before extracting epoch-level covariance features.

## Critical Bug Fix: Per-Video → Global PCA (2026-02-26)

An audit against the `glhmm` library source code revealed that PCA was being fitted **independently per video segment**, meaning each video's covariance features lived in a different coordinate system. This made cross-video regression fundamentally ill-posed.

**Fix:** Refactored the pipeline into a two-pass approach:
1. **Pass 1:** Collect TDE-embedded data from all 7 video segments and fit **one global PCA**.
2. **Pass 2:** Project each segment into the shared PCA subspace, then epoch and extract covariance features.

Additionally, based on PCA sweep results (see `pca_sweep_results.md`), `TDE_PCA_COMPONENTS` was reduced from `50` to `20`.

## Methodology
- **Input Channels**: 11 Posterior channels (`O1`, `O2`, `P3`, `P4`, `P7`, `P8`, `Pz`, `CP1`, `CP2`, `CP5`, `CP6`).
- **TDE Parameters**: `±10` lags (21 time-points per channel). **Global PCA** at `n=20` components (93.4% variance explained).
- **Epoching**: 500 ms duration, 100 ms step. Bad epochs rejected via AutoReject parameters.
- **Features**: Upper triangle of the 20-component covariance matrix per epoch ($20 \times 21 / 2 = 210$ features).
- **Target**: Z-score normalized continuous physical luminance (interpolated to epoch onsets).
- **Evaluation**: Leave-One-Video-Out Cross-Validation (LOVO-CV).
- **Model**: `StandardScaler() → Ridge()`. Alpha selection via `GridSearchCV` (scoring: Spearman ρ) with `LeaveOneGroupOut`.

## Results Comparison

### Original Results (Per-Video PCA, 50 components — BUGGY)
| TestVideo | TrainPearsonR | Test R² | Test PearsonR | RMSE | BestAlpha |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **12_a** | 0.9300 | -2.0854 | 0.0067 | 1.7565 | 0.01 |
| **12_b** | 0.9332 | -2.5855 | 0.2257 | 1.8935 | 0.01 |
| **3_a** | 0.9414 | -2.8022 | -0.0340 | 1.9499 | 0.01 |
| **7_a** | 0.9392 | -1.8394 | 0.0230 | 1.6851 | 10.0 |
| **7_b** | 0.7813 | -0.0834 | -0.0370 | 1.0409 | 10000.0 |
| **9_a** | 0.8821 | -0.3030 | 0.0136 | 1.1415 | 1000.0 |
| **9_b** | 0.7820 | -0.0614 | -0.0114 | 1.0302 | 10000.0 |

**Mean:** Train r = 0.884 | Test r = 0.027 | R² = -1.394 | RMSE = 1.500

### Updated Results (Global PCA, 20 components — FIXED)
| TestVideo | TrainPearsonR | Test R² | Test PearsonR | Test SpearmanRho | RMSE | BestAlpha |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **12_a** | 0.4215 | -0.0258 | 0.1974 | 0.1986 | 1.0128 | 100.0 |
| **12_b** | 0.3866 | **0.0267** | 0.2239 | 0.2578 | 0.9865 | 1000.0 |
| **3_a** | 0.4695 | -0.3057 | -0.0018 | 0.0318 | 1.1427 | 100.0 |
| **7_a** | 0.4156 | -0.0239 | 0.0894 | 0.0755 | 1.0119 | 1000.0 |
| **7_b** | 0.4064 | -0.0326 | 0.1186 | 0.1116 | 1.0162 | 1000.0 |
| **9_a** | 0.4131 | -0.0134 | 0.1196 | 0.1553 | 1.0067 | 1000.0 |
| **9_b** | 0.4248 | -0.0656 | 0.0631 | 0.0894 | 1.0323 | 1000.0 |

**Mean:** Train r = 0.420 | **Test r = 0.116** | **R² = -0.063** | **RMSE = 1.030**

## Interpretation

### 1. Global PCA Fix Dramatically Reduced Overfitting
The train-test gap collapsed from 0.86 (0.88 − 0.02) to 0.30 (0.42 − 0.12). The model no longer memorizes video-specific PCA artefacts.

### 2. First Evidence of Genuine Cross-Video Signal
- **6 of 7 folds have positive test r** (range 0.06–0.22).
- **Fold 12_b achieved R² > 0** (0.027) — the first time any fold's predictions outperform the mean constant baseline.
- Mean test Spearman ρ = 0.131, confirming monotonic rank preservation of luminance predictions.

### 3. Video 3 Remains the Outlier
Video 3 (r ≈ 0, R² = -0.31) remains unpredictable. This video may have fundamentally different visual characteristics incompatible with the linear model's learned mapping from other videos.

### 4. Key Parameter: 20 PCA Components
The PCA sweep (`18_pca_sweep.py`) showed that the optimum is at 10–30 components. At 50 components (original config), overfitting obscured any signal. At 20, the feature/sample ratio is 210:3200 ≈ 1:15, which is manageable for Ridge regression.

### 5. Revised Conclusion
The TDE covariance features **do contain weak but genuine linear information about luminance** when the pipeline is correctly implemented (global PCA, low dimensionality). However, the signal is modest (r ≈ 0.12), suggesting that either: (a) the covariance representation captures only a small fraction of the luminance-related neural dynamics, or (b) cross-video generalization inherently limits performance because different videos evoke fundamentally different visual processing patterns.
