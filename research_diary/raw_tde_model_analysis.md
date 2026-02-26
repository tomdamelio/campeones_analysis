# Raw TDE Luminance Model Analysis

**Date:** 2026-02-25
**Subject:** sub-27
**Script:** `13_luminance_raw_tde_model.py`

## Objective
Evaluate the baseline predictive performance of the GLHMM TDE protocol (Vidaurre et al., 2025) applied directly to the continuous preprocessed EEG before extracting epoch-level covariance features. The goal was to establish the model's accuracy before attempting dimensionality reduction via PCA sweeping. 

## Methodology
- **Input Channels**: 11 Posterior channels (`O1`, `O2`, `P3`, `P4`, `P7`, `P8`, `Pz`, `CP1`, `CP2`, `CP5`, `CP6`).
- **TDE Parameters**: `±10` lags (21 time-points per channel). Global pre-PCA at `n=50` components.
- **Epoching**: 500 ms duration, 100 ms step. Bad epochs rejected via AutoReject parameters.
- **Features**: Upper triangle of the 50-component covariance matrix per epoch ($50 \times 51 / 2 = 1275$ features).
- **Target**: Z-score normalized continuous physical luminance (interpolated to epoch onsets).
- **Evaluation**: Leave-One-Video-Out Cross-Validation (LOVO-CV).
- **Model**: `StandardScaler() -> Ridge()`. Alpha selection via `GridSearchCV` (scoring: Spearman ρ) with `LeaveOneGroupOut`.

## Results
A total of **3782 clean epochs** were analyzed across 7 test folds. 

### Metrics Breakdown
| TestVideo | TrainSize | TestSize | TrainPearsonR | Test R² | Test PearsonR | Test SpearmanRho | RMSE | BestAlpha |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **12_a** | 3241 | 541 | 0.9300 | -2.0854 | 0.0067 | -0.0763 | 1.7565 | 0.01 |
| **12_b** | 3188 | 594 | 0.9332 | -2.5855 | 0.2257 | 0.1822 | 1.8935 | 0.01 |
| **3_a** | 3225 | 557 | 0.9414 | -2.8022 | -0.0340 | -0.0676 | 1.9499 | 0.01 |
| **7_a** | 3224 | 558 | 0.9392 | -1.8394 | 0.0230 | -0.0293 | 1.6851 | 10.0 |
| **7_b** | 3216 | 566 | 0.7813 | -0.0834 | -0.0370 | -0.0223 | 1.0409 | 10000.0 |
| **9_a** | 3318 | 464 | 0.8821 | -0.3030 | 0.0136 | -0.0081 | 1.1415 | 1000.0 |
| **9_b** | 3280 | 502 | 0.7820 | -0.0614 | -0.0114 | -0.0693 | 1.0302 | 10000.0 |

**Mean Metrics:**
- **Train r**: 0.8842
- **Test R²**: -1.3943
- **Test r**: 0.0267
- **Test Spearman ρ**: -0.0130
- **Test RMSE**: 1.4997

## Interpretation & Next Steps
1. **Severe Overfitting**: The extremely high `TrainPearsonR` (mean 0.88) coupled with negligible `TestPearsonR` (mean 0.02) denotes massive overfitting. The model easily memorizes the training data but fails to generalize.
2. **Curse of Dimensionality**: The feature vector contains 1275 values per epoch, whereas the entire training set size is around ~3200 epochs. A feature-to-sample ratio of nearly 1:3 puts a huge strain on the Ridge regularizer. While extreme alphas (e.g. 10000) mitigate this slightly (see folds `7_b` and `9_b` where `R²` approaches 0), it squashes predictions to the mean.
3. **Necessity of PCA Sweeping**: The results perfectly justify the use of `18_pca_sweep.py`. The static `TDE_PCA_COMPONENTS=50` parameter yields an intractable covariance matrix for prediction. By iteratively sweeping the number of global PCA components, we can isolate the underlying neurophysiological variance corresponding to luminance without swamping the regressor in high-dimensional covariance noise.
