# Delta Luminance Model Analysis

**Date:** 2026-02-25
**Subject:** sub-27
**Script:** `19_delta_luminance_model.py`

## Objective
Following the realization that absolute physical luminance ($L_i$) fails to be predicted by linear covariance modeling (Pearson $r \approx 0.02$), we hypothesized that the visual cortex might encode the *change* or derivative of luminance ($\Delta L = L_i - L_{i-1}$) rather than absolute states, reflecting processes like optical flow or transient sensory adaptation.

We evaluated two variants of the target variable:
1. `delta_raw`: The raw unnormalized differenced luminance values.
2. `delta_zscore`: The same differenced values, but z-score normalized per video to correct for varying scales of baseline changes across levels.

## Methodology
- **Input Data**: 11 Posterior channels, `±10` TDE lags, `TDE_PCA_COMPONENTS = 20`.
- **Epoching**: 500 ms length, 100 ms step. 
- **QA**: AutoReject used for 3775 clean available epochs. The *very first* epoch of each continuous video segment was discarded to allow for purely valid $L_i - L_{i-1}$ arithmetic without crossing borders.
- **Model pipeline**: `StandardScaler() -> GridSearchCV(Ridge())` optimized for Spearman correlation.
- **Evaluation**: Leave-One-Video-Out CV (LOVO-CV). 

## Results

| Variant | Mean Train R² | Mean Test R² | Mean Test Pearson r | Mean Test Spearman ρ | Mean Test RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `delta_raw` | N/A | -10.509 | 0.0294 | 0.0390 | 3.0968 |
| `delta_zscore` | N/A | -3.0650 | 0.0336 | 0.0365 | 1.9960 |

*(Note: Maximum Fold Spearman ρ for z-score was `0.148` on video 9_a, but others fluctuated near zero).*

## Interpretation

### 1. No Evident Linear Encoding of Continuous Delta Luminance
Just like with the absolute luminance model, predicting the pure mathematically-continuous derivatives of luminance over 500 ms windows produces entirely negligible correlations across holding-out videos. A mean $r \approx 0.03$ suggests the model has virtually zero capacity to linearly map the spatial covariance matrix of occipital regions to a generic "speed of lighting change" variable.

### 2. High Variance in Out-Of-Sample Fits (Catastrophic R²)
The severely negative R² scores (-10.5 and -3.06) show the predictions are significantly worse than just guessing the training mean. The model learns completely spurious covariance signatures inside one group of videos that actively misguide it when tested on a new video. Z-scoring helps constrain the mathematical explosion (from RMSE 3.09 to 1.99) but doesn't fix the underlying lack of valid predictive structure.

### 3. Transition to Non-Linear or Categorical Tasks
We are running out of simple continuous representations. The negative answers here reinforce the need for either:
- **Change Classifier (`20_change_classifier.py`)**: Treating luminance changes purely as discrete, binary sensory events (Stable vs. High Change/Event).
- **HMM State-Space (`12_luminance_spectral_tde_model.py` / GLHMM iteration)**: Using actual Markov states to classify the non-linear configurations of the brain instead of forcing a continuous Ridge Regression map.
