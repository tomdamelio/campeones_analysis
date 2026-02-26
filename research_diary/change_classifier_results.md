# Change Classifier Analysis

**Date:** 2026-02-25
**Subject:** sub-27
**Script:** `20_change_classifier.py`

## Objective
Following the failure of continuous Ridge Regression to predict both Absolute Luminance ($L$) and Delta Luminance ($\Delta L$), we tested the hypothesis that the brain treats luminance changes as **discrete sensory events**. 

We defined a binary classification task:
- **Class 0 (Stable):** $|\Delta L| \leq 5.0$
- **Class 1 (Change):** $|\Delta L| > 5.0$

## Methodology
- **Input Data**: 11 Posterior channels, `±10` TDE lags, `TDE_PCA_COMPONENTS = 20`.
- **Epoching**: 500 ms length, 100 ms step. AutoReject applied.
- **Model pipeline**: `StandardScaler() -> LogisticRegression(solver="lbfgs")`.
- **Handling Imbalance**: The dataset was heavily imbalanced (3681 Stable vs 94 Change epochs). We used **random undersampling** on the training folds only, keeping a 50/50 balance of class 0/1 during `fit()`, while explicitly leaving the test fold unmodified to reflect real-world predictive capacity.
- **Evaluation**: Leave-One-Video-Out CV (LOVO-CV). Metrics: Accuracy, Precision, Recall, F1, AUC-ROC.

## Results
Out of 7 test folds, one fold (`7_a`) had 0 "Change" epochs leading to NaN precision/recall metrics. For the other 6 folds:

| Metric | Mean (Over 6 Valid Folds) |
| :--- | :--- |
| **Accuracy** | 0.8391 |
| **Precision** | 0.0461 |
| **Recall** | 0.1213 |
| **F1-Score** | 0.0646 |
| **AUC-ROC** | 0.5442 |

## Interpretation

### 1. No Evident Discriminative Power (AUC is random)
An **AUC-ROC of 0.544** is virtually identical to a completely random coin toss (0.50). Despite undersampling the training set to prevent the model from collapsing into "always predict stable", the logistic regression finds no generalizable hyperplane in the TDE-Covariance space that separates a 500ms epoch containing a "light flash" from a 500ms epoch of stable lighting.

### 2. High Accuracy is Deceptive
The high apparent Accuracy (~0.84) is a mathematical artifact of testing on the natural, imbalanced distribution. Because `>95%` of the test epochs are Class 0 (Stable), a model that acts almost randomly but slightly biases towards 0, or just predicts 0 most of the time, will achieve high accuracy while completely failing to detect Class 1, as evidenced by the abysmal Precision (0.04) and Recall (0.12).

### Conclusion
We have now definitively tested GLHMM TDE covariance features against:
1. Continuous Absolute Luminance (r=0.02)
2. Continuous Delta Luminance (r=0.03)
3. Binary Discretized Luminance Change Events (AUC=0.54)

All three linear mapping assumptions hold zero out-of-sample validity on sub-27. The GLHMM time-delay covariance matrix of the occipital sensors over 500ms windows does not contain any linearly decodable representation of physical on-screen light intensity or its transitions. If light is encoded, it is either strictly non-linear (e.g., specific combinations of discrete HMM state networks, independent of linear covariance magnitude), heavily localized in the time-domain (ERPs locked to exact frame changes, wiped out by 500ms covariance averaging), or confounded by higher-order cognitive visual processing in a naturalistic task.
