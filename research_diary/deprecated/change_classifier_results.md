# Change Classifier Analysis

**Date:** 2026-02-25 (updated 2026-02-26)
**Subject:** sub-27
**Script:** `20_change_classifier.py`

## Objective
Binary classification: can TDE covariance features distinguish "luminance change" from "stable" epochs?

## Final Configuration (2026-02-26)
- **Global PCA** (20 components, 93.4% variance)
- **PC standardization** (`standardise_pc=True`, glhmm canonical)
- Logistic Regression with random undersampling (extreme class imbalance: 94 vs 3681 epochs)

## Results

| Metric | Mean |
|:---|:---:|
| AUC-ROC | **0.430** |
| Accuracy | 0.609 |
| Precision | 0.032 |
| Recall | 0.323 |
| F1 | 0.058 |

AUC < 0.5 indicates performance **worse than chance** (likely noise from extreme imbalance + small positive class).

### Per-fold Detail
| Fold | Acc | Prec | Rec | F1 | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| 12_a | 0.672 | 0.051 | 0.474 | 0.092 | 0.632 |
| 12_b | 0.644 | 0.015 | 0.188 | 0.028 | 0.393 |
| 3_a | 0.567 | 0.013 | 0.333 | 0.024 | 0.478 |
| 7_a | — | — | — | — | — (0 positives) |
| 7_b | 0.593 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9_a | 0.611 | 0.052 | 0.360 | 0.091 | 0.489 |
| 9_b | 0.567 | 0.063 | 0.583 | 0.114 | 0.588 |

## Interpretation
1. **Change detection is not feasible** with TDE covariance + linear classifier.
2. Video 7 has 0–1 change events in test, making evaluation impossible for those folds.
3. The extreme class imbalance (2.5% positive) severely limits the classifier.
4. Covariance features capture continuous/graded dynamics, not discrete events.
