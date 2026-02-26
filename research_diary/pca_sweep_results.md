# PCA Sweep Analysis

**Date:** 2026-02-25 (updated 2026-02-26)
**Subject:** sub-27
**Script:** `18_pca_sweep.py`

## Objective
Determine the optimal number of PCA components for the TDE-covariance pipeline.

## Final Configuration (2026-02-26)
- **Global PCA** fitted on concatenated TDE data from all segments
- **PC standardization** (`standardise_pc=True`) matching glhmm canonical protocol
- Sweep range: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

## Results

| n_components | Var. Explained | Feature Dim | Mean r | Mean R² | Mean ρ | Mean RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10** | 85.3% | 55 | 0.103 | **-0.023** | 0.097 | **1.032** |
| **20** | 93.4% | 210 | **0.111** | -0.061 | **0.128** | 1.050 |
| **30** | 96.3% | 465 | **0.112** | -0.100 | 0.119 | 1.069 |
| 40 | 98.0% | 820 | 0.072 | -0.203 | 0.081 | 1.114 |
| 50 | 98.8% | 1275 | 0.075 | -0.184 | 0.072 | 1.107 |
| 60 | 99.3% | 1830 | 0.075 | -0.229 | 0.070 | 1.131 |
| 70 | 99.6% | 2485 | 0.092 | -0.283 | 0.088 | 1.154 |
| 80 | 99.8% | 3240 | 0.033 | -3.091 | 0.039 | 1.825 |
| 90 | 99.9% | 4095 | 0.039 | -0.870 | 0.037 | 1.371 |
| 100 | 99.9% | 5050 | 0.028 | -1.360 | 0.006 | 1.545 |

## Key Findings

1. **Optimal range: 10–30 components**. Beyond 30, performance degrades from overfitting.
2. **Best correlation (r/ρ):** n=20–30 (r≈0.111, ρ≈0.12–0.13).
3. **Best R²:** n=10 (R²=-0.023, closest to 0). This reflects lower overfitting with fewer features (55).
4. **Catastrophic overfitting at n≥80:** R² collapses to -3.1 (3240 features >> 7-fold CV budget).
5. **Recommendation:** `n=20` balances correlation signal with generalization (210 features, 93.4% variance).
