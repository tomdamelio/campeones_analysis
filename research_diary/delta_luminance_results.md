# Delta Luminance Model Analysis

**Date:** 2026-02-25 (updated 2026-02-26)
**Subject:** sub-27
**Script:** `19_delta_luminance_model.py`

## Objective
Test whether TDE covariance features predict the *change* in luminance (ΔL = L_i − L_{i-1}) rather than absolute luminance levels.

## Final Configuration (2026-02-26)
- **Global PCA** (20 components, 93.4% variance)
- **PC standardization** (`standardise_pc=True`, glhmm canonical)
- Two variants: `delta_raw` (unnormalized) and `delta_zscore` (z-scored per video)

## Results

| Variant | Mean R² | Mean r | Mean ρ | Mean RMSE |
|:---|:---:|:---:|:---:|:---:|
| `delta_raw` | -0.521 | 0.002 | -0.016 | 1.590 |
| `delta_zscore` | **-0.176** | **0.007** | **-0.005** | **1.084** |

All correlations are indistinguishable from zero. The model has no predictive power for delta luminance.

## Interpretation
1. **Delta luminance is not linearly decodable** from TDE covariance features.
2. **Comparison with absolute luminance:** The same pipeline achieves r≈0.113 for absolute luminance (z-scored) — confirming that the covariance captures slow/tonic modulations, not fast transients.
3. **Consistent with ERP findings:** Transient luminance changes produce peaked ERPs (~632ms) that are averaged away by the 500ms covariance window.
4. **Alpha instability** (BestAlpha fluctuates wildly across folds) confirms absence of stable predictive structure.
