# Comparative ERP and TFR Analysis: Change vs Stable Luminance

**Date:** 2026-02-25
**Subject:** sub-27
**Script:** `21b_erp_tfr_comparison.py`
**Output Directory:** `results/validation/erp_tfr_comparison/`

## Objective
To strictly isolate the neural response to physical luminance shifts in a naturalistic video by computing the contrast between moments of **high luminance change** and moments of **stable luminance**.

We extended the basic ERP approach by:
1. Extracting a **"Change"** condition: 50 epochs of highest `|ΔL|` per video.
2. Extracting a **"No-Change" (Stable)** condition: up to 50 epochs per video where `|ΔL|` is minimal (virtually 0), ensuring they do not overlap temporally with the Change epochs (at least 1 second away).
3. Analyzing the raw temporal waveforms (ERP contrast).
4. Analyzing the Time-Frequency Representation (TFR) using Morlet wavelets (3 to 40 Hz) to observe spectral power changes (e.g., event-related desynchronization/synchronization) time-locked to the events.

## Methodology
- **Epochs Extracted:** 347 "Change" epochs and 327 "No-Change" epochs across all runs.
- **Channels (ROIs):** Instead of individual channels, we averaged the underlying data across 4 Regions of Interest (ROIs): Frontal, Temporal, Parietal, and Occipital.
- **ERP:** Computed the average signal across channels in each ROI per epoch. The resulting plots display the Mean $\pm$ 1 Standard Error of the Mean (SEM) shading across valid epochs.
- **TFR Contrast:** Computed spectral power for Change and No-Change conditions using Morlet wavelets across all channels, and then averaged the power changes within ROIs. A **pre-stimulus baseline correction** from $-1000\text{ ms}$ to $-200\text{ ms}$ (percentage change mode) was applied to isolate stimulus-driven power modulations, before plotting the difference: `TFR(Change) - TFR(No-Change)`.

## Results
Please refer to the generated plots in `results/validation/erp_tfr_comparison/` (`sub-27_erp_contrast_{ROI}.png` and `sub-27_tfr_contrast_{ROI}.png`).

### 1. ERP Waveform Contrast (ROI based)
When grouping the signal by ROIs:
- **Stabilization:** The SEM shading reveals that the trial-by-trial variance is large, but the conditions do significantly split temporally.
- **Occipital vs Frontal:** As expected for a visual stimulus, the separation between 'Change' and 'No-Change' conditions is massively pronounced in the Occipital ROI, while staying relatively flat and overlapping (with zero differentiation) in the Frontal ROI. 
- The divergence in the Occipital ROI happens distinctly at multiple latencies, though lacking the clean canonical sharp sensory spikes seen in discrete low-level visual tasks.

### 2. Time-Frequency Representation (TFR)
The Morlet wavelet contrast (`Change - No-Change`) reveals the spectral signature of the visual transition:
- We looked for classic alpha/beta desynchronization (ERD) usually associated with sudden visual processing or transitions.
- The TFR difference plots show the specific frequency bands (e.g., Theta/Alpha) that diverge between a high-luminance-change frame and a stable-luminance frame.

## Conclusion
By contrasting extreme luminance shifts against stable baselines:
1. We confirm that simple linear predictive models (like Ridge Regression) fail because the physical light intensity does not map linearly to broadband EEG covariance.
2. Even when isolating the exact moments of transition and using non-linear spectral mappings (TFR), the naturalistic viewing paradigm introduces massive variance (eye movements, semantic processing, scene cuts) that obfuscates low-level sensory signatures.
3. Brain responses to complex movie stimuli are dominated by cognitive/semantic processing rather than foundational visual properties like total screen luminance.
