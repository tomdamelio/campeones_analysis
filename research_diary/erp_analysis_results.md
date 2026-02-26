# ERP Analysis: Luminance Changes

**Date:** 2026-02-25
**Subject:** sub-27
**Script:** `21_erp_luminance_changes.py`
**Output Directory:** `results/validation/erp/`

## Objective
To investigate whether steep absolute changes in on-screen luminance elicit a time-locked Event-Related Potential (ERP) in the visual cortex (occipital and posterior ROI channels).

Unlike the continuous Ridge/Logistic regression models that aggregated temporal dynamics into a 500ms covariance window, the ERP analysis averages the **raw broadband signal** precisely timestamped to the occurrence of the largest luminance fluctuations (`|ΔL|`), looking for canonical visual evoked potentials (e.g., P100, N170, P300).

## Methodology
- **Events:** The script identified the top 50 moments of largest absolute luminance shift within each video stimulus.
- **Epoching:** MNE Epochs were created matching those exact times, with a window of `[-200 ms, +800 ms]` relative to the change event.
- **Channels Included:** `O1`, `O2`, `P3`, `P4`, `P7`, `P8`, `Pz`, `CP1`, `CP2`, `CP5`, `CP6` (Posterior ROI).
- **Processing:** Epochs were averaged across all runs and videos to compute the Grand-Average ERP (N = 350 total epochs after validation/clipping).

## Results: Grand Average
Analysis of the `sub-27_erp_peak_amplitudes.csv` yields the following peak metrics for primary occipital sensors:

| Channel | Peak Latency (ms) | Peak Amplitude (µV) | Mean Amp [100-300 ms] (µV) |
| :--- | :--- | :--- | :--- |
| **O1** | 632.0 | 2.54 | -0.89 |
| **O2** | 632.0 | 2.58 | -1.41 |
| **Pz** | 536.0 | 2.11 | 0.20 |

### Key Observations
1. **Absence of strong early visual components:** Canonical Visual Evoked Potentials to sudden contrasting luminance usually evoke sharp responses within 100-200ms (like the P100). Our average amplitude in the `100-300 ms` window is negative (`~ -1µV`), but doesn't exhibit the sharp high-amplitude positive peaking typical of a robust low-level luminance flash.
2. **Late peaking (600+ ms):** The predominant peak occurs very late, at `~632 ms` with an amplitude of `~2.5 µV`. This late positive component resembles a cognitive ERP (like the P300/Late Positive Potential) related to processing the video content, rather than an automatic low-level sensory response to the light passing a threshold.
3. **Low Amplitude Magnitude:** A peak of `~2.5 µV` in an uncorrected average of a naturalistic free-viewing task is relatively low. 

## Conclusion
The data suggests that taking random samples of "steepest luminance changes" natively embedded within a continuous, immersive stimulus (like the Campeones video) does **not** evoke a reliable or prominent early sensory ERP in subject 27 comparable to what would be seen from a discrete, blank-screen checkerboard or flash paradigm.

Because the visual content is highly clustered, naturalistic, and continuous, strict time-locked sensory responses are likely dispersed, washed out by temporal jitter, or suppressed by higher-order cognitive processing and spatial attention mechanisms. This reinforces the previous findings: the raw intensity of light is exceptionally difficult to decode structurally or temporally from naturalistic continuous EEG data.

## Update (2026-02-26): Reconciliation with Corrected TDE Results
After fixing a critical per-video PCA bug and optimizing the number of components (50 → 20), the TDE covariance model achieves **r ≈ 0.12** for continuous luminance prediction (see `raw_tde_model_analysis.md`). This weak but genuine correlation suggests that while 500ms covariance windows cannot capture the transient ERP-like responses documented here (~632ms peak), they do retain some **graded, slow** luminance-related variance. The ERP result remains consistent: fast transient responses are washed out by covariance averaging, but slow tonic signals survive.

