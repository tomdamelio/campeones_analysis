# EEG QA AutoReject Analysis

**Date:** 2026-02-25
**Subject:** sub-27 (VR Luminance Task)

## Objective
Evaluate the impact of relaxing the AutoReject `consensus` parameter grid from `[0.25, 1.0]` to `[0.5, 1.0]` to prevent excessive data loss during QA, while maintaining the maximum interpolation limit at 8 channels (25%).

## Results and Interpretation (Consensus 0.5)
The pipeline was run exclusively on Subject 27. The rejection rate dropped significantly from **28.1%** to **9.3%**, recovering a substantial amount of training data.

### Run-by-Run Breakdown
*   **Run 002 (Video 12):** 9.1% rejection | max_interp: 8 | consensus: 0.50
*   **Run 003 (Video 9):** 22.1% rejection | max_interp: 8 | consensus: 0.50
*   **Run 004 (Video 3):** 6.4% rejection | max_interp: 8 | consensus: 0.55
*   **Run 006 (Video 7):** 6.4% rejection | max_interp: 8 | consensus: 0.55
*   **Run 007 (Video 12):** 0.2% rejection | max_interp: 8 | consensus: 0.50
*   **Run 009 (Video 9):** 15.8% rejection | max_interp: 8 | consensus: 0.50
*   **Run 010 (Video 7):** 4.9% rejection | max_interp: 8 | consensus: 0.55

### Deep Dive Insights
1. **Interpolation Limit (`n_interp`):** AutoReject consistently selected the maximum allowed interpolation limit (8 channels) across all runs. It strongly favors repairing up to 25% of the channels over discarding entire epochs.
2. **Consensus Threshold (`consensus`):** AutoReject anchored at the minimum allowed consensus (`0.50`) for the noisier runs (002, 003, 007, 009) and found an optimal point at `0.55` for the cleaner runs (004, 006, 010). This indicates that the `[0.5, 1.0]` grid is highly appropriate for this VR dataset.
3. **The "Run 004" Recovery:** Run 004 previously suffered a 54.5% data loss at `0.25` consensus. With the relaxed consensus of `0.55` and `n_interp=8`, AutoReject successfully repaired the noisy epochs via heavy interpolation, reducing the data loss to an excellent 6.4%.
4. **Spatial Distribution of Artifacts:**
   * **Frontal Channels (Fp1, Fp2, F7, F8):** Frequently interpolated, strongly pointing to ocular artifacts (heavy blinking, eyebrow movement) typical of wearing a VR headset.
   * **Muscular/Temporal Channels (FT9, FT10, CP5, CP6, TP9, TP10):** Frequently interpolated, likely due to jaw clenching or neck movements during VR orientation.
   * **Central/Motor Channels (Cz, C3, C4):** Remained surprisingly clean, preserving the core cortical signal for latent variable prediction.
5. **Temporal Distribution of Artifacts:** Rejected epochs occur in continuous "trains" rather than isolated spikes (e.g., Run 003 rejecting epochs 257-267 continuously). This implies gross physical artifacts (adjusting the headset, coughing) that corrupt too many channels simultaneously (>16) for AutoReject to repair.

---

## Future Pending Action Items
Based on these findings, there are two major architectural/methodological points to revisit in the future:

1. **Re-evaluate Preprocessing for Ocular Artifacts:** The heavy reliance on interpolating frontal channels (`Fp1/2`, `F7/8`) suggests that the current ocular artifact removal step in the preprocessing pipeline (e.g., ICA) might be insufficient. We should investigate if we can tune the ICA to better isolate and remove blink/VR-fit artifacts, which would reduce AutoReject's burden.
2. **Shift AutoReject to Preprocessing:** Currently, AutoReject is implemented as a post-processing Quality Assurance step (`scripts/qa/16_eeg_qa_autoreject.py`). Given its effectiveness at repairing epochs, we should consider integrating AutoReject directly into the core preprocessing pipeline sequence (`scripts/preprocessing/`) so that all downstream models natively utilize the interpolated/cleaned epochs, rather than managing epoch rejection separately at the modeling stage.
