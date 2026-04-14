# Respuesta a Yongjie — 2026-04-09

Interlocutor: Yongjie Duan (PhD student, co-supervisado por Diego). Tono: colega, conversacional, directo.
Contexto: Yongjie compartió su código (brightness_loso.py), confirmó 4 reps de pseudo-trials (solo para top-20 clasificación),
y preguntó qué hicimos nosotros en weight analysis.

---

## Mensaje 1 — Resultados de correr tu pipeline + observación sobre MVNN

Thanks for sharing!

I ran a version of your pipeline on my data. Quick context on my setup since it's pretty different from yours: I only have one subject so far (32 channels, 250Hz, downsampled to 100Hz), and I do leave-one-run-out CV instead of LOSO since I can't generalize across subjects yet.

I also had to adapt the design since my paradigm is controlled 3Hz light flashes, not naturalistic videos. So instead of bright vs dark videos I use PRE vs POST within each flash event: baseline window [-1.25, -0.05]s vs flash response [+0.05, +1.25]s, same 1.2s duration.

With all 32 channels I get 73.0%. Then I restricted to 8 occipital and temporal channels (O1, O2, T7, T8, FT9, FT10, TP9, TP10) and got 77.0%. Slightly better with fewer channels, the frontal and central ones seem to add noise rather than signal for this task. For comparison my previous best with bandpower Welch was 68.9%, so the raw time-series approach wins here.

---

## Mensaje 2 — Weight analysis

For the weight analysis I collected the classifier weights (coef_) from each of the 7 LORO folds separately, averaged them, and reshaped to channels x timepoints. That gives a spatiotemporal importance map that's more stable than using just a single fold.

Three panels: a signed heatmap (channel x time, red pushes toward POST, blue toward PRE), a temporal profile of mean absolute weight across channels, and a spatial bar chart with channels ranked by importance.

**[adjuntar: results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_interpretability.png]**

O1 is clearly the dominant channel, then O2

You could do the same with your data, just reshape coef_ to (127 ch x 120 tp). Would be interesting to compare given how different the paradigms are.

---

## Mensaje 3 — Pseudo-trials

One thing I wanted to clarify: the 4 reps pseudo-trials, that's for the top-20 vs top-20 case right? The code you shared looks like the extreme case (single brightest vs single darkest video, no averaging). What does the accuracy look like for the top-20 general case? Guessing it drops quite a bit from 88.5%.
