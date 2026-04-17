# Revisión del pipeline `04_preprocessing_eeg.py` vs. buenas prácticas

**Fecha:** 2026-04-15 (v3 — meta-revisión; ver changelog al final)
**Autor:** Revisión técnica (Claude)
**Alcance:** `scripts/preprocessing/04_preprocessing_eeg.py`
**Contexto del proyecto:** CAMPEONES — EEG 32-canal BrainCap (ref FCz), ECG, EOG bipolar (R_EYE/L_EYE), GSR, RESP, joystick. Paradigma VR con videos afectivos de duración variable + videos control de luminancia. Datos crudos ahora a **500 Hz** (sin resamplear). Pipeline downstream: decoding con MVNN + SVC.

**Objetivo:** Lista priorizada de cambios recomendados para `04_preprocessing_eeg.py` basada en literatura y guías de referencia. Vos decidís qué aplicar en función de tu tarea.

---

## Referencias consultadas

| Fuente | Uso en esta revisión |
|---|---|
| MNE-Python — tutoriales de preprocessing (`mne.tools/stable/auto_tutorials/preprocessing/`) | Orden canónico de pasos, diseño de filtros, ICA. |
| MNE-Python — background filtering (`.../25_background_filtering.html`) | Cutoffs HP/LP, FIR vs IIR, zero-phase, warnings sobre distorsión ERP. |
| MNE-BIDS-Pipeline — configuración de filtro y de ICA | Defaults de referencia para pipelines BIDS: `ica_algorithm="picard"`, `ica_l_freq=1.0`, `ica_eog_threshold=3.0`, `ica_ecg_threshold=0.1`, `l_freq=1`, `h_freq=40`. |
| Luck (2014) — *Applied ERP Data Analysis* (LibreTexts Cap. 4, 5, 7, 9) | Recomendaciones específicas para ERP: debate 0.1 vs 1.0 Hz HP, reference a mastoides/average, interpolación antes de reref. |
| Makoto's preprocessing pipeline (EEGLAB wiki) | Orden clásico: HPF → ch malos → interpolación → **reref average** → línea → epoching → ICA sobre datos HP 1 Hz con rango corregido → ICLabel. |
| Robbins, Touryan, Mullen, Kothe, Bigdely-Shamlo (2020) — *How sensitive are EEG results to preprocessing methods?* | Evidencia de que el pipeline de preprocesamiento impacta significativamente las conclusiones. |
| Winkler et al. (2015) — *Influence of high-pass filters on ICA* | HPF ≥ 1 Hz mejora calidad de ICA; usar HPF más laxo para datos de análisis. |
| MNE-ICALabel docs (`mne.tools/mne-icalabel`) | **Requisito explícito** de ICLabel: datos *re-referenciados a promedio* y filtrados 1–100 Hz. |
| **Cohen, M. X. (2014) — *Analyzing Neural Time Series Data: Theory and Practice*, MIT Press.** Cap. 7 (Preprocessing Steps), Cap. 8 (EEG Artifacts), Cap. 9 (Time-Domain EEG Analyses). | Orden de pasos, epochs con buffer zones para TF, filtrado sobre datos continuos, criterio conservador de exclusión ICA, problema de rank al interpolar. Referencia primaria del review. |

---

## Cohen (2014) — síntesis por capítulo

Esta sección condensa las recomendaciones de Cohen que tocan directamente los puntos del pipeline. Las citas `(§7.5)` etc. refieren a secciones numeradas del libro.

### Cap. 7 — Preprocessing Steps

- **(§7.1) Trazabilidad.** "Keep track of all details of preprocessing for each subject, such as which trials were rejected, which electrodes were interpolated, and which independent components were removed from the data." ✅ Tu pipeline ya lo hace vía el log JSON centralizado.
- **(§7.1) Consistencia entre condiciones.** "Use the same preprocessing procedures for all conditions to minimize the possibility that any biases that may have been introduced will spuriously cause condition differences." ✅ Aplica: afectivos + luminancia + calm + fixation pasan por el mismo script.
- **(§7.2) Trade-off señal/ruido.** No hay un óptimo absoluto. El criterio (cuán estricto rechazar trials, cuántos ICs quitar, qué cutoff de HP) **depende de los análisis planeados**. Para decoding con MVNN/SVC + muchos trials, el libro sugiere que se puede ser más estricto (sacrificar algo de señal) sin problemas graves.
- **(§7.3) Epochs con buffer zones para análisis TF.** Regla de oro: **buffers de 3 ciclos a la frecuencia más baja de interés** (ej. 1500 ms de buffer para analizar 2 Hz). → Relevante si en algún momento querés TF sobre los videos afectivos: **guardar continuo con anotaciones es la decisión correcta** porque permite cortar ventanas con el buffer que necesites sin reprocesar.
- **(§7.3) ICA sobre epochs superpuestas.** Advertencia explícita: si corrés ICA sobre epochs que se superponen en el tiempo, ICA ve los mismos samples más de una vez → sesgo. Tu pipeline corre ICA sobre el raw continuo con `reject_by_annotation=True`, lo cual esquiva este problema. ✅
- **(§7.5) HPF sobre datos CONTINUOS, no epoched.** Cita literal: "High-pass filters should be applied only to continuous data and not to epoched data. This is because the edge artifact of a 0.5-Hz filter may last up to 6 s, which is probably longer than your epochs." ✅ Tu pipeline filtra sobre el raw continuo antes de crear anotaciones. Confirma la decisión actual.
- **(§7.5) HPF 0.1–0.5 Hz recomendado** para minimizar drifts lentos sobre continuos. Luego los detalles de construcción del filtro están en el Cap. 14.
- **(§7.5) Notch 50/60 Hz** es útil y recomendado.
- **(§7.6) Rejection manual vs automático — Cohen prefiere manual.** "I have tried several automatic algorithmic procedures and found them unsatisfactory and therefore prefer manual trial rejection based on visual inspection." Este es un punto donde Cohen y MNE-BIDS-Pipeline divergen. Tu pipeline usa `find_bads_eog/ecg/muscle` + ICLabel (automático) pero **también permite modo interactivo** (`interactive=True` default en línea 49). Cohen estaría contento con el modo interactivo; considera mantenerlo para los primeros 5 sujetos y revisar todas las decisiones a ojo antes de delegar al modo batch.
- **(§7.7) Spatial filters para minimizar conducción volumétrica.** Cohen recomienda fuertemente el **surface Laplacian (CSD)** cuando se hacen análisis de conectividad o cuando uno quiere evitar confundir efectos por la elección de reference. Para decoding no es crítico, pero es algo que podés agregar en un script downstream (ver nueva recomendación R-15). PCA **no** resuelve la conducción volumétrica.
- **(§7.8) Reference.** Para 32 canales, average reference es aceptable pero no obligatorio; mastoides linkeados es la otra opción tradicional. Lo importante: "The reference electrode should not be close to an electrode where you expect your main effects." FCz como reference original con re-reference a average después (tu setup actual) es consistente con esto. ✅
- **(§7.9) Interpolación: rank deficiency.** Cita literal: "The problem with interpolated electrodes is that they do not provide unique data; they are a perfect weighted sum of the activity of other electrodes. This reduces the rank of the data matrix, which may lead to problems in analyses that require the matrix inverse (taking the pseudoinverse is usually an appropriate solution)." → **Soporta directamente R-3 y la preocupación por el rank de ICA**. Si interpolás antes de ICA, tenés que avisarle al ICA bajando `n_components` o pasando rank explícito.
- **(§7.9) Interpolación es especialmente importante si vas a promediar referencia.** "Interpolation can be particularly important for some spatial filters such as the surface Laplacian or source reconstruction or if you will re-reference to the average of all electrodes: the activity of one bad electrode may contaminate the clean signal of other electrodes." → Si hacés average reference (lo hacés), **tenés que interpolar canales malos antes** o la contaminación se esparce a todos.
- **(§7.9) Alternativa a interpolar: ignorar (NaN) y usar `nanmean`** a nivel grupal. No aplica a tu pipeline con solo 31 canales (preferís mantener consistencia entre sujetos).
- **(§7.10) "Start with clean data."** "Preprocessing can help turn good data into very good data, but no amount of preprocessing will turn low-quality and noisy data into very good data." Para CAMPEONES: si un sujeto tiene 10+ canales malos o EMG continuo masivo, considerar excluirlo antes de confiar en que ICA resuelva todo.

### Cap. 8 — EEG Artifacts

- **(§8.1) Criterio CONSERVADOR de exclusión ICA.** Cita literal: "You should be cautious about removing components that seem to contain signal. In general it is good to take a conservative approach and remove components from the data only if you are convinced that those components contain noise and no or very little signal. **In the best-case scenario you would remove only one component corresponding to blink artifacts**." → Esto **matiza R-7**. Usar umbrales OR + 0.80 (como propongo en R-7) es menos conservador que la postura de Cohen. Ver nota ajustada en R-7.
- **(§8.1) Criterios para juzgar componentes:** topografías, time courses, y espectros de frecuencia. Tu pipeline ya provee los plots interactivos (line 743), lo cual permite esa inspección.
- **(§8.2) Blinks: corregir con ICA mejor que rechazar trials.** "Independent components analysis seems to work better for removing blinks and other oculomotor artifacts (Hoffmann and Falkenstein 2008; Plochl, Ossandon, and Konig 2012). The success of these algorithms at removing oculomotor artifacts suggests that trials containing blinks should not be rejected." ✅ Tu pipeline no rechaza trials por blinks; los corrige vía ICA.
- **(§8.2) Excepción:** trials con blinks largos (sujeto cansado) o blinks durante estímulos visuales breves → conviene rechazar. Para CAMPEONES con videos largos no es crítico.
- **(§8.3) Saccades y EOG.** "There are also saccades and microsaccades that can contaminate EEG data, particularly at frontal and lateral frontal electrodes (or at posterior electrodes if the reference electrode is on the face)." → **Soporta directamente R-2**: movimientos horizontales (saccades) se detectan mejor con el diferencial HEOG (R_EYE − L_EYE), no con un solo canal.
- **(§8.3) Fixation en VR.** "If the subjects are supposed to be fixating throughout the experiment, trials in which subjects broke fixation indicate that the subjects were not fully engaged in the task on that trial." → Nota para CAMPEONES: en videos emocionales el sujeto **no** fija estrictamente, así que este argumento no se aplica.
- **(§8.3) Surface Laplacian** como técnica alternativa para aislar EOG de canales cerebrales.
- **(§8.4) EMG en canales EEG.** "EMG bursts are noticeable as bursts of 20- to 40-Hz activity, often has relatively large amplitude, and is typically maximal in electrodes around the face, neck, and ears." → Los canales T7, T8, FT9, FT10, TP9, TP10 son los más afectados. Relevante en VR (movimiento mandibular). Tu `find_bads_muscle(threshold=0.7)` ya captura esto.

### Cap. 9 — Filtrado de ERPs

- **(§9.2) Filtros mal diseñados → ripples (ringing).** Construir filtros con **transiciones suaves** para evitar ripples. MNE-Python usa FIR con transición automática razonable por default, así que este punto está cubierto por la elección de herramienta.
- **(§9.2) Zero-phase obligatorio para ERPs.** "Some filter settings (again, settings that are generally associated with poor filter construction) may introduce systematic biases in ERP components, such as the use of forward-only or causal filters." ✅ Tu pipeline usa `method="fir"` con MNE default, que es zero-phase.
- **(§9.2) LP cutoff reduce precisión temporal.** "A final concern of filtering ERPs is that applying the low-pass filter reduces temporal precision because the voltage value at each time point becomes a weighted average of voltage values from previous and subsequent time points." → Importante para decoding: si tu decodificador depende de timing sub-100-ms (ej. componentes visuales tempranas), un LP agresivo lo borra.
- **(§9.2) Rango típico de LP para ERPs: 20–30 Hz, ocasionalmente 5–10 Hz.** Tu LP actual es 48 Hz (más alto que lo típico de ERP, pero consistente con mantener info para decoding amplio). → Confirma R-5 como un trade-off legítimo, no un error.
- **(§9.2) Filtrar antes vs después de promediar trials es equivalente** para filtros lineales. Conviene filtrar el continuo (no los epochs) por los edge artifacts del §7.5.

---

## Resumen ejecutivo (TL;DR)

El pipeline es **sólido en lo grueso** — usa Picard, ICLabel, PyPREP-NoisyChannels, anotaciones BIDS con cobertura total, logging detallado y reporte HTML. Los puntos críticos son tres, y están todos relacionados con **el orden de operaciones** y con **supuestos de ICLabel**:

1. 🔴 **ICLabel se corre sobre datos SIN re-referenciar al promedio.** ICLabel fue entrenado asumiendo average reference + filtro 1–100 Hz. Actualmente el reref se hace *después* de ICA (línea 818).
2. 🟠 **Sólo se usa `R_EYE` para `find_bads_eog`**, cuando tenés dos canales EOG monopolares (`R_EYE`, `L_EYE`). El montage es **monocular sobre el ojo derecho**: uno lateral (canto externo → saccades horizontales) y otro infraorbital (debajo del ojo → pestañeos y verticales). Qué electrodo corresponde a cada posición **hay que verificarlo físicamente durante el setup del experimento** — la nomenclatura "R"/"L" del TSV es ambigua y no se debe suponer. Al usar sólo uno, perdés sensibilidad para un eje entero de movimiento ocular (saccades o blinks, según cuál sea el que se omite).
3. 🟠 **Los canales malos se interpolan *después* de ICA** (línea 810). El orden canónico (Luck, Makoto, MNE-BIDS-Pipeline) es: interpolar → reref → ICA. Ajá, hay posturas válidas en la literatura para hacerlo al revés, pero con rank-deficient input ICA da decomposiciones inestables si no le avisás explícitamente. **Cohen §7.9 refuerza esto**: si vas a promediar referencia, interpolar antes es casi obligatorio porque "la actividad de un electrodo malo contamina el señal limpia de otros".

El resto son ajustes menores: HPF ligeramente agresivo para ERP, lógica de exclusión ICA con un heurístico arbitrario, y algunos nice-to-haves.

**Actualización v2 (Cohen 2014 incorporado).** Cohen **confirma** el diseño general del pipeline (continuo con anotaciones, HPF sobre continuo, ICA con Picard, FIR zero-phase). El punto más relevante que agrega es un **criterio más conservador para la exclusión de componentes ICA** (§8.1): Cohen sugiere remover lo mínimo indispensable ("in the best case only one component, the blink"). Esto **matiza** R-7: mantener el umbral de ICLabel alto (0.85–0.90) y nunca sacrificar componentes con probabilidad de ser cerebrales > 0.3. Ver sección "Cohen (2014) — síntesis por capítulo" más abajo para la lista completa de aportes.

**Actualización v3 (meta-revisión contra las fuentes).** Se verificaron puntualmente las citas del review v2 contra las fuentes externas:
- **R-1 reforzado:** el requisito de CAR + 1–100 Hz de ICLabel está verbatim en el source code de `mne_icalabel` (no es interpretación de terceros).
- **R-3 corregido:** el pipeline real de Luck (Appendix 3, LibreTexts) NO interpola antes de ICA — excluye los malos del fit y los interpola después. El review v2 atribuía incorrectamente a Luck una instrucción de interpolar antes. R-3 ahora distingue **Ruta A (Makoto/MNE-BIDS, interpolar antes)** de **Ruta B (Luck, excluir del fit)** y recomienda Ruta A para CAMPEONES por compatibilidad con R-1.
- **R-4 subido a prioridad alta:** Luck Appendix 3 y el tutorial oficial de MNE-ICA explícitamente prescriben el patrón de dos HPF (0.1–30 para análisis, 1.0–30 para ICA) como standard, no como optimización opcional. Con los sanity checks de ERPs que ya estás haciendo, HP 1 Hz en la data final está distorsionando silenciosamente los componentes lentos.

---

## Tabla de pasos del pipeline actual — estado y veredicto

| # | Paso en `04_preprocessing_eeg.py` | Líneas | Estado |
|---|---|---|---|
| 1 | Load raw BIDS (BrainVision) | 186–195 | ✅ OK |
| 2 | Verify montage | 219–222 | ✅ OK |
| 3A | Notch FIR zero-phase 50 + 100 Hz | 270–306 | ✅ OK (ver R-6) |
| 3B | Bandpass FIR zero-phase **0.1 – 48 Hz** (ex 1.0 – 48 Hz) | 310–363 | ✅ R-4 implementada (analysis copy) |
| 3C | PSD 3-paneles (raw/notched/final) | 356–407 | ✅ OK |
| 4 | Guardar intermedio `desc-filtered` (drop anotaciones originales) | 410–440 | ✅ OK |
| 5 | PyPREP `NoisyChannels.find_all_bads(ransac=True)` | 449–466 | ✅ R-3 Ruta A implementada (interp. antes de ICA) |
| 6 | Cargar `merged_events` → anotaciones MNE | 470–495 | ✅ OK |
| 6B | Marcar gaps como `bad` para cobertura total | 498–594 | ✅ OK (ver R-9) |
| 7 **(NUEVO)** | Ref block: `add_reference_channels("FCz")` + `set_montage` + `interpolate_bads(reset_bads=False)` + `set_eeg_reference("average")` — corre **antes** de ICA | 606–650 (aprox) | ✅ R-1 + R-3 Ruta A implementadas |
| 8 | ICA Picard: crea `raw_for_ica` con HPF 1 Hz, fit sobre esa copia, `ica.apply` sobre `raw_filtered` (HPF 0.1 Hz) | 687–720 (aprox) | ✅ R-1 + R-4 implementadas (two-copy pattern) |
| 8.1 | `find_bads_eog(ch_name="R_EYE")` | 645–650 | ❌ **R-2 rechazada** (diagnóstico empírico, ver v4) |
| 8.2 | `find_bads_ecg(ch_name="ECG")` | 653–658 | ✅ OK |
| 8.3 | `find_bads_muscle(threshold=0.7)` | 661–663 | ✅ R-8 (documentado + logged) |
| 8.4 | ICLabel + lógica de exclusión compuesta — ahora recibe `raw_for_ica` (CAR + HPF 1 Hz) | 742+ | ✅ R-1 resuelta · 🔴 **R-7 pendiente** |
| 8.5 | Plots interactivos + `ica.apply` | 743–760 | ✅ OK |
| 9 | Sección antigua (`add_reference_channels` + `set_montage` + `interpolate_bads` + `set_eeg_reference` *después* de ICA) | — | ✅ **REEMPLAZADA por sección 7**. Queda sólo un alias `raw_interpolate = raw_ica` para compatibilidad downstream. |
| 10 | Plot final + PSDs por condición | 844–1143 | ✅ OK |
| 11 | Guardar `desc-preproc` + reporte HTML + log JSON | 1145–1197 | ✅ OK |

---

## Recomendaciones priorizadas

### 🔴 Críticas (correctness)

#### **R-1. Re-referenciar al promedio ANTES de ejecutar ICLabel (y idealmente antes de fittear ICA)** — ✅ **ACEPTADA + IMPLEMENTADA (2026-04-15)**

> **Status:** Implementada en `scripts/preprocessing/04_preprocessing_eeg.py` junto con R-3 (Ruta A) y R-4 en el mismo refactor. Ver sección "Implementación" al final de R-1.

**Problema original.** ICLabel ([Li et al. 2022, referenciado en `mne-icalabel`](https://mne.tools/mne-icalabel/stable/generated/mne_icalabel.label_components.html)) asume que los datos de entrada están **re-referenciados al promedio común y filtrados 1–100 Hz**. En el script original, ICA se fitteaba sobre datos filtrados a 1–48 Hz (ligeramente fuera del rango) y **sin rereference** (la refereciación recién ocurría en línea 818, después de `ica.apply`). Esto puede degradar la confiabilidad de la clasificación por componente.

**Por qué.** ICLabel es un modelo pre-entrenado sobre topografías y espectros en un espacio específico. Si el reference es distinto (en este caso, FCz-referenced), las topografías y varianzas de los componentes cambian, y las probabilidades de clase no son calibradas para ese espacio.

**Evidencia.**
- **Cita textual del source code de `mne_icalabel/iclabel/label_components.py`:** *"ICLabel is designed to classify ICs fitted with an extended infomax ICA decomposition algorithm on EEG datasets **referenced to a common average and filtered between [1., 100.] Hz**."* (verificado 2026-04-15 contra el repo oficial). El propio docstring aclara que correrlo fuera de estas condiciones "may negatively affect classification performance" y que el paper original de ICLabel no investigó a fondo el impacto de desviarse de ellas.
- Pipeline de Makoto (paso 8): re-reference average **antes** de ICA.
- MNE-BIDS-Pipeline ejecuta reref antes de ICA cuando `eeg_reference="average"`.

**Cambio concreto.** Reordenar así:

```python
# NUEVO orden (dentro del mismo script):
# 1. Load raw
# 2. Notch + Bandpass (0.1 - 100 Hz si querés soportar ICLabel literalmente,
#    o mantené 1-48 Hz pero aceptá que te salís del rango de entrenamiento)
# 3. PyPREP bad channels
# 4. Interpolate bads                      <-- MOVIDO desde línea 810
# 5. add_reference_channels("FCz")         <-- MOVIDO desde línea 789
# 6. set_montage                           <-- MOVIDO desde línea 802
# 7. set_eeg_reference("average")          <-- MOVIDO desde línea 818
# 8. Load merged_events + mark gaps as bad
# 9. ICA fit (reject_by_annotation=True)
# 10. ICLabel + auto-exclude
# 11. ica.apply(raw_rereferenced)
# 12. Save
```

**Alternativa menos invasiva.** Si no querés mover reref, podés hacer una copia average-referenced **solo para ICLabel**:

```python
raw_for_iclabel = raw_filtered.copy().set_eeg_reference('average')
ic_labels = label_components(raw_for_iclabel, ica, method="iclabel")
```

pero esto es un parche — lo limpio es respetar el orden canónico.

**Prioridad:** 🔴 Alta si confiás en las etiquetas de ICLabel para decidir exclusiones. Si igual revisás todos los componentes a mano, el costo es menor pero las probs de ICLabel siguen siendo ruidosas.

##### Implementación (2026-04-15) — ✅ resuelta junto con R-3 Ruta A y R-4

El refactor se concentró en una nueva **sección 7 del script** (`## 7. Reference, montage, and interpolation (run BEFORE ICA to satisfy ICLabel's CAR requirement)`), que corre **antes** del fit de ICA. Concretamente:

```python
# Sección 7 NUEVA (entre PyPREP bad channels y el fit de ICA):
raw_filtered = mne.add_reference_channels(raw_filtered.load_data(), ref_channels=["FCz"])

montage = mne.channels.read_custom_montage(bvef_file_path)  # BC-32_FCz_modified.bvef
raw_filtered.set_montage(montage)

# R-3 Ruta A: interpolar ANTES de ICA y del average reference.
# reset_bads=False preserva los nombres en info["bads"] para trazabilidad en el log JSON.
raw_filtered.interpolate_bads(reset_bads=False)

# Average reference aplicado directamente (no como proyector).
raw_filtered, _ = mne.set_eeg_reference(
    raw_filtered, ref_channels="average", copy=False
)
```

Y la vieja **sección 9** (que antes hacía `add_reference_channels` → `set_montage` → `interpolate_bads` → `set_eeg_reference` *después* de `ica.apply`) fue desmantelada y reemplazada por un alias `raw_interpolate = raw_ica` para mantener compatibilidad con el código downstream de plotting/reporting/save.

**Orden final del pipeline** (comparar con la propuesta original en la sección "Propuesta de orden canónico" más abajo):

1. Load raw  
2. Notch 50/100 Hz + Bandpass **0.1 – 48 Hz** (analysis copy — ver R-4)  
3. PyPREP `NoisyChannels.find_all_bads(ransac=True)`  
4. Load `merged_events` → anotaciones + marcar gaps como `bad`  
5. **[NUEVO] `add_reference_channels("FCz")` + `set_montage` + `interpolate_bads(reset_bads=False)` + `set_eeg_reference("average")`** ← R-1 + R-3 Ruta A  
6. **[NUEVO] Crear `raw_for_ica = raw_filtered.copy().filter(l_freq=1.0, h_freq=None)`** ← R-4 (two-copy pattern)  
7. `ica.fit(raw_for_ica, reject_by_annotation=True)`  
8. `find_bads_eog` / `find_bads_ecg` / `find_bads_muscle` sobre `raw_for_ica`  
9. `label_components(raw_for_ica, ica, method="iclabel")` ← ahora en el espacio CAR + HPF 1 Hz que ICLabel espera  
10. `ica.apply(inst=raw_filtered)` ← weights transferidas a la analysis copy (HPF 0.1 Hz)  
11. Save + report

**Requisitos satisfechos.**
- ICLabel recibe datos con **CAR + HPF 1 Hz**, exactamente el espacio para el que fue entrenado (confirmado contra el docstring de `mne_icalabel/iclabel/label_components.py`).
- Los bad channels se interpolan antes del CAR, así que la contaminación no se esparce (Cohen §7.9).
- El rank de los datos tras la interpolación se maneja automáticamente con `n_components=None` en el ICA.
- `find_bads_eog/ecg/muscle` operan sobre la misma copia que vio ICA, así que los scores son coherentes con la decomposición.

**Validación pendiente.** Correr un run end-to-end (sugerido: sub-19 task-01 acq-a run-002) y comparar el reporte HTML contra una corrida con la versión anterior del script para verificar que: (a) no hay crashes de rank, (b) ICLabel clasifica con probabilidades razonables, (c) los ERPs post-preproc mantienen los componentes lentos (efecto de R-4).

---

#### **R-2. Usar AMBOS canales EOG para detección de componentes oculares** — ❌ **RECHAZADA (2026-04-15) tras diagnóstico empírico**

> **Status:** Rechazada. Ver sección "Diagnóstico empírico" al final de R-2. El código permanece con `ch_name="R_EYE"` únicamente.

**Problema original.** Línea 647: `ica.find_bads_eog(inst=raw_filtered, ch_name="R_EYE")`. Sólo se usa uno de los dos canales EOG.

**Nota importante sobre el montage (aclaración 2026-04-15 del usuario):** el sistema **NO** tiene un electrodo en cada ojo. Ambos electrodos están sobre **el ojo derecho**:
- uno en posición **lateral** (canto externo del ojo derecho) → detecta movimientos horizontales / saccades,
- uno en posición **infraorbital** (debajo del ojo derecho) → detecta pestañeos y movimientos verticales (el sitio canónico para detectar blinks por el fenómeno de Bell).

**⚠ La correspondencia entre los nombres `R_EYE` / `L_EYE` y las posiciones físicas (lateral / infraorbital) no está verificada y se debe confirmar durante el setup del experimento antes de aplicar el cambio.** La "L" probablemente refiere a "Lower" y no a "Left", pero esto hay que chequearlo físicamente inspeccionando el cap y las anotaciones del experimentador.

**Sólo usar uno = perder un eje entero de movimiento ocular** (saccades o blinks, según cuál sea el omitido). En el caso actual, si `R_EYE` resulta ser el lateral, estás dependiendo sólo de saccades y perdés sensibilidad directa para los blinks — que típicamente son los componentes ICA más grandes y más importantes de remover.

**Cambio concreto.**

```python
# Opción A (simple, agnóstica a la verificación del montage):
# usar ambos canales. MNE correlaciona los ICs contra cada uno individualmente
# y reporta los que matchean con cualquiera. Funciona sea cual sea la
# asignación física de R_EYE y L_EYE.
eog_components, eog_scores = ica.find_bads_eog(
    inst=raw_filtered,
    ch_name=["R_EYE", "L_EYE"],
)
```

**Opción B (condicional a verificar físicamente cuál es el infraorbital):**

Si se confirma cuál de los dos canales está debajo del ojo derecho, conviene construir un **VEOG bipolar virtual** restándolo de un electrodo por encima del ojo derecho (típicamente `Fp2`). Esto cancela la actividad cerebral común entre Fp2 y el infraorbital y deja casi puro el dipolo ocular vertical — el sitio canónico para detectar pestañeos.

```python
# Sólo aplicar DESPUÉS de verificar cuál de los dos es el infraorbital.
# Supongamos que la verificación confirma que L_EYE es el infraorbital:
raw_filtered = mne.set_bipolar_reference(
    raw_filtered,
    anode="Fp2", cathode="L_EYE",   # <-- cambiar a R_EYE si la verificación dice lo contrario
    ch_name="VEOG",
    drop_refs=False,                  # mantener Fp2 y el infraorbital en el raw
)
eog_components, eog_scores = ica.find_bads_eog(
    inst=raw_filtered,
    ch_name=["R_EYE", "L_EYE", "VEOG"],
)
```

**Lo que NO sirve en este montage:** un HEOG bipolar del tipo `R_EYE − L_EYE` (restar un lateral de un infraorbital) no tiene sentido porque los dos electrodos son perpendiculares entre sí y el diferencial mezcla dos ejes. Un HEOG horizontal verdadero requeriría un electrodo **también** en el canto externo del ojo izquierdo, que no existe en este setup.

**Referencias.** Luck Cap. 9 (ICA for artifacts): "use as many EOG channels as available". Cohen §8.2: los blinks son los artefactos oculomotores más grandes y los mejor capturados por electrodos verticales (sobre/bajo el ojo). Cohen §8.3: saccades se ven mejor en electrodos laterales/frontales. La combinación de ambas posiciones cubre los dos ejes.

**Prioridad:** 🔴 Alta para la Opción A (es un cambio de una línea y es agnóstico al montage). Opción B queda **bloqueada hasta verificar físicamente** cuál es el electrodo infraorbital.

##### Diagnóstico empírico (2026-04-15) — razón del rechazo

**Contexto.** Durante la verificación física del setup, el usuario confirmó que `R_EYE` corresponde al electrodo **lateral** (a la derecha del ojo derecho), por lo que `L_EYE` debería ser el **infraorbital** (debajo del ojo derecho). Sin embargo, al tocar físicamente el electrodo `L_EYE` no se produjo ninguna deflección visible en la señal — síntoma compatible con mal contacto o electrodo desconectado.

Para decidir empíricamente si (a) usar Opción A igual (incluir L_EYE aunque esté sospechoso), (b) usar sólo R_EYE, o (c) buscar un infraorbital mislabeled en otro canal del montage, se escribió un script diagnóstico: `src/campeones_analysis/utils/diagnose_eog_channels.py`. Se corrió sobre `sub-27 ses-vr task-01 acq-a run-002` (~26.5 min de datos, filtro 1–40 Hz).

**Test 1: estadísticas de amplitud por canal**

| Canal | std | peak-to-peak | MAD |
|---|---|---|---|
| L_EYE | 464.5 µV | **9540 µV** | 61.8 µV |
| R_EYE | 40.9 µV | 1195 µV | 17.2 µV |

Contradicción std-MAD en L_EYE (std enorme, MAD razonable) → señal normal la mayor parte del tiempo con **spikes esporádicos gigantescos** (~9.5 mV p2p, fisiológicamente imposible). Patrón típico de mal contacto / electrodo flotante.

**Test 2: correlación con canales frontales (blinks forman dipolo con Fp1/Fp2)**

| Par | r |
|---|---|
| L_EYE vs Fp2 | **+0.021** (≈ cero) |
| L_EYE vs Fp1 | +0.028 |
| R_EYE vs Fp2 | **+0.541** |
| R_EYE vs Fp1 | +0.404 |

Un canal infraorbital funcional debería correlacionar fuertemente con Fp2 (forman los dos polos del dipolo de blink). L_EYE no correlaciona en absoluto → no está registrando actividad ocular coherente. R_EYE, aunque lateral, sí correlaciona bien (0.54) porque está lo suficientemente cerca del globo ocular para captar la proyección del dipolo.

**Test 3: `mne.preprocessing.find_eog_events`**

| Canal | Eventos | Tasa |
|---|---|---|
| L_EYE | 253 | 9.5/min |
| R_EYE | 40 | 1.5/min |

A primera vista L_EYE parece más activo, pero es un espejismo: `find_eog_events` ajusta su umbral a la varianza del canal, así que los spikes ruidosos de L_EYE son detectados como pseudo-blinks. **Los 253 eventos no son blinks reales** — si lo fueran, habría correlación con Fp2 (test 2), y no la hay.

**Test 4: búsqueda de infraorbital mislabeled — ranking de respuesta blink-locked en todos los canales**

Usando los 40 eventos de R_EYE como trigger, se computó el ERP blink-locked sobre los 32 canales EEG + EOG y se rankeó por amplitud pico absoluta en la ventana 0–300 ms.

| Canal | \|peak\| µV | corr vs Fp2 (waveform) |
|---|---|---|
| **FT10** | **639.5** | +0.061 |
| CP1 | 43.1 | +0.012 |
| TP9 | 42.3 | +0.106 |
| Fp1 | 40.0 | +0.951 |
| Fp2 | 34.0 | +1.000 (ref) |

FT10 apareció con una respuesta **19× mayor que Fp2**, lo cual inicialmente sugirió un posible infraorbital mislabeled. Pero tres chequeos adicionales descartaron esa hipótesis:

1. **Forma de onda incoherente con Fp2.** Correlación de 0.06 con la forma de onda de Fp2. Un infraorbital real debería correlacionar fuertemente (> 0.7 en magnitud, positiva o negativa invertida). Una respuesta time-locked con forma no dipolar no es infraorbital.
2. **Estadísticas globales de FT10 idénticas a L_EYE:** std = 351.9 µV, peak-to-peak = 9167 µV (**9 mV**), MAD = 14.2 µV. **Mismo patrón patológico que L_EYE**: línea base normal + spikes esporádicos gigantes.
3. **El "ERP" de FT10 está dominado por outliers, no por consistencia trial-a-trial:**

    | Métrica per-blink en FT10 | Valor |
    |---|---|
    | Mediana de \|pico\| | 531 µV |
    | Media de \|pico\| | 826 µV |
    | Máximo | **2968 µV** (en un solo trial) |

    Mean ≫ median → la respuesta promedio está dominada por 2–3 outliers catastróficos, no por una respuesta consistente. Un infraorbital real daría respuestas estables trial-a-trial.

**Conclusión: FT10 está roto, no es un infraorbital mislabeled.** Ningún otro canal del ranking muestra respuesta con forma dipolar (correlación fuerte con Fp2, positiva o negativa). **No existe un infraorbital mislabeled en el montage.**

**Hallazgo secundario:** FT10 tiene exactamente la misma patología que L_EYE (std/ptp gigantes, MAD normal) → debería marcarse como **bad channel** al menos para sub-27 run-002. Vale la pena correr el diagnóstico sobre otros sujetos para determinar si es un problema sistémico (electrodo físico dañado en el cap) o específico de este sujeto/run.

**Decisión final.**

1. **Mantener el código actual** con `ch_name="R_EYE"` únicamente. R_EYE es funcional como detector de blinks (correlación 0.54 con Fp2, respuesta blink-locked consistente de ~34–40 µV).
2. **Apoyarse en ICLabel** para detección topográfica de componentes oculares, que es el mecanismo robusto: ICLabel mira la topografía espacial del dipolo frontal y no depende de un canal EOG limpio.
3. **Agregar un comentario breve** al lado de `find_bads_eog` en `04_preprocessing_eeg.py` apuntando a este diagnóstico.
4. **TODO abierto:** en el próximo setup experimental revisar físicamente el contacto del electrodo infraorbital (L_EYE) y del electrodo de FT10 antes de empezar a grabar.

**Artefactos generados por el diagnóstico (guardados en `data/derivatives/diagnostics/eog_channels/`):**
- `sub-27_run-002_eog_diagnostic.png` — trazas de tiempo L_EYE / R_EYE / Fp1 / Fp2 (30 s)
- `sub-27_run-002_blink_erp_candidates.png` — ERP blink-locked de los top candidatos

**Prioridad:** ❌ Rechazada. El diagnóstico empírico demostró que L_EYE no aporta información útil (canal roto) y que no hay un infraorbital mislabeled en el montage.

---

### 🟠 Importantes (mejores prácticas estándar)

#### **R-3. Sacar los canales malos del fit de ICA (dos rutas válidas: interpolar antes, o excluir del fit)** — ✅ **ACEPTADA + IMPLEMENTADA (2026-04-15, Ruta A)**

> **Status:** Implementada como Ruta A (interpolar antes de ICA y del average reference) en `scripts/preprocessing/04_preprocessing_eeg.py`, en el mismo refactor que R-1 y R-4. Ver sección "Implementación" en R-1 arriba — la interpolación ocurre en la nueva sección 7, antes del fit de ICA, con `reset_bads=False` para mantener trazabilidad en el log JSON.


**Problema.** Orden actual: PyPREP marca canales malos → ICA fit (con los malos aún presentes en la matriz; `reject_by_annotation=True` saca los segmentos bad pero no los canales) → `ica.apply` → **recién después** se hace `interpolate_bads()` y reref.

Esto tiene dos consecuencias:
1. **ICA fittea sobre canales malos**, lo cual puede contaminar la decomposición. La práctica canónica es *sacarlos del fit* (via `picks`) **o** interpolarlos antes.
2. **Rank deficiency al interpolar + reref**. Al interpolar un canal, su valor es una combinación lineal de los vecinos → rank deficiente. Luego average reference reduce el rank en 1 más. Si `n_components=None` en ICA, MNE detecta el rank automáticamente — **pero** si después interpolás sin actualizar rank, los componentes ICA que tenés no matchean.

**Hay dos rutas canónicas en la literatura, y son igualmente válidas. Elegí una y sé consistente:**

**Ruta A (Makoto / MNE-BIDS-Pipeline):** interpolar ANTES de ICA.

```python
# Después de PyPREP NoisyChannels:
raw_filtered.info["bads"] = bads
raw_filtered = raw_filtered.interpolate_bads(reset_bads=False)
# reset_bads=False → mantener en info["bads"] para trazabilidad, pero la señal ya está interpolada
# Después: reref average → ICA fit con rank corregido automáticamente
```

**Ruta B (Luck Appendix 3):** NO interpolar antes de ICA. En cambio, excluir los malos del fit, aplicar las weights al dataset original, y **recién después** del `ica.apply` hacer interpolación + reref.

```python
# Identificar los malos pero NO interpolar
raw_filtered.info["bads"] = bads

# Fit ICA excluyendo los malos (y excluyendo EOG bipolares si los tenés)
ica.fit(
    raw_filtered,
    picks=mne.pick_types(raw_filtered.info, eeg=True, exclude='bads'),
    reject_by_annotation=True,
)
# Apply weights al raw original (los bads siguen estando)
ica.apply(raw_filtered)
# Recién ahora: interpolar + reref
```

**Recomendación para CAMPEONES: Ruta A**, por compatibilidad con ICLabel. ICLabel requiere los datos ya en average reference (ver R-1), y average reference sobre canales NO interpolados esparce la contaminación del canal malo a todos los demás (ver cita de Cohen §7.9 abajo). Si hacés Ruta B, tenés que hacer una copia aparte average-referenced solo para pasarle a ICLabel, lo cual es un parche. Ruta A es más limpia.

**Evidencia.**
- **MNE-BIDS-Pipeline y Makoto (paso 7–8)**: interpolar antes de ICA y de la rereferencia. Esta es la Ruta A.
- **Luck, *Applied ERP Data Analysis*, Appendix 3 (Example Processing Pipeline)**: verificado en LibreTexts (2026-04-15) — Luck **no interpola antes de ICA**. Su orden es: filter 0.1–30 Hz → identify bad channels → make copy filtered 1–30 Hz for ICA → exclude bads from ICA decomposition → transfer weights back → remove artifacts → re-reference → **interpolate bad channels** → epoch. Esta es la Ruta B. (Corrección sobre el review v2: la cita "Luck Cap. 7: interpolate before the average reference and before ICA" que aparecía en versiones anteriores de este documento era incorrecta y fue removida.)
- **Cohen §7.9 (cita literal):** *"Interpolation can be particularly important for some spatial filters such as the surface Laplacian or source reconstruction or **if you will re-reference to the average of all electrodes: the activity of one bad electrode may contaminate the clean signal of other electrodes**."* → Cohen apoya la Ruta A específicamente cuando vas a promediar referencia.
- **Cohen §7.9 sobre el rank:** *"The problem with interpolated electrodes is that they do not provide unique data; they are a perfect weighted sum of the activity of other electrodes. This reduces the rank of the data matrix, which may lead to problems in analyses that require the matrix inverse (taking the pseudoinverse is usually an appropriate solution)."* → Por eso, al interpolar antes de ICA, `n_components=None` (rank automático) es la respuesta correcta; no hace falta forzar `n_components=30`.

**Cuidado.** El pipeline de Bigdely-Shamlo / PREP defiende interpolar *después* de ICA para que los componentes ICA no reflejen la interpolación. Es un debate abierto y Luck está de ese lado. **Lo importante es consistencia**: si vas por Ruta A, `n_components=None` y MNE autodetecta; si vas por Ruta B, pasás `picks` explícito al fit.

**Prioridad:** 🟠 Media-alta. La estabilidad de ICA es marginal con 31 canales y el riesgo real es tener componentes "fantasma". Ruta A es la recomendada para CAMPEONES por compatibilidad con ICLabel.

---

#### **R-4. Filtrar con dos HPF distintos: uno agresivo para ICA, uno laxo para la data final** — ✅ **ACEPTADA + IMPLEMENTADA (2026-04-15)**

> **Status:** Implementada en `scripts/preprocessing/04_preprocessing_eeg.py` en el mismo refactor que R-1 y R-3. Ver "Implementación" al final de la sección.


**Problema.** Actualmente usás HPF = 1.0 Hz tanto para ICA como para la data que después pasa al decoding. Winkler et al. (2015) y Luck et al. (2018) muestran que:
- HPF ≥ 1 Hz mejora dramáticamente la calidad de la decomposición ICA (los componentes se separan mejor porque se remueven drifts lentos).
- HPF ≥ 0.5 Hz distorsiona componentes ERP lentos (N400, LPP, P300 tardío, slow waves en tareas emocionales).

Para tarea CAMPEONES con **videos emocionales** (afecto sostenido, LPP, late positive potentials), esto es relevante.

**Cambio concreto.** Dos copias:

```python
# 1. Filtro laxo sobre la data que se guarda al final
raw_analysis = raw.copy().notch_filter([50, 100], picks='eeg').filter(l_freq=0.1, h_freq=40, picks='eeg')

# 2. Filtro agresivo sobre una copia que solo se usa para fittear ICA
raw_for_ica = raw.copy().notch_filter([50, 100], picks='eeg').filter(l_freq=1.0, h_freq=40, picks='eeg')

# Fit ICA sobre la copia agresiva:
ica.fit(raw_for_ica, reject_by_annotation=True)

# Aplicar ICA a la data de análisis:
ica.apply(raw_analysis)
```

**Evidencia.**
- Winkler et al., 2015, *J Neurosci Methods* 250:28. "Highpass of 1 Hz strongly improves ICA decomposition quality".
- **Luck, *Applied ERP Data Analysis*, Appendix 3 (Example Processing Pipeline)** — verificado en LibreTexts (2026-04-15): Luck prescribe explícitamente **dos copias**. La "pre-ICA dataset" se filtra a 0.1–30 Hz. Antes del fit de ICA se crea una "ICA decomposition dataset" separada que se filtra **1–30 Hz con 48 dB/octava**. El ICA se fittea en la copia agresiva y las weights se transfieren de vuelta a la copia laxa para aplicar y seguir el pipeline. **Este es el standard en la comunidad ERP**, no una optimización opcional.
- **MNE tutorial de ICA (oficial):** *"the ICA solution found from the filtered signal can be applied to the unfiltered signal"* — MNE explícitamente respalda el patrón de dos copias. Recomienda HP 1 Hz para el fit porque "slow drifts reduce source independence, making the ICA solution less accurate."
- Pipeline de Makoto paso 4 nota: usar 0.1 Hz si analizás N400/P600, aunque limita las garantías de ICA → misma lógica, resuelta con dos copias.
- **Cohen §7.5 (cita literal verificada):** *"Applying a high-pass filter at 0.1 or 0.5 Hz to the continuous data is useful and recommended to minimize slow drifts. High-pass filters should be applied only to continuous data and not to epoched data. This is because the edge artifact of a 0.5-Hz filter may last up to 6 s, which is probably longer than your epochs."* Cohen recomienda 0.1–0.5 Hz para la data de análisis y es explícito en que el HPF debe aplicarse sobre datos continuos. Tu pipeline ya filtra sobre continuo ✅; lo que falta (según R-4) es usar un HPF más laxo (0.1 Hz) en la copia de análisis y conservar 1.0 Hz sólo para la copia de ICA.

**Para CAMPEONES específicamente:** Con videos emocionales (afecto sostenido, LPP, late positive potentials) + sanity checks de ERPs occipitales que ya estás haciendo, usar HPF 1.0 Hz para la data final **está distorsionando silenciosamente los componentes lentos** que te importan para interpretabilidad. El patrón de dos copias de Luck/MNE resuelve esto sin sacrificar calidad de ICA. Si el decoding es *lo único* que te importa y no mirás ERPs, HP 1 Hz en la data final es defendible — pero con los sanity checks en el diario eso ya no aplica.

**Prioridad:** 🔴 **Alta** (actualizado v3 — antes era "Media"). Tanto Luck como el tutorial de MNE presentan este patrón como el standard, no como una optimización. La motivación directa es que R-1 te obliga a tocar el orden del pipeline de todas formas — agregar la segunda copia en el mismo refactor es barato.

##### Implementación (2026-04-15)

Dos cambios concretos en el script:

1. **HPF de la analysis copy bajado de 1.0 a 0.1 Hz.** En la sección 3B (band-pass filtering), `hpass = 1.0` cambió a `hpass = 0.1` con un comentario extenso explicando que esta es la "analysis copy" y que una segunda copia con HPF 1 Hz se crea justo antes del fit de ICA. El log JSON registra ahora `two_copy_pattern: "analysis_0.1Hz_ica_1.0Hz"`.

2. **Segunda copia dedicada para ICA (`raw_for_ica`).** Creada en la sección 8, justo antes del fit, aplicando un filtro adicional HPF 1 Hz sobre `raw_filtered` (que ya tiene CAR + interpolación por R-1/R-3):

    ```python
    hpass_for_ica = 1.0
    raw_for_ica = raw_filtered.copy().filter(
        l_freq=hpass_for_ica, h_freq=None,
        picks='eeg', method='fir', phase='zero',
    )
    ica.fit(raw_for_ica, picks='eeg', reject_by_annotation=True)
    ```

3. **Transferencia de weights a la analysis copy.** Todo lo que opera sobre ICA corre sobre `raw_for_ica`:
   - `find_bads_eog(inst=raw_for_ica, ch_name="R_EYE")`
   - `find_bads_ecg(inst=raw_for_ica, ch_name="ECG")`
   - `find_bads_muscle(raw_for_ica, threshold=0.7)`
   - `label_components(raw_for_ica, ica, method="iclabel")`
   - `ica.plot_sources(raw_for_ica)`, `ica.plot_components(inst=raw_for_ica)`, `report.add_ica(ica, inst=raw_for_ica)`

   Pero el `ica.apply` final recibe la analysis copy:
   ```python
   raw_ica = ica.apply(inst=raw_filtered)  # raw_filtered tiene HPF 0.1, no 1.0
   ```

   Esto es matemáticamente válido porque filtrar y aplicar ICA son operaciones lineales (ver cita de Luck Appendix 3 en la sección de evidencia).

**Beneficio concreto para CAMPEONES.** La data guardada en disco ahora preserva componentes ERP lentos (LPP, late positive potentials sobre videos emocionales) que antes eran distorsionados por el HPF 1 Hz. Los sanity checks de ERPs occipitales documentados en el diario van a mostrar diferencias visibles tras re-preprocessing.

**Validación pendiente.** Mismo run de prueba que R-1 (ej. sub-19 task-01 acq-a run-002): verificar que la diferencia en los ERPs preprocesados es visible (slow components preservados con HPF 0.1 vs. la versión previa con HPF 1), y que ICLabel clasifica con probabilidades comparables o mejores.

---

#### **R-5. Reconsiderar el lowpass (48 Hz) en función del objetivo**

**Problema.** Lowpass = 48 Hz **debajo del notch 50 Hz**. Consecuencias:
- Redundancia: el notch de 50 Hz ya mata la línea eléctrica; el LP48 sólo agrega una transición adicional sin ganancia.
- Matás frecuencias beta altas (30–45 Hz) y gamma baja (>48 Hz), que pueden contener información en decoding de videos.
- Si el análisis es puramente ERP, 30 o 40 Hz es más conservador (Luck).

**Opciones.**

| Objetivo | Notch | Bandpass recomendado |
|---|---|---|
| ERP estricto (P300, LPP, N400) | 50 Hz | 0.1 – **30 Hz** |
| ERP + decoding temporal | 50 + 100 Hz | 0.1 – **40 Hz** (default MNE-BIDS-Pipeline) |
| Decoding amplio + time-frequency | 50 + 100 Hz | 1.0 – **100 Hz** |
| Análisis gamma | 50 + 100 + 150 Hz (zapline) | 1.0 – **120 Hz** |

**Recomendación para CAMPEONES:** dado que tenés decoding downstream y videos de luminancia (response temprana, occipital), **1.0 – 40 Hz** es razonable y empareja con defaults de MNE-BIDS-Pipeline. Si querés explorar gamma más adelante, conservá la data continua a 500 Hz sin LP estricto y dejá que el decoding aplique su propio filtro.

**Nota de Cohen (§9.2):** "A final concern of filtering ERPs is that applying the low-pass filter reduces temporal precision because the voltage value at each time point becomes a weighted average of voltage values from previous and subsequent time points." → Para decoding que dependa del timing fino (componentes visuales tempranas, onsets de luminancia), un LP muy bajo suaviza info útil. LP 40–48 Hz a 500 Hz es un buen compromiso.

**Prioridad:** 🟠 Media.

---

#### **R-6. Notch filtering: considerar `zapline` o banda más ajustada**

**Problema.** Estás usando notch FIR a **50 + 100 Hz**. Funciona, pero:
- Con datos VR hay movimiento → línea eléctrica puede ser **no estacionaria** (amplitud variable) → notch FIR fijo puede subcompensar o sobrecompensar.
- MNE aplica notch de ancho fijo. Para evitar artefactos de ringing, conviene chequear el ancho.

**Opciones.**
- **Statu quo (OK):** dejar `notch_filter(freqs=[50, 100])` → MNE usa filter_length automático. Es razonable.
- **Mejor:** `zapline` de `meegkit` (Robbins et al., 2020) que adapta la remoción de línea por segmentos. Instalable via pip: `pip install meegkit`. Uso: `from meegkit import dss; dss.dss_line(raw.get_data(), fline=50, sfreq=raw.info['sfreq'])`.
- **Alternativa ligera:** agregar 150 Hz al notch si ves picos en el PSD (ahora que está a 500 Hz podés resolver hasta 250 Hz).

**Prioridad:** 🟡 Baja a menos que veas contaminación de línea residual en los PSDs del reporte HTML.

---

#### **R-7. Simplificar la lógica de auto-exclusión de componentes ICA**

**Problema.** Líneas 724–733:

```python
to_exclude = []
for idx in pattern_matching_artifacts:
    if label_names[idx] in ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']:
        to_exclude.append(idx)

if len(eog_components) > 0 and eog_components[0] < 3:
    to_exclude.append(eog_components[0])

to_exclude = np.unique(to_exclude + channel_artifact_indices)
```

Problemas:
- **El heurístico `eog_components[0] < 3` es arbitrario**: sólo excluye el primer componente EOG si su índice es menor a 3. No hay justificación en la literatura para ese umbral.
- **La intersección pattern-matching ∩ ICLabel es conservadora**: excluís sólo cuando ambos métodos coinciden. Eso reduce falsos positivos pero **pierde falsos negativos** — componentes oculares claros que ICLabel detecta pero `find_bads_eog` no (o viceversa).
- **`channel noise` se excluye sólo por ICLabel**, inconsistente con la regla anterior.

**Cambio concreto.** Propuesta simple basada en umbrales de probabilidad ICLabel:

```python
from mne_icalabel import label_components

ic_labels = label_components(raw_for_iclabel, ica, method="iclabel")
labels = ic_labels["labels"]
probs = ic_labels["y_pred_proba"]

# Regla: excluir si ICLabel asigna >= 0.80 a una clase de artefacto.
ARTIFACT_LABELS = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}
ICLABEL_THRESHOLD = 0.80

iclabel_exclude = [
    i for i, (lab, p) in enumerate(zip(labels, probs))
    if lab in ARTIFACT_LABELS and p >= ICLABEL_THRESHOLD
]

# Unión (OR) con pattern matching de MNE (no intersección):
pattern_exclude = set(eog_components) | set(ecg_components) | set(muscle_components)

ica.exclude = sorted(set(iclabel_exclude) | pattern_exclude)
```

**Por qué OR y no AND:**
- `find_bads_eog/ecg` usan correlación con el canal de referencia → alta especificidad pero baja sensibilidad si el artefacto es atípico.
- ICLabel es un clasificador ML → sensibilidad alta pero puede fallar en datos fuera de distribución (VR, movimiento).
- La unión de ambos reduce el riesgo de *dejar pasar* artefactos. El costo es excluir un par de componentes de más, lo cual con 31 componentes es tolerable.

**Umbral 0.80:** es el estándar sugerido por MNE-ICALabel. Podés ajustarlo (0.7 más conservador, 0.9 más permisivo).

**⚠ Matiz de Cohen (§8.1).** Cohen es explícitamente más conservador que mi propuesta original:

> "You should be cautious about removing components that seem to contain signal. In general it is good to take a conservative approach and remove components from the data only if you are convinced that those components contain noise and no or very little signal. **In the best-case scenario you would remove only one component corresponding to blink artifacts**."

Esto es casi el extremo opuesto de la propuesta OR + 0.80. **Cómo reconciliar ambos:**

- Cohen escribe para un workflow clásico con ~64+ electrodos y análisis ERP/TF donde cada componente cerebral cuenta. Con 31 canales + decoding, remover 3–6 componentes de artefacto (blinks + ECG + músculo + ruido de línea) sigue siendo conservador en términos prácticos: te quedás con 25+ componentes cerebrales.
- **Reglas que respetan ambos puntos:**
  1. Subir el umbral ICLabel a **0.85 o 0.90** (más cerca del "convinced" de Cohen).
  2. **Siempre revisar visualmente** los componentes marcados antes de aplicar (tu modo `interactive=True` ya permite esto — **no lo elimines**).
  3. Loggear no sólo los excluidos, sino también los "casi excluidos" (prob entre 0.5 y el umbral) para auditar después.
  4. Nunca excluir componentes con prob `brain` > 0.3, aunque el clasificador los haya puesto en otra categoría.

Propuesta refinada:

```python
# Reglas conservadoras estilo Cohen:
ICLABEL_THRESHOLD = 0.85
BRAIN_FLOOR = 0.30  # nunca excluir si brain_prob > 0.3

labels = ic_labels["labels"]
probs = ic_labels["y_pred_proba"]  # shape: (n_components, 7 classes)
brain_probs = probs[:, 0]  # asumiendo brain es la primera clase
max_artifact_probs = probs.max(axis=1)  # o recorrer clases de artefacto

iclabel_exclude = [
    i for i, (lab, p, brain_p) in enumerate(zip(labels, max_artifact_probs, brain_probs))
    if lab in ARTIFACT_LABELS and p >= ICLABEL_THRESHOLD and brain_p < BRAIN_FLOOR
]
```

**Prioridad:** 🟠 Media. El criterio actual funciona pero es difícil de justificar en un paper. Con esta refinación respetás tanto el criterio ML moderno (ICLabel) como el criterio conservador clásico (Cohen).

**Implementación (2026-04-15) — Variante A aplicada.** Reescrito el bloque 744–813 del script:

- `label_components()` (que sólo devuelve la top-class probability) fue reemplazado por una llamada directa a `iclabel_label_components(raw_for_ica, ica)`, que devuelve la matriz completa `(n_components, 7)`. Eso da acceso a `brain_probabilities = iclabel_proba[:, 0]`, necesario para el brain-floor.
- **Umbrales:** `ICLABEL_THRESHOLD = 0.85` (cerca del "convinced" de Cohen), `BRAIN_FLOOR = 0.30` (Variante A: floor aplica a pattern matching Y a ICLabel — si `brain_prob >= 0.30`, el componente se veta incluso si `find_bads_eog` lo flagueó).
- **OR logic:** `candidates = pattern_set | iclabel_set`. Reemplaza la intersección AND anterior que perdía falsos negativos.
- **Heurísticos arbitrarios eliminados:** `eog_components[0] < 3` y la inclusión especial de `channel_artifact_indices` fueron removidos — ahora todo pasa por el mismo mecanismo (umbral + floor).
- **Trazabilidad en stdout:** tabla de clasificación ahora imprime `Top prob` + `Brain prob` por componente, y un bloque "R-7 EXCLUSION DECISION" resume candidatos por pattern matching, candidatos por ICLabel, componentes vetados por el brain floor, y la lista final de `ica.exclude`.
- **Logging estructurado:** nuevos campos en el JSON → `ica_exclusion_iclabel_threshold`, `ica_exclusion_brain_floor`, `ica_exclusion_pattern_candidates`, `ica_exclusion_iclabel_candidates`, `ica_exclusion_vetoed_by_brain_floor`.
- **Validación pendiente:** correr un run end-to-end (sugerido sub-19 task-01 acq-a run-002) y revisar en el report HTML cuántos componentes excluye vs la versión previa del script. Si la diferencia es grande, vale la pena inspeccionar los "vetados por brain floor" a ojo para validar el umbral.

---

### 🟡 Menores / nice-to-have

#### **R-8. `find_bads_muscle(threshold=0.7)` — justificar o ajustar**

MNE default es `threshold=0.5`. Subir a 0.7 hace que se detecten **menos** componentes musculares (más conservador). Para datos VR con movimientos esperables, puede tener sentido para evitar excluir componentes cerebrales legítimos. **Acción:** documentar la elección en un comentario y ver en el reporte HTML si los componentes marcados como muscle tienen sentido visual.

**Implementación (2026-04-15) — aplicada.** Agregado bloque de comentario extenso arriba del `find_bads_muscle` justificando la elección de `threshold=0.7`:
- Es más conservador que el default MNE 0.5.
- CAMPEONES es VR → movimiento de cabeza/cuello esperable → un threshold bajo corre riesgo de flagguear componentes cerebrales posteriores/frontales con contenido de alta frecuencia residual del micro-movimiento.
- La decisión final se re-filtra por ICLabel + brain floor (R-7), así que `find_bads_muscle` actúa como **generador de candidatos**, no como gate final.
- Criterio de ajuste escrito en el comentario: si el reporte HTML muestra componentes claramente musculares no detectados, bajar a 0.5; si componentes cerebrales están siendo vetados por R-7 después de pasar por aquí, subir a 0.8.
- Log actualizado: `find_bads_muscle_threshold=0.7` y `find_bads_muscle_components=[idx...]` en el JSON por run.

#### **R-9. Marcar como `bad` todos los gaps fuera de `merged_events`**

El paso 6B (línea 498–594) marca todo lo que no está en merged_events como 'bad'. Para ICA esto es **conservador pero razonable**: sólo usás segmentos de video presentados. Sin embargo:
- Perdés períodos de reposo que son útiles para aprender topografías de blink.
- Si el experimento tiene transiciones entre videos con el sujeto moviéndose, esos segmentos quedan fuera — bien.

**Sugerencia:** dejá el paso como está, pero considerá agregar una opción para **incluir también los segmentos de fixation baseline** al fit de ICA (ya están en merged_events, así que deberían estar incluidos — chequear en el TSV).

**Acción para verificar:** abrí `sub-19_ses-vr_task-01_acq-a_run-002_desc-merged_events.tsv` y confirmá que la línea `fixation` (stim_id=500) está marcada. Ya lo verifiqué yo: está presente, con duration 300s. ✓

#### **R-10. Logging: guardar el nombre y probabilidad ICLabel de cada componente excluido**

Actualmente el log JSON guarda los índices, pero no las probabilidades ICLabel ni la razón específica por la que se excluyó. Útil para auditar qué tan confiable fue la clasificación por run.

```python
log_preprocessing.log_detail("ica_exclusions_detail", [
    {
        "component": f"ICA{idx:03d}",
        "iclabel_class": labels[idx],
        "iclabel_prob": float(probs[idx]),
        "by_pattern_matching": idx in pattern_exclude,
        "by_iclabel": idx in iclabel_exclude,
    }
    for idx in ica.exclude
])
```

**Implementación (2026-04-15) — mini-R-10 aplicada.** En el script ahora se loggea:

```python
log_preprocessing.log_detail("ica_exclusions_detail", [
    {
        "component": f"ICA{idx:03d}",
        "iclabel_class": label_names[idx],
        "iclabel_top_prob": float(top_probabilities[idx]),
        "iclabel_brain_prob": float(brain_probabilities[idx]),
        "by_pattern_matching": idx in pattern_set,
        "by_iclabel": idx in iclabel_set,
    }
    for idx in ica.exclude
])
```

Diferencias con la versión propuesta originalmente: se loggea **tanto la top prob como la brain prob** (no sólo `y_pred_proba`), porque la versión refinada de R-7 usa el brain-floor para vetar exclusiones. Así cada componente excluido deja traza completa de por qué se excluyó y qué tan confiable era la decisión.

La versión "full R-10" (loggear también los near-misses y la matriz completa de probabilidades) no se aplicó — se reserva para si se necesita análisis de sensibilidad post-hoc del umbral.

#### **R-11. Verificar que PyPREP corra con el montage cargado**

`NoisyChannels.find_all_bads(ransac=True)` usa relaciones espaciales entre canales → necesita un montage válido. En `04_preprocessing_eeg.py:450` el raw debería tener montage cargado desde `read_raw_bids` (hay validación en línea 219–222), así que está OK. **Sólo confirmar en un sanity check** que `raw_filtered.get_montage()` retorna algo sensato inmediatamente antes de llamar a PyPREP.

**Implementación (2026-04-15) — ya aplicada en paralelo (línea 467).** El usuario implementó el hardening completo mientras se discutía R-7 y R-12, así que el cambio ya estaba vivo cuando lo revisamos. La versión actual es más robusta que la mínima propuesta ("sólo confirmar"):
- Se carga explícitamente el montage canónico `BC-32_FCz_modified.bvef` desde el directorio del script **antes** de llamar a `NoisyChannels`, usando `on_missing='ignore'` porque FCz todavía no existe como canal (es la referencia en ese punto del pipeline; se agrega en la sección 7 por R-1).
- Sanity check in-script: cuenta cuántos canales EEG tienen posiciones 3D válidas (`loc[:3]` no todo-ceros ni NaN) y lo imprime. Si falta alguno, lista los canales sin posición como warning — RANSAC no los usa, así que se explicita cuál criterio se pierde por canal.
- Log: `pyprep_montage_applied_before_prep=True` y `pyprep_channels_with_positions=<n>`.
- El comentario in-script también documenta **por qué** esto importa: sin montage, `find_bad_by_ransac` en PyPREP (que usa interpolación spline esférica sobre posiciones 3D) crashearía o silenciosamente no correría, perdiendo uno de los seis criterios — el único que captura electrodos flotantes / de alta impedancia con pickup EM consistente.
- Convive con el logging per-criterio de R-17 (ambos trabajan sobre `NoisyChannels`, pero acá se asegura el input; en R-17 se introspecta el output).

#### **R-12. Wide-band ICA copy (1–100 Hz) para mejor detección muscular e ICLabel in-distribution**

`find_bads_muscle` funciona mejor en señal con banda alta preservada (hasta ~100 Hz), porque la actividad muscular es de alta frecuencia. Si el LP para análisis está en 48 Hz (CAMPEONES), la detección muscular sobre el mismo LP cae. Además, ICLabel fue entrenado sobre datos 1–100 Hz (Pion-Tonachini 2019), y correrlo sobre 1–48 Hz es out-of-distribution.

**Solución: extender R-4 del eje HPF al eje LPF también.** El análisis copy queda en 0.1–48 Hz (sin cambios), y la ICA copy se amplía a 1–100 Hz. `ica.apply` sigue aplicando las unmixing matrix a la analysis copy — la linealidad del filtrado garantiza que la transferencia es válida.

**Por qué NO es un cambio de 1 línea.** La tentación es cambiar `h_freq=None` por `h_freq=100` en la línea actual de creación de `raw_for_ica`, pero eso es un **no-op silencioso**. La razón: `raw_for_ica = raw_filtered.copy().filter(...)` parte de `raw_filtered`, que YA pasó por el bandpass 0.1–48 Hz en la Sección 3B. El filtrado es destructivo: la energía del rango 48–100 Hz fue eliminada del buffer y no se puede recuperar filtrando de nuevo. Aplicar LP 100 sobre datos LP 48 te deja en LP 48.

Para tener banda hasta 100 Hz hay que **empezar más arriba en el pipeline**, desde `raw_notched` (después del notch pero antes del bandpass). Pero al hacerlo, perdés todo el procesamiento espacial posterior (add FCz, montage, bad channels, interpolación, CAR, anotaciones) y hay que replicarlo manualmente sobre la nueva copia.

**Cambio concreto.**

```python
hpass_for_ica = 1.0
lpass_for_ica = 100.0

# Start from raw_notched (notched but unbandpassed) to preserve the 48-100 Hz band
raw_for_ica = raw_notched.copy().filter(
    l_freq=hpass_for_ica,
    h_freq=lpass_for_ica,
    picks='eeg', method='fir', phase='zero', verbose=False,
)
# Replicate the spatial state of raw_filtered (add FCz, montage, bads, interpolate, CAR, annotations)
raw_for_ica = mne.add_reference_channels(raw_for_ica.load_data(), ref_channels=["FCz"])
raw_for_ica.set_montage(montage)
raw_for_ica.info["bads"] = list(raw_filtered.info["bads"])
raw_for_ica.interpolate_bads(reset_bads=False)
raw_for_ica, _ = mne.set_eeg_reference(raw_for_ica, ref_channels="average", copy=False)
raw_for_ica.set_annotations(raw_filtered.annotations)
```

**Bug latente arreglado como pre-requisito.** Para que este cambio funcione, `raw_notched` tiene que quedarse en el estado "notched-only". La versión previa del script hacía `raw_filtered = raw_notched.filter(...)` — pero `.filter()` es **in-place** en MNE, así que tras esa línea `raw_notched` y `raw_filtered` eran el mismo objeto en estado 0.1–48 Hz. Consecuencia: el PSD plot de la Sección 3C con título "After Notch Filter only" en realidad mostraba los datos completamente bandpasseados. Fix: agregar `.copy()` en la línea 345 para que `raw_notched` preserve su estado notch-only.

**Implementación (2026-04-15) — aplicada.** Tres cambios coordinados:

1. Línea 345: `raw_filtered = raw_notched.copy().filter(...)` (antes: `raw_notched.filter(...)` in-place). Arregla el bug latente del PSD plot.
2. Bloque de creación de `raw_for_ica` reescrito para arrancar desde `raw_notched` con LP 100 Hz y replicar el estado espacial de `raw_filtered`.
3. Log actualizado: `ica_fit_lpass = 100.0`, `analysis_lpass = 48.0`, `two_copy_pattern = "analysis_0.1-48Hz_ica_1-100Hz"`, `ica_wide_band_from_raw_notched = True`.

**Riesgo residual.** Si el futuro código tocara los "bads" o las anotaciones de `raw_filtered` después del bloque de creación de `raw_for_ica` pero antes del `ica.fit`, las dos copias divergirían silenciosamente. En la versión actual del script no pasa, pero vale como nota para mantener el orden.

**Validación pendiente.** El sanity test post-refactor debería:
- Confirmar que el PSD plot de la Sección 3C muestra ahora claramente el rango 48–100 Hz no filtrado (antes del fix parecía un duplicado del plot bandpasseado).
- Verificar que ICA sobre 1–100 Hz encuentra más componentes musculares que sobre 1–48 Hz (o los mismos con scores más altos). Un delta nulo sugeriría que el cambio no aportó nada y conviene rollbackearlo.
- Comparar la lista de exclusiones ICLabel antes/después.

**Prioridad:** 🟠 Media (mayor de lo que estaba etiquetado originalmente — la ganancia es tangible y el costo, tras el fix del bug latente, es acotado).

#### **R-13. Snap-to-sample de eventos al cargar merged_events**

Línea 488–495: creás `mne.Annotations` directamente desde los onsets del TSV. Si los onsets fueron calculados con precisión > 1 sample, al cargarlos en un raw sampleado a 500 Hz quedan "entre muestras" y MNE redondea. Para decoding temporal preciso (MVNN, SVC por ventanas), esto puede introducir jitter de hasta 1 muestra (2 ms a 500 Hz, antes eran 4 ms a 250 Hz — **mejora con el cambio a 500 Hz** 🎉).

**Acción:** nada urgente, pero vale la pena loggear la precisión sub-muestra al menos una vez por run.

**Hallazgo empírico (2026-04-15) — el jitter depende del TSV, NO del raw.** Al correr el sanity test comparando sub-19 (a 250 Hz por downsample histórico) vs sub-23 (a 500 Hz nativos), el resultado fue contraintuitivo:

| Run | sfreq raw | Max jitter | Mean jitter |
|---|---|---|---|
| sub-19 task-01 run-002 | 250 Hz | 0.053 ms | 0.027 ms |
| sub-23 task-01 run-002 | 500 Hz | 0.683 ms | 0.344 ms |

El sample a 500 Hz tiene **más** jitter que el de 250 Hz. La explicación: el TSV `merged_events` almacena onsets computados con una precisión dada (probablemente los timestamps LSL originales del XDF). En sub-19 el raw fue resampleado a 250 Hz en una versión vieja de `read_xdf.py`, y los onsets del TSV quedaron alineados a esa misma grilla → jitter ≈ 0. En sub-23 el raw está a 500 Hz nativo pero los onsets del TSV no se recomputaron para esa grilla → jitter no-nulo pero dentro del máximo teórico de 1 ms (medio sample @ 500 Hz).

**Implicancia.** El logging de R-13 mide "jitter entre TSV y grilla del raw", no "precisión absoluta de los eventos". La precisión absoluta real depende de cómo se computó el TSV upstream (marcadores LSL, sincronización de streams, etc.), que es independiente de este script. Para CAMPEONES, los valores observados (< 1 ms @ 500 Hz) son compatibles con MVNN + SVC por ventanas de decoding, así que el logging es útil como canario (detecta regresiones) pero no como metric de calidad absoluta.

**Implementación (2026-04-15) — aplicada.** Se agregó un bloque justo antes de la creación de `event_annotations` que calcula el jitter entre los onsets del TSV y la grilla de muestras del raw:

```python
sfreq = raw_filtered.info['sfreq']
onsets_samples_float = events_df['onset'].values * sfreq
onsets_samples_int = np.round(onsets_samples_float).astype(int)
jitter_samples = np.abs(onsets_samples_float - onsets_samples_int)
jitter_ms = jitter_samples * 1000.0 / sfreq
print(f"Snap-to-sample jitter (events → {sfreq:.0f} Hz grid): "
      f"max={jitter_ms.max():.4f} ms, mean={jitter_ms.mean():.4f} ms")
log_preprocessing.log_detail("event_onset_sfreq", float(sfreq))
log_preprocessing.log_detail("event_onset_jitter_max_ms", float(jitter_ms.max()))
log_preprocessing.log_detail("event_onset_jitter_mean_ms", float(jitter_ms.mean()))
```

Esto no cambia ningún dato ni anotación: es puramente instrumentación. El valor máximo posible de jitter es `0.5 / sfreq * 1000` ms (medio período de muestreo), que para 500 Hz es 1 ms. Si alguna vez el log reporta > 1 ms, es señal de que el TSV tiene onsets en otra unidad o con un origen distinto al esperado — bug de upstream, no del preproc.

#### **R-14. Reporte HTML: agregar "before/after ICA" superpuesto**

El reporte actual agrega raw, filtered, ica, y final por separado. Para detectar que ICA no sobrecorrige, MNE sugiere usar `report.add_ica(..., inst=raw_filtered)` con los plots de `plot_overlay` habilitados. Chequear que efectivamente estén saliendo en el HTML.

**Implementación (2026-04-15) — aplicada.** Dos cambios al bloque del reporte ICA:

1. **Orden confirmado.** `ica.exclude = to_exclude` está seteado en la línea ~953, **antes** de `report.add_ica` (línea ~977). Esto es necesario para que el plot de overlay refleje la decisión real de exclusión (R-7) y no un estado vacío. Ya estaba bien; el review doc queda con la verificación explícita.
2. **`inst=raw_for_ica` + `n_jobs=1` + comentario extenso.** La llamada quedó así:

   ```python
   report.add_ica(
       ica,
       title="ICA",
       inst=raw_for_ica,
       n_jobs=1,
   )
   ```

   `inst=raw_for_ica` es lo que dispara los tres plots en el HTML: topografías de todos los componentes, time courses de los excluidos, y — el crítico — `plot_overlay()` con el antes/después de aplicar `ica.exclude`. Se usa `raw_for_ica` (fit copy, 1–100 Hz CAR) en lugar de `raw_filtered` por consistencia con el decomposition: los time courses y topografías se generaron sobre `raw_for_ica`, así que el overlay también debería mostrar ese dominio. Si se pasara `raw_filtered` aquí, el overlay mezclaría dos representaciones (la decomposition vive en 1–100 Hz CAR, el overlay se vería en 0.1–48 Hz pre-CAR) y sería confuso.

   `n_jobs=1` mantiene el reporte determinístico. Se deja como hook para subir a 2/4 si en runs largos el reporte tarda demasiado.

**Nota sobre over-correction detection.** El propósito declarado de R-14 es detectar ICA over-correction — que ICA remueva señal neural junto con el artefacto. El overlay plot es la herramienta canónica para eso en la literatura (MNE tutorial, Winkler 2015). Con R-7 Variante A ya limitando exclusiones por brain floor, el riesgo de over-correction debería ser bajo, pero el overlay sigue siendo necesario como **evidencia positiva** por run (no sólo razonamiento a priori).

#### **R-15. (Cohen §7.7) Considerar surface Laplacian (CSD) como paso downstream, no en preproc**

**Contexto.** Cohen recomienda el surface Laplacian (o Current Source Density, CSD) como spatial filter para **minimizar la conducción volumétrica** y obtener topografías más localizadas — particularmente útil en análisis de conectividad, en separar efectos occipitales de parietales, y en aislar EOG del resto del scalp. PCA **no** resuelve este problema (Cohen §7.7).

**Por qué no en el preproc.** El Laplacian es una transformación que cambia drásticamente la naturaleza de los datos (pasás de voltaje absoluto a una aproximación de la segunda derivada espacial). Para decoding con MVNN + SVC, **puede** mejorar el rendimiento pero **puede** empeorarlo, según cuánto dependa tu clasificador de señales distribuidas. No querés tomar esa decisión a nivel de preproc, sino en un script downstream donde podés comparar con y sin.

**Acción sugerida.** No cambiar `04_preprocessing_eeg.py`. En cambio, agregar una opción en el script de decoding para aplicar `mne.preprocessing.compute_current_source_density()` sobre la copia previa al MVNN, y comparar accuracies. Cohen §7.7 lo describe como un preprocessing step, pero en la práctica moderna con MNE conviene tratarlo como una variante del análisis.

**Referencia.** Cohen §7.7 y §22 (surface Laplacian construcción). `mne.preprocessing.compute_current_source_density`.

**Prioridad:** 🟡 Baja. Experimentación downstream, no un cambio al preproc.

#### **R-16. Persistir el objeto ICA completo para trazabilidad post-hoc**

**Problema.** El script hoy guarda el raw post-ICA (`desc-preproc_eeg.vhdr`) y los índices excluidos en el log JSON (`ica_components_excluded`), pero **no guarda el objeto `ICA` completo** (matriz de unmixing + `ica.exclude`). Eso significa que:
- Si más adelante querés hacer análisis de sensibilidad (cambiar `ica.exclude` y re-aplicar), tenés que **re-correr ICA** entero — caro y no determinista si cambió algún hyperparam.
- No podés recuperar las topografías de componentes sin re-fittear.
- La trazabilidad del log es "qué índices se excluyeron", no "qué decomposición produjo esos índices".

**Cambio concreto.** Después de `ica.apply(inst=raw_filtered)`:

```python
ica_fname = os.path.join(
    derivatives_folder,
    f"sub-{subject}", f"ses-{session}", "eeg",
    f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_desc-ica_ica.fif",
)
os.makedirs(os.path.dirname(ica_fname), exist_ok=True)
ica.save(ica_fname, overwrite=True)
log_preprocessing.log_detail("ica_object_saved", ica_fname)
```

BIDS-derivatives convention: sufijo `_ica.fif` con `desc-ica`. Recuperable con `mne.preprocessing.read_ica(path)` — te devuelve la decomposición literal + `ica.exclude`, sin necesidad de re-fittear nada.

**Implementación (2026-04-15) — aplicada.** Está en el script inmediatamente después del bloque `ica.apply`, y loggea `ica_object_saved` con el path absoluto.

**Prioridad:** 🟠 Media. Solicitud explícita del usuario; costo bajo (6 líneas), beneficio alto si en algún momento querés re-evaluar decisiones ICA sin re-correr el pipeline.

#### **R-17. Logging extendido de canales malos por criterio PyPREP (sibling de R-10)**

**Problema.** La detección de canales malos en la Sección 5 (`NoisyChannels.find_all_bads(ransac=True)`) hoy loggeaba sólo `bad_channels` — la lista plana de canales flagueados — pero no por qué cada canal fue flagueado. PyPREP internamente corre múltiples criterios (`bad_by_ransac`, `bad_by_correlation`, `bad_by_deviation`, `bad_by_hfnoise`, `bad_by_nan`, `bad_by_flat`, `bad_by_dropout`, `bad_by_SNR`) y sus resultados quedan expuestos como atributos del objeto `NoisyChannels`, pero el pipeline los descartaba. Eso impedía responder preguntas de QC como:

- "¿Qué criterio está flagueando más canales a lo largo de los sujetos?"
- "¿El canal X fue marcado por un sólo criterio débil o por varios criterios fuertes?"
- "¿Tengo un problema sistemático de RANSAC vs un problema de high-frequency noise?"

Sin esa información, el único recurso para auditar una decisión era re-correr PyPREP.

**Cambio concreto.** Después de `nd.find_all_bads(ransac=True, ...)`, introspectar los atributos `bad_by_*` del objeto `NoisyChannels` e invertir el mapping para obtener, por canal, qué criterios lo flaguearon:

```python
# Introspect all bad_by_* attributes populated by find_all_bads
bads_by_criterion = {}
for attr in dir(nd):
    if not attr.startswith("bad_by_") or attr.startswith("_"):
        continue
    val = getattr(nd, attr)
    if isinstance(val, (list, tuple, set)):
        bads_by_criterion[attr.replace("bad_by_", "")] = list(val)

# Reverse mapping: for each flagged channel, which criteria triggered?
criteria_per_channel = {
    ch: sorted([crit for crit, chs in bads_by_criterion.items() if ch in chs])
    for ch in bads
}

log_preprocessing.log_detail("bad_channels_by_criterion", bads_by_criterion)
log_preprocessing.log_detail("bad_channels_criteria_per_channel", criteria_per_channel)
```

**Por qué introspección y no hardcodear los nombres de los atributos:** PyPREP puede agregar criterios nuevos en versiones futuras. `dir(nd)` + filtro por prefijo `bad_by_` captura automáticamente cualquier criterio nuevo sin tocar este código.

**Además se imprime** un resumen en stdout con:
- Breakdown por canal: `ch_name → crit1, crit2, ...`
- Totales por criterio: `bad_by_ransac n=3 ['Fp1', 'T7', 'O1']`

**Relación con R-10.** Es el mismo principio — **logging per-ítem con trazabilidad granular** — pero aplicado a la detección de canales malos en vez de a la exclusión de componentes ICA. Se llama R-17 y no R-10b para mantener script ↔ docs con mapping 1:1 sin sufijos.

**Implementación (2026-04-15) — aplicada.** Está en el script en la Sección 5 (líneas ~499–532), con el comentario `# R-17:` y el logging de `bad_channels_by_criterion` + `bad_channels_criteria_per_channel`.

**Prioridad:** 🟡 Baja/Media. Nice-to-have con beneficio claro para QC cross-subject.

---

## Recomendaciones específicas para la naturaleza de tus datos

### **D-1. Videos afectivos con duración variable → continuo con anotaciones es la decisión correcta**
No epochear y guardar continuo con `Annotations` de duración variable es idiomáticamente el approach correcto en MNE para este caso. Los scripts downstream (27, 27b, 38–41) ya manejan el epoching basado en anotaciones. ✅

### **D-2. Contraste decoding vs luminancia → tratar a los videos de luminancia como condición control**
En tu pipeline actual todos los videos (afectivos, luminancia, calm, fixation) pasan por el mismo preprocesamiento. Esto es correcto — **la corrección a nivel preproc debe ser idéntica entre condiciones** para que el decoding sea interpretable (no querés que la diferencia entre clases venga de un filtro distinto aplicado a cada condición).

### **D-3. VR = movimiento > EEG estándar, considerar:**
- Rejection PTP (peak-to-peak) más permisivo downstream (autoreject local en vez de global).
- Bajar `find_bads_muscle` a 0.5 (default) si ves que quedan componentes musculares que pasan el filtro 0.7.
- Incluir canales X/Y/Z (accelerómetro) como regresores en algún script de QC para ver si los artefactos correlacionan con movimiento real.

### **D-4. 500 Hz da margen para análisis gamma/time-frequency**
Con el cambio que hicimos, ahora podés resolver hasta 250 Hz. Aprovechalo: si en algún momento querés mirar beta alta o gamma en videos afectivos, **no recortes la banda a 48 Hz en preproc** — usá 100 Hz (o 120) y dejá que los scripts downstream filtren según el análisis.

### **D-5. Variabilidad entre sujetos (Jero: "los 5 no deberían tener problemas")**
Cuando corras los 5 sujetos, compará automáticamente:
- Cantidad de canales malos detectados por PyPREP (debería ser 0–3 por sujeto; >5 es sospechoso).
- Cantidad de componentes ICA excluidos por ICLabel (debería ser 3–10 de 31; >15 es sospechoso).
- Duración de segmentos marcados `bad` (gaps fuera de merged_events) — si un sujeto tiene >30% bad, algo está mal en merged_events.

Si querés, podés agregar un script de QC grupal que lea los JSON logs y tire una tabla. Bajo esfuerzo, alto valor.

---

## Orden de operaciones sugerido (síntesis)

Propuesta de orden canónico, incorporando R-1, R-3, R-4:

```
1. read_raw_bids
2. Verify montage
3. Notch 50+100 Hz
4. Crear dos copias:
   a. raw_analysis = copy.filter(l_freq=0.1, h_freq=40)
   b. raw_for_ica  = copy.filter(l_freq=1.0, h_freq=40)   # misma base, distinto HPF
5. Cargar merged_events → anotaciones (aplicar a AMBAS copias)
6. Marcar gaps como 'bad' (aplicar a AMBAS copias)
7. PyPREP NoisyChannels sobre raw_for_ica → bads
8. raw_analysis.info["bads"] = bads (misma lista)
9. Interpolate bads en AMBAS copias
10. add_reference_channels("FCz") + set_montage en AMBAS
11. set_eeg_reference("average") en AMBAS
12. ICA fit en raw_for_ica (reject_by_annotation=True)
13. find_bads_eog (R_EYE + L_EYE), find_bads_ecg (ECG), find_bads_muscle
14. ICLabel sobre raw_for_ica ya average-referenced (ICLabel OK)
15. ica.exclude = unión(pattern_matching, iclabel>=0.8)
16. ica.apply(raw_analysis)  ← aplicar sobre la copia con HPF laxo
17. Save raw_analysis como desc-preproc
18. Reporte HTML + log
```

Este orden:
- ✅ Cumple los supuestos de ICLabel (average ref antes de clasificar).
- ✅ Evita rank deficiency al interpolar antes de ICA.
- ✅ Preserva componentes lentos en la data final (HPF 0.1) pero da a ICA señal limpia (HPF 1.0).
- ✅ Mantiene la estrategia de cobertura total con anotaciones `bad` para gaps.

---

## Qué NO cambiar (está bien como está)

- ✅ **Picard como algoritmo de ICA** — más rápido y robusto que Infomax/FastICA (MNE lo recomienda explícitamente).
- ✅ **`random_state=42`** — reproducibilidad.
- ✅ **`reject_by_annotation=True`** — ICA solo aprende de segmentos válidos.
- ✅ **PyPREP NoisyChannels** — método estándar y open-source, con RANSAC es robusto.
- ✅ **Guardar continuo con anotaciones en vez de epochs** — idiomáticamente correcto para variable-duration.
- ✅ **Reporte HTML con `mne.Report`** — excelente para trazabilidad.
- ✅ **Logging JSON centralizado por sujeto/run** — excelente para post-hoc QC.
- ✅ **Interpolación spline (default de MNE)** — estándar.
- ✅ **FIR zero-phase filters** — estándar Luck/MNE/Makoto.
- ✅ **Drop de anotaciones originales antes del save intermedio** — correcto para evitar confusión con events reales.

---

#### **R-18. ICA con `picard` + `ortho=False, extended=True` para compatibilidad con ICLabel**

**Problema.** En cada run, `label_components` emite un `RuntimeWarning`:

> The provided ICA instance was fitted with a 'picard' algorithm. ICLabel was designed with extended infomax ICA decompositions. To use the extended infomax algorithm, use the 'mne.preprocessing.ICA' instance with the arguments 'ICA(method='infomax', fit_params=dict(extended=True))' (scikit-learn) or 'ICA(method='picard', fit_params=dict(ortho=False, extended=True))'.

ICLabel fue entrenado sobre decomposiciones de extended infomax. Picard con sus parámetros default produce componentes parecidos pero **no idénticos** al extended infomax — las topografías y time courses pueden variar, y las probabilidades ICLabel pueden no reflejar exactamente la distribución de entrenamiento del clasificador.

**Solución.** El propio warning indica la configuración compatible: `picard` con `fit_params=dict(ortho=False, extended=True)`. Esta variante de picard (llamada "picard-o" en la literatura) emula extended infomax numéricamente, con convergencia más rápida que infomax puro pero manteniendo la forma de la decomposition que ICLabel espera. Referencia: Ablin et al. 2018 *Faster independent component analysis by preconditioning with Hessian approximations*.

**Cambio concreto (estimado, 1 línea).** Buscar la creación del objeto ICA en el script (`ica = mne.preprocessing.ICA(n_components=..., method='picard', ...)`) y agregar `fit_params=dict(ortho=False, extended=True)` a los kwargs. Loggear la configuración.

**Impacto esperado sobre las exclusiones actuales.**
- Las topografías de componentes pueden rotar ligeramente → IDs de componentes no son comparables 1:1 contra las decomposiciones actuales (picard default).
- Las probabilidades ICLabel deberían ser más confiables, especialmente para clases de baja frecuencia como "channel noise", "line noise", o los bordes de "other".
- El brain floor (R-7) y el threshold 0.85 pueden necesitar recalibración ligera, pero la lógica OR + veto sigue siendo válida.

**Costo.** La refitting completa es determinística (random_state=42), entonces re-correr el mismo run produce output reproducible. Para un batch de 5 sujetos, significa re-correr el preproc completo (~15–20 min total). El objeto `desc-ica_ica.fif` de R-16 permite comparar antes/después sin re-fittear, si se cachea la versión previa.

**Validación pendiente.** Correr sub-23 task-01 run-002 end-to-end con el cambio y comparar:
- Lista de componentes excluidos contra la versión actual ({1,2,3,4,7,8,10,13,14,16,18,22,23,24,26,28}).
- Si cambia drásticamente (≥ 50% de overlap perdido), investigar manualmente los casos divergentes antes de batch-ear.
- Si cambia ligeramente (< 20% diferencia), aceptar el cambio y tomar la versión picard-o como baseline.

**Prioridad:** 🟡 Media. Es un one-liner, el costo es bajo, y elimina un warning que aparece en todos los runs. Pero no es urgente porque los resultados actuales ya son coherentes (brain floor, pattern matching y ICLabel se refuerzan mutuamente).

**Descubrimiento:** Warning observado en el sanity test end-to-end de sub-19 task-01 run-002 y sub-23 task-01 run-002 (2026-04-15).

---

## Resultado del sanity test end-to-end (2026-04-15)

Dos runs corridos end-to-end en modo `--auto` para validar los cambios v5–v9:

### sub-19 task-01 run-002 (250 Hz — fallback histórico)

Primer sanity test. Reveló un problema de fondo: sub-19 está a **250 Hz**, no 500 Hz como asumía el review doc. El análisis forense mostró que una versión vieja de `read_xdf.py` downsampleaba a 250 Hz; la versión actual (línea 356: "Manteniendo sampling rate nativo del EEG") ya no lo hace, pero sub-19 fue generado con la versión vieja y no se regeneró. Sub-19 tampoco tiene XDF fuente localmente disponible (`data/sourcedata/xdf/sub-19/` no existe), así que no se puede re-correr `read_xdf` sin recuperar el XDF.

**Métricas del run:**
- Bad channels (PyPREP): 3 (FC6, FC1, Fp2) — todos por correlación (R-17 confirma).
- ICA: 28 componentes, ICA fit 50.7s.
- R-7 exclusiones: 13 / 28. Pattern candidates: 18. ICLabel ≥0.85 artifact: 8. **Brain-floor vetó 5** (ICA 4, 14, 17, 23, 27).
- R-13 jitter: max 0.053 ms, mean 0.027 ms (comportamiento anómalo explicado arriba).

**Outcome:** Pipeline completa end-to-end sin errores; HTML report + `desc-ica_ica.fif` + `desc-preproc_eeg.vhdr` escritos. Todos los campos de R-8/R-10/R-11/R-12/R-13/R-14/R-16/R-17 presentes en el JSON log.

### sub-23 task-01 run-002 (500 Hz nativo — validación canónica)

Segundo sanity test tras regenerar raw BIDS con `read_xdf --subject 23` (los XDFs fuente confirmaron sfreq nativa 499.998 Hz ≈ 500 Hz). Este es el run de validación "real" del pipeline.

**Métricas del run:**
- Bad channels (PyPREP): 1 (Fz) — por correlación. Señal mucho más limpia que sub-19.
- ICA: 30 componentes, ICA fit 40.5s (más rápido que sub-19 a pesar del 2× sfreq, porque sub-19 perdió 3 canales vs 1 de sub-23).
- R-7 exclusiones: 16 / 30. Pattern candidates: 18. ICLabel ≥0.85 artifact: 7. **Brain-floor vetó 2** (ICA 11, 15).
- **ICA011 es el caso paradigmático.** Fue pattern-matched por `find_bads_ecg` (componente ECG único detectado), pero ICLabel lo clasificó como `brain` con probabilidad 0.643. El brain floor (threshold 0.30) lo vetó → KEEP. Esto es exactamente la razón por la que el brain floor existe: pattern matching de ECG puede confundirse con componentes cerebrales de baja frecuencia, e ICLabel los rescata.
- R-13 jitter: max 0.683 ms, mean 0.344 ms. Dentro del límite teórico de 1 ms (medio sample @ 500 Hz). ✓

**Outcome:** Pipeline completa end-to-end sin errores; artefactos equivalentes a sub-19 en disco. Todos los campos del log JSON presentes con valores coherentes (sfreq 499.998, ica_fit_lpass 100, two_copy_pattern, pyprep_channels_with_positions 31, etc.).

### Lecciones del sanity test

1. **R-7 Variante A hace trabajo real en ambos runs.** El brain floor vetó 5 y 2 componentes respectivamente — no es un parámetro decorativo. El caso de sub-23 ICA011 (ECG pattern rescatado por ICLabel) es ejemplar.
2. **El two-copy pattern (R-12) funcionó sin divergencia silenciosa** entre `raw_filtered` (0.1–48 Hz) y `raw_for_ica` (1–100 Hz). El bug latente del notch mislabel quedó arreglado.
3. **R-17 (PyPREP per-criterion) da data útil cross-subject.** Ambos runs flaggearon 100% por `correlation` y 0% por `ransac` — inesperado, vale la pena revisar si hay algo sistemático con el criterio RANSAC en esta configuración de datos (posible tema para el batch de 5 sujetos).
4. **Warning recurrente ICLabel + picard** → formalizado como R-18.
5. **R-13 jitter no es lo que parecía.** Mide desalineación TSV↔grid, no precisión absoluta de eventos. Los valores actuales son OK para decoding por ventanas pero vale la pena no sobre-interpretarlos.
6. **Sub-19 queda a 250 Hz** hasta que se recupere el XDF fuente. No invalida el resto del preproc, pero si los 5 sujetos del batch original incluían sub-19, conviene regenerarlo o excluirlo.

---

## Priorización sugerida (mi opinión, vos decidís)

| Status | Cambio | Costo | Beneficio | Notas |
|---|---|---|---|---|
| ✅ **Implementada (2026-04-15)** | **R-1** — reref average antes de ICLabel | Reordenar ~30 líneas | Correctness de ICLabel (verificado contra source de mne-icalabel) | Nueva sección 7 del script; ver "Implementación" en R-1 |
| ✅ **Implementada (2026-04-15)** | **R-3 (Ruta A)** — interpolar antes de ICA | Mover 2 bloques | Estabilidad de ICA + evita contaminación por average ref | Parte del mismo refactor que R-1; `reset_bads=False` para trazabilidad |
| ✅ **Implementada (2026-04-15)** | **R-4** — dos HPF (0.1 para data, 1.0 para ICA) | Agregar copia + reestructurar | ERPs lentos preservados; es el standard de Luck/MNE | `raw_for_ica = raw_filtered.copy().filter(l_freq=1.0, ...)`; `ica.apply` corre sobre `raw_filtered` |
| ✅ **Implementada (2026-04-15)** | **R-7 Variante A** — exclusión ICLabel con threshold + brain floor | Reescribir ~70 líneas | Elimina heurísticos arbitrarios; decisión defendible en paper | `iclabel_label_components` directo; OR logic; threshold 0.85; brain floor 0.30 |
| ✅ **Implementada (2026-04-15)** | **R-10 (mini)** — logging per-componente de exclusiones | 12 líneas | Auditoría completa de cada decisión ICA sin re-correr | `ica_exclusions_detail` con clase + top prob + brain prob + fuente |
| ✅ **Implementada (2026-04-15)** | **R-16** — persistir objeto ICA completo | 6 líneas | Trazabilidad post-hoc; permite sensitivity analyses sin re-fittear | Guardado en `*_desc-ica_ica.fif`, recuperable con `mne.preprocessing.read_ica` |
| ✅ **Implementada (2026-04-15)** | **R-17** — logging per-criterio de canales malos (PyPREP) | ~30 líneas | QC cross-subject ("qué criterio flaguea más"); sibling de R-10 | Introspección de `bad_by_*` en `NoisyChannels`; `bad_channels_by_criterion` + `bad_channels_criteria_per_channel` en JSON |
| ✅ **Implementada (2026-04-15)** | **R-12** — wide-band ICA copy (1–100 Hz) | ~25 líneas | Mejor muscle detection; ICLabel in-distribution; mejor separación de componentes musculares | `raw_for_ica` ahora arranca desde `raw_notched` con LP 100; spatial state replicado manualmente; bug latente de `raw_notched` in-place arreglado como pre-requisito |
| ❌ **Rechazada (2026-04-15)** | ~~**R-2** — usar R_EYE + L_EYE en `find_bads_eog`~~ | — | — | Diagnóstico empírico sobre sub-27 run-002 confirmó que L_EYE está roto (ver sección R-2) |
| ❌ **Rechazada (2026-04-15)** | ~~**R-5** — bajar LP a 40 Hz~~ | — | — | LP 48 mantenido: redundancia con notch es mínima; bajar invalida el trabajo previo; extender hacia arriba es inviable sin eye tracking (contaminación de microsacadas / EMG — Yuval-Greenberg 2008, Whitham 2007, Goncharova 2003) |
| ✅ **Implementada (2026-04-15)** | **R-8** — justificación + logging de `find_bads_muscle` threshold | 6 líneas + 2 log_detail | Trazabilidad de la elección conservadora del threshold | Comentario in-script con criterios de ajuste; candidato re-filtrado por R-7 brain floor |
| ✅ **Implementada (2026-04-15)** | **R-11** — PyPREP corre con montage explícitamente cargado | ~30 líneas (ya en paralelo) | Asegura que RANSAC use el criterio espacial (no silent-skip) | `BC-32_FCz_modified.bvef` cargado con `on_missing='ignore'` + sanity check de posiciones 3D antes de `NoisyChannels` |
| ✅ **Implementada (2026-04-15)** | **R-13** — logging de snap-to-sample jitter de eventos | ~10 líneas | QC: detecta problemas de precisión sub-muestra en el TSV | Max jitter teórico = 1 ms @ 500 Hz; `event_onset_jitter_max_ms` + `event_onset_jitter_mean_ms` en log |
| ✅ **Implementada (2026-04-15)** | **R-14** — overlay before/after ICA en reporte HTML | 7 líneas + docs | Evidencia positiva por run de que ICA no sobrecorrige | `report.add_ica(..., inst=raw_for_ica, n_jobs=1)` con `ica.exclude` ya seteado |
| 🟡 Pendiente | **R-18** — picard con `ortho=False, extended=True` (picard-o) | 1 línea (`fit_params=dict(ortho=False, extended=True)`) | Compatibilidad formal picard ↔ ICLabel (elimina warning); decomposition consistente con el training set de ICLabel (Ablin 2018; Pion-Tonachini 2019) | Descubierto en el sanity test de sub-19 + sub-23; warning recurrente en todos los runs |
| 🟡 Pendiente | R-6 (zapline) | Agregar dependencia | Mejor si hay línea no estacionaria | Sólo si detectás el problema |
| 🟡 Pendiente | R-9 | Chequeo del TSV | Evalúa si fixation baseline está en ICA fit | Ya verificado manualmente; el cambio de código es nil |
| 🟡 Pendiente | R-15 (CSD/Laplacian downstream) | Script aparte | Localización espacial, evita conducción volumétrica | Experimental, no en preproc |

---

## Próximos pasos concretos

1. **Leé esta revisión y marcá qué recomendaciones aceptás, rechazás o posponés.**
2. Para las aceptadas, decidí si aplicás los cambios antes o después de correr los 5 sujetos. Lo ideal es aplicarlos ANTES (así los 5 quedan consistentes con el re-preproc futuro de sub-27 en el paso A que mencionaste). El costo es ~1-2 hs de refactor + validación sobre 1 run de prueba.
3. Corré sub-19 task-01 acq-a run-002 como prueba end-to-end antes del batch.
4. Si querés, puedo implementar un subset de los cambios que aceptes (ej. R-1 + R-2 + R-3 es un refactor compacto y ataca los problemas más críticos).

---

**v1 — 2026-04-15:** Revisión inicial basada en MNE, MNE-BIDS-Pipeline, Luck (LibreTexts), Makoto, Winkler et al. 2015, Robbins et al. 2020, ICLabel docs.
**v2 — 2026-04-15:** Incorpora *Analyzing Neural Time Series Data* (Cohen, 2014) — Cap. 7 (Preprocessing), Cap. 8 (Artifacts), Cap. 9 (Time-Domain ERPs). Agregada sección dedicada de síntesis por capítulo, citas en R-2, R-3, R-4, R-5, R-7 y nueva R-15 (Surface Laplacian como paso downstream).
**v3 — 2026-04-15:** Meta-revisión del documento contra las fuentes externas (mne-icalabel source code, Luck Appendix 3 en LibreTexts, MNE tutorial de ICA, Cohen 2014 verificado textualmente contra el PDF). Tres cambios:
- **R-1** refuerza la evidencia con la cita textual del source code de `mne_icalabel/iclabel/label_components.py` (CAR + 1–100 Hz son requisitos explícitos del modelo, no interpretación).
- **R-3** corrige una cita inexacta a Luck Cap. 7 que decía "interpolate before the average reference and before ICA". Verificación contra el Appendix 3 de Luck en LibreTexts confirma que Luck en realidad hace lo opuesto: excluye los bads del fit de ICA y los interpola *después*. Ahora el R-3 presenta dos Rutas canónicas (A: Makoto/MNE-BIDS = interpolar antes; B: Luck = excluir del fit e interpolar después) y recomienda Ruta A para CAMPEONES por compatibilidad con R-1.
- **R-4** sube de prioridad Media a **Alta**. Luck Appendix 3 y el tutorial oficial de MNE-ICA explícitamente prescriben el patrón de dos copias (una 0.1–30 para análisis, una 1.0–30 para ICA) como standard, no como optimización. Con los sanity checks de ERPs que ya están en el diario, usar HP 1 Hz en la data final distorsiona silenciosamente los componentes lentos.

**v5 — 2026-04-15:** **R-1, R-3 (Ruta A) y R-4 aceptadas e implementadas** en `scripts/preprocessing/04_preprocessing_eeg.py` en un único refactor. Cambios concretos:
- **R-1 + R-3 Ruta A:** nueva sección 7 del script (`add_reference_channels("FCz")` + `set_montage` + `interpolate_bads(reset_bads=False)` + `set_eeg_reference("average")`) que corre **antes** del fit de ICA. La vieja sección 9 (que hacía estos mismos pasos después de `ica.apply`) fue desmantelada y reemplazada por el alias `raw_interpolate = raw_ica` para mantener compatibilidad con el código downstream de plotting/reporting/save.
- **R-4 (two-copy pattern):** HPF de la analysis copy bajado de 1.0 a 0.1 Hz (sección 3B). Se crea una segunda copia `raw_for_ica = raw_filtered.copy().filter(l_freq=1.0)` justo antes del fit de ICA (sección 8). `ica.fit`, `find_bads_eog/ecg/muscle`, `label_components` (ICLabel) y los plots de ICA operan sobre `raw_for_ica`; pero `ica.apply(inst=raw_filtered)` transfiere las weights a la analysis copy (HPF 0.1) para preservar los componentes ERP lentos (LPP, late positive potentials) relevantes para videos emocionales.
- **ICLabel ahora recibe datos en el espacio para el que fue entrenado** (common average reference + HPF 1 Hz), resolviendo el problema principal de R-1.
- **Validación pendiente:** correr un run end-to-end (sugerido sub-19 task-01 acq-a run-002) y comparar ERPs y clasificaciones ICLabel contra la versión anterior del script.
- Ver las secciones "Implementación" al final de R-1 y R-4 para el detalle técnico del refactor.

**v4 — 2026-04-15:** **R-2 rechazada tras diagnóstico empírico.** Se corrió `src/campeones_analysis/utils/diagnose_eog_channels.py` sobre sub-27 ses-vr task-01 acq-a run-002 para decidir si incluir L_EYE en `find_bads_eog`. Los tests mostraron:
- L_EYE está roto (std=464 µV pero MAD=62 µV → spikes esporádicos de hasta 9.5 mV; correlación con Fp2 = 0.02; sin dipolo de blink coherente).
- R_EYE es funcional (correlación con Fp2 = 0.54; respuesta blink-locked consistente de ~34–40 µV).
- **No existe infraorbital mislabeled** en el montage: la búsqueda por ranking de respuesta blink-locked sobre los 32 canales EEG mostró un candidato espurio (FT10, 639 µV) que resultó ser otro canal roto con el mismo patrón patológico que L_EYE (ptp=9 mV, respuesta dominada por 2–3 outliers con mean ≫ median).
- **Hallazgo secundario:** FT10 debería marcarse como bad channel en sub-27 run-002; TODO para revisarlo en el resto de los sujetos.
- **Decisión:** mantener el código con `ch_name="R_EYE"` únicamente y apoyarse en ICLabel para detección topográfica de componentes oculares.
- Artefactos del diagnóstico: `data/derivatives/diagnostics/eog_channels/sub-27_run-002_eog_diagnostic.png` y `sub-27_run-002_blink_erp_candidates.png`.
- Ver sección "Diagnóstico empírico" dentro de R-2 para el detalle completo de tests y números.

**v6 — 2026-04-15:** **R-7 (Variante A), mini-R-10 y R-16 (nuevo) aceptadas e implementadas** en `scripts/preprocessing/04_preprocessing_eeg.py`. Los tres cambios tocan el bloque ICA (líneas 744–908 aprox.) y conviven con el refactor previo (R-1 + R-3A + R-4). Resumen:

- **R-7 Variante A:** la lógica de exclusión ahora combina pattern matching (`find_bads_eog/ecg/muscle`) con ICLabel vía OR — no AND — y filtra los candidatos por un umbral de probabilidad ICLabel (`ICLABEL_THRESHOLD = 0.85`) más un brain floor (`BRAIN_FLOOR = 0.30`) que veta exclusiones de cualquier componente con probabilidad de cerebro ≥ 0.30, venga de donde venga. Eliminados el heurístico `eog_components[0] < 3` y la inclusión ad-hoc de `channel_artifact_indices`. Para obtener la matriz completa de probabilidades ICLabel (necesaria para el brain floor) se reemplazó `label_components()` por una llamada directa a `iclabel_label_components(raw_for_ica, ica)`, que devuelve `shape (n_components, 7)`. La tabla en stdout ahora imprime top prob + brain prob por componente y un resumen `R-7 EXCLUSION DECISION` con candidatos y vetados.
- **Mini-R-10:** nuevo campo `ica_exclusions_detail` en el log JSON, con clase ICLabel, top prob, brain prob, y flags `by_pattern_matching` / `by_iclabel` por componente excluido. Suficiente para auditar cada decisión sin re-correr ICA. La versión "full R-10" (near-misses + matriz completa de probabilidades) se pospone para si se necesita análisis de sensibilidad post-hoc del umbral. Además se loggean los parámetros de la decisión: `ica_exclusion_iclabel_threshold`, `ica_exclusion_brain_floor`, `ica_exclusion_pattern_candidates`, `ica_exclusion_iclabel_candidates`, `ica_exclusion_vetoed_by_brain_floor`.
- **R-16 (nuevo):** el objeto `ICA` completo (matriz de unmixing + `ica.exclude`) ahora se guarda en disco como `sub-XX_ses-XX_task-XX_acq-X_run-XXX_desc-ica_ica.fif` dentro de `data/derivatives/<pipeline>/sub-XX/ses-XX/eeg/`. Recuperable con `mne.preprocessing.read_ica(path)`. Habilita análisis de sensibilidad post-hoc (cambiar `ica.exclude` y re-aplicar) sin re-fittear ICA. El path también se loggea en el JSON bajo `ica_object_saved`.
- **Sintaxis verificada:** `python -m py_compile scripts/preprocessing/04_preprocessing_eeg.py` → OK.
- **Validación pendiente:** correr sub-19 task-01 acq-a run-002 end-to-end y comparar el set de componentes excluidos contra la versión previa del script. Si el delta es grande, inspeccionar los "vetados por brain floor" a ojo antes de correr los 5 sujetos.

**v7 — 2026-04-15:** **R-17 formalizada y renombrada.** El usuario estaba implementando en paralelo un logging extendido de canales malos por criterio PyPREP, comentado en el script como `# R-10 extended`. Esto creaba una colisión semántica con la R-10 del review (que refiere específicamente al logging de exclusiones ICA). Funcionalmente los dos cambios son independientes (secciones distintas del script, keys JSON distintas, sin overlap de variables), pero el naming ambiguo dificultaba el mapping script ↔ docs. Cambios:
- Renombrado el comentario in-script de `# R-10 extended` a `# R-17` (ver línea 499 de `04_preprocessing_eeg.py`), con nota de que es sibling de R-10 aplicado a PyPREP en vez de a ICA.
- Agregada sección **R-17** en el review doc con el contexto del problema, el código de introspección, la justificación de por qué usar `dir(nd)` en vez de hardcodear los nombres, y el estado ✅ implementada.
- Agregada fila **R-17** en la tabla de priorización.

**v8 — 2026-04-15:** **R-5 rechazada y R-12 implementada** en un mismo ciclo de discusión. Cambios:
- **R-5 rechazada.** Tras discutir la doble dirección del argumento (bajar a 40 Hz vs subir a >50 Hz):
  - Bajar a LP 40 no mejora nada concreto para CAMPEONES (la "redundancia con el notch 50" del review original es un argumento débil — el notch a 50 hace trabajo real dentro de la zona de transición del LP 48; los defaults de BIDS-Pipeline están tuneados para ERP clásico, no decoding; la ganancia esperada en SNR es nula para decoding).
  - Subir el LP hacia gamma (>50 Hz) es **mala idea** para este paradigma específico: la literatura posterior a 2007 (Yuval-Greenberg 2008 *Neuron*; Whitham et al. 2007 *Clin Neurophysiol*; Goncharova et al. 2003 *Clin Neurophysiol*; Cohen 2014 §8.4) muestra que la "gamma" de scalp es abrumadoramente artefacto muscular + microsacadas. CAMPEONES tiene 32 canales (ICA limitada), sin eye tracking de alta resolución, con videos emocionales que inducen diferencias esperables de movimiento ocular entre condiciones → alto riesgo de decodificar microsacadas/EMG en vez de señal neural, indistinguible de "éxito" real sin los controles que no tenemos. Los reviewers de papers serios rechazan gamma de scalp sin eye tracking.
  - Decisión: `lpass = 48.0` mantenido, con comentario in-script explicando la decisión (líneas 327–331 de `04_preprocessing_eeg.py`).
- **R-12 implementada.** El usuario pidió igualmente aprovechar la banda alta en la copia de ICA (sin exponerla al análisis), lo cual es una aplicación legítima de R-12 distinta a mi análisis cauteloso de "dejar para después del sanity test". Cambios:
  - Nueva creación de `raw_for_ica`: arranca desde `raw_notched` con bandpass 1–100 Hz, luego replica el estado espacial de `raw_filtered` (FCz + montage + bads + interpolación + CAR + anotaciones). ~25 líneas nuevas, reemplazando las 10 líneas del R-4 original.
  - **Bug latente arreglado como pre-requisito.** La versión previa del script tenía `raw_filtered = raw_notched.filter(...)` en la Sección 3B. `raw.filter()` es in-place en MNE, así que tras esa línea `raw_notched` quedaba mutado al estado 0.1–48 Hz (perdiendo el estado notch-only). Consecuencia: el PSD plot de la Sección 3C titulado "After Notch Filter only" estaba mostrando los datos bandpasseados (mislabel silencioso). Fix: `raw_filtered = raw_notched.copy().filter(...)` — ahora raw_notched conserva su estado correcto y el PSD plot se vuelve coherente con su título.
  - Log actualizado: `ica_fit_hpass=1.0`, `ica_fit_lpass=100.0`, `analysis_hpass=0.1`, `analysis_lpass=48.0`, `two_copy_pattern="analysis_0.1-48Hz_ica_1-100Hz"`, `ica_wide_band_from_raw_notched=True`.
  - Sintaxis verificada: `python -m py_compile` → OK.
- **Validación pendiente** del sanity test cubre ahora también: (a) el PSD plot de Sección 3C debería mostrar diferencia visible entre "After Notch Filter" (con energía en 48–100 Hz preservada) y "Final Filtered Data" (banda 0.1–48 Hz); (b) la lista de ICs excluidos por `find_bads_muscle` sobre el nuevo `raw_for_ica` vs la versión anterior; (c) que ICLabel no reclame por el estado del input.

**v9 — 2026-04-15:** **R-8 + R-11 + R-13 + R-14 implementadas en una sola tanda.** Cuatro cambios menores pero con alto valor de QC/trazabilidad, todos en `scripts/preprocessing/04_preprocessing_eeg.py`:

- **R-8** (línea ~858 del script). Comentario extenso justificando `find_bads_muscle(threshold=0.7)` — conservador vs el default MNE 0.5, con criterios escritos para subir/bajar según lo que muestre el reporte HTML. La decisión final se re-filtra por ICLabel + brain floor (R-7), así que este paso actúa como generador de candidatos. Nuevos logs: `find_bads_muscle_threshold`, `find_bads_muscle_components`.
- **R-11** (línea ~467 del script). Ya estaba aplicada en paralelo por el usuario antes de esta tanda (misma dinámica que R-17 en la v7). La versión implementada excede la mínima propuesta: además de confirmar que hay montage, carga explícitamente `BC-32_FCz_modified.bvef` con `on_missing='ignore'` (FCz no existe como canal en ese punto, es referencia), y corre un sanity check que cuenta canales con posiciones 3D válidas antes de llamar a `NoisyChannels`. Logs: `pyprep_montage_applied_before_prep`, `pyprep_channels_with_positions`. Convive con R-17 — R-11 asegura input, R-17 introspecta output.
- **R-13** (línea ~582 del script). Nuevo bloque antes de crear `event_annotations` que calcula el jitter entre los onsets del TSV y la grilla de muestras del raw. No cambia ningún dato: es instrumentación. El max posible @ 500 Hz es 1 ms; si algún run reporta > 1 ms es bug upstream. Logs: `event_onset_sfreq`, `event_onset_jitter_max_ms`, `event_onset_jitter_mean_ms`.
- **R-14** (línea ~984 del script). `report.add_ica` ahora pasa `inst=raw_for_ica` (consistente con el dominio del fit: 1–100 Hz CAR) y `n_jobs=1`, con comentario extenso explicando que `inst` es el parámetro que dispara el plot de overlay (before/after `ica.exclude`), que este overlay es la evidencia canónica contra ICA over-correction, y que el orden es correcto: `ica.exclude` está seteado ~25 líneas antes en el script. El cambio cierra el loop del pipeline ICA: decomposition → exclusion decision → visual evidence en el HTML.

Sintaxis verificada: `python -m py_compile scripts/preprocessing/04_preprocessing_eeg.py` → OK.

Validación pendiente (se acumula con los v5–v8 previos): correr sub-19 task-01 acq-a run-002 end-to-end y revisar el HTML del reporte para ver el overlay de R-14 + verificar que los logs de R-8/R-11/R-13 salen en el JSON como corresponde.

**v10 — 2026-04-15:** **Sanity test end-to-end ejecutado + R-18 formalizada + hallazgo empírico sobre R-13.** Dos runs corridos en modo `--auto` para validar los cambios v5–v9 acumulados:

- **sub-19 task-01 run-002** reveló un problema upstream: el raw está a **250 Hz**, no 500 Hz como asumía la doc. Análisis forense: la versión actual de `read_xdf.py` (línea 356) ya no downsamplea, pero sub-19 fue generado con una versión vieja y su XDF fuente no está disponible (`data/sourcedata/xdf/sub-19/` no existe). No se puede re-correr `read_xdf --subject 19` sin recuperar el XDF. Pipeline completa sin errores de todos modos; todos los campos nuevos de los logs presentes.
- **sub-23 task-01 run-002** se eligió como validación canónica (XDF fuente disponible, 500 Hz nativo confirmado). Corrido tras regenerar raw BIDS con `read_xdf --subject 23`. Resultado: 1 bad channel (Fz, correlación), 30 componentes ICA, 16/30 excluidos, brain-floor vetó 2 (ICA 11 y 15). Caso paradigmático: ICA011 fue pattern-matched por `find_bads_ecg` pero ICLabel lo clasificó como `brain` con prob 0.643 → brain floor lo rescató. Esto valida R-7 Variante A como no-decorativo. Pipeline completa sin errores; todos los logs coherentes.
- **R-18 formalizada (nueva).** El warning recurrente "The provided ICA instance was fitted with a 'picard' algorithm. ICLabel was designed with extended infomax ICA decompositions" apareció en ambos runs. Agregada nueva sección R-18 proponiendo el fix canónico (`fit_params=dict(ortho=False, extended=True)`, conocido como "picard-o", Ablin 2018). Es un one-liner pendiente, no urgente.
- **R-13 hallazgo empírico (agregado a la sección R-13).** Contraintuitivamente, el run de 500 Hz (sub-23) reportó **más** jitter (max 0.683 ms) que el de 250 Hz (sub-19, max 0.053 ms). Explicación: el TSV de sub-19 fue construido sobre la misma grilla de 250 Hz del raw (del pipeline viejo), entonces jitter TSV↔grid ≈ 0. El TSV de sub-23 tiene los timestamps LSL originales que no alinean a múltiplos exactos de 2 ms. Conclusión: **R-13 mide desalineación TSV↔grid, no precisión absoluta de eventos**. Los valores siguen siendo OK para decoding (< 1 sample @ 500 Hz) pero no hay que sobre-interpretarlos como "precisión del trigger".
- **Lecciones del sanity test** (ver nueva sección "Resultado del sanity test end-to-end"): R-7 hace trabajo real, R-12 two-copy no diverge, R-17 muestra que todos los bads son por correlación (ninguno por RANSAC — inesperado, investigar en batch), R-18 warning recurrente, R-13 jitter es TSV-dependent, sub-19 queda anómalo hasta recuperar XDF.

Ningún cambio en `scripts/preprocessing/04_preprocessing_eeg.py` en esta versión. Los cambios son todos documentales + el descubrimiento de R-18.

**Actualizar si cambian las referencias o el script.**
