### Esqueleto del pre-registro — CAMPEONES

Adaptado del template "OSF Preregistration" y del modelo rzbv8. Las secciones siguen el template oficial de OSF. El contenido entre `[corchetes]` es lo que falta completar.

* * *

#### Título

Multimodal Physiological Decoding of Affective States During Immersive Virtual Reality: A Pre-Registered Analysis of EEG, EDA, ECG, and Respiration

[Revisar con Enzo/Diego]

#### Descripción

[Párrafo de contexto general del proyecto CAMPEONES: emociones en VR, multimodal, decoding]

* * *

#### Q2 — Hypotheses

**Bloque 1 — Validación técnica (benchmarking)**

**H1 — Decodificación de cambios de luminancia:** Las features de EEG (bandpower y/o señal temporal con MVNN+SVC) permiten decodificar above chance (>50%, binary) si una época corresponde a pre o post cambio de luminancia en un video, usando Leave-One-Run-Out cross-validation dentro de cada sujeto.
- *Dirección:* accuracy > 50% (permutation test, p < 0.05).
- *Justificación:* El VEP occipital post-cambio de luminancia es un efecto fisiológico robusto. Si el pipeline no lo captura, hay un problema técnico.

**Bloque 2 — Fisiología periférica y arousal**

**H2 — EDA predice arousal:** Las features de actividad electrodérmica (SCL, SCR amplitude, SCR frequency) correlacionan positivamente con los ratings de arousal (SAM post-video).
- *Dirección:* correlación positiva (r > 0, p < 0.05, corregido por múltiples comparaciones).
- *Justificación:* La activación simpática (reflejada en EDA) es un correlato fisiológico clásico del arousal subjetivo (Boucsein et al. 2012).

**H2b — ECG predice arousal:** Las features cardíacas (HR, HRV — RMSSD, SDNN) correlacionan con los ratings de arousal.
- *Dirección:* HR correlaciona positivamente, HRV negativamente con arousal.

**H2c — Respiración predice arousal:** La tasa respiratoria y su variabilidad correlacionan con arousal.
- *Dirección:* tasa respiratoria correlaciona positivamente con arousal.

**H2d — Modelo multimodal periférico:** Un modelo combinando features de EDA + ECG + Resp predice arousal mejor que cada modalidad por separado.
- *Test:* comparación de modelos (likelihood ratio test o cross-validated accuracy).

**Bloque 3 — EEG y arousal**

**H3 — EEG predice arousal:** Las features de EEG (bandpower de 5 bandas en 32 canales) permiten predecir el nivel de arousal (high vs low, median split de ratings SAM) above chance, usando LORO-CV dentro de cada sujeto.
- *Dirección:* accuracy > 50% (permutation test, p < 0.05).

**H3b — Multimodal completo:** Un modelo combinando features de EEG + periféricos predice arousal mejor que EEG solo o periféricos solos.
- *Test:* comparación de accuracy por cross-validation + permutation test para la diferencia.

[PENDIENTE: decidir si H3 usa decoding (SVM/logistic) o modelos estadísticos (LMM). El modelo de rzbv8 usa LMM porque es un estudio de inferencia; nuestro proyecto es más de predicción. Discutir con Diego.]

[PENDIENTE: decidir la granularidad de arousal — binary (median split) vs ordinal vs continuo. Binary es más comparable con luminancia; continuo es más potente estadísticamente pero requiere regresión en lugar de clasificación.]

* * *

#### Q3 — Study type

Observational — participants are exposed to emotional stimuli in immersive VR and provide subjective reports. No experimental manipulation.

[Nota: los videos varían en contenido emocional (arousal/valencia), pero no hay manipulación experimental per se — los estímulos son el "tratamiento" natural.]

* * *

#### Q4-Q5 — Blinding

No aplica directamente (no hay grupos experimentales). Sin embargo: el análisis de datos se realizará con los labels de arousal/valencia sin conocimiento previo de qué videos corresponden a qué condición emocional.

[Considerar: análisis ciego donde un analista corra el pipeline sin ver los labels, y otro asigne los labels después.]

* * *

#### Q6 — Study design

[Describir el protocolo CAMPEONES completo:]
- N sujetos expuestos a videos emocionales en VR (HMD)
- Cada sesión: [N] videos × [duración] + fixation periods
- Mediciones simultáneas: EEG (32 canales), EDA, ECG, Resp
- Ratings post-video: SAM (valencia + arousal), [Likert?]
- Ratings durante video: joystick continuo

[Adjuntar PDF con protocolo detallado — equivalente a "Tasks and Scales.pdf" del modelo]

* * *

#### Q7 — Randomization

[Describir cómo se aleatorizó el orden de los videos entre sujetos]

* * *

#### Q8 — Data collection status

Registration following collection of data but prior to analysis.

Nota: se han realizado análisis exploratorios de validación técnica (decodificación de luminancia) sobre un subset de sujetos (sub-27, sub-23, sub-24, sub-33) para verificar que el pipeline funciona. Estos análisis se documentan como exploratorios. Los análisis confirmatorios de predicción afectiva (H2, H3) NO se han corrido al momento del registro.

[IMPORTANTE: esta transparencia es fundamental. No hay que esconder que se hicieron análisis previos — hay que declarar cuáles fueron exploratorios y cuáles son confirmatorios.]

* * *

#### Q9 — Existing data

[Declarar exactamente qué análisis se hicieron antes del pre-registro:]
- Preprocesamiento EEG de 6 sujetos (19, 23, 24, 27, 30, 33)
- Decodificación de luminancia (pre/post cambio de foto) en sub-27: bandpower Welch 68.9%, MVNN+SVC 77%
- Feature comparison (bandpower, raw_pca, tde_cov, autocorrelación) en sub-27
- Sanity checks (ERP, histogramas, single-trial) en sub-27
- Extensión de luminancia a sub-23, 24, 33 (en progreso)
- NO se han corrido análisis de predicción de arousal/valencia desde EEG ni periféricos

* * *

#### Q10 — Sample and inclusion criteria

**Inclusión:**
[Criterios de CAMPEONES — edad, salud, idioma, etc.]

**Exclusión (a nivel de sujeto):**
- Sujetos con >20% de runs con calidad insuficiente (definida por QA del pipeline: >X bad channels, ICA failure)
- [Definir umbral de calidad con métrica concreta]

**Exclusión (a nivel de trial/run):**
- Runs donde el preprocesamiento falló (ICA crash, timeout)
- [Definir umbral de artefactos]

* * *

#### Q11 — Sample size

[Se determinará en semanas 1-2, cuando se complete el preprocesamiento blitz]

N estimado: ~20 sujetos completos, de un pool de ~25 con datos recolectados.

* * *

#### Q12 — Sample size rationale

[PENDIENTE: power analysis]

Opciones:
1. **Simulación** (como en rzbv8): simular datos con efecto esperado, correr el pipeline de decoding, estimar power para distintos N
2. **Basada en literatura:** citar estudios similares de EEG emotion decoding y sus N (típicamente N=15-30 en estudios within-subject)
3. **Pragmática:** "todos los sujetos disponibles que pasen QA" + justificación de que within-subject decoding con permutation tests es robusto para N pequeños

[Nota: la opción 3 es la más honesta dado que N está limitado por los datos ya recolectados. Pero se puede complementar con la opción 1 post-hoc para estimar el poder que tenemos.]

* * *

#### Q13 — Stopping rule

No aplica — datos ya recolectados.

* * *

#### Q14 — Variables (independientes / predictores)

**Para H1 (luminancia):**
- IV: condición temporal (pre vs post cambio de luminancia), definida por onset de CHANGE_PHOTO

**Para H2 (periféricos -> arousal):**
- IV (features/predictores): SCL, SCR amplitude, SCR frequency (EDA); HR, RMSSD, SDNN (ECG); respiratory rate, respiratory rate variability (Resp)
- [Definir ventana temporal de extracción: ¿todo el video? ¿ventana alrededor de rating?]

**Para H3 (EEG -> arousal):**
- IV (features/predictores): bandpower (delta, theta, alpha, beta, gamma × 32 canales = 160 features)
- [Alternativas exploradas: raw temporal + PCA, TDE covariance, MVNN+SVC]

* * *

#### Q15 — Variables (dependientes / medidas)

**Para H1:** clase binaria (pre vs post)
**Para H2-H3:** arousal rating (SAM post-video, escala 1-9)
- Operacionalización para decoding: median split (high/low arousal) o [definir]
- Variable secundaria: |valence| (intensidad de valencia)

[PENDIENTE: definir si se usa SAM post-video, joystick continuo promediado, o ambos]

* * *

#### Q16 — Indices / Feature extraction

**EEG:**
- Bandpower Welch: 5 bandas (delta 1-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-45 Hz) × 32 canales
- Ventana: [definir — ¿época completa del video? ¿sub-ventanas?]
- Parámetros Welch: [nperseg, noverlap]
- Alternativa confirmada: MVNN+SVC (resampled a 100 Hz, señal temporal completa)

**EDA (NeuroKit2):**
- Preprocesamiento: [filtrado, decomposición tónica/fásica — método cvxEDA o Highpass]
- Features tónicas: SCL medio por trial
- Features fásicas: SCR amplitude (media), SCR frequency (n/min), SCR latency
- [Referencia: Boucsein et al. 2012, Psychophysiology — Publication Recommendations for EDA]

**ECG (NeuroKit2):**
- Preprocesamiento: [detección de R-peaks — método neurokit/pantompkins/hamilton]
- Features: HR medio, RMSSD, SDNN, pNN50
- [HRV frequency domain: LF, HF, LF/HF — ¿incluir?]
- [Referencia: Task Force of ESC/NASPE 1996 — HRV standards]

**Respiración (NeuroKit2):**
- Preprocesamiento: [detección de ciclos respiratorios]
- Features: respiratory rate medio, respiratory rate variability (SD de intervalos inter-breath)
- [Referencia: Kreibig 2010, Biological Psychology — autonomic nervous system activity in emotion]

[Adjuntar PDF con pipeline completo de preprocesamiento — equivalente a "Preprocessing and Feature Extraction.pdf" del modelo]

* * *

#### Q17 — Analysis plan

**H1 — Luminancia (within-subject decoding):**
- Clasificador: Logistic Regression (C seleccionado por inner CV) o SVM lineal
- Features: bandpower Welch (160 features) o MVNN+SVC (señal temporal 100 Hz)
- Cross-validation: Leave-One-Run-Out (LORO), N folds = N runs por sujeto
- Métrica primaria: balanced accuracy
- Test de significancia: permutation test (1000 permutaciones), p < 0.05
- Nivel de análisis: within-subject primero, luego group-level (t-test one-sample sobre accuracies individuales vs 50%)

**H2 — Periféricos -> arousal:**

Opción A (correlación): Pearson/Spearman entre cada feature periférica y rating SAM de arousal, por sujeto. Group-level: Fisher z-transform + t-test one-sample vs 0.

Opción B (decoding): mismo esquema que H1 pero con features periféricas prediciendo high/low arousal.

Opción C (LMM, más cercano al modelo rzbv8):
```
Arousal ~ B1[SCL] + B2[SCR_amp] + B3[HR] + B4[HRV] + B5[RespRate] + (1 | ParticipantID)
```

[PENDIENTE: elegir entre A, B, C. El modelo rzbv8 usa LMM (opción C). Nuestro proyecto tiene un enfoque más de decoding (opción B). Se puede hacer ambos — LMM como análisis confirmatorio principal, decoding como análisis complementario.]

**H2d — Multimodal periférico vs unimodal:**
- Comparar accuracy/R² del modelo multimodal vs modelos unimodales (EDA-only, ECG-only, Resp-only)
- Test: likelihood ratio test (LMM) o paired t-test sobre accuracies (decoding)

**H3 — EEG -> arousal (within-subject decoding):**
- Mismo framework que H1 pero prediciendo arousal
- Clasificador: Logistic Regression con LORO-CV
- Features: bandpower Welch (confirmatorio), MVNN+SVC (exploratorio)
- Permutation test, p < 0.05

**H3b — Multimodal completo:**
- Concatenar features EEG + periféricos
- Comparar vs EEG-only y periféricos-only
- [Considerar: late fusion vs early fusion]

* * *

#### Q18 — Transformations

- EEG bandpower: log-transform antes de clasificación (reduce skewness)
- EDA: [especificar transformaciones]
- HRV: [log-transform de RMSSD/SDNN si distribución skewed]
- Todas las features: z-score normalization por sujeto antes de clasificación (media=0, std=1)

* * *

#### Q19 — Inference criteria

- Significancia individual (within-subject): permutation test, p < 0.05 (1000 permutaciones)
- Significancia grupal: t-test one-sample sobre accuracies/correlaciones individuales vs chance, p < 0.05
- Corrección por múltiples comparaciones: [Bonferroni o FDR — definir N de comparaciones]
- Reporte: accuracy, AUC, CI 95%, efecto estadístico (Cohen's d para group-level)

* * *

#### Q20 — Data exclusion (trials/runs)

- Runs donde el preprocesamiento EEG falló (ICA crash, insuficientes datos post-rejection): excluidos automáticamente
- Runs con >[X]% de épocas rechazadas por artefactos: excluidos
- [Para EDA: trials donde la señal es plana (no-responder) o con artefactos de movimiento]
- [Para ECG: segmentos con >X% de R-peaks no detectables]

* * *

#### Q21 — Missing data / incomplete participants

- Sujetos con <[X] runs completos (de [8] totales): excluidos del análisis
- [Definir umbral mínimo — ej: 6/8 runs = 75%]

* * *

#### Q22 — Exploratory analyses

1. **|Valence| (intensidad de valencia):** repetir H2-H3 usando |valence| como variable dependiente. Reportar correlación arousal <-> |valence| en la muestra.
2. **Joystick continuo vs SAM post-video:** comparar si las features fisiológicas predicen mejor el rating continuo (promediado por trial) que el rating post-video.
3. **Decoding por región EEG:** comparar accuracy de canales occipito-temporales vs all-channels vs frontales para evaluar contribución topográfica.
4. **Feature set comparison:** comparar bandpower Welch, raw_pca, tde_cov, MVNN+SVC para predicción de arousal (extensión de lo hecho con luminancia).
5. **Anotaciones continuas vs resumen:** análisis de consistencia entre reportes durante y después del video.
6. **Conectividad EEG (con CSD):** si los resultados de canales individuales son prometedores, explorar conectividad entre regiones aplicando Surface Laplacian.

* * *

#### Q23 — Other

[Adjuntos previstos:]
1. **Preprocessing and Feature Extraction.pdf** — pipeline completo de EEG, EDA, ECG, Resp
2. **Experimental Protocol.pdf** — protocolo de la sesión VR
3. **Stimulus List.pdf** — lista de videos con metadata (duración, categoría afectiva)

* * *

#### Referencias clave para el pre-registro

- Boucsein et al. (2012). Publication recommendations for electrodermal measurements. *Psychophysiology*, 49(8), 1017-1034. → Standards EDA
- Pernet et al. (2020). Issues and recommendations from the OHBM COBIDAS MEEG committee for reproducible EEG and MEG research. *Nature Neuroscience*, 23, 1473-1483. → Best practices MEEG
- Botvinik-Nezer et al. (2024). EEGManyPipelines. *Journal of Neuroscience*. → Variabilidad analítica en EEG (justifica pre-registrar el pipeline)
- Keil et al. (2014). Committee report: guidelines for reporting psychophysiology. *Psychophysiology*, 51(1), 1-15. → Reporting standards
- Kreibig (2010). Autonomic nervous system activity in emotion. *Biological Psychology*, 84(3), 394-421. → Correlatos autonómicos de emoción
- Task Force of ESC/NASPE (1996). Heart rate variability standards. *European Heart Journal*, 17, 354-381. → Standards HRV
- [Agregar: papers de EEG emotion decoding relevantes — DEAP, SEED, DREAMER datasets]

* * *

### Roadmap revisado — 8-9 semanas hasta summer school

#### Semanas 1-2 (16 abril → 30 abril)

**Track A — Preprocesamiento blitz (Jero + Juli + vos supervisando):**
- Escribir protocolo de anotación de marcadores para Jero/Juli
- Correr XDF → BIDS → events para todos los sujetos pendientes
- EEG preprocessing batch para los que estén listos
- Target: tener N claro para el pre-registro

**Track B — Diseño del pre-registro (vos):**
- Completar los `[PENDIENTE]` del esqueleto de arriba
- Discutir con Enzo/Diego: hipótesis, approach estadístico, N
- Definir features de EDA/ECG/Resp en detalle (consultar NeuroKit2 docs)

**Track C — Cerrar luminancia multi-sujeto (sesión paralela 04_3):**
- Correr scripts 22, 27b, 34, 41 para sub-23, 24, 33
- Generar tabla comparativa multi-sujeto

#### Semanas 3-4 (1 mayo → 14 mayo)

**Someter pre-registro en OSF.**
- Redactar texto final
- Preparar PDFs adjuntos (pipeline, protocolo, estímulos)
- Review con Enzo/Diego
- Submit

**EDA <-> arousal (validación rápida):**
- Extraer features EDA con NeuroKit2 para todos los sujetos preprocesados
- Correlacionar con SAM arousal
- Si funciona: extender a ECG + Resp

#### Semanas 5-7 (15 mayo → 4 junio)

**Predicción afectiva (análisis confirmatorio):**
- Implementar decoding EEG → arousal (framework ya existe de luminancia)
- Multimodal: EEG + periféricos → arousal
- Multi-sujeto desde el arranque
- Permutation tests grupales

#### Semanas 8-9 (5 junio → 18 junio)

**Cierre pre-summer school:**
- Análisis cross-subject
- Documentación de resultados
- Buffer para imprevistos
- Preparar material para summer school

#### Lo que NO entra (post-summer school)

- Extracción de audio y anotaciones verbales
- Anotaciones continuas vs resumen (excepto exploración inicial)
- Surface Laplacian / CSD / conectividad
- Comparación detallada con pipeline de Yongjie
- Pantalla verde continua
