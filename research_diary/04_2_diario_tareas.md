# Diario de Tareas: sanity checks, interpretabilidad y  autocorrelación

**Proyecto:** campeones_analysis

**Fecha:** 2026-04-08

**Supervisores:** Enzo Tagliazucchi, Diego Vidaurre

**Contexto:** Luego de la reunión con Diego (2026-04-01), tenemos dos bloques de análisis validados: (1) decoding de 4 clases temporales en foto-eventos (raw_pca: 37.4%, z=10.6σ, p<0.001) y (2) decoding de 3 clases en luminancia continua (ningún feature set significativo tras corregir el balance de clases; raw_pca con z=−3.34 sugiere confound de ventanas solapadas). Diego planteó que la tarea de flash debe dar 80–90% si el EEG es visible a ojo desnudo — si no llega, podria haber un bug en el pipeline. Las tareas acordadas priorizan: (1) sanity checks fisiológicos, (2) interpretabilidad del modelo ganador (bandpower Welch), (3) nueva variante con autocorrelación por canal, y (4) contacto con Yongjie para comparar metodologías.

* * *
## Tareas pendientes (continuadas del diario anterior)

**Tarea 1 — Interpretabilidad del modelo ganador (Band Power Welch):** completada. Coeficientes extraídos, ranking canal × banda generado, plots 39a/b/c/d/e guardados. Beta ERD como mecanismo principal; delta/theta positivo en O1/O2 refleja VEP.

**Tarea 1bis — Interpretabilidad extendida a 4 clases:** completada. Alpha modulation temporal como feature dominante entre las 4 fases.

**Tarea 2 — Variante TDE con autocorrelación por canal:** completada. Implementación con 25 lags (800 features), variantes PCA y log-espaciada. Resultado: autocorr ≈ tde_cov (61.5% vs 62.2%); no supera bandpower (68.9%).

**Tarea 3 — Sanity checks de señal:** completada (excepto 3.6). Single-trial, histogramas alfa/beta, ERP+FDR, scatter. Veredicto: señal real, SNR baja, no hay bug. Tarea 3.6 (pantalla verde continua) pendiente.

**Tarea 4 — Debugging de pipeline:** no necesaria dado el resultado de Tarea 3.

**Tarea 5 — Contacto con Yongjie (Slack):** primer mensaje enviado 2026-04-08, respondió 2026-04-09. Setup: 128 canales, 100Hz, ventana 1.2s, SVM lineal, features = tiempo × canales aplanado (15240). Clases: top-20 brillantes vs top-20 oscuros. Caso extremo (1 video por clase): 88.5% LOSO. Pendiente: preguntar sobre pseudo-trials, pedir código del mejor modelo, preguntar sobre interpretabilidad de pesos.

* * *
## Sesión de hoy

### Tarea 3.1 — Revisión de plots ERP/raw overlay existentes
**Pregunta:** ¿Es el efecto de cambio de luminancia visible "a ojo desnudo" en el EEG occipital?

**Plots revisados:**
- `results/validation/photo_erp_raw_overlay/sub-27/sub-27_raw_overlay_Occipital.png` — épocas individuales superpuestas (script 25b)
- `results/validation/photo_erp_tfr/sub-27/sub-27_erp_Occipital.png` — ERP promedio CHANGE vs NO_CHANGE (script 25)
- `results/validation/photo_erp_tfr/sub-27/sub-27_erp_contrast_Occipital.png` — contraste CHANGE − NO_CHANGE (script 25)

**Hallazgo (plots agregados):** Se observa un efecto claro en O1/O2 en la ventana **200–300ms post-onset** para CHANGE_PHOTO, tanto en el overlay (semi-transparente) como en el ERP promedio y el contraste. Los plots de script 25 y 25b confirman separabilidad visual a nivel de promedio.

**Pendiente:** El raw overlay (script 25b) superpone *todas* las épocas en los mismos ejes — sigue siendo un resumen visual. Para completar 3.1 según el criterio de Diego ("se ve cerrando los ojos"), hay que verificar que el efecto sea visible en **trials individuales** (2-3 épocas en subplots separados, sin promediado ni overlay). Esto requiere el script 38.

### Tarea 3.1b — Single-trial + raw continuo interactivo (scripts 38a, 38e, 38f)

![Single-trial raw EEG — 6 CHANGE vs 6 NO_CHANGE, O1/O2, escala ±50 µV](../results/validation/sanity_checks/sub-27/sub-27_38a_single_trial_raw.png)

![Raw EEG continuo run-002 — CHANGE (rojo) vs NO_CHANGE (verde)](../results/validation/sanity_checks/sub-27/sub-27_38e_raw_continuous_run002.png)

**Scripts corridos:** `38a_single_trial_raw.py` (6 trials individuales por clase, O1/O2, escala fija ±50 µV) + `38e_raw_continuous.py` (run-002 completo, estático) + `38f_raw_interactive.py` (raw.plot() interactivo, run-002).

**Hallazgos:**
1. **No se ven diferencias claras trial-a-trial** entre CHANGE y NO_CHANGE en ningún canal. Esto es esperado: la amplitud del VEP individual (~5–10 µV) queda sepultada en el ruido espontáneo (~30–40 µV). La señal existe pero requiere promediado para emerger.
2. **Tentativa en O1/O2:** al scrollear el raw continuo, hay indicios de cambios en las frecuencias dominantes post-marcador en O1/O2, pero no en T7/T8. Hipótesis a confirmar con 38b.
3. **Confound temporal identificado (script 27, NO aplica a 27b):** en run-002 y run-007 (task-01), todos los eventos NO_CHANGE_PHOTO ocurren durante el período de fixation (inicio del run, ~300s), mientras que los CHANGE_PHOTO ocurren en la fase de tarea activa (más tarde). El script 27 (CHANGE vs NO_CHANGE) podría estar clasificando "fixation vs tarea activa" más que "cambio de luminancia vs estable", lo que inflaría artificialmente el 78.4%. El script 27b (pre vs post dentro del mismo trial CHANGE) **no tiene este confound** y por eso es más exigente (68.9%).
4. **Sobre los markers Stimulus/S X:** son triggers originales del software de presentación embebidos en el .vmrk de BrainVision. Son independientes de los eventos CHANGE/NO_CHANGE que derivamos del TSV.

**Veredicto 3.1:** Señal no visible a nivel single-trial (criterio Diego no cumplido estrictamente). Esto es consistente con el 68.9% del clasificador — la señal es real pero débil. No hay bug, hay SNR baja.

* * *

### Tarea 3.2 — Histogramas alfa/beta (script 38b)
![Alpha/Beta histogramas: pre vs post (27b windows), occipital y temporal](../results/validation/sanity_checks/sub-27/sub-27_38b_alpha_beta_histograms.png)

| ROI | Banda | p-valor (Mann-Whitney) | Interpretación |
|---|---|---|---|
| Occipital (O1/O2) | Alpha (8–13 Hz) | **p = 0.029** | Diferencia significativa post vs pre |
| Occipital (O1/O2) | Beta (13–30 Hz) | **p < 0.001** | Diferencia muy significativa — beta ERD |
| Temporal (T7/T8) | Alpha (8–13 Hz) | p = 0.998 | Sin diferencia |
| Temporal (T7/T8) | Beta (13–30 Hz) | p = 0.790 | Sin diferencia |

**Hallazgos:**
- **Beta occipital (p<0.001):** la potencia beta en O1/O2 es significativamente *menor* post-cambio que pre-cambio. Es una beta ERD (event-related desynchronization) — un marcador clásico de activación cortical visual. **Este es el feature más discriminativo del clasificador bandpower.**
- **Alpha occipital (p=0.029):** diferencia significativa pero más modesta.
- **Temporales (T7/T8):** sin diferencias en potencia. La respuesta auditiva (tono sincrónico) **no se manifiesta en potencia alfa/beta** en estos canales.
- **Confirmación de la intuición observacional:** la hipótesis del usuario sobre diferencias en frecuencias dominantes en O1/O2 fue correcta.

* * *

### Tarea 3.3 — ERP + t-test por timepoint (script 38c)
![ERP CHANGE vs NO_CHANGE con t-test FDR — occipital y temporal](../results/validation/sanity_checks/sub-27/sub-27_38c_erp_ttest.png)

| ROI | Timepoints significativos (FDR) | Pico principal |
|---|---|---|
| Occipital (O1/O2) | 30/251 | Deflexión negativa ~0–200ms, positiva ~200–400ms |
| Temporal (T7/T8) | 39/251 | Pico positivo ~100ms (más temprano que occipital) |

**Hallazgos:**
- **Occipital:** ERP clásico con deflexión negativa post-onset (N1 visual, ~100–200ms) seguida de recuperación. 30 timepoints significativos con corrección FDR.
- **Temporal:** respuesta temporal con pico a ~100ms — **más temprana que la occipital**. 39 timepoints significativos, más que en occipital. Esto indica que **el tono auditivo sincrónico genera una respuesta rápida en T7/T8** que también discrimina las condiciones.
- **Importante:** aunque el ERP temporal es significativo, la potencia (38b) no lo es. La respuesta auditiva es un ERP transitorio, no una modulación sostenida de potencia. El clasificador bandpower (ventana 500ms) captura mejor la potencia sostenida que los ERPs transitorios, lo que puede explicar por qué la potencia discrimina solo en occipital.

* * *

### Tarea 3.4 — Scatter alfa vs beta (script 38d)
![Scatter alfa vs beta: pre (azul) vs post (rojo), occipital y temporal](../results/validation/sanity_checks/sub-27/sub-27_38d_alpha_beta_scatter.png)

**Hallazgos:**
- **Occipital:** las nubes pre (azul) y post (rojo) se solapan sustancialmente pero los diamantes (medias) están desplazados — post tiene menor alfa y menor beta que pre. Separación visible pero ruidosa.
- **Temporal:** nubes casi completamente solapadas, diamantes prácticamente coincidentes. Consistente con p>0.7 en 38b.
- El solapamiento sustancial en el espacio alfa×beta es consistente con el 68.9% de accuracy — hay señal real pero la variabilidad trial-a-trial es alta.

* * *

### Síntesis tarea 3 (parcial)

**Veredicto general:** el pipeline de 27b es válido. No hay bug. La señal existe en sub-27 y es:
- **Real:** beta ERD occipital p<0.001, ERP occipital y temporal significativos con FDR
- **Débil a nivel single-trial:** SNR baja, nubes solapadas en scatter
- **Beta ERD distribuido:** aunque T7/T8 no muestran diferencia de potencia alfa/beta en 38b, la tarea 1 (39c) revela que el clasificador usa canales frontales/centrales/parietales además de occipitales — el efecto beta ERD es cortical amplio
- **Auditiva en ERP transitorio:** el tono sincrónico genera respuesta temporal a ~100ms, pero no en potencia sostenida

**¿Por qué 68.9% y no 80–90%?** El clasificador bandpower trabaja con ventanas de 500ms donde la variabilidad trial-a-trial domina sobre la señal media. El efecto existe pero es pequeño (Cohen's d bajo), lo que limita la accuracy con N=74 trials. No es un bug — es una limitación de SNR y tamaño muestral.

* * *

### Tarea 1 — Interpretabilidad del modelo ganador (script 39)
**Script:** `scripts/validation/39_interpretability_bandpower.py`
**Outputs:** `results/validation/interpretability/sub-27/`

**Metodología:**
- Modelo full-data (todos los trials, sin CV): `StandardScaler + LogisticRegression(C=1.0, lbfgs)`
- El accuracy no se reporta aquí (viene del LORO CV de los scripts originales); el modelo full-data es solo para extraer coeficientes interpretables
- `coef_` tiene escala z-score por el StandardScaler, lo que hace los β comparables entre canales/bandas

#### Figuras generadas

![Heatmaps 27b vs 27 + correlación de coeficientes](../results/validation/interpretability/sub-27/sub-27_39a_coef_heatmap_27b_vs_27.png)

![4 heatmaps por clase — script 34](../results/validation/interpretability/sub-27/sub-27_39b_coef_4class.png)

![Top-15 features por |β| para 27b y 27](../results/validation/interpretability/sub-27/sub-27_39c_top_features.png)

#### Hallazgos — 27b (pre vs post CHANGE_PHOTO)

El patrón de coeficientes revela que la clasificación descansa principalmente en la **banda beta con signo negativo** en múltiples canales:

- **β negativo en beta** → potencia beta alta predice PRE (ventana antes del cambio de luminancia)
- **Mecanismo:** beta ERD (event-related desynchronization) — la corteza desincroniza su ritmo beta después del cambio visual. El clasificador aprendió a detectar esa disminución: "beta alto = aún no cambió, beta bajo = ya cambió"
- **Distribución espacial:** el efecto es **distribuido** (Cz, Fz, CP5, T8, FT9, O2), no exclusivamente occipital. Esto es biológicamente plausible: el VEP genera un reset de la actividad beta en una red cortical extendida, no solo en V1.
- Los top-15 features del barplot (39c) confirman: la mayoría son beta negativos en canales frontales/centrales/parietales/occipitales.

| Feature top | Signo | Interpretación |
|---|---|---|
| Cz-beta (y similares frontales/centrales) | Negativo | Alto beta → PRE window |
| O2-delta, O2-gamma | Positivo | Alta broadband occipital → POST window |
| T8-beta | Negativo | Efecto extendido a temporal derecho |

#### Hallazgos — comparación 27b vs 27

| Métrica | Valor | Interpretación |
|---|---|---|
| Correlación r β(27b) vs β(27) | **r = 0.132, p = 0.095** | Correlación no significativa — los dos clasificadores capturan cosas distintas |

**Conclusión:** el clasificador de script 27 (CHANGE vs NO_CHANGE) NO está capturando el mismo patrón neural que 27b. Esto es evidencia directa del confound temporal identificado previamente (NO_CHANGE = fixation al inicio del run). El 27b es la tarea con interpretación científica más limpia.

#### Hallazgos — 4 clases (script 34)

- **Alpha domina** como banda discriminativa entre las 4 fases temporales
- Baseline y ChangeDown muestran patrones opuestos en alpha → el clasificador separa fases pre/post vía modulación de potencia alfa
- La progresión temporal (−500ms → 0 → +500ms → +1000ms) genera un patrón cambiante de alpha, posiblemente reflejando la **supresión y recuperación de alpha occipital** (alpha ERD al onset, recovery posterior)
- Los patrones de las 4 clases son complementarios (suman aprox. a cero en multinomial softmax), lo que es matemáticamente esperado en LogisticRegression multinomial

#### Análisis profundo — Tarea 1bis: perfiles temporales 4 clases (scripts 39d, 39e)
**Scripts:** `39d_band_temporal_profiles.py`, `39e_erp_vs_delta_theta.py`

![Perfil temporal por banda — β medio sobre canales por clase](../results/validation/interpretability/sub-27/sub-27_39d_band_temporal_profile.png)

![Alpha y beta por clase en canales ROI (O1, O2, Cz, Fz, T7, T8)](../results/validation/interpretability/sub-27/sub-27_39d_roi_alpha_beta.png)

![Heatmap banda × clase por canal ROI](../results/validation/interpretability/sub-27/sub-27_39d_roi_heatmap.png)

![Hipótesis contaminación ERP: scatter ERP amplitude vs β_delta/θ](../results/validation/interpretability/sub-27/sub-27_39e_erp_vs_delta_theta.png)

**Hallazgo 1 — Por qué alpha domina en 34 pero beta domina en 27b:**

| Tarea | Feature dominante | Razón |
|---|---|---|
| 27b (pre vs post, ±500ms) | **Beta ERD** | Compara exactamente la ventana inmediatamente antes/después del onset — beta ERD es máximo en ese período |
| 34 (4 clases, 0→1500ms) | **Alpha modulation** | Las 4 clases abarcan 1.5 s del trial. Alpha cicla de ERD (supresión al onset) a recovery (1+ s post-onset), capturando estructura temporal rica. Beta ERD existe pero se diluye al comparar 4 fases en 2 s. |

**Hallazgo 2 — Mecanismo de los δ/θ positivos en 27b (hipótesis contaminación ERP):**

El test en 39e reveló correlación δ β vs ERP amplitude r=0.128 (no significativo globalmente), pero con patrón claro al separar canales:

- **O1, O2:** β_delta = +1.09 / +0.82, ERP_increase = 1.81 / 1.43 µV → **hipótesis confirmada**: el VEP visual (N1/P1 en corteza occipital) aparece como potencia delta aumentada en la ventana POST
- **C3, C4:** β_delta moderado (+0.5) pero ERP_increase < 0.15 µV → mecanismo distinto, posiblemente slow cortical potentials o correlación espuria de regularización

**Distinción fisiológica ERP vs cambios espectrales (discusión teórica):**

| | ERP (evocado) | ERD/ERS (inducido) |
|---|---|---|
| Fase entre trials | Phase-locked (consistente) | Non-phase-locked (variable) |
| Sobrevive promediado | Sí | No (se cancela) |
| Mecanismo | PSPs sincronizados en dendritas apicales de pirámidales | Desacoplamiento de circuitos oscilatorios (bucles tálamo-corticales para alpha, corticales para beta) |
| Qué refleja | QUÉ se procesó y CUÁNDO | ESTADO cortical: activo vs idle |
| En bandpower Welch | Aparece como potencia de baja frecuencia (contaminación) | Capturado directamente como cambio de potencia |

**Implicación metodológica:** Welch bandpower sobre ventanas event-related captura **actividad mezclada** (evocada + inducida). Para aislar puramente el ERD habría que restar el ERP promedio de cada trial antes de computar la potencia (→ "induced power"). En nuestro caso, el clasificador usa ambas señales mezcladas, lo cual es legítimo para el decoding pero complica la interpretación fisiológica.

### Checklist Tarea 1 (completada)

- [x] 1.1 Cargar modelo Welch y extraer coeficientes β
- [x] 1.2 Mapear cada coeficiente al feature original (canal × banda)
- [x] 1.3 Graficar top-10 features → beta ERD distribuido como feature principal
- [x] 1.4 Guardar resultado: plots 39a/b/c + JSON `39_coefs.json`
- [x] 1.5 Documentar: beta ERD es el mecanismo, delta/theta positivo refleja VEP en O1/O2
- [x] 1.6 Extender a 4 clases → alpha modulation temporal como feature dominante (scripts 39d)

* * *

### Tarea 2 — Variante TDE con autocorrelación por canal (completada)

**Pregunta central:** ¿La estructura temporal *dentro de cada canal* es suficiente para decodificar el cambio de luminancia, o necesitamos coherencia entre canales?

#### Pipeline implementado

- `extract_autocorrelation_features(data, n_lags=25)` en `27b_decoding_pre_vs_post.py`
- Lags consecutivos 1..25 (4–100ms a 250Hz, 4ms/sample a 250Hz) → cubre 1 ciclo completo de alpha
- 32 canales × 25 lags = **800 features**
- Pipeline: StandardScaler + LogisticRegressionCV (L2, mismo que bandpower)
- Variantes: autocorr_pca20/50/100, autocorr_log (lags [1,2,3,4,7,12,20,25] → 256 features)

#### Resultados LORO-CV (sub-27, tarea 27b)

| Feature set | N features | Accuracy | AUC |
|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | **0.725** |
| raw_pca | ~160 | 65.5% | — |
| tde_cov | 153 | 62.2% | 0.624 |
| autocorr (1..25) | 800 | 61.5% | 0.604 |
| autocorr_pca100 | 100 | 61.5% | 0.602 |
| autocorr_log | 256 | 58.8% | 0.594 |
| autocorr_pca20 | 20 | 55.4% | 0.584 |
| chance | — | 50.0% | 0.500 |

best_C = 0.001 en casi todos los folds (regularización máxima — señal de alta dimensionalidad relativa a N=74 trials).

#### Hallazgos de ablación

1. **autocorr ≈ tde_cov (61.5% vs 62.2%)** → la coherencia inter-canal no agrega valor. La señal útil es local a cada canal.
2. **autocorr < bandpower (61.5% vs 68.9%)** → ambos detectan beta/alpha ERD. Bandpower gana porque Welch promedia más ruido trial-a-trial (estimador más robusto con N=74).
3. **PCA no mejora autocorr** — techo de 61.5% se mantiene con 20, 100 o 800 features. El cuello de botella es la relación señal-ruido intrínseca.
4. **Log-espaciado (58.8%) no mejora consecutivo** — la redundancia entre lags vecinos actuaba como promediado implícito útil.

#### Interpretabilidad (script 40, plots 40a–40f)

**Lags log-espaciados (análisis más limpio — sin redundancia):**

| Lag | Tiempo | Mean |beta| | Interpretación |
|---|---|---|---|
| lag12 | 48ms | **0.00609** | Periodo completo beta (20Hz) + medio periodo alpha (10Hz) |
| lag20 | 80ms | 0.00548 | Entre beta y alpha |
| lag25 | 100ms | 0.00441 | Periodo completo alpha (10Hz) |
| lags 1–7 | 4–28ms | ~0.003 | Gamma/broadband — menos discriminativos |

**Insight clave:** lag12 (48ms) domina porque coincide con el periodo completo de beta (20Hz → 50ms) Y el medio periodo de alpha (10Hz → 100ms, mitad = 50ms). Es la escala donde coexisten ambos mecanismos ERD. Reconcilia por qué bandpower detecta beta y autocorr consecutiva detectaba alpha: ambos contribuyen en esa zona temporal.

**Distribución espacial:** Pz, O1, TP9, O2 — occipito-parietal, consistente con Task 1.

#### Scripts generados
- `scripts/validation/27b_decoding_pre_vs_post.py` — extendido con autocorr, autocorr_log, autocorr_pca{N}, run_loro_autocorr_pca
- `scripts/validation/40_autocorr_interpretability.py` — plots 40a–40f + JSON `40_autocorr_coefs.json`

#### Checklist Tarea 2- [x] 2.1–2.3 Implementar autocorrelación canal×lag, feature vector 32×25
- [x] 2.4 Comparar con tde_cov, bandpower, raw_pca (LORO-CV)
- [x] 2.5 Registrar resultados — tabla completa
- [x] 2.6 Conclusión: autocorr sin coherencia inter-canal ≈ tde_cov; no supera bandpower
- [x] Extra: PCA variants + log-espaciado + interpretabilidad completa (scripts 40a–40f)

* * *

### Tarea 5.1 — Análisis de metodología Yongjie Duan

Yongjie Duan (PhD student co-supervisado por Diego) respondió el 2026-04-09 con slides detallando su pipeline. Se realizó un análisis comparativo de su approach vs el nuestro.

**Setup de Yongjie:**
- 128 canales, downsampled a 100Hz, ventana 0.06–1.26s (1.2s post-onset)
- Clases: top-20 videos más brillantes vs top-20 más oscuros (L = 0.299R + 0.587G + 0.114B)
- Clasificador: SVM lineal
- Features: EEG aplanado tiempo × canales → 127 canales × 120 timepoints = **15240 features**
- Validación: LOSO (Leave-One-Subject-Out), 6 sujetos
- Performance caso extremo (1 video por clase): 88.5% LOSO
- Performance caso general (20 vs 20 videos): menor, no especificado

**Comparación con nuestro pipeline:**

| | Yongjie | Sub-27 (nuestro) |
|---|---|---|
| Canales | 127 | 32 |
| SR | 100Hz | 250Hz |
| Ventana | 1.2s | 0.5s |
| Features | 15240 (raw flatten) | 160 (bandpower Welch) |
| Reducción | ninguna | — |
| Validación | LOSO (6 sujetos) | LORO (1 sujeto) |

**Puntos clave del análisis:**

1. **El 88.5% es un upper bound artificial.** El "caso extremo" selecciona el par de máxima diferencia de luminancia en todo el dataset (1 video por clase), maximizando la discriminabilidad. No hay variabilidad within-class, la diferencia de luminancia es máxima, y con muchas repeticiones el SNR es alto. No es representativo del caso general.

2. **Pseudo-trials.** Es probable que Yongjie use pseudo-trials (promediar k repeticiones del mismo video antes de clasificar), lo que sube el SNR de cada trial. Pendiente confirmar en la conversación en curso.

3. **Features = raw flatten.** 15240 features con N trials << 15240 implica régimen p >> n. El SVM maneja esto via regularización implícita (max-margin = L2) operando en el espacio dual (N×N, no p×p). Sin embargo, la selección del mejor modelo entre múltiples variantes de features ("LOSO BEST") puede inflar la performance reportada por multiple comparisons.

4. **La ventana de 1.2s es clave.** Nuestras ventanas de 0.5s cortan la respuesta antes de que termine. La ventana de Yongjie captura respuestas cognitivas tardías (P300, ondas lentas sostenidas) que nosotros no vemos.

5. **Reshape de pesos = mapa discriminativo.** Como el clasificador es lineal sobre features aplanados, los pesos pueden reshapearse a (127 canales × 120 timepoints) dando un mapa espaciotemporal de discriminabilidad — equivalente a un ERP discriminativo aprendido automáticamente. Pendiente pedirle ese análisis.

**Hipótesis sobre por qué autocorr < Welch en nuestros datos:**
Con ventanas de 500ms a 250Hz, la autocorrelación tiene solo ~5 ciclos de alpha (10Hz) disponibles. Welch con la ventana completa tiene estimación espectral más robusta. La autocorrelación pierde información espectral de baja frecuencia al estar limitada por la duración de la época.

**Pendiente (conversación en curso con Yongjie):**
- Confirmar uso de pseudo-trials y metodología exacta de generación
- Pedir código del mejor modelo para correrlo sobre sub-27
- Pedir análisis de interpretabilidad (reshape de pesos del SVM)

* * *

### Tarea 5.3 — Pipeline Yongjie-style en sub-27: MVNN + LinearSVC

**Fecha:** 2026-04-09

**Objetivo:** Replicar el approach de Yongjie (raw time-series + MVNN whitening + LinearSVC, C=1.0) sobre datos de sub-27, tarea PRE vs POST de CHANGE_PHOTO (diseño 27b). Comparar con bandpower Welch. Luego hacer ablación de canales occipitales + temporales.

**Pipeline implementado** (`scripts/validation/41_decoding_mvnn_svc.py`):
- Features: EEG raw aplanado (canales × timepoints), ventana 1.2s a 100Hz
- Normalización: MVNN (Guggenmos et al. 2018) — whitening per-timepoint, fit solo en train (sin data leakage)
- Clasificador: LinearSVC (C=1.0, dual='auto')
- CV: LORO-CV, accuracy global (predicciones concatenadas de todos los folds)
- Diseño PRE/POST: PRE = [-1.25, -0.05]s (baseline de fixation), POST = [+0.05, +1.25]s (respuesta al flash 3Hz)
- N trials: 148 (74 PRE + 74 POST, balanceado por construcción en cada run)

**Resultados:**

| Pipeline | Canales | Features | Global acc | Fold-mean | Std |
|---|---|---|---|---|---|
| MVNN + SVC (all 32ch) | 32 | 3840 | 73.0% | 73.3% | 9.8% |
| **MVNN + SVC (occ_temp 8ch)** | **8** | **960** | **77.0%** | **77.4%** | **7.3%** |
| Bandpower + SVC (all 32ch) | 32 | 160 | 61.5% | 61.4% | 7.3% |
| Bandpower + SVC (occ_temp 8ch) | 8 | 40 | 52.7% | 52.6% | 13.3% |

**Por fold — comparación occ_temp vs all 32ch (MVNN + SVC):**

| Fold | Run | N test | occ_temp | all 32ch | Δ |
|---|---|---|---|---|---|
| 1 | task-01_acq-a_run-002 | 24 | 79.2% | 58.3% | +20.9% |
| 2 | task-02_acq-a_run-003 | 16 | 87.5% | 81.2% | +6.3% |
| 3 | task-03_acq-a_run-004 | 20 | 75.0% | 80.0% | -5.0% |
| 4 | task-04_acq-a_run-006 | 24 | 70.8% | 75.0% | -4.2% |
| 5 | task-01_acq-b_run-007 | 24 | 87.5% | 87.5% | 0.0% |
| 6 | task-03_acq-b_run-009 | 16 | 75.0% | 68.8% | +6.2% |
| 7 | task-04_acq-b_run-010 | 24 | 66.7% | 62.5% | +4.2% |

**Interpretabilidad (MVNN + SVC occ_temp):**

![Interpretabilidad MVNN+SVC — 8 canales occ_temp, 7 LORO folds](../results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_interpretability.png)

- **Spatial importance (mayor a menor):** O1 > O2 > FT10 > FT9 > TP9 > T8 > T7 > TP10
- **Temporal profile:** pico dominante en ~100–150ms (N1/P1 VEP de onset visual). No se observan peaks periódicos limpios a 333ms/667ms/1000ms (3Hz SSVEP).
- **Mecanismo principal:** VEP de onset (corteza visual primaria, O1/O2), no SSVEP 3Hz sostenida.
- **Nota:** coeficientes en espacio whitened (aproximación Haufe: A ≈ w dado Σ_whitened ≈ I).

**Hallazgos clave:**

**1. MVNN+SVC (occ_temp) > MVNN+SVC (all):** 77.0% vs 73.0%. Los canales frontales y centrales agregan ruido, no señal. Reducir a 8 canales mejora performance y estabilidad fold-a-fold (std 7.3% vs 9.8%).

**2. MVNN+SVC supera bandpower en ambas condiciones de canales.** La señal discriminativa es time-locked (VEP/SSVEP), lo que raw time-series + MVNN captura mejor que potencia espectral promediada en ventanas.

**3. Bandpower colapsa con occ_temp** (52.7% ≈ chance): el feature bandpower necesita canales frontales/parietales para detectar el beta ERD distribuido. Restringido a 8 canales occipitales-temporales, no tiene suficiente señal espectral.

**4. Comparación con benchmark previo:** el mejor pipeline anterior era bandpower_welch all-channels (68.9%). MVNN+SVC occ_temp lo supera con 77.0% — con 4× menos canales y 4× menos features que MVNN all.

**Decisión metodológica:** MVNN+SVC occ_temp es el mejor pipeline single-subject hasta ahora para la tarea PRE vs POST CHANGE_PHOTO en sub-27.

**Outputs guardados:**
- `results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_vs_bandpower.json`
- `results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_interpretability.png`
- `results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_coef.npy`

* * *

## Conclusiones de la sesión

**Lo que hicimos:**
- Completamos Tarea 2: implementación, ablación, interpretabilidad de autocorrelación
- Variantes: consecutiva (800), PCA (20/50/100), log-espaciada (256)
- Análisis de interpretabilidad completo con scripts 40a–40f
- Implementamos pipeline Yongjie-style (MVNN + LinearSVC) en diseño PRE vs POST (Tarea 5.3)
- Ablación de canales: occ_temp (8ch) vs all (32ch), con interpretabilidad

**Resultados principales:**
- Techo de autocorrelación: ~61.5% (todas las variantes convergen)
- Bandpower sigue siendo el mejor feature set espectral (68.9% / AUC 0.725)
- **Mejor pipeline overall: MVNN+SVC occ_temp = 77.0%** (8 canales, 960 features)
- La señal discriminativa es **local y time-locked**: VEP de onset en O1/O2 (~100ms)
- Bandpower captura beta ERD distribuido; MVNN+SVC captura VEP time-locked — mecanismos complementarios

**Decisiones metodológicas:**
- Lags 1..25 consecutivos = representación canónica de autocorrelación para este proyecto
- No explorar más variantes de autocorrelación — el techo está claro
- MVNN+SVC occ_temp es el nuevo benchmark single-subject

**Próximos pasos:**
- [ ] 3.6 Tarea sutil — pantalla verde continua
- [ ] 5.2 Responder a Yongjie: compartir resultados de 77.0%, preguntar sobre pseudo-trials y pipeline general (20 vs 20 videos)
- [ ] Extender MVNN+SVC a más sujetos
- [ ] Permutation test para MVNN+SVC occ_temp

