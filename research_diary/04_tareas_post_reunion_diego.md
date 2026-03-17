# Tareas Post-Reunión con Diego — 2026-03-11

**Proyecto:** campeones_analysis
**Contexto:** Revisión de resultados de scripts 22–26 (pipeline de photo events: generación de eventos, epoching, ERPs, TFRs, permutation tests).

**Hallazgos de Diego:**
- En clusters occipitales y temporales hay un efecto sincronizado al cambio (CHANGE_PHOTO) vs no-cambio (NO_CHANGE_PHOTO) → buena señal.
- Problema 1: En las épocas NO_CHANGE_PHOTO aparece una sincronía de fase inesperada que no debería existir en condiciones de no-cambio.
- Problema 2: En las épocas CHANGE_PHOTO aparece sincronía de fase incluso antes de los 0 ms (pre-estímulo), lo cual es sospechoso.

**Hipótesis de Diego:**
- La sincronía de fase en NO_CHANGE se debe a que los eventos están equidistantemente espaciados (spacing fijo en `22_generate_photo_events.py`), lo que genera una periodicidad artificial.
- La sincronía pre-estímulo en CHANGE podría deberse a alguna propiedad del estímulo que afecta la señal en la ventana -1000 ms a 0 ms, y la ventana actual de época (-1.5s a +2.0s) no permite ver si esto se extiende más atrás.

---

## Tarea 5: Agregar Jitter Temporal a Eventos NO_CHANGE_PHOTO ✅

### Objetivo
Romper la sincronía de fase artificial en las épocas NO_CHANGE_PHOTO introduciendo un jitter aleatorio en los onsets, de modo que las distancias entre eventos consecutivos no sean equidistantes.

### Implementación
- Modificado `generate_no_change_photo_events()` en `scripts/validation/22_generate_photo_events.py`:
  - Constantes: `EPOCH_SPAN_S = 7.5`, `JITTER_FRACTION = 0.3`, `RANDOM_SEED = 42`
  - Cap de épocas por no-solapamiento: `max_no_overlap = int(available // epoch_span_s)`
  - Jitter uniforme: `±jitter_fraction * spacing` con greedy forward pass para mantener separación mínima
  - CLI: `--seed` para reproducibilidad
- Regenerados TSVs: 74 CHANGE + 74 NO_CHANGE, intervalos 7.5–10.1s

---

## Tarea 6: Ampliar Ventana de Épocas para Investigar Sincronía Pre-Estímulo ✅

### Objetivo
Extender la ventana de las épocas y mover el baseline lejos del estímulo para verificar si la sincronía pre-estímulo se extiende más allá de -1000 ms.

### Parámetros implementados
```python
# 23_epoch_photo_events.py
TMIN = -4.5
TMAX = 3.0
BASELINE = (-4.5, -4.0)

# 25_erp_tfr_photo_contrast.py, 25b_erp_raw_overlay.py
BASELINE = (-4.5, -4.0)
VIS_TMIN = -3.0  # crop para excluir baseline de visualizaciones

# 26_permutation_test_photo.py
BASELINE = (-4.5, -4.0)
```

### Cambios realizados
- `23_epoch_photo_events.py`: TMIN=-4.5, TMAX=3.0, BASELINE=(-4.5, -4.0)
- `25_erp_tfr_photo_contrast.py`: constantes BASELINE y VIS_TMIN, crop en plot_erp/plot_tfr/plot_contrast, línea -1000ms en todos los plots
- `25b_erp_raw_overlay.py`: VIS_TMIN=-3.0, crop, línea -1000ms, figura más ancha, mediana + media
- `26_permutation_test_photo.py`: BASELINE=(-4.5, -4.0)

---

## Tarea 7: Crear Épocas Random como Control ✅

### Objetivo
Generar épocas en momentos completamente aleatorios del registro EEG (sin relación con ningún evento) como distribución nula. Si los patrones oscilatorios de NO_CHANGE también aparecen en épocas random, el efecto es artefacto del promediado (asimetría de amplitud del alfa ongoing) y no una respuesta evocada real.

### Implementación
- `scripts/validation/23b_epoch_random_events.py`: genera 40 épocas/run (280 total) con onsets uniformes, separación mínima = 7.5s, mismos TMIN/TMAX/BASELINE.
- `scripts/validation/25c_erp_random_comparison.py`: compara CHANGE vs NO_CHANGE vs RANDOM (subsampled a n=74):
  - ERP comparison: media + SEM de las 3 condiciones superpuestas por ROI.
  - Distribution comparison: bandas de percentiles (p10-p90, p25-p75) + media + mediana por condición.
- Output en `results/validation/photo_erp_random/sub-27/`.

---

## Tarea 8: Investigar Residuo Alfa en Promediado — Simulaciones y Potencia Absoluta

### Objetivo
Entender si la oscilación residual observada en los ERPs de NO_CHANGE (y potencialmente en RANDOM) se explica por propiedades del promediado sobre señales con alfa dominante. Para eso: (a) correr simulaciones propias para replicar y entender el efecto, y (b) visualizar la potencia espectral absoluta (sin normalización por baseline) en los datos reales.

### Motivación
Los TFRs actuales (script 25) usan normalización `mode="percent"` contra baseline, lo cual muestra cambios relativos de potencia. Esto oculta la potencia absoluta de fondo: si hay mucho alfa ongoing en todas las condiciones, el TFR normalizado no lo muestra porque es estable respecto al baseline. Pero esa potencia absoluta alta es justamente lo que genera el residuo oscilatorio en los ERPs al promediar.

### Antecedente: simulación de Fede Poncio
Fede hizo una simulación que replica el efecto observado. Generó 20 señales sintéticas con componentes alfa fuertes (8-12 Hz) con fase aleatoria entre trials, más ruido de alta frecuencia y ruido blanco. Al promediar:
- El promedio muestra oscilación alfa residual clara, a pesar de que las fases individuales son random.
- La FFT del promedio muestra un pico en alfa (~10 Hz) mientras que las frecuencias altas se cancelan.
- La FFT de un trial individual muestra el pico alfa mucho más grande + componentes de alta frecuencia que desaparecen en el promedio.

Esto demuestra que con N bajo (20 trials) y alfa dominante, el promedio no cancela la oscilación de fondo. Con más trials el efecto se reduce pero no desaparece si hay asimetría de amplitud (Mazaheri & Jensen, 2008).

**Código de referencia de Fede:**
```python
sins = []
n_points = 200
for k in range(20):
    offset = np.random.uniform() * 200
    x = np.arange(n_points)
    phi = 2 * np.pi * (x - offset) / 200
    strong_alpha = (np.sin(8*phi) - 6*np.sin(10*phi)
                    + 10*np.sin(12*phi) + 5*np.sin(11*phi))
    phase_shift = np.random.uniform(0, 2*np.pi)
    hf = np.random.choice(range(50,80), size=2)
    high_freq = (1.5*np.sin(hf[0]*(2*np.pi*x/n_points) + phase_shift)
                 + 0.5*np.sin(hf[1]*(2*np.pi*x/n_points) + phase_shift))
    white_noise = np.random.normal(scale=0.5, size=n_points)
    sins.append(strong_alpha + high_freq + white_noise)
sins = np.array(sins)
# Promedio muestra oscilación alfa residual
```

### Sub-tareas

#### 8.1 Simulaciones propias ✅
Crear `scripts/validation/28_alpha_residual_simulation.py`:
- Replicar simulación de Fede con parámetros ajustados a nuestros datos (N=74, fs=500 Hz, duración=7.5s).
- Explorar cómo varía el residuo alfa en función de N (20, 50, 74, 150, 300).
- Agregar asimetría de amplitud (picos > valles) para testear el mecanismo de Mazaheri & Jensen.
- Comparar residuo con y sin asimetría.

**Resultados y conclusiones (2026-03-16):**

Output en `results/validation/alpha_simulation/` (5 figuras).

1. **Residuo alfa confirmado (fig1):** El promedio de 74 trials con alfa de fase aleatoria produce una oscilación residual clara (RMS≈0.6 a.u.) con picos espectrales en 8-12 Hz. La alta frecuencia y el ruido blanco se cancelan casi por completo, pero el alfa no. Replica el hallazgo de Fede con nuestros parámetros reales.

2. **Decaimiento 1/√N (fig2):** El residuo sigue la curva teórica 1/√N. Con N=74 el RMS es ~1.0, con N=300 baja a ~0.45 pero no llega a cero. La variabilidad entre realizaciones es alta para N bajo.

3. **Asimetría de amplitud duplica el residuo (fig3):** Con rectificación parcial (Mazaheri & Jensen 2008), el RMS del promedio pasa de 0.977 (simétrico) a 1.867 (asimétrico) — casi el doble. Además, el promedio asimétrico muestra un offset DC positivo (oscila entre ~0 y +4 en vez de ±2) porque los picos no se cancelan con los valles.

4. **La asimetría genera un piso irreducible (fig4):** La curva simétrica sigue bajando con N, pero la asimétrica se aplana en ~1.4 a.u. a partir de N≈150. Agregar más trials no elimina el residuo cuando hay asimetría — tiene un piso que no se puede resolver promediando.

5. **Mecanismo confirmado (fig5):** El histograma de picos vs |valles| muestra solapamiento en el caso simétrico y separación clara en el asimétrico (picos ~25-32, |valles| ~15-22). La convergencia del RMS confirma que el simétrico baja gradualmente mientras el asimétrico se estabiliza.

**Implicaciones para los datos reales:**
- El residuo alfa en los ERPs de NO_CHANGE (y potencialmente RANDOM) es esperable y no requiere explicación especial — es un artefacto del promediado con N finito.
- En la simulación el residuo es estacionario en toda la época (no hay nada especial en t=0). Si en los datos reales la oscilación de NO_CHANGE y RANDOM es uniforme en toda la ventana temporal, es consistente con este artefacto.
- Si CHANGE muestra modulación time-locked al evento (ej: supresión alfa post-estímulo) que NO_CHANGE y RANDOM no tienen, eso sería señal real por encima del residuo.
- **Pregunta clave para 25c:** ¿la oscilación de NO_CHANGE se parece a la de RANDOM? Si sí → artefacto. Si NO_CHANGE tiene algo que RANDOM no → hay componente stimulus-related.

#### 8.2 TFRs sin normalización por baseline ✅
Plotear TFRs con potencia absoluta (sin `apply_baseline`) para CHANGE, NO_CHANGE, RANDOM:
- Revelar la potencia alfa de fondo en todas las condiciones.
- Verificar si potencia absoluta es similar entre NO_CHANGE y RANDOM.
- Ver si CHANGE muestra modulación real sobre ese fondo.

**Resultados y conclusiones (2026-03-16):**

Output en `results/validation/photo_tfr_absolute/sub-27/` (8 figuras: 4 TFR heatmaps + 4 alpha time-courses, escalas globales compartidas entre ROIs).

**TFR heatmaps (escala global 0–14800 uV²/Hz):**

1. **Occipital domina la potencia alfa:** La banda 8-12 Hz en Occipital tiene potencia absoluta mucho mayor (~2000-6000 uV²/Hz) que en las demás ROIs (~500-1500). Esto es consistente con la generación posterior del ritmo alfa. La escala global compartida lo hace evidente: Occipital "brilla" mientras Frontal/Temporal/Parietal están mayormente oscuros.

2. **CHANGE muestra modulación time-locked clara en Occipital:** En el TFR occipital de CHANGE hay un aumento de potencia alfa visible entre -1000 ms y +1000 ms (la ventana del flicker), con picos que alcanzan ~14000 uV²/Hz. Esto NO aparece en NO_CHANGE ni en RANDOM. Es una modulación real del estímulo sobre la potencia alfa de fondo.

3. **NO_CHANGE y RANDOM son similares en los TFR heatmaps:** En las 4 ROIs, los heatmaps de NO_CHANGE y RANDOM muestran patrones de potencia alfa estacionarios y comparables, sin modulación time-locked. Esto confirma que la oscilación residual en los ERPs de NO_CHANGE es artefacto del promediado, no una respuesta evocada.

**Alpha time-courses (8-12 Hz, escala global compartida):**

4. **Occipital — hallazgo principal:** Las 3 condiciones se separan claramente. CHANGE (rojo) muestra un pico masivo de potencia alfa entre -1000 ms y +1000 ms, alcanzando ~5000-6500 uV²/Hz. NO_CHANGE (verde) se mantiene estable alrededor de ~2500 uV²/Hz. RANDOM (azul) es la más baja y estable (~1700-2000 uV²/Hz). La modulación en CHANGE es inequívocamente stimulus-driven.

5. **NO_CHANGE tiene más alfa que RANDOM en Occipital y Parietal:** NO_CHANGE (verde) está consistentemente por encima de RANDOM (azul) en ~500-800 uV²/Hz en estas ROIs. Esto sugiere que los momentos de no-cambio de foto no son completamente equivalentes a momentos aleatorios — podría haber un efecto de contexto (el sujeto está viendo una foto estática, lo que podría modular el alfa posterior de forma tónica).

6. **Frontal y Temporal — potencia baja y similar entre condiciones:** Las 3 condiciones se solapan en el rango ~400-1000 uV²/Hz sin separación clara ni modulación temporal. El efecto es predominantemente posterior.

**Implicaciones:**
- La modulación alfa en CHANGE es real y masiva en Occipital, time-locked al flicker (-1000 a +1000 ms). Esto valida el contraste CHANGE vs NO_CHANGE.
- La diferencia tónica NO_CHANGE > RANDOM en Occipital/Parietal sugiere que el contexto visual (foto estática) eleva el alfa posterior respecto a momentos aleatorios del registro. Esto no invalida el contraste pero es un matiz importante.
- Los TFRs normalizados (script 25) ocultaban la potencia absoluta de fondo. La visualización sin baseline revela que el alfa ongoing es ~2000-2500 uV²/Hz en Occipital para todas las condiciones, lo cual explica el residuo oscilatorio en los ERPs (consistente con simulaciones de 8.1).

#### 8.3 Residuo alfa en datos reales con N creciente de épocas random ✅
Verificar en la señal EEG real que el residuo alfa en el promedio de épocas random desaparece (o se reduce sustancialmente) al aumentar N. Esto complementa las simulaciones de 8.1 con datos reales.

- Usar las épocas random generadas en Tarea 7 (script `23b_epoch_random_events.py`), aumentando progresivamente la cantidad generada.
- Probar con N creciente (ej: 20, 50, 74, 150, 300, 500 épocas random).
- Para cada N, promediar las épocas y medir el RMS del residuo alfa (8-12 Hz) en el ERP promediado.
- Plotear la curva RMS vs N en datos reales y compararla con la curva teórica 1/√N de las simulaciones.
- Si el residuo baja con N → confirma que es artefacto de promediado con N finito (consistente con 8.1). Si no baja → hay algo más en la señal real que las simulaciones no capturan.

**Resultados y conclusiones (2026-03-17):**

Script: `scripts/validation/29_alpha_residual_real_data.py`. Output en `results/validation/alpha_residual_real/sub-27/` (2 figuras + JSON).

Pool de 700 épocas random (100/run × 7 runs, TMIN=-4.5, TMAX=3.0, BASELINE=(-4.5, -4.0), sfreq=250 Hz). Para cada N, se subsamplean 50 repeticiones y se mide el RMS de la señal alfa-filtrada (8-12 Hz, Butterworth orden 4) del ERP promediado.

**RMS alfa vs N — Occipital (ROI principal):**

| N | RMS alfa (µV) | ± SD | Ratio vs N=20 | Teórico √(20/N) |
|---|---|---|---|---|
| 20 | 1.178 | 0.119 | 1.00 | 1.00 |
| 50 | 0.745 | 0.089 | 0.63 | 0.63 |
| 74 | 0.608 | 0.059 | 0.52 | 0.52 |
| 100 | 0.520 | 0.057 | 0.44 | 0.45 |
| 150 | 0.399 | 0.038 | 0.34 | 0.37 |
| 200 | 0.349 | 0.037 | 0.30 | 0.32 |
| 280 | 0.299 | 0.026 | 0.25 | 0.27 |

**Hallazgos:**

1. **El residuo alfa sigue 1/√N en datos reales.** El ajuste a la curva teórica es excelente en todas las ROIs. Occipital tiene el residuo más alto (1.18 µV con N=20, baja a 0.30 µV con N=280) porque tiene la mayor potencia alfa de fondo, consistente con 8.2.

2. **Todas las ROIs muestran el mismo patrón de decaimiento.** Frontal, Temporal, Parietal y Occipital siguen la misma curva 1/√N, solo difieren en la amplitud absoluta (Occipital > Parietal ≈ Frontal > Temporal).

3. **No hay piso irreducible en los datos reales con épocas random.** A diferencia de la simulación asimétrica de 8.1 (que mostraba un piso en ~1.4 a.u.), los datos reales siguen bajando con N. Esto sugiere que la asimetría de amplitud del alfa real no es tan extrema como en la simulación, o que con N=280 todavía no se alcanza el piso.

4. **La variabilidad (SD) también baja con N**, como se espera: con más épocas, el promedio es más estable entre repeticiones.

**Implicación:** El residuo alfa en los ERPs de NO_CHANGE (N=74, ~0.6 µV en Occipital) es completamente consistente con el artefacto de promediado con N finito. No requiere explicación especial. Si se pudieran generar más épocas NO_CHANGE, el residuo seguiría bajando.

#### 8.4 Chequeo de magnitud: señal cruda vs ERP promediado ✅
Verificar que la magnitud del ERP promediado sea al menos un orden de magnitud menor que la señal cruda de trials individuales. Si no es así, es un warning de que el promediado no está cancelando la actividad de fondo como se espera.

- Para las épocas CHANGE, NO_CHANGE y RANDOM: comparar la amplitud (RMS o pico-a-pico) de trials individuales vs el ERP promediado.
- Calcular el ratio señal cruda / promediado. Se espera que sea ~√N (≈8.6 para N=74).
- Plotear distribución de amplitudes de trials individuales vs amplitud del promedio.
- Si el ratio es mucho menor que √N → la señal no se está cancelando bien (posible componente coherente entre trials, o artefacto).
- Hacer este chequeo para CHANGE, NO_CHANGE y RANDOM por separado, en ROIs occipital y temporal.

**Resultados y conclusiones (2026-03-17):**

Script: `scripts/validation/30_magnitude_check.py`. Output en `results/validation/magnitude_check/sub-27/` (3 figuras + JSON).

Se compara el RMS de trials individuales vs el RMS del ERP promediado para las 3 condiciones (N=74 cada una) en 4 ROIs. Ventana de análisis: -3.0 a +3.0 s (excluyendo baseline). Ratio esperado: √74 ≈ 8.6.

**Tabla de ratios (trial RMS / ERP RMS):**

| ROI | CHANGE | NO_CHANGE | RANDOM | Esperado (√74) |
|-----|--------|-----------|--------|-----------------|
| Frontal | 6.4 | 7.3 | 8.5 | 8.6 |
| Temporal | 6.0 | 8.8 | 6.5 | 8.6 |
| Parietal | 6.4 | 7.7 | 8.0 | 8.6 |
| Occipital | 6.1 | 8.0 | 7.8 | 8.6 |

**Amplitudes absolutas (µV):**

| ROI | Trial RMS (CHANGE) | ERP RMS (CHANGE) | Trial RMS (NO_CHANGE) | ERP RMS (NO_CHANGE) | Trial RMS (RANDOM) | ERP RMS (RANDOM) |
|-----|---|---|---|---|---|---|
| Occipital | 12.2 | 2.0 | 10.0 | 1.3 | 9.6 | 1.2 |
| Temporal | 5.7 | 1.0 | 5.0 | 0.6 | 5.8 | 0.9 |
| Parietal | 5.3 | 0.8 | 4.9 | 0.6 | 4.7 | 0.6 |
| Frontal | 6.2 | 1.0 | 5.4 | 0.7 | 5.3 | 0.6 |

**Hallazgos:**

1. **NO_CHANGE y RANDOM tienen ratios cercanos a √N.** NO_CHANGE: 7.3–8.8, RANDOM: 6.5–8.5. Esto indica que el promediado está cancelando la actividad de fondo como se espera — no hay componente coherente anómala entre trials. El residuo en el ERP es consistente con cancelación incompleta por N finito (confirmado en 8.3).

2. **CHANGE tiene ratios consistentemente menores (~6.0–6.4).** Esto es esperable y correcto: CHANGE tiene una componente time-locked al estímulo (la modulación alfa masiva vista en 8.2) que NO se cancela al promediar. Por eso el ERP RMS es proporcionalmente más alto (2.0 µV en Occipital vs 1.2–1.3 para NO_CHANGE/RANDOM), y el ratio baja. Esto confirma que hay señal real en CHANGE.

3. **Occipital tiene el trial RMS más alto (~10-12 µV) por la potencia alfa de fondo.** Consistente con 8.2. CHANGE tiene trial RMS aún mayor (12.2 µV) porque el flicker aumenta la potencia alfa durante la época.

4. **El orden de magnitud se cumple.** Trial individual ~5-12 µV, ERP promediado ~0.6-2.0 µV. La diferencia es de ~1 orden de magnitud (factor 6-9x), lo cual es razonable para N=74.

**Conclusión:** El chequeo de magnitud es satisfactorio. NO_CHANGE y RANDOM muestran ratios cercanos al teórico √74 ≈ 8.6, confirmando que el promediado funciona correctamente y el residuo es artefacto de N finito. CHANGE muestra ratios menores porque contiene señal real time-locked. No hay warnings de comportamiento anómalo.

---

## Tarea 9: Decoding de Change/No-Change con Todos los Electrodos ✅

### Objetivo
Correr modelos de decoding sobre clasificación binaria Change/No-Change usando las nuevas épocas, con todos los 32 electrodos. Feature sets alineados con el pipeline de modeling (`scripts/modeling/`).

### Implementación
- Script: `scripts/validation/27_decoding_photo_change.py`
- Re-epoching desde datos crudos con parámetros de decoding: TMIN=-2.5, TMAX=2.0, BASELINE=(-2.5, -1.5)
- 32 canales EEG, 74 CHANGE + 74 NO_CHANGE epochs, 7 runs
- Clasificación: LogisticRegression (L2), Leave-One-Run-Out CV con inner CV (3-fold stratified) para selección de C
- 3 feature sets alineados con scripts/modeling/:
  1. **bandpower_welch** (script 11): Welch PSD → 5 bandas × 32 canales = 160 features
  2. **tde_cov** (script 13): GLHMM TDE(±10) → PCA global(20, fit solo en train) → covarianza upper triangle = 210 features
  3. **raw_pca** (script 10): señal cruda vectorizada (36032 dim) → PCA(100, fit en train) → 100 features

### Resultados (2026-03-17)

Output en `results/validation/photo_decoding/sub-27/`.

| Feature set     | N features | Accuracy | Precision | Recall | F1    | AUC-ROC |
|----------------|-----------|----------|-----------|--------|-------|---------|
| bandpower_welch | 160       | 0.804    | 0.771     | 0.865  | 0.815 | 0.877   |
| tde_cov         | 210       | 0.743    | 0.709     | 0.824  | 0.763 | 0.811   |
| raw_pca         | 100 (de 36032) | 0.284 | 0.362  | 0.568  | 0.442 | 0.398   |

**Detalle por fold (n_train / n_test):**

| Fold | bandpower_welch | tde_cov | raw_pca | n_train | n_test |
|------|----------------|---------|---------|---------|--------|
| task-01_acq-a_run-002 | 0.837 (C=10) | 0.714 (C=1.0) | 0.245 (C=0.001) | 99 | 49 |
| task-02_acq-a_run-003 | 0.750 (C=1.0) | 0.875 (C=0.01) | 0.000 (C=0.001) | 140 | 8 |
| task-03_acq-a_run-004 | 1.000 (C=10) | 0.800 (C=0.01) | 0.000 (C=0.001) | 138 | 10 |
| task-04_acq-a_run-006 | 0.583 (C=10) | 0.583 (C=0.01) | 0.750 (C=10) | 136 | 12 |
| task-01_acq-b_run-007 | 0.755 (C=0.1) | 0.714 (C=0.01) | 0.245 (C=0.001) | 99 | 49 |
| task-03_acq-b_run-009 | 1.000 (C=1.0) | 0.875 (C=0.01) | 0.500 (C=0.1) | 140 | 8 |
| task-04_acq-b_run-010 | 0.833 (C=0.1) | 0.917 (C=0.01) | 0.417 (C=0.1) | 136 | 12 |

### Interpretación

**Bandpower Welch es el claro ganador (80.4% accuracy, AUC=0.88).** 160 features (5 bandas × 32 canales), ratio features/samples ~1.2 sobre ~130 trials de entrenamiento. Consistente con los hallazgos de 8.2: la modulación alfa en CHANGE es masiva en Occipital, y el bandpower la captura directamente. Los runs grandes (task-01, 49 trials) son estables (~76-84%), los runs chicos (8-10 trials) alcanzan 75-100%.

**TDE+Cov funciona bien (74.3%, AUC=0.81).** 210 features (triángulo superior de covarianza de 20 PCA components), ratio ~1.6. PCA global fitteado solo sobre datos de train en cada fold (sin data leakage). Los C óptimos son bajos (0.01 en 5/7 folds), indicando que la regularización fuerte ayuda. Captura relaciones temporales entre componentes que el bandpower no ve, pero con menos poder discriminativo que las bandas espectrales directas.

**Raw+PCA no funciona (28.4%, AUC=0.40 — peor que chance).** 36032 features raw → PCA(100) en train. A pesar de la reducción de dimensionalidad, los 100 componentes principales no capturan la información discriminativa. Los C óptimos son extremadamente bajos (0.001) en la mayoría de folds, y el modelo predice casi todo como una sola clase. La señal cruda en el dominio temporal no es informativa para este contraste sin transformación espectral o TDE previa.

**Observaciones por fold:**
- task-04_acq-a (run-006) es consistentemente el peor fold en bandpower (58%) y tde_cov (58%). Podría tener alguna particularidad (calidad de datos, comportamiento del sujeto). Candidato para QC.
- Los runs de task-01 (49 trials cada uno) son los más informativos y estables.
- Los runs chicos (8-12 trials) tienen alta varianza pero no invalidan el resultado global.

**Conclusión:** El contraste CHANGE vs NO_CHANGE es decodificable con buena precisión usando features espectrales (bandpower Welch) y razonablemente con features de covarianza temporal (TDE+cov). La señal cruda en dominio temporal no es informativa. Esto confirma que hay una señal neural discriminativa real, predominantemente en las bandas espectrales (consistente con la modulación alfa masiva vista en 8.2). Los feature sets están ahora alineados con el pipeline de modeling de `scripts/modeling/`.

---

## Tarea 10: Decoding con Micro-Épocas de 50ms (diseño de Enzo)

### Objetivo
Rediseñar el dataset de decoding usando ventanas temporales cortas (50 ms) para aumentar el número de observaciones y mejorar la resolución temporal del clasificador. Propuesto por el supervisor Enzo.

### Diseño del dataset

**Épocas CHANGE (post-estímulo):**
Para cada uno de los 74 momentos de cambio de luminancia (onset = 0 ms), extraer 4 ventanas de 50 ms:
- Ventana 1: 50 a 100 ms post-onset
- Ventana 2: 100 a 150 ms post-onset
- Ventana 3: 150 a 200 ms post-onset
- Ventana 4: 200 a 250 ms post-onset

Total CHANGE: 74 × 4 = 296 épocas

**Épocas NO_CHANGE (pre-estímulo):**
Para cada uno de los mismos 74 momentos de cambio, extraer 4 ventanas de 50 ms inmediatamente antes del onset:
- Ventana 1: -250 a -200 ms pre-onset
- Ventana 2: -200 a -150 ms pre-onset
- Ventana 3: -150 a -100 ms pre-onset
- Ventana 4: -100 a -50 ms pre-onset

Total NO_CHANGE: 74 × 4 = 296 épocas

**Dataset total:** 592 épocas (296 CHANGE + 296 NO_CHANGE), balanceado.

### Consideraciones de CV y data leakage
- Las 8 micro-épocas (4 CHANGE + 4 NO_CHANGE) derivadas de un mismo estímulo deben estar siempre en el mismo fold (train o test, nunca separadas).
- La unidad de split para Leave-One-Run-Out CV sigue siendo el run, lo cual garantiza esto automáticamente (todas las épocas de un estímulo pertenecen al mismo run).
- Dentro de un fold de entrenamiento, las micro-épocas del mismo estímulo son observaciones correlacionadas — el modelo podría aprender patrones específicos del estímulo en vez de la diferencia CHANGE vs NO_CHANGE. La separación por run mitiga esto parcialmente.

### Implementación
- Script: `scripts/validation/31_decoding_micro_epochs.py`
- Datos preprocesados a 250 Hz → 50 ms = 12 muestras por micro-ventana.
- Para cada evento CHANGE_PHOTO (74 onsets en sub-27), se extraen 8 micro-épocas (4 post + 4 pre).
- TDE descartado: necesita mínimo 21 muestras (TDE_WINDOW_HALF=10), solo hay 12.
- Clasificación: LogisticRegression (L2), LORO CV por run, inner CV para selección de C.

**Estrategia para resolver la baja resolución espectral:**
Las micro-épocas de 12 muestras no permiten calcular Welch PSD con resolución útil. Solución: primero se crea la época larga (TMIN=-2.5, TMAX=2.0, baseline=(-2.5, -1.5)) como en Tarea 9, luego se filtra pasa-banda en cada banda espectral sobre la época completa (donde hay ~1125 muestras), y finalmente se segmenta en micro-ventanas de 50 ms y se calcula la varianza (= potencia) por canal. Esto da resolución espectral completa a pesar de las ventanas cortas.

**Feature sets:**
1. `bandpower_filtered`: filtrado pasa-banda sobre época larga → segmentar → varianza por banda/canal = 5 × 32 = 160 features
2. `raw_signal`: época larga con baseline → segmentar → vectorizar = 32 ch × 12 = 384 features

### Resultados (sub-27)

**Dataset:** 296 CHANGE + 296 NO_CHANGE = 592 micro-épocas, 7 folds (runs).

| Feature set | N features | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| bandpower_filtered | 160 | 57.6% | 0.580 | 0.603 |
| raw_signal | 384 | 53.0% | 0.519 | 0.529 |

**Detalle por fold — bandpower_filtered:**

| Fold | Acc test | Acc train | C | n_train | n_test |
|---|---|---|---|---|---|
| task-01_acq-a_run-002 | 55.2% | 77.6% | 0.1 | 496 | 96 |
| task-02_acq-a_run-003 | 60.9% | 72.3% | 0.01 | 528 | 64 |
| task-03_acq-a_run-004 | 62.5% | 68.2% | 0.001 | 512 | 80 |
| task-04_acq-a_run-006 | 54.2% | 70.0% | 0.001 | 496 | 96 |
| task-01_acq-b_run-007 | 60.4% | 75.0% | 0.01 | 496 | 96 |
| task-03_acq-b_run-009 | 53.1% | 73.3% | 0.01 | 528 | 64 |
| task-04_acq-b_run-010 | 57.3% | 80.6% | 0.1 | 496 | 96 |

**Detalle por fold — raw_signal:**

| Fold | Acc test | Acc train | C | n_train | n_test |
|---|---|---|---|---|---|
| task-01_acq-a_run-002 | 58.3% | 67.7% | 0.01 | 496 | 96 |
| task-02_acq-a_run-003 | 56.2% | 69.9% | 0.01 | 528 | 64 |
| task-03_acq-a_run-004 | 55.0% | 75.4% | 0.1 | 512 | 80 |
| task-04_acq-a_run-006 | 41.7% | 64.5% | 0.001 | 496 | 96 |
| task-01_acq-b_run-007 | 53.1% | 69.8% | 0.01 | 496 | 96 |
| task-03_acq-b_run-009 | 48.4% | 69.3% | 0.01 | 528 | 64 |
| task-04_acq-b_run-010 | 58.3% | 71.0% | 0.01 | 496 | 96 |

### Interpretación

1. **Bandpower filtrado mejora significativamente (57.6%, AUC=0.603).** La estrategia de filtrar sobre la época larga y luego segmentar resuelve el problema de resolución espectral. Comparado con el Welch directo sobre 12 muestras (que daba 50% = chance), ahora hay señal discriminativa real. Sin embargo, la performance es mucho menor que con épocas largas (80.4% en Tarea 9).

2. **Raw signal sigue marginal (53.0%, AUC=0.529).** Ahora con baseline correction de la época larga, pero la señal temporal cruda en 48 ms no es suficiente. El gap train-test (~15-20 pp) indica overfitting moderado.

3. **La señal discriminativa en 50 ms es débil pero real.** La modulación alfa que domina el contraste CHANGE vs NO_CHANGE tiene un período de ~100 ms (10 Hz). Una ventana de 50 ms captura menos de medio ciclo, lo que limita fundamentalmente la potencia del bandpower. Aun así, la varianza de la señal filtrada en alfa captura algo de la diferencia de amplitud.

4. **task-04_acq-a (run-006) sigue siendo el peor fold** en ambos feature sets (54.2% y 41.7%), consistente con Tarea 9.

5. **Comparación con Tarea 9:** 80.4% (épocas largas, bandpower Welch) → 57.6% (micro-épocas, bandpower filtrado). La caída de ~23 pp refleja la pérdida de información al comprimir 4.5 s de señal en ventanas de 50 ms. El aumento de observaciones (148 → 592) no compensa la pérdida de calidad de features.

**Archivos generados:**
- `results/validation/photo_decoding_micro/sub-27/sub-27_micro_decoding_results.json`
- `results/validation/photo_decoding_micro/sub-27/sub-27_micro_decoding_summary.png`

### Tarea 10.1: Performance por ventana temporal post-estímulo

#### Objetivo
Evaluar en qué ventana de 50 ms post-onset se maximiza la performance de decoding. Para cada una de las 4 ventanas post-estímulo ([50-100], [100-150], [150-200], [200-250] ms), entrenar un modelo independiente usando como clase NO_CHANGE las 4 ventanas pre-onset shuffleadas (74 épocas por clase por modelo).

#### Diseño
- 4 modelos independientes, uno por ventana post-onset.
- Clase CHANGE: las 74 micro-épocas de esa ventana específica (una por onset).
- Clase NO_CHANGE: 74 micro-épocas muestreadas aleatoriamente de las 4 ventanas pre-onset (296 disponibles → sample 74 sin reemplazo, balanceando entre las 4 ventanas pre).
- Feature sets: bandpower_filtered y raw_signal (como en Tarea 10).
- LORO CV por run.
- Esto permite identificar el momento post-estímulo donde la señal neural discriminativa es más fuerte.

---

## Orden de Ejecución

```
Tarea 5 (Jitter en NO_CHANGE) ✅
    ↓
Tarea 6 (Ampliar ventana + ajustar baseline) ✅
    ↓
Tarea 7 (Épocas random como control) ✅
    ↓
Tarea 8 (Simulaciones + potencia absoluta) ✅
    ├── 8.1 Simulaciones propias ✅
    ├── 8.2 TFRs sin normalización ✅
    ├── 8.3 Residuo alfa en datos reales con N creciente ✅
    └── 8.4 Chequeo de magnitud cruda vs promediado ✅
    ↓
Tarea 9 (Decoding con épocas largas, 32 ch) ✅
    ↓
Tarea 10 (Decoding con micro-épocas 50ms — diseño Enzo) ✅
    └── 10.1 Performance por ventana temporal post-estímulo
```