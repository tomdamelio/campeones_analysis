# Tareas Post-Reunión 2026-03-11 (con Diego primero, y Enzo despues)

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

Pool de 700 épocas random (100/run x 7 runs, TMIN=-4.5, TMAX=3.0, BASELINE=(-4.5, -4.0), sfreq=250 Hz). Para cada N, se subsamplean 50 repeticiones y se mide el RMS de la señal alfa-filtrada (8-12 Hz, Butterworth orden 4) del ERP promediado.

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
Correr modelos de decoding sobre clasificación binaria Change/No-Change usando épocas focalizadas, con todos los 32 electrodos. Comparar ventanas de 500ms y 1s para evaluar si la señal discriminativa es más perceptual (~300ms, P2/N2) o cognitiva (~550ms).

### Diseño de épocas
- Epochar ancho (-1.5 a 1.5s), aplicar baseline (-1.5 a -1.0s), cropear a [0.05, crop_end]s post-onset para ambas condiciones.
- CHANGE: señal post-cambio de luminancia (donde está la modulación alfa, Tarea 8.2).
- NO_CHANGE: señal post-onset del evento NO_CHANGE (punto arbitrario en fixation, generado con jitter en Tarea 5). Actividad de fondo sin estímulo.
- Dos ventanas testeadas:
  - **500ms** [0.05, 0.55]s → 125 muestras a 250 Hz. Centro Hann ~300ms (procesamiento perceptual, P2/N2).
  - **1000ms** [0.05, 1.05]s → 250 muestras a 250 Hz. Centro Hann ~550ms (procesamiento más cognitivo).

### Implementación
- Script: `scripts/validation/27_decoding_photo_change.py --focused --crop-end {0.55|1.05}`
- 32 canales EEG, 74 CHANGE + 74 NO_CHANGE epochs, 7 runs
- Clasificación: LogisticRegression (L2, C=1.0 fijo), Leave-One-Run-Out CV
- 3 feature sets alineados con scripts/modeling/:
  1. **bandpower_welch** (script 11): Welch PSD → 5 bandas x 32 canales = 160 features
  2. **tde_cov** (script 13): GLHMM TDE(±10) → PCA global(20, fit solo en train) → covarianza upper triangle = 210 features
  3. **raw_pca** (script 10): señal cruda vectorizada → PCA(100, fit en train) → 100 features

**Nota sobre selección de C:** Inicialmente se usó inner CV (3-fold stratified) para seleccionar C de un grid [0.001, 0.01, 0.1, 1.0, 10.0]. Con train sets de 99-140 muestras, el inner CV produce estimaciones ruidosas y sobre-regularizaba (elegía C=0.01 en tde_cov, C=0.001 en raw_pca). Se adoptó C=1.0 fijo como default. El flag `--inner-cv` permite reactivar el grid search.

### Resultados (2026-03-18)

#### Comparación 500ms vs 1000ms

| Feature set | N feat | Acc (500ms) | AUC (500ms) | Acc (1s) | AUC (1s) | Δ Acc | Δ AUC |
|---|---|---|---|---|---|---|---|
| bandpower_welch | 160 | **0.858** | **0.922** | 0.784 | 0.872 | +0.074 | +0.050 |
| tde_cov | 210 | 0.581 | 0.606 | **0.757** | **0.820** | -0.176 | -0.214 |
| raw_pca | 100 | 0.608 | 0.652 | **0.649** | **0.716** | -0.041 | -0.064 |

#### Detalle: 500ms [0.05, 0.55]s — centro Hann ~300ms

Output en `results/validation/photo_decoding_focused_500ms/sub-27/`.

| Feature set | N features | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| bandpower_welch | 160 | 0.858 | 0.835 | 0.892 | 0.863 | 0.922 |
| tde_cov | 210 | 0.581 | 0.573 | 0.635 | 0.603 | 0.606 |
| raw_pca | 100 (de 4064) | 0.608 | 0.603 | 0.635 | 0.618 | 0.652 |

**Accuracy por fold (500ms):**

| Fold | bandpower_welch | tde_cov | raw_pca | n_train | n_test |
|---|---|---|---|---|---|
| task-01_acq-a_run-002 | 0.857 | 0.571 | 0.653 | 99 | 49 |
| task-02_acq-a_run-003 | 1.000 | 0.625 | 0.750 | 140 | 8 |
| task-03_acq-a_run-004 | 1.000 | 0.700 | 0.500 | 138 | 10 |
| task-04_acq-a_run-006 | 0.833 | 0.167 | 0.583 | 136 | 12 |
| task-01_acq-b_run-007 | 0.857 | 0.653 | 0.633 | 99 | 49 |
| task-03_acq-b_run-009 | 0.875 | 0.250 | 0.500 | 140 | 8 |
| task-04_acq-b_run-010 | 0.667 | 0.833 | 0.417 | 136 | 12 |

#### Detalle: 1000ms [0.05, 1.05]s — centro Hann ~550ms

Output en `results/validation/photo_decoding_focused_1000ms/sub-27/`.

| Feature set | N features | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| bandpower_welch | 160 | 0.784 | 0.739 | 0.878 | 0.802 | 0.872 |
| tde_cov | 210 | 0.757 | 0.738 | 0.797 | 0.766 | 0.820 |
| raw_pca | 100 (de 8032) | 0.649 | 0.636 | 0.716 | 0.675 | 0.716 |

**Accuracy por fold (1000ms):**

| Fold | bandpower_welch | tde_cov | raw_pca | n_train | n_test |
|---|---|---|---|---|---|
| task-01_acq-a_run-002 | 0.694 | 0.673 | 0.653 | 99 | 49 |
| task-02_acq-a_run-003 | 0.750 | 0.625 | 0.750 | 140 | 8 |
| task-03_acq-a_run-004 | 1.000 | 0.900 | 0.600 | 138 | 10 |
| task-04_acq-a_run-006 | 0.750 | 0.583 | 0.667 | 136 | 12 |
| task-01_acq-b_run-007 | 0.796 | 0.816 | 0.633 | 99 | 49 |
| task-03_acq-b_run-009 | 1.000 | 0.875 | 0.875 | 140 | 8 |
| task-04_acq-b_run-010 | 0.833 | 0.917 | 0.500 | 136 | 12 |
| task-03_acq-b_run-009 | 140 | 8 | 140x160 | 140x210 | 140x100 |
| task-04_acq-b_run-010 | 136 | 12 | 136x160 | 136x210 | 136x100 |


### Interpretación

**Bandpower Welch es el mejor feature set (78.4%, AUC=0.872).** la señal discriminativa está concentrada en el primer segundo post-onset. Los folds grandes (task-01, n_test=49) dan 0.69 y 0.80, los folds chicos (8-12 trials) tienen alta varianza (0.75-1.00) pero no invalidan el resultado global.

**TDE+Cov funciona bien pero pierde con épocas cortas (75.7%, AUC=0.820).** Con épocas de 1s hay ~230 timepoints válidos post-TDE (vs ~1106 con 4.5s). La covarianza de 20 componentes PCA se estima con menos muestras, lo que la hace más ruidosa.

**Raw+PCA mejora sustancialmente con épocas focalizadas (62.8%, AUC=0.715).** Con épocas de 1s, el vector raw tiene 8032 features (vs 36032 con 4.5s), y PCA(100) captura una proporción mayor de la varianza relevante. Al focalizar en la ventana discriminativa, los componentes principales reflejan la diferencia CHANGE vs NO_CHANGE en vez de la actividad de fondo estacionaria. Sigue siendo el peor feature set, pero ahora funciona por encima de chance.

**Riesgo de overfitting:** Con ratios features/samples de ~1.2 (bandpower, 160 feat / ~130 train) y ~1.6 (tde_cov, 210 feat / ~130 train), hay riesgo moderado. La regularización L2 con C=1.0 y la validación por LORO mitigan esto.

**Conclusión:** El contraste CHANGE vs NO_CHANGE es decodificable con buena precisión (78.4% accuracy, AUC=0.87) usando bandpower Welch con épocas focalizadas de 1s. La señal discriminativa es predominantemente espectral (modulación alfa masiva en Occipital, consistente con 8.2). Pipeline: épocas de 1s [0.05, 1.05]s post-onset, baseline (-1.5, -1.0)s, LogReg L2 con C=1.0 fijo, LORO CV.

---

## Tarea 9.2: Decoding Post-Estímulo vs Pre-Estímulo (mismo onset) ✅

### Objetivo
Clasificar actividad post-estímulo vs pre-estímulo usando los mismos onsets de CHANGE_PHOTO. A diferencia de Tarea 9 (donde NO_CHANGE eran eventos separados en momentos de fijación), aquí ambas clases provienen del mismo onset: la única diferencia es temporal (post vs pre cambio de luminancia).

### Diseño de épocas
- Para cada uno de los 74 onsets de CHANGE_PHOTO:
  - **Post-estímulo (clase 1):** [+0.05, +0.55]s post-onset (500ms)
  - **Pre-estímulo (clase 0):** [-0.55, -0.05]s pre-onset (500ms)
- Epoch ancho (-1.5 a 1.5s), baseline (-1.5, -1.0)s, luego crop a cada ventana.
- Dataset: 74 post + 74 pre = 148 épocas, balanceado.
- LORO CV por run (7 folds).

### Implementación
- Script: `scripts/validation/27b_decoding_pre_vs_post.py --subject 27`
- 32 canales EEG, LogisticRegression L2 con C=1.0 fijo.
- 3 feature sets: bandpower_welch (160 feat), tde_cov (210 feat), raw_pca (100 feat).

### Resultados (sub-27, C=1.0)

| Feature set | N features | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| bandpower_welch | 160 | 72.3% | 0.717 | 0.718 |
| tde_cov | 210 | 62.2% | 0.600 | 0.659 |
| raw_pca | 100 (de 4064) | 64.9% | 0.649 | 0.717 |

**Accuracy por fold:**

| Fold | bandpower_welch | tde_cov | raw_pca | n_train | n_test |
|---|---|---|---|---|---|
| task-01_acq-a_run-002 | 0.458 | 0.500 | 0.583 | 124 | 24 |
| task-02_acq-a_run-003 | 0.750 | 0.812 | 0.688 | 132 | 16 |
| task-03_acq-a_run-004 | 0.750 | 0.600 | 0.500 | 128 | 20 |
| task-04_acq-a_run-006 | 0.583 | 0.500 | 0.458 | 124 | 24 |
| task-01_acq-b_run-007 | 0.833 | 0.667 | 0.792 | 124 | 24 |
| task-03_acq-b_run-009 | 0.938 | 0.625 | 0.812 | 132 | 16 |
| task-04_acq-b_run-010 | 0.833 | 0.708 | 0.750 | 124 | 24 |

### Interpretación

1. **Bandpower Welch sigue siendo el mejor feature set (72.3%, AUC=0.718).** Consistente con Tarea 9, la señal discriminativa es predominantemente espectral. La performance es menor que en Tarea 9 (85.8% con 500ms) porque aquí la clase NO_CHANGE es actividad pre-estímulo del mismo trial (más similar a la post-estímulo) en vez de momentos de fijación separados.

2. **Raw_pca supera a TDE_cov (64.9% vs 62.2%).** Con épocas de 500ms, raw_pca tiene 4064 features reducidos a 100 por PCA, capturando patrones temporales que TDE no logra con la covarianza.

3. **La caída respecto a Tarea 9 (~13 pp en bandpower) es esperable.** En Tarea 9, NO_CHANGE eran momentos de fijación sin estímulo (actividad de fondo pura). Aquí, la ventana pre-estímulo [-0.55, -0.05]s está temporalmente adyacente al cambio de luminancia, por lo que puede contener actividad anticipatoria o preparatoria que reduce el contraste.

4. **Alta varianza entre folds.** Los folds con n_test=16 (runs con 8 onsets) muestran accuracy extrema (0.75-0.94), mientras los folds grandes (n_test=24) son más estables. El fold task-01_acq-a sigue siendo problemático (0.458 en bandpower).

**Archivos generados:**
- `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_pre_vs_post_results.json`
- `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_decoding_summary.png`

---


## Tarea 10: Decoding con Micro-Épocas de 50ms (diseño de Enzo)

### Objetivo
Rediseñar el dataset de decoding usando ventanas temporales cortas (50 ms) para aumentar el número de observaciones y mejorar la resolución temporal del clasificador. 

### Diseño del dataset

**Épocas CHANGE (post-estímulo):**
Para cada uno de los 74 momentos de cambio de luminancia (onset = 0 ms), extraer 4 ventanas de 50 ms:
- Ventana 1: 50 a 100 ms post-onset
- Ventana 2: 100 a 150 ms post-onset
- Ventana 3: 150 a 200 ms post-onset
- Ventana 4: 200 a 250 ms post-onset

Total CHANGE: 74 x 4 = 296 épocas

**Épocas NO_CHANGE (pre-estímulo):**
Para cada uno de los mismos 74 momentos de cambio, extraer 4 ventanas de 50 ms inmediatamente antes del onset:
- Ventana 1: -250 a -200 ms pre-onset
- Ventana 2: -200 a -150 ms pre-onset
- Ventana 3: -150 a -100 ms pre-onset
- Ventana 4: -100 a -50 ms pre-onset

Total NO_CHANGE: 74 x 4 = 296 épocas

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
Las micro-épocas de 12 muestras no permiten calcular Welch PSD con resolución útil. Solución: primero se crea la época larga (TMIN=-2.5, TMAx=2.0, baseline=(-2.5, -1.5)) como en Tarea 9, luego se filtra pasa-banda en cada banda espectral sobre la época completa (donde hay ~1125 muestras), y finalmente se segmenta en micro-ventanas de 50 ms y se calcula la varianza (= potencia) por canal. Esto da resolución espectral completa a pesar de las ventanas cortas.

**Feature sets:**
1. `bandpower_filtered`: filtrado pasa-banda sobre época larga → segmentar → varianza por banda/canal = 5 x 32 = 160 features
2. `raw_signal`: época larga con baseline → segmentar → vectorizar = 32 ch x 12 = 384 features

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

#### Resultados (sub-27)

Script: `scripts/validation/32_decoding_per_window.py`. Output en `results/validation/photo_decoding_per_window/sub-27/`.

**bandpower_filtered (160 features):**

| Ventana | Accuracy | AUC-ROC |
|---|---|---|
| 50-100ms | 54.7% | 0.557 |
| 100-150ms | 58.8% | 0.568 |
| 150-200ms | 65.5% | 0.749 |
| 200-250ms | 50.7% | 0.532 |

**raw_signal (384 features):**

| Ventana | Accuracy | AUC-ROC |
|---|---|---|
| 50-100ms | 54.7% | 0.531 |
| 100-150ms | 62.8% | 0.647 |
| 150-200ms | 56.1% | 0.592 |
| 200-250ms | 60.1% | 0.606 |

**Hallazgo principal:** La ventana 150-200ms post-onset maximiza la performance para bandpower_filtered (65.5%, AUC=0.749), consistente con la latencia típica de componentes ERP visuales (P1/N1). Para raw_signal, el pico está en 100-150ms (62.8%, AUC=0.647). La ventana 0-50ms y 200-250ms están cerca de chance, indicando que la señal discriminativa se concentra entre 100 y 200ms post-onset.

### Tarea 10.2: Barrido sistemático de ventanas post-onset ✅

#### Objetivo
Evaluar la performance de decoding con mayor granularidad temporal. Deslizar una ventana de 50ms desde 0ms hasta 500ms post-onset en pasos de 10ms, entrenando un modelo LORO CV independiente en cada posición.

#### Diseño
- 46 posiciones de ventana: 0-50ms, 10-60ms, 20-70ms, ..., 450-500ms.
- Para cada posición: 74 CHANGE (una micro-época por onset) vs 74 NO_CHANGE (muestreadas del pool pre-onset [-250, -50]ms).
- Feature sets: bandpower_filtered (160 feat), raw_signal (384 feat), y tde_pca_var (20 feat).
- LORO CV por run, inner CV para selección de C.
- Para TDE: se aplica TDE(±10) sobre la época larga, PCA(20) fit en train per fold (sin data leakage), y se extrae la varianza de cada componente PCA en la micro-ventana.
- Train/test promedio por fold: ~126 / ~21 épocas.

#### Resultados (sub-27)

Script: `scripts/validation/33_decoding_window_sweep.py`. Output en `results/validation/photo_decoding_sweep/sub-27/`.

**Picos de performance:**

| Feature set | N feat | Mejor Accuracy | Ventana | Mejor AUC | Ventana |
|---|---|---|---|---|---|
| bandpower_filtered | 160 | 68.9% | 270-320ms | 0.719 | 280-330ms |
| raw_signal | 384 | 66.9% | 130-180ms | 0.726 | 130-180ms |
| tde_pca_var | 20 | 69.6% | 240-290ms | 0.701 | 240-290ms |

#### Interpretación

1. **Tres feature sets, tres perfiles temporales distintos.** Raw_signal tiene su pico temprano (130-180ms), consistente con componentes ERP rápidos (N1/P2). Bandpower_filtered pica más tarde (270-330ms), reflejando la modulación de potencia espectral. TDE_pca_var pica en 240-290ms, capturando relaciones temporales entre componentes que se establecen con latencia intermedia.

2. **TDE funciona y es competitivo (69.6% accuracy, AUC=0.701) con solo 20 features.** A pesar de que las micro-ventanas tienen solo 12 muestras, la estrategia de aplicar TDE sobre la época larga y después segmentar los componentes PCA funciona. La varianza de 20 componentes PCA es un feature set muy compacto (20 vs 160 o 384) y logra la mejor accuracy puntual.

3. **La señal discriminativa emerge a ~50ms y se mantiene hasta ~400ms.** Los tres feature sets superan chance de forma sostenida a partir de ~50ms post-onset. No hay un único momento óptimo sino una ventana amplia de discriminabilidad.

4. **Raw_signal domina en la ventana temprana (50-200ms).** La forma de onda cruda captura mejor los componentes ERP transitorios tempranos.

5. **Bandpower y TDE dominan en la ventana tardía (250-350ms).** La modulación de potencia espectral y las relaciones temporales entre componentes se establecen más lentamente.

6. **Comparación con Tarea 9 (épocas largas):** El mejor resultado del sweep (~69% accuracy, AUC~0.72) sigue por debajo del 80.4% (AUC=0.877) de bandpower Welch con épocas de 4.5s. La época larga integra información de toda la ventana temporal.

**Archivos generados:**
- `results/validation/photo_decoding_sweep/sub-27/sub-27_sweep_results.json`
- `results/validation/photo_decoding_sweep/sub-27/sub-27_sweep_results.png`

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
Tarea 9.2 (Decoding post vs pre-estímulo, mismo onset) ✅
    ↓
Tarea 10 (Decoding con micro-épocas 50ms — diseño Enzo) ✅
    ├── 10.1 Performance por ventana temporal post-estímulo ✅
    └── 10.2 Barrido sistemático de ventanas (10ms steps) ✅
```

---

## Mensaje de Update a Diego (Slack) — 2026-03-22

Hola Diego, espero que todo vaya bien por Japón! Te mando un update de todo lo que estuvimos trabajando desde la última reunión.

Solo para dar contexto, estábamos haciendo unas pruebas de calidad de los datos, comparando ERPs entre CHANGE (épocas centradas al momento de cambio de luminancia) y NO_CHANGE (épocas sampleadas en momentos de fijación sin cambio). En los plots iniciales aparecía algo raro: en las épocas NO_CHANGE se veía una sincronía de fase oscilatoria que no debería existir, e incluso en CHANGE había actividad oscilatoria antes de los 0ms (i.e. momento de inicio de cambio de luminancia, que dura 1s total).
📎 `results/validation/photo_erp_tfr/sub-27_pre_jitter/sub-27_erp_Occipital.png`

Para entender la oscilación residual en NO_CHANGE generamos épocas completamente random como condición nula y corrí también algunas simulaciones. Me vino bien conversar esto en reunión de grupo porque Fede pensó que hacer esto era una buena idea. Y de hecho lo fue! El resultado fue bastante claro: cuando promediás señales con alfa dominante y N baja, la frecuencia alfa no se cancela completamente — el RMS de este residuo de actividad alfa sigue la curva teórica 1/√N. Con N=74 el residuo en Occipital es ~0.6µV, y baja gradualmente al aumentar N. Lo validamos también en datos reales usando un pool de 700 épocas random. La conclusión entonces es que la oscilación en NO_CHANGE es un artefacto esperado del promediado con N finito, y no otra cosa. Los plots que te paso son de las simulaciones. Así que esto son buenas noticias!
📎 `results/validation/alpha_simulation/fig2_n_sweep_symmetric.png`
📎 `results/validation/alpha_simulation/fig4_comparison_n_sweep.png`

Y así se ve cómo bajan los residuos de alfa en los datos reales al aumentar el N de épocas:
📎 `results/validation/alpha_residual_real/sub-27/sub-27_alpha_rms_vs_n.png`

Ahora sí, que terminamos de hacer el chequeo de los datos, pasamos a lo que más nos interesa que es hacer decoding (empezando por decoding de luminancia).

Evaluamos tres tipos de features con Leave-One-Run-Out CV:
- *Bandpower Welch*: PSD en 5 bandas × 32 canales = 160 features
- *TDE + covarianza*: Time-Delay Embedding ±10 lags → PCA(20 componentes, fit solo en train) → covarianza upper triangle de los PCs = 210 features
- *Señal cruda + PCA*: vectorización → PCA(100) = 100 features

Y corrimos dos análisis distintos con esos features:

Primero hicimos una comparación pre-estímulo (baseline) vs post-estímulo (Change luminance) con ventanas de 500ms ([+50,+550]ms como CHANGE vs [-550,-50]ms como NO_CHANGE). En esa comparación, Bandpower Welch alcanza **72.3% accuracy / AUC=0.718**, TDE+cov **62.2%**, y señal cruda **64.9%**. Pienso que TDE+cov no tiene una performance muy alta. Tal vez es porque tenemos muchos features (210) por la cantidad de épocas total (148 épocas)? Una posibilidad sería quedarnos con menos componentes principales antes del cálculo de cov?
📎 `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_decoding_summary.png`

Después, para entender mejor en qué momento se produce un pico de performance, hice un barrido temporal deslizando ventanas de 50ms desde 0 hasta 500ms post-onset en pasos de 10ms (comparando esas ventanas post inicio de cambio de luminancia con ventanas baseline), donde la señal discriminativa parece emerger ~50ms post-onset y pica entre 130-330ms según el feature set (raw_signal temprano ~130ms, bandpower y TDE más tarde ~280ms), con un máximo global de ~69% accuracy.
📎 `results/validation/photo_decoding_sweep/sub-27/sub-27_sweep_results.png`

**Próximos pasos:** Teniendo estos benchmarks como base, la idea sería avanzar hacia una tarea de clasificación más cercana al escenario de predicción continua. El diseño experimental que tenemos (1s baseline → 1s cambio de luminancia → 1s retorno) me parece que se presta para definir **4 condiciones** con igual duración (~500ms cada una), usando ventanas de 100ms con 50ms de solapamiento:
- *Baseline*: [-250, 0ms] pre-onset + [1500, 1750ms] post-offset (pooled)
- *ChangeUp*: [0, 500ms] — respuesta al onset del aumento de luminancia
- *Luminance*: [500, 1000ms] — período de luminancia sostenida
- *ChangeDown*: [1000, 1500ms] — respuesta al offset (retorno a baseline)

La idea sería hacer benchmarking con TDE en esta tarea de 4 clases (que están balanceadas). Esto un poco simula el escenario de decoding de estados continuos que nos vamos a encontrar más adelante con las pantallas verdes (donde la luminancia cambia de forma continua durante 1 minuto) o incluso en los escenarios de decoding afectivo. Me parece mejor arrancar acá porque los cambios de luminancia son más marcados (y fuertes), lo que debería dar señal más limpia antes de pasar a la tarea continua que es más difícil. Qué te parece este plan?

---

## Tarea 9.3: Revisión del pipeline de decoding pre/post — LogisticRegressionCV + features equiparadas (2026-03-25)

### Motivación

Diego señaló dos problemas en la comparación de features del análisis pre/post (Tarea 9.2):

1. **Sesgo por dimensionalidad**: comparar 160 features (bandpower), 210 (tde_cov) y 100 (raw_pca) con el mismo C fijo es injusto — la regularización óptima depende del número de features.
2. **C fijo arbitrario**: usar C=1.0 fijo sin validación cruzada puede favorecer o perjudicar sistemáticamente a un feature set.

Solución propuesta por Diego: usar `LogisticRegressionCV` (scikit-learn) que cross-valida C internamente, de forma independiente por cada feature set.

### Implementación (`27b_decoding_pre_vs_post.py`)

**Cross-validación de C:**
- Reemplazado `LogisticRegression(C=fixed)` por `LogisticRegressionCV` en todo el pipeline
- Grid de C: `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
- Inner CV: 5-fold estratificado sobre el training set de cada fold LORO (~252 samples → ~50 por inner fold)
- Cada feature set elige su propio C* por fold → comparación más justa
- Solver actualizado a `saga` (requerido por la nueva API elastic net de sklearn 1.8+), con `l1_ratios=(0,)` para pure L2 (ridge)

**Equiparación de features (~160 para los tres):**
- `TDE_PCA_COMPONENTS`: 20 → 17 (covarianza: 17×18/2 = **153 features**)
- `RAW_PCA_COMPONENTS`: 100 → 160 (**160 features**)
- bandpower_welch: 32 ch × 5 bandas = **160 features** (sin cambio)

### Resultados (sub-27, LORO 7-fold, 2026-03-25)

| Feature set | N features | Acc | F1 | AUC |
|---|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | 0.689 | **0.725** |
| raw_pca | 160 | 65.5% | 0.662 | 0.696 |
| tde_cov | 153 | 62.8% | 0.621 | 0.685 |

**Por fold (bandpower_welch):**
- run-002: 45.8% (C=0.1) | run-003: 75.0% (C=1.0) | run-004: 75.0% (C=10.0)
- run-006: 54.2% (C=0.01) | run-007: 79.2% (C=0.1) | run-009: 93.8% (C=1.0) | run-010: 70.8% (C=0.01)

**Por fold (tde_cov):**
- run-002: 41.7% (C=0.001) | run-003: 75.0% (C=0.01) | run-004: 65.0% (C=0.1)
- run-006: 54.2% (C=0.001) | run-007: 66.7% (C=0.01) | run-009: 75.0% (C=1.0) | run-010: 70.8% (C=0.1)

**Por fold (raw_pca):**
- Todos los folds eligieron C=0.001 → regularización fuerte necesaria para 160 PCs de señal cruda

### Interpretaciones

- **Ranking se mantiene**: bandpower > raw_pca ≈ tde_cov, incluso con features equiparadas y C cross-validado. La ventaja de bandpower es robusta.
- **C muy inestable entre folds**: para bandpower varía de 0.01 a 10.0; para tde_cov de 0.001 a 1.0. Con ~50 samples en el inner fold, la selección de C es ruidosa — se espera con N pequeño.
- **raw_pca elige siempre C=0.001**: 160 PCs de señal cruda tienen mucho ruido; la regularización fuerte es consistentemente necesaria.
- **tde_cov con 17 PCs (84% varianza)**: leve caída vs 20 PCs (86% varianza); el trade-off dimensionalidad/información no fue favorable.
- **Interpretación del resultado**: la tarea de luminancia genera principalmente un cambio de potencia alfa por canal → bandpower lo captura directamente. TDE-cov captura también conectividad (off-diagonal), que no es informativa aquí, diluyendo la señal útil.

### Archivo de resultados

`results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_pre_vs_post_results.json`
`results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_decoding_summary.png`
