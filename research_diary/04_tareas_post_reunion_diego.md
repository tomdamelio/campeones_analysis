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

## Tarea 5: Agregar Jitter Temporal a Eventos NO_CHANGE_PHOTO

### Objetivo
Romper la sincronía de fase artificial en las épocas NO_CHANGE_PHOTO introduciendo un jitter aleatorio en los onsets, de modo que las distancias entre eventos consecutivos no sean equidistantes.

### Problema actual
En `scripts/validation/22_generate_photo_events.py`, la función `generate_no_change_photo_events()` (línea ~135) distribuye los eventos con spacing fijo:

```python
spacing = available / n_events
for i in range(n_events):
    onset = effective_onset + i * spacing  # ← equidistante
```

Este espaciado regular genera una periodicidad artificial que se manifiesta como sincronía de fase en los ERPs y TFRs, incluso cuando no hay estímulo real.

### Implementación

#### 5.1 Modificar `generate_no_change_photo_events()` en `22_generate_photo_events.py`

Agregar un parámetro `jitter_fraction` (o similar) que desplace aleatoriamente cada onset respecto de su posición equidistante nominal.

**Lógica propuesta:**
1. Calcular las posiciones nominales equidistantes como ahora: `nominal_onset_i = effective_onset + i * spacing`
2. Para cada evento, agregar un desplazamiento aleatorio uniforme dentro de un rango que no cause solapamiento con eventos vecinos. Por ejemplo: `jitter_i ~ Uniform(-spacing/3, +spacing/3)`
3. Asegurar que:
   - Ningún onset quede antes de `effective_onset` (inicio del fixation útil)
   - Ningún onset + 1.0s (duración del evento) exceda `fixation_onset + fixation_duration` (fin del fixation)
   - No haya solapamiento entre ventanas de 1s consecutivas (onset_i + 1.0 < onset_{i+1})
4. Agregar un parámetro `--seed` al CLI para reproducibilidad del jitter (usar `RANDOM_SEED = 42` como default, consistente con `config_luminance.py`)

**Archivos a modificar:**
- `scripts/validation/22_generate_photo_events.py`: función `generate_no_change_photo_events()` y `run_pipeline()`

**Parámetros nuevos sugeridos:**
- `jitter_fraction: float = 0.3` → fracción del spacing usada como rango máximo de jitter (±30% del spacing)
- `random_seed: int = 42` → seed para `np.random.default_rng()`

#### 5.2 Regenerar los TSVs de photo_events

Ejecutar:
```bash
python scripts/validation/22_generate_photo_events.py --subject 27
```

Esto sobreescribirá los TSVs en `data/derivatives/photo_events/sub-27/ses-vr/eeg/`.

#### 5.3 Re-ejecutar el pipeline de epoching y visualización

Ejecutar en secuencia:
```bash
python scripts/validation/23_epoch_photo_events.py --subject 27
python scripts/validation/25_erp_tfr_photo_contrast.py --subject 27
python scripts/validation/26_permutation_test_photo.py --subject 27
```

#### 5.4 Verificar que la sincronía de fase desapareció en NO_CHANGE

Comparar los nuevos plots de ERP y TFR (en `results/validation/photo_erp_tfr/sub-27/`) contra los anteriores:
- Los ERPs de NO_CHANGE_PHOTO deberían mostrar una señal más plana/ruidosa (sin componentes evocados claros)
- Los TFRs de NO_CHANGE_PHOTO no deberían mostrar patrones de potencia sincronizados en el tiempo
- El contraste CHANGE - NO_CHANGE debería mantenerse o mejorar

---

## Tarea 6: Ampliar Ventana de Épocas para Investigar Sincronía Pre-Estímulo

### Objetivo
Extender la ventana de las épocas de -1.5s → -3.0s (o más) y de +2.0s → +2.5s (o más) para verificar si la sincronía de fase pre-estímulo observada en CHANGE_PHOTO se extiende más allá de -1000 ms. Si la sincronía aparece entre -3000 ms y -1000 ms, hay un problema; si no aparece, la sincronía entre -1000 ms y 0 ms se explica por la estructura del estímulo (el segundo de flicker previo al evento).

### Parámetros actuales (en `23_epoch_photo_events.py`)
```python
TMIN = -1.5    # segundos
TMAX = 2.0     # segundos
BASELINE = (-1.5, -1.0)  # corrección de línea base
```

### Parámetros nuevos propuestos
```python
TMIN = -3.5    # segundos (ampliar a -3500 ms)
TMAX = 2.5     # segundos (ampliar a +2500 ms)
BASELINE = (-3.5, -3.0)  # mover baseline a ventana lejana pre-estímulo
```

**Justificación del baseline (-3.5s a -3.0s):**
- Al mover el baseline lejos del estímulo, evitamos que la corrección de línea base contamine la ventana de interés (-1000 ms a 0 ms donde aparece el flicker pre-evento).
- La ventana -3.5s a -3.0s debería ser actividad de fondo sin influencia del estímulo.
- Este baseline se usará tanto para los ERPs como para los TFRs (modo "percent" en `25_erp_tfr_photo_contrast.py`).

### Implementación

#### 6.1 Modificar constantes en `23_epoch_photo_events.py`

Cambiar:
```python
TMIN = -3.5
TMAX = 2.5
BASELINE = (-3.5, -3.0)
```

**Consideración importante:** Verificar que las épocas ampliadas no excedan los límites de la grabación EEG. Algunos eventos cercanos al inicio o fin de un run podrían no tener suficiente señal para una ventana de 6 segundos. El código actual en `create_epochs_for_run()` ya filtra eventos fuera de los límites del recording (`valid = (samples >= eeg_raw.first_samp) & (samples < last_sample)`), pero MNE también rechazará épocas que excedan los bordes al crear el objeto `Epochs`. Esto podría reducir el número de épocas disponibles — verificar cuántas se pierden.

#### 6.2 Actualizar baseline en `25_erp_tfr_photo_contrast.py`

Actualmente el baseline para TFR está hardcodeado:
```python
tfr.apply_baseline(baseline=(-1.5, -1.0), mode="percent")
```

Cambiar a:
```python
tfr.apply_baseline(baseline=(-3.5, -3.0), mode="percent")
```

Esto aplica en tres lugares:
- `plot_tfr()`: línea donde se aplica baseline a cada condición
- `plot_contrast()`: líneas donde se aplica baseline a `tfr_a` y `tfr_b`

#### 6.3 Actualizar baseline en `26_permutation_test_photo.py`

Cambiar la constante:
```python
BASELINE = (-3.5, -3.0)
```

#### 6.4 Agregar líneas de referencia adicionales en los plots

En los plots de ERP y TFR (scripts 25 y 26), agregar líneas verticales de referencia para marcar los momentos clave de la estructura temporal del estímulo:
- `axvline(-1000)`: inicio del flicker pre-evento (1s de foto antes del evento)
- `axvline(0)`: onset del evento (ya existe)
- `axvline(1000)`: offset del flicker post-evento (ya existe)

Esto ayudará a interpretar visualmente si la sincronía pre-estímulo coincide con la ventana del flicker (-1000 ms a 0 ms) o se extiende más atrás.

#### 6.5 Re-ejecutar pipeline completo

```bash
python scripts/validation/23_epoch_photo_events.py --subject 27
python scripts/validation/25_erp_tfr_photo_contrast.py --subject 27
python scripts/validation/26_permutation_test_photo.py --subject 27
```

#### 6.6 Análisis de resultados esperados

Revisar los plots generados en `results/validation/photo_erp_tfr/sub-27/` y `results/validation/photo_stats/sub-27/`:

- **Si NO hay sincronía entre -3000 ms y -1000 ms:** La sincronía observada entre -1000 ms y 0 ms se explica por el flicker de fotodiodo pre-evento (es la respuesta esperada al estímulo visual de 1s). Esto es normal y no invalida los resultados.
- **Si SÍ hay sincronía entre -3000 ms y -1000 ms:** Hay un problema más profundo — posiblemente un artefacto del preprocesamiento, un efecto de filtrado (ringing), o alguna otra fuente de contaminación. En ese caso, habría que investigar más (conecta con Tarea 3 del diario: QC de datos crudos).

---

## Tarea 7: Decoding de Change/No-Change con Todos los Electrodos y Modelos TDE

### Objetivo
Ahora que se confirmó un efecto a nivel occipital y temporal en el contraste CHANGE vs NO_CHANGE, correr modelos de decoding sobre la tarea de clasificación binaria Change/No-Change usando las nuevas épocas (con jitter y ventana ampliada), esta vez con todos los 32 electrodos (no solo el ROI posterior de 11 canales).

### Contexto
Los modelos de decoding previos (scripts 08–20 en `scripts/modeling/`) usaban:
- Solo canales posteriores/occipitales (`POSTERIOR_CHANNELS`: 11 canales)
- Épocas de 500 ms con step de 100 ms dentro de segmentos de video de ~60s
- Target: luminancia continua (regresión) o cambio binario de luminancia (clasificación)
- Pipeline: EEG → TDE (glhmm) → PCA → Covarianza → Ridge/LogisticRegression

Ahora se quiere aplicar estos mismos métodos pero sobre la tarea de detección de CHANGE_PHOTO vs NO_CHANGE_PHOTO con las épocas definidas en la Tarea 1 del diario (scripts 22-23).

### Implementación

#### 7.1 Crear nuevo script `scripts/validation/27_decoding_photo_change.py`

Este script debe:

1. **Cargar las épocas** desde `results/validation/photo_epochs/sub-{subject}_photo-epo.fif` (generadas por script 23 con los cambios de Tareas 5 y 6).

2. **Usar todos los 32 canales EEG** (no solo `POSTERIOR_CHANNELS`). La lista completa está en `23_epoch_photo_events.py`:
   ```python
   EEG_CHANNELS = [
       'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
       'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
       'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
       'FT9', 'FT10', 'TP9', 'TP10', 'FCz',
   ]
   ```

3. **Extraer features con múltiples métodos** (los mismos que se probaron en el pipeline de luminancia):

   **Método A — Raw TDE + Covarianza** (como script 13 / 20):
   - Para cada época MNE: extraer la señal cruda `(n_channels, n_samples)`
   - Transponer a `(n_samples, n_channels)` → aplicar `apply_tde_only()` con `tde_lags=TDE_WINDOW_HALF` (10)
   - Aplicar PCA global (`fit_global_pca()` + `apply_global_pca()`) con `TDE_PCA_COMPONENTS` (20)
   - Computar covarianza por época (`compute_epoch_covariance()`) → vector de features upper-triangle
   - Features resultantes: `n_pca * (n_pca + 1) / 2 = 20 * 21 / 2 = 210` features por época

   **Método B — Spectral Band Power + TDE** (como script 12):
   - Para cada época: extraer band-power con `extract_bandpower()` usando `SPECTRAL_BANDS` (5 bandas × 32 canales = 160 features)
   - Aplicar `apply_time_delay_embedding()` con `window_half=TDE_WINDOW_HALF`
   - Reducir con PCA

   **Método C — Raw signal features (baseline)**:
   - Para cada época: extraer la señal cruda promediada en ventanas temporales (mean + variance por canal)
   - Sin TDE, como baseline de comparación

4. **Clasificación binaria:**
   - Target: `y = 1` (CHANGE_PHOTO) o `y = 0` (NO_CHANGE_PHOTO), extraído del `event_id` de las épocas MNE
   - Modelo: `StandardScaler → LogisticRegression(solver='lbfgs', max_iter=1000)` (como script 20)
   - Evaluación: Leave-One-Run-Out CV (en lugar de Leave-One-Video-Out, ya que las épocas vienen de distintos runs, no de distintos videos)
   - Métricas: Accuracy, Precision, Recall, F1, AUC-ROC (como script 20)
   - Undersampling de clase mayoritaria en training set (como script 20, función `undersample_majority_class()`)

5. **Comparación de métodos:**
   - Generar tabla CSV con métricas por fold y por método
   - Generar bar plot comparativo (como `plot_classification_cv_results()` en script 20)

#### 7.2 Parámetros de configuración

Reutilizar los parámetros de `config_luminance.py` donde aplique:
- `TDE_WINDOW_HALF = 10`
- `TDE_PCA_COMPONENTS = 20`
- `SPECTRAL_BANDS` (5 bandas)
- `RANDOM_SEED = 42`

Parámetros nuevos específicos para este script:
- `USE_ALL_CHANNELS = True` (flag para usar 32 canales en lugar de solo posterior)
- `CV_STRATEGY = "leave_one_run_out"` (en lugar de leave_one_video_out)

#### 7.3 Consideraciones sobre las épocas

Las épocas de photo events tienen una estructura diferente a las épocas de luminancia:
- **Épocas de luminancia** (scripts 08-20): ventanas de 500 ms con step de 100 ms dentro de un video continuo de ~60s. Hay cientos de épocas por video, y la señal es continua.
- **Épocas de photo events** (scripts 22-23): ventanas de -3.5s a +2.5s (6 segundos) centradas en eventos discretos. Hay ~70 épocas CHANGE y ~70 NO_CHANGE en total para sub-27.

Para aplicar TDE sobre las épocas de photo events, hay dos enfoques posibles:
- **Opción A (recomendada):** Tratar cada época MNE como un segmento independiente. Extraer la señal `(n_samples, n_channels)` de cada época, aplicar TDE, PCA, y luego computar covarianza sobre toda la época. Esto da un vector de features por época.
- **Opción B:** Concatenar todas las épocas de una condición y aplicar TDE sobre la señal concatenada. Esto es más similar al pipeline original pero pierde la estructura de épocas individuales.

Usar Opción A para mantener consistencia con la evaluación por época.

#### 7.4 Output

Guardar resultados en `results/validation/photo_decoding/sub-{subject}/`:
- `sub-{subject}_decoding_results.csv`: métricas por fold y método
- `sub-{subject}_decoding_comparison.png`: bar plot comparativo
- `sub-{subject}_decoding_summary.json`: resumen con parámetros y métricas promedio

---

## Orden de Ejecución

```
Tarea 5 (Jitter en NO_CHANGE)
    ↓
Tarea 6 (Ampliar ventana de épocas + ajustar baseline)
    ↓
Re-ejecutar pipeline 22 → 23 → 25 → 26
    ↓
Tarea 7 (Decoding con todos los electrodos)
```

Las Tareas 5 y 6 se pueden implementar en paralelo (modifican archivos distintos: 22 vs 23/25/26), pero deben ejecutarse en secuencia (primero regenerar TSVs con jitter, luego re-epochar con ventana ampliada).

---

## Dependencia con Tareas Previas del Diario

- **Tarea 3 (QC datos crudos):** Si la Tarea 6 revela sincronía pre-estímulo más allá de -1000 ms, la Tarea 3 se vuelve prioritaria para investigar artefactos de preprocesamiento.
- **Tarea 4 (Multi-sujeto):** Las Tareas 5-7 se implementan primero para sub-27. Una vez validadas, se escalan a sub-33 y sub-37 como parte de la Tarea 4.
