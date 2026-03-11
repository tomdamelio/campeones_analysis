# Diario de Investigacion — CAMPEONES Analysis
## Tarea 1: Reestructuracion de Eventos (Triggers) y Analisis Base

**Proyecto:** campeones_analysis
**Fecha:** 2026-03-10
**Estado:** Planificacion detallada

---

### Objetivo

Abandonar los cambios de luminancia como punto de anclaje (enfoque actual en `scripts/validation/21e_erp_tfr_derivative_threshold.py`) y utilizar los marcadores de fotodiodo — estimulos discretos y bien definidos — como base para buscar diferencias claras en la senal EEG.

Se definen dos nuevas condiciones de eventos:
- **CHANGE_PHOTO**: Marcas de fotodiodo de 1 segundo que flanquean cada evento del TSV
- **NO_CHANGE_PHOTO**: Ventanas de 1 segundo muestreadas dentro del periodo de fixation (baseline)

---

### 1.1 Definicion precisa de los eventos CHANGE_PHOTO

Cada evento registrado en los archivos `merged_events` TSV esta "emparedado" (sandwich) entre dos marcas de fotodiodo de exactamente 1 segundo:

**Marca PRE-evento:**
- Onset: `evento.onset - 1.0` segundos
- Duracion: 1.0 segundo
- Termina exactamente cuando comienza el evento

**Marca POST-evento:**
- Onset: `evento.onset + evento.duration` segundos
- Duracion: 1.0 segundo
- Comienza exactamente cuando termina el evento


Esto aplica a **todos** los eventos del TSV (video, video_luminance, calm, fixation). Cada evento genera 2 marcas CHANGE_PHOTO.

**Ejemplo concreto** (de `sub-27_ses-vr_task-03_acq-a_run-004_desc-merged_events.tsv`):

| evento | onset | duration | trial_type |
|--------|-------|----------|------------|
| video 9 | 45.864 | 154.008 | video |

Genera:
- CHANGE_PHOTO_PRE: onset=44.864, duration=1.0 (segundo antes del video)
- CHANGE_PHOTO_POST: onset=199.872, duration=1.0 (segundo despues del video)

**Estructura temporal:**
```
[...silencio...][1s PHOTO flicker][== EVENTO (video/lum/calm) ==][1s PHOTO flicker][...silencio...]
                 ^ CHANGE_PHOTO                                   ^ CHANGE_PHOTO
```

> Nota: La documentacion original de `03_detect_markers.py` describe la estructura del marcador audiovisual como "pantalla negra silenciosa de 1 s + parpadeo audiovisual de 1 s + pantalla negra silenciosa de 1 s" (total 3 s), insertada antes y despues de cada clip. El segundo de parpadeo es el que constituye el evento CHANGE_PHOTO.

---

### 1.2 Definicion precisa de los eventos NO_CHANGE_PHOTO

Los eventos NO_CHANGE_PHOTO se obtienen **unicamente** del primer bloque de cada sesion:
- **task-01, acq-a** (primer bloque sesion A): `sub-27_ses-vr_task-01_acq-a_run-002`
- **task-01, acq-b** (primer bloque sesion B): `sub-27_ses-vr_task-01_acq-b_run-007`

Dentro de estos bloques, el evento `fixation` (trial_type=fixation, stim_id=500, condition=baseline) tiene una duracion de ~300 segundos (5 minutos de cruz de fijacion). Este periodo de baseline no contiene estimulos visuales ni cambios de luminancia.

**Procedimiento:**
1. Contar el numero total de eventos CHANGE_PHOTO generados en el paso 1.1 para ese sujeto
2. Descartar los primeros 5 segundos del fixation para evitar efectos de novedad o transicion al inicio del bloque
3. Dentro del tiempo restante de fixation (duracion - 5s), distribuir equitativamente la misma cantidad de ventanas de 1 segundo
4. Cada ventana es un evento NO_CHANGE_PHOTO


**Ejemplo concreto** (de `sub-27_ses-vr_task-01_acq-a_run-002`):

| evento | onset | duration | trial_type |
|--------|-------|----------|------------|
| fixation | 43.176 | 300.016 | fixation |

Si se necesitan, por ejemplo, 20 eventos NO_CHANGE_PHOTO de este bloque:
- Onset efectivo del sampleo: `43.176 + 5.0 = 48.176` (se descartan los primeros 5s)
- Espacio disponible: `300.016 - 5.0 = 295.016` segundos
- Separacion entre eventos: `295.016 / 20 = 14.75` segundos
- NO_CHANGE_PHOTO_1: onset=48.176, duration=1.0
- NO_CHANGE_PHOTO_2: onset=62.926, duration=1.0
- ...
- NO_CHANGE_PHOTO_20: onset=328.426, duration=1.0

La cantidad total de NO_CHANGE_PHOTO debe ser identica a la cantidad total de CHANGE_PHOTO para tener condiciones balanceadas.

---

### 1.3 Implementacion: Generacion de nuevos TSVs con eventos agregados

**Que hacer:** Crear una funcion/script que recorra los TSVs de `merged_events` de un sujeto dado y genere nuevos TSVs (o extienda los existentes) con las filas adicionales de CHANGE_PHOTO y NO_CHANGE_PHOTO.

**Archivos de entrada:**
- `data/derivatives/merged_events/sub-{ID}/ses-vr/eeg/*_desc-merged_events.tsv`
  - Para sub-27: 8 archivos (task-01 a task-04, acq-a y acq-b)
  - Columnas: onset, duration, trial_type, stim_id, condition, stim_file

**Archivos de salida sugeridos:**
- `data/derivatives/photo_events/sub-{ID}/ses-vr/eeg/*_desc-photo_events.tsv`
  - Mismas columnas + nuevas filas con trial_type = `CHANGE_PHOTO` y `NO_CHANGE_PHOTO`

**Logica del script:**

```
Para cada sujeto:
  change_photo_count = 0

  # Paso 1: Recorrer TODOS los runs y generar CHANGE_PHOTO
  Para cada archivo merged_events TSV:
    Para cada evento en el TSV:
      Agregar CHANGE_PHOTO_PRE: onset = evento.onset - 1.0, duration = 1.0
      Agregar CHANGE_PHOTO_POST: onset = evento.onset + evento.duration, duration = 1.0
      change_photo_count += 2

  # Paso 2: Generar NO_CHANGE_PHOTO solo de task-01
  Para cada archivo merged_events de task-01 (acq-a y acq-b):
    fixation_event = evento donde trial_type == 'fixation'
    n_no_change_from_this_block = change_photo_count / 2  # mitad de cada bloque task-01
    effective_onset = fixation_event.onset + 5.0           # descartar primeros 5s
    available_duration = fixation_event.duration - 5.0
    spacing = available_duration / n_no_change_from_this_block
    Para i en range(n_no_change_from_this_block):
      Agregar NO_CHANGE_PHOTO: onset = effective_onset + i * spacing, duration = 1.0
```


**Archivos de referencia en el repositorio:**
- `scripts/preprocessing/03_detect_markers.py` -- funcion `detect_flicker_in_photo()` (linea 622)
- `scripts/preprocessing/02_create_events_tsv.py` -- genera los TSVs iniciales de eventos
- `scripts/modeling/config_luminance.py` -- define `RUNS_CONFIG` para sub-27:
  ```python
  RUNS_CONFIG = [
      {'id': '002', 'acq': 'a', 'block': 'block1', 'task': '01'},
      {'id': '003', 'acq': 'a', 'block': 'block2', 'task': '02'},
      {'id': '004', 'acq': 'a', 'block': 'block3', 'task': '03'},
      {'id': '006', 'acq': 'a', 'block': 'block4', 'task': '04'},
      {'id': '007', 'acq': 'b', 'block': 'block1', 'task': '01'},
      {'id': '009', 'acq': 'b', 'block': 'block3', 'task': '03'},
      {'id': '010', 'acq': 'b', 'block': 'block4', 'task': '04'},
  ]
  ```
  Nota: run-008 (task-02, acq-b) excluido por problemas tecnicos.

---

### 1.4 Segmentar en nuevas epocas (Epochs)

Una vez generados los TSVs con CHANGE_PHOTO y NO_CHANGE_PHOTO, crear epocas MNE centradas en estos nuevos marcadores.

**Parametros de epoching** (basados en `21e_erp_tfr_derivative_threshold.py`):
- `tmin`: -0.5 s
- `tmax`: 1.5 s (para capturar el segundo completo del flicker + post)
- `baseline`: ventana pre-estimulo para correccion de linea base
- `picks`: canales EEG (32 canales definidos en `config.py`)

**Referencia:** `create_epochs_from_threshold()` en `21e` (linea 159)

**Canales ROI** (de `config_luminance.py`):
- Posterior/Occipital: O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6
- ROIs por region (de `21e`): Frontal, Temporal, Parietal, Occipital

---

### 1.5 Visualizacion interactiva de las nuevas epocas

Antes de computar ERPs o TFRs, verificar visualmente que los nuevos eventos CHANGE_PHOTO y NO_CHANGE_PHOTO estan correctamente ubicados en la senal. Esto es analogo a lo que se hace en el paso 3 del pipeline (`03_detect_markers.py`) cuando se visualizan las marcas de AUDIO y PHOTO para validacion manual, pero aplicado a las nuevas epocas.

**Que hacer:** Para cada archivo preprocesado del sujeto 27, cargar la senal EEG junto con los canales AUDIO y PHOTO, y superponer las anotaciones de CHANGE_PHOTO y NO_CHANGE_PHOTO como marcas visuales. Esto permite confirmar que:
- Los eventos CHANGE_PHOTO coinciden con actividad real en el canal PHOTO (flicker visible)
- Los eventos NO_CHANGE_PHOTO caen en zonas de silencio/baseline (sin actividad en PHOTO ni AUDIO)
- No hay solapamiento entre epocas ni eventos mal posicionados

**Implementacion sugerida:**
1. Cargar el archivo `.vhdr` preprocesado de cada run
2. Cargar el TSV de `photo_events` generado en la Tarea 1.3
3. Convertir los eventos CHANGE_PHOTO y NO_CHANGE_PHOTO a `mne.Annotations`
4. Asignar las anotaciones al objeto Raw con `raw.set_annotations()`
5. Visualizar con `raw.plot()` mostrando canales AUDIO, PHOTO y algunos canales EEG (ej. O1, O2, Pz)

**Referencia directa:** `visualize_signals_with_annotations()` en `03_detect_markers.py` (linea 1460). Esta funcion ya:
- Selecciona canales AUDIO, PHOTO, joystick_x
- Aplica z-score para normalizar las senales
- Resamplea a 1000 Hz si la frecuencia es muy alta
- Abre el visualizador interactivo de MNE con `raw.plot(duration=20, block=True)`
- Permite edicion manual de anotaciones (agregar/eliminar/ajustar con tecla 'a' y click derecho)

Adaptar esta funcion para que muestre las nuevas anotaciones CHANGE_PHOTO (ej. en color rojo) y NO_CHANGE_PHOTO (ej. en color verde) sobre los canales AUDIO, PHOTO y EEG occipitales.

**Salida esperada:** Visualizacion interactiva por run donde se pueda scrollear por la senal y verificar que cada marca esta en el lugar correcto. Opcionalmente, permitir correccion manual si alguna marca esta mal posicionada.

---

### 1.6 Calcular y graficar ERPs

Computar ERPs promediando las epocas por condicion:
- ERP de CHANGE_PHOTO
- ERP de NO_CHANGE_PHOTO

**Referencia:** `plot_erp_3cond()` en `21e` (linea 214) -- media +/- SEM, linea vertical en t=0, eje Y invertido.

---

### 1.7 Computar mapas de tiempo-frecuencia (TFR)

**Referencia:** `plot_tfr_3cond()` en `21e` (linea 274):
- Frecuencias: `np.logspace(*np.log10([3, 40]), num=20)` (3-40 Hz)
- Ciclos: `freqs / 2.0`
- Baseline: `(-1.8, -1.0)`, modo "percent"

---

### 1.8 Contrastes estadisticos iniciales

Comparar CHANGE_PHOTO vs. NO_CHANGE_PHOTO:
- Diferencia de ERPs por ROI
- Diferencia de TFRs (mapas de potencia)
- Verificar si el contraste es evidente visualmente antes de aplicar estadistica formal (Tarea 2)

**Referencia:** `plot_null_contrast()` en `21e` (linea 359)


---

### Datos disponibles para sub-27

| Run | Task | Acq | Eventos en TSV | Fixation (300s) |
|-----|------|-----|----------------|-----------------|
| 002 | 01 | a | fixation, calm, video x3, video_luminance | Si |
| 003 | 02 | a | video x3, video_luminance | No |
| 004 | 03 | a | video x4, video_luminance | No |
| 006 | 04 | a | video x4, video_luminance, calm | No |
| 007 | 01 | b | fixation, calm, video x3, video_luminance | Si |
| 009 | 03 | b | (por verificar) | No |
| 010 | 04 | b | (por verificar) | No |

Los eventos NO_CHANGE_PHOTO se extraen exclusivamente de los runs 002 (acq-a) y 007 (acq-b).

### Estimacion de cantidad de eventos

Para sub-27 (7 runs activos, ~5 eventos por run en promedio):
- ~35 eventos en total x 2 marcas CHANGE_PHOTO cada uno = ~70 CHANGE_PHOTO
- ~70 NO_CHANGE_PHOTO distribuidos entre los dos bloques de fixation (~35 por bloque)
- Con ~295s de fixation util (300s - 5s de descarte) y 35 eventos: separacion ~8.4s

---
---

## Tarea 2: Implementacion de Validacion Estadistica Rigurosa

### Motivacion

El analisis anterior (basado en contrastes ChangeUp/ChangeDown/NoChange por derivada de luminancia dentro del estimulo de 1 minuto en `21e_erp_tfr_derivative_threshold.py`) mostro que una comparacion split-half de la condicion NoChange arrojaba diferencias significativas -- un indicador claro de falsos positivos. Ese enfoque queda descartado.

La Tarea 2 se aplica al **nuevo contraste definido en la Tarea 1**: **CHANGE_PHOTO vs. NO_CHANGE_PHOTO**. Una vez que la Tarea 1 genere las epocas de ambas condiciones (marcas de fotodiodo de 1s flanqueando cada evento vs. ventanas de 1s dentro del fixation baseline), se necesita un test estadistico robusto para determinar si las diferencias observadas en ERPs y TFRs son genuinas o producto del azar.

El repositorio ya tiene una implementacion de permutacion de etiquetas en `scripts/modeling/17_baseline_models.py` (`permute_labels_within_video_groups()`, `run_shuffle_baseline()`), pero ese enfoque esta disenado para modelos de regresion (prediccion de luminancia), no para contrastes ERP/TFR. Se necesita un test de permutaciones especifico para comparaciones de senales EEG entre CHANGE_PHOTO y NO_CHANGE_PHOTO.


### Sub-tareas

#### 2.1 Eliminar o comentar el codigo de contrastes por derivada de luminancia

En `21e_erp_tfr_derivative_threshold.py`, la funcion `plot_null_contrast()` (linea 359) realiza la comparacion split-half de la condicion NoChange (basada en derivada de luminancia del estimulo de 1 minuto). Este codigo y el enfoque de ChangeUp/ChangeDown/NoChange por umbral de derivada quedan obsoletos con la nueva definicion de condiciones. Eliminar o comentar.

#### 2.2 Desarrollar funcion de test de permutaciones para el contraste CHANGE_PHOTO vs. NO_CHANGE_PHOTO

Crear una funcion que implemente un test de permutaciones no parametrico para comparar las epocas de las dos nuevas condiciones:

1. Tomar las epocas MNE de CHANGE_PHOTO y las epocas MNE de NO_CHANGE_PHOTO (generadas en Tarea 1)
2. Calcular la metrica de diferencia real (diferencia de medias punto a punto, o estadistico t)
3. En cada iteracion (ej. 1000):
   - Mezclar aleatoriamente las etiquetas CHANGE_PHOTO / NO_CHANGE_PHOTO entre todas las epocas
   - Recalcular la metrica de diferencia con las etiquetas permutadas
4. Construir la distribucion nula empirica
5. Comparar la diferencia real contra la distribucion nula para obtener un valor p

Considerar usar `mne.stats.permutation_cluster_test()` que ademas corrige por comparaciones multiples (problema de multiples canales x tiempos x frecuencias).

#### 2.3 Programar el bucle iterativo

Implementar el bucle de 1000 iteraciones (configurable). Referencia de estructura: `run_shuffle_baseline()` en `17_baseline_models.py` que ya implementa un bucle con seeds reproducibles (`iter_seed = random_seed + iteration_idx`) y reporting de progreso cada 10 iteraciones. La configuracion actual en `config_luminance.py` define `N_PERMUTATIONS = 0` (deshabilitado) y `RANDOM_SEED = 42`.

#### 2.4 Construir distribucion nula y calcular valor p

Para cada punto temporal (o cluster de puntos), calcular que proporcion de las diferencias permutadas excede la diferencia real observada en el contraste CHANGE_PHOTO vs. NO_CHANGE_PHOTO. Esto da un valor p corregido por comparaciones multiples si se usa el enfoque de clusters.

---
---

## Tarea 3: Control de Calidad e Inspeccion de Datos Crudos (Raw Data)

### Motivacion

Se detecto una oscilacion anomala (alfa sincronizado en fase) durante los periodos de 60 segundos de luminancia, independientemente de la condicion. Esto sugiere que la oscilacion podria ser un artefacto introducido por algun paso del preprocesamiento y no una senal genuina del cerebro.

Los videos de luminancia tienen duraciones de ~60 segundos (ej. en `sub-27_ses-vr_task-03_acq-a_run-004`, el video_luminance stim_id=103 tiene `duration: 60.016`). La oscilacion alfa aparece consistentemente durante estos bloques.

El preprocesamiento en `04_preprocessing_eeg.py` aplica:
1. Filtro notch a 50 Hz
2. Filtro pasa-banda 0.5-48 Hz
3. Deteccion de canales malos con PyPrep (`NoisyChannels`)
4. ICA con etiquetado automatico (`mne_icalabel`)
5. Interpolacion de canales malos
6. Re-referencia al promedio

Cualquiera de estos pasos podria introducir artefactos.


### Sub-tareas

#### 3.1 Cargar datos crudos

Escribir un script que cargue los archivos `.vhdr` de `data/raw/sub-{ID}/ses-vr/eeg/` usando `mne.io.read_raw_brainvision()`. La funcion `set_chs_montage()` en `src/campeones_analysis/utils/preprocessing_helpers.py` define el mapeo de tipos de canal.

#### 3.2 Extraer segmentos de 60 segundos de luminancia

Usar los eventos de `merged_events` para identificar los bloques de `video_luminance`. Cropear con `raw.copy().crop(tmin=onset, tmax=onset+duration)`. Los videos de luminancia experimentales estan mapeados en `config_luminance.py` (`LUMINANCE_CSV_MAP`: videos 3, 7, 9, 12) y se identifican por `stim_id = 100 + video_id`.

#### 3.3 Graficar segmentos crudos

Generar graficos de las senales EEG crudas (32 canales) durante los bloques de luminancia:
- Series temporales de canales occipitales (O1, O2) donde alfa es mas prominente
- PSD (Power Spectral Density) del segmento para verificar si hay un pico en 8-13 Hz
- Topografia de la potencia alfa si es posible

#### 3.4 Comparar datos crudos vs. preprocesados

Cargar el mismo segmento temporal de los datos preprocesados (`data/derivatives/campeones_preproc/sub-{ID}/ses-vr/eeg/*_desc-preproc_eeg.vhdr`) y comparar visualmente:
- La oscilacion alfa esta presente en los datos crudos? -> Senal genuina o artefacto del registro
- Aparece solo despues del preprocesamiento? -> Artefacto del pipeline
- La fase es consistente entre trials? -> Posible artefacto de filtrado (ringing)

Referencia existente: `scripts/diagnostics/compare_channels.py`.

---
---

## Tarea 4: Escalado del Analisis a Multiples Participantes

### Motivacion

Todo el analisis esta limitado al Sujeto 27. Los scripts de modelado (`scripts/modeling/01-20*.py`) tienen hardcodeado `SUBJECT = '27'` en `config_luminance.py`. Con solo un sujeto y ~20-30 epocas por condicion, la potencia estadistica es insuficiente.

El repositorio tiene datos de eventos para 18 sujetos en `data/derivatives/events/`, pero solo 3 sujetos tienen datos preprocesados en `data/derivatives/campeones_preproc/`: sub-27, sub-33, sub-37.

### Sub-tareas

#### 4.1 Refactorizar scripts para aceptar ID de sujeto como parametro

Aceptar `--subject` como argumento de linea de comandos (como ya hace `04_preprocessing_eeg.py`). Referencia: `eeg_tfr.py` ya acepta multiples sujetos (`--sub 14 16 17 18`).

#### 4.2 Implementar bucle sobre participantes

Iterar sobre `data/derivatives/campeones_preproc/` para descubrir sujetos disponibles. Para cada sujeto: descubrir runs, cargar datos, ejecutar pipeline, almacenar resultados.

#### 4.3 Crear estructura de datos para Grand Average

- **ERPs Grand Average**: `mne.grand_average()` con lista de `Evoked` objects
- **TFR Grand Average**: array NumPy `(n_subjects, n_channels, n_freqs, n_times)`
- **Metadatos**: epocas por condicion, epocas rechazadas, canales interpolados

Referencia: `eeg_tfr.py` ya implementa concatenacion cross-sujeto con `all_subs_desc-morlet_tfr.npz`.

---
---

## Dependencias entre Tareas

```
Tarea 3 (QC datos crudos) --> Informa decisiones de Tarea 1
         |
         v
Tarea 1 (Reestructuracion de eventos) --> Tarea 2 (Validacion estadistica)
         |                                          |
         v                                          v
Tarea 4 (Escalado multi-sujeto) <---- Requiere Tareas 1 y 2 estabilizadas
```

Se recomienda comenzar por Tarea 3 en paralelo con Tarea 1. Tarea 2 se implementa una vez que las nuevas epocas esten definidas. Tarea 4 es la ultima.
