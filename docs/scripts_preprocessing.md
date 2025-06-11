# Scripts de Preprocesamiento

Este documento detalla los scripts de preprocesamiento disponibles en el proyecto CAMPEONES, explicando su propósito, uso, entradas y salidas.

## Índice

1. [get_video_durations.py](#get_video_durations.py) - Obtener duraciones precisas de videos
2. [create_events_tsv.py](#create_events_tsv.py) - Crear archivos events.tsv iniciales
3. [detect_markers.py](#detect_markers.py) - Detectar marcadores automáticamente
   
4. [visualize_events.py](#visualize_events.py) - Visualizar y editar manualmente eventos
5. [verify_annotations.py](#verify_annotations.py) - Verificar anotaciones guardadas

## Pipeline de Preprocesamiento

El flujo de trabajo recomendado para el preprocesamiento de datos es el siguiente:

1. Obtener las duraciones precisas de los videos con `get_video_durations.py`
2. Crear los archivos de eventos iniciales con `create_events_tsv.py` (usa las duraciones obtenidas en el paso anterior)
3. Detectar marcadores y fusionar anotaciones con `detect_markers.py`
4. (Opcional) Verificar las anotaciones con `verify_annotations.py`

---

## get_video_durations.py

Script que obtiene la duración precisa de los videos en `data/raw/stimuli` utilizando OpenCV para un cálculo exacto basado en el número de frames y framerate.

### Funcionalidad

1. Escanea todos los archivos de video (.mp4) en `data/raw/stimuli`
2. Calcula la duración precisa de cada video con una precisión de 3 decimales
3. Guarda los resultados en un archivo CSV

### Uso

```bash
python scripts/preprocessing/get_video_durations.py
```

### Parámetros

- `--output`: Ruta personalizada para guardar el CSV (opcional)

### Entrada

- Archivos de video en formato MP4
  - Ubicación: `data/raw/stimuli/*.mp4`

### Salida

- Archivo CSV con duraciones precisas de videos
  - Ubicación: `data/raw/stimuli/video_durations.csv`
  - Columnas: 
    - `filename`: Nombre del archivo de video
    - `duration`: Duración en segundos con 3 decimales de precisión

---

## create_events_tsv.py

Script que construye el archivo events.tsv inicial a partir de las planillas de órdenes y las duraciones precisas de los videos. Sigue las fases descritas en el documento W3.txt para crear anotaciones BIDS-compliant.

### Funcionalidad

1. Localiza planillas de órdenes
2. Filtra filas relevantes según la columna `path`
3. Asigna duraciones basadas en el archivo CSV de duraciones generado por `get_video_durations.py`
4. Inicializa anotaciones en formato estandarizado (tipo/id)
5. Exporta a BIDS con metadatos completos

### Uso

Para procesar un sujeto, tarea y condición específicos:

```bash
python scripts/preprocessing/create_events_tsv.py --subjects 16 --task 02 --acq a
```

Para procesar múltiples sujetos con todas sus runs:

```bash
python scripts/preprocessing/create_events_tsv.py --subjects 16 17 18 --all-runs
```

### Parámetros

- `--subjects`: Lista de IDs de sujetos a procesar (ej: 16 17 18)
- `--session`: ID de la sesión (default: 'vr')
- `--task`: ID de la tarea específica a procesar (ej: '01')
- `--acq`: Parámetro de adquisición (ej: 'a')
- `--run`: ID del run específico a procesar (ej: '003')
- `--all-runs`: Procesar todas las runs disponibles para cada sujeto

### Entrada

- Planillas de órdenes en formato Excel (.xlsx)
  - Ubicación: `data/sourcedata/xdf/sub-XX/order_matrix_XX_Y_blockZ_VR.xlsx`
  - Donde XX es el ID del sujeto, Y es la condición (A o B), y Z es el número de bloque
- Archivo CSV con duraciones de videos
  - Ubicación: `data/raw/stimuli/video_durations.csv`

### Salida

- Archivos events.tsv y events.json en formato BIDS
  - Ubicación: `data/derivatives/events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`
  - El archivo events.tsv incluye columnas:
    - onset: Tiempo de inicio del evento en segundos
    - duration: Duración del evento en segundos (basada en el archivo de duraciones)
    - trial_type: Tipo de evento (video, video_luminance, fixation, calm, practice)
    - stim_id: Identificador único del estímulo
    - condition: Condición experimental
    - stim_file: Ruta relativa al archivo de estímulo

---

## detect_markers.py

Script para la Fase C del proceso de análisis que detecta automáticamente marcadores en los canales AUDIO y PHOTO, y permite fusionar estas anotaciones con los eventos existentes.

### Funcionalidad

- Detecta automáticamente marcadores en los canales AUDIO y PHOTO
- Para AUDIO: Utiliza detección basada en amplitud o filtrado en banda (según configuración)
- Para PHOTO: Utiliza análisis de frecuencia para detectar parpadeos visuales de 2 Hz
- Busca picos coincidentes entre ambos canales (marcadores audiovisuales)
- Crea anotaciones a partir de los picos detectados, priorizando coincidencias
- Carga y muestra los eventos existentes en orden inverso
- Visualiza las señales con las anotaciones automáticas
- Permite editar manualmente las anotaciones
- Fusiona los eventos existentes con las nuevas anotaciones, actualizando onsets y duraciones
- Guarda los eventos fusionados en derivatives/merged_events

### Uso

```bash
python scripts/preprocessing/detect_markers.py --subject 16 --session vr --task 02 --run 003 --acq a
```

Por defecto, el script:
1. Cargará los eventos existentes de `data/derivatives/events`
2. Detectará marcadores en los canales AUDIO y PHOTO
3. Mostrará los eventos existentes en orden inverso
4. Visualizará un plot interactivo para editar manualmente las anotaciones
5. Fusionará los eventos originales con las anotaciones nuevas/editadas
6. Guardará el resultado en `data/derivatives/merged_events`

### Parámetros importantes

- `--subject`: ID del sujeto (requerido)
- `--session`: ID de la sesión (default: 'vr')
- `--task`: ID de la tarea (requerido)
- `--run`: ID del run (requerido)
- `--acq`: Parámetro de adquisición (opcional)
- `--photo-distance`: Distancia mínima entre picos en segundos (default: 25)
- `--audio-threshold`: Factor para el umbral de detección en el canal AUDIO (default: 2.0)
- `--photo-threshold`: Factor para el umbral de detección en el canal PHOTO (default: 1.5)
- `--no-manual-edit`: Deshabilitar edición manual de anotaciones
- `--force-save`: Forzar guardado de anotaciones editadas manualmente sin preguntar
- `--no-merge-events`: No fusionar los eventos existentes con las nuevas anotaciones
- `--merged-save-dir`: Directorio donde guardar eventos fusionados (default: merged_events)

### Entrada

- Archivos raw de EEG en formato BIDS
  - Ubicación: `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_eeg.vhdr`
- Archivos events.tsv generados por create_events_tsv.py
  - Ubicación: `data/derivatives/events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`

### Salida

- Archivos events.tsv y events.json con marcadores detectados automáticamente
  - Ubicación: `data/derivatives/auto_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-autoann_events.tsv`
- Si se realiza edición manual:
  - Ubicación: `data/derivatives/edited_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-edited_events.tsv`
- Eventos fusionados (combinando eventos originales con anotaciones nuevas):
  - Ubicación: `data/derivatives/merged_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-merged_events.tsv`

---

## visualize_events.py

Script que implementa la Fase B del proceso de análisis, permitiendo visualizar y editar manualmente eventos.

### Funcionalidad

- Carga los eventos generados en Fase A
- Visualiza canales AUDIO/PHOTO/joystick_x junto con anotaciones existentes
- Aplica z-score a las señales para facilitar su visualización conjunta
- Permite agregar anotaciones de forma manual
- Guarda las anotaciones alineadas en derivatives/aligned_events con metadatos BIDS

### Uso

```bash
python scripts/preprocessing/visualize_events.py --subject 16 --session vr --task 02 --run 003 --acq a
```

### Parámetros

- `--subject`: ID del sujeto (requerido)
- `--session`: ID de la sesión (default: 'vr')
- `--task`: ID de la tarea (requerido)
- `--run`: ID del run (requerido)
- `--acq`: Parámetro de adquisición (opcional)
- `--force-save`: Forzar guardado de anotaciones sin preguntar
- `--no-zscore`: No aplicar z-score a las señales
- `--save-dir`: Directorio dentro de derivatives donde guardar los eventos (default: 'aligned_events')

### Entrada

- Archivos raw de EEG en formato BIDS
  - Ubicación: `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_eeg.vhdr`
- Archivos events.tsv generados por create_events_tsv.py
  - Ubicación: `data/derivatives/events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`

### Salida

- Archivos events.tsv y events.json actualizados con anotaciones manuales
  - Ubicación: `data/derivatives/aligned_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-withann_events.tsv`

---

## verify_annotations.py

Script para verificar que las anotaciones se guardaron correctamente después de usar visualize_events.py.

### Funcionalidad

1. Carga el archivo de anotaciones guardado
2. Muestra las 3 columnas de anotaciones (onset, duration, trial_type)
3. Visualiza los datos con las anotaciones de forma interactiva

### Uso

```bash
python scripts/preprocessing/verify_annotations.py --subject 16 --session vr --task 02 --run 003 --acq a
```

### Parámetros

- `--subject`: ID del sujeto (requerido)
- `--session`: ID de la sesión (default: 'vr')
- `--task`: ID de la tarea (requerido)
- `--run`: ID del run (requerido)
- `--acq`: Parámetro de adquisición (opcional)
- `--show-all-channels`: Mostrar todos los canales en vez de solo AUDIO/PHOTO/joystick_x
- `--source-dir`: Directorio dentro de derivatives donde buscar los eventos (default: 'aligned_events')

### Entrada

- Archivos raw de EEG en formato BIDS
  - Ubicación: `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_eeg.vhdr`
- Archivos events.tsv generados por otros scripts (visualize_events.py o detect_markers.py)
  - Ubicación: `data/derivatives/aligned_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-withann_events.tsv`

### Salida

- Ninguna (sólo visualización y verificación)

---

## test_eeg_preprocessing.py

Script para probar el preprocesamiento de EEG para un solo participante.

### Funcionalidad

Este script demuestra un pipeline de preprocesamiento de EEG completo:
- Conversión de archivos XDF a formato MNE
- Guardado en formato FIFF y BIDS
- Filtrado
- Detección de canales ruidosos
- Referenciado
- ICA para eliminación de artefactos

### Uso

Este script está diseñado para uso interactivo en entornos como VSCode o Jupyter Notebook con bloques #%%.

Para ejecutarlo completamente:

```bash
python scripts/preprocessing/test_eeg_preprocessing.py
```

### Entrada

- Archivos XDF con datos de EEG y otros streams
  - Ubicación: `data/sub-XX/ses-YY/physio/sub-XX_ses-YY_day-Z_task-ZZ_run-NNN_eeg.xdf`

### Salida

- Archivos procesados en formato FIFF y BIDS
  - Ubicación: `data/derivatives/sub-XX/ses-YY/eeg/`
- Reportes de preprocesamiento
- Logs de preprocesamiento en JSON 