# Scripts de Preprocesamiento

Este documento detalla los scripts de preprocesamiento disponibles en el proyecto CAMPEONES, explicando su propósito, uso, entradas y salidas.

## Índice

1. [create_events_tsv.py](#create_events_tsv.py) - Crear archivos events.tsv iniciales
2. [visualize_events.py](#visualize_events.py) - Visualizar y editar manualmente eventos
3. [detect_markers.py](#detect_markers.py) - Detectar marcadores automáticamente
4. [verify_annotations.py](#verify_annotations.py) - Verificar anotaciones guardadas

---

## create_events_tsv.py

Script que construye el archivo events.tsv inicial a partir de las planillas de órdenes. Sigue las fases descritas en el documento W3.txt para crear anotaciones BIDS-compliant.

### Funcionalidad

1. Localiza planillas de órdenes
2. Filtra filas relevantes según la columna `path`
3. Asigna duraciones basadas en el catálogo de estímulos
4. Inicializa anotaciones en formato estandarizado (tipo/id)
5. Exporta a BIDS con metadatos completos

### Uso

```bash
python scripts/preprocessing/create_events_tsv.py
```

El script está configurado para procesar un sujeto específico (por defecto, el sujeto 16). Para procesar otros sujetos, modifica la línea correspondiente en la función `main()`.

### Entrada

- Planillas de órdenes en formato Excel (.xlsx)
  - Ubicación: `data/sourcedata/xdf/sub-XX/order_matrix_XX_Y_blockZ_VR.xlsx`
  - Donde XX es el ID del sujeto, Y es la condición (A o B), y Z es el número de bloque

### Salida

- Archivos events.tsv y events.json en formato BIDS
  - Ubicación: `data/derivatives/events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`
  - El archivo events.tsv incluye columnas:
    - onset: Tiempo de inicio del evento en segundos
    - duration: Duración del evento en segundos
    - trial_type: Tipo de evento (video, video_luminance, fixation, calm, practice)
    - stim_id: Identificador único del estímulo
    - condition: Condición experimental
    - stim_file: Ruta relativa al archivo de estímulo

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

## detect_markers.py

Script para la Fase C del proceso de análisis que detecta automáticamente marcadores en los canales AUDIO y PHOTO.

### Funcionalidad

- Detecta automáticamente marcadores en los canales AUDIO y PHOTO
- Para AUDIO: Utiliza detección basada en amplitud o filtrado en banda (según configuración)
- Para PHOTO: Utiliza análisis de frecuencia para detectar parpadeos visuales de 2 Hz
- Busca picos coincidentes entre ambos canales (marcadores audiovisuales)
- Crea anotaciones a partir de los picos detectados, priorizando coincidencias
- Guarda las anotaciones automáticas en derivatives/auto_events
- Visualiza las señales con las anotaciones automáticas
- Permite editar manualmente las anotaciones y guardarlas (habilitado por defecto)

### Uso

```bash
python scripts/preprocessing/detect_markers.py --subject 16 --session vr --task 02 --run 003 --acq a
```

### Parámetros importantes

- `--subject`: ID del sujeto (requerido)
- `--session`: ID de la sesión (default: 'vr')
- `--task`: ID de la tarea (requerido)
- `--run`: ID del run (requerido)
- `--acq`: Parámetro de adquisición (opcional)
- `--photo-distance`: Distancia mínima entre picos en segundos (default: 25)
- `--audio-threshold`: Factor para el umbral de detección en el canal AUDIO (default: 2.0)
- `--photo-threshold`: Factor para el umbral de detección en el canal PHOTO (default: 1.5)
- `--use-amplitude-detection`: Usar detección basada en amplitud para el canal AUDIO (default: True)
- `--whistle-freq`: Frecuencia del silbido a detectar en Hz (default: 500)
- `--whistle-bandwidth`: Ancho de banda del filtro para detectar silbidos en Hz (default: 50)
- `--whistle-duration`: Duración mínima del silbido en segundos (default: 0.05)
- `--flicker-freq`: Frecuencia del parpadeo visual a detectar en Hz (default: 2)
- `--flicker-bandwidth`: Ancho de banda del filtro para detectar parpadeos en Hz (default: 0.5)
- `--flicker-duration`: Duración mínima del parpadeo en segundos (default: 0.8)
- `--visualize-detection`: Visualizar el proceso de detección de silbidos y parpadeos
- `--compare-manual-auto`: Comparar anotaciones automáticas con manuales (si existen)
- `--enable-manual-edit`: Habilitar edición manual de anotaciones (activado por defecto)
- `--no-manual-edit`: Deshabilitar edición manual de anotaciones
- `--force-save`: Forzar guardado de anotaciones editadas manualmente sin preguntar
- `--manual-save-dir`: Directorio donde guardar eventos editados manualmente (default: edited_events)

### Entrada

- Archivos raw de EEG en formato BIDS
  - Ubicación: `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_eeg.vhdr`

### Salida

- Archivos events.tsv y events.json con marcadores detectados automáticamente
  - Ubicación: `data/derivatives/auto_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-auto_events.tsv`
- Si se realiza edición manual:
  - Ubicación: `data/derivatives/edited_events/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_desc-edited_events.tsv`

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