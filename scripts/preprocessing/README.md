# Scripts de Preprocesamiento

Este directorio contiene los scripts principales del pipeline de preprocesamiento de datos CAMPEONES.

## 📋 Orden de Ejecución

Los scripts están numerados para indicar el orden en que deben ejecutarse:

### Paso 0: Conversión XDF → BIDS (OBLIGATORIO)
**Script**: `src/campeones_analysis/physio/read_xdf.py`
```bash
micromamba run -n campeones python -m src.campeones_analysis.physio.read_xdf --subject XX
```
Convierte archivos `.xdf` originales a formato BrainVision BIDS.

---

### Paso 1: Duraciones de Videos (OPCIONAL)
**Script**: `01_get_video_durations.py`
```bash
micromamba run -n campeones python scripts/preprocessing/01_get_video_durations.py
```
Calcula duraciones precisas de estímulos visuales. Solo ejecutar si no existe `stimuli/video_durations.csv`.

---

### Paso 2: Eventos Iniciales
**Script**: `02_create_events_tsv.py`
```bash
micromamba run -n campeones python scripts/preprocessing/02_create_events_tsv.py --subjects XX --all-runs
```
Genera archivos `events.tsv` desde planillas de orden experimental.

**NOTA**: El parámetro `--run` ya no es necesario. El script detecta automáticamente el run del archivo EEG.

---

### Paso 3: Detección de Marcadores
**Script**: `03_detect_markers.py`
```bash
# Procesar TODAS las combinaciones para un sujeto (comportamiento por defecto)
micromamba run -n campeones python scripts/preprocessing/03_detect_markers.py --subject XX

# Filtrar por tarea específica
micromamba run -n campeones python scripts/preprocessing/03_detect_markers.py --subject XX --task YY

# Especificar combinación exacta
micromamba run -n campeones python scripts/preprocessing/03_detect_markers.py --subject XX --task YY --acq a
```
Detecta marcadores audiovisuales y fusiona con eventos iniciales.

**NOTA**: Si solo especificas `--subject`, el script procesará automáticamente todas las combinaciones de task/acq disponibles.

---

### Paso 4: Preprocesamiento EEG
**Script**: `04_preprocessing_eeg.py`

**Script**: `04_preprocessing_eeg.py`

Ahora soporta argumentos por línea de comandos, por lo que no es necesario editar el script.

**Uso básico (sin pausas, genera reporte HTML):**
```bash
micromamba run -n campeones python scripts/preprocessing/04_preprocessing_eeg.py --subject 18 --session vr --task 04 --run 005
```

**Modo interactivo (pausa para inspección visual de tramas y componentes ICA):**
```bash
micromamba run -n campeones python scripts/preprocessing/04_preprocessing_eeg.py --subject 18 --session vr --task 04 --run 005 --interactive
```

Aplica filtrado, ICA, y segmentación en épocas.

---

### Paso 5: Extracción de Features
**Script**: `05_physiology_features.py`
```bash
micromamba run -n campeones python scripts/preprocessing/05_physiology_features.py --subject XX
```
Extrae características de señales EEG y fisiológicas.

---

## 🔧 Scripts Auxiliares

Estos scripts no forman parte del pipeline principal pero son útiles para tareas específicas:

- `verify_annotations.py` - Verificar anotaciones guardadas
- `visualize_events.py` - Visualizar eventos interactivamente
- `test_event_id_mapping.py` - Verificar mapeo de event IDs
- `test_eeg_preprocessing.py` - Test de preprocesamiento EEG
- `create_pipeline_description.py` - Crear descripción del pipeline
- `analyze_tfr_data.py` - Análisis de datos tiempo-frecuencia
- `analyze_multi_run_psd.py` - Análisis de PSD multi-run

---

## 📚 Documentación Completa

Para documentación detallada, ver [`docs/README.md`](../../docs/README.md).
