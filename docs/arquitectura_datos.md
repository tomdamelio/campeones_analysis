# Arquitectura de datos – **BIDS 1.10-compliant**

> Ver estandares BIDS en https://zenodo.org/records/13754678

---

## 0. Principios generales

| Tema | Criterio adoptado |
|------|------------------|
| **Separación _source_ / _raw_ / _derivatives_** | Los `.xdf` originales residen en `data/sourcedata/`. El **dataset BIDS raw** vive en `data/raw/`. Los resultados se guardan (organizados por pipeline) dentro de `data/derivatives/`. |
| **Formato EEG crudo** | Los `.xdf` se convierten a **BrainVision** (`.vhdr`), uno de los formatos permitidos para EEG crudo. |
| **Trazabilidad** | Todos los derivados llevan sidecar JSON con `GeneratedBy` y `PipelineDescription`, propagando metadatos relevantes (_Inheritance Principle_). |
| **Nomenclatura de archivos derivados** | `<source_entities>[_keyword-<val>]_<suffix>.<ext>`; se usa `desc-<label>` (p. ej. `desc-sync`) para distinguir versiones. |
| **Columnas mínimas en `*_channels.tsv`** | `name`, `type`, `units` (obligatorias) + `sampling_frequency`, `low_cutoff`, `high_cutoff` (recomendadas). |

---

## 1. Fases y salidas

| Fase | Mantener único `Raw` | Objetivo | **Salida BIDS** | Documentar |
|------|----------------------|----------|-----------------|------------|
| **0 Ingesta** | ✅ | Leer `.xdf`, remapear canales y convertir a BrainVision en **`data/raw/`** | `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-<task>_run-<run>_eeg.vhdr` + `channels.tsv` | Tabla de mapeo & unidades |
| **1 Sincronía + resample** | ✅ | Re-muestreo 250 Hz y detección foto/audio | `data/derivatives/preproc/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-<task>_run-<run>_desc-sync_proc-eeg.fif` + `events.tsv` | Drift, Δlatencia |
| **2 Limpieza** | ✅ | Filtros, ICA, artefactos | `…_desc-clean_proc-eeg.fif` + `desc-clean_ica.json` | Bandas, ICA |
| **3 Segmentación (vídeo-época)** | ➡️ (`Epochs`) | Épocas 1–4 min con `events.tsv` | `…_desc-epoch<idx>_proc-eeg.fif` | Duración, % rechazo |
| **4 Features** | 🔀 | Métricas por modalidad | `data/derivatives/features/<mod>/sub-XX_ses-YY_task-<task>_run-<run>_desc-epoch<idx>_<mod>.tsv` | Parámetros + versión librería |
| **5 Fusión multimodal** | ❌ | _Merge_ tidy para estadísticas / ML | `data/derivatives/features/multimodal/sub-XX_ses-YY_task-<task>_desc-combined.tsv` | Commit hash, condición |

---

## 2. Ejemplo de conversión `.xdf` → BrainVision

```python
import mne
from mne_bids import write_raw_bids, BIDSPath

# 1 Leer archivo .xdf
raw = mne.io.read_raw_xdf("data/sourcedata/xdf/sub-12_task-vr_run-01.xdf", preload=True)

# 2 Mapear tipos de canal
raw.set_channel_types({
    "Fz":  "eeg",
    "GSR": "gsr",
    "ECG": "ecg",
    # … otros canales
})

# 3 Definir ruta BIDS y escribir BrainVision
bids_path = BIDSPath(
    subject="12", session="vr", task="vr", run="01",
    root="data/raw", datatype="eeg", suffix="eeg", extension=".vhdr"
)
write_raw_bids(raw, bids_path, overwrite=True)
```

`channels.tsv` (ejemplo):

| name | type | units | sampling_frequency | low_cutoff | high_cutoff |
| ---- | ---- | ----- | ----------------- | ---------- | ----------- |
| Fz   | EEG  | µV    | 500               | 0.1        | 48          |
| GSR  | GSR  | µS    | 32                | n/a        | 10          |
| …    | …    | …     | …                 | …          | …           |

---

## 3. Árbol de carpetas propuesto

```
project/
├── docs/arquitectura_datos.md
├── data/
│   ├── sourcedata/
│   │   └── xdf/                       # .xdf originales
│   ├── raw/                           # **dataset BIDS raw**
│   │   ├── dataset_description.json
│   │   └── sub-XX/ses-YY/eeg/…
│   └── derivatives/
│       ├── preproc/                   # sync, clean, epochs
│       ├── features/
│       │   ├── eeg/ ecg/ eda/ resp/ acc/ behav/
│       │   └── multimodal/
│       └── … otros pipelines
└── src/                              # scripts de procesamiento
```

---

## 4. Conversión automática con `read_xdf.py`

El script `read_xdf.py` ubicado en `src/campeones_analysis/physio/` permite convertir archivos `.xdf` a formato BIDS de manera automática. Este script detecta los streams EEG y joystick, realiza el remapeo de canales, y guarda los datos en formato BrainVision siguiendo la estructura BIDS.

### Uso básico

Desde la carpeta raíz del proyecto:

```bash
# Procesar un sujeto específico (procesa todas sus sesiones y tareas)
python -m src.campeones_analysis.physio.read_xdf --subject 18

# Procesar todos los sujetos disponibles
python -m src.campeones_analysis.physio.read_xdf
```

### Flags disponibles

| Flag | Descripción | Ejemplo |
|------|-------------|---------|
| `--subject` | ID del sujeto a procesar | `--subject 18` |
| `--session` | Sesión específica a procesar | `--session VR` |
| `--task` | Tarea específica a procesar | `--task 01` |
| `--run` | Run específico a procesar | `--run 001` |
| `--acq` | Parámetro de adquisición (default: "a") | `--acq a` |

### Ejemplos de uso avanzado

```bash
# Procesar una sesión específica de un sujeto
python -m src.campeones_analysis.physio.read_xdf --subject 18 --session VR

# Procesar una tarea y run específicos
python -m src.campeones_analysis.physio.read_xdf --subject 18 --session VR --task 01 --run 001

# Procesar con parámetro de adquisición específico
python -m src.campeones_analysis.physio.read_xdf --subject 18 --acq b
```

### Estructura de archivos generados

El script genera los siguientes archivos en formato BIDS:

1. **Archivos de datos EEG** (formato BrainVision):
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-RR_eeg.vhdr`
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-RR_eeg.vmrk`
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-RR_eeg.eeg`

2. **Metadatos**:
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-RR_channels.tsv`
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_acq-A_space-CapTrak_electrodes.tsv`
   - `data/raw/sub-XX/ses-YY/eeg/sub-XX_ses-YY_acq-A_space-CapTrak_coordsystem.json`

3. **Archivos de participantes**:
   - `data/raw/participants.tsv`
   - `data/raw/participants.json`
   - `data/raw/dataset_description.json`

### Notas importantes

- El script automáticamente detecta y procesa los streams EEG y joystick en los archivos XDF.
- Los datos se remuestrean a 250 Hz para estandarización.
- Se aplica automáticamente el montaje de electrodos desde el archivo `data/BC-32.bvef`.
- Los canales del joystick se agregan como canales adicionales en el archivo BrainVision.
- Los eventos se extraen y se guardan en formato BIDS (`events.tsv` y `events.json`).
