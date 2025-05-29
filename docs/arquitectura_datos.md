
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
