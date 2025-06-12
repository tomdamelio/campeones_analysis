# CAMPEONES Events Derivatives

Este directorio contiene archivos de eventos derivados del proyecto CAMPEONES.

## Contenido

- `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.tsv`: Archivos de eventos con marcas temporales
- `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-ZZ_acq-A_run-NNN_events.json`: Metadatos de los eventos

Los eventos fueron generados a partir de las planillas de órdenes y alineados con los registros EEG.

## Estructura de eventos

Cada archivo events.tsv contiene las siguientes columnas:

| Columna      | Descripción                                           |
|--------------|-------------------------------------------------------|
| onset        | Tiempo de inicio del evento en segundos               |
| duration     | Duración del evento en segundos                       |
| trial_type   | Tipo de evento (video, video_luminance, fixation, etc)|
| stim_id      | Identificador único del estímulo                      |
| condition    | Condición experimental                                |
| stim_file    | Ruta relativa al archivo de estímulo                  |
