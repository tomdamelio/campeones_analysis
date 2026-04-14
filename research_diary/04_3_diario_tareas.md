# Diario de Tareas: preprocesamiento de nuevos participantes

**Proyecto:** campeones_analysis

**Fecha:** 2026-04-14

**Supervisores:** Enzo Tagliazucchi, Diego Vidaurre

**Contexto:** Hasta ahora todo el pipeline de decoding (scripts 27, 27b, 38, 39, 40, 41) se desarrolló y validó sobre `sub-27`, que era el único sujeto con preprocesamiento estable. Para avanzar hacia análisis multi-sujeto (permutation test grupal, extensión de MVNN+SVC, comparación con el setup de Yongjie) necesitamos más participantes procesados. Jero pasó una lista de sujetos que, según él, **no deberían presentar problemas** durante el preprocesamiento: **19, 23, 24, 27, 30, 33**. `sub-27` ya está procesado y es el benchmark actual, así que el foco real es **19, 23, 24, 30, 33**.

* * *

## Tareas futuras (continuadas del diario anterior)

- [ ] Extender MVNN+SVC a más sujetos (bloqueada por preprocesamiento)
- [ ] Pasar a prediccion de pantalla verde continua


* * *

## Tarea principal — Preprocesamiento de nuevos participantes

**Objetivo:** Obtener datos preprocesados y listos para decoding (mismo formato que `sub-27`) para los sujetos pasados por Jero.

**Sujetos a procesar:**

| Sujeto | Estado | Notas |
|---|---|---|
| sub-19 | pendiente | Jero: sin problemas esperados |
| sub-23 | pendiente | Jero: sin problemas esperados |
| sub-24 | pendiente | Jero: sin problemas esperados |
| sub-27 | ya procesado | benchmark actual — no re-correr |
| sub-30 | pendiente | Jero: sin problemas esperados |
| sub-33 | pendiente | Jero: sin problemas esperados |

### Subtareas

- [ ] **1.1** Revisar que los raw BrainVision (`.vhdr`/`.vmrk`/`.eeg`) y los TSV de eventos estén disponibles en `data/raw` para los 5 sujetos nuevos.
- [ ] **1.2** Confirmar con `sub-27` los parámetros del pipeline canónico (filtros, referencia, resample, ICA, rejection) para replicarlo idénticamente.
- [ ] **1.3** Correr el pipeline de preprocesamiento para cada sujeto con `micromamba run -n campeones python -m src.campeones_analysis.<módulo>`.
- [ ] **1.4** Verificar outputs por sujeto: `raw` filtrado, épocas photo (CHANGE/NO_CHANGE), épocas luminancia 3-clases, e índices de runs.
- [ ] **1.5** Sanity check mínimo por sujeto: ERP occipital CHANGE vs NO_CHANGE (script 25) para confirmar que la señal está presente antes de pasar a decoding.
- [ ] **1.6** Registrar cualquier problema encontrado (canales ruidosos, runs corruptos, desalineación de eventos, etc.) en una tabla por sujeto al final de este diario.

### Criterio de éxito

Cada sujeto nuevo debe quedar en el mismo estado que `sub-27`: listo para ser input directo de los scripts 27b, 38, 39, 40 y 41 sin modificaciones.

* * *

## Sesión de hoy

_Pendiente de iniciar._

* * *

## Problemas encontrados por sujeto

_A completar durante la ejecución._

| Sujeto | Problema | Resolución |
|---|---|---|
| | | |
