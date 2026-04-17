# Diario de Tareas: preprocesamiento de nuevos participantes

**Proyecto:** campeones_analysis

**Fecha:** 2026-04-14

**Supervisores:** Enzo Tagliazucchi, Diego Vidaurre

**Contexto:** Hasta ahora todo el pipeline de decoding (scripts 27, 27b, 38, 39, 40, 41) se desarrolló y validó sobre `sub-27`, que era el único sujeto con preprocesamiento estable. Para avanzar hacia análisis multi-sujeto (permutation test grupal, extensión de MVNN+SVC, comparación con el setup de Yongjie) necesitamos más participantes procesados. Jero pasó una lista de sujetos que, según él, **no deberían presentar problemas** durante el preprocesamiento: **19, 23, 24, 27, 30, 33**. `sub-27` ya está procesado y es el benchmark actual, así que el foco real es **19, 23, 24, 30, 33**.

* * *

## Tareas futuras (continuadas del diario anterior)

- [x] Extender MVNN+SVC a más sujetos — completado para sub-23, 24, 33 (ver sesión 2026-04-16)
- [ ] Pasar a prediccion de pantalla verde continua
- [ ] **Aplicar Surface Laplacian / CSD como paso downstream** (no en preproc) cuando pasemos a análisis de conectividad o relaciones entre canales. Razón: la conducción volumétrica hace que fuentes cerebrales localizadas se vean difundidas por todo el scalp, produciendo conectividad espuria a delay cero. El CSD (segunda derivada espacial vía spline esférico, `mne.preprocessing.compute_current_source_density`) es reference-independent, focaliza topografías y reduce conducción volumétrica. PCA **no** lo resuelve (no conoce geometría de electrodos). Dejarlo fuera del preproc porque cambia las unidades (V → V/m²) y rompe comparabilidad con ERPs/topoplots estándar e ICLabel. Aplicar como variante dentro del script de conectividad/decoding. Referencia: Cohen 2014 §7.7, §22; ver `docs/04_preprocessing_eeg_review.md` R-15 para detalle.


* * *

## Tarea principal — Preprocesamiento de nuevos participantes

**Objetivo:** Obtener datos preprocesados y listos para decoding (mismo formato que `sub-27`) para los sujetos pasados por Jero.

**Sujetos a procesar:**

| Sujeto | Estado | Notas |
|---|---|---|
| sub-19 | preprocesado (7/8 runs) | 1 run crasheó en ICA (task-04 acq-a). Mean bads=4.6. Tier 3. |
| sub-23 | **preprocesado (8/8 runs)** | Data más limpia. Mean bads=1.4. **Tier 1 — listo para decoding.** |
| sub-24 | **preprocesado (8/8 runs)** | Buena calidad. Mean bads=2.8. **Tier 1 — listo para decoding.** |
| sub-27 | ya procesado | benchmark actual (250 Hz, 7 runs) — no re-correr |
| sub-30 | preprocesado (8/8 runs) | Frontal degradado (mean bads=5.8). Tier 2. |
| sub-33 | **preprocesado (8/8 runs)** | Muy buena calidad. Mean bads=2.2. **Tier 1 — listo para decoding.** |

### Subtareas

- [x] **1.1** Revisar que los raw BrainVision (`.vhdr`/`.vmrk`/`.eeg`) y los TSV de eventos estén disponibles en `data/raw` para los 5 sujetos nuevos. → Confirmado. read_xdf completó para los 5 sujetos. Todos a 500 Hz.
- [x] **1.2** Confirmar con `sub-27` los parámetros del pipeline canónico (filtros, referencia, resample, ICA, rejection) para replicarlo idénticamente. → Pipeline idéntico (R-12 two-copy, R-7 Var A, R-18 picard-o). Único cambio: R-18 agregado en esta sesión.
- [ ] **1.2.bis ⚠ VERIFICACIÓN FÍSICA DEL MONTAGE EOG.** (Pendiente — requiere acceso físico al equipo)
- [x] **1.3** Correr el pipeline de preprocesamiento para cada sujeto. → Batch completado: 38/40 OK. QA report generado.
- [x] **1.4** Verificar outputs por sujeto. → Confirmados preproc .vhdr + events.tsv para todos los runs exitosos. Ver tabla en sección QA.
- [ ] **1.5** Sanity check mínimo por sujeto: ERP occipital. → **PENDIENTE** — programar scripts 38a-d para sub-23, 24, 33 después de generar photo_events.
- [x] **1.6** Registrar problemas encontrados. → Documentado en tabla "Problemas encontrados por sujeto".

### Criterio de éxito

Cada sujeto nuevo debe quedar en el mismo estado que `sub-27`: listo para ser input directo de los scripts 27b, 38, 39, 40 y 41 sin modificaciones.

* * *

## Sesión 2026-04-15/16 — Preprocesamiento batch + QA

### Cambios al pipeline antes del batch

**R-18 (picard-o):** Se agregó `fit_params=dict(ortho=False, extended=True)` al constructor de ICA en `04_preprocessing_eeg.py`. Esto configura Picard como "picard-o" (Ablin et al. 2018, IEEE TSP), matemáticamente equivalente a extended infomax pero ~10× más rápido. ICLabel fue entrenado con extended infomax (Pion-Tonachini et al. 2019, NeuroImage), así que sin estos flags la calibración de probabilidades que R-7 Variant A usa (ICLABEL_THRESHOLD=0.85, BRAIN_FLOOR=0.30) estaba ligeramente fuera de distribución.

### Ejecución

1. **read_xdf batch** (scripts/preprocessing/run_read_xdf_batch.py): Convertimos XDF → BIDS para sub-19, 23, 24, 30, 33. Sub-19, 30, 33 requerieron `--force` porque tenían BIDS viejos a 250 Hz de una versión anterior del pipeline. Tras re-correr, todos confirmados a **500 Hz** (SI ≈ 2000 µs).

2. **Preprocessing batch** (scripts/preprocessing/run_batch_preprocessing.py): 40 runs totales (5 sujetos × 4 tasks × 2 acq), modo `--auto --force`, timeout 60 min por run.

3. **QA automático** (scripts/preprocessing/qa_batch_analysis.py): Reporte completo en `data/derivatives/campeones_preproc/batch_logs/qa_report_20260416_013819.md`.

### Resultado batch: 38/40 OK, 2 failures (ambos sub-19)

| Failure | Tiempo | Causa | Estado output |
|---|---|---|---|
| sub-19 task-02 acq-b run-007 | 9770s (timeout) | Timeout del subprocess, pero el preprocesamiento **SÍ completó** (archivos preproc existen, log JSON tiene `final_preprocessing_completed: true`) | **Recuperable** — data usable |
| sub-19 task-04 acq-a run-005 | 12s | `ValueError: attempt to get argmax of an empty sequence` en `ica.fit()` — PCA falló por datos insuficientes tras exclusión de bad segments + bad channels | **No recuperable** sin intervención — solo existen archivos `desc-filtered` |

### QA: Calidad por sujeto (ordenado de mejor a peor)

| Sujeto | Runs OK | Mean bads | Mean ICA excl | Mean vetos | ICA components | Tier |
|---|---|---|---|---|---|---|
| **sub-23** | **8/8** | **1.4** | 11.6 | 5.9 | 28-31 | **Tier 1** |
| **sub-33** | **8/8** | **2.2** | 12.1 | 3.9 | 28-30 | **Tier 1** |
| **sub-24** | **8/8** | **2.8** | 9.2 | 6.4 | 26-30 | **Tier 1** |
| sub-30 | 8/8 | 5.8 | 9.6 | 3.4 | 24-26 | Tier 2 (frontal degradado) |
| sub-19 | 7/8 | 4.6 | 11.7 | 6.0 | 24-29 | Tier 3 (incompleto) |

### Hallazgos principales

**FC1 sistemáticamente bad** en 29/39 runs (74%): sub-30 8/8, sub-33 8/8, sub-24 7/8, sub-19 6/7, sub-23 1/8. Probable problema de hardware/impedancia del electrodo. Siempre interpolado desde vecinos. Impacto en decoding: mínimo (FC1 no es canal visual crítico), pero documentar como limitación.

**Sub-30 — problema frontal consistente:** Fp1, Fp2, F3, Fz, FC1 bad en **todos** los runs (5-7 bads/run). Exclusivamente por criterio `correlation` de PyPREP. Canales occipitales/parietales intactos. Usable si el decoding se basa en actividad visual/occipital (que es nuestro caso).

**PyPREP criteria breakdown (batch completo):** correlation=87 (dominante), ransac=21, hf_noise=16, deviation=12. RANSAC funciona — el sanity test con sub-23 sola no lo había mostrado (0 ransac flags con 1 sujeto).

**R-7 brain-floor vetos:** mean=5.1/run. Confirma que el veto por `brain_prob > 0.30` está haciendo trabajo real — impide excluir componentes que tienen contenido brain significativo.

**Jitter:** Todos los runs < 1 ms (max=0.991 ms). Sin problemas.

**Sampling rate:** Todos 500 Hz. Los scripts de decoding (27b, 34, 41) leen `sfreq` dinámicamente — no hay hardcoding de 250 Hz. Script 41 resamplea a 100 Hz independientemente del sfreq original.

### Outliers flaggeados

12 runs con ≥5 bad channels: sub-30 (todos 8 runs) + sub-19 (2 runs: 02_run-007, 03_run-008) + sub-24 (1 run: 01_run-002).

### Verificación de outputs

| Sujeto | preproc .vhdr | preproc events.tsv | merged_events | photo_events |
|---|---|---|---|---|
| sub-23 | 8/8 | 8/8 | 8/8 | **8/8** |
| sub-24 | 8/8 | 8/8 | 8/8 | **8/8** |
| sub-33 | 8/8 | 8/8 | 8/8 | **8/8** |
| sub-30 | 8/8 | 8/8 | 8/8 | PENDIENTE |
| sub-19 | 7/8 (falta task-04_acq-a) | 7/8 | 8/8 | PENDIENTE |

* * *

## Problemas encontrados por sujeto

| Sujeto | Problema | Resolución |
|---|---|---|
| sub-19 | task-04 acq-a run-005: ICA crash (PCA empty sequence) | Pendiente. Opciones: reducir n_components, excluir run, o pre-marcar más bad channels |
| sub-19 | task-02 acq-b run-007: timeout batch (9770s) | **Resuelto** — output completo existe, el timeout fue post-procesamiento |
| sub-30 | 5-7 bad channels frontales en todos los runs | Aceptado — interpolación automática OK, canales occipitales intactos |
| Todos | FC1 bad en 74% de runs | Aceptado — hardware/impedancia, siempre interpolado |

* * *

## Decisión: sujetos para decoding

**Tier 1 (avanzar directo a decoding):** sub-23, sub-33, sub-24 — datos completos (8/8 runs), calidad alta.

**Tier 2 (usable con caveats):** sub-30 — 8/8 runs pero frontal degradado. Incluir solo si confirmamos que el decoding no depende de features frontales (probable, dado que MVNN+SVC occ_temp fue el mejor pipeline en sub-27).

**Tier 3 (requiere trabajo):** sub-19 — 7/8 runs. Incluir después de decidir qué hacer con el run faltante.

**Comparación con sub-27:** sub-27 tiene 7 runs (le falta task-02 acq-b). Los nuevos Tier 1 tienen 8 runs, lo que da 1 fold más de LORO-CV y más trials.

* * *

## Plan de decoding — Extensión a sub-23, sub-33, sub-24

### Objetivo

Replicar el pipeline completo de decoding desarrollado sobre sub-27 (diarios 04_1 y 04_2) en los 3 sujetos Tier 1. Comparar resultados por sujeto para evaluar la generalización de los hallazgos.

### Diferencia técnica: 500 Hz vs 250 Hz (sub-27)

Sub-27 fue preprocesado a 250 Hz. Los nuevos sujetos están a 500 Hz. **Impacto por script:**

| Script | Impacto de 500 Hz | Acción necesaria |
|---|---|---|
| 22 (photo_events) | Ninguno — trabaja con onsets en segundos, no samples | Solo agregar RUNS_CONFIG |
| 27b (pre/post) | Mínimo — lee sfreq dinámicamente, Welch adapta resolución | Solo agregar RUNS_CONFIG |
| 34 (4-class) | Mínimo — window samples calculados con `int(round(WIN_SIZE_S * sfreq))` | Solo agregar RUNS_CONFIG |
| 41 (MVNN+SVC) | **Ninguno** — resamplea a 100 Hz explícitamente | Solo agregar RUNS_CONFIG |

**Nota sobre autocorrelación (script 27b):** Los lags `LOG_LAGS = [1,2,3,4,7,12,20,25]` y `N_AUTOCORR_LAGS = 25` están calibrados para 250 Hz (lag 25 = 100ms = 1 ciclo alpha). A 500 Hz, lag 25 = 50ms, no cubre alpha completo. Opciones: (a) duplicar lags a [2,4,6,8,14,24,40,50] para mantener mismos tiempos, (b) ignorar — autocorrelación no es el feature principal. **Decisión:** ignorar por ahora — autocorrelación alcanzó techo de ~61.5% en sub-27, no es prioritaria. Si se necesita, ajustar lags post-hoc.

### Pasos concretos

#### Paso 0 — ✅ Agregar RUNS_CONFIG a scripts (todos los scripts comparten la misma estructura)

```python
# Sub-23, 24, 33: 8 runs = 4 tasks × 2 acq (a/b)
# Patrón uniforme: acq-a = runs 002-005, acq-b = runs 006-009
"23": [
    {"run": "002", "acq": "a", "task": "01"},
    {"run": "003", "acq": "a", "task": "02"},
    {"run": "004", "acq": "a", "task": "03"},
    {"run": "005", "acq": "a", "task": "04"},
    {"run": "006", "acq": "b", "task": "01"},
    {"run": "007", "acq": "b", "task": "02"},
    {"run": "008", "acq": "b", "task": "03"},
    {"run": "009", "acq": "b", "task": "04"},
],
# Idem para "24" y "33"
```

Scripts a actualizar: `22_generate_photo_events.py`, `27b_decoding_pre_vs_post.py`, `34_decoding_4class.py`, `41_decoding_mvnn_svc.py`.

#### Paso 1 — ✅ Generar photo_events (script 22)
```
micromamba run -n campeones python -m scripts.validation.22_generate_photo_events --subject 23
micromamba run -n campeones python -m scripts.validation.22_generate_photo_events --subject 24
micromamba run -n campeones python -m scripts.validation.22_generate_photo_events --subject 33
```

#### Paso 2 — ✅ MVNN+SVC decoding (script 41) — pipeline principal
```
micromamba run -n campeones python -m scripts.validation.41_decoding_mvnn_svc --subject 23
micromamba run -n campeones python -m scripts.validation.41_decoding_mvnn_svc --subject 24
micromamba run -n campeones python -m scripts.validation.41_decoding_mvnn_svc --subject 33
```
Esto corre MVNN+SVC con ambas variantes de canales (all 32ch + occ_temp 8ch) + permutaciones.

#### Paso 3 — ✅ Feature comparison (script 27b) — comparación con sub-27
```
micromamba run -n campeones python -m scripts.validation.27b_decoding_pre_vs_post --subject 23
micromamba run -n campeones python -m scripts.validation.27b_decoding_pre_vs_post --subject 24
micromamba run -n campeones python -m scripts.validation.27b_decoding_pre_vs_post --subject 33
```
Esto corre bandpower_welch, raw_pca, tde_cov (y variantes) para comparar con los resultados de sub-27.

#### Paso 4 — ✅ 4-class temporal (script 34) — bandpower_welch + raw_pca (tde_cov OOM a 500 Hz)
```
micromamba run -n campeones python -m scripts.validation.34_decoding_4class --subject 23
micromamba run -n campeones python -m scripts.validation.34_decoding_4class --subject 24
micromamba run -n campeones python -m scripts.validation.34_decoding_4class --subject 33
```

### Métricas a comparar (tabla objetivo)

| Sujeto | sfreq | N runs | MVNN occ_temp | MVNN all | Bandpower | raw_pca | tde_cov | 4class best |
|---|---|---|---|---|---|---|---|---|
| sub-27 | 250 | 7 | **77.0%** | 73.0% | — | — | — | 37.4% |
| sub-23 | 500 | 8 | **77.4%** | 73.8% | 74.4% | 73.8% | 65.9% | 36.3% |
| sub-24 | 500 | 8 | 57.7% (ns) | **69.2%** | 50.6% | 64.7% | 48.7% | 31.5% |
| sub-33 | 500 | 8 | **86.3%** | 79.2% | 76.8% | 71.4% | 65.5% | 33.2% |

### Hipótesis previas — evaluación post-hoc

1. **MVNN+SVC occ_temp debería seguir siendo el mejor pipeline** → **PARCIAL.** Confirmado para sub-23 y sub-33 (occ_temp > all), pero NO para sub-24 (occ_temp no significativo, p=0.10). La fisiología visual es compartida, pero la topografía discriminativa varía entre sujetos.
2. **Accuracies absolutas pueden variar** → **CONFIRMADA.** Rango masivo: 57.7%–86.3% (occ_temp), 69.2%–79.2% (all). Más variabilidad de la esperada.
3. **El ranking de feature sets debería ser estable: MVNN > bandpower > raw_pca > tde_cov** → **PARCIAL.** MVNN es consistentemente el mejor. Para 27b (2-class), bandpower ≈ raw_pca > autocorr > tde_cov (bandpower y raw_pca están más cerca de lo esperado). Para 4-class, raw_pca > bandpower.
4. **500 Hz no debería cambiar los resultados** → **CONFIRMADA para MVNN/27b.** MVNN resamplea a 100 Hz, 27b lee sfreq dinámicamente. Pero 500 Hz causó OOM en 4-class tde_cov (serialización de matrices más grandes en joblib).
5. **8 runs da estimaciones más estables** → **NO CONFIRMADA.** STD entre folds es alta (hasta 17.6% en sub-24 occ_temp). Más folds no garantiza estabilidad si la señal fluctúa entre runs.

* * *

## Sesión 2026-04-16 — Decoding multi-sujeto: MVNN+SVC completado

### Pasos ejecutados

1. **RUNS_CONFIG agregado** a scripts 22, 27b, 34, 41 para sub-23, 24, 33 (8 runs cada uno).
2. **photo_events generados** (script 22) para sub-23, 24, 33 — 8/8 por sujeto.
3. **MVNN+SVC ejecutado** (script 41) para sub-23, 24, 33 — ambos subsets (all 32ch + occ_temp 8ch), con 100 permutaciones cada uno.
4. **27b completado** para sub-23, 24, 33 (6 feature sets cada uno, sin permutaciones).
5. **34 completado** para sub-23, 24, 33 (bandpower_welch + raw_pca; tde_cov OOM a 500 Hz).

### Resultados MVNN+SVC (script 41)

#### Tabla principal

| Sujeto | Canales | Features | Trials | MVNN Acc | p-value | Null mean | Bandpower Acc |
|---|---|---|---|---|---|---|---|
| sub-27 (ref) | all (32) | 3840 | 148 | **73.0%** | N/A | N/A | 61.5% |
| sub-27 (ref) | occ_temp (8) | 960 | 148 | **77.0%** | N/A | N/A | 52.7% |
| sub-23 | all (32) | 3840 | 164 | **73.8%** | 0.0000 | 49.4% | 62.8% |
| sub-23 | occ_temp (8) | 960 | 164 | **77.4%** | 0.0000 | 50.0% | 61.6% |
| sub-24 | all (32) | 3840 | 156 | **69.2%** | 0.0000 | 49.9% | 59.0% |
| sub-24 | occ_temp (8) | 960 | 156 | 57.7% | **0.10 (ns)** | 50.2% | 59.0% |
| sub-33 | all (32) | 3840 | 168 | **79.2%** | 0.0000 | 49.4% | 68.5% |
| sub-33 | occ_temp (8) | 960 | 168 | **86.3%** | 0.0000 | 49.4% | 71.4% |

#### Per-fold accuracies (MVNN, LORO-CV)

| Sujeto | Subset | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | STD |
|---|---|---|---|---|---|---|---|---|---|---|
| sub-23 | all | 79.2 | 80.0 | 85.0 | 60.0 | 78.6 | 81.2 | 43.8 | 75.0 | 13.0% |
| sub-23 | occ_temp | 83.3 | 75.0 | 95.0 | 70.0 | 75.0 | 87.5 | 62.5 | 70.0 | 9.9% |
| sub-24 | all | 70.0 | 70.0 | 70.0 | 65.0 | 70.0 | 81.2 | 62.5 | 66.7 | 5.2% |
| sub-24 | occ_temp | 30.0 | 50.0 | 75.0 | 80.0 | 45.0 | 68.8 | 37.5 | 70.8 | 17.6% |
| sub-33 | all | 85.7 | 85.0 | 62.5 | 70.0 | 92.9 | 81.2 | 81.2 | 66.7 | 9.9% |
| sub-33 | occ_temp | **100** | 95.0 | 62.5 | 60.0 | 92.9 | 87.5 | 93.8 | 87.5 | 14.2% |

### Hallazgos principales MVNN+SVC

**1. Replicación exitosa:** 7 de 8 condiciones (sujeto × subset) significativas a p=0.0000. La decodificación PRE vs POST photo-change es robusta y replicable.

**2. Sub-33 es el mejor sujeto del dataset:** 86.3% con occ_temp (8 canales), con un fold a 100%. Supera a sub-27 por +9.3pp en occ_temp y +6.2pp en all channels.

**3. Sub-23 replica casi exactamente a sub-27:** all=73.8% (vs 73.0%), occ_temp=77.4% (vs 77.0%). Evidencia fuerte de replicabilidad cross-subject.

**4. Sub-24 — disociación occ_temp vs all:** El único sujeto donde occ_temp NO es significativo (57.7%, p=0.10). La señal discriminativa está distribuida en todo el scalp, no concentrada en canales occipito-temporales. Alta variabilidad entre folds (STD=17.6%, rango 30-80%).

**5. MVNN >> Bandpower consistente:** Ventaja promedio de +12.5pp (excluyendo sub-24 occ_temp). Las dinámicas temporales punto-a-punto dominan sobre la potencia espectral.

**6. Patrón occ_temp >= all (3/4 sujetos):** Los 24 canales extra introducen ruido que diluye la señal concentrada en regiones posteriores/temporales. Excepción: sub-24 (topografía atípica).

**7. Variabilidad intra-sujeto:** STD entre folds oscila entre 5.2% (sub-24 all, estable pero bajo) y 17.6% (sub-24 occ_temp, inestable). Incluso los mejores sujetos (sub-33 occ_temp: 86.3% global) tienen folds de 60%. Esto sugiere variabilidad temporal en la calidad de la señal neural entre runs, posiblemente por fatiga, habituación, o cambios en estrategia atencional.

### Resultados 27b — Feature comparison (pre vs post, 500ms, 2-class)

| Feature | sub-23 | sub-24 | sub-33 |
|---|---|---|---|
| bandpower_welch | **74.4%** | 50.6% | **76.8%** |
| raw_pca | **73.8%** | 64.7% | **71.4%** |
| tde_cov (full) | 65.9% | 48.7% | 65.5% |
| tde_cov_diag | 67.1% | 53.2% | 60.1% |
| tde_cov_offdiag | 61.6% | 46.8% | 62.5% |
| autocorr | 68.3% | 49.4% | **73.2%** |

**Nota:** sub-27 no tiene resultados 27b comparables (solo tenía autocorr_log=58.8% de una corrida anterior). La columna "Bandpower" de la tabla objetivo se refiere a script 27b bandpower_welch.

**Observaciones 27b:**
- **Sub-23 y sub-33 muestran el mismo ranking**: bandpower ≈ raw_pca > autocorr > tde_cov > offdiag. Bandpower y raw_pca capturan la señal discriminativa casi igual de bien (~74%).
- **Sub-24 es consistentemente el peor sujeto** — incluso bandpower (50.6%) está en chance. Solo raw_pca (64.7%) muestra señal apreciable, confirmando que la información discriminativa en sub-24 requiere la dimensión temporal completa (no solo potencia espectral).
- **tde_cov siempre inferior a bandpower/raw_pca** — la covarianza temporal de delay embedding captura menos información discriminativa que los features directos. Las variantes diag y offdiag no mejoran sobre la covarianza completa.
- **Autocorr notable en sub-33** (73.2%) — casi alcanza a bandpower (76.8%). A 500 Hz, los lags [1,2,...,25] cubren solo 50ms (medio ciclo alpha), pero para sub-33 parece capturar la señal igualmente.

### Resultados 34 — 4-class temporal decoding

| Feature | sub-27 | sub-23 | sub-24 | sub-33 |
|---|---|---|---|---|
| bandpower_welch | 31.7% | **35.7%** | 27.4% | 31.6% |
| raw_pca | **37.4%** | **36.3%** | **31.5%** | **33.2%** |
| tde_cov | 34.9% | — (OOM) | — (OOM) | — (OOM) |
| **Best** | **37.4%** | **36.3%** | **31.5%** | **33.2%** |

*Chance = 25% (4 clases equiprobables). TDE_cov crasheó por MemoryError en joblib a 500 Hz.*

**Observaciones 4-class:**
- **Todos los sujetos superan chance (25%)**, pero el accuracy absoluto es bajo (27-37%). La clasificación de 4 fases temporales (Baseline/ChangeUp/Luminance/ChangeDown) es intrínsecamente más difícil.
- **raw_pca siempre es el mejor feature** para 4-class, consistente con sub-27.
- **Sub-24 roza chance** con bandpower (27.4%) — confirma la debilidad general de la señal en este sujeto.
- **Sub-23 es el mejor en 4-class** (36.3%), superando ligeramente a sub-27 (37.4% incluye tde_cov que no corrió para los nuevos).
- **Per-class accuracy desbalanceada**: Baseline es la clase más fácil (~38-48%), ChangeDown la segunda, mientras Luminance y ChangeUp son las más difíciles. Esto refleja que los transitorios on/off (VEPs) generan respuestas más distintivas que los estados sostenidos.

### Nota sobre MemoryError en 4-class tde_cov

Script 34 usa `Parallel(n_jobs=-3)` en `run_loro_tde_cov()` para paralelizar folds. A 500 Hz, la serialización de las matrices TDE (32 ch × muchos timepoints) para la inter-process queue de joblib excede la memoria disponible. Esto afecta solo a tde_cov en 4-class (más ventanas que 2-class). Posible fix: cambiar `n_jobs=-3` a `n_jobs=1` para 500 Hz, o reducir el tamaño de los arrays antes de paralelizar. No es prioritario dado que tde_cov no es el mejor feature.

* * *

## Tareas pendientes post-decoding

- [ ] **Investigar sub-19 run-005** — crash ICA. Opciones: reducir n_components, o excluir este run y correr sub-19 con 7/8.
- [ ] **Incorporar sub-30** — correr decoding con caveat frontal documentado.
- [ ] **Sanity checks por sujeto** — scripts 38a-d para cada nuevo sujeto (ERP, histograma alpha/beta, scatter).
- [ ] **Interpretabilidad** — scripts 39/40 para cada sujeto, comparar coeficientes cross-subject.
- [x] **Comparación multi-sujeto MVNN+SVC** — tabla resumen completada (ver sesión 2026-04-16).
- [ ] **Contacto Yongjie** — compartir resultados multi-sujeto para comparar con su pipeline LOSO.
- [ ] **Pantalla verde continua** (Tarea 3.6 pendiente del diario anterior) — retomar después de validar multi-sujeto.
