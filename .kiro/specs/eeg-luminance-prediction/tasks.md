# Plan de Implementación: Predicción de Luminancia Real desde EEG

## Visión General

Implementación incremental de 5 scripts de modelado (08-12) con lógica reutilizable en `src/campeones_analysis/luminance/`. Cada tarea construye sobre la anterior. Se usa Python con las librerías existentes en `environment.yml` (mne, scikit-learn, scipy, pandas, numpy, matplotlib). Se debe agregar `hypothesis` al entorno para property-based testing.


## Tareas
- [x] 1. Configuración base y módulo de luminancia
  - [x] 1.1 Crear `scripts/modeling/config_luminance.py` con todos los parámetros centralizados
    - Épocas: 500ms duración, 400ms solapamiento, 100ms paso
    - ROI_Posterior: O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6 (revisar la existencia de estos canales para la definicion del ROI, y redefinirlos en caso no existan).
    - Bandas espectrales: delta, theta, alpha, beta, gamma
    - TDE: window_half=10, PCA components=50
    - Pipeline: PCA=100, Ridge alpha=1.0, seed=42
    - Rutas relativas con Path, LUMINANCE_CSV_MAP, RUNS_CONFIG, EXPERIMENTAL_VIDEOS
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 1.2 Crear `src/campeones_analysis/luminance/__init__.py` y estructura del módulo
    - Crear directorio `src/campeones_analysis/luminance/`
    - Exponer API pública en `__init__.py`
    - _Requirements: 6.1_

- [-] 2. Módulo de sincronización EEG-Luminancia
  - [-] 2.1 Implementar `src/campeones_analysis/luminance/sync.py`
    - `load_luminance_csv(csv_path) -> pd.DataFrame`
    - `create_epoch_onsets(n_samples_total, sfreq, epoch_duration_s, epoch_step_s) -> np.ndarray`
    - `interpolate_luminance_to_epochs(luminance_df, epoch_onsets_s, epoch_duration_s) -> np.ndarray`
    - Funciones puras, type hints, Google docstrings
    - _Requirements: 3.1, 3.2, 3.3_
  - [ ]* 2.2 Escribir property tests para sincronización (`tests/test_luminance_sync.py`)
    - **Property 4: Conteo y espaciado de épocas generadas**
    - **Validates: Requirements 3.2**
    - **Property 5: Luminancia interpolada dentro de rango válido**
    - **Validates: Requirements 3.1, 3.3**
  - [ ]* 2.3 Escribir unit tests para sincronización (`tests/test_luminance_sync.py`)
    - Test de carga de CSV con formato correcto
    - Edge case: segmento EEG demasiado corto para generar épocas
    - Edge case: CSV de luminancia no encontrado (Req 3.7)
    - _Requirements: 3.1, 3.2, 3.3, 3.7_

- [ ] 3. Script de exploración de luminancia
  - [ ] 3.1 Implementar `scripts/modeling/08_explore_luminance.py`
    - Cargar los 4 CSVs de luminancia (videos 3, 7, 9, 12) usando `load_luminance_csv`
    - Generar plots de series de tiempo crudas (un subplot por video)
    - Calcular y plotear diferencias temporales (diff t vs t+1)
    - Imprimir estadísticas descriptivas (media, std, min, max, duración)
    - Guardar figuras en `results/modeling/luminance/exploration/`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [ ]* 3.2 Escribir property tests para exploración (`tests/test_luminance_explore.py`)
    - **Property 1: La serie de diferencias temporales tiene longitud N-1**
    - **Validates: Requirements 1.2**
    - **Property 2: Invariantes de estadísticas descriptivas de luminancia**
    - **Validates: Requirements 1.4**

- [ ] 4. Script de verificación de marcas de estímulos
  - [ ] 4.1 Implementar `scripts/modeling/09_verify_luminance_markers.py`
    - Para cada run de sub-27 (Acq A y B), cargar eventos TSV y Order Matrix
    - Filtrar eventos con trial_type `video_luminance`
    - Cruzar con video_id de Order Matrix para determinar mapeo
    - Detectar y reportar discrepancias
    - Generar CSV de reporte consolidado en `results/modeling/luminance/verification/`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 4.2 Escribir property test para filtrado de eventos (`tests/test_luminance_explore.py`)
    - **Property 3: Filtrado de eventos video_luminance**
    - **Validates: Requirements 2.1**

- [ ] 5. Checkpoint - Verificar exploración y marcas
  - Ejecutar scripts 08 y 09, verificar que los plots y reportes se generan correctamente
  - Revisar el reporte de verificación de marcas para identificar discrepancias
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Modelo predictivo base con luminancia real
  - [ ] 6.1 Implementar `scripts/modeling/10_luminance_base_model.py`
    - Cargar EEG preprocesado, eventos, Order Matrix por cada run
    - Identificar segmentos video_luminance y resolver video_id correspondiente
    - Cropear EEG al segmento de video, cargar CSV de luminancia correspondiente
    - Generar épocas de 500ms/400ms overlap usando `create_epoch_onsets`
    - Vectorizar EEG crudo (32 canales), target = luminancia interpolada
    - Pipeline: Vectorizer → StandardScaler → PCA(100) → Ridge(alpha=1.0)
    - Leave-One-Video-Out CV con métricas: Pearson r, Spearman ρ, RMSE
    - Guardar resultados CSV y plots en `results/modeling/luminance/base/`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4_
  - [ ]* 6.2 Escribir property test para CV splitting (`tests/test_luminance_pipeline.py`)
    - **Property 6: Correctitud del split Leave-One-Video-Out**
    - **Validates: Requirements 3.5, 4.4, 5.4**
  - [ ]* 6.3 Escribir property test para determinismo (`tests/test_luminance_pipeline.py`)
    - **Property 11: Determinismo del pipeline con semilla fija**
    - **Validates: Requirements 6.2**

- [ ] 7. Módulo de features espectrales y TDE
  - [ ] 7.1 Implementar `src/campeones_analysis/luminance/features.py`
    - `extract_bandpower(eeg_epoch, sfreq, bands) -> np.ndarray` usando scipy.signal.welch
    - `apply_time_delay_embedding(feature_matrix, window_half) -> np.ndarray`
    - Funciones puras, type hints, Google docstrings
    - _Requirements: 4.1, 4.3, 5.1, 5.2_
  - [ ]* 7.2 Escribir property tests para features (`tests/test_luminance_features.py`)
    - **Property 7: Invariantes de extracción de bandpower**
    - **Validates: Requirements 4.1, 4.3**
    - **Property 9: Forma y contenido del Time Delay Embedding**
    - **Validates: Requirements 5.1, 5.2**
    - **Property 10: Reducción de dimensionalidad por PCA**
    - **Validates: Requirements 5.3**

- [ ] 8. Checkpoint - Verificar módulos core
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Modelo con features espectrales en ROI posterior
  - [ ] 9.1 Implementar `scripts/modeling/11_luminance_spectral_model.py`
    - Mismo flujo que script 10 pero usando `extract_bandpower` en ROI_Posterior
    - Seleccionar canales de ROI_Posterior (intersección con disponibles)
    - Features: bandpower (n_bands × n_channels_roi) por época
    - Pipeline: StandardScaler → PCA(100) → Ridge(alpha=1.0)
    - Leave-One-Video-Out CV, métricas, plots comparativos con modelo base
    - Guardar resultados en `results/modeling/luminance/spectral/`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [ ]* 9.2 Escribir property test para selección de canales ROI (`tests/test_luminance_pipeline.py`)
    - **Property 8: Selección de canales ROI es subconjunto válido**
    - **Validates: Requirements 4.2**

- [ ] 10. Modelo con TDE + PCA
  - [ ] 10.1 Implementar `scripts/modeling/12_luminance_tde_model.py`
    - Mismo flujo que script 11 pero aplicando TDE sobre features espectrales
    - `apply_time_delay_embedding` con window_half=10 sobre la secuencia de features
    - PCA inmediato post-TDE (n_components=50) para reducir dimensionalidad
    - Pipeline: StandardScaler → PCA(50) → Ridge(alpha=1.0)
    - Leave-One-Video-Out CV, métricas, plots comparativos con modelo espectral
    - Guardar resultados en `results/modeling/luminance/tde/`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 11. Checkpoint final - Verificar pipeline completo
  - Ejecutar los 5 scripts (08-12) secuencialmente
  - Verificar que todos los resultados CSV y plots se generan correctamente
  - Ensure all tests pass, ask the user if questions arise.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Se debe agregar `hypothesis` a `environment.yml` antes de implementar property tests
- Todos los scripts usan `micromamba run -n campeones python` para ejecución
- Los property tests validan propiedades universales de correctitud; los unit tests validan ejemplos y edge cases específicos
