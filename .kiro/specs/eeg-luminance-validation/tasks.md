# Implementation Plan: EEG Luminance Validation

## Overview

Plan de implementación para las 5 áreas de validación del pipeline EEG-luminancia, en orden lógico de ejecución. Cada tarea construye sobre las anteriores. Se usa Python con la infraestructura existente (config_luminance.py, módulos en src/campeones_analysis/luminance/, LOVO_CV + GridSearchCV).

## Tasks

- [x] 1. Actualizar config y módulos base
  - [x] 1.1 Agregar nuevos parámetros a `scripts/modeling/config_luminance.py`
    - Agregar `ACTIVE_MODELS`, `TARGET_ZSCORE`, `DELTA_ZSCORE`, `CHANGE_THRESHOLD`, `ERP_N_CHANGES`, `ERP_TMIN`, `ERP_TMAX`, `N_SHUFFLE_ITERATIONS`
    - _Requirements: 6.3, 7b.5, 8.4, 9.6, 10.1, 10.2, 2.5_
  - [x] 1.2 Crear `src/campeones_analysis/luminance/evaluation.py` con `compute_r2_score`
    - Función pura que calcula R² usando sklearn.metrics.r2_score
    - _Requirements: 7.1, 7.2_
  - [x] 1.3 Write property test for R² calculation
    - **Property 6: R² calculation matches sklearn**
    - **Validates: Requirements 7.1**
  - [x] 1.4 Crear `src/campeones_analysis/luminance/targets.py` con `compute_delta_luminance` y `compute_change_labels`
    - Funciones puras para calcular delta luminancia y etiquetas binarias de cambio
    - _Requirements: 8.1, 8.2, 9.1_
  - [x] 1.5 Write property tests for delta luminance and change labels
    - **Property 5: Delta luminance computation and first-epoch discard**
    - **Validates: Requirements 8.1, 8.2**
    - **Property 7: Binary change labels from threshold**
    - **Validates: Requirements 9.1**
  - [x] 1.6 Crear `src/campeones_analysis/luminance/tde_glhmm.py` con `apply_glhmm_tde_pipeline`
    - Wrapper sobre `glhmm.preproc.build_data_tde()` + `glhmm.preproc.preprocess_data()` siguiendo protocolo Vidaurre et al. (2025)
    - _Requirements: 4.1, 4.2_
  - [x] 1.7 Agregar `compute_epoch_covariance` a `src/campeones_analysis/luminance/features.py`
    - Computa covarianza completa de componentes PCA por época, extrae triángulo superior aplanado
    - _Requirements: 4.3, 4.4_
  - [x] 1.8 Write property test for covariance feature extraction
    - **Property 2: Covariance feature extraction shape and content**
    - **Validates: Requirements 4.3, 4.4**
  - [x] 1.9 Actualizar `src/campeones_analysis/luminance/__init__.py` con nuevas exportaciones
    - Exportar `compute_r2_score`, `compute_delta_luminance`, `compute_change_labels`, `apply_glhmm_tde_pipeline`, `compute_epoch_covariance`
    - _Requirements: —_

- [x] 2. Checkpoint - Verificar módulos base
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. QA EEG con Autoreject (script 16)
  - [x] 3.1 Crear `src/campeones_analysis/luminance/qa.py` con `run_autoreject_qa` y `plot_rejection_heatmap`
    - Funciones puras para aplicar autoreject y generar visualizaciones de rechazo
    - _Requirements: 1.1, 1.2, 1.4_
  - [x] 3.2 Write property test for rejection percentage calculation
    - **Property 1: Rejection percentage calculation**
    - **Validates: Requirements 1.2**
  - [x] 3.3 Crear `scripts/qa/16_eeg_qa_autoreject.py`
    - Script que carga EEG preprocesado por run, aplica autoreject, genera reporte con desglose por run/video, heatmaps, CSV + JSON sidecar
    - Guardar en `results/qa/eeg/`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 4. Modelos Baseline y evaluación z-score (script 17)
  - [x] 4.1 Crear `scripts/modeling/17_baseline_models.py`
    - Implementar Shuffle Baseline: permutar etiquetas dentro de cada video, entrenar Ridge con LOVO_CV, repetir N veces para distribución nula
    - Implementar Mean Baseline: predecir media de entrenamiento en cada fold
    - Implementar evaluación z-score vs bruta: entrenar raw TDE con ambas representaciones del target
    - Reportar R², Pearson r, Spearman ρ, RMSE para todos
    - Guardar CSVs + JSON sidecars en `results/modeling/luminance/baselines/` y `results/modeling/luminance/zscore_evaluation/`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 7b.1, 7b.2, 7b.3, 7b.4, 7b.5_
  - [x] 4.2 Write property test for mean baseline
    - **Property 3: Mean baseline prediction equals training mean**
    - **Validates: Requirements 3.1, 3.2**

- [x] 5. Modificar script 13 para usar GLHMM TDE + covarianza
  - [x] 5.1 Refactorizar `scripts/modeling/13_luminance_raw_tde_model.py`
    - Reemplazar `apply_tde_on_continuous_signal` + `_apply_pca_to_tde_matrix` por `apply_glhmm_tde_pipeline`
    - Reemplazar `_epoch_pca_timeseries` (mean+var) por epoching + `compute_epoch_covariance`
    - Agregar R² a las métricas de evaluación usando `compute_r2_score`
    - Usar luminancia bruta por defecto (respetar TARGET_ZSCORE del config)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.2, 7b.1_

- [x] 6. PCA Sweep (script 18)
  - [x] 6.1 Crear `scripts/modeling/18_pca_sweep.py`
    - Aplicar PCA con 100 componentes sobre matriz TDE concatenada, registrar varianza explicada
    - Entrenar y evaluar Ridge con LOVO_CV para n_components en [10, 20, ..., 100]
    - Generar gráfico de varianza explicada acumulada y curva de rendimiento
    - Guardar en `results/modeling/luminance/pca_sweep/`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5b.1, 5b.2, 5b.3_
  - [x] 6.2 Write property test for cumulative PCA variance
    - **Property 4: Cumulative PCA explained variance is monotonically non-decreasing**
    - **Validates: Requirements 5.2**

- [x] 7. Checkpoint - Verificar features y modelos
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Delta Luminancia (script 19)
  - [x] 8.1 Crear `scripts/modeling/19_delta_luminance_model.py`
    - Calcular target delta (y_i = L_i - L_{i-1}), descartar primera época por segmento
    - Evaluar dos variantes: delta bruto y delta z-score normalizado
    - Usar pipeline GLHMM TDE + covarianza + Ridge con LOVO_CV
    - Reportar R², Pearson r, Spearman ρ, RMSE por fold y variante
    - Guardar en `results/modeling/luminance/delta_luminance/`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 9. Clasificación Cambio vs Estabilidad (script 20)
  - [x] 9.1 Crear `scripts/modeling/20_change_classifier.py`
    - Generar target binario basado en CHANGE_THRESHOLD
    - Aplicar undersampling de clase mayoritaria en entrenamiento
    - Pipeline: StandardScaler → clasificador lineal (LogisticRegression o RidgeClassifier) con LOVO_CV
    - Reportar accuracy, precision, recall, F1, AUC-ROC por fold
    - Guardar en `results/modeling/luminance/change_classification/`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  - [x] 9.2 Write property test for undersampling balance
    - **Property 10: Undersampling produces balanced classes**
    - **Validates: Requirements 9.2**

- [x] 10. Análisis ERP en cambios de luminancia (script 21)
  - [x] 10.1 Crear `scripts/validation/21_erp_luminance_changes.py`
    - Detectar top N momentos de mayor cambio absoluto de luminancia por video
    - Crear épocas MNE centradas en esos momentos (ventana configurable)
    - Promediar épocas para ERP medio en ROI_Posterior
    - Generar gráficos de morfología temporal (O1, O2, Pz) y topomaps
    - Guardar en `results/validation/erp/`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [x] 10.2 Write property test for top N changes detection
    - **Property 8: Top N luminance changes detection**
    - **Validates: Requirements 10.1**

- [x] 11. Correlación cruzada luminancia real vs percibida (script 22)
  - [x] 11.1 Crear `scripts/validation/22_cross_correlation.py`
    - Cargar luminancia real (CSV) y luminancia reportada (joystick) por video
    - Calcular correlación cruzada normalizada, identificar lag óptimo
    - Generar gráficos de xcorr vs lag por video
    - Guardar en `results/validation/cross_correlation/`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - [x] 11.2 Write property test for cross-correlation
    - **Property 9: Cross-correlation bounds and lag detection**
    - **Validates: Requirements 11.1, 11.2**

- [x] 12. Actualizar reporte comparativo (script 15)
  - [x] 12.1 Actualizar `scripts/reporting/15_model_comparison_report.py`
    - Extender MODEL_DEFS con nuevos modelos (baselines, covarianza TDE, delta, classifier)
    - Agregar R² a métricas de regresión
    - Agregar métricas de clasificación (accuracy, F1, AUC-ROC) para change classifier
    - Filtrar modelos según ACTIVE_MODELS del config (excluir Spectral TDE)
    - Guardar en `results/modeling/luminance/comparison/` con JSON sidecar
    - _Requirements: 6.1, 6.2, 6.3, 7.3, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [x] 13. Final checkpoint - Verificar pipeline completo
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validationderere
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- All scripts use `micromamba run -n campeones` for execution
- All CSV outputs require JSON sidecar (BIDS compliance)
