# Baseline Models Analysis

**Date:** 2026-02-25
**Subject:** sub-27 (VR Luminance Task)

## Objective
Evaluate the baseline models for luminance prediction (Shuffle, Mean, and Z-score vs Raw evaluation) to establish the true null distribution and validate the target normalization strategy.

## Results and Interpretation

1. **AutoReject está funcionando según lo esperado**: Se descartaron las épocas correctas en las distintas rachas (por ejemplo 131 épocas en el video 9 del run-003, y 54 en video 12 del run-002), procesando un total final de **3782 épocas** limpias.
2. **Shuffle Baseline (Nulidad/Azar)**: Tras 100 iteraciones barajando las etiquetas por video, el R² general es **-0.040**, junto con una correlación (Pearson y Spearman) de **~0.0**. El RMSE es de **1.019**. Este es nuestro verdadero límite de "azar"; cualquier modelo útil debe superar este umbral consistentemente de R² ~ -0.04.
3. **Mean Baseline (Media de Entrenamiento)**: Como los datos objetivo están estandarizados (z-score), predecir siempre la media arroja exactamente R²=0, correlación nula y un RMSE exacto de **1.0**.
4. **Evaluación de Z-score vs Raw**:
   * Entrenar el modelo con targets **crudos (0-255)** arroja resultados desastrosos: **R² = -5.27** y **RMSE = 53.59**.
   * Entrenar utilizando los targets **estandarizados (z-score)** por video normaliza drásticamente la capacidad de generalización: el RMSE cae a **1.059** y el R² a **-0.12** (mucho más cerca de la zona nula real del shuffle).

## Conclusion
Esto demuestra enfáticamente que mapear valores crudos de reflectancia (0-255) desde la señal EEG en crudo para todas las películas es prohibitivo debido al cambio de escala por video. La decisión de normalizar vía z-score previo a modelar está matemáticamente justificada.

## Update (2026-02-26): Context After Global PCA Fix
Tras corregir un bug crítico en el pipeline TDE (PCA se fiteaba por video en lugar de globalmente), el modelo con features TDE + covarianza z-scored alcanza ahora **R² = -0.063** y **RMSE = 1.030** (con 20 componentes PCA). Esto es muy cercano al benchmark nulo:
- **Shuffle baseline**: R² = -0.040, RMSE = 1.019
- **Mean baseline**: R² = 0.000, RMSE = 1.000
- **TDE model (fixed)**: R² = -0.063, RMSE = 1.030

El modelo fijado ya no colapsa catastróficamente (R²=-1.4 → -0.06), pero todavía no supera consistentemente la media constante. El fold 12_b logra R² > 0 (0.027), indicando la primera evidencia débil de señal lineal genuina.

