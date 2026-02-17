# Documento de Requisitos: Validación del Pipeline EEG-Luminancia

## Introducción

Este feature implementa un plan de validación y mejora integral del pipeline de predicción de luminancia desde EEG para sub-27 del dataset CAMPEONES. El plan, definido por los supervisores del proyecto, abarca cinco áreas: (I) control de calidad EEG con `autoreject`, (II) modelos baseline nulos para establecer nivel de azar, (III) optimización de características con covarianza TDE completa y barrido de componentes PCA, (IV) nuevas métricas y targets de evaluación (R², delta luminancia, clasificación cambio/estabilidad), y (V) análisis neurofisiológico de validación (ERPs en cambios de luminancia y correlación cruzada). Todas las mejoras se integran sobre la infraestructura existente (scripts 10–13, módulos en `src/campeones_analysis/luminance/`, config centralizado).

## Glosario

- **Pipeline**: El sistema de predicción EEG→luminancia compuesto por los scripts de modelado (10–13) y módulos en `src/campeones_analysis/luminance/`.
- **QA_Module**: Módulo de control de calidad que aplica `autoreject` para cuantificar objetivamente la limpieza de la señal EEG.
- **Autoreject**: Librería de Python (ya en `environment.yml`) que calcula umbrales de rechazo de canales/épocas de forma automática y basada en datos.
- **Shuffle_Baseline**: Modelo baseline que entrena Ridge con etiquetas de luminancia permutadas aleatoriamente entre épocas para establecer el nivel de azar de R² y Spearman.
- **Mean_Baseline**: Modelo baseline que siempre predice la media de luminancia del conjunto de entrenamiento.
- **TDE_Covariance**: Estrategia de extracción de características que, tras aplicar TDE y PCA, computa la matriz de covarianza completa de los componentes PCA por época y extrae el triángulo superior (incluyendo diagonal) como vector de features.
- **PCA_Sweep**: Barrido sistemático del número de componentes PCA (rango 10–100) para identificar el punto donde Ridge deja de ganar rendimiento significativo.
- **R2_Score**: Coeficiente de determinación que cuantifica la proporción de variabilidad de luminancia explicada por el modelo EEG.
- **Delta_Luminance**: Target alternativo definido como y_i = L_actual − L_anterior, que evalúa si el cerebro responde más a transiciones que a estados estables de luz.
- **Change_Classifier**: Clasificador binario donde el target es 1 (cambio detectado) o 0 (constante), con técnicas de balanceo de clases.
- **ERP_Analysis**: Análisis de potenciales evocados centrado en momentos de máximo cambio de luminancia en el video, evaluando morfología de señal en canales occipitales.
- **Cross_Correlation**: Medida del desfase temporal (lag) entre la luminancia real del video y la luminancia reportada/percibida por el sujeto 27.
- **LOVO_CV**: Leave-One-Video-Out Cross-Validation, esquema de validación cruzada donde cada fold excluye todos los epochs de un video.
- **ROI_Posterior**: Electrodos posteriores/occipitales: O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6.
- **Config**: Módulo de configuración centralizado `scripts/modeling/config_luminance.py`.

## Requisitos
