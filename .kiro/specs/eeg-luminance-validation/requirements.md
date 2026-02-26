# Documento de Requisitos: Validación del Pipeline EEG-Luminancia

## Introducción

Este feature implementa un plan de validación y mejora integral del pipeline de predicción de luminancia desde EEG para sub-27 del dataset CAMPEONES. El plan, definido por los supervisores del proyecto, abarca cinco áreas: (I) control de calidad EEG con `autoreject`, (II) modelos baseline nulos para establecer nivel de azar, (III) optimización de características con covarianza TDE completa y barrido de componentes PCA, (IV) nuevas métricas y targets de evaluación (R², delta luminancia, clasificación cambio/estabilidad), y (V) análisis neurofisiológico de validación (ERPs en cambios de luminancia y correlación cruzada). Todas las mejoras se integran sobre la infraestructura existente (scripts 10–13, módulos en `src/campeones_analysis/luminance/`, config centralizado).

## Glosario

- **Pipeline**: El sistema de predicción EEG→luminancia compuesto por los scripts de modelado (10–13) y módulos en `src/campeones_analysis/luminance/`.
- **GLHMM**: Librería de Python (ya en `environment.yml`) que implementa el pipeline TDE-GLHMM descrito en Vidaurre et al. (2025, Nature Protocols). En este proyecto se reutiliza la parte de preprocesamiento TDE+PCA del pipeline, reemplazando GLHMM por Ridge regression como modelo predictivo.
- **TDE_Pipeline**: Pipeline de Time-Delay Embedding basado en el protocolo GLHMM (Vidaurre et al., 2025): datos EEG continuos → `glhmm.preproc.build_data_tde()` para embedding temporal → `glhmm.preproc.preprocess_data()` para estandarización + PCA → epoching → extracción de features de covarianza → Ridge regression.
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

### Requisito 1: Control de Calidad EEG con Autoreject

**User Story:** Como investigador, quiero cuantificar objetivamente la limpieza de la señal EEG de sub-27 usando `autoreject`, para documentar el porcentaje de épocas rechazadas y justificar si el sujeto es representativo o si ciertos segmentos deben excluirse para evitar sesgar el entrenamiento de Ridge.

#### Criterios de Aceptación

1. WHEN el QA_Module recibe datos EEG preprocesados de sub-27, THE QA_Module SHALL aplicar `autoreject` para calcular umbrales de rechazo por canal y por época.
2. WHEN `autoreject` completa el análisis, THE QA_Module SHALL documentar el porcentaje de épocas rechazadas bajo los umbrales calculados para cada run.
3. WHEN el QA_Module genera el reporte de calidad, THE QA_Module SHALL incluir un desglose por run y por video del número de épocas totales, épocas rechazadas y porcentaje de rechazo.
4. WHEN el QA_Module completa el análisis, THE QA_Module SHALL generar una visualización del patrón de rechazo (heatmap canales × épocas) para cada run.
5. THE QA_Module SHALL guardar todos los resultados y figuras en `results/qa/eeg/`.
6. WHEN el QA_Module genera resultados tabulares, THE QA_Module SHALL crear un archivo JSON sidecar con el diccionario de datos correspondiente.

### Requisito 2: Modelo Baseline con Permutación de Etiquetas (Shuffle Baseline)

**User Story:** Como investigador, quiero entrenar un modelo Ridge con etiquetas de luminancia permutadas aleatoriamente entre épocas, para establecer el nivel de azar de R² y Spearman y comparar contra el modelo real.

#### Criterios de Aceptación

1. WHEN el Shuffle_Baseline se ejecuta, THE Shuffle_Baseline SHALL permutar aleatoriamente las etiquetas de luminancia (y) entre épocas dentro de cada grupo de video, manteniendo los features (X) intactos.
2. WHEN el Shuffle_Baseline entrena el modelo, THE Shuffle_Baseline SHALL usar el mismo pipeline (StandardScaler → Ridge con GridSearchCV) y esquema LOVO_CV que los modelos reales.
3. WHEN el Shuffle_Baseline completa la evaluación, THE Shuffle_Baseline SHALL reportar R², Pearson r y Spearman ρ por fold y promedio.
4. WHEN el Shuffle_Baseline genera resultados, THE Shuffle_Baseline SHALL guardar un CSV con métricas por fold en `results/modeling/luminance/baselines/`.
5. THE Shuffle_Baseline SHALL repetir el proceso N veces (configurable, default 100) para construir una distribución nula de métricas.

### Requisito 3: Modelo Baseline de Media (Mean/Persistence Baseline)

**User Story:** Como investigador, quiero evaluar un modelo que siempre predice la media de luminancia del conjunto de entrenamiento, para tener una referencia mínima de rendimiento contra la cual comparar los modelos EEG.

#### Criterios de Aceptación

1. WHEN el Mean_Baseline se ejecuta en cada fold de LOVO_CV, THE Mean_Baseline SHALL calcular la media de los targets de luminancia del conjunto de entrenamiento de ese fold.
2. WHEN el Mean_Baseline genera predicciones, THE Mean_Baseline SHALL asignar el valor de la media de entrenamiento como predicción para todas las épocas del conjunto de test.
3. WHEN el Mean_Baseline completa la evaluación, THE Mean_Baseline SHALL reportar R², Pearson r, Spearman ρ y RMSE por fold y promedio.
4. WHEN el Mean_Baseline genera resultados, THE Mean_Baseline SHALL guardar un CSV con métricas por fold en `results/modeling/luminance/baselines/`.

### Requisito 4: Pipeline TDE con GLHMM y Extracción de Covarianza

**User Story:** Como investigador, quiero que el pipeline de TDE utilice la librería GLHMM (Vidaurre et al., 2025) para el embedding temporal y preprocesamiento (TDE + estandarización + PCA), y que tras obtener los componentes PCA por época, compute la matriz de covarianza completa y extraiga el triángulo superior como vector de features, para seguir el protocolo establecido en la literatura y capturar coherencia de fase y relaciones de lag entre componentes.

#### Criterios de Aceptación

1. WHEN el TDE_Pipeline procesa un segmento de video, THE TDE_Pipeline SHALL aplicar `glhmm.preproc.build_data_tde()` sobre la señal EEG continua de los canales ROI para generar el embedding temporal, siguiendo el protocolo de Vidaurre et al. (2025).
2. WHEN el embedding TDE es generado, THE TDE_Pipeline SHALL aplicar `glhmm.preproc.preprocess_data()` con estandarización y PCA para reducir la dimensionalidad de los features TDE-expandidos.
3. WHEN los componentes PCA de una época son obtenidos, THE TDE_Covariance SHALL computar la matriz de covarianza completa de dimensión (n_components × n_components).
4. WHEN la matriz de covarianza es computada, THE TDE_Covariance SHALL extraer el triángulo superior incluyendo la diagonal y aplanarlo en un vector 1-D de features de longitud n_components × (n_components + 1) / 2.
5. THE TDE_Covariance SHALL reemplazar el resumen mean+variance actual del script 13 como método único de extracción de features por época.
6. WHEN el TDE_Covariance se evalúa con LOVO_CV, THE Pipeline SHALL reportar las mismas métricas (R², Pearson r, Spearman ρ, RMSE) que los demás modelos.

### Requisito 5: Análisis de Varianza Explicada por PCA

**User Story:** Como investigador, quiero visualizar la varianza explicada acumulada de PCA aplicado sobre la matriz TDE (hasta 100 componentes), para entender cuánta información retiene cada número de componentes.

#### Criterios de Aceptación

1. WHEN el PCA_Sweep se ejecuta, THE PCA_Sweep SHALL aplicar PCA con 100 componentes sobre la matriz TDE concatenada de todos los segmentos de video.
2. WHEN el PCA es ajustado, THE PCA_Sweep SHALL registrar la varianza explicada individual y acumulada para cada componente de 1 a 100.
3. WHEN el PCA_Sweep completa el análisis, THE PCA_Sweep SHALL generar un gráfico de varianza explicada acumulada vs número de componentes.
4. WHEN el PCA_Sweep genera resultados, THE PCA_Sweep SHALL guardar el gráfico y un CSV con la varianza explicada por componente en `results/modeling/luminance/pca_sweep/`.

### Requisito 5b: Curva de Rendimiento de Ridge vs Número de Componentes PCA

**User Story:** Como investigador, quiero evaluar el rendimiento de Ridge al variar el número de componentes PCA (rango 10–100), para identificar el punto donde agregar más componentes deja de mejorar significativamente la predicción.

#### Criterios de Aceptación

1. WHEN el PCA_Sweep evalúa el rendimiento, THE PCA_Sweep SHALL entrenar y evaluar Ridge con LOVO_CV para cada valor de n_components en el rango [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
2. WHEN el PCA_Sweep completa la evaluación, THE PCA_Sweep SHALL generar un gráfico de curva de rendimiento con n_components en el eje X y las métricas promedio (Pearson r y R²) en el eje Y.
3. WHEN el PCA_Sweep genera resultados, THE PCA_Sweep SHALL guardar el gráfico y un CSV con métricas por n_components en `results/modeling/luminance/pca_sweep/`.

### Requisito 6: Eliminación del Modelo Spectral TDE

**User Story:** Como investigador, quiero eliminar el modelo Spectral TDE (script 12) del pipeline de comparación activo, porque no aporta información útil para la predicción de luminancia según la evaluación de los supervisores.

#### Criterios de Aceptación

1. WHEN el reporte de comparación de modelos se genera (script 15), THE Pipeline SHALL excluir el modelo Spectral TDE de las tablas y gráficos comparativos.
2. THE Pipeline SHALL mantener el script 12 en el repositorio como referencia histórica, sin eliminarlo del sistema de archivos.
3. WHEN el Config se actualiza, THE Config SHALL incluir un flag o lista que indique qué modelos están activos para comparación.

### Requisito 7: Evaluación con R² Score

**User Story:** Como investigador, quiero calcular el R² score en cada fold de validación (no solo Spearman), para cuantificar qué proporción de la variabilidad de luminancia es explicada por el EEG.

#### Criterios de Aceptación

1. WHEN el Pipeline evalúa un fold de LOVO_CV, THE Pipeline SHALL calcular el R² score además de Pearson r, Spearman ρ y RMSE.
2. WHEN el Pipeline genera resultados CSV, THE Pipeline SHALL incluir una columna R2 con el coeficiente de determinación por fold.
3. WHEN el Pipeline genera el reporte de comparación, THE Pipeline SHALL incluir R² en las tablas y gráficos comparativos de todos los modelos.
4. THE Pipeline SHALL aplicar el cálculo de R² a todos los modelos existentes (base, raw TDE) y a los nuevos modelos (baselines, covarianza TDE).

### Requisito 7b: Evaluación de Z-Score vs Luminancia Bruta

**User Story:** Como investigador, quiero comparar el rendimiento de los modelos predictivos usando luminancia bruta (0–255) vs luminancia z-score normalizada por video, para determinar empíricamente cuál representación del target produce mejores predicciones.

#### Criterios de Aceptación

1. THE Pipeline SHALL usar luminancia bruta (valores originales 0–255) como target por defecto en todos los modelos predictivos.
2. WHEN la evaluación z-score vs bruta se ejecuta, THE Pipeline SHALL entrenar y evaluar el modelo raw TDE con LOVO_CV usando ambas representaciones del target: luminancia bruta y luminancia z-score normalizada por video.
3. WHEN la evaluación completa, THE Pipeline SHALL reportar R², Pearson r, Spearman ρ y RMSE para cada representación del target.
4. WHEN la evaluación genera resultados, THE Pipeline SHALL guardar un CSV comparativo y un gráfico en `results/modeling/luminance/zscore_evaluation/`.
5. THE Config SHALL definir TARGET_ZSCORE como un parámetro booleano configurable (default False) para controlar si se aplica z-score al target de luminancia.

### Requisito 8: Predicción de Delta Luminancia (ΔL)

**User Story:** Como investigador, quiero usar como target y_i = L_actual − L_anterior para evaluar si el cerebro responde más a transiciones de luminancia que a estados estables de luz.

#### Criterios de Aceptación

1. WHEN el Pipeline calcula el target delta luminancia, THE Pipeline SHALL computar y_i = L_i − L_{i-1} para cada época, donde L_i es la luminancia media de la época actual y L_{i-1} es la luminancia media de la época anterior.
2. WHEN la primera época de un segmento de video no tiene época anterior, THE Pipeline SHALL descartar esa época del dataset.
3. WHEN el target delta luminancia es calculado, THE Pipeline SHALL evaluar dos variantes: con valores brutos de delta y con z-score normalization por video, reportando métricas para ambas.
4. THE Config SHALL definir DELTA_ZSCORE como un parámetro booleano configurable (default False) para controlar si se aplica z-score al target delta.
5. WHEN el modelo delta luminancia completa la evaluación, THE Pipeline SHALL reportar R², Pearson r, Spearman ρ y RMSE por fold y promedio para cada variante.
6. WHEN el modelo delta luminancia genera resultados, THE Pipeline SHALL guardar un CSV con métricas por fold para cada variante en `results/modeling/luminance/delta_luminance/`.

### Requisito 9: Clasificación Cambio vs Estabilidad

**User Story:** Como investigador, quiero clasificar épocas como "cambio detectado" (1) o "constante" (0) y aplicar técnicas de balanceo de clases, para evaluar si el modelo puede distinguir momentos de transición de luminancia.

#### Criterios de Aceptación

1. WHEN el Change_Classifier genera el target binario, THE Change_Classifier SHALL asignar 1 (cambio) cuando el valor absoluto de delta luminancia supera un umbral configurable, y 0 (constante) en caso contrario.
2. WHEN el Change_Classifier detecta desbalanceo de clases, THE Change_Classifier SHALL aplicar undersampling de la clase mayoritaria (épocas constantes) para equilibrar las clases en el conjunto de entrenamiento.
3. WHEN el Change_Classifier entrena el modelo, THE Change_Classifier SHALL usar un pipeline de clasificación (StandardScaler → clasificador lineal) con LOVO_CV.
4. WHEN el Change_Classifier completa la evaluación, THE Change_Classifier SHALL reportar accuracy, precision, recall, F1-score y AUC-ROC por fold y promedio.
5. WHEN el Change_Classifier genera resultados, THE Change_Classifier SHALL guardar un CSV con métricas por fold en `results/modeling/luminance/change_classification/`.
6. THE Config SHALL definir CHANGE_THRESHOLD como un parámetro configurable para el umbral de detección de cambio.

### Requisito 10: Análisis ERP Centrado en Cambios de Luminancia

**User Story:** Como investigador, quiero crear épocas centradas en los momentos de mayor cambio de luminancia en el video y analizar la morfología de la señal en canales occipitales, para verificar si existe un potencial evocado claro asociado a transiciones de luminancia.

#### Criterios de Aceptación

1. WHEN el ERP_Analysis identifica momentos de cambio, THE ERP_Analysis SHALL detectar los N momentos de mayor cambio absoluto de luminancia en cada video (N configurable, default 50).
2. WHEN los momentos de cambio son identificados, THE ERP_Analysis SHALL crear épocas EEG centradas en esos momentos con una ventana temporal configurable (default: −200 ms a +800 ms).
3. WHEN las épocas ERP son creadas, THE ERP_Analysis SHALL promediar las épocas para obtener el ERP medio en cada canal de la ROI_Posterior.
4. WHEN el ERP medio es calculado, THE ERP_Analysis SHALL generar gráficos de la morfología temporal del ERP para los canales occipitales (O1, O2, Pz como mínimo).
5. WHEN el ERP_Analysis genera resultados, THE ERP_Analysis SHALL guardar las figuras y datos del ERP en `results/validation/erp/`.
6. THE ERP_Analysis SHALL generar un gráfico topográfico (topomap) de la distribución espacial del ERP en ventanas temporales clave (e.g., 100 ms, 200 ms, 300 ms post-cambio).

### Requisito 11: Correlación Cruzada entre Luminancia Real y Percibida (Sub-27)

**User Story:** Como investigador, quiero medir el desfase temporal (lag) entre la luminancia real del video y la luminancia reportada/percibida por el sujeto 27, para entender la latencia de la respuesta perceptual.

#### Criterios de Aceptación

1. WHEN el Cross_Correlation recibe las series temporales de luminancia real y luminancia reportada (joystick), THE Cross_Correlation SHALL calcular la función de correlación cruzada normalizada entre ambas señales.
2. WHEN la correlación cruzada es calculada, THE Cross_Correlation SHALL identificar el lag (en segundos) que maximiza la correlación.
3. WHEN el Cross_Correlation completa el análisis, THE Cross_Correlation SHALL generar un gráfico de la función de correlación cruzada vs lag para cada video de luminancia.
4. WHEN el Cross_Correlation genera resultados, THE Cross_Correlation SHALL guardar las figuras y un CSV con el lag óptimo y correlación máxima por video en `results/validation/cross_correlation/`.
5. IF la señal de joystick de luminancia no está disponible para un video, THEN THE Cross_Correlation SHALL registrar una advertencia y omitir ese video.

### Requisito 12: Integración y Reporte Comparativo

**User Story:** Como investigador, quiero un reporte consolidado que compare todos los modelos (reales y baselines) con todas las métricas, para tener una visión completa del rendimiento del pipeline.

#### Criterios de Aceptación

1. WHEN el reporte comparativo se genera, THE Pipeline SHALL implementarlo como una nueva versión adaptada del script 15 existente (`scripts/reporting/15_model_comparison_report.py`), extendiendo sus definiciones de modelos y métricas.
2. WHEN el reporte comparativo se genera, THE Pipeline SHALL incluir todos los modelos activos: base, raw TDE, raw TDE con covarianza, shuffle baseline, mean baseline, delta luminancia y change classifier.
3. WHEN el reporte comparativo presenta métricas, THE Pipeline SHALL incluir R², Pearson r, Spearman ρ y RMSE para modelos de regresión, y accuracy, F1, AUC-ROC para el clasificador.
4. WHEN el reporte comparativo se genera, THE Pipeline SHALL producir una tabla resumen y un gráfico de barras agrupado comparando métricas entre modelos.
5. THE Pipeline SHALL guardar el reporte en `results/modeling/luminance/comparison/`.
6. WHEN el reporte genera resultados tabulares, THE Pipeline SHALL crear un archivo JSON sidecar con el diccionario de datos correspondiente.
