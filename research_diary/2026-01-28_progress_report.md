# Research Diary: Modelado Predictivo de estados afectivos a partir de EEG
**Fecha:** 28 de Enero de 2026
**Sujeto:** 27 (Acq A & B)
**Sesión:** VR

## 1. Objetivo
El objetivo es hace decoding estados afectivos (**Arousal, Valencia**) y perceptivos (**Luminancia**) a partir de datos de EEG de registrados durante la visualización de videos. Las etiquetas de "ground truth" fueron generadas por un reporte continuo mediante joystick.

## 2. Pipeline de procesamiento de datos

### 2.1 Preprocesamiento y Carga (Loading)
- **Fuente:** Archivos BrainVision (`.vhdr`) preprocesados mediante un pipeline semi-automatizado (filtrado, ICA, rechazo de artefactos manual).
- **Alineación Comportamental:** Los datos del joystick (`.xdf` / `.log`) se alinean con los triggers del EEG.
- **Corrección de Polaridad:**
  - La dirección del joystick para  fue aleatorizada por video.
  - **Paso de Corrección:** Si `polaridad == 'inverse'`, entonces $y_{raw} = -1 \times y_{raw}$.
  - Esto asegura que los valores positivos siempre correspondan a alta intensidad (ej. Alto Arousal).

### 2.2 Feature Engineering (Fase 1: Dominio del Tiempo)
Inicialmente, abordamos esto como un problema de decoding de series temporales crudas.
- **Epocado:** Sliding windows con solapamiento.
  - Duración: 1.0 segundo.
  - Solapamiento: 0.9 segundos (Step = 0.1s).
- **Features:**
  - Amplitud cruda del EEG de los 32 canales.
  - **Vectorización:** Aplanado de `(n_channels, n_times)` -> `(n_features,)`.
  - **Escalado:** `StandardScaler` (Normalización Z-score).
  - **Reducción de Dimensionalidad:** Análisis de Componentes Principales (**PCA**) con `n_components=100`.

## 3. Modelos predictivos

Probamos iterativamente tres enfoques para mapear el EEG a los valores del Joystick.

### Test 1: Regresión (Predicción Continua)
- **Objetivo:** Predecir el valor exacto del joystick $y \in [-1, 1]$.
- **Modelo:** `RidgeRegression` (Alpha=1.0).
- **Validación:** Leave-One-Video-Out (LOVO).
- **Dimensiones del Dataset:**
  - **Features (Input):** ~9,250 features crudos (37 canales x 250 muestras) -> Reducido a **100 Componentes Principales**.
  - **Estructura por Dimensión:**
    - **Arousal:** 14 videos (folds). ~17,000 inst. train / ~1,200 inst. test por fold.
    - **Valencia:** 10 videos (folds). ~12,000 inst. train / ~1,300 inst. test por fold.
    - **Luminancia:** 7 videos (folds). ~3,700 inst. train / ~600 inst. test por fold.

### Resultados Detallados (Regresión)

#### Arousal
| Fold (Test Video) | Train N | Test N | RMSE | Corr |
| :--- | :---: | :---: | :---: | :---: |
| 8.0_a | 17719 | 792 | 0.181 | 0.524 |
| 5.0_a | 17677 | 834 | 0.132 | 0.420 |
| 6.0_b | 16833 | 1678 | 0.560 | 0.152 |
| 2.0_b | 16135 | 2376 | 0.239 | 0.095 |
| 1.0_a | 17448 | 1063 | 0.147 | 0.083 |
| 7.0_a | 16844 | 1667 | 0.507 | 0.073 |
| 9.0_a | 16917 | 1594 | 0.099 | -0.009 |
| 12.0_b | 17885 | 626 | 0.121 | -0.038 |
| 13.0_b | 16270 | 2241 | 0.241 | -0.072 |
| 3.0_b | 17541 | 970 | 0.140 | -0.106 |
| 10.0_a | 16719 | 1792 | 0.183 | -0.184 |
| 4.0_a | 17895 | 616 | 0.502 | -0.196 |
| 11.0_a | 17448 | 1063 | 0.210 | -0.617 |
| 14.0_b | 17312 | 1199 | 0.251 | -0.671 |
| **Promedio** | - | - | **0.251** | **-0.039** |

#### Valencia
| Fold (Test Video) | Train N | Test N | RMSE | Corr |
| :--- | :---: | :---: | :---: | :---: |
| 3.0_a | 12871 | 969 | 0.633 | 0.150 |
| 2.0_a | 11464 | 2376 | 0.296 | 0.119 |
| 11.0_b | 12777 | 1063 | 0.187 | 0.093 |
| 12.0_a | 13214 | 626 | 0.507 | 0.062 |
| 10.0_b | 12048 | 1792 | 0.172 | 0.054 |
| 1.0_b | 12777 | 1063 | 0.118 | -0.034 |
| 13.0_a | 11600 | 2240 | 0.601 | -0.047 |
| 6.0_a | 12162 | 1678 | 0.673 | -0.105 |
| 5.0_b | 13006 | 834 | 0.076 | -0.158 |
| 14.0_a | 12641 | 1199 | 0.748 | -0.345 |
| **Promedio** | - | - | **0.401** | **-0.021** |

#### Luminancia
| Fold (Test Video) | Train N | Test N | RMSE | Corr |
| :--- | :---: | :---: | :---: | :---: |
| lum_009_3_b | 3690 | 615 | 0.226 | 0.039 |
| lum_006_4_a | 3690 | 615 | 0.171 | -0.029 |
| lum_002_3_a | 3690 | 615 | 0.523 | -0.102 |
| lum_010_4_b | 3690 | 615 | 0.148 | -0.151 |
| lum_007_3_b | 3690 | 615 | 0.379 | -0.288 |
| lum_004_4_a | 3690 | 615 | 0.582 | -0.408 |
| lum_003_3_a | 3690 | 615 | 0.397 | -0.439 |
| **Promedio** | - | - | **0.347** | **-0.197** |

- **Conclusión General:** Se observa una alta variabilidad entre folds (videos). Algunos videos tienen correlación positiva moderada (ej. Arousal 8.0_a con 0.52), pero otros fuertemente negativa, cancelándose en el promedio. Esto sugiere que no hay algo concreto y robusto en el EEG crudo que prediga el joystick a través de distintos videos.
De hecho, esto queda claro mirando los plots de las series de tiempo reales (anotadas con el joystick) y predichas (segun la data de EEG):

#### Visualización de Predicciones: Casos Representativos

**Arousal:**
*Mejor Caso (8.0_a, Corr=0.52)* vs *Peor Caso (14.0_b, Corr=-0.67)*
![Arousal Best](../results/modeling/predictions_timeseries/arousal/pred_ts_arousal_8_0_a.png)
![Arousal Worst](../results/modeling/predictions_timeseries/arousal/pred_ts_arousal_14_0_b.png)

**Valencia:**
*Mejor Caso (3.0_a, Corr=0.15)* vs *Peor Caso (14.0_a, Corr=-0.34)*
![Valence Best](../results/modeling/predictions_timeseries/valence/pred_ts_valence_3_0_a.png)
![Valence Worst](../results/modeling/predictions_timeseries/valence/pred_ts_valence_14_0_a.png)

**Luminancia:**
*Mejor Caso (lum_009_3_b, Corr=0.04)* vs *Peor Caso (lum_004_4_a, Corr=-0.41)*
![Luminance Best](../results/modeling/predictions_timeseries/luminance/pred_ts_luminance_lum_009_3_b.png)
![Luminance Worst](../results/modeling/predictions_timeseries/luminance/pred_ts_luminance_lum_004_4_a.png)

### Experimento 2: Clasificación Binaria (Estado)
- **Objetivo:** Clasificar el estado momentáneo como Positivo (>0) o Negativo/Bajo (<=0).
- **Modelo:** `RidgeClassifier`.
- **Lógica:** $y_{class} = 1$ si $y_{cont} > 0$ sino 0.
- **Resultados:**
  - La precisión (Accuracy) rondó el 50-60% (Nivel de azar).
  - Los puntajes **AUC** estuvieron consistentemente cerca de 0.5.
- **Conclusión:** Binarizar el valor absoluto no mejoró la detectabilidad. Esto sugiere que el nivel *absoluto* del joystick podría tener deriva o ser idiosincrásico, haciendo que el *cambio* (tendencia) sea potencialmente más informativo.

### Test 3: Clasificación de Tendencia (Dinámica)
- **Objetivo:** Predecir si el sujeto está *aumentando* o *disminuyendo* su valoración.
- **Lógica:** Calcular diferencia $\Delta = y_t - y_{t-1}$.
  - **Inicial:** Binario forzado (Subida vs Bajada). Resultado: Pobre.
  - **Refinado (Filtrado):** Se introdujo un umbral de estabilidad $\epsilon = 0.01$.
    - Si $|\Delta| \le 0.01$, el epoch es "Estable" y se excluye.
    - Si $\Delta > 0.01 \rightarrow$ **Subida (1)**.
    - Si $\Delta < -0.01 \rightarrow$ **Bajada (0)**.
- **Dataset:** Se expandió para incluir **Adquisición B** (Runs 007, 009, 010) para aumentar el tamaño de la muestra.
- **Resultados (Dataset Combinado):**
  - **Arousal:** Balanced Acc = 0.50 | AUC = 0.50 (Azar)
  - **Valencia:** Balanced Acc = 0.49 | AUC = 0.49 (Azar)
  - **Luminancia:** Balanced Acc = 0.54 | AUC = 0.55 (Ligera señal)
- **Conclusión:** Incluso modelando explícitamente la dinámica, el modelo falló. Esto apuntó fuertemente a un problema fundamental con los features de entrada (PCA de Series Temporales Crudas) más que con la lógica de etiquetado.

## 4. Exploración de Datos (El Pivote)

Para entender *por qué* fallaban los modelos, realizamos una exploración visual de la estructura de los datos.

### 4.1 Visualización de Series Temporales
- **Observación:** Los reportes de joystick (etiquetas) muestran trayectorias dinámicas claras y suaves que varían significativamente por video. El "ground truth" contiene señal.
- **Acción:** Se generaron plots por dimensión (`results/exploration/timeseries_*.png`).

### 4.2 Inspección del Espacio Latente PCA
- **Método:** Se proyectaron epochs en los primeros 3 Componentes Principales (PC1, PC2, PC3).
- **Coloreo:**
  - Por Valor de Joystick (Continuo).
  - Por Clase de Tendencia (Subida/Bajada).
- **Observación:**
  - Los gráficos muestran una "nube mezclada" (sin separación clara de colores).
  - Valores altos y bajos del joystick se solapan casi perfectamente en el espacio de componentes.
  - **Implicancia:** Las direcciones de máxima varianza en el EEG crudo (PC1, PC2) **no se alinean** con las dimensiones afectivas. La varianza dominante es probablemente ruido fisiológico o ritmo de fondo, no el efecto buscado.

### 4.3 Análisis de Varianza Explicada
- **Método:** PCA Global sobre todos los datos.
- **Resultado:**
  - Los primeros 2-3 componentes explican una fracción muy pequeña de la varianza total (< 10%).
  - La curva de Varianza Acumulada (`results/exploration/pca_global_cumulative_variance_logx.png`) crece lentamente.
- **Interpretación Matemática:**
  $$X_{eeg} = S_{afecto} + N_{ruido}$$
  En EEG crudo, $Var(N_{ruido}) \gg Var(S_{afecto})$. El PCA maximiza la varianza, por lo que los PCs 1-100 están modelando principalmente $N_{ruido}$. La señal predictiva $S_{afecto}$ está enterrada en componentes superiores o, más probablemente, en bandas de frecuencia específicas que el PCA en el dominio del tiempo ignora.

## 5. Próximos Pasos: Features Espectrales
Basándonos en la conclusión de que **el PCA en el Dominio del Tiempo está dominado por ruido**, la estrategia cambia a **Ingeniería de Features en el Dominio de la Frecuencia**.
- **Plan:**
  1.  **Extraer Band Power:** Descomponer la señal en bandas Delta, Theta, Alpha, Beta, Gamma.
  2.  **Racional:** Los estados afectivos están fisiológicamente vinculados a oscilaciones específicas (ej. asimetría Alpha para Valencia, Beta/Gamma para Arousal). Esto actúa como un filtro biológico robusto antes de que el modelo vea los datos.
  3.  **Refinamiento:** Probar lags temporales (250ms - 2000ms) para tener en cuenta el tiempo de reacción entre el estado cerebral y el movimiento del joystick.
