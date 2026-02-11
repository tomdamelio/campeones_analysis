# Documento de Requisitos: Predicción de Luminancia Real desde EEG

## Introducción

Este feature implementa un pipeline de modelos predictivos que predicen la luminancia física real (extraída de los videos de estímulo) a partir de señales EEG. El objetivo es validar que el enfoque de modelado predictivo puede capturar un estímulo físico externo claro (luz) antes de intentar predecir constructos internos complejos como arousal/valencia. El pipeline incluye exploración de datos de luminancia, verificación de marcas de estímulos, un modelo base dummy, modelos con features espectrales en electrodos posteriores/occipitales, y Time Delay Embedding (TDE) con PCA.

## Glosario

- **Sistema_Exploración**: Módulo de exploración y visualización de series de tiempo de luminancia extraídas de los videos de estímulo.
- **Sistema_Verificación**: Módulo de verificación y debugging del mapeo entre marcas de eventos EEG (trial_type `video_luminance`) y los archivos CSV de luminancia correspondientes a cada video.
- **Sistema_Sincronización**: Módulo encargado de alinear temporalmente las señales EEG con los valores de luminancia del video, generando épocas sincronizadas.
- **Sistema_Modelo_Base**: Pipeline predictivo base que usa señal EEG cruda vectorizada como features (X) y luminancia real como target (y), con épocas de 500ms y 400ms de solapamiento.
- **Sistema_Modelo_Espectral**: Pipeline predictivo que extrae features espectrales (bandas de potencia) de electrodos posteriores/occipitales como features (X) y luminancia real como target (y).
- **Sistema_TDE**: Módulo de Time Delay Embedding que expande features espectrales con una ventana temporal de ±10 puntos, seguido de PCA para reducción de dimensionalidad.
- **Pipeline_CV**: Esquema de validación cruzada Leave-One-Video-Out para evaluar la generalización de los modelos.
- **ROI_Posterior**: Región de interés compuesta por electrodos posteriores y occipitales: O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6 (revisar que tenga todos estos electrodos posteriores para la version final del ROI a utilizar)
- **Luminancia_CSV**: Archivos CSV con columnas `timestamp` y `luminance` (canal verde, 0-255) extraídos frame a frame (~60fps) de los videos de estímulo.
- **Order_Matrix**: Archivo Excel que contiene el mapeo entre bloques experimentales, videos presentados y dimensiones evaluadas para cada sujeto.

## Requisitos

### Requisito 1: Exploración de Series de Tiempo de Luminancia

**User Story:** Como investigador, quiero explorar visualmente las series de tiempo de luminancia de los 4 videos experimentales, para entender la dinámica temporal del estímulo físico antes de modelar.

#### Criterios de Aceptación

1. WHEN el Sistema_Exploración recibe los archivos Luminancia_CSV de los videos 3, 7, 9 y 12, THE Sistema_Exploración SHALL generar un gráfico de serie de tiempo de valores de luminancia cruda para cada video.
2. WHEN el Sistema_Exploración procesa un archivo Luminancia_CSV, THE Sistema_Exploración SHALL calcular las diferencias temporales (diff) entre muestras consecutivas (t y t+1) y generar un gráfico de dichas diferencias para cada video.
3. WHEN el Sistema_Exploración genera gráficos, THE Sistema_Exploración SHALL guardar las figuras en el directorio de resultados con nombres que identifiquen el video correspondiente.
4. WHEN el Sistema_Exploración carga un archivo Luminancia_CSV, THE Sistema_Exploración SHALL reportar estadísticas descriptivas básicas (media, desviación estándar, mínimo, máximo, duración total) de la serie de luminancia.

### Requisito 2: Verificación de Marcas de Estímulos de Luminancia

**User Story:** Como investigador, quiero verificar que las marcas de eventos `video_luminance` en los archivos de eventos EEG del sujeto 27 correspondan correctamente a los videos green_intensity_video_3, 7, 9 y 12, para asegurar la integridad del mapeo estímulo-dato.

#### Criterios de Aceptación

1. WHEN el Sistema_Verificación carga los eventos EEG y la Order_Matrix para cada run del sujeto 27 (Acq A y B), THE Sistema_Verificación SHALL identificar todos los eventos con trial_type `video_luminance` y reportar su onset, duración y valor.
2. WHEN el Sistema_Verificación compara los eventos `video_luminance` con la Order_Matrix, THE Sistema_Verificación SHALL determinar qué video_id de luminancia corresponde a cada evento en cada run.
3. IF el Sistema_Verificación detecta una discrepancia entre el video_id esperado (según Order_Matrix) y el video asignado, THEN el Sistema_Verificación SHALL reportar la discrepancia con detalle del run, onset, video esperado y video encontrado.
4. WHEN el Sistema_Verificación completa el análisis, THE Sistema_Verificación SHALL generar un reporte consolidado con el mapeo run → video_luminance → video_id para ambas adquisiciones (A y B).

### Requisito 3: Modelo Predictivo Base con Luminancia Real

**User Story:** Como investigador, quiero implementar un modelo predictivo base que use señal EEG como X y luminancia física real como y, para establecer una línea base de predicción de estímulos físicos.

#### Criterios de Aceptación

1. WHEN el Sistema_Sincronización recibe datos EEG y un archivo Luminancia_CSV para un segmento de video, THE Sistema_Sincronización SHALL alinear temporalmente ambas señales usando el onset del evento como punto de referencia.
2. WHEN el Sistema_Sincronización genera épocas, THE Sistema_Sincronización SHALL crear ventanas de 500ms con 400ms de solapamiento (paso de 100ms).
3. WHEN el Sistema_Sincronización calcula el target de luminancia para cada época, THE Sistema_Sincronización SHALL promediar los valores de luminancia dentro de la ventana temporal correspondiente.
4. WHEN el Sistema_Modelo_Base recibe las épocas sincronizadas, THE Sistema_Modelo_Base SHALL construir un pipeline de Vectorizer → StandardScaler → PCA → Ridge regression.
5. WHEN el Pipeline_CV evalúa el Sistema_Modelo_Base, THE Pipeline_CV SHALL usar validación cruzada Leave-One-Video-Out y reportar Pearson r, Spearman rho y RMSE por fold y promedio.
6. WHEN el Sistema_Modelo_Base completa la evaluación, THE Sistema_Modelo_Base SHALL guardar los resultados en un archivo CSV en el directorio de resultados.
7. IF el Sistema_Sincronización no encuentra un archivo Luminancia_CSV correspondiente al video de un run, THEN el Sistema_Sincronización SHALL reportar el error y omitir ese segmento sin interrumpir el pipeline.

### Requisito 4: Modelo con Features Espectrales en Electrodos Posteriores/Occipitales

**User Story:** Como investigador, quiero implementar modelos predictivos que extraigan features espectrales de electrodos posteriores/occipitales, para capturar mejor la respuesta cortical visual a cambios de luminancia.

#### Criterios de Aceptación

1. WHEN el Sistema_Modelo_Espectral recibe épocas EEG, THE Sistema_Modelo_Espectral SHALL extraer potencia espectral en bandas estándar (delta, theta, alpha, beta, gamma) para cada electrodo de la ROI_Posterior.
2. WHEN el Sistema_Modelo_Espectral selecciona electrodos, THE Sistema_Modelo_Espectral SHALL usar exclusivamente los canales de la ROI_Posterior (O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6). Definir de todas formas este ROI_Posterior chequeando que estos channels especificados existan.
3. WHEN el Sistema_Modelo_Espectral construye el vector de features, THE Sistema_Modelo_Espectral SHALL concatenar la potencia de cada banda para cada electrodo de la ROI_Posterior, resultando en un vector de dimensión (número_de_bandas × número_de_electrodos).
4. WHEN el Pipeline_CV evalúa el Sistema_Modelo_Espectral, THE Pipeline_CV SHALL usar validación cruzada Leave-One-Video-Out y reportar Pearson r, Spearman rho y RMSE por fold y promedio.
5. WHEN el Sistema_Modelo_Espectral completa la evaluación, THE Sistema_Modelo_Espectral SHALL guardar los resultados y generar gráficos comparativos con el Sistema_Modelo_Base.

### Requisito 5: Time Delay Embedding (TDE) con PCA

**User Story:** Como investigador, quiero aplicar Time Delay Embedding a los features espectrales seguido de PCA, para capturar la dinámica temporal de la respuesta cortical y mejorar la predicción de luminancia.

#### Criterios de Aceptación

1. WHEN el Sistema_TDE recibe features espectrales de una época, THE Sistema_TDE SHALL expandir cada muestra con una ventana de ±10 puntos de tiempo alrededor de la muestra actual.
2. WHEN el Sistema_TDE expande los features, THE Sistema_TDE SHALL concatenar los vectores de features de los 21 puntos de tiempo (muestra actual + 10 anteriores + 10 posteriores) en un único vector expandido.
3. WHEN el Sistema_TDE genera el vector expandido, THE Sistema_TDE SHALL aplicar PCA inmediatamente después para reducir la dimensionalidad del espacio de features expandido.
4. WHEN el Pipeline_CV evalúa el modelo con Sistema_TDE, THE Pipeline_CV SHALL usar validación cruzada Leave-One-Video-Out y reportar Pearson r, Spearman rho y RMSE por fold y promedio.
5. WHEN el Sistema_TDE completa la evaluación, THE Sistema_TDE SHALL guardar los resultados y generar gráficos comparativos con el Sistema_Modelo_Espectral (sin TDE) para demostrar la mejora en performance.

### Requisito 6: Configuración y Reproducibilidad del Pipeline

**User Story:** Como investigador, quiero que todos los parámetros del pipeline sean configurables y que los resultados sean reproducibles, para facilitar la iteración experimental.

#### Criterios de Aceptación

1. THE Sistema_Modelo_Base SHALL leer todos los parámetros (sujeto, sesión, runs, rutas, parámetros de épocas, componentes PCA, alpha de Ridge) desde un módulo de configuración centralizado.
2. THE Sistema_Modelo_Base SHALL fijar una semilla aleatoria configurable al inicio de cada ejecución para garantizar reproducibilidad.
3. WHEN cualquier script del pipeline se ejecuta, THE script SHALL usar rutas relativas al directorio raíz del proyecto, sin rutas absolutas hardcodeadas.
4. WHEN el pipeline genera resultados, THE pipeline SHALL guardar los resultados en `results/modeling/luminance/` con nombres que identifiquen el sujeto, modelo y parámetros usados.
