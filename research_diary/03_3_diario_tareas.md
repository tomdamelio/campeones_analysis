# Diario de Tareas — Decoding EEG de Luminancia (2026-03-25)

**Proyecto:** campeones_analysis
**Fecha:** 2026-03-25
**Sujeto analizado:** sub-27

---

## Contexto del proyecto

Este proyecto analiza señal EEG de sujetos expuestos a cambios de luminancia controlados en un entorno de realidad virtual. El estímulo clave es el evento **CHANGE_PHOTO**: un cambio abrupto de luminancia que dura 1 segundo y luego revierte. La estructura temporal de cada trial es:

```
  -500ms         0ms           500ms         1000ms        1500ms
    |              |              |              |              |
[Baseline]     [Onset de      [Luminancia   [Retorno a
(pre-onset)     luminancia]    sostenida]    baseline]
```

El pipeline de preprocesamiento produce datos a **250 Hz** con 32 canales EEG. El análisis de decoding busca cuánta información sobre el estado del estímulo (o del cerebro) puede extraerse de la señal EEG usando clasificadores lineales con validación cruzada.

---

## Esquema general de decoding

### Cross-validación: Leave-One-Run-Out (LORO)

Cada sesión tiene 7 runs de EEG. El esquema LORO usa uno de esos runs como test y los 6 restantes como train, y repite esto 7 veces. Esto garantiza que el modelo nunca ve datos del mismo run en train y test — importante porque los runs tienen distinta estabilidad de señal (drift, impedancia, estado del sujeto).

### Feature sets comparados

Se comparan tres tipos de features extraídos de cada ventana temporal de EEG:

**1. `bandpower_welch` (160 features):**
Para cada ventana, se calcula la densidad espectral de potencia (Welch) por canal y se integra en 5 bandas: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-45 Hz). Con 32 canales → 32×5 = 160 features. Captura directamente cuánta potencia hay en cada banda de cada electrodo. Es el feature más interpretable neurológicamente.

**2. `tde_cov` (153 features):**
Pipeline en 3 pasos:
- *Time-Delay Embedding (TDE):* Para cada canal, se concatenan copias de la señal desplazadas ±10 lags (±40ms a 250 Hz). Esto convierte cada canal en 21 dimensiones temporales → el array pasa de (T × 32 canales) a (T × 672 dimensiones TDE). El TDE captura la estructura temporal local (autocorrelaciones, relaciones entre canales con distintos retardos).
- *PCA global:* Se ajusta una PCA sobre todos los datos de entrenamiento del fold (no sobre la ventana individual), reduciendo de 672 a 17 componentes principales. La PCA se ajusta solo en train para evitar data leakage.
- *Covarianza:* Para cada ventana de 250ms (~62 muestras), se calcula la covarianza de los 17 PCs → upper triangle = 17×18/2 = **153 features**. La diagonal son las potencias de cada PC; los off-diagonals son las covarianzas (coherencia/conectividad entre PCs).

**3. `raw_pca` (160 features):**
La señal cruda de la ventana (32 canales × 62 muestras = 1984 valores) se vectoriza y se reduce con PCA(160) ajustada en train. Captura patrones espaciotemporales crudos sin suposiciones sobre frecuencias o estructura temporal.

### Equiparación de features

Para que la comparación entre modelos sea justa, los tres feature sets tienen aproximadamente el mismo número de features (~160). Con diferente número de features, la regularización óptima (C en regresión logística) es distinta, y usar el mismo C favorece sistemáticamente al modelo con menos features.

- bandpower: 32×5 = 160 (fijo por diseño)
- tde_cov: `TDE_PCA_COMPONENTS=17` → 17×18/2 = 153 ≈ 160
- raw_pca: `RAW_PCA_COMPONENTS=160` → 160 (reducción explícita)

### Clasificador: LogisticRegressionCV

En lugar de fijar C=1.0 arbitrariamente, se usa `LogisticRegressionCV` de scikit-learn, que cross-valida C internamente sobre el training set de cada fold LORO (inner CV 5-fold, grid C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]). Esto permite que cada feature set elija el nivel de regularización óptimo para su escala y dimensionalidad, haciendo la comparación más justa.

---

## Antecedentes de Tareas 9.3 y 10 (diario 03_2)

Las tareas anteriores evaluaron dos contrastes binarios en el mismo dataset:

- **Tarea 9.3 — Pre vs Post estímulo:** Clasificar 500ms de señal pre-onset (baseline) vs 500ms post-onset (CHANGE). Mismos 74 onsets, épocas del mismo trial. Resultado: bandpower 68.9% accuracy / AUC=0.725, tde_cov 62.8%, raw_pca 65.5% (chance=50%).

- **Tarea 10.2 — Barrido temporal con ventanas de 50ms:** Deslizar ventanas de 50ms desde 0 hasta 500ms post-onset en pasos de 10ms, entrenando un modelo por posición. Pico ~69% accuracy entre 130-330ms post-onset según feature set.

La conclusión general fue que la señal discriminativa es robusta (hay información real en el EEG sobre el estado del estímulo), concentrada en la modulación de potencia alfa en canales occipitales, y que el contraste "pre vs post" es el más limpio pero no el más realista para decoding continuo.

---

## Tarea 9.3 — Actualización: Fix de estandarización + paralelización (2026-03-25)

Durante el desarrollo de la Tarea 10.3 (ver abajo) se descubrió un bug conceptual en el pipeline de TDE-cov que afectaba también al script de pre/post. Se documentan aquí los fixes y sus efectos.

### Bug descubierto: `standardise_pc=True` destruye información de potencia

El pipeline TDE-cov llama a `apply_global_pca(segment, pca_model, standardise_pc=True)` donde el parámetro `standardise_pc` (activado por defecto) normaliza cada componente principal a media=0 y desviación estándar=1 **dentro de la ventana temporal**:

```python
# Dentro de apply_global_pca, si standardise_pc=True:
projected -= projected.mean(axis=0)   # resta media temporal del PC en esa ventana
projected /= projected.std(axis=0)    # divide por std temporal → std queda en 1.0
```

**Consecuencia crítica:** si forzamos std=1 en cada PC, entonces la varianza de ese PC en esa ventana es siempre 1.0, **por definición matemática**. La diagonal de la matriz de covarianza — que representa la potencia de cada componente principal — se colapsa en un vector constante de 1s, igual para todas las ventanas y todas las clases. Se destruyen así los 17 features más informativos (potencia por PC).

Lo que queda son solo los 136 features off-diagonal (covarianza entre pares de PCs), que capturan coherencia/conectividad entre componentes. Para una tarea donde la señal principal es el cambio de potencia (alfa occipital), esto es una pérdida importante.

**Analogía:** es como normalizar todas las imágenes a la misma intensidad media antes de clasificar "luz" vs "oscuridad" — el clasificador pierde justamente la información más relevante.

**Diferencia con standardise_pc=False:**
- Con `False`: diagonal = varianza real de cada PC en esa ventana (varía entre ventanas y clases → informativo)
- Con `True`: diagonal = 1.0 siempre (constante → no informativo)

**Situación en cada script:**

| Script | Estado antes del fix | Efecto |
|---|---|---|
| `27b_decoding_pre_vs_post.py` | `standardise_pc=True` en train Y test (consistente pero subóptimo) | Sin colapso, pero pierde info de potencia |
| `34_decoding_4class.py` | `standardise_pc=False` en train, `True` en test (mismatch) | Colapso total del clasificador |

El mismatch en script 34 fue el error más grave: el modelo aprendía con features que incluían potencia real (diagonal variable), pero al predecir recibía features con diagonal=1. Es como entrenar en kilos y predecir en libras — el espacio de features es fundamentalmente diferente.

**Fix:** pasar `standardise_pc=False` en todas las llamadas a `apply_global_pca`, tanto en train como en test, refactorizando el código para que ambas llamadas estén dentro de la misma función y no puedan desincronizarse.

### Paralelización de los folds LORO con joblib

Los 7 folds LORO son completamente independientes entre sí (cada fold tiene su propio train/test split, su propio PCA, su propio clasificador). Se paralelizaron con `joblib.Parallel`:

```python
from joblib import Parallel, delayed

fold_results = Parallel(n_jobs=-1)(          # usa todos los cores disponibles
    delayed(_run_one_fold_tde)(test_idx, ...)
    for test_idx in range(n_runs)
)
```

Detalles de implementación:
- El loop de folds se extrae a funciones independientes (`_run_one_fold_standard`, `_run_one_fold_tde`) que reciben todos los datos que necesitan por argumento — sin estado compartido.
- Los imports de las funciones de TDE/PCA se hacen **dentro** de cada función de fold, para que el backend `loky` (procesos separados) pueda serializarlos correctamente en cada worker.
- `n_jobs=1` dentro de `LogisticRegressionCV` para evitar sobre-suscripción: si el loop externo ya usa todos los cores, el inner CV no debe también hacerlo (causaría 7×5=35 procesos compitiendo).
- Ganancia estimada: ~4-5× en tiempo de ejecución (los 7 folds corren en paralelo en lugar de secuencialmente).

### Resultados actualizados — script 27b (sub-27, LORO 7-fold)

Tarea binaria: pre-estímulo (clase 0, ventana [−550, −50ms]) vs post-estímulo (clase 1, ventana [+50, +550ms]). 74 onsets × 2 = 148 épocas de 500ms. Chance = 50%.

| Feature set | N features | Acc | F1 | AUC |
|---|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | 0.689 | **0.725** |
| raw_pca | 160 | 65.5% | 0.662 | 0.696 |
| tde_cov (fix: standardise_pc=False) | 153 | 62.2% | 0.600 | 0.624 |

**Comparación antes/después del fix en tde_cov:**

| | Acc | AUC |
|---|---|---|
| Antes (standardise_pc=True, consistente) | 62.8% | 0.685 |
| Después (standardise_pc=False, consistente) | 62.2% | 0.624 |
| Δ | −0.6 pp | −0.061 |

El fix levemente **empeora** tde_cov en esta tarea. Esto es esperable: en 27b el código anterior era consistente (mismo parámetro en train y test), así que no había mismatch — era solo subóptimo. Para la tarea binaria pre/post, la estructura de correlación entre PCs (lo que da standardise_pc=True, equivalente a una matriz de correlación) resulta marginalmente más informativa que la covarianza completa. La razón posible: la señal principal para discriminar pre/post es el cambio de potencia alfa, que bandpower captura directamente; lo que TDE añade sobre bandpower son las correlaciones entre PCs (coherencia), y eso se pierde con standardise_pc=False porque escala las features de potencia hasta dominar. En cambio, en la tarea de 4 clases (donde los 4 estados tienen distintas potencias por PC), el fix es mucho más relevante (ver Tarea 10.3).

**Archivos:**
- `scripts/validation/27b_decoding_pre_vs_post.py`
- `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_pre_vs_post_results.json`
- `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_decoding_summary.png`

---

## Tarea 10.3: Decoding de 4 Clases Temporales

### Motivación y diseño conceptual

Diego Vidaurre propuso reformular el decoding como una tarea de **4 clases** que mapeen los 4 estados temporales del protocolo:

| Clase | Nombre | Ventana temporal | Estado del estímulo |
|---|---|---|---|
| 0 | Baseline | [−500, 0ms] pre-onset | Sin cambio, luminancia de fondo |
| 1 | ChangeUp | [0, 500ms] post-onset | Aumento de luminancia (onset) |
| 2 | Luminance | [500, 1000ms] post-onset | Luminancia elevada sostenida |
| 3 | ChangeDown | [1000, 1500ms] post-onset | Retorno a luminancia de fondo (offset) |

La motivación de Diego: esta formulación es más cercana al escenario de **decoding continuo** real (donde el modelo tiene que predecir el estado del cerebro en cualquier momento del tiempo), y sirve como benchmark antes de pasar a estímulos más complejos (pantallas verdes, donde la luminancia varía de forma continua durante 1 minuto). Las 4 clases además están balanceadas por diseño (mismo número de ventanas por clase por trial).

**Ventana deslizante:** Para maximizar el uso de datos, en lugar de tomar un único feature vector por clase por trial, se extraen múltiples ventanas deslizantes dentro de cada segmento de 500ms. Con ventanas de 250ms y paso de 50ms (80% de solapamiento), cada segmento de 500ms produce **6 ventanas**:

```
Segmento de 500ms con ventanas de 250ms, paso 50ms:
|----250ms----|
     |----250ms----|
          |----250ms----|
               |----250ms----|
                    |----250ms----|
                         |----250ms----|
→ 6 ventanas, perfectamente balanceadas entre las 4 clases
→ 24 ventanas por trial en total
→ 74 trials × 24 = ~1776 ventanas totales
```

**Épocas wide:** [−1.5, +2.0]s con baseline (−1.5, −1.0)s. El rango amplio es necesario para el TDE (que consume ±10 lags = ±40ms en los bordes) y para la ventana de Baseline que empieza en −500ms.

**Script:** `scripts/validation/34_decoding_4class.py --subject 27`

---

### Tarea 10.3.1: Intento inicial con ventanas de 100ms — diagnóstico

El diseño original del script usaba ventanas de 100ms con paso de 50ms (50% de solapamiento). Los resultados fueron:

| Feature set | N feat | Acc | F1 | AUC |
|---|---|---|---|---|
| bandpower_welch | 160 | 27.5% | 0.271 | 0.541 |
| tde_cov | 153 | 28.3% | 0.282 | 0.538 |
| raw_pca | 800* | 30.0% | 0.290 | 0.560 |

*Chance = 25.0%* — Prácticamente al azar para los tres modelos.

*\*raw_pca tenía 800 features por un error de configuración; luego se ajustó a 160.*

**Diagnóstico de por qué 100ms no funciona:**

**1. Resolución espectral insuficiente para bandpower:**
A 250 Hz, 100ms = 25 muestras. La resolución de Welch es `fs/N = 250/25 = 10 Hz/bin`. Las bandas que queremos distinguir (delta 1-4Hz, theta 4-8Hz, alpha 8-13Hz) tienen todas 0-1 bins en esta resolución. Es imposible estimar bandpower por bandas con ventanas tan cortas — todas las bandas quedan colapsadas en los mismos 2-3 bins.

**2. Covarianza mal condicionada para tde_cov:**
Con 25 muestras y 17 componentes PCA, se intenta estimar una matriz de covarianza de 17×17 = 289 entradas a partir de solo 25 observaciones. Una covarianza de dimensión 17 necesita al menos 17 muestras para ser de rango completo; con 25 muestras (solo 8 más que la dimensión), la estimación es extremadamente ruidosa e inestable.

**3. Error técnico en raw_pca:**
El cálculo de índices de ventana con aritmética flotante producía ventanas de 25 muestras en algunos casos y 26 en otros:
```python
# Código con bug:
s0 = int(round(t_start * sfreq))
s1 = int(round(t_end * sfreq))  # ← a veces 25, a veces 26

# Fix: longitud fija
s1 = s0 + int(round(WIN_SIZE_S * sfreq))  # siempre 25 o siempre 62
```
Esto generaba `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape`.

**Decisión:** cambiar a ventanas de 250ms con 80% de solapamiento (paso 50ms). 250ms a 250 Hz = 62 muestras → resolución Welch ~4 Hz/bin (suficiente para distinguir todas las bandas). La covarianza con 62 muestras y 17 PCs es mucho mejor condicionada (ratio 62/17 ≈ 3.6). El solapamiento del 80% produce el mismo balance de 6 ventanas/segmento/clase con 250ms de paso.

---

### Tarea 10.3.2: Debugging del pipeline con ventanas de 250ms

Al pasar a 250ms y empezar a debuggear el código, se descubrió el problema de `standardise_pc` descrito arriba, más algunos otros problemas de diseño.

**Problema adicional — Baseline desequilibrado:**
El diseño inicial de Baseline usaba dos sub-segmentos: [−250, 0ms] + [1500, 1750ms] (pooled), inspirado en la idea de que el baseline post-trial es análogo al pre-trial. Con ventanas de 100ms/50ms, esto daba 8 ventanas para Baseline vs 9 para las otras clases. Al rediseñar como un único segmento [−500, 0ms] de 500ms, se obtienen exactamente 6 ventanas/clase → balance perfecto.

**El bug de `standardise_pc` y su efecto en 4 clases:**

Al correr tde_cov con el código original (standardise_pc=True por defecto en todas las llamadas), los resultados de tde_cov eran idénticos a bandpower: Acc=30.9%, AUC=0.570. Esto es sospechoso — dos modelos con features completamente distintos raramente dan exactamente el mismo resultado, y en este caso el número coincidía también con otros runs previos.

La investigación reveló el bug de standardise_pc: con la diagonal de la covarianza siempre =1, los 153 features de tde_cov eran efectivamente 136 features off-diagonal (coherencia entre PCs) + 17 constantes (la diagonal). El clasificador ignoraba las constantes y solo usaba los 136 de coherencia, que para esta tarea no son suficientemente informativos.

Al intentar corregir esto (agregar `standardise_pc=False` en las líneas de train pero olvidarlo en el test), se produjo el mismatch que causó el colapso completo:

```
# Con standardise_pc inconsistente (train=False, test=True):
Acc=0.247  F1=0.165  AUC=0.524
Per-class: Baseline=0.0, ChangeUp=0.0, Luminance=0.441, ChangeDown=0.547
```

El modelo predecía solo 2 de las 4 clases. La razón: en train, la diagonal de la covarianza era variable (potencia real por PC, con valores de, por ejemplo, 0.5 a 50 µV²). En test, la diagonal era siempre 1.0. Los pesos del clasificador aprendidos sobre features de escala "potencia real" producían predicciones sin sentido cuando recibían features de escala forzada a 1.

**Fix definitivo:** refactorizar el loop de folds en una función `_run_one_fold_tde()` que contenga todas las llamadas a `apply_global_pca` — tanto train como test — con `standardise_pc=False` explícito. Al tener ambas en la misma función con el mismo parámetro, es estructuralmente imposible que vuelvan a desincronizarse.

---

### Tarea 10.3.3: Resultados finales (250ms, 80% solapamiento, standardise_pc=False)

**Dataset:** sub-27, 74 trials CHANGE_PHOTO, 24 ventanas/trial, ~1776 ventanas totales, 7 runs, LORO CV.

#### Resultados globales

| Feature set | N features | Acc | F1 macro | AUC macro |
|---|---|---|---|---|
| bandpower_welch | 160 | 30.9% | 0.308 | 0.570 |
| **tde_cov** | **153** | **35.6%** | **0.357** | **0.622** |
| raw_pca | 160 | 37.4% | 0.373 | 0.614 |

*Chance = 25.0%*. Los tres modelos superan chance de forma consistente. El mejor AUC es tde_cov (0.622), el mejor Acc/F1 es raw_pca.

#### Resultados por fold — tde_cov (mejor AUC)

| Run | n_onsets | Acc | best_C |
|---|---|---|---|
| task-01_acq-a_run-002 | 12 | 31.2% | 0.001 |
| task-02_acq-a_run-003 | 8 | 44.3% | 0.01 |
| task-03_acq-a_run-004 | 10 | 41.2% | 0.01 |
| task-04_acq-a_run-006 | 12 | 25.0% | 0.01 |
| task-01_acq-b_run-007 | 12 | 39.9% | 0.001 |
| task-03_acq-b_run-009 | 8 | 29.7% | 0.001 |
| task-04_acq-b_run-010 | 12 | 39.6% | 0.01 |

#### Resultados por fold — raw_pca (mejor Acc)

| Run | n_onsets | Acc | best_C |
|---|---|---|---|
| task-01_acq-a_run-002 | 12 | 34.7% | 0.01 |
| task-02_acq-a_run-003 | 8 | 43.8% | 0.01 |
| task-03_acq-a_run-004 | 10 | 41.2% | 0.001 |
| task-04_acq-a_run-006 | 12 | 30.9% | 0.01 |
| task-01_acq-b_run-007 | 12 | 40.6% | 0.01 |
| task-03_acq-b_run-009 | 8 | 38.5% | 0.01 |
| task-04_acq-b_run-010 | 12 | 35.4% | 0.01 |

#### Precisión por clase (qué clases se decodifican mejor)

| Feature set | Baseline (0) | ChangeUp (1) | Luminance (2) | ChangeDown (3) |
|---|---|---|---|---|
| bandpower_welch | 34.7% | 26.4% | 25.2% | 37.4% |
| tde_cov | 34.0% | 34.0% | 36.5% | 37.8% |
| raw_pca | 27.9% | **44.4%** | **44.8%** | 32.7% |

**Archivos generados:**
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_results.json`
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_confusion.png`
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_per_class_acc.png`

---

## Interpretación General

### 1. La tarea de 4 clases es decodificable pero intrínsecamente difícil

Los 3 feature sets superan chance (25%) de forma consistente: el rango va de 30.9% (bandpower) a 37.4% (raw_pca). La dificultad es esperable: las 4 clases están **temporalmente adyacentes dentro del mismo trial** (cada una ocupa 500ms de un trial de 1.5s), sin un período de "lavado" entre ellas. El cerebro no tiene tiempo de volver al estado de reposo entre clases.

Comparado con la tarea binaria pre/post (68.9% accuracy, chance=50%, es decir +18.9 pp sobre chance), la tarea de 4 clases logra hasta +12.4 pp sobre chance. La relación ganancia/complejidad es razonable.

### 2. tde_cov es el mejor modelo según AUC, con la importancia del fix

Antes del fix de standardise_pc, tde_cov daba exactamente igual que bandpower (30.9% / AUC=0.570). El fix lo lleva a 35.6% / AUC=0.622 — el mejor AUC de los tres modelos. Esto confirma que la **potencia por componente PCA** (la diagonal de la covarianza, que se destruía con standardise_pc=True) es el feature más discriminativo en TDE para esta tarea.

La distribución per-clase de tde_cov es también la más **balanceada** (0.340–0.378 para todas las clases), lo cual es una propiedad deseable en un clasificador multi-clase con clases equiprobables.

### 3. raw_pca es el mejor en Acc y destaca en las clases "de luminancia alta"

raw_pca tiene la mayor precisión global (37.4%) y destaca especialmente en ChangeUp (44.4%) y Luminance (44.8%). Pero Baseline (27.9%) está cerca de chance. La interpretación: la señal cruda captura bien el **onset** de luminancia (hay un ERP transiente marcado en el cambio) y el **estado sostenido** (la amplitud de la señal cambia durante la luminancia alta), pero tiene más dificultad para distinguir el estado pre-onset de los estados post-cambio — posiblemente porque el PCA de señal cruda promedia muchos tipos de variación no relacionados con el estímulo.

### 4. bandpower captura bien los estados de transición, no los de luminancia alta

bandpower tiene mejor precisión en Baseline (34.7%) y ChangeDown (37.4%), pero falla especialmente en Luminance (25.2%, ≈chance) y ChangeUp (26.4%). La hipótesis: el cambio espectral más marcado ocurre entre el estado de fondo y el onset/offset de luminancia (Baseline ↔ ChangeUp, Luminance ↔ ChangeDown). Dentro del período de luminancia elevada, ChangeUp (100-500ms, respuesta visual) y Luminance (500-1000ms, estado sostenido) pueden ser espectralmente similares — ambos tienen alfa suprimido y actividad visual elevada, y el cambio de uno a otro es gradual. Bandpower no tiene sensibilidad temporal fina para capturar esa transición.

### 5. El fold task-04_acq-a_run-006 es sistemáticamente el peor

Este run aparece como el peor fold en todos los análisis de decoding anteriores. En 4 clases: tde_cov=25.0% (chance exacto), raw_pca=30.9%, bandpower=33.3%. Vale la pena investigar si hay algo particular en ese run (impedancia, artefactos, diferente distribución de tipos de trial).

### 6. Alta varianza entre folds: el N por fold es muy bajo

Los folds más grandes tienen 12 onsets → 288 ventanas de test para tde_cov, que es suficiente. Pero los folds con 8 onsets tienen 192 ventanas. La varianza entre folds (ej. task-02: 44.3% vs task-04: 25.0% para tde_cov) es alta, lo que hace que el resultado global (35.6%) sea un promedio sobre una distribución amplia. Esto es esperado con N=74 trials totales y 7 runs — en esta escala, el número de trials por fold de test es pequeño.

---

## Resumen Técnico de Bugs Corregidos

| Bug | Script(s) | Descripción técnica | Efecto observado | Fix aplicado |
|---|---|---|---|---|
| `standardise_pc=True` destruye diagonal de cov | 34, 27b | `apply_global_pca` con default `True` normaliza cada PC a std=1 dentro de la ventana → diagonal de cov = 1 siempre | tde_cov ≡ bandpower (misma acc, AUC idéntico) | `standardise_pc=False` en todas las llamadas |
| Mismatch standardise_pc train/test | 34 | Train: `standardise_pc=False`; test: omitido (default `True`) → espacios de features incompatibles | Colapso: solo 2 de 4 clases predichas; Acc=24.7%, F1=0.165 | Refactorizar en `_run_one_fold_tde()` — ambas llamadas en la misma función |
| Homogeneidad de muestras en ventanas | 34 | `s0=round(t*fs)`, `s1=round((t+w)*fs)` → 25 o 26 muestras según t | `ValueError: inhomogeneous shape` al hacer `np.array(feats)` | `s1 = s0 + int(round(WIN_SIZE_S * sfreq))` — longitud siempre fija |
| Imports residuales en `run_loro_tde_cov` | 27b | `apply_global_pca`, `fit_global_pca`, `compute_epoch_covariance` importadas en la función principal pero ya solo usadas en `_run_one_fold_tde` | Ninguno funcional, solo código muerto | Limpiar: solo importar `apply_tde_only` en `run_loro_tde_cov` |

---

---

## Tarea 10.3.4: Ablación de tde_cov — diagonal vs off-diagonal vs completa (2026-03-25)

### Motivación

La diagonal de la matriz de covarianza TDE-PCA representa la **potencia por componente principal** (17 valores, análogo a bandpower pero en espacio TDE). Los elementos off-diagonal representan la **conectividad/coherencia entre PCs** (136 valores). Como el fix de `standardise_pc=False` mejoró tde_cov principalmente porque restauró la información de potencia (diagonal), la hipótesis es que la diagonal contiene la mayor parte de la señal discriminativa.

**Implementación:** función `_get_cov_mask(k, mode)` que genera una máscara booleana sobre el triángulo superior de la matriz k×k:
- `"full"` → 153 features (diagonal + off-diagonal)
- `"diag"` → 17 features (solo diagonal)
- `"offdiag"` → 136 features (solo off-diagonal)

La máscara se aplica al vector resultante de `compute_epoch_covariance()` antes de entregarlo al clasificador. No se re-calcula la PCA — el coste computacional es mínimo.

### Resultados: script 34 (4 clases, chance=25%)

| Feature set | N features | Acc | F1 macro | AUC macro | Baseline | ChangeUp | Luminance | ChangeDown |
|---|---|---|---|---|---|---|---|---|
| bandpower_welch | 160 | 30.9% | 0.308 | 0.570 | 34.7% | 26.4% | 25.2% | 37.4% |
| tde_cov_full | 153 | 35.6% | 0.357 | 0.622 | 34.0% | 34.0% | 36.5% | 37.8% |
| **tde_cov_diag** | **17** | **37.7%** | **0.377** | **0.643** | 30.9% | **42.8%** | **44.6%** | 32.7% |
| tde_cov_offdiag | 136 | 31.2% | 0.311 | 0.573 | 33.1% | 24.1% | 32.2% | 35.4% |
| raw_pca | 160 | 37.8% | 0.376 | 0.606 | 25.5% | 45.7% | 45.5% | 34.7% |

### Resultados: script 27b (binario pre/post, chance=50%)

| Feature set | N features | Acc | F1 | AUC |
|---|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | 0.689 | **0.725** |
| tde_cov_full | 153 | 62.2% | 0.600 | 0.624 |
| tde_cov_diag | 17 | 61.5% | 0.596 | 0.664 |
| tde_cov_offdiag | 136 | 59.5% | 0.538 | 0.573 |
| raw_pca | 160 | 65.5% | 0.662 | 0.696 |

### Interpretación

**Resultado principal:** `tde_cov_diag` (17 features) **supera** a `tde_cov_full` (153 features) en la tarea de 4 clases en todas las métricas (Acc: 37.7% vs 35.6%, AUC: 0.643 vs 0.622). Esto prueba que:

1. **La potencia por PC (diagonal) es la señal discriminativa en tde_cov.** Los 136 features off-diagonal no solo no ayudan, sino que añaden ruido que perjudica al clasificador.

2. **tde_cov_offdiag ≈ bandpower en rendimiento global** (31.2% vs 30.9% Acc, 0.573 vs 0.570 AUC). La conectividad entre PCs no discrimina los estados temporales de luminancia más que el espectro estándar.

3. **El patrón per-clase de tde_cov_diag se asemeja a raw_pca:** ambos destacan en ChangeUp (42.8% vs 45.7%) y Luminance (44.6% vs 45.5%), sugiriendo que la potencia por PC en el espacio TDE captura el mismo tipo de información espaciotemporal que el ERP crudo.

4. **Para la tarea binaria (27b), la diagonal da peor AUC que full** (0.664 vs 0.624 — espera, peor AUC para full). En realidad la diagonal tiene *mejor* AUC que full en binario también (0.664 > 0.624), aunque peor que bandpower. La diferencia principal es que bandpower captura directamente bandas conocidas (alpha occipital en ~10 Hz) mientras que los PCs del TDE mezclan frecuencias en componentes que pueden no corresponder limpiamente a bandas.

**Conclusión sobre el número de PCs:** dado que la diagonal (17 PCs) supera al full (153), el número de componentes PCA es un hiperparámetro relevante. Con más componentes la diagonal crece linealmente pero cada PC adicional captura menos varianza → posible trade-off rendimiento/ruido. Pendiente: analizar varianza explicada en función del número de PCs para elegir el valor óptimo.

**Archivos:**
- `scripts/validation/34_decoding_4class.py` — 5 feature sets (--features bandpower_welch tde_cov tde_cov_diag tde_cov_offdiag raw_pca)
- `scripts/validation/27b_decoding_pre_vs_post.py` — ídem
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_results.json`
- `results/validation/photo_decoding_pre_vs_post/sub-27/sub-27_pre_vs_post_results.json`

---

## Próximos Pasos

### Paso 1: Elección de número de componentes PCA para tde_cov ✅

El número actual de 17 PCs se eligió para que la diagonal (153 features) ≈ 160 features de bandpower, con fines de equiparación. Pero dado que la ablación mostró que solo la diagonal importa (17 features), la pregunta ahora es: **¿con cuántos PCs se maximiza la información en la diagonal?**

Tradeoff esperado:
- **Pocos PCs (<17):** cada PC explica más varianza, pero se descartan PCs que pueden ser informativos para discriminar clases.
- **Muchos PCs (>17):** cada PC adicional captura varianza residual, que puede ser ruido — la diagonal crece linealmente pero el clasificador tiene más dimensiones irrelevantes.
- **PCs intermedios:** posible sweet spot donde la varianza explicada marginal se vuelve pequeña (codo de la curva) pero los PCs todavía capturan señal relevante.

**Script:** `scripts/validation/35_tde_pca_variance.py` — extrae TDE de todos los runs, ajusta PCA(100), y grafica varianza explicada acumulada y marginal.

---

## Tarea 10.3.5: Comparación de npc = 13, 17, 30 (2026-03-25)

### Motivación

La ablación (10.3.4) mostró que `tde_cov_diag` (solo la diagonal) supera al full con npc=17. Dado que npc=17 se eligió originalmente por equiparación de dimensionalidad (no por optimización), se comparan npc=13 (80% varianza), 17 (83.9%) y 30 (90%) para los tres tipos de features tde_cov.

### Resultados

| Feature | npc=13 (N feat) | Acc | AUC | npc=17 (N feat) | Acc | AUC | npc=30 (N feat) | Acc | AUC |
|---|---|---|---|---|---|---|---|---|---|
| tde_cov_full    | 91  | 33.8% | 0.601 | 153 | 35.6% | 0.622 | 465 | **37.4%** | **0.637** |
| tde_cov_diag    | 13  | 37.3% | 0.626 | 17  | **37.7%** | **0.643** | 30  | 36.5% | 0.629 |
| tde_cov_offdiag | 78  | 28.4% | 0.541 | 136 | 31.2% | 0.573 | 435 | **34.8%** | **0.608** |

*Chance = 25%. Todos los valores son tarea de 4 clases, sub-27, LORO 7-fold.*

### Interpretación

**1. `tde_cov_diag` tiene su pico en npc=17 y baja con npc=30.**
Con más componentes, cada PC explica menos varianza individualmente. La potencia por PC (diagonal) se vuelve menos informativa por unidad — los componentes adicionales capturan varianza residual que no discrimina entre clases. El sweet spot para la diagonal es npc=17 (o entre 13 y 17).

**2. `tde_cov_offdiag` mejora monotónicamente con npc** (28.4% → 31.2% → 34.8%).
Con 30 PCs más especializados, las correlaciones entre componentes capturan patrones de conectividad más específicos. Con 13–17 PCs, los componentes son mezclas amplias y sus cross-correlaciones no discriminan bien. Cuantos más componentes, más "ortogonales" son entre sí y más informativas sus co-varianzas.

**3. `tde_cov_full` mejora monotónicamente** porque se beneficia de ambas fuentes: la diagonal sigue aportando y el off-diagonal mejora con npc.

**4. Señal de presión por dimensionalidad en npc=30:**
El clasificador full con npc=30 (465 features) elige C=0.001 en todos los folds — la regularización máxima del grid. Indica que el modelo está al límite de lo que puede estimar con los datos disponibles. Con más features habría que extender el grid de C.

**5. El mejor modelo individual sigue siendo `tde_cov_diag` con npc=17:** 37.7% / AUC=0.643.
Con npc=30, el full (37.4% / AUC=0.637) se acerca pero no supera a la diagonal en npc=17, y lo hace con 465 features vs 17 — una eficiencia muy inferior.

### Conclusión sobre número de PCs

npc=17 es el punto óptimo para `tde_cov_diag`. Para `tde_cov_full` y `tde_cov_offdiag`, npc=30 es mejor, pero con el costo de alta dimensionalidad. Dado que la diagonal es el feature set más informativo y eficiente, **npc=17 queda confirmado como la configuración óptima** para los análisis subsiguientes.

**Archivos:**
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_results_npc13.json`
- `results/validation/photo_decoding_4class/sub-27/sub-27_4class_results_npc30.json`

---

### Paso 2: Comparar tamaños de ventana — 250ms vs 500ms

La discusión sobre ventanas de 100ms identificó que la covarianza con 25 muestras y 17 componentes está mal condicionada (ratio muestras/dimensión = 25/17 ≈ 1.5). Con 250ms (62 muestras) el ratio sube a 62/17 ≈ 3.6, que es mejor pero sigue siendo modesto. Con 500ms (125 muestras) el ratio llegaría a 125/17 ≈ 7.4 — mucho mejor condicionado para la estimación de covarianza.

**Propuesta:** correr ambos scripts con ventanas de 500ms y solapamiento de 80% (paso 100ms):
- 500ms a 250 Hz = 125 muestras → resolución Welch ~2 Hz/bin (excelente)
- Paso de 100ms → cada segmento de 500ms produce 5 ventanas (vs 6 con 250ms/50ms)
- Ratio cov: 125/17 ≈ 7.4 (bien condicionado)
- N total de ventanas: ligeramente menor que con 250ms, pero features más estables

**Pregunta secundaria:** ¿vale la pena probar también 500ms sin solapamiento (1 ventana/clase/trial = 74 ventanas de test total)? Esto da la mejor calidad de features posible pero el menor número de observaciones. El clasificador entrenaría con solo ~6×74=444 ventanas en train, lo que puede ser poco.

**Comparación propuesta:**

| Ventana | Paso | Ventanas/segmento | Muestras | Resolución Welch | Ratio cov |
|---|---|---|---|---|---|
| 100ms | 50ms | 9 | 25 | 10 Hz/bin | 1.5 |
| 250ms | 50ms | 6 | 62 | 4 Hz/bin | 3.6 |
| **500ms** | **100ms** | **5** | **125** | **2 Hz/bin** | **7.4** |
| 500ms | 0ms (sin solap.) | 1 | 125 | 2 Hz/bin | 7.4 |

---

### Paso 3: Analizar matrices de confusión (4 clases)

Las matrices de confusión ya se generan en `sub-27_4class_confusion.png`. Analizarlas para responder:
- ¿Qué pares de clases se confunden más? Hipótesis principal: ChangeUp (1) ↔ Luminance (2), ambas durante luminancia alta, deberían ser el par más confundido.
- ¿Las confusiones son simétricas (A→B similar a B→A) o asimétricas?
- ¿Es distinto el patrón de confusiones entre feature sets? Por ejemplo, si bandpower confunde ChangeUp↔Luminance pero raw_pca no, sugeriría que la distinción entre esos dos estados es temporal (raw) pero no espectral.

---

### Paso 4: Test de permutación para significancia estadística

Para cada feature set (bandpower, tde_cov, raw_pca), verificar si la accuracy obtenida es estadísticamente significativa comparada con la distribución nula (clasificación aleatoria).

**Diseño:**
- Repetir el pipeline LORO completo N=1000 veces, pero en cada repetición permutar aleatoriamente las etiquetas de clase **dentro de cada run** (permutación preserva la estructura de runs).
- Esto genera una distribución nula de accuracies bajo H₀: "las etiquetas no tienen relación con la señal".
- El p-valor es la proporción de permutaciones que superan la accuracy real observada.
- Correr para los 3 feature sets en ambas tareas (binaria y 4 clases).

**Razón para permutar dentro de runs:** si se permutan etiquetas entre runs, la distribución nula sería artificialmente fácil (el modelo podría aprender diferencias entre runs en lugar de diferencias entre clases). Permutar dentro de cada run mantiene la variabilidad inter-run en la distribución nula.

**Output esperado:** tabla de p-valores y z-scores para cada feature set y cada tarea.

---

### Paso 5: Extender a más sujetos

Todos los resultados actuales son de sub-27. Para validar generalización se necesita correr el mismo pipeline en otros sujetos disponibles. Requiere agregar `RUNS_CONFIG` por sujeto en `34_decoding_4class.py`.

---

### Paso 6 (más adelante): Paradigma de pantallas verdes

Una vez validado el pipeline de 4 clases en luminancia (estímulo marcado y limpio), aplicarlo al paradigma afectivo: estímulos visuales continuos donde el estado emocional cambia gradualmente durante 1 minuto. El diseño del script 34 es directamente extrapolable — solo cambia la definición de clases y los segmentos temporales.

---

## Orden de Ejecución

```
[03_2] Tarea 9.3: LogisticRegressionCV + features equiparadas (~160 cada uno) ✅
    │   Script: 27b_decoding_pre_vs_post.py
    │   Resultado: bandpower 68.9%, tde_cov 62.8%, raw_pca 65.5%
    │
    ├─► [03_3] Fix standardise_pc + paralelización joblib (27b) ✅
    │   Resultado: bandpower 68.9%, tde_cov 62.2% (−0.6pp), raw_pca 65.5%
    │
    └─► [03_3] Tarea 10.3: Decoding 4 clases temporales ✅
            │   Script: 34_decoding_4class.py
            │
            ├── 10.3.1: Ventanas 100ms → todos ≈ chance → DESCARTADO
            │   Causa: resolución Welch 10Hz/bin, cov mal condicionada
            │
            ├── 10.3.2: Ventanas 250ms → descubrimiento bug standardise_pc
            │   Fix mismatch train/test → refactorización en _run_one_fold_tde()
            │
            └── 10.3.3: Resultado final (250ms, fix aplicado) ✅
            │   bandpower 30.9% / AUC=0.570
            │   tde_cov   35.6% / AUC=0.622  ← mejor AUC
            │   raw_pca   37.4% / AUC=0.614  ← mejor Acc
            │
            └── 10.3.4: Ablación tde_cov (diag/offdiag/full) ✅
            │   Script 34 (4 clases):
            │     tde_cov_diag    37.7% / AUC=0.643  ← MEJOR (17 features!)
            │     tde_cov_full    35.6% / AUC=0.622
            │     tde_cov_offdiag 31.2% / AUC=0.573
            │   Script 27b (binario):
            │     bandpower       68.9% / AUC=0.725  ← sigue siendo el mejor
            │     tde_cov_full    62.2% / AUC=0.624
            │     tde_cov_diag    61.5% / AUC=0.664
            │     tde_cov_offdiag 59.5% / AUC=0.573
            │
            └── 10.3.5: Comparación npc=13/17/30 ✅
                tde_cov_diag:    npc=13: 37.3%/0.626 | npc=17: 37.7%/0.643★ | npc=30: 36.5%/0.629
                tde_cov_full:    npc=13: 33.8%/0.601 | npc=17: 35.6%/0.622  | npc=30: 37.4%/0.637
                tde_cov_offdiag: npc=13: 28.4%/0.541 | npc=17: 31.2%/0.573  | npc=30: 34.8%/0.608
                → npc=17 confirmado como óptimo para tde_cov_diag
```
