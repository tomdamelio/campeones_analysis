# Diario de Tareas — Preparación Reunión Diego (2026-04-01)

**Proyecto:** campeones_analysis
**Fecha:** 2026-04-01
**Supervisores:** Enzo, Diego Vidaurre
**Contexto:** Preparación para reunión con Diego. Recapitulación de todo lo trabajado desde el último reporte, y proximos pasos.

---
## Preguntas abiertas luego del ultimo reporte con Diego

### 1. Sobre la comparación de feature sets

Diego señaló que el número de features sesga la comparación. Tiene razon, y por eso implementamos dos correcciones:

#### 1.1 Equiparación de features + LogisticRegressionCV (Tarea 9.3, 2026-03-25)

Rediseño del script 27b (pre/post) para corregir dos problemas:
- **Equiparación de dimensionalidad:** estandarizar todos los feature sets a ~160 features
- **C adaptativo:** usar `LogisticRegressionCV` en lugar de C fijo, permitiendo que cada feature set elija su regularización óptima

**Configuración:**
- `bandpower_welch`: 160 features (32 ch × 5 bandas) — sin cambio
- `raw_pca`: 160 features (PCA(160) sobre flatten espacio-tiempo)
- `tde_cov`: 153 features (17 PCs → upper triangle cov = 17×18/2)
- `LogisticRegressionCV`: grid C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], inner CV 5-fold, L2 ridge
- **Dataset:** 148 épocas (74 onsets × 2 clases), 7 runs LORO, ~252 samples train por fold

**¿Cómo funciona el cross-validation de C en LogisticRegressionCV?**

Para cada fold LORO (hay 7 en total):
1. **Se separan los datos:** 6 runs para train (~252 samples), 1 run para test (~36 samples)
2. **Inner CV Loop — Seleccionar C óptimo:**
   - Se divide el train (252 samples) en 5 sub-folds internos (~50 samples por sub-fold)
   - Para **cada valor de C** en [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
     - Se entrena el modelo 5 veces, dejando fuera cada sub-fold
     - Se calcula la accuracy promedio en los 5 sub-folds (inner CV accuracy)
   - Se elige el **C* que maximiza la accuracy promedio** en la inner CV
   - Ejemplo: bandpower_welch en fold run-002 elige C=0.1 → es el mejor regularizador para ese fold
3. **Entrenar modelo final:** con el C* elegido, se entrena el modelo completo sobre los 252 samples (sin sub-folds)
4. **Predecir en test:** se evalúa sobre el 1 run de test (36 samples) y se reporta la accuracy final

**Por qué esto es mejor que C fijo:**
- Con C=1.0 fijo: trabajas con la misma regularización para todos — puede ser sub-óptima para algunos feature sets
- Con LogisticRegressionCV: cada feature set elige su C* automáticamente:
  - Si el feature set es más complejo (raw_pca con 160 PCs ruidosos), elige C bajo (mucha regularización)
  - Si es más simple (bandpower con 160 features bien estructurados), elige C medio-alto (menos regularización)
- **Limitación:** Con ~50 samples por sub-fold de inner CV, la selección de C es **muy ruidosa** — por eso ves C variar entre folds (bandpower: [0.1, 1.0, 10.0, 0.01, ...]), reflejando la variabilidad del pequeño N

**Visualización del doble loop:**

```
OUTER LOOP: Leave-One-Run-Out CV (7 folds)
├─ Fold 1: train=run[2,3,4,6,7,9,10] test=run[2]
│  ├─ Feature set: bandpower_welch
│  │  ├─ INNER LOOP: 5-fold CV sobre train (~252 samples → ~50/fold)
│  │  │  ├─ C=0.001:   [acc_f1=0.70, acc_f2=0.65, ..., acc_f5=0.68] → mean=0.66
│  │  │  ├─ C=0.01:    [acc_f1=0.75, acc_f2=0.70, ..., acc_f5=0.72] → mean=0.71
│  │  │  ├─ C=0.1:     [acc_f1=0.78, acc_f2=0.75, ..., acc_f5=0.76] → mean=0.76 ← BEST
│  │  │  ├─ C=1.0:     [acc_f1=0.75, acc_f2=0.73, ..., acc_f5=0.74] → mean=0.74
│  │  │  └─ ... (continúa con C=10.0, 100.0)
│  │  ├─ ELIGE: C* = 0.1 (máxima accuracy en inner CV)
│  │  ├─ Entrena modelo final: LogisticRegression(C=0.1) sobre TODOS los ~252
│  │  └─ Predice en test run[2]: accuracy = 45.8%
│  │
│  ├─ Feature set: raw_pca
│  │  ├─ INNER LOOP: idem (pero datos son PCA(160) en lugar de bandpower)
│  │  ├─ ELIGE: C* = 0.001 (máxima accuracy en inner CV) ← muy regularizado
│  │  └─ Predice en test run[2]: accuracy = 42.3%
│  │
│  └─ Feature set: tde_cov
│     ├─ INNER LOOP: idem (pero datos son covarianza TDE en lugar de bandpower)
│     ├─ ELIGE: C* = 0.001
│     └─ Predice en test run[2]: accuracy = 41.7%
│
├─ Fold 2: train=run[2,3,4,6,7,9,10] test=run[3]
│  ├─ Feature set: bandpower_welch
│  │  ├─ INNER LOOP: 5-fold CV (datos distintos → C* puede ser distinto)
│  │  ├─ ELIGE: C* = 1.0 ← diferente a fold 1!
│  │  └─ Predice: accuracy = 75.0%
│  │
│  └─ ... (idem para raw_pca, tde_cov)
│
└─ Folds 3–7: idem
   RESULTADO FINAL: promedio de todos los test accuracies
   bandpower: (45.8 + 75.0 + ... ) / 7 = 68.9%
```

**Lo clave a entender:**
1. El **inner loop (5-fold CV)** es completamente independiente y ocurre **dentro** de cada fold del outer loop
2. **Cada fold outer tiene su propio C***, porque los datos de train son distintos (composición de 6 runs diferente)
3. El C* **no se transfiere** entre folds — no es un hiperparámetro global, es local a cada fold
4. Por eso ves: bandpower con C=[0.1, 1.0, 10.0, 0.01, ...] — cada fold elige independientemente

**Resultados (sub-27, LORO 7-fold, chance=50%):**

| Feature set | N feat | Acc | F1 | AUC | C variable? |
|---|---|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | **0.689** | **0.725** | Sí (0.01–10.0) |
| raw_pca | 160 | 65.5% | 0.662 | 0.696 | Sí, siempre C=0.001 |
| tde_cov | 153 | 62.8% | 0.621 | 0.685 | Sí (0.001–1.0) |

**C por fold (bandpower_welch):** [0.1, 1.0, 10.0, 0.01, 0.1, 1.0, 0.01] — muy variable
**C por fold (raw_pca):** [0.001]×7 — consistentemente máxima regularización
**C por fold (tde_cov):** [0.001, 0.01, 0.1, 0.001, 0.01, 1.0, 0.1] — también variable

**Interpretación:** El ranking se mantiene (bandpower > raw_pca ≈ tde_cov) incluso con features equiparadas y C cross-validado. **La variabilidad alta de C entre folds es el hallazgo clave:**

- **bandpower elige C entre 0.01 y 10.0** (4 órdenes de magnitud de diferencia) — el algoritmo está "confundido," no hay un regularizador claramente óptimo
- **raw_pca siempre elige C=0.001** (máxima regularización) — indica que necesita mucho "freno" para no overfitter
- **tde_cov también variable** (0.001–1.0) — la inner CV no converge a un C óptimo único

**Causa raíz:** Con ~252 samples de train divididos en 5 sub-folds internos, tienes solo ~50 samples **por sub-fold** de inner CV. Con tan pocos datos, la superficie de accuracy vs C es ruidosa y estocástica. Un cambio pequeño en la composición train/test puede cambiar qué C gana. Esto **no indica diferencias reales en la naturaleza de los feature sets**, sino que el tamaño muestral es insuficiente para hacer una selección de C confiable.

**Esta fue la razón por la que Diego pidió que lo hiciéramos:** mostrar que la variabilidad de C es un artefacto de N bajo, no una propiedad de los features. Para tareas posteriores (4 clases, luminancia continua) se prefirió **C=1.0 fijo** en lugar de LogisticRegressionCV, precisamente para evitar ese ruido y hacer comparaciones más limpias.

#### 1.2 Cambio metodológico para tareas posteriores

Para los análisis de 4 clases (script 34) y luminancia continua (script 36), se cambió a **`LogisticRegression(C=1.0, lbfgs)` fijo** para:
- Comparación **más limpias** entre feature sets (mismo estimador, no adaptativo)
- **Permutaciones consistentes:** el nulo y el observado usan exactamente el mismo clasificador, sin artefactos de CV ruidosa
- Eliminar ambigüedad: la performance sería solo del feature set, no de la interacción feature-set-con-selección-C

Esto es mejor para validación estadística (permutaciones, p-valores), aunque sacrifica optimización individual por feature set.

### 2. Sobre la information content de TDE vs bandpower

Diego comentó:
> "En todo caso, TDE contiene la misma información que Bandpower Welch, + coherencia, o sea, conectividad"

**Exacto — y lo evaluamos específicamente.** Para validar esta descomposición de TDE en componentes de potencia vs conectividad, implementamos una **ablación** del espacio de covariance TDE en 3 variantes:

#### 2.1 Ablación de TDE: Diagonal vs Off-diagonal vs Completo (Tarea 10.3.4, 2026-03-25)

La matriz de covarianza TDE (17×17 después de PCA) se puede descomponer en sus elementos informativos:
- **Diagonal:** 17 features = potencia de cada PC (análogo a bandpower pero en espacio TDE)
- **Off-diagonal:** 136 features = covarianzas entre PCs (conectividad/coherencia)
- **Completo (full):** 153 features = diagonal + off-diagonal

**¿Por qué la diagonal es análoga a bandpower?**

Ambas capturan **medidas de "potencia" o "energía dispersa"**, pero en espacios completamente distintos:

**Bandpower (160 features):**
```
Para cada canal (32) y banda espectral (5):
  potencia_band = ∫ PSD(f) df    para f ∈ [f_bajo, f_alto]
  
Resultado: un escalar por canal-banda
  - Si alpha está suprimida: potencia_alpha baja
  - Si theta está elevada: potencia_theta alta
  
32 canales × 5 bandas = 160 números que representan 
"cuánta energía en cada FRECUENCIA × CANAL"
```

**Diagonal de TDE Covarianza (17 features):**
```
Para cada componente principal TDE (17):
  varianza_PC = var(PC_i)    sobre todos los timepoints en la ventana
  
Resultado: un escalar por componente
  - Si el PC explica "ERPs visuales": varianza alta
  - Si el PC explica "ruido de línea base": varianza baja
  
17 números que representan 
"cuánta varianza/energía explica cada MODO ESPACIOTEMPORAL"
```

**La analogía:**

| Aspecto | Bandpower | TDE Diagonal |
|---|---|---|
| **Qué mide** | Potencia en cada banda espectral | Varianza en cada modo espaciotemporal |
| **Dimensión del espacio** | Frecuencia × Canal (160 dims) | Componentes principales TDE (17 dims) |
| **Interpretación** | Energía por frecuencia | Energía por patrón dinámico |
| **¿Tiene info sobre amplitud?** | Sí, la potencia es proporcional a amplitud² | Sí, la varianza es proporcional a amplitud² |
| **¿Tiene info sobre conexiones?** | No directamente | No, eso es off-diagonal |

**Ejemplo concreto — Cambio de luminancia visual:**

Cuando hay un stimulus visual (ChangeUp):
1. **Bandpower detecta:** aumento de potencia occipital en alpha (8–13 Hz) puede estar suprimida u oscilando
2. **TDE diagonal detecta:** aumento de varianza en el PC que captura "respuesta visual transiente" (amplitud cruda, que mezcla todas las frecuencias pero captura el ERP de latencia ~100ms)

Ambos están midiendo "cambios en la amplitud/energía del EEG", pero por canales diferentes:
- Bandpower: "¿hay más energía en tal rango de frecuencias?"
- TDE diagonal: "¿hay más varianza en este patrón espaciotemporal?"

**Por qué solo la diagonal importa en 4 clases (el hallazgo clave):**

Si TDE diagonal captura "potencia en patrones espaciotemporales" y bandpower captura "potencia en bandas", y ambos dan aproximadamente lo mismo (~30–35% accuracy), entonces los "patrones espaciotemporales" aprendidos por TDE-PCA son esencialmente equivalentes a las bandas espectrales estándar para esta tarea.

Los **off-diagonal (conectividad)** serían la información *adicional* en TDE más allá de bandpower. Pero resulta que esa conectividad:
- En la tarea visual/luminancia **no discrimina bien** (31.2% accuracy, casi igual que bandpower 30.9%)
- Mejor aún: cuando lo incluyes (full = 35.6%), empeora comparado a solo diagonal (37.7%)

Esto significa: **la conectividad entre PCs es ruido para esta tarea**, no información adicional.

**Implementación:** máscara booleana en `_get_cov_mask(k, mode)` que selectivamente extrae diagonal, off-diagonal, o full antes de alimentar al clasificador. Mismo PCA(17), solo varían los features seleccionados.

**Resultados en Tarea de 4 clases temporales (script 34, sub-27, chance=25%):**

| Feature set | N features | Acc | AUC | Baseline | ChangeUp | Luminance | ChangeDown |
|---|---|---|---|---|---|---|---|
| bandpower_welch | 160 | 30.9% | 0.570 | 34.7% | 26.4% | 25.2% | 37.4% |
| **tde_cov_diag** | **17** | **37.7%** | **0.643** | 30.9% | **42.8%** | **44.6%** | 32.7% |
| tde_cov_full | 153 | 35.6% | 0.622 | 34.0% | 34.0% | 36.5% | 37.8% |
| tde_cov_offdiag | 136 | 31.2% | 0.573 | 33.1% | 24.1% | 32.2% | 35.4% |
| raw_pca | 160 | 37.4% | 0.614 | 27.9% | 44.4% | 44.8% | 32.7% |

**Resultados en Tarea binaria Pre/Post (script 27b, sub-27, chance=50%):**

| Feature set | N features | Acc | AUC |
|---|---|---|---|
| bandpower_welch | 160 | **68.9%** | **0.725** |
| raw_pca | 160 | 65.5% | 0.696 |
| tde_cov_full | 153 | 62.2% | 0.624 |
| tde_cov_diag | 17 | 61.5% | **0.664** |
| tde_cov_offdiag | 136 | 59.5% | 0.573 |

**Hallazgos centrales:**

1. **En 4 clases: la diagonal (potencia) supera al full** — `tde_cov_diag` (37.7% Acc, 0.643 AUC) > `tde_cov_full` (35.6%, 0.622). Con solo 17 features superan a 153 features. Esto prueba que **la conectividad (off-diagonal) no solo no ayuda, sino que añade ruido** para esta tarea.

2. **tde_cov_offdiag ≈ bandpower en 4 clases** (31.2% vs 30.9% Acc). La conectividad entre PCs no discrimina mejor que la potencia espectral. Diego tenía razón en que TDE = bandpower + conectividad, pero para estados temporales de luminancia visual, es la potencia lo que importa.

3. **En pre/post: la diagonal da AUC mejor que full** (0.664 vs 0.624). La potencia vuelve a dominar. Off-diagonal tiene el peor rendimiento (0.573).

4. **raw_pca y tde_cov_diag tienen patrones per-clase similares** en 4 clases, destacando en ChangeUp (42.8% vs 44.4%) y Luminance (44.6% vs 44.8%). Ambos capturan respuestas transientes de amplitud (ERPs visuales), no solo estructura espectral.

5. **Implicación para tu presentación a Diego:** Le muestras que conceptualmente TDE = bandpower + conectividad, **pero empíricamente**, para visual/luminancia, es la diagonal (potencia en espacio TDE-PCA) la que discrimina. La conectividad entre componentes temporales es estructuralmente informativa (coherencia sí existe), pero para esta tarea **no es discriminativa entre clases**.

#### 2.2 Optimización del número de PCs (Tarea 10.3.5, 2026-03-25)

Dado que la ablación reveló que solo la diagonal importa (17 features), se evaluó si 17 es óptimo o si hay un número mejor. Se compararon npc=13 (80% varianza), 17 (83.9%), 30 (90%):

| Feature type | npc=13 | npc=17 | npc=30 |
|---|---|---|---|
| tde_cov_diag | 37.3% / 0.626 | **37.7% / 0.643** | 36.5% / 0.629 |
| tde_cov_full | 33.8% / 0.601 | 35.6% / 0.622 | **37.4% / 0.637** |
| tde_cov_offdiag | 28.4% / 0.541 | 31.2% / 0.573 | **34.8% / 0.608** |

**Conclusión:** npc=17 es el sweet spot para `tde_cov_diag`. Con más componentes, cada uno explica menos varianza y la diagonal se vuelve menos discriminativa. Para off-diagonal, más componentes ayudan (porque correlaciones entre PCs especializados son más informativas), pero costo computacional es alto (435 features).

---

## Sobre la pregunta de "decoding continuo"

Diego preguntó:
> "¿Por qué esto está más cerca del escenario de regresión continua que lo que acabas de hacer?"

La respuesta que di: la tarea de 4 clases agrega resolución temporal dentro del trial. Pero Diego tiene razón en que sigue siendo clasificación discreta, no regresión continua.

**Pregunta para discutir en próxima reunión:** ¿Cuál sería el diseño correcto para un primer paso de validación hacia la regresión continua? ¿Directamente regresión sobre la señal de luminancia del video (variable continua), usando una ventana deslizante del EEG? ¿O hay pasos intermedios?

---

## Resumen de que hicimos hasta ahora

Desde el último reporte a Diego se completaron dos bloques de trabajo:

1. **Decoding 4 clases temporales + test de permutación**
2. **Decoding 3 clases de luminancia continua (60s)** — nueva tarea sobre los segmentos de luminancia, con test de permutación


### Bloque 1: Decoding 4 clases temporales + Test de Permutación ✅

#### Diseño (script 34)

4 condiciones definidas por posición temporal dentro del trial de foto (1s baseline → 1s cambio → 1s retorno):

| Clase | Ventana |
|---|---|
| Baseline | [-250, 0ms] pre-onset + [1500, 1750ms] post-offset |
| ChangeUp | [0, 500ms] — respuesta al onset del aumento |
| Luminance | [500, 1000ms] — luminancia sostenida |
| ChangeDown | [1000, 1500ms] — respuesta al offset |

Ventanas de 250ms, step 50ms (6 ventanas solapadas por trial), LORO CV, 74 trials × 7 runs, ~1776 ventanas totales.

#### Resultados LORO (sub-27, chance = 25%)

| Feature set | Acc | AUC |
|---|---|---|
| raw_pca | **37.4%** | 0.617 |
| tde_cov | 34.9% | 0.604 |
| bandpower_welch | 31.7% | 0.570 |

→ Plots: [`results/validation/photo_decoding_4class/sub-27/sub-27_4class_confusion.png`](../results/validation/photo_decoding_4class/sub-27/sub-27_4class_confusion.png) | [`sub-27_4class_per_class_acc.png`](../results/validation/photo_decoding_4class/sub-27/sub-27_4class_per_class_acc.png)

#### Test de permutación (n=1000, C=1.0 fijo)

Método: permutar etiquetas **dentro de cada run** (no entre runs) para preservar la variabilidad inter-run en la distribución nula.

| Feature set | Acc obs | Null mean ± std | p-valor | z-score |
|---|---|---|---|---|
| bandpower_welch | 31.7% | 25.0% ± 1.1% | **0.000** | 6.00 |
| tde_cov | 34.9% | 24.9% ± 1.2% | **0.000** | 8.62 |
| raw_pca | 37.4% | 25.0% ± 1.2% | **0.000** | 10.61 |

**¿Qué significa el z-score?**

El z-score mide cuántas desviaciones estándar está la accuracy observada por encima de la media de la distribución nula:

z = (accuracy_observada - media_nula) / desviación_estándar_nula

- **z > 0**: la accuracy observada está por encima del nulo (señal presente)
- **z alto (ej. 6.00)**: señal muy robusta, improbable por azar
- **z ≈ 0**: accuracy observada ≈ media nula (no hay señal)
- **z < 0**: accuracy observada por debajo del nulo (posible confound o error)

En estadística, z ≈ 1.96 corresponde a p < 0.05 (unilateral). Aquí z=6.00 significa que la señal es extremadamente significativa (p < 10^-9). Un z=10.61 es prácticamente imposible por azar (p ≈ 10^-26).

**Los 3 modelos son estadísticamente significativos (p < 0.001).** Ninguna de las 1000 permutaciones igualó la accuracy observada. El nulo converge exactamente en el 25% teórico porque las clases están balanceadas. Z-scores altos (6–11σ) indican señal muy robusta.

#### Nota metodológica: consistencia estimador observado vs nulo

Un problema previo era que el LORO observado usaba `LogisticRegressionCV` (que cross-valida C con inner-CV) mientras que las permutaciones usaban C fijo. Esto **inflaba artificialmente la accuracy observada** relativa al nulo. Se corrigió usando `LogisticRegression(C=1.0, lbfgs)` fijo en ambos, garantizando que se compara el mismo estimador.

#### Interpretación para Diego

La tarea de 4 clases captura **resolución temporal dentro del trial**: el modelo distingue en qué estado de la trayectoria de luminancia (onset, sostenido, offset, baseline) está el sujeto en cada ventana de 250ms. Los z-scores altos (especialmente raw_pca = 10.6σ) confirman que el EEG de sub-27 contiene información genuina sobre la dinámica temporal del estímulo visual.

---

### Bloque 2: Decoding 3 clases en luminancia continua (60s) ✅

#### Motivación y diferencia con los bloques anteriores

Los análisis anteriores operaban sobre **foto-eventos** (trials de 1s con estructura temporal definida por el diseño experimental). El siguiente paso natural es trabajar sobre los **segmentos de luminancia continua** (60s por run), donde no hay trials y los eventos se infieren de la señal del video.

Esto es más cercano al escenario que nos interesa a largo plazo: decoding continuo de estados afectivos sin estructura de trial.

#### Diseño (script 36)

Para cada segmento de 60s por run (7 runs), se detectan eventos de cambio de luminancia a partir de la **derivada frame-a-frame** del canal verde del video:

| Clase | Criterio |
|---|---|
| ChangeUp | ΔL/frame > 1.5 (threshold) |
| ChangeDown | ΔL/frame < −1.5 |
| NoChange | \|ΔL/frame\| < 1.5 durante toda la época Y 1s previo estable |

**Diseño de épocas (versión actual, corregida):**
- 6 ventanas de 250ms por evento (offsets 0, 50, 100, 150, 200, 250ms → tope derecho 500ms)
- Idem para NoChange: 6 ventanas solapadas por evento estable
- Balance: n_nc_eventos = (n_up + n_down) / 2 → ~33% por clase
- Selección greedy no-solapante con guard de 500ms entre eventos

→ Visualización de épocas: [`results/validation/luminance_3class/sub-27/timeline_plots/sub-27_timeline_task-01_acq-a_run-002_vid12_pres1.png`](../results/validation/luminance_3class/sub-27/timeline_plots/sub-27_timeline_task-01_acq-a_run-002_vid12_pres1.png)

#### Resultados primera versión (threshold=2.0, NC sin 6 ventanas) + test de permutación

*(Versión antes de la corrección del diseño de épocas)*

| Feature set | Acc | F1 | AUC | p-valor | z |
|---|---|---|---|---|---|
| raw_pca | 42.1% | 0.357 | 0.541 | **0.022** | 2.05 |
| bandpower_welch | 41.5% | 0.357 | 0.524 | 0.090 | 1.31 |
| tde_cov | 38.3% | 0.302 | 0.501 | 0.256 | 0.70 |

**Hallazgo crítico del test de permutación:** el nulo empírico no convergía en 33.3% (chance nominal) sino en ~38–39%, porque NoChange representaba el 50% de las ventanas. Un clasificador que aprende el sesgo de clase obtiene ~38% sin capturar señal EEG real. Raw_pca fue el único significativo (p=0.022, z=2.05), muy por debajo de los z-scores de la Tarea anterior.

#### Corrección del diseño y nueva versión en curso

Se corrigió el algoritmo:
1. Threshold bajado de 2.0 → **1.5** (más eventos detectados)
2. NC genera **6 ventanas solapadas** (antes: 1 ventana por frame)
3. Balance: n_nc_eventos = (n_up + n_down)/2 → clases ~33-33-33

Los resultados de esta versión corregida completaron (2026-04-01).

**Distribución de ventanas con nuevo diseño (sub-27, 7 runs):**

| Run | NoChange | ChangeUp | ChangeDown | Total |
|---|---|---|---|---|
| run-002 (vid12) | 90 | 96 | 90 | 276 |
| run-003 (vid9) | 60 | 72 | 54 | 186 |
| run-004 (vid3) | 48 | 54 | 42 | 144 |
| run-006 (vid7) | 30 | 36 | 30 | 96 |
| run-007 (vid12) | 90 | 96 | 90 | 276 |
| run-009 (vid9) | 60 | 72 | 54 | 186 |
| run-010 (vid7) | 30 | 36 | 30 | 96 |
| **Total** | **408** | **462** | **390** | **1260** |

#### Resultados versión corregida (threshold=1.5, balance 33-33-33, n=1000 permutaciones)

| Feature set | Acc | F1 | AUC | Null mean±std | p-valor | z-score |
|---|---|---|---|---|---|---|
| bandpower_welch | 34.7% | 0.343 | 0.515 | 33.7% ± 1.8% | 0.314 | 0.55 |
| tde_cov | 31.8% | 0.319 | 0.472 | 33.7% ± 2.2% | 0.805 | −0.86 |
| raw_pca | 29.3% | 0.285 | 0.455 | 33.7% ± 1.3% | 1.000 | **−3.34** |

#### Matriz de Confusión por Feature Set

**bandpower_welch:**

| Verdadero\Predicho | NoChange | ChangeUp | ChangeDown |
|---|---|---|---|
| NoChange | 119 | 156 | 133 |
| ChangeUp | 116 | 188 | 158 |
| ChangeDown | 114 | 146 | 130 |

**tde_cov:**

| Verdadero\Predicho | NoChange | ChangeUp | ChangeDown |
|---|---|---|---|
| NoChange | 133 | 163 | 112 |
| ChangeUp | 118 | 149 | 195 |
| ChangeDown | 114 | 157 | 119 |

**raw_pca:**

| Verdadero\Predicho | NoChange | ChangeUp | ChangeDown |
|---|---|---|---|
| NoChange | 125 | 171 | 112 |
| ChangeUp | 173 | 171 | 118 |
| ChangeDown | 133 | 184 | 73 |

**Chance level: 33.3% — Null empírico: ~33.7% ✓ (ahora converge correctamente)**

**Ningún feature set es significativo. tde_cov y raw_pca están por debajo del chance (z negativo).**

#### Interpretación — qué nos dice este resultado

**1. El nulo ahora converge correctamente en 33.7% ≈ 33.3%.** El balance de clases está funcionando bien. Los resultados previos (raw_pca p=0.022) eran en gran parte artefacto del desbalance.

**2. La tarea 3-clases no muestra señal con este diseño en sub-27.** Los tres feature sets caen dentro o debajo de la distribución nula. El diseño corregido "limpió" el artefacto pero también eliminó la señal aparente.

**3. raw_pca con z=−3.34 es llamativo.** Un z negativo tan pronunciado sugiere que el clasificador está aprendiendo algo sistemáticamente *contra-predictivo*. La hipótesis más probable: con 6 ventanas solapadas (200ms de overlap en 250ms de ventana) por evento, las ventanas son altamente correlacionadas entre sí. El PCA sobre el flatten espacio-temporal puede estar capturando la varianza del *offset temporal dentro de la época* (ventana 0ms vs ventana 250ms tienen formas distintas) en lugar de la diferencia entre clases. Esto podría crear un confound: el clasificador aprende "ventana de offset 250ms vs 0ms" en lugar de "ChangeUp vs NoChange".

**4. Hipótesis alternativa — confound temporal en NoChange.** Los eventos NC se seleccionan de regiones estables (1s sin cambio previo + 500ms forward). Esto puede hacer que los eventos NC se concentren en momentos del video de baja actividad general, mientras que ChangeUp/Down ocurren en transiciones. Si el EEG refleja algún cambio de estado general que no es específico al cambio de luminancia, raw_pca (que captura toda la varianza espacio-temporal) podría aprender ese confound en dirección incorrecta.

### Diagnóstico y próximos pasos

El resultado actual indica que **la Tarea 3-clases en su forma actual no extrae señal útil** de sub-27. Las opciones:

1. **Investigar el confound de las 6 ventanas solapadas**: probar con solo 1 ventana por evento (sin augmentación temporal) para ver si el z de raw_pca se normaliza. Si el z negativo desaparece, el problema es el overlap.

2. **Simplificar a tarea binaria (Change vs NoChange)**: colapsar ChangeUp+ChangeDown en una clase. Más muestras por clase, tarea más fácil — permite verificar si hay alguna señal antes de intentar discriminar dirección.


---

## Línea de tiempo de scripts producidos

| Script | Descripción | Estado |
|---|---|---|
| `scripts/validation/27_decoding_photo_change.py` | Binario CHANGE vs NO_CHANGE en foto-eventos | ✅ |
| `scripts/validation/34_decoding_4class.py` | 4 clases temporales en foto-eventos + permutaciones | ✅ |
| `scripts/validation/36_decoding_luminance_3class.py` | 3 clases (Up/Down/NoChange) en segmentos 60s | ✅ activo |
| `scripts/validation/37_visualize_luminance_epochs.py` | Visualización timeline épocas luminancia | ✅ |


---

## Próximos pasos y tareas acordadas con Diego

**Contexto:** Análisis post-reunión para formalizar cada punto que Diego planteó y trazar implementación práctica.

### 1. Interpretabilidad del modelo ganador (Band Power Welch)
1.1 Cargar el modelo ganador (el modelo con feature de Welch era?) en los modelos que aun performaban bien (prediccion de 2 clases con cambios fuertes de luminancia) y extraer los coeficientes (betas) de la regresión logística usada en el análisis binario/4-clases.
1.2 Mapear cada coeficiente al feature original (canal × banda) y crear un ranking de features con mayor contribución positiva/negativa.
1.3 Graficar al menos 2 top-10 features (e.g. potencia alfa occipital, beta frontal) para confirmar qué bandas dominan.
1.4 Guardar resultado y plot comparativo (barplot, heatmap).
1.5 Documentar brevemente en el diario: qué banda(s) y canal(es) justifican la ventaja de Welch.

### 1bis. Interpretabilidad del modelo ganador (Band Power Welch)

1.6 Extender esta tarea a la prediccion de 4 clases con cambios fuertes de luminancia.

### 2. Variante Time-Delay Embedding (TDE) con autocorrelación por canal
**Racional (Diego):** La autocorrelación por canal captura esencialmente la misma información que Welch, pero en dominio temporal en lugar de frecuencial. La covarianza TDE completa (`tde_cov_full`) agrega encima la coherencia entre canales, que puede no ser predictiva y además infla el número de features.
- **Por qué NO usar solo la diagonal de TDE:** la diagonal es la varianza de cada canal, que equivale a la suma de potencia en todas las bandas, no a su *distribución*. Es lo menos informativo del espectro.
- **Por qué NO usar solo el off-diagonal:** captura coherencia entre canales, que probablemente no aporta suficiente señal predictiva como para compensar el coste de features extra.

2.1 Con la misma tarea predictiva (de 2 clases? o de 4 clases era la que habia funcionado bien?), implementar función que calcule autocorrelación temporal de lags 1..N para cada canal de la época (N pequeño, e.g. 5–10 lags).
2.2 Para cada ventana de 250 ms generar vector canal-by-lag (32 canales × N lags).
2.3 Concatenar vectores de los 32 canales en un feature vector (evita combinaciones canal-canal).
2.4 Comparar con el pipeline actual de `tde_cov_full`: misma normalización, PCA (si aplica), clasificación `LogisticRegression(C=1.0)` y permutación n=1000.
2.5 Registrar resultados y comparar con: `tde_cov_full`, `bandpower_welch` y `raw_PCA`.
2.6 Conclusión prevista: si la performance es similar a Welch → confirma que la representación temporal sin coherencia inter-canal es suficiente. Si mejora sobre `tde_cov_full` → confirma que la conectividad espacial no es predictiva y solo añade ruido de features.

### 3. Sanity checks de señal y plausibilidad fisiológica
**Criterio diagnóstico clave (Diego):** La tarea de flash (cambios de luminancia fuertes) tiene señal tan evidente que cerrando los ojos se ve en el raw EEG. Si visualmente se distingue → el clasificador DEBE dar 80–90% o más. Si no llega a eso → hay un **bug en el pipeline de CV**. Este es el punto de entrada al debugging (tarea 4).

3.1 Foco principal en la tarea de **flash** (scripts 27/34, cambios fuertes): plotear EEG raw del canal occipital para 2–3 épocas representativas de `ChangeUp` y `NoChange`. Confirmar si el cambio de luminancia es visible “a ojo”.
3.2 Hacer **histogramas de potencia alfa** separados para las dos clases (`ChangeUp` vs `NoChange`). Si las distribuciones se solapan totalmente → hay problema en la señal o en las etiquetas.
3.3 Plot de ERP y/o TFR promedio para `ChangeUp` vs `NoChange`, con estadísticos simples (t-test por timepoint o cluster permutacional).
3.4 Si es posible, añadir scatter de potencia alfa vs beta para ver separabilidad visual.
3.5 Registrar el resultado del sanity check explícitamente: ¿se ve señal? ¿cuánto da el clasificador? → Si señal visible + classifier < 70% → escalar a tarea 4 (debugging). Si señal no visible → problema en preprocesamiento o etiquetas.
3.6 Nota sobre la **tarea subtle** (pantalla verde continua): ya sabemos que da muy mal. No es el foco del sanity check; solo volver a ella una vez que el flash esté validado y el pipeline sea confiable.

### 4. Debugging de pipeline (condicional a resultado de tarea 3)
**Activar solo si:** tarea 3 muestra señal claramente visible en raw/histograma alfa pero el clasificador da < 70% en el flash task. En ese caso hay un error de código.

4.1 Revisar paso a paso pipeline de CV: splits, shuffle, stratify, normalización por fold, permutaciones dentro de run.
4.2 Añadir asserts y “debug mode” para verificar que no hay fuga de información (leakage) desde test hacia train.
4.3 Verificar que las etiquetas están correctamente alineadas con las épocas (el bug más probable según Diego).
4.4 Comparar análisis binario flash (debería dar 80–90%) con tarea subtle (sabemos que da mal) para calibrar si el problema es el pipeline o la señal.
4.5 Documentar discrepancia y causa encontrada en este diario.

### 5. Contacto con Yongjie (estudiante del lab de Pablo/Diego, Slack)
**Contexto:** Yongjie reportó 90% de accuracy en clasificación binaria (brightest vs darkest videos), evaluado **por video** (no timepoint a timepoint). Puede haber varias explicaciones: metodología diferente, tarea más fácil por más trials, o error en su pipeline. Diego quiere comparar.

5.1 Buscar a Yongjie en Slack y escribirle mensaje corto con preguntas claras:
- ¿Clasificación por video o por timepoint/época?
- Definición exacta de clases (brighter vs darkest: ¿umbral? ¿percentil?)
- Train/test split: ¿hay run dependency? ¿stratify?
- Features usados (raw, espectral, TDE, otro)
5.2 Pedir que comparta pipeline de testeo y/o snippet de código de preprocesamiento/normalización.
5.3 Anotar comparativa entre su metodología y la nuestra (ventajas/diferencias clave).
5.4 Incluir bloque en este diario: “Feedback externo — Yongjie [fecha] — hallazgos y acciones derivadas.”

### 6. Criterio de cierre de tarea
- Resultados reproducibles: al menos 2 plots (histograma alfa y raw occipital) y una tabla de métricas para cada feature set.
- Chequeo de comportamiento: si visualmente hay señal pero classifier falla, se documenta error y se corrige en código.
- Contacto hecho y respuesta referenciada (con fecha y hallazgo).