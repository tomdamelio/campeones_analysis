# Diario de Tareas — Validación Estadística y Próximos Pasos (2026-03-31)

**Proyecto:** campeones_analysis
**Fecha:** 2026-03-31
**Supervisores:** Enzo, Diego Vidaurre
**Contexto:** Reunión de seguimiento con Enzo; definición de la agenda de trabajo siguiente.

---

## Estado del arte al inicio de este ciclo

Al cierre del diario anterior (`03_3_diario_tareas.md`), el pipeline de decoding había producido los siguientes resultados sobre sub-27:

**Tarea binaria Pre vs Post (script 27b) — chance = 50%:**

| Feature set | Acc | AUC |
|---|---|---|
| bandpower_welch | 68.9% | 0.725 |
| raw_pca | 65.5% | 0.696 |
| tde_cov_full | 62.2% | 0.624 |
| tde_cov_diag | 61.5% | 0.664 |

**Tarea 4 clases temporales (script 34) — chance = 25%:**

| Feature set | Acc | AUC |
|---|---|---|
| tde_cov_diag (npc=17) | 37.7% | 0.643 |
| raw_pca | 37.4% | 0.614 |
| tde_cov_full | 35.6% | 0.622 |
| bandpower_welch | 30.9% | 0.570 |

**Pregunta central de Enzo:** ¿estas performances son estadísticamente significativas, o podrían ser producto del azar?

La respuesta a esta pregunta determina si las interpretaciones actuales son válidas o deben revisarse. Todo lo que sigue se ordena en torno a ella.

---

## Tarea 1: Validación de Significancia Estadística (Test de Permutación) ✅

### Definición

Verificar si las accuracies observadas (script 34, 4 clases) son robustas comparadas con una distribución nula construida con etiquetas aleatorias.

**Modelos a evaluar:** 3 feature sets del script 34 — bandpower_welch, tde_cov, raw_pca.

**Diseño del test:**

1. Repetir el pipeline LORO completo N veces.
2. En cada repetición, permutar las etiquetas de clase **dentro de cada run** (no entre runs, para conservar la variabilidad inter-run en la distribución nula).
3. El p-valor es la proporción de permutaciones que superan o igualan la accuracy observada: `p = #{acc_perm >= acc_obs} / N`.
4. Criterio de significancia: p < 0.05.

**Por qué permutar dentro de runs:** si se permutan entre runs, el clasificador podría aprender diferencias sistemáticas entre runs (drift de impedancia, estado del sujeto) que no tienen nada que ver con el estímulo.

**Implementación:** flag `--permute N` en `scripts/validation/34_decoding_4class.py`. Optimizaciones: C=1.0 fijo con lbfgs (mismo clasificador en LORO observado y permutaciones), splits pre-computados, paralelización across permutaciones (threads, n_jobs=-1).

**Nota metodológica:** tanto el LORO observado como las permutaciones usan `LogisticRegression(C=1.0, lbfgs)`. Esto garantiza que se compara el mismo estimador con etiquetas reales vs aleatorias, sin inflar artificialmente la accuracy observada con inner CV.

### Resultados definitivos (n=1000 — 2026-03-31, C=1.0 fijo)

| Feature set | Acc obs | Null mean ± std | p-valor | z-score |
|---|---|---|---|---|
| bandpower_welch | 31.7% | 25.0% ± 1.1% | **0.000** | 6.00 |
| tde_cov | 34.9% | 24.9% ± 1.2% | **0.000** | 8.62 |
| raw_pca | 37.4% | 25.0% ± 1.2% | **0.000** | 10.61 |

Los 3 modelos son estadísticamente significativos (p < 0.001). Ninguna permutación de las 1000 igualó la accuracy observada. Los z-scores confirman señal robusta, especialmente en raw_pca (10.6σ) y tde_cov (8.6σ).

**Tiempos reales (12 workers, n_jobs=-3):**

| Feature | 1 perm | n=1000 (wall) |
|---|---|---|
| bandpower_welch | 0.4s | 8.6 min |
| tde_cov | 0.2s | 6.4 min |
| raw_pca | 0.4s | 9.0 min |

### Criterio de éxito y consecuencias

- **Si p < 0.05:** validar las interpretaciones de `03_3_diario_tareas.md` y proceder con la Tarea 3.
- **Si p ≥ 0.05:** revisar pipeline; candidatos — más sujetos, revisión del preprocesamiento, features alternativos.

---

## Tarea 2: Clarificación Metodológica del PCA ✅

### Definición

Entender con precisión qué hace el PCA en cada pipeline, para poder explicarlo ante Enzo y Diego. La duda técnica central: en `raw_pca`, ¿el recorte a 160 componentes ocurre sobre el dominio del tiempo, los canales, o un flatten espacio-temporal?

### Solución

*Verificado leyendo `scripts/validation/34_decoding_4class.py`, `src/campeones_analysis/luminance/tde_glhmm.py` y `features.py`. Shapes para sub-27, sfreq=250 Hz, ventana 250ms.*

#### Tabla comparativa

| Aspecto | bandpower_welch | raw_pca | tde_cov |
|---|---|---|---|
| **Input por ventana** | (32 ch, 63 muestras) | (32 ch, 63 muestras) | (32 ch, 63 muestras) |
| **Transformación previa** | Ninguna | Flatten → (2016,) | TDE ±10 lags → (63, 672) |
| **¿Se aplica PCA?** | No | Sí | Sí |
| **Observaciones en el fit** | — | ~6.000 ventanas de train (por fold) | ~75.600 timepoints concatenados (por fold) |
| **Features en el fit** | — | 2.016 (32 ch × 63 muestras) | 672 (32 canales × 21 lags TDE) |
| **Dominio del PCA** | — | Espacio-tiempo aplanado | Temporal (cada timepoint = observación) |
| **Componentes** | — | 160 | 17 |
| **Feature final** | (160,) = 32×5 bandas | (160,) scores de PCA | (153,) covarianza upper triangle |

#### Pipeline 1: `bandpower_welch` — Sin PCA

```
Ventana: (32, 63)
→ Por canal: Welch PSD → integrar 5 bandas
→ Output: (160,) = 32 canales × 5 bandas
→ Directo al clasificador
```

Los 160 features son directamente interpretables: potencia espectral por banda por canal.

#### Pipeline 2: `raw_pca` — PCA sobre espacio-tiempo aplanado

```
Ventana: (32, 63)
→ .reshape(-1) → (2016,)   # flatten: [ch0_t0, ch0_t1, ..., ch31_t62]

Por fold LORO:
  X_train: (~6.000 ventanas, 2.016 features)   ← PCA se ajusta aquí
  StandardScaler → PCA(160)
  → X_train_pca: (~6.000, 160)
  → X_test_pca:  (~250, 160)
```

**Respuesta a la duda técnica:** el PCA opera sobre el espacio **espacio-temporal aplanado** de 2.016 dimensiones — no sobre tiempo ni canales por separado. Cada componente es una combinación lineal de todos los pares (canal, muestra_temporal). Captura patrones espacio-temporales conjuntos (e.g. un ERP con forma y topografía específicas). Los 160 componentes son las 160 direcciones de mayor varianza estimadas sobre ~6.000 ventanas de entrenamiento.

**Implicación:** `raw_pca` puede capturar ERPs que `bandpower` no ve, porque opera sobre amplitud cruda sin asumir estructura frecuencial.

#### Pipeline 3: `tde_cov` — PCA sobre espacio TDE-expandido

```
Ventana: (32, 63) → TDE ±10 lags → (63, 672)   # 32 ch × 21 lags

Por fold LORO (PCA global):
  Concatenar todos los timepoints de train:
    ~75.600 timepoints × 672 features   ← PCA se ajusta aquí
  PCA(17) → 17 componentes = 83-85% varianza explicada

Por ventana individual:
  (63, 672) → proyectar → (63, 17)
  Covarianza temporal → (17, 17) → upper triangle → (153,) features
```

**Dominio del PCA:** temporal — cada timepoint es una observación en el espacio de 672 dimensiones TDE. Los 17 PCs son los 17 modos dinámicos más comunes en la señal.

**Diferencia clave con `raw_pca`:**
- `raw_pca`: observación = ventana completa (250ms). Captura variabilidad *entre* ventanas.
- `tde_cov`: observación = 1 timepoint en espacio TDE. Captura estructura *dentro* de la dinámica temporal, luego la resume via covarianza.

La diagonal de la covarianza (potencia por PC) captura cuánta energía hay en cada modo dinámico; los off-diagonal capturan coherencia entre modos.

---

## Tarea 3: Decoding 3 clases en Ensayos de Luminancia (60s) ✅ Completada

*Condicionada al éxito de la Tarea 1 — **desbloqueada** dado p=0.000 en los 3 feature sets.*

### Definición

Dentro de cada ensayo de **luminancia continua (60 segundos)** — hay uno por run, ~7 en total — predecir a nivel de ventana si el EEG corresponde a un momento de:

- **ChangeUp**: subida de luminancia en el canal verde
- **ChangeDown**: bajada de luminancia en el canal verde
- **NoChange**: sin cambio significativo

Tarea de **clasificación 3 clases**, con LORO CV (Leave-One-Run-Out) y los **mismos 3 feature sets** que la Tarea 1 (bandpower_welch, tde_cov, raw_pca).

### Diferencia con intentos anteriores

Dos scripts previos abordaron problemas parecidos pero no idénticos:

- [`scripts/validation/27_decoding_photo_change.py`](../scripts/validation/27_decoding_photo_change.py): clasificación **binaria** (CHANGE_PHOTO vs NO_CHANGE_PHOTO) en los eventos breves de foto — no en los ensayos de 60s. Resultados no alentadores.
- [`scripts/modeling/20_change_classifier.py`](../scripts/modeling/20_change_classifier.py): clasificación **binaria** (cambio vs estabilidad) dentro de los ensayos de 60s, con pipeline de GLHMM distinta. Resultados pobres.

**Lo nuevo:** diseño **3 clases** (ChangeUp / ChangeDown / NoChange) sobre los ensayos de 60s, con la misma arquitectura de ventanas deslizantes y feature sets benchmarkeados en la Tarea 1.

### Diseño técnico

| Elemento | Descripción |
|---|---|
| Datos | Epoch `video_luminance` (~60s, 1 por run, 7 runs) |
| Señal de referencia | CSV verde: `green_intensity_video_{id}.csv` (`timestamp`, `luminance` 0–255) |
| Ventanas EEG | 250ms, 6 offsets [0,50,100,150,200,250ms], tope derecho 500ms |
| Clases | 0=NoChange, 1=ChangeUp, 2=ChangeDown |
| Derivada de luminancia | frame-level: `ΔL[t] = L[t] - L[t-1]` |
| Umbral ChangeUp/Down | `\|ΔL\| > 2.0` (frame-level) |
| Umbral NoChange | `\|ΔL\| < 1.0` en ventana actual **Y** estabilidad en el último 1s previo |
| Feature sets | bandpower_welch, tde_cov, raw_pca |
| CV | LORO (Leave-One-Run-Out, 7 folds) |
| Clasificador | LogisticRegression(C=1.0, lbfgs) |
| Chance level | 33.3% (3 clases) |

**Nota sobre NoChange:** el 1s de estabilidad previa filtra ventanas de "post-cambio" que aún no volvieron al baseline. Solo se etiquetan como NoChange los períodos donde el sistema lleva ≥1s sin ningún cambio detectable.

### Distribución de ventanas (sub-27, 7 runs)

| Run | NoChange | ChangeUp | ChangeDown | Total |
|---|---|---|---|---|
| run-002 (vid12 p1) | 108 | 48 | 60 | 216 |
| run-003 (vid9 p1) | 66 | 36 | 30 | 132 |
| run-004 (vid3 p1) | 66 | 30 | 36 | 132 |
| run-006 (vid7 p1) | 24 | 18 | 6 | 48 |
| run-007 (vid12 p2) | 108 | 48 | 60 | 216 |
| run-009 (vid9 p2) | 66 | 36 | 30 | 132 |
| run-010 (vid7 p2) | 24 | 18 | 6 | 48 |
| **Total** | **462** | **234** | **228** | **924** |

NoChange representa ~50% de las ventanas (vs 33.3% de chance uniforme). Las clases ChangeUp y ChangeDown están aproximadamente balanceadas (~25% cada una).

### Resultados definitivos con test de permutación (sub-27, 2026-04-01, n=1000)

| Feature set | Acc | F1 (macro) | AUC | Null mean±std | p-valor | z-score |
|---|---|---|---|---|---|---|
| bandpower_welch | 41.5% | 0.357 | 0.524 | 38.6% ± 2.1% | 0.090 | 1.31 |
| tde_cov | 38.3% | 0.302 | 0.501 | 36.8% ± 2.1% | 0.256 | 0.70 |
| raw_pca | 42.1% | 0.357 | 0.541 | 38.9% ± 1.6% | **0.022** | 2.05 |

**Chance nominal: 33.3% — Chance efectivo (nulo): ~37–39%**

#### Accuracy por clase

| Feature set | NoChange | ChangeUp | ChangeDown |
|---|---|---|---|
| bandpower_welch | 58.7% | 18.4% | 30.3% |
| tde_cov | 61.0% | 11.1% | 20.2% |
| raw_pca | 60.8% | 25.6% | 21.1% |

### Interpretación de los resultados

**1. El hallazgo central: el chance efectivo no es 33.3% sino ~38–39%.**

Este es el resultado más importante del test de permutación. La distribución nula (etiquetas aleatorizadas dentro de cada run) no converge en 33.3% sino en **~37–39%**, porque NoChange representa el 50% de las ventanas. Un clasificador que aprende el sesgo de la distribución de clases — sin capturar ninguna señal EEG real — naturalmente predice "NoChange" con más frecuencia y obtiene ~38% de accuracy global. El test de permutación revela ese baseline real.

**Consecuencia directa:** la "ganancia" observada de 38–42% sobre el 33.3% nominal es en gran parte artefacto del desbalance de clases. La ganancia real sobre el nulo es solo **0–3 puntos porcentuales**.

**2. Solo raw_pca es estadísticamente significativo (p=0.022).**

raw_pca supera el umbral α=0.05, pero con z=2.05 — un efecto modesto. bandpower_welch está en el límite (p=0.090) y tde_cov es claramente no significativo (p=0.256). Comparado con la Tarea 1, donde los 3 feature sets tenían p=0.000 y z>6, aquí la señal es mucho más débil.

**3. Por qué la señal es más débil que en la Tarea 1.**

En la Tarea 1 (4 clases en foto-eventos), los eventos eran temporalmente precisos: un flash de foto tiene un onset definido al milisegundo. En la Tarea 3, los "eventos" de cambio de luminancia son graduales (el video cambia suavemente) y la derivada del canal verde es ruidosa. La señal EEG en las ventanas de 250–500ms post-onset es más variable y el clasificador tiene menos qué aprender.

Además, los 6 offsets por evento (0–250ms post-onset) asumen que hay una respuesta EEG distribuida en esa ventana, pero si el cambio de luminancia es muy gradual, el onset preciso del procesamiento neural es incierto.

**4. tde_cov falla completamente en esta tarea (z=0.70, p=0.256).**

tde_cov captura estructura dinámica a través de la covarianza de modos TDE. Esto funciona bien para estados sostenidos (Tarea 1: 4 clases en epochs de 6s con ~24 ventanas por trial). Pero en la Tarea 3 cada run tiene solo 8–36 eventos de cambio, y las ventanas son cortas (250ms). Con tan pocos datos, la covarianza TDE per-ventana es muy ruidosa y no captura diferencias entre clases.

**5. raw_pca lidera porque captura ERPs transientes.**

El único modelo significativo (raw_pca) opera sobre la amplitud cruda de la señal, lo que le permite detectar respuestas N1/P2 occipitales que aparecen ~100–200ms post-onset de cualquier cambio visual. Esto es exactamente lo que distingue ChangeUp/Down de NoChange, aunque no necesariamente la *dirección* del cambio.

**6. Comparación directa con Tarea 1.**

| Métrica | Tarea 1 (4 clases, foto) | Tarea 3 (3 clases, luminancia) |
|---|---|---|
| Mejor acc obs. | 37.4% (raw_pca) | 42.1% (raw_pca) |
| Mejor z | 10.61 (raw_pca) | 2.05 (raw_pca) |
| N feature sets significativos | 3/3 | 1/3 |
| Null mean | ~25% (= chance nominal) | ~38–39% (>> chance nominal) |

La Tarea 1 es estadísticamente mucho más sólida. El null de la Tarea 1 converge en el 25% teórico porque las clases estaban balanceadas. Aquí el desbalance (50% NC, 25% cada cambio) enmascara la señal real.

### Diagnóstico y recomendaciones

El resultado actual no es un fracaso del paradigma — es una señal de que hay un **problema de diseño en la Tarea 3** que vale la pena corregir antes de generalizar a más sujetos:

1. **Rebalancear clases estrictamente**: submuestrear NoChange al mismo N que ChangeUp+ChangeDown/2, de modo que las 3 clases queden en 33-33-33. El null convergería en 33.3% y los resultados serían comparables con la Tarea 1.

2. **Examinar la distribución de eventos por run**: run-006 y run-010 tienen solo 48 ventanas totales (muy pocas), lo que puede hacer que esos folds sean muy ruidosos.

3. **Considerar una tarea binaria**: colapsar ChangeUp+ChangeDown en una clase "Change" contra NoChange (binario, como script 27). Con más muestras de cambio, el clasificador tendría más que aprender.

### Outputs guardados

- `results/validation/luminance_3class/sub-27/sub-27_3class_results.json`
- `results/validation/luminance_3class/sub-27/sub-27_3class_permutation.json`
- `results/validation/luminance_3class/sub-27/sub-27_3class_confusion.png`
- `results/validation/luminance_3class/sub-27/sub-27_3class_per_class_acc.png`
- `results/validation/luminance_3class/sub-27/diagnostic_plots/*_diagnostic.png` (7 plots)

---

## Orden de Ejecución

```
[Inicio] Estado actual: scripts 27b y 34 funcionando, sub-27 ✅
    │
    ├─► Tarea 2: Clarificación PCA ✅
    │   → Shapes verificados en el código
    │   → raw_pca = flatten espacio-temporal (2016 dims)
    │   → tde_cov = PCA sobre timepoints en espacio TDE (672 dims)
    │
    └─► Tarea 1: Test de permutación
            │
            ├── Prueba de concepto (N=10, 3 features) ✅
            │   → bandpower: z=4.90 | tde_cov: z=7.32 | raw_pca: z=14.39
            │   → Todos superan el nulo; tiempo estimado ~5 min para N=1000
            │
            ├── Permutaciones N=1000 (3 features) ✅
            │   → bandpower: p=0.000, z=6.00
            │   → tde_cov:   p=0.000, z=8.62
            │   → raw_pca:   p=0.000, z=10.61
            │
            ├── [Si p < 0.05 en al menos un modelo]
            │   → Validar interpretaciones de 03_3_diario_tareas.md
            │   └── Tarea 3: Predicción de luminancia percibida
            │
            └── [Si p ≥ 0.05 en todos]
                → Revisión de pipeline y supuestos
```

---

## Cierre e Iteración

Al finalizar, presentar resultados ante **Enzo** y **Diego**:

- **Hallazgos significativos:** proceder con el paradigma de pantallas verdes — decoding continuo de estados emocionales/afectivos.
- **Hallazgos no significativos:** redefinir el alcance; posiblemente ampliar a más sujetos.

La reunión de cierre debe incluir:
1. Tabla de p-valores por feature set (Tarea 1).
2. Descripción precisa de la arquitectura PCA (Tarea 2 ✅).
3. Propuesta de diseño para la predicción de luminancia percibida (Tarea 3, si aplica).
