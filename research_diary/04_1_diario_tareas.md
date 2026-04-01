# Diario de Tareas — Preparación Reunión Diego (2026-04-01)

**Proyecto:** campeones_analysis
**Fecha:** 2026-04-01
**Supervisores:** Enzo, Diego Vidaurre
**Contexto:** Preparación para reunión con Diego. Recapitulación de todo lo trabajado desde el último reporte, con referencias a plots y resultados clave.

---
## Preguntas abiertas luego del ultimo reporte con Diego

### 1. Sobre la comparación de feature sets

Diego señaló que el número de features sesga la comparación. En la configuración actual:
- bandpower_welch: 160 features (32 ch × 5 bandas)
- tde_cov: 153 features (upper triangle de covarianza 17×17)
- raw_pca: 160 features (PCA fijo a 160)

Los features ya son razonablemente comparables en número. Sin embargo, hay un antecedente relevante: los benchmarks binarios Pre/Post se corrieron con **`LogisticRegressionCV`** (cross-valida C con inner-CV dentro de cada fold LORO), y se observó que **C era muy variable entre folds**. La hipótesis es que con ~127 ventanas de entrenamiento por fold, la superficie de cross-validación de C es ruidosa — el estimador elegido varía entre folds, lo que hace menos limpia la comparación entre feature sets. Esta variabilidad probablemente refleja el **bajo N de muestras**, no una diferencia real en la complejidad del problema.

Para los análisis de 4 clases y luminancia continua se cambió a `LogisticRegression(C=1.0, lbfgs)` fijo, eliminando ese ruido. Queda pendiente explorar ridge con CV de C — ¿tiene sentido implementarlo como próximo paso, o es suficiente con C=1.0 como primer análisis?

### 2. Sobre la señal en luminancia continua vs foto-eventos

Los z-scores en la Tarea 4 clases de foto-eventos (6–11σ) son mucho más altos que en la Tarea 3 de luminancia continua (0.7–2.0σ). Las hipótesis:
- Los cambios de luminancia en los videos son graduales (no abruptos como los flashes de foto) → onset neural menos preciso
- Los foto-eventos tienen estructura de trial definida con precisión de ms; los eventos de luminancia continua son inferidos con ruido
- Menor N de eventos por run en luminancia continua

¿Es razonable esperar señal más débil en este paradigma, o hay algo más que ajustar antes de escalar a más sujetos?

### 3. Sobre la pregunta de "decoding continuo"

Diego preguntó en el chat:
> "¿Por qué esto está más cerca del escenario de regresión continua que lo que acabas de hacer?"

La respuesta que di: la tarea de 4 clases agrega resolución temporal dentro del trial. Pero Diego tiene razón en que sigue siendo clasificación discreta, no regresión continua.

**Pregunta para discutir:** ¿Cuál sería el diseño correcto para un primer paso de validación hacia la regresión continua? ¿Directamente regresión sobre la señal de luminancia del video (variable continua), usando una ventana deslizante del EEG? ¿O hay pasos intermedios?

### 4. Sobre la generalización a más sujetos

Todos los resultados hasta ahora son de **sub-27** (un solo sujeto). ¿Cuándo es el momento adecuado para escalar a los demás sujetos? ¿Esperamos a tener el pipeline de 1 sujeto completamente validado, o deberíamos correr al menos 2–3 para tener una primera idea de la variabilidad inter-sujeto?





---

## Resumen ejecutivo

Desde el último reporte a Diego se completaron cuatro bloques de trabajo:

1. **Decoding 4 clases temporales + test de permutación** — validación estadística sólida (Tarea 1 del ciclo anterior)
2. **Decoding 3 clases de luminancia continua (60s)** — nueva tarea sobre los segmentos de luminancia, con test de permutación


## Bloque 1: Decoding 4 clases temporales + Test de Permutación ✅

### Diseño (script 34)

4 condiciones definidas por posición temporal dentro del trial de foto (1s baseline → 1s cambio → 1s retorno):

| Clase | Ventana |
|---|---|
| Baseline | [-250, 0ms] pre-onset + [1500, 1750ms] post-offset |
| ChangeUp | [0, 500ms] — respuesta al onset del aumento |
| Luminance | [500, 1000ms] — luminancia sostenida |
| ChangeDown | [1000, 1500ms] — respuesta al offset |

Ventanas de 250ms, step 50ms (6 ventanas solapadas por trial), LORO CV, 74 trials × 7 runs, ~1776 ventanas totales.

### Resultados LORO (sub-27, chance = 25%)

| Feature set | Acc | AUC |
|---|---|---|
| raw_pca | **37.4%** | 0.617 |
| tde_cov | 34.9% | 0.604 |
| bandpower_welch | 31.7% | 0.570 |

→ Plots: [`results/validation/photo_decoding_4class/sub-27/sub-27_4class_confusion.png`](../results/validation/photo_decoding_4class/sub-27/sub-27_4class_confusion.png) | [`sub-27_4class_per_class_acc.png`](../results/validation/photo_decoding_4class/sub-27/sub-27_4class_per_class_acc.png)

### Test de permutación (n=1000, C=1.0 fijo)

Método: permutar etiquetas **dentro de cada run** (no entre runs) para preservar la variabilidad inter-run en la distribución nula.

| Feature set | Acc obs | Null mean ± std | p-valor | z-score |
|---|---|---|---|---|
| bandpower_welch | 31.7% | 25.0% ± 1.1% | **0.000** | 6.00 |
| tde_cov | 34.9% | 24.9% ± 1.2% | **0.000** | 8.62 |
| raw_pca | 37.4% | 25.0% ± 1.2% | **0.000** | 10.61 |

**Los 3 modelos son estadísticamente significativos (p < 0.001).** Ninguna de las 1000 permutaciones igualó la accuracy observada. El nulo converge exactamente en el 25% teórico porque las clases están balanceadas. Z-scores altos (6–11σ) indican señal muy robusta.

### Nota metodológica: consistencia estimador observado vs nulo

Un problema previo era que el LORO observado usaba `LogisticRegressionCV` (que cross-valida C con inner-CV) mientras que las permutaciones usaban C fijo. Esto **inflaba artificialmente la accuracy observada** relativa al nulo. Se corrigió usando `LogisticRegression(C=1.0, lbfgs)` fijo en ambos, garantizando que se compara el mismo estimador.

### Interpretación para Diego

La tarea de 4 clases captura **resolución temporal dentro del trial**: el modelo distingue en qué estado de la trayectoria de luminancia (onset, sostenido, offset, baseline) está el sujeto en cada ventana de 250ms. Los z-scores altos (especialmente raw_pca = 10.6σ) confirman que el EEG de sub-27 contiene información genuina sobre la dinámica temporal del estímulo visual.

---

## Bloque 2: Decoding 3 clases en luminancia continua (60s) ✅

### Motivación y diferencia con los bloques anteriores

Los análisis anteriores operaban sobre **foto-eventos** (trials de 1s con estructura temporal definida por el diseño experimental). El siguiente paso natural es trabajar sobre los **segmentos de luminancia continua** (60s por run), donde no hay trials y los eventos se infieren de la señal del video.

Esto es más cercano al escenario que nos interesa a largo plazo: decoding continuo de estados afectivos sin estructura de trial.

### Diseño (script 36)

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

### Resultados primera versión (threshold=2.0, NC sin 6 ventanas) + test de permutación

*(Versión antes de la corrección del diseño de épocas)*

| Feature set | Acc | F1 | AUC | p-valor | z |
|---|---|---|---|---|---|
| raw_pca | 42.1% | 0.357 | 0.541 | **0.022** | 2.05 |
| bandpower_welch | 41.5% | 0.357 | 0.524 | 0.090 | 1.31 |
| tde_cov | 38.3% | 0.302 | 0.501 | 0.256 | 0.70 |

**Hallazgo crítico del test de permutación:** el nulo empírico no convergía en 33.3% (chance nominal) sino en ~38–39%, porque NoChange representaba el 50% de las ventanas. Un clasificador que aprende el sesgo de clase obtiene ~38% sin capturar señal EEG real. Raw_pca fue el único significativo (p=0.022, z=2.05), muy por debajo de los z-scores de la Tarea anterior.

### Corrección del diseño y nueva versión en curso

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

### Resultados versión corregida (threshold=1.5, balance 33-33-33, n=1000 permutaciones)

| Feature set | Acc | F1 | AUC | Null mean±std | p-valor | z-score |
|---|---|---|---|---|---|---|
| bandpower_welch | 34.7% | 0.343 | 0.515 | 33.7% ± 1.8% | 0.314 | 0.55 |
| tde_cov | 31.8% | 0.319 | 0.472 | 33.7% ± 2.2% | 0.805 | −0.86 |
| raw_pca | 29.3% | 0.285 | 0.455 | 33.7% ± 1.3% | 1.000 | **−3.34** |

**Chance level: 33.3% — Null empírico: ~33.7% ✓ (ahora converge correctamente)**

**Ningún feature set es significativo. tde_cov y raw_pca están por debajo del chance (z negativo).**

### Interpretación — qué nos dice este resultado

**1. El nulo ahora converge correctamente en 33.7% ≈ 33.3%.** El balance de clases está funcionando bien. Los resultados previos (raw_pca p=0.022) eran en gran parte artefacto del desbalance.

**2. La tarea 3-clases no muestra señal con este diseño en sub-27.** Los tres feature sets caen dentro o debajo de la distribución nula. El diseño corregido "limpió" el artefacto pero también eliminó la señal aparente.

**3. raw_pca con z=−3.34 es llamativo.** Un z negativo tan pronunciado sugiere que el clasificador está aprendiendo algo sistemáticamente *contra-predictivo*. La hipótesis más probable: con 6 ventanas solapadas (200ms de overlap en 250ms de ventana) por evento, las ventanas son altamente correlacionadas entre sí. El PCA sobre el flatten espacio-temporal puede estar capturando la varianza del *offset temporal dentro de la época* (ventana 0ms vs ventana 250ms tienen formas distintas) en lugar de la diferencia entre clases. Esto podría crear un confound: el clasificador aprende "ventana de offset 250ms vs 0ms" en lugar de "ChangeUp vs NoChange".

**4. Hipótesis alternativa — confound temporal en NoChange.** Los eventos NC se seleccionan de regiones estables (1s sin cambio previo + 500ms forward). Esto puede hacer que los eventos NC se concentren en momentos del video de baja actividad general, mientras que ChangeUp/Down ocurren en transiciones. Si el EEG refleja algún cambio de estado general que no es específico al cambio de luminancia, raw_pca (que captura toda la varianza espacio-temporal) podría aprender ese confound en dirección incorrecta.

### Diagnóstico y próximos pasos

El resultado actual indica que **la Tarea 3-clases en su forma actual no extrae señal útil** de sub-27. Las opciones:

1. **Investigar el confound de las 6 ventanas solapadas**: probar con solo 1 ventana por evento (sin augmentación temporal) para ver si el z de raw_pca se normaliza. Si el z negativo desaparece, el problema es el overlap.

2. **Simplificar a tarea binaria (Change vs NoChange)**: colapsar ChangeUp+ChangeDown en una clase. Más muestras por clase, tarea más fácil — permite verificar si hay alguna señal antes de intentar discriminar dirección.

3. **Revisar el criterio de NC**: el requerimiento de 1s de estabilidad previa Y 500ms forward puede estar seleccionando períodos muy específicos del video que son sistemáticamente distintos en el EEG por razones no relacionadas al cambio de luminancia.

---


---

## Resumen de lo que está corriendo ahora

Script 36 con el nuevo diseño de épocas (threshold=1.5, balance 33-33-33, 6 ventanas NC) + permutaciones n=1000 para los 3 feature sets. Resultados esperados en ~10–15 minutos.

---

## Línea de tiempo de scripts producidos

| Script | Descripción | Estado |
|---|---|---|
| `scripts/validation/27_decoding_photo_change.py` | Binario CHANGE vs NO_CHANGE en foto-eventos | ✅ |
| `scripts/validation/34_decoding_4class.py` | 4 clases temporales en foto-eventos + permutaciones | ✅ |
| `scripts/validation/36_decoding_luminance_3class.py` | 3 clases (Up/Down/NoChange) en segmentos 60s | ✅ activo |
| `scripts/validation/37_visualize_luminance_epochs.py` | Visualización timeline épocas luminancia | ✅ |
