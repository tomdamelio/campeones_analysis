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

## Tarea 1: Validación de Significancia Estadística (Test de Permutación)

### Objetivos

Verificar si las accuracies observadas (scripts 27b y 34) son robustas comparadas con una distribución nula construida con etiquetas aleatorias.

### Modelos a evaluar

- **Script 27b** (Pre vs Post): 3 feature sets — bandpower_welch, TDE, raw_pca.
- **Script 34** (4 clases): los mismos 3 feature sets.

### Diseño del test de permutación

**Procedimiento:**

1. Repetir el pipeline LORO completo N veces (ver abajo la decisión sobre N).
2. En cada repetición, permutar aleatoriamente las etiquetas de clase **dentro de cada run** (no entre runs, para conservar la estructura de variabilidad inter-run en la distribución nula).
3. Registrar la accuracy de cada permutación.
4. El p-valor es la proporción de permutaciones que superan o igualan la accuracy real observada:
   ```
   p = #{accuracy_perm >= accuracy_real} / N
   ```
5. Criterio de significancia: p < 0.05.

**Por qué permutar dentro de runs y no globalmente:** si se permutan etiquetas entre runs, el clasificador podría aprender diferencias sistemáticas entre runs (drift de impedancia, estado del sujeto, artefactos) que no tienen nada que ver con el estímulo. Permutar dentro de cada run mantiene esas diferencias intactas en la distribución nula, de modo que lo que estamos testeando es exclusivamente si la etiqueta de clase (pre/post, o clase 0-3) está relacionada con la señal.

### Decisión sobre número de permutaciones (N)

**Criterio de precisión estadística:**
- Con N=100: el p-valor tiene resolución de 0.01 → suficiente para una prueba de concepto.
- Con N=1.000: resolución de 0.001 → adecuado para reportar p < 0.05 con confianza.
- Con N=10.000: resolución de 0.0001 → necesario si se quiere reportar p < 0.001.

**Propuesta escalonada:**

**Paso 1 — Prueba de concepto (N=100, un solo feature set):**
- Correr solo con `bandpower_welch` en el script 34 (4 clases), que es el más rápido de calcular.
- Medir el tiempo de una permutación individual.
- Extrapolar el tiempo total para N=1.000 y N=10.000.
- Decidir N final basándose en esa estimación.

**Paso 2 — Permutaciones completas (N=1.000 o el N que resulte viable):**
- Correr para los 3 feature sets en los 2 scripts.
- Reportar: accuracy observada, distribución nula (media ± std), p-valor, z-score.

**Estimación de tiempo esperada (orden de magnitud):**
- Un fold LORO tarda ~X segundos (a medir en Paso 1). El pipeline completo son 7 folds.
- 1 permutación = 7 folds en paralelo (con joblib) → ~X segundos / factor_paralelización.
- N=1.000 permutaciones × ese tiempo = total estimado.
- Si el tiempo es > 1 hora para N=1.000 con un feature set, reconsiderar reducir N o usar un subset de folds.

**Output del test de permutación:**
```
Script 27b — Pre vs Post:
  bandpower_welch:  obs=68.9%  null_mean=??%  null_std=??%  p=??  z=??
  tde_cov:          obs=62.2%  ...
  raw_pca:          obs=65.5%  ...

Script 34 — 4 clases:
  bandpower_welch:  obs=30.9%  null_mean=??%  null_std=??%  p=??  z=??
  tde_cov:          obs=35.6%  ...
  raw_pca:          obs=37.4%  ...
```

### Criterio de éxito y consecuencias

- **Si las accuracies son significativas (p < 0.05):** validar las interpretaciones ya planteadas en `03_3_diario_tareas.md` (alfa occipital, ERPs transientes, etc.) y proceder con la Tarea 3 (predicción de luminancia percibida).
- **Si no son significativas:** revisar las interpretaciones; el pipeline puede estar capturando artefactos, estructura de run, o simplemente hay insuficiente señal en sub-27. Alternativas a explorar: más sujetos, features alternativos, revisión del preprocesamiento.

---

## Tarea 2: Clarificación Metodológica del PCA

*Esta tarea es conceptual — no requiere análisis adicionales sino entender con precisión lo que ya está implementado, para poder explicarlo correctamente ante Enzo y Diego.*

### El problema

En las conversaciones del proyecto se mencionan dos usos distintos de PCA que pueden confundirse fácilmente:

1. **PCA global en `tde_cov`**: se ajusta sobre todos los datos de entrenamiento del fold (no sobre una ventana individual). Reduce las 672 dimensiones TDE a 17 componentes. Luego, **para cada ventana**, se proyectan los datos sobre esos 17 PCs y se calcula la covarianza de 62 muestras × 17 PCs.

2. **PCA en `raw_pca`**: se ajusta sobre vectores aplanados (flatten) de cada ventana completa — es decir, para una ventana de 62 muestras × 32 canales = 1984 valores, se concatenan en un vector de 1984 dimensiones y se aplica PCA(160). Reduce esos 1984 valores a 160 componentes.

### Duda técnica específica a resolver

Para `raw_pca`: ¿el recorte a 160 componentes ocurre en el dominio del **tiempo**, del **espacio** (canales), o del **espacio-tiempo conjunto** (flatten)?

**Respuesta:** en `raw_pca` el flatten es espacio-temporal — se aplana la ventana entera (tiempo × canales) en un vector y luego PCA opera sobre ese vector. Los 160 componentes resultantes son combinaciones lineales de todas las muestras de todos los canales de la ventana. Esto contrasta con:
- `bandpower_welch`: opera por canal (espectro de cada canal independientemente) → los features son combinaciones de frecuencia × espacio.
- `tde_cov`: el PCA opera sobre las 672 dimensiones TDE en el eje **temporal** (cada muestra del segmento de entrenamiento es una observación), luego la covarianza opera sobre el eje temporal de la ventana → los features son combinaciones de **tiempo × canales** de una forma estructurada (primero TDE, luego PCA, luego cov).

### Por qué importa clarificarlo

La presentación ante Diego requiere poder responder con precisión: "¿qué está aprendiendo el PCA?". La respuesta incorrecta ("el PCA reduce canales") puede llevar a interpretaciones equivocadas sobre la naturaleza de los features.

### Acción concreta

Leer el código de las tres funciones de extracción de features en `34_decoding_4class.py` y `27b_decoding_pre_vs_post.py`, trazar el shape de los arrays en cada paso, y redactar una descripción en 3-5 líneas por feature set que sea verificable y presentable.

---

## Tarea 3: Predicción de Luminancia Percibida

*Esta tarea está condicionada al éxito de la Tarea 1 — solo se procede si las accuracies son estadísticamente significativas.*

### Motivación

El pipeline actual discrimina entre **estados temporales discretos** del protocolo (Pre/Post, o las 4 clases). El siguiente paso es aplicar el decoding a un problema más continuo y ecológico: predecir la **respuesta subjetiva a la luminancia** a partir de la señal EEG.

La variable objetivo no es la luminancia física del estímulo (conocida y controlada), sino la **luminancia percibida**, operacionalizada a partir de los **cambios en el canal verde** de la pantalla (proxy de luminancia visual).

### Diseño experimental

**Definición de eventos:**
- **Evento de "cambio":** muestrear momentos en que hay un cambio real de luminancia en el canal verde (eventos CHANGE_PHOTO o equivalentes del paradigma de pantallas verdes).
- **Control ("no cambio"):** muestrear momentos aleatorios del canal verde donde no hay cambio de luminancia, con la misma distribución temporal que los eventos reales.

**Tarea de decoding:**
- **Contraste:** "momento de cambio de luminancia" vs "momento sin cambio".
- **Modelo base:** usar el mejor feature set de las tareas anteriores (presumiblemente `bandpower_welch` para binario, `tde_cov_diag` para 4 clases, o el que resulte óptimo tras la validación).
- **Variable predictora:** cambios en el canal verde (proxy de luminancia percibida).

**Análisis comparativo — luminancia real vs percibida:**
- Modelar el **delay temporal** entre el cambio físico de luminancia (onset del estímulo) y la respuesta neural (pico de decodificación).
- Comparar con el barrido temporal del script 10.2 (pico de accuracy ~130-330ms post-onset), que ya es una medida indirecta de este delay.
- Posible análisis adicional: correlación entre la magnitud del cambio de luminancia (verde) y la magnitud de la respuesta neural (accuracy del clasificador o peso de los features).

### Pendientes antes de empezar

1. ✅ Tarea 1: confirmar significancia estadística.
2. ✅ Tarea 2: aclarar arquitectura del PCA para seleccionar el modelo correcto.
3. Definir exactamente qué señal del canal verde usamos como proxy de luminancia — ¿intensidad media del frame? ¿cambio frame a frame? ¿banda de frecuencia del video?
4. Verificar disponibilidad de la señal de canal verde sincronizada con el EEG en los archivos de datos.

---

## Orden de Ejecución

```
[Inicio] Estado actual: scripts 27b y 34 funcionando con 3 feature sets, sub-27 ✅
    │
    ├─► Tarea 2: Clarificación PCA (conceptual, sin análisis)
    │   → Leer código, trazar shapes, redactar descripción por feature set
    │   → Criterio de cierre: puedo explicar sin ambigüedad qué hace el PCA en cada pipeline
    │
    └─► Tarea 1: Test de permutación
            │
            ├── Paso 1a: Prueba de concepto (N=100, bandpower, script 34)
            │   → Medir tiempo de 1 permutación
            │   → Estimar tiempo total para N=1.000 y N=10.000
            │
            ├── Paso 1b: Decidir N final según estimación de tiempo
            │   → Umbral propuesto: si N=1.000 tarda < 2h para los 3 features, proceder
            │   → Si no: N=500 o reducir a 1 script primero
            │
            └── Paso 1c: Permutaciones completas (N elegido)
                → Correr 3 features × 2 scripts
                → Output: p-valor y z-score por feature set y tarea
                │
                ├── [Si p < 0.05 en al menos un modelo]
                │   → Validar interpretaciones de 03_3_diario_tareas.md
                │   └── Tarea 3: Predicción de luminancia percibida
                │       → Definir proxy de luminancia (canal verde)
                │       → Diseñar contraste cambio vs no-cambio
                │       → Modelar delay luminancia real vs percibida
                │
                └── [Si p >= 0.05 en todos]
                    → Revisión de pipeline y supuestos
                    → Candidatos a investigar:
                        • Más sujetos (salir de sub-27)
                        • Revisar preprocesamiento (artefactos, filtrado)
                        • Features alternativos
```

---

## Cierre e Iteración

Al finalizar los análisis de esta agenda, se presentarán los resultados ante **Enzo** y **Diego** para definir la continuidad del proyecto:

- **Hallazgos significativos:** proceder con el paradigma de pantallas verdes (Paso 6 del diario anterior) — decoding continuo de estados emocionales/afectivos.
- **Hallazgos no significativos:** redefinir el alcance del proyecto; posiblemente ampliar a más sujetos antes de interpretar cualquier resultado.

La reunión de cierre debe incluir:
1. Tabla de p-valores por feature set y tarea.
2. Descripción precisa de la arquitectura PCA (Tarea 2).
3. Propuesta de diseño para la predicción de luminancia percibida (si aplica).
