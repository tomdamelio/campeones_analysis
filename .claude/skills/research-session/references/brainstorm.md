  # Referencia: modo --brainstorm

  Este documento define el flujo que activa `/research-session --brainstorm <tema>`.

  ## Invocación

  ```
  /research-session --brainstorm <tema o pregunta abierta>
  ```

  Ejemplos:
  - `/research-session --brainstorm rol del beta ERD en respuestas afectivas`
  - `/research-session --brainstorm alternativas al feature set actual`
  - `/research-session --brainstorm hipótesis para los outliers de sub-27`

  ---

  ## Fase 1 — Encuadre

  Antes de invocar nada, identificar el tipo de brainstorm. Preguntar al usuario:

  > "¿Qué tipo de brainstorm querés?
  > (a) Explorar un fenómeno neurocientífico — preguntas abiertas sobre mecanismos
  > (b) Generar hipótesis nuevas — qué podríamos testear que todavía no testeamos
  > (c) Explorar alternativas metodológicas — para una decisión que ya está sobre la mesa"

  Confirmar el tipo antes de continuar. Esto define qué skill se invoca en Fase 2.

  ---

  ## Fase 2 — Invocar la skill apropiada

  ### Tipo (a) o (b) — exploración divergente

  Invocar la skill `scientific-brainstorming`. Es abierta, divergente, busca conexiones interdisciplinarias y desafía supuestos. Pasarle:
  - El tema/pregunta del usuario
  - Contexto del proyecto: EEG decoding de respuestas afectivas y perceptuales (luminancia), supervisores Enzo Tagliazucchi y Diego Vidaurre
  - Restricción: no asumir que ya hay datos — explorar como si la pregunta fuera nueva

  ### Tipo (c) — convergencia hacia una decisión

  Invocar `/ars-plan` en modo Socrático. Es más estructurada, hace preguntas guiadas para llegar a una opción justificable. Pasarle:
  - La decisión a tomar
  - Las alternativas conocidas que ya consideró el usuario
  - Las constraints del proyecto (compute disponible, tamaño muestral actual, herramientas en uso, Windows 11 sin paralelismo agresivo)

  **Importante:** `/ars-plan` está diseñado para planificar capítulos de paper. En este modo se usa **solo como motor Socrático** — ignorar cualquier salida que apunte a estructura de paper. Lo que interesa es el razonamiento guiado.

  ---

  ## Fase 3 — Documentar

  Guardar el output del brainstorm en:

  ```
  research_diary/brainstorm/brainstorm_<slug_tema>_YYYY-MM-DD.md
  ```

  Donde `<slug_tema>` es una versión kebab-case corta del tema (ej: `beta-erd-afectivo`, `alternativas-feature-set`).

  Estructura:

  ```markdown
  # Brainstorm: [tema]

  **Fecha:** YYYY-MM-DD
  **Tipo:** (a) fenómeno / (b) hipótesis / (c) metodológico
  **Contexto:** [1-2 líneas conectando con el estado actual del proyecto]

  * * *
  ## Pregunta de partida

  [la pregunta exacta como la formuló el usuario]

  * * *
  ## Exploración

  [transcript estructurado del brainstorm — preguntas hechas, respuestas exploradas, conexiones encontradas]

  * * *
  ## Ideas / hipótesis surgidas

  [lista numerada de ideas concretas que salieron, cada una con 2-3 líneas]

  * * *
  ## Próximos pasos sugeridos

  [lista de cosas que se podrían hacer si alguna idea se persigue: análisis, papers a leer, experimentos]

  * * *
  ## Caveats

  [supuestos del brainstorm, límites de la exploración, qué quedó afuera]
  ```

  **Regla de formato:** usar `* * *` como separador, NUNCA `---` (compatibilidad con pandoc del flujo HTML del diario principal, por si se referencia desde ahí).

  ---

  ## Fase 4 — Conexión con el diario activo

  Preguntar al usuario:

  > "¿Querés que agregue una mención a este brainstorm en el diario actual, y/o que alguna idea se convierta en tarea pendiente?"

  Si dice sí:
  - Agregar al diario activo (`MM_NN_diario_tareas.md`) una línea bajo `## Sesión de hoy`:
    ```markdown
    **Brainstorm — [tema]:** explorado en `brainstorm/brainstorm_<slug>_YYYY-MM-DD.md`. Ideas principales: [1-2 bullets].
    ```
  - Si alguna idea se convierte en tarea, agregarla a `## Tareas pendientes` del diario en formato estándar (párrafo con bold, NO checkbox).

  Si dice no, cerrar el modo dejando el archivo en `brainstorm/` para referencia futura.

  ---

  ## Lo que este modo NO hace

  - No escribe paper ni introducción de paper
  - No fuerza una conclusión: el brainstorm puede cerrarse con "más preguntas que respuestas" si así sale
  - No descarta ideas por ser ambiciosas — el filtro de viabilidad es para Fase 2 del flujo principal de research-session, no para acá
  - No toma decisiones por el usuario; solo expone el espacio de posibilidades
  - No se mezcla con `--lit-review`: si una idea requiere revisión bibliográfica, se sugiere correr `--lit-review` después como paso separado