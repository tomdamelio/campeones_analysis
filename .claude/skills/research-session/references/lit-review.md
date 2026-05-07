  # Referencia: modo --lit-review

  Este documento define el flujo que activa `/research-session --lit-review <tema>`.

  ## Invocación

  ```
  /research-session --lit-review <tema>
  ```

  Ejemplos:
  - `/research-session --lit-review decoding EEG de luminancia con MVNN`
  - `/research-session --lit-review feature engineering en BCI afectivo`
  - `/research-session --lit-review LORO vs LOSO en single-trial EEG`

  ---

  ## Fase 1 — Definir scope

  Antes de buscar nada, preguntar al usuario:

  > "Para encuadrar la búsqueda:
  > (a) ¿Tema concreto o pregunta específica? [si no quedó claro en la invocación]
  > (b) Ventana temporal — default: últimos 5 años. ¿Querés extenderla?
  > (c) Tipo de papers — ¿metodológicos, fenómeno neural, o ambos?
  > (d) Volumen aproximado — ¿5-10 / 10-20 / 20+ papers?"

  Confirmar las cuatro respuestas antes de continuar.

  ---

  ## Fase 2 — Invocar `/ars-lit-review`

  Disparar el modo lit-review de academic-research-skills, pasándole:
  - El scope definido en Fase 1 (tema, ventana, tipo, volumen)
  - Contexto del proyecto: EEG decoding, paradigma de luminancia, supervisores Enzo Tagliazucchi y Diego Vidaurre, herramientas usadas (MNE-Python, scikit-learn, GLHMM)
  - Instrucción explícita: **"esto NO es para escribir un paper. Es bibliografía anotada para decisiones de proyecto y reuniones con supervisores. Sin draft narrativo, sin discusión, solo bibliografía + anotación."**

  ARS produce bibliografía anotada en formato académico: cada entrada con cita, resumen breve, hallazgo principal, método, relevancia para el proyecto.

  ---

  ## Fase 3 — Verificación de citas (siempre)

  Para CADA paper del output de ARS, verificar que existe:

  - **PubMed**: usar `mcp__claude_ai_PubMed__search_articles` o `mcp__claude_ai_PubMed__lookup_article_by_citation`. Confirmar título, autores, año, PMID/DOI.
  - **bioRxiv / medRxiv**: usar `mcp__claude_ai_bioRxiv__search_preprints` o `mcp__claude_ai_bioRxiv__get_preprint` para preprints biológicos.
  - **arXiv / otros**: si ARS cita un paper que no aparece en PubMed ni bioRxiv (común para papers de ML puros), usar WebSearch para confirmar que existe en arXiv u otro repositorio.

  **Regla:** si una cita no se puede verificar, marcarla como `[NO VERIFICADO]` en el output final y explicarle al usuario qué se buscó. No silenciar citas no verificadas — exponerlas explícitamente. Las LLMs alucinan papers con frecuencia; este paso es no negociable.

  Llevar registro de:
  - Papers verificados → mantienen cita completa con DOI/PMID/arXiv ID
  - Papers no verificados → marcados con flag `[NO VERIFICADO]` y razón ("no encontrado en PubMed/bioRxiv/WebSearch con queries Q1, Q2")
  - Papers que ARS omitió pero la verificación encontró como relevantes → agregar manualmente con anotación propia

  ---

  ## Fase 4 — Síntesis con synthesis-supervisor (opcional)

  Una vez verificada la bibliografía, ofrecer al usuario:

  > "Ya tengo [N] papers verificados. ¿Querés que invoque al `synthesis-supervisor` para producir convergencias / contradicciones / gaps, o preferís dejar la lit-review como bibliografía anotada plana?"

  Si dice sí, invocar el agente (`agents/synthesis-supervisor.md`) pasándole los N papers verificados y la pregunta de fondo del scope. El output del agente se incorpora como sección final del documento de lit-review (ver Fase 5, sección "Síntesis").

  Si dice no, saltear esta fase.

  ---

  ## Fase 5 — Documentar

  Output va a:

  ```
  research_diary/lit_reviews/lit_review_<slug_tema>_YYYY-MM-DD.md
  ```

  Estructura:

  ```markdown
  # Lit-Review: [tema]

  **Fecha:** YYYY-MM-DD
  **Scope:** [tema, ventana temporal, tipo, volumen objetivo]
  **Volumen final:** [N verificados / M no verificados / K total]

  * * *
  ## Pregunta de fondo

  [la pregunta concreta que motiva la búsqueda]

  * * *
  ## Bibliografía anotada

  [tabla o lista con: cita completa | hallazgo principal | método | relevancia para el proyecto | estado de verificación]

  * * *
  ## Síntesis (si se hizo Fase 4)

  ### Convergencias
  ### Contradicciones
  ### Gaps

  * * *
  ## Caveats

  [ventana temporal usada, sesgos potenciales de selección, qué se buscó y NO se encontró, papers no verificados con razón concreta]
  ```

  **Regla de formato:** usar `* * *` como separador, NUNCA `---`.

  ---

  ## Fase 6 — Conexión con diario activo

  Preguntar al usuario:

  > "¿Alguna fila de la bibliografía debe convertirse en tarea pendiente? Ejemplos:
  > - 'leer paper X en profundidad'
  > - 'replicar metodología Y'
  > - 'comparar nuestros números con los de Z'"

  Agregar las tareas resultantes al diario activo (`MM_NN_diario_tareas.md`) bajo `## Tareas pendientes` en formato estándar (párrafo con bold, NO checkbox).

  Si el usuario quiere, también agregar una línea bajo `## Sesión de hoy` mencionando el lit-review:

  ```markdown
  **Lit-Review — [tema]:** [N] papers verificados en `lit_reviews/lit_review_<slug>_YYYY-MM-DD.md`. [1-2 hallazgos clave].
  ```

  ---

  ## Lo que este modo NO hace

  - No escribe introducción ni discusión de paper
  - No fuerza una narrativa: si los papers no convergen, lo expone tal cual
  - No descarta papers por contradecir resultados propios — al contrario, esos son los más valiosos para llevar a la reunión
  - No omite citas no verificables: las marca explícitamente con `[NO VERIFICADO]`
  - No reemplaza lectura profunda: la bibliografía anotada es índice, no resumen exhaustivo