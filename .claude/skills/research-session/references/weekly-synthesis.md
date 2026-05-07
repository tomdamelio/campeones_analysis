  # Referencia: modo --weekly-synthesis

  Este documento define el flujo que activa `/research-session --weekly-synthesis [N_diarios]`.

  ## Invocación

  ```
  /research-session --weekly-synthesis              # default: desde la última reunión documentada
  /research-session --weekly-synthesis 3            # override: últimos 3 diarios
  ```

  ---

  ## Fase 1 — Definir ventana y audiencia

  ### Detección automática de la ventana (default)

  Buscar en los diarios recientes (`research_diary/MM_NN_diario_tareas.md`, ordenados de más nuevo a más viejo) la sección `## Respuesta a Diego` o `## Respuesta a Enzo` más reciente. Esa es la fecha de la última reunión / interacción documentada con supervisores.

  La ventana cubre desde el diario donde aparece esa sección (inclusive) hasta el diario más reciente (inclusive).

  Si no se encuentra ninguna sección `## Respuesta a [supervisor]`, default: últimos 4 diarios.

  ### Override manual

  Si el usuario pasa un número (`--weekly-synthesis 3`), usar los últimos N diarios sin auto-detección.

  ### Identificar audiencia

  Preguntar:

  > "¿Esta síntesis es para Diego (énfasis metodológico — CV, regularización, pipeline ML) o para Enzo (énfasis neurofisiológico — interpretación de señal, mecanismos, paradigma)?"

  Si la respuesta es "ambos", producir la síntesis con ambos énfasis señalados explícitamente en cada tema.

  Confirmar ventana detectada + audiencia antes de continuar:

  > "Voy a sintetizar [N] diarios — desde [MM_NN, fecha] hasta [MM_NN, fecha] — para [supervisor]. ¿Arrancamos?"

  ---

  ## Fase 2 — Leer y consolidar diarios

  Leer todos los `MM_NN_diario_tareas.md` en la ventana. Para cada uno extraer:

  - **Tareas completadas** (de la sección `## Conclusiones de la sesión` → "Lo que hicimos", si existe; de la sección `## Sesión de hoy` si la sesión está en progreso)
  - **Resultados numéricos** (accuracy, AUC, p-valores, effect sizes, con su sujeto / feature set / condición asociada)
  - **Decisiones metodológicas** y su justificación
  - **Preguntas abiertas / próximos pasos**
  - **Plots generados** (paths absolutos, identificados por nombre de archivo)

  Consolidar en una estructura intermedia (no escrita a disco aún) ordenada por **tema**, no por fecha. Detectar temas recurrentes que aparecen en múltiples diarios — esos suelen ser los más relevantes para la reunión.

  ---

  ## Fase 3 — Invocar `/ars-outline`

  Disparar el modo outline de academic-research-skills. **Instrucción explícita:**

  > "Esto NO es un paper. Es input para una reunión semanal con [supervisor]. Producir outline temático + evidence map. Sin prosa narrativa, sin introducción, sin discusión, sinabstract, sin conclusión narrativa. Solo: temas, sub-puntos, qué resultado / plot /
  decisión respalda cada sub-punto, y de qué diario viene cada cosa."

  ARS produce un outline jerárquico con evidence map: cada nodo del outline apunta a la fuente concreta dentro de los diarios consolidados.

  **Si ARS devuelve prosa de paper a pesar de la instrucción:** descartarla y pedir reformulación. La regla es estricta — outline + evidence map, nada más.

  ---

  ## Fase 4 — Identificación de gaps (opcional)

  Ofrecer al usuario:

  > "¿Querés que invoque al `synthesis-supervisor` sobre el outline para detectar:
  > - Contradicciones entre semanas (ej: una decisión metodológica de hace 2 diarios que ya no es coherente con un resultado del último)
  > - Gaps que valga la pena traer a la reunión
  > - Tensiones entre resultados propios y la literatura citada en los diarios?"

  Si dice sí, invocar `agents/synthesis-supervisor.md` con el outline + resultados consolidados como input. El output se agrega como última sección del documento de síntesis (ver Fase 5).

  ---

  ## Fase 5 — Documentar

  Output va a:

  ```
  research_diary/weekly_synthesis/synthesis_YYYY-MM-DD.md
  ```

  Estructura:

  ```markdown
  # Síntesis semanal — YYYY-MM-DD

  **Audiencia:** [Diego / Enzo / ambos]
  **Ventana cubierta:** [diario X.Y a diario X.Z, fechas concretas]
  **Diarios incluidos:** [lista]

  * * *
  ## Temas

  ### Tema 1 — [nombre]
  **Estado:** [completado / en progreso / bloqueado]
  **Evidencia:**
  - Resultado: [número concreto, sujeto, condición] — diario MM_NN, plot path
  - Decisión: [decisión + justificación] — diario MM_NN
  **Pendiente:** [qué falta de este tema]

  ### Tema 2 — [nombre]
  [...]

  * * *
  ## Evidence map

  [tabla compacta: claim → diario_origen → tipo (resultado / decisión / plot / cita)]

  * * *
  ## Preguntas para la reunión

  [lista de cosas concretas que necesitan input del supervisor]

  * * *
  ## Decisiones pendientes

  [cosas que el usuario quiere decidir antes de la reunión, o quiere validar en la reunión]

  * * *
  ## Gaps / contradicciones detectadas (si se hizo Fase 4)

  [output del synthesis-supervisor]
  ```

  **Regla de formato:** usar `* * *` como separador, NUNCA `---`.

  ---

  ## Fase 6 — Conexión con cierre normal del skill

  Este output es **input opcional** para la Fase 4 del flujo principal de research-session (cierre con HTML + Slack). Preguntar:

  > "¿Querés usar esta síntesis como base para los mensajes de Slack (Paso 3-4 de la Fase 4 del cierre normal), o solo como nota personal para la reunión?"

  **Si dice sí:** pasar la síntesis al flujo de generación de mensajes Slack — los mensajes se construyen sobre la estructura temática del outline en lugar de sobre el diario cronológico. Mantener todas las reglas de estilo del Slack (primera persona, números
  concretos, sin emojis, honesto sobre limitaciones, terminar con pregunta concreta).

  **Si dice no:** dejar el archivo en `research_diary` como referencia personal del usuario para la reunión.

  ---

  ## Lo que este modo NO hace

  - No genera HTML — eso lo hace la Fase 4 del flujo principal de research-session
  - No escribe los mensajes de Slack automáticamente — mantiene el paso de aprobación manual del usuario
  - No modifica los diarios fuente — solo los lee
  - No escribe paper: la regla "outline + evidence map, sin prosa de paper" se aplica estrictamente
  - No reemplaza la Fase 4 del cierre normal: la complementa cuando el usuario quiere una vista temática además de la cronológica
  - No inventa resultados que no estén en los diarios fuente: todo claim debe poder mapearse a un diario concreto vía el evidence map