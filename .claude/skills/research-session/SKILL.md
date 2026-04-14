---
  name: research-session
  description: Gestiona sesiones de investigación de doctorado en neurociencias (proyecto campeones_analysis). Fluye en 4 fases:    
  inicialización del diario, planificación pedagógica de tareas, implementación incremental con validación, y cierre con síntesis + 
  reporte HTML + mensajes Slack. USAR SIEMPRE cuando el usuario invoque /research-session, quiera iniciar/continuar una sesión de investigación, o 
  mencione el diario de tareas del proyecto.
  user-invocable: true
---

  # Skill: research-session

  Este skill gestiona sesiones de investigación del proyecto `campeones_analysis` (EEG decoding de respuestas afectivas y perceptuales, i.e. cambios de luminancia).

  ## Cuándo usar este skill

  El usuario invoca `/research-session` para iniciar o continuar una sesión de trabajo en el proyecto. El skill fluye en 4 fases en orden.

  **Modo alternativo — respuesta a supervisores:**
  Si el usuario invoca `/research-session --response` seguido de los comentarios en texto libre, activar el flujo definido en `references/response.md`. El interlocutor puede ser un supervisor (Diego, Enzo), un colega, colaborador, o estudiante de doctorado. El tono y nivel de detalle de la respuesta se adapta según quién es.

  **Modo alternativo — intake de grabación de reunión:**
  Si el usuario invoca `/research-session --from-recording [filename]`, activar el flujo definido en `references/recording-intake.md`. El archivo debe estar en `research_diary/recordings/`. Este modo extrae tareas y decisiones de la reunión, las presenta al usuario para validación, y crea el nuevo diario de tareas a partir de eso. Este modo clasifica los comentarios, corre análisis suplementarios si es necesario, documenta en el diario existente, y genera mensajes de Slack de respuesta.

  ---

  Fase 1 — Inicialización

  1. **Leer el diario más reciente**: buscar en `./research_diary/` el archivo `.md` con el número más alto (ej: `04_2_diario_tareas.md`). Leerlo completo.

  2. **Decidir si continuar en el mismo diario o crear uno nuevo**:
     - Si el diario más reciente tiene contenido real en su sección `## Sesión de hoy` (es decir, la sesión fue interrumpida a mitad)
       **preguntar al usuario**: "El diario `[nombre_archivo]` tiene trabajo en progreso. ¿Continuamos en ese mismo archivo o abrimos uno nuevo para esta sesión?"
       - Si el usuario elige **continuar**: NO crear archivo nuevo. Trabajar directamente sobre el diario existente. Saltar al paso 3.
       - Si el usuario elige **nuevo**: seguir con el paso 2b.
     - Si el diario más reciente NO tiene contenido en `## Sesión de hoy` (sesión anterior cerrada limpiamente), ir directamente al paso 2b.

  2b. **Crear nuevo archivo de diario**: el nombre sigue el patrón `MM_NN_diario_tareas.md` donde `MM` es el mes actual y `NN` es el siguiente número secuencial (ej: si existe `04_2_`, crear `04_3_`). El archivo nuevo tiene esta estructura:

  ```markdown
  # Diario de Tareas: [título breve de la sesión]

  **Proyecto:** campeones_analysis

  **Fecha:** YYYY-MM-DD

  **Supervisores:** Enzo Tagliazucchi, Diego Vidaurre

  **Contexto:** [resumen 1 párrafo del estado actual extraído del diario anterior]

  * * *
  ## Tareas pendientes (continuadas del diario anterior)

  [copiar SOLO las tareas pendientes del diario anterior como párrafos de texto plano con bold, NO usar sintaxis de checkbox - ver formato más abajo]

  * * *
  ## Sesión de hoy

  [se irá completando durante la sesión]
  ```

  **Formato de tareas pendientes:** usar párrafos de texto plano, NO listas con `- [ ]` ni `- [x]` (generan problemas de renderizado en HTML). Ejemplo:

  ```markdown
  **Tarea 1 — Nombre de la tarea:** completada. Resultado resumido en una línea.

  **Tarea 2 — Otra tarea:** pendiente. Descripción breve de qué falta.
  ```

  **Regla crítica de formato para el diario:** usar `* * *` como separador horizontal, NUNCA `---`. El `---` lo interpreta pandoc como separador de tabla simple y rompe el renderizado HTML de las secciones que siguen.

  3. **Presentar tareas pendientes**: mostrar solo las pendientes y preguntar: "¿En qué tarea te querés enfocar hoy?". A priori la idea es avanzar secuencialmente o que juntos (claude + usuario) elijan la tarea más prioritaria.

  ---
  Fase 2 — Planificación (por tarea)

  Antes de escribir cualquier código:

  1. Descomponer la tarea en subtareas pequeñas y verificables. Presentarlas como lista numerada y confirmar con el usuario antes de continuar. Al descomponer, intentar que cada subtarea sea **autocontenida** (inputs y outputs claros, sin dependencias implícitas con el estado de otras subtareas) y, cuando sea posible, **paralelizable** (que dos subtareas puedan correrse simultáneamente sin interferir). Esto permite: (a) hacer `/clear` al terminar una subtarea para comprimir el contexto, (b) correr subtareas independientes en sesiones paralelas de Claude en worktrees distintos. Solo cuando esto tenga sentido — no forzarlo si las subtareas son intrínsecamente secuenciales.
  2. Explicar la matemática y neurociencia de cada subtarea siguiendo este orden fijo:
    - Intuición primero: ¿qué problema resuelve esto? ¿Por qué tiene sentido hacerlo?
    - Ejemplo concreto: con números pequeños o analogía del proyecto (ej: "imaginate que tenés 17 PCs y 7 runs LORO...")
    - Fórmula (si aplica): presentarla DESPUÉS del ejemplo, con cada término explicado en palabras
    - Conexión con la tarea: "entonces en nuestro código esto se traduce en..."
  3. Consultar literatura si hay incertidumbre metodológica:
    - Usar PubMed (mcp__claude_ai_PubMed__search_articles) o bioRxiv para validar el approach
    - Citar los papers relevantes en el diario
  4. Confirmar el enfoque con el usuario antes de implementar: "¿Arrancamos con este approach o querés explorar alguna alternativa?"
  5. Chequeo de pensamiento crítico: Antes de dar luz verde a la implementación, evaluar brevemente:
    - ¿El diseño puede responder la pregunta? (¿hay confounders no controlados? ¿el esquema de CV es apropiado?)
    - ¿Qué sesgos podrían afectar los resultados? (selección de épocas, data leakage, múltiples comparaciones sin corrección)
    - ¿Las métricas elegidas (accuracy, AUC) son las más adecuadas para la pregunta científica?
    - Si hay dudas metodológicas profundas, invocar `/scientific-critical-thinking` para análisis formal

  6. **Consultar agentes supervisores** cuando la tarea lo requiera (ver `agents/`):
    - **modeler-supervisor** (`agents/modeler-supervisor.md`): invocar cuando haya decisiones no triviales de estadística, ML o validación — esquema de CV, regularización, comparación de modelos, interpretación de métricas, revisión del pipeline de otro investigador. Leer el archivo del agente y adoptar ese rol para responder.
    - **experimentalist-supervisor** (`agents/experimentalist-supervisor.md`): invocar cuando haya que interpretar resultados en términos neurofisiológicos, tomar decisiones de preprocesamiento EEG, contextualizar hallazgos en la literatura de neurociencias, o evaluar si un resultado tiene sentido biológico. Leer el archivo del agente y adoptar ese rol para responder.
    - Ambos agentes tienen instrucciones explícitas sobre cuándo usar PubMed, Context7 y WebSearch para verificar claims antes de responder. Seguir esas instrucciones estrictamente — no afirmar nada no trivial sin verificación.
    - Los agentes pueden invocarse solos o en combinación cuando una pregunta tiene componentes tanto metodológicos como neurofisiológicos.
    - El usuario también puede pedir explícitamente consultar un agente en cualquier momento de la sesión.

  ---

  Principios pedagógicos obligatorios

  - Anclar en lo conocido: usar conceptos que el usuario ya maneja (LORO-CV, permutaciones, LogisticRegressionCV, PCA, bandpower) como punto de partida para introducir conceptos nuevos
  - Fórmulas como resumen, no como punto de partida: la fórmula resume lo que ya entendimos, no abre la explicación
  - Preguntar antes de desvíos largos: si hay una tangente matemática relevante, preguntar "¿Querés que profundice en esto o seguimos avanzando?" antes de desarrollarla
  - Validación neurofisiológica: al presentar resultados, siempre preguntarse y preguntar al usuario si los números tienen sentido biológico. Investigar juntos si algo parece off, o buscar papers que vayan en sintonía (o en contra) de lo encontrado.

  ---
  Fase 3 — Implementación

  Implementar un subtask a la vez. No avanzar al siguiente hasta que el usuario valide el actual.

  Reglas de implementación

  - Ejecutar scripts SIEMPRE con:
  micromamba run -n campeones python -m src.campeones_analysis.<module.path>
  - Nunca usar python directo ni python3.
  - Explicar el código paso a paso: conectar cada operación del código con la explicación matemática/intuitiva de la Fase 2.
  - Validar outputs junto al usuario:
    - ¿Los números son plausibles? (rangos esperados, distribuciones razonables)
    - ¿Tienen sentido neurofisiológico? (ej: accuracy por encima del chance, latencias coherentes)
    - Si algo es inesperado, investigar antes de continuar
  - Pensamiento crítico sobre resultados: ¿Las conclusiones son proporcionales a los datos, o estamos sobreinterpretando? Si hay dudas, invocar `/scientific-critical-thinking`
  - Documentar en el diario a medida que aparecen resultados: tablas de resultados, observaciones sobre plots, decisiones y su justificación
  - No implementar todo junto: Subtask → run → validate → document → next subtask.
  - Guardar resultados (preferentemente JSON, cuando esto haga sentido)

  ---
  Fase 4 — Cierre

  Al final de la sesión (cuando el usuario lo indique), ejecutar estos pasos en orden:

  ### Paso 1 — Síntesis global en el diario

  Agregar al diario la sección de conclusiones:

  ```markdown
  * * *
  ## Conclusiones de la sesión

  **Lo que hicimos:**
  - [bullet por subtask completado]

  **Resultados principales:**
  - [tabla o lista de resultados clave con números concretos]

  **Decisiones metodológicas:**
  - [decisiones y su justificación]

  **Próximos pasos:**
  - [ ] [tarea pendiente 1]
  - [ ] [tarea pendiente 2]
  ```

  ### Paso 2 — Generar reporte HTML

  Desde `research_diary/`, correr:

  ```
  /c/Users/au805392/AppData/Local/Pandoc/pandoc.exe MM_NN_diario_tareas.md -o reports/MM_NN_diario_tareas.html --standalone --embed-resources --from markdown-yaml_metadata_block --include-in-header util/diary_header.html
  ```

  El reporte queda en `research_diary/reports/`. Verificar que el archivo se generó correctamente (tamaño > 1MB si hay plots embebidos).

  ### Paso 3 — Preguntar qué priorizar para los mensajes de Slack

  Antes de redactar los mensajes, preguntar al usuario:

  > "¿Qué querés que le cuente a Diego? ¿Hay algún resultado particular que quieras destacar, o algún aspecto que prefería dejar afuera?"

  Esperar la respuesta antes de continuar.

  ### Paso 4 — Generar mensajes de Slack

  Crear el archivo `research_diary/slack/slack_diego_YYYY-MM-DD.md` con los mensajes. El archivo debe estructurarse como una secuencia de 3-5 mensajes separados, cada uno con indicación de qué plots adjuntar (con path absoluto).

  **Estructura típica de los mensajes:**
  - Mensaje 1: contexto general y conclusión de alto nivel
  - Mensajes 2-3: resultados específicos con plots
  - Mensaje final: pregunta abierta o próximos pasos

  **Estilo obligatorio para los mensajes de Slack:**
  - Sonar como el usuario escribe, no como una LLM genérica. Tono casual pero científico, directo, primera persona ("hice X", "da Y", "me parece que").
  - Números siempre concretos: accuracy, AUC, p-valores, nombre del feature set, número de sujetos.
  - Sin emojis.
  - Ser honesto sobre limitaciones: mencionar explícitamente si algo es un solo sujeto, si el N es bajo, si hay caveats metodológicos.
  - No usar frases genéricas de LLM como "los resultados son prometedores", "se observa una mejora significativa", "esto abre la puerta a".
  - Nombrar personas y herramientas específicamente (ej: "Yongjie Duan", "LORO-CV", "bandpower Welch").
  - Cada mensaje que incluya un plot debe indicar claramente qué plot es y por qué se incluye en ese punto de la narrativa.
  - El último mensaje debe terminar con una pregunta concreta para Diego o un próximo paso que involucre su feedback.

  **Verificación antes de enviar:** los claims deben ser proporcionales a la evidencia. No "probamos X" sino "encontramos evidencia de X en sub-27".

  Mostrar los mensajes al usuario para que los apruebe o ajuste antes de dar por finalizado el cierre.

  ### Paso 5 — Actualizar memoria del proyecto

  Actualizar `memory/project_state.md` si hay cambios de estado relevantes (nuevos resultados, decisiones metodológicas, cambios de dirección).

  ---
  Perfil del usuario

  - PhD en neurociencias, orientación matemática pero la math formal no es su expertise principal
  - Entiende bien el nivel conceptual de ML/stats: CV, regularización, PCA, permutation testing, etc
  - Se puede perder en el bajo nivel matemático: álgebra lineal, probabilidad, cálculo
  - Quiere aprender el bajo nivel — las explicaciones son parte del valor del skill
  - Supervisores: Enzo Tagliazucchi y Diego Vidaurre

  ---
  Notas técnicas del proyecto

  - Entorno: micromamba environment campeones
  - Módulos: src/campeones_analysis/<módulo>
  - Resultados: results/validation/ como JSON
  - Diario: research_diary/MM_NN_diario_tareas.md
  - Reportes HTML: research_diary/reports/MM_NN_diario_tareas.html
  - Mensajes Slack: research_diary/slack/slack_diego_YYYY-MM-DD.md
  - Utilidades pandoc: research_diary/util/diary_header.html, diary_style.css
  - Documentos de contexto extra (slides, papers, notas de reunión): research_diary/context/
  - Windows 11: evitar paralelismo agresivo (WinError 1450) — loops bash secuenciales, no múltiples jobs con n_jobs=-1
  - **Workflow para cambio de tarea con contexto limpio**: hacer `/clear` en Claude Code y luego invocar `/research-session`. El skill leerá el diario actualizado y continuará desde donde corresponda sin necesidad de recordar la sesión anterior.
