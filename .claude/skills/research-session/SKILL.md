---
  name: research-session
  description: Gestiona sesiones de investigación de doctorado en neurociencias (proyecto campeones_analysis). Fluye en 4 fases:    
  inicialización del diario, planificación pedagógica de tareas, implementación incremental con validación, y cierre con síntesis + 
  presentación de avances (skill `slides`) + mensajes Slack. USAR SIEMPRE cuando el usuario invoque /research-session, quiera iniciar/continuar una sesión de investigación, o 
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

    **Modo alternativo — brainstorming estructurado:**
    Si el usuario invoca `/research-session --brainstorm <tema>`, activar el flujo definido en `references/brainstorm.md`. Sirve para explorar un tema antes de planificar implementación: preguntas Socráticas, alternativas, gaps. Output queda en
  `research_diary/brainstorm/`.

    **Modo alternativo — revisión de literatura estructurada:**
    Si el usuario invoca `/research-session --lit-review <tema>`, activar el flujo definido en `references/lit-review.md`. Produce bibliografía anotada en formato académico (no draft de paper), con verificación de citas en PubMed/bioRxiv/arXiv. Output queda en
  `research_diary/lit_reviews/`.

    **Modo alternativo — síntesis semanal para supervisores:**
    Si el usuario invoca `/research-session --weekly-synthesis [N_diarios]`, activar el flujo definido en `references/weekly-synthesis.md`. Produce una vista temática (no cronológica) del trabajo reciente — desde la última reunión por default — pensada como input al HTML + Slack del cierre, NO como draft de paper.

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

    1. **Descomponer la tarea en subtareas paralelizables por default.** Presentarlas como lista numerada y confirmar con el usuario antes de continuar. La
  descomposición **debe** buscar activamente paralelización en dos niveles:

       **Nivel 1 — Paralelización entre subtareas:**
       Cada subtarea debe ser **autocontenida**: inputs explícitos, output explícito (idealmente un archivo o conjunto de archivos con path conocido), cero estado       
  compartido con otras subtareas. Si dos subtareas son autocontenidas respecto a la misma tarea madre, por default deben poder correrse simultáneamente en pestañas      
  independientes de Claude Code, cada una con su propio `/clear`.

       **Nivel 2 — Paralelización dentro de una subtarea:**
       Cuando una subtarea se aplica sobre una colección de unidades independientes (N papers, N sujetos, N feature sets, N modelos, N condiciones experimentales),      
  descomponerla en **un prompt por unidad**. Cada prompt:
         - Se escribe a un archivo bajo una carpeta `prompts/` relativa al workspace de la subtarea.
         - Declara su input específico (ej: el PDF concreto que procesa), su output específico (ej: el `.md` que va a generar), y las restricciones (herramientas        
  permitidas, qué NO tocar).
         - Incluye el contexto mínimo necesario para que una pestaña fresca de Claude pueda ejecutarlo sin leer la conversación madre.
         - Termina con una **consolidación explícita**: un prompt adicional que lee todos los outputs por-unidad y los integra en un único documento.

       **Regla de diseño:**
       Preferir N subtareas/prompts autocontenidos y un paso final de consolidación, antes que una subtarea monolítica secuencial. El costo adicional (escribir N+1      
  prompts en lugar de 1) se amortiza con creces por el paralelismo real (N pestañas corriendo a la vez) y por el contexto limpio de cada pestaña.

       **Cuándo NO forzar paralelización:**
       - Si las subtareas son intrínsecamente secuenciales (la subtarea 2 necesita el output validado de la subtarea 1 para existir). En ese caso, documentar la
  dependencia explícitamente ("bloqueada por subtarea N") en lugar de paralelizar.
       - Si el costo de preparar prompts autocontenidos supera el beneficio (ej.: una subtarea de 10 minutos que no se repite).
       - Si la subtarea requiere interacción iterativa con el usuario que sería costosa de replicar en N pestañas.

       **Patrón de referencia aplicado en este proyecto:**
       La Subtarea 0 del pre-registro OSF se descompuso en 6 frentes (A–F) paralelizables entre sí (Nivel 1), más paralelización interna en los frentes A y F (Nivel 2:  
  un prompt por paper, 6 + 7 unidades independientes, más dos prompts de consolidación). Ver `research_diary/preregistro_notes/prompts/` como template reproducible. El  
  máximo paralelismo efectivo fue 17 pestañas simultáneas.

       **Presentación al usuario:**
       Cuando se le muestre la descomposición, **marcar explícitamente**:
       - Cuáles subtareas corren en paralelo (Nivel 1) y cuáles son secuenciales.
       - Qué subtareas se sub-paralelizan internamente (Nivel 2) y en cuántas unidades.
       - Dónde va a vivir cada prompt autocontenido (path al `prompts/`).
       - Cuál es el paso de consolidación y qué va a leer.
       Esto permite al usuario validar la estrategia de paralelización antes de escribir nada.
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
      - **modeler-supervisor** (`agents/modeler-supervisor.md`): invocar cuando haya decisiones no triviales de estadística, ML o validación — esquema de CV, regularización,
  comparación de modelos, interpretación de métricas, revisión del pipeline de otro investigador. Leer el archivo del agente y adoptar ese rol para responder.
      - **experimentalist-supervisor** (`agents/experimentalist-supervisor.md`): invocar cuando haya que interpretar resultados en términos neurofisiológicos, tomar decisiones de
   preprocesamiento EEG, contextualizar hallazgos en la literatura de neurociencias, o evaluar si un resultado tiene sentido biológico. Leer el archivo del agente y adoptar ese
  rol para responder.
      - **synthesis-supervisor** (`agents/synthesis-supervisor.md`): invocar cuando haya que integrar múltiples fuentes (papers, resultados de varios sujetos, varios feature
  sets), detectar contradicciones entre papers o entre resultados propios y la literatura, o identificar gaps. Útil sobre todo después de `--lit-review` o antes de
  `--weekly-synthesis`. Leer el archivo del agente y adoptar ese rol para responder.
      - Los tres agentes tienen instrucciones explícitas sobre cuándo usar PubMed, Context7, WebSearch (y bioRxiv para synthesis) para verificar claims antes de responder. Seguir
   esas instrucciones estrictamente — no afirmar nada no trivial sin verificación.
      - Los agentes pueden invocarse solos o en combinación cuando una pregunta tiene componentes metodológicos, neurofisiológicos, y/o de integración cross-source.
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

  ### Paso 2 — Armar la presentación de avances (skill `slides`)

  **En vez del reporte HTML**, invocar el skill **`slides`** (Skill tool con `skill: "slides"`) para
  construir —de forma colaborativa y slide por slide— una presentación Beamer plot-céntrica de los
  resultados de la sesión (theme SimpleDarkBlue + Helvetica + paleta tab20c). El deck queda en
  `research_diary/context/MM_NN/presentation/`. Seguir el flujo del SKILL.md de `slides`: inventario
  de plots (Glob de `context/MM_NN/**/figures/*.png` + los CSV de `tables/`) → outline topico→slides
  → esqueleto (portada/Pregunta/recap/roadmap) → slide por slide, con el **usuario marcando el ritmo**
  y dando los ajustes por slide (titulos-titular, recorte de suptitles, leyendas, etc.).

  (El HTML por pandoc queda deprecado como cierre; usarlo solo si el usuario lo pide explícitamente.)

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

  ### Paso 6 — Commit y push de cambios de código                                     
                                                         
  - Flujo diario al cerrar la sesión:
    a. git status para ver lo no commiteado/no trackeado.
    b. Agrupar en unidades de cambio coherentes (un commit por frente lógico, no commitear todo junto).
    c. Por cada unidad: git add <paths específicos> (nunca -A ni .) + mensaje con prefijo Conventional Commits (feat:, fix:, etc.), ≤72 chars, foco en el "porqué".
    d. git push una sola vez al final.
  - Cierre semanal extra (viernes / último día): verificar sync con origin/main, revisar ?? colgados, repasar git log --oneline -20 para coherencia narrativa, decidir
  qué hacer con branches abiertas.
  - Reglas: no mezclar código con cambios de entorno, nunca secretos, no pushear si hay tests rotos, omitir explícitamente el paso si la sesión no tocó código
  versionado.
  - Aclaración: aplica sólo a archivos versionados — research_diary/ y reportes ya están en .gitignore.

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
