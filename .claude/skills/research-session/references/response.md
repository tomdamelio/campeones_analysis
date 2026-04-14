# Referencia: modo --response

Este documento define el flujo que activa `/research-session --response`.

## Invocación

```
/research-session --response
[pegar aquí los comentarios, en texto libre]
```

El interlocutor puede ser cualquier persona relevante al proyecto: supervisor (Diego Vidaurre, Enzo Tagliazucchi), colaborador, colega, estudiante de doctorado, etc. El tono y nivel de detalle de la respuesta final se adapta según quién es.

---

## Fase 1 — Identificar el interlocutor y parsear los comentarios

Antes de clasificar, identificar quién escribió y qué tipo de intercambio es:

- **Supervisor** (Diego / Enzo): los comentarios suelen ser preguntas metodológicas profundas, pedidos de análisis, o redirecciones. La respuesta debe ser más formal y exhaustiva.
- **Colega / colaborador / estudiante de doctorado**: el intercambio es más horizontal. Los comentarios pueden ser curiosidad, comparación de metodologías, o pedido de compartir código/resultados. La respuesta puede ser más conversacional y directa.

Separar cada punto distinguible y clasificar en:

**Tipo A — Respondible con datos existentes**
Ya tenemos los resultados o plots necesarios. Solo hay que articularlos bien.

**Tipo B — Requiere análisis nuevo**
Los datos existentes no alcanzan. Hay que correr algo nuevo.

**Tipo C — Requiere búsqueda bibliográfica o razonamiento experto**
La pregunta es conceptual o metodológica. Consultar los agentes supervisores (modeler-supervisor o experimentalist-supervisor) usando PubMed o Context7.

Presentar la clasificación al usuario antes de continuar:

> "Identifiqué N puntos en los comentarios de [nombre / rol]:
> - Punto 1 [Tipo A]: ...
> - Punto 2 [Tipo B]: necesitaría correr X
> - Punto 3 [Tipo C]: lo busco en literatura
> ¿Arrancamos con este plan?"

---

## Fase 2 — Ejecutar por tipo

### Tipo A
Localizar los resultados relevantes en el diario o JSONs. Draftar la respuesta directamente.

### Tipo B
Flujo condensado de Fases 2+3 de research-session:
- Plantear el análisis al usuario y confirmar antes de implementar
- Correr el análisis
- Documentar en el diario (ver Fase 3)

### Tipo C
Invocar el agente correspondiente:
- Duda metodológica → `agents/modeler-supervisor.md`
- Duda neurofisiológica o de interpretación → `agents/experimentalist-supervisor.md`
Seguir las instrucciones de verificación con herramientas del agente antes de responder.

---

## Fase 3 — Documentar en el diario existente

Agregar al final del diario más reciente (`MM_NN_diario_tareas.md`):

```markdown
* * *
## Respuesta a [nombre] — YYYY-MM-DD

**Comentarios recibidos:**
[pegar o resumir los comentarios originales]

**Punto 1 — [resumen del punto]**
[respuesta, con resultados o análisis que respaldan]

**Punto 2 — [resumen del punto]**
[respuesta]
```

Si se corrieron análisis nuevos (Tipo B), incluir tabla de resultados y plots dentro del punto correspondiente.

Regenerar el reporte HTML después de actualizar el diario:
```
/c/Users/au805392/AppData/Local/Pandoc/pandoc.exe MM_NN_diario_tareas.md -o reports/MM_NN_diario_tareas.html --standalone --embed-resources --from markdown-yaml_metadata_block --include-in-header util/diary_header.html
```

---

## Fase 4 — Generar mensajes de respuesta

Crear `research_diary/slack/slack_[nombre]_YYYY-MM-DD_response.md` con los mensajes.

**Adaptar el tono según el interlocutor:**

- **Supervisor**: más estructurado, con resultados numéricos explícitos, reconociendo limitaciones, terminando con pregunta o próximo paso concreto.
- **Colega / colaborador / estudiante de doctorado**: más conversacional, directo, puede ser más especulativo ("me parece que...", "no estoy seguro pero..."). Si el intercambio implica compartir código o datos, señalarlo explícitamente.

**Reglas de estilo comunes:**
- Primera persona, casual pero científico
- Números concretos siempre
- Sin emojis ni frases genéricas de LLM
- Honesto sobre limitaciones y sobre qué no sabemos
- Indicar qué plots adjuntar (con path absoluto) en cada mensaje que los incluya

Mostrar los mensajes al usuario para aprobación antes de dar el modo por terminado.
