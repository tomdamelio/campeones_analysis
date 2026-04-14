# Referencia: modo --from-recording

Este documento define el flujo que activa `/research-session --from-recording [filename]`.

## Invocación

```
/research-session --from-recording nombre_archivo.txt
```

El archivo debe estar en `research_diary/recordings/`. Puede ser:
- `.txt` o `.md`: transcript generado externamente (ej: Gemini) — leer directamente
- En el futuro: `.mp3`, `.m4a`, `.wav` → transcribir con Gemini API antes de procesar (ver extensión pendiente)

---

## Fase 1 — Leer y transcribir

Leer el archivo desde `research_diary/recordings/[filename]`.

Si es texto: proceder directamente a Fase 2.

---

## Fase 2 — Extraer información de la reunión

Parsear el contenido con atención al detalle. El objetivo es no perder ningún análisis mencionado, ninguna decisión tomada, ninguna pregunta abierta. Extraer:

**A. Contexto de la reunión**
- Fecha, participantes, tema general
- Estado del proyecto según lo discutido (qué funciona, qué no, qué cambió)

**B. Análisis a hacer**
- Listar TODOS los análisis mencionados, incluso los vagos o condicionales ("si X da bien, entonces probar Y")
- Para cada uno: qué se pide, por qué (si se explicó), y si hay orden de prioridad implícito o explícito

**C. Decisiones tomadas**
- Cambios metodológicos acordados
- Cosas descartadas y por qué
- Direcciones nuevas

**D. Preguntas abiertas**
- Dudas que quedaron sin resolver
- Cosas que el supervisor quiere entender mejor

**E. Próxima reunión / deadlines**
- Fechas mencionadas (convertir fechas relativas a absolutas usando la fecha actual)

---

## Fase 3 — Presentar extracción al usuario

Mostrar lo extraído en formato estructurado y preguntar:

> "Esto es lo que extraje de la grabación. ¿Hay algo que esté mal interpretado, que falte, o que quieras reformular antes de armar el diario?"

Esperar confirmación o correcciones antes de continuar.

---

## Fase 4 — Crear nuevo diario de tareas

Una vez confirmada la extracción, crear el nuevo archivo de diario siguiendo el template estándar del skill (`MM_NN_diario_tareas.md`). Las tareas pendientes vienen de la reunión:

- Cada análisis del punto B → tarea o subtarea
- Las decisiones del punto C → documentadas en el Contexto del diario
- Las preguntas abiertas del punto D → tareas de tipo "investigar / buscar literatura"
- Los deadlines del punto E → anotados en las tareas correspondientes

El diario nuevo arranca con `## Tareas pendientes` vacío de sesiones anteriores (ya todo viene de la reunión) y `## Sesión de hoy` vacío, listo para la próxima sesión de trabajo.

Preguntar al usuario si quiere arrancar a trabajar en alguna tarea ahora mismo o prefiere cerrar y arrancar en otra sesión.

---

## Notas

- La grabación original queda en `recordings/` como referencia permanente
- Si el transcript viene de Gemini, puede tener imprecisiones en nombres técnicos (ej: "loro cv" en lugar de "LORO-CV", nombres de librerías mal escritos) — corregirlos al extraer
- En el paso B, ser exhaustivo: mejor listar de más que perder un análisis que el supervisor mencionó de pasada
