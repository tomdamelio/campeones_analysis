# Log de Preprocesamiento - CAMPEONES

Este documento registra el estado del preprocesamiento de cada participante y sesión.

## Leyenda de Estados

- ✅ **OK**: Procesado correctamente, marcas válidas
- ⚠️ **PARCIAL**: Algunas sesiones válidas, otras perdidas
- ❌ **PERDIDO**: Sesión perdida, no se pueden generar marcas válidas
- ⏳ **PENDIENTE**: Aún no procesado
- 🔄 **REVISAR**: Requiere revisión manual adicional

---

## Registro por Participante

### Sub-25

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| 01 | a | 002 | ✅ OK | Marcas audio/photo válidas. Ajustadas manualmente correctamente. | 2025-01-18 |
| 01 | b | 006 | ❌ PERDIDO | Problemas con audio/photo. No se pudieron generar marcas válidas. Sesión perdida. | 2025-01-18 |
| 02 | a | 003 | ✅ OK | Marcas audio/photo válidas. Ajustadas manualmente correctamente. | 2025-01-18 |
| 02 | b | 007 | ❌ PERDIDO | Problemas con audio/photo. No se pudieron generar marcas válidas. Sesión perdida. | 2025-01-18 |
| 03 | a | 004 | ✅ OK | Marcas audio/photo válidas. Ajustadas manualmente correctamente. | 2025-01-18 |
| 03 | b | 008 | ❌ PERDIDO | Problemas con audio/photo. No se pudieron generar marcas válidas. Sesión perdida. | 2025-01-18 |
| 04 | a | 005 | ✅ OK | Marcas audio/photo válidas. Ajustadas manualmente correctamente. | 2025-01-18 |
| 04 | b | 009 | ❌ PERDIDO | Problemas con audio/photo. No se pudieron generar marcas válidas. Sesión perdida. | 2025-01-18 |

**Resumen Sub-25**: 4/8 sesiones válidas (50.0%) - **Todas las sesiones del day A válidas, todas las del day B perdidas**

---

### Sub-12

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

### Sub-13

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

### Sub-14

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

### Sub-16

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

### Sub-17

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

### Sub-18

| Task | Acq | Run | Estado | Notas | Fecha |
|------|-----|-----|--------|-------|-------|
| - | - | - | ⏳ PENDIENTE | - | - |

---

## Resumen General

| Participante | Sesiones Válidas | Sesiones Perdidas | Sesiones Pendientes | % Válido |
|--------------|------------------|-------------------|---------------------|----------|
| Sub-25 | 4 | 4 | 0 | 50.0% |
| Sub-12 | 0 | 0 | ? | - |
| Sub-13 | 0 | 0 | ? | - |
| Sub-14 | 0 | 0 | ? | - |
| Sub-16 | 0 | 0 | ? | - |
| Sub-17 | 0 | 0 | ? | - |
| Sub-18 | 0 | 0 | ? | - |

**Total procesado**: 4/8 sesiones válidas de Sub-25

---

## Problemas Comunes Identificados

### Audio/Photo Issues - Sub-25 Day B (Acq B)
- **Afectado**: Sub-25, todas las sesiones acq-b (runs 006, 007, 008, 009)
- **Problema**: Señal de audio extremadamente débil o ausente
  - Detección típica en acq-a: 8 picos de audio, 10 picos de photo, 3 coincidencias
  - Detección en acq-b: 1 pico de audio, 24 picos de photo, 0 coincidencias
- **Causa probable**: Fallo técnico en el canal de audio durante la grabación del day B
- **Impacto**: Todas las sesiones del day B perdidas (4/8 sesiones totales)
- **Solución**: No hay solución retroactiva. Las sesiones están perdidas definitivamente.
- **Recomendación**: Para futuros participantes, verificar la calidad de la señal de audio antes de iniciar la grabación

---

## Notas Metodológicas

### Criterios de Validación
1. **Marcas válidas**: Se detectan marcadores audiovisuales coincidentes entre canales AUDIO y PHOTO
2. **Ajuste manual**: Las anotaciones se ajustan manualmente para coincidir con los 7 eventos esperados
3. **Sesión perdida**: Cuando no es posible generar 7 anotaciones válidas debido a problemas técnicos

### Proceso de Preprocesamiento
1. Ejecutar `02_create_events_tsv.py` para generar eventos iniciales
2. Ejecutar `03_detect_markers.py` para detectar marcadores y fusionar con eventos
3. Editar manualmente las anotaciones en el visualizador MNE
4. Verificar que se generó el archivo `merged_events` correctamente

---

## Historial de Cambios

| Fecha | Cambio | Autor |
|-------|--------|-------|
| 2025-01-18 | Creación del documento. Registro inicial de Sub-25. | - |
| 2025-01-18 | Completado preprocesamiento de Sub-25. 4/8 sesiones válidas (day A completo, day B perdido). | - |

