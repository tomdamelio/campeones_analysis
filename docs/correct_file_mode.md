# Modo de Corrección de Archivos - detect_markers.py

## Descripción

El modo de corrección (`--correct-file`) permite editar manualmente archivos de eventos ya procesados por `detect_markers.py`. Esta funcionalidad es útil cuando:

- Se detectan discrepancias significativas en las duraciones de eventos
- Se necesita ajustar manualmente onsets o duraciones
- Se quieren corregir errores en la detección automática
- Se necesita añadir o eliminar eventos específicos

## Uso Básico

```bash
python scripts/preprocessing/detect_markers.py \
    --subject 14 \
    --session vr \
    --task 01 \
    --run 006 \
    --acq b \
    --correct-file
```

## Parámetros Adicionales

### Especificar directorio y descripción del archivo

```bash
python scripts/preprocessing/detect_markers.py \
    --subject 14 \
    --session vr \
    --task 01 \
    --run 006 \
    --acq b \
    --correct-file \
    --correct-file-dir merged_events \
    --correct-file-desc merged
```

### Guardado automático sin confirmación

```bash
python scripts/preprocessing/detect_markers.py \
    --subject 14 \
    --session vr \
    --task 01 \
    --run 006 \
    --acq b \
    --correct-file \
    --force-save
```

## Flujo de Trabajo

1. **Carga del archivo**: El script busca y carga el archivo especificado
2. **Análisis previo**: Muestra estadísticas del archivo (número de eventos, duraciones, tipos)
3. **Visualización detallada**: Lista todos los eventos línea por línea
4. **Edición interactiva**: Abre la ventana de MNE para edición manual
5. **Detección de cambios**: Compara las anotaciones originales con las editadas
6. **Resumen de cambios**: Muestra qué eventos fueron modificados
7. **Backup y guardado**: Crea backup del archivo original y sobrescribe con los eventos corregidos

## Instrucciones de Edición

### En la ventana interactiva de MNE:

- **Ajustar duraciones**: Arrastra los bordes de las anotaciones
- **Mover eventos**: Arrastra desde el centro de la anotación
- **Eliminar eventos**: Haz clic derecho sobre la anotación
- **Crear nuevos eventos**: Presiona 'a' y arrastra para seleccionar región
- **Ajustar escala**: Presiona 'j'/'k' para cambiar escala vertical
- **Finalizar**: Cierra la ventana para guardar cambios

## Estructura de Archivos y Backup

Los archivos se actualizan in-situ, creando un backup automático:
```
data/derivatives/merged_events/
└── sub-{subject}/
    └── ses-{session}/
        └── eeg/
            ├── sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-merged_events.tsv         # ← Actualizado
            ├── sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-merged_events.json        # ← Actualizado  
            ├── sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-merged_events.tsv.backup  # ← Backup del TSV
            └── sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-merged_events.json.backup # ← Backup del JSON
```

## Metadatos

El archivo JSON se actualiza automáticamente e incluye:
- **ProcessingHistory**: Historial de todas las correcciones realizadas
- **BackupCreated**: Ruta del archivo de backup creado
- **LastModified**: Fecha y hora de la última modificación
- **GeneratedBy**: Información de la herramienta utilizada

## Casos de Uso Típicos

### 1. Corrección de duración significativa

Cuando encuentres diferencias como:
```
¡ADVERTENCIA! Diferencia significativa en la duración del evento 2:
  Original: 104.00s
  Nueva: 87.15s
```

### 2. Ajuste fino de onsets

Para alinear mejor los eventos con las señales visuales.

### 3. Eliminación de eventos espurios

Cuando la detección automática incluye falsos positivos.

### 4. Adición de eventos perdidos

Cuando la detección automática pierde algunos eventos reales.

## Ejemplo Completo

```bash
# 1. Corregir archivo merged_events
python scripts/preprocessing/detect_markers.py \
    --subject 14 \
    --session vr \
    --task 01 \
    --run 006 \
    --acq b \
    --correct-file

# 2. El script mostrará:
#    - Información del archivo cargado
#    - Estadísticas de eventos
#    - Lista detallada de eventos
#    - Abrirá ventana interactiva

# 3. Después de editar y cerrar la ventana:
#    - Muestra resumen de cambios
#    - Guarda archivo corregido
#    - Proporciona próximos pasos

# 4. Validar resultado (archivo original actualizado)
bids-validator data/derivatives/merged_events
```

## Próximos Pasos

Después de la corrección:

1. **Revisar el archivo actualizado**
2. **Validar que los cambios son correctos**
3. **Si hay problemas, restaurar desde el backup (.tsv.backup)**
4. **Documentar los cambios realizados**
5. **Los backups pueden eliminarse una vez confirmada la corrección**

## Troubleshooting

### Archivo no encontrado
- Verificar que el archivo existe en el directorio especificado
- Usar `--correct-file-dir` y `--correct-file-desc` correctos

### Error en la visualización
- Verificar que los datos raw están disponibles
- Comprobar que MNE puede cargar los datos

### Cambios no detectados
- Asegurar que se realizaron modificaciones visibles
- Usar `--force-save` si es necesario 