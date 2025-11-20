# Guía para Testear Nuevos Sujetos

Esta guía describe el proceso completo para verificar las señales periféricas (EDA, ECG, RESP) de nuevos participantes en el experimento CAMPEONES.

## Requisitos Previos

- Entorno conda `campeones` activado:
  ```bash
  conda activate campeones
  ```

## Estructura de Archivos Esperada

Los datos crudos deben estar organizados siguiendo esta estructura:

```
data/sourcedata/xdf/
├── sub-XX/                                    # Carpeta del participante
│   ├── ses-VR/                               # Sesión VR
│   │   └── physio/                           # Archivos XDF aquí
│   │       ├── sub-XX_ses-vr_day-a_task-01_run-001_eeg.xdf
│   │       ├── sub-XX_ses-vr_day-a_task-02_run-002_eeg.xdf
│   │       └── ...
│   └── order_matrix_XX_A_block1_VR.xlsx      # Planillas de orden
│       order_matrix_XX_A_block2_VR.xlsx
│       order_matrix_XX_B_block1_VR.xlsx
│       └── ...
```

## Pasos para Testear un Nuevo Sujeto

### Paso 1: Subir los Datos

Colocar los archivos XDF del nuevo participante en:
```
data/sourcedata/xdf/sub-{subject}/ses-{session}/physio/
```

Ejemplo para el sujeto 20:
```
data/sourcedata/xdf/sub-20/ses-VR/physio/
```

### Paso 2: Convertir XDF a Formato BIDS (BrainVision)

Ejecutar el script de conversión para transformar los archivos `.xdf` a formato BrainVision (`.vhdr`, `.vmrk`, `.eeg`):

```bash
python -m src.campeones_analysis.physio.read_xdf --subject {subject}
```

**Ejemplo:**
```bash
python -m src.campeones_analysis.physio.read_xdf --subject 20
```

Este script:
- Procesa todos los archivos XDF del sujeto
- Convierte los datos a formato BrainVision
- Guarda los archivos en `data/raw/sub-XX/ses-vr/eeg/`
- Genera metadatos BIDS completos

**Salida esperada:**
```
INFO - ✅ Successfully processed sub-20_ses-vr_day-a_task-01_run-003_eeg.xdf
INFO - ✅ Successfully processed sub-20_ses-vr_day-b_task-01_run-008_eeg.xdf
...
INFO - 📊 RESUMEN SESIÓN VR:
INFO -    Archivos procesados: 11
INFO -    Archivos omitidos: 0
INFO -    Total archivos encontrados: 11
```

### Paso 3: Verificar Señales Fisiológicas

Ejecutar el script de verificación manual para inspeccionar las señales periféricas de cada run:

```bash
python scripts/sanity_check/test_check_physiology_manual.py --subject {subject} --task {task} --acq {acq} --run {run}
```

**Ejemplo para el sujeto 20, tarea 04, condición a, run 007:**
```bash
python scripts/sanity_check/test_check_physiology_manual.py --subject 20 --task 04 --acq a --run 007
```

#### Parámetros:
- `--subject`: ID del sujeto (ej: 20)
- `--task`: Número de tarea (ej: 01, 02, 03, 04)
- `--acq`: Condición experimental (a o b)
- `--run`: Número de run (ej: 001, 002, 003, etc.)

#### Comportamiento del Script:

El script abre ventanas interactivas de matplotlib en secuencia:

1. **Primero:** Abre plots de **EDA** (Actividad Electrodérmica / SCR)
   - Revisar la señal cruda
   - Verificar componentes tónico y fásico
   - Cerrar la ventana para continuar

2. **Segundo:** Abre plots de **ECG** (Electrocardiograma)
   - Revisar la señal cardíaca
   - Verificar detección de picos R
   - Cerrar la ventana para continuar

3. **Tercero:** Abre plots de **RESP** (Respiración)
   - Revisar la señal respiratoria
   - Verificar ciclos respiratorios
   - Cerrar la ventana para finalizar

### Paso 4: Repetir para Todas las Runs

Repetir el Paso 3 para cada combinación de tarea, condición y run del sujeto.

**Ejemplo de runs típicas para un sujeto:**
```bash
# Tarea 01, condición a
python scripts/sanity_check/test_check_physiology_manual.py --subject 20 --task 01 --acq a --run 003

# Tarea 01, condición b
python scripts/sanity_check/test_check_physiology_manual.py --subject 20 --task 01 --acq b --run 008

# Tarea 02, condición a
python scripts/sanity_check/test_check_physiology_manual.py --subject 20 --task 02 --acq a --run 004

# Tarea 02, condición b
python scripts/sanity_check/test_check_physiology_manual.py --subject 20 --task 02 --acq b --run 009

# ... y así sucesivamente
```

## Verificación de Archivos Generados

Después del Paso 2, verificar que se generaron los siguientes archivos en `data/raw/`:

```
data/raw/sub-XX/ses-vr/eeg/
├── sub-XX_ses-vr_task-01_acq-a_run-003_eeg.vhdr
├── sub-XX_ses-vr_task-01_acq-a_run-003_eeg.vmrk
├── sub-XX_ses-vr_task-01_acq-a_run-003_eeg.eeg
├── sub-XX_ses-vr_task-01_acq-a_run-003_channels.tsv
├── sub-XX_ses-vr_task-01_acq-a_run-003_events.tsv
├── sub-XX_ses-vr_task-01_acq-a_run-003_events.json
└── ...
```

## Qué Revisar en las Señales

### EDA (Actividad Electrodérmica)
- ✅ Señal sin saturación
- ✅ Componente tónico estable
- ✅ Respuestas fásicas visibles durante estímulos
- ❌ Artefactos o desconexiones

### ECG (Electrocardiograma)
- ✅ Picos R claramente detectados
- ✅ Ritmo cardíaco estable
- ✅ Señal sin inversión (si está invertida, el script la corrige automáticamente)
- ❌ Ruido excesivo o pérdida de señal

### RESP (Respiración)
- ✅ Ciclos respiratorios regulares
- ✅ Amplitud de señal adecuada
- ✅ Frecuencia respiratoria dentro de rango normal
- ❌ Artefactos de movimiento

## Troubleshooting

### Error: "No se encontraron archivos FIF"
- **Causa:** El script `test_check_physiology.py` busca archivos `.fif` en lugar de `.vhdr`
- **Solución:** Usar `test_check_physiology_manual.py` que lee directamente desde `data/raw/`

### Error: "No se encontró el archivo"
- **Causa:** Desajuste entre el número de run esperado y el archivo real
- **Solución:** Verificar los archivos disponibles en `data/raw/sub-XX/ses-vr/eeg/` y usar el número de run correcto

### Error: "unrecognized arguments"
- **Causa:** Error de sintaxis en el comando (espacios extra)
- **Solución:** Verificar que no haya espacios entre `--` y el nombre del argumento

### Advertencia: "Conflicting BIDSVersion"
- **Causa:** Versión de BIDS en `dataset_description.json` difiere
- **Solución:** Esta advertencia es informativa y no afecta el procesamiento

## Notas Adicionales

- El script de conversión (Paso 2) puede tardar varios minutos dependiendo del número de archivos
- Las ventanas de matplotlib deben cerrarse manualmente para avanzar a la siguiente señal
- Se recomienda revisar al menos una run de cada tarea para verificar la calidad general de los datos
- Los plots se muestran de forma interactiva y no se guardan automáticamente

## Referencias

- Ver `docs/arquitectura_datos.md` para más detalles sobre la estructura BIDS
- Ver `docs/scripts_preprocessing.md` para información sobre otros scripts de preprocesamiento
