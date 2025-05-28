# Tests Documentation

Este directorio contiene los scripts de prueba y utilidades para el procesamiento de datos del proyecto. Los scripts están organizados en tres directorios principales:

- `./tests/read_files/`: Scripts para lectura y conversión de archivos
- `./tests/sanity_check/`: Scripts para verificación y validación de datos
- `./tests/preprocessing/`: Scripts para preprocesamiento de datos

A continuación se describe cada archivo y su uso:

## Archivos de Test en `./tests/read_files/`

### `test_read_xdf.py`
**Descripción**: Script interactivo para leer y procesar archivos XDF, diseñado para ser ejecutado en un entorno Jupyter/IPython. Este script realiza un procesamiento completo de datos de EEG, joystick y marcadores, con soporte para el formato BIDS.
```python
# Ejecutar en un entorno interactivo (Jupyter/IPython)
# Modificar las variables al inicio del script:
subject = '16'
session = 'VR'
task = '02'
run = '003'
acq = 'a'  # Reemplaza 'day' por 'acq' para compatibilidad BIDS
```
**Funcionalidad**:
- Lee y procesa archivos XDF con múltiples streams (EEG, joystick, marcadores)
- Procesamiento de EEG:
  - Corrección de tipos de canales
  - Resampleo a 250 Hz
  - Aplicación de montaje de canales
- Procesamiento de Joystick:
  - Extracción de datos de ejes X e Y
  - Resampleo a 250 Hz para sincronización con EEG
  - Mapeo y normalización de valores
- Procesamiento de Marcadores:
  - Conversión de marcadores a formato numérico
  - Creación de canal de estimulación (STI 014)
  - Sincronización con datos EEG
- Generación de estructura BIDS completa:
  - Archivos EEG en formato BrainVision (.vhdr)
  - Archivos de eventos (.tsv)
  - Metadatos de canales y eventos
  - Documentación de mapeo de canales y unidades
- Manejo de metadatos BIDS:
  - Creación/actualización de archivos de metadatos
  - Actualización de información de participantes
  - Generación de archivos channels.tsv
- Validaciones y verificaciones:
  - Comprobación de existencia de archivos
  - Verificación de sincronización entre streams
  - Validación de longitudes de señales

### `test_convert_xdf_to_bids.py`
**Descripción**: Script de línea de comandos para procesar archivos XDF con soporte completo para estructura BIDS.
**Uso**:
```bash
# Para datos reales:
python tests/read_files/test_convert_xdf_to_bids.py --subject <ID_SUJETO> [--skip-if-exists]

# Para datos de test:
python tests/read_files/test_convert_xdf_to_bids.py --subject <ID_SUJETO> --data-folder tests/test_data [--skip-if-exists]
```
**Funcionalidad**:
- Procesa **todos** los archivos XDF de un sujeto, sin importar el nombre del archivo.
- Extrae streams de EEG y markers.
- Crea estructura BIDS completa:
  - `/eeg/` para archivos FIF
  - `/events/` para archivos NPY de markers
- Realiza downsampling a 500 Hz
- Soporta saltar archivos existentes con `--skip-if-exists`
- Si el nombre del archivo no sigue el patrón esperado, igual lo procesa y usa valores por defecto para los nombres de salida.
- Permite especificar el directorio de datos con `--data-folder` (útil para test).

## Archivos de Test en `./tests/sanity_check/`

### `test_check_physiology.py`
**Descripción**: Script automatizado para generar visualizaciones de señales fisiológicas (EDA, ECG, RESP) para todos los archivos de un sujeto.
**Uso**:
```bash
python tests/sanity_check/test_check_physiology.py --subject <ID_SUJETO>
```
**Funcionalidad**:
- Procesa automáticamente todos los archivos FIF de un sujeto
- Genera y guarda plots de:
  - Actividad Electrodérmica (EDA)
  - Electrocardiograma (ECG)
  - Respiración (RESP)
- Guarda los plots en estructura de directorios:
  - `/plots/eda/` para plots de EDA
  - `/plots/ecg/` para plots de ECG
  - `/plots/resp/` para plots de RESP

### `test_check_physiology_manual.py`
**Descripción**: Script interactivo para inspección detallada de señales fisiológicas de *un* archivo específico.
**Uso**:
```python
# Ejecutar en un entorno interactivo (Jupyter/IPython)
# Modificar las variables al inicio del script:
subject = "18"  # ID del sujeto
session = "vr"  # Sesión
task = "01"     # Tarea
run = "001"     # Run
```
**Funcionalidad**:
- Permite inspección interactiva de señales fisiológicas
- Visualiza en tiempo real:
  - Actividad Electrodérmica (EDA)
  - Electrocardiograma (ECG) con inversión de señal
  - Respiración (RESP)
- Usa ventanas interactivas de matplotlib
- Permite modificar parámetros y reprocesar fácilmente

### `test_check_eda.py`
**Descripción**: Script automatizado para generar visualizaciones de Actividad Electrodérmica (EDA) para todos los archivos de un sujeto.
**Uso**:
```bash
python tests/sanity_check/test_check_eda.py --subject <ID_SUJETO>
```
**Funcionalidad**:
- Procesa todos los archivos FIF de un sujeto
- Genera y guarda plots de EDA usando NeuroKit2
- Guarda los plots en `/eda_plots/` con nombres basados en el archivo FIF
- Incluye:
  - Señal EDA cruda
  - Componentes de EDA (tonic/phasic)
  - Métricas de EDA
- Genera un plot por archivo FIF procesado

### `test_process_subject.py`
**Descripción**: Script automatizado para procesar todos los datos de uno o todos los sujetos en secuencia, tanto para datos reales como de test.
**Uso**:
```bash
# Para datos reales:
python tests/sanity_check/test_process_subject.py [--subject <ID_SUJETO>]

# Para datos de test:
python tests/sanity_check/test_process_subject.py --subject <ID_SUJETO> --test
```
**Funcionalidad**:
- Procesa uno o todos los sujetos disponibles en el directorio de datos correspondiente (`data/` o `tests/test_data/` según el flag `--test`).
- Ejecuta en secuencia:
  1. `test_convert_xdf_to_bids.py` para convertir XDF a FIF (usando el directorio correcto de datos)
  2. `test_check_eda.py` para generar plots de EDA
- Verifica existencia de archivos FIF para evitar reprocesamiento
- Genera reporte final de procesamiento
- Maneja errores sin interrumpir el procesamiento de otros sujetos
- El flag `--test` hace que busque los datos en `tests/test_data/` en vez de `data/` y ajusta el pipeline automáticamente.

### `test_check_markers.py`
**Descripción**: Script interactivo para visualizar y verificar el canal de audio que contiene los markers.
**Uso**:
```python
# Ejecutar en un entorno interactivo (Jupyter/IPython)
# Modificar la lista de sujetos al inicio del script:
subjects = ["16"]  # Lista de IDs de sujetos a verificar
```
**Funcionalidad**:
- Visualiza el canal de audio de archivos FIF específicos
- Permite inspección interactiva de la señal de audio
- Usa ventanas interactivas de matplotlib
- Escala la señal para mejor visualización

## Archivos de Test en `./tests/preprocessing/`

### `test_eeg_preprocessing.py`
**Descripción**: Pipeline completo de preprocesamiento EEG modular y compatible con BIDS.
**Uso**:
```python
# Ejecutar en un entorno interactivo (Jupyter/VSCode)
# Modificar los parámetros al inicio del script:
subject = "12"
session = "vr"
task = "01"
run = "002"
day = "a"
data = "eeg"
```
**Funcionalidad**:
- Pipeline modular y reproducible
- Procesa archivos XDF y los convierte a formato FIF
- Implementa preprocesamiento completo:
  1. Filtrado (bandpass 0.5-48 Hz, notch 50 Hz)
  2. Detección automática de canales ruidosos
  3. Rechazo automático de épocas
  4. ICA para corrección de artefactos
- Guarda resultados en estructura BIDS:
  - `/data/derivatives/` para archivos procesados
  - Genera reportes de preprocesamiento
  - Mantiene logs detallados del procesamiento
- Compatible con herramientas BIDS (mne-bids)

### `conftest.py`
**Descripción**: Archivo de configuración para pytest.
**Uso**: No se ejecuta directamente, es usado por pytest.
**Funcionalidad**: Define fixtures y configuraciones comunes para las pruebas.

## Directorios

- `test_data/`: Contiene datos de prueba para los tests
- `test_outputs/`: Directorio donde se guardan los resultados de los tests

## Notas de Uso

1. Los archivos de salida se guardan en `test_outputs/` siguiendo la estructura BIDS.
