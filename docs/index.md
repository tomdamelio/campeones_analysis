# CAMPEONES Analysis

Bienvenido a la documentación del Proyecto CAMPEONES, un pipeline para el análisis de datos de un experimento con datos de VR, incluyendo EEG y medidas fisiológicas periféricas.

## Descripción

Este proyecto proporciona un flujo de trabajo modular y simplificado para el preprocesamiento, extracción de características y machine learning de datos de experimentos de emociones, utilizando herramientas de código abierto.

## Características Principales

- **Procesamiento de Datos**:
  - EEG y datos fisiológicos periféricos (MNE, NeuroKit2)
  - Compatibilidad con BIDS (MNE-BIDS)
  - Machine Learning (scikit-learn)

- **Reproducibilidad**:
  - Gestión de entornos (micromamba)
  - Control de versiones de datos (DVC, Google Drive)

- **Documentación**:
  - Guías de uso y ejemplos
  - Documentación de código
  - Diario de investigación

## Filosofía de Desarrollo

- **Iteraciones Rápidas**: Enfoque en ciclos cortos de desarrollo para facilitar la experimentación.
- **Simplicidad**: Estructura minimalista que prioriza la funcionalidad sobre la formalidad.
- **Validación Manual**: Comprobación directa de resultados durante el desarrollo.

## Estructura del Proyecto

```
campeones_analysis/
├── data/                # Datos del proyecto (versionados con DVC)
│   ├── raw/             # Datos en formato BIDS
│   ├── sourcedata/      # Datos originales
│   └── derivatives/     # Datos procesados
├── docs/                # Documentación
├── scripts/             # Scripts de utilidad
├── src/                 # Código fuente del proyecto
│   └── campeones_analysis/
│       ├── eeg/         # Análisis de EEG
│       ├── physio/      # Procesamiento de señales fisiológicas
│       ├── behav/       # Análisis comportamental
│       ├── fusion/      # Integración multimodal
│       ├── models/      # Modelos y análisis estadístico
│       └── utils/       # Utilidades comunes
└── results/             # Resultados de análisis
```

## Licencia

Este proyecto está licenciado bajo MIT - ver [LICENSE](../LICENSE) para más detalles.
