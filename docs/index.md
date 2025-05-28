# CAMPEONES Analysis

Bienvenido a la documentación del Proyecto CAMPEONES, un pipeline reproducible para el análisis de datos de un experimento con datos de VR, incluyendo EEG y medidas fisiológicas periféricas.

## Descripción

Este proyecto proporciona un flujo de trabajo reproducible, modular y automatizado para el preprocesamiento, extracción de características y machine learning de datos de experimentos de emociones, utilizando herramientas de código abierto.

## Características Principales

- **Procesamiento de Datos**:
  - EEG y datos fisiológicos periféricos (MNE, NeuroKit2)
  - Compatibilidad con BIDS (MNE-BIDS)
  - Machine Learning (scikit-learn)

- **Reproducibilidad**:
  - Gestión de entornos (micromamba, conda-lock)
  - Control de versiones de datos (DVC, Google Drive)
  - Gates de calidad automatizados (Nox, pre-commit, ruff, pyright)

- **Documentación**:
  - Guías de uso y ejemplos
  - Documentación de código
  - Diario de investigación

## Contenido

- [Documentación de Tests](tests.md) - Guía completa de los scripts de test y su uso
- [Diario de Investigación](./research_diary) - Registro detallado del proceso de investigación
- [Guía de Instalación](installation.md) - Instrucciones para configurar el entorno de desarrollo # PENDIENTE
- [Guía de Uso](usage.md) - Ejemplos y tutoriales de uso del pipeline # PENDIENTE

## Estructura del Proyecto

```
campeones_analysis/
├── data/               # Datos del proyecto (versionados con DVC)
├── docs/              # Documentación
├── research_diary/    # Diario de investigación
├── scripts/           # Scripts de utilidad
├── src/               # Código fuente del proyecto
├── tests/             # Tests y scripts de validación
└── results/           # Resultados de análisis
```

## Licencia

Este proyecto está licenciado bajo MIT - ver [LICENSE](../LICENSE) para más detalles.
