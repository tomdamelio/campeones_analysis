# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Initial project scaffold and reproducible environment setup.
- Integración de DVC con Google Drive usando Service Account:
  - Documentado el flujo recomendado: backup manual en Drive, descarga local, versionado con DVC, y push/pull a un remoto DVC separado.
  - Añadidas buenas prácticas: nunca subir datos manualmente al remoto DVC, siempre versionar y sincronizar con DVC para reproducibilidad.
  - Instrucciones paso a paso para configurar Service Account y permisos en Drive.
