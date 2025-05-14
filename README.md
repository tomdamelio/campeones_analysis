# campeones_analysis

> TL;DR: Reproducible pipeline for immersive emotion experiment data (EEG + peripheral measures) with  Python.

## Description

`campeones_analysis` is a Python project for analyzing data from experiment on emotions in immersive contexts, including EEG and peripheral physiological measures. It provides a reproducible, modular, and automated workflow for preprocessing, feature extraction, and machine learning, using state-of-the-art open-source tools.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/campeones_analysis.git
cd campeones_analysis

# Create and activate the environment
micromamba create -n campeones -f environment.yml
micromamba activate campeones

# Install dev tools (optional, for development)
pip install -e .["dev"]

# Pull data (if DVC and remote configured)
dvc pull -j 4
```

## 30-second Usage Demo

```python
import mne
from campeones_analysis import *

# Example: Load and preprocess EEG data
raw = mne.io.read_raw_fif("data/sub-01/eeg/raw.fif")
# ...your analysis here...
```

## Features

- EEG and peripheral data preprocessing (MNE, NeuroKit2)
- BIDS compatibility (MNE-BIDS)
- Machine learning (scikit-learn)
- Reproducible environments (micromamba, conda-lock)
- Automated quality gates (Nox, pre-commit, ruff, pyright)
- Data versioning (DVC, Google Drive remote)
- Documentation (MkDocs)

## Dependency Management Policy

- For fast prototyping, you may install new dependencies with `pip install` during development.
- Immediately add any new dependency to `environment.yml` to keep the environment reproducible.
- Regenerate the lockfile (`conda-lock.yml`) periodically (e.g., at project milestones, before releases, or after a batch of changes).
- Document all dependency changes in `CHANGELOG.md` and in commit messages for traceability.
- If the dependency is a pure Python dev tool or only needed for development/automation, add it to `[project.optional-dependencies]` in `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE)

## Uso de DVC para datos reproducibles

### Flujo recomendado
1. **Backup manual:** Sube los datos originales a una carpeta de Google Drive (solo como respaldo, no para reproducibilidad).
2. **Descarga local:** Descarga los datos a tu máquina local en la carpeta `data/`.
3. **Versionado con DVC:**
   - Ejecuta `dvc add data/` para versionar los datos.
   - Haz `git add data.dvc .gitignore` y commitea los archivos de control.
4. **Configura el remoto DVC:**
   - Usa una carpeta separada de Google Drive como remoto DVC.
   - Configura el remoto con: `dvc remote add -d gdrive gdrive://<ID-carpeta-DVC>`
   - Habilita Service Account:
     ```bash
     dvc remote modify gdrive gdrive_use_service_account true
     dvc remote modify gdrive gdrive_service_account_json_file_path gdrive-sa.json
     ```
   - Comparte la carpeta de Drive con el email de la Service Account.
5. **Sube los datos versionados:**
   - Ejecuta `dvc push` para subir los datos al remoto DVC.
6. **Colaboradores:**
   - Clonan el repo, configuran la Service Account y hacen `dvc pull` para obtener los datos.

### Buenas prácticas y advertencias
- **Nunca subas datos manualmente a la carpeta de Drive usada como remoto DVC.**
- **Solo versiona y sincroniza datos con DVC para garantizar reproducibilidad.**
- **El backup manual es opcional y no debe usarse como fuente de verdad para análisis reproducibles.**
- **Asegúrate de que `.gitignore` incluya los datos, pero siempre commitea los archivos `.dvc` y `.gitignore`.
- Si tienes problemas de autenticación, revisa la configuración de la Service Account y los permisos de la carpeta de Drive.
