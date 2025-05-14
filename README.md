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
