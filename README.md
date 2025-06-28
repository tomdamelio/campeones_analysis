# campeones_analysis

> TL;DR: Reproducible pipeline for data from experiments on immersive emotions (EEG + peripheral measurements) with Python.

## Description

`campeones_analysis` is a Python project for analyzing data from experiments on emotions in immersive contexts, including EEG and peripheral physiological measurements. It provides a reproducible, modular, and automated workflow for preprocessing, feature extraction, and machine learning, using state-of-the-art open-source tools.

## Quick start

```bash
# Clone the repository
git clone https://github.com/tomdamelio/campeones_analysis.git
cd campeones_analysis

# Create and activate the environment
micromamba create -n campeones -f environment.yml
micromamba activate campeones

# Install development tools (optional, for development)
pip install -e .[dev]

# Note: Data files should be obtained separately and placed in the data/ directory
# Contact the project maintainers for data access instructions
```

## XDF data processing

To generate BIDS files from the original data in XDF format:

```bash
# Process all available XDF files
python -m src.campeones_analysis.physio.read_xdf

# Process a specific subject
python -m src.campeones_analysis.physio.read_xdf --subject 01

# Process a specific session
python -m src.campeones_analysis.physio.read_xdf --subject 01 --session VR

# Test mode (process only the first XDF file found)
python -m src.campeones_analysis.physio.read_xdf --test

# Enable detailed logging
python -m src.campeones_analysis.physio.read_xdf --debug

# Continue processing other files if one fails
python -m src.campeones_analysis.physio.read_xdf --continue-on-error
```

The original files must be placed in the structure:
`data/sourcedata/xdf/sub-{subject}/ses-{session}/physio/`

The processed files will be saved in BIDS format in:
`data/raw/`

## Features

- EEG and peripheral data preprocessing (MNE, NeuroKit2)
- BIDS compatibility (MNE-BIDS) with standardized event ID mapping
- Machine learning (scikit-learn)
- Reproducible environments (micromamba, conda-lock)
- Documentation (MkDocs)

## Documentation

Key documentation files:
- [`docs/event_id_mapping_implementation.md`](docs/event_id_mapping_implementation.md) - Event ID mapping for BIDS compliance
- [`docs/scripts_preprocessing.md`](docs/scripts_preprocessing.md) - Preprocessing scripts documentation

## Dependency management policy

- For rapid prototyping, you can install new dependencies with `pip install` during development.
- Immediately add any new dependencies to `environment.yml` to maintain reproducibility of the environment.
- Regenerate the lock file (`conda-lock.yml`) periodically (e.g., at project milestones, before releases, or after a batch of changes).
- Document all dependency changes in `CHANGELOG.md` and in commit messages to ensure traceability.
- If the dependency is a pure Python development tool or is only needed for development/automation, add it to `[project.optional-dependencies]` in `pyproject.toml`.

## Data management

### Getting the data

Data files are stored externally and should be obtained separately from the project maintainers. Once obtained, place them in the `data/` directory following the BIDS structure.

### Data structure

The project expects data to be organized in BIDS format:
- Raw data: `data/raw/`
- Processed data: `data/derivatives/`
- Source data: `data/sourcedata/`

## License

MIT — see [LICENSE](LICENSE)
