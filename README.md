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
pip install -e .[“dev”]

# Extract data (if DVC and remote are configured)
dvc pull -j 4
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
- BIDS compatibility (MNE-BIDS)
- Machine learning (scikit-learn)
- Reproducible environments (micromamba, conda-lock)
- Data version control (DVC, remote Google Drive)
- Documentation (MkDocs)

## Dependency management policy

- For rapid prototyping, you can install new dependencies with `pip install` during development.
- Immediately add any new dependencies to `environment.yml` to maintain reproducibility of the environment.
- Regenerate the lock file (`conda-lock.yml`) periodically (e.g., at project milestones, before releases, or after a batch of changes).
- Document all dependency changes in `CHANGELOG.md` and in commit messages to ensure traceability.
- If the dependency is a pure Python development tool or is only needed for development/automation, add it to `[project.optional-dependencies]` in `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE)

## Using DVC for reproducible data

### Recommended workflow
1. **Manual backup:** Upload the original data to a Google Drive folder (for backup only, not for reproducibility).
2. **Local download:** Download the data to your local machine in the `data/` folder.
3. **Versioning with DVC:**
   - Run `dvc add data/` to version the data.
   - Do `git add data.dvc .gitignore` and commit the control files.
4. **Set up the DVC remote:**
   - Use a separate Google Drive folder as the DVC remote.
   - Set up the remote with: `dvc remote add -d gdrive gdrive://<DVC-folder-ID>`
   - Enable the service account:
```bash
     dvc remote modify gdrive gdrive_use_service_account true
     dvc remote modify gdrive gdrive_service_account_json_file_path gdrive-sa.json
     ```
   - Share the Drive folder with the service account email.
5. **Upload the versioned data:**
   - Run `dvc push` to upload the data to the remote DVC.
6. **Collaborators:**
   - Clone the repository, configure the service account, and run `dvc pull` to get the data.

### Best practices and warnings
- **Never manually upload data to the Drive folder used as remote DVC.**
- **Only version and sync data with DVC to ensure reproducibility.**
- **Manual backup is optional and should not be used as the source of truth for reproducible analysis.**
- **Ensure that `.gitignore` includes the data, but always commit the `.dvc` and `.gitignore` files.**
- If you encounter authentication issues, check the service account settings and Drive folder permissions.