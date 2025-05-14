# 🛠️ Project Checklist — **`campeones_analysis`**

> Follow the tasks **in order**.
> Tick each box once the item is complete or leave a short note if something needs attention.

---

## 1 Environment & Tooling

[x] **Install micromamba** (skip if already in PATH)

*Notes:*

- Instalación realizada en Windows 10 usando PowerShell:
  - Se ejecutó: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - Luego: `iwr -UseBasicParsing https://micro.mamba.pm/install.ps1 | iex`
  - Micromamba quedó accesible en el PATH y verificado con `micromamba --version`

[x] **Create the local env from lock-file**

```bash
micromamba create -n campeones -f environment.yml
micromamba activate campeones
micromamba install -c conda-forge conda-lock
conda-lock lock --mamba
```

*Notes:*

- El entorno se creó desde environment.yml usando micromamba.
- Se instaló conda-lock en el entorno y se generó conda-lock.yml para máxima reproducibilidad.
- Si el entorno no aparece tras la creación, verifica que micromamba esté correctamente instalado y en el PATH.


[x] **Install extra dev tools via `pyproject.toml`**

```bash
pip install -e .["dev"]            # ruff, pyright, nox, pre-commit, commitlint…
```

*Notes:*
- Se creó el archivo pyproject.toml con el grupo de dependencias [project.optional-dependencies]["dev"] para herramientas de desarrollo.
- Se instalaron herramientas de calidad, testing, formateo, type checking y automatización (ruff, pyright, pytest, nox, pre-commit, mkdocs, vulture, commitlint).
- Esto sigue las reglas del proyecto para mantener el entorno de producción limpio y reproducible, separando dependencias de desarrollo y facilitando la colaboración y la calidad del código.

[x] **Run Nox bootstrap** (installs virtual envs used by sessions)

```bash
nox -s format lint type_check --install-only
```

*Notes:*
- Nox automatiza y aísla tareas de calidad de código (formateo, lint, type checking, tests) en entornos virtuales temporales.
- Esto asegura que los chequeos se ejecuten siempre igual, en cualquier máquina, y facilita el flujo de trabajo profesional y reproducible.

[x] **Enable pre-commit hooks**

```bash
pre-commit install
```

*Notes:*
- Se creó el archivo .pre-commit-config.yaml con hooks para ruff (lint y formato), pyright (type checking) y chequeos básicos de archivos (tamaño, conflictos, YAML, espacios, etc.).
- Los hooks de pre-commit se ejecutan automáticamente antes de cada commit para asegurar la calidad y consistencia del código, previniendo errores comunes y automatizando buenas prácticas.

---

## 2 · Data & DVC (Google Drive remote)

[ ] Configure Drive credentials (*first time only; paste link or gdrive id*)

```bash
dvc remote modify gdrive gdrive_use_service_account true   # if SA json
```

*Notes:*
[ ] **Pull BIDS dataset & derivatives**

```bash
dvc pull -j 4  # parallel download
```

*Notes:*
[ ] Verify workspace is clean

```bash
dvc status -c          # should output "Pipeline is up to date."
```

*Notes:*

---

## 3 · Quality Gates

[ ] **Run fast linters & type checks**

```bash
nox -s lint type_check
```

*Notes:* 
[ ] **Run unit + property tests with coverage**

```bash
nox -s tests           # wraps: pytest --cov --type-check
```

*Notes:* 
[ ] Coverage ≥ 90 %? If not, open an issue and add missing tests.
*Notes:* 

---

## 4 · Documentation

[ ] **Build docs locally**

```bash
nox -s docs            # runs mkdocs build --strict
```

*Notes:* 
[ ] Add quick-start snippet & TL;DR to **README.md** (30-second demo).
*Notes:*

---

## 5 · Automation & CI

[ ] **Push branch and open first PR** (tests, lint, docs must pass in CI).
*Notes:*
[ ] Confirm reusable GitHub Actions workflow shows green checks.
*Notes:*

---

## 6 · First Data Pipeline (example)

* [ ] Create a minimal **nox session** `preproc_demo` that:

1. Loads `sub-01` raw,
2. Runs EEG band-pass + ICA from `configs/preprocessing_eeg.yaml`,
3. Saves the cleaned Raw to `data/derivatives/clean/`.
*Notes:*
[ ] Add a **DVC stage** in `dvc.yaml` for that session and push artefacts.
*Notes:*

---

## 7 · Commit & Tag

[ ] Use **Cursor: AI Commit** → message follows Conventional Commits.
*Notes:*
* [ ] Tag the repo `v0.1.0-alpha` once the above boxes are green.

```bash
git tag -a v0.1.0-alpha -m "initial reproducible scaffold"
git push --tags
```

*Notes:*
