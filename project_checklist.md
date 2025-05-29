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

[x] **Backup manual:** Sube los datos originales a una carpeta de Google Drive (solo como respaldo, no para reproducibilidad).

[x] **Descarga local:** Descarga los datos a tu máquina local en la carpeta `data/`.

[x] **Versiona los datos con DVC:**

```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "data: track data/ directory with DVC"
```

[x] **Configura el remoto DVC (solo la primera vez):**

```bash
dvc remote add -d gdrive gdrive://<ID-carpeta-DVC>
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path gdrive-sa.json

```
- Comparte la carpeta de Drive con el email de la Service Account.

[x] **Sube los datos versionados al remoto DVC:**

```bash
dvc push
```

[x] **Colaboradores:**

Clonan el repo, configuran la Service Account y hacen:

```bash
dvc pull -j 4  # descarga paralela de datos versionados
```

[x] **Verifica que el workspace esté limpio**

```bash
dvc status -c          # debería decir "Pipeline is up to date."
```
"Cache and remote 'gdrive' are in sync."

**Notas:**
- Nunca subas datos manualmente a la carpeta de Drive usada como remoto DVC.
- Solo versiona y sincroniza datos con DVC para garantizar reproducibilidad.
- El backup manual es opcional y no debe usarse como fuente de verdad para análisis reproducibles.
- Asegúrate de que `.gitignore` incluya los datos, pero siempre commitea los archivos `.dvc` y `.gitignore`.
- Si tienes problemas de autenticación, revisa la configuración de la Service Account y los permisos de la carpeta de Drive.

---

## 3 · Quality Gates

[x] **Run fast linters & type checks**

```bash
nox -s lint type_check
```

[x] **Run unit + property tests with coverage**

```bash
nox -s tests           # wraps: pytest --cov --type-check
```

[ ] Coverage ≥ 90 %? If not, open an issue and add missing tests.
*Notes:* 

---

## 4 · Documentation

[x] **Build docs locally**

```bash
nox -s docs            # runs mkdocs build --strict
```

*Notes:* 
[x] Add quick-start snippet & TL;DR to **README.md** (30-second demo).
*Notes:*

---

## 5 · Automation & CI

[ ] **Push branch and open first PR** (tests, lint, docs must pass in CI).
*Notes:*
[ ] Confirm reusable GitHub Actions workflow shows green checks.
*Notes:*

---

## 6 · First Data Pipeline (EEG Preprocessing)

- [ ] Crear un script reproducible de preprocesamiento EEG para un participante:
    1. Leer datos BIDS de un sujeto/sesión/tarea.
    2. Filtrar (bandpass 0.5–45 Hz) y notch.
    3. Detección automática y visual de canales ruidosos.
    4. Segmentar en epochs (-0.3 a 1.2s) según eventos.
    5. Rechazo automático y manual de epochs (AutoReject).
    6. ICA + clasificación automática de componentes (ICLabel).
    7. Interpolación de canales malos y rereferencia.
    8. Guardar epochs preprocesados y reporte HTML en `data/derivatives/`.
    9. Loggear todos los pasos y parámetros en JSON.
- [ ] Crear una **nox session** (`nox -s preproc_demo`) que ejecute el script.
- [ ] Agregar un **DVC stage** en `dvc.yaml` para este pipeline y pushear artefactos.

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
