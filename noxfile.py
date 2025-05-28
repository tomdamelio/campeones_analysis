import nox

# Configuración global para todas las sesiones
nox.options.sessions = ["lint"]  # Solo ejecutar lint por defecto
nox.options.reuse_existing_virtualenvs = True  # Reutilizar entornos virtuales


@nox.session
def format(session):
    """Formatea el código usando ruff."""
    session.install("ruff")
    session.run("ruff", "format", ".", external=True)


@nox.session
def lint(session):
    """Ejecuta el linter."""
    session.install("ruff")
    session.run("ruff", "check", ".", external=True)


@nox.session
def type_check(session):
    """Verifica los tipos."""
    session.install("pyright")
    session.run("pyright", "src", "tests", external=True)


@nox.session
def tests(session):
    """Ejecuta los tests unitarios."""
    # Instalar solo las dependencias necesarias para tests unitarios
    session.install("pytest", "pytest-cov")
    session.install("-e", ".", "--no-deps")  # Instalar el proyecto sin dependencias
    # Ejecutar solo los tests en el directorio tests/ y no fallar si no hay tests
    session.run(
        "pytest",
        "tests/",
        "--no-header",
        "--no-summary",
        external=True,
        success_codes=[0, 5],
    )


@nox.session
def docs(session):
    """Construye la documentación."""
    session.install("mkdocs-material")
    session.run("mkdocs", "build", "--strict", external=True)


# Comando para ejecutar todas las sesiones en paralelo
@nox.session
def all(session):
    """Ejecuta todas las sesiones en paralelo."""
    session.notify("format")
    session.notify("lint")
    session.notify("type_check")
    session.notify("tests")
    session.notify("docs")
