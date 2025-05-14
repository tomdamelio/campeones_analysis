import nox


@nox.session
def format(session):
    session.install("ruff")
    session.run("ruff", "format", ".", external=True)


@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", ".", external=True)


@nox.session
def type_check(session):
    session.install("pyright")
    session.run("pyright", "src", "tests", external=True)


@nox.session
def tests(session):
    session.install(".[dev]")  # Install project with dev dependencies
    session.run("pytest", ".", external=True)


@nox.session
def docs(session):
    session.install("mkdocs-material")
    session.run("mkdocs", "build", "--strict", external=True)
