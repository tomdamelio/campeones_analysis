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
    session.run("pyright", ".", external=True)


@nox.session
def tests(session):
    session.install("pytest")
    session.run("pytest", ".", external=True)
