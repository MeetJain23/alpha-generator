"""What the repository may and may not contain.

Two halves, and the second is the one that gets forgotten.

The first asserts that nothing which is the asset is tracked: no registry, no
pool, no panel, no credential. That failure is loud when it happens, because
someone sees a strategy in a public repository.

The second asserts that everything which is infrastructure *is* tracked. That
failure is silent. An unanchored ``data/`` in .gitignore matches
``alpha/data/`` as well, and when it did, the package sat on disk, the README
referenced it, the imports resolved, every test passed, and git said nothing.
The gap only appears on a fresh clone. It is the same failure shape the rest
of this system is built against: silent, and flattering, so nothing prompts
anyone to look.

Credentials are here too, because where a key may live is the same question as
what may be committed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> list[str]:
    """Run a git command in the repository, or skip if there is no git."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable or not a repository: {exc}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    return git("ls-files")


# --------------------------------------------------------------------------
# nothing that is the asset may be tracked
# --------------------------------------------------------------------------

FORBIDDEN_SUFFIXES = (
    ".parquet",
    ".npz",
    ".csv",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
    ".ipynb",
    ".log",
)

FORBIDDEN_PREFIXES = ("data/", "cache/", "runs/", "scratch/")


def test_no_data_or_registry_file_is_tracked(tracked: list[str]) -> None:
    """The registry and the pool are the asset. See docs/REPRODUCIBILITY.md."""
    offenders = [
        path for path in tracked if path.lower().endswith(FORBIDDEN_SUFFIXES)
    ]
    assert not offenders, f"asset or data files are tracked: {offenders}"


def test_no_output_directory_is_tracked(tracked: list[str]) -> None:
    offenders = [
        path for path in tracked if path.startswith(FORBIDDEN_PREFIXES)
    ]
    assert not offenders, f"output directories are tracked: {offenders}"


def test_no_env_file_is_tracked_except_the_example(tracked: list[str]) -> None:
    """A .env is the single most common way a key reaches a public repo."""
    env_files = [path for path in tracked if Path(path).name.startswith(".env")]
    assert env_files == [".env.example"], f"unexpected env files tracked: {env_files}"


def test_the_env_example_holds_no_values() -> None:
    """Names only. A worked example is how a real key gets committed by
    someone filling in the blank and forgetting which file they were in."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, value = stripped.partition("=")
        assert value == "", f"{name} has a value in .env.example"


# --------------------------------------------------------------------------
# everything that is infrastructure must be tracked
# --------------------------------------------------------------------------


def _on_disk(directory: str) -> set[str]:
    root = REPO_ROOT / directory
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }


@pytest.mark.parametrize("directory", ["alpha", "tests", "scripts"])
def test_every_source_file_on_disk_is_tracked(
    tracked: list[str], directory: str
) -> None:
    """The test that would have caught the original bug.

    An unanchored ignore pattern excludes a whole package without any local
    symptom. Nothing else in the suite notices, because the files are right
    there on disk and every import works.
    """
    present = _on_disk(directory)
    assert present, f"no python files found under {directory}/"
    missing = sorted(present - set(tracked))
    assert not missing, (
        f"{len(missing)} source files exist on disk but are not tracked: "
        f"{missing}. A .gitignore pattern is probably matching at a depth it "
        f"was not meant to."
    )


def test_the_ignore_patterns_for_output_are_anchored() -> None:
    """Anchored to the repo root, so they cannot match a source package.

    Asserted on the file rather than trusted to review, because the
    unanchored form looks more natural and reads as a simplification.
    """
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    for name in ("data", "cache", "runs", "scratch"):
        assert f"/{name}/" in lines, f"/{name}/ must be anchored in .gitignore"
        assert f"{name}/" not in lines, (
            f"unanchored '{name}/' matches alpha/{name}/ and would silently "
            f"untrack a source package"
        )


def test_the_package_is_importable_from_what_is_tracked(tracked: list[str]) -> None:
    """Every module the package exposes has a tracked file behind it."""
    for module in (
        "alpha/__init__.py",
        "alpha/config.py",
        "alpha/data/types.py",
        "alpha/expr/grammar.py",
        "alpha/registry/db.py",
        "alpha/screen/screen.py",
    ):
        assert module in tracked, f"{module} is not tracked"


# --------------------------------------------------------------------------
# credentials never render
# --------------------------------------------------------------------------


def test_a_missing_credential_names_the_variable(monkeypatch) -> None:
    """Not a 401 an hour later pointing at the wrong system."""
    from alpha.config import MissingCredential, require

    monkeypatch.delenv("ALPHA_TEST_KEY", raising=False)
    with pytest.raises(MissingCredential, match="ALPHA_TEST_KEY"):
        require("ALPHA_TEST_KEY")


def test_an_empty_credential_counts_as_missing(monkeypatch) -> None:
    """An empty string is how a blank .env line reaches a request."""
    from alpha.config import MissingCredential, require

    monkeypatch.setenv("ALPHA_TEST_KEY", "")
    with pytest.raises(MissingCredential):
        require("ALPHA_TEST_KEY")


def test_a_credential_has_no_default() -> None:
    """require takes one argument, so a default cannot be passed by habit."""
    import inspect

    from alpha.config import require

    parameters = inspect.signature(require).parameters
    assert list(parameters) == ["name"]
    assert parameters["name"].default is inspect.Parameter.empty


# Deliberately key-shaped, so the redaction tests exercise a realistic value.
# detect-secrets flags this and is right to: the allowlist pragma marks the one
# line, at the site, where a reader can see what it is. Baselining the file
# would hide the next real finding in it.
SECRET_VALUE = "sk-live-do-not-log-this-value"  # pragma: allowlist secret


@pytest.fixture()
def secret(monkeypatch):
    from alpha.config import require

    monkeypatch.setenv("ALPHA_TEST_KEY", SECRET_VALUE)
    return require("ALPHA_TEST_KEY")


@pytest.mark.parametrize(
    "render",
    [repr, str, lambda s: f"{s}", lambda s: "{}".format(s), lambda s: f"{s!s}"],
    ids=["repr", "str", "fstring", "format", "fstring-conversion"],
)
def test_a_secret_never_renders_its_value(secret, render) -> None:
    """Every path a logger or a traceback would take gives the redaction."""
    assert SECRET_VALUE not in render(secret)
    assert "redacted" in render(secret)


def test_a_secret_inside_a_structure_still_redacts(secret) -> None:
    """Structured logging attaches objects to a record, and a record is
    rendered by repr on the way to a log file."""
    assert SECRET_VALUE not in repr({"credentials": secret})
    assert SECRET_VALUE not in repr([secret])


def test_the_credentials_object_redacts(monkeypatch) -> None:
    from alpha.config import load_data_credentials

    monkeypatch.setenv("NDL_API_KEY", SECRET_VALUE)
    assert SECRET_VALUE not in repr(load_data_credentials())


def test_the_value_comes_out_only_by_asking(secret) -> None:
    """One greppable method name, so an audit for where a key is used is a
    search rather than a reading of every format string."""
    assert secret.reveal() == SECRET_VALUE


def test_a_secret_reports_emptiness_without_disclosure(secret) -> None:
    assert bool(secret) is True
    assert len(secret) == len(SECRET_VALUE)


def test_no_credential_is_read_from_a_file() -> None:
    """config.py touches os.environ and nothing else."""
    source = (REPO_ROOT / "alpha" / "config.py").read_text(encoding="utf-8")
    for forbidden in ("open(", "read_text", "Path(", "json.load", "dotenv"):
        assert forbidden not in source, (
            f"config.py references {forbidden!r}; credentials come from the "
            f"environment only"
        )
    assert "os.environ" in source
