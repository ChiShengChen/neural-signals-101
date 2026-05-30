"""Smoke test: every chapter notebook executes top-to-bottom without error.

Runs in *smoke mode* (``NEURO101_SMOKE=1``) so loaders pull the smallest data
slice and each notebook finishes quickly on CPU. Marked ``slow``/``network``
because the first run downloads public datasets; deselect with
``pytest -m 'not network'`` for a no-download run.

The notebooks are (re)built from their ``notebooks/_src/*.py`` sources first, so
this also verifies the jupytext build step.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks" / "_src"
NB_DIR = ROOT / "notebooks"


def _notebook_sources():
    return sorted(SRC_DIR.glob("[0-9][0-9]_*.py"))


@pytest.fixture(scope="session", autouse=True)
def _build_notebooks():
    """Build .ipynb from sources once before the smoke tests run."""
    if _notebook_sources():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_notebooks.py")],
            check=True, cwd=ROOT,
        )


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize(
    "nb_name",
    [p.stem for p in _notebook_sources()],
    ids=[p.stem for p in _notebook_sources()],
)
def test_notebook_executes(nb_name):
    import nbformat
    from nbclient import NotebookClient

    nb_path = NB_DIR / f"{nb_name}.ipynb"
    assert nb_path.exists(), f"{nb_path} not built"

    env = dict(os.environ, NEURO101_SMOKE="1")
    os.environ.update(env)

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=int(os.environ.get("NEURO101_NB_TIMEOUT", "600")),
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    client.execute()  # raises CellExecutionError if any cell fails
