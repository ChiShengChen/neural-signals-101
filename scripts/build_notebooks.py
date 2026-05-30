#!/usr/bin/env python
"""Build notebooks/*.ipynb from the percent-format sources in notebooks/_src/.

We keep the *authoritative* notebook source as plain ``.py`` files in
``notebooks/_src`` (percent format: ``# %%`` code cells, ``# %% [markdown]``
text cells). They are diff-friendly, easy to review, and the committed
``.ipynb`` are generated from them with no execution outputs. This is the
jupytext workflow recommended in CONTRIBUTING.md.

Run: ``python scripts/build_notebooks.py`` (or ``make notebooks``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks" / "_src"
OUT_DIR = ROOT / "notebooks"


def build_one(py_path: Path) -> Path:
    nb = jupytext.read(py_path)
    # Ensure a clean, output-free notebook with a stable kernel spec.
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (neuro101)",
        "language": "python",
        "name": "python3",
    }
    out_path = OUT_DIR / (py_path.stem + ".ipynb")
    jupytext.write(nb, out_path, fmt="notebook")
    return out_path


def main() -> int:
    sources = sorted(SRC_DIR.glob("*.py"))
    if not sources:
        print(f"No sources found in {SRC_DIR}", file=sys.stderr)
        return 1
    for py in sources:
        out = build_one(py)
        print(f"  built {out.relative_to(ROOT)}")
    print(f"✅ built {len(sources)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
