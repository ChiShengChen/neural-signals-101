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
# (source dir, output dir) pairs. The main spine lives in notebooks/; the optional
# advanced side-quests live in deep-dives/.
BUILD_DIRS = [
    (ROOT / "notebooks" / "_src", ROOT / "notebooks"),
    (ROOT / "deep-dives" / "_src", ROOT / "deep-dives"),
    # Traditional-Chinese mirrors (same code, translated prose/comments).
    (ROOT / "notebooks" / "zh-TW" / "_src", ROOT / "notebooks" / "zh-TW"),
    (ROOT / "deep-dives" / "zh-TW" / "_src", ROOT / "deep-dives" / "zh-TW"),
]


def build_one(py_path: Path, out_dir: Path) -> Path:
    nb = jupytext.read(py_path)
    # Make the committed .ipynb indistinguishable from a Jupyter-saved notebook so
    # GitHub's (picky) renderer is happy: drop the jupytext-paired marker and write
    # COMPLETE, standard metadata. jupytext omits `language_info` when generating
    # from a .py, and GitHub errors ("An error occurred...") on notebooks without it.
    nb.metadata.pop("jupytext", None)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.11",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }
    out_path = out_dir / (py_path.stem + ".ipynb")
    jupytext.write(nb, out_path, fmt="notebook")
    return out_path


def main() -> int:
    total = 0
    for src_dir, out_dir in BUILD_DIRS:
        sources = sorted(src_dir.glob("*.py"))
        if not sources:
            continue
        for py in sources:
            out = build_one(py, out_dir)
            print(f"  built {out.relative_to(ROOT)}")
        total += len(sources)
    if total == 0:
        print("No notebook sources found.", file=sys.stderr)
        return 1
    print(f"✅ built {total} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
