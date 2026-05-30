"""Guard against zh-TW ↔ English code drift.

The Traditional-Chinese notebooks (``*/zh-TW/_src``) are translations: their
markdown, code comments and human-facing print text are in Chinese, but the
**executable code must stay identical** to the English source. As the repo grows
it's easy to change logic in English and forget the translation (or vice-versa).

This test parses each EN/zh notebook pair and asserts that:
  1. they have the same number of cells and the same cell-type sequence, and
  2. every *code* cell has the same executable structure — compared as a Python
     token stream with comments dropped and **all string literals blanked**
     (so translated comments / print text / f-string wording are allowed, but a
     changed function call, parameter, number, or variable name is caught).

Translated dict keys / argument literals would instead surface as an execution
failure in the notebook smoke test, so blanking strings here is safe.
"""
import io
import tokenize
from pathlib import Path

import jupytext
import pytest

ROOT = Path(__file__).resolve().parents[1]

_PAIR_DIRS = [
    (ROOT / "notebooks" / "_src", ROOT / "notebooks" / "zh-TW" / "_src"),
    (ROOT / "deep-dives" / "_src", ROOT / "deep-dives" / "zh-TW" / "_src"),
]


def _pairs():
    out = []
    for en_dir, zh_dir in _PAIR_DIRS:
        if not en_dir.exists():
            continue
        for en in sorted(en_dir.glob("*.py")):
            zh = zh_dir / en.name
            if zh.exists():
                out.append(pytest.param(en, zh, id=f"{en_dir.parent.name}/{en.stem}"))
    return out


def _normalize_code(src: str):
    """Token stream of a code cell, comments dropped and strings blanked.

    IPython magics (``!pip``, ``%cd``) aren't valid Python, so we replace those
    lines with ``pass`` before tokenizing (they are identical across EN/zh anyway).
    """
    lines = []
    for ln in src.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(" " * (len(ln) - len(stripped)) + "pass")
        else:
            lines.append(ln)
    code = "\n".join(lines)

    toks = []
    skip = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in skip:
                continue
            if tok.type in (tokenize.STRING, getattr(tokenize, "FSTRING_START", -1)):
                toks.append("<STR>")
            else:
                toks.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        # Unparseable fragment — fall back to a whitespace-collapsed comparison.
        return ("RAW", " ".join(code.split()))
    return tuple(toks)


@pytest.mark.parametrize("en_path, zh_path", _pairs())
def test_zh_matches_en_code(en_path, zh_path):
    en = jupytext.read(en_path)
    zh = jupytext.read(zh_path)

    assert len(en.cells) == len(zh.cells), (
        f"cell count differs: {en_path.name} has {len(en.cells)}, "
        f"zh-TW has {len(zh.cells)} — a cell was added/removed in one language."
    )

    for i, (ce, cz) in enumerate(zip(en.cells, zh.cells)):
        assert ce.cell_type == cz.cell_type, (
            f"cell {i} type differs in {en_path.name}: "
            f"EN={ce.cell_type} zh={cz.cell_type}"
        )
        if ce.cell_type == "code":
            # Skip the import-bootstrap / Colab-setup cells: they are environment-
            # and folder-depth-specific plumbing (the zh-TW copies sit one level
            # deeper), legitimately differ, and contain no teaching logic.
            if "sys.path" in ce.source or "google.colab" in ce.source:
                continue
            ne, nz = _normalize_code(ce.source), _normalize_code(cz.source)
            assert ne == nz, (
                f"CODE DRIFT in {en_path.name} cell {i}: the executable code "
                f"differs between English and zh-TW (comments/strings are ignored). "
                f"Re-sync the translation."
            )


def test_every_english_notebook_has_translation():
    """Every English notebook source must have a zh-TW counterpart (and vice-versa)."""
    missing = []
    for en_dir, zh_dir in _PAIR_DIRS:
        if not en_dir.exists():
            continue
        for en in sorted(en_dir.glob("*.py")):
            if not (zh_dir / en.name).exists():
                missing.append(f"missing zh-TW for {en.relative_to(ROOT)}")
        for zh in sorted(zh_dir.glob("*.py")):
            if not (en_dir / zh.name).exists():
                missing.append(f"orphan zh-TW {zh.relative_to(ROOT)} (no English source)")
    assert not missing, "translation set is incomplete:\n  " + "\n  ".join(missing)
