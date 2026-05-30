#!/usr/bin/env python
"""Execute every notebook top-to-bottom and report pass/fail.

Used by ``make run-all`` and shared with the CI smoke test. Honours
``NEURO101_SMOKE=1`` (set by CI) so notebooks run on tiny data slices in
seconds. Executed notebooks are written to ``notebooks/_executed/`` so the
committed sources stay output-free.

Exit code is non-zero if any notebook raises.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = NB_DIR / "_executed"


def run_one(path: Path, timeout: int) -> tuple[bool, float, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, time.time() - t0, str(exc).splitlines()[-1][:200]
    finally:
        OUT_DIR.mkdir(exist_ok=True)
        nbformat.write(nb, OUT_DIR / path.name)
    return True, time.time() - t0, ""


def main(argv: list[str]) -> int:
    timeout = int(os.environ.get("NEURO101_NB_TIMEOUT", "600"))
    selected = argv[1:] if len(argv) > 1 else None
    notebooks = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if selected:
        notebooks = [n for n in notebooks if any(s in n.name for s in selected)]
    if not notebooks:
        print("No notebooks found — run `make notebooks` first.", file=sys.stderr)
        return 1

    smoke = os.environ.get("NEURO101_SMOKE", "0")
    print(f"Executing {len(notebooks)} notebooks (NEURO101_SMOKE={smoke}, "
          f"timeout={timeout}s each)\n")
    failures = []
    for nb_path in notebooks:
        ok, dt, err = run_one(nb_path, timeout)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {nb_path.name:45s} {dt:6.1f}s  {err}")
        if not ok:
            failures.append(nb_path.name)

    print()
    if failures:
        print(f"❌ {len(failures)} notebook(s) failed: {', '.join(failures)}")
        return 1
    print(f"✅ all {len(notebooks)} notebooks executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
