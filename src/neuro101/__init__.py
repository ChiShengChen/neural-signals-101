"""neuro101 — shared, tested helpers for the *Neural Signals 101* tutorial.

Import the pieces you need, e.g.::

    from neuro101 import eval, preprocessing, features, viz, io, datasets
    from neuro101.eval import make_subject_split, leakage_safe_pipeline

The package is intentionally small and readable: every function is meant to be
opened and understood by a learner, not treated as a black box.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Re-export the safety-critical evaluation API at the top level so notebooks can
# do `from neuro101 import make_subject_split` without remembering submodules.
from .eval import (  # noqa: F401
    evaluate_with_variance,
    leakage_safe_pipeline,
    make_block_split,
    make_subject_split,
)

__all__ = [
    "__version__",
    "make_subject_split",
    "make_block_split",
    "leakage_safe_pipeline",
    "evaluate_with_variance",
]
