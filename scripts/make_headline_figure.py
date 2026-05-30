#!/usr/bin/env python
"""Generate the README headline figure: inflated (wrong) vs honest (right) score.

The contrast comes from the tutorial's most important lesson (Chapter 09,
pitfall #2): pooling many subjects and then splitting *randomly* lets the model
peek at trials from the same person in both train and test, inflating the score.
The honest number comes from Leave-One-Subject-Out (LOSO): test only on people
the model has never seen.

Both pipelines are identical (CSP -> LDA, fit inside the pipeline so there is no
*feature* leakage) — the ONLY difference is how the data is split. That is the
whole point: the method, not the model, creates the fake gain.

Usage:
    python scripts/make_headline_figure.py [--subjects N] [--out PATH]

Downloads BCI IV 2a on first run (~0.2 GB/subject, cached). Saves PNG to
docs/headline.png and prints both scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from neuro101 import (
    io,  # noqa: E402
    viz,  # noqa: E402
)
from neuro101.eval import (  # noqa: E402
    evaluate_with_variance,
    leakage_safe_pipeline,
    make_subject_split,
)
from neuro101.features import make_csp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline():
    """CSP (4 components) -> LDA, fit on train only inside the pipeline."""
    return leakage_safe_pipeline(
        [("csp", make_csp(n_components=4)), ("lda", LinearDiscriminantAnalysis())]
    )


def wrong_pooled_random(X, y, n_splits=5, seed=0):
    """WRONG: pool all subjects, split RANDOMLY (subject identity leaks)."""
    pipe = build_pipeline()
    res = evaluate_with_variance(
        pipe, X, y,
        cv=lambda: StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(X, y),
        scoring="accuracy", seeds=(0, 1, 2),
    )
    return res["accuracy"]["mean"], res["accuracy"]["std"]


def right_loso(X, y, subjects):
    """RIGHT: Leave-One-Subject-Out (test on unseen people)."""
    pipe = build_pipeline()
    res = evaluate_with_variance(
        pipe, X, y,
        cv=lambda: make_subject_split(subjects),
        scoring="accuracy", seeds=(0, 1, 2),
    )
    return res["accuracy"]["mean"], res["accuracy"]["std"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=4,
                        help="number of BCI IV 2a subjects to use")
    parser.add_argument("--out", type=str, default=str(ROOT / "docs" / "headline.png"))
    args = parser.parse_args()

    print(f"Loading BCI IV 2a ({args.subjects} subjects, left vs right hand)...")
    X, y, subjects = io.load_bnci_2a_epochs(
        subjects=list(range(1, args.subjects + 1))
    )
    print(f"  X={X.shape}, classes={np.bincount(y)}, subjects={np.unique(subjects)}")

    print("Scoring WRONG (pooled + random shuffle)...")
    w_mean, w_std = wrong_pooled_random(X, y)
    print(f"  inflated accuracy = {w_mean:.3f} ± {w_std:.3f}")

    print("Scoring RIGHT (Leave-One-Subject-Out)...")
    r_mean, r_std = right_loso(X, y, subjects)
    print(f"  honest accuracy   = {r_mean:.3f} ± {r_std:.3f}")

    chance = 1.0 / len(np.unique(y))
    fig, ax = plt.subplots(figsize=(6, 5))
    viz.plot_wrong_vs_right(
        w_mean, r_mean, wrong_err=w_std, right_err=r_std, chance=chance,
        metric="Accuracy (left vs right hand)",
        wrong_label="WRONG\npool + random split",
        right_label="RIGHT\nleave-one-subject-out",
        title="Same model, same data — only the evaluation differs",
        ax=ax,
    )
    fig.text(0.5, 0.005,
             "BCI IV 2a · CSP→LDA · the red bar is a mirage: it tests on people "
             "the model already trained on.",
             ha="center", fontsize=7.5, color="gray", wrap=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n✅ saved {out}")
    print(f"   inflated {w_mean:.2f}  ->  honest {r_mean:.2f}  "
          f"(fake gain {w_mean - r_mean:+.2f})")

    if w_mean <= r_mean:
        print("⚠️  WARNING: contrast did not hold (wrong <= right). "
              "Increase --subjects or revisit setup.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
