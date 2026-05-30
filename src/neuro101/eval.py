"""Leakage-safe splitting and evaluation for neural-signal machine learning.

This module is the **single source of truth** for how data is split and scored
in this tutorial. It deliberately does **not** provide a random-shuffle splitter,
because random shuffling is the #1 cause of fake-high scores on neural data
(adjacent samples are correlated, so a shuffled test set "peeks" at training).

Glossary (terms defined on first use):

* **Trial / epoch** — a short, labelled segment of a recording (e.g. 4 seconds of
  EEG while a person imagines moving their left hand). Samples *within* a trial are
  highly correlated, so a trial must never be split across train and test.
* **Subject** — one person (or animal) who was recorded. Different subjects have
  different brains, electrode placements and skull shapes, so a model that works
  on the subjects it trained on may fail on a new person.
* **LOSO** — Leave-One-Subject-Out cross-validation: hold out *all* trials of one
  subject for testing, train on everyone else, repeat for each subject. This
  measures whether a model generalises to **new people**.
* **Leakage** — when information from the test set sneaks into training (directly,
  or through a preprocessing step fit on all the data). Leakage inflates scores.

The four public functions required by the tutorial's "hard rules":
:func:`make_subject_split`, :func:`make_block_split`,
:func:`leakage_safe_pipeline`, :func:`evaluate_with_variance`.
"""
from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

__all__ = [
    "make_subject_split",
    "make_block_split",
    "leakage_safe_pipeline",
    "evaluate_with_variance",
]


def make_subject_split(
    subjects: np.ndarray,
    *,
    n_splits: int | None = None,
    return_indices: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-One-Subject-Out (LOSO) cross-validation splitter.

    Yields ``(train_idx, test_idx)`` pairs where every test fold contains the
    trials of exactly **one** held-out subject and the train fold contains all
    the others. This is the splitter behind the tutorial's *headline*,
    subject-independent metric: a model is only credited for generalising to
    people it has never seen.

    Parameters
    ----------
    subjects : array of shape (n_trials,)
        Subject id for each trial/epoch. Trials are grouped by this id and never
        split across folds.
    n_splits : int, optional
        If given, only this many subjects are held out (the first ``n_splits``
        unique subject ids, in sorted order). Use it to cap runtime on CPU.
        Default: one fold per unique subject.
    return_indices : bool, default True
        If True, yield integer index arrays. If False, yield boolean masks of
        length ``n_trials``.

    Yields
    ------
    (train, test) : tuple of np.ndarray
        Integer indices (or boolean masks) selecting train and test trials.

    Raises
    ------
    ValueError
        If fewer than two unique subjects are present (LOSO is meaningless).

    Notes
    -----
    Use this — never ``sklearn.model_selection.train_test_split(shuffle=True)`` —
    on grouped neural data. See Chapter 09, pitfall #2.

    Examples
    --------
    >>> import numpy as np
    >>> subj = np.array([0, 0, 1, 1, 2, 2])
    >>> folds = list(make_subject_split(subj))
    >>> len(folds)  # one fold per subject
    3
    >>> tr, te = folds[0]
    >>> sorted(subj[te])  # first fold tests on subject 0 only
    [0, 0]
    """
    subjects = np.asarray(subjects)
    if subjects.ndim != 1:
        raise ValueError("`subjects` must be a 1-D array, one id per trial.")
    unique = np.unique(subjects)
    if unique.size < 2:
        raise ValueError(
            f"LOSO needs >= 2 subjects, found {unique.size}. "
            "With a single subject use make_block_split instead."
        )
    if n_splits is not None:
        unique = unique[:n_splits]

    all_idx = np.arange(subjects.shape[0])
    for s in unique:
        test_mask = subjects == s
        if return_indices:
            yield all_idx[~test_mask], all_idx[test_mask]
        else:
            yield ~test_mask, test_mask


def make_block_split(
    n_samples: int,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Contiguous block / trial-aware split for within-subject time series.

    Splits data into temporally **contiguous** blocks so that adjacent (and
    therefore correlated) samples never straddle the train/test boundary.
    Each fold uses one contiguous block as the test set and the remaining
    samples as the training set. Optionally:

    * inserts a ``gap`` of discarded samples on each side of the test block to
      remove autocorrelation leakage across the boundary, and
    * respects ``groups`` (e.g. trial ids) so a single trial/epoch is never cut
      in half — block boundaries are snapped to group boundaries.

    Parameters
    ----------
    n_samples : int
        Number of samples/epochs, assumed to be in temporal order.
    groups : array of shape (n_samples,), optional
        Trial/block id per sample. If given, the data is split at group
        boundaries only (so each fold holds out whole groups). The array must be
        "contiguous by group": all samples of a group sit next to each other.
    n_splits : int, default 5
        Number of contiguous folds.
    gap : int, default 0
        Number of samples to drop on each side of the test block (a "purge")
        to avoid autocorrelation leak between train and test. Ignored at the
        ends of the recording.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
        Integer indices selecting train and test samples (the purged ``gap``
        samples belong to neither).

    Raises
    ------
    ValueError
        If ``n_splits`` exceeds the number of available blocks/groups.

    Notes
    -----
    Use this for within-subject evaluation. See Chapter 09, pitfall #1.

    Examples
    --------
    >>> folds = list(make_block_split(10, n_splits=2))
    >>> tr, te = folds[0]
    >>> te.tolist()           # first contiguous half is the test block
    [0, 1, 2, 3, 4]
    >>> tr.tolist()
    [5, 6, 7, 8, 9]
    >>> # With a gap=1 purge, the sample adjacent to the test block is dropped:
    >>> tr, te = next(make_block_split(10, n_splits=2, gap=1))
    >>> 5 in tr.tolist()       # sample 5 is purged
    False
    """
    if groups is None:
        # Work on raw sample indices (each sample is its own "unit").
        unit_of_sample = np.arange(n_samples)
        n_units = n_samples
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != n_samples:
            raise ValueError("`groups` must have length n_samples.")
        # Map each sample to a 0-based contiguous "unit" index.
        _, unit_of_sample = np.unique(groups, return_inverse=True)
        # Verify contiguity (each unit appears as one run).
        change = np.flatnonzero(np.diff(unit_of_sample)) + 1
        run_ids = unit_of_sample[np.r_[0, change]]
        if run_ids.size != np.unique(run_ids).size:
            raise ValueError(
                "`groups` is not contiguous: samples of the same group must be "
                "adjacent (data is assumed to be in temporal order)."
            )
        n_units = int(unit_of_sample.max()) + 1

    if n_splits > n_units:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of blocks ({n_units})."
        )

    all_idx = np.arange(n_samples)
    # Partition the *units* into n_splits contiguous chunks (as even as possible).
    bounds = np.linspace(0, n_units, n_splits + 1).astype(int)
    for k in range(n_splits):
        u_lo, u_hi = bounds[k], bounds[k + 1]
        test_unit_mask = (unit_of_sample >= u_lo) & (unit_of_sample < u_hi)
        test_idx = all_idx[test_unit_mask]

        # Purge `gap` samples on each side of the contiguous test block.
        train_mask = ~test_unit_mask
        if gap > 0 and test_idx.size:
            lo, hi = test_idx.min(), test_idx.max()
            purge_lo = max(0, lo - gap)
            purge_hi = min(n_samples, hi + gap + 1)
            train_mask[purge_lo:purge_hi] = False

        yield all_idx[train_mask], test_idx


def leakage_safe_pipeline(
    steps: Sequence[tuple[str, BaseEstimator]],
    *,
    memory: str | None = None,
) -> Pipeline:
    """Wrap preprocessing + estimator so every transform is fit on TRAIN ONLY.

    This is a thin, documented wrapper around :class:`sklearn.pipeline.Pipeline`.
    Its purpose is pedagogical: any scaler / CSP / covariance / feature step
    placed inside the pipeline is **re-fit inside each cross-validation fold on
    the training data alone**, so statistics from the test fold (means, spatial
    filters, class info) never leak into training.

    The common mistake this prevents: calling ``scaler.fit_transform(X)`` or
    ``CSP().fit(X, y)`` on the *whole* dataset before splitting. See Chapter 09,
    pitfall #3 — the honest score is lower but real.

    Parameters
    ----------
    steps : sequence of (name, estimator)
        Standard sklearn ``(name, estimator)`` steps. The final step is the
        classifier; earlier steps are transforms (scaler, CSP, covariance, ...).
    memory : str, optional
        Optional joblib cache directory to memoise fitted transforms across
        identical fits (purely a speed optimisation; does not affect results).

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline that is safe to pass to ``cross_val_score`` / our
        :func:`evaluate_with_variance`.

    Raises
    ------
    ValueError
        If ``steps`` is empty.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = leakage_safe_pipeline([
    ...     ("scale", StandardScaler()),
    ...     ("clf", LogisticRegression()),
    ... ])
    >>> [name for name, _ in pipe.steps]
    ['scale', 'clf']
    """
    if not steps:
        raise ValueError("`steps` must contain at least the final estimator.")
    return Pipeline(list(steps), memory=memory)


def _resolve_scoring(scoring: str | Sequence[str]) -> dict[str, Callable]:
    """Return an ordered {name: scorer_callable} map from sklearn scorer names."""
    names = [scoring] if isinstance(scoring, str) else list(scoring)
    return {name: get_scorer(name) for name in names}


def evaluate_with_variance(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: Iterator[tuple[np.ndarray, np.ndarray]] | Callable[[], Iterator],
    scoring: str | Sequence[str] = ("accuracy", "balanced_accuracy", "f1_macro"),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    return_per_fold: bool = True,
) -> dict:
    """Run cross-validation across multiple seeds and report mean ± std.

    Never report a single number from a single run. This helper repeats the
    cross-validation over several random seeds, re-fitting a fresh clone of the
    estimator on each fold, and returns mean, standard deviation and the raw
    per-fold/per-seed scores so that a paired statistical test across folds is
    possible. This is the antidote to the "lucky seed" pitfall (Chapter 09,
    pitfall #6).

    Parameters
    ----------
    estimator : sklearn Pipeline
        Preferably built with :func:`leakage_safe_pipeline` so transforms are
        fit on train folds only.
    X : np.ndarray of shape (n_trials, ...)
        Features or epochs. The first axis indexes trials.
    y : np.ndarray of shape (n_trials,)
        Integer class labels.
    cv : iterable of (train_idx, test_idx), or a zero-arg callable returning one
        The cross-validation splits. **Pass a callable** (e.g.
        ``lambda: make_subject_split(subjects)``) so the splits can be
        regenerated for each seed; a one-shot iterator is consumed after the
        first seed. A plain iterator is accepted but will be materialised to a
        list and reused (the same splits for every seed).
    scoring : str or sequence of str, default ("accuracy", "balanced_accuracy", "f1_macro")
        sklearn scorer name(s).
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Random seeds. The estimator's ``random_state`` (if it has one) is set to
        each seed before fitting, and numpy's global seed is set too, so the
        spread reflects genuine run-to-run variance.
    return_per_fold : bool, default True
        If True, include the full ``(n_seeds, n_folds)`` score matrix per metric
        under the ``"per_fold"`` key.

    Returns
    -------
    dict
        ``{metric: {"mean": float, "std": float, "per_fold": np.ndarray}, ...}``
        plus top-level keys ``"n_folds"``, ``"n_seeds"`` and ``"seeds"``.
        The ``per_fold`` matrix has shape ``(n_seeds, n_folds)`` — flatten it for
        an overall spread, or take column means for a paired test across folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 5)); y = (X[:, 0] > 0).astype(int)
    >>> subjects = np.repeat(np.arange(4), 10)
    >>> pipe = leakage_safe_pipeline([("s", StandardScaler()),
    ...                               ("clf", LogisticRegression())])
    >>> res = evaluate_with_variance(
    ...     pipe, X, y,
    ...     cv=lambda: make_subject_split(subjects),
    ...     scoring="accuracy", seeds=(0, 1))
    >>> set(res["accuracy"]) == {"mean", "std", "per_fold"}
    True
    >>> res["n_folds"], res["n_seeds"]
    (4, 2)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    scorers = _resolve_scoring(scoring)

    # Materialise the splits so they can be reused across seeds.
    if callable(cv):
        make_splits = cv
    else:
        cached = list(cv)
        make_splits = lambda: iter(cached)  # noqa: E731

    # metric -> list over seeds of list over folds
    raw: dict[str, list[list[float]]] = {name: [] for name in scorers}

    for seed in seeds:
        np.random.seed(seed)
        per_metric_fold: dict[str, list[float]] = {name: [] for name in scorers}
        for train_idx, test_idx in make_splits():
            model = clone(estimator)
            if "random_state" in model.get_params():
                model.set_params(random_state=seed)
            model.fit(X[train_idx], y[train_idx])
            for name, scorer in scorers.items():
                per_metric_fold[name].append(
                    float(scorer(model, X[test_idx], y[test_idx]))
                )
        for name in scorers:
            raw[name].append(per_metric_fold[name])

    out: dict = {}
    n_folds = 0
    for name, seed_folds in raw.items():
        mat = np.asarray(seed_folds, dtype=float)  # (n_seeds, n_folds)
        n_folds = mat.shape[1]
        entry = {"mean": float(mat.mean()), "std": float(mat.std())}
        if return_per_fold:
            entry["per_fold"] = mat
        out[name] = entry

    out["n_folds"] = n_folds
    out["n_seeds"] = len(seeds)
    out["seeds"] = list(seeds)
    return out
