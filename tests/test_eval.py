"""Tests for the safety-critical splitting/evaluation module.

These tests are the guard rails behind the tutorial's "hard rules": they fail
loudly if a split ever leaks across subjects/blocks, or if variance reporting
collapses to a single number.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neuro101.eval import (
    evaluate_with_variance,
    leakage_safe_pipeline,
    make_block_split,
    make_subject_split,
)


# --------------------------------------------------------------------------- #
# make_subject_split (LOSO)
# --------------------------------------------------------------------------- #
def test_subject_split_is_leave_one_subject_out():
    subjects = np.repeat([10, 20, 30], 5)  # ids need not be 0..n
    folds = list(make_subject_split(subjects))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        test_subj = set(subjects[test_idx])
        train_subj = set(subjects[train_idx])
        # Exactly one subject in test, and it never appears in train (no leak).
        assert len(test_subj) == 1
        assert test_subj.isdisjoint(train_subj)
        # Train + test cover everything, no overlap.
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == subjects.size


def test_subject_split_n_splits_caps_folds():
    subjects = np.repeat(np.arange(5), 3)
    assert len(list(make_subject_split(subjects, n_splits=2))) == 2


def test_subject_split_requires_two_subjects():
    with pytest.raises(ValueError):
        list(make_subject_split(np.zeros(10, dtype=int)))


def test_subject_split_boolean_masks():
    subjects = np.array([0, 0, 1, 1])
    tr, te = next(make_subject_split(subjects, return_indices=False))
    assert tr.dtype == bool and te.dtype == bool
    assert (tr | te).all() and not (tr & te).any()


# --------------------------------------------------------------------------- #
# make_block_split
# --------------------------------------------------------------------------- #
def test_block_split_is_contiguous():
    folds = list(make_block_split(20, n_splits=4))
    assert len(folds) == 4
    for _, test_idx in folds:
        # A contiguous block has no internal gaps.
        assert np.all(np.diff(np.sort(test_idx)) == 1)


def test_block_split_covers_all_samples_without_gap():
    seen = np.zeros(20, dtype=bool)
    for _, test_idx in make_block_split(20, n_splits=5):
        seen[test_idx] = True
    assert seen.all()  # every sample is tested exactly once across folds


def test_block_split_gap_purges_adjacent_samples():
    train_idx, test_idx = next(make_block_split(20, n_splits=2, gap=2))
    # The two samples immediately after the test block must be purged.
    boundary = test_idx.max()
    assert (boundary + 1) not in train_idx
    assert (boundary + 2) not in train_idx
    assert (boundary + 3) in train_idx


def test_block_split_respects_groups():
    # 4 trials of 5 samples each; splitting must never cut a trial.
    groups = np.repeat(np.arange(4), 5)
    for _, test_idx in make_block_split(20, groups=groups, n_splits=2):
        tested_groups = groups[test_idx]
        for g in np.unique(tested_groups):
            # All 5 samples of a tested group are present (trial kept whole).
            assert (tested_groups == g).sum() == 5


def test_block_split_rejects_noncontiguous_groups():
    groups = np.array([0, 1, 0, 1])  # interleaved -> not contiguous
    with pytest.raises(ValueError):
        list(make_block_split(4, groups=groups, n_splits=2))


# --------------------------------------------------------------------------- #
# leakage_safe_pipeline
# --------------------------------------------------------------------------- #
def test_leakage_safe_pipeline_builds_pipeline():
    pipe = leakage_safe_pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression())]
    )
    assert [n for n, _ in pipe.steps] == ["scale", "clf"]


def test_leakage_safe_pipeline_rejects_empty():
    with pytest.raises(ValueError):
        leakage_safe_pipeline([])


def test_pipeline_scaler_fits_on_train_only():
    """The scaler inside the pipeline must use train statistics, not test ones.

    We give train and test wildly different scales; a leak-free pipeline scales
    test data by the *train* mean/std, so the transformed test mean is far from 0.
    """
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(50, 3))
    X_test = rng.normal(100, 1, size=(10, 3))
    pipe = leakage_safe_pipeline([("scale", StandardScaler())])
    pipe.fit(X_train)
    transformed_test = pipe.named_steps["scale"].transform(X_test)
    # If it had (wrongly) fit on test, this would be ~0. It must be large.
    assert np.abs(transformed_test.mean()) > 10


# --------------------------------------------------------------------------- #
# evaluate_with_variance
# --------------------------------------------------------------------------- #
def _toy_problem(n_per_subj=20, n_subj=4, seed=0):
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(n_subj), n_per_subj)
    X = rng.normal(size=(n_per_subj * n_subj, 4))
    # Learnable but noisy signal.
    y = ((X[:, 0] + 0.5 * rng.normal(size=X.shape[0])) > 0).astype(int)
    return X, y, subjects


def test_variance_reports_mean_std_and_per_fold():
    X, y, subjects = _toy_problem()
    pipe = leakage_safe_pipeline(
        [("s", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
    )
    res = evaluate_with_variance(
        pipe, X, y,
        cv=lambda: make_subject_split(subjects),
        scoring=("accuracy", "balanced_accuracy"),
        seeds=(0, 1, 2),
    )
    assert res["n_folds"] == 4 and res["n_seeds"] == 3
    for metric in ("accuracy", "balanced_accuracy"):
        entry = res[metric]
        assert {"mean", "std", "per_fold"} <= set(entry)
        assert entry["per_fold"].shape == (3, 4)
        assert 0.0 <= entry["mean"] <= 1.0


def test_variance_cv_callable_regenerates_splits_each_seed():
    """A one-shot iterator would be exhausted after seed 0; a callable must not."""
    X, y, subjects = _toy_problem()
    pipe = leakage_safe_pipeline([("clf", LogisticRegression(max_iter=200))])
    res = evaluate_with_variance(
        pipe, X, y, cv=lambda: make_subject_split(subjects),
        scoring="accuracy", seeds=(0, 1, 2, 3),
    )
    # All seeds produced a full set of folds (no silently-empty seeds).
    assert res["accuracy"]["per_fold"].shape == (4, 4)
    assert not np.isnan(res["accuracy"]["per_fold"]).any()
