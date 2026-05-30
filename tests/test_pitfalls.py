"""Soul-guard tests: the WRONG→RIGHT contrasts of Chapter 12 must keep holding.

These pin the *teaching claims* of the most important chapter. If a refactor ever
makes a leaky method stop looking inflated (or an honest method stop looking
honest), CI fails here — protecting the point of the whole tutorial.

The synthetic tests run in the fast lane (no download). The real-data subject-
pooling test is marked ``network``/``slow``.
"""
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def test_feature_selection_leak_inflates_on_pure_noise():
    """Pitfall #3: selecting features on ALL data manufactures signal from noise.

    With pure-noise features and random labels the honest score must be ~chance,
    while fitting the selector on the whole dataset before CV inflates it.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2000))   # 2000 meaningless features
    y = rng.integers(0, 2, 200)            # labels unrelated to X

    # WRONG: select the 30 "best" features using all data, then cross-validate.
    X_leaked = SelectKBest(f_classif, k=30).fit(X, y).transform(X)
    wrong = np.mean([
        cross_val_score(LDA(), X_leaked, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
        for s in range(3)
    ])

    # RIGHT: selection inside the pipeline (re-fit on each train fold).
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=30)), ("lda", LDA())])
    right = np.mean([
        cross_val_score(pipe, X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
        for s in range(3)
    ])

    assert right < 0.60, f"honest score should be ~chance, got {right:.3f}"
    assert wrong > right + 0.10, f"leak should inflate: wrong={wrong:.3f} right={right:.3f}"


def test_accuracy_hides_failure_on_imbalanced_data():
    """Pitfall #4: accuracy looks great while the model detects nothing."""
    y = np.array([0] * 95 + [1] * 5)        # 5% positive (rare) class
    majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y), 1)), y)
    pred = majority.predict(np.zeros((len(y), 1)))

    assert accuracy_score(y, pred) >= 0.90          # looks excellent...
    assert balanced_accuracy_score(y, pred) == 0.5  # ...but detects none of class 1


@pytest.mark.network
@pytest.mark.slow
def test_subject_pooling_inflates_vs_loso():
    """Pitfall #2 on real data: pooled-random split > subject-independent (LOSO)."""
    from sklearn.model_selection import StratifiedKFold as SKF

    from neuro101 import features as ft
    from neuro101 import io
    from neuro101.eval import (
        evaluate_with_variance,
        leakage_safe_pipeline,
        make_subject_split,
    )

    X, y, subj = io.load_bnci_2a_epochs(n_subjects=3)
    pipe = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])

    wrong = evaluate_with_variance(
        pipe, X, y,
        cv=lambda: SKF(5, shuffle=True, random_state=0).split(X, y),
        scoring="accuracy", seeds=(0, 1),
    )["accuracy"]["mean"]
    right = evaluate_with_variance(
        pipe, X, y, cv=lambda: make_subject_split(subj),
        scoring="accuracy", seeds=(0, 1),
    )["accuracy"]["mean"]

    assert wrong > right, f"pooled-random should inflate: wrong={wrong:.3f} right={right:.3f}"
