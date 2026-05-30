# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 06 — Classical Machine Learning (the right way)
#
# Now we classify motor imagery with classical models — and we do it **honestly**.
# The single most important idea in this whole tutorial appears here:
# **time-series / grouped data cannot be split by random shuffle.**
#
# ## Learning objectives
# 1. Build sklearn **Pipelines** that fit preprocessing on **train only** (no leakage).
# 2. Compare **LDA**, **SVM**, **random forest** + the **Riemannian** baseline.
# 3. Use the **right cross-validation**: block-aware (within subject) and
#    Leave-One-Subject-Out (across subjects).
# 4. Read evaluation **metrics**: accuracy, balanced accuracy, F1.
#
# **Runtime:** ~2–3 min on CPU.

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neuro101 import io, datasets as ds
from neuro101 import features as ft
from neuro101.eval import (
    evaluate_with_variance, leakage_safe_pipeline,
    make_block_split, make_subject_split,
)

SMOKE = ds.is_smoke()
n_subj = 2 if SMOKE else 4
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X.shape} | classes={np.bincount(y)} | subjects={np.unique(subj)}")

# %% [markdown]
# ## The golden rule: fit transforms on train only
#
# Any step that *learns* from data — a scaler's mean/std, CSP's spatial filters,
# covariance whitening — must see **only the training fold**. sklearn `Pipeline`
# does this automatically: when you call `cross_val_score`, every step is re-fit on
# each fold's training data. Our `leakage_safe_pipeline` is just a documented
# `Pipeline` to make that intent obvious.

# %%
# Four model pipelines. CSP and covariance are fit inside the pipeline => no leak.
models = {
    "CSP + LDA": leakage_safe_pipeline(
        [("csp", ft.make_csp(4)), ("lda", LinearDiscriminantAnalysis())]),
    "CSP + SVM": leakage_safe_pipeline(
        [("csp", ft.make_csp(4)), ("scale", StandardScaler()), ("svm", SVC(C=1, kernel="rbf"))]),
    "Bandpower + RF": leakage_safe_pipeline(
        [("clf", RandomForestClassifier(n_estimators=200, random_state=0))]),
    "Riemann + LogReg": leakage_safe_pipeline(
        ft.make_riemann_pipeline_steps() + [("clf", LogisticRegression(max_iter=500))]),
}

# Bandpower+RF needs 2-D features; the others take raw epochs. Precompute bandpower.
X_bp = ft.bandpower(X, sf)

# %% [markdown]
# ## Within-subject evaluation (block-aware split)
#
# First, the easier question: can we decode *one* person, testing on held-out
# *blocks* of their trials (never random shuffle)? We use subject 1 and a
# trial-aware block split.

# %%
s0 = subj == np.unique(subj)[0]
Xs, ys = X[s0], y[s0]
Xs_bp = X_bp[s0]
n_blocks = 5

print("Within-subject (subject 1), 5 block-aware folds:")
for name, model in models.items():
    data = Xs_bp if name == "Bandpower + RF" else Xs
    res = evaluate_with_variance(
        model, data, ys,
        cv=lambda: make_block_split(len(ys), n_splits=n_blocks),
        scoring=("accuracy", "balanced_accuracy", "f1_macro"),
        seeds=(0, 1),
    )
    a = res["accuracy"]; print(f"  {name:18s} acc {a['mean']:.3f} ± {a['std']:.3f}")

# %% [markdown]
# ## The headline question: subject-independent (Leave-One-Subject-Out)
#
# The honest, deployment-relevant question is: does it work on a **new person**?
# We hold out each subject in turn (LOSO). Expect **lower** numbers — generalising
# across brains is hard. This is the metric we report as the headline.

# %%
results = {}
for name, model in models.items():
    data = X_bp if name == "Bandpower + RF" else X
    res = evaluate_with_variance(
        model, data, y,
        cv=lambda: make_subject_split(subj),
        scoring=("accuracy", "balanced_accuracy", "f1_macro"),
        seeds=(0, 1),
    )
    results[name] = res
    a, ba, f1 = res["accuracy"], res["balanced_accuracy"], res["f1_macro"]
    print(f"  {name:18s} acc {a['mean']:.3f}±{a['std']:.3f}  "
          f"bal_acc {ba['mean']:.3f}  f1 {f1['mean']:.3f}")

# %% [markdown]
# ## Compare models honestly (mean ± std across subjects)

# %%
names = list(results)
means = [results[n]["accuracy"]["mean"] for n in names]
stds = [results[n]["accuracy"]["std"] for n in names]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, means, yerr=stds, capsize=6, color="#4c72b0")
ax.axhline(0.5, ls="--", color="gray", label="chance (2-class)")
ax.set(ylabel="LOSO accuracy", title="Subject-independent accuracy (mean ± std over held-out subjects)")
ax.set_ylim(0, 1); plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Metrics, briefly (full treatment in Chapter 09)
#
# - **Accuracy**: fraction correct. Misleading when classes are imbalanced.
# - **Balanced accuracy**: average recall across classes — fair under imbalance.
# - **F1**: harmonic mean of precision and recall; good for "did we catch the
#   positive class without crying wolf".
# - Always report **variance** (± std over folds/subjects), never a single number.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **`train_test_split(shuffle=True)` on epochs or time series.** Adjacent /
#   same-subject samples leak. Use `make_block_split` (within subject) or
#   `make_subject_split` (across subjects). *This repo does not even provide a
#   random-shuffle splitter.*
# - **`scaler.fit_transform(X)` before cross-validation.** Test statistics leak
#   into training. Put the scaler **in the pipeline**.
# - **Reporting the within-subject number as your headline.** It is optimistic.
#   The subject-independent (LOSO) number is the honest one.
# - **One run, one number.** Seeds and folds vary; report mean ± std.
# - **Tuning hyper-parameters on the test fold.** Use nested CV or a separate
#   validation split — never peek at test.
#
# **Next:** Chapter 07 — deep learning (EEGNet & friends) with braindecode, still
# obeying every rule above.
