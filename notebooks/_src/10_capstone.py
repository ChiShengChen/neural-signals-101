# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 10 — Capstone: Raw → Honest Report
#
# Time to put it together. You will take **one public dataset** from raw recordings
# all the way to an **honest, subject-independent report** — filling in the `TODO`s
# yourself. The scaffolding makes the *structure* correct (no leakage) so you can
# focus on the modelling choices.
#
# ## Your mission
# Build a motor-imagery decoder for BCI IV 2a and report its honest performance,
# then try to beat the baseline **without** introducing any leakage.
#
# ## Rules (the same ones the whole tutorial enforces)
# 1. Headline metric is **subject-independent** (Leave-One-Subject-Out).
# 2. Every learned transform goes **inside a pipeline** (fit on train only).
# 3. Report **mean ± std**, never a single number.
# 4. Seed everything.
#
# **Runtime:** ~2–4 min once your TODOs are filled in.

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt

from neuro101 import io, datasets as ds, features as ft, viz
from neuro101.eval import (
    leakage_safe_pipeline, make_subject_split, make_block_split, evaluate_with_variance,
)

SMOKE = ds.is_smoke()
SEED = 0
np.random.seed(SEED)

# %% [markdown]
# ## Step 1 — Load the data
#
# We provide this step. (In smoke/CI mode fewer subjects are loaded automatically.)

# %%
n_subj = 2 if SMOKE else 4
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X.shape} | classes={np.bincount(y)} | subjects={np.unique(subj)}")

# %% [markdown]
# ## Step 2 — TODO: build your pipeline
#
# Replace the baseline below with your own. Ideas: more CSP components, a different
# classifier (SVM, LogReg, random forest), or the Riemannian steps
# (`ft.make_riemann_pipeline_steps()`). **Keep every learned step inside the
# pipeline** so it is fit on train folds only.

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- baseline (replace me) ---
my_pipeline = leakage_safe_pipeline([
    ("csp", ft.make_csp(n_components=4)),
    ("clf", LinearDiscriminantAnalysis()),
])
# TODO: try, e.g.
# from sklearn.svm import SVC
# my_pipeline = leakage_safe_pipeline(
#     ft.make_riemann_pipeline_steps() + [("clf", SVC(C=1))]
# )

# %% [markdown]
# ## Step 3 — Honest evaluation (subject-independent)
#
# We provide the correct evaluation call. Do **not** change it to a random split!

# %%
report = evaluate_with_variance(
    my_pipeline, X, y,
    cv=lambda: make_subject_split(subj),
    scoring=("accuracy", "balanced_accuracy", "f1_macro"),
    seeds=(0, 1),
)
for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
    m = report[metric]
    print(f"  {metric:18s}: {m['mean']:.3f} ± {m['std']:.3f}")
print(f"  (over {report['n_folds']} held-out subjects × {report['n_seeds']} seeds)")

# %% [markdown]
# ## Step 4 — TODO: an optimistic baseline for contrast
#
# Compute a **within-subject** (block-aware) score on a single subject and notice
# it is higher than the LOSO number above. This is the gap between "works on me"
# and "works on a new person".

# %%
# TODO: pick one subject, evaluate with make_block_split, compare to LOSO.
s0 = subj == np.unique(subj)[0]
within = evaluate_with_variance(
    my_pipeline, X[s0], y[s0],
    cv=lambda: make_block_split(int(s0.sum()), n_splits=4),
    scoring="accuracy", seeds=(0,),
)["accuracy"]
print(f"Within-subject (optimistic): {within['mean']:.3f} ± {within['std']:.3f}")
print(f"Subject-independent (honest): {report['accuracy']['mean']:.3f} ± {report['accuracy']['std']:.3f}")

# %% [markdown]
# ## Step 5 — The report figure
#
# A clean summary: honest subject-independent accuracy with its spread vs the
# optimistic within-subject number and chance.

# %%
viz.plot_wrong_vs_right(
    within["mean"], report["accuracy"]["mean"],
    wrong_err=within["std"], right_err=report["accuracy"]["std"],
    chance=0.5,
    wrong_label="Within-subject\n(optimistic)",
    right_label="Subject-independent\n(honest headline)",
    metric="Accuracy", title="My capstone decoder — honest report",
)
plt.show()

# %% [markdown]
# ## Step 6 — TODO: write your conclusions
#
# In the markdown cell below, answer:
# 1. What is your **honest** (subject-independent) accuracy and its spread?
# 2. How big is the optimism gap vs within-subject?
# 3. Did your changes actually help, given the ± std (or could it be noise)?
# 4. What would you try next (more data? a different feature? per-subject calibration)?

# %% [markdown]
# > **Your conclusions here.**
# >
# > _Example:_ "CSP+LDA reached 0.66 ± 0.10 subject-independent vs 0.78 within-subject.
# > Switching to Riemannian features changed it by less than one std, so I can't
# > claim it helped. Next I'd add a few calibration trials from the target subject."

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Reporting the within-subject number as your result.** It is the optimistic
#   one; the headline must be subject-independent.
# - **Sneaking in a random split** "because the score is nicer". That is the exact
#   trap this whole tutorial exists to prevent.
# - **Claiming an improvement smaller than the std.** If model A is 0.66 ± 0.10 and
#   model B is 0.68 ± 0.10, you have *not* shown B is better. Use a paired test
#   across folds (Chapter 09).
# - **Tuning on the test subjects.** Choose hyper-parameters with nested CV or a
#   held-out validation subject.
#
# 🎓 **Congratulations** — you can now take neural signals from raw recordings to an
# honest, leakage-free report. That honesty is the rarest and most valuable skill
# in applied neural-signal ML.
