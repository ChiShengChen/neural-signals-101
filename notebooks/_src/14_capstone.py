# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/14_capstone.ipynb)
#
# > **Running on Google Colab?** Run the next cell first — it installs everything and
# > fetches the helper package. **Running locally (after `make setup`)?** The next
# > cell does nothing; just run it and continue.

# %%
# --- Colab bootstrap: installs deps + the neuro101 package ONLY on Colab ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab setup complete — continue to the chapter below.")

# %% [markdown]
# # Chapter 14 — Capstone: The Honest Leaderboard
#
# ## Your mission
# Build a motor-imagery decoder for BCI IV 2a, iterate on the **DEV** set only,
# then submit once to the **hidden held-out leaderboard** to see how honest your
# estimate really was. Fill in the `TODO`s yourself.
#
# ## The game
# - A **hidden set** (one or more whole subjects) is carved off at the start.
# - You iterate with Leave-One-Subject-Out on the remaining **DEV subjects** only.
# - When you are happy, call `score_on_hidden(my_pipeline)` **once** — that is your
#   leaderboard submission.
# - We also show a ⚠️ **CHEATER's leaderboard**: someone who peeks at the hidden set
#   while picking between pipelines, then reports that cherry-picked score.
#   You will see exactly how much it inflates.
#
# ## Rules (the same ones the whole tutorial enforces)
# 1. Develop and tune on DEV subjects only.
# 2. Call `score_on_hidden` **once**, at the very end.
# 3. Every learned transform goes **inside a pipeline** (fit on train only).
# 4. Report **mean ± std**, never a single number.
# 5. Seed everything.
#
# > **Prerequisites:** Chapter 12.
# > **Difficulty:** ★★★★☆
# > **Runtime:** ~2–4 min on CPU once your TODOs are filled in.

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
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from neuro101 import io, datasets as ds, features as ft, viz
from neuro101.eval import (
    leakage_safe_pipeline, make_subject_split, make_block_split, evaluate_with_variance,
)

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## Step 1 — Load the data and carve the hidden set
#
# We load a fixed number of subjects, then **deterministically** reserve the last
# subject(s) as the hidden held-out set. You will not touch `X_hidden` / `y_hidden`
# until the final "leaderboard submission" cell.

# %%
# In smoke/CI mode we use 3 total subjects (1 hidden, 2 dev).
# Full mode: 5 total subjects (1 hidden, 4 dev).
n_total  = 3 if SMOKE else 5
n_hidden = 1            # number of subjects reserved as the hidden leaderboard set

X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_total)
all_subjects = np.unique(subj)

# Deterministic split: last subject(s) are hidden.
LOCKED_SUBJECTS = list(all_subjects[-n_hidden:])
DEV_SUBJECTS    = list(all_subjects[:-n_hidden])

dev_mask  = np.isin(subj, DEV_SUBJECTS)
X_dev, y_dev, subj_dev = X[dev_mask],  y[dev_mask],  subj[dev_mask]
X_hidden, y_hidden     = X[~dev_mask], y[~dev_mask]

print(f"DEV subjects : {DEV_SUBJECTS}  — {X_dev.shape[0]} trials")
print(f"LOCKED subjects: {LOCKED_SUBJECTS}  — {X_hidden.shape[0]} trials")
print(f"                  ^^^ do NOT inspect this until the final cell ^^^")

# %% [markdown]
# ### The `score_on_hidden` helper
#
# Call this **exactly once** — at the end of the notebook after you are done
# iterating. It trains on all DEV data and evaluates on the hidden set.

# %%
def score_on_hidden(pipeline):
    """Train on all DEV subjects, evaluate on the locked held-out subject(s).

    Returns the accuracy on the hidden set.
    Call this exactly ONCE — at the very end of your development cycle.
    """
    model = clone(pipeline)
    model.fit(X_dev, y_dev)
    acc = accuracy_score(y_hidden, model.predict(X_hidden))
    print(f"[LEADERBOARD] Hidden held-out accuracy: {acc:.3f}")
    return acc

# %% [markdown]
# ## Step 2 — TODO: build your pipeline
#
# Replace the baseline below with your own. Ideas: more CSP components, a different
# classifier (SVM, LogReg), or the Riemannian steps
# (`ft.make_riemann_pipeline_steps()`). **Keep every learned step inside the
# pipeline** so it is fit on train folds only.

# %%
# --- baseline (replace me) ---
my_pipeline = leakage_safe_pipeline([
    ("csp", ft.make_csp(n_components=4)),
    ("clf", LinearDiscriminantAnalysis()),
])
# TODO: try, e.g.
# from sklearn.svm import SVC
# my_pipeline = leakage_safe_pipeline(
#     ft.make_riemann_pipeline_steps() + [("clf", SVC(C=1, random_state=0))]
# )

# %% [markdown]
# ## Step 3 — DEV loop: honest LOSO on DEV subjects only
#
# This is the ONLY evaluation you should look at while building your model.
# If you have ≥ 2 DEV subjects, we use Leave-One-Subject-Out (LOSO).
# In smoke mode with 1 DEV subject we fall back to a block-aware split.

# %%
n_dev_subjects = len(DEV_SUBJECTS)

if n_dev_subjects >= 2:
    cv_fn = lambda: make_subject_split(subj_dev)          # noqa: E731
    cv_label = f"LOSO ({n_dev_subjects}-fold)"
else:
    # Smoke mode: only one dev subject — use block-aware split instead.
    cv_fn = lambda: make_block_split(len(y_dev), n_splits=4)   # noqa: E731
    cv_label = "block-aware split (smoke/single-subject)"

dev_report = evaluate_with_variance(
    my_pipeline, X_dev, y_dev,
    cv=cv_fn,
    scoring=("accuracy", "balanced_accuracy"),
    seeds=(0, 1),
)
for metric in ("accuracy", "balanced_accuracy"):
    m = dev_report[metric]
    print(f"  {metric:18s}: {m['mean']:.3f} ± {m['std']:.3f}")
print(f"  (CV: {cv_label}, {dev_report['n_folds']} folds × {dev_report['n_seeds']} seeds)")

dev_acc_mean = dev_report["accuracy"]["mean"]
dev_acc_std  = dev_report["accuracy"]["std"]

# %% [markdown]
# ### Your DEV estimate is your honest expectation
#
# Write it down before the next cells — that number is your best guess for what the
# hidden set will return.

# %% [markdown]
# ## Step 4 — ⚠️ The CHEATER's leaderboard (wrong — do not do this)
#
# A cheater does not develop on DEV only. They build several pipelines, peek at the
# hidden set for **each one**, pick the highest score they see, and report it as their
# result. The cell below re-enacts that mistake so you can see the inflation.
#
# **Every lookup of the hidden set is a peek — it costs you a degree of freedom.**

# %%
# Three pipelines the cheater tries on the hidden set.
_cheat_pipelines = {
    "CSP-4 + LDA (baseline)": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=4)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "CSP-8 + LDA": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=8)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "CSP-4 + LDA (seed 42)": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=4)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
}

_cheat_scores = {}
print("⚠️  CHEATER peeks at the hidden set for EACH pipeline:")
for name, pipe in _cheat_pipelines.items():
    m = clone(pipe)
    m.fit(X_dev, y_dev)
    s = accuracy_score(y_hidden, m.predict(X_hidden))
    _cheat_scores[name] = s
    print(f"   {name:30s} → hidden={s:.3f}")

_cheater_best_name  = max(_cheat_scores, key=_cheat_scores.get)
_cheater_best_score = _cheat_scores[_cheater_best_name]
print(f"\n⚠️  Cheater reports: '{_cheater_best_name}' → {_cheater_best_score:.3f}")

# Now get an honest DEV estimate of the SAME selected pipeline for comparison.
_honest_of_winner = evaluate_with_variance(
    _cheat_pipelines[_cheater_best_name], X_dev, y_dev,
    cv=cv_fn,
    scoring="accuracy", seeds=(0, 1),
)["accuracy"]

print(f"\n   Honest DEV estimate of the 'winner': {_honest_of_winner['mean']:.3f} ± {_honest_of_winner['std']:.3f}")
print(f"   Cheater's claim (best of {len(_cheat_scores)} peeks): {_cheater_best_score:.3f}")
_inflation = _cheater_best_score - _honest_of_winner["mean"]
print(f"   Apparent inflation from selection bias:  {_inflation:+.3f}")
print()
print("   The cheater can't know if their reported score is the 'lucky' one of several.")
print("   Each peek burns a degree of freedom on an irreplaceable dataset.")

# %% [markdown]
# ### Why does selection on the test set inflate?
#
# When you evaluate K pipelines on the same fixed test set and pick the maximum,
# you are doing the equivalent of a *multiple-comparisons* search.
# The maximum of K scores is almost always above the true expected value — even if
# none of the pipelines is genuinely better than the others.
# The more models you try, the worse the inflation.
#
# The only protection is to **never select on the test set**: pick your architecture
# on DEV (via LOSO / nested CV), lock it, then submit once.

# %% [markdown]
# ## Step 5 — TODO: plug in your own pipeline and submit to the leaderboard
#
# 1. Iterate in Step 2 / 3 until you are happy with your DEV LOSO score.
# 2. Replace `my_pipeline` in Step 2 with your final choice.
# 3. Run the cell below **once** — this is your leaderboard submission.

# %%
# >>> YOUR FINAL SUBMISSION — run this cell exactly ONCE <<<
hidden_acc = score_on_hidden(my_pipeline)

# %% [markdown]
# ## Step 6 — The report figure
#
# A clean summary: your DEV estimate vs the honest hidden held-out result.
# The gap shows how close (or far) your LOSO estimate was to reality.

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
labels  = ["DEV estimate\n(LOSO honest)", "Hidden held-out\n(leaderboard)"]
heights = [dev_acc_mean, hidden_acc]
errs    = [dev_acc_std, 0.0]
colors  = ["#4878CF", "#6ACC65"]
bars = ax.bar([0, 1], heights, width=0.55, color=colors,
              yerr=errs, capsize=6)
ax.axhline(0.25, ls="--", color="gray", lw=1)
ax.text(1.45, 0.25, "chance = 0.25", color="gray", va="center", ha="left", fontsize=8)
for b, h in zip(bars, heights):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}",
            ha="center", va="bottom", fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.0)
gap = dev_acc_mean - hidden_acc
ax.set_title(f"Capstone: DEV vs Hidden  (gap = {gap:+.3f})")
fig.tight_layout()
plt.show()

print(f"\nDEV estimate : {dev_acc_mean:.3f} ± {dev_acc_std:.3f}")
print(f"Hidden score : {hidden_acc:.3f}")
print(f"Gap          : {gap:+.3f}  "
      + ("(DEV was optimistic)" if gap > 0 else "(DEV was pessimistic)"))

# %% [markdown]
# ## Step 7 — TODO: write your conclusions
#
# In the markdown cell below, answer:
# 1. What is your **honest** hidden-held-out accuracy?
# 2. How close was your DEV estimate to the hidden result?
# 3. Did your pipeline changes actually help vs the baseline (given ± std)?
# 4. What would you try next with more data or time?

# %% [markdown]
# > **Your conclusions here.**
# >
# > _Example:_ "CSP+LDA reached 0.56 hidden (vs 0.70 DEV estimate).
# > The ~0.14 gap is typical for subject-independent transfer on BCI 2a.
# > I could not beat the baseline significantly given the spread; I'd try
# > per-subject fine-tuning with a small calibration set next."

# %% [markdown]
# ## ⚠️ Common mistakes
#
# - **Calling `score_on_hidden` more than once.** Every call is a peek; you can
#   no longer trust that number.
# - **Reporting the DEV LOSO number as your final result** without noting it is
#   an estimate. The hidden score is the ground truth.
# - **Claiming an improvement smaller than the std.** If model A is 0.66 ± 0.10
#   and model B is 0.68 ± 0.10, you have *not* shown B is better.
# - **Sneaking in a random split** "because the score is nicer". That is the exact
#   trap this whole tutorial exists to prevent.
# - **Tuning hyper-parameters on the test subject.** Use nested CV or a separate
#   validation subject for all tuning; the held-out subject is for final scoring only.
#
# ---
#
# 🎓 **Congratulations — you've finished the tutorial!**
#
# You can now take neural signals from raw recordings all the way to an honest,
# leakage-free report. You understand *why* evaluation method matters, *how* to
# spot common pitfalls, and *what* the score actually means. That combination —
# technical skill plus evaluation rigour — is the rarest thing in applied
# neural-signal ML. Go build something real.
