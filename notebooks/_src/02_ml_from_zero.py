# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/02_ml_from_zero.ipynb)
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
# # Chapter 02 — ML from Zero
#
# ## Learning objectives
# 1. Understand what **features** (X) and **labels** (y) are, and visualise them on
#    a 2-D scatter plot.
# 2. Split data into **train / validation / test** sets and know exactly what each
#    part is *for*.
# 3. Recognise **overfitting**: why train accuracy rising to 1.0 is a warning sign,
#    not a success.
# 4. Read a **decision boundary** plot and identify underfit, good, and overfit
#    models by eye.
# 5. Understand why picking your model by peeking at the test set **inflates your
#    reported score** — and how to avoid it.
#
# > **Prerequisites:** none — this is a from-scratch start.
# > **Difficulty:** ★★☆☆☆
# > **Runtime:** ~1 min (toy data, CPU).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

# Additional sklearn imports used throughout this chapter
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1 — Features and labels: what X and y are
#
# Every supervised machine-learning pipeline starts with two arrays:
#
# | Symbol | Shape | Plain-English meaning |
# |---|---|---|
# | **X** | `(n_samples, n_features)` | The *measurements* for each sample — what the model **sees** |
# | **y** | `(n_samples,)` | The *correct answer* for each sample — what the model must **predict** |
#
# A **sample** is one thing you measured (here: one data point).
# A **feature** is one number that describes that sample (here: its x- and
# y-coordinate on the plane).
# A **label** is the class that sample belongs to (here: 0 = blue, 1 = orange).
#
# We use `sklearn.datasets.make_moons` to generate a toy two-class problem —
# two interleaved crescent shapes that are easy to visualise but *not* linearly
# separable (a straight line can't perfectly divide them).

# %%
# ----- Generate toy data -----
X, y = make_moons(n_samples=500, noise=0.30, random_state=0)

print(f"X shape : {X.shape}   <- (n_samples, n_features)")
print(f"y shape : {y.shape}   <- (n_samples,)")
print(f"Classes : {np.unique(y)}   <- 0 = moon-A, 1 = moon-B")

fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.6, edgecolors="k",
                     linewidths=0.4, s=30)
ax.set_xlabel("Feature 0  (X[:, 0])")
ax.set_ylabel("Feature 1  (X[:, 1])")
ax.set_title("Our toy dataset — two moons\n"
             "Colour = label y  (red=1, blue=0)")
plt.colorbar(scatter, ax=ax, label="y (class label)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Notice that blue and red are tangled — no single straight cut separates them
# perfectly. That's the point: real data rarely is linearly separable, and we
# need a model flexible enough to capture curved boundaries.

# %% [markdown]
# ## 2 — Train / validation / test split
#
# We split our dataset into **three non-overlapping** parts before touching a
# model:
#
# | Part | Typical size | Purpose |
# |---|---|---|
# | **Train** | 60–70% | The model *learns* from this |
# | **Validation** | 15–20% | We tune hyperparameters here; look at it as often as you like |
# | **Test** | 15–20% | **SACRED** — touch it only once, at the very end |
#
# ### Why is the test set sacred?
#
# Every time you look at the test set and make a decision (e.g. "complexity 7
# is better than complexity 5"), you are *implicitly* fitting to it.  After many
# such peeks the test accuracy no longer tells you how the model will do on
# truly unseen data — it tells you how well you searched.  You'll see a concrete
# demonstration of this in Section 5.
#
# > **Note on shuffling — READ THIS before Chapter 12.**
# > Below we shuffle data before splitting with `train_test_split`.  This is
# > correct *here* because `make_moons` generates **i.i.d.** (independently and
# > identically distributed) random points — each point is independent of every
# > other.  EEG recordings are **time series**: nearby samples are correlated,
# > so shuffling would leak future information into the training set and give
# > wildly optimistic results.  Chapter 12 covers the right way to split
# > time-series data.

# %%
# ----- Three-way split -----
# First: carve off the test set (20 %)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y
)
# Second: split the remainder into train (75 % of 80 % = 60 %) + val (20 %)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval
)

print(f"Total samples   : {len(X)}")
print(f"  Train         : {len(X_train)}  ({len(X_train)/len(X):.0%})")
print(f"  Validation    : {len(X_val)}   ({len(X_val)/len(X):.0%})")
print(f"  Test (SACRED) : {len(X_test)}  ({len(X_test)/len(X):.0%})")

# %% [markdown]
# ## 3 — Overfitting
#
# A **Decision Tree** splits the feature space into rectangular regions; the
# deeper the tree, the more splits, the more complex the boundary.
#
# `max_depth=1` can only make a single cut — it will **underfit** (too simple).
# `max_depth=15` can memorise every training point — it will **overfit** (too
# complex, wiggly boundary that won't generalise).
#
# ### Before you run the next cell — make a prediction!
#
# As `max_depth` increases from 1 to 15:
# - What do you expect to happen to **training accuracy**?
# - What do you expect to happen to **validation accuracy**?
#
# Write your guesses down (or just think about them), then run the cell.

# %%
depths = list(range(1, 16))
train_accs, val_accs = [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
    val_accs.append(accuracy_score(y_val, clf.predict(X_val)))

best_val_depth = depths[np.argmax(val_accs)]
print(f"Best depth by VALIDATION accuracy : {best_val_depth}")
print(f"  Train acc at best depth : {train_accs[best_val_depth-1]:.3f}")
print(f"  Val   acc at best depth : {max(val_accs):.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, train_accs, "o-", color="steelblue", label="Train accuracy")
ax.plot(depths, val_accs,   "s-", color="tomato",    label="Validation accuracy")
ax.axvline(best_val_depth, color="gray", linestyle="--", alpha=0.7,
           label=f"Best val depth = {best_val_depth}")
ax.set_xlabel("Tree max_depth  (model complexity →)")
ax.set_ylabel("Accuracy")
ax.set_title("The overfitting U-gap curve\n"
             "Train keeps climbing; validation peaks then falls")
ax.legend()
ax.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What you should see:**
#
# - **Train accuracy** climbs steadily toward 1.0 as `max_depth` grows.
#   At depth 15 the tree has memorised every training point — 100% train accuracy.
# - **Validation accuracy** peaks around a modest depth, then *falls* as the tree
#   starts memorising training noise.
# - The gap between the two curves is the **overfitting gap**.  A large gap is a
#   warning sign that your model won't generalise to new data.
#
# The "right" depth is the one that peaks on the **validation** curve — we never
# use the test set to make this choice.

# %% [markdown]
# ## 4 — Decision boundary visualisation
#
# A **decision boundary** is the line (or curve) where the model changes its
# prediction from one class to the other.  We draw it by predicting the class
# at every point on a fine grid, then colour-filling the grid (this is called a
# **meshgrid contourf** plot).
#
# We will look at three depths side by side:
#
# | depth | Expected behaviour |
# |---|---|
# | 1 | **Underfit** — one horizontal or vertical cut; misses most of the structure |
# | best_val | **Good fit** — curved boundary that roughly follows both moons |
# | 15 | **Overfit** — jagged, memorised islands that fit the *noise* in training data |
#
# ### Before you run: predict the shape of the boundary
#
# For depth 1, depth `best_val_depth`, and depth 15 — what shape do you expect
# the coloured regions to have?  Will they look smooth or jagged?

# %%
def plot_boundary(ax, clf, X_plot, y_plot, title):
    """Draw a meshgrid decision boundary with training points overlaid."""
    x_min, x_max = X_plot[:, 0].min() - 0.4, X_plot[:, 0].max() + 0.4
    y_min, y_max = X_plot[:, 1].min() - 0.4, X_plot[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="bwr", levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap="bwr",
               edgecolors="k", linewidths=0.4, s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, depth, label in zip(
    axes,
    [1, best_val_depth, 15],
    ["Underfit (depth=1)", f"Good fit (depth={best_val_depth})", "Overfit (depth=15)"]
):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    plot_boundary(ax, clf, X_trainval, y_trainval, label)

fig.suptitle("Decision boundaries — underfit / good / overfit", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What you should see:**
#
# - **Left (depth=1):** a single horizontal or vertical stripe — the model has
#   no idea about the curved moons.  Many points are wrong.
# - **Middle (depth=best):** a smooth-ish curved boundary that follows the moons
#   without being too wiggly.  Some mistakes near the noisy overlap region — that
#   is fine and expected.
# - **Right (depth=15):** jagged, island-filled regions that wrap tightly around
#   individual training points.  On *new* data those islands are mostly wrong.

# %% [markdown]
# ## 5 — The punchline: peeking at the test set inflates your score
#
# Suppose we are impatient and skip the validation set.  We train a tree at
# every depth, evaluate it on the **test** set each time, and pick the depth
# that maximises test accuracy.  What score do we report?

# %%
# ----- "Cheating" scenario: choose depth by peeking at TEST -----
test_accs_all = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)            # train on training set only
    test_accs_all.append(accuracy_score(y_test, clf.predict(X_test)))

best_cheat_depth = depths[np.argmax(test_accs_all)]
cheat_score      = max(test_accs_all)

# ----- Honest scenario: choose depth by VALIDATION, report TEST once -----
honest_clf = DecisionTreeClassifier(max_depth=best_val_depth, random_state=0)
honest_clf.fit(X_train, y_train)
honest_score = accuracy_score(y_test, honest_clf.predict(X_test))

print("=" * 50)
print(f"CHEATING  — chose depth={best_cheat_depth} by maximising TEST acc")
print(f"  Reported test accuracy : {cheat_score:.3f}  <- OPTIMISTIC, do not trust")
print()
print(f"HONEST    — chose depth={best_val_depth} by validation, report test ONCE")
print(f"  Reported test accuracy : {honest_score:.3f}  <- trustworthy")
print("=" * 50)
print()
print("Difference (optimism bias):", round(cheat_score - honest_score, 3))

# %% [markdown]
# The **honest** number is lower.  That gap is called **optimism bias**: every
# time you look at the test set and use it to make a decision, you burn a little
# of its independence.  After enough peeks, the test set has effectively become
# another training set.
#
# **The rule:**
# - Use **validation** to tune hyperparameters (tree depth, number of neighbours,
#   regularisation strength, …).
# - Evaluate on **test** exactly **once**, at the very end, and report that number.
#   Then stop.  Do not go back and try to squeeze out more.
#
# If you have very little data you can use **k-fold cross-validation** on the
# train+val pool instead of a single split — but the test set stays sacred
# regardless.

# %% [markdown]
# ## ✅ Concept check
#
# Answer these before moving on:
#
# 1. A model achieves 99% train accuracy and 72% test accuracy.  Is this good or
#    bad?  What is the technical term for this situation?
# 2. You train five models with different hyperparameters, evaluate each on the
#    test set, and pick the best one.  Why is the reported test accuracy
#    misleading?
# 3. What is the purpose of the **validation** set?  How is it different from the
#    test set?
# 4. In this chapter we shuffled the data before splitting.  Why would shuffling
#    be *wrong* for EEG time-series data?  (Hint: think about what "i.i.d."
#    means.)
#
# **Answers:**
#
# 1. Bad — this is **overfitting**. The model memorised the training data but
#    fails to generalise.  The 27-point gap between train and test accuracy is the
#    overfitting gap.
# 2. You effectively used the test set as a second validation set.  Each
#    comparison was a chance to pick the result that looks best by luck, so the
#    winning number is optimistically biased.
# 3. The validation set is used to tune hyperparameters during model development;
#    you may look at it many times.  The test set is held out until the very end
#    and touched only once, so its accuracy is an unbiased estimate of future
#    performance.
# 4. EEG samples at time t and t+1 are correlated (not i.i.d.).  Shuffling breaks
#    that structure, lets the model "see" future time points during training, and
#    produces a wildly optimistic accuracy.  Chapter 12 covers time-series-safe
#    splitting strategies.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Choosing your model by looking at the test set.** Even "just one peek" counts.
#   Every peek lets the test set influence your choice, inflating your reported
#   accuracy.  Use the validation set for all tuning decisions.
#
# - **No held-out test set at all.** Reporting only train accuracy (or even
#   cross-validation accuracy on the full dataset) tells you nothing about
#   generalisation.  Always keep a test set you have never touched.
#
# - **Shuffling time-series data.** Shuffling is fine for i.i.d. toy datasets like
#   the ones in this chapter.  For EEG (or any time series) it leaks future
#   information into the training set.  Chapter 12 shows the right approach:
#   group-aware or time-ordered splits.
#
# - **Reading one accuracy number without a chance baseline.** If your dataset has
#   90% class-A samples, a model that always predicts A scores 90% with zero
#   knowledge.  Always compare against the **majority-class baseline** (or random
#   chance).
#
# - **Forgetting to stratify.** Without `stratify=y` in `train_test_split`, a
#   random split might put almost all of one class in train and none in test (a
#   problem with small or imbalanced datasets).  Always pass `stratify=y` for
#   classification tasks.

# %% [markdown]
# **Next:** Chapter 03 — *Math you can see* — builds the geometric intuition
# (dot products, distances, projections) that underlies every model in this
# tutorial, with interactive plots and no heavy algebra.
