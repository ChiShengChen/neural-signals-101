# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/11_statistics_intuition.ipynb)
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
# # Chapter 11 — Statistics Intuition
#
# Before we dissect the pitfalls in Chapter 12, we need a set of **statistical
# reflexes** — quick gut-checks that tell you whether a number you're reading
# (or reporting) actually means what it appears to mean.
#
# This chapter is almost entirely **simulation-based**. We invent situations where
# we know the ground truth, and then we watch what our measurements do. That lets
# us see the gap between "what we measured" and "what is actually true" — a gap
# that is invisible in a real experiment but absolutely real.
#
# ## Learning objectives
# 1. Feel how much a single accuracy estimate can swing due to **sampling variation**.
# 2. Know what **mean ± std** and a **95 % confidence interval** really say.
# 3. Understand that **chance level is not always 1/n_classes** (imbalance!).
# 4. Recognise that a small difference between two bars is probably **just noise**
#    unless a paired test says otherwise.
#
# > **Prerequisites:** Chapter 02.
# > **Difficulty:** ★★★☆☆
# > **Runtime:** ~1–2 min.

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# %% [markdown]
# ---
# ## Section 1 — Sampling Variation: your estimate is noisy
#
# Imagine a classifier that has a **true accuracy of 0.65**.  That number is
# engraved in the universe — but you can never read it directly.  All you can
# do is run the classifier on a test set and count how often it was right.
#
# If your test set has only **20 examples**, you are effectively flipping a
# weighted coin 20 times and counting heads.  Twenty flips is not very many.
#
# > **Before running:** with only 20 test trials, how far off can a single
# > accuracy estimate be from the true value of 0.65?  Write down your guess,
# > then run the cell.

# %%
# --- simulation setup ---
TRUE_ACC = 0.65        # ground truth, unknown in practice
N_REPEATS = 5_000     # how many "experiments" we imagine running

def simulate_accuracy(n_test, n_repeats, true_acc, rng):
    """Return array of measured accuracies from n_repeats independent experiments."""
    # Each experiment: draw n_test Bernoulli(true_acc) outcomes, compute mean
    outcomes = rng.random(size=(n_repeats, n_test)) < true_acc
    return outcomes.mean(axis=1)

accs_20 = simulate_accuracy(20, N_REPEATS, TRUE_ACC, rng)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(accs_20, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(TRUE_ACC, color="crimson", lw=2.5, label=f"True accuracy = {TRUE_ACC}")
ax.axvline(accs_20.mean(), color="orange", lw=1.5, linestyle="--",
           label=f"Mean of estimates = {accs_20.mean():.3f}")
ax.set_xlabel("Measured accuracy (n = 20 test trials)", fontsize=12)
ax.set_ylabel("Number of simulated experiments", fontsize=12)
ax.set_title("Distribution of accuracy estimates with only 20 test trials\n"
             f"(true accuracy = {TRUE_ACC}, {N_REPEATS:,} simulated experiments)")
ax.legend()
pct5, pct95 = np.percentile(accs_20, [5, 95])
ax.text(0.02, 0.95,
        f"Middle 90 % range:\n[{pct5:.2f}, {pct95:.2f}]",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.show()

print(f"True accuracy:              {TRUE_ACC:.3f}")
print(f"Mean measured accuracy:     {accs_20.mean():.3f}")
print(f"Std of measured accuracies: {accs_20.std():.3f}")
print(f"90 % of experiments fell in [{pct5:.2f}, {pct95:.2f}]")
print(f"=> A single n=20 estimate can be off by ±{(pct95-pct5)/2:.2f} (90 % range)")

# %% [markdown]
# **What that histogram is telling you:**
# Each bar is "in this many simulated experiments, the measured accuracy landed
# here."  The true value is the red line — but a single experiment gives you
# just one draw from this whole spread.
#
# The spread is *enormous* for n=20.  Some experiments even report accuracy
# below 0.50 (chance!), even though the true accuracy is 0.65.
#
# ### How does sample size help?
#
# > **Before running:** predict which sample size will narrow the histogram the
# > most — going from n=20 to n=100, or from n=100 to n=500?

# %%
ns = [20, 100, 500]
colors = ["steelblue", "darkorange", "seagreen"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for ax, n, c in zip(axes, ns, colors):
    accs = simulate_accuracy(n, N_REPEATS, TRUE_ACC, rng)
    p5, p95 = np.percentile(accs, [5, 95])
    ax.hist(accs, bins=35, color=c, edgecolor="white", alpha=0.85)
    ax.axvline(TRUE_ACC, color="crimson", lw=2)
    ax.set_title(f"n = {n} test trials\nstd = {accs.std():.3f}   90 % in [{p5:.2f}, {p95:.2f}]",
                 fontsize=10)
    ax.set_xlabel("Measured accuracy")
    if ax is axes[0]:
        ax.set_ylabel("Count")
    ax.set_xlim(0.25, 1.0)

fig.suptitle(f"Sampling variation shrinks as test-set size grows  (true acc = {TRUE_ACC})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("n_test | std of estimate | 90 % range width")
for n in ns:
    accs = simulate_accuracy(n, N_REPEATS, TRUE_ACC, rng)
    p5, p95 = np.percentile(accs, [5, 95])
    print(f"  {n:>4}  |     {accs.std():.4f}      |    {p95-p5:.3f}")

# %% [markdown]
# **Key insight:** the standard deviation of the estimate shrinks roughly as
# 1/√n. Doubling the test set halves the standard deviation.
#
# * With n=20 you can barely tell whether your model beats chance.
# * With n=500 your estimate is tight enough to be meaningful.
#
# **Practical rule:** report how many test examples (or subjects, or trials)
# you used.  That number tells readers how much to trust your accuracy.

# %% [markdown]
# ---
# ## Section 2 — What mean ± std and confidence intervals really say
#
# In EEG papers you often see results like:
# *"LOSO accuracy: 0.68 ± 0.12 (N = 9 subjects)"*
#
# What does that actually mean, and how certain should we be?
#
# ### A concrete example: 9 subjects' LOSO accuracies

# %%
# Simulate 9 subjects' per-subject LOSO accuracies
# (imagine a real experiment with N=9 participants)
n_subjects = 9
true_subject_mean = 0.68

# Each subject's accuracy has real variability (between-subject differences)
subject_accs = np.array([0.61, 0.74, 0.55, 0.79, 0.63, 0.72, 0.58, 0.81, 0.68])
# (hand-crafted to be realistic; seeded so results are reproducible)

mean_acc = subject_accs.mean()
std_acc = subject_accs.std(ddof=1)   # sample std (divide by N-1)
se_acc = std_acc / np.sqrt(n_subjects)  # standard error of the mean

print(f"Subject accuracies: {subject_accs}")
print(f"Mean  = {mean_acc:.3f}")
print(f"Std   = {std_acc:.3f}  (spread across subjects)")
print(f"SE    = {se_acc:.3f}  (uncertainty about the mean)")

# %% [markdown]
# ### Bootstrap 95 % confidence interval
#
# A **confidence interval (CI)** answers: "Given these 9 numbers, what range
# would plausibly contain the true mean if we repeated the study?"
#
# We'll use **bootstrapping** — a resampling trick that requires no math
# assumptions:
# 1. Draw 9 subjects **with replacement** from our 9 subjects (some repeat, some
#    are skipped by chance).
# 2. Compute the mean of that resample.
# 3. Repeat 10 000 times.
# 4. The middle 95 % of those bootstrap means is our confidence interval.

# %%
N_BOOT = 10_000
boot_means = np.array([
    rng.choice(subject_accs, size=n_subjects, replace=True).mean()
    for _ in range(N_BOOT)
])

ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot_means, bins=60, color="mediumslateblue", edgecolor="white", alpha=0.85)
ax.axvline(mean_acc, color="crimson", lw=2.5, label=f"Observed mean = {mean_acc:.3f}")
ax.axvspan(ci_low, ci_high, alpha=0.18, color="gold",
           label=f"95 % CI = [{ci_low:.3f}, {ci_high:.3f}]")
ax.axvline(ci_low, color="darkorange", lw=1.5, linestyle="--")
ax.axvline(ci_high, color="darkorange", lw=1.5, linestyle="--")
# mark individual subjects
for a in subject_accs:
    ax.axvline(a, color="steelblue", lw=0.8, alpha=0.5)
ax.set_xlabel("Bootstrap mean accuracy", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Bootstrap distribution of the mean  (N = {n_subjects} subjects)\n"
             f"Width of 95 % CI = {ci_high - ci_low:.3f}  — nearly 20 percentage points wide!",
             fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print(f"\n95 % Bootstrap CI:  [{ci_low:.3f}, {ci_high:.3f}]")
print(f"CI width:            {ci_high - ci_low:.3f}")
print(f"\nInterpretation: with N={n_subjects} subjects, the true mean could plausibly")
print(f"be anywhere from {ci_low:.0%} to {ci_high:.0%}.")
print("That is a huge range! Always report N alongside mean ± std.")

# %% [markdown]
# **What the bootstrap distribution shows:**
# Each bar represents "this bootstrap resample of 9 subjects gave this mean."
# The shaded gold region is the 95 % CI.
#
# Notice how **wide** the CI is — nearly 20 percentage points!  A study that
# reports "accuracy = 0.68" with only 9 subjects leaves enormous uncertainty.
#
# This is not a flaw in the analysis — it is an honest description of what
# 9 subjects can and cannot tell us.  The only way to narrow the CI is to
# recruit more participants.
#
# **Std vs SE vs CI — which one to report?**
# | Quantity | Measures | Use when |
# |---|---|---|
# | Std (σ) | Spread *across subjects* | Showing individual variability |
# | SE = σ/√N | Uncertainty about the *mean* | Comparisons, inference |
# | 95 % CI | Plausible range for the mean | Publication, clearest communication |
#
# Many papers report mean ± std when they should show a CI.  Know the difference.

# %% [markdown]
# ---
# ## Section 3 — Chance level is NOT always 1/n_classes
#
# The most common mistake in classification papers: assuming that because there
# are 2 classes, the chance baseline is 50 %.  **This is only true when classes
# are balanced.**
#
# ### The imbalanced case
# Suppose you are detecting seizures: 90 % of 5-minute windows are normal
# (class 0) and 10 % contain a seizure (class 1).  A model that *always*
# predicts "normal" gets **90 % accuracy for free** — without learning anything.

# %%
# Simulate an imbalanced binary classification problem
rng2 = np.random.default_rng(42)

N_TRIALS = 200
P_CLASS1 = 0.10   # seizure rate (rare)
P_CLASS0 = 1 - P_CLASS1

# True labels (90 % zeros, 10 % ones)
y_true = (rng2.random(N_TRIALS) < P_CLASS1).astype(int)
print(f"Class counts: 0 → {(y_true==0).sum()}, 1 → {(y_true==1).sum()}")
print(f"Class proportions: {np.bincount(y_true) / N_TRIALS}")

# Majority-class dummy model: always predict 0
y_majority = np.zeros(N_TRIALS, dtype=int)
majority_acc = (y_majority == y_true).mean()

# Random model: predict 0 or 1 with equal probability
y_random = rng2.integers(0, 2, size=N_TRIALS)
random_acc = (y_random == y_true).mean()

# Perfect model (oracle) — upper bound
perfect_acc = 1.0

print(f"\nAlways-predict-majority accuracy: {majority_acc:.3f}")
print(f"Random (50/50) accuracy:          {random_acc:.3f}")
print(f"Perfect model accuracy:           {perfect_acc:.3f}")
print(f"\n=> The real baseline is {majority_acc:.2f}, not 0.50!")

# %% [markdown]
# ### Visualising the baselines

# %%
# Also show: even at TRUE chance (flipping a fair coin), you can get lucky
# with a small test set
N_BOOT_CHANCE = 5_000
n_test_small = 20

# Fair-coin predictions on N_TRIALS labels
chance_accs_small = np.array([
    ((rng2.integers(0, 2, n_test_small) == y_true[:n_test_small]).mean())
    for _ in range(N_BOOT_CHANCE)
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: bar chart of model baselines
ax = axes[0]
models_bar = ["Always\npredict 0\n(majority)", "Random\n50/50", "Your\nmodel\nmust beat"]
scores_bar = [majority_acc, random_acc, majority_acc]
bar_colors = ["tomato", "orange", "crimson"]
bars = ax.bar(models_bar, scores_bar, color=bar_colors, width=0.5, edgecolor="black")
ax.axhline(0.5, color="gray", linestyle="--", lw=1.5, label="Naive 1/k chance = 0.50")
ax.axhline(majority_acc, color="crimson", linestyle="-", lw=1.5,
           label=f"Real baseline = {majority_acc:.2f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Imbalanced classes (90 % class 0)\n"
             "The majority classifier scores 0.90 without learning anything", fontsize=10)
ax.legend(fontsize=9)
for bar, score in zip(bars, scores_bar):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.01, f"{score:.2f}",
            ha="center", fontsize=10, fontweight="bold")

# Right: "lucky chance" distribution with small n
ax2 = axes[1]
ax2.hist(chance_accs_small, bins=25, color="slategray", edgecolor="white", alpha=0.85)
ax2.axvline(0.5, color="navy", lw=2, label="True chance = 0.50")
ax2.axvline(majority_acc, color="crimson", lw=2,
            label=f"Majority baseline = {majority_acc:.2f}")
lucky = (chance_accs_small >= 0.80).mean()
ax2.set_xlabel(f"Accuracy of a fair-coin model on n={n_test_small} trials", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title(f"Even a random model can score ≥ 0.80 in {lucky:.0%} of experiments\n"
              f"(n = {n_test_small} test trials)", fontsize=10)
ax2.legend(fontsize=9)

plt.suptitle("Imbalance: the real baseline is majority accuracy, not 1/n_classes",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nWith only n={n_test_small} test trials, a purely random model")
print(f"scores ≥ 0.80 accuracy in {lucky:.1%} of experiments.")
print("=> Small test sets + imbalanced classes = very unreliable evaluation.")

# %% [markdown]
# **Key takeaways from Section 3:**
# * Always report the **majority-class baseline** alongside your model's accuracy.
# * For imbalanced problems, prefer **balanced accuracy**, **F1**, or **AUC**
#   instead of raw accuracy (covered in Chapter 12).
# * Even a coin-flip model can score impressively high when the test set is small
#   *and* classes are skewed — the binomial distribution has a long right tail.

# %% [markdown]
# ---
# ## Section 4 — A small gap between two bars is probably noise
#
# You report two models.  The bar chart looks like:
#
# ```
# Model A: 0.66    Model B: 0.68
# ```
#
# Is Model B better?  Almost certainly **no** — not without a statistical test.
#
# ### The setup
# We simulate 9 cross-validation folds for two models where the true
# performance is identical (both ~ 0.67).  Due to sampling noise, one will
# happen to average slightly higher than the other.

# %%
rng3 = np.random.default_rng(7)

n_folds = 9
# Per-fold accuracies (paired: same folds for both models)
# True means are very close; std is large (typical for small N)
fold_accs_A = np.array([0.60, 0.70, 0.55, 0.75, 0.65, 0.72, 0.58, 0.78, 0.62])
fold_accs_B = fold_accs_A + rng3.normal(loc=0.02, scale=0.04, size=n_folds)
fold_accs_B = np.clip(fold_accs_B, 0.0, 1.0)

mean_A, std_A = fold_accs_A.mean(), fold_accs_A.std(ddof=1)
mean_B, std_B = fold_accs_B.mean(), fold_accs_B.std(ddof=1)

print(f"Model A: mean={mean_A:.3f}  std={std_A:.3f}")
print(f"Model B: mean={mean_B:.3f}  std={std_B:.3f}")
print(f"Difference B - A = {mean_B - mean_A:.3f}")

# %% [markdown]
# ### Visualise: overlapping error bars tell you the difference could be zero

# %%
from scipy import stats

# Paired t-test (same folds => paired, not independent)
t_stat, p_ttest = stats.ttest_rel(fold_accs_A, fold_accs_B)
# Wilcoxon signed-rank (non-parametric alternative)
w_stat, p_wilcox = stats.wilcoxon(fold_accs_A, fold_accs_B)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: bar chart with error bars (what most papers show)
ax = axes[0]
x = np.array([0, 1])
means = [mean_A, mean_B]
stds  = [std_A, std_B]
bar_h = ax.bar(x, means, yerr=stds, width=0.4, capsize=8,
               color=["steelblue", "darkorange"], edgecolor="black",
               error_kw=dict(elinewidth=2, ecolor="black"))
ax.set_xticks(x)
ax.set_xticklabels(["Model A", "Model B"], fontsize=13)
ax.set_ylabel("Mean accuracy (± std across folds)", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title(f"Bar chart: B looks better by {mean_B - mean_A:.3f}\n"
             f"But error bars OVERLAP — difference is within noise", fontsize=10)
for xi, (m, s) in enumerate(zip(means, stds)):
    ax.text(xi, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=12, fontweight="bold")

# annotate the gap
ax.annotate("", xy=(1, mean_B), xytext=(0, mean_A),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2))
ax.text(0.5, (mean_A + mean_B)/2 + 0.01, f"Δ={mean_B-mean_A:.3f}",
        ha="center", fontsize=10, color="crimson")

# Right: per-fold scatter (what you should also show)
ax2 = axes[1]
folds_x = np.arange(1, n_folds + 1)
ax2.plot(folds_x, fold_accs_A, "o-", color="steelblue", label=f"Model A (mean={mean_A:.3f})", lw=2)
ax2.plot(folds_x, fold_accs_B, "s--", color="darkorange", label=f"Model B (mean={mean_B:.3f})", lw=2)
ax2.set_xlabel("Fold", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title(f"Per-fold accuracy  (n={n_folds} folds)\n"
              f"Paired t-test: p = {p_ttest:.3f}  |  Wilcoxon: p = {p_wilcox:.3f}", fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0.3, 1.0)
ax2.axhline(mean_A, color="steelblue", linestyle=":", alpha=0.5)
ax2.axhline(mean_B, color="darkorange", linestyle=":", alpha=0.5)
ax2.set_xticks(folds_x)

sig_label = "Not significant" if p_ttest > 0.05 else "Significant"
color_sig  = "firebrick" if p_ttest > 0.05 else "seagreen"
ax2.text(0.98, 0.05, f"Paired t-test: {sig_label}\n(p = {p_ttest:.3f} > 0.05)",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=10,
         color=color_sig,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

plt.suptitle("Is Model B really better? The paired test says: probably not.",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nPaired t-test:    t = {t_stat:.3f},  p = {p_ttest:.4f}")
print(f"Wilcoxon test:    W = {w_stat:.1f},   p = {p_wilcox:.4f}")
print()
if p_ttest > 0.05:
    print("=> p > 0.05: the difference is NOT statistically significant.")
    print("   We cannot claim Model B is better than Model A.")
else:
    print("=> p < 0.05: the difference IS statistically significant.")
print()
print(f"The gap of {mean_B - mean_A:.3f} is smaller than 1 std ({std_A:.3f}).")
print("Reporting just the means without a test would be misleading.")

# %% [markdown]
# **What the paired test does and why "paired" matters:**
#
# A paired test asks: "On the *same* folds, does Model B consistently beat
# Model A, or does each model win some folds by luck?"  Because both models
# see the same data split in each fold, their fold scores are *correlated*.
# Using a paired test (ttest_rel or Wilcoxon) instead of an independent test
# gives a fairer comparison and more power.
#
# **Practical checklist before claiming Model B beats Model A:**
# - [ ] Report mean ± std for *both* models.
# - [ ] Make sure error bars are shown in the figure.
# - [ ] Run a **paired** test across folds/subjects.
# - [ ] Report the p-value; if p > 0.05, say "no significant difference".
# - [ ] Ideally also report effect size (Cohen's d) and confidence interval on
#       the *difference*.

# %% [markdown]
# ---
# ## ✅ Concept check
#
# Test your understanding before moving on.
#
# **Q1.** A paper reports "accuracy = 0.72" on a 2-class EEG dataset.  Without
# knowing how many test examples were used, can you assess whether this result
# is meaningful?  Why or why not?
#
# **Q2.** You run LOSO cross-validation on 8 subjects and get
# mean accuracy = 0.74 ± 0.15.  A colleague says "great, CI is ±0.15."
# What is wrong with that statement?  What should the confidence interval
# actually be computed from?
#
# **Q3.** A seizure detector achieves 88 % accuracy on a test set where
# 85 % of windows are seizure-free.  Is this a good result?  What baseline
# should you compare against?
#
# **Q4.** Model A averages 0.70 and Model B averages 0.73 across 8 CV folds,
# both with std ≈ 0.12.  A reviewer says "B clearly wins."  What would you
# do to check this claim?
#
# ---
# **Answers:**
#
# **A1.** No.  Without n_test you cannot know the sampling variation.  With
# n=10 test trials the estimate swings ± 0.15; with n=500 it's ± 0.04.  The
# same reported number means something very different depending on n.
#
# **A2.** ± std is NOT the confidence interval.  The CI on the mean must
# account for sample size: SE = std/√N.  With N=8 subjects and std=0.15,
# SE ≈ 0.053, and a 95 % CI ≈ ±0.12 (not ±0.15).  Use bootstrapping or a
# t-interval for the mean, not the raw std.
#
# **A3.** No (not clearly).  The majority-class baseline (always predict
# "seizure-free") already scores 85 %.  A model at 88 % is only 3 percentage
# points above a model that learns nothing.  Use balanced accuracy or F1
# to fairly evaluate imbalanced problems.
#
# **A4.** Run a **paired** statistical test (ttest_rel or Wilcoxon) on the
# per-fold scores.  If p > 0.05, you cannot claim B is better.  The gap
# (0.03) is much smaller than 1 std (0.12), which strongly suggests noise.

# %% [markdown]
# ---
# ## ⚠️ Common mistakes / why this is wrong
#
# | Mistake | Why it matters | What to do instead |
# |---|---|---|
# | Reporting only one number ("accuracy = 0.72") | Hides all uncertainty; reader cannot judge reliability | Report mean ± std *and* n_test (or n_subjects) |
# | Assuming chance = 1/n_classes | Wrong for imbalanced data; inflates apparent improvement | Always compute majority-class baseline; use balanced accuracy |
# | Claiming improvement < 1 std is real | High probability the difference is sampling noise | Always run a paired significance test across folds |
# | Ignoring N | N=5 subjects and N=50 are not the same confidence | State N prominently; report CI width |
# | Treating std as the CI | std measures spread across subjects; CI/√N measures uncertainty of the mean | Report SE or bootstrap CI for inference |
# | Running an independent test when data are paired | Loses statistical power; can inflate or deflate p | Use ttest_rel / Wilcoxon when comparing same folds |
#
# ---
# **Next:** Chapter 12 — Evaluation & Pitfalls — where we apply these reflexes
# to real decoding pipelines and see exactly how each mistake inflates scores.
