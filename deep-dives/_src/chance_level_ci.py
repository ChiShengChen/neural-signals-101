# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Chance Level & Confidence Intervals
#
# What does "above chance" actually mean for a BCI result, and how confident
# should you be in a reported classification accuracy?
#
# > **Prerequisites:** main Chapter 11.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget.**

# %%
# --- bootstrap: try the installed package first, fall back to repo src ---
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# %% [markdown]
# ---
# ## Part 1 — The Binomial Test: formalising "better than chance"
#
# When a classifier makes predictions on **n** independent test trials and gets
# **k** correct, the number of correct predictions under a chance classifier is
# distributed as:
#
# $$K_\text{chance} \sim \text{Binomial}(n,\; p_\text{chance}), \quad
#   p_\text{chance} = 1/k_\text{classes}$$
#
# The one-sided binomial test asks: if the true accuracy were exactly
# $p_\text{chance}$, how likely is it to observe **at least** k correct by luck
# alone?  That probability is the **p-value**.
#
# $$p = P(K \ge k \mid n,\; p_\text{chance}) = \sum_{j=k}^{n} \binom{n}{j} p_\text{chance}^j (1-p_\text{chance})^{n-j}$$
#
# `scipy.stats.binomtest` computes this exactly (no normal approximation).

# %%
def binomial_test(k_correct: int, n: int, p_chance: float = 0.5) -> dict:
    """
    One-sided binomial test: H0 accuracy = p_chance, H1 accuracy > p_chance.

    Parameters
    ----------
    k_correct : int   Number of correctly classified trials.
    n         : int   Total number of test trials.
    p_chance  : float Chance level (1/n_classes).

    Returns
    -------
    dict with keys: acc, p_value, significant (alpha=0.05)
    """
    acc = k_correct / n
    result = stats.binomtest(k_correct, n, p_chance, alternative="greater")
    return {
        "acc": acc,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
    }


# Demonstrate across a range of (n, observed accuracy) combinations
print(f"{'n':>6} {'obs_acc':>8} {'k':>5} {'p_value':>10} {'sig?':>6}")
print("-" * 42)
for n, obs_acc in [(20, 0.65), (20, 0.80), (100, 0.65), (100, 0.58), (400, 0.58)]:
    k = round(obs_acc * n)
    r = binomial_test(k, n, p_chance=0.5)
    sig = "YES" if r["significant"] else "no"
    print(f"{n:>6} {obs_acc:>8.2f} {k:>5} {r['p_value']:>10.4f} {sig:>6}")

# %% [markdown]
# Notice that **65 % accuracy with only 20 trials is not significant** (p = 0.13),
# whereas 58 % with 400 trials is highly significant (p < 0.001).  The raw accuracy
# number tells you almost nothing without knowing n.

# %% [markdown]
# ---
# ## Part 2 — Confidence Intervals on Accuracy: Wald vs Wilson vs Clopper-Pearson
#
# An observed accuracy $\hat{p} = k/n$ is a point estimate.  We always need an
# interval.  Three common choices differ substantially for small n or extreme p.
#
# ### 2a  The Wald (normal approximation) interval
# $$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
# Simple, but breaks badly when $\hat{p}$ is near 0 or 1, or when n is small:
# it can produce impossible intervals that extend outside [0, 1].
#
# ### 2b  The Wilson score interval (recommended for most uses)
# $$\frac{\hat{p} + \tfrac{z^2}{2n} \pm z\sqrt{\tfrac{\hat{p}(1-\hat{p})}{n} + \tfrac{z^2}{4n^2}}}{1 + z^2/n}$$
# Stays within [0,1] and has much better coverage for small n.
#
# ### 2c  The Clopper-Pearson exact interval
# Inverts the binomial CDF directly — the "gold standard" for guaranteed coverage,
# but tends to be conservative (wider than necessary).

# %%
def wald_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Normal approximation (Wald) confidence interval."""
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval (Brown et al. 2001)."""
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval via beta distribution quantiles."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


# --- Compare the three methods for a fixed observed accuracy ---
print("Comparison for obs_acc = 0.60, three values of n")
print(f"{'n':>5} {'Method':>18} {'Lo':>7} {'Hi':>7} {'Width':>7}")
print("-" * 46)
for n in [10, 20, 100]:
    k = round(0.60 * n)
    p_hat = k / n
    methods = {
        "Wald": wald_ci(k, n),
        "Wilson": wilson_ci(k, n),
        "Clopper-Pearson": clopper_pearson_ci(k, n),
    }
    for name, (lo, hi) in methods.items():
        print(f"{n:>5} {name:>18} {lo:>7.4f} {hi:>7.4f} {hi-lo:>7.4f}")
    print()

# %% [markdown]
# ### Visualisation 1 — CI width vs n for three methods
#
# For a fixed "true" accuracy of 0.60, how does the width of the 95 % CI shrink
# as we collect more test trials?

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

obs_acc_vals = [0.60, 0.90]   # moderate vs extreme observed accuracy
ns = np.arange(5, 301, 1)

for ax, obs_acc_fixed in zip(axes, obs_acc_vals):
    wald_widths, wilson_widths, cp_widths = [], [], []
    for n in ns:
        k = round(obs_acc_fixed * n)
        k = max(0, min(n, k))
        wald_widths.append(wald_ci(k, n)[1] - wald_ci(k, n)[0])
        wilson_widths.append(wilson_ci(k, n)[1] - wilson_ci(k, n)[0])
        cp_widths.append(clopper_pearson_ci(k, n)[1] - clopper_pearson_ci(k, n)[0])

    ax.plot(ns, wald_widths,    label="Wald",            color="tomato",      lw=1.8)
    ax.plot(ns, wilson_widths,  label="Wilson",          color="steelblue",   lw=1.8)
    ax.plot(ns, cp_widths,      label="Clopper-Pearson", color="seagreen",    lw=1.8, linestyle="--")
    ax.set_title(f"CI width vs n  (obs acc = {obs_acc_fixed:.0%})", fontsize=12)
    ax.set_xlabel("Number of test trials  (n)", fontsize=11)
    ax.set_ylabel("95 % CI width", fontsize=11)
    ax.legend(fontsize=10)
    ax.axvline(20,  color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.axvline(100, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.text(21, max(wald_widths) * 0.95, "n=20",  fontsize=8, color="gray")
    ax.text(101, max(wald_widths) * 0.95, "n=100", fontsize=8, color="gray")
    ax.set_xlim(5, 300)
    ax.set_ylim(0, None)

fig.suptitle("Width of 95 % confidence interval on classification accuracy\n"
             "Wald breaks for extreme p (right panel); Wilson ≈ Clopper-Pearson",
             fontsize=11)
fig.tight_layout()
plt.savefig("/tmp/dd_chance_ci_width.png", dpi=110, bbox_inches="tight")
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# Key observations:
# * **Wald** produces a narrower (overconfident) interval for p_hat = 0.90 at
#   small n, because the normal approximation assumes symmetric tails even when
#   the distribution is right-skewed.
# * **Wilson** and **Clopper-Pearson** track each other closely and stay properly
#   bounded.  Use Wilson in practice (closed-form, well-calibrated).
# * A 95 % CI on accuracy with only 20 trials is roughly ±20 percentage points
#   wide — enormous.

# %% [markdown]
# ---
# ## Part 3 — The Real Significance Threshold: not just 1/k
#
# A common mistake is to declare "above chance" whenever accuracy > 1/n_classes.
# But **at small n**, even a pure chance classifier will exceed 1/k often enough
# to mislead.
#
# The correct question: what is the **minimum observed accuracy** required so that
# the one-sided binomial test rejects H0 at significance level α?
#
# This is the smallest $\hat{p}$ such that
# $$P(K \ge k \mid n, p_\text{chance}) < \alpha,$$
# i.e., the smallest k such that the p-value drops below α, divided by n.
#
# Mueller-Putz et al. (2008)* introduced this idea in the BCI literature to
# provide a simple table that reviewers and practitioners could apply directly.
#
# *Müller-Putz G. R. et al., "Evaluating Causal Relations in Neural Systems …",
# Clinical Neurophysiology, 2008.*

# %%
def significance_threshold(n: int, p_chance: float, alpha: float = 0.05) -> float:
    """
    Minimum accuracy (fraction) an observed result must reach to be
    statistically significant (one-sided binomial test, level alpha).

    Returns the threshold as a proportion in [0, 1].  If even n/n does not
    reach significance (can happen for very small n), returns 1.0.
    """
    for k in range(1, n + 1):
        pval = stats.binomtest(k, n, p_chance, alternative="greater").pvalue
        if pval < alpha:
            return k / n
    return 1.0  # no threshold found at this n


# --- Tabulate thresholds for common (n, k_classes) combinations ---
ns_table = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
k_classes_list = [2, 3, 4]

print("Minimum accuracy to be significant at alpha=0.05 (binomial test)")
print(f"{'n':>5}  " + "  ".join(f"{'k='+str(k):>9}" for k in k_classes_list))
print("-" * 38)

# Store for the plots below
thresh_k2 = []
thresh_k4 = []

for n in ns_table:
    row_vals = []
    for kc in k_classes_list:
        t = significance_threshold(n, 1.0 / kc)
        row_vals.append(t)
    thresh_k2.append(significance_threshold(n, 0.5))
    thresh_k4.append(significance_threshold(n, 0.25))
    print(f"{n:>5}  " + "  ".join(f"{v:>9.1%}" for v in row_vals))

# Pull out the two specific values requested
n20_k2  = significance_threshold(20,  0.5)
n100_k2 = significance_threshold(100, 0.5)
print(f"\n=> Threshold for n=20,  k=2 (chance=0.50): {n20_k2:.4f}  ({n20_k2:.1%})")
print(f"=> Threshold for n=100, k=2 (chance=0.50): {n100_k2:.4f} ({n100_k2:.1%})")

# %% [markdown]
# ### Visualisation 2 — Significance threshold vs n for k=2 and k=4
#
# The dashed horizontal lines mark the naive 1/k "chance" levels.  The
# **solid curves show the actual accuracy you need** to be significant at α=0.05.
# For small n, this can be dramatically higher than 1/k.

# %%
ns_plot = np.arange(5, 501)
thresh_k2_plot = [significance_threshold(n, 0.5)  for n in ns_plot]
thresh_k4_plot = [significance_threshold(n, 0.25) for n in ns_plot]

fig, ax = plt.subplots(figsize=(11, 5))

ax.step(ns_plot, thresh_k2_plot, where="post",
        label="k=2  (required threshold)", color="steelblue",   lw=2.2)
ax.step(ns_plot, thresh_k4_plot, where="post",
        label="k=4  (required threshold)", color="darkorange",  lw=2.2)

ax.axhline(0.50, color="steelblue", lw=1.2, linestyle="--", alpha=0.6, label="1/2 naive chance (k=2)")
ax.axhline(0.25, color="darkorange", lw=1.2, linestyle="--", alpha=0.6, label="1/4 naive chance (k=4)")

# annotate the n=20, k=2 and n=100, k=2 points
ax.scatter([20, 100], [n20_k2, n100_k2], zorder=5, color="steelblue", s=60)
ax.annotate(f"n=20: need ≥{n20_k2:.0%}", xy=(20, n20_k2),
            xytext=(35, n20_k2 + 0.06), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1.0))
ax.annotate(f"n=100: need ≥{n100_k2:.0%}", xy=(100, n100_k2),
            xytext=(130, n100_k2 + 0.04), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1.0))

ax.set_xlabel("Number of test trials  (n)", fontsize=12)
ax.set_ylabel("Minimum accuracy for p < 0.05", fontsize=12)
ax.set_title("Significance threshold vs n  (one-sided binomial test, α = 0.05)\n"
             "After Müller-Putz et al. 2008 — the threshold converges to 1/k only slowly",
             fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(5, 500)
ax.set_ylim(0.2, 1.05)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("/tmp/dd_chance_threshold.png", dpi=110, bbox_inches="tight")
plt.show()
print("Figure 2 saved.")

# %% [markdown]
# The step-function shape arises because k (number correct) is discrete.
# For n = 20 and a 2-class task you need **≥ 75 % accuracy** to be significant
# at α = 0.05 — far above the naive 50 % chance level.  With n = 100 the bar
# drops to 59 %, and it only approaches 50 % as n → ∞.

# %% [markdown]
# ---
# ## Part 4 — Practical Reporting Checklist
#
# Formalising the guidance from Chapter 11 with the math now in hand:
#
# | What to always report | Why it matters |
# |---|---|
# | n (number of test trials) | Determines significance threshold and CI width |
# | Chance level 1/k_classes | Anchors the null; never assume the reader knows k |
# | Observed accuracy (point estimate) | The raw number |
# | 95 % CI (Wilson or Clopper-Pearson) | Shows uncertainty; lower CI > chance = clear evidence |
# | Binomial test p-value | Exact test, no normal approximation |
# | The significance threshold for your n | Lets the reader apply the Müller-Putz check |
#
# ### Quick demo: the same accuracy can be conclusive or meaningless

# %%
scenarios = [
    dict(label="n=20,  acc=0.75", k=15, n=20,  p_chance=0.5),
    dict(label="n=20,  acc=0.65", k=13, n=20,  p_chance=0.5),
    dict(label="n=100, acc=0.65", k=65, n=100, p_chance=0.5),
    dict(label="n=400, acc=0.58", k=232,n=400, p_chance=0.5),
]

print(f"\n{'Scenario':>22}  {'Acc':>6} {'95% CI':>18}  {'p-value':>9}  {'sig?':>5}  {'thresh':>8}")
print("-" * 80)
for sc in scenarios:
    k, n, pc = sc["k"], sc["n"], sc["p_chance"]
    acc = k / n
    lo, hi = wilson_ci(k, n)
    pval = stats.binomtest(k, n, pc, alternative="greater").pvalue
    thresh = significance_threshold(n, pc)
    sig = "YES" if pval < 0.05 else "no"
    print(f"{sc['label']:>22}  {acc:>6.2%} [{lo:.3f}, {hi:.3f}]  {pval:>9.4f}  {sig:>5}  {thresh:>8.1%}")

# %% [markdown]
# **Key takeaway:** a tiny test set demands a surprisingly high accuracy to be
# significant.  Always report n, always report the CI, and always compare against
# the *true* significance threshold for your n — not just the naive 1/k.

# %% [markdown]
# ---
# ## ⚠️ A subtler trap: effective n is smaller than your trial count
#
# Everything above assumes the n trials are **independent**.  In an EEG
# classification experiment they almost never are.
#
# ### Why EEG trials are autocorrelated
#
# EEG epochs within a session share slow drift, fatigue, vigilance fluctuations,
# impedance drift, and subject state.  The signal in epoch i and epoch i+1 are
# not independent draws from the same distribution.  In practice this means
# the **effective sample size** n_eff < n.
#
# If the intraclass correlation (ICC, ρ) across epochs is ρ, and the average
# cluster size is m epochs per "independent block", then the classic formula
# (Kish 1965) gives:
#
# $$n_\text{eff} = \frac{n}{1 + (m - 1)\rho}$$
#
# ### Consequence for the significance threshold
#
# Plugging n_eff into the binomial test instead of n raises the significance
# threshold.  The trap is that you might compute p < 0.05 using the raw trial
# count, announce a significant BCI result, but the correct calculation using
# n_eff would not be significant.
#
# A concrete example:

# %%
def effective_n(n_total: int, m_cluster: int, rho: float) -> int:
    """
    Kish (1965) effective sample size under intraclass correlation rho
    with average cluster size m.
    """
    n_eff = n_total / (1 + (m_cluster - 1) * rho)
    return max(1, int(np.floor(n_eff)))


print("Effect of intraclass correlation on effective n and significance threshold")
print(f"{'rho':>6}  {'n_raw':>6}  {'n_eff':>6}  {'thresh_raw':>11}  {'thresh_eff':>11}  {'verdict'}")
print("-" * 70)

n_raw   = 100
m_block = 10     # 10 epochs per independent block (e.g. one 2-s run)
obs_acc = 0.62   # 62 correct out of 100
k_obs   = round(obs_acc * n_raw)

for rho in [0.00, 0.05, 0.10, 0.20, 0.35]:
    n_eff   = effective_n(n_raw, m_block, rho)
    thresh_raw = significance_threshold(n_raw, 0.5)
    thresh_eff = significance_threshold(n_eff, 0.5)
    # p-value should be re-derived with n_eff
    k_eff_scaled = round(obs_acc * n_eff)
    pval_eff = stats.binomtest(k_eff_scaled, n_eff, 0.5, alternative="greater").pvalue
    sig_eff = "significant" if pval_eff < 0.05 else "NOT significant"
    print(f"{rho:>6.2f}  {n_raw:>6}  {n_eff:>6}  {thresh_raw:>11.1%}  {thresh_eff:>11.1%}  {sig_eff}")

# %% [markdown]
# Even a modest intraclass correlation of ρ = 0.10 reduces n_eff from 100 to
# about 52, pushing the significance threshold from ~59 % up to ~62 % or higher.
# An observed accuracy of 62 % that looked significant under the naive count is
# no longer significant once the correlation structure is accounted for.
#
# ### The per-fold trap
#
# A related pitfall: some papers report that accuracy was "above chance in 8 out
# of 10 cross-validation folds" and treat each fold independently.  But the folds
# are **not** independent — they share training data and the same subject's brain.
# The correct test is on the **pooled out-of-fold predictions** (a single binomial
# test with n = total test examples), not on per-fold p-values.
#
# ### How to protect yourself
# 1. Estimate ρ from your data (fit a one-way ANOVA per subject, compute ICC).
# 2. Compute n_eff and use it in the binomial test.
# 3. Report the raw n, n_eff, ρ, and the threshold derived from n_eff.
# 4. If you use cross-validation, pool out-of-fold predictions and run one
#    binomial test on the pooled result — never average per-fold p-values.
#
# These corrections are rarely done in published BCI papers, which means many
# "significant" results in the literature are likely false positives at the
# stated α level.
