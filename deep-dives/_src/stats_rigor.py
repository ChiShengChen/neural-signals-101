# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Statistical Rigor for Model Comparison
#
# Formal machinery behind the intuitions introduced in main Chapter 11.
#
# > **Prerequisites:** main Chapter 11.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget.**

# %%
# --- Bootstrap: ensure neuro101 is importable ---
import sys, os
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent.parent / "src"))

import warnings
import numpy as np
import matplotlib
# Headless backend for CI / smoke runs
if os.environ.get("NEURO101_SMOKE") == "1" or os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Suppress non-interactive backend warning from plt.show() in headless mode
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.base import clone

rng = np.random.default_rng(42)

SMOKE = os.environ.get("NEURO101_SMOKE") == "1"
# In smoke mode reduce iterations so CI finishes fast
N_EXPERIMENTS = 300  if SMOKE else 3_000   # outer Monte Carlo repetitions
K_FOLDS       = 5
N_SAMPLES     = 150                         # samples per synthetic dataset
print(f"SMOKE={SMOKE}  N_EXPERIMENTS={N_EXPERIMENTS}  K_FOLDS={K_FOLDS}  N_SAMPLES={N_SAMPLES}")

# %% [markdown]
# ---
# ## Section 1 — Why naive k-fold paired t-tests are anti-conservative
#
# ### The theoretical problem
#
# When comparing two models with k-fold cross-validation, the standard recipe is:
#
# 1. Compute k per-fold differences $d_i = \text{acc}_B^{(i)} - \text{acc}_A^{(i)}$.
# 2. Run a one-sample t-test on $H_0: \mu_d = 0$.  The t-statistic is
#    $t = \bar{d} \,/\, (s_d / \sqrt{k})$.
# 3. Reject $H_0$ if $p < 0.05$.
#
# This is **wrong** because the t-test requires the $d_i$ to be
# **independent**.  In k-fold CV two folds share $(k-2)/(k-1)$ of their training
# examples.  That positive correlation means $s_d^2$ **underestimates** the true
# variance of $\bar{d}$, the denominator shrinks, $|t|$ is inflated, and the test
# rejects far more often than the stated $\alpha = 0.05$.
#
# ### Simulation strategy
#
# To measure the empirical Type-I error we need experiments where the **true
# difference is exactly zero**.  The cleanest way is to simulate fold scores
# directly from a multivariate normal that encodes the known correlation structure
# of k-fold CV, then run the test.
#
# Nadeau & Bengio (2003) show the intra-experiment covariance between fold scores
# under $H_0$ is approximately $\rho \sigma^2$ where
# $\rho = n_{\text{test}} / n_{\text{train}}$ and $\sigma^2$ is the per-fold
# score variance.  We simulate directly from this distribution.

# %%
def simulate_correlated_fold_diffs(k, n_train, n_test, sigma=0.12, rng_s=None):
    """Simulate k CV fold differences under H0 (true mean = 0).

    Fold scores are correlated: any two folds share ~(k-2)/(k-1) of their
    training data, inducing positive covariance rho * sigma^2.
    We model this as a multivariate normal with the Nadeau-Bengio covariance.

    Parameters
    ----------
    k : int        Number of folds.
    n_train : int  Training set size per fold.
    n_test : int   Test set size per fold.
    sigma : float  Standard deviation of a single fold score (typical ~0.10-0.15).
    rng_s : np.random.Generator  Random state.
    """
    if rng_s is None:
        rng_s = np.random.default_rng()
    rho = n_test / n_train          # N&B overlap ratio
    # Covariance matrix: diagonal sigma^2, off-diagonal rho * sigma^2
    cov = sigma**2 * (rho * np.ones((k, k)) + (1 - rho) * np.eye(k))
    mean = np.zeros(k)
    return rng_s.multivariate_normal(mean, cov)


n_test_sim  = N_SAMPLES // K_FOLDS
n_train_sim = N_SAMPLES - n_test_sim
sigma_sim   = 0.12       # typical single-fold accuracy std in BCI settings

print(f"Simulation: k={K_FOLDS}, n_train={n_train_sim}, n_test={n_test_sim}")
print(f"Intra-experiment correlation rho = n_test/n_train = {n_test_sim/n_train_sim:.4f}")

# --- Monte Carlo: count how often each test rejects under H0 ---
print(f"\nRunning {N_EXPERIMENTS} Monte Carlo experiments…")
naive_rejects     = []
corrected_rejects = []
pvals_naive_all   = []
pvals_corr_all    = []

for exp_i in range(N_EXPERIMENTS):
    exp_rng = np.random.default_rng(rng.integers(0, 2**31))
    diffs = simulate_correlated_fold_diffs(
        K_FOLDS, n_train_sim, n_test_sim, sigma=sigma_sim, rng_s=exp_rng
    )

    # --- Naive paired t-test (WRONG) ---
    _, p_naive = stats.ttest_1samp(diffs, popmean=0)
    naive_rejects.append(p_naive < 0.05)
    pvals_naive_all.append(p_naive)

    # --- Nadeau-Bengio corrected variance ---
    n      = len(diffs)
    mean_d = diffs.mean()
    var_d  = diffs.var(ddof=1)
    rho_cv = n_test_sim / n_train_sim
    corrected_var = var_d * (1.0 / n + rho_cv)
    if corrected_var > 0:
        t_corr = mean_d / np.sqrt(corrected_var)
        p_corr = 2 * stats.t.sf(abs(t_corr), df=n - 1)
    else:
        p_corr = 1.0
    corrected_rejects.append(p_corr < 0.05)
    pvals_corr_all.append(p_corr)

naive_type1     = float(np.mean(naive_rejects))
corrected_type1 = float(np.mean(corrected_rejects))
pvals_naive_all = np.array(pvals_naive_all)
pvals_corr_all  = np.array(pvals_corr_all)

print(f"\nNaive paired t-test   Type-I error: {naive_type1:.3f}  (nominal 0.05)")
print(f"Corrected (N&B 2003)  Type-I error: {corrected_type1:.3f}  (nominal 0.05)")
print(f"\nThe naive test rejects {naive_type1/0.05:.1f}x as often as it should.")

# %% [markdown]
# **Reading the result:** both tests see fold differences whose true mean is
# exactly zero.  Every rejection is a false positive.  The naive test fires far
# more often than the nominal 5 % because it ignores the positive correlation
# between fold scores.  The corrected test's rejection rate is much closer to 5 %.

# %%
# --- Visualisation 1: Type-I error bar comparison + p-value histograms ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: bar chart of empirical Type-I error rates
ax = axes[0]
labels = ["Naive\npaired t-test", "Nadeau-Bengio\ncorrected"]
rates  = [naive_type1, corrected_type1]
colors = ["tomato", "steelblue"]
bars = ax.bar(labels, rates, color=colors, width=0.42, edgecolor="black", linewidth=1.2)
ax.axhline(0.05, color="black", linestyle="--", lw=2.5, label="Nominal α = 0.05")
ax.set_ylim(0, max(0.40, naive_type1 * 1.45))
ax.set_ylabel("Empirical Type-I error rate", fontsize=12)
ax.set_title(f"False positive rate when H₀ is TRUE\n"
             f"({N_EXPERIMENTS:,} simulated experiments, {K_FOLDS}-fold CV, "
             f"ρ = n_test/n_train = {n_test_sim}/{n_train_sim})",
             fontsize=10)
ax.legend(fontsize=11)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2,
            rate + 0.006, f"{rate:.3f}",
            ha="center", fontsize=14, fontweight="bold")
ax.text(0.97, 0.93, "← ideal (0.05)", transform=ax.transAxes,
        ha="right", fontsize=10, color="black", style="italic")

# Right: p-value histograms (should be Uniform[0,1] under H0 → flat)
ax2 = axes[1]
bins = np.linspace(0, 1, 21)
exp_count = len(pvals_naive_all) / 20          # expected bar height for Uniform
ax2.hist(pvals_naive_all, bins=bins, alpha=0.68, color="tomato",
         label=f"Naive  (α̂={naive_type1:.3f})", edgecolor="white")
ax2.hist(pvals_corr_all,  bins=bins, alpha=0.68, color="steelblue",
         label=f"Corrected (α̂={corrected_type1:.3f})", edgecolor="white")
ax2.axhline(exp_count, color="black", linestyle="--", lw=2,
            label="Expected (Uniform[0,1])")
ax2.axvline(0.05, color="darkorange", lw=2, linestyle=":", label="α = 0.05")
ax2.set_xlabel("p-value", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("p-value distribution under H₀\n"
              "(ideal = flat histogram)", fontsize=11)
ax2.legend(fontsize=9)

plt.suptitle("Naive k-fold paired t-test is anti-conservative: too many false positives",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fig1_type1_error.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# **Key pattern in the p-value histogram:** under $H_0$ a correctly calibrated
# test produces p-values that are uniformly distributed — a flat histogram.  The
# naive test's histogram is skewed heavily toward small p-values (the left bar is
# far taller than expected), which directly translates into an inflated rejection
# rate.  The corrected histogram is much closer to flat.

# %% [markdown]
# ---
# ## Section 2 — The Nadeau & Bengio (2003) Corrected Resampled t-test
#
# ### The fix: one extra term in the variance
#
# Nadeau & Bengio (2003) derive the correct variance estimator for the mean
# difference $\bar{d}$ across $n$ CV folds:
#
# $$
# \widehat{\mathrm{Var}}[\bar{d}]
#   = \left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2
# $$
#
# where $s_d^2 = \frac{1}{n-1}\sum_i (d_i - \bar{d})^2$ is the ordinary sample
# variance of the fold differences, $n_{\text{test}}$ is the test-fold size, and
# $n_{\text{train}}$ is the training-fold size.
#
# The ratio $\rho = n_{\text{test}} / n_{\text{train}}$ is the **overlap
# correction**.  For standard k-fold CV $\rho = 1/(k-1)$, so the corrected
# denominator is larger and the t-statistic is appropriately deflated.
#
# $$
# t = \frac{\bar{d}}{\sqrt{\left(\frac{1}{n} + \rho\right) s_d^2}}
#   \;\overset{\text{approx}}{\sim}\; t_{n-1}
# $$
#
# **References:**
# * Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
#   *Machine Learning*, 52(3), 239–281.
# * Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of
#   significance tests for comparing learning algorithms.
#   *PAKDD 2004, LNAI 3056*, 3–12.
# * Dietterich, T. G. (1998). Approximate statistical tests for comparing
#   supervised classification learning algorithms. *Neural Computation*, 10(7).

# %%
def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau-Bengio (2003) variance-corrected paired t-test for CV fold differences.

    Parameters
    ----------
    diffs : array-like, shape (n_folds,)
        Per-fold score differences: score_B[i] - score_A[i].
    n_train : int or float
        Number of training examples per fold (average for unequal folds).
    n_test : int or float
        Number of test examples per fold (average for unequal folds).

    Returns
    -------
    t_stat : float
        Corrected t-statistic.
    p_value : float
        Two-tailed p-value against t_{n-1}.
    ci_95 : tuple of (float, float)
        95 % confidence interval on the true mean difference.

    Notes
    -----
    The correction adds rho = n_test/n_train to the usual 1/n term, accounting
    for the positive correlation between fold scores induced by overlapping
    training sets.

    Reference: Nadeau & Bengio (2003), Machine Learning 52(3).
    """
    diffs   = np.asarray(diffs, float)
    n       = len(diffs)
    mean_d  = diffs.mean()
    var_d   = diffs.var(ddof=1)
    rho     = n_test / n_train                        # overlap ratio
    corrected_var = var_d * (1.0 / n + rho)           # N&B eq. (5)
    if corrected_var <= 0:
        return 0.0, 1.0, (mean_d, mean_d)
    se      = np.sqrt(corrected_var)
    t_stat  = mean_d / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    t_crit  = stats.t.ppf(0.975, df=n - 1)
    ci_95   = (mean_d - t_crit * se, mean_d + t_crit * se)
    return t_stat, p_value, ci_95


# --- Worked example: two real models on synthetic data ---
rng_demo = np.random.default_rng(99)
n_samples_demo = 200
X_demo, y_demo = make_classification(
    n_samples=n_samples_demo, n_features=20, n_informative=8,
    n_redundant=4, random_state=7
)

pipe_A = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=1.0,  max_iter=300, random_state=0))])
pipe_B = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=0.01, max_iter=300, random_state=0))])

kf_demo  = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
scores_A = cross_val_score(pipe_A, X_demo, y_demo, cv=kf_demo, scoring="accuracy")
scores_B = cross_val_score(pipe_B, X_demo, y_demo, cv=kf_demo, scoring="accuracy")
fold_diffs_demo = scores_B - scores_A

n_test_demo  = n_samples_demo // K_FOLDS
n_train_demo = n_samples_demo - n_test_demo

_, p_naive_demo = stats.ttest_1samp(fold_diffs_demo, popmean=0)
t_corr_demo, p_corr_demo, ci_corr_demo = corrected_resampled_ttest(
    fold_diffs_demo, n_train_demo, n_test_demo
)

print("Worked example: LR C=1.0  vs  LR C=0.01  (same dataset, same folds)")
print(f"  Per-fold differences: {np.round(fold_diffs_demo, 3)}")
print(f"  Mean diff (B - A):   {fold_diffs_demo.mean():.4f}")
print(f"  Naive  t-test:        p = {p_naive_demo:.4f}")
print(f"  Corrected (N&B):      p = {p_corr_demo:.4f}   t = {t_corr_demo:.3f}")
print(f"  95 % CI on diff:      [{ci_corr_demo[0]:.4f}, {ci_corr_demo[1]:.4f}]")
print(f"  rho = {n_test_demo}/{n_train_demo} = {n_test_demo/n_train_demo:.4f}")

# %% [markdown]
# The corrected p-value is larger (more conservative) than the naive one.
# The 95 % confidence interval tells you the **range of plausible true
# differences** — often more informative than the p-value because it shows
# whether any practically meaningful advantage is excluded.

# %% [markdown]
# ---
# ## Section 3 — Nested Cross-validation for Hyperparameter Selection
#
# ### The optimism problem
#
# A common workflow is:
# 1. Run k-fold CV, sweeping many hyperparameters.
# 2. Pick the best configuration.
# 3. **Report the CV score of that best configuration.**
#
# This is wrong.  The CV score used to *select* the configuration is
# **contaminated**: you peeked at it to make your choice, so it is optimistically
# biased.  The model that "won" may simply have been lucky on those folds.
#
# **Nested CV** separates model selection from model evaluation:
#
# ```
# ┌──────────────────────── outer fold k ──────────────────────────┐
# │  outer train (k-1 folds)          │   outer test fold k        │
# │  ┌── inner CV (grid search) ──┐   │   ^ held out entirely      │
# │  │  find best hyperparams    │   │   | score reported here     │
# │  └───────────────────────────┘   │   |                         │
# │  refit best config on all train ──────┘                         │
# └─────────────────────────────────────────────────────────────────┘
# ```
#
# The outer test fold is **never** seen during hyperparameter selection, so
# the outer-fold score is an honest estimate of generalisation performance.

# %%
# --- Synthetic dataset ---
N_SYNTH      = 300
N_FEATURES   = 30
N_INFORMATIVE = 6

X_nest, y_nest = make_classification(
    n_samples=N_SYNTH, n_features=N_FEATURES,
    n_informative=N_INFORMATIVE, n_redundant=8,
    random_state=17
)

param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
pipe_base  = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=500, random_state=0))])

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# --- Non-nested (WRONG): select on the same CV you later report ---
gs_nonnested = GridSearchCV(pipe_base, param_grid, cv=outer_cv,
                            scoring="accuracy", refit=True, n_jobs=1)
gs_nonnested.fit(X_nest, y_nest)
score_nonnested  = gs_nonnested.best_score_     # optimistically biased

# --- Nested CV (CORRECT) ---
nested_scores = []
best_Cs       = []
for train_idx, test_idx in outer_cv.split(X_nest):
    X_tr, X_te = X_nest[train_idx], X_nest[test_idx]
    y_tr, y_te = y_nest[train_idx], y_nest[test_idx]
    gs_inner = GridSearchCV(pipe_base, param_grid, cv=inner_cv,
                            scoring="accuracy", refit=True, n_jobs=1)
    gs_inner.fit(X_tr, y_tr)
    best_Cs.append(gs_inner.best_params_["clf__C"])
    nested_scores.append(gs_inner.score(X_te, y_te))

nested_scores     = np.array(nested_scores)
score_nested_mean = nested_scores.mean()
score_nested_std  = nested_scores.std(ddof=1)

print("Single-dataset comparison:")
print(f"  Non-nested (biased):  {score_nonnested:.4f}  <-- selection leaked into estimate")
print(f"  Nested CV:  {score_nested_mean:.4f} ± {score_nested_std:.4f}  <-- honest")
print(f"  Bias:       {score_nonnested - score_nested_mean:+.4f}")
print(f"  Best C per outer fold: {best_Cs}")

# %%
# --- Bootstrap over many random datasets to characterise the bias distribution ---
N_BOOT_NEST = 80 if SMOKE else 250
boot_rng       = np.random.default_rng(55)
boot_nonnested = []
boot_nested    = []

for _ in range(N_BOOT_NEST):
    seed_i = int(boot_rng.integers(0, 2**31))
    Xb, yb = make_classification(
        n_samples=N_SYNTH, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, n_redundant=8,
        random_state=seed_i
    )
    cv_b       = KFold(n_splits=5, shuffle=True, random_state=seed_i)
    cv_inner_b = KFold(n_splits=3, shuffle=True, random_state=0)

    # Non-nested
    gs_nn = GridSearchCV(pipe_base, param_grid, cv=cv_b,
                         scoring="accuracy", refit=True, n_jobs=1)
    gs_nn.fit(Xb, yb)
    boot_nonnested.append(gs_nn.best_score_)

    # Nested
    fold_sc = []
    for tr, te in cv_b.split(Xb):
        gs_in = GridSearchCV(pipe_base, param_grid, cv=cv_inner_b,
                             scoring="accuracy", refit=True, n_jobs=1)
        gs_in.fit(Xb[tr], yb[tr])
        fold_sc.append(gs_in.score(Xb[te], yb[te]))
    boot_nested.append(float(np.mean(fold_sc)))

boot_nonnested = np.array(boot_nonnested)
boot_nested    = np.array(boot_nested)
bias_per_run   = boot_nonnested - boot_nested

# --- Visualisation 2: non-nested vs nested distributions ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
lo = min(boot_nonnested.min(), boot_nested.min()) - 0.01
hi = max(boot_nonnested.max(), boot_nested.max()) + 0.01
bins_nest = np.linspace(lo, hi, 26)
ax.hist(boot_nonnested, bins=bins_nest, alpha=0.70, color="tomato",
        label=f"Non-nested  mean={boot_nonnested.mean():.3f}", edgecolor="white")
ax.hist(boot_nested,    bins=bins_nest, alpha=0.70, color="steelblue",
        label=f"Nested CV   mean={boot_nested.mean():.3f}",    edgecolor="white")
ax.axvline(boot_nonnested.mean(), color="tomato",    lw=2.5, linestyle="--")
ax.axvline(boot_nested.mean(),    color="steelblue", lw=2.5, linestyle="--")
ax.set_xlabel("Reported accuracy", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Score distributions over {N_BOOT_NEST} random datasets\n"
             f"Mean bias = {bias_per_run.mean():.4f}", fontsize=11)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.hist(bias_per_run, bins=20, color="darkorchid", edgecolor="white", alpha=0.85)
ax2.axvline(0, color="black", lw=1.5, linestyle="--", label="Zero bias")
ax2.axvline(bias_per_run.mean(), color="crimson", lw=2.5,
            label=f"Mean bias = {bias_per_run.mean():.4f}")
pct_pos = (bias_per_run > 0).mean()
ax2.set_xlabel("Non-nested minus nested score", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title(f"Optimism bias per experiment\n"
              f"{pct_pos:.0%} of runs: non-nested > nested", fontsize=11)
ax2.legend(fontsize=10)

plt.suptitle("Non-nested CV is optimistically biased; nested CV gives honest estimates",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fig2_nested_vs_nonnested.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 2 saved.")
print(f"\nSummary across {N_BOOT_NEST} datasets:")
print(f"  Non-nested mean ± std:  {boot_nonnested.mean():.4f} ± {boot_nonnested.std():.4f}")
print(f"  Nested CV  mean ± std:  {boot_nested.mean():.4f} ± {boot_nested.std():.4f}")
print(f"  Mean optimism bias:     {bias_per_run.mean():+.4f}")
print(f"  Fraction: non-nested > nested:  {pct_pos:.1%}")

# %% [markdown]
# The non-nested distribution (red) is consistently shifted right relative to
# the nested distribution (blue).  The optimism bias is systematic: in the
# majority of experiments the non-nested score overstates what the model will
# actually achieve on truly new data.

# %% [markdown]
# ---
# ## Section 4 — Practical Recommendations
#
# | Recommendation | Rationale |
# |---|---|
# | Use the **Nadeau-Bengio corrected t-test** | Naive test is anti-conservative; NB correction accounts for training-set overlap |
# | Report **effect size** (Cohen's d) and **95 % CI** on the difference | p-values say nothing about magnitude; a tiny but "significant" improvement may be irrelevant in practice |
# | Use **nested CV** whenever hyperparameters are tuned | Non-nested CV leaks selection information into the reported score |
# | Correct for **multiple comparisons** when many configurations are tried | With 20 configs and α = 0.05, ~1 false positive is expected even under the global null |
# | Use **subjects as the unit of replication**, not folds | Fold scores within a subject are correlated; see the subtler trap below |

# %%
# --- Effect size: Cohen's d on paired fold differences ---
def cohens_d_paired(diffs):
    """Cohen's d for paired differences (effect size = mean/std)."""
    return diffs.mean() / diffs.std(ddof=1)

d_demo = cohens_d_paired(fold_diffs_demo)
magnitude = ("negligible" if abs(d_demo) < 0.2 else
             "small"      if abs(d_demo) < 0.5 else
             "medium"     if abs(d_demo) < 0.8 else "large")
print("Effect size for the worked example (LR C=1.0 vs C=0.01):")
print(f"  Cohen's d = {d_demo:.3f}  ({magnitude})")
print(f"  Mean diff = {fold_diffs_demo.mean():.4f}")
print(f"  Corrected 95% CI: [{ci_corr_demo[0]:.4f}, {ci_corr_demo[1]:.4f}]")
ci_contains_zero = ci_corr_demo[0] < 0 < ci_corr_demo[1]
print(f"  CI contains zero: {ci_contains_zero}  "
      f"{'→ cannot reject H0 at 5%' if ci_contains_zero else '→ reject H0'}")

# --- Multiple comparisons: FWER inflation ---
k_configs = np.arange(1, 21)
fwer = 1 - (1 - 0.05) ** k_configs

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_configs, fwer, "o-", color="tomato", lw=2.5, markersize=6,
        label="FWER = 1 − 0.95^k")
ax.axhline(0.05, color="black", linestyle="--", lw=1.5, label="Target α = 0.05")
ax.axhline(fwer[9], color="darkorange", linestyle=":", lw=1.5,
           label=f"k=10: FWER ≈ {fwer[9]:.2f}")
ax.fill_between(k_configs, fwer, 0.05, alpha=0.18, color="tomato")
ax.set_xlabel("Number of configurations tested", fontsize=12)
ax.set_ylabel("Family-wise error rate", fontsize=12)
ax.set_title("Testing more configs inflates the false-positive rate\n"
             "(Bonferroni threshold = 0.05 / k)", fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("/tmp/dd_fig3_fwer.png", dpi=120, bbox_inches="tight")
plt.show()

# Holm-Bonferroni on toy p-values
toy_pvals = np.array([0.001, 0.04, 0.03, 0.07, 0.002, 0.80, 0.15, 0.045])
order     = np.argsort(toy_pvals)
n_tests   = len(toy_pvals)
holm_thr  = 0.05 / (n_tests - np.arange(n_tests))
print("\nHolm-Bonferroni correction on 8 toy p-values:")
for rank, (idx, thr) in enumerate(zip(order, holm_thr)):
    rej = toy_pvals[idx] <= thr
    print(f"  rank {rank+1}: p={toy_pvals[idx]:.3f}  Holm threshold={thr:.4f}  "
          f"{'REJECT' if rej else 'retain'}")

# %% [markdown]
# ---
# %% [markdown]
# ## ⚠️ A subtler trap: the garden of forking paths and subject-level exchangeability
#
# ### The garden of forking paths
#
# Everything above assumes you run a pre-specified set of comparisons and apply a
# multiple-comparisons correction.  There is a deeper problem that no correction
# fully fixes: the **garden of forking paths** (Gelman & Loken, 2014).
#
# In a typical BCI study a researcher might try:
# * 3 preprocessing pipelines (bandpass ranges, artifact rejection thresholds)
# * 4 feature methods (CSP, Riemannian, band power, CNN)
# * 5 classifiers (LDA, SVM-RBF, SVM-linear, LR, EEGNet)
# * 3 CV strategies (k-fold, LOSO, block)
#
# That is $3 \times 4 \times 5 \times 3 = 180$ combinations.  If the researcher
# reports only the combination with the smallest p-value, even a Bonferroni
# correction for 180 tests is **invalid** — because the space of combinations
# tried was not pre-registered.  The researcher may not consciously test all 180;
# incremental intuitions, discarded "non-working" configs, and survivor
# presentation achieve the same effect without deliberate cheating.  This is the
# ordinary scientific process, and it invalidates frequentist guarantees.
#
# **The corrected t-test and nested CV protect you from the comparisons you
# declare.  They do not protect you from the comparisons you forgot you made.**
#
# ### Subject-level exchangeability (even more subtle)
#
# Even with a scrupulous multiple-comparisons correction and the corrected
# t-test, one structural assumption hides in the fine print:
# **exchangeability of the units used in the test**.
#
# When you run k-fold CV within a multi-session dataset from a single subject,
# the test treats k fold scores as $k$ near-independent observations.  In a
# **multi-subject** study the true unit of replication is the **subject**, not the
# fold.
#
# Concretely: 9 subjects × 5 folds = 45 fold scores.  Running the corrected
# t-test on all 45 values treats them as 45 nearly-independent observations.
# They are not: the 5 folds from subject 1 share the same brain, same electrode
# placement, same recording session, and the same idiosyncratic noise sources.
# Their effective degrees of freedom is closer to 9 than to 45.
#
# **What to do:** aggregate to subject level first (mean accuracy per subject),
# then run a paired test on those 9 values.  Better yet, model subject as a
# random effect.  The corrected t-test is "correct" only relative to the
# exchangeability structure you feed it — getting that structure right is
# the analyst's responsibility, and no formula can rescue a mismatch between
# statistical model and data-generating process.

# %%
# --- Illustration: naively pooling 45 fold scores vs 9 subject means ---
rng_subj = np.random.default_rng(13)

n_subjects_illus  = 9
n_folds_per_subj  = 5
# Each subject has a true effect drawn from N(0, 0.04) — global null is true
subject_true_diff = rng_subj.normal(0.0, 0.04, size=n_subjects_illus)

fold_diffs_flat    = []
subject_mean_diffs = []
for s in range(n_subjects_illus):
    # Within-subject folds: correlated via shared subject intercept
    folds_s = subject_true_diff[s] + rng_subj.normal(0, 0.02, size=n_folds_per_subj)
    fold_diffs_flat.extend(folds_s.tolist())
    subject_mean_diffs.append(folds_s.mean())

fold_diffs_flat    = np.array(fold_diffs_flat)
subject_mean_diffs = np.array(subject_mean_diffs)

n_train_illus = 80
n_test_illus  = 20

# Test 1: naive t on all 45 fold scores — WRONG (inflated df)
_, p_naive_45 = stats.ttest_1samp(fold_diffs_flat, popmean=0)

# Test 2: N&B corrected on all 45 fold scores — still WRONG (wrong unit)
t_nb45, p_nb45, ci_nb45 = corrected_resampled_ttest(
    fold_diffs_flat, n_train_illus, n_test_illus
)

# Test 3: correct — aggregate to 9 subject means first
_, p_subj = stats.ttest_1samp(subject_mean_diffs, popmean=0)
t_nb9, p_nb9, ci_nb9 = corrected_resampled_ttest(
    subject_mean_diffs, n_train_illus, n_test_illus
)

print("Exchangeability check: 9 subjects × 5 folds, true mean diff ≈ 0")
print(f"  True subject-level diffs: {np.round(subject_true_diff, 3)}")
print(f"  True grand mean diff:     {subject_true_diff.mean():.4f}")
print()
print(f"  WRONG  — naive t on 45 fold scores:       p = {p_naive_45:.4f}  (df=44, n=45)")
print(f"  WRONG  — N&B corrected on 45 fold scores: p = {p_nb45:.4f}  (df=44, n=45)")
print(f"  CORRECT— N&B corrected on  9 subj means:  p = {p_nb9:.4f}  (df=8,  n=9)")
print()
print("The correct approach uses n=9 subject-level observations.")
print("More observations ≠ more subjects; the formula cannot fix the wrong unit.")
