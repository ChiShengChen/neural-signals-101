# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Riemannian Methods on Small Data
#
# Why covariance-based pipelines stay competitive when you only have a handful of
# labelled EEG trials — and what the SPD manifold has to do with it.
#
# > **Prerequisites:** main Chapter 07.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget.**

# %% Bootstrap — import neuro101 from repo src if not installed as a package
import sys
import os
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mne.decoding import CSP

from neuro101 import io, datasets as ds
from neuro101.eval import make_block_split

rng = np.random.default_rng(42)
SMOKE = ds.is_smoke()
print(f"Smoke mode: {SMOKE}")

# %% [markdown]
# ## Part 1 — Every trial lives on a curved manifold
#
# ### From raw epochs to covariance matrices
#
# Given a trial **X** of shape *(channels, time)*, the **sample covariance
# matrix** is:
#
# $$
# \mathbf{C} = \frac{1}{T-1}\,\mathbf{X}\,\mathbf{X}^\top \;\in\; \mathbb{R}^{p \times p}
# $$
#
# where $p$ is the number of channels and $T$ the number of time samples.
# This single $p\!\times\!p$ matrix encodes **all pairwise channel
# co-activations** for that trial — a rich, compact fingerprint.
#
# ### Why covariances are SPD
#
# $\mathbf{C}$ is always:
#
# * **Symmetric** ($\mathbf{C}^\top = \mathbf{C}$) — trivially, because
#   $\text{Cov}(i,j) = \text{Cov}(j,i)$.
# * **Positive semi-definite** — $\mathbf{v}^\top\mathbf{C}\mathbf{v} \ge 0$
#   for any vector **v** (it measures variance of the projection, which can't
#   be negative).
# * **Strictly positive definite** (SPD) when the channels are not perfectly
#   linearly dependent — which holds after any reasonable whitening or OAS
#   regularisation.
#
# ### SPD matrices form a Riemannian manifold — not flat Euclidean space
#
# The set of all $p\!\times\!p$ SPD matrices, denoted $\text{Sym}^+_p$,
# is **not** a vector subspace: the Euclidean midpoint of two SPD matrices
# is SPD, but averaging or interpolating naïvely in the "matrix entries"
# sense violates the geometry that the space actually has.
#
# Concretely, $\text{Sym}^+_p$ is a **smooth Riemannian manifold** — a
# curved surface embedded in the space of symmetric matrices. Its natural
# ("affine-invariant") distance between two matrices $\mathbf{A}$ and
# $\mathbf{B}$ is:
#
# $$
# d_R(\mathbf{A}, \mathbf{B})
# = \left\|\log\!\left(\mathbf{A}^{-1/2}\,\mathbf{B}\,\mathbf{A}^{-1/2}\right)\right\|_F
# = \sqrt{\sum_i \log^2\!\lambda_i}
# $$
#
# where $\lambda_i$ are the joint eigenvalues of $\mathbf{A}$ and $\mathbf{B}$.
# This distance is invariant to congruence transforms ($\mathbf{C} \mapsto
# \mathbf{W}\mathbf{C}\mathbf{W}^\top$), which means it doesn't care about
# the physical scale of the signal — only about its *shape*.

# %% [markdown]
# ## Part 2 — Visual intuition: Euclidean vs. Riemannian mean (the swelling effect)
#
# We illustrate the key pathology with **2×2** SPD matrices, where the manifold
# can be drawn and the determinant has a clear geometric meaning: it equals the
# area of the ellipse whose axes are the square roots of the eigenvalues.
#
# The **swelling effect**: the Euclidean (entry-wise) mean of a set of SPD
# matrices always has a determinant **larger than or equal to** the Riemannian
# (geometric) mean. Averaging in flat space "over-inflates" the ellipse. The
# Riemannian mean stays on the manifold and avoids this.

# %%
def make_spd_2x2(angle_deg: float, lam1: float, lam2: float) -> np.ndarray:
    """Build a 2x2 SPD matrix with given eigenvalues and principal direction."""
    theta = np.deg2rad(angle_deg)
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return Q @ np.diag([lam1, lam2]) @ Q.T


def draw_ellipse(ax, cov: np.ndarray, center=(0, 0), n_std=1.5,
                 color="steelblue", alpha=0.25, lw=2, label=None):
    """Overlay the ellipse corresponding to a 2x2 covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                  color=color, alpha=alpha, lw=lw, fill=True,
                  edgecolor=color, linewidth=lw, zorder=3, label=label)
    ax.add_patch(ell)


# Build a set of 2x2 SPD matrices spread around the "identity direction"
angles   = np.linspace(-50, 50, 7)    # principal axes tilted between -50° and +50°
lam_vals = np.linspace(1.5, 4.0, 7)   # eigenvalue spread varies across matrices

mats = np.stack([make_spd_2x2(a, l, 0.4) for a, l in zip(angles, lam_vals)])

# Euclidean mean (flat average of matrix entries)
euclid_mean = mats.mean(axis=0)

# Riemannian mean (minimises sum of squared affine-invariant distances)
riemann_mean = mean_riemann(mats)

det_euc = np.linalg.det(euclid_mean)
det_rie = np.linalg.det(riemann_mean)

print(f"Determinant of Euclidean mean  : {det_euc:.4f}")
print(f"Determinant of Riemannian mean : {det_rie:.4f}")
print(f"Swelling ratio (Euc / Rie)     : {det_euc / det_rie:.3f}  "
      f"  <- Euclidean is {100*(det_euc/det_rie-1):.1f}% larger")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(mats)))

for ax, title, mean_cov, mean_color, mean_label in [
    (axes[0], "Euclidean mean  (flat average)",
     euclid_mean, "#e74c3c", f"Euclidean mean\n(det={det_euc:.2f})"),
    (axes[1], "Riemannian mean  (geometric / geodesic)",
     riemann_mean, "#2ecc71", f"Riemannian mean\n(det={det_rie:.2f})"),
]:
    for i, m in enumerate(mats):
        draw_ellipse(ax, m, color=palette[i], alpha=0.18, lw=1.2)
    draw_ellipse(ax, mean_cov, color=mean_color, alpha=0.65, lw=3,
                 label=mean_label)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal"); ax.axhline(0, lw=0.5, c="k"); ax.axvline(0, lw=0.5, c="k")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(loc="upper right", fontsize=9)

fig.suptitle(
    "Swelling effect: Euclidean mean over-inflates the ellipse\n"
    "(semi-transparent ellipses = individual matrices; filled = mean)",
    fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_swelling.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure saved.")

# %% [markdown]
# ### Reading the figure
#
# Each semi-transparent ellipse is one 2×2 SPD matrix visualised as its
# "confidence ellipse". The bold filled ellipse is the mean.
#
# * **Left panel:** The Euclidean mean ellipse is noticeably *larger* than any
#   individual ellipse — it has "swollen". Its determinant is
#   $\approx {det_euc:.2f}$, well above the constituents.
# * **Right panel:** The Riemannian mean ellipse sits *inside* the cloud,
#   properly centred on the manifold. Its determinant is
#   $\approx {det_rie:.2f}$.
#
# **Why this matters for EEG:** if you Euclidean-average covariance matrices
# across trials to estimate a "reference" or to whiten your data, you are
# operating with a systematically inflated matrix — your distances and
# projections will be wrong. The Riemannian mean uses the geodesic (curved)
# path between matrices, staying on the SPD manifold throughout.
#
# **Formal guarantee (Ando–Li–Mathias, 2004):**
#
# $$
# \det\!\left(\frac{\mathbf{A}+\mathbf{B}}{2}\right)
# \;\ge\;
# \det\!\left(\mathbf{A} \#\mathbf{B}\right)
# $$
#
# where $\mathbf{A}\#\mathbf{B} = \mathbf{A}^{1/2}
# (\mathbf{A}^{-1/2}\mathbf{B}\mathbf{A}^{-1/2})^{1/2}\mathbf{A}^{1/2}$
# is the geometric mean. Equality holds iff $\mathbf{A}=\mathbf{B}$.

# %%
# Verify swelling holds across many random SPD sets
swelling_ratios = []
for _ in range(200):
    Z = rng.standard_normal((8, 3, 3))
    S = np.einsum("nij,nkj->nik", Z, Z) + np.eye(3)[None] * 0.5
    eu = S.mean(0)
    ri = mean_riemann(S)
    swelling_ratios.append(np.linalg.det(eu) / np.linalg.det(ri))

swelling_ratios = np.array(swelling_ratios)
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(swelling_ratios, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
ax.axvline(1.0, color="red", lw=2, linestyle="--", label="ratio = 1 (no swelling)")
ax.set(xlabel="det(Euclidean mean) / det(Riemannian mean)",
       ylabel="Count (200 random trials)",
       title="Swelling is always ≥ 1 — Euclidean mean always over-inflates det")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/dd_rie_swelling_hist.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Swelling ratio: min={swelling_ratios.min():.3f} "
      f"mean={swelling_ratios.mean():.3f} "
      f"max={swelling_ratios.max():.3f}  (always >= 1)")

# %% [markdown]
# ## Part 3 — The tangent-space trick
#
# Operating on the manifold directly is expensive (every mean computation is
# iterative). The **tangent-space projection** gives us the best of both worlds:
# the correct curved geometry *and* cheap flat arithmetic for the classifier.
#
# ### How it works
#
# 1. **Compute the Riemannian mean** $\mathbf{M}$ of the *training* covariance
#    matrices (this is the "reference point" on the manifold).
# 2. **Map each trial to the tangent space** at $\mathbf{M}$ via the matrix
#    logarithm:
#
#    $$
#    \mathbf{S}_i = \text{Log}_\mathbf{M}(\mathbf{C}_i)
#    = \mathbf{M}^{1/2}
#      \log\!\left(\mathbf{M}^{-1/2}\mathbf{C}_i\mathbf{M}^{-1/2}\right)
#      \mathbf{M}^{1/2}
#    $$
#
#    $\mathbf{S}_i$ is a **symmetric** matrix (not necessarily positive
#    definite) — we are now in flat space.
# 3. **Vectorise** $\mathbf{S}_i$ (upper triangle, off-diagonals scaled by
#    $\sqrt{2}$ to preserve the Frobenius norm) into a feature vector.
# 4. **Fit any linear classifier** on those vectors.
#
# The intuition: near its reference point, a smooth manifold looks flat —
# just like a map of a city ignores the Earth's curvature over small distances.
# The tangent space gives us that local flat map, centred at the geometric
# mean of the training set.
#
# **Critically, the reference point $\mathbf{M}$ must be fit on the training
# fold only** — projecting test-set covariances onto a reference that was
# estimated using the test set is leakage. `pyriemann.TangentSpace` handles
# this automatically inside a sklearn `Pipeline`.

# %%
# Tiny synthetic demo of the tangent-space projection
from pyriemann.utils.tangentspace import tangent_space, untangent_space  # noqa

# Generate 20 random 4x4 SPD matrices (simulated covariances)
Z = rng.standard_normal((20, 4, 10))
C_demo = np.einsum("nij,nkj->nik", Z, Z) + np.eye(4)[None] * 0.3

# Compute the Riemannian mean
M_demo = mean_riemann(C_demo)

# Project to tangent space at M_demo
S_demo = tangent_space(C_demo, M_demo)
print(f"Original SPD shape : {C_demo.shape}  (n_matrices, p, p)")
print(f"Tangent vectors    : {S_demo.shape}  (n_matrices, p*(p+1)//2)")
print(f"→ Each 4x4 covariance → {S_demo.shape[1]}-d flat vector")
print()

# Verify: the Riemannian mean maps to the zero vector in tangent space
S_mean = tangent_space(M_demo[np.newaxis], M_demo)
print(f"Tangent vector of the reference (should be ~0): "
      f"max|coeff| = {np.abs(S_mean).max():.2e}")

# %% [markdown]
# ## Part 4 — Empirical: Riemannian vs CSP+LDA as training set shrinks
#
# Now the key experiment. We compare two pipelines on BCI IV 2a (motor imagery):
#
# | Pipeline | Steps |
# |---|---|
# | **Riemann** | `Covariances(oas)` → `TangentSpace` → `LogisticRegression` |
# | **CSP+LDA** | `CSP(n_components=4)` → `LinearDiscriminantAnalysis` |
#
# We evaluate within one subject using `make_block_split` (contiguous folds,
# no leakage), and sweep the **training fraction** from 20 % to 100 % of the
# available training trials.
#
# **Hypothesis:** Riemannian features rely on covariance matrices — each matrix
# is a sufficient statistic for the full spatial covariation pattern. With few
# trials, covariance matrices are still estimable (especially with OAS shrinkage),
# while CSP has to identify spatial filters from scratch from small, noisy data.
# We expect Riemannian to degrade more gracefully.

# %%
# Load data — 2 subjects in smoke mode, up to 3 otherwise
n_subj = 2 if SMOKE else 3
print(f"Loading {n_subj} subject(s) from BCI IV 2a …")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=n_subj)
print(f"  X={X_all.shape}, classes={np.bincount(y_all)}")

# Work within subject 1 only (most trials)
s1_mask = subj_all == subj_all.min()
X1, y1 = X_all[s1_mask], y_all[s1_mask]
print(f"\nSubject {subj_all.min()}: {X1.shape[0]} trials, "
      f"{X1.shape[1]} channels, {X1.shape[2]} time-points")
print(f"Class balance: {np.bincount(y1)}")

# %%
# Define the two pipelines
pipe_riemann = Pipeline([
    ("cov",  Covariances(estimator="oas")),
    ("ts",   TangentSpace(metric="riemann")),
    ("clf",  LogisticRegression(C=1.0, max_iter=500, random_state=0)),
])

pipe_csp = Pipeline([
    ("csp",  CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("clf",  LinearDiscriminantAnalysis()),
])

# %%
# Sweep training fractions and evaluate with block-split CV
# make_block_split gives contiguous honest folds
N_SPLITS = 4 if SMOKE else 5
TRAIN_FRACS = [0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS = [0, 1, 2]  # keep fast; more seeds = tighter CIs

def eval_pipeline_at_frac(pipe, X, y, frac, n_splits, seeds):
    """
    For each seed: take the first `frac` of each training fold's trials,
    fit the pipeline, score on the held-out test fold.
    Returns array of accuracies shape (n_seeds, n_splits).
    """
    from sklearn.base import clone as sk_clone
    scores = np.zeros((len(seeds), n_splits))
    folds = list(make_block_split(len(y), n_splits=n_splits))

    for si, seed in enumerate(seeds):
        rng_s = np.random.default_rng(seed)
        for fi, (train_idx, test_idx) in enumerate(folds):
            # Subsample training set to `frac` of available trials
            n_keep = max(2, int(len(train_idx) * frac))
            # Always take a contiguous prefix so we don't shuffle time order
            train_sub = train_idx[:n_keep]

            model = sk_clone(pipe)
            try:
                model.fit(X[train_sub], y[train_sub])
                preds = model.predict(X[test_idx])
                scores[si, fi] = (preds == y[test_idx]).mean()
            except Exception:
                scores[si, fi] = np.nan
    return scores


results = {}  # frac -> {"riemann": arr, "csp": arr}

for frac in TRAIN_FRACS:
    print(f"  frac={frac:.0%} …", end=" ", flush=True)
    r_scores = eval_pipeline_at_frac(pipe_riemann, X1, y1, frac, N_SPLITS, SEEDS)
    c_scores = eval_pipeline_at_frac(pipe_csp,     X1, y1, frac, N_SPLITS, SEEDS)
    results[frac] = {"riemann": r_scores, "csp": c_scores}
    print(f"Riemann={np.nanmean(r_scores):.3f}  CSP={np.nanmean(c_scores):.3f}")

print("\nDone.")

# %%
# Summarise: mean accuracy ± std across seeds×folds
fracs    = TRAIN_FRACS
rie_mean = [np.nanmean(results[f]["riemann"]) for f in fracs]
rie_std  = [np.nanstd(results[f]["riemann"])  for f in fracs]
csp_mean = [np.nanmean(results[f]["csp"])     for f in fracs]
csp_std  = [np.nanstd(results[f]["csp"])      for f in fracs]

print("Training fraction | Riemann acc     | CSP+LDA acc     | Riemann advantage")
print("-" * 75)
for f, rm, rs, cm, cs in zip(fracs, rie_mean, rie_std, csp_mean, csp_std):
    adv = rm - cm
    print(f"       {f:5.0%}        | {rm:.3f} ± {rs:.3f}  | {cm:.3f} ± {cs:.3f}  | {adv:+.3f}")

# %%
# Plot: accuracy vs training fraction
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    [f * 100 for f in fracs], rie_mean, yerr=rie_std,
    marker="o", linewidth=2.0, capsize=4, color="#2ecc71",
    label="Riemann (Cov + TangentSpace + LogReg)", zorder=5,
)
ax.errorbar(
    [f * 100 for f in fracs], csp_mean, yerr=csp_std,
    marker="s", linewidth=2.0, capsize=4, color="#e74c3c",
    label="CSP (4 components) + LDA", zorder=5,
)
ax.axhline(0.5, color="gray", lw=1.0, linestyle=":", label="Chance (50 %)")

# Shade the low-data regime
ax.axvspan(0, 45, alpha=0.07, color="gold", label="Low-data regime")

ax.set(
    xlabel="Training set size  (% of one subject's trials)",
    ylabel="Accuracy (block-split CV)",
    title="Riemannian vs. CSP+LDA: accuracy as training data shrinks\n"
          f"BCI IV 2a, subject {subj_all.min()}, "
          f"{N_SPLITS}-fold block-split, {len(SEEDS)} seeds",
    xlim=(10, 105), ylim=(0.35, 1.05),
)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_smalldata.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure saved.")

# %% [markdown]
# ## Part 5 — Why does covariance pool information more robustly?
#
# ### Counting parameters vs. counting trials
#
# **CSP** must estimate $k$ spatial filters of length $p$ (channels). Those
# filters are the solution to a generalised eigenvalue problem on the
# *class-conditional covariance matrices* — but each class covariance has
# $p(p+1)/2$ free parameters. With 22 channels and only 20 training trials per
# class, you are estimating $22 \times 23/2 = 253$ numbers from 20 observations:
# a hugely under-determined problem. CSP therefore picks up noise.
#
# **Covariance + Tangent Space (Riemann)** *also* has to estimate a $p \times p$
# covariance per trial. The difference is:
#
# 1. **OAS shrinkage**: the `Covariances(estimator="oas")` step applies
#    Oracle-Approximating Shrinkage, which pulls the sample covariance towards
#    $\alpha \hat{\mathbf{C}} + (1-\alpha)\,\text{tr}(\hat{\mathbf{C}})/p\,\mathbf{I}$.
#    This keeps the matrix well-conditioned even with very few trials, at the
#    cost of a small bias.
# 2. **No explicit filter learning**: the classifier operates in the full
#    $p(p+1)/2$-dimensional tangent space and uses $\ell_2$-regularised logistic
#    regression. There is no intermediate step that can catastrophically overfit
#    to a bad spatial filter.
# 3. **Geometric mean as a robust reference**: the Riemannian mean of even a
#    handful of covariance matrices is already a meaningful, stable point on the
#    manifold — because the curvature of $\text{Sym}^+_p$ is non-positive
#    (Hadamard space), the mean is unique and the iterative algorithm converges
#    even from a few points.
#
# ### The informal intuition
#
# > Each covariance matrix "pools" information from all $p \times T$ individual
# > measurements into a single $p \times p$ object. Even a single trial yields a
# > rich descriptor of the brain state. CSP, by contrast, needs enough *labelled*
# > trials to make two class-conditional pools statistically distinguishable before
# > it can find a useful filter. With tiny $n$, the pools overlap by chance, and
# > the estimated filters are arbitrary.

# %%
# Quick illustration: condition number of sample vs OAS covariance as n_trials grows
from pyriemann.estimation import Covariances as PyrCov

p = 22  # BCI IV 2a has 22 EEG channels
trial_counts = [5, 10, 20, 40, 80, 160] if not SMOKE else [5, 10, 20, 40]
n_reps = 30 if not SMOKE else 10

cond_scm = []  # sample covariance (no regularisation)
cond_oas = []  # OAS shrinkage

for n in trial_counts:
    c_scm_r, c_oas_r = [], []
    for _ in range(n_reps):
        # Simulate n trials of p-channel EEG (random, no true signal)
        Z = rng.standard_normal((n, p, 50))  # 50 time samples per trial
        C_scm = PyrCov(estimator="scm").fit_transform(Z)
        C_oas = PyrCov(estimator="oas").fit_transform(Z)
        c_scm_r.append(np.mean([np.linalg.cond(c) for c in C_scm]))
        c_oas_r.append(np.mean([np.linalg.cond(c) for c in C_oas]))
    cond_scm.append(np.mean(c_scm_r))
    cond_oas.append(np.mean(c_oas_r))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(trial_counts, cond_scm, "o-", color="#e74c3c",
            lw=2, label="Sample covariance (SCM, no regularisation)")
ax.semilogy(trial_counts, cond_oas, "s-", color="#2ecc71",
            lw=2, label="OAS shrinkage (pyriemann default)")
ax.set(
    xlabel="Number of trials (simulated, p=22 channels)",
    ylabel="Mean condition number (log scale)",
    title="OAS shrinkage keeps covariance matrices well-conditioned\n"
          "even when n_trials << n_channels²",
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_condnum.png", dpi=100, bbox_inches="tight")
plt.show()
print("Condition number (SCM vs OAS) plotted.")

# %% [markdown]
# ### Summary of results
#
# The condition-number plot shows the core problem:
# when $n\_trials \ll p(p+1)/2$, the sample covariance is nearly singular
# (huge condition number → numerically unstable inverses → bad Riemannian
# distances). OAS shrinkage keeps the condition number bounded regardless
# of training set size, which is why the Riemannian pipeline remains
# competitive with as few as 10–20 trials per class.
#
# The accuracy-vs-fraction plot (Part 4) shows the consequence:
# at 20 % of the training set the Riemannian pipeline typically leads CSP+LDA
# by several percentage points; by the time the full training set is available
# the gap narrows or reverses because CSP has enough data to identify good
# spatial filters.

# %% [markdown]
# ## ⚠️ A subtler trap: cross-session covariance shift and the leaking reference
#
# ### The problem
#
# The tangent-space projection requires a **reference matrix** $\mathbf{M}$
# (the Riemannian mean). In a typical workflow you fit $\mathbf{M}$ on the
# training covariances. Every covariance — train *and* test — is then projected
# relative to the *same* reference.
#
# This works perfectly within a single session. But EEG covariance matrices
# **shift between sessions and subjects** due to:
#
# * electrode impedance changes (gel drying, cap repositioning),
# * skin–electrode contact drift,
# * subject fatigue or arousal level changes,
# * subtle differences in cap placement (channel locations shift slightly).
#
# When you train on Session 1 and test on Session 2, the test covariances live
# on a *different part of the manifold* from the training covariances. Projecting
# both onto the Session-1 reference $\mathbf{M}_1$ stretches and rotates the
# test-session features in the tangent space — the classifier sees a
# distribution shift that looks like a new task.
#
# ### The (sometimes) hidden leakage
#
# A common mistake: **compute $\mathbf{M}$ using all trials (train + test) and
# then split.** This is leakage. The reference has "seen" the test sessions and
# re-centres the test covariances to exactly where the model was trained — making
# scores look better than they are. In a within-session CV this effect is mild
# (the sessions are close on the manifold). Across sessions or subjects the
# inflation can be dramatic.
#
# ### The fix: Riemannian alignment (re-centring per session)
#
# The standard remedy is **Riemannian Alignment (RA)**, introduced by He & Wu
# (2019): before concatenating sessions, whiten each session's covariances to a
# common reference (usually the identity matrix):
#
# $$
# \tilde{\mathbf{C}}_i^{(s)} = \mathbf{M}_s^{-1/2}\,\mathbf{C}_i^{(s)}\,\mathbf{M}_s^{-1/2}
# $$
#
# where $\mathbf{M}_s$ is the Riemannian mean of session $s$ **computed only
# from the training trials of that session**. After alignment, all sessions'
# covariances are centred at the identity, removing the inter-session shift.
#
# **The critical rule:**
#
# > $\mathbf{M}_s$ for the *test* session must be estimated from the test
# > session's own (unlabelled) trials, not from the training session.
# > If you use test-session labels or the training-session mean to align the
# > test session, you either leak or you fail to remove the shift.
#
# This is why Riemannian methods, despite their elegance, still require careful
# bookkeeping about *which data was used to fit the reference*. The manifold
# geometry solves the within-session averaging problem beautifully — it does
# not automatically solve the cross-session distribution shift problem.
#
# ### Minimal code sketch (do NOT run inside a train/test fold on all data)
#
# ```python
# # Leaking version (WRONG for cross-session eval)
# M_all  = mean_riemann(np.concatenate([C_train, C_test]))   # sees test!
# S_train = tangent_space(C_train, M_all)
# S_test  = tangent_space(C_test,  M_all)
#
# # Correct version: reference fit on train only
# M_train = mean_riemann(C_train)
# S_train = tangent_space(C_train, M_train)
# S_test  = tangent_space(C_test,  M_train)   # test projected onto train reference
#
# # Best practice for cross-session (Riemannian alignment):
# M_train_sess = mean_riemann(C_train)         # from training session
# M_test_sess  = mean_riemann(C_test_unlabelled)  # from test session (no labels needed)
# C_train_aligned = M_train_sess @ np.linalg.inv(M_train_sess) @ C_train   # = identity centred
# # ... more precisely via matrix square root; pyriemann.utils.mean provides helpers
# ```
#
# ### Take-away
#
# Riemannian geometry gives you a principled, parameter-efficient representation
# of EEG covariance structure that shines with small training sets. But it does
# *not* make the data-collection problem disappear. The reference point
# $\mathbf{M}$ is a learned quantity — mis-fitting it (on too little data or on
# contaminated data) propagates errors to every downstream distance and
# projection. On small data, use OAS shrinkage, fit the reference on the
# training fold only, and consider per-session alignment when sessions span
# more than a few minutes or different recording conditions.
