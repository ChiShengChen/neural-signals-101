# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — CSP Geometry & Derivation
#
# You will leave here with a rigorous understanding of *why* CSP works, not just
# *what* it does: the Rayleigh-quotient objective, the generalized eigenvalue
# reduction, the whitening + PCA geometric view, and a synthetic demo that makes
# each step visible.
#
# > **Prerequisites:** main Chapters 03 and 07.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget** (this is a side-quest).

# %%
# --- bootstrap: locate the neuro101 package whether running locally or from
#     deep-dives/_src/ (one extra parent level compared with notebooks/_src/).
import sys
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg

rng = np.random.default_rng(0)

# Use a clean, non-interactive backend for CI.
matplotlib.use("Agg")

print("numpy", np.__version__, " | scipy", linalg.__name__)

# %% [markdown]
# ---
# ## 1  The objective: maximize the Rayleigh quotient
#
# Given labelled EEG epochs from **class 1** and **class 2**, let
#
# $$\Sigma_1 = \mathbb{E}[\,x x^\top \mid \text{class 1}\,],\qquad
#   \Sigma_2 = \mathbb{E}[\,x x^\top \mid \text{class 2}\,]$$
#
# be the *trial-averaged* spatial covariance matrices (shape $C \times C$,
# where $C$ is the number of EEG channels).
#
# A **spatial filter** $w \in \mathbb{R}^C$ projects the multi-channel signal to
# a scalar channel $z = w^\top x$.  The variance of $z$ in class $k$ is
#
# $$\text{Var}_k(z) = w^\top \Sigma_k\, w.$$
#
# **CSP goal:** find $w$ that **maximises** the variance in class 1 while
# **minimising** it in class 2.  Formally, maximise the **Rayleigh quotient**:
#
# $$\mathcal{R}(w) = \frac{w^\top \Sigma_1\, w}{w^\top \Sigma_2\, w}.$$
#
# *Why log-variance later?*  The variance ratio is multiplicative across trials;
# taking logs maps it to an additive, roughly-Gaussian feature that linear
# classifiers prefer.

# %% [markdown]
# ---
# ## 2  Reduction to a generalized eigenvalue problem
#
# ### Setting up the stationarity condition
#
# At the maximum of $\mathcal{R}(w)$ the gradient vanishes:
#
# $$\nabla_w \mathcal{R}(w) = 0
#   \implies
#   \frac{2\Sigma_1 w}{w^\top \Sigma_2 w}
#   - \frac{w^\top \Sigma_1 w}{(w^\top \Sigma_2 w)^2}\,2\Sigma_2 w = 0.$$
#
# Multiply through by $(w^\top \Sigma_2 w)/2$ and let $\lambda = \mathcal{R}(w)$:
#
# $$\boxed{\Sigma_1\, w = \lambda\, \Sigma_2\, w}$$
#
# This is the **generalized eigenvalue problem** (GEP).  Every stationary point
# of the Rayleigh quotient — maximum, minimum, or saddle — is an eigenpair of
# this GEP.  The *maximum* corresponds to the **largest** eigenvalue $\lambda_1$,
# and the *minimum* (most discriminative in the opposite direction) to the
# **smallest** $\lambda_C$.
#
# ### The whitening + PCA derivation (key steps, no heavy proofs)
#
# **Step 1 — Whiten w.r.t. the composite covariance.**
# Let $\Sigma_c = \Sigma_1 + \Sigma_2$.  Because $\Sigma_c \succ 0$ (positive
# definite, assuming enough trials), write its eigendecomposition
# $\Sigma_c = U D U^\top$ and form the whitening matrix
# $P = D^{-1/2} U^\top$.  Define $\tilde\Sigma_k = P\,\Sigma_k\,P^\top$.
#
# **Step 2 — Turn the GEP into an ordinary eigenvalue problem.**
# Substituting $w = P^\top v$ into $\Sigma_1 w = \lambda \Sigma_2 w$ gives, after
# multiplying on the left by $P$:
#
# $$\tilde\Sigma_1\, v = \lambda\,\tilde\Sigma_2\, v.$$
#
# Because whitening enforces $\tilde\Sigma_1 + \tilde\Sigma_2 = I$, we have
# $\tilde\Sigma_2 = I - \tilde\Sigma_1$.  The GEP then becomes
#
# $$\tilde\Sigma_1\, v = \lambda (I - \tilde\Sigma_1)\, v
#   \implies \tilde\Sigma_1\, v = \frac{\lambda}{1+\lambda}\, v.$$
#
# So $v$ is just an **ordinary eigenvector of** $\tilde\Sigma_1$!
#
# **Step 3 — Map back.**
# The spatial filter in original space is $w = P^\top v$; the corresponding
# **pattern** (what the source looks like on the scalp) is $a = \Sigma_c w /
# (w^\top \Sigma_c w)$ — not $w$ itself (a common mistake).
#
# **Geometric intuition:** whitening makes both covariances "round" in the
# composite metric, then the ordinary PCA of $\tilde\Sigma_1$ finds the axis
# of maximum variance for class 1 — which is simultaneously the axis of
# *minimum* variance for class 2 (because they sum to the identity).

# %%
# --- Numeric check: GEP vs scipy.linalg.eigh --------------------------------
# scipy.linalg.eigh(A, B) solves A v = lambda B v for symmetric (A,B).
# We also verify the whitening route gives the same eigenvectors (up to sign).

def csp_filters(Sigma1: np.ndarray, Sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CSP spatial filters (columns of W) and eigenvalues via eigh.

    eigh(Sigma1, Sigma1+Sigma2) solves Sigma1 w = lambda*(Sigma1+Sigma2)*w.
    This is equivalent to the Rayleigh quotient Sigma1 / Sigma2 because
    w'Sigma1 w / w'Sigma2 w = rho  <=>  Sigma1 w = rho Sigma2 w,
    and normalising the denominator to Sigma1+Sigma2 is a numerically stable
    re-parameterisation (eigenvalues now live in [0,1], lambda_new = rho/(1+rho)).
    """
    Sigma_c = Sigma1 + Sigma2
    # eigh returns eigenvalues in ascending order; columns of v are eigenvectors.
    vals, vecs = linalg.eigh(Sigma1, Sigma_c)
    # Largest eigenvalue = most class-1 variance; return both ends.
    return vals, vecs


# Quick sanity: random 3x3 SPD matrices
A = rng.standard_normal((3, 3)); A = A @ A.T + np.eye(3)
B = rng.standard_normal((3, 3)); B = B @ B.T + np.eye(3)

vals_eigh, vecs_eigh = linalg.eigh(A, A + B)
# Directly via GEP Aw = lambda Bw
vals_gep, vecs_gep = linalg.eigh(A, B)

# Rayleigh quotients should match (up to the rho/(1+rho) transform)
rho_gep = vals_gep        # rho = lambda from A w = lambda B w
rho_eigh = vals_eigh / (1 - vals_eigh + 1e-15)   # invert lambda_new = rho/(1+rho)
print("GEP eigenvalues (Sigma1 w = λ Sigma2 w):", np.round(rho_gep, 4))
print("Recovered from eigh parameterisation:    ", np.round(rho_eigh, 4))
print("Max abs difference:", np.max(np.abs(np.sort(rho_gep) - np.sort(rho_eigh))))

# %% [markdown]
# The two sets of eigenvalues match (up to rounding).  The
# `scipy.linalg.eigh(Sigma1, Sigma1+Sigma2)` parameterisation is what MNE uses
# internally — eigenvalues land in $[0,1]$, avoiding unbounded ratios.

# %% [markdown]
# ---
# ## 3  Visual demo: 2-channel, 2-class synthetic data
#
# We construct two classes whose covariance matrices differ in *rotation* but share
# the same total spread.  This mimics motor imagery: the same overall EEG "energy"
# but a different spatial distribution depending on which hand is imagined.

# %%
# --- Synthetic dataset -------------------------------------------------------
n_trials = 200      # per class
n_times  = 400      # samples per trial

# Class 1: variance is elongated along direction [cos θ, sin θ]
theta1 = np.deg2rad(30)
u1 = np.array([np.cos(theta1), np.sin(theta1)])
v1 = np.array([-np.sin(theta1), np.cos(theta1)])
# Eigenvalues: large along u1, small along v1 (high channel-1 power)
D1 = np.diag([4.0, 0.5])
cov1 = u1[:, None] * D1[0, 0] * u1[None, :] + v1[:, None] * D1[1, 1] * v1[None, :]

# Class 2: variance elongated along direction [cos φ, sin φ]
theta2 = np.deg2rad(-40)
u2 = np.array([np.cos(theta2), np.sin(theta2)])
v2 = np.array([-np.sin(theta2), np.cos(theta2)])
D2 = np.diag([4.0, 0.5])
cov2 = u2[:, None] * D2[0, 0] * u2[None, :] + v2[:, None] * D2[1, 1] * v2[None, :]

# Draw trials: each trial x is shape (2, n_times); covariance is estimated per trial.
# For simplicity we draw iid time-points from the class covariance (zero-mean Gaussian).
L1 = np.linalg.cholesky(cov1)
L2 = np.linalg.cholesky(cov2)

X1 = (L1 @ rng.standard_normal((2, n_trials * n_times))).reshape(2, n_trials, n_times)
X2 = (L2 @ rng.standard_normal((2, n_trials * n_times))).reshape(2, n_trials, n_times)
# X_k shape: (channels=2, trials, times) → transpose to (trials, channels, times)
X1 = X1.transpose(1, 0, 2)   # (n_trials, 2, n_times)
X2 = X2.transpose(1, 0, 2)

# Estimate class covariance matrices from the data (trial-average of xx')
Sigma1_hat = np.mean([x @ x.T / n_times for x in X1], axis=0)
Sigma2_hat = np.mean([x @ x.T / n_times for x in X2], axis=0)

print("Estimated Sigma1:\n", np.round(Sigma1_hat, 3))
print("True Sigma1:\n", np.round(cov1, 3))

# %%
# --- Compute CSP filters ----------------------------------------------------
vals, W = csp_filters(Sigma1_hat, Sigma2_hat)
# Columns of W are filters; eigh returns ascending order,
# so last column = maximum-Rayleigh filter for class 1,
# first column = minimum-Rayleigh (= maximum for class 2).
w_max = W[:, -1]   # most class-1 variance
w_min = W[:,  0]   # least class-1 variance (most class-2)

print(f"\nCSP eigenvalues (in [0,1]): {vals.round(4)}")
print(f"Filter w_max (favours class 1): {w_max.round(4)}")
print(f"Filter w_min (favours class 2): {w_min.round(4)}")

# Verify Rayleigh quotients
def rayleigh(w, S1, S2):
    return (w @ S1 @ w) / (w @ S2 @ w)

print(f"\nRayleigh(w_max) = {rayleigh(w_max, Sigma1_hat, Sigma2_hat):.4f}  "
      f"(should be large)")
print(f"Rayleigh(w_min) = {rayleigh(w_min, Sigma1_hat, Sigma2_hat):.4f}  "
      f"(should be small)")

# %% [markdown]
# ### Figure 1 — Data clouds + CSP filter directions
#
# The ellipses show the covariance structure of each class in 2-D channel space.
# The CSP filters (arrows) are the directions that maximally separate the classes
# by variance.

# %%
def cov_ellipse(cov, center=(0, 0), n_std=2.0, n_pts=200):
    """Return (x, y) for an ellipse representing a 2-D covariance matrix."""
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.array([np.cos(t), np.sin(t)])
    L = np.linalg.cholesky(cov)
    ellipse = n_std * (L @ circle)
    return ellipse[0] + center[0], ellipse[1] + center[1]


# Sample a few trial-mean points for scatter (one point per trial = mean of x)
mean1 = X1.mean(axis=-1)   # (n_trials, 2)
mean2 = X2.mean(axis=-1)

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(mean1[:, 0], mean1[:, 1], s=8, alpha=0.3,
           color="#4c72b0", label="Class 1 trial means")
ax.scatter(mean2[:, 0], mean2[:, 1], s=8, alpha=0.3,
           color="#dd8452", label="Class 2 trial means")

# Covariance ellipses
ex1, ey1 = cov_ellipse(Sigma1_hat)
ex2, ey2 = cov_ellipse(Sigma2_hat)
ax.plot(ex1, ey1, color="#4c72b0", lw=2, label="Class 1 cov ellipse (2σ)")
ax.plot(ex2, ey2, color="#dd8452", lw=2, label="Class 2 cov ellipse (2σ)")

# CSP filter arrows
scale = 2.5
ax.annotate("", xy=w_max * scale, xytext=-w_max * scale,
            arrowprops=dict(arrowstyle="<->", color="green", lw=2.5))
ax.annotate("", xy=w_min * scale, xytext=-w_min * scale,
            arrowprops=dict(arrowstyle="<->", color="purple", lw=2.5))

ax.text(*(w_max * (scale + 0.3)), "w_max\n(↑ var class 1)", color="green",
        ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(*(w_min * (scale + 0.3)), "w_min\n(↑ var class 2)", color="purple",
        ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_aspect("equal")
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Channel 1", fontsize=11)
ax.set_ylabel("Channel 2", fontsize=11)
ax.set_title("2-D EEG channel space: covariance ellipses + CSP filter directions",
             fontsize=11)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig1.png", dpi=120)
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# **Reading Figure 1.**  The green bidirectional arrow (`w_max`) passes through the
# long axis of the *blue* (class 1) ellipse and through the *short* axis of the
# orange (class 2) ellipse.  Projecting any data point onto this axis gives a
# scalar whose variance is *high* for class 1 and *low* for class 2 — exactly the
# Rayleigh objective.  The purple arrow (`w_min`) does the opposite.
#
# Neither filter is aligned with any original channel axis; they are *oblique*
# mixtures optimised for discrimination.  This is the core gain of CSP over simply
# picking one electrode.

# %%
# --- Project trials onto CSP filters and compute log-variance features -------
def logvar_features(X_class, w):
    """Project each trial onto w, return log-variance across time.

    X_class : (n_trials, n_channels, n_times)
    w       : (n_channels,)
    Returns : (n_trials,) of log-variance
    """
    # w @ x[trial] for each trial -> (n_trials, n_times)
    z = np.einsum("c,tcs->ts", w, X_class)
    return np.log(z.var(axis=1) + 1e-12)


lv1_max = logvar_features(X1, w_max)   # class 1, filter w_max
lv2_max = logvar_features(X2, w_max)   # class 2, filter w_max
lv1_min = logvar_features(X1, w_min)   # class 1, filter w_min
lv2_min = logvar_features(X2, w_min)   # class 2, filter w_min

# Collect into feature matrix: [logvar(w_max), logvar(w_min)]
F1 = np.column_stack([lv1_max, lv1_min])   # (n_trials, 2)
F2 = np.column_stack([lv2_max, lv2_min])

# %% [markdown]
# ### Figure 2 — Log-variance feature scatter (the classifier's view)
#
# After projecting onto the two CSP filters, the log-variances separate the two
# classes cleanly in 2-D feature space.  A linear classifier (dashed line) can
# split them near-perfectly here, even though the original 2-D channel data
# overlapped substantially.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: 1-D distributions per filter
ax = axes[0]
bins = np.linspace(-2, 4, 50)
ax.hist(lv1_max, bins=bins, alpha=0.6, color="#4c72b0", label="Class 1")
ax.hist(lv2_max, bins=bins, alpha=0.6, color="#dd8452", label="Class 2")
ax.set(xlabel="log-variance (w_max projection)", ylabel="Count",
       title="Filter w_max: variance separated by class")
ax.legend()
ax.axvline(0.5 * (lv1_max.mean() + lv2_max.mean()), color="k", ls="--",
           label="midpoint")

# Right: 2-D scatter in CSP feature space
ax = axes[1]
ax.scatter(F1[:, 0], F1[:, 1], s=12, alpha=0.5, color="#4c72b0", label="Class 1")
ax.scatter(F2[:, 0], F2[:, 1], s=12, alpha=0.5, color="#dd8452", label="Class 2")

# Simple linear decision boundary (perpendicular bisector of class means)
m1, m2 = F1.mean(0), F2.mean(0)
mid = 0.5 * (m1 + m2)
direction = m2 - m1
normal = np.array([-direction[1], direction[0]])
t_vals = np.linspace(-3, 3, 100)
boundary = mid[:, None] + normal[:, None] * t_vals
ax.plot(boundary[0], boundary[1], "k--", lw=1.5, label="Linear boundary")
ax.set(xlabel="log-var(w_max)  [class 1 ↑]",
       ylabel="log-var(w_min)  [class 2 ↑]",
       title="CSP feature space: classes linearly separable")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig2.png", dpi=120)
plt.show()
print("Figure 2 saved.")

# %% [markdown]
# **Connection to Chapter 03.**  In Chapter 03 we saw that the eigenvectors of a
# covariance matrix point along the directions of maximum variance.  CSP extends
# this: instead of diagonalising *one* covariance, it simultaneously diagonalises
# *two* via the generalized eigenproblem.  The resulting filters are the axes that
# make class 1 look "big" and class 2 look "small" at the same time.

# %% [markdown]
# ---
# ## 4  The whitening view — CSP as whiten + PCA (geometric)
#
# The derivation in §2 reveals a clean geometric pipeline:
#
# ```
# original channel space
#       │
#       │  whiten w.r.t. Σ_c = Σ₁ + Σ₂
#       │  (makes both classes "round" in the composite metric)
#       ▼
# whitened space  (Σ̃₁ + Σ̃₂ = I)
#       │
#       │  ordinary PCA of Σ̃₁
#       │  (first PC = direction with most class-1 variance)
#       ▼
# CSP component space
# ```
#
# Whitening **equalises** the total power across directions, removing any
# channel-amplitude differences that are irrelevant to the class distinction.
# PCA of $\tilde\Sigma_1$ then finds the discrimination axis.
#
# Because $\tilde\Sigma_2 = I - \tilde\Sigma_1$, the PC that maximises class-1
# variance *simultaneously minimises* class-2 variance.  This is the algebraic
# reason why CSP is simultaneously optimal for both directions.

# %%
# --- Visualise the whitening step -------------------------------------------
# Whiten the sample cloud w.r.t. Sigma_c, then show that Sigma_tilde_1 + Sigma_tilde_2 = I.

Sigma_c = Sigma1_hat + Sigma2_hat
vals_c, U_c = np.linalg.eigh(Sigma_c)
P = (U_c / np.sqrt(vals_c)[None, :]).T          # whitening matrix (D^{-1/2} U')

Sigma1_tilde = P @ Sigma1_hat @ P.T
Sigma2_tilde = P @ Sigma2_hat @ P.T
identity_check = Sigma1_tilde + Sigma2_tilde

print("Σ̃₁ + Σ̃₂ (should be identity):\n", np.round(identity_check, 6))

# Eigenvectors of Sigma1_tilde (ordinary PCA of whitened class-1 covariance)
vals_t, V = np.linalg.eigh(Sigma1_tilde)
print(f"\nEigenvalues of Σ̃₁: {vals_t.round(4)}  (sum={vals_t.sum():.4f})")
print(f"Eigenvalues of Σ̃₂: {(1-vals_t).round(4)}  (sum={(1-vals_t).sum():.4f})")

# Map eigenvectors back to original space
W_whitening_route = P.T @ V    # columns = CSP filters (whitening route)

# Compare with scipy.linalg.eigh filters (should match up to sign)
cos_sim = np.abs(np.diag(W_whitening_route.T @ W))
print(f"\nCosine similarity between both routes' filters: {cos_sim.round(6)}")
print("(1.0 = identical up to sign, as expected)")

# %% [markdown]
# The two routes — direct `scipy.linalg.eigh` and whitening + PCA — give
# identical spatial filters (cosine similarity = 1.0).  Implementing one is
# equivalent to the other; the whitening route is more transparent geometrically,
# while `eigh` is numerically superior because it uses a specialised solver.

# %%
# --- Figure 3: whitened space vs original space -----------------------------
# Show both class clouds before and after whitening.

# Sample a few representative points (trial means)
pts1 = mean1   # (n_trials, 2)
pts2 = mean2

# Whiten
pts1_w = (P @ pts1.T).T
pts2_w = (P @ pts2.T).T

Sigma1_w_est = np.cov(pts1_w.T)
Sigma2_w_est = np.cov(pts2_w.T)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Original space
ax = axes[0]
ax.scatter(pts1[:, 0], pts1[:, 1], s=8, alpha=0.3, color="#4c72b0")
ax.scatter(pts2[:, 0], pts2[:, 1], s=8, alpha=0.3, color="#dd8452")
ex1o, ey1o = cov_ellipse(Sigma1_hat)
ex2o, ey2o = cov_ellipse(Sigma2_hat)
ax.plot(ex1o, ey1o, "#4c72b0", lw=2, label="Class 1 ellipse")
ax.plot(ex2o, ey2o, "#dd8452", lw=2, label="Class 2 ellipse")
# Draw axes
for label, vec, col in [("w_max", w_max, "green"), ("w_min", w_min, "purple")]:
    ax.annotate("", xy=vec * 2, xytext=-vec * 2,
                arrowprops=dict(arrowstyle="<->", color=col, lw=2))
ax.set_aspect("equal"); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
ax.set_title("Original channel space", fontsize=11)
ax.set_xlabel("Channel 1"); ax.set_ylabel("Channel 2")
ax.legend(fontsize=8)

# Whitened space
ax = axes[1]
ax.scatter(pts1_w[:, 0], pts1_w[:, 1], s=8, alpha=0.3, color="#4c72b0")
ax.scatter(pts2_w[:, 0], pts2_w[:, 1], s=8, alpha=0.3, color="#dd8452")
if Sigma1_w_est.shape == (2, 2):
    ex1w, ey1w = cov_ellipse(Sigma1_w_est)
    ex2w, ey2w = cov_ellipse(Sigma2_w_est)
    ax.plot(ex1w, ey1w, "#4c72b0", lw=2, label="Class 1 ellipse (whitened)")
    ax.plot(ex2w, ey2w, "#dd8452", lw=2, label="Class 2 ellipse (whitened)")
# Draw PCA directions of Sigma1_tilde in whitened space
for label, vec, col in [("PC1(Σ̃₁)=w_max", V[:, -1], "green"),
                         ("PC2(Σ̃₁)=w_min", V[:,  0], "purple")]:
    ax.annotate("", xy=vec * 1.5, xytext=-vec * 1.5,
                arrowprops=dict(arrowstyle="<->", color=col, lw=2))
# Draw unit circle (composite covariance is identity after whitening)
theta_c = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta_c), np.sin(theta_c), "k:", lw=1, label="Unit circle (Σ_c = I)")
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
ax.set_title("Whitened space (Σ_c = I)\nCSP = ordinary PCA of Σ̃₁", fontsize=11)
ax.set_xlabel("Whitened dim 1"); ax.set_ylabel("Whitened dim 2")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig3.png", dpi=120)
plt.show()
print("Figure 3 saved.")

# %% [markdown]
# **Reading Figure 3.**  On the left, the two class ellipses are tilted in different
# directions — the raw channel data has direction-dependent power.  After whitening
# (right), the composite covariance becomes a circle (dashed unit circle), equalising
# all directions.  The class-1 ellipse (blue) is still elongated — just in a
# direction that is now *purely* about class discrimination.  The PCA axes of the
# whitened class-1 covariance (green / purple arrows) are the CSP filters.

# %% [markdown]
# ---
# ## 5  Optional: MNE CSP on BCI IV 2a (real data, quick version)
#
# The section below loads a couple of real subjects to confirm that the geometric
# picture translates to real EEG.  It uses `NEURO101_SMOKE=1` awareness so it
# stays fast in CI.

# %%
import os
from neuro101 import io, datasets as ds

SMOKE = ds.is_smoke()
n_subj = 1 if SMOKE else 2    # keep it small; this is a side-quest not a benchmark

print(f"Loading {n_subj} subject(s) of BCI IV 2a  (SMOKE={SMOKE}) …")
X_real, y_real, subj_real = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X_real.shape}  y classes={np.bincount(y_real)}  sfreq={sf} Hz")

# %%
from neuro101.features import make_csp

csp_mne = make_csp(n_components=4)
F_csp_real = csp_mne.fit_transform(X_real, y_real)
print("MNE CSP log-var features:", F_csp_real.shape)

# %%
# --- Figure 4: MNE CSP scatter on real data ---------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
colors = {0: "#4c72b0", 1: "#dd8452"}
for cls in np.unique(y_real):
    mask = y_real == cls
    ax.scatter(F_csp_real[mask, 0], F_csp_real[mask, -1],
               s=15, alpha=0.55, color=colors[cls],
               label=f"class {cls} ({'left' if cls==0 else 'right'} hand)")

ax.set(xlabel="CSP comp 1 (log-var, ↑ class 1 variance)",
       ylabel="CSP comp 4 (log-var, ↑ class 2 variance)",
       title=f"MNE CSP on BCI IV 2a ({n_subj} subject(s))\n"
             f"[demo-only fit on all data — see §6 for the trap]")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig4.png", dpi=120)
plt.show()
print("Figure 4 saved.")

# %% [markdown]
# The two CSP log-variance features already show class separation even without a
# classifier.  The scatter matches the synthetic demo: comp 1 is high for
# left-hand imagery and low for right-hand (the top CSP component captures the
# contralateral ERD asymmetry), while comp 4 inverts this.
#
# **Patterns vs filters — one more subtlety.**  The spatial *filter* $w$ is what
# you multiply the data by.  The spatial *pattern* $a$ (what brain source you are
# isolating) is the corresponding column of $(W^{-1})^\top$, or equivalently
# $\Sigma_c w / (w^\top \Sigma_c w)$.  Plotting $w$ on a topomap is **wrong**;
# plotting $a$ is correct.  MNE's `CSP.patterns_` stores $a$ for you.

# %% [markdown]
# ---
# ## 6  ⚠️  A subtler trap: CSP overfits silently when trials are scarce
#
# The pitfall most practitioners discover too late is not the obvious "fit CSP on
# all data" mistake (that is Chapter 12 leakage, already flagged in Chapter 07).
# The subtler trap concerns **the number of CSP components selected**, and it
# operates *inside* an otherwise correct cross-validation loop.
#
# ### The problem: variance of the eigenspectrum
#
# CSP solves a sample-based eigenvalue problem.  With $T$ trials and $C$ channels,
# the sample covariance $\hat\Sigma_k$ has estimation noise of order
# $O(\sqrt{C/T})$.  When $C$ is large (22 channels in BCI IV 2a, 64 in PhysioNet)
# and $T$ is small (tens of trials per subject), the *order* of eigenvectors can
# flip: the filter that happened to capture the most class-1 variance in this
# fold's training set may be tracking a noise direction, not the true motor-imagery
# source.
#
# ### The experiment that reveals it
#
# A common workflow (wrong, but easy to make):
#
# 1. Fit CSP on all labelled training data in a CV fold.
# 2. **Select** the $k$ components (out of all $C$) that have the highest
#    test-set CSP log-variance separation — i.e. use test performance to choose $k$.
# 3. Report the best test accuracy.
#
# Step 2 peeks at the test set and inflates the result.  The correct approach is to
# either (a) fix $k$ before the loop on biological grounds (e.g., always use the
# 2 extreme components), or (b) select $k$ via an *inner* CV loop on the training
# fold only.
#
# ### Why this is non-obvious
#
# - The model is still fit on training data only — the standard "no leakage" check
#   passes.
# - The bias appears only in the **model-selection** step, which feels like a
#   hyperparameter choice, not a data split.
# - With 22 channels you have 22 CSP components; the probability of at least one
#   noise direction appearing "good" on a held-out fold purely by chance is high
#   when trials are scarce.
# - Regularised CSP variants (Ledoit-Wolf shrinkage of $\hat\Sigma_k$,
#   `mne.decoding.CSP(reg="ledoit_wolf")`) reduce the estimation noise but do not
#   eliminate the selection bias — they must still be combined with proper
#   inner-fold component selection.
#
# ### The correct recipe
#
# ```python
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from mne.decoding import CSP
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# pipe = Pipeline([
#     ("csp", CSP(reg="ledoit_wolf", log=True)),
#     ("lda", LinearDiscriminantAnalysis()),
# ])
#
# # Inner CV selects n_components; outer CV estimates generalisation.
# param_grid = {"csp__n_components": [2, 4, 6, 8]}
# inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#
# # Outer loop (call cross_val_score on the GridSearchCV object):
# # scores = cross_val_score(GridSearchCV(pipe, param_grid, cv=inner_cv),
# #                          X, y, cv=outer_cv)
# ```
#
# The outer loop sees only the test performance of the *inner-optimised*
# pipeline; the inner loop never touches the outer test fold.  This nested CV
# adds compute (25 fits for the grid shown) but is the only honest way to report
# accuracy when $k$ was tuned.
#
# **Bottom line:** CSP's silent overfitter is not in the spatial filter fit itself
# but in the dimensionality choice.  Treat the number of CSP components as a
# hyperparameter and select it with an inner cross-validation loop — not by
# peeking at test performance.
