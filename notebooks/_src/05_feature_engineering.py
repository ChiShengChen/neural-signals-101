# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 05 — Feature Engineering
#
# A classifier is only as good as the numbers you feed it. This chapter turns raw
# epochs into **features**: compact descriptions that expose the structure a model
# can learn from.
#
# ## Learning objectives
# 1. **Time-domain** and **frequency-domain** features (recap + use).
# 2. **Connectivity**: coherence and **PLV** (phase-locking value).
# 3. **CSP** (Common Spatial Patterns) for motor imagery.
# 4. **Riemannian / covariance** features with `pyriemann` — a standard modern baseline.
# 5. Which features *learn* from data (and so must be fit on train only).
#
# **Runtime:** ~1–2 min (a few BCI IV 2a subjects, cached).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
from neuro101 import io, datasets as ds
from neuro101 import features as ft

SMOKE = ds.is_smoke()
n_subj = 2 if SMOKE else 3
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X.shape} (trials, channels, time) @ {sf} Hz | classes={np.bincount(y)}")

# %% [markdown]
# ## Two kinds of features (this distinction matters for honest evaluation)
#
# - **Stateless features** are computed from each trial alone (band power,
#   variance, PLV). Computing them before splitting is *fine* — they don't peek at
#   other trials.
# - **Learned features** estimate parameters from a *set* of labelled trials
#   (CSP learns spatial filters; covariance-whitening estimates a reference). These
#   **must be fit on the training fold only** or they leak. We handle that in
#   Chapter 06 with pipelines; here we just build them.

# %% [markdown]
# ## 1. Time-domain features
#
# Simple per-channel statistics: variance, mean absolute value, and a Hjorth
# "mobility" (how fast the signal wiggles). Cheap and surprisingly useful.

# %%
F_time = ft.time_domain_features(X)
print("time-domain features:", F_time.shape)

# %% [markdown]
# ## 2. Frequency-domain features: band power
#
# Motor imagery suppresses the mu/beta rhythm over the hand area
# (event-related desynchronisation). Band power captures exactly this. Let's see
# whether alpha/beta power over central channels differs between left and right
# imagined hand movements.

# %%
F_bp = ft.bandpower(X, sf)
print("band-power features:", F_bp.shape)

# Average beta power per class (reshaped to band x channel via our layout).
n_ch = X.shape[1]
bp_by_band = F_bp.reshape(F_bp.shape[0], len(ft.BANDS), n_ch)
beta_idx = list(ft.BANDS).index("beta")
fig, ax = plt.subplots(figsize=(9, 3))
for cls, color in [(0, "#4c72b0"), (1, "#dd8452")]:
    ax.plot(bp_by_band[y == cls, beta_idx].mean(0), color=color,
            label=f"class {cls} ({'left' if cls==0 else 'right'} hand)")
ax.set(xlabel="Channel index", ylabel="log beta power",
       title="Mean beta-band power per channel, by class"); ax.legend()
plt.show()

# %% [markdown]
# ## 3. Connectivity: coherence and PLV
#
# Connectivity measures how channels relate, not just their individual power.
# - **Coherence**: correlation in the frequency domain (do two channels share
#   power at the same frequencies?).
# - **PLV** (phase-locking value): are two channels *phase-synchronised*,
#   regardless of amplitude? 1 = locked, 0 = unrelated.
#
# These are O(channels²), so we compute them on a small channel subset for speed.

# %%
picks = slice(0, 6)  # first 6 channels for a quick demo
F_coh = ft.coherence_features(X[:, picks, :], sf)
F_plv = ft.plv_features(X[:, picks, :])
print("coherence features:", F_coh.shape, "| PLV features:", F_plv.shape)

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(F_plv.ravel(), bins=30, color="#55a868")
ax.set(xlabel="PLV", title="Distribution of pairwise PLV (6 channels)")
plt.show()

# %% [markdown]
# ## 4. CSP — Common Spatial Patterns (the motor-imagery workhorse)
#
# CSP **learns** spatial filters (channel mixtures) that make the two classes
# differ as much as possible in variance. The log-variance of the top components
# is a tiny, powerful feature set. Because CSP uses the labels, it is a *learned*
# feature — we fit it on the whole set here only to *visualise* it; in a real
# evaluation it goes inside a train-only pipeline.

# %%
csp = ft.make_csp(n_components=4)
F_csp = csp.fit_transform(X, y)   # demo-only fit on all data (see warning below)
print("CSP features:", F_csp.shape)

# Show the first two CSP features separate the classes.
fig, ax = plt.subplots(figsize=(6, 4))
for cls, color in [(0, "#4c72b0"), (1, "#dd8452")]:
    ax.scatter(F_csp[y == cls, 0], F_csp[y == cls, -1], s=12, alpha=0.6,
               color=color, label=f"class {cls}")
ax.set(xlabel="CSP comp 1 (log-var)", ylabel="CSP comp 4 (log-var)",
       title="CSP makes the classes linearly separable"); ax.legend()
plt.show()

# %% [markdown]
# > ⚠️ **The `fit_transform(X, y)` above used the WHOLE dataset, including labels.**
# > That is *feature leakage* and would inflate any score computed afterwards. It
# > is fine here because we only *plotted* — we did not measure accuracy. In
# > Chapter 06 we put CSP inside a pipeline so it is re-fit on each training fold.

# %% [markdown]
# ## 5. Riemannian / covariance features (a strong modern baseline)
#
# Represent each trial by its **channel covariance matrix** (how channels co-vary).
# These matrices live on a curved space; `pyriemann` projects them to a flat
# **tangent space** where ordinary linear classifiers work well. This "covariance +
# tangent space" recipe is a top BCI baseline and often beats hand-tuned features.

# %%
from pyriemann.estimation import Covariances
covs = Covariances(estimator="oas").fit_transform(X)
print("covariance matrices:", covs.shape, "(trials, channels, channels)")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, cls in zip(axes, [0, 1]):
    ax.imshow(covs[y == cls].mean(0), cmap="RdBu_r")
    ax.set_title(f"Mean covariance — class {cls}")
plt.tight_layout(); plt.show()

# %% [markdown]
# The two class-average covariance matrices differ — that difference is exactly
# what a Riemannian classifier exploits. We build the full `Covariances →
# TangentSpace → classifier` pipeline in the next chapter.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Fitting CSP / covariance whitening / scalers on all data, then evaluating.**
#   The #1 feature-leakage mistake. Learned features must be fit on the train fold
#   only (Chapter 06 makes this automatic).
# - **Comparing absolute band power across subjects/sessions.** Use relative power
#   or per-subject normalisation (fit on train!).
# - **Throwing thousands of connectivity features at a small dataset.** O(channels²)
#   features + few trials = overfitting. Select channels or reduce dimensionality.
# - **Forgetting units/log.** Power is heavy-tailed; log-power is far friendlier to
#   linear models (our helpers log it for you).
# - **Treating CSP as magic.** CSP assumes the discriminative info is in band-power
#   spatial patterns — great for motor imagery, not for everything.
#
# **Next:** Chapter 06 — classical ML with *proper* cross-validation, where these
# features finally meet a classifier the honest way.
