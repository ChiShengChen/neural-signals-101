# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/07_feature_engineering.ipynb)
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
# # Chapter 07 — Feature Engineering
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
# > **Prerequisites:** Chapters 03 and 06.
# > **Difficulty:** ★★★★☆
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
#   Chapter 08 with pipelines; here we just build them.

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
# **Reading the plot above (ERD in action):** BCI Competition IV 2a records left-hand
# vs right-hand *imagined* movements. When you imagine moving your **left** hand, the
# motor cortex on the **right** hemisphere (contralateral) decreases its beta power —
# this dip is called **ERD (Event-Related Desynchronization)**. The two coloured lines
# above should diverge most over central channels (roughly the middle of the x-axis,
# near C3/Cz/C4): that divergence is the contralateral ERD signature a classifier
# learns to exploit.

# %% [markdown]
# ## What these features mean in the brain
#
# Features are not just numbers — each one ties to a specific neuroscientific
# phenomenon. Understanding *why* a feature works makes it much easier to debug a
# broken classifier or design a better one.
#
# ### Band power ↔ brain rhythms
# The brain generates rhythmic electrical oscillations at characteristic frequencies:
# - **Alpha (~8–13 Hz):** strongest when your eyes are closed and you are relaxed
#   ("idling" visual cortex). Opening your eyes suppresses it.
# - **Mu/Beta (~8–30 Hz) over sensorimotor cortex:** the motor system's equivalent
#   of alpha. When you are sitting still the sensorimotor cortex hums along at mu/beta
#   frequencies — a kind of ready-but-idle state.
#
# ### ERD / ERS — the core motor-imagery signal
# **Event-Related Desynchronization (ERD):** as soon as you move *or imagine moving*
# a hand, the mu/beta rhythm over the **opposite** (contralateral) sensorimotor cortex
# **drops sharply in power**. The brain is "waking up" that area, and the synchronised
# idling rhythm breaks apart.
#
# **Event-Related Synchronization (ERS):** after the movement ends, power rebounds —
# sometimes above baseline — as the cortex returns to its resting state.
#
# Why contralateral? The motor cortex is anatomically wired so that the **left**
# hemisphere controls the **right** body side and vice versa. So imagining a left-hand
# movement desynchronises the *right* sensorimotor cortex, and imagining a right-hand
# movement desynchronises the *left* sensorimotor cortex. A motor-imagery BCI
# classifier literally detects which side of the brain has lower beta power.
#
# ### CSP ↔ geometry from Chapter 03
# In Chapter 03 we saw that EEG channels record a **mixture** of underlying sources
# (because electricity spreads through the skull). CSP finds **spatial filters** —
# weighted sums of channels — chosen so that one class has high variance and the other
# has low variance. For left-vs-right hand imagery, the first CSP filter will
# emphasise channels over the *right* sensorimotor cortex (high variance for left-hand
# imagery, low for right), and the last filter will do the opposite. The resulting
# log-variances directly encode the left-vs-right ERD asymmetry in just a handful of
# numbers.
#
# ### Covariance / Riemannian features
# The channel covariance matrix records the **full spatial pattern** of how every pair
# of channels co-varies during a trial. When the right sensorimotor cortex
# desynchronises (left-hand imagery), the correlations involving channels over that
# region change. The Riemannian approach uses the geometry of the space of covariance
# matrices to compare these whole-brain co-activation patterns across trials — no hand
# selection of channels required.

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

# %% [markdown]
# > **Predict before you run:** BCI Competition IV 2a uses left-hand (class 0) and
# > right-hand (class 1) imagined movements. Based on what you just read about ERD:
# >
# > 1. Over which hemisphere do you expect a **beta-power drop** for *left-hand*
# >    imagery — left or right?
# > 2. CSP will learn a spatial filter that gives high variance for one class and low
# >    for the other. Should that filter emphasise channels on the *ipsilateral* (same
# >    side) or *contralateral* (opposite side) hemisphere relative to the imagined hand?
# > 3. Do you expect the two CSP scatter-plot clusters to overlap a lot or separate
# >    cleanly? Why?
# >
# > *Run the cell, then check whether the plot matches your prediction.*

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
# > Chapter 08 we put CSP inside a pipeline so it is re-fit on each training fold.

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
# ## ✅ Concept check
#
# Test your understanding before moving on.
#
# **Question 1 — ERD meaning:** During imagined hand movement, the mu/beta power over
# the contralateral sensorimotor cortex *decreases*. What is the technical term for
# this power decrease, and why is it useful for a BCI?
#
# **Question 2 — Why contralateral?** A participant imagines moving their *left* hand.
# Over which hemisphere (left or right) do you expect the strongest beta-power drop,
# and why?
#
# **Question 3 — What CSP optimises:** CSP finds spatial filters by solving a
# generalised eigenvalue problem on the two class covariance matrices. In plain
# language, what property of the filtered signal does it maximise for one class while
# minimising it for the other?
#
# ---
# **Answers:**
#
# 1. The power decrease is called **ERD (Event-Related Desynchronization)**. It is
#    useful because it is a reliable, spatially specific marker that appears even
#    during *imagined* (not actual) movement — so a classifier can detect movement
#    intent without any physical action.
#
# 2. The strongest beta-power drop occurs over the **right** hemisphere (contralateral
#    to the left hand). The motor cortex is wired contralaterally: left hand →
#    right motor cortex → right hemisphere EEG channels show ERD.
#
# 3. CSP maximises the **variance** (signal power) of the filtered signal for one class
#    while simultaneously minimising it for the other. For left-vs-right hand imagery,
#    this isolates the spatial pattern of channels whose power is high for one class
#    (e.g., right-hemisphere channels during left-hand imagery) and low for the other —
#    directly capturing the contralateral ERD asymmetry.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Fitting CSP / covariance whitening / scalers on all data, then evaluating.**
#   The #1 feature-leakage mistake. Learned features must be fit on the train fold
#   only (Chapter 08 makes this automatic).
# - **Comparing absolute band power across subjects/sessions.** Use relative power
#   or per-subject normalisation (fit on train!).
# - **Throwing thousands of connectivity features at a small dataset.** O(channels²)
#   features + few trials = overfitting. Select channels or reduce dimensionality.
# - **Forgetting units/log.** Power is heavy-tailed; log-power is far friendlier to
#   linear models (our helpers log it for you).
# - **Treating CSP as magic.** CSP assumes the discriminative info is in band-power
#   spatial patterns — great for motor imagery, not for everything.
#
# **Next:** Chapter 08 — classical ML with *proper* cross-validation, where these
# features finally meet a classifier the honest way.
