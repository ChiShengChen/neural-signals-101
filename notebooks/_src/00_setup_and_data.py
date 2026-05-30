# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/00_setup_and_data.ipynb)
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
# # Chapter 00 — Setup & Data
#
# **Welcome!** This is the first notebook of *ML & Signal Processing on Neural
# Signals 101*. By the end you will have working code that downloads real public
# brain recordings and plots them, so you have something to play with immediately.
#
# ## Learning objectives
# 1. Understand the Python ecosystem we use (what each library is *for*).
# 2. Recognise the common neural-data **file formats** (EDF, BDF, FIF, BrainVision).
# 3. Download and plot a sample from each public dataset used in this tutorial.
# 4. Understand how this repo keeps everything **reproducible** and **CPU-friendly**.
#
# > **Prerequisites:** none — start here.
# > **Difficulty:** ★☆☆☆☆
# > **Runtime:** ~3–5 min the *first* time (downloads are cached afterwards). In
# > "smoke mode" (`NEURO101_SMOKE=1`, used by CI) the largest download is skipped.
#
# No prior neuroscience needed. Every term is defined on first use. New to the
# jargon? Keep [`docs/GLOSSARY.md`](../docs/GLOSSARY.md) open in another tab.

# %% [markdown]
# ## The Python ecosystem we use
#
# | Library | What it does for us |
# |---|---|
# | **numpy / scipy** | arrays and signal processing (filters, FFT) |
# | **matplotlib** | plotting |
# | **mne** | the standard library for EEG/MEG: loading, filtering, epoching, plotting |
# | **scikit-learn** | classical machine learning + the *correct* cross-validation tools |
# | **pytorch / braindecode** | deep learning models built for EEG |
# | **pyriemann** | covariance / Riemannian features (a strong modern BCI baseline) |
# | **moabb** | one-line access to standard BCI datasets |
#
# Our own helper package is **`neuro101`** (in `src/`). It holds the loaders,
# preprocessing, features and — most importantly — the *leakage-safe* evaluation
# tools every later chapter relies on.

# %%
# Bootstrap: make `import neuro101` work whether or not you ran `pip install -e .`
import sys, os
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
    import neuro101  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from neuro101 import datasets as ds
from neuro101 import io, viz

SMOKE = ds.is_smoke()  # True in CI -> use the tiniest data slices
print("neuro101 version:", neuro101.__version__)
print("smoke mode:", SMOKE, "| data cache:", ds.cache_dir())

# %% [markdown]
# ## The datasets (and their sizes)
#
# Nothing is bundled with this repo — every dataset is **downloaded on first use
# and cached**. Here is the full registry, including approximate download sizes
# so you know what you are committing to before you fetch gigabytes.

# %%
print(ds.describe())

# %% [markdown]
# ## File formats you will meet
#
# Neural recordings come in a handful of container formats. They all ultimately
# store *samples × channels* plus metadata (channel names, sampling rate, events):
#
# - **EDF / EDF+** (`.edf`) — *European Data Format*. Very common for EEG and
#   sleep data (PhysioNet uses it). Open and widely supported.
# - **BDF** (`.bdf`) — BioSemi's 24-bit variant of EDF.
# - **FIF** (`.fif`) — Elekta/MEGIN format, the native format of MNE (MEG+EEG).
# - **BrainVision** (`.vhdr` + `.eeg` + `.vmrk`) — a 3-file format from Brain
#   Products; `.vhdr` is the header you actually open.
# - **GDF** (`.gdf`) — used by some BCI competition datasets.
#
# **You rarely parse these by hand.** MNE has a `read_raw_*` function for each,
# and they all return the same kind of object: an MNE `Raw`.

# %% [markdown]
# ## Dataset 1 — PhysioNet Motor Imagery (small, downloads fast)
#
# 64-channel EEG while subjects imagine moving their hands/feet. We load one
# subject and plot a few seconds of a few channels. This is "raw EEG": wiggly
# voltage traces over time, one per electrode.

# %%
X_pn, y_pn, subj_pn = io.load_physionet_mi_epochs(n_subjects=1)
print("PhysioNet epochs:", X_pn.shape, "labels:", np.bincount(y_pn))

# Plot the first epoch, first 3 channels, stacked with vertical offsets.
sf_pn = ds.DATASETS["physionet_mi"].sfreq_hz
fig, ax = plt.subplots(figsize=(9, 3.5))
offsets = np.arange(3) * 50e-6  # 50 µV spacing so traces don't overlap
for ch in range(3):
    ax.plot(np.arange(X_pn.shape[-1]) / sf_pn,
            X_pn[0, ch] + offsets[ch], lw=0.7)
ax.set(xlabel="Time (s)", ylabel="Channel (offset)", title="PhysioNet MI — raw EEG (1 epoch, 3 channels)")
plt.show()

# %% [markdown]
# ## Dataset 2 — BCI Competition IV 2a (the headline dataset)
#
# 22-channel EEG, 9 subjects, 4-class motor imagery. We use the **left-hand vs
# right-hand** subset throughout the tutorial. Below we load a minimal slice and
# show the shape of the data arrays we will feed to models:
# `(trials, channels, time)`.

# %%
n_subj = 2 if SMOKE else 2  # keep Ch00 quick; later chapters load more
Xb, yb, subjb = io.load_bnci_2a_epochs(n_subjects=n_subj)
print("BCI IV 2a epochs:", Xb.shape, "| classes (0=left,1=right):", np.bincount(yb))
print("subjects present:", np.unique(subjb))

sf_b = ds.DATASETS["bnci_2a"].sfreq_hz
viz.plot_signal(Xb[0, 0], sf_b, title="BCI IV 2a — one channel of one motor-imagery trial",
                units="µV (scaled)")
plt.show()

# %% [markdown]
# ## ⭐ The array-shape mental model (read this twice)
#
# Almost every beginner bug in EEG code is a **shape bug**: an axis wired the wrong
# way, a silent broadcast, or a stray transpose. Build the mental model now and
# you will save yourself hours later.
#
# Our epoched data is always a **3-D array** with this meaning:
#
# ```
#            axis 0          axis 1           axis 2
#   X  =  ( n_trials  ,   n_channels   ,    n_times   )
#          "which        "which            "which time
#           trial?"       electrode?"       sample?"
# ```
#
# - **`X[i]`** → one trial: a 2-D `(channels, times)` slice.
# - **`X[i, c]`** → one channel of one trial: a 1-D time series (what we plotted).
# - **`X[:, c, :]`** → channel `c` for *every* trial.
# - **`X.mean(axis=2)`** → average over **time** → `(trials, channels)`.
# - **`X.mean(axis=0)`** → average over **trials** → `(channels, times)` (an ERP).
#
# The labels line up with **axis 0**: `y` has shape `(n_trials,)`, and `y[i]` is the
# label of `X[i]`. Keep that pairing sacred — if you ever reorder trials, reorder
# `y` the same way.

# %%
print("X shape:", Xb.shape, " -> (n_trials, n_channels, n_times)")
print("one trial   X[0]    :", Xb[0].shape, "  (channels, times)")
print("one channel X[0, 0] :", Xb[0, 0].shape, "  (times,) -> a 1-D signal")
print("labels      y       :", yb.shape, " -> one label per trial; y[0] =", yb[0])

# A classic silent bug: averaging over the WRONG axis.
over_time = Xb.mean(axis=2)      # correct: collapse time -> (trials, channels)
over_trials = Xb.mean(axis=0)    # different meaning: an average trial (channels, times)
print("\nmean over time  (axis=2):", over_time.shape, "= one number per channel per trial")
print("mean over trials(axis=0):", over_trials.shape, "= the average trial (an ERP)")
print("These are NOT interchangeable — picking the wrong axis silently gives garbage.")

# %% [markdown]
# > **Broadcasting & transpose traps.** NumPy will happily *broadcast* a
# > `(channels, 1)` array against `(channels, times)` without complaint — convenient,
# > but it also means a wrong-shaped array can produce numbers instead of an error.
# > And `X[i].T` (transpose to `(times, channels)`) is exactly what some libraries
# > want and others reject — always check what shape a function expects. **When a
# > result looks weird, print `.shape` first.** It is the fastest debugging habit in
# > all of neural-signal ML. (See `docs/TROUBLESHOOTING.md` for the common errors.)

# %% [markdown]
# ## Dataset 3 — Sleep-EDF (used later for sleep staging & class imbalance)
#
# Overnight recordings with expert **sleep-stage** labels. We load one night and
# show how many 30-second epochs fall in each stage — notice the classes are very
# **imbalanced** (lots of N2). We will use exactly this imbalance in Chapter 12
# to show why *accuracy* can lie.

# %%
Xs, ys, subjs = io.load_sleep_edf_epochs(n_subjects=1)
from neuro101.io import SLEEP_STAGE_NAMES
counts = np.bincount(ys, minlength=len(SLEEP_STAGE_NAMES))
print("Sleep-EDF epochs:", Xs.shape)
for name, c in zip(SLEEP_STAGE_NAMES, counts):
    print(f"  {name:4s}: {c}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(SLEEP_STAGE_NAMES, counts, color="#4c72b0")
ax.set(title="Sleep-EDF — epochs per stage (note the imbalance)", ylabel="count")
plt.show()

# %% [markdown]
# ## Dataset 4 — MNE sample (MEG+EEG; large, optional)
#
# The classic MNE teaching dataset (audio/visual task). It is **~1.5 GB**, so in
# smoke/CI mode we skip the download and just describe it. Run this locally to
# see a real MEG+EEG recording and its event markers.

# %%
if SMOKE:
    print("Smoke mode: skipping the 1.5 GB MNE sample download.")
    print("Run locally (without NEURO101_SMOKE=1) to fetch and plot it.")
else:
    raw = io.load_mne_sample_raw()
    print(raw)
    events, event_id = io.load_mne_sample_events(raw)
    print("event types:", event_id, "| n events:", len(events))
    # Plot 5 seconds of a few EEG channels.
    picks = raw.copy().pick("eeg").ch_names[:4]
    raw.copy().pick(picks).plot(duration=5.0, n_channels=4, show=True, block=False)

# %% [markdown]
# ## How this repo stays reproducible & fast
#
# - **Pinned `requirements.txt`** and a Python 3.11 venv (`make setup`).
# - **Seeded RNGs** everywhere (you will see `random_state=0`).
# - **Subsampling**: loaders read the env var `NEURO101_SMOKE` and shrink the data
#   so every notebook runs on a laptop CPU in a few minutes. We always *tell you*
#   when we subsample.
# - **Cached downloads** in `~/neuro101_data` (override with `NEURO101_DATA`).

# %% [markdown]
# ## ✅ Concept check
#
# 1. An epoched array has shape `(n_trials, n_channels, n_times)`. Which axis do you
#    average over to get one number **per channel per trial**?
# 2. Why must `X` and `y` agree on axis 0, and what breaks if you reorder trials in
#    `X` but not in `y`?
# 3. Two datasets here have different sampling rates (e.g. 160 vs 250 Hz). Why does a
#    filter or FFT written for one rate misbehave on the other?
#
# **Answers:** (1) `axis=2` (time) → `(n_trials, n_channels)`. (2) `y[i]` is the label
# of `X[i]`; reordering one without the other silently mislabels every trial. (3) Both
# operate in units of *fraction of the sampling rate*, so the same cutoff in Hz maps to
# a different normalized frequency at a different `sfreq` — always carry `sfreq` with the data.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Assuming data lives on your disk.** Never hard-code `/Users/me/eeg.edf`.
#   Always download via code so others can reproduce your work. (This repo never
#   reads a local file you have to provide.)
# - **Ignoring the sampling rate.** Every dataset here has a *different* sampling
#   rate (160, 250, 100 Hz...). A filter or FFT written for one rate is wrong for
#   another. Always carry `sfreq` alongside your data.
# - **Plant-then-forget downloads.** A 1.5 GB download in a notebook that runs in
#   CI will time out. Guard big downloads (as we did for the MNE sample) and
#   document sizes.
# - **Editing the generated `.ipynb`.** In this repo the *source* of truth is
#   `notebooks/_src/*.py`. Edit those and run `make notebooks` (see CONTRIBUTING).
#
# **Next:** Chapter 01 — what these signals physically *are*, and where the noise
# comes from.
