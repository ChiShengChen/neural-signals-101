# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 01 — What Neural Signals Are
#
# Before we filter or classify anything, let's build intuition for *what we are
# even measuring* and *where the noise comes from*.
#
# ## Learning objectives
# 1. Name the main neural signal types and their **physical origin**.
# 2. Compare them on **spatial scale**, **temporal scale**, and **signal-to-noise
#    ratio (SNR)** — how strong the signal is relative to the noise.
# 3. Look at a real EEG trace and *point at* the noise.
#
# **Runtime:** ~1 min (uses one small PhysioNet subject already cached from Ch00).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
from neuro101 import io, datasets as ds, features as ft

# %% [markdown]
# ## A field guide to neural signals
#
# All of these are ways of measuring the electrical (or metabolic) activity of
# neurons, but at very different scales and with very different trade-offs.
#
# | Signal | What it measures | Where it's recorded | Spatial scale | Temporal scale | SNR |
# |---|---|---|---|---|---|
# | **EEG** | summed electrical activity of many neurons | scalp (non-invasive) | cm (blurry) | ms (fast) | **low** |
# | **MEG** | magnetic fields from the same currents | outside the head | cm | ms | low–medium |
# | **ECoG** | electrical activity from the cortical surface | on the brain (surgery) | mm | ms | high |
# | **LFP** | local field potential of a small population | electrode in tissue | sub-mm | ms | high |
# | **Spikes** | individual neuron action potentials | electrode next to a cell | single neuron | sub-ms | high |
# | **fNIRS** | blood oxygenation (proxy for activity) | scalp (light) | cm | **seconds** (slow) | medium |
# | **EMG** | muscle electrical activity | skin over muscle | muscle | ms | high |
#
# **The core trade-off:** the less invasive the method, the more the signal is
# blurred and buried in noise. EEG is easy to record but has *low SNR*; spikes
# have beautiful SNR but require putting an electrode next to a neuron.

# %% [markdown]
# ## Orders of magnitude (why units matter)
#
# - **EEG**: tens of **microvolts** (µV, millionths of a volt). A blink is ~100 µV
#   and *dwarfs* the brain signal you care about.
# - **Spikes**: ~100 µV but at the electrode tip, sharp and brief (~1 ms).
# - **fNIRS**: changes over **seconds**, because blood flow is slow.
#
# Mixing these scales up (e.g. treating EEG like an audio signal) is a classic
# beginner error. Let's look at real EEG.

# %%
X, y, subj = io.load_physionet_mi_epochs(n_subjects=1)
sf = ds.DATASETS["physionet_mi"].sfreq_hz
print(f"Loaded {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} samples each @ {sf} Hz")

# %% [markdown]
# ## Where is the noise? Look at one channel
#
# Real EEG is a sum of: (1) brain rhythms you want, (2) **biological artefacts**
# (eye blinks, muscle, heartbeat), and (3) **environmental noise** (50/60 Hz mains
# hum, electrode drift). The plot below shows one channel; the annotations point
# at the usual suspects.

# %%
trial = X[0, 0] * 1e6  # convert to µV for readable axis
t = np.arange(trial.size) / sf

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t, trial, lw=0.8, color="#333")
ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)",
       title="One EEG channel — a mix of brain signal and noise")
ax.axhline(0, color="gray", lw=0.5)
plt.show()

# %% [markdown]
# ## Make the noise visible: where does the power live?
#
# A quick way to "see" noise is to look at how power is spread across frequencies.
# Brain rhythms cluster at low frequencies (< ~30 Hz); muscle artefacts spread to
# high frequencies; mains hum is a sharp spike at exactly 50 or 60 Hz.

# %%
from scipy.signal import welch
freqs, psd = welch(X[:, 0, :], fs=sf, nperseg=int(sf))  # PSD per epoch, channel 0
mean_psd = psd.mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.semilogy(freqs, mean_psd, color="#c44e52")
for f0, name in [(10, "alpha (~10 Hz)\nbrain rhythm"), (60, "60 Hz\nmains hum (US)")]:
    ax.axvline(f0, ls="--", color="gray", lw=1)
    ax.text(f0 + 1, mean_psd.max() * 0.3, name, fontsize=8)
ax.set(xlim=(0, 80), xlabel="Frequency (Hz)", ylabel="Power (log)",
       title="Power spectrum — brain rhythms vs noise")
plt.show()

# %% [markdown]
# The hump around ~10 Hz is the **alpha rhythm** (brain). Any sharp line at 50/60
# Hz is **mains noise** (we remove it with a notch filter in Chapter 02). Broadband
# power climbing at high frequency is often **muscle (EMG)** contamination.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Assuming a clean signal.** EEG is *mostly* noise from your model's point of
#   view. The single biggest determinant of BCI accuracy is often artefact
#   handling, not the classifier.
# - **Confusing temporal scales.** fNIRS changes over seconds; spikes over a
#   millisecond. A method (or sampling rate) tuned for one is wrong for another.
# - **Treating channels as independent sensors of different things.** Nearby EEG
#   electrodes see *overlapping* sources (volume conduction) — they are highly
#   correlated. This is exactly why we cannot shuffle samples freely later.
# - **Ignoring units.** Reporting "amplitude 0.00007" instead of "70 µV" hides
#   bugs. Keep physical units; sanity-check that EEG is tens of µV.
#
# **Next:** Chapter 02 — the digital signal processing foundations (sampling,
# aliasing, filters). Do not skip it.
