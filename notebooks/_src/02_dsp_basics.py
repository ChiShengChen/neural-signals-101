# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 02 — Digital Signal Processing (DSP) Basics
#
# **Do not skip this chapter.** Almost every later mistake (and every later
# feature) rests on these ideas. We will build each one with a picture.
#
# ## Learning objectives
# 1. The **sampling theorem** and **aliasing**: why sampling rate matters.
# 2. **Quantization**: turning continuous voltage into numbers.
# 3. **Filters** (FIR vs IIR), **band-pass**, and the **50/60 Hz notch**.
# 4. **Re-referencing** and **montage**: what "channel value" even means in EEG.
#
# **Runtime:** ~1 min (mostly synthetic signals so the maths is crystal clear).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
from neuro101 import preprocessing as pp

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Sampling: turning a continuous signal into numbers
#
# A recording device measures voltage **`sfreq`** times per second (the *sampling
# rate*, in Hz). The **Nyquist frequency** is half the sampling rate: it is the
# highest frequency you can faithfully represent.
#
# ### Aliasing: the danger of sampling too slowly
# If a signal contains a frequency *above* Nyquist, it doesn't disappear — it
# masquerades as a *lower* frequency. That fake low frequency is an **alias**, and
# you can never undo it. Below: a 30 Hz sine sampled at 40 Hz (Nyquist = 20 Hz)
# looks like a slow ~10 Hz wave.

# %%
f_true = 30.0       # true frequency (Hz)
fs_low = 40.0       # too-low sampling rate -> Nyquist = 20 Hz < 30 Hz
fs_high = 1000.0    # "continuous" reference

t_cont = np.arange(0, 0.5, 1 / fs_high)
t_samp = np.arange(0, 0.5, 1 / fs_low)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t_cont, np.sin(2 * np.pi * f_true * t_cont), color="#bbb", label="true 30 Hz signal")
ax.plot(t_samp, np.sin(2 * np.pi * f_true * t_samp), "o-", color="#c44e52",
        label="sampled at 40 Hz (looks ~10 Hz!)")
ax.set(xlabel="Time (s)", title="Aliasing: a 30 Hz wave sampled at 40 Hz fakes a slow wave")
ax.legend()
plt.show()

# %% [markdown]
# **Takeaway:** always sample at *more than twice* the highest frequency you care
# about, and use an **anti-aliasing low-pass filter** before downsampling. Real
# amplifiers do this in hardware; if you downsample in software, filter first.

# %% [markdown]
# ## 2. Quantization: voltage becomes integers
#
# An analog-to-digital converter rounds each sample to the nearest of a finite set
# of levels (e.g. 16-bit = 65,536 levels). Too few bits adds **quantization
# noise** — a staircase error. Modern EEG uses 16–24 bits, so this is usually
# negligible, but it's good to *see* it.

# %%
t = np.arange(0, 0.1, 1 / 1000)
clean = np.sin(2 * np.pi * 40 * t)
def quantize(x, n_bits):
    levels = 2 ** n_bits
    return np.round((x + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, clean, color="#bbb", label="continuous")
ax.step(t, quantize(clean, 3), where="mid", color="#c44e52", label="3-bit (8 levels)")
ax.set(xlabel="Time (s)", title="Quantization with only 3 bits — visible staircase noise")
ax.legend()
plt.show()

# %% [markdown]
# ## 3. Filters: keep the frequencies you want
#
# A **filter** changes how much of each frequency passes through.
# - **Low-pass**: keeps low frequencies. **High-pass**: keeps high frequencies.
# - **Band-pass**: keeps a band (e.g. 8–30 Hz for motor imagery).
# - **Notch / band-stop**: removes a narrow band (e.g. 50/60 Hz mains hum).
#
# Two families:
# - **FIR** (Finite Impulse Response): always stable, exactly linear phase (no
#   distortion of timing), but needs many coefficients (slower).
# - **IIR** (Infinite Impulse Response): cheap and sharp, but can distort phase.
#   We use **zero-phase** filtering (`filtfilt`, forward+backward) to cancel the
#   phase distortion — at the cost of being non-causal (uses future samples), which
#   is fine for offline analysis.
#
# Let's build a noisy signal and clean it.

# %%
sf = 250.0
t = np.arange(0, 4, 1 / sf)
alpha = np.sin(2 * np.pi * 10 * t)              # 10 Hz brain rhythm we want
mains = 0.8 * np.sin(2 * np.pi * 50 * t)        # 50 Hz mains hum (unwanted)
drift = 2.0 * np.sin(2 * np.pi * 0.3 * t)       # slow drift (unwanted)
noise = 0.3 * rng.standard_normal(t.size)
signal = (alpha + mains + drift + noise)[None, None, :]  # shape (1,1,T) for our helpers

# Band-pass 8-30 Hz removes both the slow drift and most of the 50 Hz.
bp = pp.bandpass_filter(signal, sf, 8, 30)[0, 0]
# Notch is the targeted tool for mains specifically.
notched = pp.notch_filter(signal, sf, 50.0)[0, 0]

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, signal[0, 0], color="#333"); axes[0].set_title("Raw: alpha + 50 Hz + drift + noise")
axes[1].plot(t, notched, color="#dd8452"); axes[1].set_title("After 50 Hz notch (drift remains)")
axes[2].plot(t, bp, color="#2e8b57"); axes[2].set_title("After 8–30 Hz band-pass (clean ~10 Hz)")
axes[2].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### See the filter work in the frequency domain
# The clearest proof a filter did its job is the **power spectrum** before/after.

# %%
from scipy.signal import welch
f0, p_raw = welch(signal[0, 0], sf, nperseg=int(sf))
_, p_bp = welch(bp, sf, nperseg=int(sf))
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.semilogy(f0, p_raw, label="raw", color="#333")
ax.semilogy(f0, p_bp, label="band-passed 8–30 Hz", color="#2e8b57")
for fx in (0.3, 50): ax.axvline(fx, ls="--", color="gray", lw=0.8)
ax.set(xlim=(0, 70), xlabel="Frequency (Hz)", ylabel="Power",
       title="Band-pass removes the 0.3 Hz drift and the 50 Hz mains line")
ax.legend(); plt.show()

# %% [markdown]
# ## 4. Re-referencing & montage
#
# EEG measures **voltage differences**, so every number is "channel minus
# reference". Choosing the reference changes all values consistently:
# - **Common reference** (e.g. a mastoid electrode): simple but biased toward that site.
# - **Average reference**: subtract the mean across all electrodes at each instant
#   — a common, location-neutral default.
# - **Bipolar / Laplacian**: differences between neighbours; sharpens local activity.
#
# The **montage** is the map from channel names (Cz, C3, ...) to 3-D positions on
# the head. It's needed for topographic plots and for spatial methods like CSP.
#
# Below: a toy demo of average referencing on multi-channel data.

# %%
n_ch = 5
common_brain = np.sin(2 * np.pi * 10 * t)         # shared signal seen by all
data = np.stack([common_brain + 0.2 * rng.standard_normal(t.size) + ch
                 for ch in range(n_ch)])           # each channel has an offset
avg_ref = data - data.mean(axis=0, keepdims=True)  # average reference

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ch in range(n_ch):
    axes[0].plot(t[:250], data[ch, :250] + ch, lw=0.7)
    axes[1].plot(t[:250], avg_ref[ch, :250] + ch, lw=0.7)
axes[0].set_title("Before: per-channel offsets dominate")
axes[1].set_title("After average reference: shared offset removed")
for a in axes: a.set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Downsampling without an anti-aliasing filter.** You permanently fold
#   high-frequency content into your band of interest. Always low-pass first.
# - **Filtering too aggressively, then doing connectivity/causality.** Sharp IIR
#   filters distort phase; phase-based features (PLV, Chapter 05) become garbage.
#   Use zero-phase (`filtfilt`) or FIR linear-phase filters.
# - **High-pass at too high a cutoff (e.g. 2 Hz) for slow signals (ERPs).** You can
#   filter away the very effect you want to measure.
# - **Forgetting the reference.** Two papers reporting "channel Cz" can mean
#   different things if their references differ. Always state your reference & montage.
# - **Notching at 50 Hz when your mains is 60 Hz (or vice-versa).** Match your
#   region: 50 Hz in Europe/Asia/Africa, 60 Hz in the Americas.
#
# **Next:** Chapter 03 — removing real biological artefacts (blinks, muscle) with
# ICA and friends, on the *same* segment so you can see before/after.
