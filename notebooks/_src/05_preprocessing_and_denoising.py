# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/05_preprocessing_and_denoising.ipynb)
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
# # Chapter 05 — Preprocessing & Denoising
#
# Real recordings are full of **artefacts**: signals that are not brain activity.
# This chapter shows how to spot and remove the big ones, and ends with a
# **before/after comparison on the same segment** so you can see the cleaning work.
#
# ## Learning objectives
# 1. Identify common artefacts: **EOG** (eyes), **EMG** (muscle), **motion**, **line noise**.
# 2. Use **ICA** (Independent Component Analysis) to remove eye-blink artefacts.
# 3. Understand **ASR**-style amplitude cleaning, **epoching** and **baseline correction**.
# 4. Compare the same segment before and after cleaning.
#
# > **Prerequisites:** Chapter 04.
# > **Difficulty:** ★★★☆☆
#
# **Runtime:** ~1–2 min (one small PhysioNet subject; ICA on CPU).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")
from neuro101 import io, preprocessing as pp

# %% [markdown]
# ## A catalogue of artefacts
#
# | Artefact | Looks like | Frequency | Fix |
# |---|---|---|---|
# | **Eye blink (EOG)** | large slow deflection, frontal | < 4 Hz | ICA / regression |
# | **Muscle (EMG)** | bursts of high-frequency fuzz | > 20 Hz | ICA / reject epoch |
# | **Motion / electrode pop** | sudden jumps / steps | broadband | reject / ASR-style clip |
# | **Line noise** | constant hum | exactly 50/60 Hz | notch filter (Ch 02) |
# | **Heartbeat (ECG)** | regular ~1 Hz spikes | ~1 Hz | ICA |
#
# We load one subject's continuous recording and clean it.

# %%
raw = io.load_physionet_mi_raw(subject=1)
raw_clean_filt = pp.basic_clean_raw(raw, l_freq=1.0, h_freq=40.0, notch=60.0)
print(raw_clean_filt)
print("channels:", raw_clean_filt.ch_names[:6], "...")

# %% [markdown]
# ## ICA: separate the recording into independent sources
#
# ICA assumes the recording is a mixture of statistically independent sources
# (brain rhythms + a blink source + a heartbeat source + ...). It **unmixes**
# them so we can zero out the artefact sources and mix the rest back. We use a
# frontal electrode (Fpz/Fp1) as an eye-blink proxy because this dataset has no
# dedicated EOG channel.

# %%
ica = pp.fit_ica(raw_clean_filt, n_components=15, random_state=0)

# Find components that look like eye movements using a frontal channel proxy.
frontal = next((c for c in ("Fpz", "Fp1", "Fp2", "AFz") if c in raw_clean_filt.ch_names), None)
eog_idx = pp.detect_eog_components(ica, raw_clean_filt, ch_name=frontal)
print("frontal proxy channel:", frontal, "| ICA components flagged as eye-related:", eog_idx)

# If nothing was flagged automatically (data-dependent), fall back to the
# component most correlated with the frontal channel so the demo always shows
# *something* being removed.
if not eog_idx and frontal is not None:
    sources = ica.get_sources(raw_clean_filt).get_data()
    frontal_sig = raw_clean_filt.copy().pick([frontal]).get_data()[0]
    corr = [abs(np.corrcoef(sources[i], frontal_sig)[0, 1]) for i in range(sources.shape[0])]
    eog_idx = [int(np.argmax(corr))]
    print("fallback: removing most frontal-correlated component:", eog_idx)

# %% [markdown]
# ## Before / after on the SAME segment
#
# We apply ICA (removing the flagged components) and overlay the same frontal
# channel before and after. Blink deflections should shrink.
#
# > **Before running:** guess whether the ICA removal will completely eliminate the
# > blink peaks on the frontal channel, or only reduce them — and why a perfect
# > removal might actually be a warning sign of over-cleaning.

# %%
raw_ica = raw_clean_filt.copy()
ica.exclude = eog_idx
ica.apply(raw_ica)

ch = frontal or raw_clean_filt.ch_names[0]
seg = slice(0, int(10 * raw_clean_filt.info["sfreq"]))  # first 10 s
t = np.arange(seg.stop) / raw_clean_filt.info["sfreq"]
before = raw_clean_filt.copy().pick([ch]).get_data()[0][seg] * 1e6
after = raw_ica.copy().pick([ch]).get_data()[0][seg] * 1e6

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(t, before, color="#c44e52", lw=0.8, label="before ICA (blinks present)")
ax.plot(t, after, color="#2e8b57", lw=0.8, label="after ICA (blinks reduced)")
ax.set(xlabel="Time (s)", ylabel="µV", title=f"ICA artefact removal on channel {ch}")
ax.legend(); plt.show()

# %% [markdown]
# ## ASR-style amplitude cleaning (a transparent stand-in)
#
# **ASR** (Artifact Subspace Reconstruction) reconstructs badly corrupted
# segments from a clean reference subspace. A full ASR is beyond this chapter, but
# the *spirit* is "tame extreme excursions". Our `clip_extreme_amplitudes`
# robustly clips spikes — here on a segment with an injected motion artefact.

# %%
sig = raw_clean_filt.copy().pick([ch]).get_data()[0][:1000].copy()
sig[500:505] += 400e-6  # inject a big motion spike (µV scale)
cleaned = pp.clip_extreme_amplitudes(sig[None, None, :], z_thresh=5.0)[0, 0]
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(sig * 1e6, color="#c44e52", label="with motion spike")
ax.plot(cleaned * 1e6, color="#2e8b57", label="after robust clipping")
ax.set(title="ASR-style clipping tames extreme amplitudes", ylabel="µV"); ax.legend()
plt.show()

# %% [markdown]
# ## Epoching & baseline correction
#
# **Epoching** cuts the continuous recording into labelled trials around events.
# **Baseline correction** subtracts the mean of a pre-event window from each
# epoch, so all epochs start at a common zero — this removes slow drifts that
# would otherwise swamp the effect.

# %%
events, _ = mne.events_from_annotations(raw_clean_filt, event_id=dict(T1=0, T2=1))
epochs = pp.make_epochs(
    raw_clean_filt, events, dict(left=0, right=1),
    tmin=-0.5, tmax=2.0, baseline=(None, 0),
)
print("epochs:", epochs.get_data(copy=False).shape, "(trials, channels, time)")
evoked = epochs["left"].average()
fig = evoked.plot(spatial_colors=False, show=False)
plt.show()

# %% [markdown]
# ## ✅ Concept check
#
# 1. ICA decomposes the recording into independent components. Why must the number of
#    components be ≤ the number of EEG channels?
# 2. Baseline correction subtracts the mean of a pre-stimulus window. What artefact
#    does this remove, and what assumption does it make about that artefact?
# 3. You fit ICA on the entire continuous recording (before cross-validation) and use
#    the resulting component weights as features. Why does this constitute data leakage?
#
# **Answers:**
# 1. ICA solves a system with as many equations as channels; you cannot recover more
#    independent sources than you have observed mixtures (channels).
# 2. Baseline correction removes slow DC drift/offset; it assumes the drift is
#    constant (or slowly varying) within the baseline window and continues into
#    the epoch — which may not hold for rapid drift changes.
# 3. ICA learned the mixing matrix using signal statistics from all trials, including
#    the future test trials. Component activations on test data are therefore
#    influenced by test-set statistics, violating train/test independence.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Removing too many ICA components.** Each component you delete also removes
#   some brain signal. Remove the few clear artefacts, not "everything that looks weird".
# - **Fitting ICA / artefact thresholds on the *whole* dataset, then classifying.**
#   ICA is *unsupervised* so this is less dangerous than supervised leakage, but if
#   your downstream feature depends on it, fit cleaning **inside** your train fold.
#   (We make this airtight in Chapter 08 with pipelines.)
# - **Rejecting epochs by eye after seeing the labels.** That biases results. Set
#   rejection thresholds *before* looking at class membership.
# - **Skipping baseline correction for ERPs.** Slow drift will dominate and your
#   averaged waveform will be meaningless.
# - **Over-cleaning.** The goal is *honest* data, not pretty data. A model that only
#   works on heavily hand-cleaned data won't survive real-time use.
#
# **Next:** Chapter 06 — the frequency domain (FFT, PSD, spectrograms, wavelets)
# and the time–frequency trade-off.
