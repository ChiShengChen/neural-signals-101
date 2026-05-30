# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 06 — The Frequency Domain
#
# Brain rhythms live in frequency bands, so frequency analysis is central to
# neural signal processing. We'll build the FFT, Welch PSD, the spectrogram and
# wavelets, and make the **time–frequency trade-off** concrete.
#
# ## Learning objectives
# 1. **FFT** and **PSD** (Welch's method): how power is distributed across frequency.
# 2. **STFT / spectrogram** and **wavelets**: how that distribution changes over time.
# 3. **Band power** (delta/theta/alpha/beta/gamma) as features.
# 4. The **time–frequency trade-off**: you cannot have perfect resolution in both.
#
# > **Prerequisites:** Chapters 03 and 04.
# > **Difficulty:** ★★★☆☆
#
# **Runtime:** ~1 min.

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, stft
from neuro101 import io, datasets as ds, viz
from neuro101.features import bandpower, BANDS

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. The FFT: a signal as a sum of sine waves
#
# The **Fourier transform** rewrites a signal as a sum of sine waves of different
# frequencies. The **FFT** computes it fast. Below we make a signal with known
# 6 Hz and 13 Hz components plus noise, and recover those peaks.

# %%
sf = 250.0
t = np.arange(0, 4, 1 / sf)
x = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 13 * t) + 0.4 * rng.standard_normal(t.size)

freqs = np.fft.rfftfreq(t.size, 1 / sf)
amp = np.abs(np.fft.rfft(x)) / t.size

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(t, x, lw=0.7); axes[0].set(title="Signal (time domain)", xlabel="Time (s)")
axes[1].plot(freqs, amp, color="#c44e52"); axes[1].set(xlim=(0, 40),
            title="FFT amplitude spectrum", xlabel="Frequency (Hz)")
for f0 in (6, 13): axes[1].axvline(f0, ls="--", color="gray", lw=0.8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Welch's PSD: a stable power spectrum
#
# A raw FFT of noisy data is itself noisy. **Welch's method** splits the signal
# into overlapping windows, computes the spectrum of each, and averages them. The
# result — the **power spectral density (PSD)** — is much smoother and is what we
# use for band-power features. Let's compare a raw FFT-power vs Welch on real EEG.

# %%
X, y, subj = io.load_physionet_mi_epochs(n_subjects=1)
sf = ds.DATASETS["physionet_mi"].sfreq_hz
sig = X[0, 0]  # one channel, one epoch

f_w, psd_w = welch(sig, fs=sf, nperseg=int(sf))
raw_power = (np.abs(np.fft.rfft(sig)) ** 2) / sig.size
f_raw = np.fft.rfftfreq(sig.size, 1 / sf)

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.semilogy(f_raw, raw_power, color="#bbb", lw=0.7, label="raw FFT power (noisy)")
ax.semilogy(f_w, psd_w, color="#2e8b57", lw=1.5, label="Welch PSD (smooth)")
ax.set(xlim=(0, 60), xlabel="Frequency (Hz)", ylabel="Power", title="Welch averaging tames the spectrum")
ax.legend(); plt.show()

# %% [markdown]
# ## 3. Band power: features from the PSD
#
# We summarise the PSD into the classic EEG bands. These are among the most-used
# EEG features. Our helper returns **log** band power per channel.

# %%
print("Bands (Hz):", BANDS)
bp = bandpower(X[:1], sf)  # (1 trial, n_channels * n_bands)
n_ch = X.shape[1]
bp_grid = bp.reshape(len(BANDS), n_ch)  # rows: bands, cols: channels (our layout)
fig, ax = plt.subplots(figsize=(9, 3))
im = ax.imshow(bp_grid, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(BANDS))); ax.set_yticklabels(list(BANDS))
ax.set(xlabel="Channel index", title="Log band power per channel (one trial)")
fig.colorbar(im, ax=ax, fraction=0.025); plt.show()

# %% [markdown]
# ## 4. Time–frequency: when does each frequency happen?
#
# A PSD throws away *time*. But brain activity is non-stationary — an alpha burst
# may come and go. The **spectrogram** (Short-Time Fourier Transform, STFT) slides
# a window along the signal and computes a spectrum at each position.
#
# We build a **chirp** (a tone whose frequency rises over time) so the spectrogram
# clearly shows a diagonal ridge.
#
# > **Before running:** guess the shape of the chirp's ridge in the spectrogram — will
# > it be a flat horizontal line, a diagonal line rising from bottom-left to top-right,
# > or a curved arc? Sketch your prediction, then compare to the output.

# %%
t = np.arange(0, 4, 1 / sf)
chirp = np.sin(2 * np.pi * (5 + 8 * t) * t)  # frequency rises from ~5 Hz upward
viz.plot_spectrogram(chirp, sf, title="Spectrogram of a rising chirp (frequency increases with time)")
plt.show()

# %% [markdown]
# ## 5. The time–frequency trade-off (made concrete)
#
# The STFT window length forces a choice:
# - **Short window** → good *time* resolution, poor *frequency* resolution.
# - **Long window** → good *frequency* resolution, poor *time* resolution.
#
# You cannot have both (this is the uncertainty principle for signals). Below, the
# same chirp analysed with a short vs long window — note how the ridge is either
# *time-sharp* or *frequency-sharp*, never both.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, nperseg, label in [(axes[0], 64, "short window (64): time-sharp"),
                           (axes[1], 512, "long window (512): frequency-sharp")]:
    f, tt, Sxx = spectrogram(chirp, fs=sf, nperseg=nperseg)
    ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-20), shading="gouraud")
    ax.set(ylim=(0, 60), xlabel="Time (s)", ylabel="Hz", title=label)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Wavelets: a smarter trade-off
#
# **Wavelets** use *short* windows for high frequencies and *long* windows for low
# frequencies — matching how we usually want to look at brain signals (precise
# timing of fast events, precise frequency of slow rhythms). MNE has rich wavelet
# tools (`mne.time_frequency.tfr_morlet`); here we show the idea with a Morlet-like
# analysis via STFT magnitude across scales.

# %%
f_stft, t_stft, Zxx = stft(chirp, fs=sf, nperseg=128)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading="gouraud")
ax.set(ylim=(0, 60), xlabel="Time (s)", ylabel="Frequency (Hz)",
       title="STFT magnitude of the chirp (a stepping stone to wavelets)")
plt.show()

# %% [markdown]
# ## ✅ Concept check
#
# 1. Welch's method averages spectra computed on overlapping windows. Doubling the
#    number of windows (by halving the window length) improves frequency or time
#    resolution — which one, and why?
# 2. The alpha band is typically 8–13 Hz. If you compare absolute alpha power between
#    two subjects and subject A has much thicker skull, what confound arises and how
#    would you address it?
# 3. A short STFT window gives good time resolution but poor frequency resolution.
#    Name one brain-science scenario where you would deliberately choose a short
#    window despite its poor frequency resolution.
#
# **Answers:**
# 1. Halving the window length halves frequency resolution (fewer samples → coarser
#    frequency bins) but the variance of each estimate decreases because more windows
#    are averaged. You trade frequency resolution for statistical stability, not time
#    resolution (the window position still determines time resolution).
# 2. Thicker skull attenuates the EEG signal, so subject A will show lower absolute
#    power even if their actual brain alpha is the same. Use relative band power
#    (alpha ÷ total broadband power) to normalise across individuals.
# 3. Any scenario where *when* an event happens matters more than *exactly which*
#    frequency — for example, detecting the onset of a motor-evoked high-gamma burst
#    (> 70 Hz) where millisecond timing is critical and the exact frequency within
#    the gamma band is less important.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Reading a single noisy FFT as truth.** Use Welch (averaging) or you'll chase
#   noise peaks. Report the window length you used.
# - **Comparing absolute band power across subjects.** Skull thickness and
#   electrode contact change overall amplitude. Use **relative** band power (each
#   band ÷ total) when comparing people — we expose `relative=True` for this.
# - **Ignoring spectral leakage.** Hard window edges smear power across
#   frequencies; windowing (Welch uses Hann by default) mitigates it.
# - **Expecting perfect time *and* frequency resolution.** It's mathematically
#   impossible. Choose your window to match the question.
# - **Mismatched `nperseg` and sampling rate.** A window of 256 samples means a
#   different duration at 100 Hz vs 250 Hz. Think in *seconds*, then convert.
#
# **Next:** Chapter 07 — turning these spectra (and more) into features:
# connectivity, CSP, and Riemannian covariance.
