# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/03_math_you_can_see.ipynb)
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
# # Chapter 03 — Math You Can See
#
# Three ideas hide inside almost every later chapter: **Fourier**, **covariance**,
# and **eigenvectors**. None of them require calculus to *understand* — they just
# need the right picture. This chapter builds those pictures from scratch using
# toy signals so that when these words appear in Chapter 06 or Chapter 08, you
# already have a mental image to anchor them.
#
# ## Learning objectives
# 1. Describe the **Fourier transform** as "how much of each rhythm is in a signal."
# 2. Read a **covariance matrix** and say what its entries mean.
# 3. Explain what **eigenvectors** of a covariance matrix point at, and why that
#    matters for separating two classes of brain signals.
#
# > **Prerequisites:** Chapter 02.
# > **Difficulty:** ★★★☆☆
# > **Runtime:** ~1 min (synthetic, CPU).

# %%
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# %% [markdown]
# ---
# ## Part 1 — Fourier: signals are sums of rhythms
#
# ### What is a rhythm?
#
# A **sine wave** is the purest possible oscillation — it goes up and down
# smoothly at exactly one speed, called its **frequency** (measured in **Hz**,
# cycles per second). EEG rhythms like "alpha" are named because they show up as
# bumps in the sine-wave decomposition of the signal.
#
# The **Fourier transform** answers one question:
# *"If I were to write this signal as a sum of sine waves, how much of each
# frequency would I need?"*
#
# The answer is the **spectrum** — a bar chart of frequency vs amplitude.
#
# Let's build a signal ourselves from three sine waves so we know exactly what
# the right answer should be.

# %%
# --- Build the signal ---
sfreq = 200          # sampling rate: 200 samples per second
duration = 2.0       # seconds
t = np.arange(0, duration, 1.0 / sfreq)   # time axis

f1, f2, f3 = 6.0, 10.0, 20.0             # Hz — three rhythms

amp1, amp2, amp3 = 1.0, 0.6, 0.4          # how loud each rhythm is

sine1 = amp1 * np.sin(2 * np.pi * f1 * t)
sine2 = amp2 * np.sin(2 * np.pi * f2 * t)
sine3 = amp3 * np.sin(2 * np.pi * f3 * t)

noise = 0.15 * rng.standard_normal(t.size)  # small amount of random noise

signal = sine1 + sine2 + sine3 + noise

print(f"Signal: {len(t)} samples, {sfreq} Hz, {duration} s")
print(f"Components: {f1} Hz (amp {amp1}), {f2} Hz (amp {amp2}), {f3} Hz (amp {amp3})")

# %% [markdown]
# ### Figure 1a — The raw time-domain signal
#
# This is what you would see on an oscilloscope or in an EEG viewer.
# Can you spot any repeating patterns? It's hard, because three rhythms overlap.

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, signal, color="#2c7bb6", lw=0.9)
ax.set(xlabel="Time (s)", ylabel="Amplitude",
       title="Time-domain signal (6 Hz + 10 Hz + 20 Hz + noise)")
ax.axhline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 1b — The individual components (stacked)
#
# Here we "open the hood" and display each ingredient separately.
# This is the ground truth we want the Fourier transform to recover.

# %%
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

components = [
    (sine1, f"{f1} Hz component  (amp {amp1})", "#d7191c"),
    (sine2, f"{f2} Hz component  (amp {amp2})", "#fdae61"),
    (sine3, f"{f3} Hz component  (amp {amp3})", "#1a9641"),
    (noise, "Noise", "#888888"),
]
for ax, (comp, label, color) in zip(axes, components):
    ax.plot(t, comp, color=color, lw=0.9, label=label)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_ylabel("Amplitude")

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Individual sine-wave ingredients", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 1c — Adding components one at a time (partial sums)
#
# This is the key geometric insight: **a signal IS a sum of rhythms**.
# We start with just the 6 Hz wave, then add 10 Hz, then 20 Hz, then noise.
# Each panel brings the waveform closer to the final signal we recorded.

# %%
partial_sums = [
    (sine1,                        f"6 Hz only"),
    (sine1 + sine2,                f"6 + 10 Hz"),
    (sine1 + sine2 + sine3,        f"6 + 10 + 20 Hz"),
    (sine1 + sine2 + sine3 + noise, "6 + 10 + 20 Hz + noise  (= full signal)"),
]

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
colors = ["#d7191c", "#fdae61", "#1a9641", "#2c7bb6"]

for ax, (ps, label), color in zip(axes, partial_sums, colors):
    ax.plot(t, ps, color=color, lw=0.9)
    ax.set_ylabel("Amplitude")
    ax.set_title(label, fontsize=10, loc="left")
    ax.axhline(0, color="gray", lw=0.4)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Building a signal by adding sine waves one at a time", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ### Before you run the next cell — make a prediction!
#
# **Predict before running:** We built the signal from sine waves at 6 Hz,
# 10 Hz, and 20 Hz with amplitudes 1.0, 0.6, and 0.4 respectively.
#
# - Which frequencies do you expect to see as peaks in the spectrum?
# - Which peak should be tallest? Which shortest?
# - Should there be any noise floor between peaks?
#
# Write down (or just think through) your prediction before revealing the answer.

# %%
# --- Compute the FFT magnitude spectrum ---
N = len(signal)
fft_vals = np.fft.rfft(signal)                   # real FFT (positive freqs only)
fft_mag  = np.abs(fft_vals) / N * 2              # normalise: divide by N, ×2 for one-sided
freqs    = np.fft.rfftfreq(N, d=1.0 / sfreq)    # frequency axis in Hz

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, fft_mag, color="#2c7bb6", lw=1.2)
ax.set(xlabel="Frequency (Hz)", ylabel="Amplitude",
       title="FFT magnitude spectrum — peaks reveal which rhythms are present",
       xlim=(0, sfreq / 2))

# Annotate the three known peaks
for f0, amp in [(f1, amp1), (f2, amp2), (f3, amp3)]:
    idx = np.argmin(np.abs(freqs - f0))
    ax.axvline(f0, ls="--", color="gray", lw=0.8)
    ax.annotate(f"{f0} Hz\n(amp ≈ {amp})",
                xy=(f0, fft_mag[idx]),
                xytext=(f0 + 2, fft_mag[idx] * 0.8),
                fontsize=9, color="#c44e52",
                arrowprops=dict(arrowstyle="->", color="#c44e52"))

plt.tight_layout()
plt.show()

print("Peak frequencies found at approximately:")
peak_mask = fft_mag > 0.1
for f, m in zip(freqs[peak_mask], fft_mag[peak_mask]):
    print(f"  {f:.1f} Hz  →  amplitude {m:.3f}")

# %% [markdown]
# **Did the spectrum match your prediction?**
#
# Key takeaways:
# - Each sine-wave ingredient shows up as a **sharp spike** at its frequency.
# - The **height** of the spike equals the amplitude we chose (1.0, 0.6, 0.4).
# - Between the spikes there is a small noise floor — the random noise spreads
#   its tiny energy across all frequencies.
#
# The spectrum is a *recipe card*: it tells you exactly which rhythms, and how
# much of each, are baked into the signal.
#
# > **One practical note:** the x-axis of the FFT only makes sense if you know
# > the **sampling rate**. The same FFT output with a different `sfreq` would
# > label those peaks at completely different frequencies. Always record `sfreq`!

# %% [markdown]
# ---
# ## Part 2 — Covariance: how two channels move together
#
# ### Why we care
#
# EEG has many channels (electrodes). They are **not independent** — nearby
# electrodes pick up the same underlying brain source (this is called
# *volume conduction*). Understanding how channels co-vary is essential for
# spatial filtering (like CSP, used in Chapter 08).
#
# ### What is covariance?
#
# **Variance** of one channel: how much that channel wiggles on its own.
#
# **Covariance** of two channels: do they tend to be *high at the same time*
# (positive covariance) or *go in opposite directions* (negative covariance)?
#
# The **covariance matrix** collects all pairwise covariances in one table.
# For two channels A and B:
#
# ```
# C = | Var(A)    Cov(A,B) |
#     | Cov(B,A)  Var(B)   |
# ```
#
# The diagonal is each channel's own variance; the off-diagonal is the coupling.

# %%
# --- Generate two correlated channels ---
n_samples = 300

# Channel 1: pure random
ch1_raw = rng.standard_normal(n_samples)

# Channel 2: 70% the same signal + 30% independent noise  → strong positive correlation
ch2_raw = 0.7 * ch1_raw + 0.3 * rng.standard_normal(n_samples)

# Scale them to different variances to make the covariance matrix more interesting
ch1 = 2.0 * ch1_raw          # channel 1 has larger variance
ch2 = 1.0 * ch2_raw          # channel 2 has smaller variance

data2d = np.stack([ch1, ch2], axis=0)   # shape (2, n_samples)

# Covariance matrix
C = np.cov(data2d)   # numpy computes pairwise covariance

print("Covariance matrix C:")
print(C.round(3))
print()
print(f"  C[0,0] = Var(ch1) = {C[0,0]:.3f}   — ch1's own variance")
print(f"  C[1,1] = Var(ch2) = {C[1,1]:.3f}   — ch2's own variance")
print(f"  C[0,1] = Cov(ch1, ch2) = {C[0,1]:.3f}  — how they co-vary (positive = same direction)")

# %% [markdown]
# ### Figure 2 — Scatter plot with covariance ellipse and eigenvectors
#
# A scatter plot of ch1 vs ch2 shows the joint distribution.
# We overlay:
# - A **covariance ellipse**: the region of ≈68 % of the data (1-sigma).
#   Its shape is stretched along the directions of greatest joint variation.
# - **Eigenvectors** scaled by the square-root of their eigenvalue.
#   They point along those stretching directions.

# %%
# Eigen-decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(C)   # eigh: symmetric matrix, sorted ascending
# Sort descending so eigenvector[:,0] is the "most variance" direction
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Draw the ellipse by rotating a unit circle
theta = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
# Scale each axis by sqrt(eigenvalue), then rotate by eigenvectors
ellipse = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle

# Centre of the data (mean)
mu = data2d.mean(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ch1, ch2, alpha=0.3, s=12, color="#2c7bb6", label="data points")
ax.plot((mu[0] + ellipse[0]), (mu[1] + ellipse[1]),
        color="#d7191c", lw=2, label="1-sigma ellipse")

# Draw eigenvectors as arrows
arrow_colors = ["#d7191c", "#fdae61"]
arrow_labels = ["1st eigenvector\n(most variance)", "2nd eigenvector\n(least variance)"]
for i in range(2):
    scale = np.sqrt(eigenvalues[i])
    dx = eigenvectors[0, i] * scale
    dy = eigenvectors[1, i] * scale
    ax.annotate("", xy=(mu[0, 0] + dx, mu[1, 0] + dy),
                xytext=(mu[0, 0], mu[1, 0]),
                arrowprops=dict(arrowstyle="-|>", color=arrow_colors[i], lw=2.5))
    ax.text(mu[0, 0] + dx * 1.15, mu[1, 0] + dy * 1.15,
            arrow_labels[i], fontsize=9, color=arrow_colors[i], ha="center")

ax.set(xlabel="Channel 1", ylabel="Channel 2",
       title="Covariance ellipse and eigenvectors\n"
             "Eigenvectors point along directions of greatest joint variation")
ax.set_aspect("equal")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the picture:**
#
# - The elongated shape of the cloud tells you ch1 and ch2 are **correlated** —
#   when ch1 is high, ch2 tends to be high too.
# - The **red arrow** (first eigenvector) points along the long axis of the
#   ellipse — the direction where the data varies the most.
# - The **orange arrow** (second eigenvector) is perpendicular — the direction
#   of least variation.
# - The **length** of each arrow equals `sqrt(eigenvalue)` which is the
#   standard deviation of the data *projected onto that arrow's direction*.
#
# If ch1 and ch2 were completely independent, the ellipse would be a circle
# and the two eigenvectors would point along the x- and y-axes.

# %% [markdown]
# ---
# ## Part 3 — Eigenvectors and CSP geometry
#
# ### The core idea of CSP (Common Spatial Patterns)
#
# In a motor-imagery BCI we ask: *"Is the person imagining moving their left hand
# or right hand?"* Each mental state produces EEG with a **different spatial
# covariance structure** — some channels become more active, some less.
#
# **CSP** finds a spatial direction (a weighted combination of channels) where:
# - **Class A** has HIGH variance (the signal varies a lot).
# - **Class B** has LOW variance (the signal barely moves).
#
# If we project the raw multi-channel signal onto that direction, the two classes
# become easy to separate: class A gives big values, class B gives small values.
#
# We can see this geometry with 2-D toy data — no CSP library needed.

# %%
# --- Create two classes with different covariance structure ---
n_per_class = 200

# Class A: varies mostly along the 45-degree diagonal (/ direction)
angle_A = np.pi / 4         # 45 degrees
cov_A_axes = np.array([[3.0, 0], [0, 0.3]])   # long axis in first principal direction

def make_class(n, angle, cov_axes, rng):
    """Sample from a 2D Gaussian, rotate it by `angle`."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    raw = rng.standard_normal((n, 2)) @ np.diag(np.sqrt(np.diag(cov_axes)))
    return (R @ raw.T).T

classA = make_class(n_per_class, angle_A,  cov_A_axes, rng)
# Class B: varies mostly along the other diagonal (\ direction)
angle_B = -np.pi / 4        # -45 degrees (perpendicular to A)
classB = make_class(n_per_class, angle_B, cov_A_axes, rng)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(classA[:, 0], classA[:, 1], alpha=0.4, s=15,
           color="#d7191c", label="Class A (e.g. left-hand imagery)")
ax.scatter(classB[:, 0], classB[:, 1], alpha=0.4, s=15,
           color="#2c7bb6", label="Class B (e.g. right-hand imagery)")
ax.set(xlabel="Channel 1", ylabel="Channel 2",
       title="Two classes with different covariance structure\n"
             "Class A stretches along /, Class B along \\")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# In the scatter plot you can see:
# - Class A (red) is elongated along the **top-left to bottom-right** diagonal.
# - Class B (blue) is elongated along the **other** diagonal.
#
# Now the question: **which single axis should we project onto to best separate
# the two classes?**
#
# A natural choice is the direction along which A varies a lot and B varies a
# little — i.e. the **first eigenvector of class A's covariance matrix**.
# That direction maximises the ratio `Var_A / Var_B`.

# %% [markdown]
# ---
# ### Before you run the next cell — another prediction!
#
# **Predict before running:** We will project both classes onto the first
# eigenvector of class A's covariance matrix (the direction where A varies most).
#
# - Do you expect class A's histogram to be wide or narrow on that axis?
# - Do you expect class B's histogram to be wide or narrow?
# - Will the two histograms overlap a lot, or will they be well separated?

# %%
# Compute the first eigenvector of class A's covariance
cov_A = np.cov(classA.T)
eigvals_A, eigvecs_A = np.linalg.eigh(cov_A)
# eigh returns ascending order; take the last (largest)
w_csp = eigvecs_A[:, -1]    # the direction that maximises Var_A

print(f"Chosen direction (eigenvector of class A): [{w_csp[0]:.3f}, {w_csp[1]:.3f}]")

# Project both classes onto w_csp
proj_A = classA @ w_csp
proj_B = classB @ w_csp

var_A_proj = np.var(proj_A)
var_B_proj = np.var(proj_B)
print(f"\nVariance of class A projected onto w_csp: {var_A_proj:.3f}")
print(f"Variance of class B projected onto w_csp: {var_B_proj:.3f}")
print(f"Variance ratio (A/B): {var_A_proj / var_B_proj:.2f}x")

# %% [markdown]
# ### Figure 3 — Geometric view + 1-D histograms

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: scatter with the chosen direction
ax = axes[0]
ax.scatter(classA[:, 0], classA[:, 1], alpha=0.35, s=12,
           color="#d7191c", label="Class A")
ax.scatter(classB[:, 0], classB[:, 1], alpha=0.35, s=12,
           color="#2c7bb6", label="Class B")

# Draw w_csp as a line through the origin
scale = 3.0
ax.annotate("", xy=(w_csp[0] * scale, w_csp[1] * scale),
            xytext=(-w_csp[0] * scale, -w_csp[1] * scale),
            arrowprops=dict(arrowstyle="-|>", color="#1a9641", lw=2.5))
ax.text(w_csp[0] * scale * 1.1, w_csp[1] * scale * 1.1,
        "CSP axis\n(max A, min B)", color="#1a9641", fontsize=9, ha="center")

ax.set(xlabel="Channel 1", ylabel="Channel 2",
       title="CSP axis: maximise class A, minimise class B variance")
ax.set_aspect("equal")
ax.legend(fontsize=9)

# Right panel: 1-D histograms after projection
ax2 = axes[1]
bins = np.linspace(-5, 5, 40)
ax2.hist(proj_A, bins=bins, alpha=0.55, color="#d7191c", label=f"Class A  (var={var_A_proj:.2f})")
ax2.hist(proj_B, bins=bins, alpha=0.55, color="#2c7bb6", label=f"Class B  (var={var_B_proj:.2f})")
ax2.set(xlabel="Projected value (along CSP axis)",
        ylabel="Count",
        title="1-D histograms after projection\n"
              "Class A is spread out; class B is compressed")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the picture:**
#
# - On the **CSP axis** (green arrow), class A is spread wide — high variance.
# - Class B is compressed — low variance — because its elongated direction is
#   *perpendicular* to the CSP axis.
# - This variance difference is **separability**: a simple threshold on the
#   projected value (or on the log-variance of a short window) can distinguish
#   the two mental states.
#
# In a real EEG motor-imagery pipeline (Chapter 08) the "channels" are dozens
# of electrodes and the covariance matrices are 64×64 rather than 2×2, but the
# geometric intuition is exactly the same.

# %% [markdown]
# ---
# ## ✅ Concept check
#
# **1.** You add two sine waves: one at 8 Hz with amplitude 2 and one at 13 Hz
# with amplitude 0.5. In the FFT magnitude spectrum, which peak is taller? Why?
#
# **2.** A 2×2 covariance matrix has off-diagonal values close to zero. What does
# that tell you about the relationship between the two channels?
#
# **3.** You compute the covariance matrix eigenvectors for class A EEG data.
# You find the eigenvector with the *smallest* eigenvalue. What kind of direction
# does that eigenvector point — high variance or low variance for class A?
#
# **4.** Why do you need to know the **sampling rate** before you can interpret
# the x-axis of an FFT?
#
# ---
# **Answers:**
#
# 1. The 8 Hz peak is taller (amplitude 2 vs 0.5). In the magnitude spectrum,
#    height directly reflects amplitude. The 13 Hz peak is shorter.
#
# 2. Off-diagonal ≈ 0 means the channels are **uncorrelated** — knowing that
#    channel 1 is high gives you no useful information about where channel 2 is.
#    The scatter plot would look like a round cloud, not an elongated ellipse.
#
# 3. The smallest eigenvalue corresponds to the direction of **lowest variance**.
#    For CSP you would want this direction for the *other* class, not for class A.
#
# 4. The FFT produces a vector of complex numbers indexed by *bin number*, not
#    Hz. To convert bin index to Hz you compute `bin / (N / sfreq)`. Without
#    `sfreq`, the same FFT output could represent any range of frequencies.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Confusing amplitude and power.** The FFT gives amplitude (|F|). Power is
#   amplitude *squared* (|F|²). A peak that looks twice as tall in amplitude is
#   *four times* more powerful. Power spectra are usually shown on a log scale;
#   amplitude spectra on linear. Don't mix them up.
#
# - **Forgetting the sampling rate sets the frequency axis.** The FFT only
#   produces bin indices. Without knowing `sfreq` you cannot label the x-axis in
#   Hz. A common bug: dividing by the number of samples N instead of by
#   the sampling period 1/sfreq. Always use `np.fft.rfftfreq(N, d=1/sfreq)`.
#
# - **Treating EEG channels as independent.** Because electricity spreads through
#   the skull (volume conduction), nearby electrodes record overlapping mixtures
#   of the same brain sources. Their covariance is never zero. Algorithms that
#   assume independent channels (e.g. naive Bayes on raw samples) will overfit.
#
# - **Thinking eigenvectors are magic.** Eigenvectors just give you the axes of
#   the covariance ellipse. They are only meaningful relative to the covariance
#   matrix you computed. If you change the data (different session, different
#   subject), the eigenvectors change too. The CSP filters you compute in one
#   session can degrade badly if applied to a different session without
#   recalibration.
#
# - **Ignoring normalisation in the FFT.** `np.fft.rfft` returns un-normalised
#   complex numbers. To get back physical amplitudes, divide by N and multiply
#   by 2 (for the one-sided spectrum). Forgetting this gives you numbers that
#   scale with signal length, making comparison across recordings misleading.
#
# **Next:** Chapter 04 — DSP basics: now that you have intuition for the
# frequency domain, we build the practical tools — filters, notch removal, and
# re-referencing — that clean raw EEG before any feature extraction.
