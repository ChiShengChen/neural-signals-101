# %% [markdown]
# # Deep-dive — ICA & ASR Internals
#
# How the artifact-removal machinery from Chapter 05 actually works under the hood.
#
# > **Prerequisites:** main Chapter 05.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget.**

# %% [markdown]
# ## 0 — Bootstrap

# %%
import sys
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path.cwd().parent.parent / "src"))
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in nbconvert
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
mne.set_log_level("ERROR")

from sklearn.decomposition import FastICA

from neuro101.io import load_physionet_mi_raw
from neuro101.preprocessing import fit_ica, clip_extreme_amplitudes

rng = np.random.default_rng(42)

print("All imports OK  |  MNE", mne.__version__)

# %% [markdown]
# ---
# ## 1 — The mixing model: X = A S
#
# Independent Component Analysis (ICA) starts from a **generative assumption**:
# what the sensors record is a linear superposition of a small set of
# *statistically independent* source signals.
#
# ### The model
#
# Let:
# * **S** ∈ ℝ^{n_sources × T} — the source matrix (each row is one source signal).
# * **A** ∈ ℝ^{n_channels × n_sources} — the **mixing matrix** (unknown).
# * **X** ∈ ℝ^{n_channels × T} — what the electrodes record.
#
# Then:
# $$X = A \, S$$
#
# ICA goal: find an **unmixing matrix** W ≈ A⁻¹ such that
# $$\hat{S} = W X$$
# where the rows of $\hat{S}$ are as statistically independent as possible.
#
# ### Why non-Gaussianity?
#
# The Central Limit Theorem says that *sums* of independent random variables
# tend toward Gaussianity — so any single observed channel (a mixture) is
# *more* Gaussian than the individual sources. ICA exploits this in reverse:
# we search for directions W that **maximise non-Gaussianity** (e.g. maximise
# kurtosis or negentropy). The direction of maximum non-Gaussianity is the one
# that has "un-mixed" the superposition back to a single source.
#
# **The failure case:** if a source is truly Gaussian, its distribution is
# symmetric and has no unique direction of maximum non-Gaussianity — any
# rotation of a Gaussian is still Gaussian. ICA cannot separate Gaussian
# sources; they must be non-Gaussian (leptokurtic blink pulses, EMG bursts,
# sinusoidal oscillations — all qualify).

# %% [markdown]
# ---
# ## 2 — Cocktail-party demo with synthetic signals
#
# We create three independent source signals (a sine, a sawtooth, and sparse
# Laplacian noise), mix them with a known matrix **A**, then recover them with
# `sklearn.decomposition.FastICA`.  FastICA uses the *negentropy* objective
# (equivalent to maximising non-Gaussianity) and is the standard fast algorithm
# due to Hyvärinen & Oja (1997).

# %%
# ---------- generate three independent sources ----------
T      = 2000
t      = np.linspace(0, 4 * np.pi, T)

s1 = np.sin(t * 3)                                   # sine wave
s2 = 2 * (t % (np.pi) / np.pi) - 1                  # sawtooth
s3 = rng.laplace(scale=0.3, size=T)                  # sparse (Laplacian) noise
s3 = s3 / np.std(s3)                                 # unit variance

S = np.vstack([s1, s2, s3])   # (3, T)

# standardise: ICA needs (approx) unit variance sources
S = S / S.std(axis=1, keepdims=True)

# ---------- mix with a random full-rank matrix ----------
A_true = rng.standard_normal((4, 3))                 # 4 "channels" from 3 sources
# make sure it is well-conditioned
A_true /= np.linalg.norm(A_true, axis=0, keepdims=True)

X_mix = A_true @ S                                   # (4, T)  — the "recordings"

# add a little sensor noise
X_mix += 0.05 * rng.standard_normal(X_mix.shape)

# ---------- recover with FastICA ----------
ica_syn = FastICA(n_components=3, random_state=0, max_iter=2000, tol=1e-4)
S_rec = ica_syn.fit_transform(X_mix.T).T             # (3, T)

print("Source shapes:", S.shape)
print("Mixture shape:", X_mix.shape)
print("Recovered shape:", S_rec.shape)

# quick correlation check (up to permutation/sign)
from scipy.optimize import linear_sum_assignment

corr = np.abs(np.corrcoef(S, S_rec)[:3, 3:])        # 3×3 block
row_idx, col_idx = linear_sum_assignment(-corr)
for r, c in zip(row_idx, col_idx):
    print(f"  source {r+1}  <->  recovered {c+1}   |r| = {corr[r, c]:.3f}")

# %% [markdown]
# ### Figure 1 — Sources, mixtures, and recovered signals
#
# ICA recovers the sources **up to arbitrary permutation and sign flip**.
# That is not a bug — there is no principled reason to prefer +sin over −sin
# or to assign a particular ordering.  What matters is that each recovered
# component time-course matches exactly one source (up to a scalar).

# %%
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

source_labels   = ["Sine (s₁)", "Sawtooth (s₂)", "Laplacian noise (s₃)"]
mixture_labels  = [f"Electrode {i+1}" for i in range(4)]
col_titles      = ["Independent Sources (S)", "Observed Mixtures (X = AS)", "Recovered (Ŝ = WX)"]

colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- column 0: sources ---
for i in range(3):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(t, S[i], color=colours[i], lw=0.8)
    ax.set_ylabel(source_labels[i], fontsize=8)
    ax.set_yticks([])
    if i == 0:
        ax.set_title(col_titles[0], fontsize=10, fontweight="bold")
    if i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("time (a.u.)", fontsize=8)

# --- column 1: 4 mixtures squeezed into 3 rows ---
for i in range(3):
    ax = fig.add_subplot(gs[i, 1])
    j  = i if i < 3 else 3       # map 3 rows -> 4 mixtures (show 3 for symmetry)
    ax.plot(t, X_mix[j], color="grey", lw=0.7)
    ax.set_ylabel(mixture_labels[j], fontsize=8)
    ax.set_yticks([])
    if i == 0:
        ax.set_title(col_titles[1], fontsize=10, fontweight="bold")
    if i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("time (a.u.)", fontsize=8)

# --- column 2: recovered (sorted by correlation) ---
for plot_i, (r, c) in enumerate(zip(row_idx, col_idx)):
    ax = fig.add_subplot(gs[plot_i, 2])
    # flip sign to match source if needed
    sign = np.sign(np.corrcoef(S[r], S_rec[c])[0, 1])
    ax.plot(t, sign * S_rec[c], color=colours[r], lw=0.8)
    ax.set_ylabel(f"IC {c+1} ↔ s{r+1}", fontsize=8)
    ax.set_yticks([])
    if plot_i == 0:
        ax.set_title(col_titles[2], fontsize=10, fontweight="bold")
    if plot_i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("time (a.u.)", fontsize=8)

fig.suptitle(
    "Cocktail-party demo: ICA unmixes 4 sensor signals back to 3 independent sources",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig1_synthetic.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# **Read-out.** The recovered column should look visually identical to the
# source column (up to a permutation of rows and a possible sign flip).
# The correlations printed above quantify the alignment; values close to 1.0
# confirm successful separation.
#
# **Key maths.** FastICA alternates between:
# 1. **Fixed-point update** — for each column $\mathbf{w}_i$ of W:
#    $$\mathbf{w}_i \leftarrow E[\mathbf{x}\, g(\mathbf{w}_i^\top \mathbf{x})]
#    - E[g'(\mathbf{w}_i^\top \mathbf{x})]\,\mathbf{w}_i$$
#    where $g = \tanh$ (the derivative of the log-cosh contrast function).
# 2. **Gram-Schmidt decorrelation** — $W \leftarrow (WW^\top)^{-1/2} W$ to
#    keep components mutually uncorrelated.
#
# Convergence is cubic (much faster than gradient ascent) and typically takes
# < 200 iterations on well-conditioned problems.

# %% [markdown]
# ---
# ## 3 — Real EEG: ICA on a PhysioNet motor-imagery recording
#
# We load one subject's continuous recording, apply ICA with MNE, and inspect
# the component topographies and time-courses.  The frontal components (large
# weights at Fp1/Fp2/Fpz) are the telltale eye-blink sources.

# %%
import os

smoke = os.environ.get("NEURO101_SMOKE", "0") == "1"

print("Loading PhysioNet subject 1 …")
raw = load_physionet_mi_raw(subject=1, runs=(4,) if smoke else (4, 8, 12))
print(raw)

# Basic cleaning before ICA (ICA needs ≥1 Hz high-pass; fit_ica handles this)
raw_clean = raw.copy().filter(1.0, 40.0, verbose="ERROR")
raw_clean.set_eeg_reference("average", projection=False, verbose="ERROR")

n_ica = 10 if smoke else 15
print(f"Fitting ICA with {n_ica} components …")
ica = fit_ica(raw_clean, n_components=n_ica, random_state=0)
print("ICA fitted:", ica)

# %% [markdown]
# ### Figure 2 — ICA component time-courses and topography weights
#
# Each panel shows the first few seconds of a component's **activation** (the
# row of $\hat{S} = WX$) together with the **topography** (the corresponding
# column of A = W⁻¹, which tells you how that source projects onto the scalp).
# A component with high frontal weight and slow, large deflections is a blink.

# %%
sfreq    = raw_clean.info["sfreq"]
data_ica = ica.get_sources(raw_clean).get_data()   # (n_comp, n_times)
times    = np.arange(data_ica.shape[1]) / sfreq

# Show up to 5 components in time + their weights as a bar chart
n_show  = min(5, n_ica)
t_end   = min(20.0, times[-1])        # first 20 s
mask    = times <= t_end

# mixing matrix: columns are component topographies (scalings onto channels)
mixing  = ica.mixing_matrix_           # (n_channels, n_components)
ch_names = raw_clean.ch_names

fig2, axes = plt.subplots(n_show, 2, figsize=(13, 2.5 * n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

for k in range(n_show):
    ax_ts  = axes[k, 0]
    ax_top = axes[k, 1]

    # time-course
    ts = data_ica[k, mask]
    ax_ts.plot(times[mask], ts, lw=0.6, color="#333333")
    ax_ts.set_ylabel(f"IC {k:02d}", fontsize=8)
    ax_ts.set_yticks([])
    if k == 0:
        ax_ts.set_title("Component activation (first 20 s)", fontsize=9, fontweight="bold")
    if k < n_show - 1:
        ax_ts.set_xticks([])
    else:
        ax_ts.set_xlabel("time (s)", fontsize=8)

    # topography: bar chart of weights per channel (sorted by |weight|)
    weights = mixing[:, k]
    order   = np.argsort(np.abs(weights))[::-1][:12]   # top-12 channels
    ax_top.barh(
        range(len(order)),
        weights[order],
        color=["#d62728" if w > 0 else "#1f77b4" for w in weights[order]],
        height=0.7,
    )
    ax_top.set_yticks(range(len(order)))
    ax_top.set_yticklabels([ch_names[i] for i in order], fontsize=7)
    ax_top.axvline(0, color="k", lw=0.7)
    ax_top.set_xlabel("weight", fontsize=8)
    if k == 0:
        ax_top.set_title("Top-12 channel weights (topography)", fontsize=9, fontweight="bold")

fig2.suptitle("ICA components — activations and topographies", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig2_components.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 2 saved.")

# %% [markdown]
# ### Identifying the blink component
#
# Eye blinks appear as **large, slow (~1–4 Hz) deflections** concentrated on
# **frontal channels** (Fp1, Fp2, Fpz).  Look for a component whose:
# * Time-course has infrequent large spikes (blink rate ~15/min = ~0.25 Hz).
# * Top-weighted channels are frontal.
#
# MNE's `find_bads_eog` automates this by correlating each component activation
# with a frontal channel used as a proxy EOG signal.

# %%
# Try automated EOG detection using a frontal channel as proxy
frontal_candidates = ["Fp1", "Fp2", "Fpz", "AF3", "AF4"]
proxy_ch = next((c for c in frontal_candidates if c in raw_clean.ch_names), None)

blink_idx = []
if proxy_ch is not None:
    try:
        blink_idx, scores = ica.find_bads_eog(
            raw_clean, ch_name=proxy_ch, threshold=2.5, verbose="ERROR"
        )
        print(f"Proxy channel: {proxy_ch}")
        print(f"Flagged as blink component(s): {blink_idx}")
        if len(scores) > 0:
            for idx, sc in enumerate(scores):
                print(f"  IC {idx:02d}  EOG-correlation score = {sc:+.3f}")
    except Exception as e:
        print(f"Automated detection skipped ({e}); inspect topographies manually.")
else:
    print("No frontal channel found — inspect topographies manually.")

if blink_idx:
    print(f"\nComponent(s) {blink_idx} will be excluded as eye-blink artefact(s).")

# %% [markdown]
# ### Apply ICA and compare before/after on the frontal channel
#
# We zero out the flagged component(s) in source space and mix back.
# The result should show **reduced** blink deflections on frontal channels
# with the rest of the spectrum intact.

# %%
if blink_idx:
    raw_corrected = raw_clean.copy()
    ica.exclude = blink_idx
    ica.apply(raw_corrected, verbose="ERROR")
    ica.exclude = []          # reset so later cells are not affected

    # Compare on the frontal proxy channel (first 30 s)
    t_seg = 30.0
    mask_seg = times <= t_seg

    ch_idx = raw_clean.ch_names.index(proxy_ch)
    before = raw_clean.get_data()[ch_idx, mask_seg] * 1e6    # V → µV
    after  = raw_corrected.get_data()[ch_idx, mask_seg] * 1e6

    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    ax1.plot(times[mask_seg], before, lw=0.7, color="#d62728", label="before ICA")
    ax1.set_ylabel("amplitude (µV)", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"Channel {proxy_ch} — before ICA correction", fontsize=10)

    ax2.plot(times[mask_seg], after, lw=0.7, color="#1f77b4", label="after ICA")
    ax2.set_ylabel("amplitude (µV)", fontsize=9)
    ax2.set_xlabel("time (s)", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"Channel {proxy_ch} — after ICA correction", fontsize=10)

    fig3.suptitle("ICA blink removal: before vs after", fontsize=11)
    plt.tight_layout()
    plt.savefig("/tmp/dd_ica_fig3_before_after.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Figure 3 saved.")
    print(f"Peak-to-peak before: {np.ptp(before):.1f} µV  |  after: {np.ptp(after):.1f} µV")
else:
    print("No blink components flagged; skipping before/after plot.")

# %% [markdown]
# ---
# ## 4 — ASR internals: subspace projection and the simplified stand-in
#
# ### What true ASR does (the maths)
#
# Artifact Subspace Reconstruction (Mullen et al. 2015) works in three stages:
#
# **Stage 1 — Build a clean reference subspace.**
# Collect a segment of data deemed artifact-free (typically the quietest 30 s).
# Compute its covariance:
# $$\Sigma_\text{clean} = \frac{1}{T_\text{clean}} X_\text{clean} X_\text{clean}^\top$$
# Perform PCA: $\Sigma_\text{clean} = U \Lambda U^\top$.
# Retain all principal components (the full space); record the eigenvalues
# $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_C$.
#
# **Stage 2 — Slide and flag corrupted windows.**
# For each short window $X_w$ (e.g. 0.5 s), compute the Riemannian distance or
# the maximum reconstructed variance in the clean subspace:
# $$v_k = \mathbf{u}_k^\top X_w X_w^\top \mathbf{u}_k \;/\; (T_w \lambda_k)$$
# A window is flagged if $\max_k v_k > \rho^2$ for a user-chosen threshold $\rho$
# (default 5 in MNE-based implementations).
#
# **Stage 3 — Reconstruct corrupted windows.**
# Decompose the flagged window: $X_w = X_w^{(\text{keep})} + X_w^{(\text{bad})}$
# where the "bad" subspace is spanned by the components whose variance exceeds the
# threshold.  Replace the bad subspace with zeros (or a Wiener-filtered estimate
# from neighbouring clean windows):
# $$\hat{X}_w = P_\text{keep}\, X_w, \quad P_\text{keep} = \sum_{k:\,v_k \leq \rho^2}
# \mathbf{u}_k \mathbf{u}_k^\top$$
#
# ### The simplified stand-in: `clip_extreme_amplitudes`
#
# `neuro101.preprocessing.clip_extreme_amplitudes` is a **channel-wise** robust
# clipper: for each channel it estimates the median and MAD, then clamps any
# sample beyond $\pm z_\text{thresh} \times 1.4826 \times \text{MAD}$.
# This handles single-channel spike artefacts (electrode pops, movement) but
# does **not** project across the full sensor space — it misses artefacts that
# are diffuse across channels (e.g. amplifier saturation affecting multiple
# channels coherently).

# %%
# Illustrate the difference numerically
# -----------------------------------------------------------------------
# Inject two types of artefact into a clean segment:
#   (a) a single-channel spike  -> clip_extreme_amplitudes handles this
#   (b) a multi-channel saturation (all channels simultaneously) -> clip handles it
#       but a proper ASR would reconstruct from the subspace

n_ch, n_t = 15, 5000
sfreq_sim  = 160.0
t_sim      = np.arange(n_t) / sfreq_sim

# --- simulate clean multi-channel EEG with covariance structure ---
# draw a random covariance
A_sim = rng.standard_normal((n_ch, n_ch))
cov   = A_sim @ A_sim.T / n_ch
L = np.linalg.cholesky(cov)
clean = (L @ rng.standard_normal((n_ch, n_t))) * 20e-6   # ~20 µV rms

# --- inject artefacts ---
dirty = clean.copy()
# (a) single-channel spike
dirty[3, 1200:1210] += 500e-6     # 500 µV spike on channel 4
# (b) broadband saturation on a cluster of channels
dirty[7:12, 3000:3100] += 300e-6  # 300 µV offset on channels 8-12

# --- apply clip_extreme_amplitudes ---
clipped = clip_extreme_amplitudes(dirty, z_thresh=5.0)

# --- simplified ASR: PCA-based subspace reconstruction ---
def simple_asr(X, clean_ref, threshold_z=5.0):
    """
    Minimal PCA-based ASR demo (educational, not production-grade).

    Parameters
    ----------
    X : (n_ch, n_t)   data to clean
    clean_ref : (n_ch, n_t_ref)  reference clean data
    threshold_z : float   variance threshold (in units of clean PC variance)

    Returns
    -------
    X_rec : (n_ch, n_t)   reconstructed data
    """
    # Step 1: build clean subspace from reference
    cov_ref = np.cov(clean_ref)
    eigvals, U = np.linalg.eigh(cov_ref)          # ascending order
    eigvals = eigvals[::-1]; U = U[:, ::-1]       # descending

    # Step 2: project each window and check variance ratio
    win_len = int(0.5 * clean_ref.shape[1] / 10)  # small windows
    X_rec   = X.copy()
    n_t_x   = X.shape[1]

    for start in range(0, n_t_x, win_len):
        end  = min(start + win_len, n_t_x)
        Xw   = X[:, start:end]
        T_w  = end - start

        # variance of each PC in this window
        pc_var = np.array([
            (U[:, k] @ Xw @ Xw.T @ U[:, k]) / T_w
            for k in range(len(eigvals))
        ])
        ratio  = pc_var / (eigvals + 1e-30)

        # keep only PCs within threshold
        keep   = ratio <= threshold_z ** 2
        if keep.all():
            continue   # window is clean

        P_keep = U[:, keep] @ U[:, keep].T
        X_rec[:, start:end] = P_keep @ Xw

    return X_rec

asr_out = simple_asr(dirty, clean, threshold_z=5.0)

# %%
# --- Figure 4: compare the three approaches on two channels ---
fig4, axes4 = plt.subplots(3, 2, figsize=(13, 8), sharex=True)

channels_to_show = [3, 9]   # spike channel and saturation cluster channel

for col, ch in enumerate(channels_to_show):
    scale = 1e6   # V -> µV
    for row, (label, sig, colour) in enumerate([
        ("Dirty signal", dirty[ch] * scale, "#d62728"),
        ("clip_extreme_amplitudes", clipped[ch] * scale, "#ff7f0e"),
        ("Simple PCA-ASR",  asr_out[ch] * scale, "#1f77b4"),
    ]):
        ax = axes4[row, col]
        ax.plot(t_sim, clean[ch] * scale, lw=0.5, color="#aaaaaa", label="clean ref")
        ax.plot(t_sim, sig, lw=0.7, color=colour, label=label)
        ax.set_ylabel("µV", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        if row == 0:
            ax.set_title(f"Channel {ch+1}", fontsize=10, fontweight="bold")
        if row == 2:
            ax.set_xlabel("time (s)", fontsize=8)

fig4.suptitle(
    "ASR vs clip_extreme_amplitudes on injected artefacts\n"
    "(grey = clean reference; left = single spike, right = multi-channel saturation)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig4_asr_compare.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 4 saved.")

# %% [markdown]
# ### Key differences: clip vs. true ASR
#
# | Property | `clip_extreme_amplitudes` | True ASR |
# |---|---|---|
# | **Operates on** | each channel independently | all channels jointly (PCA subspace) |
# | **Artefact model** | amplitude exceeds robust z-score | window variance in any PC exceeds clean-data variance |
# | **Reconstruction** | hard clipping (not reconstruction) | back-projects retained PCs onto sensor space |
# | **Preserves signal?** | yes, outside clipped samples | yes, in subspace not flagged as corrupted |
# | **Fails on** | diffuse low-amplitude correlated artefacts | artefacts aligned with largest neural PCs |
# | **Computational cost** | O(C · T) | O(C² · T) |
#
# **When to use which:** `clip_extreme_amplitudes` is a quick first pass for
# isolated electrode pops or brief saturation events. ASR is appropriate when
# the artefact is broadband, cross-channel correlated (e.g. head movement that
# induces field artefacts across the whole cap).

# %% [markdown]
# ## ⚠️ A subtler trap
#
# ### The ICA/ASR "threshold shopping" leak — and why ICA is *still* not free
#
# **The obvious leak** (already noted in Chapter 05 and Chapter 12): if you fit
# ICA on the whole dataset and then use component weights as features in a
# supervised classifier, the unmixing matrix was shaped by test-set statistics —
# a standard form of data leakage.
#
# **The subtler trap:** ICA is *unsupervised*, so fitting it on all data (train +
# test) before splitting is less harmful than fitting a supervised model on all
# data.  But "less harmful" is not "harmless":
#
# 1. **Label-correlated artefacts.**  Suppose subjects blink more during one
#    class (e.g. they relax and blink after the "rest" cue fires).  The blink
#    component then captures label-correlated variance.  If you select which
#    components to *remove* by visually inspecting component topographies
#    **after** you already know the class labels, you may inadvertently keep
#    components that carry label information through their artefact signal —
#    and you will never notice because the topography looks plausible.
#
# 2. **ASR threshold tuning on the same recording.**  ASR has a threshold
#    parameter $\rho$ (default ~5).  If you sweep $\rho$ values and pick the
#    one that maximises downstream classification accuracy on the *same*
#    recording you are cleaning, you have performed hyperparameter search on
#    the test set.  The cleaned data now has lower noise at the exact setting
#    that happens to align residual artefacts favourably with the class
#    boundary — a subtle, almost invisible form of circular analysis.
#
# 3. **Component selection by eyeballing after seeing labels.**  Human
#    inspection of ICA components is not free from bias.  If a researcher
#    knows the expected effect direction, they may be more likely to exclude
#    a component that "looks clean" but whose removal weakens the expected
#    effect, or to keep one that "looks like EMG" but whose removal also
#    weakens a confound.  This is not fraud; it is the ordinary human
#    confirmation bias operating on an underspecified decision.
#
# **The fix:** commit to a component-selection rule *before* seeing accuracy
# numbers — e.g. "exclude all components whose correlation with the frontal
# EOG proxy exceeds 0.4 and that explain < 2 % of total variance" — and do
# not revisit the rule after evaluation.  Log the threshold, the number of
# components removed, and run a permutation test to confirm the cleaned data
# beats chance by more than the threshold search can explain.
