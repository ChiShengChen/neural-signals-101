# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Source Localization (the inverse problem)
#
# *Where in the brain did the scalp signal come from?*
#
# > **Prerequisites:** main Chapters 01 and 02.
# > **Level:** advanced ★★★★☆
# > **Downloads the fsaverage template (cached). Headless 2-D visuals only.**

# %% [markdown]
# ## 0 — Bootstrap

# %%
# --------------------------------------------------------------------------- #
# Bootstrap — locate neuro101 whether run from the repo root or deep-dives/_src/
# --------------------------------------------------------------------------- #
import sys
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(6):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src"))
            break
        _p = _p.parent
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe under nbconvert
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds

SMOKE = ds.is_smoke()
RNG = np.random.default_rng(42)

print(f"MNE {mne.__version__}  |  smoke={SMOKE}  |  imports OK")

# %% [markdown]
# ---
# ## 1 — The big picture: forward vs. inverse
#
# ### Volume conduction: why EEG is blurry
#
# Chapter 01 showed that every scalp electrode is a **spatial average** of many
# underlying neural currents. The skull, cerebrospinal fluid and scalp tissue are
# ohmic conductors — they smear current like a fog. A single cortical patch
# (a "dipole") spreads its potential across the entire head. Conversely, a single
# electrode picks up contributions from dozens of cortical patches simultaneously.
#
# ### The forward problem (sources → scalp) — well-posed
#
# If we *know* where and how strongly each cortical patch fires, we can compute the
# expected scalp potential exactly using Maxwell's equations. MNE calls this the
# **forward solution** or **leadfield matrix** **L** ∈ ℝ^{n_channels × 3·n_sources}:
#
# $$\mathbf{x}(t) = \mathbf{L}\, \mathbf{j}(t) + \boldsymbol{\varepsilon}(t)$$
#
# where **x** is the electrode data, **j** is the vector of source amplitudes, and
# **ε** is noise. Given **j**, computing **x** is unique and stable — the forward
# problem is *well-posed*.
#
# ### The inverse problem (scalp → sources) — ill-posed
#
# We observe **x** and want **j**. The system is *hugely underdetermined*:
# 64 electrodes vs. ≥10,000 cortical source locations. An infinite family of
# source configurations produce exactly the same scalp map. This is the
# **ill-posed inverse problem**.
#
# **Regularization** breaks the deadlock by adding a prior: the minimum-norm
# estimate (MNE) prefers the solution with the smallest total source power. dSPM
# normalises MNE by a noise estimate so the map becomes a *z*-score-like quantity.
# sLORETA and LCMV beamformers add different flavors of prior. None of them
# recover "ground truth" — they recover the *most parsimonious* solution consistent
# with the data *and* the prior.

# %% [markdown]
# ---
# ## 2 — Build the pieces (fsaverage template)
#
# We use FreeSurfer's **fsaverage** template brain — no subject MRI needed. MNE
# ships pre-built source spaces and BEM solutions for fsaverage, so we only need a
# single `fetch_fsaverage()` call (downloaded once, then cached).
#
# For the data to localize we construct a clean **simulated evoked response** on the
# standard 10-20 montage. This is the most robust headless choice: no internet
# dependency beyond the fsaverage template, no BIDS complexity, no EEG hardware
# quirks. The simulated waveform contains a P300-like component peaking at ~300 ms.

# %%
# ------------------------------------------------------------------ #
# 2a. Fetch fsaverage (downloads once, then cached in ~/mne_data/)   #
# ------------------------------------------------------------------ #
_FSAVERAGE_OK = False
_FWD_OK = False
stc = None
stc_mne_sol = None
evoked = None

try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = str(Path(fs_dir).parent)
    print(f"fsaverage directory: {fs_dir}")
    _FSAVERAGE_OK = True
except Exception as e:
    print(f"[WARNING] Could not fetch fsaverage: {e}")
    print("         Falling back to simulated-only mode.")

# %%
# ------------------------------------------------------------------ #
# 2b. Build a simulated evoked response on the standard_1020 montage #
# ------------------------------------------------------------------ #
montage = mne.channels.make_standard_montage("standard_1020")
# Use 32 channels for speed; full 10-20 would also work.
_CH_NAMES = montage.ch_names[:32]
_SFREQ = 250.0
_N_TIMES = 62 if SMOKE else 125   # 250 ms in smoke, 500 ms normally
_T = np.linspace(0.0, _N_TIMES / _SFREQ, _N_TIMES)

# Simulate data: white noise + P300-like Gaussian bump at 300 ms
_data = RNG.standard_normal((len(_CH_NAMES), _N_TIMES)) * 1e-7
for _i in range(len(_CH_NAMES)):
    _amp = 3e-6 * (1.0 + 0.3 * RNG.standard_normal())
    _data[_i] += _amp * np.exp(-((_T - 0.30) ** 2) / (2 * 0.04 ** 2))

info = mne.create_info(ch_names=_CH_NAMES, sfreq=_SFREQ, ch_types="eeg")
evoked = mne.EvokedArray(_data, info, tmin=0.0, comment="simulated P300-like ERP")
evoked.set_montage(montage)
# EEG average reference is required for the inverse operator
evoked.set_eeg_reference(projection=True)

print(
    f"Evoked: {evoked.info['nchan']} channels × {evoked.times.size} time points  "
    f"({evoked.tmin:.2f} – {evoked.tmax:.2f} s)"
)

# %%
# ------------------------------------------------------------------ #
# 2c. Source space + BEM + forward solution                          #
# ------------------------------------------------------------------ #
if _FSAVERAGE_OK:
    try:
        _bem_dir = os.path.join(fs_dir, "bem")
        _src_file = os.path.join(_bem_dir, "fsaverage-ico-5-src.fif")
        _bem_sol_file = os.path.join(_bem_dir, "fsaverage-5120-5120-5120-bem-sol.fif")
        _trans_file = os.path.join(_bem_dir, "fsaverage-trans.fif")

        src = mne.read_source_spaces(_src_file)
        bem = mne.read_bem_solution(_bem_sol_file)
        print(
            f"Source space: {src[0]['nuse'] + src[1]['nuse']:,} vertices  |  "
            f"BEM: {len(bem['surfs'])} surfaces"
        )

        fwd = mne.make_forward_solution(
            evoked.info,
            trans=_trans_file,
            src=src,
            bem=bem,
            eeg=True,
            meg=False,
            verbose=False,
        )
        print(f"Forward solution: {fwd['nsource']:,} sources × {fwd['nchan']} channels")
        _FWD_OK = True
    except Exception as e:
        print(f"[WARNING] Forward solution failed: {e}")
        print("         Falling back to simulated-stc mode.")

# %%
# ------------------------------------------------------------------ #
# 2d. Noise covariance + inverse operator + dSPM / MNE              #
# ------------------------------------------------------------------ #
if _FWD_OK:
    try:
        # Ad-hoc noise covariance (diagonal, sensor-type-appropriate)
        noise_cov = mne.make_ad_hoc_cov(evoked.info, verbose=False)

        inv_op = mne.minimum_norm.make_inverse_operator(
            evoked.info, fwd, noise_cov, verbose=False
        )
        print("Inverse operator built successfully")

        _lambda2 = 1.0 / 9.0  # SNR = 3 → standard regularization choice
        stc = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2=_lambda2, method="dSPM", verbose=False
        )
        stc_mne_sol = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2=_lambda2, method="MNE", verbose=False
        )
        print(f"dSPM SourceEstimate: {stc.data.shape[0]:,} vertices × {stc.data.shape[1]} times")
        print(
            f"dSPM range: {stc.data.min():.2f} – {stc.data.max():.2f} (pseudo-z scores)"
        )
    except Exception as e:
        print(f"[WARNING] Inverse solution failed: {e}")

# Fallback: if MNE inverse failed, generate a dummy stc-like structure
if stc is None:
    print("[INFO] Using synthetic SourceEstimate for visualization.")
    _n_src = 500
    _stc_data = RNG.standard_normal((_n_src, _N_TIMES)) * 0.5
    # Add a bump at ~300 ms in a few vertices
    for _vi in [10, 42, 99]:
        _stc_data[_vi] += 5.0 * np.exp(-((_T - 0.30) ** 2) / (2 * 0.04 ** 2))

    class _FakeStc:
        data = _stc_data
        times = _T

    stc = _FakeStc()
    stc_mne_sol = _FakeStc()

# %% [markdown]
# ---
# ## 3 — dSPM vs. MNE: what's the difference?
#
# Both methods solve the same under-determined least-squares problem, differing
# only in post-processing:
#
# | Method | Formula | Interpretation |
# |--------|---------|----------------|
# | **MNE** | $\hat{\mathbf{j}} = \mathbf{W} \mathbf{x}$ where **W** minimises $\|\mathbf{j}\|^2$ | Current amplitude (A·m); biased toward superficial sources |
# | **dSPM** | MNE ÷ estimated noise std per source | Pseudo-*z* score; better spatial selectivity than raw MNE |
# | **sLORETA** | MNE ÷ estimated source variance | Exact zero-error localization for a single dipole |
#
# The **regularization parameter λ²** controls the noise-vs-smoothness trade-off.
# A common choice is λ² = 1/SNR² with SNR = 3 → λ² ≈ 0.11.
# *Larger λ²* → smoother (more spread-out), less noisy map.
# *Smaller λ²* → sharper but noise-amplifying.

# %% [markdown]
# ---
# ## 4 — Visualization (headless 2-D only)
#
# `stc.plot()` would open a 3-D VTK/PyVista window and crash under nbconvert.
# We plot the source estimates in 2-D instead: a source × time image, peak-vertex
# time courses, and the scalp topomap that anchors interpretation.

# %%
# ------------------------------------------------------------------ #
# Figure 1 — Scalp topomap of the simulated evoked response          #
# ------------------------------------------------------------------ #
_times_plot = [t for t in [0.10, 0.20, 0.30, 0.40] if t <= evoked.tmax]
fig1 = evoked.plot_topomap(times=_times_plot, show=False)
fig1.suptitle(
    "Simulated P300-like ERP — scalp topomaps\n"
    "(input to the source localization pipeline)",
    fontsize=10, y=1.02,
)
# Note: plot_topomap manages its own layout engine (constrained); do not call
# tight_layout() on it or a second conflicting engine is set.
plt.show()
print("Figure 1: scalp topomaps rendered OK")

# %%
# ------------------------------------------------------------------ #
# Figure 2 — Source × time image (dSPM)                              #
# ------------------------------------------------------------------ #
# Showing only the first 500 vertices keeps the image readable;
# for the full ico-5 solution there are 20 484 cortical vertices.
_n_show = min(500, stc.data.shape[0])
_vmax = np.percentile(np.abs(stc.data[:_n_show]), 99)

fig2, axes2 = plt.subplots(
    1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]},
    layout="constrained",
)

# Left: heat-map of source amplitudes
im = axes2[0].imshow(
    stc.data[:_n_show],
    aspect="auto",
    origin="lower",
    extent=[float(stc.times[0]), float(stc.times[-1]), 0, _n_show],
    cmap="RdBu_r",
    vmin=-_vmax,
    vmax=_vmax,
)
axes2[0].axvline(x=0.30, color="k", linestyle="--", linewidth=0.8, label="t = 300 ms")
axes2[0].set_xlabel("Time (s)", fontsize=11)
axes2[0].set_ylabel(f"Cortical vertex index (first {_n_show})", fontsize=11)
axes2[0].set_title("dSPM source estimates", fontsize=12)
axes2[0].legend(fontsize=9)
fig2.colorbar(im, ax=axes2[0], label="dSPM (pseudo-z)")

# Right: histogram of peak-time amplitudes
_peak_t_idx = int(np.argmin(np.abs(stc.times - 0.30)))
_peak_amps = stc.data[:, _peak_t_idx]
axes2[1].hist(
    _peak_amps, bins=40, orientation="horizontal", color="steelblue", edgecolor="white", linewidth=0.4
)
axes2[1].axhline(0, color="k", linewidth=0.6)
axes2[1].set_xlabel("Count", fontsize=10)
axes2[1].set_ylabel("dSPM amplitude at t = 300 ms", fontsize=10)
axes2[1].set_title("Amplitude\ndistribution", fontsize=11)

fig2.suptitle(
    "Source space view of the dSPM inverse solution\n"
    f"({stc.data.shape[0]:,} cortical vertices, λ² = 1/9)",
    fontsize=11,
)
plt.show()
print("Figure 2: source × time image rendered OK")

# %%
# ------------------------------------------------------------------ #
# Figure 3 — Peak-vertex time courses: dSPM vs. MNE                 #
# ------------------------------------------------------------------ #
_top_k = 5
_peak_vertices = np.argsort(np.abs(stc.data).max(axis=1))[-_top_k:]

fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

_colors = plt.cm.tab10(np.linspace(0, 0.5, _top_k))

for _rank, _vi in enumerate(_peak_vertices[::-1]):
    _lbl = f"vertex {_vi}" if _rank > 0 else f"vertex {_vi} (peak)"
    axes3[0].plot(stc.times, stc.data[_vi], color=_colors[_rank], label=_lbl, linewidth=1.2)
    axes3[1].plot(stc_mne_sol.times, stc_mne_sol.data[_vi], color=_colors[_rank], linewidth=1.2)

axes3[0].axvline(0.30, color="k", linestyle="--", linewidth=0.8)
axes3[1].axvline(0.30, color="k", linestyle="--", linewidth=0.8, label="t = 300 ms")
axes3[0].set_ylabel("dSPM (pseudo-z)", fontsize=10)
axes3[1].set_ylabel("MNE (A·m)", fontsize=10)
axes3[1].set_xlabel("Time (s)", fontsize=10)
axes3[0].set_title(f"Top-{_top_k} peak-amplitude vertices: dSPM (upper) vs. MNE (lower)", fontsize=11)
axes3[0].legend(fontsize=8, ncol=2, loc="upper left")
axes3[1].legend(fontsize=9, loc="upper left")
for _ax in axes3:
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.grid(True, alpha=0.3)

fig3.tight_layout()
plt.show()
print("Figure 3: peak-vertex time courses rendered OK")

# %% [markdown]
# ---
# ## 5 — What these maps mean (and don't mean)
#
# ### How to read Figure 2
# Each row is one cortical vertex; color encodes its estimated source amplitude at
# that moment. The bulk of the cortex has near-zero activity (light / white). A
# handful of vertices light up near t = 300 ms — that is where the minimum-norm
# solution "places" the generator of our simulated P300-like component. The
# histogram (right) shows how sparse the activated set is.
#
# ### Why dSPM ≠ ground truth
# The simulated data contained no true dipole at all — the activity was spatially
# uniform Gaussian noise plus a temporally Gaussian envelope. dSPM nonetheless
# produces a plausible-looking focal blob. The "localization" is a product of the
# *minimum-norm prior* (assume the fewest, smallest sources), not of a physical
# measurement of cortical current. Real EEG source localization is the same:
# the map tells you what the regularizer *guesses*, not where the neurons fired.

# %% [markdown]
# ---
# ## ⚠️ A subtler trap
#
# It is tempting to report "we localized the P300 to the posterior parietal cortex
# at (x, y, z) mm precision." Three reasons that claim overstates the evidence:
#
# **1. The ill-posedness never goes away.**
# Regularization makes the inverse problem *solvable*, but the solution is not
# unique. Different regularizers (MNE, dSPM, sLORETA, LCMV, eLORETA, …) will
# return overlapping but shifted source estimates from identical scalp data. The
# "blob" you see is the intersection of your data and your *prior*, not a unbiased
# estimator of the true generator.
#
# **2. Millimetre precision from centimetre electrodes is physically implausible.**
# The skull smears potentials across ~3 cm. Source localization accuracy in
# empirical studies with known intracranial sources is 5–20 mm on average even
# with 64+ channels. Reporting sub-centimetre coordinates from 32-channel EEG
# is wishful thinking.
#
# **3. Source-space features do not magically remove volume-conduction leakage.**
# A common pattern in BCI papers: apply source localization, extract
# source-time-series features, feed to a classifier, celebrate. But the inverse
# operator is a *linear* combination of the electrode signals. If two cortical
# regions are correlated at the scalp (because their dipoles project similarly),
# the estimated source signals will also be correlated. The leakage moves from
# sensor space to source space; it does not disappear. True functional
# connectivity between source estimates requires additional methods (e.g.
# imaginary coherence, orthogonalization, or beamformer spatial filters with
# careful orientation selection).
#
# **In short:** source localization is a powerful hypothesis-generation tool.
# Treat its output as *probabilistic spatial priors* on where activity *might*
# be, not as anatomically verified measurements. Use it to motivate electrode
# placement or constrain group-level neuroimaging analyses — never as a
# substitute for invasive recordings or high-resolution MRI-guided source
# modelling.
