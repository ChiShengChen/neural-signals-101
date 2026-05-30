# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Is It Brain or Artifact? (decoding confounds)
#
# Many "BCI" papers report high accuracy that actually comes from EOG (eye) or EMG
# (muscle) artifacts, not from neural activity at all.  This deep-dive shows you how
# to detect that trap and defend against it.
#
# > **Prerequisites:** main Chapters 03, 05 and 12.
# > **Level:** advanced ★★★★☆
# > **This is the physiological twin of leakage (Chapter 12).**

# %%
# --------------------------------------------------------------------------- #
# Bootstrap — locate neuro101 whether we are run from the repo root or from
# deep-dives/_src/  (one extra parent level vs. the main notebooks).
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

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe under nbconvert
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mne
mne.set_log_level("ERROR")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

from neuro101 import datasets as ds
from neuro101.io import load_bnci_2a_epochs

SMOKE = ds.is_smoke()
RNG = np.random.default_rng(42)

print(f"neuro101 OK  |  MNE {mne.__version__}  |  SMOKE={SMOKE}")

# %% [markdown]
# ---
# ## 1  The problem, in plain English
#
# A classifier is an optimization machine: it finds *any* pattern in the input
# features that predicts the label.  It does **not care** whether that pattern
# comes from neural activity in motor cortex, from muscle twitches recorded on
# frontal electrodes, or from the slow corneoretinal potential of an eye movement.
#
# ### Why eye-movement artifacts are especially dangerous for motor-imagery BCI
#
# | Task design | What actually happens |
# |---|---|
# | Subject imagines moving the **left** hand | Subject also (subtly) tends to glance slightly **left** |
# | Subject imagines moving the **right** hand | Subject also (subtly) tends to glance slightly **right** |
#
# The EOG (electrooculogram) signal for a leftward saccade is a large, slow positive
# deflection at left-frontal electrodes and negative at right-frontal.  This signal
# volume-conducts broadly to the entire frontal EEG — right through channels like
# **Fz, FCz, FC1, FC2**.  These channels are present in almost every BCI montage.
#
# A CSP+LDA decoder will happily pick up that frontal asymmetry and report it as
# "motor imagery decoding accuracy".  When the decoder is moved to a context where
# the eye-movement correlation is broken (different instructions, gaze-fixed
# condition, or simply a different day), accuracy collapses.
#
# ### Why this is *not* just bad preprocessing
#
# Even with ICA eye-blink removal, a **saccade-correlated** EOG component may not
# be cleanly removed (blinks and saccades look different in component space).
# Worse, if the saccade itself was *caused by the task design*, no amount of
# preprocessing can fix it — the only remedy is better experimental control.

# %% [markdown]
# ---
# ## 2  Controlled demo: inject a synthetic "eye-movement" artifact
#
# ### Setup
#
# We load BCI IV 2a (left-hand vs. right-hand motor imagery) and establish a
# baseline CSP+LDA accuracy.  Then we **synthetically inject** a label-correlated
# power modulation onto the frontal/fronto-central channels — mimicking what
# happens when subjects systematically move their eyes with the imagined movement.
#
# **Injection rule:**
# * Class 0 (left-hand): frontal channels (Fz, FC1–4, FCz) get **3× the amplitude**
#   → large increase in frontal alpha/beta power for one class only.
# * Class 1 (right-hand): frontal channels left as-is.
#
# This amplification mimics the frontal power asymmetry produced by a real
# left-biased saccade (left frontal positive EOG → large positive-going
# oscillation when projected onto the bandpass-filtered EEG).
#
# The data already comes bandpass-filtered 8–30 Hz (mu/beta range) by the MOABB
# paradigm, so the artifact must be *in-band* to survive — the multiplicative
# amplitude change achieves this by scaling existing in-band activity.

# %%
# --- Load data ---------------------------------------------------------------
n_subj = 2 if SMOKE else 3
print(f"Loading {n_subj} subject(s) of BCI IV 2a …")
X, y, subj = load_bnci_2a_epochs(n_subjects=n_subj)
n_trials, n_ch, n_times = X.shape
sfreq = ds.DATASETS["bnci_2a"].sfreq_hz

# Channel names for BCI IV 2a (22 EEG channels from MOABB, in order)
CH_NAMES = [
    "Fz",  "FC3", "FC1", "FCz", "FC2", "FC4",   # indices 0-5  (frontal/FC)
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4", "C6",  # indices 6-12 (sensorimotor)
    "CP3", "CP1", "CPz", "CP2", "CP4",            # indices 13-17 (centro-parietal)
    "P1",  "Pz",  "P2",  "POz",                   # indices 18-21 (parietal)
]
assert len(CH_NAMES) == 22

# Channel group indices
FRONTAL_IDX    = list(range(0, 6))                # Fz, FC3, FC1, FCz, FC2, FC4
SENSORIMOTOR_IDX = [7, 9, 11]                     # C3, Cz, C4 — the key MI channels
ABLATION_IDX   = list(range(6, 22))               # everything except frontal/FC

print(f"X shape: {X.shape}  |  class counts: {np.bincount(y)}")
print(f"Frontal channels:     {[CH_NAMES[i] for i in FRONTAL_IDX]}")
print(f"Sensorimotor (C3/Cz/C4): {[CH_NAMES[i] for i in SENSORIMOTOR_IDX]}")

# %%
# --- Inject synthetic label-correlated artifact ------------------------------
# Class 0 (left): scale up frontal by 3×  →  large frontal alpha/beta power
# Class 1 (right): no change
X_art = X.copy()
ARTIFACT_SCALE = 3.0

for i, trial_y in enumerate(y):
    if trial_y == 0:
        X_art[i, FRONTAL_IDX, :] *= ARTIFACT_SCALE

frontal_std_before = X[:, FRONTAL_IDX, :].std()
frontal_std_after  = X_art[y == 0, :, :][:, FRONTAL_IDX, :].std()
print(f"Frontal channel std: before={frontal_std_before:.2f} µV  "
      f"| after (class 0): {frontal_std_after:.2f} µV")

# %%
# --- Build a standard CSP+LDA pipeline and evaluate --------------------------
def make_pipe():
    return Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("Evaluating baseline (no artifact) …")
acc_base = cross_val_score(make_pipe(), X, y, cv=cv, scoring="accuracy")

print("Evaluating with injected artifact …")
acc_art = cross_val_score(make_pipe(), X_art, y, cv=cv, scoring="accuracy")

print("Evaluating artifact data after frontal-channel ablation …")
X_art_abl = X_art[:, ABLATION_IDX, :]   # drop Fz/FC channels
acc_abl = cross_val_score(make_pipe(), X_art_abl, y, cv=cv, scoring="accuracy")

print()
print(f"Baseline accuracy:               {acc_base.mean():.3f} ± {acc_base.std():.3f}")
print(f"With injected artifact:          {acc_art.mean():.3f} ± {acc_art.std():.3f}")
print(f"After frontal-channel ablation:  {acc_abl.mean():.3f} ± {acc_abl.std():.3f}")

# %% [markdown]
# **Reading the numbers.**
# The "with artifact" condition shows a large accuracy jump — the CSP filters
# immediately latch onto the frontal power asymmetry.  After ablating the frontal
# channels (dropping Fz, FC1–4, FCz), accuracy returns to near-baseline.  The
# decoder's apparent gain was entirely artifactual.

# %% [markdown]
# ### Figure 1 — Accuracy: baseline / artifact / ablation
#
# The three-bar plot makes the story undeniable.  The red dashed line marks chance
# level (50 %).  A "significant" accuracy jump appears when the artifact is present,
# and disappears the moment the contaminated channels are removed.

# %%
fig, ax = plt.subplots(figsize=(7, 5))

labels_bar = ["Baseline\n(no artifact)", "Injected\nartifact", "Ablation\n(drop frontal)"]
means = [acc_base.mean(), acc_art.mean(), acc_abl.mean()]
stds  = [acc_base.std(),  acc_art.std(),  acc_abl.std()]
colors = ["#4c72b0", "#c44e52", "#55a868"]

bars = ax.bar(labels_bar, means, yerr=stds, capsize=6,
              color=colors, edgecolor="k", linewidth=0.8, width=0.5,
              error_kw=dict(elinewidth=1.5, ecolor="black"))

# Annotate values on top of bars
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
            f"{m:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Chance (50 %)")
ax.set_ylim(0.3, 1.15)
ax.set_ylabel("Accuracy (5-fold CV)", fontsize=12)
ax.set_title("CSP+LDA accuracy — does the decoder read brain or artifact?\n"
             f"BCI IV 2a, {n_subj} subject(s), left vs. right hand", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlabel("Condition", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/artifact_confounds_fig1.png", dpi=120)
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# ---
# ## 3  Honesty check B — inspect which channels the model weights most
#
# If a motor-imagery decoder is genuinely reading cortical oscillations, its
# **spatial patterns** (not filters — see Note below) should load most heavily on
# **central/sensorimotor channels**: C3, Cz, C4, and their neighbours.
#
# > **Note on patterns vs. filters.**  The spatial *filter* $w$ is what you
# > multiply the data by; it is *not* the source topography.  The spatial
# > *pattern* $a$ is what the source looks like on the scalp, obtained via
# > $a = \Sigma_c\, w / (w^\top \Sigma_c w)$ where $\Sigma_c$ is the composite
# > covariance.  MNE stores this as `CSP.patterns_`.  **Always plot patterns,
# > not filters**, when you want to know which channels are physiologically active.
#
# We fit CSP on the full data for each condition (not inside CV — this is a
# diagnostic, not a performance estimate) and compare the mean absolute pattern
# weight across channels.

# %%
# Fit CSP on full data (diagnostic only — no CV split needed here)
csp_base_full = CSP(n_components=4, reg="ledoit_wolf", log=True)
csp_art_full  = CSP(n_components=4, reg="ledoit_wolf", log=True)
csp_base_full.fit(X, y)
csp_art_full.fit(X_art, y)

# patterns_ shape: (n_channels, n_channels) — full set of 22 components
# The 4 components used are: [0, 1, -2, -1]  (2 most class-0, 2 most class-1)
USED_COMPS = [0, 1, -2, -1]
base_patterns = np.abs(csp_base_full.patterns_[USED_COMPS]).mean(axis=0)  # (22,)
art_patterns  = np.abs(csp_art_full.patterns_[USED_COMPS]).mean(axis=0)   # (22,)

# Normalise each to its own max for easy visual comparison
base_patterns_norm = base_patterns / base_patterns.max()
art_patterns_norm  = art_patterns  / art_patterns.max()

print("Mean abs pattern weight per channel (normalised to 1.0)")
print(f"{'Channel':8s}  {'Baseline':10s}  {'Artifact':10s}  {'Group':15s}")
for i, ch in enumerate(CH_NAMES):
    grp = ("frontal/FC" if i < 6 else
           "sensorimotor" if i in SENSORIMOTOR_IDX else
           "other")
    print(f"{ch:8s}  {base_patterns_norm[i]:10.3f}  {art_patterns_norm[i]:10.3f}  {grp}")

# %% [markdown]
# **Key observation.**  Under the artifact condition, the frontal/FC channels
# dominate the CSP pattern with weights that are orders of magnitude larger than
# in the baseline.  Real motor imagery shows a more balanced picture, with central
# channels (C3/Cz/C4) carrying substantial weight.  A physiologically implausible
# pattern — all the weight on frontal, almost nothing on sensorimotor — is a red flag.

# %% [markdown]
# ### Figure 2 — Channel weight profiles: baseline vs. injected artifact
#
# A bar chart per channel (sorted by anatomical position: frontal → central →
# parietal) reveals the shift in spatial emphasis.  **Red** bars = artifact
# condition.  **Blue** bars = clean baseline.  The frontal channels (left section)
# should only be lightly weighted for a genuine motor decoder.

# %%
fig, ax = plt.subplots(figsize=(13, 5))

x_pos = np.arange(n_ch)
width = 0.38

ax.bar(x_pos - width / 2, base_patterns_norm, width,
       label="Baseline (no artifact)", color="#4c72b0", alpha=0.85, edgecolor="k", lw=0.5)
ax.bar(x_pos + width / 2, art_patterns_norm, width,
       label="With injected artifact", color="#c44e52", alpha=0.85, edgecolor="k", lw=0.5)

# Shade frontal region
ax.axvspan(-0.5, 5.5, alpha=0.08, color="orange", label="Frontal/FC channels")
# Shade sensorimotor region
ax.axvspan(6.5, 12.5, alpha=0.08, color="green", label="Sensorimotor (C3/Cz/C4)")

ax.set_xticks(x_pos)
ax.set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Normalised mean |CSP pattern weight|", fontsize=10)
ax.set_title("Spatial emphasis of the CSP decoder — which channels does it trust?\n"
             "(high weight = decoder relies heavily on that channel)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-0.6, n_ch - 0.4)
ax.set_ylim(0, 1.25)

# Annotate the two channel groups
ax.text(2.5, 1.18, "Frontal / FC\n(artifact target)", ha="center",
        fontsize=8, color="darkorange", fontweight="bold")
ax.text(9.0, 1.18, "Sensorimotor\n(true MI region)", ha="center",
        fontsize=8, color="darkgreen", fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/artifact_confounds_fig2.png", dpi=120)
plt.show()
print("Figure 2 saved.")

# %% [markdown]
# **Reading Figure 2.**  In the artifact condition (red), the frontal/FC channels
# tower over everything else — the decoder is essentially a frontal power meter.
# In the clean baseline (blue), central and parietal channels contribute
# meaningfully alongside fronto-central ones.  For a genuine motor decoder you
# want to see C3/Cz/C4 prominent; if those are dwarfed by Fz/FCz, suspect an
# artifact-driven result.

# %% [markdown]
# ---
# ## 4  Real-data caution — decodability can sit in non-sensorimotor channels
#
# Even without our synthetic injection, a portion of decodability in BCI IV 2a
# comes from channels outside the canonical C3/Cz/C4 sensorimotor strip.  This
# is not necessarily artifactual — it could be genuine premotor or supplementary
# motor area activity — but it is worth checking.
#
# A quick sanity check: compare cross-validated accuracy when using **only**
# the sensorimotor channels (C3, Cz, C4 and their immediate neighbours) vs. the
# full 22-channel set.  If the full set substantially outperforms the
# sensorimotor-only set, ask yourself: where does the extra information come from?

# %%
SM_EXTENDED = [6, 7, 8, 9, 10, 11, 12]   # C5, C3, C1, Cz, C2, C4, C6
X_sm = X[:, SM_EXTENDED, :]

print("Evaluating sensorimotor-only subset (C5/C3/C1/Cz/C2/C4/C6) …")
acc_sm = cross_val_score(make_pipe(), X_sm, y, cv=cv, scoring="accuracy")
print(f"Sensorimotor-only accuracy: {acc_sm.mean():.3f} ± {acc_sm.std():.3f}")
print(f"Full-montage accuracy:      {acc_base.mean():.3f} ± {acc_base.std():.3f}")
delta = acc_base.mean() - acc_sm.mean()
print(f"Gain from non-sensorimotor channels: {delta:+.3f}")
print()
print("If the gain is small (< 0.05), the extra channels contribute little.")
print("If large, investigate WHICH channels drive it — and whether those are")
print("physiologically plausible for the task.")

# %% [markdown]
# ### Practical artifact-rejection advice
#
# The gold standard for ruling out artifact-driven decoding is:
#
# 1. **ICA + EOG regression** — remove the independent components identified as
#    eye or muscle artifacts *before* fitting the decoder.  Compare accuracy before
#    and after; if accuracy collapses, you were reading artifacts.
#
# 2. **Channel ablation test** — iteratively remove frontal/peripheral channels
#    and check whether accuracy survives.  A motor-imagery decoder should be
#    robust to losing Fz and FC channels.
#
# 3. **Spatial plausibility check** — plot the CSP activation *patterns* (not
#    filters!) as a topographic map.  For left-vs-right hand imagery you expect a
#    bilateral contralateral distribution centred around C3/C4.
#
# 4. **Gaze-control condition** — collect a separate block where subjects fixate a
#    cross and the correlation between gaze direction and class label is broken.
#    If accuracy does not hold up, the original result was artifactual.

# %%
# --- Summary printout --------------------------------------------------------
print("=" * 58)
print("SUMMARY — Artifact confound demo")
print("=" * 58)
print(f"Baseline acc (all 22 ch, no artifact):   {acc_base.mean():.3f}")
print(f"With injected frontal artifact:           {acc_art.mean():.3f}")
print(f"After ablating frontal channels (Fz/FC):  {acc_abl.mean():.3f}")
print(f"Sensorimotor-only subset:                 {acc_sm.mean():.3f}")
print()
print("Pattern diagnosis:")
base_f   = base_patterns_norm[FRONTAL_IDX].mean()
base_sm  = base_patterns_norm[SENSORIMOTOR_IDX].mean()
art_f    = art_patterns_norm[FRONTAL_IDX].mean()
art_sm   = art_patterns_norm[SENSORIMOTOR_IDX].mean()
print(f"  Baseline  — mean frontal pattern: {base_f:.2f}  |  SM: {base_sm:.2f}"
      f"  |  ratio: {base_f/base_sm:.2f}")
print(f"  Artifact  — mean frontal pattern: {art_f:.2f}  |  SM: {art_sm:.2f}"
      f"  |  ratio: {art_f/art_sm:.2f}")
print()
print("Red flag threshold: frontal/SM ratio > 3 warrants investigation.")

# %% [markdown]
# ---
# ## ⚠️ A subtler trap
#
# The demo above uses a dramatic 3× amplitude injection — easy to catch.  Real-world
# confounds are subtler.  Here is one that survives standard ICA cleaning and fools
# even experienced researchers:
#
# ### Volume-conducted residual: ICA removes the *component* but not the *signal*
#
# Standard ICA-based EOG removal works by identifying the component whose time
# course correlates with an external EOG channel, then projecting it out of the
# EEG.  This works well for **vertical blinks** because they have a canonical,
# strong, unique topography.
#
# **Saccades (horizontal eye movements) are harder.**
#
# * A saccade produces a slow corneoretinal potential that volume-conducts
#   differentially across the scalp — large at ipsilateral frontal, small at
#   central, near-zero at occipital.
# * ICA may decompose this into *two or more* components: a dominant frontal one
#   and a weaker residual that projects onto fronto-central channels (FCz, FC1,
#   FC2).  The dominant component is flagged and removed; the residual is not.
# * After ICA cleaning, the frontal channels appear clean by visual inspection
#   and by EOG correlation metrics.  Yet a weak, task-correlated saccade signal
#   persists in FCz/FC1/FC2.
#
# **Why no preprocessing can fix this if the saccade is task-correlated:**
#
# Suppose your motor-imagery paradigm shows a *directional cue* (an arrow pointing
# left or right) before each trial.  Subjects reflexively saccade toward the arrow.
# Even a covert, suppressed saccade leaves a corneoretinal trace.  This trace is
# perfectly correlated with the class label — not because of motor imagery, but
# because of *cue design*.
#
# No artifact rejection step can decorrelate a signal that is perfectly aligned
# with the label.  An ICA component that correlates with the label will be *kept*
# by standard ICA cleaning pipelines (which flag components by EOG-channel
# correlation, not by class-label correlation).  The residual eye signal survives
# in the data, the decoder learns it, and the result looks like motor imagery
# decoding.
#
# **The only fix is experimental design:**
# * Use a central fixation cross throughout; never show directional visual cues.
# * Use auditory or neutral symbolic cues (the words "LEFT" / "RIGHT") that do
#   not trigger reflexive gaze shifts.
# * Record and analyse gaze position; confirm it does not differ between classes.
# * Test that removing all channels anterior to Cz does not collapse accuracy.
#
# **The meta-lesson:**  Artifact confounds come in two flavours.  *Preprocessing-
# fixable* confounds (blink artifacts, cable noise) can be removed after data
# collection.  *Design-level* confounds (systematic eye movements correlated with
# the task structure) cannot — they are baked into the data.  The only protection
# is prospective experimental control, not retrospective signal processing.
