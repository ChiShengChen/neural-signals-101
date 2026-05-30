# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 10 — Paradigms & Applications
#
# A quick tour of the main things people *do* with neural signals, with one
# minimal runnable example each. Some use the real datasets you already have
# cached; a few use clearly-labelled **simulations** where the real dataset is too
# large for a 101 course (with pointers to the real data).
#
# ## Learning objectives
# Recognise and run a tiny example of each paradigm:
# 1. **Motor imagery** (move-by-thinking) — real BCI IV 2a.
# 2. **P300 / ERP** (brain's response to a rare stimulus) — simulated.
# 3. **SSVEP** (steady-state response to flicker) — simulated.
# 4. **Sleep staging** — real Sleep-EDF.
# 5. **Seizure detection** — simulated (imbalanced, links to Chapter 12).
# 6. **Neural decoding / brain-to-text** — concept + small multi-class decode.
#
# **Runtime:** ~2–3 min.

# %% [markdown]
# > **Prerequisites:** Chapters 01 and 07.
# > **Difficulty:** ★★★☆☆

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from neuro101 import io, datasets as ds, features as ft
from neuro101.eval import leakage_safe_pipeline, make_subject_split, evaluate_with_variance, make_block_split

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## 1. Motor imagery (real: BCI IV 2a)
#
# Decode imagined left- vs right-hand movement from the mu/beta rhythm. We did the
# full treatment in Chapters 05–06; here is the one-cell version with an honest
# subject-independent score.
#
# **Why it works:** When you *imagine* moving your hand, the primary motor cortex
# on the **opposite (contralateral) side** of your brain becomes active — even
# without any real movement. This activity suppresses the **mu rhythm (8–12 Hz)**
# and **beta rhythm (18–26 Hz)** over the sensorimotor cortex, a process called
# **event-related desynchronisation (ERD)** (introduced in Chapter 07). Because
# left-hand imagery drives the *right* hemisphere and right-hand imagery drives the
# *left* hemisphere, the spatial asymmetry is the decodable signal.

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Xmi, ymi, smi = io.load_bnci_2a_epochs(n_subjects=2)
pipe = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LinearDiscriminantAnalysis())])
res = evaluate_with_variance(pipe, Xmi, ymi, cv=lambda: make_subject_split(smi),
                             scoring="accuracy", seeds=(0,))
print(f"Motor imagery LOSO accuracy: {res['accuracy']['mean']:.3f} (chance 0.5)")

# %% [markdown]
# ## 2. P300 / ERP (simulated)
#
# An **ERP** (Event-Related Potential) is the brain's averaged response to an
# event. The **P300** is a positive bump ~300 ms after a rare, attended stimulus —
# the basis of P300 "spellers". Single trials are buried in noise; **averaging many
# trials** reveals the ERP. We simulate target (rare) vs non-target trials.
#
# *Real data:* MOABB P300 datasets (e.g. `BNCI2014_009`), or the MNE sample auditory task.
#
# **Why it works:** The P300 is generated when the brain detects something
# *surprising and relevant* — a rare item in a stream of common ones. This is
# called the **oddball paradigm**: targets are deliberately made rare (e.g. 1 in 6
# flashes in a speller) so the brain produces this large positive deflection only
# for them. Because non-targets are common, no such response appears, giving a
# clean contrast signal once you average across many repetitions to reduce noise.

# %%
sf, T = 250.0, int(0.8 * 250)
t = np.arange(T) / sf
def p300_wave(amp):  # a bump centred ~300 ms
    return amp * np.exp(-((t - 0.30) ** 2) / (2 * 0.05 ** 2))
n = 200
targets = np.array([p300_wave(8e-6) + 5e-6 * rng.standard_normal(T) for _ in range(n)])
nontar  = np.array([p300_wave(0.0)  + 5e-6 * rng.standard_normal(T) for _ in range(n)])

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(t * 1000, targets[0] * 1e6, color="#bbb", lw=0.6, label="single target trial (noisy)")
ax.plot(t * 1000, targets.mean(0) * 1e6, color="#c44e52", lw=2, label="target average (P300!)")
ax.plot(t * 1000, nontar.mean(0) * 1e6, color="#4c72b0", lw=2, label="non-target average")
ax.axvline(300, ls="--", color="gray"); ax.set(xlabel="Time (ms)", ylabel="µV",
          title="P300: averaging reveals the ~300 ms bump"); ax.legend()
plt.show()

# %% [markdown]
# ## 3. SSVEP (simulated)
#
# If you stare at a light flickering at, say, 12 Hz, your visual cortex produces a
# **steady-state** oscillation at 12 Hz (and harmonics). SSVEP BCIs show several
# flickering targets at different frequencies and read off which one you look at —
# just find the strongest peak in the PSD.
#
# *Real data:* MOABB SSVEP datasets (e.g. `SSVEPExo`, `Nakanishi2015`).
#
# **Why it works:** The visual cortex is wired to follow the rhythm of its input —
# a property called **frequency tagging**. When a flickering stimulus at frequency
# *f* continuously drives the retina, the entire visual pathway (retina → thalamus
# → V1) entrains to *f* and its harmonics. Because each target in the BCI interface
# flickers at a *different* frequency, reading out which frequency dominates the
# occipital EEG immediately tells you where the user is looking — no training data
# needed.

# %% [markdown]
# **Before running:** the user attends the 12 Hz flicker — which frequency will
# dominate the power spectrum: 8 Hz, 12 Hz, or 15 Hz? Write your prediction, then
# run the cell and check the plot.

# %%
sf = 250.0; t = np.arange(int(3 * sf)) / sf
stim_freqs = [8.0, 12.0, 15.0]
attended = 12.0  # the frequency the user is "looking at"
eeg = sum((1.0 if f == attended else 0.05) * np.sin(2 * np.pi * f * t) for f in stim_freqs)
eeg = eeg + 1.5 * rng.standard_normal(t.size)
f, psd = welch(eeg, sf, nperseg=int(sf * 2))
detected = stim_freqs[int(np.argmax([psd[np.argmin(abs(f - sf0))] for sf0 in stim_freqs]))]
print(f"Attended {attended} Hz -> detected {detected} Hz (peak picking)")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(f, psd, color="#55a868")
for sf0 in stim_freqs: ax.axvline(sf0, ls="--", color="gray")
ax.set(xlim=(5, 20), xlabel="Frequency (Hz)", ylabel="Power",
       title="SSVEP: the attended flicker frequency dominates the spectrum")
plt.show()

# %% [markdown]
# ## 4. Sleep staging (real: Sleep-EDF)
#
# Classify each 30-second epoch of overnight EEG into a sleep stage
# (Wake/N1/N2/N3/REM). Classes are very imbalanced — we use **balanced accuracy**
# (Chapter 12 explains why) and a block-aware split within the night.
#
# **Why it works:** Each sleep stage has a distinct EEG fingerprint. During **N2**
# you see **sleep spindles** (brief 12–15 Hz bursts) and **K-complexes** (large
# sharp waves). **N3** (deep sleep) is dominated by slow **delta waves** (< 4 Hz,
# high amplitude). **REM** looks more like wakefulness — low-amplitude
# mixed-frequency activity with bursts of rapid eye movements. These stage-specific
# rhythms arise from thalamocortical circuits switching between different operating
# modes, making them reliably classifiable from band power features.

# %%
from sklearn.ensemble import RandomForestClassifier
Xsl, ysl, ssl = io.load_sleep_edf_epochs(n_subjects=1)
sf_sl = ds.DATASETS["sleep_edf"].sfreq_hz
Fsl = ft.bandpower(Xsl, sf_sl)  # band power per 30-s epoch
clf = leakage_safe_pipeline([("rf", RandomForestClassifier(n_estimators=100, random_state=0))])
res_sl = evaluate_with_variance(clf, Fsl, ysl,
                                cv=lambda: make_block_split(len(ysl), n_splits=4),
                                scoring=("accuracy", "balanced_accuracy"), seeds=(0,))
print(f"Sleep staging  accuracy={res_sl['accuracy']['mean']:.3f}  "
      f"balanced_accuracy={res_sl['balanced_accuracy']['mean']:.3f}")
print("(accuracy looks higher because most epochs are stage N2 — see Ch09)")

# %% [markdown]
# ## 5. Seizure detection (simulated, imbalanced)
#
# Seizures are **rare** events with high-amplitude rhythmic activity. The key
# challenge is **class imbalance**: 99% of the recording is normal. We simulate
# "normal" vs "seizure" segments and preview the metric trap that Chapter 12
# dissects in full.
#
# *Real data:* CHB-MIT Scalp EEG, TUH Seizure Corpus (both large).
#
# **Why it works:** A seizure is caused by **hypersynchronous** neuronal firing —
# thousands of neurons that normally fire independently all lock together and
# discharge in high-amplitude rhythmic bursts. This abnormal synchrony produces
# unusually large, regular oscillations in the EEG that stand out clearly from the
# lower-amplitude, irregular background activity of a healthy brain, making
# amplitude and spectral features reliable discriminators.

# %%
sf = 256.0; T = int(2 * sf); t = np.arange(T) / sf
def normal_seg():  return rng.standard_normal(T)
def seizure_seg(): return 3 * np.sin(2 * np.pi * 4 * t) + rng.standard_normal(T)  # big slow rhythm
n_total, n_seiz = 300, 15  # 5% seizures (imbalanced!)
# Spread the rare seizures throughout the recording (every ~20th segment) so they
# are not all bunched at the end — otherwise a block split would isolate them.
seiz_positions = set(np.linspace(0, n_total - 1, n_seiz).astype(int))
segs, yse = [], []
for i in range(n_total):
    if i in seiz_positions:
        segs.append(seizure_seg()); yse.append(1)
    else:
        segs.append(normal_seg()); yse.append(0)
Xse = np.array(segs)[:, None, :]
yse = np.array(yse)
Fse = ft.bandpower(Xse, sf)
print(f"Seizure dataset: {np.bincount(yse)} (class 1 = seizure, only {100*yse.mean():.0f}%)")

from sklearn.linear_model import LogisticRegression
clf = leakage_safe_pipeline([("clf", LogisticRegression(max_iter=500))])
res_se = evaluate_with_variance(clf, Fse, yse,
                                cv=lambda: make_block_split(len(yse), groups=np.arange(len(yse)), n_splits=5),
                                scoring=("accuracy", "balanced_accuracy", "f1_macro"), seeds=(0,))
print(f"  accuracy={res_se['accuracy']['mean']:.3f}  "
      f"balanced_accuracy={res_se['balanced_accuracy']['mean']:.3f}  "
      f"f1_macro={res_se['f1_macro']['mean']:.3f}")
print("  -> accuracy can look great while the detector misses seizures (Ch09).")

# %% [markdown]
# ## 6. Neural decoding / brain-to-text (concept + small decode)
#
# **Neural decoding** maps brain activity to an external variable: a cursor, a
# robotic arm, or — at the cutting edge — **text/speech** from intracortical
# implants (e.g. recent "brain-to-text" speech neuroprostheses). The principle is
# the same multi-class decoding you have already done. Here we decode the
# **4-class** motor imagery (left/right hand, feet, tongue) as a stand-in for a
# multi-target decoder.
#
# **Why it works:** Speech and language involve rapid, fine-grained patterns of
# activity across many small cortical areas (e.g. the phoneme representations in
# motor cortex are only millimetres apart). Scalp EEG cannot resolve this spatial
# detail — skull and scalp smear the signals together. The breakthrough results in
# brain-to-text rely on **invasive (intracortical) electrode arrays** (e.g.
# Utah arrays) that record individual neurons or small neural populations at
# sub-millimetre resolution. Chapter 13 discusses the ethical and practical
# honesty issues around scalp-EEG "mind-reading" claims.

# %%
X4, y4, s4 = io.load_bnci_2a_epochs(n_subjects=2,
                                    classes=("left_hand", "right_hand", "feet", "tongue"))
pipe4 = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() +
                              [("clf", LogisticRegression(max_iter=500))])
res4 = evaluate_with_variance(pipe4, X4, y4, cv=lambda: make_subject_split(s4),
                              scoring="accuracy", seeds=(0,))
print(f"4-class decode LOSO accuracy: {res4['accuracy']['mean']:.3f} (chance 0.25)")

# %% [markdown]
# ## ✅ Concept check
#
# Test your understanding before moving on.
#
# **Q1.** In the P300 paradigm, why are targets deliberately made *rare* rather
# than appearing as often as non-targets?
#
# **Q2.** What is *frequency tagging*, and why does it let an SSVEP BCI work
# without any training data?
#
# **Q3.** Motor imagery decoding exploits a contralateral asymmetry — what does
# *contralateral* mean here, and which rhythms change?
#
# **Answers:**
# 1. The P300 response is driven by *surprise* — the brain only emits the large
#    ~300 ms positivity when a stimulus is rare and task-relevant (the **oddball**
#    effect). If targets appeared as often as non-targets the brain would adapt and
#    the P300 would disappear, removing the decodable contrast.
# 2. **Frequency tagging** is the tendency of the visual cortex to oscillate at the
#    same frequency as a rhythmic visual stimulus. Because each BCI target flickers
#    at a unique frequency, the dominant peak in the occipital EEG directly
#    identifies the attended target — no subject-specific calibration needed.
# 3. *Contralateral* means "on the opposite side": imagining a *right*-hand
#    movement activates the *left* motor cortex, and vice versa. The activated
#    hemisphere suppresses the **mu (8–12 Hz)** and **beta (18–26 Hz)** rhythms
#    (event-related desynchronisation, ERD), creating a spatial asymmetry that CSP
#    and similar methods exploit.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Using accuracy for rare events (seizure).** A model that predicts "no
#   seizure" forever scores 95% accuracy and is useless. Use balanced accuracy /
#   F1 / ROC-AUC (Chapter 12).
# - **Single-trial ERP claims.** ERPs need averaging; a "P300 detected in one
#   trial" claim demands very careful stats.
# - **SSVEP without enough data length.** Frequency resolution needs seconds of
#   signal; too short a window blurs neighbouring stimulus frequencies together.
# - **Mixing nights/subjects in sleep staging via random split.** Consecutive
#   epochs are highly correlated — split by night/subject (Chapter 12, pitfall #1/#2).
# - **Over-claiming "brain-to-text".** Real speech decoding uses invasive implants,
#   large models and huge data. Scalp-EEG "mind reading" demos are usually leakage.
#
# **Next:** Chapter 11 — *statistics intuition* (what mean ± std means, why a small
# gap between two bars is usually noise), the groundwork for Chapter 12 — the most
# important chapter — which shows every way evaluation lies as WRONG → RIGHT pairs.
