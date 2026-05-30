# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 15 — Real-Time Streaming & Low-Cost Hardware  *(Appendix)*
#
# > **Prerequisites:** Chapter 08.
# > **Difficulty:** ★★☆☆☆
# > **Runtime:** ~1–2 min.
#
# Everything in the tutorial so far was *offline*: you had the whole recording on
# disk, could shuffle it, re-run it, peek at future samples.
# In a real BCI the signal arrives **sample by sample, right now**. That one word —
# *now* — changes almost every decision you make.
#
# This chapter shows:
# 1. A **simulated real-time** demo that makes causal inference visible.
# 2. What *actually* changes when you go from offline to online.
# 3. Where to find **low-cost hardware** so you can record your own brain signals.
#
# ## Learning objectives
# 1. Understand what "causal" means in a streaming context.
# 2. Visualise per-trial predictions + smoothed accuracy over a pseudo-stream.
# 3. Know the main practical differences: latency, drift, calibration.
# 4. Know the open-source hardware options and streaming protocols.

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from neuro101 import io, datasets as ds, features as ft
from neuro101.eval import leakage_safe_pipeline

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## Part 1 — Simulated real-time demo
#
# ### The setup: offline data, online mindset
#
# We cannot plug a real headset into a Jupyter notebook, but we can **pretend**.
# The idea:
# - **Train** a classifier on "historical" data (earlier subjects / early trials).
# - **Stream** later trials one at a time, making a prediction for each as it arrives.
# - The classifier is **never updated** during streaming (frozen, causal).
# - We plot the sequence of per-trial predictions and a rolling accuracy window.
#
# This mimics a closed-loop system: at time *t* you classify the current trial
# using only the model trained before the session started.

# %%
# Load data: use first subject(s) as "prior session" training data,
# last subject as the simulated live stream.
n_subj = 2 if SMOKE else 3
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
all_subjects = np.unique(subj)

train_subjs  = all_subjects[:-1]          # earlier subjects → training
stream_subj  = all_subjects[-1]           # last subject → simulated live stream

train_mask   = np.isin(subj, train_subjs)
stream_mask  = subj == stream_subj

X_train, y_train = X[train_mask], y[train_mask]
X_stream, y_stream = X[stream_mask], y[stream_mask]

print(f"Training on subject(s) {list(train_subjs)}: {X_train.shape[0]} trials")
print(f"Streaming subject {stream_subj}: {X_stream.shape[0]} trials")
print(f"Signal shape per trial: {X_train.shape[1]} channels × {X_train.shape[2]} samples "
      f"({X_train.shape[2]/sf:.2f} s @ {sf} Hz)")

# %% [markdown]
# ### Train the classifier — once, before streaming begins
#
# **Causal rule:** the classifier sees *only* training-session data.
# It will never be shown any sample from the streaming session during inference.

# %%
rt_pipeline = leakage_safe_pipeline([
    ("csp", ft.make_csp(n_components=4)),
    ("clf", LinearDiscriminantAnalysis()),
])
rt_pipeline.fit(X_train, y_train)
print("Classifier trained. Freezing weights — streaming begins now.")

# %% [markdown]
# ### Predict before you run
#
# > **Before running the next cell:** Will per-trial accuracy be noisy or smooth?
# > Will smoothing predictions over a rolling window make them steadier?
# > Write your answer here, then check.
# >
# > *Your prediction:* _______

# %%
# Simulate the stream: one trial at a time, predict, log.
n_stream = len(y_stream)
preds   = np.empty(n_stream, dtype=int)
conf    = np.empty(n_stream)          # max class probability = confidence
correct = np.empty(n_stream)

for t in range(n_stream):
    # !! CAUSAL: only X_stream[t] is available at step t — no future samples.
    trial = X_stream[t : t + 1]          # shape (1, channels, samples)
    preds[t]   = rt_pipeline.predict(trial)[0]
    probas_t   = rt_pipeline.predict_proba(trial)[0]
    conf[t]    = probas_t.max()
    correct[t] = int(preds[t] == y_stream[t])

instant_acc = correct.mean()
print(f"Instant (per-trial) accuracy on stream: {instant_acc:.3f}  "
      f"(chance = {1/len(np.unique(y)):.2f})")

# Rolling accuracy with a window of W trials.
W = min(10, n_stream // 4)             # adaptive window for smoke/full mode
rolling_acc = np.full(n_stream, np.nan)
for t in range(W - 1, n_stream):
    rolling_acc[t] = correct[t - W + 1 : t + 1].mean()

# Rolling average of confidence.
rolling_conf = np.full(n_stream, np.nan)
for t in range(W - 1, n_stream):
    rolling_conf[t] = conf[t - W + 1 : t + 1].mean()

# %% [markdown]
# ### Plot: the simulated live feed

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

trial_idx = np.arange(n_stream)

# --- Row 1: predicted class vs true class ---
ax0 = axes[0]
ax0.step(trial_idx, y_stream, where="mid", color="gray",  lw=1.5, label="True class")
ax0.step(trial_idx, preds,    where="mid", color="#4878CF", lw=1,  label="Predicted", alpha=0.8)
ax0.set_yticks(np.unique(y_stream))
ax0.set_ylabel("Class label")
ax0.legend(loc="upper right", fontsize=8)
ax0.set_title("Simulated real-time BCI stream (causal, frozen classifier)")

# --- Row 2: per-trial correct / incorrect + rolling accuracy ---
ax1 = axes[1]
hit_idx  = trial_idx[correct == 1]
miss_idx = trial_idx[correct == 0]
ax1.scatter(hit_idx,  np.ones(len(hit_idx)),  marker="|", s=80,
            color="#6ACC65", label="Correct", zorder=3)
ax1.scatter(miss_idx, np.zeros(len(miss_idx)), marker="|", s=80,
            color="#D65F5F", label="Wrong",   zorder=3)
ax1.plot(trial_idx, rolling_acc, color="black", lw=2,
         label=f"Rolling acc (W={W})")
ax1.axhline(instant_acc, ls="--", color="steelblue", lw=1,
            label=f"Mean acc={instant_acc:.2f}")
ax1.axhline(1 / len(np.unique(y_stream)), ls=":", color="gray", lw=1, label="Chance")
ax1.set_ylim(-0.2, 1.3)
ax1.set_ylabel("Accuracy")
ax1.legend(loc="upper right", fontsize=8, ncol=2)

# --- Row 3: confidence over time ---
ax2 = axes[2]
ax2.fill_between(trial_idx, 0.5, conf, where=(conf >= 0.5),
                 alpha=0.4, color="#6ACC65", label="Confident")
ax2.fill_between(trial_idx, conf, 0.5, where=(conf < 0.5),
                 alpha=0.4, color="#D65F5F", label="Uncertain")
ax2.plot(trial_idx, rolling_conf, color="black", lw=1.5,
         label=f"Rolling conf (W={W})")
ax2.axhline(0.5, ls="--", color="gray", lw=1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Trial index (→ time)")
ax2.set_ylabel("Max class probability")
ax2.legend(loc="upper right", fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# **What you just saw:**
#
# - Per-trial accuracy is noisy — individual predictions can easily flip.
# - A rolling window smooths the noise and reveals the *trend*: is the classifier
#   staying calibrated or drifting?
# - High confidence does not always mean high accuracy (the CSP+LDA confidence is
#   not perfectly calibrated). Use it as a quality signal, not ground truth.
# - The classifier was **causal throughout**: it only used information from the
#   training session, never future streaming trials.
#
# Compare this to what a **cheating real-time system** looks like: if you trained
# on all the streaming data first and then "replayed" it, you'd get a much higher
# accuracy — but it would be completely irreproducible in deployment.

# %% [markdown]
# ## Part 2 — What changes in real-time vs offline?
#
# | Concern | Offline | Real-time |
# |---|---|---|
# | **Latency** | Don't care; all data on disk | Must classify within tens of ms of trial end |
# | **Non-stationarity / drift** | Can average across sessions | The statistics shift hour-by-hour, even minute-by-minute |
# | **Calibration** | Full historical dataset | Often only a short per-session calibration run (≤ 5 min) |
# | **Prediction smoothing** | Not needed | Rolling vote or exponential moving average reduce flicker |
# | **Feedback loop** | None | User adapts to system; system may adapt to user (co-adaptation) |
#
# ### Drift and non-stationarity (Chapter 12, Pitfall 5)
#
# EEG statistics drift within a session: electrode impedance changes, the user gets
# tired or more focused, sweat builds up under electrodes, the amplifier baseline
# wanders. A classifier trained at 9 AM may be noticeably worse by 10 AM — not
# because the model is bad, but because the *distribution* it was trained on no
# longer matches the incoming signal.
#
# Offline numbers therefore **systematically overestimate** real-time performance.
# The gap is often 5–15 percentage points on a typical 4-class motor imagery task.
#
# **Mitigations:**
# - **Per-session re-calibration**: fit (or fine-tune) on a short block at the
#   start of each session.
# - **Adaptive classifiers**: update parameters online as new labelled data
#   accumulates (with care about catastrophic forgetting).
# - **Riemannian alignment** (Chapter 08 Riemannian pipeline): the covariance
#   geometry is more stable across sessions than raw band-power.

# %% [markdown]
# ## Part 3 — Low-cost hardware: record your own signals
#
# You do not need a research-grade 256-channel amplifier to experiment. Several
# affordable options exist:
#
# ### Consumer EEG headsets
#
# | Device | Channels | Price (approx.) | Notes |
# |---|---|---|---|
# | **OpenBCI Ganglion** | 4 | ~$200 | Open-source firmware; good community; dry or wet electrodes |
# | **OpenBCI Cyton** | 8 (+ Daisy for 16) | ~$500–800 | Research-quality ADC; most popular open-source choice |
# | **Muse 2 / Muse S** | 4 (+ PPG) | ~$200–300 | Very easy to wear; good for relaxation/focus; Bluetooth |
# | **Emotiv EPOC X** | 14 | ~$850 | Saline electrodes; semi-professional |
#
# **Caveats:** Consumer headsets have fewer channels, higher noise floors, and often
# worse electrode contact than research systems. For clean motor imagery you typically
# want at least 8 channels over motor cortex (C3, Cz, C4 and neighbours). Muse covers
# mainly frontal/temporal areas and is better suited for attention/meditation experiments.
#
# ### Streaming protocol: Lab Streaming Layer (LSL)
#
# **LSL** is the de-facto standard for real-time biosignal streaming in research.
# It is transport-agnostic (USB, Bluetooth, Wi-Fi) and has drivers for nearly every
# headset. Key libraries:
#
# - **`pylsl`** — Python LSL bindings (receive / send streams).
# - **`MNE-LSL`** (`pip install mne-lsl`) — wraps LSL into MNE `Raw` objects so your
#   MNE/scikit-learn pipeline can consume live data with minimal changes.
# - **`BrainFlow`** (`pip install brainflow`) — unified Python (and C++) API for
#   OpenBCI, Muse, Neurosity, synthetic boards, and more. One line of code changes
#   the hardware target.
#
# ### A first experiment: see your own eye-blink
#
# ```python
# # Pseudocode — requires BrainFlow + a Cyton or Ganglion connected via USB dongle
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# params = BrainFlowInputParams()
# params.serial_port = "/dev/ttyUSB0"       # adjust for your OS
# board = BoardShim(BoardIds.CYTON_BOARD, params)
# board.prepare_session()
# board.start_stream()
# # ... collect data, plot, see the big spike when you blink ...
# board.stop_stream()
# board.release_session()
# ```
#
# Eye-blinks produce large (~100–500 µV) slow waves most visible on frontal
# channels (Fp1, Fp2). Alpha waves (8–12 Hz) appear on occipital channels when
# you close your eyes. Both are immediately visible even in noisy consumer recordings
# — a satisfying first proof that the headset is working.

# %% [markdown]
# ## ✅ Concept check
#
# **Q1.** A researcher trains a CSP+LDA classifier offline and reports 78% accuracy.
# When they deploy it in a real-time session two weeks later, accuracy drops to 61%.
# Name two factors that could explain the gap.
#
# **A1.** (a) *Non-stationarity/drift* — EEG statistics change between sessions (impedance,
# fatigue, electrode position shifts) so the classifier trained on the old distribution
# may not match the new one. (b) *Overfitting to a fixed test set* — if the 78% was
# measured on a set that was used for model selection (the selection-bias problem from
# Chapter 14), the reported number was already inflated. The true generalisation
# performance was always closer to 61%.
#
# ---
# **Q2.** You build a real-time classifier that updates its weights each time a new
# labelled trial arrives. What could go wrong if the labels come from the user's
# button presses (self-reported)?
#
# **A2.** Button presses may be *delayed or mislabelled* (the user presses after the
# mental imagery, introducing a timing mismatch). If the label is always one trial
# behind, the model trains with shifted labels — effectively adding noise to the
# supervision signal. Worse, if the system also uses the classifier output to trigger
# feedback, errors compound: a wrong prediction gets trained on as a correct label.
# Robust co-adaptive systems use only high-confidence predictions or separate label
# collection phases.
#
# ---
# **Q3.** You want to try a consumer EEG headset for a motor-imagery experiment.
# You have a Muse 2 and an OpenBCI Cyton. Which would you choose, and why?
#
# **A3.** The **OpenBCI Cyton** is the better choice for motor imagery. Cyton has 8
# channels that can be placed over motor cortex (C3, Cz, C4), uses active or
# passive wet electrodes for better signal quality, and its open-source firmware
# integrates cleanly with BrainFlow and pylsl. The Muse 2 places its 4 electrodes
# on the forehead and behind the ears — covering frontal and temporal areas rather
# than the sensorimotor strip — so it is poorly suited for left/right hand imagery.

# %% [markdown]
# ## ⚠️ Common mistakes
#
# - **Testing real-time accuracy with future data leaking in.** The most common form:
#   "I'll fit the scaler on the whole streaming session, then replay it." That scaler
#   has seen future data. The fix: freeze all preprocessing at training time and only
#   apply `transform` (never `fit`) during streaming.
# - **Ignoring drift.** Offline numbers measured on the same session as training will
#   not hold in a new session recorded days or weeks later. Always measure
#   cross-session or cross-day accuracy when claiming real-time readiness.
# - **Trusting offline numbers for online deployment.** An offline accuracy of X does
#   not mean a real-time system will achieve X. Latency, prediction smoothing, the
#   feedback loop, and user adaptation all introduce new variables that are invisible
#   in offline evaluation.
# - **Reporting per-trial accuracy without a rolling window.** Single-trial EEG
#   classification is noisy. A rolling or exponential-moving-average display gives
#   the user feedback that is stable enough to act on.
# - **Skipping per-session calibration.** Even 2–5 minutes of labelled data from the
#   current session can dramatically close the drift gap — it is almost always worth
#   the time.
#
# ---
#
# 🏁 **End of the appendix.** You now have the conceptual toolkit to take what you
# learned offline all the way to a working real-time prototype. The jump from
# notebook to headset is smaller than it looks — the same sklearn pipelines, the same
# causal rules, just with data arriving in a stream instead of from a file. Good luck!
