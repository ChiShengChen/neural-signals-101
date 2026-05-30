# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 13 — Neuroethics & Anti-Hype
#
# Science journalism loves a good "mind-reading" headline. Neural-decoding papers are
# disproportionately covered, and disproportionately misrepresented. This chapter
# argues that **over-claiming to the public is the ethical twin of leakage** — both
# manufacture a false impression of what the technology can do. One happens in the
# code; the other happens in the press release.
#
# We will look at *why* numbers inflate, *what* gets lost in translation, and *what
# responsibilities* come with holding a dataset of someone else's brain activity.
#
# ## Learning objectives
# 1. Distinguish offline cross-validated accuracy from real-time, deployable BCI
#    performance, and articulate why the gap is almost always large.
# 2. Map common "mind-reading" headline failures to specific Chapter 12 pitfalls
#    (leakage, tiny N, cherry-picking, no held-out subject).
# 3. Describe the core concepts in brain-data ethics: privacy, informed consent,
#    neuro-rights, and dual-use risk.
# 4. Summarise the genuine state of brain-to-text / speech neuroprostheses and
#    contrast it with scalp-EEG "telepathy" claims.
# 5. Apply a responsible-reporting checklist to any neural-decoding result.
#
# > **Prerequisites:** Chapters 10 and 12.
# > **Difficulty:** ★★☆☆☆
# > **Runtime:** ~1 min.

# %%
# Bootstrap: make `import neuro101` work whether or not you ran `pip install -e .`
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
    import neuro101  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from neuro101 import viz

# %% [markdown]
# ---
# # Part 1 — "Offline 95%" is NOT a working BCI
#
# ## What does "offline" mean?
#
# When a researcher reports "95% accuracy", they almost always mean **offline,
# cross-validated accuracy**: a recording was made earlier, epochs were extracted,
# features were computed, and a classifier was evaluated using held-out *portions of
# that same recording*. The human participant is long gone. There is no feedback
# loop, no real-time pressure, no fatigue, and no deployment infrastructure.
#
# A **real-time BCI** is something completely different:
#
# | Dimension | Offline evaluation | Online / deployed BCI |
# |---|---|---|
# | When does inference happen? | After the recording is done | While the user is trying to control the device |
# | Is the signal stationary? | Approximately yes (one session) | No — it shifts minute-to-minute |
# | Is there feedback? | No | Yes — the user adapts *to* the system |
# | Latency constraint? | None | Hundreds of milliseconds |
# | Calibration? | Usually baked into the training set | Must be re-done for every new session |
# | User fatigue | Irrelevant | Grows over time, shifts the signal |
#
# ## The four killers of offline accuracy
#
# **1. Non-stationarity.** EEG signals drift. Electrode impedance changes, the user
# shifts, attention wanders, and the neural patterns that looked clean at 10 AM are
# slightly different by 10:15. A model trained on the first half of a session that
# is then tested live in the second half may face a distribution it has never seen.
# We demonstrated this concretely in **Chapter 12, Pitfall 5** (cross-session /
# domain shift).
#
# **2. Calibration cost.** Many BCI systems require several minutes of labelled
# trials from a new user before they can be used at all. If calibration is hard or
# tiring, adoption collapses. "95% in the lab after 20 minutes of sitting still" is
# very different from "80% on trial one with a new user who has never tried this
# before".
#
# **3. User fatigue.** Motor-imagery BCIs require sustained mental effort. Performance
# degrades over a session as attention wanders, which is why BCI studies often limit
# sessions to under an hour. The cross-validated number on a tidy, freshly collected
# dataset does not contain this degradation.
#
# **4. Latency.** Cross-validation has no latency constraint. A deployed system must
# produce a decision in real time — typically within 100–500 ms of the command being
# issued. Complex pipelines that are fast "offline" may miss the timing window when
# running on embedded hardware.
#
# > **Key insight:** A model that scores 95% offline can be unusable live. The gap
# > between offline accuracy and usable real-time performance is one of the most
# > consistent and underreported findings in the BCI literature.

# %% [markdown]
# ---
# # Part 2 — Why "EEG mind-reading" headlines are almost always inflated
#
# Let us trace the typical lifecycle of a hyped result. Each failure mode below is
# tied to a specific Chapter 12 pitfall.
#
# ## Failure mode A: Subject leakage (→ Pitfall 2)
#
# The authors recruit 10 subjects, pool all their trials into one dataset, split it
# randomly into train and test, and report accuracy. Because some trials from each
# subject appear in both train and test, the classifier learns *who the person is*
# (their personal EEG fingerprint) rather than *what they are thinking*. This is
# **Chapter 12, Pitfall 2** — subject-dependent evaluation masquerading as
# generalisation.
#
# **The honest version** uses Leave-One-Subject-Out (LOSO): the test subject was
# never seen during training. LOSO accuracies are typically 10–20 percentage points
# lower, especially for scalp EEG where inter-subject variability is huge.
#
# ## Failure mode B: Tiny N (→ Pitfall 6)
#
# With five subjects and very high inter-subject variance, a LOSO cross-validation
# has five folds. The variance across folds is so large that the *best* fold (the
# most cooperative subject) may be 20 points above the mean. If the paper reports the
# best subject or the best seed, that is **Chapter 12, Pitfall 6** — cherry-picking
# the lucky fold.
#
# The responsible practice is to report **mean ± std over all folds/seeds** and
# to be explicit that N = 5 is barely sufficient to estimate variance, let alone to
# generalise.
#
# ## Failure mode C: Preprocessing / feature leakage (→ Pitfall 3)
#
# The feature selector, scaler, or CSP is fit on the entire dataset (all subjects,
# all trials) before the cross-validation loop. The "best" features are those that
# happen to correlate with the labels in the full dataset — including the test fold.
# The classifier then has an unfair preview of the test labels. This is **Chapter 12,
# Pitfall 3** and it can manufacture signal from pure random noise, as the demo
# below will show.
#
# ## Failure mode D: No held-out subject; demos vs claims
#
# A demo is not a product. Demonstrating that *one specific person*, who trained the
# model and rehearsed the task, can control a device is an existence proof, not a
# deployability claim. Real generalisation requires a participant who has never been
# in the lab, calibrating with a short session and then achieving reasonable accuracy.
# Many "impressive" BCI demonstrations are essentially one person running their own
# personalised, heavily calibrated system — the neural equivalent of "works on my
# machine".
#
# ## Why journalists amplify these failures
#
# Journalists are not adversaries — they are incentivised by clicks, and "brain-
# computer interface decodes thoughts at 95% accuracy" gets more clicks than
# "cross-validated band-power classifier achieves 0.73 ± 0.12 LOSO on 9 subjects
# in a constrained imagined speech task, with p = 0.04 before multiple-comparison
# correction". Scientists who issue breathless press releases share responsibility
# for the inflation.

# %% [markdown]
# ---
# # Part 3 — Data ethics: brain data is not like other data
#
# ## Why brain data deserves special treatment
#
# An EEG recording is a time series of electrical potentials. At first glance it
# looks like sensor data — not obviously more sensitive than a step counter. But
# researchers have already demonstrated that EEG can reveal:
#
# - **Cognitive states**: attention, workload, stress, drowsiness.
# - **Emotional responses**: liking or disliking a product, a person, a political
#   message.
# - **Medical conditions**: epilepsy, sleep disorders, early signs of cognitive
#   decline.
# - **Identity**: EEG can serve as a biometric identifier — your brain rhythm is as
#   unique as a fingerprint.
#
# Unlike a password, you cannot change your brain. A breach of neural data is
# permanent.
#
# ## Informed consent
#
# In academic research, an Institutional Review Board (IRB) or Ethics Committee
# reviews the study protocol. Participants must be told, in plain language:
#
# - What data will be recorded.
# - What it will be used for now *and* in the future.
# - Who will have access to it.
# - How it will be stored, anonymised, and eventually deleted.
# - That they can withdraw at any time without penalty.
#
# The challenge for machine learning researchers is that secondary uses — e.g.,
# training a foundation model on pooled BCI datasets — may not have been anticipated
# when the original consent was given. Using that data for the new purpose may
# violate the spirit of the original consent even if it is technically permitted by
# a broad data-sharing clause.
#
# ## Neuro-rights: an emerging framework
#
# Several ethicists and legal scholars (notably Rafael Yuste and Sara Goering) have
# proposed a framework of **neuro-rights** — fundamental rights specific to the
# neural domain. The core proposals, now incorporated into Chile's constitution (2021)
# and discussed in the UN context, are:
#
# | Neuro-right | What it protects |
# |---|---|
# | **Mental privacy** | The right to keep your thoughts private; no scanning or decoding without consent. |
# | **Mental integrity** | Protection against unauthorised alteration of neural activity. |
# | **Cognitive liberty** | The right to choose whether to enhance or augment your own cognition. |
# | **Psychological continuity** | Protection of a stable sense of self against manipulation. |
# | **Equal access to mental augmentation** | Ensuring that cognitive enhancement does not entrench inequality. |
#
# These are forward-looking: current EEG is far too noisy to read detailed thoughts.
# But the legal and ethical frameworks must be built *before* the technology
# matures — not after.
#
# ## Dual-use risk
#
# A BCI decoder built for a wheelchair user or a locked-in patient is a genuine
# medical good. The **same decoder architecture**, applied to covert recordings in
# an interrogation or employment-screening context, becomes a tool for surveillance
# or coercion. This is the dual-use problem, and it is not hypothetical:
#
# - Emotion-detection headsets are already marketed to employers to monitor worker
#   engagement (China has piloted this in factories and classrooms).
# - EEG-based lie detection products exist, despite lacking scientific validation,
#   and have been used in some legal proceedings.
# - Military researchers have funded BCI work with obvious surveillance applications
#   (target detection from passive EEG in drone operators, for example).
#
# The researcher who publishes an open-source BCI decoder cannot fully control how
# it is used. Thinking carefully about downstream risk — and saying so in the paper —
# is part of scientific responsibility.
#
# > **Balance note:** none of this means we should stop doing BCI research. It means
# > we should do it with eyes open, publish thoughtfully, advocate for protective
# > regulation, and build communities of practice that take these norms seriously.

# %% [markdown]
# ---
# # Part 4 — Brain-to-text / speech neuroprostheses: the honest story
#
# ## What the real research actually does
#
# Between 2021 and 2024 several landmark papers demonstrated real-time decoding of
# attempted speech from **intracortical implants** in participants with paralysis
# (e.g., Frank Willett et al. 2023 in *Nature*, reaching ~62 words/minute). These
# are genuinely impressive results. Here is what made them possible:
#
# | Factor | Real speech neuroprosthesis | Scalp-EEG "mind-reading" |
# |---|---|---|
# | Signal source | Intracortical electrode array (Utah array, ECoG grid) — microvolts resolved at neuron level | Scalp EEG — millivolt potentials smeared through skull and scalp, ~10,000× noisier |
# | Spatial resolution | Single-unit / small population level | Centimetres |
# | Participants | 1–5 highly motivated, extensively calibrated participants who have given consent to surgery | Many, but typically N < 20 naive participants |
# | Calibration | Hundreds of hours of recordings, per-participant models | Minutes to hours per session |
# | Model size | Large sequence models (RNN, Transformer) trained on participant-specific data | Small classifiers on simple features |
# | Task | Continuous attempted speech, full vocabulary | Typically a handful of words or commands |
# | Deployment | Closed-loop system with real-time feedback, tested over months | Offline CV on a single recording session |
#
# ## Why the gap matters
#
# When a journalist writes "EEG decodes speech at 90% accuracy", they are almost
# always describing a result where:
# - The "speech" is 5 imagined words.
# - The 90% is offline, subject-dependent accuracy.
# - The feature leakage (Chapter 12, Pitfall 3) has inflated the number.
# - There is no comparison to a chance baseline.
# - No one has tested a new participant.
#
# Contrast this with the real neuroprosthesis work, which is humble about its scope:
# it works for *this specific participant*, with *this specific implant*, after
# *this much calibration*, for *this vocabulary*, at *this level of accuracy on
# held-out utterances*. Every one of those qualifiers matters.
#
# ## The scientific progress is real — but modest on scalp EEG
#
# It is fair to say that scalp EEG can classify a handful of mental states with
# moderate accuracy under careful, controlled conditions. It is not fair to say it
# "reads thoughts". The distinction is not pedantry — it matters for policy,
# funding, patient expectations, and public trust in science.

# %% [markdown]
# ---
# # The demo — Before you run: think first
#
# We are about to generate **pure random noise** as "EEG features" and assign
# **completely random binary labels**. There is, by construction, zero relationship
# between the features and the labels.
#
# **Before running: predict what each method will report.**
#
# - The **WRONG** method selects the 30 "best" features using *all* the data (leaky),
#   then cross-validates. What accuracy do you expect?
# - The **RIGHT** method puts feature selection inside a pipeline so selection only
#   sees training data in each fold. What accuracy do you expect?
#
# Write your prediction here (or just think it):
# - Leaky method: ____%
# - Honest method: ____%
#
# Then run the cell and see what happens.

# %%
# --- Demo: Leakage manufactures a fake "mind-reading" result ---
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# Pure random noise — NO real brain signal, NO relationship to labels.
X = rng.standard_normal((200, 2000))   # 200 "trials", 2000 meaningless features
y = rng.integers(0, 2, 200)            # random binary labels

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- WRONG: fit feature selector on ALL data, then cross-validate ---
# SelectKBest sees the test-fold labels when it picks the "best" features.
selector_leaky = SelectKBest(f_classif, k=30)
X_selected_leaky = selector_leaky.fit(X, y).transform(X)  # peeks at ALL labels!

acc_leaky = np.mean([
    cross_val_score(
        LogisticRegression(max_iter=500),
        X_selected_leaky, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=s),
    ).mean()
    for s in (0, 1, 2)
])

# --- RIGHT: feature selection inside a Pipeline (fit per fold only) ---
pipe_honest = Pipeline([
    ("sel", SelectKBest(f_classif, k=30)),
    ("clf", LogisticRegression(max_iter=500)),
])

acc_honest = np.mean([
    cross_val_score(
        pipe_honest, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=s),
    ).mean()
    for s in (0, 1, 2)
])

print(f"Data: {X.shape[0]} samples, {X.shape[1]} RANDOM features, RANDOM labels.")
print(f"Chance level = 0.50")
print()
print(f"WRONG (leaky feature selection):  accuracy = {acc_leaky:.3f}  <-- on PURE NOISE!")
print(f"RIGHT (selection inside pipeline): accuracy = {acc_honest:.3f}  <-- back to chance, correctly")
print()
print("The 'WRONG' number is what a hyped 'we decoded thoughts at X%!' result often is:")
print("leakage, not telepathy.")

# %%
# --- Plot: the two-bar "headline vs reality" ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))

viz.plot_wrong_vs_right(
    wrong_score=acc_leaky,
    right_score=acc_honest,
    chance=0.5,
    metric="Accuracy",
    wrong_label="Leaky method\n('mind-reading!')",
    right_label="Honest method\n(pure chance)",
    title="Leakage manufactures 'mind-reading' from noise",
    ax=ax,
)
ax.text(
    0.5, -0.13,
    "This is what a hyped 'we decoded thoughts at 90%!' result\noften is — leakage, not telepathy.",
    ha="center", va="top", transform=ax.transAxes,
    fontsize=8.5, style="italic", color="#555555",
)
plt.show()

# %% [markdown]
# ---
# # Part 5 — Responsible reporting checklist
#
# Copy this into your next project. Every item has a reason.
#
# ```text
# STUDY DESIGN
# [ ] State the number of participants (N) prominently.
# [ ] State whether the evaluation is subject-DEPENDENT or subject-INDEPENDENT (LOSO).
#     If subject-dependent, label it as such — do not present it as a generalisation claim.
# [ ] Include at least one completely held-out test participant not used in any
#     tuning or model selection decision.
#
# METRICS
# [ ] Report chance-level baseline (e.g., 0.5 for binary, 1/k for k-class).
# [ ] Report mean ± std over folds/seeds — never a single number.
# [ ] Use balanced accuracy / F1 / ROC-AUC for imbalanced problems (not just accuracy).
# [ ] If comparing two models, use a paired test across folds (not a t-test on two scalars).
#
# EVALUATION INTEGRITY
# [ ] Every learned transform (scaler, feature selection, CSP, ICA, PCA) is inside
#     a Pipeline and fit on training folds only.
# [ ] Hyperparameters were tuned on a validation set or via nested CV — NOT on the test set.
# [ ] Distinguish offline cross-validated numbers from any online / real-time demo result.
# [ ] If you ran a real-time demo, report both the offline CV score AND the online score.
#
# ETHICS & CONSENT
# [ ] Participants gave written informed consent for this specific use of their data.
# [ ] Secondary uses (sharing, pooling, training larger models) were covered in consent.
# [ ] Data is stored securely and anonymised per IRB / ethics committee requirements.
# [ ] The paper discusses dual-use risk if the technique could plausibly be misused.
#
# COMMUNICATION
# [ ] Do not over-generalise from a task-specific result (e.g., 5-word imagined speech
#     ≠ "reading thoughts").
# [ ] Do not present a lab demo as a deployable product.
# [ ] Release code (and data if consent allows) so others can verify your numbers.
# [ ] Offer to review press releases before they go out — catch the headline before
#     it becomes the fact.
# ```

# %% [markdown]
# ---
# ## ✅ Concept check
#
# Work through these before moving on. The answers are below.
#
# **Q1.** A paper reports 94% accuracy decoding emotion from EEG using 8 subjects.
# The evaluation was a random 80/20 train-test split on the pooled dataset. Name two
# Chapter 12 pitfalls this almost certainly commits and describe how each inflates the
# reported number.
#
# **Q2.** A startup markets a "thought-to-text" EEG headset citing a university
# paper achieving "88% decoding of imagined words". List three questions you would ask
# to assess whether that number reflects real-world usability.
#
# **Q3.** A researcher wants to share a large, rich EEG dataset with the community.
# Original consent covered "use in our lab for this study". What ethical steps should
# they take before publishing the data openly?
#
# **Q4.** In the demo above, why does the leaky pipeline produce high accuracy on
# *pure random data*? Why does the honest pipeline correctly return ~0.50?
#
# ---
# **Answers:**
#
# **A1.** (1) **Subject leakage (Pitfall 2):** pooling subjects and splitting randomly
# means the same subject's trials appear in train and test, so the model can learn
# the person's EEG fingerprint rather than the emotion — inflating accuracy on
# familiar subjects. (2) **Lucky seed / no variance (Pitfall 6):** with N = 8 there
# are very few effective "folds"; reporting a single run without variance hides the
# fact that the result may be lucky. A bonus third pitfall: **feature leakage
# (Pitfall 3)** if any preprocessing was fit on the full dataset.
#
# **A2.** Any three of: (1) Is the 88% offline or online (real-time)? (2) Is it
# subject-independent (LOSO) or subject-dependent? (3) How many words / how large is
# the vocabulary? (4) What is the chance level? (5) Has it been tested on a new user
# who never provided training data? (6) Has it been tested outside the lab?
#
# **A3.** Re-contact participants for supplementary consent covering public data
# release; if that is impossible, carefully anonymise (remove identifying metadata,
# consider whether EEG fingerprinting could re-identify); consult the ethics
# committee about whether the original broad-use clause is sufficient; consider a
# data access agreement requiring ethical oversight for secondary users.
#
# **A4.** The leaky pipeline runs `SelectKBest.fit(X, y)` on the **entire** dataset
# before cross-validation. With 2000 random features, some will correlate with the
# random labels just by chance. By picking exactly those 30 "winners", we hand the
# classifier features that are spuriously aligned with the test-fold labels — the
# model appears to generalise what is actually memorisation of statistical noise.
# The honest pipeline fits `SelectKBest` only on the training portion of each fold,
# so no information about test labels leaks in; the selected features are truly
# uninformative about the test set, and accuracy correctly returns to ~0.50.

# %% [markdown]
# ---
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Equating offline accuracy with deployability.** "It works in cross-validation"
#   is a necessary but not sufficient condition for a usable system. Non-stationarity,
#   fatigue, latency, and calibration cost all drive real-world performance below the
#   offline number. A deployed BCI must be re-validated in real-time conditions.
#
# - **Ignoring consent for secondary data use.** Downloading a publicly available EEG
#   dataset and using it to train a model for a purpose the participants never
#   consented to is ethically problematic even when it is technically legal. Always
#   check consent scope.
#
# - **Over-generalising from a single subject (or a highly cooperative subject
#   pool).** A system demonstrated on the researcher themselves, or on volunteers
#   recruited from a neuroscience lab, will almost certainly perform worse on a naive
#   general population. The gap between "works for us" and "works for everyone" is
#   where most BCI products fail.
#
# - **Treating a demo as a product.** A compelling real-time demo in a conference
#   booth — with a calibrated user, a clean recording environment, and a carefully
#   chosen task — is very far from a product. The honest path from demo to product
#   involves rigorous held-out validation on new users in messy environments.
#
# - **Dismissing neuro-rights as science fiction.** The legal frameworks are ahead
#   of the current technology *on purpose*. Building ethical norms now, before the
#   technology matures, is far easier than retrofitting them afterward.
#
# - **Assuming dual-use is someone else's problem.** If you publish a method, you
#   are part of the chain of responsibility for how it is used. This does not mean
#   you must refuse to publish; it means you should think carefully, document risks,
#   and engage with policy conversations.
#
# ---
#
# **Next:** Chapter 14 — the capstone. You will build your own full pipeline on a
# held-out test set you have never seen, following every checklist from Chapters 12
# and 13. The grader will check for leakage, proper variance reporting, a chance
# baseline, and an honest online-vs-offline discussion.
