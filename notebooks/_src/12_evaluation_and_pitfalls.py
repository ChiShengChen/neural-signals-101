# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/12_evaluation_and_pitfalls.ipynb)
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
# # Chapter 12 — Evaluation & Pitfalls  ⭐ (the most important chapter)
#
# Almost every "amazing" neural-decoding result that fails to reproduce died of one
# of the mistakes below. We teach each one as a **WRONG → RIGHT pair**:
#
# 1. Run the **WRONG** way → get a deceptively **high** score.
# 2. Explain in plain language *why it's inflated*.
# 3. Run the **RIGHT** way → the score **drops to a realistic number**.
# 4. A one-sentence **takeaway**.
#
# **The drop is the lesson.** We print and plot both numbers so the contrast is
# unmissable. Every ⚠️ cell is wrong on purpose — never copy it into real work.
#
# ## The six pitfalls
# 1. Random shuffle split on continuous/epoched data (autocorrelation leakage).
# 2. Subject-dependent vs subject-independent evaluation.
# 3. Preprocessing / feature leakage (fitting on all data before the split).
# 4. Class imbalance & the wrong metric.
# 5. Cross-session / domain shift.
# 6. Lucky seed / no variance reporting.
#
# > **Prerequisites:** Chapters 02, 08 and 11.
# > **Difficulty:** ★★★★☆
#
# **Runtime:** ~2–4 min on CPU (smoke mode shrinks data).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score)

from neuro101 import io, datasets as ds, features as ft, viz
from neuro101.eval import (
    make_subject_split, make_block_split, leakage_safe_pipeline, evaluate_with_variance,
)

SMOKE = ds.is_smoke()
sf = ds.DATASETS["bnci_2a"].sfreq_hz
scoreboard = {}  # collects (wrong, right) per pitfall for the final summary plot

# %% [markdown]
# ---
# # Pitfall 1 — Random shuffle split on continuous/epoched data
#
# **The setup.** We slice each motor-imagery trial into many *overlapping* windows.
# Neighbouring windows are almost identical (they share most of their samples). If
# we then split windows **randomly**, a window in the test set has a near-twin in
# the training set — the model can "recognise" it instead of generalising.

# %%
X1, y1, s1 = io.load_bnci_2a_epochs(subjects=[1])
win, step = 200, 25
W, L, G = [], [], []
for ti in range(X1.shape[0]):
    for start in range(0, X1.shape[-1] - win + 1, step):
        W.append(X1[ti, :, start:start + win]); L.append(y1[ti]); G.append(ti)
W, L, G = np.array(W), np.array(L), np.array(G)
F1 = ft.bandpower(W, sf)
print(f"{len(L)} overlapping windows from {X1.shape[0]} trials")

# %% [markdown]
# ### ⚠️ WRONG: random shuffle split (windows from one trial land in both sets)

# %%
wrong1 = np.mean([
    cross_val_score(LDA(), F1, L, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"⚠️ WRONG (random shuffle) accuracy = {wrong1:.3f}")

# %% [markdown]
# **Why it's inflated:** adjacent windows are correlated, so random splitting leaks
# near-duplicate samples across the train/test boundary. The model is partly
# *memorising*, not generalising.

# %% [markdown]
# ### ✅ RIGHT: trial-aware block split (whole trials stay together)

# %%
right1 = np.mean([
    LDA().fit(F1[tr], L[tr]).score(F1[te], L[te])
    for tr, te in make_block_split(len(L), groups=G, n_splits=5)
])
print(f"✅ RIGHT (trial-aware split) accuracy = {right1:.3f}")
scoreboard["1. Autocorrelation\nleakage"] = (wrong1, right1)

# %% [markdown]
# > **Takeaway 1:** Never random-shuffle correlated time series — keep whole
# > trials/blocks on one side of the split (`make_block_split`).

# %% [markdown]
# ---
# # Pitfall 2 — Subject-dependent vs subject-independent
#
# **The setup.** We pool trials from several subjects. The honest question is "does
# it work on a *new person*?" — but it's tempting to split the pooled trials
# randomly, which lets the same subject appear in both train and test.

# %%
nsub = 2 if SMOKE else 4
X, y, subj = io.load_bnci_2a_epochs(n_subjects=nsub)
pipe2 = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])

# %% [markdown]
# ### ⚠️ WRONG: pool subjects, then split randomly (subject identity leaks)

# %%
wrong2 = evaluate_with_variance(
    pipe2, X, y,
    cv=lambda: StratifiedKFold(5, shuffle=True, random_state=0).split(X, y),
    scoring="accuracy", seeds=(0, 1, 2),
)["accuracy"]["mean"]
print(f"⚠️ WRONG (pooled + random) accuracy = {wrong2:.3f}")

# %% [markdown]
# **Why it's inflated:** the model can learn each *subject's* idiosyncrasies (skull,
# electrode placement, personal rhythms) and reuse them on that subject's other
# trials in the test set — which won't happen for a genuinely new user.

# %% [markdown]
# ### ✅ RIGHT: Leave-One-Subject-Out (test only on unseen people)

# %%
right2 = evaluate_with_variance(
    pipe2, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0, 1, 2),
)["accuracy"]["mean"]
print(f"✅ RIGHT (LOSO) accuracy = {right2:.3f}")
scoreboard["2. Subject\nleakage"] = (wrong2, right2)

# %% [markdown]
# > **Takeaway 2:** The headline metric must be **subject-independent** (LOSO).
# > Subject-dependent numbers are optimistic — label them as such.

# %% [markdown]
# ---
# # Pitfall 3 — Preprocessing / feature leakage
#
# **The setup (deliberately stark).** We use **pure random noise** as features with
# **random labels** — there is *no* real relationship, so the honest accuracy must
# be chance (0.5). Watch a leaky feature-selection step manufacture "signal" out of
# nothing. (The same thing happens, less obviously, when you fit a scaler / CSP /
# ICA on the whole dataset before splitting.)

# %%
rng = np.random.default_rng(0)
Xnoise = rng.standard_normal((200, 2000))   # 2000 meaningless features
ynoise = rng.integers(0, 2, 200)            # labels unrelated to the features

# %% [markdown]
# ### ⚠️ WRONG: select the "best" features using ALL the data, then cross-validate

# %%
selected = SelectKBest(f_classif, k=30).fit(Xnoise, ynoise).transform(Xnoise)  # peeks at all labels
wrong3 = np.mean([
    cross_val_score(LDA(), selected, ynoise, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"⚠️ WRONG (select on all data) accuracy = {wrong3:.3f}  <-- on PURE NOISE!")

# %% [markdown]
# **Why it's inflated:** choosing the 30 features most correlated with the labels
# *using the whole dataset* lets information about the test labels leak into the
# feature set. With 2000 random features, some will correlate with the labels **by
# chance** — and we hand-picked exactly those. The score is 100% mirage.

# %% [markdown]
# ### ✅ RIGHT: put selection inside the pipeline (fit on each train fold only)

# %%
pipe3 = Pipeline([("sel", SelectKBest(f_classif, k=30)), ("lda", LDA())])
right3 = np.mean([
    cross_val_score(pipe3, Xnoise, ynoise, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"✅ RIGHT (selection inside pipeline) accuracy = {right3:.3f}  <-- back to chance, correctly")
scoreboard["3. Feature\nleakage"] = (wrong3, right3)

# %% [markdown]
# > **Takeaway 3:** Every step that *learns* from data (scaler, feature selection,
# > CSP, ICA, PCA) must be fit on the **training fold only** — wrap it in a Pipeline.

# %% [markdown]
# ---
# # Pitfall 4 — Class imbalance & the wrong metric
#
# **The setup.** Real Sleep-EDF data. We try to detect a **rare** sleep stage (the
# least frequent one). Most epochs are *not* that stage, so a model can score high
# accuracy by always saying "no".

# %%
Xsl, ysl, ssl = io.load_sleep_edf_epochs(n_subjects=1)
counts = np.bincount(ysl)
minority = int(np.argmin([c if c > 0 else 10**9 for c in counts]))
y_bin = (ysl == minority).astype(int)
print(f"Detecting the rare stage (id {minority}); it is only {100*y_bin.mean():.1f}% of epochs")

# A "model" that always predicts the majority class (detects nothing).
lazy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_bin), 1)), y_bin)
pred = lazy.predict(np.zeros((len(y_bin), 1)))

# %% [markdown]
# ### ⚠️ WRONG: report accuracy

# %%
acc4 = accuracy_score(y_bin, pred)
print(f"⚠️ WRONG metric — accuracy = {acc4:.3f}   (looks excellent!)")

# %% [markdown]
# **Why it's misleading:** accuracy rewards predicting the majority class. A model
# that *never* detects the rare stage still scores very high (≈ the majority
# fraction) because the event is rare. Accuracy hides total failure on the class
# you actually care about — balanced accuracy collapses to 0.5 (chance).

# %% [markdown]
# ### ✅ RIGHT: balanced accuracy, F1, confusion matrix (and ROC-AUC)

# %%
bal4 = balanced_accuracy_score(y_bin, pred)
f1_4 = f1_score(y_bin, pred, zero_division=0)
cm = confusion_matrix(y_bin, pred)
print(f"✅ RIGHT metrics — balanced accuracy = {bal4:.3f}, F1 = {f1_4:.3f}")
print("   confusion matrix (rows=true, cols=pred):"); print(cm)

fig, ax = plt.subplots(figsize=(4.2, 3.8))
viz.plot_confusion(cm, ["not-rare", "rare"], ax=ax,
                   title="It detects NONE of the rare class")
plt.show()
scoreboard["4. Wrong metric\n(acc vs bal-acc)"] = (acc4, bal4)

# %% [markdown]
# > **Takeaway 4:** On imbalanced problems, accuracy lies. Report balanced
# > accuracy / F1 / ROC-AUC and look at the confusion matrix.

# %% [markdown]
# ---
# # Pitfall 5 — Cross-session / domain shift
#
# **The setup.** Brains (and electrodes) change between recording days: different
# impedance, slightly different cap placement, different mood. A model trained on
# **session 1** and deployed on **session 2** faces a *distribution shift*.
#
# On BCI IV 2a the real session-to-session drop is modest, so to make the mechanism
# unmistakable we apply a **clearly-simulated "new recording day"** transform to the
# real data: per-channel gain changes (impedance) plus DC offsets (drift). This is
# what a different day/device genuinely does to the signal.

# %%
nsub5 = 2 if SMOKE else 3
Xs, ys, subs, sess = io.load_bnci_2a_epochs(n_subjects=nsub5, return_session=True)

def simulate_new_day(X_in, seed):
    """Realistic session shift: random per-channel gain + DC offset."""
    r = np.random.default_rng(seed)
    gain = r.uniform(0.5, 2.0, size=(1, X_in.shape[1], 1))   # impedance differences
    offset = r.uniform(-1, 1, size=(1, X_in.shape[1], 1)) * X_in.std()
    return X_in * gain + offset

pipe5 = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])

# %% [markdown]
# ### ⚠️ WRONG: evaluate on same-distribution (session-1-style) data

# %%
within = []
for su in np.unique(subs):
    m = (subs == su) & (sess == 0)
    Xa, ya = Xs[m], ys[m]
    within.append(np.mean([pipe5.fit(Xa[tr], ya[tr]).score(Xa[te], ya[te])
                           for tr, te in make_block_split(len(ya), n_splits=4)]))
wrong5 = float(np.mean(within))
print(f"⚠️ WRONG (test on same recording day) accuracy = {wrong5:.3f}")

# %% [markdown]
# **Why it's optimistic:** your validation data came from the *same* session as
# training, so it shares all the day-specific quirks. Deployment never does.

# %% [markdown]
# ### ✅ RIGHT: train on day 1, test on a (simulated) different day

# %%
cross = []
for su in np.unique(subs):
    m = (subs == su) & (sess == 0)
    Xa, ya = Xs[m], ys[m]
    Xnew = simulate_new_day(Xa, seed=int(su))   # the "next day" recording
    cross.append(pipe5.fit(Xa, ya).score(Xnew, ya))
right5 = float(np.mean(cross))
print(f"✅ RIGHT (test on a different day) accuracy = {right5:.3f}")
scoreboard["5. Domain\nshift"] = (wrong5, right5)

# %% [markdown]
# > **Takeaway 5:** Test across the shift you'll face in deployment (sessions,
# > days, devices, subjects). Same-session scores overstate real-world performance.
# > Fixes: per-session normalisation, domain adaptation, or a short calibration.

# %% [markdown]
# ---
# # Pitfall 6 — Lucky seed / no variance
#
# **The setup.** Subject-independent decoding varies a lot from one held-out subject
# to the next. If you run once and quote the *best* fold, you are cherry-picking.

# %%
per_fold = evaluate_with_variance(
    pipe2, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0,),
)["accuracy"]["per_fold"].ravel()
print("per-subject accuracies:", np.round(per_fold, 3))

# %% [markdown]
# ### ⚠️ WRONG: report the single best fold

# %%
wrong6 = float(per_fold.max())
print(f"⚠️ WRONG (best single fold) accuracy = {wrong6:.3f}")

# %% [markdown]
# **Why it's misleading:** with few folds and high variance, the maximum is a lucky
# draw, not the expected performance. A different random seed or test subject would
# give a different "best".

# %% [markdown]
# ### ✅ RIGHT: report mean ± std over folds (and test significance properly)

# %%
right6 = float(per_fold.mean())
print(f"✅ RIGHT (mean ± std) accuracy = {right6:.3f} ± {per_fold.std():.3f}")
scoreboard["6. Lucky fold\n(best vs mean)"] = (wrong6, right6)

# A paired test across folds, when comparing two models, respects the pairing:
from scipy.stats import wilcoxon, ttest_rel
pipe_riem = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() +
                                  [("lda", LDA())])
fold_csp = per_fold
fold_riem = evaluate_with_variance(
    pipe_riem, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0,),
)["accuracy"]["per_fold"].ravel()
if len(fold_csp) == len(fold_riem) and len(fold_csp) >= 3:
    t, p = ttest_rel(fold_riem, fold_csp)
    print(f"Paired t-test CSP vs Riemann across {len(fold_csp)} subjects: "
          f"mean diff {np.mean(fold_riem - fold_csp):+.3f}, p = {p:.3f}")
    print("(A difference smaller than the std, with p>0.05, is NOT a real improvement.)")

# %% [markdown]
# > **Takeaway 6:** One number is not a result. Report mean ± std over folds/seeds,
# > and use a **paired** test across folds before claiming one model beats another.

# %% [markdown]
# ---
# # The whole story in one figure
#
# Every pitfall, WRONG (red) vs RIGHT (green). In each pair the red bar is the
# number you'd be tempted to publish; the green bar is the truth.
#
# > **Before running:** guess how large the WRONG-vs-RIGHT accuracy gap will be for
# > Pitfall 3 (feature leakage on pure noise) — do you expect the inflated score to
# > be near 0.6, 0.7, 0.8, or higher, given that 2000 random features were available?

# %%
labels = list(scoreboard)
wrongs = [scoreboard[k][0] for k in labels]
rights = [scoreboard[k][1] for k in labels]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - 0.2, wrongs, 0.4, label="WRONG (inflated)", color=viz.WRONG_COLOR)
ax.bar(x + 0.2, rights, 0.4, label="RIGHT (honest)", color=viz.RIGHT_COLOR)
for xi, (w, r) in enumerate(zip(wrongs, rights)):
    ax.text(xi - 0.2, w + 0.01, f"{w:.2f}", ha="center", fontsize=8)
    ax.text(xi + 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title("Six ways evaluation lies — and the honest number in each case")
ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# # ✅ Evaluation checklist (copy this into your next project)
#
# ```text
# DATA SPLITTING
# [ ] No random-shuffle split on time series / epochs (use block- or subject-aware).
# [ ] Whole trials/blocks stay on one side of the split.
# [ ] Headline metric is subject-INDEPENDENT (Leave-One-Subject-Out).
# [ ] (If continuous) a gap/purge separates train and test in time.
#
# LEAKAGE
# [ ] Every learned transform (scaler, CSP, ICA, PCA, feature selection) is in a
#     Pipeline and fit on TRAIN folds only.
# [ ] No normalisation/statistics computed on the full dataset before splitting.
# [ ] Hyper-parameters tuned with nested CV / a separate validation set (not test).
#
# METRICS
# [ ] Reported balanced accuracy / F1 / ROC-AUC for imbalanced problems (not just accuracy).
# [ ] Looked at the confusion matrix.
# [ ] Compared against chance AND a simple baseline (e.g. majority class, CSP+LDA).
#
# ROBUSTNESS
# [ ] Evaluated across the shift you'll deploy under (session/day/device/subject).
# [ ] Seeded all RNGs.
# [ ] Reported mean ± std over folds/seeds (never a single number).
# [ ] Used a PAIRED test across folds before claiming model A beats model B.
# ```

# %% [markdown]
# # Recap table — each pitfall → its one-line fix
#
# | # | Pitfall (the trap) | One-line fix |
# |---|---|---|
# | 1 | Random shuffle on correlated time series | Split by trial/block (`make_block_split`) |
# | 2 | Pool subjects, then random split | Leave-One-Subject-Out (`make_subject_split`) |
# | 3 | Fit scaler/CSP/selection on all data | Fit inside a Pipeline, train folds only |
# | 4 | Accuracy on imbalanced classes | Balanced accuracy / F1 / ROC-AUC + confusion matrix |
# | 5 | Test on the same session as training | Test across sessions/days/devices/subjects |
# | 6 | One run, report the best number | Mean ± std over folds/seeds + paired test |
#
# > **The big idea:** in neural-signal ML, the honest number is almost always
# > *lower* than the first number you get. Chasing the drop — not the peak — is what
# > separates results that ship from results that vanish.

# %% [markdown]
# ## ✅ Concept check
#
# 1. In Pitfall 1, overlapping windows from the same trial are split randomly.
#    Name the statistical property that makes this splitting strategy invalid for
#    time-series data and explain why it inflates the score.
# 2. You have a 4-class dataset where one class makes up 70% of trials. Your model
#    achieves 72% accuracy. Without looking at the confusion matrix, why can you not
#    conclude the model is useful?
# 3. A colleague reports a paired t-test between CSP+LDA and a deep net across 9
#    held-out subjects, obtaining p = 0.04 and a mean difference of +1.2%. Should
#    you treat this as a meaningful improvement? What additional information do you need?
#
# **Answers:**
# 1. Adjacent overlapping windows share most of their samples, making them highly
#    autocorrelated. Random splitting places near-duplicate windows on both sides of
#    the train/test boundary, so the model can "recognise" a test window by its
#    near-twin in training — inflating accuracy through memorisation, not generalisation.
# 2. A model that always predicts the majority class would achieve ~70% accuracy
#    without learning anything. You need balanced accuracy, F1, and the confusion
#    matrix to determine whether the model actually detects the minority classes.
# 3. Statistical significance (p = 0.04) does not imply practical significance.
#    With only 9 folds the test has low power, and a 1.2% mean difference is likely
#    smaller than the standard deviation across subjects. You need the effect size
#    (mean ± std of per-subject differences) and should check whether the result
#    survives correction for multiple comparisons across all models you tested.

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong (meta)
#
# - **Treating this chapter as optional.** It is the difference between real and
#   fake results.
# - **Fixing one leak and trusting the number.** These pitfalls stack — audit all six.
# - **Reporting the WRONG numbers above as if they were achievements.** Every red
#   bar here is a cautionary tale, not a benchmark.
#
# **Next:** Chapter 13 — *neuroethics & anti-hype*: over-claiming to the public is
# the ethical twin of leakage. Then Chapter 14 — the capstone, where you build a
# leakage-free report yourself against a hidden held-out set.
