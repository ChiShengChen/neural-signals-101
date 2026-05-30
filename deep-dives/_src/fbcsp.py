# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Filter-Bank CSP (FBCSP)
#
# The classic BCI-competition-winning extension of single-band CSP (Ang et al. 2008).
#
# > **Prerequisites:** main Chapters 06 and 07.
# > **Level:** advanced ★★★★☆
# > **The classic strong motor-imagery baseline.**

# %% [markdown]
# ## 0 — Bootstrap

# %%
import sys
import os
from pathlib import Path

# Robust upward search for src/neuro101
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path.cwd()
    for _parent in [_here, *_here.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101" / "__init__.py").exists():
            sys.path.insert(0, str(_candidate))
            break
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — safe in CI and nbconvert
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mne
mne.set_log_level("ERROR")

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuro101.io import load_bnci_2a_epochs
from neuro101.preprocessing import bandpass_filter
from neuro101.features import make_csp
from neuro101.eval import make_block_split

rng = np.random.default_rng(42)
np.random.seed(42)

SMOKE = os.environ.get("NEURO101_SMOKE") == "1"
N_SUBJECTS = 2 if SMOKE else 3   # subjects to benchmark (smoke: 2 to keep runtime tiny)
N_SPLITS   = 3 if SMOKE else 5   # block-CV folds per subject
N_SEEDS    = 1 if SMOKE else 3   # seeds for variance estimation

print(f"SMOKE={SMOKE}  N_SUBJECTS={N_SUBJECTS}  N_SPLITS={N_SPLITS}  N_SEEDS={N_SEEDS}")

# %% [markdown]
# ---
# ## 1 — Motivation: why a single band is suboptimal
#
# Classic CSP is typically applied to a single broad band (e.g. 8–30 Hz) because that
# captures both the mu (~8–12 Hz) and beta (~13–30 Hz) rhythms that are suppressed
# during motor imagery (event-related desynchronisation, ERD).
#
# **The problem:** the exact frequency of peak ERD varies substantially across:
# * **subjects** — one person's mu peak may be at 9 Hz, another's at 11 Hz,
# * **tasks** — hand imagery favours mu+low-beta; foot imagery tends to shift upward,
# * **sessions** — fatigue and attention shift the spectral profile.
#
# A single 8–30 Hz band conflates all these sub-bands. CSP spatial filters tuned on
# this broad range blend useful and useless frequency content together, diluting the
# discriminative signal. **Filter-Bank CSP** (FBCSP; Ang et al., 2008, *IJCNN*)
# resolves this by:
#
# 1. Band-passing the epochs into several narrow sub-bands independently,
# 2. Running a separate CSP inside each sub-band,
# 3. Concatenating all per-band log-variance features into one wide vector,
# 4. Using **mutual-information feature selection** to keep only the most
#    informative band×component pairs, then
# 5. Classifying with LDA.
#
# The key insight is that feature selection step (4) is data-driven: if a subject's
# discriminative rhythm lives entirely in 8–12 Hz, the selector will automatically
# up-weight those components and ignore the rest.

# %% [markdown]
# ### Sub-band layout
#
# We use seven 4-Hz-wide sub-bands covering 4–32 Hz, which span the theta, mu, and
# beta ranges where motor imagery effects are found.

# %%
# Sub-bands: list of (low_hz, high_hz) tuples
SUBBANDS = [
    (4,  8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
]
BAND_LABELS = [f"{lo}-{hi} Hz" for lo, hi in SUBBANDS]
N_BANDS = len(SUBBANDS)
print(f"Sub-bands ({N_BANDS} total): {BAND_LABELS}")

# %% [markdown]
# ---
# ## 2 — FBCSP implementation
#
# We build FBCSP as an **sklearn-compatible transformer** so that it fits entirely
# on the training fold when placed inside a cross-validation loop — no leakage.

# %%
class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """Filter-Bank Common Spatial Patterns transformer.

    Applies bandpass filtering into ``subbands``, fits an independent CSP per
    band on the training data, then concatenates the log-variance features from
    all bands into a single feature vector per trial.

    Parameters
    ----------
    subbands : list of (float, float)
        Frequency ranges (Hz) for each sub-band.
    sfreq : float
        Sampling frequency of the input epochs (Hz).
    n_components : int
        Number of CSP components per band (top + bottom n_components//2).
    """

    def __init__(self, subbands=None, sfreq=250.0, n_components=4):
        self.subbands = subbands or SUBBANDS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y):
        """Fit one CSP per sub-band on (X, y)."""
        self.csps_ = []
        for low, high in self.subbands:
            Xf = bandpass_filter(X, self.sfreq, low, high)
            csp = make_csp(n_components=self.n_components)
            csp.fit(Xf, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        """Return concatenated log-variance features from all bands.

        Returns
        -------
        np.ndarray of shape (n_trials, n_bands * n_components)
        """
        feats = []
        for (low, high), csp in zip(self.subbands, self.csps_):
            Xf = bandpass_filter(X, self.sfreq, low, high)
            feats.append(csp.transform(Xf))   # (n_trials, n_components)
        return np.concatenate(feats, axis=1)  # (n_trials, n_bands * n_components)


def make_fbcsp_pipeline(subbands, sfreq, n_components=4, k_best=8):
    """Return a leakage-safe FBCSP → SelectKBest → LDA pipeline.

    All steps — CSP fitting, feature selection, and LDA — are fit on the
    training fold only when used inside a CV loop.
    """
    steps = [
        ("fbcsp",    FBCSPTransformer(subbands=subbands, sfreq=sfreq,
                                      n_components=n_components)),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=k_best)),
        ("lda",      LDA()),
    ]
    return Pipeline(steps)


def make_single_band_csp_pipeline(sfreq, fmin=8.0, fmax=30.0, n_components=4):
    """Baseline: single-band 8-30 Hz CSP → LDA pipeline."""
    from neuro101.features import make_csp as _make_csp

    class SingleBandCSP(BaseEstimator, TransformerMixin):
        def __init__(self, sfreq=250.0, fmin=8.0, fmax=30.0, n_components=4):
            self.sfreq = sfreq
            self.fmin = fmin
            self.fmax = fmax
            self.n_components = n_components

        def fit(self, X, y):
            Xf = bandpass_filter(X, self.sfreq, self.fmin, self.fmax)
            self.csp_ = _make_csp(n_components=self.n_components)
            self.csp_.fit(Xf, y)
            return self

        def transform(self, X):
            Xf = bandpass_filter(X, self.sfreq, self.fmin, self.fmax)
            return self.csp_.transform(Xf)

    return Pipeline([
        ("csp", SingleBandCSP(sfreq=sfreq, fmin=fmin, fmax=fmax,
                              n_components=n_components)),
        ("lda", LDA()),
    ])


print("FBCSP and single-band CSP pipeline factories defined.")

# %% [markdown]
# ---
# ## 3 — Data loading
#
# We load BCI IV 2a (left-hand vs right-hand motor imagery) for the first few
# subjects. The loader returns broadband epochs; we will do our own sub-band
# filtering inside the FBCSP transformer.
#
# **Important:** we load with the *raw* 0.5–100 Hz paradigm band instead of the
# usual 8–30 Hz pre-filtering, so our filter bank has full access to the spectrum.

# %%
print(f"Loading BCI IV 2a for {N_SUBJECTS} subject(s) …")
X_all, y_all, subjects_all = load_bnci_2a_epochs(
    n_subjects=N_SUBJECTS,
    fmin=0.5,     # minimal pre-filtering — let our filter bank do the work
    fmax=40.0,
    tmin=0.5,
    tmax=2.5,
)
sfreq = 250.0   # BCI IV 2a sampling rate
print(f"X shape: {X_all.shape}  |  classes: {np.unique(y_all)}  |  sfreq: {sfreq} Hz")
print(f"Trials per subject: { {s: int((subjects_all == s).sum()) for s in np.unique(subjects_all)} }")

# %% [markdown]
# ---
# ## 4 — Within-subject (block) cross-validation
#
# We evaluate both pipelines within each subject using `make_block_split`, which
# respects temporal order and prevents adjacent-trial leakage. Feature selection
# with `SelectKBest` is fit inside each fold alongside the CSP filters — this is
# the critical design choice that prevents leakage.

# %%
def run_cv_for_subject(X_subj, y_subj, sfreq, n_splits, n_seeds):
    """Return per-fold accuracy arrays for FBCSP and single-band CSP.

    Returns
    -------
    fb_accs : list of float    per-fold accuracies for FBCSP
    sb_accs : list of float    per-fold accuracies for single-band CSP
    """
    # Number of features to select: 2 per band (top + bottom CSP) × n_bands
    # or up to total available features
    n_total = N_BANDS * 4   # 4 components per band
    k_best  = min(n_total, max(2, N_BANDS * 2))

    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    sb_pipe = make_single_band_csp_pipeline(sfreq, n_components=4)

    fb_accs, sb_accs = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        for tr, te in make_block_split(len(y_subj), n_splits=n_splits):
            # Clone so each fold starts fresh (no state bleed across folds)
            fb = clone(fb_pipe)
            sb = clone(sb_pipe)

            fb.fit(X_subj[tr], y_subj[tr])
            sb.fit(X_subj[tr], y_subj[tr])

            fb_accs.append(fb.score(X_subj[te], y_subj[te]))
            sb_accs.append(sb.score(X_subj[te], y_subj[te]))

    return fb_accs, sb_accs


results = {}
unique_subjects = np.unique(subjects_all)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]
    print(f"  Subject {s}: {Xs.shape[0]} trials, {N_SPLITS}-fold CV × {N_SEEDS} seeds …")
    fb_accs, sb_accs = run_cv_for_subject(Xs, ys, sfreq,
                                          n_splits=N_SPLITS, n_seeds=N_SEEDS)
    results[s] = {
        "fb_accs": np.array(fb_accs),
        "sb_accs": np.array(sb_accs),
        "fb_mean": float(np.mean(fb_accs)),
        "fb_std":  float(np.std(fb_accs)),
        "sb_mean": float(np.mean(sb_accs)),
        "sb_std":  float(np.std(sb_accs)),
    }
    print(f"    FBCSP: {results[s]['fb_mean']:.3f} ± {results[s]['fb_std']:.3f}  |  "
          f"Single-band CSP: {results[s]['sb_mean']:.3f} ± {results[s]['sb_std']:.3f}")

# Overall (pooled across subjects and folds)
all_fb = np.concatenate([results[s]["fb_accs"] for s in unique_subjects])
all_sb = np.concatenate([results[s]["sb_accs"] for s in unique_subjects])
print(f"\nOverall FBCSP:        {all_fb.mean():.3f} ± {all_fb.std():.3f}")
print(f"Overall Single-band:  {all_sb.mean():.3f} ± {all_sb.std():.3f}")

# %% [markdown]
# ---
# ## 5 — Visualisation 1: per-subject accuracy comparison
#
# The bar chart below shows FBCSP vs single-band CSP accuracy for each subject.
# Error bars show ± 1 std across folds × seeds.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.32
x = np.arange(len(unique_subjects))

fb_means = [results[s]["fb_mean"] for s in unique_subjects]
fb_stds  = [results[s]["fb_std"]  for s in unique_subjects]
sb_means = [results[s]["sb_mean"] for s in unique_subjects]
sb_stds  = [results[s]["sb_std"]  for s in unique_subjects]

bars_fb = ax.bar(x - bar_width / 2, fb_means, bar_width,
                 yerr=fb_stds, capsize=5,
                 color="steelblue", alpha=0.85, label="FBCSP (filter-bank)")
bars_sb = ax.bar(x + bar_width / 2, sb_means, bar_width,
                 yerr=sb_stds, capsize=5,
                 color="tomato", alpha=0.85, label="Single-band CSP (8–30 Hz)")

ax.axhline(0.5, color="black", linestyle="--", lw=1.5, label="Chance (50%)")
ax.set_xticks(x)
ax.set_xticklabels([f"Subject {s}" for s in unique_subjects], fontsize=11)
ax.set_ylabel("Accuracy (block CV, mean ± std)", fontsize=11)
ax.set_title("FBCSP vs Single-band CSP — BCI IV 2a (left vs right hand)\n"
             f"Block CV ({N_SPLITS} folds × {N_SEEDS} seeds)", fontsize=11)
ax.set_ylim(0.35, 1.02)
ax.legend(fontsize=10)

# Annotate the delta on top of FBCSP bars
for xi, (fm, sm) in enumerate(zip(fb_means, sb_means)):
    delta = fm - sm
    sign  = "+" if delta >= 0 else ""
    ax.text(xi - bar_width / 2, fm + (fb_stds[xi] if fb_stds else 0.02) + 0.025,
            f"{sign}{delta:.2f}", ha="center", fontsize=9, color="steelblue",
            fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/dd_fbcsp_subject_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# ---
# ## 6 — Visualisation 2: feature-band selection profile
#
# One of FBCSP's great interpretive virtues is that the feature selector tells us
# **which sub-bands were most discriminative** for each subject. By fitting the full
# pipeline on each subject's training data (here, all folds combined as an illustrative
# fit) we can inspect which band×component pairs survive the mutual-information gate.
#
# The bar chart below shows the **count of selected features from each sub-band**,
# summed across subjects. Sub-bands in the mu (8–12 Hz) and low-beta (12–16 Hz) range
# should dominate for left-vs-right hand imagery.

# %%
band_selection_counts = np.zeros(N_BANDS, dtype=int)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]

    n_total = N_BANDS * 4
    k_best  = min(n_total, max(2, N_BANDS * 2))

    # Fit FBCSP on the full subject data (illustrative only — within-subject
    # generalisation is measured by the block CV above, not this fit).
    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    fb_pipe.fit(Xs, ys)

    # The selector knows which features (0..N_BANDS*4-1) were kept.
    selector = fb_pipe.named_steps["selector"]
    selected_features = selector.get_support(indices=True)
    # Feature i comes from band i // n_components
    for feat_idx in selected_features:
        band_idx = feat_idx // 4   # 4 components per band
        band_selection_counts[band_idx] += 1

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: selected feature count per band
ax = axes[0]
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, N_BANDS))
bars = ax.bar(range(N_BANDS), band_selection_counts, color=colors,
              edgecolor="black", linewidth=0.8)
ax.set_xticks(range(N_BANDS))
ax.set_xticklabels(BAND_LABELS, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Selected feature count (across subjects)", fontsize=11)
ax.set_title(f"Which sub-bands does SelectKBest choose?\n"
             f"(mutual_info_classif, {N_SUBJECTS} subject(s))", fontsize=11)

# Annotate physiology
for band_name, band_range in [("mu", (8, 12)), ("beta", (12, 20))]:
    lo, hi = band_range
    for i, (bl, bh) in enumerate(SUBBANDS):
        if bl >= lo and bh <= hi + 4:
            ax.bar(i, band_selection_counts[i], color=colors[i],
                   edgecolor="navy", linewidth=2.0)

# Add a rough physiology legend
mu_patch   = mpatches.Patch(edgecolor="navy", facecolor="none",
                              linewidth=2, label="mu / beta band")
ax.legend(handles=[mu_patch], fontsize=9)

for bar, count in zip(bars, band_selection_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            str(count), ha="center", fontsize=11, fontweight="bold")

ax.set_ylim(0, band_selection_counts.max() * 1.3 + 1)

# Right: mutual information scores averaged across subjects (per band)
ax2 = axes[1]
mi_band_avg = np.zeros(N_BANDS)
mi_band_counts = np.zeros(N_BANDS, dtype=int)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]

    n_total = N_BANDS * 4
    k_best  = min(n_total, max(2, N_BANDS * 2))

    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    fb_pipe.fit(Xs, ys)

    selector = fb_pipe.named_steps["selector"]
    mi_scores = selector.scores_  # shape: (N_BANDS * n_components,)

    for feat_idx, score in enumerate(mi_scores):
        band_idx = feat_idx // 4
        mi_band_avg[band_idx] += score
        mi_band_counts[band_idx] += 1

mi_band_avg /= mi_band_counts

bars2 = ax2.bar(range(N_BANDS), mi_band_avg, color=colors,
                edgecolor="black", linewidth=0.8)
ax2.set_xticks(range(N_BANDS))
ax2.set_xticklabels(BAND_LABELS, rotation=35, ha="right", fontsize=10)
ax2.set_ylabel("Mean mutual information score (nats)", fontsize=11)
ax2.set_title("Average mutual-information score per sub-band\n"
              "(averaged over components and subjects)", fontsize=11)
ax2.set_ylim(0, mi_band_avg.max() * 1.35 + 0.01)

for bar, score in zip(bars2, mi_band_avg):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mi_band_avg.max() * 0.02,
             f"{score:.3f}", ha="center", fontsize=9)

plt.suptitle("FBCSP band-importance analysis — mu/beta physiology should emerge",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fbcsp_band_selection.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure 2 saved.")

# %% [markdown]
# **Reading the plots:** Sub-bands overlapping the mu (8–12 Hz) and low-beta
# (12–16 Hz) ranges typically accumulate the most selected features and the
# highest mutual-information scores for left-vs-right hand imagery, consistent
# with the known ERD physiology. The 4–8 Hz (theta) and 28–32 Hz (high-beta)
# bands are usually less informative, but this is subject-dependent — exactly
# the variability that FBCSP's data-driven selection is designed to handle.

# %% [markdown]
# ---
# ## 7 — Summary table

# %%
print("=" * 62)
print(f"{'Method':<25} {'Mean Acc':>10} {'Std':>8}")
print("=" * 62)
for s in unique_subjects:
    r = results[s]
    print(f"  S{s} FBCSP           {r['fb_mean']:>10.3f} {r['fb_std']:>8.3f}")
    print(f"  S{s} Single-band CSP {r['sb_mean']:>10.3f} {r['sb_std']:>8.3f}")
    print()

print("-" * 62)
print(f"  Overall FBCSP        {all_fb.mean():>10.3f} {all_fb.std():>8.3f}")
print(f"  Overall Single-band  {all_sb.mean():>10.3f} {all_sb.std():>8.3f}")
print("=" * 62)
print(f"\nFBCSP vs Single-band delta: {all_fb.mean() - all_sb.mean():+.3f}")

# %% [markdown]
# ---
# ## ⚠️ A subtler trap: feature-selection leakage looks just like a clean pipeline
#
# FBCSP concatenates features from N_BANDS × n_components sources (e.g. 7 × 4 = 28
# features for our setup). `SelectKBest` then picks the *k* most informative ones
# using mutual information between features and labels.
#
# Here is the trap: **if `SelectKBest` is fit on all the data — even for a moment —
# before the train/test split, the selected features carry label information from
# the test set into training.** This is one of the most common and hardest-to-spot
# leakage patterns in BCI papers, because:
#
# 1. The code *looks* correct — the CSP fitting and LDA are inside a pipeline.
# 2. The leakage does not come from fitting a covariance matrix on test data; it
#    comes from the *ranking* of features. Even a single mutual-information
#    computation over the full (train+test) set biases which features survive,
#    because the selector has "seen" which bands were predictive on the test trials.
# 3. The bias is **multiplicative with the number of features tried**: with 28
#    candidate features and `k=8`, 20 features are discarded. If those 20 include
#    test-leakage signal, the selected 8 are systematically enriched for test
#    patterns that will not generalise.
#
# ### Why it is subtle
#
# A naive sanity check — "did I call `fit` on test data?" — passes. The selection
# is done on scores computed *before* any explicit `fit` call. A correct check is:
# "does any object that uses label information observe test-set samples **before**
# the train/test boundary?" For `mutual_info_classif`, the answer is yes if you
# call it outside the pipeline.
#
# ### The correct pattern (what this notebook does)
#
# ```python
# # CORRECT: SelectKBest is inside Pipeline; sklearn clones and re-fits it per fold
# pipeline = Pipeline([
#     ("fbcsp",    FBCSPTransformer(...)),   # fit on train only
#     ("selector", SelectKBest(mutual_info_classif, k=k)),  # fit on train only
#     ("lda",      LDA()),                   # fit on train only
# ])
# # Inside a CV loop, pipeline.fit(X_train, y_train) fits ALL steps on train only.
# ```
#
# ### A related pitfall: band selection by test performance
#
# A subtler variant: instead of `SelectKBest`, a researcher manually inspects which
# sub-band gives the highest *test-set* accuracy across a grid search, then reports
# that accuracy as the score for "FBCSP with the best band". This is optimism bias
# (selection from test performance), not leakage in the strict sense, but the result
# is equally invalid — the reported number answers the question "how good is FBCSP
# when we know the answer?" rather than "how good is FBCSP on new data?".
#
# **The rule:** everything that touches labels and data must be inside the CV fold,
# fit exclusively on the training partition.
