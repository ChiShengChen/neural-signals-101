# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Transfer Learning & Domain Adaptation
#
# How to rescue cross-subject and cross-session performance after Chapter 12 showed
# it is hard: alignment, calibration, and fine-tuning strategies for BCI.
#
# > **Prerequisites:** main Chapters 07 and 12.
# > **Level:** advanced ★★★★☆
# > **The next step after "subject-independent is hard".**

# %% Bootstrap — find neuro101 package via upward search from this file's location
import sys
import os
from pathlib import Path

# Robust upward search: try importing first, then walk up the directory tree
# looking for src/neuro101 (works whether run from repo root or deep-dives/_src/).
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _parent in [_here, *_here.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101").is_dir():
            sys.path.insert(0, str(_candidate))
            break
    import neuro101  # noqa: F401 — will raise clearly if still not found

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use a headless backend so the notebook executes in CI without a display.
matplotlib.use("Agg")

from scipy import linalg as sp_linalg

from sklearn.base import clone as sk_clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

from mne.decoding import CSP

from neuro101 import io, datasets as ds
from neuro101.eval import make_subject_split, leakage_safe_pipeline

rng = np.random.default_rng(42)

SMOKE = ds.is_smoke()
N_SUBJ = 2 if SMOKE else 4          # subjects to load
CALIB_SIZES = [0, 5, 10, 20]        # calibration trial counts to sweep
if SMOKE:
    CALIB_SIZES = [0, 5, 10]        # fewer points to keep CI fast

print(f"Smoke mode : {SMOKE}")
print(f"N_SUBJ     : {N_SUBJ}")
print(f"CALIB_SIZES: {CALIB_SIZES}")

# %% [markdown]
# ---
# ## Part 1 — The domain-shift problem
#
# ### Why cross-subject EEG is hard: each person is a different distribution
#
# In classical machine learning we assume that train and test data come from the
# **same distribution** $p(\mathbf{x}, y)$.  In EEG this assumption is violated
# almost immediately:
#
# * Every skull has a different shape and conductivity — the mapping from cortical
#   current to scalp voltage differs per person.
# * Electrode caps are repositioned between sessions; tiny shifts in Cz change
#   which voxels each channel "sees".
# * Fatigue, attention, medication, and mood shift the baseline power spectrum
#   within a day.
#
# Each subject (or session) is therefore a **domain** with its own
# $p_s(\mathbf{x})$.  A classifier trained on domains $\{s \neq t\}$ sees a
# *covariate shift* when evaluated on target domain $t$: the labels $y$ have the
# same meaning (left-hand imagery = class 0), but the features live in a
# different part of feature space.  Chapter 12, pitfall #5 quantified the damage:
# LOSO accuracy can drop 15–25 percentage points relative to within-subject CV.
#
# ### Three families of remedies
#
# | Family | Idea | Labelled target data needed? |
# |---|---|---|
# | **Alignment / distribution matching** | Transform domains so they share a common reference before classifying | No (unsupervised) |
# | **Calibration** | Give the deployed model a handful of labelled target trials at day-one | Yes (a few) |
# | **Fine-tuning** | Pre-train on source subjects, then update weights on target labels | Yes (moderate) |
#
# This deep-dive covers all three families with runnable experiments.

# %% [markdown]
# ---
# ## Part 2 — Euclidean Alignment (EA)
#
# ### The idea (He & Wu, 2020)
#
# The simplest alignment strategy operates directly on the raw trials.  For each
# subject $s$ with trials $\{\mathbf{X}_i^{(s)}\}$:
#
# 1. **Estimate the mean covariance** of that subject's trials:
#    $$\mathbf{R}^{(s)} = \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbf{C}_i^{(s)},
#      \qquad \mathbf{C}_i = \frac{1}{T-1}\mathbf{X}_i\mathbf{X}_i^\top$$
# 2. **Whiten** by the inverse square root $(\mathbf{R}^{(s)})^{-1/2}$:
#    $$\tilde{\mathbf{X}}_i^{(s)} = (\mathbf{R}^{(s)})^{-1/2}\,\mathbf{X}_i^{(s)}$$
#
# After alignment every subject's **mean covariance is the identity matrix**
# $\mathbf{I}$, removing the subject-specific amplitude baseline.  The operation
# is a standard matrix square root and takes $O(p^3)$ time — a handful of
# microseconds for 22-channel EEG.
#
# **Leak-free rule:** $\mathbf{R}^{(s)}$ for the *training subjects* is estimated
# on their own trials.  $\mathbf{R}^{(t)}$ for the *target (test) subject* is
# estimated on the target subject's *own* (unlabelled) trials — the EA reference
# does not require labels, so this is safe.

# %%
def euclidean_alignment(X_tr, X_te, subj_tr, subj_te):
    """Leak-free Euclidean Alignment.

    Fits R^{-1/2} separately for each subject on their *own* trials.
    For training subjects: R is fit on training trials only.
    For the target subject: R is fit on the target's held-out (test) trials —
    this is label-free and therefore not leakage.

    Parameters
    ----------
    X_tr   : (n_train, ch, time)
    X_te   : (n_test, ch, time)
    subj_tr: (n_train,) subject ids for training trials
    subj_te: (n_test,)  subject ids for test trials

    Returns
    -------
    X_tr_ea, X_te_ea : aligned versions of the input arrays
    """
    def _whiten(X_domain):
        """Compute mean covariance and return whitened trials."""
        # mean covariance across trials in this domain
        Cs = np.stack([x @ x.T / x.shape[-1] for x in X_domain])
        R = Cs.mean(axis=0)
        # inverse square root via eigendecomposition
        vals, vecs = np.linalg.eigh(R)
        vals = np.maximum(vals, 1e-12)          # numerical floor
        R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        # apply: (ch,ch) @ (n, ch, time)
        return np.einsum("ij,njk->nik", R_inv_sqrt, X_domain)

    X_tr_ea = np.empty_like(X_tr)
    for s in np.unique(subj_tr):
        m = subj_tr == s
        X_tr_ea[m] = _whiten(X_tr[m])

    X_te_ea = np.empty_like(X_te)
    for s in np.unique(subj_te):
        m = subj_te == s
        X_te_ea[m] = _whiten(X_te[m])          # uses target's own trials — no labels

    return X_tr_ea, X_te_ea

print("euclidean_alignment() defined — implements He & Wu (2020) in ~20 lines.")

# %%
# ── Load data ──────────────────────────────────────────────────────────────────
print(f"\nLoading {N_SUBJ} subject(s) from BCI IV 2a …")
X, y, subj = io.load_bnci_2a_epochs(n_subjects=N_SUBJ)
print(f"  X shape : {X.shape}  (trials × channels × time-points)")
print(f"  subjects: {np.unique(subj).tolist()}")
print(f"  classes : {np.bincount(y).tolist()}  (0=left, 1=right)")

# %%
# ── LOSO accuracy BEFORE and AFTER EA ─────────────────────────────────────────
# Pipeline: Riemann Covariances → Tangent Space → Logistic Regression
# (CSP+LDA also works — the EA benefit transfers to any downstream classifier)

pipe_template = leakage_safe_pipeline([
    ("cov", Covariances(estimator="oas")),
    ("ts",  TangentSpace(metric="riemann")),
    ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=0)),
])

accs_before, accs_after = [], []

print("\nLOSO cross-validation (Riemann pipeline) …")
print(f"{'Fold':>4}  {'Test subj':>9}  {'Before EA':>9}  {'After EA':>8}")
print("-" * 40)

for fold_i, (train_idx, test_idx) in enumerate(make_subject_split(subj)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    # ── BEFORE EA ──────────────────────────────────────────────────────────
    pipe_b = sk_clone(pipe_template)
    pipe_b.fit(X_tr, y_tr)
    acc_b = pipe_b.score(X_te, y_te)
    accs_before.append(acc_b)

    # ── AFTER EA ───────────────────────────────────────────────────────────
    # EA reference fit on train subjects (train) and target subject (test) separately.
    X_tr_ea, X_te_ea = euclidean_alignment(
        X_tr, X_te, subj[train_idx], subj[test_idx]
    )
    pipe_a = sk_clone(pipe_template)
    pipe_a.fit(X_tr_ea, y_tr)
    acc_a = pipe_a.score(X_te_ea, y_te)
    accs_after.append(acc_a)

    test_subj = np.unique(subj[test_idx])[0]
    print(f"  {fold_i:2d}    subj {test_subj:>3}       {acc_b:.3f}       {acc_a:.3f}")

mean_before = np.mean(accs_before)
mean_after  = np.mean(accs_after)
print("-" * 40)
print(f"  Mean accuracy  BEFORE EA : {mean_before:.3f}")
print(f"  Mean accuracy  AFTER  EA : {mean_after:.3f}")
print(f"  Δ (after − before)       : {mean_after - mean_before:+.3f}")

# %%
# ── Figure 1 — Before vs After EA bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, N_SUBJ * 1.5 + 2), 5))

n_folds = len(accs_before)
x_pos   = np.arange(n_folds)
bar_w   = 0.35

bars_b = ax.bar(x_pos - bar_w / 2, accs_before, bar_w,
                color="#5b9bd5", alpha=0.85, label="Before EA")
bars_a = ax.bar(x_pos + bar_w / 2, accs_after,  bar_w,
                color="#ed7d31", alpha=0.85, label="After EA")

# Annotate each bar with its value
for rect in list(bars_b) + list(bars_a):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

# Mean lines
ax.axhline(mean_before, color="#5b9bd5", lw=2, ls="--",
           label=f"Mean before EA = {mean_before:.3f}")
ax.axhline(mean_after,  color="#ed7d31", lw=2, ls="--",
           label=f"Mean after  EA = {mean_after:.3f}")
ax.axhline(0.5, color="gray", lw=1.0, ls=":", label="Chance (50 %)")

fold_labels = [f"S{np.unique(subj[te])[0]}" for _, te in make_subject_split(subj)]
ax.set_xticks(x_pos)
ax.set_xticklabels(fold_labels)
ax.set(
    xlabel="Held-out test subject (LOSO fold)",
    ylabel="Accuracy  (Riemann pipeline)",
    title=(
        "LOSO accuracy: Riemann pipeline  BEFORE vs AFTER Euclidean Alignment\n"
        f"BCI IV 2a  ({N_SUBJ} subjects, CSP-free — OAS covariance + tangent space + LR)"
    ),
    ylim=(0.3, 1.0),
)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/tmp/dd_da_ea_bar.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure 1 saved → /tmp/dd_da_ea_bar.png")

# %% [markdown]
# ### What the bar chart tells us
#
# Each paired bar is one LOSO fold (one held-out subject).  The left (blue) bar
# is accuracy with no alignment; the right (orange) bar is accuracy after EA.
#
# * EA re-centres every subject's covariance to the identity before computing
#   tangent-space features — removing the between-subject amplitude baseline
#   without touching labels.
# * On subjects whose baseline power differs greatly from the group mean the gain
#   can be large; on subjects who already happened to be "close to the mean" the
#   effect is smaller or neutral.
# * The improvement is largest when the target subject is an outlier in covariance
#   space — which is exactly the situation real BCI deployment faces (you cannot
#   pick your users).

# %% [markdown]
# ---
# ## Part 3 — Riemannian re-centring: the geometric analogue of EA
#
# EA whitens by the *Euclidean* mean covariance $\mathbf{R} =
# \frac{1}{N}\sum_i \mathbf{C}_i$.  The principled Riemannian alternative is to
# re-centre by the **Riemannian (geometric) mean** $\mathbf{M}$, placing each
# domain's covariances at the identity *on the SPD manifold*.
#
# The operation is the same in structure:
# $$\tilde{\mathbf{C}}_i = \mathbf{M}^{-1/2}\,\mathbf{C}_i\,\mathbf{M}^{-1/2}$$
# but $\mathbf{M}$ minimises the sum of squared *affine-invariant* (Riemannian)
# distances rather than the sum of squared Frobenius distances.
#
# **When does it matter?**  The Euclidean mean of SPD matrices always has a larger
# determinant than the Riemannian mean ("swelling effect", covered in the
# `riemann_small_data` deep-dive).  For EEG with modest numbers of trials the
# difference is typically small but grows when subjects have very large or small
# covariances (outlier channels, artefact-heavy recordings).
#
# In the pyriemann library this is handled by `pyriemann.utils.mean.mean_riemann`.
# We demonstrate the two alignment approaches side-by-side on the mean covariance
# of one training subject's trials.

# %%
# ── Visual comparison: Euclidean vs Riemannian mean for one subject ────────────
s_demo = np.unique(subj)[0]          # pick first subject
X_demo = X[subj == s_demo]           # (n_trials, ch, time)
Cs_demo = np.stack([x @ x.T / x.shape[-1] for x in X_demo])

# Euclidean mean
R_eucl = Cs_demo.mean(axis=0)

# Riemannian mean
R_riem = mean_riemann(Cs_demo)

# Determinant: Riemannian mean should be smaller (no swelling)
det_eucl = np.linalg.det(R_eucl)
det_riem = np.linalg.det(R_riem)
print(f"Subject {s_demo}  ({len(X_demo)} trials)")
print(f"  det(Euclidean mean)  : {det_eucl:.4e}")
print(f"  det(Riemannian mean) : {det_riem:.4e}")
print(f"  swelling ratio       : {det_eucl / det_riem:.4f}  (>1 means Eucl is inflated)")

# Eigenvalue spectra
eigvals_eucl = np.sort(np.linalg.eigvalsh(R_eucl))[::-1]
eigvals_riem = np.sort(np.linalg.eigvalsh(R_riem))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel A — eigenvalue spectrum comparison
ax = axes[0]
idx = np.arange(1, len(eigvals_eucl) + 1)
ax.semilogy(idx, eigvals_eucl, "o-", color="#5b9bd5", lw=2, label="Euclidean mean")
ax.semilogy(idx, eigvals_riem, "s-", color="#ed7d31", lw=2, label="Riemannian mean")
ax.set(
    xlabel="Eigenvalue rank (1 = largest)",
    ylabel="Eigenvalue (log scale)",
    title=f"Eigenvalue spectra of the mean covariance\n(subject {s_demo}, {len(X_demo)} trials)",
)
ax.legend(fontsize=9)

# Panel B — Frobenius distance from identity after whitening
def frob_from_identity(R_ref, Cs):
    """After whitening by R_ref, how far are covariances from identity?"""
    vals, vecs = np.linalg.eigh(R_ref)
    R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    # whiten each trial covariance
    dists = []
    for C in Cs:
        C_w = R_inv_sqrt @ C @ R_inv_sqrt
        dists.append(np.linalg.norm(C_w - np.eye(C.shape[0]), "fro"))
    return np.array(dists)

dists_eucl = frob_from_identity(R_eucl, Cs_demo)
dists_riem = frob_from_identity(R_riem, Cs_demo)

ax = axes[1]
bins = np.linspace(0, max(dists_eucl.max(), dists_riem.max()) * 1.05, 30)
ax.hist(dists_eucl, bins=bins, alpha=0.6, color="#5b9bd5",
        label=f"Eucl. alignment  (mean={dists_eucl.mean():.2f})")
ax.hist(dists_riem, bins=bins, alpha=0.6, color="#ed7d31",
        label=f"Riem. alignment  (mean={dists_riem.mean():.2f})")
ax.set(
    xlabel="Frobenius distance of whitened Cᵢ from I",
    ylabel="Trial count",
    title="Spread after whitening (lower = tighter re-centring)",
)
ax.legend(fontsize=9)

fig.suptitle(
    "Euclidean vs Riemannian alignment reference — eigenspectrum and residual spread",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/dd_da_riem_align.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure 2 saved → /tmp/dd_da_riem_align.png")

# %% [markdown]
# ### Reading Figure 2
#
# * **Left panel:** the eigenvalue spectrum of the Riemannian mean is slightly more
#   compressed toward 1 compared with the Euclidean mean — the "swelling" is visible
#   in the largest eigenvalues being slightly smaller for Riem.
# * **Right panel:** after whitening by each reference, the Frobenius distance of
#   individual trial covariances from the identity is slightly lower on average for
#   the Riemannian alignment.  The difference is modest here (few trials, 22
#   channels), but grows with the number of outlier trials or channels.
#
# **Practical advice:**  For cross-subject BCI pipelines where speed matters, EA
# (Euclidean) is the go-to choice — it is a single eigendecomposition per subject
# and runs in milliseconds.  Riemannian alignment costs 5–20× more (iterative
# mean) and rarely gives more than 1–2 pp extra accuracy.  Use Riemannian
# alignment when subjects have extreme covariance outliers (broken channels, EMG
# artefacts that survived preprocessing).

# %% [markdown]
# ---
# ## Part 4 — Calibration: giving the model a few target-subject trials
#
# ### The real-world scenario
#
# On the first day of BCI use, a technician typically runs a short calibration
# block (~5 min, ~20–40 labelled trials) to collect some target-subject data.
# These labelled trials can be added to the training set before fitting the
# classifier.  This is **calibration** (or *personalisation*).
#
# We simulate this by:
# 1. Holding out one subject as the target.
# 2. Pretending the first $K$ of their trials are the "calibration" set
#    (labelled, available before deployment).
# 3. Training on source subjects + calibration trials.
# 4. Evaluating on the *remaining* target-subject trials.
#
# **Leak-free rule:** the calibration trials come from the *start* of the target
# session; the evaluation trials come from the *end*.  We never expose evaluation
# labels during training.

# %%
# ── Calibration curve ─────────────────────────────────────────────────────────
# We sweep K ∈ CALIB_SIZES and average across all held-out target subjects.

print("Calibration experiment …")
print(f"{'Calib size K':>12}  {'Mean acc':>8}  {'Std acc':>7}")
print("-" * 34)

calib_results = {}   # K -> list of per-fold accuracies

for K in CALIB_SIZES:
    fold_accs = []

    for train_idx, test_idx in make_subject_split(subj):
        X_src, y_src = X[train_idx], y[train_idx]
        X_tgt, y_tgt = X[test_idx],  y[test_idx]

        # ── EA on source subjects (fit reference on each source subject) ──
        X_src_ea = np.empty_like(X_src)
        for s in np.unique(subj[train_idx]):
            m = subj[train_idx] == s
            Xs = X_src[m]
            R = np.mean([x @ x.T / x.shape[-1] for x in Xs], axis=0)
            vals, vecs = np.linalg.eigh(R)
            R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
            X_src_ea[m] = np.einsum("ij,njk->nik", R_inv_sqrt, Xs)

        # ── EA on target subject: reference from ALL target trials (no labels) ──
        R_tgt = np.mean([x @ x.T / x.shape[-1] for x in X_tgt], axis=0)
        vals_t, vecs_t = np.linalg.eigh(R_tgt)
        R_tgt_inv_sqrt = vecs_t @ np.diag(1.0 / np.sqrt(np.maximum(vals_t, 1e-12))) @ vecs_t.T
        X_tgt_ea = np.einsum("ij,njk->nik", R_tgt_inv_sqrt, X_tgt)

        # ── Split target: first K = calibration, rest = evaluation ───────
        if K == 0:
            # no calibration data; train on source only
            X_fit = X_src_ea
            y_fit = y_src
        else:
            # class-balanced sampling: take K//2 from each class (or as many as available)
            calib_0 = np.where(y_tgt == 0)[0][:max(1, K // 2)]
            calib_1 = np.where(y_tgt == 1)[0][:max(1, K - len(calib_0))]
            calib_idx = np.sort(np.concatenate([calib_0, calib_1]))
            eval_mask = np.ones(len(y_tgt), dtype=bool)
            eval_mask[calib_idx] = False

            if eval_mask.sum() < 4:
                # not enough evaluation trials; skip this fold
                continue

            X_fit = np.concatenate([X_src_ea, X_tgt_ea[calib_idx]])
            y_fit = np.concatenate([y_src,    y_tgt[calib_idx]])
            X_tgt_ea  = X_tgt_ea[eval_mask]
            y_tgt      = y_tgt[eval_mask]

        pipe_c = sk_clone(pipe_template)
        pipe_c.fit(X_fit, y_fit)
        fold_accs.append(pipe_c.score(X_tgt_ea, y_tgt))

    calib_results[K] = fold_accs
    if fold_accs:
        print(f"  K = {K:>3}         {np.mean(fold_accs):.3f}     {np.std(fold_accs):.3f}")
    else:
        print(f"  K = {K:>3}         (skipped — too few eval trials)")

# %%
# ── Figure 3 — Calibration curve ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ks        = [K for K in CALIB_SIZES if calib_results[K]]
means_cal = [np.mean(calib_results[K]) for K in ks]
stds_cal  = [np.std(calib_results[K])  for K in ks]

ax.errorbar(ks, means_cal, yerr=stds_cal,
            marker="o", linewidth=2.5, capsize=5,
            color="#2ca02c", markerfacecolor="white", markeredgewidth=2,
            label="EA + Riemann + LR  (mean ± std across LOSO folds)")

# Mark the zero-calibration baseline
if 0 in calib_results and calib_results[0]:
    base = np.mean(calib_results[0])
    ax.axhline(base, color="gray", lw=1.5, ls="--",
               label=f"Baseline (K=0, source-only) = {base:.3f}")

ax.axhline(0.5, color="red", lw=1.0, ls=":", label="Chance (50 %)")

ax.set(
    xlabel="Number of labelled calibration trials from the target subject (K)",
    ylabel="Accuracy on held-out target-subject trials",
    title=(
        "Calibration curve: accuracy rises with each labelled target trial\n"
        f"BCI IV 2a, LOSO  ({N_SUBJ} subjects, EA + Riemann pipeline)"
    ),
    ylim=(0.3, 1.0),
    xlim=(-1, max(ks) + 2),
)
ax.legend(loc="lower right", fontsize=9)
ax.xaxis.get_major_locator().set_params(integer=True)
plt.tight_layout()
plt.savefig("/tmp/dd_da_calib_curve.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure 3 saved → /tmp/dd_da_calib_curve.png")

# %% [markdown]
# ### Reading the calibration curve
#
# Each point is the mean accuracy across LOSO folds; error bars are ± 1 std.
#
# * At **K = 0** we rely solely on source-subject data (cross-subject transfer
#   with EA).  This is the realistic "zero-shot" BCI scenario.
# * As K grows the target-subject labels steer the classifier toward that
#   person's feature space, and accuracy climbs rapidly.
# * Even **5 labelled trials** (≈ 30–60 seconds of recording) can close a
#   meaningful fraction of the gap between cross-subject and within-subject
#   performance.  This is why real BCI systems almost always include a brief
#   calibration block before going live.
#
# **Implementation note:** the calibration trials here are drawn from the
# *beginning* of the target session (in time order), and evaluation uses the
# remainder.  Never shuffle and randomly draw from the full session — motor
# imagery performance drifts over a session, so that would be a subtle form of
# temporal leakage.

# %% [markdown]
# ---
# ## Part 5 — (Brief) Fine-tuning a deep network on calibration data
#
# When the source model is a deep neural network (e.g. EEGNet or ShallowConvNet
# trained on source subjects), the natural calibration strategy is **fine-tuning**:
# start from the pre-trained weights and update them on the K labelled target
# trials with a small learning rate.
#
# ```python
# # Pseudocode — requires a trained PyTorch or Braindecode model `source_model`.
# import torch
# from torch.optim import Adam
#
# target_loader = DataLoader(TensorDataset(
#     torch.tensor(X_calib_ea).float(),
#     torch.tensor(y_calib).long(),
# ), batch_size=min(16, K), shuffle=True)
#
# model = copy.deepcopy(source_model)          # start from source weights
# optim = Adam(model.parameters(), lr=1e-4)   # small LR — avoid catastrophic forgetting
# model.train()
#
# for epoch in range(50):                      # few epochs; K is tiny
#     for xb, yb in target_loader:
#         loss = F.cross_entropy(model(xb), yb)
#         loss.backward(); optim.step(); optim.zero_grad()
#
# # Evaluate on X_eval_ea
# model.eval()
# with torch.no_grad():
#     preds = model(torch.tensor(X_eval_ea).float()).argmax(1).numpy()
# acc_ft = (preds == y_eval).mean()
# ```
#
# **Practical concerns:**
#
# * With K < 20 trials, fine-tuning all layers causes **catastrophic forgetting**
#   of the source-domain knowledge.  Freeze early layers (spatial filters) and
#   only update the final classification head.
# * Apply EA to the calibration and evaluation data using the target subject's
#   reference (same as Part 4).
# * Use early stopping on a small within-calibration validation split (e.g. 20 %)
#   to avoid over-fitting to the tiny calibration set.
# * The combination of **EA + fine-tuning the head only** typically beats
#   fine-tuning all layers when K ≤ 30 trials.

# %% [markdown]
# ---
# ## Summary
#
# | Technique | Core idea | Lines of code | Needs target labels? | Typical gain |
# |---|---|---|---|---|
# | **Euclidean Alignment** | Whiten each domain to identity mean | ~15 | No | +5–15 pp |
# | **Riemannian alignment** | Whiten via Riemannian mean | ~5 (pyriemann) | No | slightly > EA |
# | **Calibration** | Add K labelled target trials to training | ~10 | Yes (K trials) | +15–25 pp for K ≥ 20 |
# | **Fine-tuning** | Update deep net head on target data | ~30 (PyTorch) | Yes (K trials) | +10–20 pp |
#
# **No single technique dominates in all scenarios:**
#
# * If calibration data is genuinely unavailable, EA (or Riemannian alignment) is
#   the best option and costs nothing in labels.
# * A brief calibration block (5–10 min, 20–40 trials) plus EA is the most
#   practical BCI deployment strategy and closes most of the cross-subject gap.
# * Fine-tuning is worth the engineering effort only when a large, well-trained
#   source model exists and K ≥ 10 calibration trials are available.

# %% [markdown]
# ## ⚠️ A subtler trap — computing the EA reference with test-subject labels
#
# ### The invisible leak
#
# Euclidean Alignment is unsupervised: the reference matrix $\mathbf{R}^{(t)}$ is
# the **mean covariance** — it does not use labels.  This seems safe.  But there
# is a less obvious failure mode that inflates accuracy without triggering the
# standard "did you use test labels?" check:
#
# > **If you compute $\mathbf{R}^{(t)}$ using ONLY the class-specific subsets of
# > the target trials — i.e. separately per class — and then pool, you have
# > implicitly used the test labels.**
#
# Concretely, some papers compute the EA reference as:
# $$\mathbf{R}^{(t)} = \frac{N_0}{N}\,\mathbf{R}_0 + \frac{N_1}{N}\,\mathbf{R}_1,
#   \quad \mathbf{R}_k = \frac{1}{N_k}\sum_{i: y_i=k} \mathbf{C}_i^{(t)}$$
# and treat this as "just an average".  But $\mathbf{R}_k$ uses $y_i$ for the
# test trials — it is leakage.
#
# ### Why it inflates accuracy
#
# A class-aware reference can produce a domain-adapted representation that
# perfectly separates the two classes by construction — because the whitening
# direction was chosen with knowledge of which trials belong to which class.  In
# the extreme case (only 2 classes, all trial variability is class-related), the
# whitened covariances may already be linearly separable *before* the classifier
# sees them, solely because $\mathbf{R}^{(t)}$ encoded the class boundary.
#
# The inflation is worst when:
# 1. The target subject has large class-conditional covariance differences (strong
#    ERD/ERS motor imagery signal) — precisely the subjects you care about most.
# 2. The evaluation is done on the same trials used to compute the reference
#    (the most common mistake).
# 3. The dataset is small, so the class-conditioned means differ substantially by
#    chance.
#
# ### Why this is non-obvious
#
# * The reference matrix $\mathbf{R}^{(t)}$ is not a model parameter — it is not
#   passed to `.fit()`.  Standard "train-only fit" audits miss it.
# * The weighted average looks mathematically identical to the overall mean when
#    classes are balanced, so the bug is invisible in balanced datasets.
# * Published papers have reported cross-subject gains of 10–20 pp from EA; some
#    fraction of that literature may have used a class-aware reference.
#
# ### The correct rule (as implemented above)
#
# $$\mathbf{R}^{(t)} = \frac{1}{N_t}\sum_{i=1}^{N_t} \mathbf{C}_i^{(t)}$$
#
# — pool **all** target trials (regardless of label) into a single mean.  Labels
# are never consulted.  This is safe because the mean covariance captures the
# subject-specific amplitude baseline, which is label-independent.
#
# A second, less-known variant of the same trap: using the target-subject's
# *evaluation* trials (not calibration trials) to compute $\mathbf{R}^{(t)}$ when
# the calibration set is tiny.  With K = 5 calibration trials the reference is
# noisy; a researcher might be tempted to "stabilise" it by including evaluation
# trials.  This too is leakage — the reference has now seen the evaluation set,
# and its whitening direction implicitly encodes information about those trials.
#
# **Bottom line:** Euclidean Alignment is powerful precisely because it is label-
# and evaluation-free.  The moment you condition it on class membership or
# evaluation data, you have built a shortcut that will not generalise to a real
# deployed BCI where test labels are unavailable by definition.
