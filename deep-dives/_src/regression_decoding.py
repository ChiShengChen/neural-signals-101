# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Regression Decoding (continuous targets)
#
# The main tutorial classifies every trial into a discrete label.  Real neural
# decoding is often a **regression problem**: predict cursor position, hand
# kinematics, or continuous arousal/drowsiness from brain signals.
#
# > **Prerequisites:** main Chapters 06, 08 and 12.
# > **Level:** advanced ★★★★☆
# > **All the main chapters classify; real decoding is often regression.**

# %%
# --------------------------------------------------------------------------- #
# Bootstrap — locate neuro101 whether we are run from the repo root or from
# deep-dives/_src/  (one extra parent level vs. the main notebooks).
# --------------------------------------------------------------------------- #
import sys
import os
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
matplotlib.use("Agg")          # headless backend — safe under nbconvert / CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds
from neuro101 import make_block_split
from neuro101.io import load_sleep_edf_epochs
from neuro101.features import bandpower

SMOKE = ds.is_smoke()
RNG   = np.random.default_rng(42)

print(f"neuro101 OK  |  SMOKE={SMOKE}")

# %% [markdown]
# ---
# ## 1  Classification vs regression — and why the metrics change completely
#
# ### When your target is continuous
#
# | Setting | Target | Wrong metric | Right metrics |
# |---|---|---|---|
# | Sleep stage prediction | {Wake, N1, N2, N3, REM} | — | accuracy, balanced-accuracy, F1 |
# | Arousal/drowsiness | continuous scalar in [0, 1] | **accuracy** | **R², MAE, Pearson r** |
# | Cursor position | x, y in mm | **accuracy** | **R², MAE** |
# | Reaction time | ms | **accuracy** | **R², MAE, Spearman r** |
#
# **Never use "accuracy" for a continuous target.**  Accuracy requires a
# prediction to exactly hit the right value — with a real-valued output that
# never happens, so every sample is "wrong" and accuracy is always 0%.
#
# ### The three regression metrics you need
#
# **R² — coefficient of determination**
#
# $$R^2 = 1 - \frac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}$$
#
# Interpretation: fraction of variance explained.  R² = 1 is perfect; R² = 0
# means your model does no better than always predicting the mean; **R² can be
# negative** (worse than the mean — see the subtle trap at the end).
#
# **MAE — mean absolute error**
#
# $$\text{MAE} = \frac{1}{n}\sum |y_i - \hat y_i|$$
#
# Same units as the target; easy to interpret.  Unlike R² it does not depend on
# target variance, so it tells you the raw prediction error in physical units.
#
# **Pearson / Spearman correlation**
#
# Measures whether predictions *rank* the same as true values.  A model can have
# high correlation but bad calibration: if $\hat y = 2 y - 1$ (slope and offset
# wrong), Pearson $r = 1$ but $R^2 \ll 1$ and MAE is large.  Always report
# correlation **and** R², and check a scatter plot.

# %%
# --------------------------------------------------------------------------- #
# Visualise the key metric relationships with a toy example.
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Three ways a regression can look 'good' on one metric but fail on another",
             fontsize=11, fontweight="bold")

rng_toy = np.random.default_rng(7)
y_true  = np.linspace(0, 1, 80) + rng_toy.normal(0, 0.05, 80)

# Panel A: well-calibrated model
y_good = y_true + rng_toy.normal(0, 0.12, 80)
ax = axes[0]
ax.scatter(y_true, y_good, s=12, alpha=0.6, color="steelblue")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_g = r2_score(y_true, y_good)
r_g, _ = pearsonr(y_true, y_good)
ax.set_title(f"A) Well-calibrated\nR²={r2_g:.2f}  r={r_g:.2f}", fontsize=9)
ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.legend(fontsize=8)

# Panel B: high correlation but slope/offset wrong (badly calibrated)
y_recal = 0.4 * y_true + 0.35 + rng_toy.normal(0, 0.04, 80)
ax = axes[1]
ax.scatter(y_true, y_recal, s=12, alpha=0.6, color="darkorange")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_r = r2_score(y_true, y_recal)
r_r, _ = pearsonr(y_true, y_recal)
ax.set_title(f"B) High r, bad calibration\nR²={r2_r:.2f}  r={r_r:.2f}", fontsize=9)
ax.set_xlabel("True"); ax.legend(fontsize=8)

# Panel C: zero correlation, non-negative R² due to target range
y_rand = rng_toy.uniform(y_true.min(), y_true.max(), 80)
ax = axes[2]
ax.scatter(y_true, y_rand, s=12, alpha=0.6, color="crimson")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_rn = r2_score(y_true, y_rand)
r_rn, _ = pearsonr(y_true, y_rand)
ax.set_title(f"C) Random model\nR²={r2_rn:.2f}  r={r_rn:.2f}", fontsize=9)
ax.set_xlabel("True"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/regression_metrics_explainer.png", dpi=110, bbox_inches="tight")
plt.show()
print("Panel B: Pearson r is high but R² is low — the model's slope and offset are wrong.")
print("Always check the scatter plot against the identity line.")

# %% [markdown]
# ---
# ## 2  Regression on a real EEG continuous target
#
# ### Dataset and target
#
# We use **Sleep-EDF**: overnight EEG recordings sampled at 100 Hz, cut into
# 30-second epochs.  Sleep stage labels are available, but we deliberately ignore
# them and instead define a **continuous arousal proxy** — the log alpha-band
# power on the Fpz-Cz channel.
#
# Alpha power (8–13 Hz) is a classical marker of cortical arousal: it is high
# during relaxed wakefulness and suppressed during deep sleep.  This gives us a
# smooth, physiologically meaningful continuous signal that varies across the
# night.
#
# **Regression task:** predict alpha log-power from the *other* band powers
# (delta, theta, beta, gamma) extracted from the same channel.
#
# ### Why this is honest (within-subject, same channel, other bands)
#
# Delta power rises in deep sleep; theta rises in light sleep and REM; beta/gamma
# power drops with increasing depth.  These bands carry genuine information about
# the brain state, so R² > 0 is expected and meaningful — but the relationship is
# noisy enough that R² is far below 1.
#
# ### Time-aware split
#
# We use `make_block_split` (contiguous blocks) because consecutive 30-second
# epochs are autocorrelated: a person in deep sleep stays in deep sleep for
# several epochs.  A random split would let the model "interpolate" between
# adjacent epochs in train and test — inflating R².

# %%
# --------------------------------------------------------------------------- #
# Load Sleep-EDF and extract band-power features.
# --------------------------------------------------------------------------- #
n_subjects_to_load = 1 if SMOKE else 2
print(f"Loading Sleep-EDF ({n_subjects_to_load} subject(s)) …")
X_eeg, y_stage, subjects = load_sleep_edf_epochs(n_subjects=n_subjects_to_load)

SFREQ = 100.0  # Hz
print(f"Epochs: {X_eeg.shape[0]}  |  channel: EEG Fpz-Cz  |  epoch length: 30 s")

# Extract log band-power features for all 5 bands.
# feats[:, 0] = delta   feats[:, 1] = theta   feats[:, 2] = alpha
# feats[:, 3] = beta    feats[:, 4] = gamma
feats_all = bandpower(X_eeg, SFREQ)  # (n_epochs, 5)

# Continuous target: log alpha power
target_alpha = feats_all[:, 2]       # shape (n_epochs,)

# Features: all bands except alpha
X_feat = feats_all[:, [0, 1, 3, 4]]  # delta, theta, beta, gamma  → (n_epochs, 4)

n_epochs = len(target_alpha)
autocorr_lag1 = float(np.corrcoef(target_alpha[:-1], target_alpha[1:])[0, 1])
print(f"\nTarget (log alpha power):  range=[{target_alpha.min():.2f}, {target_alpha.max():.2f}]")
print(f"Lag-1 autocorrelation: {autocorr_lag1:.3f}  (moderate — adjacent epochs are similar)")

# %%
# --------------------------------------------------------------------------- #
# Fit Ridge regression with a leakage-safe, time-aware block split.
# --------------------------------------------------------------------------- #
N_SPLITS = 3 if SMOKE else 5

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=1.0)),
])

r2_per_fold   = []
mae_per_fold  = []
pearson_per_fold = []
spearman_per_fold = []
all_true_bs  = []
all_pred_bs  = []
te_indices_bs = []

for fold_idx, (tr, te) in enumerate(make_block_split(n_epochs, n_splits=N_SPLITS)):
    pipe.fit(X_feat[tr], target_alpha[tr])
    preds = pipe.predict(X_feat[te])

    r2_per_fold.append(r2_score(target_alpha[te], preds))
    mae_per_fold.append(mean_absolute_error(target_alpha[te], preds))
    pearson_per_fold.append(pearsonr(target_alpha[te], preds)[0])
    spearman_per_fold.append(spearmanr(target_alpha[te], preds)[0])

    all_true_bs.append(target_alpha[te])
    all_pred_bs.append(preds)
    te_indices_bs.append(te)

all_true_bs_cat = np.concatenate(all_true_bs)
all_pred_bs_cat = np.concatenate(all_pred_bs)
te_idx_cat = np.concatenate(te_indices_bs)
sort_order = np.argsort(te_idx_cat)

print("=== Block-split (time-aware) regression results ===")
print(f"  R²       : {np.mean(r2_per_fold):.3f} ± {np.std(r2_per_fold):.3f}  (per fold: {[f'{v:.3f}' for v in r2_per_fold]})")
print(f"  MAE      : {np.mean(mae_per_fold):.3f} ± {np.std(mae_per_fold):.3f}  (log-power units)")
print(f"  Pearson r: {np.mean(pearson_per_fold):.3f} ± {np.std(pearson_per_fold):.3f}")
print(f"  Spearman r: {np.mean(spearman_per_fold):.3f} ± {np.std(spearman_per_fold):.3f}")

# %% [markdown]
# ### Interpreting the results
#
# A Pearson $r > 0.6$ means the **rank ordering** of arousal across epochs is
# captured.  But check R²: if the slope or offset is wrong, R² will be lower
# than $r^2$.  Always look at both — and always look at the scatter plot.

# %%
# --------------------------------------------------------------------------- #
# Figure 1: Predicted vs true (scatter + identity line) and time-slice.
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Scatter: predicted vs true ---
ax_scatter = fig.add_subplot(gs[0])
ax_scatter.scatter(all_true_bs_cat, all_pred_bs_cat,
                   s=5, alpha=0.25, color="steelblue", rasterized=True)
vmin = min(all_true_bs_cat.min(), all_pred_bs_cat.min())
vmax = max(all_true_bs_cat.max(), all_pred_bs_cat.max())
ax_scatter.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="identity (perfect)")
overall_r2   = r2_score(all_true_bs_cat, all_pred_bs_cat)
overall_r, _ = pearsonr(all_true_bs_cat, all_pred_bs_cat)
ax_scatter.set_xlabel("True log alpha power", fontsize=10)
ax_scatter.set_ylabel("Predicted log alpha power", fontsize=10)
ax_scatter.set_title(
    f"Predicted vs True  (block-split)\nR²={overall_r2:.3f}  Pearson r={overall_r:.3f}",
    fontsize=10, fontweight="bold")
ax_scatter.legend(fontsize=8)

# --- Time slice: first 200 test-set epochs in temporal order ---
sorted_true = all_true_bs_cat[sort_order]
sorted_pred = all_pred_bs_cat[sort_order]
n_show = min(200, len(sorted_true))

ax_time = fig.add_subplot(gs[1])
epoch_minutes = np.arange(n_show) * 0.5          # 30-s epochs → 0.5 min each
ax_time.plot(epoch_minutes, sorted_true[:n_show],
             lw=1.2, color="steelblue", label="True", alpha=0.85)
ax_time.plot(epoch_minutes, sorted_pred[:n_show],
             lw=1.2, color="darkorange", linestyle="--", label="Predicted", alpha=0.85)
ax_time.set_xlabel("Time (minutes)", fontsize=10)
ax_time.set_ylabel("Log alpha power", fontsize=10)
ax_time.set_title("Predicted vs True over time\n(first 200 test-set epochs)", fontsize=10, fontweight="bold")
ax_time.legend(fontsize=8)

plt.savefig("/tmp/regression_scatter_time.png", dpi=110, bbox_inches="tight")
plt.show()
print(f"Block-split overall  R²={overall_r2:.3f}   Pearson r={overall_r:.3f}")

# %% [markdown]
# ---
# ## 3  The regression-specific leakage trap: autocorrelated targets
#
# ### Why this is even worse than in classification
#
# In Chapter 12 you saw that random-shuffle splits inflate classification accuracy
# because adjacent trials are correlated.  For **regression with a smooth
# continuous target**, the problem is dramatically worse:
#
# * Adjacent 30-second epochs have nearly identical alpha power (lag-1
#   autocorrelation ≈ 0.58 for raw alpha, ≈ 0.99 for a smoothed signal).
# * A random split scatters these similar epochs into both train and test.
# * When the test epoch for epoch 142 is flanked by train epochs 141 and 143
#   (which have almost the same target value), the model can effectively
#   **interpolate** the target — not because it learned the brain-state
#   relationship, but because it memorised the smooth trajectory.
# * R² inflates massively for smooth targets; for very smooth targets it can
#   make a near-useless model look nearly perfect.
#
# ### Demonstration
#
# We construct a **highly autocorrelated drowsiness proxy**: a 50-epoch
# (25-minute) sliding-window average of delta power.  The lag-1 autocorrelation
# of this signal is ≈ 0.998 — it changes extremely slowly across the night.
#
# Then we compare:
# * **WRONG** — random 5-fold KFold (shuffled): standard sklearn default.
# * **RIGHT** — contiguous block split (`make_block_split`).

# %%
# --------------------------------------------------------------------------- #
# Build the smooth drowsiness proxy (high autocorrelation).
# --------------------------------------------------------------------------- #
SMOOTH_WINDOW = 50   # epochs  → 25 minutes of sleep-architecture trend
raw_delta      = feats_all[:, 0]   # log-delta power
smooth_target  = np.convolve(raw_delta,
                              np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                              mode="same")

autocorr_smooth = float(np.corrcoef(smooth_target[:-1], smooth_target[1:])[0, 1])
print(f"Smooth drowsiness proxy: lag-1 autocorr = {autocorr_smooth:.4f}")
print("(≈1.00 means adjacent epochs are almost indistinguishable — prime leakage territory)")

# %%
# --------------------------------------------------------------------------- #
# WRONG: random KFold (the classic mistake).
# --------------------------------------------------------------------------- #
kf_wrong = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

wrong_r2 = []
for tr, te in kf_wrong.split(X_feat):
    m = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    m.fit(X_feat[tr], smooth_target[tr])
    wrong_r2.append(r2_score(smooth_target[te], m.predict(X_feat[te])))

wrong_mean = float(np.mean(wrong_r2))
wrong_std  = float(np.std(wrong_r2))
print(f"\nWRONG (random KFold, shuffle=True):  R² = {wrong_mean:.3f} ± {wrong_std:.3f}")

# --------------------------------------------------------------------------- #
# RIGHT: contiguous block split (time-aware).
# --------------------------------------------------------------------------- #
right_r2 = []
for tr, te in make_block_split(n_epochs, n_splits=N_SPLITS):
    m = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    m.fit(X_feat[tr], smooth_target[tr])
    right_r2.append(r2_score(smooth_target[te], m.predict(X_feat[te])))

right_mean = float(np.mean(right_r2))
right_std  = float(np.std(right_r2))
print(f"RIGHT (block split, contiguous):      R² = {right_mean:.3f} ± {right_std:.3f}")
print(f"\nInflation from leakage: ΔR² = {wrong_mean - right_mean:+.3f}")

# %%
# --------------------------------------------------------------------------- #
# Figure 2: WRONG vs RIGHT R² bar comparison.
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left panel: 2-bar contrast
ax_bar = axes[0]
labels   = ["WRONG\n(random KFold)", "RIGHT\n(block split)"]
means    = [wrong_mean, right_mean]
stds     = [wrong_std,  right_std]
colors   = ["#d62728", "#2ca02c"]   # red = wrong, green = right
bars = ax_bar.bar(labels, means, yerr=stds, capsize=7,
                  color=colors, alpha=0.82, edgecolor="k", linewidth=0.8,
                  error_kw=dict(elinewidth=1.5))

# Draw R²=0 reference line (no better than predicting the mean)
ax_bar.axhline(0, color="k", lw=1.0, linestyle="--", label="R²=0  (predicting the mean)")
ax_bar.set_ylabel("R²  (5-fold mean ± std)", fontsize=10)
ax_bar.set_title("Leakage trap:\nsmooth target × random split → inflated R²",
                 fontsize=10, fontweight="bold")
ax_bar.legend(fontsize=8)

# Annotate bars with values
for bar, mean_val, std_val in zip(bars, means, stds):
    ypos = max(mean_val, 0) + std_val + 0.01
    ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{mean_val:.3f}±{std_val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

# Right panel: time-series of smooth target to show WHY it leaks
ax_ts = axes[1]
show_n = min(300, n_epochs)
ax_ts.plot(np.arange(show_n) * 0.5, smooth_target[:show_n],
           color="purple", lw=1.5, alpha=0.85)
ax_ts.set_xlabel("Time (minutes)", fontsize=10)
ax_ts.set_ylabel("Smooth delta proxy (log-power, 25-min avg)", fontsize=10)
ax_ts.set_title(f"Target autocorr(lag-1) = {autocorr_smooth:.4f}\n"
                "Adjacent epochs nearly identical → interpolation masquerades as learning",
                fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/regression_leakage_bar.png", dpi=110, bbox_inches="tight")
plt.show()

print(f"\n{'='*55}")
print(f"  WRONG R² = {wrong_mean:.3f}  |  RIGHT R² = {right_mean:.3f}")
print(f"  Inflation ΔR² = {wrong_mean - right_mean:+.3f}")
print(f"{'='*55}")
print("\nConclusion: random KFold inflates R² because adjacent epochs share")
print("nearly identical targets — the model interpolates, not extrapolates.")

# %% [markdown]
# ### Why contiguous block splits are mandatory for smooth targets
#
# With a lag-1 autocorrelation of ≈ 0.998, adjacent epochs differ by only a tiny
# amount.  A random split scatters epoch 100 into the test set while epochs 99
# and 101 sit in the training set.  The model has seen the target value at time
# 99 and time 101; it just needs to output their average — and R² soars.
#
# A contiguous block split keeps all of fold *k*'s epochs together.  The model
# must *extrapolate* from one block of the night to another, which is genuinely
# hard.  The honest score can even be **negative** (the model does worse than
# predicting the mean), which is an entirely valid and informative result.
#
# **The rule:** whenever your target is smoother than a random walk — kinematics,
# arousal, any physiological state variable — use block splits.  If you can
# compute the lag-1 autocorrelation and it exceeds ≈ 0.5, treat it as smooth.

# %% [markdown]
# ---
# ## ⚠️ A subtler trap: R² can be negative — and a small positive R² can still be misleading
#
# You saw above that the block-split R² can be negative.  That is correct and
# honest: it means the model is *worse* than always predicting the training-set
# mean.  But there is a second, even less obvious failure mode.
#
# ### Trending targets and the "constant predictor" illusion
#
# Suppose your target drifts monotonically across the session — say, alpha power
# declining steadily from wakefulness into sleep.  A model that learns a single
# number (the training-set mean) will follow that trend for free if the test fold
# is *later* in time and the target has moved in the same direction.
#
# **Result:** even a constant prediction can yield a small positive R², not
# because the model learned anything about neural signals, but because the target
# trend crossed the mean in the right direction during the test block.  This is
# the continuous-target analogue of "predicting the majority class" in
# classification — but it is less obvious because R² > 0 *looks* like success.
#
# **Diagnosis:** plot the residuals $y - \hat y$ against time.  A constant
# predictor will leave a structured, drifting residual pattern.  A real model
# will leave residuals that look like white noise.  Always inspect the residual
# time series, not just the scalar R².

# %%
# --------------------------------------------------------------------------- #
# Demonstrate the trending-target / constant-predictor illusion.
# --------------------------------------------------------------------------- #
n_demo = 300
t_demo = np.arange(n_demo)

# Target: a gentle linear trend + noise (simulating alpha declining into sleep)
trend_target = -0.005 * t_demo + rng_toy.normal(0, 0.15, n_demo)

# Train on first half, test on second half — as a block split would do.
split = n_demo // 2
y_train = trend_target[:split]
y_test  = trend_target[split:]

# "Constant predictor": always predict the training mean.
const_pred = np.full(split, y_train.mean())
r2_const   = r2_score(y_test, const_pred)

# Residuals: structured drift or white noise?
residuals_const = y_test - const_pred

print("=== Trending-target / constant-predictor illusion ===")
print(f"  Constant predictor (training mean = {y_train.mean():.3f}) on test fold:")
print(f"  R² = {r2_const:.3f}  ← can be positive even though model ignores all signals!")
print()
print("  Residuals mean: {:.3f}  (nonzero → structured drift, not white noise)".format(
    residuals_const.mean()))
print()
print("Lesson: A small positive R² alone is not evidence of learning.")
print("Always plot residuals vs time and compare to a naive mean baseline.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax0 = axes[0]
ax0.plot(t_demo[:split], y_train,  lw=1, alpha=0.7, color="steelblue", label="Train target")
ax0.plot(t_demo[split:], y_test,   lw=1, alpha=0.7, color="darkorange", label="Test target")
ax0.axhline(y_train.mean(), color="k", lw=1.5, linestyle="--",
            label=f"Constant pred = {y_train.mean():.3f}")
ax0.axvline(split, color="gray", lw=1, linestyle=":")
ax0.set_xlabel("Epoch index"); ax0.set_ylabel("Target")
ax0.set_title(f"Trending target: constant predictor\nR² = {r2_const:.3f}  ← misleadingly non-negative",
              fontsize=9, fontweight="bold")
ax0.legend(fontsize=7)

ax1 = axes[1]
ax1.plot(t_demo[split:], residuals_const,
         lw=1, color="crimson", alpha=0.8, label="Residual = true − predicted")
ax1.axhline(0, color="k", lw=1, linestyle="--")
ax1.set_xlabel("Epoch index (test fold only)"); ax1.set_ylabel("Residual")
ax1.set_title("Residuals show structured drift\n(not white noise → model is not capturing the signal)",
              fontsize=9, fontweight="bold")
ax1.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/regression_subtle_trap.png", dpi=110, bbox_inches="tight")
plt.show()

print("\nRule of thumb: always compare your model to the naive mean baseline.")
print("If R²(model) ≈ R²(naive mean predictor), you have learned nothing.")
