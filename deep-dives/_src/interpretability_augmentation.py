# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Interpreting EEG Nets & Data Augmentation
#
# What does EEGNet actually look at, and how can we squeeze more signal
# out of tiny labelled datasets via carefully designed augmentation?
#
# > **Prerequisites:** main Chapter 09.
# > **Level:** advanced ★★★★☆
# > **Not bound by the 5-min CPU budget.**

# %% Bootstrap — import neuro101 from repo src if not installed as package
import sys
import os
from pathlib import Path

# Robust upward search: try installed, then walk up looking for src/neuro101
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _candidate in [_here, *_here.parents]:
        _src = _candidate / "src"
        if (_src / "neuro101" / "__init__.py").exists():
            sys.path.insert(0, str(_src))
            break
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from neuro101 import io, datasets as ds

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)

DEVICE = "cpu"
SMOKE = ds.is_smoke()
print(f"Smoke mode : {SMOKE}  |  torch {torch.__version__}  |  device={DEVICE}")

# %% [markdown]
# ---
# ## Part A — Interpretability: what is EEGNet actually using?
#
# EEGNet (Lawhern et al. 2018) is a compact convolutional architecture designed
# for EEG classification. Its three blocks — temporal convolution, depthwise
# spatial convolution, and a separable convolution — are small enough to train
# on CPU in minutes. But "it classifies" is not the same as "it classifies for
# the right reasons." We will train a tiny EEGNet on BCI IV 2a (left vs right
# motor imagery) and then apply **gradient-based saliency** to ask: which
# electrodes and which moments in time drive the model's confidence?

# %% [markdown]
# ### Step 1 — Load data (1–2 subjects, left vs right MI)

# %%
N_SUBJ = 1 if SMOKE else 2
print(f"Loading {N_SUBJ} subject(s) from BCI IV 2a …")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=N_SUBJ)
print(f"  X={X_all.shape}  classes={np.bincount(y_all)}  dtype={X_all.dtype}")

# BCI IV 2a channel order (22 EEG channels after MOABB strips EOG/stim)
CH_NAMES = [
    "Fz",  "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",  "POz",
]
assert len(CH_NAMES) == X_all.shape[1], (
    f"Channel name mismatch: {len(CH_NAMES)} names but {X_all.shape[1]} channels"
)

N_CHANNELS = X_all.shape[1]   # 22
N_TIMES    = X_all.shape[2]   # 501
SFREQ      = 250.0             # Hz (BCI IV 2a)
print(f"  Channels={N_CHANNELS}  Times={N_TIMES}  sfreq={SFREQ} Hz")

# %% [markdown]
# ### Step 2 — Honest train / test split (standardise on train only)

# %%
# Use a single subject's data; stratified 70/30 split preserving class balance.
s_mask = (subj_all == subj_all.min())
X_s, y_s = X_all[s_mask], y_all[s_mask]

# Stratified split: for each class, take the first 70% as train, rest as test.
train_idx_list, test_idx_list = [], []
for c in np.unique(y_s):
    c_idx = np.where(y_s == c)[0]
    n_tr  = max(1, int(0.70 * len(c_idx)))
    train_idx_list.append(c_idx[:n_tr])
    test_idx_list.append(c_idx[n_tr:])

train_idx = np.sort(np.concatenate(train_idx_list))
test_idx  = np.sort(np.concatenate(test_idx_list))

X_tr_raw, y_tr = X_s[train_idx], y_s[train_idx]
X_te_raw, y_te = X_s[test_idx],  y_s[test_idx]

# Standardise: zero-mean, unit-variance per channel on TRAIN set only.
mu  = X_tr_raw.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
sig = X_tr_raw.std(axis=(0, 2), keepdims=True) + 1e-8
X_tr = (X_tr_raw - mu) / sig
X_te = (X_te_raw - mu) / sig   # apply train stats to test

print(f"Train: {X_tr.shape}  classes={np.bincount(y_tr)}")
print(f"Test : {X_te.shape}  classes={np.bincount(y_te)}")

# Convert to float32 PyTorch tensors; EEGNet expects (batch, 1, channels, times)
def to_tensor(X, y):
    # braindecode v0.8 expects (batch, channels, times) — no extra dim needed
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    return Xt, yt

Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
Xte_t, yte_t = to_tensor(X_te, y_te)
print(f"Tensor shape: {Xtr_t.shape}  (batch, channels, times)")

# %% [markdown]
# ### Step 3 — Build and train a tiny EEGNet
#
# We use braindecode's `EEGNetv4` with tiny hyper-parameters (`F1=4, D=2, F2=8`)
# so it trains in a few seconds on CPU.

# %%
from braindecode.models import EEGNetv4

def build_eegnet(n_chans: int, n_times: int, n_classes: int = 2) -> nn.Module:
    """Tiny EEGNetv4 that fits on CPU in under a minute.

    braindecode v0.8 EEGNetv4 expects input shape (batch, channels, times).
    """
    model = EEGNetv4(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        F1=4,
        D=2,
        F2=8,
        kernel_length=32,   # ~128 ms at 250 Hz
        drop_prob=0.25,
        final_conv_length="auto",
    )
    return model

model = build_eegnet(N_CHANNELS, N_TIMES)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"EEGNetv4: {n_params:,} parameters")

# %%
EPOCHS      = 5 if SMOKE else 20
BATCH_SIZE  = 32
LR          = 1e-3

train_ds  = TensorDataset(Xtr_t, ytr_t)
train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                       generator=torch.Generator().manual_seed(42))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model.train()
loss_history = []
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(train_ds)
    loss_history.append(epoch_loss)
    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    logits_te = model(Xte_t.to(DEVICE))
    preds_te  = logits_te.argmax(dim=1).cpu().numpy()
baseline_acc = (preds_te == y_te).mean()
print(f"\nBaseline test accuracy (no augmentation, full train set): {baseline_acc:.3f}")

# %% [markdown]
# ### Step 4 — Gradient-based saliency: |d(logit)/d(input)|
#
# The simplest interpretability method: for each test trial we compute the
# **absolute gradient of the winning class logit with respect to every input
# sample**. Large gradients → the model's confidence is sensitive to that
# (channel, time) location. We average across test trials to get a stable
# importance map.
#
# **Important caveat:** this tells us where the loss *surface* is steep, not
# necessarily what the network "decided" on. See the ⚠️ section at the end.

# %%
def compute_saliency(model: nn.Module, X_tensor: torch.Tensor) -> np.ndarray:
    """Return |d logit_pred / d input| averaged over trials.

    Returns
    -------
    saliency : np.ndarray  shape (n_channels, n_times)
        Mean absolute gradient, collapsed over the trial dimension.
    """
    model.eval()
    all_grads = []
    for i in range(len(X_tensor)):
        x = X_tensor[i:i+1].clone().requires_grad_(True).to(DEVICE)
        logits = model(x)
        # Gradient w.r.t. the argmax class (predicted class)
        pred_class = logits.argmax(dim=1).item()
        score = logits[0, pred_class]
        score.backward()
        grad = x.grad.detach().cpu().numpy()   # (1, C, T)  — braindecode v0.8
        all_grads.append(np.abs(grad[0]))      # (C, T)

    saliency = np.mean(all_grads, axis=0)      # (C, T)
    return saliency

print("Computing saliency on test set …")
saliency_map = compute_saliency(model, Xte_t)
print(f"Saliency map shape: {saliency_map.shape}  (channels, times)")

# Per-channel importance: average over time
ch_importance = saliency_map.mean(axis=1)   # (C,)
rank_order    = np.argsort(ch_importance)[::-1]

print("\nTop-10 channels by saliency:")
for r, ci in enumerate(rank_order[:10]):
    print(f"  {r+1:2d}. {CH_NAMES[ci]:<6s}  importance={ch_importance[ci]:.4f}")

# %% [markdown]
# ### Visualisation 1 — Per-channel saliency (bar chart)
#
# For genuine motor imagery the network should emphasise **central sensorimotor
# channels** (C3, Cz, C4 and their neighbours FC/CP) because ERD/ERS in the
# 8–30 Hz mu/beta band is strongest over the hand area (Brodmann area 4).
#
# If the model instead emphasises frontal channels (Fz, FC3/FC4) or peripheral
# channels (POz, Pz), it may be tracking **eye-movement artifacts or cue-related
# potentials** rather than true motor imagery — a known confound. Cross-reference
# with the artifact_confounds deep-dive if you see that pattern.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- left panel: sorted bar chart ---
ax = axes[0]
sorted_imp  = ch_importance[rank_order]
sorted_names = [CH_NAMES[i] for i in rank_order]
colors = []
for nm in sorted_names:
    if nm in ("C3", "Cz", "C4", "C1", "C2", "C5", "C6",
               "CP1", "CPz", "CP2", "CP3", "CP4",
               "FC1", "FCz", "FC2", "FC3", "FC4"):
        colors.append("#2ecc71")    # sensorimotor — expected
    elif nm in ("Fz", "FC5", "FC6"):
        colors.append("#e74c3c")    # frontal — suspect artifact
    else:
        colors.append("#3498db")    # other

bars = ax.barh(range(len(sorted_names)), sorted_imp[::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names[::-1], fontsize=7)
ax.set_xlabel("Mean |gradient|", fontsize=10)
ax.set_title("Per-channel gradient saliency\n(green = sensorimotor, red = frontal/artifact-prone)", fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#2ecc71", label="Sensorimotor (expected)"),
    Patch(facecolor="#e74c3c", label="Frontal (artifact-prone)"),
    Patch(facecolor="#3498db", label="Other"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

# --- right panel: 2-D heat map (channels × time) ---
ax2 = axes[1]
times_ms = np.linspace(500, 2500, N_TIMES)   # 0.5–2.5 s post-cue
im = ax2.imshow(
    saliency_map,
    aspect="auto",
    origin="upper",
    extent=[times_ms[0], times_ms[-1], N_CHANNELS - 0.5, -0.5],
    cmap="hot",
)
ax2.set_yticks(range(N_CHANNELS))
ax2.set_yticklabels(CH_NAMES, fontsize=6)
ax2.set_xlabel("Time post-cue (ms)", fontsize=10)
ax2.set_title("Saliency heat map (channels × time)\nhot = high importance", fontsize=9)
plt.colorbar(im, ax=ax2, label="Mean |gradient|")

fig.suptitle(
    f"EEGNet gradient saliency — BCI IV 2a (left vs right MI)\n"
    f"Test accuracy: {baseline_acc:.1%}   |   {N_SUBJ} subject(s)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/dd_int_saliency.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saliency figure saved.")

# %% [markdown]
# ### Step 5 — Per-time-sample importance
#
# Averaging saliency over channels gives us a temporal profile of model attention.
# We expect higher importance in the **sustained MI window** (roughly 0.5–2 s
# post-cue), not at the very beginning of the epoch (cue-locked visual response)
# or at the very end.

# %%
time_importance = saliency_map.mean(axis=0)   # (T,)
times_ms = np.linspace(500, 2500, N_TIMES)

fig, ax = plt.subplots(figsize=(9, 3))
ax.fill_between(times_ms, time_importance, alpha=0.4, color="#8e44ad")
ax.plot(times_ms, time_importance, color="#8e44ad", lw=1.5)
ax.set_xlabel("Time post-cue (ms)")
ax.set_ylabel("Mean |gradient| across channels")
ax.set_title("Temporal saliency profile — when does EEGNet pay attention?")
ax.axvspan(500, 2500, alpha=0.05, color="gray", label="MI epoch window")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_int_time_saliency.png", dpi=100, bbox_inches="tight")
plt.show()
print("Temporal saliency figure saved.")

# %% [markdown]
# ---
# ## Part B — Data augmentation for small EEG data
#
# BCI datasets are expensive to collect; a typical study yields 100–300 labelled
# trials per subject — a regime where deep networks overfit easily. Augmentation
# creates additional synthetic training examples by applying **physiologically
# plausible** perturbations to existing trials. The perturbations should be
# small enough not to change the class label, but diverse enough to act as a
# regulariser.
#
# ### The four augmentations we implement
#
# | # | Name | Operation | Rationale |
# |---|------|-----------|-----------|
# | a | **Time shift** | Roll signal by ±τ samples (random τ) | MI signal is not perfectly time-locked to cue |
# | b | **Gaussian noise** | Add i.i.d. N(0, σ²) to every sample | Models electrode noise, skin impedance drift |
# | c | **Channel/frequency masking** | Zero out a random channel OR a random 4-Hz band | Forces spatial and spectral robustness |
# | d | **Mixup** | Convex combo of two trials + soft labels | Regularises the output layer; originally from CV |

# %% [markdown]
# ### Step 6 — Implement augmentation functions

# %%
def aug_time_shift(X: np.ndarray, max_shift: int = 25, rng=None) -> np.ndarray:
    """Roll each trial independently by a random number of samples in ±max_shift.

    X : (N, C, T)
    """
    rng = rng or np.random.default_rng()
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(X))
    out = np.empty_like(X)
    for i, s in enumerate(shifts):
        out[i] = np.roll(X[i], s, axis=-1)
    return out


def aug_gaussian_noise(X: np.ndarray, sigma: float = 0.05, rng=None) -> np.ndarray:
    """Add zero-mean Gaussian noise with std = sigma * signal_std."""
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal(X.shape).astype(X.dtype)
    # Scale noise relative to per-channel std (already standardised, so ~1)
    return X + sigma * noise


def aug_channel_freq_mask(X: np.ndarray, sfreq: float = 250.0,
                          p_channel: float = 0.5, rng=None) -> np.ndarray:
    """Zero out either a random channel or a random 4-Hz frequency band.

    With probability p_channel, mask one channel; otherwise zero a 4-Hz band.
    """
    rng = rng or np.random.default_rng()
    out = X.copy()
    N, C, T = out.shape
    for i in range(N):
        if rng.random() < p_channel:
            # Channel masking
            ch = rng.integers(0, C)
            out[i, ch, :] = 0.0
        else:
            # Frequency-band masking via FFT
            # Pick a random 4-Hz band within the signal (up to Nyquist)
            nyq    = sfreq / 2.0
            f_lo   = rng.uniform(4.0, nyq - 5.0)
            f_hi   = f_lo + 4.0
            freqs  = np.fft.rfftfreq(T, d=1.0 / sfreq)
            mask   = (freqs >= f_lo) & (freqs <= f_hi)
            spec   = np.fft.rfft(out[i], axis=-1)
            spec[:, mask] = 0.0
            out[i] = np.fft.irfft(spec, n=T, axis=-1).astype(X.dtype)
    return out


def aug_mixup(X: np.ndarray, y: np.ndarray, alpha: float = 0.2,
              rng=None):
    """Mixup augmentation: convex combinations of trial pairs + soft labels.

    Returns X_mix (N, C, T) and y_mix (N, 2) as soft one-hot.
    """
    rng   = rng or np.random.default_rng()
    N     = len(X)
    n_cls = int(y.max()) + 1
    lam   = rng.beta(alpha, alpha, size=N).astype(np.float32)  # (N,)
    perm  = rng.permutation(N)

    X_mix = (lam[:, None, None] * X +
             (1 - lam[:, None, None]) * X[perm]).astype(X.dtype)

    # Soft one-hot labels
    y_oh      = np.eye(n_cls, dtype=np.float32)[y]
    y_perm_oh = np.eye(n_cls, dtype=np.float32)[y[perm]]
    y_mix     = (lam[:, None] * y_oh +
                 (1 - lam[:, None]) * y_perm_oh)
    return X_mix, y_mix


def apply_augmentation(X: np.ndarray, y: np.ndarray, rng=None,
                       do_mixup: bool = True):
    """Apply all augmentations and concatenate with the original data.

    Returns X_aug, y_aug (hard labels); mixup trials are assigned the
    *majority* label (argmax of soft labels) so we stay compatible with
    standard cross-entropy.
    """
    rng = rng or np.random.default_rng(0)
    X_aug_list = [X]
    y_aug_list = [y]

    # a) Time shift
    X_aug_list.append(aug_time_shift(X, rng=rng))
    y_aug_list.append(y.copy())

    # b) Gaussian noise
    X_aug_list.append(aug_gaussian_noise(X, rng=rng))
    y_aug_list.append(y.copy())

    # c) Channel/freq masking
    X_aug_list.append(aug_channel_freq_mask(X, rng=rng))
    y_aug_list.append(y.copy())

    # d) Mixup
    if do_mixup:
        X_mx, y_soft = aug_mixup(X, y, rng=rng)
        X_aug_list.append(X_mx)
        y_aug_list.append(y_soft.argmax(axis=1))  # hard label (argmax)

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)
    return X_out, y_out

print("Augmentation functions defined.")
# Quick smoke-check
_X_chk, _y_chk = apply_augmentation(X_tr[:10], y_tr[:10], rng=rng)
print(f"  apply_augmentation: {X_tr[:10].shape} → {_X_chk.shape}")

# %% [markdown]
# ### Step 7 — Train WITH vs WITHOUT augmentation on a small subset
#
# We deliberately use only a **small fraction of the training set** (30 trials
# by default, 16 in smoke mode) to stress-test augmentation.  The held-out test
# set is the same 30 % split from above, so the comparison is apples-to-apples.
#
# We repeat for `N_SEEDS` random seeds and compare the mean and spread.
# Chapter 09 note: one seed teaches plumbing — we use a few seeds here to get
# an honest sense of variance, though with this few trials, variance is high.

# %%
# In smoke mode we keep things minimal but still need balanced classes.
# We take the first N_PER_CLASS trials from each class (stratified small set).
N_PER_CLASS  = 8 if SMOKE else 15    # per-class count for the small training set
N_SEEDS      = 2 if SMOKE else 4
EPOCHS_AUG   = 5 if SMOKE else 15


def make_small_balanced(X, y, n_per_class, rng_seed=0):
    """Return a balanced small training set (n_per_class per class)."""
    rng2 = np.random.default_rng(rng_seed)
    idx_list = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n_take = min(n_per_class, len(c_idx))
        chosen = rng2.choice(c_idx, size=n_take, replace=False)
        idx_list.append(chosen)
    idx_all = np.concatenate(idx_list)
    return X[idx_all], y[idx_all]


def train_eval(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               augment: bool, seed: int, epochs: int) -> float:
    """Train a fresh EEGNet and return test accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    aug_rng = np.random.default_rng(seed)

    if augment:
        X_tr_use, y_tr_use = apply_augmentation(X_train, y_train, rng=aug_rng)
    else:
        X_tr_use, y_tr_use = X_train.copy(), y_train.copy()

    Xtr_t2, ytr_t2 = to_tensor(X_tr_use, y_tr_use)
    Xte_t2, _      = to_tensor(X_test, y_test)

    net = build_eegnet(N_CHANNELS, N_TIMES)
    net.to(DEVICE)
    opt  = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    ds2 = TensorDataset(Xtr_t2, ytr_t2)
    dl2 = DataLoader(ds2, batch_size=min(16, len(y_tr_use)), shuffle=True,
                     generator=torch.Generator().manual_seed(seed))

    net.train()
    for _ in range(epochs):
        for xb, yb in dl2:
            opt.zero_grad()
            loss = crit(net(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        preds = net(Xte_t2.to(DEVICE)).argmax(1).cpu().numpy()
    return float((preds == y_test).mean())


# Build a balanced small training set
X_small, y_small = make_small_balanced(X_tr, y_tr, N_PER_CLASS, rng_seed=42)
SMALL_N = len(y_small)
print(f"Small training set: {SMALL_N} trials  "
      f"(classes: {np.bincount(y_small.astype(int))})")
print(f"Test set           : {len(X_te)} trials")
print(f"Training {N_SEEDS} seeds × 2 conditions …")

accs_no_aug  = []
accs_with_aug = []

for seed in range(N_SEEDS):
    acc_no  = train_eval(X_small, y_small, X_te, y_te,
                         augment=False, seed=seed, epochs=EPOCHS_AUG)
    acc_yes = train_eval(X_small, y_small, X_te, y_te,
                         augment=True,  seed=seed, epochs=EPOCHS_AUG)
    accs_no_aug.append(acc_no)
    accs_with_aug.append(acc_yes)
    print(f"  seed={seed}  no-aug={acc_no:.3f}  with-aug={acc_yes:.3f}")

mean_no  = float(np.mean(accs_no_aug))
mean_yes = float(np.mean(accs_with_aug))
std_no   = float(np.std(accs_no_aug))
std_yes  = float(np.std(accs_with_aug))

print(f"\nResults ({N_SEEDS} seeds):")
print(f"  Without augmentation : {mean_no:.3f} ± {std_no:.3f}")
print(f"  With augmentation    : {mean_yes:.3f} ± {std_yes:.3f}")
print(f"  Delta                : {mean_yes - mean_no:+.3f}")
print(f"  Chance level         : 0.500")

# %% [markdown]
# ### Visualisation 2 — With vs Without augmentation accuracy

# %%
fig, ax = plt.subplots(figsize=(7, 5))

labels_bar = ["No augmentation", "With augmentation"]
means_bar  = [mean_no, mean_yes]
stds_bar   = [std_no,  std_yes]
colors_bar = ["#e74c3c", "#2ecc71"]
x_pos      = [0, 1]

bars2 = ax.bar(x_pos, means_bar, yerr=stds_bar, capsize=8,
               color=colors_bar, edgecolor="white", width=0.5,
               error_kw=dict(elinewidth=2, ecolor="black"))

# Jitter individual seed dots on top of bars
for xi, accs in enumerate([accs_no_aug, accs_with_aug]):
    jitter = rng.uniform(-0.05, 0.05, size=len(accs))
    ax.scatter(xi + jitter, accs, color="black", s=40, zorder=5, alpha=0.7)

ax.axhline(0.5, color="gray", lw=1.5, linestyle="--", label="Chance (50%)")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_bar, fontsize=12)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel("Test accuracy", fontsize=11)
ax.set_title(
    f"Augmentation effect on small training sets\n"
    f"EEGNet · {SMALL_N} labelled trials · {N_SEEDS} seeds · BCI IV 2a",
    fontsize=10,
)
ax.legend(fontsize=9)

# Annotate bars with mean ± std text
for xi, (m, s, col) in enumerate(zip(means_bar, stds_bar, colors_bar)):
    ax.text(xi, m + s + 0.015, f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/dd_int_augmentation.png", dpi=100, bbox_inches="tight")
plt.show()
print("Augmentation figure saved.")
print(f"\n[SUMMARY]  no-aug={mean_no:.3f}  with-aug={mean_yes:.3f}  delta={mean_yes-mean_no:+.3f}")

# %% [markdown]
# ### Interpretation
#
# With only ~30 labelled trials, EEGNet has barely enough data to learn the
# spatial + temporal filters simultaneously. Augmentation acts as a regulariser:
#
# * **Time shift** prevents the model from overfitting to exact temporal offsets
#   of the motor response, which varies ±100 ms across trials naturally.
# * **Gaussian noise** discourages the model from memorising electrode noise
#   patterns that are idiosyncratic to individual trials.
# * **Channel masking** forces redundancy across spatial filters — similar to
#   dropout but applied at the sensor level.
# * **Mixup** smooths the decision boundary between left and right MI, which
#   reduces overconfident predictions on test examples near the boundary.
#
# With more data (full training set used in Part A) augmentation has less impact
# — the model already sees enough natural variability. This is the "data regime"
# interaction that Chapter 09 discusses.
#
# **Variance caveat:** with only 30 training trials and a handful of seeds the
# error bars are wide. Do not read too much into the specific numbers — the
# plumbing is what matters for extending this to your own dataset.

# %% [markdown]
# ## ⚠️ A subtler trap
#
# ### Gradient saliency can silently mislead you
#
# The saliency map we computed shows where the **loss surface is steep**, not
# where the model has learned a reliable causal feature. These two things
# diverge in at least three important ways:
#
# **1. Input-gradient instability.** Vanilla gradient saliency is notoriously
# noisy: a tiny perturbation to the input can flip the saliency map entirely
# (Ghorbani et al., "Interpretation of Neural Networks is Fragile", 2019).
# Two trials with nearly identical EEG and the same predicted class can produce
# completely different saliency patterns. Averaging over the test set (as we did)
# helps, but the average of meaningless patterns is still meaningless. Methods
# like **Integrated Gradients** or **SmoothGrad** are more stable, but still
# not causal.
#
# **2. High saliency ≠ causal use.** Suppose the model assigns high gradient
# magnitude to channel C3. This might mean:
# * The model is genuinely using mu-rhythm suppression over the left motor cortex
#   ✓ (the story we want).
# * Or: C3 and Cz happen to be correlated in the training set, and the gradient
#   lands on C3 because it is downstream in the depth-wise convolution weight
#   ordering, not because it is physiologically special ✗.
# * Or: An artifact present in certain training trials (e.g. a slow drift that
#   the baseline correction did not fully remove) coincidentally correlates with
#   the label, and C3 is where that drift was strongest ✗.
#
# You cannot distinguish these cases from the saliency map alone. Only a
# **causal intervention** — physically blocking C3 (electrode removal) or
# ablating it in post-hoc analysis and retraining — would tell you.
#
# **3. The augmentation / saliency interaction trap.** If you apply channel
# masking during training (as we did in aug `c`) and then compute gradient
# saliency on the *trained* model, the saliency will be artificially spread
# across channels. Masking taught the model to distribute its reliance; the
# saliency map then looks "more sensorimotor" simply because frontal channels
# were occasionally zeroed out during training — not because the biology demands
# it. This makes the model appear more neurologically interpretable than it is.
#
# **The take-away:** treat saliency as a debugging tool for hunting obvious
# failures (e.g. the model staring at an EOG channel), not as a scientific
# claim about which brain regions drive motor imagery. For that you need
# ablation studies, occlusion sensitivity, or — better — a pre-registered
# decoding pipeline with held-out data from a separate lab.
