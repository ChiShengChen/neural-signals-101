# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Self-Supervised Learning for EEG
#
# Pre-train a small encoder on *unlabeled* EEG via a pretext task, then fine-tune
# on a tiny labeled set — the idea behind EEG "foundation models".
#
# > **Prerequisites:** main Chapter 09.
# > **Level:** advanced ★★★★☆
# > **The idea behind EEG "foundation models". Not bound by the 5-min budget.**

# %% Bootstrap — find neuro101 via upward search
import sys
import os
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _parent in [_here, *_here.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101").is_dir():
            sys.path.insert(0, str(_candidate))
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

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

DEVICE = "cpu"
SMOKE  = ds.is_smoke()

print(f"Smoke mode : {SMOKE}")
print(f"Device     : {DEVICE}")
print(f"torch      : {torch.__version__}")

# %% [markdown]
# ---
# ## Part 1 — Why self-supervised learning for EEG?
#
# ### The label scarcity problem
#
# In a typical motor-imagery BCI study a subject must sit still, imagine movements
# on cue, and repeat this hundreds of times across multiple sessions.  Each labeled
# trial costs roughly 4–8 seconds of careful experimental control.  Getting 200
# labeled trials from one subject costs 15–30 minutes of calibration — and those
# labels are valid only for *that* subject on *that* day.
#
# Contrast this with *unlabeled* EEG:
#
# * Resting-state recordings run for hours with no experimenter intervention.
# * Sleep EEG studies produce 8-hour continuous streams.
# * Clinical archives hold millions of patient-hours.
# * All of this data is collected without per-trial behavioral labels.
#
# **Self-supervised learning (SSL)** turns the structure of the raw signal itself
# into a training objective.  A *pretext task* is defined purely from the data —
# no human labeling needed — and a neural encoder is trained to solve it.  If the
# pretext task requires understanding meaningful temporal or spectral structure of
# EEG, the encoder's internal representations should transfer to downstream
# tasks such as sleep staging, emotion recognition, or motor imagery.
#
# ### Canonical EEG pretext tasks
#
# | Pretext task | Source | Signal structure exploited |
# |---|---|---|
# | **Relative positioning** (RP) | Banville et al. 2021 (MOABB, "Uncovering the structure of clinical EEG") | Close-in-time windows share similar neural state; far-apart windows do not |
# | **Temporal shuffling** (TS) | Banville et al. 2021 | The correct time-order of a triplet of windows is predictable from neural dynamics |
# | **Contrastive multi-view** (SimCLR-EEG) | Cheng et al. 2020 | Two augmented views of the same window should have similar representations |
# | **Masked prediction** (BERT-style) | Kostas et al. 2022 | Masked channel patches can be reconstructed from context |
#
# This deep-dive implements **relative positioning (RP)** — the simplest of the
# four and the one introduced by Banville et al. 2021.  It requires zero class
# labels and can be run on any continuous EEG recording.

# %% [markdown]
# ---
# ## Part 2 — Data loading and pretext-task construction
#
# We load BCI IV 2a (left vs. right hand motor imagery) but **only use the raw
# signal** during pre-training.  Labels are reserved for the fine-tuning and
# evaluation phases.
#
# ### Leak-free protocol
#
# | Phase | Data used | Labels used? |
# |---|---|---|
# | Pre-training (SSL) | Pretrain subjects (unlabeled) | **No** |
# | Fine-tune / linear probe | Small portion of held-out subject's train set | **Yes (few)** |
# | Evaluation | Remainder of held-out subject's test set | Ground-truth only |
#
# We hold out one subject completely for the downstream evaluation.  The
# pretrained encoder never sees that subject's trials.

# %%
# ── Data parameters ───────────────────────────────────────────────────────────
N_SUBJ_TOTAL = 3 if SMOKE else 6   # total subjects to load
# Subject 1 is held out for downstream eval; the rest are used for SSL pretraining
PRETRAIN_SUBJ_COUNT = max(1, N_SUBJ_TOTAL - 1)

print(f"Loading {N_SUBJ_TOTAL} subject(s) …")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=N_SUBJ_TOTAL)
n_trials, N_CH, N_TIMES = X_all.shape
print(f"  X={X_all.shape}  (trials × channels × time-points)")
print(f"  classes={np.bincount(y_all)}  subjects={np.unique(subj_all).tolist()}")

# Split: subjects 2..N for pretraining, subject 1 for downstream
subj_ids   = np.unique(subj_all)
pretrain_subjects = subj_ids[1:]        # subjects 2, 3, …
eval_subject      = subj_ids[0]         # subject 1 — never seen during pretraining

mask_pretrain = np.isin(subj_all, pretrain_subjects)
mask_eval     = subj_all == eval_subject

X_pretrain = X_all[mask_pretrain]       # shape (n_pre, ch, times)
X_eval     = X_all[mask_eval]           # shape (n_eval, ch, times)
y_eval     = y_all[mask_eval]

print(f"\nPre-train pool : {X_pretrain.shape[0]} trials  (subjects {pretrain_subjects.tolist()})")
print(f"Eval subject   : {X_eval.shape[0]} trials  (subject {eval_subject})")

# ── Normalise per channel (pretrain stats) ────────────────────────────────────
mu_pre  = X_pretrain.mean(axis=(0, 2), keepdims=True)
sig_pre = X_pretrain.std(axis=(0, 2),  keepdims=True) + 1e-8
X_pretrain_n = (X_pretrain - mu_pre) / sig_pre

# Apply same normalisation to eval data (no leakage — mu/sig from pretrain pool)
X_eval_n = (X_eval - mu_pre) / sig_pre

# %% [markdown]
# ### Building relative-positioning (RP) pairs
#
# **Algorithm (Banville et al. 2021 §2.1):**
#
# 1. Pick a random "anchor" window $w_a$ of length $L$ from a trial.
# 2. Sample a second window $w_b$:
#    - **Positive** (close): $w_b$ starts within $\tau_{\text{pos}}$ samples of $w_a$.
#    - **Negative** (far):   $w_b$ starts at least $\tau_{\text{neg}}$ samples away.
# 3. The binary label is 1 (close) or 0 (far).
# 4. Crucially: **no motor-imagery labels are used**.  The anchor and the second window
#    can come from any trial regardless of condition.

# %%
def make_rp_pairs(
    X: np.ndarray,
    n_pairs: int,
    win_len: int,
    tau_pos: int,
    tau_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample relative-positioning (anchor, sample, label) triplets.

    Parameters
    ----------
    X       : (n_trials, n_ch, n_times)
    n_pairs : how many pairs to generate
    win_len : window length in samples
    tau_pos : max temporal offset for a "close" positive pair
    tau_neg : min temporal offset for a "far" negative pair
    rng     : numpy random Generator

    Returns
    -------
    anc  : (n_pairs, n_ch, win_len)
    samp : (n_pairs, n_ch, win_len)
    lbl  : (n_pairs,) int — 1=close, 0=far
    """
    n_trials, n_ch, n_times = X.shape
    max_start = n_times - win_len

    if max_start <= 0:
        raise ValueError(f"win_len={win_len} too long for n_times={n_times}")

    anc_list, samp_list, lbl_list = [], [], []

    for _ in range(n_pairs):
        # Pick a random trial and anchor start position
        trial_idx = rng.integers(0, n_trials)
        a_start   = rng.integers(0, max_start + 1)

        if rng.random() < 0.5:
            # Positive pair: sample within tau_pos of anchor
            low  = max(0,          a_start - tau_pos)
            high = min(max_start,  a_start + tau_pos)
            b_start = int(rng.integers(low, high + 1))
            label   = 1
        else:
            # Negative pair: sample at least tau_neg away
            # Build a valid range away from anchor
            candidates = []
            if a_start - tau_neg >= 0:
                candidates.append((0, a_start - tau_neg))
            if a_start + tau_neg <= max_start:
                candidates.append((a_start + tau_neg, max_start))
            if not candidates:
                # Edge case: short trial — fall back to far end or near start
                b_start = 0 if a_start > max_start // 2 else max_start
            else:
                seg = candidates[rng.integers(len(candidates))]
                b_start = int(rng.integers(seg[0], seg[1] + 1))
            label = 0

        anc_list.append(X[trial_idx, :, a_start : a_start + win_len])
        samp_list.append(X[trial_idx, :, b_start : b_start + win_len])
        lbl_list.append(label)

    return (
        np.stack(anc_list).astype(np.float32),
        np.stack(samp_list).astype(np.float32),
        np.array(lbl_list, dtype=np.int64),
    )

# ── Pretext task hyper-parameters ─────────────────────────────────────────────
WIN_LEN   = min(100, N_TIMES)          # 100 samples ≈ 400 ms at 250 Hz
TAU_POS   = max(10, WIN_LEN // 4)      # "close" = within 1/4 window
TAU_NEG   = max(WIN_LEN // 2, TAU_POS + 10)  # "far" = > half window away
N_PAIRS   = 500 if SMOKE else 3000

anc_np, samp_np, lbl_np = make_rp_pairs(
    X_pretrain_n, N_PAIRS, WIN_LEN, TAU_POS, TAU_NEG, rng
)
print(f"RP pairs: {anc_np.shape}  |  pos={lbl_np.sum()}  neg={(lbl_np==0).sum()}")

# Convert to tensors
anc_t  = torch.from_numpy(anc_np)
samp_t = torch.from_numpy(samp_np)
lbl_t  = torch.from_numpy(lbl_np)

rp_dataset = TensorDataset(anc_t, samp_t, lbl_t)
rp_loader  = DataLoader(
    rp_dataset, batch_size=64, shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
print(f"RP DataLoader: {len(rp_loader)} batches of up to 64 pairs")

# %% [markdown]
# ---
# ## Part 3 — Encoder architecture and pretext pre-training
#
# ### The encoder
#
# We keep the encoder deliberately tiny so it runs in seconds on CPU.  A small
# stack of 1-D convolutional layers over the time dimension (applied independently
# to each channel) followed by a global average pool produces a fixed-length
# embedding regardless of input length.
#
# ### The pretext head
#
# Given embeddings $z_a$ and $z_b$ for the anchor and sample, the RP head
# concatenates them and predicts close (1) vs. far (0):
#
# $$\hat{y} = \sigma(W_2\,\text{ReLU}(W_1 [z_a \| z_b]))$$
#
# Both the encoder and the head are trained jointly on the RP loss.  The head
# is discarded after pretraining; only the encoder weights are reused.

# %%
class TinyEEGEncoder(nn.Module):
    """Small conv encoder for EEG windows.

    Input : (batch, n_ch, win_len)
    Output: (batch, embed_dim)
    """

    def __init__(self, n_ch: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Temporal conv block 1
            nn.Conv1d(n_ch, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),          # halve time
            # Temporal conv block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),          # halve again
            # Temporal conv block 3
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # (batch, embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)               # (batch, embed_dim, T')
        h = self.pool(h).squeeze(-1)  # (batch, embed_dim)
        return h


class RPHead(nn.Module):
    """Relative-positioning binary head: predict close (1) vs. far (0)."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_a, z_b], dim=1))


EMBED_DIM = 64
encoder  = TinyEEGEncoder(n_ch=N_CH, embed_dim=EMBED_DIM).to(DEVICE)
rp_head  = RPHead(embed_dim=EMBED_DIM).to(DEVICE)

n_params_enc  = sum(p.numel() for p in encoder.parameters())
n_params_head = sum(p.numel() for p in rp_head.parameters())
print(f"Encoder params : {n_params_enc:,}")
print(f"RP head params : {n_params_head:,}")
print(f"Total          : {n_params_enc + n_params_head:,}")

# ── Pre-training loop ─────────────────────────────────────────────────────────
PRETRAIN_EPOCHS = 3 if SMOKE else 15
PRETRAIN_LR     = 3e-4

optimizer_ssl = torch.optim.Adam(
    list(encoder.parameters()) + list(rp_head.parameters()),
    lr=PRETRAIN_LR,
)
criterion_ssl = nn.CrossEntropyLoss()

pretrain_losses = []
print(f"\nPre-training ({PRETRAIN_EPOCHS} epochs, relative positioning) …")

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    encoder.train()
    rp_head.train()
    running_loss, running_correct, n_total = 0.0, 0, 0

    for anc_b, samp_b, lbl_b in rp_loader:
        anc_b  = anc_b.to(DEVICE)
        samp_b = samp_b.to(DEVICE)
        lbl_b  = lbl_b.to(DEVICE)

        z_a = encoder(anc_b)
        z_b = encoder(samp_b)
        logits = rp_head(z_a, z_b)
        loss   = criterion_ssl(logits, lbl_b)

        optimizer_ssl.zero_grad()
        loss.backward()
        optimizer_ssl.step()

        running_loss    += loss.item() * len(lbl_b)
        running_correct += (logits.argmax(1) == lbl_b).sum().item()
        n_total         += len(lbl_b)

    epoch_loss = running_loss / n_total
    epoch_acc  = running_correct / n_total
    pretrain_losses.append(epoch_loss)

    if epoch == 1 or epoch % max(1, PRETRAIN_EPOCHS // 5) == 0 or epoch == PRETRAIN_EPOCHS:
        print(f"  Epoch {epoch:3d}/{PRETRAIN_EPOCHS}  loss={epoch_loss:.4f}  RP-acc={epoch_acc:.3f}")

print("Pre-training complete. Encoder weights frozen for downstream use.")

# %% [markdown]
# ### Figure 1 — Pre-training loss curve
#
# A declining loss curve means the encoder is learning to distinguish "close in
# time" from "far apart" — it is building a representation that captures temporal
# autocorrelation structure in EEG without ever seeing a behavioral label.
#
# Perfect RP accuracy would be 100 % (trivially overfitting); we want the encoder
# to generalise, not memorise pairs.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(pretrain_losses) + 1), pretrain_losses,
        "o-", color="#2078b4", linewidth=2, markersize=5, label="RP pretext loss")
ax.axhline(np.log(2), color="gray", ls="--", lw=1.5,
           label=f"Chance CE = {np.log(2):.3f}  (random binary classifier)")
ax.set(
    xlabel="Pre-training epoch",
    ylabel="Cross-entropy loss (RP pretext task)",
    title=(
        "Self-supervised pre-training: relative-positioning loss\n"
        "Encoder learns temporal structure with NO class labels"
    ),
    xlim=(0.5, len(pretrain_losses) + 0.5),
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_ssl_pretrain_curve.png", dpi=100, bbox_inches="tight")
plt.show()
print("Figure 1 saved → /tmp/dd_ssl_pretrain_curve.png")

# %% [markdown]
# ---
# ## Part 4 — Downstream evaluation: linear probe vs. from-scratch
#
# ### Protocol
#
# We evaluate on the held-out subject (subject 1, never seen during pre-training).
# For each budget of labeled trials $K \in \{4, 8, 16, 32, \ldots\}$:
#
# 1. **Pretrained (linear probe):** freeze the SSL encoder, train only a
#    2-class linear head on the $K$ labeled trials.
# 2. **From scratch:** train the *same* encoder architecture from random
#    initialization on the same $K$ labeled trials (encoder + linear head,
#    all weights trainable).
# 3. Evaluate both on the *remaining* held-out trials.
#
# If SSL pre-training helps, the pretrained linear probe should beat the
# from-scratch model at small $K$, where the latter has very few labeled
# examples to learn from.

# %%
# ── Downstream data: held-out subject ─────────────────────────────────────────
# Time-ordered split: first 60% as pool for labeled selection, last 40% as test.
n_eval = len(y_eval)
split_pt = int(0.6 * n_eval)

X_pool = X_eval_n[:split_pt]
y_pool = y_eval[:split_pt]
X_test = X_eval_n[split_pt:]
y_test = y_eval[split_pt:]

# Extract full-trial windows for downstream (use the middle WIN_LEN samples)
start_ds = (N_TIMES - WIN_LEN) // 2
X_pool_w = X_pool[:, :, start_ds : start_ds + WIN_LEN].astype(np.float32)
X_test_w = X_test[:, :, start_ds : start_ds + WIN_LEN].astype(np.float32)

print(f"Pool for labeled selection : {X_pool_w.shape}  (classes={np.bincount(y_pool)})")
print(f"Test set (never used in fit): {X_test_w.shape}  (classes={np.bincount(y_test)})")

Xte_t_ds = torch.from_numpy(X_test_w)
yte_np    = y_test


# ── Label-budget sweep ────────────────────────────────────────────────────────
def get_balanced_subset(X, y, n_per_class, rng):
    """Return at most n_per_class examples from each class, randomly sampled."""
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=min(n_per_class, len(c_idx)), replace=False)
        idx.append(chosen)
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def evaluate_encoder_probe(
    encoder_model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te_tensor: torch.Tensor,
    y_te: np.ndarray,
    freeze_encoder: bool,
    n_epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> float:
    """Train a linear head (or full fine-tune) and return test accuracy."""
    import copy
    enc = copy.deepcopy(encoder_model).to(device)

    if freeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    else:
        enc.train()

    head = LinearHead(enc.embed_dim).to(device)
    torch.manual_seed(seed)

    params = list(head.parameters())
    if not freeze_encoder:
        params += list(enc.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32))
    ytr_t = torch.from_numpy(y_tr.astype(np.int64))

    ds_tr = TensorDataset(Xtr_t, ytr_t)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=min(16, len(y_tr)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    for _ in range(n_epochs):
        if freeze_encoder:
            enc.eval()
        else:
            enc.train()
        head.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            z = enc(xb)
            logits = head(z)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    enc.eval()
    head.eval()
    with torch.no_grad():
        z_te = enc(X_te_tensor.to(device))
        preds = head(z_te).argmax(1).cpu().numpy()
    return float((preds == y_te).mean())


# ── Sweep budgets ─────────────────────────────────────────────────────────────
if SMOKE:
    BUDGETS = [2, 4, 8]
    FINETUNE_EPOCHS = 10
    SCRATCH_EPOCHS  = 10
else:
    # Maximum n_per_class is limited to what's in the pool
    max_per_class = min(np.bincount(y_pool))
    raw_budgets = [2, 4, 8, 16, 32]
    BUDGETS = [b for b in raw_budgets if b <= max_per_class]
    if not BUDGETS:
        BUDGETS = [max(1, max_per_class // 2), max_per_class]
    FINETUNE_EPOCHS = 30
    SCRATCH_EPOCHS  = 50

PROBE_LR   = 1e-3
SCRATCH_LR = 3e-4

print(f"\nLabel-budget sweep: {BUDGETS} (per class)")
print(f"Fine-tune epochs (pretrained): {FINETUNE_EPOCHS}")
print(f"Train epochs (from-scratch)  : {SCRATCH_EPOCHS}")
print(f"Test set size: {len(y_test)}\n")

# Build a fresh encoder (same architecture, random weights) for scratch baseline
scratch_encoder_init = TinyEEGEncoder(n_ch=N_CH, embed_dim=EMBED_DIM)

acc_pretrained = []
acc_scratch    = []

for n_per_class in BUDGETS:
    X_sub, y_sub = get_balanced_subset(X_pool_w, y_pool, n_per_class, rng)

    # -- Pretrained linear probe --
    acc_pt = evaluate_encoder_probe(
        encoder_model   = encoder,           # SSL-pretrained weights
        X_tr            = X_sub,
        y_tr            = y_sub,
        X_te_tensor     = Xte_t_ds,
        y_te            = yte_np,
        freeze_encoder  = True,              # linear probe only
        n_epochs        = FINETUNE_EPOCHS,
        lr              = PROBE_LR,
        device          = DEVICE,
        seed            = SEED,
    )

    # -- From scratch (same arch, random init, full training) --
    acc_sc = evaluate_encoder_probe(
        encoder_model   = scratch_encoder_init,
        X_tr            = X_sub,
        y_tr            = y_sub,
        X_te_tensor     = Xte_t_ds,
        y_te            = yte_np,
        freeze_encoder  = False,             # train encoder + head from scratch
        n_epochs        = SCRATCH_EPOCHS,
        lr              = SCRATCH_LR,
        device          = DEVICE,
        seed            = SEED,
    )

    acc_pretrained.append(acc_pt)
    acc_scratch.append(acc_sc)
    print(f"  K={n_per_class:3d}/class  |  pretrained={acc_pt:.3f}  |  scratch={acc_sc:.3f}")

# %% [markdown]
# ### Figure 2 — Pretrained vs. from-scratch accuracy as a function of label budget
#
# This is the central figure of the notebook.  Each point is the test accuracy on
# the held-out subject.  The x-axis is the number of labeled trials *per class*
# available for fine-tuning.
#
# **What to look for:**
# * At the smallest label budget (leftmost), does the pretrained linear probe
#   outperform the from-scratch model?  This is where SSL pre-training is most
#   useful — the encoder already extracts structured features even before it sees
#   any labels.
# * As K grows, the from-scratch model should approach or match the pretrained
#   model (given enough data it can learn its own features).
# * **Caveat:** with a tiny encoder, few pretraining pairs, and a single evaluation
#   subject, the effect may be modest or noisy.  The plumbing matters more than the
#   exact numbers at this tutorial scale.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

label_counts = [2 * b for b in BUDGETS]   # total labels = 2 × n_per_class

ax.plot(label_counts, acc_pretrained, "o-", color="#2ca02c", linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="SSL pretrained → linear probe")
ax.plot(label_counts, acc_scratch, "s--", color="#d62728", linewidth=2.0,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="From-scratch encoder + head")
ax.axhline(0.5, color="gray", lw=1.2, ls=":", label="Chance (50 %)")

# Annotate values
for x, yp, ys in zip(label_counts, acc_pretrained, acc_scratch):
    ax.annotate(f"{yp:.2f}", (x, yp), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#2ca02c")
    ax.annotate(f"{ys:.2f}", (x, ys), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="#d62728")

ax.set(
    xlabel="Total labeled trials used for fine-tuning / training (both classes)",
    ylabel=f"Test accuracy (subject {eval_subject}, held-out)",
    title=(
        "SSL pre-training vs. from-scratch: accuracy vs. label budget\n"
        "Relative positioning pretext task on BCI IV 2a  (single-seed — see ⚠️)"
    ),
    ylim=(0.3, 1.05),
)
ax.legend(fontsize=10, loc="lower right")
ax.xaxis.get_major_locator().set_params(integer=True)
plt.tight_layout()
plt.savefig("/tmp/dd_ssl_pretrained_vs_scratch.png", dpi=100, bbox_inches="tight")
plt.show()

# Print the smallest-budget result for the return value
smallest_k = BUDGETS[0]
acc_pt_min = acc_pretrained[0]
acc_sc_min = acc_scratch[0]
print(f"\nFigure 2 saved → /tmp/dd_ssl_pretrained_vs_scratch.png")
print(f"\nSmallest label budget: K={smallest_k}/class  ({2*smallest_k} total trials)")
print(f"  Pretrained linear probe : {acc_pt_min:.3f}")
print(f"  From-scratch            : {acc_sc_min:.3f}")
print(f"  Delta (pretrained-scratch): {acc_pt_min - acc_sc_min:+.3f}")

# %% [markdown]
# ---
# ## Part 5 — Honest caveats (echoing Chapter 09)
#
# ### What this demo teaches — and what it does not
#
# | This demo shows | This demo does NOT show |
# |---|---|
# | How to implement RP pretext task from scratch | A rigorous SSL benchmark |
# | The plumbing: encoder → pretext → transfer | Statistical significance (single seed) |
# | That the idea compiles and runs on CPU | That SSL always beats supervised on EEG |
# | A leak-free pretraining/eval split | Effect size reproducibility |
#
# **Single-seed results:** one run of this notebook is a single point estimate with
# no error bars.  The variance across random seeds, subjects, and datasets is large
# at small K.  Banville et al. 2021 ran this across 11 datasets and 5 seeds.
#
# **Tiny scale:** our encoder has ~50k parameters, pretrained on a few hundred
# pairs for ≤15 epochs.  Real EEG SSL models use millions of parameters and hours
# of pretraining on large corpora.  At tutorial scale the pretrained model may
# only marginally (or sometimes not) outperform scratch — this is expected and
# honest.
#
# **Label budget granularity:** with a held-out subject providing ~100 trials
# (~50 per class), the "small K" regime is poorly sampled.  A robust study would
# sweep 10–20 random draws of the label subset and report mean ± std.
#
# **Pretext task quality:** RP is the simplest pretext task.  Temporal shuffling,
# contrastive multi-view, and masked prediction typically learn richer
# representations.  For EEG "foundation models" see:
# * Banville et al. 2021 — "Uncovering the structure of clinical EEG signals with
#   self-supervised learning" (NeuroImage / arXiv 2007.16104)
# * Kostas et al. 2022 — "BENDR: Using transformers and a contrastive
#   self-supervised learning task to learn from massive amounts of EEG data"
#   (Frontiers Hum. Neurosci.)
# * Jiang et al. 2024 — "LaBraM: Large Brain Model for EEG" (ICLR 2024)

# %% [markdown]
# ---
# ## ⚠️ A subtler trap — pretext-task solutions via physiological artifacts
#
# ### The invisible shortcut
#
# The relative-positioning task asks the encoder to predict whether two EEG windows
# are close or far apart in time.  This *sounds* like it forces the encoder to
# learn neural dynamics.  But there are at least two physiological confounds that
# let the encoder solve RP without learning anything neuroscientifically meaningful:
#
# ### Confound 1 — Electrode drift and slow-baseline wandering
#
# EEG amplifiers are DC-coupled (or have very long time constants).  Over the course
# of a trial or session, skin-electrode impedance changes produce a **slow
# electro-dermal drift** that shifts the baseline voltage of each channel by
# microvolts to tens of microvolts over tens of seconds.  This drift is highly
# temporally autocorrelated: two windows taken 50 ms apart have almost identical
# drift levels; two windows taken 30 seconds apart may differ substantially.
#
# **Consequence:** the encoder can achieve high RP accuracy simply by learning to
# compare the *mean voltage offset* of each window.  Close windows share a similar
# offset; far windows do not.  The learned features are not oscillatory or
# event-related — they are slow drift features.  When this encoder is used for
# motor-imagery classification (which depends on alpha/beta power modulation, not
# DC level), the pretrained features may be *less* useful than features learned
# from scratch on the actual task.
#
# **Detection:** compare RP accuracy when the data is **high-pass filtered** at
# 1 Hz (removes drift) vs. raw.  If removing drift causes a large drop in RP
# accuracy, the encoder was relying on drift, not neural dynamics.  The
# `artifact_confounds` deep-dive covers the broader pattern of artifact-driven
# decoding.
#
# ### Confound 2 — Line-noise amplitude modulation
#
# Powerline interference (50/60 Hz) and its harmonics appear in EEG whenever
# electrode impedance increases during a session.  The amplitude of the line-noise
# component typically *drifts* over the session as the electrode gel dries —
# meaning two temporally close windows have similar line-noise amplitudes, and two
# far-apart windows do not.  A convolutional encoder that learns to detect 50 Hz
# amplitude as a proxy for temporal distance solves RP well but learns an artifact
# feature, not a neural one.
#
# **Detection:** apply a notch filter at 50/60 Hz before pretraining.  If RP
# accuracy drops significantly, line-noise amplitude was the shortcut.
#
# ### The critical insight (cross-reference `artifact_confounds.py`)
#
# Both confounds share the same root cause: **any signal property that is
# temporally autocorrelated at a shorter scale than $\tau_{\text{neg}}$ but
# variable at a longer scale will solve the RP task**.  This includes:
#
# * Electrode drift (seconds to minutes)
# * Line-noise amplitude modulation (minutes)
# * Movement-related EMG envelope (seconds)
# * Breathing-correlated slow waves (~4 s period)
#
# **None of these are neural.**
#
# ### The correct mitigation
#
# 1. **Pre-process before pretraining:** 1 Hz high-pass, notch at power-line
#    frequency, epoch-level rejection for large artefacts.  The same cleaning you
#    would apply for supervised learning.
# 2. **Choose $\tau_{\text{pos}}$ and $\tau_{\text{neg}}$ carefully:** if
#    $\tau_{\text{neg}}$ is larger than the timescale of the drift, RP is mostly
#    a drift-detection task.  Use $\tau_{\text{neg}} \leq 2$ s for mu/beta neural
#    dynamics.
# 3. **Validate the learned features:** after pretraining, compute the correlation
#    between the embedding and known artifact channels (EOG, EMG, slow drift).  If
#    the first principal component of the embedding correlates with EOG amplitude,
#    the encoder learned eye-movement features, not motor imagery features.
# 4. **Cross-subject pretraining:** if the encoder is pretrained on different
#    subjects than the evaluation subject, subject-specific drift patterns cannot
#    be memorised — which is exactly the leak-free protocol we used above.
#
# **The meta-lesson:** SSL is not immune to the artifact confounds that afflict
# supervised EEG decoding.  The pretext task defines what "useful structure" means,
# and if that structure is dominated by slow artifacts, the pretrained features will
# capture artifacts.  Clean data going in → useful representations coming out.
