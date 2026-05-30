# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Chapter 07 — Deep Learning for Neural Signals
#
# Deep nets can learn features end-to-end from raw EEG. We start with the
# **convolutional** models designed for EEG (EEGNet, ShallowConvNet, DeepConvNet),
# then a small **LSTM** and a tiny **Transformer**, and end with a note on
# **self-supervised / foundation models**.
#
# Everything here is kept **tiny so it trains on a CPU in a few minutes**, and we
# still obey the golden rule: split by subject, standardise on train only.
#
# ## Learning objectives
# 1. Train **EEGNet / ShallowConvNet / DeepConvNet** via `braindecode`.
# 2. Build a minimal **LSTM** and **Transformer** for EEG in PyTorch.
# 3. Keep training honest (subject-aware split) and reproducible (seeded).
#
# **Runtime:** ~3–5 min on CPU (smoke mode trains for very few epochs).

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from neuro101 import io, datasets as ds
from neuro101.eval import make_subject_split

# Reproducibility: seed everything.
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.use_deterministic_algorithms(False)
DEVICE = "cpu"  # this tutorial is CPU-only by design

SMOKE = ds.is_smoke()
EPOCHS = 3 if SMOKE else 15
n_subj = 2 if SMOKE else 3

# %% [markdown]
# ## Data: BCI IV 2a, subject-aware split
#
# We train on some subjects and test on a **held-out** subject (subject-independent
# — the honest setting). We standardise each channel using **training** statistics
# only.

# %%
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
X = X.astype(np.float32)
n_chans, n_times = X.shape[1], X.shape[2]
print(f"X={X.shape}, classes={np.bincount(y)}, subjects={np.unique(subj)}")

# Use the first LOSO fold: hold out one subject for testing.
train_idx, test_idx = next(make_subject_split(subj))
mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
sd = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-7
Xtr = (X[train_idx] - mu) / sd
Xte = (X[test_idx] - mu) / sd
ytr, yte = y[train_idx], y[test_idx]
print(f"train trials={len(ytr)} (subjects {np.unique(subj[train_idx])}), "
      f"test trials={len(yte)} (held-out subject {np.unique(subj[test_idx])})")

# %% [markdown]
# ## A tiny, transparent training loop
#
# We write the loop by hand (rather than hide it) so you can see exactly what
# happens: forward pass → loss → backward → step. Batches are small; we report
# test accuracy after training.

# %%
def train_eval(model, Xtr, ytr, Xte, yte, epochs=EPOCHS, lr=1e-3, batch=32):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte, dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossfn(out, ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
        history.append(acc)
    return history

# %% [markdown]
# ## 1. EEGNet, ShallowConvNet, DeepConvNet (braindecode)
#
# These convolutional architectures are the standard EEG deep-learning baselines.
# `braindecode` provides them ready to use. They expect input shaped
# `(batch, channels, time)`.

# %%
from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net

def make_model(name):
    if name == "EEGNet":
        return EEGNetv4(n_chans=n_chans, n_outputs=2, n_times=n_times)
    if name == "ShallowConvNet":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=2, n_times=n_times,
                               final_conv_length="auto")
    if name == "DeepConvNet":
        return Deep4Net(n_chans=n_chans, n_outputs=2, n_times=n_times,
                        final_conv_length="auto")
    raise ValueError(name)

conv_results = {}
for name in ["EEGNet", "ShallowConvNet", "DeepConvNet"]:
    torch.manual_seed(SEED)
    hist = train_eval(make_model(name), Xtr, ytr, Xte, yte)
    conv_results[name] = hist
    print(f"  {name:15s} held-out-subject accuracy: {hist[-1]:.3f}")

# %% [markdown]
# ## 2. A minimal LSTM
#
# Recurrent nets read the signal time-step by time-step. EEG is long, so we treat
# channels as features at each time-step and downsample time for speed. This is a
# *baseline*, not a competitive model — convolutional EEG nets usually win.

# %%
class TinyLSTM(nn.Module):
    def __init__(self, n_chans, hidden=32, n_classes=2, stride=4):
        super().__init__()
        self.stride = stride
        self.lstm = nn.LSTM(n_chans, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):              # x: (batch, chans, time)
        x = x[:, :, ::self.stride]     # downsample time
        x = x.transpose(1, 2)          # (batch, time, chans)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])   # last time-step

torch.manual_seed(SEED)
hist_lstm = train_eval(TinyLSTM(n_chans), Xtr, ytr, Xte, yte)
print(f"  TinyLSTM held-out-subject accuracy: {hist_lstm[-1]:.3f}")

# %% [markdown]
# ## 3. A tiny Transformer
#
# A small Transformer encoder over downsampled time-steps. Self-attention lets
# every time-step look at every other — powerful but data-hungry, so on this tiny
# dataset don't expect it to shine.

# %%
class TinyTransformer(nn.Module):
    def __init__(self, n_chans, d_model=32, nhead=4, layers=1, n_classes=2, stride=8):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(n_chans, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):              # (batch, chans, time)
        x = x[:, :, ::self.stride].transpose(1, 2)  # (batch, time, chans)
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x.mean(1))    # mean over time

torch.manual_seed(SEED)
hist_tr = train_eval(TinyTransformer(n_chans), Xtr, ytr, Xte, yte)
print(f"  TinyTransformer held-out-subject accuracy: {hist_tr[-1]:.3f}")

# %% [markdown]
# ## Compare learning curves

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for name, hist in {**conv_results, "TinyLSTM": hist_lstm, "TinyTransformer": hist_tr}.items():
    ax.plot(range(1, len(hist) + 1), hist, marker="o", ms=3, label=name)
ax.axhline(0.5, ls="--", color="gray", label="chance")
ax.set(xlabel="Epoch", ylabel="Held-out-subject accuracy",
       title="Tiny EEG deep nets on CPU (subject-independent)")
ax.legend(); ax.set_ylim(0, 1); plt.show()

# %% [markdown]
# ## Further reading: self-supervised & foundation models
#
# Labels are scarce in neuroscience, so a growing trend is **self-supervised
# learning** — pre-train on lots of *unlabelled* EEG (predicting masked segments,
# contrastive learning between augmented views) then fine-tune on your small
# labelled task. Recent "EEG **foundation models**" (e.g. BENDR, BIOT, LaBraM,
# Neuro-GPT) follow this recipe. They are beyond a 101 course, but the takeaway is:
# *if you have little labelled data, pre-training on unlabelled data often helps more
# than a fancier classifier.*

# %% [markdown]
# ## ⚠️ Common mistakes / why this is wrong
#
# - **Standardising with statistics from the whole dataset.** Compute mean/std on
#   **train only** (as we did), then apply to test. Otherwise you leak.
# - **Random-shuffling epochs into train/test.** Same leakage as classical ML —
#   split by subject (or block). Deep models leak *just as eagerly*.
# - **Comparing to classical baselines unfairly.** On small EEG datasets, CSP+LDA
#   or Riemannian methods often **beat** deep nets. Always include a classical
#   baseline (Chapter 06) before claiming a deep model is better.
# - **Not seeding.** Deep training is noisy; report mean ± std over seeds for a
#   fair comparison (we use one seed here only to stay fast).
# - **Training to a fixed epoch count and reporting the best test accuracy seen.**
#   That peeks at test — use a validation split for early stopping.
#
# **Next:** Chapter 08 — a tour of BCI paradigms (P300, SSVEP, sleep, seizure),
# one minimal runnable example each.
