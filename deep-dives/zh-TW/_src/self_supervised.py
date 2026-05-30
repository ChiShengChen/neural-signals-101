# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度探討 — 腦電圖的自監督學習（Self-Supervised Learning for EEG）
#
# 以前置任務（pretext task）在*無標籤*腦電圖上預訓練小型編碼器（encoder），
# 再以少量有標籤資料進行微調（fine-tune）——此即腦電圖「基礎模型（foundation models）」的核心概念。
#
# > **前置知識：** 主課程第 09 章。
# > **難度：** 進階 ★★★★☆
# > **腦電圖「基礎模型」背後的概念。不受 5 分鐘預算限制。**

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

# ── 可重現性（Reproducibility）────────────────────────────────────────────────
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
# ## 第一部分 — 為何要對腦電圖使用自監督學習？
#
# ### 標籤稀缺問題
#
# 在典型的運動想像（motor-imagery）腦機介面（BCI）研究中，受試者必須靜止不動，
# 依指示想像動作，並在多個實驗回合（session）中重複數百次。每筆有標籤的試次（trial）
# 約需 4–8 秒的精確實驗控制。從單一受試者取得 200 筆有標籤試次，
# 需要 15–30 分鐘的校正時間——而這些標籤僅對*當天*的*該位*受試者有效。
#
# 相較之下，*無標籤*腦電圖則豐富許多：
#
# * 靜息態（resting-state）記錄可連續進行數小時，無需實驗者介入。
# * 睡眠腦電圖研究產生 8 小時的連續資料流。
# * 臨床資料庫存有數百萬病人小時的記錄。
# * 以上所有資料均無需每試次的行為標籤即可收集。
#
# **自監督學習（Self-Supervised Learning，SSL）** 將原始訊號本身的結構轉化為訓練目標。
# *前置任務（pretext task）* 完全由資料本身定義，無需人工標籤，
# 神經編碼器（neural encoder）透過解決此任務而被訓練。若前置任務要求理解腦電圖
# 有意義的時間或頻譜結構，編碼器的內部表示（representation）應能遷移至
# 下游任務，例如睡眠分期（sleep staging）、情緒辨識（emotion recognition）
# 或運動想像。
#
# ### 典型腦電圖前置任務
#
# | 前置任務 | 出處 | 利用的訊號結構 |
# |---|---|---|
# | **相對定位（Relative Positioning，RP）** | Banville et al. 2021 (MOABB, "Uncovering the structure of clinical EEG") | 時間上相近的窗口（window）具有相似的神經狀態；時間上相隔較遠的則否 |
# | **時間順序打亂（Temporal Shuffling，TS）** | Banville et al. 2021 | 三個窗口的正確時間順序可由神經動態預測 |
# | **對比多視角（Contrastive Multi-view，SimCLR-EEG）** | Cheng et al. 2020 | 同一窗口的兩種增強（augmented）視角應有相似的表示 |
# | **遮罩預測（Masked Prediction，BERT-style）** | Kostas et al. 2022 | 被遮罩的通道（channel）片段可由上下文重建 |
#
# 本深度探討實作 **相對定位（RP）**——四者中最簡單，由 Banville et al. 2021 提出。
# 此方法完全不需要類別標籤，可應用於任何連續腦電圖記錄。

# %% [markdown]
# ---
# ## 第二部分 — 資料載入與前置任務建構
#
# 我們載入 BCI IV 2a 資料集（左手 vs. 右手運動想像），但預訓練期間**僅使用原始訊號**。
# 標籤保留至微調（fine-tuning）與評估階段使用。
#
# ### 無洩漏（Leak-free）協定
#
# | 階段 | 使用的資料 | 使用標籤？ |
# |---|---|---|
# | 預訓練（SSL） | 預訓練受試者（無標籤） | **否** |
# | 微調 / 線性探針（linear probe） | 保留受試者訓練集的一小部分 | **是（少量）** |
# | 評估 | 保留受試者測試集的其餘部分 | 僅用於對照真實值 |
#
# 我們將一位受試者完全保留用於下游評估。預訓練過的編碼器從未見過該受試者的試次。

# %%
# ── 資料參數 ──────────────────────────────────────────────────────────────────
N_SUBJ_TOTAL = 3 if SMOKE else 6   # 要載入的受試者總數
# 受試者 1 保留用於下游評估；其餘用於 SSL 預訓練
PRETRAIN_SUBJ_COUNT = max(1, N_SUBJ_TOTAL - 1)

print(f"Loading {N_SUBJ_TOTAL} subject(s) …")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=N_SUBJ_TOTAL)
n_trials, N_CH, N_TIMES = X_all.shape
print(f"  X={X_all.shape}  (trials × channels × time-points)")
print(f"  classes={np.bincount(y_all)}  subjects={np.unique(subj_all).tolist()}")

# 分割：受試者 2..N 用於預訓練，受試者 1 用於下游評估
subj_ids   = np.unique(subj_all)
pretrain_subjects = subj_ids[1:]        # 受試者 2、3、…
eval_subject      = subj_ids[0]         # 受試者 1 — 預訓練期間從未見過

mask_pretrain = np.isin(subj_all, pretrain_subjects)
mask_eval     = subj_all == eval_subject

X_pretrain = X_all[mask_pretrain]       # 形狀 (n_pre, ch, times)
X_eval     = X_all[mask_eval]           # 形狀 (n_eval, ch, times)
y_eval     = y_all[mask_eval]

print(f"\nPre-train pool : {X_pretrain.shape[0]} trials  (subjects {pretrain_subjects.tolist()})")
print(f"Eval subject   : {X_eval.shape[0]} trials  (subject {eval_subject})")

# ── 以通道（channel）為單位正規化（使用預訓練統計量）──────────────────────────
mu_pre  = X_pretrain.mean(axis=(0, 2), keepdims=True)
sig_pre = X_pretrain.std(axis=(0, 2),  keepdims=True) + 1e-8
X_pretrain_n = (X_pretrain - mu_pre) / sig_pre

# 對評估資料套用相同的正規化（無洩漏——mu/sig 來自預訓練池）
X_eval_n = (X_eval - mu_pre) / sig_pre

# %% [markdown]
# ### 建構相對定位（RP）配對
#
# **演算法（Banville et al. 2021 §2.1）：**
#
# 1. 從某試次中隨機選取一個長度為 $L$ 的「錨點（anchor）」窗口 $w_a$。
# 2. 取樣第二個窗口 $w_b$：
#    - **正例（Positive）**（相近）：$w_b$ 的起始位置在 $w_a$ 的 $\tau_{\text{pos}}$ 個取樣點範圍內。
#    - **負例（Negative）**（相遠）：$w_b$ 的起始位置至少距 $w_a$ $\tau_{\text{neg}}$ 個取樣點。
# 3. 二元標籤為 1（相近）或 0（相遠）。
# 4. 關鍵：**完全不使用運動想像標籤**。錨點與第二個窗口可來自任何試次，
#    不限條件（condition）。

# %%
def make_rp_pairs(
    X: np.ndarray,
    n_pairs: int,
    win_len: int,
    tau_pos: int,
    tau_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """取樣相對定位（relative-positioning）（錨點、樣本、標籤）三元組。

    Parameters
    ----------
    X       : (n_trials, n_ch, n_times)
    n_pairs : 要生成的配對數量
    win_len : 窗口長度（取樣點數）
    tau_pos : 「相近」正例配對的最大時間偏移
    tau_neg : 「相遠」負例配對的最小時間偏移
    rng     : numpy 隨機數生成器（Generator）

    Returns
    -------
    anc  : (n_pairs, n_ch, win_len)
    samp : (n_pairs, n_ch, win_len)
    lbl  : (n_pairs,) int — 1=相近, 0=相遠
    """
    n_trials, n_ch, n_times = X.shape
    max_start = n_times - win_len

    if max_start <= 0:
        raise ValueError(f"win_len={win_len} too long for n_times={n_times}")

    anc_list, samp_list, lbl_list = [], [], []

    for _ in range(n_pairs):
        # 隨機選取試次與錨點起始位置
        trial_idx = rng.integers(0, n_trials)
        a_start   = rng.integers(0, max_start + 1)

        if rng.random() < 0.5:
            # 正例配對：在錨點的 tau_pos 範圍內取樣
            low  = max(0,          a_start - tau_pos)
            high = min(max_start,  a_start + tau_pos)
            b_start = int(rng.integers(low, high + 1))
            label   = 1
        else:
            # 負例配對：至少距錨點 tau_neg 取樣點
            # 建立遠離錨點的有效範圍
            candidates = []
            if a_start - tau_neg >= 0:
                candidates.append((0, a_start - tau_neg))
            if a_start + tau_neg <= max_start:
                candidates.append((a_start + tau_neg, max_start))
            if not candidates:
                # 邊界情況：試次過短——退回至遠端或起始處
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

# ── 前置任務超參數（hyper-parameters）────────────────────────────────────────
WIN_LEN   = min(100, N_TIMES)          # 100 個取樣點 ≈ 250 Hz 下的 400 ms
TAU_POS   = max(10, WIN_LEN // 4)      # 「相近」= 在 1/4 窗口範圍內
TAU_NEG   = max(WIN_LEN // 2, TAU_POS + 10)  # 「相遠」= 超過半個窗口距離
N_PAIRS   = 500 if SMOKE else 3000

anc_np, samp_np, lbl_np = make_rp_pairs(
    X_pretrain_n, N_PAIRS, WIN_LEN, TAU_POS, TAU_NEG, rng
)
print(f"RP pairs: {anc_np.shape}  |  pos={lbl_np.sum()}  neg={(lbl_np==0).sum()}")

# 轉換為張量（tensor）
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
# ## 第三部分 — 編碼器架構與前置任務預訓練
#
# ### 編碼器（Encoder）
#
# 為使其能在 CPU 上快速執行，我們刻意保持編碼器非常小。
# 對時間維度進行一疊 1-D 卷積層（每個通道獨立處理），
# 再接全域平均池化（global average pooling），無論輸入長度如何，
# 均可產生固定長度的嵌入向量（embedding）。
#
# ### 前置任務頭部（Pretext Head）
#
# 給定錨點與樣本的嵌入 $z_a$ 和 $z_b$，RP 頭部（head）將兩者拼接（concatenate）
# 後預測相近（1）或相遠（0）：
#
# $$\hat{y} = \sigma(W_2\,\text{ReLU}(W_1 [z_a \| z_b]))$$
#
# 編碼器與頭部在 RP 損失（loss）上聯合訓練。預訓練後頭部被捨棄；
# 僅重複使用編碼器的權重。

# %%
class TinyEEGEncoder(nn.Module):
    """用於腦電圖窗口的小型卷積編碼器（conv encoder）。

    Input : (batch, n_ch, win_len)
    Output: (batch, embed_dim)
    """

    def __init__(self, n_ch: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 時間卷積區塊（Temporal conv block）1
            nn.Conv1d(n_ch, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),          # 時間維度減半
            # 時間卷積區塊 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),          # 再次減半
            # 時間卷積區塊 3
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
    """相對定位二元分類頭部（head）：預測相近（1）或相遠（0）。"""

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

# ── 預訓練迴圈（Pre-training loop）───────────────────────────────────────────
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
# ### 圖一 — 預訓練損失曲線
#
# 下降的損失曲線代表編碼器正在學習區分「時間上相近」與「時間上相遠」——
# 它在從未接觸行為標籤的情況下，建立了能捕捉腦電圖時間自相關（temporal autocorrelation）
# 結構的表示。
#
# RP 準確率完美會達到 100%（即簡單地過擬合（overfit）配對）；
# 我們希望編碼器能夠泛化（generalise），而非記憶配對。

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
# ## 第四部分 — 下游評估：線性探針（linear probe）vs. 從頭訓練（from-scratch）
#
# ### 協定
#
# 我們在保留受試者（受試者 1，預訓練期間從未見過）上進行評估。
# 對每個有標籤試次預算 $K \in \{4, 8, 16, 32, \ldots\}$：
#
# 1. **預訓練（線性探針）：** 凍結（freeze）SSL 編碼器，僅在 $K$ 筆有標籤試次上
#    訓練一個 2 類線性頭部（linear head）。
# 2. **從頭訓練：** 在相同的 $K$ 筆有標籤試次上，以隨機初始化（random initialization）
#    訓練*相同*編碼器架構（編碼器 + 線性頭部，所有權重均可訓練）。
# 3. 在*剩餘*的保留試次上評估兩者。
#
# 若 SSL 預訓練有幫助，在小 $K$ 時，預訓練的線性探針應優於從頭訓練的模型——
# 因為後者僅有極少量有標籤樣本可學習。

# %%
# ── 下游資料：保留受試者 ──────────────────────────────────────────────────────
# 依時間順序分割：前 60% 作為有標籤選取池，後 40% 作為測試集。
n_eval = len(y_eval)
split_pt = int(0.6 * n_eval)

X_pool = X_eval_n[:split_pt]
y_pool = y_eval[:split_pt]
X_test = X_eval_n[split_pt:]
y_test = y_eval[split_pt:]

# 擷取下游任務的完整試次窗口（使用中間 WIN_LEN 個取樣點）
start_ds = (N_TIMES - WIN_LEN) // 2
X_pool_w = X_pool[:, :, start_ds : start_ds + WIN_LEN].astype(np.float32)
X_test_w = X_test[:, :, start_ds : start_ds + WIN_LEN].astype(np.float32)

print(f"Pool for labeled selection : {X_pool_w.shape}  (classes={np.bincount(y_pool)})")
print(f"Test set (never used in fit): {X_test_w.shape}  (classes={np.bincount(y_test)})")

Xte_t_ds = torch.from_numpy(X_test_w)
yte_np    = y_test


# ── 標籤預算（label budget）掃描 ──────────────────────────────────────────────
def get_balanced_subset(X, y, n_per_class, rng):
    """從每個類別中隨機取樣至多 n_per_class 筆資料並回傳。"""
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
    """訓練線性頭部（或全面微調）並回傳測試準確率（test accuracy）。"""
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


# ── 掃描各標籤預算 ────────────────────────────────────────────────────────────
if SMOKE:
    BUDGETS = [2, 4, 8]
    FINETUNE_EPOCHS = 10
    SCRATCH_EPOCHS  = 10
else:
    # 每類最大數量受限於池中的可用量
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

# 建立一個全新的編碼器（相同架構，隨機權重）作為從頭訓練的基準線
scratch_encoder_init = TinyEEGEncoder(n_ch=N_CH, embed_dim=EMBED_DIM)

acc_pretrained = []
acc_scratch    = []

for n_per_class in BUDGETS:
    X_sub, y_sub = get_balanced_subset(X_pool_w, y_pool, n_per_class, rng)

    # -- 預訓練線性探針（Pretrained linear probe）--
    acc_pt = evaluate_encoder_probe(
        encoder_model   = encoder,           # SSL 預訓練權重
        X_tr            = X_sub,
        y_tr            = y_sub,
        X_te_tensor     = Xte_t_ds,
        y_te            = yte_np,
        freeze_encoder  = True,              # 僅線性探針
        n_epochs        = FINETUNE_EPOCHS,
        lr              = PROBE_LR,
        device          = DEVICE,
        seed            = SEED,
    )

    # -- 從頭訓練（相同架構，隨機初始化，完整訓練）--
    acc_sc = evaluate_encoder_probe(
        encoder_model   = scratch_encoder_init,
        X_tr            = X_sub,
        y_tr            = y_sub,
        X_te_tensor     = Xte_t_ds,
        y_te            = yte_np,
        freeze_encoder  = False,             # 從頭訓練編碼器 + 頭部
        n_epochs        = SCRATCH_EPOCHS,
        lr              = SCRATCH_LR,
        device          = DEVICE,
        seed            = SEED,
    )

    acc_pretrained.append(acc_pt)
    acc_scratch.append(acc_sc)
    print(f"  K={n_per_class:3d}/class  |  pretrained={acc_pt:.3f}  |  scratch={acc_sc:.3f}")

# %% [markdown]
# ### 圖二 — 預訓練 vs. 從頭訓練準確率隨標籤預算的變化
#
# 此為本筆記本的核心圖表。每個點代表在保留受試者上的測試準確率。
# X 軸為可用於微調的每類有標籤試次數量。
#
# **觀察重點：**
# * 在最小標籤預算（最左側），預訓練的線性探針是否優於從頭訓練的模型？
#   這是 SSL 預訓練最有幫助的情境——即使在見到任何標籤之前，
#   編碼器已能提取結構化特徵。
# * 隨著 K 增大，從頭訓練的模型應逐漸接近或追上預訓練模型
#   （有足夠的資料後，它可以自行學習特徵）。
# * **注意：** 使用微型編碼器、少量預訓練配對、且只有單一評估受試者時，
#   效果可能不顯著或有噪音。在本教學規模下，管道（plumbing）的正確性
#   比確切的數字更重要。

# %%
fig, ax = plt.subplots(figsize=(8, 5))

label_counts = [2 * b for b in BUDGETS]   # 總標籤數 = 2 × n_per_class

ax.plot(label_counts, acc_pretrained, "o-", color="#2ca02c", linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="SSL pretrained → linear probe")
ax.plot(label_counts, acc_scratch, "s--", color="#d62728", linewidth=2.0,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="From-scratch encoder + head")
ax.axhline(0.5, color="gray", lw=1.2, ls=":", label="Chance (50 %)")

# 標注數值
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

# 印出最小預算的結果作為回傳值
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
# ## 第五部分 — 誠實的注意事項（呼應第 09 章）
#
# ### 本示範教了什麼——以及它沒教什麼
#
# | 本示範展示的內容 | 本示範未展示的內容 |
# |---|---|
# | 如何從頭實作 RP 前置任務 | 嚴謹的 SSL 基準測試 |
# | 管道結構：編碼器 → 前置任務 → 遷移 | 統計顯著性（單一隨機種子） |
# | 該想法能在 CPU 上編譯並執行 | SSL 總能在腦電圖上勝過監督學習 |
# | 無洩漏的預訓練／評估分割 | 效應量（effect size）的可重現性 |
#
# **單一隨機種子結果：** 本筆記本執行一次僅為單點估計，無誤差棒（error bar）。
# 在小 K 時，隨機種子、受試者與資料集之間的變異相當大。
# Banville et al. 2021 對 11 個資料集執行 5 個隨機種子。
#
# **微型規模：** 我們的編碼器約有 5 萬個參數，以數百個配對預訓練不超過 15 個輪次（epoch）。
# 真正的腦電圖 SSL 模型使用數百萬個參數，並在大型語料庫上預訓練數小時。
# 在教學規模下，預訓練模型可能只略微（甚至有時無法）優於從頭訓練——
# 這是預期中的誠實結果。
#
# **標籤預算的粒度：** 保留受試者提供約 100 個試次（每類約 50 個），
# 「小 K」區間取樣不足。健全的研究應對標籤子集進行 10–20 次隨機抽取，
# 並回報均值 ± 標準差。
#
# **前置任務品質：** RP 是最簡單的前置任務。時間順序打亂（temporal shuffling）、
# 對比多視角（contrastive multi-view）與遮罩預測（masked prediction）
# 通常能學習更豐富的表示。關於腦電圖「基礎模型」，請參考：
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
# ### 看不見的捷徑（The invisible shortcut）
#
# 相對定位任務要求編碼器預測兩個腦電圖窗口在時間上是相近還是相遠。
# 這*聽起來*像是迫使編碼器學習神經動態。但至少有兩種生理性混淆因素（physiological confound）
# 讓編碼器不需學習任何具有神經科學意義的內容，就能解決 RP 任務：
#
# ### 混淆因素 1 — 電極漂移（electrode drift）與緩慢基線游移（slow-baseline wandering）
#
# 腦電圖放大器採直流耦合（DC-coupled）（或具有非常長的時間常數）。
# 在試次或實驗回合的過程中，皮膚-電極阻抗變化產生**緩慢的電皮膚漂移（electro-dermal drift）**，
# 使每個通道的基線電壓在數十秒內偏移微伏至數十微伏。此漂移具有高度的時間自相關性：
# 相隔 50 毫秒的兩個窗口幾乎具有相同的漂移水準；
# 相隔 30 秒的兩個窗口則可能差異顯著。
#
# **後果：** 編碼器可透過學習比較每個窗口的*平均電壓偏移量*，達到高 RP 準確率。
# 相近的窗口有相似的偏移量；相遠的則否。學到的特徵不是振盪性或事件相關的——
# 它們是緩慢漂移特徵。當此編碼器用於運動想像分類（依賴 alpha/beta 功率調製，
# 而非直流電平）時，預訓練特徵可能*不如*從頭針對實際任務學習的特徵有用。
#
# **偵測方法：** 比較資料**高通濾波（high-pass filtered）**至 1 Hz（去除漂移）
# 與原始資料的 RP 準確率。若去除漂移導致 RP 準確率大幅下降，
# 則編碼器依賴的是漂移而非神經動態。
# `artifact_confounds` 深度探討涵蓋了偽影驅動解碼的更廣泛模式。
#
# ### 混淆因素 2 — 市電雜訊（line-noise）振幅調製
#
# 每當實驗回合中電極阻抗升高，電力線干擾（50/60 Hz）及其諧波就會出現在腦電圖中。
# 市電雜訊成分的振幅通常隨著實驗回合推進而*漂移*（因電極凝膠乾燥）——
# 這意味著時間上相近的兩個窗口有相似的市電雜訊振幅，
# 而時間上相遠的兩個窗口則否。學習以 50 Hz 振幅作為時間距離代理的卷積編碼器，
# 雖能很好地解決 RP 任務，卻學習的是偽影特徵，而非神經特徵。
#
# **偵測方法：** 在預訓練前對 50/60 Hz 進行陷波濾波（notch filter）。
# 若 RP 準確率顯著下降，則市電雜訊振幅就是捷徑。
#
# ### 關鍵洞察（交叉參考 `artifact_confounds.py`）
#
# 兩種混淆因素具有相同的根本原因：**任何在比 $\tau_{\text{neg}}$ 更短尺度上
# 具有時間自相關性、但在更長尺度上有所變化的訊號屬性，都能解決 RP 任務**。
# 這包括：
#
# * 電極漂移（數秒至數分鐘）
# * 市電雜訊振幅調製（數分鐘）
# * 動作相關的肌電（EMG）包絡（數秒）
# * 呼吸相關的緩慢腦波（週期約 4 秒）
#
# **以上均非神經訊號。**
#
# ### 正確的緩解措施
#
# 1. **預訓練前進行預處理：** 1 Hz 高通濾波，對市電頻率進行陷波濾波，
#    對大型偽影進行試次級別的排除。與監督學習相同的清理流程。
# 2. **謹慎選擇 $\tau_{\text{pos}}$ 和 $\tau_{\text{neg}}$：** 若 $\tau_{\text{neg}}$
#    大於漂移的時間尺度，RP 主要是一個漂移偵測任務。
#    對於 mu/beta 神經動態，使用 $\tau_{\text{neg}} \leq 2$ 秒。
# 3. **驗證學到的特徵：** 預訓練後，計算嵌入向量與已知偽影通道（EOG、EMG、緩慢漂移）
#    之間的相關性。若嵌入的第一主成分與 EOG 振幅相關，
#    則編碼器學習的是眼動特徵，而非運動想像特徵。
# 4. **跨受試者預訓練：** 若編碼器在不同於評估受試者的受試者上預訓練，
#    則受試者特定的漂移模式無法被記憶——這正是我們上面所使用的無洩漏協定。
#
# **元教訓（meta-lesson）：** SSL 並不能免疫困擾監督式腦電圖解碼的偽影混淆因素。
# 前置任務定義了「有用結構」的含義，若該結構主要由緩慢偽影主導，
# 則預訓練特徵將捕捉到偽影。輸入乾淨的資料 → 產出有用的表示。
