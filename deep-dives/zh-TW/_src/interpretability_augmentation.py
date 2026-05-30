# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 解釋 EEG 神經網路 & 資料增強（Data Augmentation）
#
# EEGNet 究竟在「看」什麼？我們又如何透過精心設計的增強策略，
# 從少量標記資料中榨出更多訊號？
#
# > **先修條件：** 主課程第 09 章。
# > **難度：** 進階 ★★★★☆
# > **不受 5 分鐘 CPU 預算限制。**

# %% Bootstrap — 若尚未安裝為套件，則從 repo src 匯入 neuro101
import sys
import os
from pathlib import Path

# 穩健的向上搜尋：先嘗試已安裝版本，再逐層往上找 src/neuro101
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

# 重現性設定
torch.manual_seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)

DEVICE = "cpu"
SMOKE = ds.is_smoke()
print(f"Smoke mode : {SMOKE}  |  torch {torch.__version__}  |  device={DEVICE}")

# %% [markdown]
# ---
# ## 第 A 部分 — 可解釋性：EEGNet 究竟用了什麼特徵？
#
# EEGNet（Lawhern et al. 2018）是一個為 EEG 分類而設計的緊湊卷積架構（convolutional architecture）。
# 其三個區塊——時間卷積（temporal convolution）、深度空間卷積（depthwise spatial convolution）
# 與可分離卷積（separable convolution）——小到可以在幾分鐘內於 CPU 上完成訓練。
# 但「它能分類」和「它基於正確原因進行分類」並不相同。
# 我們將在 BCI IV 2a（左 vs 右運動想像）上訓練一個小型 EEGNet，
# 然後套用**梯度式顯著圖（gradient-based saliency）**：哪些電極和哪些時刻
# 驅動了模型的信心？

# %% [markdown]
# ### 步驟 1 — 載入資料（1–2 位受試者，左 vs 右運動想像）

# %%
N_SUBJ = 1 if SMOKE else 2
print(f"正在從 BCI IV 2a 載入 {N_SUBJ} 位受試者資料 …")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=N_SUBJ)
print(f"  X={X_all.shape}  classes={np.bincount(y_all)}  dtype={X_all.dtype}")

# BCI IV 2a 電極順序（MOABB 移除 EOG/刺激通道後，保留 22 個 EEG 通道）
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
SFREQ      = 250.0             # Hz（BCI IV 2a 取樣率）
print(f"  Channels={N_CHANNELS}  Times={N_TIMES}  sfreq={SFREQ} Hz")

# %% [markdown]
# ### 步驟 2 — 誠實的訓練／測試分割（標準化僅依訓練集計算）

# %%
# 使用單一受試者的資料；以分層 70/30 分割（stratified split）保持類別平衡。
s_mask = (subj_all == subj_all.min())
X_s, y_s = X_all[s_mask], y_all[s_mask]

# 分層分割：對每個類別取前 70% 作為訓練集，其餘作為測試集。
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

# 標準化（standardisation）：以訓練集計算各通道的零均值、單位變異數。
mu  = X_tr_raw.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
sig = X_tr_raw.std(axis=(0, 2), keepdims=True) + 1e-8
X_tr = (X_tr_raw - mu) / sig
X_te = (X_te_raw - mu) / sig   # 以訓練集統計量套用於測試集

print(f"Train: {X_tr.shape}  classes={np.bincount(y_tr)}")
print(f"Test : {X_te.shape}  classes={np.bincount(y_te)}")

# 轉換為 float32 PyTorch 張量（tensor）；EEGNet 期望 (batch, 1, channels, times)
def to_tensor(X, y):
    # braindecode v0.8 期望 (batch, channels, times)——不需要額外維度
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    return Xt, yt

Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
Xte_t, yte_t = to_tensor(X_te, y_te)
print(f"Tensor shape: {Xtr_t.shape}  (batch, channels, times)")

# %% [markdown]
# ### 步驟 3 — 建構並訓練一個小型 EEGNet
#
# 我們使用 braindecode 的 `EEGNetv4`，配置極小的超參數（`F1=4, D=2, F2=8`），
# 使其能在幾秒內於 CPU 上完成訓練。

# %%
from braindecode.models import EEGNetv4

def build_eegnet(n_chans: int, n_times: int, n_classes: int = 2) -> nn.Module:
    """建構可在一分鐘內於 CPU 訓練完畢的小型 EEGNetv4。

    braindecode v0.8 的 EEGNetv4 期望輸入形狀為 (batch, channels, times)。
    """
    model = EEGNetv4(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        F1=4,
        D=2,
        F2=8,
        kernel_length=32,   # 在 250 Hz 下約為 128 ms
        drop_prob=0.25,
        final_conv_length="auto",
    )
    return model

model = build_eegnet(N_CHANNELS, N_TIMES)
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"EEGNetv4: {n_params:,} 個參數")

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
        print(f"  第 {epoch:3d}/{EPOCHS} 輪  loss={epoch_loss:.4f}")

# 在測試集上評估
model.eval()
with torch.no_grad():
    logits_te = model(Xte_t.to(DEVICE))
    preds_te  = logits_te.argmax(dim=1).cpu().numpy()
baseline_acc = (preds_te == y_te).mean()
print(f"\n基準測試準確率（無增強，完整訓練集）：{baseline_acc:.3f}")

# %% [markdown]
# ### 步驟 4 — 梯度式顯著圖（gradient-based saliency）：|d(logit)/d(input)|
#
# 最簡單的可解釋性方法：對每個測試試次（trial），計算**獲勝類別的 logit
# 對每個輸入樣本的絕對梯度**。梯度越大 → 模型信心對該（通道, 時間點）位置越敏感。
# 我們對所有測試試次取平均，得到穩定的重要性地圖（importance map）。
#
# **重要注意：** 這告訴我們損失**曲面（loss surface）**的陡峭程度，
# 不一定反映網路「決策」的依據。詳見末尾的 ⚠️ 小節。

# %%
def compute_saliency(model: nn.Module, X_tensor: torch.Tensor) -> np.ndarray:
    """回傳對所有試次取平均的 |d logit_pred / d input|。

    Returns
    -------
    saliency : np.ndarray  shape (n_channels, n_times)
        對試次維度取平均後的絕對梯度均值。
    """
    model.eval()
    all_grads = []
    for i in range(len(X_tensor)):
        x = X_tensor[i:i+1].clone().requires_grad_(True).to(DEVICE)
        logits = model(x)
        # 計算對 argmax 類別（預測類別）的梯度
        pred_class = logits.argmax(dim=1).item()
        score = logits[0, pred_class]
        score.backward()
        grad = x.grad.detach().cpu().numpy()   # (1, C, T)  — braindecode v0.8
        all_grads.append(np.abs(grad[0]))      # (C, T)

    saliency = np.mean(all_grads, axis=0)      # (C, T)
    return saliency

print("正在計算測試集的顯著圖 …")
saliency_map = compute_saliency(model, Xte_t)
print(f"顯著圖形狀：{saliency_map.shape}  (channels, times)")

# 各通道重要性：對時間維度取平均
ch_importance = saliency_map.mean(axis=1)   # (C,)
rank_order    = np.argsort(ch_importance)[::-1]

print("\n依顯著度排列前 10 名通道：")
for r, ci in enumerate(rank_order[:10]):
    print(f"  {r+1:2d}. {CH_NAMES[ci]:<6s}  importance={ch_importance[ci]:.4f}")

# %% [markdown]
# ### 視覺化 1 — 各通道顯著度（長條圖）
#
# 對於真實的運動想像（motor imagery），網路應著重**中央感覺運動通道**
#（C3、Cz、C4 及其鄰近的 FC/CP 通道），因為 8–30 Hz 的 mu/beta 頻帶
# ERD/ERS（事件相關去同步化／同步化）在手部區域（Brodmann area 4）最強。
#
# 若模型反而著重額葉通道（Fz、FC3/FC4）或周邊通道（POz、Pz），
# 則可能追蹤的是**眼動偽跡（eye-movement artifacts）或提示相關電位（cue-related potentials）**，
# 而非真正的運動想像——這是已知的混淆因素（confound）。
# 若發現此情形，請與 artifact_confounds 深度解析交叉比對。

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左圖：依重要性排序的長條圖 ---
ax = axes[0]
sorted_imp  = ch_importance[rank_order]
sorted_names = [CH_NAMES[i] for i in rank_order]
colors = []
for nm in sorted_names:
    if nm in ("C3", "Cz", "C4", "C1", "C2", "C5", "C6",
               "CP1", "CPz", "CP2", "CP3", "CP4",
               "FC1", "FCz", "FC2", "FC3", "FC4"):
        colors.append("#2ecc71")    # 感覺運動通道——符合預期
    elif nm in ("Fz", "FC5", "FC6"):
        colors.append("#e74c3c")    # 額葉通道——疑似偽跡
    else:
        colors.append("#3498db")    # 其他通道

bars = ax.barh(range(len(sorted_names)), sorted_imp[::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names[::-1], fontsize=7)
ax.set_xlabel("平均 |梯度|", fontsize=10)
ax.set_title("各通道梯度顯著度\n（綠色 = 感覺運動，紅色 = 額葉／易受偽跡影響）", fontsize=9)

# 圖例
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#2ecc71", label="感覺運動（符合預期）"),
    Patch(facecolor="#e74c3c", label="額葉（易受偽跡影響）"),
    Patch(facecolor="#3498db", label="其他"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

# --- 右圖：2D 熱力圖（通道 × 時間）---
ax2 = axes[1]
times_ms = np.linspace(500, 2500, N_TIMES)   # 提示後 0.5–2.5 秒
im = ax2.imshow(
    saliency_map,
    aspect="auto",
    origin="upper",
    extent=[times_ms[0], times_ms[-1], N_CHANNELS - 0.5, -0.5],
    cmap="hot",
)
ax2.set_yticks(range(N_CHANNELS))
ax2.set_yticklabels(CH_NAMES, fontsize=6)
ax2.set_xlabel("提示後時間 (ms)", fontsize=10)
ax2.set_title("顯著度熱力圖（通道 × 時間）\n亮色 = 高重要性", fontsize=9)
plt.colorbar(im, ax=ax2, label="平均 |梯度|")

fig.suptitle(
    f"EEGNet 梯度顯著度 — BCI IV 2a（左 vs 右運動想像）\n"
    f"測試準確率：{baseline_acc:.1%}   |   {N_SUBJ} 位受試者",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/dd_int_saliency.png", dpi=100, bbox_inches="tight")
plt.show()
print("顯著度圖已儲存。")

# %% [markdown]
# ### 步驟 5 — 各時間點重要性
#
# 將顯著度對通道維度取平均，得到模型注意力的時間輪廓。
# 我們預期**持續的運動想像窗口**（提示後約 0.5–2 秒）的重要性較高，
# 而非 epoch 的最初（提示鎖定的視覺反應）或最末。

# %%
time_importance = saliency_map.mean(axis=0)   # (T,)
times_ms = np.linspace(500, 2500, N_TIMES)

fig, ax = plt.subplots(figsize=(9, 3))
ax.fill_between(times_ms, time_importance, alpha=0.4, color="#8e44ad")
ax.plot(times_ms, time_importance, color="#8e44ad", lw=1.5)
ax.set_xlabel("提示後時間 (ms)")
ax.set_ylabel("各通道平均 |梯度|")
ax.set_title("時間顯著度輪廓 — EEGNet 何時「注意」？")
ax.axvspan(500, 2500, alpha=0.05, color="gray", label="運動想像 epoch 窗口")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_int_time_saliency.png", dpi=100, bbox_inches="tight")
plt.show()
print("時間顯著度圖已儲存。")

# %% [markdown]
# ---
# ## 第 B 部分 — 小樣本 EEG 資料的增強策略
#
# BCI 資料集採集成本高昂；典型研究每位受試者僅有 100–300 個標記試次——
# 在此資料量下，深度網路極易過擬合（overfit）。
# 增強（augmentation）透過對現有試次施加**符合生理意義**的擾動，
# 創造額外的合成訓練樣本。
# 擾動幅度應小到不改變類別標籤，但足夠多樣以發揮正則化（regularisation）效果。
#
# ### 我們實作的四種增強方法
#
# | # | 名稱 | 操作 | 理由 |
# |---|------|------|------|
# | a | **時間平移（Time shift）** | 以隨機 ±τ 個取樣點滾動訊號 | 運動想像訊號並非精確鎖定於提示 |
# | b | **高斯雜訊（Gaussian noise）** | 對每個樣本加入 i.i.d. N(0, σ²) | 模擬電極雜訊與皮膚阻抗漂移 |
# | c | **通道／頻率遮罩（Channel/frequency masking）** | 隨機將一個通道或一個 4 Hz 頻帶歸零 | 強迫空間與頻譜的穩健性 |
# | d | **Mixup** | 兩個試次的凸組合（convex combination）+ 軟標籤 | 正則化輸出層；源自電腦視覺（CV） |

# %% [markdown]
# ### 步驟 6 — 實作增強函式

# %%
def aug_time_shift(X: np.ndarray, max_shift: int = 25, rng=None) -> np.ndarray:
    """對每個試次獨立地以 ±max_shift 範圍內的隨機取樣點數滾動訊號。

    X : (N, C, T)
    """
    rng = rng or np.random.default_rng()
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(X))
    out = np.empty_like(X)
    for i, s in enumerate(shifts):
        out[i] = np.roll(X[i], s, axis=-1)
    return out


def aug_gaussian_noise(X: np.ndarray, sigma: float = 0.05, rng=None) -> np.ndarray:
    """加入標準差為 sigma * signal_std 的零均值高斯雜訊。"""
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal(X.shape).astype(X.dtype)
    # 雜訊幅度相對於各通道標準差縮放（已標準化，故約為 1）
    return X + sigma * noise


def aug_channel_freq_mask(X: np.ndarray, sfreq: float = 250.0,
                          p_channel: float = 0.5, rng=None) -> np.ndarray:
    """以機率 p_channel 遮罩一個隨機通道，否則將一個隨機 4 Hz 頻帶歸零。

    以機率 p_channel 遮罩一個通道；否則將一個 4 Hz 頻帶歸零。
    """
    rng = rng or np.random.default_rng()
    out = X.copy()
    N, C, T = out.shape
    for i in range(N):
        if rng.random() < p_channel:
            # 通道遮罩
            ch = rng.integers(0, C)
            out[i, ch, :] = 0.0
        else:
            # 透過 FFT 進行頻帶遮罩
            # 在訊號範圍內（至 Nyquist）隨機選取一個 4 Hz 頻帶
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
    """Mixup 增強：試次對的凸組合 + 軟標籤（soft labels）。

    Returns X_mix (N, C, T) 以及 y_mix (N, 2) 的軟性 one-hot 向量。
    """
    rng   = rng or np.random.default_rng()
    N     = len(X)
    n_cls = int(y.max()) + 1
    lam   = rng.beta(alpha, alpha, size=N).astype(np.float32)  # (N,)
    perm  = rng.permutation(N)

    X_mix = (lam[:, None, None] * X +
             (1 - lam[:, None, None]) * X[perm]).astype(X.dtype)

    # 軟性 one-hot 標籤
    y_oh      = np.eye(n_cls, dtype=np.float32)[y]
    y_perm_oh = np.eye(n_cls, dtype=np.float32)[y[perm]]
    y_mix     = (lam[:, None] * y_oh +
                 (1 - lam[:, None]) * y_perm_oh)
    return X_mix, y_mix


def apply_augmentation(X: np.ndarray, y: np.ndarray, rng=None,
                       do_mixup: bool = True):
    """套用所有增強方法，並與原始資料合併串接。

    回傳 X_aug、y_aug（硬標籤）；Mixup 試次以軟標籤的
    *多數類別*（argmax）指定，以與標準交叉熵相容。
    """
    rng = rng or np.random.default_rng(0)
    X_aug_list = [X]
    y_aug_list = [y]

    # a) 時間平移
    X_aug_list.append(aug_time_shift(X, rng=rng))
    y_aug_list.append(y.copy())

    # b) 高斯雜訊
    X_aug_list.append(aug_gaussian_noise(X, rng=rng))
    y_aug_list.append(y.copy())

    # c) 通道／頻率遮罩
    X_aug_list.append(aug_channel_freq_mask(X, rng=rng))
    y_aug_list.append(y.copy())

    # d) Mixup
    if do_mixup:
        X_mx, y_soft = aug_mixup(X, y, rng=rng)
        X_aug_list.append(X_mx)
        y_aug_list.append(y_soft.argmax(axis=1))  # 硬標籤（argmax）

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)
    return X_out, y_out

print("增強函式已定義。")
# 快速冒煙測試（smoke check）
_X_chk, _y_chk = apply_augmentation(X_tr[:10], y_tr[:10], rng=rng)
print(f"  apply_augmentation: {X_tr[:10].shape} → {_X_chk.shape}")

# %% [markdown]
# ### 步驟 7 — 比較有／無增強在小樣本子集上的訓練效果
#
# 我們刻意只使用**一小部分訓練集**（預設 30 個試次，冒煙模式下為 16 個）
# 來壓力測試增強效果。保留測試集與上方相同的 30% 分割，確保比較公平。
#
# 我們以 `N_SEEDS` 個隨機種子重複實驗，比較均值與離散程度。
# 第 09 章提醒：單一種子只能驗證管線正確性——這裡使用數個種子，
# 以誠實反映變異程度，但試次數如此之少，變異本就偏高。

# %%
# 冒煙模式下保持最精簡，但仍需平衡各類別。
# 從每個類別取前 N_PER_CLASS 個試次（分層小集合）。
N_PER_CLASS  = 8 if SMOKE else 15    # 小訓練集中每類別的試次數
N_SEEDS      = 2 if SMOKE else 4
EPOCHS_AUG   = 5 if SMOKE else 15


def make_small_balanced(X, y, n_per_class, rng_seed=0):
    """回傳平衡的小訓練集（每類別 n_per_class 個試次）。"""
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
    """訓練一個全新的 EEGNet，回傳測試準確率。"""
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


# 建立平衡的小訓練集
X_small, y_small = make_small_balanced(X_tr, y_tr, N_PER_CLASS, rng_seed=42)
SMALL_N = len(y_small)
print(f"小訓練集：{SMALL_N} 個試次  "
      f"（各類別：{np.bincount(y_small.astype(int))}）")
print(f"測試集   ：{len(X_te)} 個試次")
print(f"正在訓練 {N_SEEDS} 個種子 × 2 個條件 …")

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

print(f"\n結果（{N_SEEDS} 個種子）：")
print(f"  無增強  ：{mean_no:.3f} ± {std_no:.3f}")
print(f"  有增強  ：{mean_yes:.3f} ± {std_yes:.3f}")
print(f"  差異    ：{mean_yes - mean_no:+.3f}")
print(f"  機率水準：0.500")

# %% [markdown]
# ### 視覺化 2 — 有／無增強的準確率比較

# %%
fig, ax = plt.subplots(figsize=(7, 5))

labels_bar = ["無增強", "有增強"]
means_bar  = [mean_no, mean_yes]
stds_bar   = [std_no,  std_yes]
colors_bar = ["#e74c3c", "#2ecc71"]
x_pos      = [0, 1]

bars2 = ax.bar(x_pos, means_bar, yerr=stds_bar, capsize=8,
               color=colors_bar, edgecolor="white", width=0.5,
               error_kw=dict(elinewidth=2, ecolor="black"))

# 在長條圖上方以抖動（jitter）方式呈現各種子的個別結果
for xi, accs in enumerate([accs_no_aug, accs_with_aug]):
    jitter = rng.uniform(-0.05, 0.05, size=len(accs))
    ax.scatter(xi + jitter, accs, color="black", s=40, zorder=5, alpha=0.7)

ax.axhline(0.5, color="gray", lw=1.5, linestyle="--", label="機率水準（50%）")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_bar, fontsize=12)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel("測試準確率", fontsize=11)
ax.set_title(
    f"增強對小訓練集的效果\n"
    f"EEGNet · {SMALL_N} 個標記試次 · {N_SEEDS} 個種子 · BCI IV 2a",
    fontsize=10,
)
ax.legend(fontsize=9)

# 在長條上標注均值 ± 標準差文字
for xi, (m, s, col) in enumerate(zip(means_bar, stds_bar, colors_bar)):
    ax.text(xi, m + s + 0.015, f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/dd_int_augmentation.png", dpi=100, bbox_inches="tight")
plt.show()
print("增強比較圖已儲存。")
print(f"\n[摘要]  no-aug={mean_no:.3f}  with-aug={mean_yes:.3f}  delta={mean_yes-mean_no:+.3f}")

# %% [markdown]
# ### 詮釋
#
# 僅有約 30 個標記試次時，EEGNet 幾乎沒有足夠資料同時學習空間與時間濾波器。
# 增強作為正則化器：
#
# * **時間平移**防止模型過度擬合運動反應的精確時間偏移量，
#   這在不同試次間自然存在 ±100 ms 的差異。
# * **高斯雜訊**阻止模型記憶個別試次特有的電極雜訊模式。
# * **通道遮罩**強迫空間濾波器之間的冗餘——類似 dropout，
#   但作用於感測器層級。
# * **Mixup** 平滑化左右運動想像之間的決策邊界，
#   減少對邊界附近測試樣本的過度自信預測。
#
# 資料量較多時（如第 A 部分使用完整訓練集），增強的效益較小——
# 模型已能從自然變異中學習。這正是第 09 章討論的「資料量制度」
#（data regime）交互效應。
#
# **變異注意事項：** 只有 30 個訓練試次加上少量種子，誤差棒會很寬。
# 請勿過度解讀具體數字——管線本身才是擴展到您自有資料集的重點。

# %% [markdown]
# ## ⚠️ A subtler trap
#
# ### 梯度顯著圖可能悄然誤導你
#
# 我們計算的顯著圖呈現的是**損失曲面的陡峭程度**，而非
# 模型已學習到可靠因果特徵的位置。這兩者在至少三個重要方面會出現分歧：
#
# **1. 輸入梯度不穩定性（input-gradient instability）。** 原始梯度顯著圖
# 出了名地雜亂：對輸入施加微小擾動就可能完全翻轉顯著圖
#（Ghorbani et al., "Interpretation of Neural Networks is Fragile", 2019）。
# 兩個幾乎相同的 EEG 試次、預測相同類別，卻可能產生完全不同的顯著圖模式。
# 對測試集取平均（如我們所做）有所幫助，但無意義模式的平均仍然無意義。
# **整合梯度（Integrated Gradients）**或 **SmoothGrad** 更為穩健，
# 但仍非因果方法。
#
# **2. 高顯著度 ≠ 因果使用。** 假設模型在 C3 通道分配了高梯度量值，這可能意味著：
# * 模型確實在使用左側運動皮質上方的 mu 節律抑制
#   ✓（我們期望的故事）。
# * 或者：C3 和 Cz 在訓練集中恰好相關，梯度落在 C3 上
#   是因為它在深度卷積權重排序中位於下游，而非因其具有生理特殊性 ✗。
# * 或者：某些訓練試次中存在的偽跡（例如基線校正未完全移除的緩慢漂移）
#   偶然與標籤相關，而 C3 恰好是該漂移最強的位置 ✗。
#
# 僅憑顯著圖無法區分這些情況。只有**因果干預**——
# 物理移除 C3（拔除電極）或在事後分析中剔除並重新訓練——
# 才能告訴你答案。
#
# **3. 增強／顯著度交互作用陷阱（augmentation/saliency interaction trap）。**
# 若在訓練期間套用通道遮罩（如我們的增強方法 c），然後對*已訓練*模型計算梯度顯著圖，
# 顯著度將被人為地分散到各個通道。遮罩訓練讓模型學會分散依賴；
# 顯著圖因此看起來「更符合感覺運動分佈」，純粹是因為額葉通道
# 在訓練期間偶爾被歸零——而非因為生物學的要求。
# 這讓模型看起來比實際上更具神經學可解釋性。
#
# **結論：** 將顯著圖視為尋找明顯失敗的除錯工具
#（例如，模型盯著 EOG 通道），而非關於哪些腦區驅動運動想像的科學主張。
# 若要做出科學聲明，你需要消融研究（ablation studies）、遮擋敏感度分析，
# 或者更理想地——使用來自獨立實驗室的保留資料預先登錄解碼管線。
