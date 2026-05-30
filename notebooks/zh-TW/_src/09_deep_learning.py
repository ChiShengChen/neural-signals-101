# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/09_deep_learning.ipynb)
#
# > **在 Google Colab 上執行？** 請先執行下一個儲存格——它會安裝所有套件並取得輔助模組。**在本機執行（執行 `make setup` 之後）？** 下一個儲存格不會做任何事；直接執行並繼續即可。

# %%
# --- Colab 啟動引導：僅在 Colab 環境下安裝相依套件與 neuro101 套件 ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab 設定完成——請繼續閱讀下方章節。")

# %% [markdown]
# # 第 09 章 — 神經訊號的深度學習（Deep Learning for Neural Signals）
#
# 深度網路（deep nets）可以直接從原始腦電圖（EEG）端對端地學習特徵。我們從專為 EEG
# 設計的**卷積（convolutional）**模型（EEGNet、ShallowConvNet、DeepConvNet）開始，
# 接著介紹小型 **LSTM** 和迷你 **Transformer**，最後說明**自監督式／基礎模型（self-supervised / foundation models）**。
#
# 這裡所有內容都刻意設計得**非常小，以便在 CPU 上幾分鐘內完成訓練**，
# 同時我們仍遵守黃金法則：依受試者（subject）分割資料，且標準化只使用訓練集統計量。
#
# ## 學習目標
# 1. 透過 `braindecode` 訓練 **EEGNet / ShallowConvNet / DeepConvNet**。
# 2. 在 PyTorch 中為 EEG 建立最小化的 **LSTM** 與 **Transformer**。
# 3. 保持訓練的誠實性（依受試者分割）與可重現性（固定隨機種子）。
#
# > **前置條件：** 第 08 章。
# > **難度：** ★★★★☆
#
# **執行時間：** CPU 上約 3–5 分鐘（煙霧測試模式下訓練輪數極少）。

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    # 向上搜尋專案根目錄（最多 5 層），找到後將 src 加入路徑
    _p = Path.cwd()
    for _ in range(5):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from neuro101 import io, datasets as ds
from neuro101.eval import make_subject_split

# 可重現性：固定所有隨機種子
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
torch.use_deterministic_algorithms(False)
DEVICE = "cpu"  # 本教程僅使用 CPU

SMOKE = ds.is_smoke()
EPOCHS = 3 if SMOKE else 15
n_subj = 2 if SMOKE else 3

# %% [markdown]
# ## 資料：BCI IV 2a，依受試者分割
#
# 我們在部分受試者上訓練，並在一位**保留不動的（held-out）**受試者上測試（跨受試者——誠實的設定）。
# 每個通道的標準化**僅使用訓練集**的統計量。

# %%
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
X = X.astype(np.float32)
n_chans, n_times = X.shape[1], X.shape[2]
print(f"X={X.shape}, classes={np.bincount(y)}, subjects={np.unique(subj)}")

# 使用第一個 LOSO 折疊：保留一位受試者作為測試集
train_idx, test_idx = next(make_subject_split(subj))
mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
sd = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-7
Xtr = (X[train_idx] - mu) / sd
Xte = (X[test_idx] - mu) / sd
ytr, yte = y[train_idx], y[test_idx]
print(f"訓練試次={len(ytr)} (受試者 {np.unique(subj[train_idx])}), "
      f"測試試次={len(yte)} (保留受試者 {np.unique(subj[test_idx])})")

# %% [markdown]
# ## 一個小巧、透明的訓練迴圈（training loop）
#
# 我們手動撰寫迴圈（而非隱藏起來），讓你清楚看到每一步：
# 前向傳播（forward pass）→ 損失（loss）→ 反向傳播（backward）→ 更新（step）。
# 批次（batch）很小；訓練完成後回報測試準確率。

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
# ## 1. EEGNet、ShallowConvNet、DeepConvNet（braindecode）
#
# 這些卷積架構（convolutional architectures）是 EEG 深度學習的標準基線（baseline）。
# `braindecode` 提供了現成可用的版本。它們期望的輸入形狀為 `(batch, channels, time)`。
#
# > **執行前：** 猜猜 EEGNet、ShallowConvNet 或 DeepConvNet 哪一個在保留受試者上的準確率最高——
# > 以及它們中是否有任何一個能在這個小型資料集上超越第 08 章的 CSP+LDA 基線。

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
    print(f"  {name:15s} 保留受試者準確率: {hist[-1]:.3f}")

# %% [markdown]
# ## ⚠️ 比較上述任意兩個數字前，請先閱讀這段說明
#
# 看到三個準確率數字後，很容易就宣布「模型 X 最好」。**先停下來。**
# 每一個都是*單次訓練、單一隨機種子*在*一位*保留受試者上的結果。深度訓練充滿雜訊——
# 換個隨機種子，數字就會移動。讓我們用不同種子重新訓練**相同的** EEGNet 來測量這種雜訊：

# %% [markdown]
# **執行前：** 下方各次執行的架構和資料完全相同——唯一改變的是隨機種子。
# 準確率相差多遠？（猜猜看範圍有多大。）

# %%
seed_accs = []
for s in range(3):
    torch.manual_seed(s)
    np.random.seed(s)
    h = train_eval(make_model("EEGNet"), Xtr, ytr, Xte, yte)
    seed_accs.append(h[-1])
    print(f"  EEGNet 種子 {s}: {h[-1]:.3f}")
seed_accs = np.array(seed_accs)
print(f"  -> 相同模型，相同資料: {seed_accs.mean():.3f} ± {seed_accs.std():.3f} "
      f"(差距 {seed_accs.max() - seed_accs.min():.3f})")

# %% [markdown]
# > **誠實聲明（適用於本章所有數字）：** 這些都是**單一種子執行，目的是教導*架構*的用法**——
# > 如何建構、訓練與評估 CPU 上的 EEG 網路——**而非**對架構進行排名。
# > 上方種子間的差距往往和*不同*模型之間的差距一樣大，
# > 因此任何「EEGNet 打敗 DeepConvNet」的單次執行宣稱都毫無意義。
# > 真正的架構比較需要**多個種子 × 多位受試者 + 配對檢定（Chapter 11）**，
# > 以及傳統基線（第 07–08 章）——在小型 EEG 資料集上，傳統方法往往更勝一籌。

# %% [markdown]
# ## 2. 最小化 LSTM（Long Short-Term Memory）
#
# 遞迴網路（Recurrent nets）逐步讀取訊號時間序列。EEG 序列較長，
# 因此我們將通道（channels）視為每個時間步的特徵，並對時間進行降採樣（downsample）以加速。
# 這是一個*基線*，而非競爭性模型——卷積式 EEG 網路通常更好。

# %%
class TinyLSTM(nn.Module):
    def __init__(self, n_chans, hidden=32, n_classes=2, stride=4):
        super().__init__()
        self.stride = stride
        self.lstm = nn.LSTM(n_chans, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):              # x: (batch, chans, time)
        x = x[:, :, ::self.stride]     # 對時間降採樣
        x = x.transpose(1, 2)          # (batch, time, chans)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])   # 最後一個時間步

torch.manual_seed(SEED)
hist_lstm = train_eval(TinyLSTM(n_chans), Xtr, ytr, Xte, yte)
print(f"  TinyLSTM 保留受試者準確率: {hist_lstm[-1]:.3f}")

# %% [markdown]
# ## 3. 迷你 Transformer（Transformer）
#
# 在降採樣後的時間步上建立一個小型 Transformer 編碼器（encoder）。
# 自注意力（self-attention）讓每個時間步都能查看其他所有時間步——功能強大但需要大量資料，
# 因此在這個微小資料集上不要期待它表現亮眼。

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
        return self.head(x.mean(1))    # 對時間取平均

torch.manual_seed(SEED)
hist_tr = train_eval(TinyTransformer(n_chans), Xtr, ytr, Xte, yte)
print(f"  TinyTransformer 保留受試者準確率: {hist_tr[-1]:.3f}")

# %% [markdown]
# ## 比較學習曲線（learning curves）

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for name, hist in {**conv_results, "TinyLSTM": hist_lstm, "TinyTransformer": hist_tr}.items():
    ax.plot(range(1, len(hist) + 1), hist, marker="o", ms=3, label=name)
ax.axhline(0.5, ls="--", color="gray", label="機率水準（chance）")
ax.set(xlabel="訓練輪數（Epoch）", ylabel="保留受試者準確率",
       title="CPU 上的迷你 EEG 深度網路（跨受試者）")
ax.legend(); ax.set_ylim(0, 1); plt.show()

# %% [markdown]
# ## 延伸閱讀：自監督式與基礎模型（self-supervised & foundation models）
#
# 神經科學中標記資料（labels）非常稀缺，因此一個日益成長的趨勢是**自監督式學習（self-supervised learning）**——
# 先在大量*未標記*的 EEG 上預訓練（預測遮蔽片段、在增強視圖之間進行對比學習），
# 再在你的小型標記任務上微調（fine-tune）。
# 近期的「EEG **基礎模型（foundation models）**」（例如 BENDR、BIOT、LaBraM、Neuro-GPT）
# 都遵循這個方法。它們超出了 101 課程的範圍，但重點是：
# *如果標記資料很少，在未標記資料上預訓練往往比使用更複雜的分類器效果更好。*

# %% [markdown]
# ## ✅ 概念自測（Concept check）
#
# 1. EEGNet 對通道使用深度卷積（depthwise convolution），然後對時間使用可分離卷積（separable convolution）。
#    與標準卷積相比，深度可分離卷積（depthwise separable convolutions）對 EEG 資料的主要實際優勢是什麼？
# 2. TinyLSTM 在遞迴層前將時間降採樣 4 倍。這有什麼取捨：可能丟失什麼資訊？獲得什麼？
# 3. 你訓練 EEGNet 共 15 個 epoch，每個 epoch 追蹤測試準確率，並回報任何 epoch 中達到的最高測試準確率。
#    為什麼這樣做有問題？應該怎麼做？
#
# **解答：**
# 1. 深度可分離卷積通過將完整卷積分解為逐通道空間濾波器（per-channel spatial filter）
#    和逐點混合步驟（pointwise mixing step），大幅減少參數數量（進而降低過擬合風險）——
#    當 EEG 資料集很小時，這一點至關重要。
# 2. 降採樣 4 倍會丟棄高頻時間細節（新奈奎斯特頻率以上的所有成分），
#    這對快速 gamma 帶瞬態（gamma-band transients）可能有影響。獲得的是更短的序列，
#    減少訓練時間並緩解長序列輸入的 LSTM 梯度消失（vanishing gradients）問題。
# 3. 根據測試準確率選擇最佳 epoch 等於讓測試集效能引導模型選擇——這就是測試集洩漏（test-set leakage）。
#    應使用獨立的驗證受試者（或驗證折疊）進行早停（early stopping）；
#    測試受試者僅用於最終報告的數字。

# %% [markdown]
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# - **使用整個資料集的統計量進行標準化。** 只在**訓練集**上計算均值／標準差（如我們所做），
#   再應用到測試集。否則就會洩漏（leakage）。
# - **將試次（epochs）隨機打亂後分成訓練／測試集。** 與傳統機器學習一樣的洩漏——
#   應依受試者（或區塊）分割。深度模型同樣會*急切地*洩漏。
# - **與傳統基線不公平比較。** 在小型 EEG 資料集上，CSP+LDA 或黎曼方法（Riemannian methods）
#   往往**超越**深度網路。在宣稱深度模型更好之前，務必包含傳統基線（第 08 章）。
# - **未固定隨機種子。** 深度訓練充滿雜訊；應回報多個種子的均值 ± 標準差以進行公平比較
#   （我們這裡只使用一個種子是為了保持快速）。
# - **訓練固定 epoch 數後回報見過的最佳測試準確率。**
#   這會偷看測試集——應使用驗證集分割進行早停。
#
# **下一章：** 第 10 章 — BCI 範式（P300、SSVEP、睡眠、癲癇）巡禮，每個範式提供一個最小可執行範例。
