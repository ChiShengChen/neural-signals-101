# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/04_dsp_basics.ipynb)
#
# > **在 Google Colab 上執行？** 請先執行下一個儲存格——它會安裝所有套件並取得輔助程式包。**在本機執行（執行 `make setup` 之後）？** 下一個儲存格什麼也不做；直接執行並繼續即可。

# %%
# --- Colab 啟動程式：僅在 Colab 環境安裝相依套件與 neuro101 套件 ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab 設定完成——請繼續往下執行本章內容。")

# %% [markdown]
# # 第 04 章 — 數位訊號處理（Digital Signal Processing，DSP）基礎
#
# **請勿跳過本章。** 幾乎所有後續的錯誤（以及後續的所有特徵）都建立在這些概念之上。我們將搭配圖示逐一建立每個概念。
#
# ## 學習目標
# 1. **取樣定理（sampling theorem）** 與 **混疊（aliasing）**：為什麼取樣率很重要。
# 2. **量化（quantization）**：將連續電壓轉換為數字。
# 3. **濾波器（filters）**（FIR vs IIR）、**帶通（band-pass）** 以及 **50/60 Hz 陷波（notch）**。
# 4. **重參考（re-referencing）** 與 **導程（montage）**：EEG 中「通道數值」究竟代表什麼。
#
# > **先修條件：** 第 03 章。
# > **難度：** ★★★☆☆
#
# **執行時間：** 約 1 分鐘（主要使用合成訊號，以便清楚呈現數學原理）。

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(5):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent
import numpy as np
import matplotlib.pyplot as plt
from neuro101 import preprocessing as pp

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 取樣（Sampling）：將連續訊號轉換為數字
#
# 錄製裝置每秒量測電壓 **`sfreq`** 次（即*取樣率（sampling rate）*，單位為 Hz）。**奈奎斯特頻率（Nyquist frequency）** 是取樣率的一半：它是你能忠實重現的最高頻率。
#
# ### 混疊（Aliasing）：取樣過慢的危險
# 若訊號包含*高於*奈奎斯特頻率的成分，它不會消失——它會偽裝成一個*更低*的頻率。這個假的低頻就是**混疊（alias）**，而且無法還原。以下範例：一個 30 Hz 的正弦波以 40 Hz 取樣（奈奎斯特 = 20 Hz < 30 Hz），看起來像慢速的約 10 Hz 波形。

# %%
f_true = 30.0       # 真實頻率（Hz）
fs_low = 40.0       # 過低的取樣率 -> 奈奎斯特 = 20 Hz < 30 Hz
fs_high = 1000.0    # 「連續」參考訊號

t_cont = np.arange(0, 0.5, 1 / fs_high)
t_samp = np.arange(0, 0.5, 1 / fs_low)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t_cont, np.sin(2 * np.pi * f_true * t_cont), color="#bbb", label="真實 30 Hz 訊號")
ax.plot(t_samp, np.sin(2 * np.pi * f_true * t_samp), "o-", color="#c44e52",
        label="以 40 Hz 取樣（看起來像 ~10 Hz！）")
ax.set(xlabel="時間（秒）", title="混疊：30 Hz 波形以 40 Hz 取樣後偽裝成慢速波")
ax.legend()
plt.show()

# %% [markdown]
# **重點：** 取樣率必須*超過*你所關心的最高頻率的兩倍，並在降取樣之前使用**抗混疊低通濾波器（anti-aliasing low-pass filter）**。真實的放大器在硬體中完成此步驟；若在軟體中降取樣，請先進行濾波。

# %% [markdown]
# ## 2. 量化（Quantization）：電壓變為整數
#
# 類比數位轉換器（analog-to-digital converter）將每個取樣值四捨五入至有限個等級中最近的一個（例如 16 位元 = 65,536 個等級）。位元數太少會產生**量化雜訊（quantization noise）**——一種階梯狀誤差。現代 EEG 使用 16–24 位元，因此通常可忽略不計，但親眼*看到*這個現象仍然很有幫助。

# %%
t = np.arange(0, 0.1, 1 / 1000)
clean = np.sin(2 * np.pi * 40 * t)
def quantize(x, n_bits):
    levels = 2 ** n_bits
    return np.round((x + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, clean, color="#bbb", label="連續訊號")
ax.step(t, quantize(clean, 3), where="mid", color="#c44e52", label="3 位元（8 個等級）")
ax.set(xlabel="時間（秒）", title="僅用 3 位元量化——可見的階梯狀雜訊")
ax.legend()
plt.show()

# %% [markdown]
# ## 3. 濾波器（Filters）：保留你想要的頻率
#
# **濾波器（filter）** 改變每個頻率通過的程度。
# - **低通（Low-pass）**：保留低頻。**高通（High-pass）**：保留高頻。
# - **帶通（Band-pass）**：保留一個頻帶（例如 8–30 Hz，用於運動想像）。
# - **陷波／帶阻（Notch / band-stop）**：移除窄頻帶（例如 50/60 Hz 市電雜訊）。
#
# 兩大類型：
# - **FIR**（有限脈衝響應，Finite Impulse Response）：永遠穩定、嚴格線性相位（不扭曲時序），但需要較多係數（較慢）。
# - **IIR**（無限脈衝響應，Infinite Impulse Response）：計算量少且過渡帶陡峭，但可能扭曲相位。我們使用**零相位（zero-phase）** 濾波（`filtfilt`，正向＋反向）來抵消相位扭曲——代價是非因果性（使用未來的取樣值），這在離線分析中沒問題。
#
# 接下來建立一個含雜訊的訊號並加以清理。

# %%
sf = 250.0
t = np.arange(0, 4, 1 / sf)
alpha = np.sin(2 * np.pi * 10 * t)              # 10 Hz 腦波節律（我們想要的）
mains = 0.8 * np.sin(2 * np.pi * 50 * t)        # 50 Hz 市電雜訊（不想要的）
drift = 2.0 * np.sin(2 * np.pi * 0.3 * t)       # 緩慢漂移（不想要的）
noise = 0.3 * rng.standard_normal(t.size)
signal = (alpha + mains + drift + noise)[None, None, :]  # 形狀 (1,1,T)，供輔助函式使用

# 帶通 8-30 Hz 同時移除緩慢漂移和大部分 50 Hz。
bp = pp.bandpass_filter(signal, sf, 8, 30)[0, 0]
# 陷波是專門針對市電雜訊的工具。
notched = pp.notch_filter(signal, sf, 50.0)[0, 0]

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, signal[0, 0], color="#333"); axes[0].set_title("原始訊號：alpha + 50 Hz + 漂移 + 雜訊")
axes[1].plot(t, notched, color="#dd8452"); axes[1].set_title("50 Hz 陷波後（漂移仍存在）")
axes[2].plot(t, bp, color="#2e8b57"); axes[2].set_title("8–30 Hz 帶通後（乾淨的 ~10 Hz）")
axes[2].set_xlabel("時間（秒）")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 在頻域中觀察濾波效果
# 濾波器是否達到效果，最清楚的證明是濾波前後的**功率頻譜（power spectrum）**。
#
# > **執行前猜測：** 在 8–30 Hz 帶通後的功率頻譜中，50 Hz 處你預期會看到什麼——完全消失、部分衰減，還是不變？同時猜測 0.3 Hz 的漂移峰值是否仍然可見。

# %%
from scipy.signal import welch
f0, p_raw = welch(signal[0, 0], sf, nperseg=int(sf))
_, p_bp = welch(bp, sf, nperseg=int(sf))
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.semilogy(f0, p_raw, label="原始訊號", color="#333")
ax.semilogy(f0, p_bp, label="帶通 8–30 Hz 後", color="#2e8b57")
for fx in (0.3, 50): ax.axvline(fx, ls="--", color="gray", lw=0.8)
ax.set(xlim=(0, 70), xlabel="頻率（Hz）", ylabel="功率",
       title="帶通濾波移除了 0.3 Hz 漂移和 50 Hz 市電線雜訊")
ax.legend(); plt.show()

# %% [markdown]
# ## 4. 重參考（Re-referencing）與導程（Montage）
#
# EEG 量測的是**電壓差（voltage differences）**，因此每個數值都是「通道減去參考點」。選擇參考點會一致地改變所有數值：
# - **共同參考（Common reference）**（例如乳突電極）：簡單，但偏向該位置。
# - **平均參考（Average reference）**：在每個時間點減去所有電極的平均值——是常用的、位置中立的預設值。
# - **雙極／拉普拉斯（Bipolar / Laplacian）**：鄰近電極之間的差值；強化局部活動。
#
# **導程（montage）** 是從通道名稱（Cz、C3 等）到頭部三維位置的對應關係。地形圖（topographic plots）和空間方法（如 CSP）都需要它。
#
# 以下是多通道資料的平均參考示範。

# %%
n_ch = 5
common_brain = np.sin(2 * np.pi * 10 * t)         # 所有通道都能看到的共享訊號
data = np.stack([common_brain + 0.2 * rng.standard_normal(t.size) + ch
                 for ch in range(n_ch)])           # 每個通道都有一個偏移量
avg_ref = data - data.mean(axis=0, keepdims=True)  # 平均參考

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ch in range(n_ch):
    axes[0].plot(t[:250], data[ch, :250] + ch, lw=0.7)
    axes[1].plot(t[:250], avg_ref[ch, :250] + ch, lw=0.7)
axes[0].set_title("之前：每個通道的偏移量主導訊號")
axes[1].set_title("平均參考後：共同偏移已移除")
for a in axes: a.set_xlabel("時間（秒）")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## ✅ 概念確認
#
# 1. 一個訊號包含 90 Hz 成分，以 100 Hz 取樣（奈奎斯特 = 50 Hz）。混疊頻率會出現在記錄資料的哪個頻率？
# 2. 在對 ERP 資料進行分段（epoching）之前，你套用了 1 Hz 高通濾波器。請說明這對峰值在 0.5 Hz 的慢速皮質電位可能產生一個後果。
# 3. 平均參考後，每個時間點的跨通道均值在數學上有什麼保證？
#
# **解答：**
# 1. 混疊出現在 |90 − 100| = 10 Hz——一個假的 10 Hz 振盪。
# 2. 1 Hz 高通濾波器會衰減（並扭曲）0.5 Hz 的 ERP 成分，可能反轉或消除你想要量測的效應。
# 3. 平均參考後，每個取樣點的跨通道均值恰好為零（由構造保證：你從每個通道中減去了均值）。

# %% [markdown]
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# - **降取樣前未使用抗混疊濾波器。** 你會永久地將高頻內容折疊到你關心的頻帶中。請務必先進行低通濾波。
# - **過度積極地濾波，然後進行連通性／因果分析。** 陡峭的 IIR 濾波器會扭曲相位；基於相位的特徵（如 PLV，第 07 章）會因此失效。請使用零相位（`filtfilt`）或 FIR 線性相位濾波器。
# - **對緩慢訊號（ERP）使用過高截止頻率的高通濾波（例如 2 Hz）。** 你可能會濾掉你想要量測的效應本身。
# - **忘記參考點。** 兩篇論文都說「通道 Cz」，但如果它們的參考點不同，意義就不同。請務必說明你的參考點與導程。
# - **在 60 Hz 市電地區使用 50 Hz 陷波（或反之）。** 請配合你的地區：歐洲／亞洲／非洲為 50 Hz，美洲為 60 Hz。
#
# **下一章：** 第 05 章——使用 ICA 及相關方法移除真實的生物偽跡（眨眼、肌電），並在*同一段資料*上呈現前後對比。
