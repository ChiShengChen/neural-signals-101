# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/03_math_you_can_see.ipynb)
#
# > **在 Google Colab 上執行？** 請先執行下一個儲存格——它會安裝所有套件並
# > 下載輔助工具包。**在本地端執行（執行 `make setup` 之後）？** 下一個
# > 儲存格不會做任何事；直接執行並繼續即可。

# %%
# --- Colab 啟動程序：僅在 Colab 環境中安裝相依套件與 neuro101 套件 ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab 設定完成——請繼續進入下方章節。")

# %% [markdown]
# # 第三章 — 看得見的數學（Math You Can See）
#
# 幾乎每個後續章節中都隱藏著三個概念：**傅立葉（Fourier）**、**共變異數（covariance）**
# 和**特徵向量（eigenvectors）**。這些概念都不需要微積分就能*理解*——
# 它們只需要正確的圖像。本章用玩具訊號從頭建立這些圖像，
# 這樣當這些詞彙出現在第六章或第八章時，你已有心理圖像可以錨定。
#
# ## 學習目標
# 1. 將**傅立葉轉換（Fourier transform）**描述為「訊號中每種節律有多少」。
# 2. 閱讀**共變異數矩陣（covariance matrix）**並說明其條目的含義。
# 3. 解釋共變異數矩陣的**特徵向量（eigenvectors）**指向何處，以及為何這對
#    分離兩類腦電訊號至關重要。
#
# > **先備知識：** 第二章。
# > **難度：** ★★★☆☆
# > **執行時間：** 約 1 分鐘（合成資料，使用 CPU）。

# %%
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# %% [markdown]
# ---
# ## 第一部分 — 傅立葉：訊號是節律的總和
#
# ### 什麼是節律？
#
# **正弦波（sine wave）**是最純粹的振盪——它以恰好一種速度平滑地上下起伏，
# 這種速度稱為其**頻率（frequency）**（以**赫茲（Hz）**，即每秒循環次數為單位）。
# EEG 節律如「alpha（阿爾法）」之所以如此命名，是因為它們在訊號的正弦波分解中以突峰的形式出現。
#
# **傅立葉轉換**回答一個問題：
# *「如果我把這個訊號寫成正弦波的總和，我需要每個頻率多少？」*
#
# 答案是**頻譜（spectrum）**——頻率對應振幅的長條圖。
#
# 讓我們自己用三個正弦波建立一個訊號，這樣我們就確切知道正確答案應該是什麼。

# %%
# --- 建立訊號 ---
sfreq = 200          # 取樣率：每秒 200 個樣本
duration = 2.0       # 秒
t = np.arange(0, duration, 1.0 / sfreq)   # 時間軸

f1, f2, f3 = 6.0, 10.0, 20.0             # 赫茲——三種節律

amp1, amp2, amp3 = 1.0, 0.6, 0.4          # 每種節律的響度

sine1 = amp1 * np.sin(2 * np.pi * f1 * t)
sine2 = amp2 * np.sin(2 * np.pi * f2 * t)
sine3 = amp3 * np.sin(2 * np.pi * f3 * t)

noise = 0.15 * rng.standard_normal(t.size)  # 少量隨機雜訊

signal = sine1 + sine2 + sine3 + noise

print(f"訊號：{len(t)} 個樣本，{sfreq} Hz，{duration} 秒")
print(f"組成：{f1} Hz（振幅 {amp1}），{f2} Hz（振幅 {amp2}），{f3} Hz（振幅 {amp3}）")

# %% [markdown]
# ### 圖 1a — 原始時域訊號
#
# 這是你在示波器或 EEG 檢視器上看到的樣子。
# 你能發現任何重複的模式嗎？很難，因為三種節律重疊在一起。

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, signal, color="#2c7bb6", lw=0.9)
ax.set(xlabel="時間（秒）", ylabel="振幅",
       title="時域訊號（6 Hz + 10 Hz + 20 Hz + 雜訊）")
ax.axhline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 圖 1b — 各個獨立組成（堆疊顯示）
#
# 這裡我們「打開引擎蓋」，分別顯示每個成分。
# 這是我們希望傅立葉轉換能夠恢復的真實情況。

# %%
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

components = [
    (sine1, f"{f1} Hz 組成（振幅 {amp1}）", "#d7191c"),
    (sine2, f"{f2} Hz 組成（振幅 {amp2}）", "#fdae61"),
    (sine3, f"{f3} Hz 組成（振幅 {amp3}）", "#1a9641"),
    (noise, "雜訊", "#888888"),
]
for ax, (comp, label, color) in zip(axes, components):
    ax.plot(t, comp, color=color, lw=0.9, label=label)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_ylabel("振幅")

axes[-1].set_xlabel("時間（秒）")
fig.suptitle("各個正弦波組成", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 圖 1c — 逐一疊加組成（部分和）
#
# 這是關鍵的幾何洞察：**訊號就是節律的總和**。
# 我們從 6 Hz 波開始，然後加上 10 Hz，再加上 20 Hz，最後加上雜訊。
# 每個面板使波形越來越接近我們記錄的最終訊號。

# %%
partial_sums = [
    (sine1,                        f"僅 6 Hz"),
    (sine1 + sine2,                f"6 + 10 Hz"),
    (sine1 + sine2 + sine3,        f"6 + 10 + 20 Hz"),
    (sine1 + sine2 + sine3 + noise, "6 + 10 + 20 Hz + 雜訊（= 完整訊號）"),
]

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
colors = ["#d7191c", "#fdae61", "#1a9641", "#2c7bb6"]

for ax, (ps, label), color in zip(axes, partial_sums, colors):
    ax.plot(t, ps, color=color, lw=0.9)
    ax.set_ylabel("振幅")
    ax.set_title(label, fontsize=10, loc="left")
    ax.axhline(0, color="gray", lw=0.4)

axes[-1].set_xlabel("時間（秒）")
fig.suptitle("逐一疊加正弦波來建立訊號", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ### 執行下一個儲存格之前——先做預測！
#
# **執行前先預測：** 我們用 6 Hz、10 Hz 和 20 Hz 的正弦波建立訊號，
# 振幅分別為 1.0、0.6 和 0.4。
#
# - 你預期頻譜中哪些頻率會出現峰值？
# - 哪個峰值應該最高？哪個最矮？
# - 峰值之間應該有雜訊底線嗎？
#
# 在揭曉答案之前，先寫下（或只是想清楚）你的預測。

# %%
# --- 計算 FFT（快速傅立葉轉換）振幅頻譜 ---
N = len(signal)
fft_vals = np.fft.rfft(signal)                   # 實數 FFT（僅正頻率）
fft_mag  = np.abs(fft_vals) / N * 2              # 正規化：除以 N，×2 用於單側
freqs    = np.fft.rfftfreq(N, d=1.0 / sfreq)    # 頻率軸（赫茲）

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, fft_mag, color="#2c7bb6", lw=1.2)
ax.set(xlabel="頻率（Hz）", ylabel="振幅",
       title="FFT 振幅頻譜——峰值揭示存在哪些節律",
       xlim=(0, sfreq / 2))

# 標注三個已知峰值
for f0, amp in [(f1, amp1), (f2, amp2), (f3, amp3)]:
    idx = np.argmin(np.abs(freqs - f0))
    ax.axvline(f0, ls="--", color="gray", lw=0.8)
    ax.annotate(f"{f0} Hz\n(振幅 ≈ {amp})",
                xy=(f0, fft_mag[idx]),
                xytext=(f0 + 2, fft_mag[idx] * 0.8),
                fontsize=9, color="#c44e52",
                arrowprops=dict(arrowstyle="->", color="#c44e52"))

plt.tight_layout()
plt.show()

print("找到的峰值頻率約為：")
peak_mask = fft_mag > 0.1
for f, m in zip(freqs[peak_mask], fft_mag[peak_mask]):
    print(f"  {f:.1f} Hz  →  振幅 {m:.3f}")

# %% [markdown]
# **頻譜與你的預測相符嗎？**
#
# 重點摘要：
# - 每個正弦波成分在其頻率處顯示為一個**尖銳的峰值**。
# - 峰值的**高度**等於我們選擇的振幅（1.0、0.6、0.4）。
# - 峰值之間有一個小的雜訊底線——隨機雜訊將其微小的能量分散到所有頻率。
#
# 頻譜是一張*食譜卡*：它告訴你訊號中烘焙了哪些節律，以及每種節律有多少。
#
# > **一個實用注意事項：** FFT 的 x 軸只有在你知道**取樣率**的情況下才有意義。
# > 相同的 FFT 輸出搭配不同的 `sfreq` 會將那些峰值標記在完全不同的頻率上。
# > 永遠記錄 `sfreq`！

# %% [markdown]
# ---
# ## 第二部分 — 共變異數：兩個通道如何一起移動
#
# ### 為什麼我們要在意這個
#
# EEG 有許多通道（電極）。它們**並非獨立**——附近的電極接收相同的底層大腦來源
#（這稱為*體積傳導（volume conduction）*）。理解通道如何共同變化對於
# 空間濾波（如第八章使用的 CSP，Common Spatial Patterns）至關重要。
#
# ### 什麼是共變異數？
#
# 單一通道的**變異數（variance）**：該通道自身抖動的程度。
#
# 兩個通道的**共變異數（covariance）**：它們是否傾向於*同時偏高*
#（正共變異數）或*朝相反方向移動*（負共變異數）？
#
# **共變異數矩陣（covariance matrix）**將所有成對共變異數收集在一個表格中。
# 對於通道 A 和 B：
#
# ```
# C = | Var(A)    Cov(A,B) |
#     | Cov(B,A)  Var(B)   |
# ```
#
# 對角線是每個通道自身的變異數；非對角線是耦合程度。

# %%
# --- 生成兩個相關的通道 ---
n_samples = 300

# 通道 1：純粹隨機
ch1_raw = rng.standard_normal(n_samples)

# 通道 2：70% 相同訊號 + 30% 獨立雜訊 → 強正相關
ch2_raw = 0.7 * ch1_raw + 0.3 * rng.standard_normal(n_samples)

# 縮放至不同的變異數，使共變異數矩陣更有趣
ch1 = 2.0 * ch1_raw          # 通道 1 具有較大的變異數
ch2 = 1.0 * ch2_raw          # 通道 2 具有較小的變異數

data2d = np.stack([ch1, ch2], axis=0)   # 形狀為 (2, n_samples)

# 共變異數矩陣
C = np.cov(data2d)   # numpy 計算成對共變異數

print("共變異數矩陣 C：")
print(C.round(3))
print()
print(f"  C[0,0] = Var(ch1) = {C[0,0]:.3f}   — ch1 自身的變異數")
print(f"  C[1,1] = Var(ch2) = {C[1,1]:.3f}   — ch2 自身的變異數")
print(f"  C[0,1] = Cov(ch1, ch2) = {C[0,1]:.3f}  — 它們如何共同變化（正值 = 同方向）")

# %% [markdown]
# ### 圖 2 — 帶有共變異數橢圓和特徵向量的散點圖
#
# ch1 對 ch2 的散點圖顯示聯合分佈。
# 我們疊加：
# - **共變異數橢圓**：約 68% 的資料區域（1-sigma）。
#   其形狀沿最大聯合變化的方向拉伸。
# - **特徵向量**按其特徵值平方根縮放。
#   它們沿著這些拉伸方向指向。

# %%
# 共變異數矩陣的特徵分解
eigenvalues, eigenvectors = np.linalg.eigh(C)   # eigh：對稱矩陣，按升序排列
# 按降序排列，使 eigenvector[:,0] 是「最大變異數」方向
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 通過旋轉單位圓來繪製橢圓
theta = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
# 將每個軸按 sqrt(特徵值) 縮放，然後由特徵向量旋轉
ellipse = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle

# 資料的中心（均值）
mu = data2d.mean(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ch1, ch2, alpha=0.3, s=12, color="#2c7bb6", label="資料點")
ax.plot((mu[0] + ellipse[0]), (mu[1] + ellipse[1]),
        color="#d7191c", lw=2, label="1-sigma 橢圓")

# 以箭頭繪製特徵向量
arrow_colors = ["#d7191c", "#fdae61"]
arrow_labels = ["第 1 特徵向量\n（最大變異數）", "第 2 特徵向量\n（最小變異數）"]
for i in range(2):
    scale = np.sqrt(eigenvalues[i])
    dx = eigenvectors[0, i] * scale
    dy = eigenvectors[1, i] * scale
    ax.annotate("", xy=(mu[0, 0] + dx, mu[1, 0] + dy),
                xytext=(mu[0, 0], mu[1, 0]),
                arrowprops=dict(arrowstyle="-|>", color=arrow_colors[i], lw=2.5))
    ax.text(mu[0, 0] + dx * 1.15, mu[1, 0] + dy * 1.15,
            arrow_labels[i], fontsize=9, color=arrow_colors[i], ha="center")

ax.set(xlabel="通道 1", ylabel="通道 2",
       title="共變異數橢圓與特徵向量\n"
             "特徵向量指向最大聯合變化的方向")
ax.set_aspect("equal")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **解讀圖像：**
#
# - 雲點的細長形狀告訴你 ch1 和 ch2 是**相關的（correlated）**——
#   當 ch1 偏高時，ch2 也傾向於偏高。
# - **紅色箭頭**（第一特徵向量）沿橢圓的長軸方向指向——資料變化最大的方向。
# - **橘色箭頭**（第二特徵向量）垂直於此——變化最小的方向。
# - 每個箭頭的**長度**等於 `sqrt(特徵值)`，即資料*投影到該箭頭方向上*的標準差。
#
# 如果 ch1 和 ch2 完全獨立，橢圓將是一個圓形，
# 兩個特徵向量將沿 x 軸和 y 軸方向指向。

# %% [markdown]
# ---
# ## 第三部分 — 特徵向量與 CSP 幾何（Eigenvectors and CSP Geometry）
#
# ### CSP（Common Spatial Patterns，共同空間模式）的核心概念
#
# 在運動想像（motor-imagery）BCI（腦機介面）中，我們問：
# *「這個人在想像移動左手還是右手？」*
# 每種心理狀態產生的 EEG 具有**不同的空間共變異數結構**——
# 某些通道變得更活躍，某些則減少。
#
# **CSP** 找到一個空間方向（通道的加權組合），使得：
# - **A 類**具有高變異數（訊號變化很大）。
# - **B 類**具有低變異數（訊號幾乎不移動）。
#
# 如果我們將原始多通道訊號投影到該方向，兩個類別就變得容易分離：
# A 類給出大值，B 類給出小值。
#
# 我們可以用二維玩具資料看到這個幾何結構——不需要任何 CSP 函式庫。

# %%
# --- 建立具有不同共變異數結構的兩個類別 ---
n_per_class = 200

# A 類：主要沿 45 度對角線（/ 方向）變化
angle_A = np.pi / 4         # 45 度
cov_A_axes = np.array([[3.0, 0], [0, 0.3]])   # 長軸在第一主要方向上

def make_class(n, angle, cov_axes, rng):
    """從 2D 高斯分佈取樣，並按 `angle` 旋轉。"""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    raw = rng.standard_normal((n, 2)) @ np.diag(np.sqrt(np.diag(cov_axes)))
    return (R @ raw.T).T

classA = make_class(n_per_class, angle_A,  cov_A_axes, rng)
# B 類：主要沿另一條對角線（\ 方向）變化
angle_B = -np.pi / 4        # -45 度（垂直於 A）
classB = make_class(n_per_class, angle_B, cov_A_axes, rng)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(classA[:, 0], classA[:, 1], alpha=0.4, s=15,
           color="#d7191c", label="A 類（例如：左手想像）")
ax.scatter(classB[:, 0], classB[:, 1], alpha=0.4, s=15,
           color="#2c7bb6", label="B 類（例如：右手想像）")
ax.set(xlabel="通道 1", ylabel="通道 2",
       title="兩個具有不同共變異數結構的類別\n"
             "A 類沿 / 方向拉伸，B 類沿 \\ 方向拉伸")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 在散點圖中你可以看到：
# - A 類（紅色）沿**左上到右下**的對角線方向拉伸。
# - B 類（藍色）沿**另一條**對角線方向拉伸。
#
# 現在的問題是：**我們應該投影到哪個單一軸上，才能最好地分離兩個類別？**
#
# 一個自然的選擇是 A 變化很大而 B 變化很小的方向——
# 即 **A 類共變異數矩陣的第一特徵向量**。
# 該方向使比值 `Var_A / Var_B` 最大化。

# %% [markdown]
# ---
# ### 執行下一個儲存格之前——再做一次預測！
#
# **執行前先預測：** 我們將把兩個類別投影到 A 類共變異數矩陣的第一特徵向量上
#（A 變化最大的方向）。
#
# - 你預期 A 類的直方圖在那條軸上是寬的還是窄的？
# - 你預期 B 類的直方圖是寬的還是窄的？
# - 兩個直方圖會大量重疊，還是會明顯分開？

# %%
# 計算 A 類共變異數的第一特徵向量
cov_A = np.cov(classA.T)
eigvals_A, eigvecs_A = np.linalg.eigh(cov_A)
# eigh 返回升序；取最後一個（最大的）
w_csp = eigvecs_A[:, -1]    # 使 Var_A 最大化的方向

print(f"選擇的方向（A 類的特徵向量）：[{w_csp[0]:.3f}, {w_csp[1]:.3f}]")

# 將兩個類別投影到 w_csp 上
proj_A = classA @ w_csp
proj_B = classB @ w_csp

var_A_proj = np.var(proj_A)
var_B_proj = np.var(proj_B)
print(f"\nA 類投影到 w_csp 的變異數：{var_A_proj:.3f}")
print(f"B 類投影到 w_csp 的變異數：{var_B_proj:.3f}")
print(f"變異數比值（A/B）：{var_A_proj / var_B_proj:.2f}x")

# %% [markdown]
# ### 圖 3 — 幾何視圖 + 一維直方圖

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左圖：帶有選擇方向的散點圖
ax = axes[0]
ax.scatter(classA[:, 0], classA[:, 1], alpha=0.35, s=12,
           color="#d7191c", label="A 類")
ax.scatter(classB[:, 0], classB[:, 1], alpha=0.35, s=12,
           color="#2c7bb6", label="B 類")

# 將 w_csp 繪製為穿過原點的線
scale = 3.0
ax.annotate("", xy=(w_csp[0] * scale, w_csp[1] * scale),
            xytext=(-w_csp[0] * scale, -w_csp[1] * scale),
            arrowprops=dict(arrowstyle="-|>", color="#1a9641", lw=2.5))
ax.text(w_csp[0] * scale * 1.1, w_csp[1] * scale * 1.1,
        "CSP 軸\n（最大化 A，最小化 B）", color="#1a9641", fontsize=9, ha="center")

ax.set(xlabel="通道 1", ylabel="通道 2",
       title="CSP 軸：最大化 A 類，最小化 B 類的變異數")
ax.set_aspect("equal")
ax.legend(fontsize=9)

# 右圖：投影後的一維直方圖
ax2 = axes[1]
bins = np.linspace(-5, 5, 40)
ax2.hist(proj_A, bins=bins, alpha=0.55, color="#d7191c", label=f"A 類（var={var_A_proj:.2f}）")
ax2.hist(proj_B, bins=bins, alpha=0.55, color="#2c7bb6", label=f"B 類（var={var_B_proj:.2f}）")
ax2.set(xlabel="投影值（沿 CSP 軸）",
        ylabel="計數",
        title="投影後的一維直方圖\n"
              "A 類分散；B 類被壓縮")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# **解讀圖像：**
#
# - 在 **CSP 軸**（綠色箭頭）上，A 類分散得很寬——高變異數。
# - B 類被壓縮——低變異數——因為其拉伸方向*垂直於* CSP 軸。
# - 這種變異數差異就是**可分離性（separability）**：投影值上的一個簡單閾值
#   （或短時窗的對數變異數）可以區分兩種心理狀態。
#
# 在真實的 EEG 運動想像流程（第八章）中，「通道」是數十個電極，
# 共變異數矩陣是 64×64 而不是 2×2，但幾何直覺完全相同。

# %% [markdown]
# ---
# ## ✅ 概念確認（Concept Check）
#
# **1.** 你疊加兩個正弦波：一個 8 Hz 振幅為 2，另一個 13 Hz 振幅為 0.5。
# 在 FFT 振幅頻譜中，哪個峰值更高？為什麼？
#
# **2.** 一個 2×2 的共變異數矩陣其非對角線值接近零。
# 這告訴你兩個通道之間的關係是什麼？
#
# **3.** 你計算了 A 類 EEG 資料的共變異數矩陣特徵向量。
# 你找到具有*最小*特徵值的特徵向量。這個特徵向量指向什麼樣的方向——
# A 類的高變異數還是低變異數？
#
# **4.** 為什麼在你能解讀 FFT 的 x 軸之前，你需要知道**取樣率**？
#
# ---
# **答案：**
#
# 1. 8 Hz 的峰值更高（振幅 2 對比 0.5）。在振幅頻譜中，高度直接反映振幅。
#    13 Hz 的峰值更矮。
#
# 2. 非對角線 ≈ 0 表示通道是**不相關的（uncorrelated）**——知道通道 1 偏高
#    對於通道 2 在哪裡不提供任何有用資訊。散點圖看起來像圓形雲點，而不是細長的橢圓。
#
# 3. 最小特徵值對應**最低變異數**的方向。對於 CSP，你會希望另一個類別的這個方向，
#    而不是 A 類的。
#
# 4. FFT 產生一個由*箱號（bin number）*索引的複數向量，而不是赫茲。
#    要將箱號轉換為赫茲，你需要計算 `箱號 / (N / sfreq)`。
#    沒有 `sfreq`，相同的 FFT 輸出可以代表任何頻率範圍。

# %% [markdown]
# ## ⚠️ 常見錯誤 / 為什麼這樣做是錯的
#
# - **混淆振幅和功率。** FFT 給出振幅（|F|）。功率是振幅的*平方*（|F|²）。
#   在振幅上看起來高兩倍的峰值功率是*四倍*。功率頻譜通常以對數刻度顯示；
#   振幅頻譜以線性刻度顯示。不要混淆它們。
#
# - **忘記取樣率設定頻率軸。** FFT 只產生箱號索引。不知道 `sfreq` 就無法以
#   赫茲標記 x 軸。一個常見的錯誤：除以樣本數 N 而不是除以取樣週期 1/sfreq。
#   永遠使用 `np.fft.rfftfreq(N, d=1/sfreq)`。
#
# - **將 EEG 通道視為獨立的。** 由於電流通過頭骨傳播（體積傳導），
#   附近的電極記錄相同大腦來源的重疊混合。它們的共變異數從不為零。
#   假設獨立通道的演算法（例如對原始樣本使用朴素貝葉斯）會過擬合。
#
# - **認為特徵向量是魔法。** 特徵向量只是給你共變異數橢圓的軸。
#   它們只相對於你計算的共變異數矩陣才有意義。如果你改變資料（不同的實驗階段、
#   不同的受試者），特徵向量也會改變。你在一個階段計算的 CSP 濾波器如果在
#   不重新校準的情況下應用於不同階段，可能會嚴重退化。
#
# - **忽略 FFT 中的正規化。** `np.fft.rfft` 返回非正規化的複數。要取回物理振幅，
#   需除以 N 並乘以 2（用於單側頻譜）。忘記這一點會給你隨訊號長度縮放的數字，
#   使不同錄音之間的比較產生誤導。
#
# **下一章：** 第四章——DSP 基礎：現在你對頻域有了直覺，我們建立實用工具——
# 濾波器、陷波消除和重新參考——在任何特徵提取之前清理原始 EEG。
