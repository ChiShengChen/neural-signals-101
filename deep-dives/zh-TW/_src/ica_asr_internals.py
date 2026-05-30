# %% [markdown]
# # 深入探討 — ICA 與 ASR 內部機制
#
# 第 05 章的偽跡（artifact）去除機制實際上是如何運作的。
#
# > **前置條件：** 主課程第 05 章。
# > **難度：** advanced ★★★★☆
# > **不受 5 分鐘 CPU 預算限制。**

# %% [markdown]
# ## 0 — Bootstrap（啟動引導）

# %%
import sys
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ImportError:
    _p = Path.cwd()
    for _ in range(6):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 非互動式後端，在 nbconvert 中安全使用
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
mne.set_log_level("ERROR")

from sklearn.decomposition import FastICA

from neuro101.io import load_physionet_mi_raw
from neuro101.preprocessing import fit_ica, clip_extreme_amplitudes

rng = np.random.default_rng(42)

print("所有匯入成功  |  MNE", mne.__version__)

# %% [markdown]
# ---
# ## 1 — 混合模型（mixing model）：X = A S
#
# 獨立成分分析（ICA，Independent Component Analysis）從一個**生成假設（generative assumption）**出發：
# 感測器記錄的是一小組*統計獨立*源訊號（source signals）的線性疊加。
#
# ### 模型
#
# 設：
# * **S** ∈ ℝ^{n_sources × T} — 源矩陣（每行為一個源訊號）。
# * **A** ∈ ℝ^{n_channels × n_sources} — **混合矩陣（mixing matrix）**（未知）。
# * **X** ∈ ℝ^{n_channels × T} — 電極記錄到的訊號。
#
# 則：
# $$X = A \, S$$
#
# ICA 目標：找到一個**解混矩陣（unmixing matrix）** W ≈ A⁻¹，使得
# $$\hat{S} = W X$$
# 其中 $\hat{S}$ 的各行儘可能統計獨立。
#
# ### 為何需要非高斯性（non-Gaussianity）？
#
# 中央極限定理表明，獨立隨機變數的*和*趨向高斯分布——
# 因此任何單一觀測通道（混合訊號）都比各個源*更*高斯。
# ICA 反向利用這一點：
# 我們尋找**最大化非高斯性**的方向 W（例如最大化峰度（kurtosis）或負熵（negentropy））。
# 最大非高斯性的方向就是已將疊加「解混」回單一源的方向。
#
# **失敗情形：** 若某個源真的是高斯分布，其分布是對稱的，
# 沒有唯一的最大非高斯性方向——高斯分布的任何旋轉仍是高斯分布。
# ICA 無法分離高斯源；源必須是非高斯的
# （超峰度眨眼脈衝、EMG 爆發、正弦振盪——這些都符合條件）。

# %% [markdown]
# ---
# ## 2 — 雞尾酒會（cocktail-party）示範：合成訊號
#
# 我們建立三個獨立源訊號（正弦波、鋸齒波和稀疏拉普拉斯雜訊），
# 用已知矩陣 **A** 混合它們，然後用 `sklearn.decomposition.FastICA` 恢復它們。
# FastICA 使用*負熵*目標（等價於最大化非高斯性），
# 是 Hyvärinen & Oja（1997）提出的標準快速算法。

# %%
# ---------- 生成三個獨立源 ----------
T      = 2000
t      = np.linspace(0, 4 * np.pi, T)

s1 = np.sin(t * 3)                                   # 正弦波
s2 = 2 * (t % (np.pi) / np.pi) - 1                  # 鋸齒波
s3 = rng.laplace(scale=0.3, size=T)                  # 稀疏（拉普拉斯）雜訊
s3 = s3 / np.std(s3)                                 # 單位方差

S = np.vstack([s1, s2, s3])   # (3, T)

# 標準化：ICA 需要（近似）單位方差源
S = S / S.std(axis=1, keepdims=True)

# ---------- 用隨機滿秩矩陣混合 ----------
A_true = rng.standard_normal((4, 3))                 # 從 3 個源生成 4 個「通道」
# 確保條件良好
A_true /= np.linalg.norm(A_true, axis=0, keepdims=True)

X_mix = A_true @ S                                   # (4, T)  — 「記錄」訊號

# 添加少量感測器雜訊
X_mix += 0.05 * rng.standard_normal(X_mix.shape)

# ---------- 用 FastICA 恢復 ----------
ica_syn = FastICA(n_components=3, random_state=0, max_iter=2000, tol=1e-4)
S_rec = ica_syn.fit_transform(X_mix.T).T             # (3, T)

print("源的形狀:", S.shape)
print("混合訊號形狀:", X_mix.shape)
print("恢復訊號形狀:", S_rec.shape)

# 快速相關性檢查（不計排列/符號）
from scipy.optimize import linear_sum_assignment

corr = np.abs(np.corrcoef(S, S_rec)[:3, 3:])        # 3×3 區塊
row_idx, col_idx = linear_sum_assignment(-corr)
for r, c in zip(row_idx, col_idx):
    print(f"  源 {r+1}  <->  恢復成分 {c+1}   |r| = {corr[r, c]:.3f}")

# %% [markdown]
# ### 圖 1 — 源、混合訊號與恢復訊號
#
# ICA 恢復源訊號時**存在任意排列和符號翻轉**。
# 這不是錯誤——沒有原則性的理由偏好 +sin 而非 −sin，
# 或指定特定的排序。重要的是每個恢復的成分時間序列
# 精確匹配一個源（最多差一個純量）。

# %%
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

source_labels   = ["正弦波 (s₁)", "鋸齒波 (s₂)", "拉普拉斯雜訊 (s₃)"]
mixture_labels  = [f"電極 {i+1}" for i in range(4)]
col_titles      = ["獨立源 (S)", "觀測混合訊號 (X = AS)", "恢復訊號 (Ŝ = WX)"]

colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- 第 0 列：源 ---
for i in range(3):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(t, S[i], color=colours[i], lw=0.8)
    ax.set_ylabel(source_labels[i], fontsize=8)
    ax.set_yticks([])
    if i == 0:
        ax.set_title(col_titles[0], fontsize=10, fontweight="bold")
    if i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("時間（任意單位）", fontsize=8)

# --- 第 1 列：4 個混合訊號壓縮至 3 行 ---
for i in range(3):
    ax = fig.add_subplot(gs[i, 1])
    j  = i if i < 3 else 3       # 將 3 行映射至 4 個混合訊號（顯示 3 個以保持對稱）
    ax.plot(t, X_mix[j], color="grey", lw=0.7)
    ax.set_ylabel(mixture_labels[j], fontsize=8)
    ax.set_yticks([])
    if i == 0:
        ax.set_title(col_titles[1], fontsize=10, fontweight="bold")
    if i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("時間（任意單位）", fontsize=8)

# --- 第 2 列：恢復訊號（按相關性排序）---
for plot_i, (r, c) in enumerate(zip(row_idx, col_idx)):
    ax = fig.add_subplot(gs[plot_i, 2])
    # 若需要則翻轉符號以匹配源
    sign = np.sign(np.corrcoef(S[r], S_rec[c])[0, 1])
    ax.plot(t, sign * S_rec[c], color=colours[r], lw=0.8)
    ax.set_ylabel(f"IC {c+1} ↔ s{r+1}", fontsize=8)
    ax.set_yticks([])
    if plot_i == 0:
        ax.set_title(col_titles[2], fontsize=10, fontweight="bold")
    if plot_i < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel("時間（任意單位）", fontsize=8)

fig.suptitle(
    "雞尾酒會示範：ICA 將 4 個感測器訊號解混回 3 個獨立源",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig1_synthetic.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 1 已儲存。")

# %% [markdown]
# **解讀。** 恢復欄應在視覺上與源欄完全相同（允許行的排列和可能的符號翻轉）。
# 上方列印的相關性量化了對齊程度；接近 1.0 的值確認分離成功。
#
# **核心數學。** FastICA 交替執行：
# 1. **不動點更新（Fixed-point update）** — 對 W 的每列 $\mathbf{w}_i$：
#    $$\mathbf{w}_i \leftarrow E[\mathbf{x}\, g(\mathbf{w}_i^\top \mathbf{x})]
#    - E[g'(\mathbf{w}_i^\top \mathbf{x})]\,\mathbf{w}_i$$
#    其中 $g = \tanh$（對數餘弦對比函數的導數）。
# 2. **Gram-Schmidt 去相關** — $W \leftarrow (WW^\top)^{-1/2} W$ 以
#    保持成分相互不相關。
#
# 收斂是三次方的（遠快於梯度上升），在條件良好的問題上通常少於 200 次迭代。

# %% [markdown]
# ---
# ## 3 — 真實 EEG：PhysioNet 運動想像記錄上的 ICA
#
# 我們載入一位受試者的連續記錄，用 MNE 應用 ICA，並檢查成分地形圖（topography）
# 和時間序列。額葉成分（在 Fp1/Fp2/Fpz 上有大權重）是眨眼源的典型特徵。

# %%
import os

smoke = os.environ.get("NEURO101_SMOKE", "0") == "1"

print("正在載入 PhysioNet 受試者 1 的資料…")
raw = load_physionet_mi_raw(subject=1, runs=(4,) if smoke else (4, 8, 12))
print(raw)

# ICA 前的基本清理（ICA 需要 ≥1 Hz 高通濾波；fit_ica 會處理此問題）
raw_clean = raw.copy().filter(1.0, 40.0, verbose="ERROR")
raw_clean.set_eeg_reference("average", projection=False, verbose="ERROR")

n_ica = 10 if smoke else 15
print(f"正在以 {n_ica} 個成分擬合 ICA…")
ica = fit_ica(raw_clean, n_components=n_ica, random_state=0)
print("ICA 已擬合:", ica)

# %% [markdown]
# ### 圖 2 — ICA 成分時間序列與地形圖權重
#
# 每個面板顯示成分**激活（activation）**（$\hat{S} = WX$ 的某行）的前幾秒，
# 以及**地形圖（topography）**（A = W⁻¹ 的對應列，告訴你該源如何投影到頭皮上）。
# 具有高額葉權重和緩慢、大幅偏轉的成分是眨眼成分。

# %%
sfreq    = raw_clean.info["sfreq"]
data_ica = ica.get_sources(raw_clean).get_data()   # (n_comp, n_times)
times    = np.arange(data_ica.shape[1]) / sfreq

# 顯示最多 5 個成分的時間序列及其權重條形圖
n_show  = min(5, n_ica)
t_end   = min(20.0, times[-1])        # 前 20 秒
mask    = times <= t_end

# 混合矩陣：各列為成分地形圖（通道上的縮放比例）
mixing  = ica.mixing_matrix_           # (n_channels, n_components)
ch_names = raw_clean.ch_names

fig2, axes = plt.subplots(n_show, 2, figsize=(13, 2.5 * n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

for k in range(n_show):
    ax_ts  = axes[k, 0]
    ax_top = axes[k, 1]

    # 時間序列
    ts = data_ica[k, mask]
    ax_ts.plot(times[mask], ts, lw=0.6, color="#333333")
    ax_ts.set_ylabel(f"IC {k:02d}", fontsize=8)
    ax_ts.set_yticks([])
    if k == 0:
        ax_ts.set_title("成分激活（前 20 秒）", fontsize=9, fontweight="bold")
    if k < n_show - 1:
        ax_ts.set_xticks([])
    else:
        ax_ts.set_xlabel("時間（秒）", fontsize=8)

    # 地形圖：每通道權重的條形圖（按 |權重| 排序）
    weights = mixing[:, k]
    order   = np.argsort(np.abs(weights))[::-1][:12]   # 前 12 個通道
    ax_top.barh(
        range(len(order)),
        weights[order],
        color=["#d62728" if w > 0 else "#1f77b4" for w in weights[order]],
        height=0.7,
    )
    ax_top.set_yticks(range(len(order)))
    ax_top.set_yticklabels([ch_names[i] for i in order], fontsize=7)
    ax_top.axvline(0, color="k", lw=0.7)
    ax_top.set_xlabel("權重", fontsize=8)
    if k == 0:
        ax_top.set_title("前 12 個通道權重（地形圖）", fontsize=9, fontweight="bold")

fig2.suptitle("ICA 成分 — 激活與地形圖", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig2_components.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 2 已儲存。")

# %% [markdown]
# ### 識別眨眼成分
#
# 眨眼偽跡以**集中於額葉通道（Fp1、Fp2、Fpz）的大而緩慢（約 1–4 Hz）偏轉**出現。
# 尋找時間序列中具有不頻繁大幅尖峰的成分（眨眼頻率約 15 次/分鐘 = 約 0.25 Hz），
# 以及頂部加權通道為額葉的成分。
#
# MNE 的 `find_bads_eog` 透過將每個成分激活與用作替代 EOG（眼電圖）訊號的額葉通道相關來自動化此過程。

# %%
# 嘗試使用額葉通道作為替代的自動 EOG 偵測
frontal_candidates = ["Fp1", "Fp2", "Fpz", "AF3", "AF4"]
proxy_ch = next((c for c in frontal_candidates if c in raw_clean.ch_names), None)

blink_idx = []
if proxy_ch is not None:
    try:
        blink_idx, scores = ica.find_bads_eog(
            raw_clean, ch_name=proxy_ch, threshold=2.5, verbose="ERROR"
        )
        print(f"替代通道: {proxy_ch}")
        print(f"標記為眨眼成分: {blink_idx}")
        if len(scores) > 0:
            for idx, sc in enumerate(scores):
                print(f"  IC {idx:02d}  EOG 相關分數 = {sc:+.3f}")
    except Exception as e:
        print(f"自動偵測略過（{e}）；請手動檢查地形圖。")
else:
    print("未找到額葉通道——請手動檢查地形圖。")

if blink_idx:
    print(f"\n成分 {blink_idx} 將作為眨眼偽跡被排除。")

# %% [markdown]
# ### 應用 ICA 並比較額葉通道的前後效果
#
# 我們在源空間中將標記的成分歸零後再混合回去。
# 結果應顯示額葉通道上**減少的**眨眼偏轉，同時其餘頻譜保持完整。

# %%
if blink_idx:
    raw_corrected = raw_clean.copy()
    ica.exclude = blink_idx
    ica.apply(raw_corrected, verbose="ERROR")
    ica.exclude = []          # 重置，避免影響後續儲存格

    # 比較額葉替代通道（前 30 秒）
    t_seg = 30.0
    mask_seg = times <= t_seg

    ch_idx = raw_clean.ch_names.index(proxy_ch)
    before = raw_clean.get_data()[ch_idx, mask_seg] * 1e6    # V → µV
    after  = raw_corrected.get_data()[ch_idx, mask_seg] * 1e6

    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    ax1.plot(times[mask_seg], before, lw=0.7, color="#d62728", label="ICA 前")
    ax1.set_ylabel("振幅（µV）", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"通道 {proxy_ch} — ICA 校正前", fontsize=10)

    ax2.plot(times[mask_seg], after, lw=0.7, color="#1f77b4", label="ICA 後")
    ax2.set_ylabel("振幅（µV）", fontsize=9)
    ax2.set_xlabel("時間（秒）", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"通道 {proxy_ch} — ICA 校正後", fontsize=10)

    fig3.suptitle("ICA 眨眼去除：前後比較", fontsize=11)
    plt.tight_layout()
    plt.savefig("/tmp/dd_ica_fig3_before_after.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("圖 3 已儲存。")
    print(f"校正前峰對峰: {np.ptp(before):.1f} µV  |  校正後: {np.ptp(after):.1f} µV")
else:
    print("未標記眨眼成分；略過前後比較圖。")

# %% [markdown]
# ---
# ## 4 — ASR 內部機制：子空間投影（subspace projection）與簡化替代
#
# ### 真實 ASR 的數學原理
#
# 偽跡子空間重建（ASR，Artifact Subspace Reconstruction）（Mullen et al. 2015）分三個階段運作：
#
# **第一階段 — 建立乾淨參考子空間（clean reference subspace）。**
# 收集被認為無偽跡的資料段（通常是最安靜的 30 秒）。
# 計算其共變異數：
# $$\Sigma_\text{clean} = \frac{1}{T_\text{clean}} X_\text{clean} X_\text{clean}^\top$$
# 執行 PCA（主成分分析）：$\Sigma_\text{clean} = U \Lambda U^\top$。
# 保留所有主成分（全空間）；記錄特徵值
# $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_C$。
#
# **第二階段 — 滑動並標記損壞窗口。**
# 對每個短窗口 $X_w$（例如 0.5 秒），計算黎曼距離或
# 乾淨子空間中的最大重建方差：
# $$v_k = \mathbf{u}_k^\top X_w X_w^\top \mathbf{u}_k \;/\; (T_w \lambda_k)$$
# 若 $\max_k v_k > \rho^2$（用戶選擇的閾值 $\rho$，MNE 實現中預設為 5），則標記該窗口。
#
# **第三階段 — 重建損壞窗口。**
# 分解標記窗口：$X_w = X_w^{(\text{keep})} + X_w^{(\text{bad})}$，
# 其中「壞」子空間由方差超過閾值的成分張成。
# 將壞子空間替換為零（或來自鄰近乾淨窗口的 Wiener 濾波估計）：
# $$\hat{X}_w = P_\text{keep}\, X_w, \quad P_\text{keep} = \sum_{k:\,v_k \leq \rho^2}
# \mathbf{u}_k \mathbf{u}_k^\top$$
#
# ### 簡化替代：`clip_extreme_amplitudes`
#
# `neuro101.preprocessing.clip_extreme_amplitudes` 是一個**逐通道**穩健裁剪器（clipper）：
# 對每個通道估計中位數（median）和 MAD（絕對中位差，median absolute deviation），
# 然後截斷超出 $\pm z_\text{thresh} \times 1.4826 \times \text{MAD}$ 的任何樣本。
# 這可以處理單通道尖峰偽跡（電極彈出、移動），但**不**跨整個感測器空間投影——
# 它無法捕捉跨通道漫射的偽跡
# （例如影響多個通道的放大器飽和（amplifier saturation））。

# %%
# 以數值說明差異
# -----------------------------------------------------------------------
# 向乾淨段注入兩種類型的偽跡：
#   (a) 單通道尖峰 -> clip_extreme_amplitudes 可以處理
#   (b) 多通道飽和（所有通道同時） -> clip 可以處理
#       但適當的 ASR 會從子空間重建

n_ch, n_t = 15, 5000
sfreq_sim  = 160.0
t_sim      = np.arange(n_t) / sfreq_sim

# --- 模擬具有共變異數結構的乾淨多通道 EEG ---
# 抽取隨機共變異數
A_sim = rng.standard_normal((n_ch, n_ch))
cov   = A_sim @ A_sim.T / n_ch
L = np.linalg.cholesky(cov)
clean = (L @ rng.standard_normal((n_ch, n_t))) * 20e-6   # ~20 µV rms

# --- 注入偽跡 ---
dirty = clean.copy()
# (a) 單通道尖峰
dirty[3, 1200:1210] += 500e-6     # 通道 4 上的 500 µV 尖峰
# (b) 一組通道上的寬頻飽和
dirty[7:12, 3000:3100] += 300e-6  # 通道 8-12 上的 300 µV 偏移

# --- 應用 clip_extreme_amplitudes ---
clipped = clip_extreme_amplitudes(dirty, z_thresh=5.0)

# --- 簡化 ASR：基於 PCA 的子空間重建 ---
def simple_asr(X, clean_ref, threshold_z=5.0):
    """
    最小 PCA 型 ASR 示範（教學用途，非生產級別）。

    參數
    ----------
    X : (n_ch, n_t)   待清理資料
    clean_ref : (n_ch, n_t_ref)  乾淨參考資料
    threshold_z : float   方差閾值（以乾淨 PC 方差為單位）

    回傳
    -------
    X_rec : (n_ch, n_t)   重建資料
    """
    # 步驟 1：從參考建立乾淨子空間
    cov_ref = np.cov(clean_ref)
    eigvals, U = np.linalg.eigh(cov_ref)          # 升序
    eigvals = eigvals[::-1]; U = U[:, ::-1]       # 降序

    # 步驟 2：投影每個窗口並檢查方差比
    win_len = int(0.5 * clean_ref.shape[1] / 10)  # 小窗口
    X_rec   = X.copy()
    n_t_x   = X.shape[1]

    for start in range(0, n_t_x, win_len):
        end  = min(start + win_len, n_t_x)
        Xw   = X[:, start:end]
        T_w  = end - start

        # 此窗口中每個 PC 的方差
        pc_var = np.array([
            (U[:, k] @ Xw @ Xw.T @ U[:, k]) / T_w
            for k in range(len(eigvals))
        ])
        ratio  = pc_var / (eigvals + 1e-30)

        # 僅保留在閾值內的 PC
        keep   = ratio <= threshold_z ** 2
        if keep.all():
            continue   # 窗口是乾淨的

        P_keep = U[:, keep] @ U[:, keep].T
        X_rec[:, start:end] = P_keep @ Xw

    return X_rec

asr_out = simple_asr(dirty, clean, threshold_z=5.0)

# %%
# --- 圖 4：比較兩個通道上的三種方法 ---
fig4, axes4 = plt.subplots(3, 2, figsize=(13, 8), sharex=True)

channels_to_show = [3, 9]   # 尖峰通道和飽和叢集通道

for col, ch in enumerate(channels_to_show):
    scale = 1e6   # V -> µV
    for row, (label, sig, colour) in enumerate([
        ("污染訊號", dirty[ch] * scale, "#d62728"),
        ("clip_extreme_amplitudes", clipped[ch] * scale, "#ff7f0e"),
        ("簡化 PCA-ASR",  asr_out[ch] * scale, "#1f77b4"),
    ]):
        ax = axes4[row, col]
        ax.plot(t_sim, clean[ch] * scale, lw=0.5, color="#aaaaaa", label="乾淨參考")
        ax.plot(t_sim, sig, lw=0.7, color=colour, label=label)
        ax.set_ylabel("µV", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        if row == 0:
            ax.set_title(f"通道 {ch+1}", fontsize=10, fontweight="bold")
        if row == 2:
            ax.set_xlabel("時間（秒）", fontsize=8)

fig4.suptitle(
    "ASR vs. clip_extreme_amplitudes 於注入偽跡上的比較\n"
    "（灰色 = 乾淨參考；左 = 單一尖峰，右 = 多通道飽和）",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("/tmp/dd_ica_fig4_asr_compare.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 4 已儲存。")

# %% [markdown]
# ### 關鍵差異：clip vs. 真實 ASR
#
# | 屬性 | `clip_extreme_amplitudes` | 真實 ASR |
# |---|---|---|
# | **操作對象** | 各通道獨立 | 所有通道聯合（PCA 子空間） |
# | **偽跡模型** | 振幅超過穩健 z 分數 | 任意 PC 的窗口方差超過乾淨資料方差 |
# | **重建方式** | 硬裁剪（非重建） | 將保留的 PC 反投影到感測器空間 |
# | **保留訊號？** | 是，裁剪樣本之外 | 是，在未被標記為損壞的子空間中 |
# | **失效於** | 漫射低振幅相關偽跡 | 與最大神經 PC 對齊的偽跡 |
# | **計算成本** | O(C · T) | O(C² · T) |
#
# **何時使用哪種：** `clip_extreme_amplitudes` 是針對孤立電極彈出或短暫飽和事件的快速初步處理。
# ASR 適用於偽跡是寬頻、跨通道相關的情況
# （例如頭部移動在整個帽子上引起場偽跡）。

# %% [markdown]
# ## ⚠️ A subtler trap
#
# ### ICA/ASR「閾值搜索」洩漏——以及為何 ICA *仍然*不是免費的
#
# **明顯的洩漏**（已在第 05 章和第 12 章中指出）：若你在整個資料集上擬合 ICA，
# 然後使用成分權重作為監督分類器中的特徵，
# 解混矩陣就被測試集統計數據塑造了——這是標準形式的資料洩漏（data leakage）。
#
# **更微妙的陷阱：** ICA 是*無監督*的，所以在切分前對所有資料（訓練 + 測試）擬合它
# 比在所有資料上擬合監督模型危害更小。但「危害更小」並不等於「無害」：
#
# 1. **與標籤相關的偽跡（Label-correlated artifacts）。**
#    假設受試者在某個類別時眨眼更頻繁
#    （例如他們在「休息」提示後放鬆並眨眼）。
#    眨眼成分就會捕捉到與標籤相關的方差。
#    若你**在已知類別標籤之後**透過視覺檢查成分地形圖來選擇要*去除*哪些成分，
#    你可能無意中保留了透過偽跡訊號攜帶標籤資訊的成分——
#    而你永遠不會注意到，因為地形圖看起來合理。
#
# 2. **在同一記錄上調整 ASR 閾值。**
#    ASR 有一個閾值參數 $\rho$（預設約 5）。
#    若你掃描 $\rho$ 值並選擇在*同一*你正在清理的記錄上最大化下游分類準確率的那個，
#    你已對測試集進行了超參數搜索。
#    清理後的資料現在在恰好使殘留偽跡與類別邊界有利對齊的確切設置下具有更低的雜訊——
#    這是一種微妙的、幾乎看不見的循環分析形式。
#
# 3. **在看到標籤後用眼睛觀察選擇成分。**
#    人工檢查 ICA 成分並非不受偏見影響。
#    若研究人員知道預期的效應方向，他們可能更傾向於排除一個「看起來乾淨」
#    但其去除會削弱預期效應的成分，或保留一個「看起來像 EMG」
#    但其去除也削弱了一個混淆因素的成分。
#    這不是欺詐；這是在未充分規定的決策中運作的普通人類確認偏誤（confirmation bias）。
#
# **解決方案：** 在看到準確率數字*之前*確定成分選擇規則——例如
# 「排除所有與額葉 EOG 替代的相關性超過 0.4 且解釋總方差 < 2% 的成分」——
# 並且在評估後不要重新審視規則。
# 記錄閾值、去除的成分數量，
# 並進行置換測試（permutation test）以確認清理後的資料
# 超過機會水準的程度多於閾值搜索所能解釋的。
