# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/05_preprocessing_and_denoising.ipynb)
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
# # 第 05 章 — 前處理（Preprocessing）與去雜訊（Denoising）
#
# 真實的錄製資料充滿了**偽跡（artefacts）**：不屬於大腦活動的訊號。本章將示範如何找出並移除主要的偽跡，最後以**同一段資料的前後對比**讓你親眼看見清理效果。
#
# ## 學習目標
# 1. 識別常見偽跡：**EOG**（眼動）、**EMG**（肌電）、**動作偽跡**、**市電雜訊**。
# 2. 使用 **ICA**（獨立成分分析，Independent Component Analysis）移除眨眼偽跡。
# 3. 理解 **ASR** 式振幅清理、**分段（epoching）** 與 **基線校正（baseline correction）**。
# 4. 比較同一段資料在清理前後的差異。
#
# > **先修條件：** 第 04 章。
# > **難度：** ★★★☆☆
#
# **執行時間：** 約 1–2 分鐘（一位 PhysioNet 受試者的小型資料；在 CPU 上執行 ICA）。

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
import mne
mne.set_log_level("ERROR")
from neuro101 import io, preprocessing as pp

# %% [markdown]
# ## 偽跡目錄
#
# | 偽跡 | 外觀 | 頻率 | 處理方式 |
# |---|---|---|---|
# | **眨眼（EOG）** | 大幅慢速偏折，前額 | < 4 Hz | ICA／迴歸 |
# | **肌電（EMG）** | 高頻模糊爆發 | > 20 Hz | ICA／拒絕分段 |
# | **動作／電極跳動** | 突然跳變／階梯 | 寬頻 | 拒絕／ASR 式截斷 |
# | **市電雜訊** | 固定頻率嗡嗡聲 | 恰好 50/60 Hz | 陷波濾波（第 02 章） |
# | **心跳（ECG）** | 規律約 1 Hz 尖波 | ~1 Hz | ICA |
#
# 我們載入一位受試者的連續錄製資料並加以清理。

# %%
raw = io.load_physionet_mi_raw(subject=1)
raw_clean_filt = pp.basic_clean_raw(raw, l_freq=1.0, h_freq=40.0, notch=60.0)
print(raw_clean_filt)
print("通道：", raw_clean_filt.ch_names[:6], "...")

# %% [markdown]
# ## ICA：將錄製資料分解為獨立來源
#
# ICA 假設錄製資料是統計上獨立的來源之混合（大腦節律＋眨眼來源＋心跳來源＋……）。它對這些來源進行**解混（unmix）**，讓我們能將偽跡來源歸零後再重新混合其餘部分。由於此資料集沒有專用的 EOG 通道，我們使用前額電極（Fpz/Fp1）作為眨眼的代理指標。

# %%
ica = pp.fit_ica(raw_clean_filt, n_components=15, random_state=0)

# 使用前額通道代理尋找看起來像眼動的成分。
frontal = next((c for c in ("Fpz", "Fp1", "Fp2", "AFz") if c in raw_clean_filt.ch_names), None)
eog_idx = pp.detect_eog_components(ica, raw_clean_filt, ch_name=frontal)
print("前額代理通道：", frontal, "| 被標記為眼動相關的 ICA 成分：", eog_idx)

# 若自動偵測未標記任何成分（視資料而定），則回退至與前額通道相關性最高的成分，
# 以確保示範中*一定*有成分被移除。
if not eog_idx and frontal is not None:
    sources = ica.get_sources(raw_clean_filt).get_data()
    frontal_sig = raw_clean_filt.copy().pick([frontal]).get_data()[0]
    corr = [abs(np.corrcoef(sources[i], frontal_sig)[0, 1]) for i in range(sources.shape[0])]
    eog_idx = [int(np.argmax(corr))]
    print("回退策略：移除與前額通道相關性最高的成分：", eog_idx)

# %% [markdown]
# ## 在「同一段」資料上比較前後差異
#
# 我們套用 ICA（移除被標記的成分）並疊加同一個前額通道在清理前後的波形。眨眼造成的偏折應該會縮小。
#
# > **執行前猜測：** ICA 移除後，前額通道上的眨眼峰值會完全消除，還是只會減少？為什麼「完美移除」反而可能是過度清理的警訊？

# %%
raw_ica = raw_clean_filt.copy()
ica.exclude = eog_idx
ica.apply(raw_ica)

ch = frontal or raw_clean_filt.ch_names[0]
seg = slice(0, int(10 * raw_clean_filt.info["sfreq"]))  # 前 10 秒
t = np.arange(seg.stop) / raw_clean_filt.info["sfreq"]
before = raw_clean_filt.copy().pick([ch]).get_data()[0][seg] * 1e6
after = raw_ica.copy().pick([ch]).get_data()[0][seg] * 1e6

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(t, before, color="#c44e52", lw=0.8, label="ICA 前（眨眼存在）")
ax.plot(t, after, color="#2e8b57", lw=0.8, label="ICA 後（眨眼已減少）")
ax.set(xlabel="時間（秒）", ylabel="µV", title=f"通道 {ch} 的 ICA 偽跡移除")
ax.legend(); plt.show()

# %% [markdown]
# ## ASR 式振幅清理（透明的替代方案）
#
# **ASR**（Artifact Subspace Reconstruction，偽跡子空間重建）從乾淨的參考子空間重建嚴重受損的片段。完整的 ASR 超出本章範圍，但其精神是「馴服極端的偏折」。我們的 `clip_extreme_amplitudes` 函式穩健地截斷尖波——這裡對一個注入了動作偽跡的片段進行處理。

# %%
sig = raw_clean_filt.copy().pick([ch]).get_data()[0][:1000].copy()
sig[500:505] += 400e-6  # 注入一個大的動作尖波（µV 級別）
cleaned = pp.clip_extreme_amplitudes(sig[None, None, :], z_thresh=5.0)[0, 0]
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(sig * 1e6, color="#c44e52", label="含動作尖波")
ax.plot(cleaned * 1e6, color="#2e8b57", label="穩健截斷後")
ax.set(title="ASR 式截斷馴服極端振幅", ylabel="µV"); ax.legend()
plt.show()

# %% [markdown]
# ## 分段（Epoching）與基線校正（Baseline Correction）
#
# **分段（Epoching）** 將連續錄製資料切割為以事件為中心、帶有標籤的試次（trials）。**基線校正（Baseline correction）** 從每個分段中減去事件前視窗的均值，使所有分段都從一個共同的零點開始——這能移除否則會淹沒效應的緩慢漂移。

# %%
events, _ = mne.events_from_annotations(raw_clean_filt, event_id=dict(T1=0, T2=1))
epochs = pp.make_epochs(
    raw_clean_filt, events, dict(left=0, right=1),
    tmin=-0.5, tmax=2.0, baseline=(None, 0),
)
print("分段：", epochs.get_data(copy=False).shape, "（試次, 通道, 時間）")
evoked = epochs["left"].average()
fig = evoked.plot(spatial_colors=False, show=False)
plt.show()

# %% [markdown]
# ## ✅ 概念確認
#
# 1. ICA 將錄製資料分解為獨立成分。為什麼成分數量必須 ≤ EEG 通道數？
# 2. 基線校正減去刺激前視窗的均值。這能移除什麼偽跡？它對該偽跡做了什麼假設？
# 3. 你在整段連續錄製資料（交叉驗證之前）上擬合 ICA，並將所得的成分權重用作特徵。為什麼這構成資料洩漏（data leakage）？
#
# **解答：**
# 1. ICA 求解一個方程組，方程數等於通道數；你無法從比觀測到的混合訊號（通道）更多的獨立來源中還原資訊。
# 2. 基線校正移除緩慢的 DC 漂移／偏移；它假設漂移在基線視窗內是恆定的（或緩慢變化的），並持續進入分段——對於快速的漂移變化，這個假設可能不成立。
# 3. ICA 使用所有試次的訊號統計來學習混合矩陣，包括未來的測試試次。因此測試資料上的成分激活受到測試集統計的影響，違反了訓練／測試獨立性。

# %% [markdown]
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# - **移除過多 ICA 成分。** 每刪除一個成分，也會移除一些大腦訊號。只移除少數明確的偽跡，而非「所有看起來奇怪的東西」。
# - **在*整個*資料集上擬合 ICA／偽跡閾值，然後再進行分類。** ICA 是*無監督*的，因此危險性低於監督式的資料洩漏，但若你的下游特徵依賴它，請在訓練折（train fold）*內部*進行清理的擬合。（第 08 章的管線（pipelines）會讓這一點無懈可擊。）
# - **在看到標籤後才用肉眼拒絕分段。** 這會使結果產生偏差。請在查看類別資訊*之前*設定拒絕閾值。
# - **對 ERP 跳過基線校正。** 緩慢漂移會主導訊號，你的平均波形將毫無意義。
# - **過度清理。** 目標是*誠實*的資料，而非漂亮的資料。一個只在大量手動清理的資料上才有效的模型，在即時使用時將無法存活。
#
# **下一章：** 第 06 章——頻域（FFT、PSD、頻譜圖、小波）以及時間－頻率取捨。
