# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/07_feature_engineering.ipynb)
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
# # 第 07 章 — 特徵工程（Feature Engineering）
#
# 分類器的效能取決於你餵給它的數值。本章將原始的 epoch（時段）轉換為
# **特徵（features）**：能讓模型學習到資料結構的精簡描述。
#
# ## 學習目標
# 1. **時域（time-domain）** 與 **頻域（frequency-domain）** 特徵（複習與應用）。
# 2. **連結性（Connectivity）**：相干性（coherence）與 **PLV**（相位鎖定值，phase-locking value）。
# 3. **CSP**（共同空間模式，Common Spatial Patterns）用於運動想像（motor imagery）。
# 4. **黎曼（Riemannian）／共變異數（covariance）** 特徵（使用 `pyriemann`）——現代標準基線。
# 5. 哪些特徵需要從資料中*學習*（因此必須僅在訓練集上擬合）。
#
# > **前置條件：** 第 03 章與第 06 章。
# > **難度：** ★★★★☆
#
# **執行時間：** 約 1–2 分鐘（使用少數 BCI IV 2a 受試者，已快取）。

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
from neuro101 import io, datasets as ds
from neuro101 import features as ft

SMOKE = ds.is_smoke()
n_subj = 2 if SMOKE else 3
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X.shape} (試驗數, 通道數, 時間點) @ {sf} Hz | 類別分布={np.bincount(y)}")

# %% [markdown]
# ## 兩種特徵類型（此區別對誠實評估至關重要）
#
# - **無狀態特徵（Stateless features）** 僅由單一試驗計算得出（頻帶功率、變異數、PLV）。
#   在資料切分前計算它們是*沒問題*的——它們不會偷看其他試驗。
# - **學習型特徵（Learned features）** 需從一組帶標籤的試驗中估計參數
#   （CSP 學習空間濾波器；共變異數白化估計參考值）。這些特徵
#   **必須僅在訓練折（training fold）上擬合**，否則會造成資料洩漏（data leakage）。我們在
#   第 08 章使用流水線（pipeline）處理此問題；本章只負責建構這些特徵。

# %% [markdown]
# ## 1. 時域特徵（Time-domain features）
#
# 簡單的逐通道統計量：變異數（variance）、平均絕對值（mean absolute value），以及
# Hjorth「移動率（mobility）」（訊號振動速度的指標）。計算成本低，但出乎意料地有效。

# %%
F_time = ft.time_domain_features(X)
print("時域特徵:", F_time.shape)

# %% [markdown]
# ## 2. 頻域特徵（Frequency-domain features）：頻帶功率（band power）
#
# 運動想像會抑制手部區域上方的 mu/beta 節律
# （事件相關去同步，event-related desynchronisation）。頻帶功率正好能捕捉此現象。讓我們看看
# 中央通道（central channels）的 alpha/beta 功率在左右手想像運動之間是否有所差異。

# %%
F_bp = ft.bandpower(X, sf)
print("頻帶功率特徵:", F_bp.shape)

# 計算每個類別的平均 beta 功率（重塑為頻帶 x 通道的格式）。
n_ch = X.shape[1]
bp_by_band = F_bp.reshape(F_bp.shape[0], len(ft.BANDS), n_ch)
beta_idx = list(ft.BANDS).index("beta")
fig, ax = plt.subplots(figsize=(9, 3))
for cls, color in [(0, "#4c72b0"), (1, "#dd8452")]:
    ax.plot(bp_by_band[y == cls, beta_idx].mean(0), color=color,
            label=f"class {cls} ({'左手' if cls==0 else '右手'})")
ax.set(xlabel="通道索引", ylabel="log beta 功率",
       title="各類別每通道平均 beta 頻帶功率"); ax.legend()
plt.show()

# %% [markdown]
# **解讀上圖（ERD 的實際表現）：** BCI Competition IV 2a 資料集記錄左手
# vs 右手的*想像*運動。當你想像移動**左手**時，**右側**大腦半球（對側，contralateral）的
# 運動皮質 beta 功率下降——這個下降稱為 **ERD（事件相關去同步，Event-Related Desynchronization）**。
# 上方的兩條彩色曲線應在中央通道（x 軸中間附近，接近 C3/Cz/C4）最為分歧：
# 這種分歧就是分類器所學習利用的對側 ERD 特徵。

# %% [markdown]
# ## 這些特徵在大腦中的意義
#
# 特徵不只是數字——每個特徵都對應一個特定的神經科學現象。理解*為什麼*某個特徵有效，
# 能讓除錯分類器或設計更好的特徵變得容易許多。
#
# ### 頻帶功率 ↔ 大腦節律（brain rhythms）
# 大腦在特定頻率產生節律性電振盪：
# - **Alpha（約 8–13 Hz）：** 閉眼且放鬆時最強
#   （視覺皮質「閒置」狀態）。睜眼會抑制它。
# - **Mu/Beta（約 8–30 Hz）位於感覺運動皮質（sensorimotor cortex）上方：** 運動系統的
#   alpha 等效節律。靜止不動時，感覺運動皮質以 mu/beta 頻率運作——一種備而不動的閒置狀態。
#
# ### ERD / ERS — 運動想像的核心訊號
# **事件相關去同步（ERD，Event-Related Desynchronization）：** 一旦你移動*或想像移動*
# 一隻手，**對側（contralateral）** 感覺運動皮質上方的 mu/beta 節律功率**急劇下降**。
# 大腦正在「喚醒」該區域，原本同步的閒置節律因此瓦解。
#
# **事件相關同步（ERS，Event-Related Synchronization）：** 運動結束後，功率反彈——
# 有時超過基線——因為皮質恢復到靜息狀態。
#
# 為何是對側？運動皮質的解剖連接使**左側**半球控制**右側**身體，反之亦然。因此，
# 想像左手運動會使*右側*感覺運動皮質去同步，而想像右手運動會使*左側*感覺運動皮質去同步。
# 運動想像 BCI 分類器實際上是在偵測大腦哪一側的 beta 功率較低。
#
# ### CSP ↔ 第 03 章的幾何概念
# 在第 03 章，我們看到 EEG 通道記錄的是底層來源的**混合訊號**
# （因為電流會穿透頭骨擴散）。CSP 尋找**空間濾波器（spatial filters）**——通道的加權組合——
# 使得一個類別的訊號變異數高而另一個類別的訊號變異數低。對於左右手想像而言，
# 第一個 CSP 濾波器會強調*右側*感覺運動皮質上方的通道（左手想像時變異數高，右手時低），
# 最後一個濾波器則相反。所得的 log 變異數直接以少數幾個數字編碼了左右 ERD 的不對稱性。
#
# ### 共變異數（Covariance）／黎曼（Riemannian）特徵
# 通道共變異數矩陣記錄了試驗期間每對通道如何共同變化的**完整空間模式**。當右側感覺運動皮質
# 去同步（左手想像）時，涉及該區域通道的相關性會改變。黎曼方法利用共變異數矩陣空間的幾何結構，
# 跨試驗比較這些全腦共激活模式——無需手動選擇通道。

# %% [markdown]
# ## 3. 連結性（Connectivity）：相干性（coherence）與 PLV
#
# 連結性衡量通道之間的關聯，而不僅是各自的功率。
# - **相干性（Coherence）**：頻域中的相關性（兩個通道是否在相同頻率共享功率？）。
# - **PLV**（相位鎖定值，phase-locking value）：兩個通道是否*相位同步*，
#   與振幅無關？1 = 完全鎖定，0 = 完全無關。
#
# 這些計算的複雜度為 O(通道數²)，因此我們只對少數通道子集計算以加速。

# %%
picks = slice(0, 6)  # 取前 6 個通道進行快速示範
F_coh = ft.coherence_features(X[:, picks, :], sf)
F_plv = ft.plv_features(X[:, picks, :])
print("相干性特徵:", F_coh.shape, "| PLV 特徵:", F_plv.shape)

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(F_plv.ravel(), bins=30, color="#55a868")
ax.set(xlabel="PLV", title="成對 PLV 分布（6 個通道）")
plt.show()

# %% [markdown]
# ## 4. CSP — 共同空間模式（Common Spatial Patterns，運動想像的主力方法）
#
# CSP **學習**空間濾波器（通道混合方式），使兩個類別的訊號變異數盡可能不同。
# 頂部成分的 log 變異數構成了一個小巧而強大的特徵集。由於 CSP 使用標籤，它是*學習型*
# 特徵——我們在此對全部資料做擬合只是為了*視覺化*；在實際評估中，它必須放在僅訓練的流水線內。

# %% [markdown]
# > **執行前先預測：** BCI Competition IV 2a 使用左手（class 0）和右手（class 1）的想像運動。
# > 根據你剛才讀到的 ERD 知識：
# >
# > 1. 對於*左手*想像，你預期 **beta 功率下降**發生在哪個半球——左側還是右側？
# > 2. CSP 會學習一個對某個類別給出高變異數、對另一個類別給出低變異數的空間濾波器。
# >    該濾波器應強調相對於想像手的*同側（ipsilateral）* 還是*對側（contralateral）* 半球的通道？
# > 3. 你預期兩個 CSP 散點圖群集大量重疊還是清晰分離？為什麼？
# >
# > *執行儲存格後，確認圖形是否與你的預測相符。*

# %%
csp = ft.make_csp(n_components=4)
F_csp = csp.fit_transform(X, y)   # 僅示範用，對全部資料做擬合（見下方警告）
print("CSP 特徵:", F_csp.shape)

# 顯示前兩個 CSP 特徵如何分離類別。
fig, ax = plt.subplots(figsize=(6, 4))
for cls, color in [(0, "#4c72b0"), (1, "#dd8452")]:
    ax.scatter(F_csp[y == cls, 0], F_csp[y == cls, -1], s=12, alpha=0.6,
               color=color, label=f"class {cls}")
ax.set(xlabel="CSP 成分 1 (log 變異數)", ylabel="CSP 成分 4 (log 變異數)",
       title="CSP 使類別線性可分"); ax.legend()
plt.show()

# %% [markdown]
# > ⚠️ **上方的 `fit_transform(X, y)` 使用了整個資料集，包含標籤。**
# > 這是*特徵洩漏（feature leakage）*，會使事後計算的分數虛高。
# > 此處僅用於*繪圖*，沒有測量準確率，所以沒問題。在
# > 第 08 章，我們將 CSP 放在流水線內，使它在每個訓練折上重新擬合。

# %% [markdown]
# ## 5. 黎曼（Riemannian）／共變異數特徵（強大的現代基線）
#
# 用每個試驗的**通道共變異數矩陣**（通道如何共同變化）來表示該試驗。
# 這些矩陣存在於一個彎曲空間；`pyriemann` 將它們投影到平坦的
# **切空間（tangent space）**，使普通的線性分類器能有效運作。這種「共變異數 +
# 切空間」方案是頂尖的 BCI 基線，通常優於手工調整的特徵。

# %%
from pyriemann.estimation import Covariances
covs = Covariances(estimator="oas").fit_transform(X)
print("共變異數矩陣:", covs.shape, "（試驗數, 通道數, 通道數）")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, cls in zip(axes, [0, 1]):
    ax.imshow(covs[y == cls].mean(0), cmap="RdBu_r")
    ax.set_title(f"平均共變異數矩陣 — class {cls}")
plt.tight_layout(); plt.show()

# %% [markdown]
# 兩個類別的平均共變異數矩陣有所不同——這個差異正是黎曼分類器所利用的。
# 我們在下一章建構完整的 `Covariances → TangentSpace → 分類器` 流水線。

# %% [markdown]
# ## ✅ 概念確認（Concept check）
#
# 繼續前請先測試你的理解。
#
# **問題 1 — ERD 的意義：** 在想像手部運動期間，對側感覺運動皮質上方的 mu/beta 功率*下降*。
# 這種功率下降的專業術語是什麼？它為何對 BCI 有用？
#
# **問題 2 — 為何是對側？** 受試者想像移動*左手*。你預期最強的 beta 功率下降出現在
# 哪個半球（左或右）？為什麼？
#
# **問題 3 — CSP 的優化目標：** CSP 透過對兩個類別共變異數矩陣求解廣義特徵值問題來
# 找到空間濾波器。用白話文說，它對一個類別的濾波後訊號最大化哪個屬性，同時對另一個類別最小化？
#
# ---
# **解答：**
#
# 1. 功率下降稱為 **ERD（事件相關去同步，Event-Related Desynchronization）**。它
#    之所以有用，是因為它是一個可靠且具有空間特異性的標誌，甚至在*想像*（非實際）運動時也會出現——
#    因此分類器能在沒有任何肢體動作的情況下偵測運動意圖。
#
# 2. 最強的 beta 功率下降出現在**右側**半球（左手的對側）。運動皮質的連接方式是對側的：
#    左手 → 右側運動皮質 → 右側半球 EEG 通道顯示 ERD。
#
# 3. CSP 對一個類別的濾波後訊號最大化**變異數**（訊號功率），同時對另一個類別最小化。
#    對於左右手想像而言，這能隔離出功率對某一類別高（例如左手想像時的右側半球通道）而對另一類別低的
#    空間模式——直接捕捉對側 ERD 的不對稱性。

# %% [markdown]
# ## ⚠️ 常見錯誤 / 為什麼這樣做是錯的
#
# - **在全部資料上擬合 CSP／共變異數白化／縮放器，再進行評估。**
#   這是最常見的特徵洩漏錯誤 (#1)。學習型特徵必須僅在訓練折上擬合
#   （第 08 章讓這一步自動化）。
# - **跨受試者／跨場次比較絕對頻帶功率。** 應使用相對功率
#   或逐受試者標準化（在訓練集上擬合！）。
# - **在小資料集上使用大量連結性特徵。** O(通道數²) 特徵 + 少量試驗 = 過擬合（overfitting）。
#   請選擇通道或降低維度。
# - **忘記單位／對數變換。** 功率分布為重尾（heavy-tailed）；log 功率對線性模型友善得多
#   （我們的輔助函式已自動取 log）。
# - **將 CSP 視為萬能。** CSP 假設判別資訊存在於頻帶功率的空間模式中——
#   對運動想像很有效，但並非通用。
#
# **下一步：** 第 08 章 — 採用*正確*交叉驗證的傳統機器學習，這些特徵將以誠實的方式與分類器相遇。
