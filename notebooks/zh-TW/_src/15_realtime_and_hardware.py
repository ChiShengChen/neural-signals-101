# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/15_realtime_and_hardware.ipynb)
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
# # 第 15 章 — 即時串流（Real-Time Streaming）與低成本硬體  *（附錄）*
#
# > **前置條件：** 第 08 章。
# > **難度：** ★★☆☆☆
# > **執行時間：** 約 1–2 分鐘。
#
# 到目前為止，教程中的一切都是*離線（offline）*的：你把整段錄製放在磁碟上，
# 可以隨意打亂順序、重新執行、預覽未來的樣本。
# 在真實的腦機介面（BCI，Brain-Computer Interface）中，訊號是**逐樣本即時到達的**。
# 就是那一個詞——*現在（now）*——幾乎改變了你做的每一個決定。
#
# 本章展示：
# 1. 一個**模擬即時（simulated real-time）**示範，讓因果推斷（causal inference）變得可見。
# 2. 從離線到線上，*實際上*改變了什麼。
# 3. 在哪裡可以找到**低成本硬體**，讓你能錄製自己的腦部訊號。
#
# ## 學習目標
# 1. 理解在串流情境中「因果（causal）」的含義。
# 2. 視覺化每次試驗的預測結果與偽串流上的平滑準確率。
# 3. 了解主要的實際差異：延遲（latency）、訊號漂移（drift）、校準（calibration）。
# 4. 了解開源硬體選項與串流協定（streaming protocol）。

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from neuro101 import io, datasets as ds, features as ft
from neuro101.eval import leakage_safe_pipeline

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## 第一部分 — 模擬即時示範
#
# ### 設定：離線資料，線上思維
#
# 我們無法把真實的頭戴裝置接進 Jupyter notebook，但我們可以**假裝**。
# 概念如下：
# - **訓練（Train）**一個分類器，使用「歷史」資料（較早的受試者 / 早期試驗）。
# - **串流（Stream）**後續試驗，一次一個，在每次試驗到達時進行預測。
# - 在串流過程中，分類器**從不更新**（凍結、因果）。
# - 我們繪製每次試驗預測的序列，以及滾動準確率視窗（rolling accuracy window）。
#
# 這模擬了一個閉環（closed-loop）系統：在時間 *t*，你使用在本次會話開始前訓練好的模型
# 對當前試驗進行分類。

# %%
# 載入資料：使用前幾個受試者作為「先前會話」的訓練資料，
# 最後一個受試者作為模擬的即時串流。
n_subj = 2 if SMOKE else 3
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
all_subjects = np.unique(subj)

train_subjs  = all_subjects[:-1]          # 較早的受試者 → 訓練
stream_subj  = all_subjects[-1]           # 最後一個受試者 → 模擬即時串流

train_mask   = np.isin(subj, train_subjs)
stream_mask  = subj == stream_subj

X_train, y_train = X[train_mask], y[train_mask]
X_stream, y_stream = X[stream_mask], y[stream_mask]

print(f"在受試者 {list(train_subjs)} 上訓練：{X_train.shape[0]} 次試驗")
print(f"串流受試者 {stream_subj}：{X_stream.shape[0]} 次試驗")
print(f"每次試驗的訊號形狀：{X_train.shape[1]} 個通道 × {X_train.shape[2]} 個樣本 "
      f"（{X_train.shape[2]/sf:.2f} 秒 @ {sf} Hz）")

# %% [markdown]
# ### 訓練分類器——在串流開始前，只訓練一次
#
# **因果規則：** 分類器只看到訓練會話的資料。
# 在推論過程中，它永遠不會看到來自串流會話的任何樣本。

# %%
rt_pipeline = leakage_safe_pipeline([
    ("csp", ft.make_csp(n_components=4)),
    ("clf", LinearDiscriminantAnalysis()),
])
rt_pipeline.fit(X_train, y_train)
print("分類器訓練完成。凍結權重——串流現在開始。")

# %% [markdown]
# ### 執行前先預測
#
# > **在執行下一個儲存格之前：** 每次試驗的準確率會是嘈雜的還是平滑的？
# > 對預測結果使用滾動視窗平滑，會讓它們更穩定嗎？
# > 先在此寫下你的答案，然後再驗證。
# >
# > *你的預測：* _______

# %%
# 模擬串流：一次一個試驗，進行預測並記錄。
n_stream = len(y_stream)
preds   = np.empty(n_stream, dtype=int)
conf    = np.empty(n_stream)          # 最大類別機率 = 信心度（confidence）
correct = np.empty(n_stream)

for t in range(n_stream):
    # !! 因果：在步驟 t 時，只有 X_stream[t] 可用——沒有未來樣本。
    trial = X_stream[t : t + 1]          # 形狀為 (1, channels, samples)
    preds[t]   = rt_pipeline.predict(trial)[0]
    probas_t   = rt_pipeline.predict_proba(trial)[0]
    conf[t]    = probas_t.max()
    correct[t] = int(preds[t] == y_stream[t])

instant_acc = correct.mean()
print(f"串流上的即時（每試驗）準確率：{instant_acc:.3f}  "
      f"（機率基線 = {1/len(np.unique(y)):.2f}）")

# 使用 W 次試驗視窗計算滾動準確率（rolling accuracy）。
W = min(10, n_stream // 4)             # 根據煙霧模式 / 完整模式自適應視窗
rolling_acc = np.full(n_stream, np.nan)
for t in range(W - 1, n_stream):
    rolling_acc[t] = correct[t - W + 1 : t + 1].mean()

# 信心度的滾動平均。
rolling_conf = np.full(n_stream, np.nan)
for t in range(W - 1, n_stream):
    rolling_conf[t] = conf[t - W + 1 : t + 1].mean()

# %% [markdown]
# ### 圖表：模擬即時串流

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

trial_idx = np.arange(n_stream)

# --- 第 1 列：預測類別 vs 真實類別 ---
ax0 = axes[0]
ax0.step(trial_idx, y_stream, where="mid", color="gray",  lw=1.5, label="真實類別")
ax0.step(trial_idx, preds,    where="mid", color="#4878CF", lw=1,  label="預測類別", alpha=0.8)
ax0.set_yticks(np.unique(y_stream))
ax0.set_ylabel("類別標籤")
ax0.legend(loc="upper right", fontsize=8)
ax0.set_title("模擬即時 BCI 串流（因果、凍結分類器）")

# --- 第 2 列：每次試驗正確 / 錯誤 + 滾動準確率 ---
ax1 = axes[1]
hit_idx  = trial_idx[correct == 1]
miss_idx = trial_idx[correct == 0]
ax1.scatter(hit_idx,  np.ones(len(hit_idx)),  marker="|", s=80,
            color="#6ACC65", label="正確", zorder=3)
ax1.scatter(miss_idx, np.zeros(len(miss_idx)), marker="|", s=80,
            color="#D65F5F", label="錯誤",   zorder=3)
ax1.plot(trial_idx, rolling_acc, color="black", lw=2,
         label=f"滾動準確率（W={W}）")
ax1.axhline(instant_acc, ls="--", color="steelblue", lw=1,
            label=f"平均準確率={instant_acc:.2f}")
ax1.axhline(1 / len(np.unique(y_stream)), ls=":", color="gray", lw=1, label="機率基線")
ax1.set_ylim(-0.2, 1.3)
ax1.set_ylabel("準確率（Accuracy）")
ax1.legend(loc="upper right", fontsize=8, ncol=2)

# --- 第 3 列：信心度隨時間的變化 ---
ax2 = axes[2]
ax2.fill_between(trial_idx, 0.5, conf, where=(conf >= 0.5),
                 alpha=0.4, color="#6ACC65", label="高信心度")
ax2.fill_between(trial_idx, conf, 0.5, where=(conf < 0.5),
                 alpha=0.4, color="#D65F5F", label="低信心度")
ax2.plot(trial_idx, rolling_conf, color="black", lw=1.5,
         label=f"滾動信心度（W={W}）")
ax2.axhline(0.5, ls="--", color="gray", lw=1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("試驗編號（→ 時間）")
ax2.set_ylabel("最大類別機率")
ax2.legend(loc="upper right", fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# **你剛才看到的：**
#
# - 每次試驗的準確率是嘈雜的——單次預測可以輕易翻轉。
# - 滾動視窗能平滑噪聲，並揭示*趨勢*：分類器是保持校準狀態還是在漂移？
# - 高信心度並不總是意味著高準確率（CSP+LDA 的信心度並非完美校準）。
#   把它當作品質訊號，而不是基準事實。
# - 分類器**自始至終都是因果的**：它只使用訓練會話的資訊，從不使用未來的串流試驗。
#
# 對比一下**作弊的即時系統**是什麼樣子：如果你先在所有串流資料上訓練，
# 再「重播」它，你會得到更高的準確率——但在實際部署中完全無法重現。

# %% [markdown]
# ## 第二部分 — 即時 vs 離線，哪些東西改變了？
#
# | 問題 | 離線 | 即時 |
# |---|---|---|
# | **延遲（Latency）** | 不在乎；所有資料在磁碟上 | 必須在試驗結束後數十毫秒內完成分類 |
# | **非平穩性（Non-stationarity）/ 漂移（drift）** | 可以跨會話取平均 | 統計特性每小時、甚至每分鐘都在變化 |
# | **校準（Calibration）** | 完整的歷史資料集 | 通常只有短暫的每會話校準（≤ 5 分鐘）|
# | **預測平滑（Prediction smoothing）** | 不需要 | 滾動投票或指數移動平均能減少畫面閃爍 |
# | **回饋迴路（Feedback loop）** | 無 | 使用者會適應系統；系統也可能適應使用者（共同適應，co-adaptation）|
#
# ### 漂移與非平穩性（第 12 章，陷阱 5）
#
# 腦電圖（EEG，Electroencephalogram）統計特性在一次會話中會發生漂移：電極阻抗
# （electrode impedance）會改變、使用者會感到疲倦或更加專注、汗水積聚在電極下方、
# 放大器基線（amplifier baseline）會漂移。早上 9 點訓練的分類器到了 10 點可能明顯變差
# ——不是因為模型不好，而是它訓練時的*資料分布*與傳入訊號不再匹配。
#
# 因此，離線數字會**系統性地高估**即時效能。
# 在典型的 4 類運動想像任務中，差距通常為 5–15 個百分點。
#
# **緩解措施：**
# - **每會話重新校準（Per-session re-calibration）**：在每次會話開始時，
#   對一小段資料進行擬合（或微調）。
# - **自適應分類器（Adaptive classifiers）**：隨著新標記資料的累積，線上更新參數
#   （需注意災難性遺忘，catastrophic forgetting）。
# - **黎曼對齊（Riemannian alignment）**（第 08 章黎曼流水線）：協方差
#   （covariance）幾何結構在跨會話時比原始頻帶功率（band-power）更穩定。

# %% [markdown]
# ## 第三部分 — 低成本硬體：錄製你自己的訊號
#
# 你不需要研究級的 256 通道放大器就可以進行實驗。有幾個平價選項：
#
# ### 消費級 EEG 頭戴裝置
#
# | 設備 | 通道數 | 價格（約） | 備註 |
# |---|---|---|---|
# | **OpenBCI Ganglion** | 4 | ~$200 | 開源韌體；社群活躍；乾式或濕式電極 |
# | **OpenBCI Cyton** | 8（+ Daisy 擴展至 16）| ~$500–800 | 研究品質 ADC；最受歡迎的開源選擇 |
# | **Muse 2 / Muse S** | 4（+ PPG）| ~$200–300 | 非常容易穿戴；適合放鬆 / 專注實驗；藍牙 |
# | **Emotiv EPOC X** | 14 | ~$850 | 生理食鹽水電極；半專業級 |
#
# **注意事項：** 消費級頭戴裝置的通道較少、噪聲基底（noise floor）較高，
# 且電極接觸品質通常不如研究系統。對於清晰的運動想像，你通常需要在運動皮質
# （motor cortex）上方至少 8 個通道（C3、Cz、C4 及鄰近位置）。
# Muse 主要覆蓋額葉 / 顳葉區域，更適合注意力 / 冥想實驗。
#
# ### 串流協定：Lab Streaming Layer（LSL）
#
# **LSL** 是研究中即時生物訊號串流（biosignal streaming）的事實標準。
# 它與傳輸方式無關（USB、藍牙、Wi-Fi），幾乎每款頭戴裝置都有驅動程式。主要函式庫：
#
# - **`pylsl`** — Python LSL 綁定（接收 / 傳送串流）。
# - **`MNE-LSL`**（`pip install mne-lsl`）——將 LSL 封裝成 MNE `Raw` 物件，
#   讓你的 MNE/scikit-learn 流水線幾乎不需改動就能消費即時資料。
# - **`BrainFlow`**（`pip install brainflow`）——統一的 Python（和 C++）API，
#   支援 OpenBCI、Muse、Neurosity、合成板（synthetic board）等。更換硬體目標只需改一行程式碼。
#
# ### 第一個實驗：看見你自己的眼睛眨動
#
# ```python
# # 虛擬碼（Pseudocode）——需要 BrainFlow + 透過 USB 軟體狗連接的 Cyton 或 Ganglion
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# params = BrainFlowInputParams()
# params.serial_port = "/dev/ttyUSB0"       # 請根據你的作業系統調整
# board = BoardShim(BoardIds.CYTON_BOARD, params)
# board.prepare_session()
# board.start_stream()
# # ... 收集資料、繪圖、看見眨眼時的大幅峰值 ...
# board.stop_stream()
# board.release_session()
# ```
#
# 眨眼會在額葉通道（Fp1、Fp2）上產生較大的（約 100–500 µV）慢波。
# 當你閉上眼睛時，阿爾法波（Alpha waves，8–12 Hz）會出現在枕葉通道上。
# 即使在嘈雜的消費級錄製中，這兩者都能立即看到
# ——這是令人滿意的第一個證明，確認頭戴裝置正在運作。

# %% [markdown]
# ## ✅ 概念驗證
#
# **Q1.** 一位研究人員在離線情況下訓練了一個 CSP+LDA 分類器，並回報 78% 的準確率。
# 兩週後，當他們將其部署在即時會話中時，準確率下降到 61%。
# 請列舉兩個可能解釋這個差距的因素。
#
# **A1.** (a) *非平穩性 / 漂移*——EEG 統計特性在會話之間會改變（阻抗、疲勞、電極位置偏移），
# 因此在舊分布上訓練的分類器可能與新的訊號不匹配。(b) *對固定測試集的過度擬合*——如果
# 78% 是在用於模型選擇的集合上測量的（第 14 章的選擇偏誤問題），
# 那麼回報的數字早已虛高。真實的泛化效能始終更接近 61%。
#
# ---
# **Q2.** 你建立了一個即時分類器，每次新的標記試驗到達時就更新其權重。
# 如果標籤來自使用者的按鍵（自我回報），可能出現什麼問題？
#
# **A2.** 按鍵可能*延遲或被錯誤標記*（使用者在心理想像之後才按下，導致時間錯位）。
# 如果標籤總是落後一次試驗，模型就在帶有偏移標籤的情況下訓練——實際上是在監督訊號中
# 加入噪聲。更糟的是，如果系統也用分類器輸出來觸發回饋，錯誤就會累積：
# 錯誤的預測被當成正確的標籤來訓練。
# 強健的共同自適應系統只使用高信心度的預測，或採用獨立的標籤收集階段。
#
# ---
# **Q3.** 你想用消費級 EEG 頭戴裝置進行運動想像實驗。
# 你有一台 Muse 2 和一台 OpenBCI Cyton。你會選哪個，為什麼？
#
# **A3.** **OpenBCI Cyton** 是運動想像的更好選擇。Cyton 有 8 個通道，可以放置在
# 運動皮質上方（C3、Cz、C4），使用主動或被動濕式電極以獲得更好的訊號品質，
# 其開源韌體與 BrainFlow 和 pylsl 整合順暢。Muse 2 將其 4 個電極放在額頭和耳後
# ——覆蓋額葉和顳葉區域，而不是感覺運動區（sensorimotor strip）——
# 因此不適合左 / 右手想像任務。

# %% [markdown]
# ## ⚠️ 常見錯誤
#
# - **測試即時準確率時讓未來資料洩漏進來。** 最常見的形式：
#   「我先在整個串流會話上擬合縮放器，然後重播它。」那個縮放器已經看過未來資料。
#   解決方法：在訓練時凍結所有前處理步驟，在串流期間只使用 `transform`（絕不使用 `fit`）。
# - **忽略漂移。** 在與訓練相同的會話上測量的離線數字，在幾天或幾週後錄製的新會話中
#   將無法維持。當聲稱具有即時準備性時，請務必測量跨會話或跨天的準確率。
# - **將離線數字用於線上部署的信任依據。** 離線準確率 X 並不意味著即時系統會達到 X。
#   延遲、預測平滑、回饋迴路和使用者適應性都引入了離線評估中看不到的新變量。
# - **回報每次試驗的準確率而不使用滾動視窗。** 單試驗 EEG 分類是嘈雜的。
#   滾動或指數移動平均（exponential-moving-average）顯示能給使用者穩定到足以採取行動的回饋。
# - **跳過每會話校準。** 即使是來自當前會話的 2–5 分鐘標記資料，也能大幅縮短漂移差距
#   ——幾乎總是值得花費這點時間。
#
# ---
#
# 🏁 **附錄結束。** 你現在擁有概念上的工具包，可以將離線學到的一切延伸到可運作的即時原型。
# 從 notebook 到頭戴裝置的跳躍比看起來更小——相同的 sklearn 流水線、相同的因果規則，
# 只是資料從串流而來，而不是從檔案讀取。祝你好運！
