# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/10_paradigms_and_applications.ipynb)
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
# # 第 10 章 — 範式與應用（Paradigms & Applications）
#
# 快速瀏覽人們*利用*神經訊號所做的主要事情，每個範式提供一個最小可執行範例。
# 部分使用你已快取的真實資料集；少數使用有清楚標注的**模擬資料（simulations）**——
# 這是因為真實資料集對 101 課程而言過於龐大（並附有真實資料的指引連結）。
#
# ## 學習目標
# 認識並執行每個範式（paradigm）的迷你範例：
# 1. **運動想像（Motor imagery）**（用思想移動）— 真實 BCI IV 2a。
# 2. **P300 / 事件相關電位（ERP，Event-Related Potential）**（大腦對稀少刺激的反應）— 模擬。
# 3. **穩態視覺誘發電位（SSVEP，Steady-State Visual Evoked Potential）**（閃光的穩態反應）— 模擬。
# 4. **睡眠分期（Sleep staging）** — 真實 Sleep-EDF。
# 5. **癲癇偵測（Seizure detection）** — 模擬（不平衡，連結至第 12 章）。
# 6. **神經解碼／腦對文字（Neural decoding / brain-to-text）** — 概念 + 小型多類別解碼。
#
# **執行時間：** 約 2–3 分鐘。

# %% [markdown]
# > **前置條件：** 第 01 章與第 07 章。
# > **難度：** ★★★☆☆

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
from scipy.signal import welch

from neuro101 import io, datasets as ds, features as ft
from neuro101.eval import leakage_safe_pipeline, make_subject_split, evaluate_with_variance, make_block_split

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## 1. 運動想像（Motor imagery）（真實資料：BCI IV 2a）
#
# 從 mu/beta 節律解碼想像的左手與右手動作。我們在第 05–06 章做了完整處理；
# 這裡是使用誠實的跨受試者分數的單儲存格版本。
#
# **為何有效：** 當你*想像*移動手部時，大腦的初級運動皮質（primary motor cortex）
# 在**對側（contralateral）**（即對側腦半球）會被啟動——即使沒有真實動作。
# 這種活動會壓制感覺運動皮質（sensorimotor cortex）上的 **mu 節律（8–12 Hz）**
# 和 **beta 節律（18–26 Hz）**，這個過程稱為**事件相關去同步（ERD，Event-Related Desynchronisation）**
# （在第 07 章中介紹）。因為左手想像驅動*右*腦半球，右手想像驅動*左*腦半球，
# 空間不對稱性就是可解碼的訊號。

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Xmi, ymi, smi = io.load_bnci_2a_epochs(n_subjects=2)
pipe = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LinearDiscriminantAnalysis())])
res = evaluate_with_variance(pipe, Xmi, ymi, cv=lambda: make_subject_split(smi),
                             scoring="accuracy", seeds=(0,))
print(f"運動想像 LOSO 準確率: {res['accuracy']['mean']:.3f}（機率水準 0.5）")

# %% [markdown]
# ## 2. P300 / 事件相關電位（ERP）（模擬）
#
# **ERP**（Event-Related Potential，事件相關電位）是大腦對事件的平均反應。
# **P300** 是稀少、受關注刺激出現後約 300 毫秒的正向電位——這是 P300「拼字器（speller）」的基礎。
# 單次試次（single trials）埋沒在雜訊中；**對多次試次取平均**才能揭示 ERP。
# 我們模擬目標（稀少）與非目標試次。
#
# *真實資料：* MOABB P300 資料集（例如 `BNCI2014_009`）或 MNE 樣本聽覺任務。
#
# **為何有效：** P300 在大腦偵測到*令人驚訝且相關*的事物時產生——
# 即一連串常見刺激中的稀少項目。這被稱為**奇球範式（oddball paradigm）**：
# 目標刻意設計得很稀少（例如拼字器中每 6 次閃光中只有 1 次），
# 使大腦只對它們產生這個大型正向偏折（large positive deflection）。
# 因為非目標很常見，不會產生此類反應，多次重複平均後可獲得清晰的對比訊號以減少雜訊。

# %%
sf, T = 250.0, int(0.8 * 250)
t = np.arange(T) / sf
def p300_wave(amp):  # 以 ~300 ms 為中心的波形
    return amp * np.exp(-((t - 0.30) ** 2) / (2 * 0.05 ** 2))
n = 200
targets = np.array([p300_wave(8e-6) + 5e-6 * rng.standard_normal(T) for _ in range(n)])
nontar  = np.array([p300_wave(0.0)  + 5e-6 * rng.standard_normal(T) for _ in range(n)])

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(t * 1000, targets[0] * 1e6, color="#bbb", lw=0.6, label="單次目標試次（含雜訊）")
ax.plot(t * 1000, targets.mean(0) * 1e6, color="#c44e52", lw=2, label="目標平均（P300！）")
ax.plot(t * 1000, nontar.mean(0) * 1e6, color="#4c72b0", lw=2, label="非目標平均")
ax.axvline(300, ls="--", color="gray"); ax.set(xlabel="時間（ms）", ylabel="µV",
          title="P300：取平均後可揭示 ~300 ms 的波峰"); ax.legend()
plt.show()

# %% [markdown]
# ## 3. 穩態視覺誘發電位（SSVEP）（模擬）
#
# 如果你盯著以 12 Hz 閃爍的光，你的視覺皮質（visual cortex）會以 12 Hz（及其諧波）
# 產生**穩態（steady-state）**振盪。SSVEP 腦機介面（BCI）以不同頻率顯示多個閃爍目標，
# 並讀出你注視的那個——只需找出功率頻譜密度（PSD）中最強的波峰即可。
#
# *真實資料：* MOABB SSVEP 資料集（例如 `SSVEPExo`、`Nakanishi2015`）。
#
# **為何有效：** 視覺皮質天生就會跟隨其輸入的節律——
# 這種特性稱為**頻率標記（frequency tagging）**。當頻率 *f* 的閃爍刺激持續驅動視網膜時，
# 整個視覺通路（視網膜 → 視丘 → V1）都會鎖定在 *f* 及其諧波上。
# 由於 BCI 介面中每個目標以*不同*頻率閃爍，
# 讀出枕葉（occipital）EEG 的主導頻率即可立即判斷使用者注視的位置——無需訓練資料。

# %% [markdown]
# **執行前：** 使用者注意 12 Hz 閃光——哪個頻率將主導功率頻譜：8 Hz、12 Hz 還是 15 Hz？
# 寫下你的預測，然後執行儲存格並對照圖形。

# %%
sf = 250.0; t = np.arange(int(3 * sf)) / sf
stim_freqs = [8.0, 12.0, 15.0]
attended = 12.0  # 使用者「注視」的頻率
eeg = sum((1.0 if f == attended else 0.05) * np.sin(2 * np.pi * f * t) for f in stim_freqs)
eeg = eeg + 1.5 * rng.standard_normal(t.size)
f, psd = welch(eeg, sf, nperseg=int(sf * 2))
detected = stim_freqs[int(np.argmax([psd[np.argmin(abs(f - sf0))] for sf0 in stim_freqs]))]
print(f"注意 {attended} Hz -> 偵測到 {detected} Hz（波峰挑選）")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(f, psd, color="#55a868")
for sf0 in stim_freqs: ax.axvline(sf0, ls="--", color="gray")
ax.set(xlim=(5, 20), xlabel="頻率（Hz）", ylabel="功率",
       title="SSVEP：受注意的閃爍頻率主導頻譜")
plt.show()

# %% [markdown]
# ## 4. 睡眠分期（Sleep staging）（真實資料：Sleep-EDF）
#
# 將一整夜 EEG 的每個 30 秒片段分類為睡眠階段
# （清醒（Wake）／N1／N2／N3／快速眼動（REM））。各類別非常不平衡——
# 我們使用**平衡準確率（balanced accuracy）**（第 12 章解釋原因），
# 並在夜間內使用區塊感知（block-aware）分割。
#
# **為何有效：** 每個睡眠階段都有獨特的 EEG 指紋（fingerprint）。
# 在 **N2** 期間可見**睡眠紡錘波（sleep spindles）**（短暫的 12–15 Hz 爆發）
# 和 **K 複合波（K-complexes）**（大型尖波）。
# **N3**（深度睡眠）以緩慢的 **delta 波（< 4 Hz，高振幅）**為主。
# **REM** 看起來更像清醒——低振幅混頻活動，伴隨快速眼球運動爆發。
# 這些特定階段的節律源自於丘腦皮質迴路（thalamocortical circuits）
# 在不同運作模式之間切換，使其能從頻帶功率（band power）特徵中可靠地分類。

# %%
from sklearn.ensemble import RandomForestClassifier
Xsl, ysl, ssl = io.load_sleep_edf_epochs(n_subjects=1)
sf_sl = ds.DATASETS["sleep_edf"].sfreq_hz
Fsl = ft.bandpower(Xsl, sf_sl)  # 每個 30 秒片段的頻帶功率
clf = leakage_safe_pipeline([("rf", RandomForestClassifier(n_estimators=100, random_state=0))])
res_sl = evaluate_with_variance(clf, Fsl, ysl,
                                cv=lambda: make_block_split(len(ysl), n_splits=4),
                                scoring=("accuracy", "balanced_accuracy"), seeds=(0,))
print(f"睡眠分期 accuracy={res_sl['accuracy']['mean']:.3f}  "
      f"balanced_accuracy={res_sl['balanced_accuracy']['mean']:.3f}")
print("（accuracy 看起來較高是因為大多數片段是 N2 階段——見第 09 章）")

# %% [markdown]
# ## 5. 癲癇偵測（Seizure detection）（模擬，不平衡）
#
# 癲癇發作（seizures）是**罕見**事件，具有高振幅節律性活動。
# 主要挑戰是**類別不平衡（class imbalance）**：99% 的錄製片段是正常的。
# 我們模擬「正常」與「癲癇發作」片段，並預覽第 12 章將深入剖析的指標陷阱。
#
# *真實資料：* CHB-MIT 頭皮 EEG、TUH 癲癇語料庫（兩者都很龐大）。
#
# **為何有效：** 癲癇發作是由**過度同步（hypersynchronous）**神經元放電引起——
# 數千個通常獨立放電的神經元全部鎖定在一起，以高振幅節律性爆發放電。
# 這種異常同步在 EEG 中產生異常大且規律的振盪，
# 與健康大腦的低振幅、不規則背景活動明顯不同，
# 使得振幅和頻譜特徵成為可靠的鑑別指標。

# %%
sf = 256.0; T = int(2 * sf); t = np.arange(T) / sf
def normal_seg():  return rng.standard_normal(T)
def seizure_seg(): return 3 * np.sin(2 * np.pi * 4 * t) + rng.standard_normal(T)  # 大型慢節律
n_total, n_seiz = 300, 15  # 5% 癲癇發作（不平衡！）
# 將稀少的癲癇發作均勻分散於整個錄製中（每約第 20 個片段），
# 以避免它們全部集中在末尾——否則區塊分割會將它們隔離。
seiz_positions = set(np.linspace(0, n_total - 1, n_seiz).astype(int))
segs, yse = [], []
for i in range(n_total):
    if i in seiz_positions:
        segs.append(seizure_seg()); yse.append(1)
    else:
        segs.append(normal_seg()); yse.append(0)
Xse = np.array(segs)[:, None, :]
yse = np.array(yse)
Fse = ft.bandpower(Xse, sf)
print(f"癲癇資料集: {np.bincount(yse)}（類別 1 = 癲癇發作，僅 {100*yse.mean():.0f}%）")

from sklearn.linear_model import LogisticRegression
clf = leakage_safe_pipeline([("clf", LogisticRegression(max_iter=500))])
res_se = evaluate_with_variance(clf, Fse, yse,
                                cv=lambda: make_block_split(len(yse), groups=np.arange(len(yse)), n_splits=5),
                                scoring=("accuracy", "balanced_accuracy", "f1_macro"), seeds=(0,))
print(f"  accuracy={res_se['accuracy']['mean']:.3f}  "
      f"balanced_accuracy={res_se['balanced_accuracy']['mean']:.3f}  "
      f"f1_macro={res_se['f1_macro']['mean']:.3f}")
print("  -> accuracy 看起來很好，但偵測器可能漏掉癲癇發作（第 09 章）。")

# %% [markdown]
# ## 6. 神經解碼／腦對文字（Neural decoding / brain-to-text）（概念 + 小型解碼）
#
# **神經解碼（Neural decoding）**將大腦活動映射到外部變數：游標、機械手臂，
# 或——在最前沿——從皮質內植入物（intracortical implants）解碼**文字／語音**
# （例如近期的「腦對文字」語音神經義肢）。原理與你已做過的多類別解碼相同。
# 這裡我們解碼 **4 類別**運動想像（左手／右手、腳部、舌頭）作為多目標解碼器的替代。
#
# **為何有效：** 語音和語言涉及許多小型皮質區域（例如運動皮質中的音素表示僅相距數毫米）
# 的快速、精細活動模式。頭皮 EEG 無法解析這種空間細節——頭骨和頭皮會將訊號混合在一起。
# 腦對文字的突破性成果依賴**侵入性（intracortical）電極陣列**（例如猶他陣列（Utah arrays））——
# 以亞毫米解析度記錄個別神經元或小型神經元群體。
# 第 13 章討論關於頭皮 EEG「心靈感應」宣稱的倫理和誠實議題。

# %%
X4, y4, s4 = io.load_bnci_2a_epochs(n_subjects=2,
                                    classes=("left_hand", "right_hand", "feet", "tongue"))
pipe4 = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() +
                              [("clf", LogisticRegression(max_iter=500))])
res4 = evaluate_with_variance(pipe4, X4, y4, cv=lambda: make_subject_split(s4),
                              scoring="accuracy", seeds=(0,))
print(f"4 類別解碼 LOSO 準確率: {res4['accuracy']['mean']:.3f}（機率水準 0.25）")

# %% [markdown]
# ## ✅ 概念自測（Concept check）
#
# 繼續前請先測試你的理解。
#
# **Q1.** 在 P300 範式中，為什麼目標要刻意設計得*稀少*，而不是和非目標一樣頻繁出現？
#
# **Q2.** 什麼是*頻率標記（frequency tagging）*？為什麼它讓 SSVEP BCI 不需要任何訓練資料就能運作？
#
# **Q3.** 運動想像解碼利用了對側不對稱性——*對側（contralateral）*在這裡是什麼意思？哪些節律會改變？
#
# **解答：**
# 1. P300 反應由*驚訝感*驅動——只有當刺激稀少且與任務相關時（**奇球（oddball）**效應），
#    大腦才會在 ~300 ms 時發出大型正向電位。
#    如果目標出現的頻率和非目標一樣，大腦會適應，P300 就會消失，失去可解碼的對比訊號。
# 2. **頻率標記（frequency tagging）**是視覺皮質以節律性視覺刺激的相同頻率振盪的傾向。
#    因為每個 BCI 目標以唯一頻率閃爍，枕葉 EEG 中的主導波峰直接識別受注意的目標——
#    無需受試者特定的校準。
# 3. *對側*意指「在對面一側」：想像*右*手動作會啟動*左*運動皮質，反之亦然。
#    被啟動的腦半球會壓制 **mu（8–12 Hz）** 和 **beta（18–26 Hz）** 節律
#    （事件相關去同步，ERD），創造 CSP 和類似方法所利用的空間不對稱性。

# %% [markdown]
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# - **對罕見事件（癲癇）使用準確率（accuracy）。** 永遠預測「無癲癇發作」的模型準確率達 95%，
#   卻毫無用處。應使用平衡準確率（balanced accuracy）／F1／ROC-AUC（第 12 章）。
# - **單次試次 ERP 宣稱。** ERP 需要取平均；「在一次試次中偵測到 P300」的宣稱需要非常謹慎的統計。
# - **SSVEP 資料時間不足。** 頻率解析度需要數秒的訊號；時窗太短會使相鄰刺激頻率模糊在一起。
# - **在睡眠分期中以隨機分割混合多夜／多位受試者。** 連續片段高度相關——
#   應依夜晚／受試者分割（第 12 章，陷阱 #1/#2）。
# - **過度宣稱「腦對文字（brain-to-text）」。** 真正的語音解碼使用侵入性植入物、大型模型和大量資料。
#   頭皮 EEG「心靈感應」示範通常是洩漏（leakage）。
#
# **下一章：** 第 11 章 — *統計直覺*（均值 ± 標準差的含義，為何兩個長條間的小差距通常是雜訊），
# 這是第 12 章（最重要的章節——展示評估如何以 WRONG → RIGHT 配對方式說謊）的基礎。
