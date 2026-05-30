# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度探討 — 是大腦還是偽影？（解碼混淆因素）
#
# 許多「BCI」論文回報的高準確率，其實來自眼電（EOG）或肌電（EMG）
# 偽影（artifact），而非神經活動本身。本深度探討說明如何偵測這個陷阱並加以防範。
#
# > **前置知識：** 主課程第 03、05 及 12 章。
# > **難度：** 進階 ★★★★☆
# > **本篇是第 12 章「資料洩漏」的生理學孿生篇。**

# %%
# --------------------------------------------------------------------------- #
# 引導程式 — 無論從儲存庫根目錄或 deep-dives/_src/ 執行，均可找到 neuro101
#（比主要筆記本多一層父目錄）。
# --------------------------------------------------------------------------- #
import sys
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(6):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src"))
            break
        _p = _p.parent
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 非互動式後端 — nbconvert 環境下安全使用
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mne
mne.set_log_level("ERROR")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

from neuro101 import datasets as ds
from neuro101.io import load_bnci_2a_epochs

SMOKE = ds.is_smoke()
RNG = np.random.default_rng(42)

print(f"neuro101 OK  |  MNE {mne.__version__}  |  SMOKE={SMOKE}")

# %% [markdown]
# ---
# ## 1  問題說明
#
# 分類器（classifier）本質上是一台最佳化機器：它會在輸入特徵中找到
# *任何*能預測標籤的模式，完全**不在意**該模式究竟來自運動皮質的神經活動、
# 額葉電極記錄到的肌肉抽搐，還是眼球運動產生的緩慢角膜視網膜電位。
#
# ### 為何眼動偽影對運動想像 BCI 特別危險
#
# | 實驗設計 | 實際發生的狀況 |
# |---|---|
# | 受試者想像移動**左手** | 受試者同時（隱微地）傾向略微向**左**看 |
# | 受試者想像移動**右手** | 受試者同時（隱微地）傾向略微向**右**看 |
#
# 向左掃視（saccade）產生的眼電（EOG，electrooculogram）訊號，在左側額葉電極
# 呈大幅緩慢正偏轉，右側額葉呈負偏轉。此訊號透過體積導通（volume conduction）
# 廣泛傳至整個額葉腦電——直接影響 **Fz、FCz、FC1、FC2** 等通道，
# 而這些通道幾乎出現在所有 BCI 電極蒙太奇（montage）中。
#
# CSP（共空間模式，Common Spatial Pattern）+ LDA 解碼器會樂於擷取這種額葉不對稱，
# 並將其回報為「運動想像解碼準確率」。一旦解碼器移至眼動與任務相關性被打破
# 的情境（不同指令、固定注視條件，或只是換了一天），準確率就會崩潰。
#
# ### 為何這*不僅僅*是前處理不當的問題
#
# 即使已使用 ICA 去除眨眼偽影，**與掃視相關**的 EOG 成分也可能無法被
# 乾淨移除（眨眼與掃視在成分空間中的表現不同）。更糟的是，如果掃視
# 本身是由*實驗設計造成*的，再多的前處理都無濟於事——唯一的解方是
# 更好的實驗控制。

# %% [markdown]
# ---
# ## 2  受控示範：注入合成「眼動」偽影
#
# ### 設置
#
# 我們載入 BCI IV 2a（左手對右手運動想像）並建立基線 CSP+LDA 準確率。
# 接著**合成注入**一個與標籤相關的功率調變至額葉/額中央通道——
# 模擬受試者隨想像動作系統性移動眼球時所發生的狀況。
#
# **注入規則：**
# * 第 0 類（左手）：額葉通道（Fz、FC1–4、FCz）的振幅放大為**3 倍**
#   → 單一類別的額葉 alpha/beta 功率大幅增加。
# * 第 1 類（右手）：額葉通道維持不變。
#
# 此放大模擬真實的左偏掃視所產生的額葉功率不對稱（左側額葉 EOG 正偏
# → 投影到帶通濾波後的 EEG 上呈現大振幅正向振盪）。
#
# MOABB 典範已將資料帶通濾波至 8–30 Hz（mu/beta 範圍），因此偽影必須
# *在頻帶內*才能存活——乘法振幅放大透過縮放現有的帶內活動達到此目的。

# %%
# --- 載入資料 ---------------------------------------------------------------
n_subj = 2 if SMOKE else 3
print(f"正在載入 {n_subj} 位受試者的 BCI IV 2a 資料 …")
X, y, subj = load_bnci_2a_epochs(n_subjects=n_subj)
n_trials, n_ch, n_times = X.shape
sfreq = ds.DATASETS["bnci_2a"].sfreq_hz

# BCI IV 2a 的通道名稱（MOABB 提供的 22 個 EEG 通道，按順序排列）
CH_NAMES = [
    "Fz",  "FC3", "FC1", "FCz", "FC2", "FC4",   # 索引 0-5  （額葉/FC）
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4", "C6",  # 索引 6-12（感覺運動區）
    "CP3", "CP1", "CPz", "CP2", "CP4",            # 索引 13-17（中頂葉）
    "P1",  "Pz",  "P2",  "POz",                   # 索引 18-21（頂葉）
]
assert len(CH_NAMES) == 22

# 通道群組索引
FRONTAL_IDX    = list(range(0, 6))                # Fz, FC3, FC1, FCz, FC2, FC4
SENSORIMOTOR_IDX = [7, 9, 11]                     # C3, Cz, C4 — 關鍵的 MI 通道
ABLATION_IDX   = list(range(6, 22))               # 除額葉/FC 以外的所有通道

print(f"X 形狀: {X.shape}  |  各類別試次數: {np.bincount(y)}")
print(f"額葉通道：     {[CH_NAMES[i] for i in FRONTAL_IDX]}")
print(f"感覺運動區（C3/Cz/C4）: {[CH_NAMES[i] for i in SENSORIMOTOR_IDX]}")

# %%
# --- 注入合成的與標籤相關偽影 ----------------------------------------------
# 第 0 類（左手）：額葉通道放大 3 倍 → 大幅增加額葉 alpha/beta 功率
# 第 1 類（右手）：不做任何更改
X_art = X.copy()
ARTIFACT_SCALE = 3.0

for i, trial_y in enumerate(y):
    if trial_y == 0:
        X_art[i, FRONTAL_IDX, :] *= ARTIFACT_SCALE

frontal_std_before = X[:, FRONTAL_IDX, :].std()
frontal_std_after  = X_art[y == 0, :, :][:, FRONTAL_IDX, :].std()
print(f"額葉通道標準差：注入前={frontal_std_before:.2f} µV  "
      f"| 注入後（第 0 類）: {frontal_std_after:.2f} µV")

# %%
# --- 建立標準 CSP+LDA 管線並評估 -----------------------------------------
def make_pipe():
    return Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("評估基線（無偽影）…")
acc_base = cross_val_score(make_pipe(), X, y, cv=cv, scoring="accuracy")

print("評估注入偽影後的資料 …")
acc_art = cross_val_score(make_pipe(), X_art, y, cv=cv, scoring="accuracy")

print("評估偽影資料去除額葉通道後的結果 …")
X_art_abl = X_art[:, ABLATION_IDX, :]   # 去除 Fz/FC 通道
acc_abl = cross_val_score(make_pipe(), X_art_abl, y, cv=cv, scoring="accuracy")

print()
print(f"基線準確率：               {acc_base.mean():.3f} ± {acc_base.std():.3f}")
print(f"注入偽影後：               {acc_art.mean():.3f} ± {acc_art.std():.3f}")
print(f"去除額葉通道後：           {acc_abl.mean():.3f} ± {acc_abl.std():.3f}")

# %% [markdown]
# **解讀數據。**
# 「含偽影」條件顯示出準確率大幅躍升——CSP 濾波器立即抓住額葉功率不對稱。
# 去除額葉通道（Fz、FC1–4、FCz）後，準確率回落至接近基線。
# 解碼器的表觀提升完全源自偽影。

# %% [markdown]
# ### 圖一 — 準確率：基線 / 偽影 / 消融
#
# 三欄長條圖讓結論一目了然。紅色虛線標示機會水準（50%）。
# 偽影存在時出現「顯著」的準確率躍升，一旦去除受汙染的通道隨即消失。

# %%
fig, ax = plt.subplots(figsize=(7, 5))

labels_bar = ["基線\n（無偽影）", "注入\n偽影", "消融\n（去除額葉）"]
means = [acc_base.mean(), acc_art.mean(), acc_abl.mean()]
stds  = [acc_base.std(),  acc_art.std(),  acc_abl.std()]
colors = ["#4c72b0", "#c44e52", "#55a868"]

bars = ax.bar(labels_bar, means, yerr=stds, capsize=6,
              color=colors, edgecolor="k", linewidth=0.8, width=0.5,
              error_kw=dict(elinewidth=1.5, ecolor="black"))

# 在長條頂端標註數值
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
            f"{m:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Chance (50 %)")
ax.set_ylim(0.3, 1.15)
ax.set_ylabel("Accuracy (5-fold CV)", fontsize=12)
ax.set_title("CSP+LDA accuracy — does the decoder read brain or artifact?\n"
             f"BCI IV 2a, {n_subj} subject(s), left vs. right hand", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlabel("Condition", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/artifact_confounds_fig1.png", dpi=120)
plt.show()
print("圖一已儲存。")

# %% [markdown]
# ---
# ## 3  誠實性檢查 B — 檢視模型最倚重哪些通道的權重
#
# 若一個運動想像解碼器真正在讀取皮質振盪，其**空間模式**（spatial patterns，
# 注意：非濾波器，見下方說明）應主要載荷於**中央/感覺運動區通道**：
# C3、Cz、C4 及其鄰近通道。
#
# > **模式（patterns）與濾波器（filters）的差異。** 空間*濾波器* $w$ 是用來
# > 乘以資料的向量；它*不是*來源的頭皮地形。空間*模式* $a$ 才代表來源在頭皮
# > 上的樣貌，計算方式為
# > $a = \Sigma_c\, w / (w^\top \Sigma_c w)$，其中 $\Sigma_c$ 為複合協方差矩陣。
# > MNE 將此儲存於 `CSP.patterns_`。**繪製模式，而非濾波器**，才能了解哪些
# > 通道具有生理活性。
#
# 我們在完整資料上（非在交叉驗證內——這是診斷用途，不是效能估計）分別針對
# 每個條件擬合 CSP，並比較各通道平均絕對模式權重。

# %%
# 在完整資料上擬合 CSP（僅用於診斷——此處不需要 CV 切分）
csp_base_full = CSP(n_components=4, reg="ledoit_wolf", log=True)
csp_art_full  = CSP(n_components=4, reg="ledoit_wolf", log=True)
csp_base_full.fit(X, y)
csp_art_full.fit(X_art, y)

# patterns_ 形狀：(n_channels, n_channels) — 完整的 22 個成分
# 使用的 4 個成分為：[0, 1, -2, -1]（2 個最屬第 0 類，2 個最屬第 1 類）
USED_COMPS = [0, 1, -2, -1]
base_patterns = np.abs(csp_base_full.patterns_[USED_COMPS]).mean(axis=0)  # (22,)
art_patterns  = np.abs(csp_art_full.patterns_[USED_COMPS]).mean(axis=0)   # (22,)

# 各自正規化至最大值，方便視覺比較
base_patterns_norm = base_patterns / base_patterns.max()
art_patterns_norm  = art_patterns  / art_patterns.max()

print("各通道平均絕對模式權重（正規化至 1.0）")
print(f"{'通道':8s}  {'基線':10s}  {'偽影':10s}  {'群組':15s}")
for i, ch in enumerate(CH_NAMES):
    grp = ("frontal/FC" if i < 6 else
           "sensorimotor" if i in SENSORIMOTOR_IDX else
           "other")
    print(f"{ch:8s}  {base_patterns_norm[i]:10.3f}  {art_patterns_norm[i]:10.3f}  {grp}")

# %% [markdown]
# **關鍵觀察。** 在偽影條件下，額葉/FC 通道主導 CSP 模式，其權重比基線
# 大了數個量級。真實的運動想像呈現較為平衡的分布，中央通道（C3/Cz/C4）
# 具有相當的權重。若模式的生理解讀不合理——所有權重集中於額葉，感覺
# 運動區幾乎為零——則是紅旗警示。

# %% [markdown]
# ### 圖二 — 通道權重分布：基線 vs. 注入偽影
#
# 按解剖位置排序（額葉 → 中央 → 頂葉）的各通道長條圖，
# 揭示空間重心的轉移。**紅色**長條 = 偽影條件；**藍色**長條 = 乾淨基線。
# 左側的額葉通道在真正的運動解碼器中應僅獲得輕微的權重。

# %%
fig, ax = plt.subplots(figsize=(13, 5))

x_pos = np.arange(n_ch)
width = 0.38

ax.bar(x_pos - width / 2, base_patterns_norm, width,
       label="Baseline (no artifact)", color="#4c72b0", alpha=0.85, edgecolor="k", lw=0.5)
ax.bar(x_pos + width / 2, art_patterns_norm, width,
       label="With injected artifact", color="#c44e52", alpha=0.85, edgecolor="k", lw=0.5)

# 標示額葉區域
ax.axvspan(-0.5, 5.5, alpha=0.08, color="orange", label="Frontal/FC channels")
# 標示感覺運動區域
ax.axvspan(6.5, 12.5, alpha=0.08, color="green", label="Sensorimotor (C3/Cz/C4)")

ax.set_xticks(x_pos)
ax.set_xticklabels(CH_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Normalised mean |CSP pattern weight|", fontsize=10)
ax.set_title("Spatial emphasis of the CSP decoder — which channels does it trust?\n"
             "(high weight = decoder relies heavily on that channel)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-0.6, n_ch - 0.4)
ax.set_ylim(0, 1.25)

# 標注兩個通道群組
ax.text(2.5, 1.18, "Frontal / FC\n(artifact target)", ha="center",
        fontsize=8, color="darkorange", fontweight="bold")
ax.text(9.0, 1.18, "Sensorimotor\n(true MI region)", ha="center",
        fontsize=8, color="darkgreen", fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/artifact_confounds_fig2.png", dpi=120)
plt.show()
print("圖二已儲存。")

# %% [markdown]
# **解讀圖二。** 在偽影條件（紅色）中，額葉/FC 通道遠超其他所有通道——
# 解碼器本質上成了一個額葉功率計。在乾淨基線（藍色）中，中央與頂葉
# 通道對額葉中央通道提供有意義的補充。對於真正的運動解碼器，應看到
# C3/Cz/C4 顯著突出；如果這些通道被 Fz/FCz 所掩蓋，則應懷疑結果
# 由偽影驅動。

# %% [markdown]
# ---
# ## 4  真實資料的注意事項 — 解碼能力可能來自非感覺運動通道
#
# 即使沒有我們的合成注入，BCI IV 2a 中有一部分解碼能力來自
# 標準 C3/Cz/C4 感覺運動條帶以外的通道。這不一定是偽影——
# 可能是真實的前運動區或輔助運動區活動——但仍值得檢查。
#
# 一個快速的完整性檢查：比較**僅使用**感覺運動通道（C3、Cz、C4
# 及其緊鄰通道）與使用完整 22 通道組的交叉驗證準確率。
# 若完整組的表現顯著優於感覺運動組，請自問：額外的資訊從哪裡來？

# %%
SM_EXTENDED = [6, 7, 8, 9, 10, 11, 12]   # C5, C3, C1, Cz, C2, C4, C6
X_sm = X[:, SM_EXTENDED, :]

print("評估僅限感覺運動通道的子集（C5/C3/C1/Cz/C2/C4/C6）…")
acc_sm = cross_val_score(make_pipe(), X_sm, y, cv=cv, scoring="accuracy")
print(f"僅感覺運動通道準確率：{acc_sm.mean():.3f} ± {acc_sm.std():.3f}")
print(f"完整電極蒙太奇準確率：{acc_base.mean():.3f} ± {acc_base.std():.3f}")
delta = acc_base.mean() - acc_sm.mean()
print(f"來自非感覺運動通道的提升：{delta:+.3f}")
print()
print("若提升幅度小（< 0.05），額外通道貢獻甚微。")
print("若幅度大，請調查是哪些通道推動了提升——以及這些通道是否")
print("在生理學上對該任務而言是合理的。")

# %% [markdown]
# ### 實用的偽影排除建議
#
# 排除偽影驅動解碼的黃金標準為：
#
# 1. **ICA + EOG 回歸** — 在擬合解碼器*之前*，先移除被識別為眼球或
#    肌肉偽影的獨立成分（independent components）。比較前後準確率；
#    若準確率崩潰，代表您讀取的是偽影。
#
# 2. **通道消融測試** — 迭代去除額葉/外圍通道，檢查準確率是否仍能維持。
#    運動想像解碼器應對去除 Fz 和 FC 通道具有穩健性。
#
# 3. **空間合理性檢查** — 將 CSP 激活*模式*（非濾波器！）繪製為頭皮地形圖。
#    對於左右手想像，預期應見到以 C3/C4 為中心的雙側對側分布。
#
# 4. **眼球控制條件** — 收集一個受試者固定注視十字的獨立段落，
#    使注視方向與類別標籤之間的相關性被打破。
#    若準確率無法維持，則原始結果屬於偽影驅動。

# %%
# --- 摘要輸出 ----------------------------------------------------------------
print("=" * 58)
print("摘要 — 偽影混淆因素示範")
print("=" * 58)
print(f"基線準確率（全 22 通道，無偽影）：  {acc_base.mean():.3f}")
print(f"注入額葉偽影後：                    {acc_art.mean():.3f}")
print(f"去除額葉通道後（Fz/FC）：           {acc_abl.mean():.3f}")
print(f"僅感覺運動通道子集：                {acc_sm.mean():.3f}")
print()
print("模式診斷：")
base_f   = base_patterns_norm[FRONTAL_IDX].mean()
base_sm  = base_patterns_norm[SENSORIMOTOR_IDX].mean()
art_f    = art_patterns_norm[FRONTAL_IDX].mean()
art_sm   = art_patterns_norm[SENSORIMOTOR_IDX].mean()
print(f"  基線 — 平均額葉模式: {base_f:.2f}  |  SM: {base_sm:.2f}"
      f"  |  比值: {base_f/base_sm:.2f}")
print(f"  偽影 — 平均額葉模式: {art_f:.2f}  |  SM: {art_sm:.2f}"
      f"  |  比值: {art_f/art_sm:.2f}")
print()
print("紅旗門檻：額葉/SM 比值 > 3 需要調查。")

# %% [markdown]
# ---
# ## ⚠️ A subtler trap
#
# 上述示範使用了戲劇性的 3 倍振幅注入——很容易被發現。真實世界的
# 混淆因素更為隱微。以下是一種能通過標準 ICA 清理、甚至欺騙資深研究者的陷阱：
#
# ### 體積導通殘留：ICA 移除的是*成分*，但訊號依然存在
#
# 標準 ICA 眼電移除的做法，是識別出時間序列與外部 EOG 通道相關的成分，
# 再將其從 EEG 中投影去除。這對**垂直眨眼**效果良好，因為眨眼具有
# 典型且強烈的唯一地形。
#
# **掃視（水平眼動）則較難處理。**
#
# * 掃視產生的緩慢角膜視網膜電位（corneoretinal potential）以不同程度
#   透過體積導通傳至頭皮——同側額葉最大、中央區很小、枕葉近乎為零。
# * ICA 可能將其分解為*兩個或更多*成分：一個主要的額葉成分和一個
#   較弱的殘留成分，後者投影至額葉中央通道（FCz、FC1、FC2）。
#   主要成分被標記並移除；殘留成分則否。
# * ICA 清理後，額葉通道在視覺檢查及 EOG 相關指標下看似乾淨，
#   但 FCz/FC1/FC2 中仍有微弱的、與任務相關的掃視訊號殘留。
#
# **若掃視與任務相關，任何前處理都無法修復：**
#
# 假設您的運動想像典範在每次試次前顯示*方向提示*（向左或向右的箭頭），
# 受試者會反射性地看向箭頭。即使是隱蔽的、被抑制的掃視也會留下角膜
# 視網膜痕跡，且此痕跡與類別標籤完全相關——不是因為運動想像，
# 而是因為*提示設計*。
#
# 任何偽影排除步驟都無法去相關一個與標籤完全對齊的訊號。與標籤相關
# 的 ICA 成分會被標準 ICA 清理管線*保留*（標準做法依 EOG 通道相關性
# 標記成分，而非依類別標籤相關性）。殘留的眼動訊號留存於資料中，
# 解碼器從中學習，結果看起來像是運動想像解碼。
#
# **唯一的修復方案是實驗設計：**
# * 全程使用中央固視十字；絕不顯示方向性視覺提示。
# * 使用聽覺或中性符號提示（文字「LEFT」/「RIGHT」），
#   不會引發反射性注視偏移。
# * 記錄並分析眼球位置；確認各類別之間的注視位置無差異。
# * 測試移除 Cz 以前的所有通道是否不會造成準確率崩潰。
#
# **後設教訓：** 偽影混淆因素有兩種類型。*可透過前處理修復*的混淆
# （眨眼偽影、電纜雜訊）可在資料收集後移除。*設計層面*的混淆
# （與任務結構相關的系統性眼動）則無法——它們已烙印在資料中。
# 唯一的防護是前瞻性的實驗控制，而非事後的訊號處理。
