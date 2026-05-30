# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/08_classical_ml.ipynb)
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
# # 第 08 章 — 傳統機器學習（Classical Machine Learning，正確的做法）
#
# 現在我們用傳統模型對運動想像進行分類——而且要**誠實地**做。
# 本教程中最重要的核心概念就出現在這裡：
# **時間序列／分組資料不能以隨機打亂（random shuffle）的方式切分。**
#
# ## 學習目標
# 1. 建構 sklearn **流水線（Pipelines）**，使前處理步驟僅在**訓練集上擬合**（無資料洩漏）。
# 2. 比較 **LDA**（線性判別分析）、**SVM**（支持向量機）、**隨機森林（random forest）** 與 **黎曼（Riemannian）** 基線。
# 3. 使用**正確的交叉驗證（cross-validation）**：區塊感知（block-aware，受試者內）與
#    留一受試者法（Leave-One-Subject-Out，跨受試者）。
# 4. 解讀評估**指標（metrics）**：準確率（accuracy）、平衡準確率（balanced accuracy）、F1 分數。
#
# > **前置條件：** 第 02 章與第 07 章。
# > **難度：** ★★★☆☆
#
# **執行時間：** CPU 上約 2–3 分鐘。

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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neuro101 import io, datasets as ds
from neuro101 import features as ft
from neuro101.eval import (
    evaluate_with_variance, leakage_safe_pipeline,
    make_block_split, make_subject_split,
)

SMOKE = ds.is_smoke()
n_subj = 2 if SMOKE else 4
X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X.shape} | 類別分布={np.bincount(y)} | 受試者={np.unique(subj)}")

# %% [markdown]
# ## 黃金法則：僅在訓練集上擬合轉換器
#
# 任何需要*從資料中學習*的步驟——縮放器（scaler）的均值／標準差、CSP 的空間濾波器、
# 共變異數白化——都必須只看到**訓練折的資料**。sklearn `Pipeline`
# 會自動完成這件事：當你呼叫 `cross_val_score` 時，每個步驟都會在
# 每個折的訓練資料上重新擬合。我們的 `leakage_safe_pipeline` 只是一個有說明文件的
# `Pipeline`，讓這個意圖更加明確。

# %%
# 四個模型流水線。CSP 和共變異數在流水線內擬合 => 無洩漏。
models = {
    "CSP + LDA": leakage_safe_pipeline(
        [("csp", ft.make_csp(4)), ("lda", LinearDiscriminantAnalysis())]),
    "CSP + SVM": leakage_safe_pipeline(
        [("csp", ft.make_csp(4)), ("scale", StandardScaler()), ("svm", SVC(C=1, kernel="rbf"))]),
    "Bandpower + RF": leakage_safe_pipeline(
        [("clf", RandomForestClassifier(n_estimators=200, random_state=0))]),
    "Riemann + LogReg": leakage_safe_pipeline(
        ft.make_riemann_pipeline_steps() + [("clf", LogisticRegression(max_iter=500))]),
}

# Bandpower+RF 需要二維特徵；其他模型直接接受原始 epoch。預先計算頻帶功率。
X_bp = ft.bandpower(X, sf)

# %% [markdown]
# ## 受試者內評估（區塊感知切分，block-aware split）
#
# 首先回答較容易的問題：我們能否對*單一*受試者解碼，並在其保留的
# *區塊（blocks）* 試驗上測試（絕不使用隨機打亂）？我們使用受試者 1 和
# 一種考量試驗順序的區塊切分方式。

# %%
s0 = subj == np.unique(subj)[0]
Xs, ys = X[s0], y[s0]
Xs_bp = X_bp[s0]
n_blocks = 5

print("受試者內評估（受試者 1），5 個區塊感知折：")
for name, model in models.items():
    data = Xs_bp if name == "Bandpower + RF" else Xs
    res = evaluate_with_variance(
        model, data, ys,
        cv=lambda: make_block_split(len(ys), n_splits=n_blocks),
        scoring=("accuracy", "balanced_accuracy", "f1_macro"),
        seeds=(0, 1),
    )
    a = res["accuracy"]; print(f"  {name:18s} 準確率 {a['mean']:.3f} ± {a['std']:.3f}")

# %% [markdown]
# ## 核心問題：受試者獨立（留一受試者法，Leave-One-Subject-Out）
#
# 誠實且與實際部署相關的問題是：它能否對**新受試者**有效？
# 我們依次保留每位受試者（LOSO）。預期**數值會較低**——跨大腦泛化是困難的。
# 這是我們作為主要結果報告的指標。
#
# > **執行前先預測：** 猜猜 LOSO 準確率會比你剛才看到的受試者內區塊切分準確率
# > 高還是低——你預期差距大約有多少個百分點？

# %%
results = {}
for name, model in models.items():
    data = X_bp if name == "Bandpower + RF" else X
    res = evaluate_with_variance(
        model, data, y,
        cv=lambda: make_subject_split(subj),
        scoring=("accuracy", "balanced_accuracy", "f1_macro"),
        seeds=(0, 1),
    )
    results[name] = res
    a, ba, f1 = res["accuracy"], res["balanced_accuracy"], res["f1_macro"]
    print(f"  {name:18s} 準確率 {a['mean']:.3f}±{a['std']:.3f}  "
          f"bal_acc {ba['mean']:.3f}  f1 {f1['mean']:.3f}")

# %% [markdown]
# ## 誠實地比較模型（受試者間平均值 ± 標準差）

# %%
names = list(results)
means = [results[n]["accuracy"]["mean"] for n in names]
stds = [results[n]["accuracy"]["std"] for n in names]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, means, yerr=stds, capsize=6, color="#4c72b0")
ax.axhline(0.5, ls="--", color="gray", label="機率水準（2 類別）")
ax.set(ylabel="LOSO 準確率", title="受試者獨立準確率（保留受試者的平均值 ± 標準差）")
ax.set_ylim(0, 1); plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 指標簡介（完整說明請見第 12 章）
#
# - **準確率（Accuracy）**：正確預測的比例。當類別不平衡時具有誤導性。
# - **平衡準確率（Balanced accuracy）**：各類別召回率（recall）的平均值——在不平衡情況下較公平。
# - **F1 分數**：精確率（precision）與召回率的調和平均數；適合評估「是否在不誤報的情況下捕捉到正類」。
# - 永遠要報告**變異數**（± 折／受試者的標準差），不要只報告單一數字。

# %% [markdown]
# ## ✅ 概念確認（Concept check）
#
# 1. CSP（共同空間模式，Common Spatial Patterns）找到空間濾波器，使兩個類別之間的
#    變異數比最大化。為什麼 CSP 必須在流水線的訓練折*內部*擬合，而不能事先在整個資料集上擬合？
# 2. 你的模型受試者內準確率達到 95%，但 LOSO 只有 55%。這個差距最可能說明了
#    模型所學到的內容是什麼？
# 3. 當類別不平衡時，平衡準確率優於準確率。請用敏感度（sensitivity）和特異度（specificity）
#    寫出平衡準確率的公式。
#
# **解答：**
# 1. CSP 使用類別標籤計算共變異數矩陣及其廣義特徵分解。在完整資料集上擬合，
#    會讓測試折的標籤資訊影響空間濾波器，導致折外分數虛高（前處理洩漏）。
# 2. 受試者內與 LOSO 之間的大差距通常意味著模型學到了
#    受試者特有的特性（電極位置、頭骨幾何結構、個體腦節律），而非可泛化的運動想像特徵。
# 3. 平衡準確率 = （敏感度 + 特異度）/ 2
#    = (TP/(TP+FN) + TN/(TN+FP)) / 2，即兩個類別召回率的平均值。

# %% [markdown]
# ## ⚠️ 常見錯誤 / 為什麼這樣做是錯的
#
# - **對 epoch 或時間序列使用 `train_test_split(shuffle=True)`。** 相鄰／同受試者的樣本會洩漏。
#   請使用 `make_block_split`（受試者內）或 `make_subject_split`（跨受試者）。
#   *本專案甚至不提供隨機打亂切分器。*
# - **在交叉驗證之前使用 `scaler.fit_transform(X)`。** 測試統計量會洩漏到訓練中。
#   請將縮放器**放在流水線內**。
# - **將受試者內的數字作為主要結果報告。** 這是過於樂觀的。
#   受試者獨立（LOSO）的數字才是誠實的。
# - **只做一次，只報告一個數字。** 隨機種子和折的選擇會造成差異；報告平均值 ± 標準差。
# - **在測試折上調整超參數（hyperparameters）。** 請使用嵌套交叉驗證（nested CV）或單獨的
#   驗證集切分——絕不偷看測試集。
#
# **下一步：** 第 09 章 — 使用 braindecode 進行深度學習（EEGNet 等），並繼續遵守以上所有規則。
