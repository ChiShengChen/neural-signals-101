# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/12_evaluation_and_pitfalls.ipynb)
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
# # 第 12 章 — 評估與常見陷阱（Evaluation & Pitfalls）⭐（最重要的章節）
#
# 幾乎每一個「驚人」的神經解碼成果，只要無法重現，都是因為犯了以下某個錯誤。
# 我們以 **WRONG → RIGHT（錯誤 → 正確）** 對比的方式逐一講解：
#
# 1. 用**錯誤**的方式執行 → 得到**虛高**的分數。
# 2. 用白話說明**為何分數被誇大**。
# 3. 用**正確**的方式執行 → 分數**回落至合理數值**。
# 4. 一句話的**重點結論**。
#
# **落差就是教訓。** 我們同時列印並繪圖，讓對比一目了然。每個 ⚠️ 儲存格都是故意寫錯的——永遠不要複製到真實專案中。
#
# ## 七大陷阱
# 1. 對連續／分段（epoch）資料進行隨機打亂切分（自相關洩漏）。
# 2. 受試者依賴（subject-dependent）vs 受試者獨立（subject-independent）——以及讓評估方式與你的宣稱相符。
# 3. 前處理（preprocessing）／特徵洩漏（feature leakage，切分前就在全部資料上擬合）。
# 4. 類別不平衡（class imbalance）與錯誤指標。
# 5. 跨場次（cross-session）／領域偏移（domain shift）。
# 6. 幸運隨機種子（lucky seed）／未報告變異數（no variance reporting）。
# 7. 在測試集上調整超參數／選模型（最常見的隱性殺手）→ 巢狀交叉驗證（nested CV）。
#
# > **前置條件：** 第 02、08 與 11 章。
# > **難度：** ★★★★☆
#
# **執行時間：** CPU 上約 2–4 分鐘（煙霧測試（smoke）模式會縮小資料量）。

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score)

from neuro101 import io, datasets as ds, features as ft, viz
from neuro101.eval import (
    make_subject_split, make_block_split, leakage_safe_pipeline, evaluate_with_variance,
)

SMOKE = ds.is_smoke()
sf = ds.DATASETS["bnci_2a"].sfreq_hz
scoreboard = {}  # 收集每個陷阱的（錯誤, 正確）分數，用於最終彙總圖

# %% [markdown]
# ---
# # 陷阱 1 — 對連續／分段資料進行隨機打亂切分
#
# **情境設定。** 我們將每個運動想像（motor-imagery）試驗切分成多個*重疊*的時間窗。
# 相鄰時間窗幾乎相同（它們共享大部分樣本）。如果我們接著**隨機**切分時間窗，
# 測試集中的某個時間窗在訓練集中幾乎有一個「雙胞胎」——模型可以「辨認」它，而非真正泛化。

# %%
X1, y1, s1 = io.load_bnci_2a_epochs(subjects=[1])
win, step = 200, 25
W, L, G = [], [], []
for ti in range(X1.shape[0]):
    for start in range(0, X1.shape[-1] - win + 1, step):
        W.append(X1[ti, :, start:start + win]); L.append(y1[ti]); G.append(ti)
W, L, G = np.array(W), np.array(L), np.array(G)
F1 = ft.bandpower(W, sf)
print(f"來自 {X1.shape[0]} 個試驗的 {len(L)} 個重疊時間窗")

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：隨機打亂切分（同一試驗的時間窗分散到訓練集與測試集）

# %%
wrong1 = np.mean([
    cross_val_score(LDA(), F1, L, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"⚠️ WRONG（隨機打亂）準確率 = {wrong1:.3f}")

# %% [markdown]
# **為何分數被誇大：** 相鄰時間窗高度相關，因此隨機切分會讓近乎重複的樣本出現在訓練／測試邊界兩側。
# 模型部分在「記憶」，而非真正泛化。

# %% [markdown]
# ### ✅ RIGHT（正確）：試驗感知（trial-aware）的區塊切分（整個試驗保持在同一側）

# %%
right1 = np.mean([
    LDA().fit(F1[tr], L[tr]).score(F1[te], L[te])
    for tr, te in make_block_split(len(L), groups=G, n_splits=5)
])
print(f"✅ RIGHT（試驗感知切分）準確率 = {right1:.3f}")
scoreboard["1. Autocorrelation\nleakage"] = (wrong1, right1)

# %% [markdown]
# > **重點結論 1：** 永遠不要對相關時間序列進行隨機打亂——讓整個試驗／區塊保持在切分的同一側（`make_block_split`）。

# %% [markdown]
# ---
# # 陷阱 2 — 受試者依賴 vs 受試者獨立
#
# **情境設定。** 我們將多個受試者的試驗合併。誠實的問題是「對*新的人*是否有效？」——
# 但很容易把合併後的試驗隨機切分，讓同一個受試者同時出現在訓練集和測試集中。

# %%
nsub = 2 if SMOKE else 4
X, y, subj = io.load_bnci_2a_epochs(n_subjects=nsub)
pipe2 = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：合併受試者後隨機切分（受試者身分資訊洩漏）

# %%
wrong2 = evaluate_with_variance(
    pipe2, X, y,
    cv=lambda: StratifiedKFold(5, shuffle=True, random_state=0).split(X, y),
    scoring="accuracy", seeds=(0, 1, 2),
)["accuracy"]["mean"]
print(f"⚠️ WRONG（合併 + 隨機）準確率 = {wrong2:.3f}")

# %% [markdown]
# **為何分數被誇大：** 模型可以學習每個*受試者*的特徵（頭骨、電極位置、個人節律），
# 並在測試集中重用這些特徵——而真正的新使用者不會有這種情況。

# %% [markdown]
# ### ✅ RIGHT（正確）：留一受試者法（Leave-One-Subject-Out，LOSO）（僅在未見過的人身上測試）

# %%
right2 = evaluate_with_variance(
    pipe2, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0, 1, 2),
)["accuracy"]["mean"]
print(f"✅ RIGHT（LOSO）準確率 = {right2:.3f}")
scoreboard["2. Subject\nleakage"] = (wrong2, right2)

# %% [markdown]
# ### 真正的原則不是「永遠用 LOSO」——而是*讓評估方式與你的宣稱相符*
#
# 很容易得出「LOSO 才是真相，受試者內評估是謊言」的結論，但這樣矯枉過正了。
# 誠實的原則是：**讓你的評估方式反映你實際的部署方式**：
#
# | 如果你宣稱… | …那麼誠實的評估方式是 | 範例 |
# |---|---|---|
# | 「對**新使用者**開箱即用」 | **受試者獨立**（LOSO） | 通用消費型腦機介面（BCI） |
# | 「在**個人校準後**有效」 | **受試者內**、無洩漏的切分 | 按患者校準的臨床裝置；SSVEP 拼寫器 |
# | 「在**單一場次內**有效」 | 場次內（within-session）區塊切分 | 一次性實驗室實驗 |
#
# 受試者內評估**並非**不誠實——*只要切分本身不洩漏*
# （陷阱 #1 和 #3 仍然適用），且你**明確說明你回答的是哪個問題**。
# 問題在於：報告了受試者內的數字，卻*暗示*它可以推廣到新受試者。
# 不要用一個神話（「隨機切分沒問題」）換來另一個教條（「只有 LOSO 算數」）。

# %% [markdown]
# > **重點結論 2：** 報告與你的部署宣稱相符的指標，並**誠實地標示它**。
# > 如果你宣稱對新受試者有效，頭條數字必須是受試者獨立（LOSO）的。
# > 受試者內的數字對已校準的個人模型來說沒問題——但永遠不要讓它偽裝成跨受試者的結果。

# %% [markdown]
# ---
# # 陷阱 3 — 前處理（Preprocessing）／特徵洩漏（Feature Leakage）
#
# **情境設定（故意設計得很極端）。** 我們用**純隨機噪音**作為特徵，
# 加上**隨機標籤**——不存在任何真實關聯，因此誠實的準確率必須是隨機水準（0.5）。
# 觀察有洩漏的特徵選取步驟如何從虛無中製造出「信號」。
# （當你在切分前就對整個資料集擬合縮放器（scaler）／CSP／ICA 時，同樣的事會發生，只是沒那麼明顯。）

# %%
rng = np.random.default_rng(0)
Xnoise = rng.standard_normal((200, 2000))   # 2000 個無意義的特徵
ynoise = rng.integers(0, 2, 200)            # 與特徵無關的標籤

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：用**全部**資料選出「最佳」特徵，然後進行交叉驗證

# %%
selected = SelectKBest(f_classif, k=30).fit(Xnoise, ynoise).transform(Xnoise)  # 偷看了所有標籤
wrong3 = np.mean([
    cross_val_score(LDA(), selected, ynoise, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"⚠️ WRONG（在全部資料上選特徵）準確率 = {wrong3:.3f}  <-- 純噪音！")

# %% [markdown]
# **為何分數被誇大：** 使用*整個資料集*選出與標籤最相關的 30 個特徵，讓測試標籤的資訊洩漏進特徵集。
# 在 2000 個隨機特徵中，有些會**偶然**與標籤相關——而我們正好挑選了這些。分數是 100% 的海市蜃樓。

# %% [markdown]
# ### ✅ RIGHT（正確）：將選取步驟放入流水線（Pipeline）中（僅在每個訓練折疊上擬合）

# %%
pipe3 = Pipeline([("sel", SelectKBest(f_classif, k=30)), ("lda", LDA())])
right3 = np.mean([
    cross_val_score(pipe3, Xnoise, ynoise, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
    for s in (0, 1, 2)
])
print(f"✅ RIGHT（流水線內選取）準確率 = {right3:.3f}  <-- 正確地回到隨機水準")
scoreboard["3. Feature\nleakage"] = (wrong3, right3)

# %% [markdown]
# > **重點結論 3：** 每個*從資料中學習*的步驟（縮放器、特徵選取、CSP、ICA、PCA）
# > 都必須只在**訓練折疊上**擬合——將它們封裝進 Pipeline 中。

# %% [markdown]
# ---
# # 陷阱 4 — 類別不平衡（Class Imbalance）與錯誤指標
#
# **情境設定。** 真實的 Sleep-EDF 資料。我們嘗試偵測一個**稀有**睡眠階段（最少見的那個）。
# 大多數時期都不是該階段，因此模型可以靠著永遠預測「不是」來獲得高準確率。

# %%
Xsl, ysl, ssl = io.load_sleep_edf_epochs(n_subjects=1)
counts = np.bincount(ysl)
minority = int(np.argmin([c if c > 0 else 10**9 for c in counts]))
y_bin = (ysl == minority).astype(int)
print(f"偵測稀有階段（id {minority}）；它只佔所有時期的 {100*y_bin.mean():.1f}%")

# 一個永遠預測多數類別的「模型」（什麼都偵測不到）。
lazy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_bin), 1)), y_bin)
pred = lazy.predict(np.zeros((len(y_bin), 1)))

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：報告準確率（accuracy）

# %%
acc4 = accuracy_score(y_bin, pred)
print(f"⚠️ WRONG 指標 — 準確率 = {acc4:.3f}   （看起來很棒！）")

# %% [markdown]
# **為何會造成誤導：** 準確率獎勵預測多數類別的行為。一個*從不*偵測稀有階段的模型
# 仍然能得到很高的分數（≈ 多數類別的比例），因為事件很稀有。
# 準確率隱藏了對你真正關心的類別的完全失敗——平衡準確率（balanced accuracy）會崩潰到 0.5（隨機水準）。

# %% [markdown]
# ### ✅ RIGHT（正確）：平衡準確率（balanced accuracy）、F1、混淆矩陣（confusion matrix）（以及 ROC-AUC）

# %%
bal4 = balanced_accuracy_score(y_bin, pred)
f1_4 = f1_score(y_bin, pred, zero_division=0)
cm = confusion_matrix(y_bin, pred)
print(f"✅ RIGHT 指標 — 平衡準確率 = {bal4:.3f}, F1 = {f1_4:.3f}")
print("   混淆矩陣（列=真實，欄=預測）："); print(cm)

fig, ax = plt.subplots(figsize=(4.2, 3.8))
viz.plot_confusion(cm, ["not-rare", "rare"], ax=ax,
                   title="它完全偵測不到稀有類別")
plt.show()
scoreboard["4. Wrong metric\n(acc vs bal-acc)"] = (acc4, bal4)

# %% [markdown]
# > **重點結論 4：** 在不平衡問題上，準確率會說謊。
# > 報告平衡準確率／F1／ROC-AUC，並查看混淆矩陣。

# %% [markdown]
# ---
# # 陷阱 5 — 跨場次（Cross-Session）／領域偏移（Domain Shift）
#
# **情境設定。** 大腦（和電極）會在不同的錄製天之間發生變化：不同的阻抗、
# 稍微不同的帽子位置、不同的情緒狀態。在**場次 1** 訓練的模型，部署到**場次 2** 時面對的是*分布偏移*。
#
# 在 BCI IV 2a 資料集上，真實的場次間下降幅度適中，所以為了讓機制更清晰，
# 我們對真實資料應用一個**明顯模擬的「新錄製日」**轉換：
# 每通道的增益變化（阻抗）加上直流偏移（漂移）。這正是不同天／裝置對信號的真實影響。

# %%
nsub5 = 2 if SMOKE else 3
Xs, ys, subs, sess = io.load_bnci_2a_epochs(n_subjects=nsub5, return_session=True)

def simulate_new_day(X_in, seed):
    """真實的場次偏移：隨機的每通道增益 + 直流偏移。"""
    r = np.random.default_rng(seed)
    gain = r.uniform(0.5, 2.0, size=(1, X_in.shape[1], 1))   # 阻抗差異
    offset = r.uniform(-1, 1, size=(1, X_in.shape[1], 1)) * X_in.std()
    return X_in * gain + offset

pipe5 = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：在相同分布（場次 1 風格）的資料上評估

# %%
within = []
for su in np.unique(subs):
    m = (subs == su) & (sess == 0)
    Xa, ya = Xs[m], ys[m]
    within.append(np.mean([pipe5.fit(Xa[tr], ya[tr]).score(Xa[te], ya[te])
                           for tr, te in make_block_split(len(ya), n_splits=4)]))
wrong5 = float(np.mean(within))
print(f"⚠️ WRONG（在相同錄製日測試）準確率 = {wrong5:.3f}")

# %% [markdown]
# **為何過於樂觀：** 你的驗證資料來自與訓練相同的場次，因此共享所有當天特有的特性。部署時從不會如此。

# %% [markdown]
# ### ✅ RIGHT（正確）：在第 1 天訓練，在（模擬的）不同日期測試

# %%
cross = []
for su in np.unique(subs):
    m = (subs == su) & (sess == 0)
    Xa, ya = Xs[m], ys[m]
    Xnew = simulate_new_day(Xa, seed=int(su))   # 「隔天」的錄製
    cross.append(pipe5.fit(Xa, ya).score(Xnew, ya))
right5 = float(np.mean(cross))
print(f"✅ RIGHT（在不同日期測試）準確率 = {right5:.3f}")
scoreboard["5. Domain\nshift"] = (wrong5, right5)

# %% [markdown]
# > **重點結論 5：** 在你部署時會面對的偏移下進行測試（場次、日期、裝置、受試者）。
# > 同一場次的分數高估了真實世界的效能。
# > 修正方法：場次內標準化（per-session normalisation）、領域適應（domain adaptation），或短暫校準。

# %% [markdown]
# ---
# # 陷阱 6 — 幸運種子（Lucky Seed）／無變異數報告
#
# **情境設定。** 受試者獨立的解碼在不同的留一受試者（held-out subject）之間差異很大。
# 如果你只跑一次就引用*最佳*折疊的結果，你就是在挑選。

# %%
per_fold = evaluate_with_variance(
    pipe2, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0,),
)["accuracy"]["per_fold"].ravel()
print("每位受試者的準確率：", np.round(per_fold, 3))

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：報告單一最佳折疊

# %%
wrong6 = float(per_fold.max())
print(f"⚠️ WRONG（單一最佳折疊）準確率 = {wrong6:.3f}")

# %% [markdown]
# **為何會造成誤導：** 在折疊數少且變異高的情況下，最大值是幸運的抽選，而非預期效能。
# 不同的隨機種子或測試受試者會給出不同的「最佳值」。

# %% [markdown]
# ### ✅ RIGHT（正確）：報告折疊間的平均值 ± 標準差（並適當地進行顯著性檢定）

# %%
right6 = float(per_fold.mean())
print(f"✅ RIGHT（平均值 ± 標準差）準確率 = {right6:.3f} ± {per_fold.std():.3f}")
scoreboard["6. Lucky fold\n(best vs mean)"] = (wrong6, right6)

# 跨折疊的配對檢定，在比較兩個模型時，尊重其配對性：
from scipy.stats import wilcoxon, ttest_rel
pipe_riem = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() +
                                  [("lda", LDA())])
fold_csp = per_fold
fold_riem = evaluate_with_variance(
    pipe_riem, X, y, cv=lambda: make_subject_split(subj),
    scoring="accuracy", seeds=(0,),
)["accuracy"]["per_fold"].ravel()
if len(fold_csp) == len(fold_riem) and len(fold_csp) >= 3:
    t, p = ttest_rel(fold_riem, fold_csp)
    print(f"跨 {len(fold_csp)} 位受試者的 CSP vs 黎曼（Riemann）配對 t 檢定："
          f"平均差異 {np.mean(fold_riem - fold_csp):+.3f}, p = {p:.3f}")
    print("（差異小於標準差且 p>0.05，並非真正的改進。）")

# %% [markdown]
# > **重點結論 6：** 一個數字不是結果。報告跨折疊／種子的平均值 ± 標準差，
# > 並在宣稱一個模型優於另一個之前，使用**配對**檢定。

# %% [markdown]
# ---
# # 陷阱 7 — 在測試集上調整超參數（最常見的隱性殺手）
#
# 這個陷阱藏在明處。你嘗試了幾種前處理選擇、模型或超參數，
# **挑選在評估資料上得分最高的那個，然後報告那個最高分。**
# 但「多次嘗試中的最佳值」是樂觀偏誤的——是一種*贏家詛咒（winner's curse）*。
# 你嘗試的配置越多，最佳值就越被誇大。
#
# 修正方法是**巢狀交叉驗證（nested cross-validation）**：*內層*迴圈選擇超參數，
# *外層*迴圈在內層從未見過的資料上對所選模型進行評分。

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

Xncv, yncv, _ = io.load_bnci_2a_epochs(subjects=[1])
Fncv = ft.bandpower(Xncv, sf)
C_grid = np.logspace(-3, 3, 13)   # 13 個候選超參數可供選擇
inner = StratifiedKFold(5, shuffle=False)

# %% [markdown]
# ### ⚠️ WRONG（錯誤）：網格搜索後，報告*最佳*配置自身的交叉驗證分數

# %%
cv_scores = [cross_val_score(SVC(C=c), Fncv, yncv, cv=inner).mean() for c in C_grid]
wrong7 = float(np.max(cv_scores))   # 我們選擇 C 就是因為它最大化了這個分數
print(f"⚠️ WRONG（{len(C_grid)} 個配置中的最佳，使用相同 CV）準確率 = {wrong7:.3f}")

# %% [markdown]
# **為何分數被誇大：** 你使用評估分數來*選擇*模型，然後引用那個相同的分數作為結果。
# 有 13 次嘗試，最大值捕捉到的是幸運的折疊切分——而真實專案探索的幾十種前處理／模型選擇，
# 通膨效果會累積疊加。（試著擴展 `C_grid`：差距會增大。）

# %% [markdown]
# ### ✅ RIGHT（正確）：巢狀交叉驗證（nested CV）——在內部選擇，在外部評分

# %%
outer = StratifiedKFold(5, shuffle=False)
nested = []
for tr, te in outer.split(Fncv, yncv):
    gs = GridSearchCV(SVC(), {"C": C_grid}, cv=StratifiedKFold(4, shuffle=False))
    gs.fit(Fncv[tr], yncv[tr])                 # 僅在訓練集上選擇 C
    nested.append(gs.score(Fncv[te], yncv[te]))  # 在未碰過的測試折疊上評分
right7 = float(np.mean(nested))
print(f"✅ RIGHT（巢狀 CV）準確率 = {right7:.3f}   誠實的下降幅度 = {wrong7 - right7:+.3f}")
scoreboard["7. Tuning on\ntest (nested CV)"] = (wrong7, right7)

# %% [markdown]
# > **重點結論 7：** 永遠不要報告你也用來*選擇*模型的分數。
# > 選擇超參數／前處理／模型本身是訓練的一部分——將它包在**巢狀 CV**（或獨立的驗證集）中。
# > 這裡在單一受試者上的下降看起來很小，但選擇偏誤是誠實論文誇大結果的最常見方式。

# %% [markdown]
# ---
# # 完整故事，盡在一張圖
#
# 每個陷阱，WRONG（紅色）vs RIGHT（綠色）。每一對中，紅色長條是你可能想發表的數字；綠色長條是真相。
#
# > **執行前：** 猜猜陷阱 3（純噪音上的特徵洩漏）的 WRONG vs RIGHT 準確率差距會有多大——
# > 考慮到有 2000 個隨機特徵可用，你預期被誇大的分數接近 0.6、0.7、0.8 還是更高？

# %%
labels = list(scoreboard)
wrongs = [scoreboard[k][0] for k in labels]
rights = [scoreboard[k][1] for k in labels]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - 0.2, wrongs, 0.4, label="WRONG（被誇大）", color=viz.WRONG_COLOR)
ax.bar(x + 0.2, rights, 0.4, label="RIGHT（誠實）", color=viz.RIGHT_COLOR)
for xi, (w, r) in enumerate(zip(wrongs, rights)):
    ax.text(xi - 0.2, w + 0.01, f"{w:.2f}", ha="center", fontsize=8)
    ax.text(xi + 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("分數"); ax.set_ylim(0, 1.05)
ax.set_title("評估說謊的七種方式——以及每種情況下的誠實數字")
ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# # ✅ 評估核查清單（複製到你的下一個專案）
#
# ```text
# 資料切分（DATA SPLITTING）
# [ ] 不對時間序列／分段資料進行隨機打亂切分（使用區塊或受試者感知的方式）。
# [ ] 整個試驗／區塊保持在切分的同一側。
# [ ] 評估與部署宣稱相符（「新使用者」用 LOSO；「校準後」用無洩漏的受試者內切分）
#     ——並標示清楚是哪種評估方式。
# [ ] （如果是連續資料）訓練集與測試集之間用時間間隔／清除區（gap/purge）分隔。
#
# 洩漏（LEAKAGE）
# [ ] 每個學習型轉換（縮放器、CSP、ICA、PCA、特徵選取）都在
#     Pipeline 中，並僅在訓練折疊上擬合。
# [ ] 切分前不在整個資料集上計算標準化／統計量。
# [ ] 超參數用巢狀 CV／獨立驗證集調整（不用測試集）。
#
# 指標（METRICS）
# [ ] 對不平衡問題報告平衡準確率／F1／ROC-AUC（不只是準確率）。
# [ ] 查看混淆矩陣。
# [ ] 與隨機水準及簡單基線（如多數類別、CSP+LDA）進行比較。
#
# 穩健性（ROBUSTNESS）
# [ ] 在你部署時會面對的偏移下進行評估（場次／日期／裝置／受試者）。
# [ ] 為所有隨機數生成器（RNG）設定種子。
# [ ] 報告跨折疊／種子的平均值 ± 標準差（永遠不要只報告單一數字）。
# [ ] 在宣稱模型 A 優於模型 B 之前，使用跨折疊的配對檢定。
# ```

# %% [markdown]
# # 彙總表——每個陷阱及其一行修正方法
#
# | # | 陷阱（陷阱所在） | 一行修正方法 |
# |---|---|---|
# | 1 | 對相關時間序列進行隨機打亂 | 按試驗／區塊切分（`make_block_split`） |
# | 2 | 合併受試者後隨機切分 | 留一受試者法（`make_subject_split`） |
# | 3 | 在全部資料上擬合縮放器／CSP／選取步驟 | 在 Pipeline 中擬合，僅用訓練折疊 |
# | 4 | 在不平衡類別上用準確率 | 平衡準確率／F1／ROC-AUC + 混淆矩陣 |
# | 5 | 在訓練的相同場次上測試 | 跨場次／日期／裝置／受試者測試 |
# | 6 | 單次執行，報告最佳數字 | 跨折疊／種子的平均值 ± 標準差 + 配對檢定 |
# | 7 | 使用測試分數調整／選模型 | 巢狀 CV（內部選擇，外部評分） |
#
# > **核心概念：** 在神經信號機器學習（ML）中，誠實的數字幾乎總是
# > *低於*你得到的第一個數字。追求下降——而非峰值——才是讓結果
# > 能夠被部署而非消失的關鍵。

# %% [markdown]
# ## ✅ 概念測驗
#
# 1. 在陷阱 1 中，來自同一試驗的重疊時間窗被隨機切分。
#    說出使這種切分策略對時間序列資料無效的統計性質，並解釋為何它會誇大分數。
# 2. 你有一個 4 類資料集，其中一個類別佔試驗的 70%。你的模型達到了 72% 的準確率。
#    不看混淆矩陣，為何你不能得出模型有用的結論？
# 3. 同事報告了 CSP+LDA 與深度網絡在 9 個留一受試者上的配對 t 檢定，
#    獲得 p = 0.04，平均差異為 +1.2%。你應該將此視為有意義的改進嗎？你還需要什麼額外資訊？
#
# **答案：**
# 1. 相鄰的重疊時間窗共享大部分樣本，使它們高度自相關（autocorrelated）。隨機切分將
#    近乎重複的時間窗放在訓練／測試邊界的兩側，因此模型可以通過訓練集中的「雙胞胎」
#    來「辨認」測試時間窗——通過記憶而非泛化來誇大準確率。
# 2. 永遠預測多數類別的模型在不學習任何東西的情況下也能達到約 70% 的準確率。
#    你需要平衡準確率、F1 以及混淆矩陣來確定模型是否真的能偵測到少數類別。
# 3. 統計顯著性（p = 0.04）並不意味著實際顯著性（practical significance）。
#    只有 9 個折疊的話，檢定的效力（power）很低，而 1.2% 的平均差異可能小於受試者間的標準差。
#    你需要效應量（effect size，即每位受試者差異的平均值 ± 標準差），
#    並應檢查結果是否在對所有測試過的模型進行多重比較校正後仍然成立。

# %% [markdown]
# ## ⚠️ 常見錯誤／為何這樣做是錯的（後設觀點）
#
# - **將本章視為可選章節。** 這正是真實結果與虛假結果之間的分水嶺。
# - **修正了一個洩漏就相信數字。** 這些陷阱會疊加——審查全部七個。
# - **將上述 WRONG 數字作為成就來報告。** 這裡的每個紅色長條都是一個警示故事，而非基準（benchmark）。
#
# **下一步：** 第 13 章——*神經倫理學（neuroethics）與反炒作（anti-hype）*：
# 向公眾過度宣稱是洩漏的倫理學孿生。然後是第 14 章——終章，
# 你將在一個隱藏的留一測試集上，自行建構一個無洩漏的報告。
