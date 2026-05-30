# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/11_statistics_intuition.ipynb)
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
# # 第 11 章 — 統計直覺（Statistics Intuition）
#
# 在第 12 章剖析陷阱之前，我們需要一套**統計反射（statistical reflexes）**——
# 快速的直覺檢查，告訴你正在閱讀（或回報）的數字是否真的代表它看起來的含義。
#
# 本章幾乎完全以**模擬為基礎**。我們創造我們知道真相的情境，
# 然後觀察我們的測量結果。這讓我們能看到「我們測量到的」與「實際上真實的」之間的差距——
# 在真實實驗中這個差距是不可見的，但卻是真實存在的。
#
# ## 學習目標
# 1. 感受單次準確率估計因**抽樣變異（sampling variation）**而大幅波動的程度。
# 2. 了解**均值 ± 標準差（mean ± std）**和 **95% 信賴區間（confidence interval，CI）**的真正含義。
# 3. 理解**機率水準（chance level）並非永遠是 1/n_classes**（類別不平衡！）。
# 4. 認識到兩個長條間的小差距很可能只是**雜訊（noise）**，除非配對檢定（paired test）另有說法。
#
# > **前置條件：** 第 02 章。
# > **難度：** ★★★☆☆
# > **執行時間：** 約 1–2 分鐘。

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
rng = np.random.default_rng(0)

# %% [markdown]
# ---
# ## 第 1 節 — 抽樣變異（Sampling Variation）：你的估計充滿雜訊
#
# 想像一個**真實準確率為 0.65**的分類器。這個數字刻在宇宙中——
# 但你永遠無法直接讀取它。你能做的只是在測試集上執行分類器，
# 計算它答對了幾次。
#
# 如果你的測試集只有 **20 個樣本**，你實際上是把一枚有偏差的硬幣投 20 次並計算正面次數。
# 20 次投擲並不多。
#
# > **執行前：** 只有 20 個測試試次時，單次準確率估計距離真實值 0.65 能偏差多遠？
# > 寫下你的猜測，然後執行儲存格。

# %%
# --- 模擬設定 ---
TRUE_ACC = 0.65        # 真實準確率，在實際情況中是未知的
N_REPEATS = 5_000     # 我們想像執行多少次「實驗」

def simulate_accuracy(n_test, n_repeats, true_acc, rng):
    """從 n_repeats 個獨立實驗中回傳測量準確率的陣列。"""
    # 每個實驗：抽取 n_test 個 Bernoulli(true_acc) 結果，計算均值
    outcomes = rng.random(size=(n_repeats, n_test)) < true_acc
    return outcomes.mean(axis=1)

accs_20 = simulate_accuracy(20, N_REPEATS, TRUE_ACC, rng)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(accs_20, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(TRUE_ACC, color="crimson", lw=2.5, label=f"真實準確率 = {TRUE_ACC}")
ax.axvline(accs_20.mean(), color="orange", lw=1.5, linestyle="--",
           label=f"估計值均值 = {accs_20.mean():.3f}")
ax.set_xlabel("測量準確率（n = 20 個測試試次）", fontsize=12)
ax.set_ylabel("模擬實驗次數", fontsize=12)
ax.set_title("只有 20 個測試試次時的準確率估計分佈\n"
             f"（真實準確率 = {TRUE_ACC}，{N_REPEATS:,} 個模擬實驗）")
ax.legend()
pct5, pct95 = np.percentile(accs_20, [5, 95])
ax.text(0.02, 0.95,
        f"中間 90% 範圍：\n[{pct5:.2f}, {pct95:.2f}]",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.show()

print(f"真實準確率：              {TRUE_ACC:.3f}")
print(f"測量準確率均值：          {accs_20.mean():.3f}")
print(f"測量準確率標準差：        {accs_20.std():.3f}")
print(f"90% 的實驗落在 [{pct5:.2f}, {pct95:.2f}]")
print(f"=> 單次 n=20 估計可能偏差 ±{(pct95-pct5)/2:.2f}（90% 範圍）")

# %% [markdown]
# **直方圖告訴你什麼：**
# 每個長條是「在這麼多次模擬實驗中，測量準確率落在這裡」。
# 真實值是紅線——但單次實驗只給你從這整個分佈中抽取一個樣本。
#
# 對 n=20 而言，分佈*非常寬*。有些實驗甚至回報準確率低於 0.50（機率水準！），
# 即使真實準確率是 0.65。
#
# ### 樣本大小如何有幫助？
#
# > **執行前：** 預測哪個樣本大小會讓直方圖縮窄最多——
# > 從 n=20 增加到 n=100，還是從 n=100 增加到 n=500？

# %%
ns = [20, 100, 500]
colors = ["steelblue", "darkorange", "seagreen"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for ax, n, c in zip(axes, ns, colors):
    accs = simulate_accuracy(n, N_REPEATS, TRUE_ACC, rng)
    p5, p95 = np.percentile(accs, [5, 95])
    ax.hist(accs, bins=35, color=c, edgecolor="white", alpha=0.85)
    ax.axvline(TRUE_ACC, color="crimson", lw=2)
    ax.set_title(f"n = {n} 個測試試次\nstd = {accs.std():.3f}   90% 在 [{p5:.2f}, {p95:.2f}]",
                 fontsize=10)
    ax.set_xlabel("測量準確率")
    if ax is axes[0]:
        ax.set_ylabel("次數")
    ax.set_xlim(0.25, 1.0)

fig.suptitle(f"抽樣變異隨測試集大小增加而縮小（真實準確率 = {TRUE_ACC}）",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("n_test | 估計標準差 | 90% 範圍寬度")
for n in ns:
    accs = simulate_accuracy(n, N_REPEATS, TRUE_ACC, rng)
    p5, p95 = np.percentile(accs, [5, 95])
    print(f"  {n:>4}  |     {accs.std():.4f}      |    {p95-p5:.3f}")

# %% [markdown]
# **關鍵洞察：** 估計的標準差大致以 1/√n 縮小。測試集加倍，標準差減半。
#
# * n=20 時，你幾乎無法判斷你的模型是否超越機率水準。
# * n=500 時，你的估計已足夠緊密而有意義。
#
# **實用規則：** 回報你使用了多少個測試樣本（或受試者、或試次）。
# 這個數字告訴讀者應該多信任你的準確率。

# %% [markdown]
# ---
# ## 第 2 節 — 均值 ± 標準差與信賴區間的真正含義
#
# 在 EEG 論文中，你經常看到這樣的結果：
# *「LOSO 準確率：0.68 ± 0.12（N = 9 位受試者）」*
#
# 這實際上意味著什麼？我們應該有多確信？
#
# ### 具體範例：9 位受試者的 LOSO 準確率

# %%
# 模擬 9 位受試者的逐受試者 LOSO 準確率
# （想像一個有 N=9 名參與者的真實實驗）
n_subjects = 9
true_subject_mean = 0.68

# 每位受試者的準確率具有真實變異性（受試者間差異）
subject_accs = np.array([0.61, 0.74, 0.55, 0.79, 0.63, 0.72, 0.58, 0.81, 0.68])
# （手工設定以符合真實情境；固定種子確保結果可重現）

mean_acc = subject_accs.mean()
std_acc = subject_accs.std(ddof=1)   # 樣本標準差（除以 N-1）
se_acc = std_acc / np.sqrt(n_subjects)  # 均值的標準誤（standard error of the mean）

print(f"受試者準確率: {subject_accs}")
print(f"均值  = {mean_acc:.3f}")
print(f"標準差 = {std_acc:.3f}  （受試者間的分散程度）")
print(f"標準誤 = {se_acc:.3f}  （關於均值的不確定性）")

# %% [markdown]
# ### Bootstrap 95% 信賴區間（Confidence Interval，CI）
#
# **信賴區間（CI）**回答的問題是：「給定這 9 個數字，如果我們重複這項研究，
# 哪個範圍合理地包含了真實均值？」
#
# 我們將使用**自助法（bootstrapping）**——一種不需要數學假設的重抽樣技巧：
# 1. 從我們的 9 位受試者中**有放回地**抽取 9 位（有些重複，有些被跳過）。
# 2. 計算該重抽樣的均值。
# 3. 重複 10,000 次。
# 4. 這些 bootstrap 均值的中間 95% 就是我們的信賴區間。

# %%
N_BOOT = 10_000
boot_means = np.array([
    rng.choice(subject_accs, size=n_subjects, replace=True).mean()
    for _ in range(N_BOOT)
])

ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot_means, bins=60, color="mediumslateblue", edgecolor="white", alpha=0.85)
ax.axvline(mean_acc, color="crimson", lw=2.5, label=f"觀測均值 = {mean_acc:.3f}")
ax.axvspan(ci_low, ci_high, alpha=0.18, color="gold",
           label=f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
ax.axvline(ci_low, color="darkorange", lw=1.5, linestyle="--")
ax.axvline(ci_high, color="darkorange", lw=1.5, linestyle="--")
# 標記個別受試者
for a in subject_accs:
    ax.axvline(a, color="steelblue", lw=0.8, alpha=0.5)
ax.set_xlabel("Bootstrap 均值準確率", fontsize=12)
ax.set_ylabel("次數", fontsize=12)
ax.set_title(f"均值的 Bootstrap 分佈（N = {n_subjects} 位受試者）\n"
             f"95% CI 寬度 = {ci_high - ci_low:.3f}  — 將近 20 個百分點那麼寬！",
             fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print(f"\n95% Bootstrap CI:  [{ci_low:.3f}, {ci_high:.3f}]")
print(f"CI 寬度:            {ci_high - ci_low:.3f}")
print(f"\n解讀：以 N={n_subjects} 位受試者，真實均值合理地可能在")
print(f"{ci_low:.0%} 到 {ci_high:.0%} 之間的任何地方。")
print("這是一個很大的範圍！請務必在均值 ± 標準差旁邊回報 N。")

# %% [markdown]
# **Bootstrap 分佈告訴你什麼：**
# 每個長條代表「這次對 9 位受試者的 bootstrap 重抽樣得到了這個均值」。
# 金色陰影區域是 95% CI。
#
# 注意 CI 有多**寬**——將近 20 個百分點！只有 9 位受試者的研究回報「準確率 = 0.68」
# 留下了巨大的不確定性。
#
# 這不是分析中的缺陷——這是對 9 位受試者能告訴我們什麼和不能告訴我們什麼的誠實描述。
# 縮窄 CI 的唯一方法是招募更多參與者。
#
# **標準差 vs. 標準誤 vs. CI — 應該回報哪個？**
# | 量 | 測量 | 使用時機 |
# |---|---|---|
# | 標準差（σ） | *受試者間*的分散程度 | 顯示個別變異性 |
# | 標準誤 = σ/√N | 關於*均值*的不確定性 | 比較、推論 |
# | 95% CI | 均值的合理範圍 | 發表、最清晰的溝通 |
#
# 許多論文在應該顯示 CI 時回報均值 ± 標準差。要知道兩者的區別。

# %% [markdown]
# ---
# ## 第 3 節 — 機率水準（chance level）並非永遠是 1/n_classes
#
# 分類論文中最常見的錯誤：假設因為有 2 個類別，機率基線就是 50%。
# **只有在類別平衡時才成立。**
#
# ### 不平衡的情況
# 假設你在偵測癲癇：90% 的 5 分鐘時窗是正常的（類別 0），10% 含有癲癇發作（類別 1）。
# *永遠*預測「正常」的模型**免費獲得 90% 準確率**——無需學習任何東西。

# %%
# 模擬不平衡二元分類問題
rng2 = np.random.default_rng(42)

N_TRIALS = 200
P_CLASS1 = 0.10   # 癲癇發作率（罕見）
P_CLASS0 = 1 - P_CLASS1

# 真實標籤（90% 零、10% 一）
y_true = (rng2.random(N_TRIALS) < P_CLASS1).astype(int)
print(f"類別計數: 0 → {(y_true==0).sum()}, 1 → {(y_true==1).sum()}")
print(f"類別比例: {np.bincount(y_true) / N_TRIALS}")

# 多數類別虛擬模型：永遠預測 0
y_majority = np.zeros(N_TRIALS, dtype=int)
majority_acc = (y_majority == y_true).mean()

# 隨機模型：以相等機率預測 0 或 1
y_random = rng2.integers(0, 2, size=N_TRIALS)
random_acc = (y_random == y_true).mean()

# 完美模型（oracle）— 上界
perfect_acc = 1.0

print(f"\n永遠預測多數類別的準確率: {majority_acc:.3f}")
print(f"隨機（50/50）準確率:       {random_acc:.3f}")
print(f"完美模型準確率:            {perfect_acc:.3f}")
print(f"\n=> 真實基線是 {majority_acc:.2f}，而非 0.50！")

# %% [markdown]
# ### 視覺化基線（baselines）

# %%
# 同時顯示：即使在真正的機率水準下（投公平硬幣），小測試集也可能「幸運」地得到高分
N_BOOT_CHANCE = 5_000
n_test_small = 20

# 對 N_TRIALS 個標籤進行公平硬幣預測
chance_accs_small = np.array([
    ((rng2.integers(0, 2, n_test_small) == y_true[:n_test_small]).mean())
    for _ in range(N_BOOT_CHANCE)
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 左：模型基線長條圖
ax = axes[0]
models_bar = ["永遠\n預測 0\n（多數類別）", "隨機\n50/50", "你的\n模型\n必須超越"]
scores_bar = [majority_acc, random_acc, majority_acc]
bar_colors = ["tomato", "orange", "crimson"]
bars = ax.bar(models_bar, scores_bar, color=bar_colors, width=0.5, edgecolor="black")
ax.axhline(0.5, color="gray", linestyle="--", lw=1.5, label="樸素 1/k 機率 = 0.50")
ax.axhline(majority_acc, color="crimson", linestyle="-", lw=1.5,
           label=f"真實基線 = {majority_acc:.2f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("準確率", fontsize=12)
ax.set_title("不平衡類別（90% 為類別 0）\n"
             "多數類別分類器不需學習任何東西就能得到 0.90", fontsize=10)
ax.legend(fontsize=9)
for bar, score in zip(bars, scores_bar):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.01, f"{score:.2f}",
            ha="center", fontsize=10, fontweight="bold")

# 右：小 n 下的「幸運機率」分佈
ax2 = axes[1]
ax2.hist(chance_accs_small, bins=25, color="slategray", edgecolor="white", alpha=0.85)
ax2.axvline(0.5, color="navy", lw=2, label="真實機率 = 0.50")
ax2.axvline(majority_acc, color="crimson", lw=2,
            label=f"多數類別基線 = {majority_acc:.2f}")
lucky = (chance_accs_small >= 0.80).mean()
ax2.set_xlabel(f"公平硬幣模型在 n={n_test_small} 個試次上的準確率", fontsize=11)
ax2.set_ylabel("次數", fontsize=11)
ax2.set_title(f"即使隨機模型也可能在 {lucky:.0%} 的實驗中得到 ≥ 0.80 的分數\n"
              f"（n = {n_test_small} 個測試試次）", fontsize=10)
ax2.legend(fontsize=9)

plt.suptitle("不平衡：真實基線是多數類別準確率，而非 1/n_classes",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\n只有 n={n_test_small} 個測試試次時，純隨機模型")
print(f"在 {lucky:.1%} 的實驗中準確率 ≥ 0.80。")
print("=> 小測試集 + 不平衡類別 = 非常不可靠的評估。")

# %% [markdown]
# **第 3 節的關鍵要點：**
# * 請務必在你的模型準確率旁邊回報**多數類別基線（majority-class baseline）**。
# * 對於不平衡問題，優先使用**平衡準確率（balanced accuracy）**、**F1** 或 **AUC**，
#   而非原始準確率（第 12 章涵蓋）。
# * 當測試集很小*且*類別不均衡時，即使硬幣投擲模型也能得到令人印象深刻的高分——
#   二項分佈（binomial distribution）有長長的右尾。

# %% [markdown]
# ### 第 3b 節 — 「我的準確率*顯著高於*機率水準嗎？」（二項檢定，binomial test）
#
# 由於小測試集充滿雜訊，你應該詢問你的準確率是否可能是*靠運氣*在真實機率水準下出現的。
# 有 `n` 個測試試次且 `k` 個正確時，在機率模型下，正確數服從 **Binomial(n, p_chance)**。
# 兩個初學者友好的工具：
#
# 1. **二項檢定（binomial test）**p 值：靠運氣達到*這麼好或更好*結果的機率。
# 2. 準確率的**信賴區間下界（lower confidence bound）**——
#    [Müller-Putz 等人 2008] 的 BCI 經驗法則：回報準確率及其信賴區間，
#    只有當**下界超過機率水準**時才宣稱「高於機率水準」。

# %%
from scipy import stats as _st


def acc_above_chance(k_correct, n_trials, p_chance=0.5, conf=0.95):
    """分類準確率的二項檢定與 Wilson 下界。"""
    acc = k_correct / n_trials
    p_value = _st.binomtest(k_correct, n_trials, p_chance, alternative="greater").pvalue
    lo, hi = _st.binom.interval(conf, n_trials, acc)  # 計數的粗略 CI
    return acc, p_value, lo / n_trials, hi / n_trials


for n_trials in (20, 100, 400):
    k = round(0.65 * n_trials)  # 真實得分 65% 的模型
    acc, p, lo, hi = acc_above_chance(k, n_trials, p_chance=0.5)
    verdict = "高於機率水準" if lo > 0.5 else "無法與機率水準區分"
    print(f"n={n_trials:4d}: acc={acc:.2f}  95% CI≈[{lo:.2f},{hi:.2f}]  "
          f"binom p={p:.1e}  -> {verdict}")

# %% [markdown]
# 注意，*相同的* 0.65 準確率在 400 個試次時令人信服地高於機率水準，
# 但**在只有 20 個試次時，其信賴區間跨越了 0.50**——相同的點估計，完全相反的結論。
# 這就是為什麼微小的 BCI 測試集需要機率水準檢定，而不只是一個數字。

# %% [markdown]
# ---
# ## 第 4 節 — 兩個長條間的小差距很可能只是雜訊
#
# 你回報了兩個模型。長條圖看起來像是：
#
# ```
# 模型 A: 0.66    模型 B: 0.68
# ```
#
# 模型 B 更好嗎？幾乎可以確定**不是**——沒有統計檢定的話不行。
#
# ### 設定
# 我們模擬兩個模型在 9 個交叉驗證（cross-validation）折疊上的情況，
# 其中兩個模型的真實效能完全相同（都約為 0.67）。由於抽樣雜訊，
# 其中一個碰巧比另一個的平均值略高。

# %%
rng3 = np.random.default_rng(7)

n_folds = 9
# 逐折疊準確率（配對：兩個模型使用相同折疊）
# 真實均值非常接近；標準差很大（典型的小 N 情況）
fold_accs_A = np.array([0.60, 0.70, 0.55, 0.75, 0.65, 0.72, 0.58, 0.78, 0.62])
fold_accs_B = fold_accs_A + rng3.normal(loc=0.02, scale=0.04, size=n_folds)
fold_accs_B = np.clip(fold_accs_B, 0.0, 1.0)

mean_A, std_A = fold_accs_A.mean(), fold_accs_A.std(ddof=1)
mean_B, std_B = fold_accs_B.mean(), fold_accs_B.std(ddof=1)

print(f"模型 A: mean={mean_A:.3f}  std={std_A:.3f}")
print(f"模型 B: mean={mean_B:.3f}  std={std_B:.3f}")
print(f"差異 B - A = {mean_B - mean_A:.3f}")

# %% [markdown]
# ### 視覺化：重疊的誤差條告訴你差異可能為零

# %%
from scipy import stats

# 配對 t 檢定（相同折疊 => 配對，而非獨立）
t_stat, p_ttest = stats.ttest_rel(fold_accs_A, fold_accs_B)
# Wilcoxon 符號秩檢定（非參數替代）
w_stat, p_wilcox = stats.wilcoxon(fold_accs_A, fold_accs_B)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：帶有誤差條的長條圖（大多數論文展示的）
ax = axes[0]
x = np.array([0, 1])
means = [mean_A, mean_B]
stds  = [std_A, std_B]
bar_h = ax.bar(x, means, yerr=stds, width=0.4, capsize=8,
               color=["steelblue", "darkorange"], edgecolor="black",
               error_kw=dict(elinewidth=2, ecolor="black"))
ax.set_xticks(x)
ax.set_xticklabels(["模型 A", "模型 B"], fontsize=13)
ax.set_ylabel("平均準確率（± 折疊間標準差）", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title(f"長條圖：B 看起來好了 {mean_B - mean_A:.3f}\n"
             f"但誤差條重疊——差異在雜訊範圍內", fontsize=10)
for xi, (m, s) in enumerate(zip(means, stds)):
    ax.text(xi, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=12, fontweight="bold")

# 標注差距
ax.annotate("", xy=(1, mean_B), xytext=(0, mean_A),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2))
ax.text(0.5, (mean_A + mean_B)/2 + 0.01, f"Δ={mean_B-mean_A:.3f}",
        ha="center", fontsize=10, color="crimson")

# 右：逐折疊散點圖（你也應該展示的）
ax2 = axes[1]
folds_x = np.arange(1, n_folds + 1)
ax2.plot(folds_x, fold_accs_A, "o-", color="steelblue", label=f"模型 A (mean={mean_A:.3f})", lw=2)
ax2.plot(folds_x, fold_accs_B, "s--", color="darkorange", label=f"模型 B (mean={mean_B:.3f})", lw=2)
ax2.set_xlabel("折疊", fontsize=12)
ax2.set_ylabel("準確率", fontsize=12)
ax2.set_title(f"逐折疊準確率（n={n_folds} 個折疊）\n"
              f"配對 t 檢定: p = {p_ttest:.3f}  |  Wilcoxon: p = {p_wilcox:.3f}", fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0.3, 1.0)
ax2.axhline(mean_A, color="steelblue", linestyle=":", alpha=0.5)
ax2.axhline(mean_B, color="darkorange", linestyle=":", alpha=0.5)
ax2.set_xticks(folds_x)

sig_label = "不顯著" if p_ttest > 0.05 else "顯著"
color_sig  = "firebrick" if p_ttest > 0.05 else "seagreen"
ax2.text(0.98, 0.05, f"配對 t 檢定: {sig_label}\n(p = {p_ttest:.3f} > 0.05)",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=10,
         color=color_sig,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

plt.suptitle("模型 B 真的更好嗎？配對檢定說：可能不是。",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\n配對 t 檢定:    t = {t_stat:.3f},  p = {p_ttest:.4f}")
print(f"Wilcoxon 檢定:  W = {w_stat:.1f},   p = {p_wilcox:.4f}")
print()
if p_ttest > 0.05:
    print("=> p > 0.05: 差異在統計上不顯著。")
    print("   我們無法宣稱模型 B 比模型 A 更好。")
else:
    print("=> p < 0.05: 差異在統計上顯著。")
print()
print(f"差距 {mean_B - mean_A:.3f} 小於 1 個標準差（{std_A:.3f}）。")
print("僅回報均值而不進行檢定將會誤導讀者。")

# %% [markdown]
# **配對檢定的作用以及為何「配對」很重要：**
#
# 配對檢定問的是：「在*相同的*折疊上，模型 B 是否始終比模型 A 好，
# 還是每個模型都會在某些折疊上靠運氣贏？」因為兩個模型在每個折疊中看到相同的資料分割，
# 它們的折疊分數是*相關的*。使用配對檢定（ttest_rel 或 Wilcoxon）而非獨立檢定
# 可以得到更公平的比較和更大的統計功效（power）。
#
# **宣稱模型 B 打敗模型 A 前的實用清單：**
# - [ ] 回報*兩個*模型的均值 ± 標準差。
# - [ ] 確保圖形中顯示誤差條。
# - [ ] 在折疊／受試者上執行**配對**檢定。
# - [ ] 回報 p 值；如果 p > 0.05，請說「無顯著差異」。
# - [ ] 最好也回報效果量（Cohen's d）和差異的信賴區間。

# %% [markdown]
# ### 第 4b 節 — 交叉驗證違反了 t 檢定的假設（以及如何修正）
#
# 這裡有一個幾乎所有人都搞錯的細節。標準配對 t 檢定假設折疊分數是**獨立的**。
# 但在 k 折交叉驗證（k-fold CV）中，訓練集*大量重疊*（任意兩個折疊共享大部分訓練資料），
# 因此折疊差異是**相關的**。這使得樸素 t 檢定**過於自信**——
# 它回報的 p 值比應有的*更小*，所以你「發現」了實際上並不存在的改進。
#
# **Nadeau & Bengio（2003）**提供了一個修正方差的版本，考慮了訓練／測試重疊。
# 這是一行的修正，比較 CV 折疊上的模型時應作為你的預設方法。

# %%
def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau-Bengio 方差修正配對 t 檢定，適用於重疊 CV 折疊。

    `diffs` 是逐折疊分數差異（模型 B - 模型 A）；`n_train`/`n_test`
    是每個折疊訓練／測試分割中的樣本數。
    """
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    corrected_var = var * (1.0 / n + n_test / n_train)   # 修正項
    if corrected_var <= 0:
        return 0.0, 1.0
    t = mean / np.sqrt(corrected_var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return t, p


diffs = fold_accs_B - fold_accs_A
n_total = 100  # 假設每個折疊測試約 1/9 的 100 個樣本
n_test = n_total // n_folds
n_train = n_total - n_test
t_corr, p_corr = corrected_resampled_ttest(diffs, n_train, n_test)
print(f"樸素配對 t 檢定:         p = {p_ttest:.3f}")
print(f"修正後（Nadeau-Bengio）: p = {p_corr:.3f}   <- 較大 = 更誠實")
print("修正使 p 值變大：重疊的 CV 折疊不是獨立的，")
print("因此樸素檢定誇大了顯著性。預設應使用修正後的版本。")

# %% [markdown]
# ---
# ## ✅ 概念自測（Concept check）
#
# 繼續前請先測試你的理解。
#
# **Q1.** 一篇論文在 2 類別 EEG 資料集上回報「accuracy = 0.72」。
# 不知道使用了多少個測試樣本，你能評估這個結果是否有意義嗎？為什麼？
#
# **Q2.** 你在 8 位受試者上執行 LOSO 交叉驗證，得到均值準確率 = 0.74 ± 0.15。
# 一位同事說「很好，CI 是 ±0.15」。這個說法有什麼問題？
# 信賴區間實際上應該從什麼計算？
#
# **Q3.** 一個癲癇偵測器在測試集上達到 88% 準確率，其中 85% 的時窗沒有癲癇發作。
# 這是個好結果嗎？你應該與什麼基線比較？
#
# **Q4.** 模型 A 在 8 個 CV 折疊上平均 0.70，模型 B 平均 0.73，
# 兩者標準差都約為 0.12。審稿人說「B 明顯獲勝」。你會怎麼做來檢驗這個說法？
#
# ---
# **解答：**
#
# **A1.** 不行。沒有 n_test，你無法知道抽樣變異。10 個測試試次時估計波動 ±0.15；
# 500 個時是 ±0.04。相同的回報數字根據 n 的不同代表著非常不同的含義。
#
# **A2.** ± 標準差（std）**不是**信賴區間。均值的 CI 必須考慮樣本大小：
# 標準誤（SE）= std/√N。以 N=8 位受試者且 std=0.15，SE ≈ 0.053，
# 95% CI ≈ ±0.12（而非 ±0.15）。應使用 bootstrap 或 t 區間計算均值的 CI，
# 而非直接使用原始標準差。
#
# **A3.** 不（不明確）。多數類別基線（永遠預測「無癲癇發作」）已達 85%。
# 88% 的模型只比什麼都不學的模型高 3 個百分點。
# 應使用平衡準確率（balanced accuracy）或 F1 來公平評估不平衡問題。
#
# **A4.** 在逐折疊分數上執行**配對**統計檢定（ttest_rel 或 Wilcoxon）。
# 如果 p > 0.05，你無法宣稱 B 更好。差距（0.03）遠小於 1 個標準差（0.12），
# 這強烈暗示這只是雜訊。

# %% [markdown]
# ---
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# | 錯誤 | 為何重要 | 應該怎麼做 |
# |---|---|---|
# | 只回報一個數字（「accuracy = 0.72」） | 隱藏所有不確定性；讀者無法判斷可靠性 | 回報均值 ± 標準差*和* n_test（或 n_subjects） |
# | 假設機率水準 = 1/n_classes | 對不平衡資料不正確；誇大表面上的改進 | 總是計算多數類別基線；使用平衡準確率 |
# | 宣稱小於 1 個標準差的改進是真實的 | 差異是抽樣雜訊的機率很高 | 總是在折疊間執行配對顯著性檢定 |
# | 忽略 N | N=5 位受試者和 N=50 的信心水準不同 | 明確說明 N；回報 CI 寬度 |
# | 將標準差視為 CI | 標準差測量受試者間的分散；CI/√N 測量均值的不確定性 | 對推論回報標準誤或 bootstrap CI |
# | 資料配對時使用獨立檢定 | 損失統計功效；可能誇大或縮小 p 值 | 比較相同折疊時使用 ttest_rel / Wilcoxon |
#
# ---
# **下一章：** 第 12 章 — 評估與陷阱——我們將把這些反射應用到真實的解碼流水線上，
# 並以 WRONG → RIGHT 配對的方式逐一呈現每個錯誤如何誇大分數。
