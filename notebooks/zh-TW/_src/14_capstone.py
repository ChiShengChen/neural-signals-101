# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/14_capstone.ipynb)
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
# # 第 14 章 — 綜合專題：誠實的排行榜（Honest Leaderboard）
#
# ## 你的任務
# 為 BCI IV 2a 資料集建立一個運動想像（motor-imagery）解碼器，僅在 **DEV** 集上反覆迭代，
# 然後向**隱藏的保留排行榜（hidden held-out leaderboard）**提交一次，看看你的估計有多誠實。
# 請自行填入 `TODO` 的部分。
#
# ## 遊戲規則
# - 一開始就切出一個**隱藏集（hidden set）**（一個或多個完整受試者）。
# - 你只能在剩下的 **DEV 受試者**上，使用留一受試者法（Leave-One-Subject-Out）反覆迭代。
# - 當你滿意之後，呼叫 `score_on_hidden(my_pipeline)` **一次**——那就是你的排行榜提交。
# - 我們也會展示 ⚠️ **作弊者排行榜**：有人在選擇流水線時偷看了隱藏集，然後把精心挑選的分數
#   當成自己的結果。你將親眼看到這會讓分數虛高多少。
#
# ## 規則（與整個教程一致）
# 1. 只在 DEV 受試者上開發與調校。
# 2. 在最後才呼叫 `score_on_hidden`，且**只呼叫一次**。
# 3. 所有學習過的轉換步驟都必須放在**流水線（pipeline）內部**（只在訓練集上擬合）。
# 4. 回報**平均值 ± 標準差（mean ± std）**，絕不只回報單一數字。
# 5. 固定所有隨機種子（seed）。
#
# > **前置條件：** 第 12 章。
# > **難度：** ★★★★☆
# > **執行時間：** 填完 TODO 後，CPU 上約 2–4 分鐘。

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

from neuro101 import io, datasets as ds, features as ft, viz
from neuro101.eval import (
    leakage_safe_pipeline, make_subject_split, make_block_split, evaluate_with_variance,
)

rng = np.random.default_rng(0)
SMOKE = ds.is_smoke()

# %% [markdown]
# ## 步驟 1 — 載入資料並切出隱藏集
#
# 我們載入固定數量的受試者，然後**確定性地**保留最後幾個受試者作為隱藏保留集（hidden held-out set）。
# 在最終「排行榜提交」儲存格之前，你不會碰到 `X_hidden` / `y_hidden`。

# %%
# 在煙霧測試（smoke/CI）模式下使用 3 個受試者（1 個隱藏，2 個 DEV）。
# 完整模式：5 個受試者（1 個隱藏，4 個 DEV）。
n_total  = 3 if SMOKE else 5
n_hidden = 1            # 保留作為隱藏排行榜集的受試者數量

X, y, subj = io.load_bnci_2a_epochs(n_subjects=n_total)
all_subjects = np.unique(subj)

# 確定性切分：最後幾個受試者為隱藏集。
LOCKED_SUBJECTS = list(all_subjects[-n_hidden:])
DEV_SUBJECTS    = list(all_subjects[:-n_hidden])

dev_mask  = np.isin(subj, DEV_SUBJECTS)
X_dev, y_dev, subj_dev = X[dev_mask],  y[dev_mask],  subj[dev_mask]
X_hidden, y_hidden     = X[~dev_mask], y[~dev_mask]

print(f"DEV 受試者：{DEV_SUBJECTS}  — {X_dev.shape[0]} 次試驗")
print(f"LOCKED 受試者：{LOCKED_SUBJECTS}  — {X_hidden.shape[0]} 次試驗")
print(f"                  ^^^ 在最後一個儲存格之前請勿查看此資料 ^^^")

# %% [markdown]
# ### `score_on_hidden` 輔助函式
#
# 請**恰好呼叫一次**——在你完成所有迭代之後的筆記本末尾。
# 它會在所有 DEV 資料上訓練，並在隱藏集上進行評估。

# %%
def score_on_hidden(pipeline):
    """在所有 DEV 受試者上訓練，對鎖定的保留受試者進行評估。

    回傳隱藏集上的準確率（accuracy）。
    請在開發週期結束時恰好呼叫一次（ONCE）。
    """
    model = clone(pipeline)
    model.fit(X_dev, y_dev)
    acc = accuracy_score(y_hidden, model.predict(X_hidden))
    print(f"[排行榜] 隱藏保留集準確率：{acc:.3f}")
    return acc

# %% [markdown]
# ## 步驟 2 — TODO：建立你的流水線
#
# 用你自己的流水線取代下方的基準線。一些想法：更多的 CSP 分量、不同的分類器
#（SVM、LogReg），或黎曼步驟（`ft.make_riemann_pipeline_steps()`）。
# **讓所有學習過的步驟保持在流水線內部**，以確保它們只在訓練折（train fold）上擬合。

# %%
# --- 基準線（請替換此處）---
my_pipeline = leakage_safe_pipeline([
    ("csp", ft.make_csp(n_components=4)),
    ("clf", LinearDiscriminantAnalysis()),
])
# TODO: 試試看，例如：
# from sklearn.svm import SVC
# my_pipeline = leakage_safe_pipeline(
#     ft.make_riemann_pipeline_steps() + [("clf", SVC(C=1, random_state=0))]
# )

# %% [markdown]
# ## 步驟 3 — DEV 迴圈：僅在 DEV 受試者上進行誠實的 LOSO 評估
#
# 這是你在建立模型過程中**唯一**應該參考的評估結果。
# 如果有 ≥ 2 個 DEV 受試者，我們使用留一受試者法（Leave-One-Subject-Out，LOSO）。
# 在只有 1 個 DEV 受試者的煙霧模式下，則退而使用區塊感知（block-aware）切分。

# %%
n_dev_subjects = len(DEV_SUBJECTS)

if n_dev_subjects >= 2:
    cv_fn = lambda: make_subject_split(subj_dev)          # noqa: E731
    cv_label = f"LOSO（{n_dev_subjects} 折）"
else:
    # 煙霧模式：只有一個 DEV 受試者——改用區塊感知切分。
    cv_fn = lambda: make_block_split(len(y_dev), n_splits=4)   # noqa: E731
    cv_label = "區塊感知切分（煙霧／單受試者）"

dev_report = evaluate_with_variance(
    my_pipeline, X_dev, y_dev,
    cv=cv_fn,
    scoring=("accuracy", "balanced_accuracy"),
    seeds=(0, 1),
)
for metric in ("accuracy", "balanced_accuracy"):
    m = dev_report[metric]
    print(f"  {metric:18s}: {m['mean']:.3f} ± {m['std']:.3f}")
print(f"  （CV：{cv_label}，{dev_report['n_folds']} 折 × {dev_report['n_seeds']} 個種子）")

dev_acc_mean = dev_report["accuracy"]["mean"]
dev_acc_std  = dev_report["accuracy"]["std"]

# %% [markdown]
# ### 你的 DEV 估計值就是你的誠實預期
#
# 在執行下一個儲存格之前先記下這個數字——它是你對隱藏集結果的最佳猜測。

# %% [markdown]
# ## 步驟 4 — ⚠️ 作弊者排行榜（錯誤做法——請勿如此）
#
# 作弊者不是只在 DEV 集上開發。他們建立多個流水線，對**每個流水線**都偷看隱藏集，
# 挑選看到的最高分數，然後把它當成自己的結果。下方的儲存格重現了這個錯誤，
# 讓你看到分數虛高了多少。
#
# **每次查看隱藏集都是一次偷窺——它會消耗一個自由度（degree of freedom）。**

# %%
# 作弊者在隱藏集上嘗試的三個流水線。
_cheat_pipelines = {
    "CSP-4 + LDA (baseline)": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=4)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "CSP-8 + LDA": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=8)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "CSP-4 + LDA (seed 42)": leakage_safe_pipeline([
        ("csp", ft.make_csp(n_components=4)),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
}

_cheat_scores = {}
print("⚠️  作弊者對每個流水線都偷看了隱藏集：")
for name, pipe in _cheat_pipelines.items():
    m = clone(pipe)
    m.fit(X_dev, y_dev)
    s = accuracy_score(y_hidden, m.predict(X_hidden))
    _cheat_scores[name] = s
    print(f"   {name:30s} → hidden={s:.3f}")

_cheater_best_name  = max(_cheat_scores, key=_cheat_scores.get)
_cheater_best_score = _cheat_scores[_cheater_best_name]
print(f"\n⚠️  作弊者回報：'{_cheater_best_name}' → {_cheater_best_score:.3f}")

# 現在對「獲勝者」同一流水線計算誠實的 DEV 估計值以進行比較。
_honest_of_winner = evaluate_with_variance(
    _cheat_pipelines[_cheater_best_name], X_dev, y_dev,
    cv=cv_fn,
    scoring="accuracy", seeds=(0, 1),
)["accuracy"]

print(f"\n   「獲勝者」的誠實 DEV 估計值：{_honest_of_winner['mean']:.3f} ± {_honest_of_winner['std']:.3f}")
print(f"   作弊者的聲稱（{len(_cheat_scores)} 次偷看的最佳值）：{_cheater_best_score:.3f}")
_inflation = _cheater_best_score - _honest_of_winner["mean"]
print(f"   選擇偏誤（selection bias）造成的明顯虛高：  {_inflation:+.3f}")
print()
print("   作弊者無法知道他們回報的分數是否是多次中「幸運」的那一次。")
print("   每次偷窺都會在不可替換的資料集上消耗一個自由度。")

# %% [markdown]
# ### 為什麼在測試集上進行選擇會造成虛高？
#
# 當你在同一個固定測試集上評估 K 個流水線並挑選最大值時，
# 你實際上是在做等同於*多重比較（multiple-comparisons）*的搜尋。
# K 個分數中的最大值幾乎總是高於真實的期望值——即使這些流水線中
# 沒有一個真的比其他的更好。你嘗試的模型越多，虛高就越嚴重。
#
# 唯一的保護措施是**絕不在測試集上進行選擇**：在 DEV 集上（透過 LOSO / 巢狀 CV）
# 挑選你的架構，鎖定它，然後提交一次。

# %% [markdown]
# ## 步驟 5 — TODO：接上你自己的流水線並提交至排行榜
#
# 1. 在步驟 2 / 3 反覆迭代，直到你對 DEV LOSO 分數感到滿意。
# 2. 用你的最終選擇替換步驟 2 中的 `my_pipeline`。
# 3. **執行下方儲存格一次**——這就是你的排行榜提交。

# %%
# >>> 你的最終提交——只執行這個儲存格一次 <<<
hidden_acc = score_on_hidden(my_pipeline)

# %% [markdown]
# ## 步驟 6 — 報告圖表
#
# 清晰的摘要：你的 DEV 估計值與誠實的隱藏保留集結果。
# 差距顯示你的 LOSO 估計值與現實有多接近（或多遙遠）。

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
labels  = ["DEV 估計值\n（LOSO 誠實）", "隱藏保留集\n（排行榜）"]
heights = [dev_acc_mean, hidden_acc]
errs    = [dev_acc_std, 0.0]
colors  = ["#4878CF", "#6ACC65"]
bars = ax.bar([0, 1], heights, width=0.55, color=colors,
              yerr=errs, capsize=6)
ax.axhline(0.25, ls="--", color="gray", lw=1)
ax.text(1.45, 0.25, "機率基線 = 0.25", color="gray", va="center", ha="left", fontsize=8)
for b, h in zip(bars, heights):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}",
            ha="center", va="bottom", fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel("準確率（Accuracy）")
ax.set_ylim(0, 1.0)
gap = dev_acc_mean - hidden_acc
ax.set_title(f"綜合專題：DEV vs 隱藏集（差距 = {gap:+.3f}）")
fig.tight_layout()
plt.show()

print(f"\nDEV 估計值：{dev_acc_mean:.3f} ± {dev_acc_std:.3f}")
print(f"隱藏集分數：{hidden_acc:.3f}")
print(f"差距        ：{gap:+.3f}  "
      + ("（DEV 過於樂觀）" if gap > 0 else "（DEV 過於悲觀）"))

# %% [markdown]
# ## 步驟 7 — TODO：寫下你的結論
#
# 在下方的 markdown 儲存格中回答：
# 1. 你的**誠實**隱藏保留集準確率是多少？
# 2. 你的 DEV 估計值與隱藏結果相差多少？
# 3. 你的流水線改動相對於基準線真的有所提升嗎（考量到 ± std）？
# 4. 如果有更多資料或時間，你接下來會嘗試什麼？

# %% [markdown]
# > **在此填入你的結論。**
# >
# > _範例：_ "CSP+LDA 在隱藏集上達到 0.56（vs DEV 估計值 0.70）。
# > ~0.14 的差距對於 BCI 2a 的跨受試者遷移來說是典型的。
# > 考量到分數的分散程度，我無法顯著超越基準線；下一步我會嘗試
# > 用少量校準資料進行每位受試者的微調（per-subject fine-tuning）。"

# %% [markdown]
# ## ⚠️ 常見錯誤
#
# - **多次呼叫 `score_on_hidden`。** 每次呼叫都是一次偷窺；你將無法再信任那個數字。
# - **將 DEV LOSO 數字當成最終結果回報**，而不說明它只是一個估計值。隱藏集分數才是基準事實。
# - **聲稱小於標準差的改進。** 如果模型 A 是 0.66 ± 0.10，模型 B 是 0.68 ± 0.10，
#   你並*沒有*證明 B 更好。
# - **偷偷換成隨機切分**「因為分數比較好看」。這正是整個教程要防止的陷阱。
# - **在測試受試者上調校超參數（hyper-parameters）。** 所有調校都應使用巢狀 CV 或
#   獨立的驗證受試者；保留受試者只用於最終評分。
#
# ---
#
# 🎓 **恭喜——你已完成整個教程！**
#
# 你現在可以從原始錄製的神經訊號，一路走到誠實、無資料洩漏的報告。你理解了*為什麼*
# 評估方法很重要，*如何*發現常見陷阱，以及*分數究竟代表什麼意義*。這種組合——
# 技術能力加上評估嚴謹性——是應用神經訊號機器學習中最稀缺的東西。去建造一些真實的東西吧。
