# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 機會水準（Chance Level）與信賴區間（Confidence Intervals）
#
# 「高於機率水準」對 BCI 結果究竟意味著什麼，
# 以及你對報告的分類準確率應有多大的信心？
#
# > **先備知識：** 主要章節 11。
# > **難度：** 進階 ★★★★☆
# > **不受 5 分鐘 CPU 預算限制。**

# %%
# --- 啟動程序：優先使用已安裝的套件，退而尋找 repo 中的 src ---
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(6):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# %% [markdown]
# ---
# ## 第一部分 — 二項式檢定（Binomial Test）：形式化「優於機率」
#
# 當分類器對 **n** 個獨立測試試次做出預測並正確分類 **k** 個時，
# 機率分類器的正確預測數服從：
#
# $$K_\text{chance} \sim \text{Binomial}(n,\; p_\text{chance}), \quad
#   p_\text{chance} = 1/k_\text{classes}$$
#
# 單尾二項式檢定（one-sided binomial test）問的是：若真實準確率恰好為
# $p_\text{chance}$，僅憑運氣觀察到**至少** k 個正確的概率是多少？
# 該概率就是 **p 值（p-value）**。
#
# $$p = P(K \ge k \mid n,\; p_\text{chance}) = \sum_{j=k}^{n} \binom{n}{j} p_\text{chance}^j (1-p_\text{chance})^{n-j}$$
#
# `scipy.stats.binomtest` 精確計算此值（無正態近似）。

# %%
def binomial_test(k_correct: int, n: int, p_chance: float = 0.5) -> dict:
    """
    單尾二項式檢定：H0 準確率 = p_chance，H1 準確率 > p_chance。

    Parameters
    ----------
    k_correct : int   正確分類的試次數。
    n         : int   測試試次總數。
    p_chance  : float 機率水準（1/n_classes）。

    Returns
    -------
    包含以下鍵的 dict：acc、p_value、significant（alpha=0.05）
    """
    acc = k_correct / n
    result = stats.binomtest(k_correct, n, p_chance, alternative="greater")
    return {
        "acc": acc,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
    }


# 示範多種 (n, 觀察準確率) 組合
print(f"{'n':>6} {'obs_acc':>8} {'k':>5} {'p_value':>10} {'sig?':>6}")
print("-" * 42)
for n, obs_acc in [(20, 0.65), (20, 0.80), (100, 0.65), (100, 0.58), (400, 0.58)]:
    k = round(obs_acc * n)
    r = binomial_test(k, n, p_chance=0.5)
    sig = "YES" if r["significant"] else "no"
    print(f"{n:>6} {obs_acc:>8.2f} {k:>5} {r['p_value']:>10.4f} {sig:>6}")

# %% [markdown]
# 注意：**65% 準確率在僅有 20 次試次時不顯著**（p = 0.13），
# 而 400 次試次的 58% 卻高度顯著（p < 0.001）。
# 若不知道 n，原始準確率數字幾乎毫無意義。

# %% [markdown]
# ---
# ## 第二部分 — 準確率的信賴區間（CI）：Wald 法、Wilson 法與 Clopper-Pearson 法
#
# 觀察到的準確率 $\hat{p} = k/n$ 是點估計（point estimate）。我們始終需要一個區間。
# 三種常見選擇在小 n 或極端 p 下差異相當大。
#
# ### 2a  Wald 法（正態近似）區間
# $$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
# 簡單，但當 $\hat{p}$ 接近 0 或 1，或 n 較小時嚴重失效：
# 可能產生超出 [0, 1] 範圍的不可能區間。
#
# ### 2b  Wilson 分數區間（score interval，大多數情況下的推薦）
# $$\frac{\hat{p} + \tfrac{z^2}{2n} \pm z\sqrt{\tfrac{\hat{p}(1-\hat{p})}{n} + \tfrac{z^2}{4n^2}}}{1 + z^2/n}$$
# 保持在 [0,1] 內，對小 n 具有更佳的覆蓋率（coverage）。
#
# ### 2c  Clopper-Pearson 精確區間
# 直接反推二項式 CDF——覆蓋保證的「黃金標準」，
# 但傾向保守（比必要的更寬）。

# %%
def wald_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """正態近似（Wald 法）信賴區間。"""
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 分數信賴區間（Brown et al. 2001）。"""
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z ** 2
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """透過 beta 分布分位數的精確 Clopper-Pearson 區間。"""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


# --- 比較三種方法（固定觀察準確率）---
print("觀察準確率 = 0.60，三種 n 值的比較")
print(f"{'n':>5} {'方法':>18} {'下界':>7} {'上界':>7} {'寬度':>7}")
print("-" * 46)
for n in [10, 20, 100]:
    k = round(0.60 * n)
    p_hat = k / n
    methods = {
        "Wald": wald_ci(k, n),
        "Wilson": wilson_ci(k, n),
        "Clopper-Pearson": clopper_pearson_ci(k, n),
    }
    for name, (lo, hi) in methods.items():
        print(f"{n:>5} {name:>18} {lo:>7.4f} {hi:>7.4f} {hi-lo:>7.4f}")
    print()

# %% [markdown]
# ### 視覺化 1 — 三種方法的 CI 寬度與 n 的關係
#
# 對於固定「真實」準確率 0.60，隨著測試試次增加，95% CI 的寬度如何縮小？

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

obs_acc_vals = [0.60, 0.90]   # 中等 vs 極端觀察準確率
ns = np.arange(5, 301, 1)

for ax, obs_acc_fixed in zip(axes, obs_acc_vals):
    wald_widths, wilson_widths, cp_widths = [], [], []
    for n in ns:
        k = round(obs_acc_fixed * n)
        k = max(0, min(n, k))
        wald_widths.append(wald_ci(k, n)[1] - wald_ci(k, n)[0])
        wilson_widths.append(wilson_ci(k, n)[1] - wilson_ci(k, n)[0])
        cp_widths.append(clopper_pearson_ci(k, n)[1] - clopper_pearson_ci(k, n)[0])

    ax.plot(ns, wald_widths,    label="Wald",            color="tomato",      lw=1.8)
    ax.plot(ns, wilson_widths,  label="Wilson",          color="steelblue",   lw=1.8)
    ax.plot(ns, cp_widths,      label="Clopper-Pearson", color="seagreen",    lw=1.8, linestyle="--")
    ax.set_title(f"CI 寬度與 n 的關係（觀察準確率 = {obs_acc_fixed:.0%}）", fontsize=12)
    ax.set_xlabel("測試試次數（n）", fontsize=11)
    ax.set_ylabel("95% CI 寬度", fontsize=11)
    ax.legend(fontsize=10)
    ax.axvline(20,  color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.axvline(100, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.text(21, max(wald_widths) * 0.95, "n=20",  fontsize=8, color="gray")
    ax.text(101, max(wald_widths) * 0.95, "n=100", fontsize=8, color="gray")
    ax.set_xlim(5, 300)
    ax.set_ylim(0, None)

fig.suptitle("分類準確率 95% 信賴區間的寬度\n"
             "Wald 法在極端 p 下失效（右圖）；Wilson ≈ Clopper-Pearson",
             fontsize=11)
fig.tight_layout()
plt.savefig("/tmp/dd_chance_ci_width.png", dpi=110, bbox_inches="tight")
plt.show()
print("圖 1 已儲存。")

# %% [markdown]
# 主要觀察：
# * **Wald 法**對小 n 的 p_hat = 0.90 產生較窄（過度自信）的區間，
#   因為正態近似假設對稱尾部，即使分布右偏也如此。
# * **Wilson 法**與 **Clopper-Pearson 法**彼此接近，且保持在合理範圍內。
#   實務中使用 Wilson 法（封閉形式、校準良好）。
# * 僅有 20 次試次的 95% CI 大約寬達 ±20 個百分點——相當大。

# %% [markdown]
# ---
# ## 第三部分 — 真正的顯著性閾值：不只是 1/k
#
# 一個常見錯誤是只要準確率 > 1/n_classes 就宣稱「高於機率水準」。
# 但**在 n 較小時**，即使是純機率分類器也會足夠頻繁地超過 1/k 而誤導人。
#
# 正確的問題是：在顯著水準 α 下，單尾二項式檢定拒絕 H0 所需的
# **最低觀察準確率**是多少？
#
# 這是最小的 $\hat{p}$，使得
# $$P(K \ge k \mid n, p_\text{chance}) < \alpha，$$
# 即最小的 k 使 p 值降至 α 以下，除以 n。
#
# Mueller-Putz et al. (2008)* 在 BCI 文獻中引入了這一想法，
# 提供了一個評審者和從業者可以直接應用的簡易表格。
#
# *Müller-Putz G. R. et al., "Evaluating Causal Relations in Neural Systems …",
# Clinical Neurophysiology, 2008.*

# %%
def significance_threshold(n: int, p_chance: float, alpha: float = 0.05) -> float:
    """
    觀察結果達到統計顯著性所需的最低準確率（比例）
    （單尾二項式檢定，顯著水準 alpha）。

    回傳 [0, 1] 中的閾值比例。若即使 n/n 也無法達到顯著性
    （在非常小的 n 時可能發生），則回傳 1.0。
    """
    for k in range(1, n + 1):
        pval = stats.binomtest(k, n, p_chance, alternative="greater").pvalue
        if pval < alpha:
            return k / n
    return 1.0  # 在此 n 下未找到閾值


# --- 列出常見 (n, k_classes) 組合的閾值 ---
ns_table = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
k_classes_list = [2, 3, 4]

print("在 alpha=0.05 達到顯著性所需的最低準確率（二項式檢定）")
print(f"{'n':>5}  " + "  ".join(f"{'k='+str(k):>9}" for k in k_classes_list))
print("-" * 38)

# 儲存以供下方繪圖使用
thresh_k2 = []
thresh_k4 = []

for n in ns_table:
    row_vals = []
    for kc in k_classes_list:
        t = significance_threshold(n, 1.0 / kc)
        row_vals.append(t)
    thresh_k2.append(significance_threshold(n, 0.5))
    thresh_k4.append(significance_threshold(n, 0.25))
    print(f"{n:>5}  " + "  ".join(f"{v:>9.1%}" for v in row_vals))

# 提取兩個特定值
n20_k2  = significance_threshold(20,  0.5)
n100_k2 = significance_threshold(100, 0.5)
print(f"\n=> n=20，k=2（機率=0.50）的閾值：{n20_k2:.4f}  ({n20_k2:.1%})")
print(f"=> n=100，k=2（機率=0.50）的閾值：{n100_k2:.4f} ({n100_k2:.1%})")

# %% [markdown]
# ### 視覺化 2 — k=2 與 k=4 的顯著性閾值與 n 的關係
#
# 虛線水平線標示樸素的 1/k「機率水準」。
# **實線曲線顯示你實際需要的準確率**才能在 α=0.05 下顯著。
# 對於小 n，這可能遠高於 1/k。

# %%
ns_plot = np.arange(5, 501)
thresh_k2_plot = [significance_threshold(n, 0.5)  for n in ns_plot]
thresh_k4_plot = [significance_threshold(n, 0.25) for n in ns_plot]

fig, ax = plt.subplots(figsize=(11, 5))

ax.step(ns_plot, thresh_k2_plot, where="post",
        label="k=2（所需閾值）", color="steelblue",   lw=2.2)
ax.step(ns_plot, thresh_k4_plot, where="post",
        label="k=4（所需閾值）", color="darkorange",  lw=2.2)

ax.axhline(0.50, color="steelblue", lw=1.2, linestyle="--", alpha=0.6, label="1/2 樸素機率（k=2）")
ax.axhline(0.25, color="darkorange", lw=1.2, linestyle="--", alpha=0.6, label="1/4 樸素機率（k=4）")

# 標註 n=20、k=2 與 n=100、k=2 的點
ax.scatter([20, 100], [n20_k2, n100_k2], zorder=5, color="steelblue", s=60)
ax.annotate(f"n=20: need ≥{n20_k2:.0%}", xy=(20, n20_k2),
            xytext=(35, n20_k2 + 0.06), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1.0))
ax.annotate(f"n=100: need ≥{n100_k2:.0%}", xy=(100, n100_k2),
            xytext=(130, n100_k2 + 0.04), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1.0))

ax.set_xlabel("測試試次數（n）", fontsize=12)
ax.set_ylabel("p < 0.05 所需的最低準確率", fontsize=12)
ax.set_title("顯著性閾值與 n 的關係（單尾二項式檢定，α = 0.05）\n"
             "依 Müller-Putz et al. 2008——閾值緩慢趨近 1/k",
             fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(5, 500)
ax.set_ylim(0.2, 1.05)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("/tmp/dd_chance_threshold.png", dpi=110, bbox_inches="tight")
plt.show()
print("圖 2 已儲存。")

# %% [markdown]
# 階梯函數的形狀源於 k（正確數）是離散的。
# 對於 n = 20 的二類任務，你需要 **≥ 75% 準確率**才能在 α = 0.05 下顯著——
# 遠高於樸素的 50% 機率水準。在 n = 100 時，門檻降至 59%，
# 只有當 n → ∞ 時才趨近 50%。

# %% [markdown]
# ---
# ## 第四部分 — 實務報告清單
#
# 以現有數學將第 11 章的指引正式化：
#
# | 必須報告的項目 | 重要原因 |
# |---|---|
# | n（測試試次數） | 決定顯著性閾值與 CI 寬度 |
# | 機率水準 1/k_classes | 錨定虛無假設；勿假設讀者知道 k |
# | 觀察準確率（點估計） | 原始數字 |
# | 95% CI（Wilson 法或 Clopper-Pearson 法） | 顯示不確定性；CI 下界 > 機率 = 清楚的證據 |
# | 二項式檢定 p 值 | 精確檢定，無正態近似 |
# | 你的 n 對應的顯著性閾值 | 讓讀者可以套用 Müller-Putz 檢查 |
#
# ### 快速示範：相同準確率可能是決定性的，也可能毫無意義

# %%
scenarios = [
    dict(label="n=20,  acc=0.75", k=15, n=20,  p_chance=0.5),
    dict(label="n=20,  acc=0.65", k=13, n=20,  p_chance=0.5),
    dict(label="n=100, acc=0.65", k=65, n=100, p_chance=0.5),
    dict(label="n=400, acc=0.58", k=232,n=400, p_chance=0.5),
]

print(f"\n{'情境':>22}  {'準確率':>6} {'95% CI':>18}  {'p 值':>9}  {'sig?':>5}  {'閾值':>8}")
print("-" * 80)
for sc in scenarios:
    k, n, pc = sc["k"], sc["n"], sc["p_chance"]
    acc = k / n
    lo, hi = wilson_ci(k, n)
    pval = stats.binomtest(k, n, pc, alternative="greater").pvalue
    thresh = significance_threshold(n, pc)
    sig = "YES" if pval < 0.05 else "no"
    print(f"{sc['label']:>22}  {acc:>6.2%} [{lo:.3f}, {hi:.3f}]  {pval:>9.4f}  {sig:>5}  {thresh:>8.1%}")

# %% [markdown]
# **關鍵要點：** 小型測試集需要令人驚訝地高準確率才能達到顯著性。
# 始終報告 n，始終報告 CI，
# 並始終與你的 n 對應的*真正*顯著性閾值比較——而不只是樸素的 1/k。

# %% [markdown]
# ---
# ## ⚠️ 一個更隱蔽的陷阱：有效 n 比你的試次數更小
#
# 以上所有內容都假設 n 次試次是**獨立的（independent）**。
# 在 EEG 分類實驗中，它們幾乎從不獨立。
#
# ### 為何 EEG 試次存在自相關（autocorrelated）
#
# 同一次測試中的 EEG 試次共享慢速漂移、疲勞、警覺波動、
# 阻抗漂移以及受試者狀態。試次 i 和試次 i+1 的訊號不是來自同一分布的獨立樣本。
# 實務中這意味著**有效樣本數（effective sample size）** n_eff < n。
#
# 若跨試次的組內相關係數（intraclass correlation，ICC，ρ）為 ρ，
# 且每個「獨立區塊」的平均試次數為 m，則 Kish（1965）的經典公式給出：
#
# $$n_\text{eff} = \frac{n}{1 + (m - 1)\rho}$$
#
# ### 對顯著性閾值的影響
#
# 在二項式檢定中使用 n_eff 取代 n 會提高顯著性閾值。
# 陷阱在於：你可能使用原始試次計數計算得到 p < 0.05，
# 宣告 BCI 結果顯著，但使用 n_eff 的正確計算卻不顯著。
#
# 一個具體例子：

# %%
def effective_n(n_total: int, m_cluster: int, rho: float) -> int:
    """
    在組內相關係數 rho、平均叢集大小 m 下，Kish（1965）的有效樣本數。
    """
    n_eff = n_total / (1 + (m_cluster - 1) * rho)
    return max(1, int(np.floor(n_eff)))


print("組內相關係數對有效 n 與顯著性閾值的影響")
print(f"{'rho':>6}  {'n_raw':>6}  {'n_eff':>6}  {'thresh_raw':>11}  {'thresh_eff':>11}  {'結論'}")
print("-" * 70)

n_raw   = 100
m_block = 10     # 每個獨立區塊 10 個試次（例如一次 2 秒的執行）
obs_acc = 0.62   # 100 次中有 62 次正確
k_obs   = round(obs_acc * n_raw)

for rho in [0.00, 0.05, 0.10, 0.20, 0.35]:
    n_eff   = effective_n(n_raw, m_block, rho)
    thresh_raw = significance_threshold(n_raw, 0.5)
    thresh_eff = significance_threshold(n_eff, 0.5)
    # p 值應以 n_eff 重新推導
    k_eff_scaled = round(obs_acc * n_eff)
    pval_eff = stats.binomtest(k_eff_scaled, n_eff, 0.5, alternative="greater").pvalue
    sig_eff = "significant" if pval_eff < 0.05 else "NOT significant"
    print(f"{rho:>6.2f}  {n_raw:>6}  {n_eff:>6}  {thresh_raw:>11.1%}  {thresh_eff:>11.1%}  {sig_eff}")

# %% [markdown]
# 即使是 ρ = 0.10 的中等組內相關係數，也會將 n_eff 從 100 降至約 52，
# 將顯著性閾值從 ~59% 提高到 ~62% 甚至更高。
# 在樸素計數下看起來顯著的 62% 觀察準確率，
# 一旦考慮相關結構，就不再顯著。
#
# ### 每折（per-fold）的陷阱
#
# 相關的陷阱：某些論文報告準確率「在 10 個交叉驗證折中的 8 個中高於機率水準」，
# 並將每個折視為獨立。但這些折**不是**獨立的——它們共享訓練資料和同一位受試者的大腦。
# 正確的檢定是針對**彙總的折外預測**（pooled out-of-fold predictions）進行
# 一個二項式檢定（n = 測試樣本總數），而不是對每折的 p 值分開檢定。
#
# ### 如何保護自己
# 1. 從你的資料估計 ρ（對每位受試者擬合單向 ANOVA，計算 ICC）。
# 2. 計算 n_eff 並在二項式檢定中使用它。
# 3. 報告原始 n、n_eff、ρ，以及從 n_eff 推導的閾值。
# 4. 若使用交叉驗證，彙總折外預測並對彙總結果進行一次
#    二項式檢定——絕不要對每折的 p 值取平均。
#
# 這些修正在已發表的 BCI 論文中很少被執行，
# 這意味著文獻中許多「顯著」結果在所聲明的 α 水準下很可能是偽陽性（false positives）。
