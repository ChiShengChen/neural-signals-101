# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 模型比較的統計嚴謹性（Statistical Rigor for Model Comparison）
#
# 主章節 11 所介紹直覺的正式數學框架。
#
# > **先備知識（Prerequisites）：** 主章節 11。
# > **難度（Level）：** 進階 ★★★★☆
# > **不受 5 分鐘 CPU 預算限制。**

# %%
# --- 啟動程序：確保 neuro101 可以被匯入 ---
import sys, os
from pathlib import Path
_p = Path.cwd()
for _ in range(6):
    if (_p / "src" / "neuro101").exists():
        sys.path.insert(0, str(_p / "src")); break
    _p = _p.parent

import warnings
import numpy as np
import matplotlib
# 無頭後端（Headless backend）供 CI / smoke 執行使用
if os.environ.get("NEURO101_SMOKE") == "1" or os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 在無頭模式下，抑制 plt.show() 的非互動後端警告
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.base import clone

rng = np.random.default_rng(42)

SMOKE = os.environ.get("NEURO101_SMOKE") == "1"
# 在 smoke 模式下減少迭代次數，使 CI 快速完成
N_EXPERIMENTS = 300  if SMOKE else 3_000   # 外層蒙地卡羅（Monte Carlo）重複次數
K_FOLDS       = 5
N_SAMPLES     = 150                         # 每個合成資料集的樣本數
print(f"SMOKE={SMOKE}  N_EXPERIMENTS={N_EXPERIMENTS}  K_FOLDS={K_FOLDS}  N_SAMPLES={N_SAMPLES}")

# %% [markdown]
# ---
# ## 第一節 — 為什麼天真的 k 折配對 t 檢定（k-fold paired t-test）過於寬鬆
#
# ### 理論問題
#
# 使用 k 折交叉驗證（k-fold cross-validation）比較兩個模型時，標準做法是：
#
# 1. 計算 k 個折疊的逐折差異（per-fold differences）$d_i = \text{acc}_B^{(i)} - \text{acc}_A^{(i)}$。
# 2. 對 $H_0: \mu_d = 0$ 執行單樣本 t 檢定。t 統計量（t-statistic）為
#    $t = \bar{d} \,/\, (s_d / \sqrt{k})$。
# 3. 若 $p < 0.05$ 則拒絕 $H_0$。
#
# 這是**錯誤的**，因為 t 檢定要求 $d_i$ 彼此**獨立（independent）**。
# 在 k 折交叉驗證中，兩個折疊共享 $(k-2)/(k-1)$ 的訓練樣本。
# 這種正相關（positive correlation）導致 $s_d^2$ **低估**了 $\bar{d}$ 的真實變異數，
# 分母縮小，$|t|$ 被放大，檢定拒絕的頻率遠超過設定的 $\alpha = 0.05$。
#
# ### 模擬策略（Simulation strategy）
#
# 為了測量實際第一型錯誤率（Type-I error），我們需要**真實差異恰好為零**的實驗。
# 最乾淨的方式是直接從多變量常態分布（multivariate normal）模擬折疊分數，
# 該分布編碼了 k 折交叉驗證已知的相關結構，再執行檢定。
#
# Nadeau & Bengio (2003) 顯示，在 $H_0$ 下折疊分數之間的實驗內共變異數（intra-experiment covariance）
# 約為 $\rho \sigma^2$，其中
# $\rho = n_{\text{test}} / n_{\text{train}}$，$\sigma^2$ 為逐折分數變異數。
# 我們直接從此分布進行模擬。

# %%
def simulate_correlated_fold_diffs(k, n_train, n_test, sigma=0.12, rng_s=None):
    """在 H0 下模擬 k 個交叉驗證折疊差異（真實均值 = 0）。

    折疊分數存在相關性：任意兩個折疊共享約 (k-2)/(k-1) 的訓練資料，
    產生正共變異數 rho * sigma^2。
    我們以 Nadeau-Bengio 共變異數矩陣的多變量常態分布建模。

    Parameters
    ----------
    k : int        折疊數。
    n_train : int  每折的訓練集大小。
    n_test : int   每折的測試集大小。
    sigma : float  單一折疊分數的標準差（典型值約 0.10-0.15）。
    rng_s : np.random.Generator  隨機狀態。
    """
    if rng_s is None:
        rng_s = np.random.default_rng()
    rho = n_test / n_train          # N&B 重疊比率（overlap ratio）
    # 共變異數矩陣：對角線為 sigma^2，非對角線為 rho * sigma^2
    cov = sigma**2 * (rho * np.ones((k, k)) + (1 - rho) * np.eye(k))
    mean = np.zeros(k)
    return rng_s.multivariate_normal(mean, cov)


n_test_sim  = N_SAMPLES // K_FOLDS
n_train_sim = N_SAMPLES - n_test_sim
sigma_sim   = 0.12       # BCI 設定中單一折疊準確率標準差的典型值

print(f"模擬設定：k={K_FOLDS}, n_train={n_train_sim}, n_test={n_test_sim}")
print(f"實驗內相關係數 rho = n_test/n_train = {n_test_sim/n_train_sim:.4f}")

# --- 蒙地卡羅（Monte Carlo）：計算每個檢定在 H0 下的拒絕次數 ---
print(f"\n執行 {N_EXPERIMENTS} 次蒙地卡羅實驗…")
naive_rejects     = []
corrected_rejects = []
pvals_naive_all   = []
pvals_corr_all    = []

for exp_i in range(N_EXPERIMENTS):
    exp_rng = np.random.default_rng(rng.integers(0, 2**31))
    diffs = simulate_correlated_fold_diffs(
        K_FOLDS, n_train_sim, n_test_sim, sigma=sigma_sim, rng_s=exp_rng
    )

    # --- 天真配對 t 檢定（錯誤做法）---
    _, p_naive = stats.ttest_1samp(diffs, popmean=0)
    naive_rejects.append(p_naive < 0.05)
    pvals_naive_all.append(p_naive)

    # --- Nadeau-Bengio 修正變異數 ---
    n      = len(diffs)
    mean_d = diffs.mean()
    var_d  = diffs.var(ddof=1)
    rho_cv = n_test_sim / n_train_sim
    corrected_var = var_d * (1.0 / n + rho_cv)
    if corrected_var > 0:
        t_corr = mean_d / np.sqrt(corrected_var)
        p_corr = 2 * stats.t.sf(abs(t_corr), df=n - 1)
    else:
        p_corr = 1.0
    corrected_rejects.append(p_corr < 0.05)
    pvals_corr_all.append(p_corr)

naive_type1     = float(np.mean(naive_rejects))
corrected_type1 = float(np.mean(corrected_rejects))
pvals_naive_all = np.array(pvals_naive_all)
pvals_corr_all  = np.array(pvals_corr_all)

print(f"\n天真配對 t 檢定   第一型錯誤率：{naive_type1:.3f}  （名目值 0.05）")
print(f"修正版（N&B 2003）第一型錯誤率：{corrected_type1:.3f}  （名目值 0.05）")
print(f"\n天真檢定拒絕的頻率是應有水準的 {naive_type1/0.05:.1f} 倍。")

# %% [markdown]
# **解讀結果：** 兩個檢定看到的折疊差異其真實均值恰好為零。每次拒絕都是偽陽性（false positive）。
# 天真檢定觸發的頻率遠超過名目 5%，因為它忽略了折疊分數之間的正相關。
# 修正版檢定的拒絕率則接近 5%。

# %%
# --- 視覺化 1：第一型錯誤率長條圖 + p 值直方圖 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：實際第一型錯誤率的長條圖
ax = axes[0]
labels = ["天真\n配對 t 檢定", "Nadeau-Bengio\n修正版"]
rates  = [naive_type1, corrected_type1]
colors = ["tomato", "steelblue"]
bars = ax.bar(labels, rates, color=colors, width=0.42, edgecolor="black", linewidth=1.2)
ax.axhline(0.05, color="black", linestyle="--", lw=2.5, label="名目 α = 0.05")
ax.set_ylim(0, max(0.40, naive_type1 * 1.45))
ax.set_ylabel("實際第一型錯誤率", fontsize=12)
ax.set_title(f"H₀ 為真時的偽陽性率\n"
             f"（{N_EXPERIMENTS:,} 次模擬實驗，{K_FOLDS} 折交叉驗證，"
             f"ρ = n_test/n_train = {n_test_sim}/{n_train_sim}）",
             fontsize=10)
ax.legend(fontsize=11)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2,
            rate + 0.006, f"{rate:.3f}",
            ha="center", fontsize=14, fontweight="bold")
ax.text(0.97, 0.93, "← 理想值 (0.05)", transform=ax.transAxes,
        ha="right", fontsize=10, color="black", style="italic")

# 右：p 值直方圖（在 H0 下應為 Uniform[0,1] → 平坦）
ax2 = axes[1]
bins = np.linspace(0, 1, 21)
exp_count = len(pvals_naive_all) / 20          # Uniform 分布下的期望長條高度
ax2.hist(pvals_naive_all, bins=bins, alpha=0.68, color="tomato",
         label=f"天真版  (α̂={naive_type1:.3f})", edgecolor="white")
ax2.hist(pvals_corr_all,  bins=bins, alpha=0.68, color="steelblue",
         label=f"修正版 (α̂={corrected_type1:.3f})", edgecolor="white")
ax2.axhline(exp_count, color="black", linestyle="--", lw=2,
            label="期望值（Uniform[0,1]）")
ax2.axvline(0.05, color="darkorange", lw=2, linestyle=":", label="α = 0.05")
ax2.set_xlabel("p 值", fontsize=12)
ax2.set_ylabel("次數", fontsize=12)
ax2.set_title("H₀ 下的 p 值分布\n"
              "（理想情況 = 平坦直方圖）", fontsize=11)
ax2.legend(fontsize=9)

plt.suptitle("天真 k 折配對 t 檢定過於寬鬆：偽陽性過多",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fig1_type1_error.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 1 已儲存。")

# %% [markdown]
# **p 值直方圖的關鍵型態：** 在 $H_0$ 下，一個校準正確的檢定產生均勻分布（uniformly distributed）的 p 值——
# 即平坦的直方圖。天真版的直方圖嚴重偏向小 p 值（左側長條遠高於期望值），
# 直接導致拒絕率膨脹。修正版的直方圖則接近平坦。

# %% [markdown]
# ---
# ## 第二節 — Nadeau & Bengio (2003) 修正重採樣 t 檢定（Corrected Resampled t-test）
#
# ### 修正方法：變異數多加一項
#
# Nadeau & Bengio (2003) 推導出跨 $n$ 個交叉驗證折疊的均值差異 $\bar{d}$ 的正確變異數估計量：
#
# $$
# \widehat{\mathrm{Var}}[\bar{d}]
#   = \left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2
# $$
#
# 其中 $s_d^2 = \frac{1}{n-1}\sum_i (d_i - \bar{d})^2$ 是折疊差異的普通樣本變異數，
# $n_{\text{test}}$ 是測試折疊大小，$n_{\text{train}}$ 是訓練折疊大小。
#
# 比率 $\rho = n_{\text{test}} / n_{\text{train}}$ 是**重疊修正（overlap correction）**。
# 對於標準 k 折交叉驗證，$\rho = 1/(k-1)$，因此修正後的分母更大，t 統計量被適當地縮小。
#
# $$
# t = \frac{\bar{d}}{\sqrt{\left(\frac{1}{n} + \rho\right) s_d^2}}
#   \;\overset{\text{approx}}{\sim}\; t_{n-1}
# $$
#
# **參考文獻：**
# * Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
#   *Machine Learning*, 52(3), 239–281.
# * Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of
#   significance tests for comparing learning algorithms.
#   *PAKDD 2004, LNAI 3056*, 3–12.
# * Dietterich, T. G. (1998). Approximate statistical tests for comparing
#   supervised classification learning algorithms. *Neural Computation*, 10(7).

# %%
def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau-Bengio (2003) 變異數修正配對 t 檢定，用於交叉驗證折疊差異。

    Parameters
    ----------
    diffs : array-like, shape (n_folds,)
        逐折分數差異：score_B[i] - score_A[i]。
    n_train : int or float
        每折的訓練樣本數（不等大小折疊時取平均值）。
    n_test : int or float
        每折的測試樣本數（不等大小折疊時取平均值）。

    Returns
    -------
    t_stat : float
        修正後的 t 統計量。
    p_value : float
        相對於 t_{n-1} 的雙尾 p 值。
    ci_95 : tuple of (float, float)
        真實均值差異的 95% 信賴區間（confidence interval）。

    Notes
    -----
    修正項將 rho = n_test/n_train 加入通常的 1/n 項，
    以補償因訓練集重疊所產生的折疊分數正相關。

    Reference: Nadeau & Bengio (2003), Machine Learning 52(3).
    """
    diffs   = np.asarray(diffs, float)
    n       = len(diffs)
    mean_d  = diffs.mean()
    var_d   = diffs.var(ddof=1)
    rho     = n_test / n_train                        # 重疊比率（overlap ratio）
    corrected_var = var_d * (1.0 / n + rho)           # N&B 公式 (5)
    if corrected_var <= 0:
        return 0.0, 1.0, (mean_d, mean_d)
    se      = np.sqrt(corrected_var)
    t_stat  = mean_d / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    t_crit  = stats.t.ppf(0.975, df=n - 1)
    ci_95   = (mean_d - t_crit * se, mean_d + t_crit * se)
    return t_stat, p_value, ci_95


# --- 實作範例：在合成資料上比較兩個真實模型 ---
rng_demo = np.random.default_rng(99)
n_samples_demo = 200
X_demo, y_demo = make_classification(
    n_samples=n_samples_demo, n_features=20, n_informative=8,
    n_redundant=4, random_state=7
)

pipe_A = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=1.0,  max_iter=300, random_state=0))])
pipe_B = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=0.01, max_iter=300, random_state=0))])

kf_demo  = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
scores_A = cross_val_score(pipe_A, X_demo, y_demo, cv=kf_demo, scoring="accuracy")
scores_B = cross_val_score(pipe_B, X_demo, y_demo, cv=kf_demo, scoring="accuracy")
fold_diffs_demo = scores_B - scores_A

n_test_demo  = n_samples_demo // K_FOLDS
n_train_demo = n_samples_demo - n_test_demo

_, p_naive_demo = stats.ttest_1samp(fold_diffs_demo, popmean=0)
t_corr_demo, p_corr_demo, ci_corr_demo = corrected_resampled_ttest(
    fold_diffs_demo, n_train_demo, n_test_demo
)

print("實作範例：LR C=1.0  vs  LR C=0.01（相同資料集，相同折疊）")
print(f"  逐折差異：{np.round(fold_diffs_demo, 3)}")
print(f"  均值差異 (B - A)：   {fold_diffs_demo.mean():.4f}")
print(f"  天真 t 檢定：        p = {p_naive_demo:.4f}")
print(f"  修正版（N&B）：      p = {p_corr_demo:.4f}   t = {t_corr_demo:.3f}")
print(f"  差異的 95% CI：      [{ci_corr_demo[0]:.4f}, {ci_corr_demo[1]:.4f}]")
print(f"  rho = {n_test_demo}/{n_train_demo} = {n_test_demo/n_train_demo:.4f}")

# %% [markdown]
# 修正後的 p 值比天真版更大（更保守）。
# 95% 信賴區間（confidence interval）告訴你**真實差異的合理範圍**——
# 通常比 p 值更具資訊量，因為它顯示是否排除了任何實際意義上的改善。

# %% [markdown]
# ---
# ## 第三節 — 超參數選擇的巢狀交叉驗證（Nested Cross-validation）
#
# ### 過度樂觀問題（Optimism problem）
#
# 常見的工作流程是：
# 1. 執行 k 折交叉驗證，掃描多個超參數（hyperparameters）。
# 2. 選出最佳配置。
# 3. **回報該最佳配置的交叉驗證分數。**
#
# 這是錯誤的。用來*選擇*配置的交叉驗證分數已被**污染（contaminated）**：
# 你偷看了它才做出選擇，因此它帶有樂觀偏誤（optimistically biased）。
# 「獲勝」的模型可能只是在那些折疊上運氣較好。
#
# **巢狀交叉驗證（Nested CV）** 將模型選擇與模型評估分開：
#
# ```
# ┌──────────────────────── outer fold k ──────────────────────────┐
# │  outer train (k-1 folds)          │   outer test fold k        │
# │  ┌── inner CV (grid search) ──┐   │   ^ held out entirely      │
# │  │  find best hyperparams    │   │   | score reported here     │
# │  └───────────────────────────┘   │   |                         │
# │  refit best config on all train ──────┘                         │
# └─────────────────────────────────────────────────────────────────┘
# ```
#
# 外層測試折疊在超參數選擇過程中**從未**被看到，
# 因此外層折疊分數是泛化效能（generalisation performance）的誠實估計。

# %%
# --- 合成資料集 ---
N_SYNTH      = 300
N_FEATURES   = 30
N_INFORMATIVE = 6

X_nest, y_nest = make_classification(
    n_samples=N_SYNTH, n_features=N_FEATURES,
    n_informative=N_INFORMATIVE, n_redundant=8,
    random_state=17
)

param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
pipe_base  = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=500, random_state=0))])

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# --- 非巢狀（錯誤做法）：在同一個交叉驗證上選擇並回報 ---
gs_nonnested = GridSearchCV(pipe_base, param_grid, cv=outer_cv,
                            scoring="accuracy", refit=True, n_jobs=1)
gs_nonnested.fit(X_nest, y_nest)
score_nonnested  = gs_nonnested.best_score_     # 帶有樂觀偏誤

# --- 巢狀交叉驗證（正確做法）---
nested_scores = []
best_Cs       = []
for train_idx, test_idx in outer_cv.split(X_nest):
    X_tr, X_te = X_nest[train_idx], X_nest[test_idx]
    y_tr, y_te = y_nest[train_idx], y_nest[test_idx]
    gs_inner = GridSearchCV(pipe_base, param_grid, cv=inner_cv,
                            scoring="accuracy", refit=True, n_jobs=1)
    gs_inner.fit(X_tr, y_tr)
    best_Cs.append(gs_inner.best_params_["clf__C"])
    nested_scores.append(gs_inner.score(X_te, y_te))

nested_scores     = np.array(nested_scores)
score_nested_mean = nested_scores.mean()
score_nested_std  = nested_scores.std(ddof=1)

print("單一資料集比較：")
print(f"  非巢狀（有偏誤）：  {score_nonnested:.4f}  <-- 選擇資訊洩漏至估計值")
print(f"  巢狀交叉驗證：  {score_nested_mean:.4f} ± {score_nested_std:.4f}  <-- 誠實估計")
print(f"  偏誤（Bias）：       {score_nonnested - score_nested_mean:+.4f}")
print(f"  各外層折疊最佳 C：{best_Cs}")

# %%
# --- 跨多個隨機資料集的 Bootstrap 以描述偏誤分布 ---
N_BOOT_NEST = 80 if SMOKE else 250
boot_rng       = np.random.default_rng(55)
boot_nonnested = []
boot_nested    = []

for _ in range(N_BOOT_NEST):
    seed_i = int(boot_rng.integers(0, 2**31))
    Xb, yb = make_classification(
        n_samples=N_SYNTH, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, n_redundant=8,
        random_state=seed_i
    )
    cv_b       = KFold(n_splits=5, shuffle=True, random_state=seed_i)
    cv_inner_b = KFold(n_splits=3, shuffle=True, random_state=0)

    # 非巢狀
    gs_nn = GridSearchCV(pipe_base, param_grid, cv=cv_b,
                         scoring="accuracy", refit=True, n_jobs=1)
    gs_nn.fit(Xb, yb)
    boot_nonnested.append(gs_nn.best_score_)

    # 巢狀
    fold_sc = []
    for tr, te in cv_b.split(Xb):
        gs_in = GridSearchCV(pipe_base, param_grid, cv=cv_inner_b,
                             scoring="accuracy", refit=True, n_jobs=1)
        gs_in.fit(Xb[tr], yb[tr])
        fold_sc.append(gs_in.score(Xb[te], yb[te]))
    boot_nested.append(float(np.mean(fold_sc)))

boot_nonnested = np.array(boot_nonnested)
boot_nested    = np.array(boot_nested)
bias_per_run   = boot_nonnested - boot_nested

# --- 視覺化 2：非巢狀 vs 巢狀分布 ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
lo = min(boot_nonnested.min(), boot_nested.min()) - 0.01
hi = max(boot_nonnested.max(), boot_nested.max()) + 0.01
bins_nest = np.linspace(lo, hi, 26)
ax.hist(boot_nonnested, bins=bins_nest, alpha=0.70, color="tomato",
        label=f"非巢狀  均值={boot_nonnested.mean():.3f}", edgecolor="white")
ax.hist(boot_nested,    bins=bins_nest, alpha=0.70, color="steelblue",
        label=f"巢狀交叉驗證  均值={boot_nested.mean():.3f}",    edgecolor="white")
ax.axvline(boot_nonnested.mean(), color="tomato",    lw=2.5, linestyle="--")
ax.axvline(boot_nested.mean(),    color="steelblue", lw=2.5, linestyle="--")
ax.set_xlabel("回報準確率", fontsize=12)
ax.set_ylabel("次數", fontsize=12)
ax.set_title(f"{N_BOOT_NEST} 個隨機資料集的分數分布\n"
             f"均值偏誤 = {bias_per_run.mean():.4f}", fontsize=11)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.hist(bias_per_run, bins=20, color="darkorchid", edgecolor="white", alpha=0.85)
ax2.axvline(0, color="black", lw=1.5, linestyle="--", label="零偏誤")
ax2.axvline(bias_per_run.mean(), color="crimson", lw=2.5,
            label=f"均值偏誤 = {bias_per_run.mean():.4f}")
pct_pos = (bias_per_run > 0).mean()
ax2.set_xlabel("非巢狀分數減去巢狀分數", fontsize=12)
ax2.set_ylabel("次數", fontsize=12)
ax2.set_title(f"每次實驗的樂觀偏誤\n"
              f"{pct_pos:.0%} 的執行次數：非巢狀 > 巢狀", fontsize=11)
ax2.legend(fontsize=10)

plt.suptitle("非巢狀交叉驗證帶有樂觀偏誤；巢狀交叉驗證給出誠實的估計",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fig2_nested_vs_nonnested.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 2 已儲存。")
print(f"\n跨 {N_BOOT_NEST} 個資料集的摘要：")
print(f"  非巢狀均值 ± 標準差：  {boot_nonnested.mean():.4f} ± {boot_nonnested.std():.4f}")
print(f"  巢狀交叉驗證均值 ± 標準差：  {boot_nested.mean():.4f} ± {boot_nested.std():.4f}")
print(f"  均值樂觀偏誤：     {bias_per_run.mean():+.4f}")
print(f"  比例：非巢狀 > 巢狀：  {pct_pos:.1%}")

# %% [markdown]
# 非巢狀分布（紅色）相對於巢狀分布（藍色）持續偏右。
# 樂觀偏誤是系統性的：在大多數實驗中，非巢狀分數高估了模型在真實新資料上的實際表現。

# %% [markdown]
# ---
# ## 第四節 — 實務建議
#
# | 建議 | 理由 |
# |---|---|
# | 使用 **Nadeau-Bengio 修正 t 檢定** | 天真版檢定過於寬鬆；NB 修正補償了訓練集重疊 |
# | 回報**效果量（Cohen's d）**和差異的 **95% CI** | p 值不說明大小；微小但「顯著」的改善在實務上可能無關緊要 |
# | 調整超參數時使用**巢狀交叉驗證** | 非巢狀交叉驗證將選擇資訊洩漏至回報分數 |
# | 測試多個配置時修正**多重比較（multiple comparisons）** | 以 α = 0.05 測試 20 個配置，即使全局虛無假說為真，也預期約有 1 個偽陽性 |
# | 使用**受試者作為複製單位**，而非折疊 | 同一受試者的折疊分數相關；見下方更細微的陷阱 |

# %%
# --- 效果量：配對折疊差異的 Cohen's d ---
def cohens_d_paired(diffs):
    """配對差異的 Cohen's d（效果量 = 均值/標準差）。"""
    return diffs.mean() / diffs.std(ddof=1)

d_demo = cohens_d_paired(fold_diffs_demo)
magnitude = ("可忽略" if abs(d_demo) < 0.2 else
             "小"      if abs(d_demo) < 0.5 else
             "中"      if abs(d_demo) < 0.8 else "大")
print("實作範例的效果量（LR C=1.0 vs C=0.01）：")
print(f"  Cohen's d = {d_demo:.3f}  ({magnitude})")
print(f"  均值差異 = {fold_diffs_demo.mean():.4f}")
print(f"  修正版 95% CI: [{ci_corr_demo[0]:.4f}, {ci_corr_demo[1]:.4f}]")
ci_contains_zero = ci_corr_demo[0] < 0 < ci_corr_demo[1]
print(f"  CI 包含零：{ci_contains_zero}  "
      f"{'→ 在 5% 水準下無法拒絕 H0' if ci_contains_zero else '→ 拒絕 H0'}")

# --- 多重比較：家族誤差率（FWER）膨脹 ---
k_configs = np.arange(1, 21)
fwer = 1 - (1 - 0.05) ** k_configs

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_configs, fwer, "o-", color="tomato", lw=2.5, markersize=6,
        label="FWER = 1 − 0.95^k")
ax.axhline(0.05, color="black", linestyle="--", lw=1.5, label="目標 α = 0.05")
ax.axhline(fwer[9], color="darkorange", linestyle=":", lw=1.5,
           label=f"k=10: FWER ≈ {fwer[9]:.2f}")
ax.fill_between(k_configs, fwer, 0.05, alpha=0.18, color="tomato")
ax.set_xlabel("測試的配置數量", fontsize=12)
ax.set_ylabel("家族誤差率（Family-wise error rate）", fontsize=12)
ax.set_title("測試更多配置會膨脹偽陽性率\n"
             "（Bonferroni 門檻 = 0.05 / k）", fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("/tmp/dd_fig3_fwer.png", dpi=120, bbox_inches="tight")
plt.show()

# Holm-Bonferroni 應用於玩具 p 值
toy_pvals = np.array([0.001, 0.04, 0.03, 0.07, 0.002, 0.80, 0.15, 0.045])
order     = np.argsort(toy_pvals)
n_tests   = len(toy_pvals)
holm_thr  = 0.05 / (n_tests - np.arange(n_tests))
print("\nHolm-Bonferroni 修正應用於 8 個玩具 p 值：")
for rank, (idx, thr) in enumerate(zip(order, holm_thr)):
    rej = toy_pvals[idx] <= thr
    print(f"  排名 {rank+1}: p={toy_pvals[idx]:.3f}  Holm 門檻={thr:.4f}  "
          f"{'拒絕' if rej else '保留'}")

# %% [markdown]
# ---
# %% [markdown]
# ## ⚠️ 更細微的陷阱：分析決策分岔路（Garden of Forking Paths）與受試者層級可互換性（Subject-level Exchangeability）
#
# ### 分析決策分岔路（Garden of forking paths）
#
# 以上所有內容假設你執行預先指定的比較集合並應用多重比較修正。
# 有一個更深層的問題，任何修正都無法完全解決：
# **分析決策分岔路（garden of forking paths）**（Gelman & Loken, 2014）。
#
# 在典型的 BCI 研究中，研究者可能嘗試：
# * 3 種前處理流程（頻帶範圍、偽跡排除門檻）
# * 4 種特徵方法（CSP、黎曼、頻帶能量、CNN）
# * 5 種分類器（LDA、SVM-RBF、SVM-linear、LR、EEGNet）
# * 3 種交叉驗證策略（k 折、LOSO、區塊）
#
# 共 $3 \times 4 \times 5 \times 3 = 180$ 種組合。若研究者只回報 p 值最小的組合，
# 即使對 180 次測試進行 Bonferroni 修正也是**無效的**——
# 因為嘗試的組合空間未事先登記（pre-registered）。
# 研究者不一定刻意測試所有 180 種；
# 漸進式直覺、捨棄「無效」配置和倖存者呈現（survivor presentation）在沒有刻意作弊的情況下達到同樣效果。
# 這是普通的科學過程，卻使頻率論保證失效。
#
# **修正的 t 檢定和巢狀交叉驗證能保護你免於你*聲明*的比較所帶來的問題。
# 它們無法保護你免於你*忘記自己做過*的比較。**
#
# ### 受試者層級可互換性（Subject-level exchangeability，更細微）
#
# 即使仔細進行多重比較修正和使用修正的 t 檢定，
# 細則中仍隱藏著一個結構性假設：
# **檢定中所用單位的可互換性（exchangeability of the units）**。
#
# 當你對單一受試者的多個療程資料集執行 k 折交叉驗證，
# 該檢定將 k 個折疊分數視為 $k$ 個近似獨立的觀測值。
# 在**多受試者**研究中，真正的複製單位是**受試者（subject）**，而非折疊。
#
# 具體而言：9 名受試者 × 5 折 = 45 個折疊分數。
# 對所有 45 個值執行修正的 t 檢定，等同於將它們視為 45 個近似獨立的觀測值。
# 但它們並非如此：受試者 1 的 5 個折疊共享相同的大腦、相同的電極位置、
# 相同的記錄療程，以及相同的特異性雜訊來源。
# 其有效自由度（effective degrees of freedom）接近 9 而非 45。
#
# **因應方法：** 先匯總至受試者層級（每位受試者的平均準確率），
# 再對這 9 個值執行配對檢定。更好的做法是將受試者建模為隨機效應（random effect）。
# 修正的 t 檢定「正確」的前提是相對於你輸入它的可互換性結構——
# 正確理解該結構是分析者的責任，沒有任何公式能修復統計模型與資料生成過程之間的不匹配。

# %%
# --- 說明：天真地匯總 45 個折疊分數 vs 9 個受試者均值 ---
rng_subj = np.random.default_rng(13)

n_subjects_illus  = 9
n_folds_per_subj  = 5
# 每位受試者的真實效應來自 N(0, 0.04) — 全局虛無假說為真
subject_true_diff = rng_subj.normal(0.0, 0.04, size=n_subjects_illus)

fold_diffs_flat    = []
subject_mean_diffs = []
for s in range(n_subjects_illus):
    # 受試者內折疊：透過共享受試者截距（subject intercept）相關聯
    folds_s = subject_true_diff[s] + rng_subj.normal(0, 0.02, size=n_folds_per_subj)
    fold_diffs_flat.extend(folds_s.tolist())
    subject_mean_diffs.append(folds_s.mean())

fold_diffs_flat    = np.array(fold_diffs_flat)
subject_mean_diffs = np.array(subject_mean_diffs)

n_train_illus = 80
n_test_illus  = 20

# 檢定 1：對所有 45 個折疊分數執行天真 t 檢定 — 錯誤（自由度膨脹）
_, p_naive_45 = stats.ttest_1samp(fold_diffs_flat, popmean=0)

# 檢定 2：對所有 45 個折疊分數執行 N&B 修正 — 仍然錯誤（錯誤的分析單位）
t_nb45, p_nb45, ci_nb45 = corrected_resampled_ttest(
    fold_diffs_flat, n_train_illus, n_test_illus
)

# 檢定 3：正確做法 — 先匯總至 9 個受試者均值
_, p_subj = stats.ttest_1samp(subject_mean_diffs, popmean=0)
t_nb9, p_nb9, ci_nb9 = corrected_resampled_ttest(
    subject_mean_diffs, n_train_illus, n_test_illus
)

print("可互換性檢查：9 位受試者 × 5 折，真實均值差異 ≈ 0")
print(f"  受試者層級的真實差異：{np.round(subject_true_diff, 3)}")
print(f"  真實總體均值差異：     {subject_true_diff.mean():.4f}")
print()
print(f"  錯誤  — 對 45 個折疊分數執行天真 t 檢定：       p = {p_naive_45:.4f}  (df=44, n=45)")
print(f"  錯誤  — 對 45 個折疊分數執行 N&B 修正：         p = {p_nb45:.4f}  (df=44, n=45)")
print(f"  正確  — 對  9 個受試者均值執行 N&B 修正：       p = {p_nb9:.4f}  (df=8,  n=9)")
print()
print("正確做法使用 n=9 個受試者層級觀測值。")
print("更多觀測值 ≠ 更多受試者；公式無法修復錯誤的分析單位。")
