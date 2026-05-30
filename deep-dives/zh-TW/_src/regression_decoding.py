# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 迴歸解碼（連續目標）
#
# 主要教學將每個試次分類為離散標籤。真實的神經解碼（neural decoding）
# 通常是**迴歸問題（regression problem）**：從腦電訊號預測游標位置、
# 手部運動學，或連續的清醒度／嗜睡程度。
#
# > **前置條件：** 主課程第 06、08 及 12 章。
# > **難度：** 進階 ★★★★☆
# > **所有主章節都在分類；真實解碼通常是迴歸。**

# %%
# --------------------------------------------------------------------------- #
# 啟動引導 — 無論從儲存庫根目錄或 deep-dives/_src/ 執行，
# 都能找到 neuro101（比主筆記本多一層父目錄）。
# --------------------------------------------------------------------------- #
import sys
import os
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
matplotlib.use("Agg")          # 無頭後端 — 在 nbconvert / CI 下安全執行
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds
from neuro101 import make_block_split
from neuro101.io import load_sleep_edf_epochs
from neuro101.features import bandpower

SMOKE = ds.is_smoke()
RNG   = np.random.default_rng(42)

print(f"neuro101 OK  |  SMOKE={SMOKE}")

# %% [markdown]
# ---
# ## 1  分類與迴歸的差異 — 以及為何評估指標截然不同
#
# ### 當目標為連續值時
#
# | 情境 | 目標 | 錯誤指標 | 正確指標 |
# |---|---|---|---|
# | 睡眠階段預測 | {清醒, N1, N2, N3, 快速動眼期} | — | 準確率、平衡準確率、F1 |
# | 清醒度／嗜睡程度 | [0, 1] 中的連續純量 | **準確率** | **R²、MAE、Pearson r** |
# | 游標位置 | 毫米單位的 x、y | **準確率** | **R²、MAE** |
# | 反應時間 | 毫秒 | **準確率** | **R²、MAE、Spearman r** |
#
# **永遠不要對連續目標使用「準確率」。** 準確率要求預測值完全等於真實值——
# 對於實數值輸出，這永遠不會發生，因此每個樣本都「錯誤」，準確率始終為 0%。
#
# ### 你需要的三種迴歸指標
#
# **R² — 決定係數（coefficient of determination）**
#
# $$R^2 = 1 - \frac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}$$
#
# 解讀方式：可解釋的變異比例。R² = 1 表示完美；R² = 0 表示模型不比
# 永遠預測平均值更好；**R² 可以是負數**（比平均值還差——請見結尾的細微陷阱）。
#
# **MAE — 平均絕對誤差（mean absolute error）**
#
# $$\text{MAE} = \frac{1}{n}\sum |y_i - \hat y_i|$$
#
# 單位與目標相同；易於解讀。與 R² 不同，它不依賴目標變異量，
# 因此可用物理單位告訴你原始預測誤差。
#
# **Pearson / Spearman 相關**
#
# 衡量預測值與真實值的*排序*是否一致。模型可能具有高相關但校準（calibration）很差：
# 若 $\hat y = 2 y - 1$（斜率和截距都錯），Pearson $r = 1$，但 $R^2 \ll 1$，
# 且 MAE 很大。務必同時報告相關係數**與** R²，並檢查散佈圖。

# %%
# --------------------------------------------------------------------------- #
# 用玩具範例呈現關鍵指標之間的關係。
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("三種迴歸看似在某項指標表現良好、但在另一項卻失敗的情況",
             fontsize=11, fontweight="bold")

rng_toy = np.random.default_rng(7)
y_true  = np.linspace(0, 1, 80) + rng_toy.normal(0, 0.05, 80)

# 圖板 A：校準良好的模型
y_good = y_true + rng_toy.normal(0, 0.12, 80)
ax = axes[0]
ax.scatter(y_true, y_good, s=12, alpha=0.6, color="steelblue")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_g = r2_score(y_true, y_good)
r_g, _ = pearsonr(y_true, y_good)
ax.set_title(f"A) 校準良好\nR²={r2_g:.2f}  r={r_g:.2f}", fontsize=9)
ax.set_xlabel("真實值"); ax.set_ylabel("預測值"); ax.legend(fontsize=8)

# 圖板 B：高相關但斜率/截距錯誤（校準差）
y_recal = 0.4 * y_true + 0.35 + rng_toy.normal(0, 0.04, 80)
ax = axes[1]
ax.scatter(y_true, y_recal, s=12, alpha=0.6, color="darkorange")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_r = r2_score(y_true, y_recal)
r_r, _ = pearsonr(y_true, y_recal)
ax.set_title(f"B) 高 r，校準差\nR²={r2_r:.2f}  r={r_r:.2f}", fontsize=9)
ax.set_xlabel("真實值"); ax.legend(fontsize=8)

# 圖板 C：零相關，但因目標範圍而 R² 非負
y_rand = rng_toy.uniform(y_true.min(), y_true.max(), 80)
ax = axes[2]
ax.scatter(y_true, y_rand, s=12, alpha=0.6, color="crimson")
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        "k--", lw=1.5, label="identity")
r2_rn = r2_score(y_true, y_rand)
r_rn, _ = pearsonr(y_true, y_rand)
ax.set_title(f"C) 隨機模型\nR²={r2_rn:.2f}  r={r_rn:.2f}", fontsize=9)
ax.set_xlabel("真實值"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/regression_metrics_explainer.png", dpi=110, bbox_inches="tight")
plt.show()
print("圖板 B：Pearson r 很高但 R² 很低——模型的斜率和截距都錯了。")
print("務必對照恆等線（identity line）檢查散佈圖。")

# %% [markdown]
# ---
# ## 2  在真實 EEG 連續目標上進行迴歸
#
# ### 資料集與目標
#
# 我們使用 **Sleep-EDF**：以 100 Hz 取樣的整夜 EEG 錄製，切成 30 秒的時期（epoch）。
# 睡眠階段標籤雖然可用，但我們刻意忽略它們，改而定義一個**連續清醒度代理指標**——
# Fpz-Cz 通道上的對數 Alpha 頻帶功率。
#
# Alpha 功率（8–13 Hz）是皮質清醒度（cortical arousal）的經典標記：
# 在放鬆的清醒狀態下偏高，在深度睡眠中受到抑制。
# 這給了我們一個在整夜之間平滑變化、具有生理意義的連續訊號。
#
# **迴歸任務：** 從同一通道提取的*其他*頻帶功率（delta、theta、beta、gamma）
# 預測 Alpha 對數功率。
#
# ### 為何此設計是誠實的（受試者內、同一通道、其他頻帶）
#
# Delta 功率在深度睡眠中升高；theta 在淺眠和快速動眼期（REM）升高；
# beta/gamma 功率隨睡眠深度增加而下降。這些頻帶攜帶了腦狀態的真實資訊，
# 因此 R² > 0 是合理且有意義的——但關係夠嘈雜，使 R² 遠低於 1。
#
# ### 時間感知分割
#
# 我們使用 `make_block_split`（連續區塊）是因為相鄰的 30 秒時期具有自相關性：
# 一個人在深度睡眠中會連續多個時期維持深度睡眠。隨機分割會讓模型在
# 訓練集和測試集中相鄰時期之間「內插」——進而虛增 R²。

# %%
# --------------------------------------------------------------------------- #
# 載入 Sleep-EDF 並提取頻帶功率特徵。
# --------------------------------------------------------------------------- #
n_subjects_to_load = 1 if SMOKE else 2
print(f"正在載入 Sleep-EDF（{n_subjects_to_load} 位受試者）……")
X_eeg, y_stage, subjects = load_sleep_edf_epochs(n_subjects=n_subjects_to_load)

SFREQ = 100.0  # Hz
print(f"時期數：{X_eeg.shape[0]}  |  通道：EEG Fpz-Cz  |  時期長度：30 秒")

# 提取所有 5 個頻帶的對數頻帶功率特徵。
# feats[:, 0] = delta   feats[:, 1] = theta   feats[:, 2] = alpha
# feats[:, 3] = beta    feats[:, 4] = gamma
feats_all = bandpower(X_eeg, SFREQ)  # (n_epochs, 5)

# 連續目標：對數 alpha 功率
target_alpha = feats_all[:, 2]       # shape (n_epochs,)

# 特徵：除 alpha 以外的所有頻帶
X_feat = feats_all[:, [0, 1, 3, 4]]  # delta, theta, beta, gamma  → (n_epochs, 4)

n_epochs = len(target_alpha)
autocorr_lag1 = float(np.corrcoef(target_alpha[:-1], target_alpha[1:])[0, 1])
print(f"\n目標（對數 alpha 功率）：範圍=[{target_alpha.min():.2f}, {target_alpha.max():.2f}]")
print(f"滯後-1 自相關（lag-1 autocorrelation）：{autocorr_lag1:.3f}  （中等——相鄰時期相似）")

# %%
# --------------------------------------------------------------------------- #
# 使用無資料洩漏的時間感知區塊分割進行嶺迴歸（Ridge regression）。
# --------------------------------------------------------------------------- #
N_SPLITS = 3 if SMOKE else 5

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=1.0)),
])

r2_per_fold   = []
mae_per_fold  = []
pearson_per_fold = []
spearman_per_fold = []
all_true_bs  = []
all_pred_bs  = []
te_indices_bs = []

for fold_idx, (tr, te) in enumerate(make_block_split(n_epochs, n_splits=N_SPLITS)):
    pipe.fit(X_feat[tr], target_alpha[tr])
    preds = pipe.predict(X_feat[te])

    r2_per_fold.append(r2_score(target_alpha[te], preds))
    mae_per_fold.append(mean_absolute_error(target_alpha[te], preds))
    pearson_per_fold.append(pearsonr(target_alpha[te], preds)[0])
    spearman_per_fold.append(spearmanr(target_alpha[te], preds)[0])

    all_true_bs.append(target_alpha[te])
    all_pred_bs.append(preds)
    te_indices_bs.append(te)

all_true_bs_cat = np.concatenate(all_true_bs)
all_pred_bs_cat = np.concatenate(all_pred_bs)
te_idx_cat = np.concatenate(te_indices_bs)
sort_order = np.argsort(te_idx_cat)

print("=== 區塊分割（時間感知）迴歸結果 ===")
print(f"  R²       : {np.mean(r2_per_fold):.3f} ± {np.std(r2_per_fold):.3f}  （各折：{[f'{v:.3f}' for v in r2_per_fold]}）")
print(f"  MAE      : {np.mean(mae_per_fold):.3f} ± {np.std(mae_per_fold):.3f}  （對數功率單位）")
print(f"  Pearson r: {np.mean(pearson_per_fold):.3f} ± {np.std(pearson_per_fold):.3f}")
print(f"  Spearman r: {np.mean(spearman_per_fold):.3f} ± {np.std(spearman_per_fold):.3f}")

# %% [markdown]
# ### 解讀結果
#
# Pearson $r > 0.6$ 表示跨時期的清醒度**排序**已被捕捉到。但請查看 R²：
# 若斜率或截距有誤，R² 將低於 $r^2$。務必同時查看兩者——也務必查看散佈圖。

# %%
# --------------------------------------------------------------------------- #
# 圖一：預測值對真實值（散佈圖 + 恆等線）及時間切片。
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- 散佈圖：預測值對真實值 ---
ax_scatter = fig.add_subplot(gs[0])
ax_scatter.scatter(all_true_bs_cat, all_pred_bs_cat,
                   s=5, alpha=0.25, color="steelblue", rasterized=True)
vmin = min(all_true_bs_cat.min(), all_pred_bs_cat.min())
vmax = max(all_true_bs_cat.max(), all_pred_bs_cat.max())
ax_scatter.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="恆等線（完美預測）")
overall_r2   = r2_score(all_true_bs_cat, all_pred_bs_cat)
overall_r, _ = pearsonr(all_true_bs_cat, all_pred_bs_cat)
ax_scatter.set_xlabel("真實對數 alpha 功率", fontsize=10)
ax_scatter.set_ylabel("預測對數 alpha 功率", fontsize=10)
ax_scatter.set_title(
    f"預測值對真實值（區塊分割）\nR²={overall_r2:.3f}  Pearson r={overall_r:.3f}",
    fontsize=10, fontweight="bold")
ax_scatter.legend(fontsize=8)

# --- 時間切片：前 200 個測試集時期（依時序排列）---
sorted_true = all_true_bs_cat[sort_order]
sorted_pred = all_pred_bs_cat[sort_order]
n_show = min(200, len(sorted_true))

ax_time = fig.add_subplot(gs[1])
epoch_minutes = np.arange(n_show) * 0.5          # 30 秒時期 → 每個 0.5 分鐘
ax_time.plot(epoch_minutes, sorted_true[:n_show],
             lw=1.2, color="steelblue", label="真實值", alpha=0.85)
ax_time.plot(epoch_minutes, sorted_pred[:n_show],
             lw=1.2, color="darkorange", linestyle="--", label="預測值", alpha=0.85)
ax_time.set_xlabel("時間（分鐘）", fontsize=10)
ax_time.set_ylabel("對數 alpha 功率", fontsize=10)
ax_time.set_title("預測值對真實值的時間序列\n（前 200 個測試集時期）", fontsize=10, fontweight="bold")
ax_time.legend(fontsize=8)

plt.savefig("/tmp/regression_scatter_time.png", dpi=110, bbox_inches="tight")
plt.show()
print(f"區塊分割整體  R²={overall_r2:.3f}   Pearson r={overall_r:.3f}")

# %% [markdown]
# ---
# ## 3  迴歸專屬的資料洩漏（leakage）陷阱：自相關目標
#
# ### 為何這比分類中更嚴重
#
# 在第 12 章你看到隨機打亂分割會因相鄰試次相關而虛增分類準確率。
# 對於**具有平滑連續目標的迴歸**，問題要嚴重得多：
#
# * 相鄰的 30 秒時期具有幾乎相同的 alpha 功率（原始 alpha 的滯後-1 自相關 ≈ 0.58，
#   平滑訊號 ≈ 0.99）。
# * 隨機分割將這些相似時期散落到訓練集和測試集中。
# * 當時期 142 的測試時期被訓練時期 141 和 143 包圍
#   （這兩個時期的目標值幾乎相同），模型實際上可以**內插**目標——
#   不是因為它學到了腦狀態關係，而是因為它記住了平滑軌跡。
# * 對平滑目標而言，R² 大幅虛增；對非常平滑的目標，它能讓近乎無用的模型看起來接近完美。
#
# ### 示範
#
# 我們建立一個**高度自相關的嗜睡代理指標**：50 個時期（25 分鐘）
# delta 功率的滑動視窗平均。此訊號的滯後-1 自相關 ≈ 0.998——
# 它在整夜中變化極為緩慢。
#
# 接著我們比較：
# * **錯誤做法** — 隨機 5 折 KFold（打亂）：sklearn 標準預設。
# * **正確做法** — 連續區塊分割（`make_block_split`）。

# %%
# --------------------------------------------------------------------------- #
# 建立平滑嗜睡代理指標（高自相關）。
# --------------------------------------------------------------------------- #
SMOOTH_WINDOW = 50   # 時期數  → 25 分鐘的睡眠結構趨勢
raw_delta      = feats_all[:, 0]   # 對數 delta 功率
smooth_target  = np.convolve(raw_delta,
                              np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                              mode="same")

autocorr_smooth = float(np.corrcoef(smooth_target[:-1], smooth_target[1:])[0, 1])
print(f"平滑嗜睡代理指標：滯後-1 自相關 = {autocorr_smooth:.4f}")
print("（≈1.00 表示相鄰時期幾乎無法區分——資料洩漏的溫床）")

# %%
# --------------------------------------------------------------------------- #
# 錯誤做法：隨機 KFold（典型錯誤）。
# --------------------------------------------------------------------------- #
kf_wrong = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

wrong_r2 = []
for tr, te in kf_wrong.split(X_feat):
    m = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    m.fit(X_feat[tr], smooth_target[tr])
    wrong_r2.append(r2_score(smooth_target[te], m.predict(X_feat[te])))

wrong_mean = float(np.mean(wrong_r2))
wrong_std  = float(np.std(wrong_r2))
print(f"\n錯誤做法（隨機 KFold，shuffle=True）：R² = {wrong_mean:.3f} ± {wrong_std:.3f}")

# --------------------------------------------------------------------------- #
# 正確做法：連續區塊分割（時間感知）。
# --------------------------------------------------------------------------- #
right_r2 = []
for tr, te in make_block_split(n_epochs, n_splits=N_SPLITS):
    m = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    m.fit(X_feat[tr], smooth_target[tr])
    right_r2.append(r2_score(smooth_target[te], m.predict(X_feat[te])))

right_mean = float(np.mean(right_r2))
right_std  = float(np.std(right_r2))
print(f"正確做法（區塊分割，連續）：          R² = {right_mean:.3f} ± {right_std:.3f}")
print(f"\n因資料洩漏造成的虛增：ΔR² = {wrong_mean - right_mean:+.3f}")

# %%
# --------------------------------------------------------------------------- #
# 圖二：錯誤做法 vs 正確做法 R² 長條圖比較。
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左圖：2 條長條對比
ax_bar = axes[0]
labels   = ["錯誤做法\n（隨機 KFold）", "正確做法\n（區塊分割）"]
means    = [wrong_mean, right_mean]
stds     = [wrong_std,  right_std]
colors   = ["#d62728", "#2ca02c"]   # 紅色 = 錯誤，綠色 = 正確
bars = ax_bar.bar(labels, means, yerr=stds, capsize=7,
                  color=colors, alpha=0.82, edgecolor="k", linewidth=0.8,
                  error_kw=dict(elinewidth=1.5))

# 繪製 R²=0 參考線（不優於預測平均值）
ax_bar.axhline(0, color="k", lw=1.0, linestyle="--", label="R²=0  （預測平均值）")
ax_bar.set_ylabel("R²  （5 折平均 ± 標準差）", fontsize=10)
ax_bar.set_title("資料洩漏陷阱：\n平滑目標 × 隨機分割 → R² 虛增",
                 fontsize=10, fontweight="bold")
ax_bar.legend(fontsize=8)

# 在長條上標註數值
for bar, mean_val, std_val in zip(bars, means, stds):
    ypos = max(mean_val, 0) + std_val + 0.01
    ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{mean_val:.3f}±{std_val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

# 右圖：平滑目標的時間序列，說明為何會洩漏
ax_ts = axes[1]
show_n = min(300, n_epochs)
ax_ts.plot(np.arange(show_n) * 0.5, smooth_target[:show_n],
           color="purple", lw=1.5, alpha=0.85)
ax_ts.set_xlabel("時間（分鐘）", fontsize=10)
ax_ts.set_ylabel("平滑 delta 代理指標（對數功率，25 分鐘平均）", fontsize=10)
ax_ts.set_title(f"目標自相關（滯後-1）= {autocorr_smooth:.4f}\n"
                "相鄰時期幾乎相同 → 內插偽裝成學習",
                fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/regression_leakage_bar.png", dpi=110, bbox_inches="tight")
plt.show()

print(f"\n{'='*55}")
print(f"  錯誤做法 R² = {wrong_mean:.3f}  |  正確做法 R² = {right_mean:.3f}")
print(f"  虛增量 ΔR² = {wrong_mean - right_mean:+.3f}")
print(f"{'='*55}")
print("\n結論：隨機 KFold 虛增 R² 是因為相鄰時期共享")
print("幾乎相同的目標值——模型在內插，而非外插。")

# %% [markdown]
# ### 為何連續區塊分割對平滑目標是必要的
#
# 滯後-1 自相關 ≈ 0.998 時，相鄰時期相差極微。隨機分割將時期 100 散落到
# 測試集，而時期 99 和 101 留在訓練集。模型已見過時間 99 和 101 的目標值；
# 它只需輸出它們的平均值——R² 就飆升了。
#
# 連續區塊分割將折疊 *k* 的所有時期保持在一起。模型必須從整夜的一個
# 區塊*外插*到另一個區塊，這真的很困難。誠實的得分甚至可能是**負數**
# （模型表現比預測平均值更差），這是完全有效且具資訊量的結果。
#
# **規則：** 只要你的目標比隨機漫步更平滑——運動學、清醒度、任何生理狀態變數——
# 就使用區塊分割。如果你能計算滯後-1 自相關且其超過 ≈ 0.5，就將其視為平滑目標。

# %% [markdown]
# ---
# ## ⚠️ A subtler trap: R² can be negative — and a small positive R² can still be misleading
#
# 你在上面看到區塊分割的 R² 可以是負數。這是正確且誠實的：
# 這意味著模型*比永遠預測訓練集平均值更差*。但還有第二種更不明顯的失敗模式。
#
# ### 趨勢目標與「常數預測器」的幻覺
#
# 假設你的目標在整個訓練過程中單調漂移——比方說，alpha 功率從清醒穩定
# 下降到睡眠。如果測試折疊在時間上*較晚*且目標朝同方向移動，
# 一個學習單一數值（訓練集平均值）的模型將免費跟隨那個趨勢。
#
# **結果：** 即使是常數預測也能產生小的正 R²，不是因為模型學到了
# 任何關於神經訊號的東西，而是因為目標趨勢在測試區塊期間朝正確方向越過了平均值。
# 這是連續目標版本的「預測多數類」——但較不明顯，因為 R² > 0 *看起來*像成功。
#
# **診斷方法：** 將殘差 $y - \hat y$ 對時間作圖。常數預測器會留下結構化、
# 漂移的殘差模式。真實模型的殘差看起來像白雜訊（white noise）。
# 務必檢查殘差時間序列，而不只是純量 R²。

# %%
# --------------------------------------------------------------------------- #
# 示範趨勢目標／常數預測器的幻覺。
# --------------------------------------------------------------------------- #
n_demo = 300
t_demo = np.arange(n_demo)

# 目標：溫和的線性趨勢 + 雜訊（模擬 alpha 進入睡眠時下降）
trend_target = -0.005 * t_demo + rng_toy.normal(0, 0.15, n_demo)

# 在前半部訓練，在後半部測試——如區塊分割所做的。
split = n_demo // 2
y_train = trend_target[:split]
y_test  = trend_target[split:]

# 「常數預測器」：永遠預測訓練集平均值。
const_pred = np.full(split, y_train.mean())
r2_const   = r2_score(y_test, const_pred)

# 殘差：結構化漂移或白雜訊？
residuals_const = y_test - const_pred

print("=== 趨勢目標／常數預測器的幻覺 ===")
print(f"  常數預測器（訓練平均值 = {y_train.mean():.3f}）在測試折疊上：")
print(f"  R² = {r2_const:.3f}  ← 即使模型忽略所有訊號也可能為正！")
print()
print("  殘差平均值：{:.3f}  （非零 → 結構化漂移，非白雜訊）".format(
    residuals_const.mean()))
print()
print("教訓：單一小正 R² 不能作為學習的證據。")
print("務必將殘差對時間作圖，並與樸素平均基準線比較。")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax0 = axes[0]
ax0.plot(t_demo[:split], y_train,  lw=1, alpha=0.7, color="steelblue", label="訓練目標")
ax0.plot(t_demo[split:], y_test,   lw=1, alpha=0.7, color="darkorange", label="測試目標")
ax0.axhline(y_train.mean(), color="k", lw=1.5, linestyle="--",
            label=f"常數預測 = {y_train.mean():.3f}")
ax0.axvline(split, color="gray", lw=1, linestyle=":")
ax0.set_xlabel("時期索引"); ax0.set_ylabel("目標值")
ax0.set_title(f"趨勢目標：常數預測器\nR² = {r2_const:.3f}  ← 具有誤導性的非負值",
              fontsize=9, fontweight="bold")
ax0.legend(fontsize=7)

ax1 = axes[1]
ax1.plot(t_demo[split:], residuals_const,
         lw=1, color="crimson", alpha=0.8, label="殘差 = 真實值 − 預測值")
ax1.axhline(0, color="k", lw=1, linestyle="--")
ax1.set_xlabel("時期索引（僅測試折疊）"); ax1.set_ylabel("殘差")
ax1.set_title("殘差顯示結構化漂移\n（非白雜訊 → 模型未捕捉到訊號）",
              fontsize=9, fontweight="bold")
ax1.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/regression_subtle_trap.png", dpi=110, bbox_inches="tight")
plt.show()

print("\n經驗法則：務必將你的模型與樸素平均基準線比較。")
print("若 R²（模型）≈ R²（樸素平均預測器），你什麼也沒學到。")
