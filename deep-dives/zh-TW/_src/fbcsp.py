# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 濾波器組共空間模式（Filter-Bank CSP，FBCSP）
#
# 贏得 BCI 競賽的經典單頻段 CSP 延伸版（Ang et al. 2008）。
#
# > **先備知識：** 主要章節 06 與 07。
# > **難度：** 進階 ★★★★☆
# > **運動想像（motor imagery）的經典強效基線。**

# %% [markdown]
# ## 0 — 啟動程序（Bootstrap）

# %%
import sys
import os
from pathlib import Path

# 向上搜尋 src/neuro101 的穩健迴圈
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path.cwd()
    for _parent in [_here, *_here.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101" / "__init__.py").exists():
            sys.path.insert(0, str(_candidate))
            break
    import neuro101  # noqa: F401

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 無頭模式 — 在 CI 與 nbconvert 中皆安全
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mne
mne.set_log_level("ERROR")

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuro101.io import load_bnci_2a_epochs
from neuro101.preprocessing import bandpass_filter
from neuro101.features import make_csp
from neuro101.eval import make_block_split

rng = np.random.default_rng(42)
np.random.seed(42)

SMOKE = os.environ.get("NEURO101_SMOKE") == "1"
N_SUBJECTS = 2 if SMOKE else 3   # 參與分析的受試者數（smoke 模式：2 位以縮短執行時間）
N_SPLITS   = 3 if SMOKE else 5   # 每位受試者的區塊交叉驗證（block-CV）折數
N_SEEDS    = 1 if SMOKE else 3   # 用於變異量估計的隨機種子數

print(f"SMOKE={SMOKE}  N_SUBJECTS={N_SUBJECTS}  N_SPLITS={N_SPLITS}  N_SEEDS={N_SEEDS}")

# %% [markdown]
# ---
# ## 1 — 動機：為何單一頻段效果不佳
#
# 經典的共空間模式（CSP，Common Spatial Patterns）通常套用在單一寬頻段（例如 8–30 Hz），
# 因為這涵蓋了運動想像期間被抑制的 mu（~8–12 Hz）與 beta（~13–30 Hz）節律
# （事件相關去同步，ERD，event-related desynchronisation）。
#
# **問題：** 每個人的 ERD 峰值頻率因以下因素而有顯著差異：
# * **受試者** — 某人的 mu 峰值可能在 9 Hz，另一人在 11 Hz，
# * **任務** — 手部想像偏重 mu + 低 beta；腳部想像往往向高頻移動，
# * **場次** — 疲勞與注意力會改變頻譜輪廓。
#
# 單一 8–30 Hz 頻段將所有子頻段混為一談。在此寬頻範圍調整的 CSP 空間濾波器
# 會將有用與無用的頻率內容混合，稀釋了具辨別力的訊號。
# **濾波器組 CSP**（FBCSP；Ang et al., 2008, *IJCNN*）透過以下步驟解決此問題：
#
# 1. 將試次各自帶通濾波（bandpass filtering）至多個獨立的窄子頻段，
# 2. 在每個子頻段內各自執行 CSP，
# 3. 將所有頻段的對數方差（log-variance）特徵串接成一個寬特徵向量，
# 4. 使用**互資訊特徵選取**（mutual-information feature selection）保留最具資訊量的
#    頻段×成分組合，再
# 5. 以 LDA（線性判別分析）進行分類。
#
# 關鍵洞見在於步驟 (4) 的特徵選取是資料驅動的：若某受試者的辨別節律完全在 8–12 Hz，
# 選取器將自動提升這些成分的權重並忽略其餘部分。

# %% [markdown]
# ### 子頻段配置
#
# 我們使用七個 4 Hz 寬的子頻段，涵蓋 4–32 Hz，跨越運動想像效果所在的
# theta（θ）、mu（μ）與 beta（β）範圍。

# %%
# 子頻段：(低頻_Hz, 高頻_Hz) 的元組列表
SUBBANDS = [
    (4,  8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
]
BAND_LABELS = [f"{lo}-{hi} Hz" for lo, hi in SUBBANDS]
N_BANDS = len(SUBBANDS)
print(f"子頻段（共 {N_BANDS} 個）：{BAND_LABELS}")

# %% [markdown]
# ---
# ## 2 — FBCSP 實作
#
# 我們將 FBCSP 建構為 **sklearn 相容的轉換器（transformer）**，
# 使其在放入交叉驗證迴圈時完全只在訓練折（training fold）上擬合——不發生資料洩漏。

# %%
class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """濾波器組共空間模式（Filter-Bank Common Spatial Patterns）轉換器。

    將輸入帶通濾波至 ``subbands`` 各子頻段，在訓練資料上為每個頻段各自擬合
    一個獨立的 CSP，然後將所有頻段的對數方差特徵串接成每個試次的單一特徵向量。

    Parameters
    ----------
    subbands : list of (float, float)
        各子頻段的頻率範圍（Hz）。
    sfreq : float
        輸入試次的取樣率（Hz）。
    n_components : int
        每個頻段的 CSP 成分數（最高 + 最低各 n_components//2）。
    """

    def __init__(self, subbands=None, sfreq=250.0, n_components=4):
        self.subbands = subbands or SUBBANDS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y):
        """在每個子頻段上各自對 (X, y) 擬合一個 CSP。"""
        self.csps_ = []
        for low, high in self.subbands:
            Xf = bandpass_filter(X, self.sfreq, low, high)
            csp = make_csp(n_components=self.n_components)
            csp.fit(Xf, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        """回傳所有頻段串接後的對數方差特徵。

        Returns
        -------
        np.ndarray，形狀為 (n_trials, n_bands * n_components)
        """
        feats = []
        for (low, high), csp in zip(self.subbands, self.csps_):
            Xf = bandpass_filter(X, self.sfreq, low, high)
            feats.append(csp.transform(Xf))   # (n_trials, n_components)
        return np.concatenate(feats, axis=1)  # (n_trials, n_bands * n_components)


def make_fbcsp_pipeline(subbands, sfreq, n_components=4, k_best=8):
    """回傳一個防洩漏的 FBCSP → SelectKBest → LDA 管線（pipeline）。

    所有步驟——CSP 擬合、特徵選取、LDA——在 CV 迴圈中都只在訓練折上擬合。
    """
    steps = [
        ("fbcsp",    FBCSPTransformer(subbands=subbands, sfreq=sfreq,
                                      n_components=n_components)),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=k_best)),
        ("lda",      LDA()),
    ]
    return Pipeline(steps)


def make_single_band_csp_pipeline(sfreq, fmin=8.0, fmax=30.0, n_components=4):
    """基線（baseline）：單頻段 8-30 Hz CSP → LDA 管線。"""
    from neuro101.features import make_csp as _make_csp

    class SingleBandCSP(BaseEstimator, TransformerMixin):
        def __init__(self, sfreq=250.0, fmin=8.0, fmax=30.0, n_components=4):
            self.sfreq = sfreq
            self.fmin = fmin
            self.fmax = fmax
            self.n_components = n_components

        def fit(self, X, y):
            Xf = bandpass_filter(X, self.sfreq, self.fmin, self.fmax)
            self.csp_ = _make_csp(n_components=self.n_components)
            self.csp_.fit(Xf, y)
            return self

        def transform(self, X):
            Xf = bandpass_filter(X, self.sfreq, self.fmin, self.fmax)
            return self.csp_.transform(Xf)

    return Pipeline([
        ("csp", SingleBandCSP(sfreq=sfreq, fmin=fmin, fmax=fmax,
                              n_components=n_components)),
        ("lda", LDA()),
    ])


print("FBCSP 與單頻段 CSP 管線工廠已定義。")

# %% [markdown]
# ---
# ## 3 — 資料載入
#
# 我們為前幾位受試者載入 BCI IV 2a 資料集（左手 vs 右手運動想像）。
# 載入器回傳寬頻試次；我們將在 FBCSP 轉換器內部自行進行子頻段濾波。
#
# **重要：** 我們以 *原始* 0.5–100 Hz 典範頻段載入，而非常見的 8–30 Hz 預濾波，
# 讓我們的濾波器組能完整存取頻譜。

# %%
print(f"正在載入 {N_SUBJECTS} 位受試者的 BCI IV 2a 資料 …")
X_all, y_all, subjects_all = load_bnci_2a_epochs(
    n_subjects=N_SUBJECTS,
    fmin=0.5,     # 最小預濾波——讓我們的濾波器組來做主要工作
    fmax=40.0,
    tmin=0.5,
    tmax=2.5,
)
sfreq = 250.0   # BCI IV 2a 取樣率
print(f"X 形狀：{X_all.shape}  |  類別：{np.unique(y_all)}  |  取樣率：{sfreq} Hz")
print(f"各受試者試次數：{ {s: int((subjects_all == s).sum()) for s in np.unique(subjects_all)} }")

# %% [markdown]
# ---
# ## 4 — 受試者內（within-subject）區塊交叉驗證
#
# 我們使用 `make_block_split` 對每位受試者評估兩條管線，
# 該函式遵守時間順序並防止相鄰試次洩漏。
# `SelectKBest` 的特徵選取與 CSP 濾波器一起在每個折內擬合——
# 這是防止洩漏的關鍵設計選擇。

# %%
def run_cv_for_subject(X_subj, y_subj, sfreq, n_splits, n_seeds):
    """回傳 FBCSP 與單頻段 CSP 的每折準確率陣列。

    Returns
    -------
    fb_accs : list of float    FBCSP 每折準確率
    sb_accs : list of float    單頻段 CSP 每折準確率
    """
    # 要選取的特徵數：每頻段 2 個（CSP 上下各一）× n_bands
    # 或最多取可用特徵總數
    n_total = N_BANDS * 4   # 每個頻段 4 個成分
    k_best  = min(n_total, max(2, N_BANDS * 2))

    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    sb_pipe = make_single_band_csp_pipeline(sfreq, n_components=4)

    fb_accs, sb_accs = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        for tr, te in make_block_split(len(y_subj), n_splits=n_splits):
            # 複製（clone）以確保每折重新開始（不殘留跨折狀態）
            fb = clone(fb_pipe)
            sb = clone(sb_pipe)

            fb.fit(X_subj[tr], y_subj[tr])
            sb.fit(X_subj[tr], y_subj[tr])

            fb_accs.append(fb.score(X_subj[te], y_subj[te]))
            sb_accs.append(sb.score(X_subj[te], y_subj[te]))

    return fb_accs, sb_accs


results = {}
unique_subjects = np.unique(subjects_all)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]
    print(f"  受試者 {s}：{Xs.shape[0]} 個試次，{N_SPLITS} 折 CV × {N_SEEDS} 種子 …")
    fb_accs, sb_accs = run_cv_for_subject(Xs, ys, sfreq,
                                          n_splits=N_SPLITS, n_seeds=N_SEEDS)
    results[s] = {
        "fb_accs": np.array(fb_accs),
        "sb_accs": np.array(sb_accs),
        "fb_mean": float(np.mean(fb_accs)),
        "fb_std":  float(np.std(fb_accs)),
        "sb_mean": float(np.mean(sb_accs)),
        "sb_std":  float(np.std(sb_accs)),
    }
    print(f"    FBCSP：{results[s]['fb_mean']:.3f} ± {results[s]['fb_std']:.3f}  |  "
          f"單頻段 CSP：{results[s]['sb_mean']:.3f} ± {results[s]['sb_std']:.3f}")

# 整體結果（跨受試者與折的匯總）
all_fb = np.concatenate([results[s]["fb_accs"] for s in unique_subjects])
all_sb = np.concatenate([results[s]["sb_accs"] for s in unique_subjects])
print(f"\n整體 FBCSP：       {all_fb.mean():.3f} ± {all_fb.std():.3f}")
print(f"整體單頻段 CSP：   {all_sb.mean():.3f} ± {all_sb.std():.3f}")

# %% [markdown]
# ---
# ## 5 — 視覺化 1：各受試者準確率比較
#
# 下方長條圖顯示每位受試者的 FBCSP 與單頻段 CSP 準確率。
# 誤差棒（error bars）為跨折 × 種子的 ± 1 標準差。

# %%
fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.32
x = np.arange(len(unique_subjects))

fb_means = [results[s]["fb_mean"] for s in unique_subjects]
fb_stds  = [results[s]["fb_std"]  for s in unique_subjects]
sb_means = [results[s]["sb_mean"] for s in unique_subjects]
sb_stds  = [results[s]["sb_std"]  for s in unique_subjects]

bars_fb = ax.bar(x - bar_width / 2, fb_means, bar_width,
                 yerr=fb_stds, capsize=5,
                 color="steelblue", alpha=0.85, label="FBCSP（濾波器組）")
bars_sb = ax.bar(x + bar_width / 2, sb_means, bar_width,
                 yerr=sb_stds, capsize=5,
                 color="tomato", alpha=0.85, label="單頻段 CSP（8–30 Hz）")

ax.axhline(0.5, color="black", linestyle="--", lw=1.5, label="機會水準（50%）")
ax.set_xticks(x)
ax.set_xticklabels([f"受試者 {s}" for s in unique_subjects], fontsize=11)
ax.set_ylabel("準確率（區塊 CV，均值 ± 標準差）", fontsize=11)
ax.set_title("FBCSP vs 單頻段 CSP — BCI IV 2a（左手 vs 右手）\n"
             f"區塊 CV（{N_SPLITS} 折 × {N_SEEDS} 種子）", fontsize=11)
ax.set_ylim(0.35, 1.02)
ax.legend(fontsize=10)

# 在 FBCSP 長條頂端標注差值
for xi, (fm, sm) in enumerate(zip(fb_means, sb_means)):
    delta = fm - sm
    sign  = "+" if delta >= 0 else ""
    ax.text(xi - bar_width / 2, fm + (fb_stds[xi] if fb_stds else 0.02) + 0.025,
            f"{sign}{delta:.2f}", ha="center", fontsize=9, color="steelblue",
            fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/dd_fbcsp_subject_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 1 已儲存。")

# %% [markdown]
# ---
# ## 6 — 視覺化 2：特徵頻段選取輪廓
#
# FBCSP 的一大詮釋優勢在於特徵選取器告訴我們**哪些子頻段對每位受試者最具辨別力**。
# 透過在每位受試者的訓練資料上（此處將所有折合併作為示意性擬合）
# 擬合完整管線，我們可以檢視哪些頻段×成分配對通過了互資訊（mutual-information）閘道。
#
# 下方長條圖顯示**各子頻段被選取的特徵數**（跨受試者加總）。
# mu（8–12 Hz）與低 beta（12–16 Hz）範圍的子頻段應在左手 vs 右手想像中佔主導地位。

# %%
band_selection_counts = np.zeros(N_BANDS, dtype=int)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]

    n_total = N_BANDS * 4
    k_best  = min(n_total, max(2, N_BANDS * 2))

    # 在受試者全部資料上擬合 FBCSP（僅供示意——
    # 受試者內泛化能力由上方區塊 CV 衡量，非此擬合）。
    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    fb_pipe.fit(Xs, ys)

    # 選取器記錄了哪些特徵（0..N_BANDS*4-1）被保留。
    selector = fb_pipe.named_steps["selector"]
    selected_features = selector.get_support(indices=True)
    # 特徵 i 來自頻段 i // n_components
    for feat_idx in selected_features:
        band_idx = feat_idx // 4   # 每個頻段 4 個成分
        band_selection_counts[band_idx] += 1

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左圖：各頻段被選取的特徵數
ax = axes[0]
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, N_BANDS))
bars = ax.bar(range(N_BANDS), band_selection_counts, color=colors,
              edgecolor="black", linewidth=0.8)
ax.set_xticks(range(N_BANDS))
ax.set_xticklabels(BAND_LABELS, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("被選取特徵數（跨受試者）", fontsize=11)
ax.set_title(f"SelectKBest 選擇了哪些子頻段？\n"
             f"（mutual_info_classif，{N_SUBJECTS} 位受試者）", fontsize=11)

# 標注生理學對應頻段
for band_name, band_range in [("mu", (8, 12)), ("beta", (12, 20))]:
    lo, hi = band_range
    for i, (bl, bh) in enumerate(SUBBANDS):
        if bl >= lo and bh <= hi + 4:
            ax.bar(i, band_selection_counts[i], color=colors[i],
                   edgecolor="navy", linewidth=2.0)

# 添加生理學圖例
mu_patch   = mpatches.Patch(edgecolor="navy", facecolor="none",
                              linewidth=2, label="mu / beta 頻段")
ax.legend(handles=[mu_patch], fontsize=9)

for bar, count in zip(bars, band_selection_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            str(count), ha="center", fontsize=11, fontweight="bold")

ax.set_ylim(0, band_selection_counts.max() * 1.3 + 1)

# 右圖：各頻段跨受試者平均互資訊分數
ax2 = axes[1]
mi_band_avg = np.zeros(N_BANDS)
mi_band_counts = np.zeros(N_BANDS, dtype=int)

for s in unique_subjects:
    mask = subjects_all == s
    Xs, ys = X_all[mask], y_all[mask]

    n_total = N_BANDS * 4
    k_best  = min(n_total, max(2, N_BANDS * 2))

    fb_pipe = make_fbcsp_pipeline(SUBBANDS, sfreq, n_components=4, k_best=k_best)
    fb_pipe.fit(Xs, ys)

    selector = fb_pipe.named_steps["selector"]
    mi_scores = selector.scores_  # 形狀：(N_BANDS * n_components,)

    for feat_idx, score in enumerate(mi_scores):
        band_idx = feat_idx // 4
        mi_band_avg[band_idx] += score
        mi_band_counts[band_idx] += 1

mi_band_avg /= mi_band_counts

bars2 = ax2.bar(range(N_BANDS), mi_band_avg, color=colors,
                edgecolor="black", linewidth=0.8)
ax2.set_xticks(range(N_BANDS))
ax2.set_xticklabels(BAND_LABELS, rotation=35, ha="right", fontsize=10)
ax2.set_ylabel("平均互資訊分數（nats）", fontsize=11)
ax2.set_title("各子頻段平均互資訊分數\n"
              "（跨成分與受試者平均）", fontsize=11)
ax2.set_ylim(0, mi_band_avg.max() * 1.35 + 0.01)

for bar, score in zip(bars2, mi_band_avg):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mi_band_avg.max() * 0.02,
             f"{score:.3f}", ha="center", fontsize=9)

plt.suptitle("FBCSP 頻段重要性分析 — mu/beta 生理學特性應浮現",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/dd_fbcsp_band_selection.png", dpi=120, bbox_inches="tight")
plt.show()
print("圖 2 已儲存。")

# %% [markdown]
# **讀圖說明：** 對於左手 vs 右手想像，與 mu（8–12 Hz）和低 beta（12–16 Hz）
# 重疊的子頻段通常累積最多被選取的特徵與最高的互資訊分數，
# 這與已知的 ERD 生理學一致。4–8 Hz（theta）和 28–32 Hz（高 beta）頻段
# 通常資訊量較低，但這取決於受試者——這正是 FBCSP 資料驅動選取設計
# 旨在處理的個體變異性。

# %% [markdown]
# ---
# ## 7 — 結果摘要表

# %%
print("=" * 62)
print(f"{'方法':<25} {'平均準確率':>10} {'標準差':>8}")
print("=" * 62)
for s in unique_subjects:
    r = results[s]
    print(f"  S{s} FBCSP           {r['fb_mean']:>10.3f} {r['fb_std']:>8.3f}")
    print(f"  S{s} Single-band CSP {r['sb_mean']:>10.3f} {r['sb_std']:>8.3f}")
    print()

print("-" * 62)
print(f"  整體 FBCSP        {all_fb.mean():>10.3f} {all_fb.std():>8.3f}")
print(f"  整體單頻段 CSP    {all_sb.mean():>10.3f} {all_sb.std():>8.3f}")
print("=" * 62)
print(f"\nFBCSP vs 單頻段 CSP 差值：{all_fb.mean() - all_sb.mean():+.3f}")

# %% [markdown]
# ---
# ## ⚠️ A subtler trap: feature-selection leakage looks just like a clean pipeline
#
# FBCSP 將 N_BANDS × n_components 個來源的特徵串接（例如我們的設定中為 7 × 4 = 28 個特徵）。
# `SelectKBest` 接著使用特徵與標籤之間的互資訊挑選最具資訊量的 *k* 個。
#
# 陷阱在此：**若 `SelectKBest` 在訓練/測試分割之前——哪怕只是一瞬間——
# 對所有資料進行擬合，所選特徵就會攜帶來自測試集的標籤資訊進入訓練。**
# 這是 BCI 論文中最常見且最難被發現的資料洩漏（leakage）模式之一，原因如下：
#
# 1. 程式碼看起來*正確*——CSP 擬合與 LDA 都在管線內。
# 2. 洩漏並非來自對測試資料擬合共變異數矩陣；它來自特徵的*排名*。
#    即使對整個（訓練+測試）集進行單次互資訊計算，也會偏向哪些特徵能存活，
#    因為選取器「看到」了哪些頻段對測試試次有預測能力。
# 3. 偏差與**嘗試的特徵數成正比**：有 28 個候選特徵而 `k=8`，
#    20 個特徵被捨棄。若這 20 個包含測試洩漏訊號，
#    被選取的 8 個就系統性地富含不能泛化的測試模式。
#
# ### 為何如此隱蔽
#
# 樸素的完整性檢查——「我有沒有對測試資料呼叫 `fit`？」——通過了。
# 選取是在任何明確的 `fit` 呼叫之前、根據計算出的分數完成的。
# 正確的檢查是：「任何使用標籤資訊的物件，是否在訓練/測試邊界**之前**
# 觀察到了測試集樣本？」對於 `mutual_info_classif`，若在管線外呼叫，答案是肯定的。
#
# ### 正確模式（本筆記本的做法）
#
# ```python
# # 正確：SelectKBest 在 Pipeline 內；sklearn 每折複製並重新擬合它
# pipeline = Pipeline([
#     ("fbcsp",    FBCSPTransformer(...)),   # 只在訓練集上擬合
#     ("selector", SelectKBest(mutual_info_classif, k=k)),  # 只在訓練集上擬合
#     ("lda",      LDA()),                   # 只在訓練集上擬合
# ])
# # 在 CV 迴圈中，pipeline.fit(X_train, y_train) 只對訓練集擬合所有步驟。
# ```
#
# ### 相關陷阱：以測試集表現選取頻段
#
# 更隱蔽的變體：研究者手動檢視哪個子頻段在網格搜尋中產生最高的*測試集*準確率，
# 然後將該準確率作為「最佳頻段 FBCSP」的分數回報。
# 這是樂觀偏差（optimism bias）（從測試表現進行選取），而非嚴格意義上的洩漏，
# 但結果同樣無效——所回報的數字回答的是「當我們知道答案時 FBCSP 有多好？」
# 而非「FBCSP 對新資料有多好？」。
#
# **規則：** 任何接觸標籤與資料的操作都必須在 CV 折內，
# 並且只在訓練分區上擬合。
