# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深入探討 — 黎曼方法（Riemannian Methods）於小資料集的應用
#
# 為何共變異數（covariance）管線在僅有少量標記 EEG 試驗時仍具競爭力——
# 以及 SPD 流形（manifold）與此有何關聯。
#
# > **前置條件：** 主課程第 07 章。
# > **難度：** advanced ★★★★☆
# > **不受 5 分鐘 CPU 預算限制。**

# %% Bootstrap — 若未以套件形式安裝，從 repo src 匯入 neuro101
import sys
import os
from pathlib import Path

try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(6):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mne.decoding import CSP

from neuro101 import io, datasets as ds
from neuro101.eval import make_block_split

rng = np.random.default_rng(42)
SMOKE = ds.is_smoke()
print(f"Smoke 模式: {SMOKE}")

# %% [markdown]
# ## 第一部分 — 每個試驗都存在於一個彎曲的流形上
#
# ### 從原始 epoch 到共變異數矩陣
#
# 給定一個形狀為 *(通道數, 時間點數)* 的試驗 **X**，其**樣本共變異數矩陣（sample covariance matrix）**為：
#
# $$
# \mathbf{C} = \frac{1}{T-1}\,\mathbf{X}\,\mathbf{X}^\top \;\in\; \mathbb{R}^{p \times p}
# $$
#
# 其中 $p$ 為通道數，$T$ 為時間取樣數。
# 這個 $p\!\times\!p$ 矩陣編碼了該試驗中**所有通道對之間的共同激活（co-activation）**——
# 一個豐富而精簡的大腦狀態指紋。
#
# ### 為何共變異數矩陣是 SPD
#
# $\mathbf{C}$ 永遠是：
#
# * **對稱（Symmetric）**（$\mathbf{C}^\top = \mathbf{C}$）——顯然成立，因為
#   $\text{Cov}(i,j) = \text{Cov}(j,i)$。
# * **正半定（Positive semi-definite）**——對任意向量 **v** 均有
#   $\mathbf{v}^\top\mathbf{C}\mathbf{v} \ge 0$（它衡量投影方差，不可為負）。
# * **嚴格正定（SPD，Strictly positive definite）**——當通道之間不存在完全線性相依時成立，
#   任何合理的白化（whitening）或 OAS 正則化後均滿足此條件。
#
# ### SPD 矩陣構成黎曼流形（Riemannian manifold），而非平坦歐氏空間
#
# 所有 $p\!\times\!p$ SPD 矩陣的集合，記為 $\text{Sym}^+_p$，
# **並非**向量子空間：兩個 SPD 矩陣的歐氏中點仍是 SPD，
# 但在「矩陣元素」意義下樸素地取平均或插值會違背該空間實際具有的幾何結構。
#
# 具體而言，$\text{Sym}^+_p$ 是一個**光滑黎曼流形（smooth Riemannian manifold）**——
# 嵌入於對稱矩陣空間中的彎曲曲面。兩個矩陣 $\mathbf{A}$ 與 $\mathbf{B}$
# 之間的自然（「仿射不變」）距離為：
#
# $$
# d_R(\mathbf{A}, \mathbf{B})
# = \left\|\log\!\left(\mathbf{A}^{-1/2}\,\mathbf{B}\,\mathbf{A}^{-1/2}\right)\right\|_F
# = \sqrt{\sum_i \log^2\!\lambda_i}
# $$
#
# 其中 $\lambda_i$ 是 $\mathbf{A}$ 與 $\mathbf{B}$ 的聯合特徵值（joint eigenvalues）。
# 此距離對全等變換（congruence transforms）不變（$\mathbf{C} \mapsto
# \mathbf{W}\mathbf{C}\mathbf{W}^\top$），意即它不關心訊號的物理尺度，只關心其*形狀*。

# %% [markdown]
# ## 第二部分 — 視覺直覺：歐氏均值（Euclidean mean）vs. 黎曼均值（膨脹效應）
#
# 我們用 **2×2** SPD 矩陣來說明核心病理現象，在此維度下流形可以被繪製，
# 且行列式（determinant）有清晰的幾何意義：等於以特徵值平方根為軸的橢圓面積。
#
# **膨脹效應（swelling effect）**：一組 SPD 矩陣的歐氏（逐元素）均值的行列式
# 永遠**大於或等於**黎曼（幾何）均值的行列式。
# 在平坦空間中取平均會「過度膨脹」橢圓。
# 黎曼均值保持在流形上，避免了這個問題。

# %%
def make_spd_2x2(angle_deg: float, lam1: float, lam2: float) -> np.ndarray:
    """建立具有給定特徵值和主方向的 2x2 SPD 矩陣。"""
    theta = np.deg2rad(angle_deg)
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return Q @ np.diag([lam1, lam2]) @ Q.T


def draw_ellipse(ax, cov: np.ndarray, center=(0, 0), n_std=1.5,
                 color="steelblue", alpha=0.25, lw=2, label=None):
    """在 2x2 共變異數矩陣對應的橢圓上疊加繪製。"""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                  color=color, alpha=alpha, lw=lw, fill=True,
                  edgecolor=color, linewidth=lw, zorder=3, label=label)
    ax.add_patch(ell)


# 建立一組分布在「單位矩陣方向」附近的 2x2 SPD 矩陣
angles   = np.linspace(-50, 50, 7)    # 主軸在 -50° 到 +50° 之間傾斜
lam_vals = np.linspace(1.5, 4.0, 7)   # 特徵值差距在各矩陣間變化

mats = np.stack([make_spd_2x2(a, l, 0.4) for a, l in zip(angles, lam_vals)])

# 歐氏均值（矩陣元素的平面平均）
euclid_mean = mats.mean(axis=0)

# 黎曼均值（最小化仿射不變距離的平方和）
riemann_mean = mean_riemann(mats)

det_euc = np.linalg.det(euclid_mean)
det_rie = np.linalg.det(riemann_mean)

print(f"歐氏均值的行列式    : {det_euc:.4f}")
print(f"黎曼均值的行列式    : {det_rie:.4f}")
print(f"膨脹比（歐氏 / 黎曼）: {det_euc / det_rie:.3f}  "
      f"  <- 歐氏比黎曼大 {100*(det_euc/det_rie-1):.1f}%")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(mats)))

for ax, title, mean_cov, mean_color, mean_label in [
    (axes[0], "歐氏均值（平面平均）",
     euclid_mean, "#e74c3c", f"歐氏均值\n(det={det_euc:.2f})"),
    (axes[1], "黎曼均值（幾何 / 測地線）",
     riemann_mean, "#2ecc71", f"黎曼均值\n(det={det_rie:.2f})"),
]:
    for i, m in enumerate(mats):
        draw_ellipse(ax, m, color=palette[i], alpha=0.18, lw=1.2)
    draw_ellipse(ax, mean_cov, color=mean_color, alpha=0.65, lw=3,
                 label=mean_label)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal"); ax.axhline(0, lw=0.5, c="k"); ax.axvline(0, lw=0.5, c="k")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("維度 1"); ax.set_ylabel("維度 2")
    ax.legend(loc="upper right", fontsize=9)

fig.suptitle(
    "膨脹效應：歐氏均值過度膨脹橢圓\n"
    "（半透明橢圓 = 各矩陣；填充橢圓 = 均值）",
    fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_swelling.png", dpi=100, bbox_inches="tight")
plt.show()
print("圖形已儲存。")

# %% [markdown]
# ### 解讀圖形
#
# 每個半透明橢圓代表一個 2×2 SPD 矩陣的「信賴橢圓」視覺化。
# 粗體填充橢圓為均值。
#
# * **左圖：** 歐氏均值橢圓明顯*大於*任何單一橢圓——它已「膨脹」。
#   其行列式約為 $\approx {det_euc:.2f}$，遠超各組成矩陣。
# * **右圖：** 黎曼均值橢圓位於雲團*內部*，
#   適當地在流形上置中。其行列式約為 $\approx {det_rie:.2f}$。
#
# **為何這對 EEG 重要：** 若你跨試驗對共變異數矩陣取歐氏平均以估計「參考值」
# 或白化資料，你是在使用一個系統性膨脹的矩陣——你的距離和投影將會錯誤。
# 黎曼均值使用矩陣間的測地線（彎曲）路徑，全程保持在 SPD 流形上。
#
# **形式保證（Ando–Li–Mathias, 2004）：**
#
# $$
# \det\!\left(\frac{\mathbf{A}+\mathbf{B}}{2}\right)
# \;\ge\;
# \det\!\left(\mathbf{A} \#\mathbf{B}\right)
# $$
#
# 其中 $\mathbf{A}\#\mathbf{B} = \mathbf{A}^{1/2}
# (\mathbf{A}^{-1/2}\mathbf{B}\mathbf{A}^{-1/2})^{1/2}\mathbf{A}^{1/2}$
# 為幾何均值。等號成立若且唯若 $\mathbf{A}=\mathbf{B}$。

# %%
# 驗證膨脹效應在多個隨機 SPD 集合上均成立
swelling_ratios = []
for _ in range(200):
    Z = rng.standard_normal((8, 3, 3))
    S = np.einsum("nij,nkj->nik", Z, Z) + np.eye(3)[None] * 0.5
    eu = S.mean(0)
    ri = mean_riemann(S)
    swelling_ratios.append(np.linalg.det(eu) / np.linalg.det(ri))

swelling_ratios = np.array(swelling_ratios)
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(swelling_ratios, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
ax.axvline(1.0, color="red", lw=2, linestyle="--", label="比值 = 1（無膨脹）")
ax.set(xlabel="det(歐氏均值) / det(黎曼均值)",
       ylabel="次數（200 個隨機試驗）",
       title="膨脹效應永遠 ≥ 1 — 歐氏均值永遠過度膨脹行列式")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/dd_rie_swelling_hist.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"膨脹比：min={swelling_ratios.min():.3f} "
      f"mean={swelling_ratios.mean():.3f} "
      f"max={swelling_ratios.max():.3f}  (永遠 >= 1)")

# %% [markdown]
# ## 第三部分 — 切線空間（tangent space）技巧
#
# 直接在流形上操作代價高昂（每次均值計算都是迭代的）。
# **切線空間投影（tangent-space projection）**讓我們兼得兩者之長：
# 正確的彎曲幾何 *以及* 分類器所需的廉價平面算術。
#
# ### 運作原理
#
# 1. **計算黎曼均值** $\mathbf{M}$——僅使用*訓練集*共變異數矩陣
#    （這是流形上的「參考點」）。
# 2. **透過矩陣對數（matrix logarithm）將每個試驗映射到 $\mathbf{M}$ 處的切線空間：**
#
#    $$
#    \mathbf{S}_i = \text{Log}_\mathbf{M}(\mathbf{C}_i)
#    = \mathbf{M}^{1/2}
#      \log\!\left(\mathbf{M}^{-1/2}\mathbf{C}_i\mathbf{M}^{-1/2}\right)
#      \mathbf{M}^{1/2}
#    $$
#
#    $\mathbf{S}_i$ 是一個**對稱**矩陣（不必然正定）——我們現在處於平坦空間中。
# 3. **向量化** $\mathbf{S}_i$（上三角，非對角元素乘以 $\sqrt{2}$
#    以保留 Frobenius 範數）為特徵向量。
# 4. **對這些向量擬合任意線性分類器。**
#
# 直覺：在參考點附近，光滑流形看起來是平坦的——
# 就像城市地圖在短距離內忽略地球曲率一樣。
# 切線空間給出以訓練集幾何均值為中心的局部平面地圖。
#
# **關鍵點：參考點 $\mathbf{M}$ 必須僅在訓練折（training fold）上擬合**——
# 將測試集共變異數投影到使用測試集估計的參考點上是資料洩漏（data leakage）。
# `pyriemann.TangentSpace` 在 sklearn `Pipeline` 內部自動處理此問題。

# %%
# 切線空間投影的微型合成示範
from pyriemann.utils.tangentspace import tangent_space, untangent_space  # noqa

# 生成 20 個隨機 4x4 SPD 矩陣（模擬共變異數）
Z = rng.standard_normal((20, 4, 10))
C_demo = np.einsum("nij,nkj->nik", Z, Z) + np.eye(4)[None] * 0.3

# 計算黎曼均值
M_demo = mean_riemann(C_demo)

# 投影至 M_demo 處的切線空間
S_demo = tangent_space(C_demo, M_demo)
print(f"原始 SPD 形狀 : {C_demo.shape}  (矩陣數, p, p)")
print(f"切線向量      : {S_demo.shape}  (矩陣數, p*(p+1)//2)")
print(f"→ 每個 4x4 共變異數 → {S_demo.shape[1]} 維平面向量")
print()

# 驗證：黎曼均值在切線空間中映射到零向量
S_mean = tangent_space(M_demo[np.newaxis], M_demo)
print(f"參考點的切線向量（應接近 0）: "
      f"max|coeff| = {np.abs(S_mean).max():.2e}")

# %% [markdown]
# ## 第四部分 — 實驗：黎曼 vs. CSP+LDA 於訓練集縮減時的比較
#
# 現在進行核心實驗。我們在 BCI IV 2a（運動想像）上比較兩條管線：
#
# | 管線 | 步驟 |
# |---|---|
# | **黎曼（Riemann）** | `Covariances(oas)` → `TangentSpace` → `LogisticRegression` |
# | **CSP+LDA** | `CSP(n_components=4)` → `LinearDiscriminantAnalysis` |
#
# 我們在單一受試者內使用 `make_block_split`（連續折，無洩漏）進行評估，
# 並將**訓練比例**從可用訓練試驗的 20% 掃描至 100%。
#
# **假設：** 黎曼特徵依賴於共變異數矩陣——每個矩陣是完整空間共變化模式的充分統計量。
# 在少量試驗的情況下，共變異數矩陣仍可估計（尤其配合 OAS 收縮），
# 而 CSP 必須從小而嘈雜的資料中從頭識別空間濾波器。
# 我們預期黎曼方法的退化更為緩慢。

# %%
# 載入資料——smoke 模式下使用 2 位受試者，否則最多 3 位
n_subj = 2 if SMOKE else 3
print(f"正在從 BCI IV 2a 載入 {n_subj} 位受試者的資料…")
X_all, y_all, subj_all = io.load_bnci_2a_epochs(n_subjects=n_subj)
print(f"  X={X_all.shape}, 類別分布={np.bincount(y_all)}")

# 僅在受試者 1 內作業（試驗數最多）
s1_mask = subj_all == subj_all.min()
X1, y1 = X_all[s1_mask], y_all[s1_mask]
print(f"\n受試者 {subj_all.min()}: {X1.shape[0]} 個試驗, "
      f"{X1.shape[1]} 個通道, {X1.shape[2]} 個時間點")
print(f"類別平衡: {np.bincount(y1)}")

# %%
# 定義兩條管線
pipe_riemann = Pipeline([
    ("cov",  Covariances(estimator="oas")),
    ("ts",   TangentSpace(metric="riemann")),
    ("clf",  LogisticRegression(C=1.0, max_iter=500, random_state=0)),
])

pipe_csp = Pipeline([
    ("csp",  CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("clf",  LinearDiscriminantAnalysis()),
])

# %%
# 掃描訓練比例並用區塊切分交叉驗證（block-split CV）評估
# make_block_split 給出連續誠實折
N_SPLITS = 4 if SMOKE else 5
TRAIN_FRACS = [0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS = [0, 1, 2]  # 保持快速；更多種子 = 更緊的信賴區間

def eval_pipeline_at_frac(pipe, X, y, frac, n_splits, seeds):
    """
    對每個種子：取每個訓練折試驗的前 `frac` 比例，
    擬合管線，並在保留測試折上評分。
    回傳形狀為 (n_seeds, n_splits) 的準確率陣列。
    """
    from sklearn.base import clone as sk_clone
    scores = np.zeros((len(seeds), n_splits))
    folds = list(make_block_split(len(y), n_splits=n_splits))

    for si, seed in enumerate(seeds):
        rng_s = np.random.default_rng(seed)
        for fi, (train_idx, test_idx) in enumerate(folds):
            # 將訓練集子取樣至可用試驗的 `frac` 比例
            n_keep = max(2, int(len(train_idx) * frac))
            # 始終取連續前綴以不打亂時間順序
            train_sub = train_idx[:n_keep]

            model = sk_clone(pipe)
            try:
                model.fit(X[train_sub], y[train_sub])
                preds = model.predict(X[test_idx])
                scores[si, fi] = (preds == y[test_idx]).mean()
            except Exception:
                scores[si, fi] = np.nan
    return scores


results = {}  # frac -> {"riemann": arr, "csp": arr}

for frac in TRAIN_FRACS:
    print(f"  frac={frac:.0%} …", end=" ", flush=True)
    r_scores = eval_pipeline_at_frac(pipe_riemann, X1, y1, frac, N_SPLITS, SEEDS)
    c_scores = eval_pipeline_at_frac(pipe_csp,     X1, y1, frac, N_SPLITS, SEEDS)
    results[frac] = {"riemann": r_scores, "csp": c_scores}
    print(f"黎曼={np.nanmean(r_scores):.3f}  CSP={np.nanmean(c_scores):.3f}")

print("\n完成。")

# %%
# 彙總：跨種子×折的平均準確率 ± 標準差
fracs    = TRAIN_FRACS
rie_mean = [np.nanmean(results[f]["riemann"]) for f in fracs]
rie_std  = [np.nanstd(results[f]["riemann"])  for f in fracs]
csp_mean = [np.nanmean(results[f]["csp"])     for f in fracs]
csp_std  = [np.nanstd(results[f]["csp"])      for f in fracs]

print("訓練比例 | 黎曼準確率      | CSP+LDA 準確率  | 黎曼優勢")
print("-" * 75)
for f, rm, rs, cm, cs in zip(fracs, rie_mean, rie_std, csp_mean, csp_std):
    adv = rm - cm
    print(f"       {f:5.0%}        | {rm:.3f} ± {rs:.3f}  | {cm:.3f} ± {cs:.3f}  | {adv:+.3f}")

# %%
# 繪圖：準確率 vs. 訓練比例
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    [f * 100 for f in fracs], rie_mean, yerr=rie_std,
    marker="o", linewidth=2.0, capsize=4, color="#2ecc71",
    label="黎曼（Cov + TangentSpace + LogReg）", zorder=5,
)
ax.errorbar(
    [f * 100 for f in fracs], csp_mean, yerr=csp_std,
    marker="s", linewidth=2.0, capsize=4, color="#e74c3c",
    label="CSP（4 個成分）+ LDA", zorder=5,
)
ax.axhline(0.5, color="gray", lw=1.0, linestyle=":", label="機會水準（50 %）")

# 標記低資料區間
ax.axvspan(0, 45, alpha=0.07, color="gold", label="低資料區間")

ax.set(
    xlabel="訓練集大小（% 單一受試者的試驗數）",
    ylabel="準確率（區塊切分交叉驗證）",
    title="黎曼 vs. CSP+LDA：訓練資料縮減時的準確率\n"
          f"BCI IV 2a, 受試者 {subj_all.min()}, "
          f"{N_SPLITS} 折區塊切分, {len(SEEDS)} 個種子",
    xlim=(10, 105), ylim=(0.35, 1.05),
)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_smalldata.png", dpi=100, bbox_inches="tight")
plt.show()
print("圖形已儲存。")

# %% [markdown]
# ## 第五部分 — 為何共變異數更穩健地彙集資訊？
#
# ### 參數數量 vs. 試驗數量
#
# **CSP** 必須估計長度為 $p$（通道數）的 $k$ 個空間濾波器。
# 這些濾波器是**類別條件共變異數矩陣**廣義特徵值問題的解——
# 但每個類別的共變異數有 $p(p+1)/2$ 個自由參數。
# 以 22 個通道和每類僅 20 個訓練試驗，
# 你是在從 20 個觀測中估計 $22 \times 23/2 = 253$ 個數值：
# 一個嚴重欠定的問題。CSP 因此容易拾取雜訊。
#
# **共變異數 + 切線空間（黎曼）** *同樣*需要對每個試驗估計 $p \times p$ 共變異數。
# 差異在於：
#
# 1. **OAS 收縮（OAS shrinkage）**：`Covariances(estimator="oas")` 步驟應用
#    Oracle-Approximating Shrinkage（OAS，甲骨文近似收縮），
#    將樣本共變異數拉向
#    $\alpha \hat{\mathbf{C}} + (1-\alpha)\,\text{tr}(\hat{\mathbf{C}})/p\,\mathbf{I}$。
#    即使在試驗數極少的情況下，也能保持矩陣良好條件，代價是微小的偏差。
# 2. **無明確濾波器學習**：分類器在完整
#    $p(p+1)/2$ 維切線空間中操作，並使用 $\ell_2$ 正則化邏輯迴歸。
#    不存在可能對不良空間濾波器災難性過擬合的中間步驟。
# 3. **幾何均值作為穩健參考**：即使只有少量共變異數矩陣的黎曼均值，
#    也已是流形上有意義、穩定的點——因為 $\text{Sym}^+_p$ 的曲率為非正
#    （Hadamard 空間），均值唯一，且迭代算法即使從少量點也能收斂。
#
# ### 非正式直覺
#
# > 每個共變異數矩陣將所有 $p \times T$ 個別測量值「彙集」成一個 $p \times p$ 物件。
# > 即使單一試驗也能產生豐富的大腦狀態描述符。
# > 相比之下，CSP 需要足夠多的*標記*試驗，使兩個類別條件池在統計上可區分，
# > 才能找到有用的濾波器。在 $n$ 極小的情況下，兩個池偶然重疊，
# > 估計出的濾波器是任意的。

# %%
# 快速示範：樣本共變異數 vs. OAS 共變異數的條件數隨試驗數增長的變化
from pyriemann.estimation import Covariances as PyrCov

p = 22  # BCI IV 2a 有 22 個 EEG 通道
trial_counts = [5, 10, 20, 40, 80, 160] if not SMOKE else [5, 10, 20, 40]
n_reps = 30 if not SMOKE else 10

cond_scm = []  # 樣本共變異數（無正則化）
cond_oas = []  # OAS 收縮

for n in trial_counts:
    c_scm_r, c_oas_r = [], []
    for _ in range(n_reps):
        # 模擬 n 個 p 通道 EEG 試驗（隨機，無真實訊號）
        Z = rng.standard_normal((n, p, 50))  # 每個試驗 50 個時間樣本
        C_scm = PyrCov(estimator="scm").fit_transform(Z)
        C_oas = PyrCov(estimator="oas").fit_transform(Z)
        c_scm_r.append(np.mean([np.linalg.cond(c) for c in C_scm]))
        c_oas_r.append(np.mean([np.linalg.cond(c) for c in C_oas]))
    cond_scm.append(np.mean(c_scm_r))
    cond_oas.append(np.mean(c_oas_r))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(trial_counts, cond_scm, "o-", color="#e74c3c",
            lw=2, label="樣本共變異數（SCM，無正則化）")
ax.semilogy(trial_counts, cond_oas, "s-", color="#2ecc71",
            lw=2, label="OAS 收縮（pyriemann 預設）")
ax.set(
    xlabel="試驗數（模擬，p=22 個通道）",
    ylabel="平均條件數（對數尺度）",
    title="OAS 收縮保持共變異數矩陣良好條件\n"
          "即使在 n_trials << n_channels² 的情況下",
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/dd_rie_condnum.png", dpi=100, bbox_inches="tight")
plt.show()
print("條件數（SCM vs. OAS）已繪製。")

# %% [markdown]
# ### 結果摘要
#
# 條件數圖揭示了核心問題：
# 當 $n\_trials \ll p(p+1)/2$ 時，樣本共變異數幾乎奇異
# （條件數極大 → 數值上不穩定的逆矩陣 → 不良的黎曼距離）。
# OAS 收縮無論訓練集大小如何，都能限制條件數，
# 這就是為何黎曼管線在每類僅 10–20 個試驗的情況下仍能保持競爭力。
#
# 準確率 vs. 比例圖（第四部分）顯示了後果：
# 在訓練集的 20% 時，黎曼管線通常領先 CSP+LDA 幾個百分點；
# 到完整訓練集時，差距縮小甚至逆轉，因為 CSP 有足夠資料識別良好的空間濾波器。

# %% [markdown]
# ## ⚠️ A subtler trap: cross-session covariance shift and the leaking reference
#
# ### 問題
#
# 切線空間投影需要一個**參考矩陣** $\mathbf{M}$（黎曼均值）。
# 在典型工作流程中，你在訓練共變異數上擬合 $\mathbf{M}$。
# 每個共變異數——訓練 *和* 測試——然後相對於*同一個*參考投影。
#
# 這在單一會話（session）內運作完美。但由於以下原因，
# EEG 共變異數矩陣**在會話和受試者之間會發生漂移（shift）**：
#
# * 電極阻抗變化（凝膠乾燥、帽子重新定位），
# * 皮膚-電極接觸漂移，
# * 受試者疲勞或喚醒水準變化，
# * 帽子放置的細微差異（通道位置略有偏移）。
#
# 當你在會話 1 訓練並在會話 2 測試時，測試共變異數處於流形上*與*訓練共變異數不同的部分。
# 將兩者都投影到會話 1 的參考 $\mathbf{M}_1$ 上，
# 會在切線空間中拉伸和旋轉測試會話的特徵——
# 分類器看到的分布偏移（distribution shift）看起來像是一個新任務。
#
# ### （有時）隱藏的洩漏
#
# 一個常見錯誤：**使用所有試驗（訓練 + 測試）計算 $\mathbf{M}$，然後再切分。**
# 這是資料洩漏（data leakage）。參考已「看過」測試會話，
# 並將測試共變異數精確地重置回模型訓練的位置——使分數看起來比實際更好。
# 在會話內交叉驗證中，此效應輕微（兩個會話在流形上很接近）。
# 跨會話或受試者時，膨脹可能是巨大的。
#
# ### 解決方案：黎曼對齊（Riemannian alignment，RA）（每會話重置中心）
#
# 標準方法是 He & Wu（2019）提出的**黎曼對齊（RA）**：
# 在串接會話之前，將每個會話的共變異數白化（whiten）到一個共同參考（通常是單位矩陣）：
#
# $$
# \tilde{\mathbf{C}}_i^{(s)} = \mathbf{M}_s^{-1/2}\,\mathbf{C}_i^{(s)}\,\mathbf{M}_s^{-1/2}
# $$
#
# 其中 $\mathbf{M}_s$ 是**僅從該會話訓練試驗計算**的會話 $s$ 的黎曼均值。
# 對齊後，所有會話的共變異數都以單位矩陣為中心，消除了會話間漂移。
#
# **關鍵規則：**
#
# > *測試*會話的 $\mathbf{M}_s$ 必須從測試會話自己的（未標記）試驗估計，
# > 而非從訓練會話估計。
# > 若你使用測試會話標籤或訓練會話均值來對齊測試會話，
# > 你要麼造成洩漏，要麼未能消除漂移。
#
# 這就是為何黎曼方法儘管優雅，仍然需要仔細記錄*哪些資料用於擬合參考*。
# 流形幾何完美解決了會話內平均問題——但並不自動解決跨會話分布漂移問題。
#
# ### 最小程式碼示意（請勿在所有資料的訓練/測試折內執行）
#
# ```python
# # 洩漏版本（跨會話評估時錯誤）
# M_all  = mean_riemann(np.concatenate([C_train, C_test]))   # 看到了測試集！
# S_train = tangent_space(C_train, M_all)
# S_test  = tangent_space(C_test,  M_all)
#
# # 正確版本：參考僅在訓練集上擬合
# M_train = mean_riemann(C_train)
# S_train = tangent_space(C_train, M_train)
# S_test  = tangent_space(C_test,  M_train)   # 測試集投影到訓練集參考上
#
# # 跨會話最佳實踐（黎曼對齊）：
# M_train_sess = mean_riemann(C_train)         # 來自訓練會話
# M_test_sess  = mean_riemann(C_test_unlabelled)  # 來自測試會話（不需要標籤）
# C_train_aligned = M_train_sess @ np.linalg.inv(M_train_sess) @ C_train   # = 以單位矩陣為中心
# # ... 更精確地透過矩陣平方根；pyriemann.utils.mean 提供輔助函數
# ```
#
# ### 結論
#
# 黎曼幾何為 EEG 共變異數結構提供了一個原則性、參數高效的表示，
# 在訓練集小的時候表現出色。但它*並不*讓資料收集問題消失。
# 參考點 $\mathbf{M}$ 是一個已學習的量——錯誤地擬合它
# （資料太少或使用了受污染的資料）會將誤差傳播到每個下游距離和投影中。
# 在小資料情況下，使用 OAS 收縮，僅在訓練折上擬合參考，
# 並在會話跨越數分鐘以上或不同記錄條件時考慮每會話對齊。
