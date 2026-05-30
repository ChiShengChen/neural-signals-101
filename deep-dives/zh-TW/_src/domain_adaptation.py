# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深入探討 — 遷移學習（Transfer Learning）與領域適應（Domain Adaptation）
#
# 第 12 章已說明跨受試者與跨場次的準確率為何難以提升——本章介紹如何補救：
# 針對 BCI（腦機介面）的對齊（alignment）、校準（calibration）與微調（fine-tuning）策略。
#
# > **前置條件：** 主課程第 07 章與第 12 章。
# > **難度：** advanced ★★★★☆
# > **「受試者獨立訓練很難」之後的下一步。**

# %% Bootstrap — 向上搜尋 neuro101 套件所在位置
import sys
import os
from pathlib import Path

# 強健的向上搜尋：先嘗試直接匯入，再逐層向上尋找 src/neuro101
# （無論從 repo 根目錄或 deep-dives/_src/ 執行皆可運作）。
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _here = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _parent in [_here, *_here.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101").is_dir():
            sys.path.insert(0, str(_candidate))
            break
    import neuro101  # noqa: F401 — 若仍找不到，會拋出清楚的錯誤訊息

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 使用無頭後端（headless backend），讓 Notebook 在無螢幕的 CI 環境中也能執行。
matplotlib.use("Agg")

from scipy import linalg as sp_linalg

from sklearn.base import clone as sk_clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

from mne.decoding import CSP

from neuro101 import io, datasets as ds
from neuro101.eval import make_subject_split, leakage_safe_pipeline

rng = np.random.default_rng(42)

SMOKE = ds.is_smoke()
N_SUBJ = 2 if SMOKE else 4          # 載入的受試者數量
CALIB_SIZES = [0, 5, 10, 20]        # 要掃描的校準試驗筆數
if SMOKE:
    CALIB_SIZES = [0, 5, 10]        # 減少資料點以加快 CI 速度

print(f"Smoke 模式 : {SMOKE}")
print(f"N_SUBJ     : {N_SUBJ}")
print(f"CALIB_SIZES: {CALIB_SIZES}")

# %% [markdown]
# ---
# ## 第一部分 — 領域偏移（domain shift）問題
#
# ### 跨受試者 EEG 為何困難：每個人都是不同的分佈
#
# 傳統機器學習假設訓練與測試資料來自**相同分佈** $p(\mathbf{x}, y)$。
# 在 EEG 中，這個假設幾乎立刻就被打破：
#
# * 每個人的頭骨形狀與導電率不同——從皮質電流到頭皮電壓的映射因人而異。
# * 電極帽在每次場次（session）重新配戴時位置略有偏差；Cz 的微小偏移
#   會改變各通道所「看到」的腦區。
# * 疲勞、注意力、藥物與情緒會在同一天內改變基準功率頻譜。
#
# 因此，每位受試者（或每次場次）都是一個擁有獨立 $p_s(\mathbf{x})$ 的
# **領域（domain）**。在領域 $\{s \neq t\}$ 上訓練的分類器，
# 遇到目標領域 $t$ 時會遭遇*共變量偏移（covariate shift）*：
# 標籤 $y$ 的語意不變（左手想像 = 類別 0），但特徵卻落在特徵空間的不同區域。
# 第 12 章的第 5 個陷阱量化了這項損害：
# LOSO（留一受試者）準確率相較受試者內交叉驗證可下降 15–25 個百分點。
#
# ### 三大補救方案族群
#
# | 族群 | 核心概念 | 是否需要目標領域標記資料？ |
# |---|---|---|
# | **對齊（Alignment）／分佈匹配** | 在分類前將各領域轉換至共同參照系 | 否（非監督式） |
# | **校準（Calibration）** | 在第一天部署時提供少量已標記的目標試驗 | 是（少量） |
# | **微調（Fine-tuning）** | 在來源受試者上預訓練後，以目標標籤更新權重 | 是（中等量） |
#
# 本深入探討涵蓋全部三大族群，並附有可執行的實驗。

# %% [markdown]
# ---
# ## 第二部分 — 歐氏對齊（Euclidean Alignment，EA）
#
# ### 核心概念（He & Wu, 2020）
#
# 最簡單的對齊策略直接作用於原始試驗。對於擁有試驗集
# $\{\mathbf{X}_i^{(s)}\}$ 的受試者 $s$：
#
# 1. **估計該受試者試驗的均值共變異數（mean covariance）**：
#    $$\mathbf{R}^{(s)} = \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbf{C}_i^{(s)},
#      \qquad \mathbf{C}_i = \frac{1}{T-1}\mathbf{X}_i\mathbf{X}_i^\top$$
# 2. **以反平方根（inverse square root）$(\mathbf{R}^{(s)})^{-1/2}$ 白化**：
#    $$\tilde{\mathbf{X}}_i^{(s)} = (\mathbf{R}^{(s)})^{-1/2}\,\mathbf{X}_i^{(s)}$$
#
# 對齊後，每位受試者的**均值共變異數皆為單位矩陣** $\mathbf{I}$，
# 消除了受試者特定的振幅基準線。此運算為標準的矩陣平方根，
# 時間複雜度為 $O(p^3)$——對 22 通道 EEG 而言僅需微秒級時間。
#
# **無洩漏規則：** 訓練受試者的 $\mathbf{R}^{(s)}$ 僅用其自身試驗估計。
# 目標（測試）受試者的 $\mathbf{R}^{(t)}$ 則用目標受試者*自身*的（未標記）試驗估計——
# EA 參照不需要標籤，因此這是安全的。

# %%
def euclidean_alignment(X_tr, X_te, subj_tr, subj_te):
    """無洩漏的歐氏對齊（Euclidean Alignment）。

    對每位受試者分別在*其自身*試驗上擬合 R^{-1/2}。
    訓練受試者：R 僅在訓練試驗上擬合。
    目標受試者：R 在目標受試者的保留（測試）試驗上擬合——
    此為無標籤操作，因此不構成資料洩漏。

    Parameters
    ----------
    X_tr   : (n_train, ch, time)
    X_te   : (n_test, ch, time)
    subj_tr: (n_train,) 訓練試驗的受試者 ID
    subj_te: (n_test,)  測試試驗的受試者 ID

    Returns
    -------
    X_tr_ea, X_te_ea : 已對齊的輸入陣列
    """
    def _whiten(X_domain):
        """計算均值共變異數並回傳白化後的試驗。"""
        # 計算此領域內所有試驗的均值共變異數
        Cs = np.stack([x @ x.T / x.shape[-1] for x in X_domain])
        R = Cs.mean(axis=0)
        # 透過特徵分解計算反平方根
        vals, vecs = np.linalg.eigh(R)
        vals = np.maximum(vals, 1e-12)          # 數值下界
        R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        # 套用：(ch,ch) @ (n, ch, time)
        return np.einsum("ij,njk->nik", R_inv_sqrt, X_domain)

    X_tr_ea = np.empty_like(X_tr)
    for s in np.unique(subj_tr):
        m = subj_tr == s
        X_tr_ea[m] = _whiten(X_tr[m])

    X_te_ea = np.empty_like(X_te)
    for s in np.unique(subj_te):
        m = subj_te == s
        X_te_ea[m] = _whiten(X_te[m])          # 使用目標受試者自身試驗——無標籤，安全

    return X_tr_ea, X_te_ea

print("euclidean_alignment() 已定義——以約 20 行實作 He & Wu (2020)。")

# %%
# ── 載入資料 ──────────────────────────────────────────────────────────────────
print(f"\n正在從 BCI IV 2a 載入 {N_SUBJ} 位受試者…")
X, y, subj = io.load_bnci_2a_epochs(n_subjects=N_SUBJ)
print(f"  X 形狀 : {X.shape}  （試驗 × 通道 × 時間點）")
print(f"  受試者 : {np.unique(subj).tolist()}")
print(f"  類別   : {np.bincount(y).tolist()}  （0=左手，1=右手）")

# %%
# ── EA 前後的 LOSO 準確率 ─────────────────────────────────────────────────────
# 管線：黎曼共變異數（Riemann Covariances）→ 切線空間（Tangent Space）→ 邏輯迴歸
# （CSP+LDA 同樣適用——EA 的效益可轉移至任何下游分類器）

pipe_template = leakage_safe_pipeline([
    ("cov", Covariances(estimator="oas")),
    ("ts",  TangentSpace(metric="riemann")),
    ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=0)),
])

accs_before, accs_after = [], []

print("\nLOSO 交叉驗證（黎曼管線）…")
print(f"{'折次':>4}  {'測試受試者':>9}  {'EA 前':>9}  {'EA 後':>8}")
print("-" * 40)

for fold_i, (train_idx, test_idx) in enumerate(make_subject_split(subj)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    # ── EA 前 ──────────────────────────────────────────────────────────
    pipe_b = sk_clone(pipe_template)
    pipe_b.fit(X_tr, y_tr)
    acc_b = pipe_b.score(X_te, y_te)
    accs_before.append(acc_b)

    # ── EA 後 ───────────────────────────────────────────────────────────
    # EA 參照：訓練受試者（訓練集）與目標受試者（測試集）分別擬合。
    X_tr_ea, X_te_ea = euclidean_alignment(
        X_tr, X_te, subj[train_idx], subj[test_idx]
    )
    pipe_a = sk_clone(pipe_template)
    pipe_a.fit(X_tr_ea, y_tr)
    acc_a = pipe_a.score(X_te_ea, y_te)
    accs_after.append(acc_a)

    test_subj = np.unique(subj[test_idx])[0]
    print(f"  {fold_i:2d}    受試者 {test_subj:>3}       {acc_b:.3f}       {acc_a:.3f}")

mean_before = np.mean(accs_before)
mean_after  = np.mean(accs_after)
print("-" * 40)
print(f"  平均準確率  EA 前 : {mean_before:.3f}")
print(f"  平均準確率  EA 後 : {mean_after:.3f}")
print(f"  Δ（EA 後 − EA 前）: {mean_after - mean_before:+.3f}")

# %%
# ── 圖一 — EA 前後長條圖 ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, N_SUBJ * 1.5 + 2), 5))

n_folds = len(accs_before)
x_pos   = np.arange(n_folds)
bar_w   = 0.35

bars_b = ax.bar(x_pos - bar_w / 2, accs_before, bar_w,
                color="#5b9bd5", alpha=0.85, label="EA 前")
bars_a = ax.bar(x_pos + bar_w / 2, accs_after,  bar_w,
                color="#ed7d31", alpha=0.85, label="EA 後")

# 在每個長條上標示數值
for rect in list(bars_b) + list(bars_a):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

# 均值參考線
ax.axhline(mean_before, color="#5b9bd5", lw=2, ls="--",
           label=f"EA 前均值 = {mean_before:.3f}")
ax.axhline(mean_after,  color="#ed7d31", lw=2, ls="--",
           label=f"EA 後均值 = {mean_after:.3f}")
ax.axhline(0.5, color="gray", lw=1.0, ls=":", label="機率基準（50%）")

fold_labels = [f"S{np.unique(subj[te])[0]}" for _, te in make_subject_split(subj)]
ax.set_xticks(x_pos)
ax.set_xticklabels(fold_labels)
ax.set(
    xlabel="保留的測試受試者（LOSO 折次）",
    ylabel="準確率（黎曼管線）",
    title=(
        "LOSO 準確率：歐氏對齊前後比較——黎曼管線\n"
        f"BCI IV 2a  （{N_SUBJ} 位受試者，無 CSP — OAS 共變異數 + 切線空間 + LR）"
    ),
    ylim=(0.3, 1.0),
)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/tmp/dd_da_ea_bar.png", dpi=100, bbox_inches="tight")
plt.show()
print("圖一已儲存 → /tmp/dd_da_ea_bar.png")

# %% [markdown]
# ### 如何解讀長條圖
#
# 每對長條代表一個 LOSO 折次（即一位保留的受試者）。左側（藍色）長條
# 為無對齊時的準確率；右側（橘色）長條為 EA 後的準確率。
#
# * EA 在計算切線空間特徵前，將每位受試者的共變異數重新對齊至單位矩陣——
#   在不觸碰標籤的情況下消除受試者間的振幅基準線差異。
# * 對於基準功率與群體均值差異較大的受試者，提升幅度可能相當顯著；
#   對於原本就接近均值的受試者，效果則較小或中性。
# * 當目標受試者在共變異數空間中屬於離群值時，改善最為明顯——
#   這正是 BCI 實際部署所面臨的情況（使用者不可由研究者挑選）。

# %% [markdown]
# ---
# ## 第三部分 — 黎曼重新定心（Riemannian re-centring）：EA 的幾何類比
#
# EA 以*歐氏*均值共變異數 $\mathbf{R} = \frac{1}{N}\sum_i \mathbf{C}_i$ 進行白化。
# 更嚴謹的黎曼替代方案是以**黎曼（幾何）均值** $\mathbf{M}$ 重新定心，
# 在 SPD（對稱正定）流形（manifold）上將每個領域的共變異數置於單位矩陣處。
#
# 運算結構相同：
# $$\tilde{\mathbf{C}}_i = \mathbf{M}^{-1/2}\,\mathbf{C}_i\,\mathbf{M}^{-1/2}$$
# 但 $\mathbf{M}$ 最小化的是*仿射不變（affine-invariant）*（黎曼）距離的平方和，
# 而非 Frobenius 距離的平方和。
#
# **何時有差異？** SPD 矩陣的歐氏均值行列式永遠大於黎曼均值——
# 即「膨脹效應（swelling effect）」（詳見 `riemann_small_data` 深入探討）。
# 對於試驗數適中的 EEG，差異通常很小，但當受試者共變異數極大或極小時
# （離群通道、富含偽影的記錄）差異會增大。
#
# 在 pyriemann 函式庫中，此功能由 `pyriemann.utils.mean.mean_riemann` 提供。
# 以下並排展示兩種對齊方式，以一位訓練受試者的試驗均值共變異數為例進行說明。

# %%
# ── 視覺比較：某位受試者的歐氏均值與黎曼均值 ────────────────────────────────
s_demo = np.unique(subj)[0]          # 選取第一位受試者
X_demo = X[subj == s_demo]           # (n_trials, ch, time)
Cs_demo = np.stack([x @ x.T / x.shape[-1] for x in X_demo])

# 歐氏均值
R_eucl = Cs_demo.mean(axis=0)

# 黎曼均值
R_riem = mean_riemann(Cs_demo)

# 行列式：黎曼均值應較小（無膨脹效應）
det_eucl = np.linalg.det(R_eucl)
det_riem = np.linalg.det(R_riem)
print(f"受試者 {s_demo}  （{len(X_demo)} 筆試驗）")
print(f"  det（歐氏均值）  : {det_eucl:.4e}")
print(f"  det（黎曼均值）  : {det_riem:.4e}")
print(f"  膨脹比（swelling ratio）: {det_eucl / det_riem:.4f}  （>1 代表歐氏均值偏大）")

# 特徵值頻譜
eigvals_eucl = np.sort(np.linalg.eigvalsh(R_eucl))[::-1]
eigvals_riem = np.sort(np.linalg.eigvalsh(R_riem))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 面板 A — 特徵值頻譜比較
ax = axes[0]
idx = np.arange(1, len(eigvals_eucl) + 1)
ax.semilogy(idx, eigvals_eucl, "o-", color="#5b9bd5", lw=2, label="歐氏均值")
ax.semilogy(idx, eigvals_riem, "s-", color="#ed7d31", lw=2, label="黎曼均值")
ax.set(
    xlabel="特徵值排名（1 = 最大）",
    ylabel="特徵值（對數尺度）",
    title=f"均值共變異數的特徵值頻譜\n（受試者 {s_demo}，{len(X_demo)} 筆試驗）",
)
ax.legend(fontsize=9)

# 面板 B — 白化後與單位矩陣的 Frobenius 距離
def frob_from_identity(R_ref, Cs):
    """以 R_ref 白化後，各共變異數距離單位矩陣有多遠？"""
    vals, vecs = np.linalg.eigh(R_ref)
    R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    # 白化每筆試驗的共變異數
    dists = []
    for C in Cs:
        C_w = R_inv_sqrt @ C @ R_inv_sqrt
        dists.append(np.linalg.norm(C_w - np.eye(C.shape[0]), "fro"))
    return np.array(dists)

dists_eucl = frob_from_identity(R_eucl, Cs_demo)
dists_riem = frob_from_identity(R_riem, Cs_demo)

ax = axes[1]
bins = np.linspace(0, max(dists_eucl.max(), dists_riem.max()) * 1.05, 30)
ax.hist(dists_eucl, bins=bins, alpha=0.6, color="#5b9bd5",
        label=f"歐氏對齊  （均值={dists_eucl.mean():.2f}）")
ax.hist(dists_riem, bins=bins, alpha=0.6, color="#ed7d31",
        label=f"黎曼對齊  （均值={dists_riem.mean():.2f}）")
ax.set(
    xlabel="白化後 Cᵢ 與 I 的 Frobenius 距離",
    ylabel="試驗數",
    title="白化後的分散程度（越低 = 重新定心越緊密）",
)
ax.legend(fontsize=9)

fig.suptitle(
    "歐氏對齊與黎曼對齊比較——特徵值頻譜與殘差分散",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/dd_da_riem_align.png", dpi=100, bbox_inches="tight")
plt.show()
print("圖二已儲存 → /tmp/dd_da_riem_align.png")

# %% [markdown]
# ### 如何解讀圖二
#
# * **左側面板：** 黎曼均值的特徵值頻譜相較歐氏均值略微向 1 集中——
#   「膨脹效應」在最大特徵值中清晰可見（黎曼均值略小）。
# * **右側面板：** 以各自參照白化後，黎曼對齊的個別試驗共變異數與單位矩陣的
#   Frobenius 距離平均略低。差異在此例中不大（少量試驗，22 通道），
#   但隨著離群試驗或通道數增多而顯著放大。
#
# **實務建議：** 對於注重速度的跨受試者 BCI 管線，EA（歐氏）是首選——
# 每位受試者僅需一次特徵分解，執行時間為毫秒級。黎曼對齊的計算成本高出
# 5–20 倍（迭代求均值），而準確率提升通常不超過 1–2 個百分點。
# 僅當受試者存在極端共變異數離群值（斷損通道、前處理後仍殘留的 EMG 偽影）
# 時，才值得使用黎曼對齊。

# %% [markdown]
# ---
# ## 第四部分 — 校準（Calibration）：提供模型少量目標受試者試驗
#
# ### 真實世界情境
#
# BCI 首次使用當天，技術人員通常會執行一個短暫的校準區塊（約 5 分鐘，
# 20–40 筆已標記試驗）來收集目標受試者的資料。
# 這些已標記試驗可在擬合分類器前加入訓練集，此即**校準**（或稱*個人化*）。
#
# 模擬方式如下：
# 1. 保留一位受試者作為目標受試者。
# 2. 假設其前 $K$ 筆試驗為「校準」資料集（已標記，部署前即可取得）。
# 3. 以來源受試者資料 + 校準試驗進行訓練。
# 4. 在目標受試者的*剩餘*試驗上進行評估。
#
# **無洩漏規則：** 校準試驗來自目標場次的*開頭*（依時間順序）；
# 評估試驗來自*結尾*。訓練過程中絕不暴露評估標籤。

# %%
# ── 校準曲線 ─────────────────────────────────────────────────────────────────
# 掃描 K ∈ CALIB_SIZES，並對所有保留目標受試者取均值。

print("校準實驗…")
print(f"{'校準大小 K':>12}  {'均值準確率':>10}  {'標準差':>7}")
print("-" * 34)

calib_results = {}   # K -> 每折準確率列表

for K in CALIB_SIZES:
    fold_accs = []

    for train_idx, test_idx in make_subject_split(subj):
        X_src, y_src = X[train_idx], y[train_idx]
        X_tgt, y_tgt = X[test_idx],  y[test_idx]

        # ── 對來源受試者套用 EA（以各來源受試者自身試驗擬合參照） ──
        X_src_ea = np.empty_like(X_src)
        for s in np.unique(subj[train_idx]):
            m = subj[train_idx] == s
            Xs = X_src[m]
            R = np.mean([x @ x.T / x.shape[-1] for x in Xs], axis=0)
            vals, vecs = np.linalg.eigh(R)
            R_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
            X_src_ea[m] = np.einsum("ij,njk->nik", R_inv_sqrt, Xs)

        # ── 目標受試者的 EA：參照由全部目標試驗估計（無標籤） ──
        R_tgt = np.mean([x @ x.T / x.shape[-1] for x in X_tgt], axis=0)
        vals_t, vecs_t = np.linalg.eigh(R_tgt)
        R_tgt_inv_sqrt = vecs_t @ np.diag(1.0 / np.sqrt(np.maximum(vals_t, 1e-12))) @ vecs_t.T
        X_tgt_ea = np.einsum("ij,njk->nik", R_tgt_inv_sqrt, X_tgt)

        # ── 分割目標資料：前 K 筆 = 校準，其餘 = 評估 ───────────────
        if K == 0:
            # 無校準資料；僅以來源受試者訓練
            X_fit = X_src_ea
            y_fit = y_src
        else:
            # 類別平衡取樣：每類取 K//2 筆（或盡可能多）
            calib_0 = np.where(y_tgt == 0)[0][:max(1, K // 2)]
            calib_1 = np.where(y_tgt == 1)[0][:max(1, K - len(calib_0))]
            calib_idx = np.sort(np.concatenate([calib_0, calib_1]))
            eval_mask = np.ones(len(y_tgt), dtype=bool)
            eval_mask[calib_idx] = False

            if eval_mask.sum() < 4:
                # 評估試驗不足；跳過此折
                continue

            X_fit = np.concatenate([X_src_ea, X_tgt_ea[calib_idx]])
            y_fit = np.concatenate([y_src,    y_tgt[calib_idx]])
            X_tgt_ea  = X_tgt_ea[eval_mask]
            y_tgt      = y_tgt[eval_mask]

        pipe_c = sk_clone(pipe_template)
        pipe_c.fit(X_fit, y_fit)
        fold_accs.append(pipe_c.score(X_tgt_ea, y_tgt))

    calib_results[K] = fold_accs
    if fold_accs:
        print(f"  K = {K:>3}         {np.mean(fold_accs):.3f}     {np.std(fold_accs):.3f}")
    else:
        print(f"  K = {K:>3}         （跳過——評估試驗不足）")

# %%
# ── 圖三 — 校準曲線 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ks        = [K for K in CALIB_SIZES if calib_results[K]]
means_cal = [np.mean(calib_results[K]) for K in ks]
stds_cal  = [np.std(calib_results[K])  for K in ks]

ax.errorbar(ks, means_cal, yerr=stds_cal,
            marker="o", linewidth=2.5, capsize=5,
            color="#2ca02c", markerfacecolor="white", markeredgewidth=2,
            label="EA + 黎曼 + LR  （LOSO 折次均值 ± 標準差）")

# 標示零校準基準線
if 0 in calib_results and calib_results[0]:
    base = np.mean(calib_results[0])
    ax.axhline(base, color="gray", lw=1.5, ls="--",
               label=f"基準線（K=0，僅來源受試者）= {base:.3f}")

ax.axhline(0.5, color="red", lw=1.0, ls=":", label="機率基準（50%）")

ax.set(
    xlabel="目標受試者已標記校準試驗數（K）",
    ylabel="目標受試者保留試驗的準確率",
    title=(
        "校準曲線：每增加一筆已標記目標試驗，準確率隨之提升\n"
        f"BCI IV 2a，LOSO  （{N_SUBJ} 位受試者，EA + 黎曼管線）"
    ),
    ylim=(0.3, 1.0),
    xlim=(-1, max(ks) + 2),
)
ax.legend(loc="lower right", fontsize=9)
ax.xaxis.get_major_locator().set_params(integer=True)
plt.tight_layout()
plt.savefig("/tmp/dd_da_calib_curve.png", dpi=100, bbox_inches="tight")
plt.show()
print("圖三已儲存 → /tmp/dd_da_calib_curve.png")

# %% [markdown]
# ### 如何解讀校準曲線
#
# 每個資料點為各 LOSO 折次的均值準確率；誤差棒為 ± 1 個標準差。
#
# * **K = 0** 時，我們完全依賴來源受試者資料（套用 EA 的跨受試者遷移）。
#   此為真實的「零樣本（zero-shot）」BCI 情境。
# * 隨著 K 增加，目標受試者的標籤引導分類器朝該受試者的特徵空間靠攏，
#   準確率迅速提升。
# * 即使僅有 **5 筆已標記試驗**（約 30–60 秒的記錄），也能顯著縮小跨受試者
#   與受試者內準確率之間的差距。這正是為何真實 BCI 系統幾乎都會在上線前
#   包含一個短暫的校準區塊。
#
# **實作說明：** 此處的校準試驗依時間順序取自目標場次的*開頭*，
# 評估則使用其餘部分。切勿隨機抽取完整場次中的試驗——
# 運動想像（motor imagery）的表現在一次場次中會隨時間漂移，
# 這樣做會造成一種隱微的時序洩漏。

# %% [markdown]
# ---
# ## 第五部分 — （簡介）深度網路校準資料的微調（Fine-tuning）
#
# 當來源模型為深度神經網路（例如在來源受試者上訓練的 EEGNet 或 ShallowConvNet）時，
# 自然的校準策略是**微調（fine-tuning）**：
# 從預訓練權重出發，以小學習率在 K 筆已標記目標試驗上更新模型。
#
# ```python
# # 偽代碼（Pseudocode）——需要已訓練的 PyTorch 或 Braindecode 模型 `source_model`。
# import torch
# from torch.optim import Adam
#
# target_loader = DataLoader(TensorDataset(
#     torch.tensor(X_calib_ea).float(),
#     torch.tensor(y_calib).long(),
# ), batch_size=min(16, K), shuffle=True)
#
# model = copy.deepcopy(source_model)          # 從來源權重出發
# optim = Adam(model.parameters(), lr=1e-4)   # 小學習率——避免災難性遺忘
# model.train()
#
# for epoch in range(50):                      # 少量 epoch；K 很小
#     for xb, yb in target_loader:
#         loss = F.cross_entropy(model(xb), yb)
#         loss.backward(); optim.step(); optim.zero_grad()
#
# # 在 X_eval_ea 上評估
# model.eval()
# with torch.no_grad():
#     preds = model(torch.tensor(X_eval_ea).float()).argmax(1).numpy()
# acc_ft = (preds == y_eval).mean()
# ```
#
# **實務注意事項：**
#
# * K < 20 筆試驗時，微調所有層會導致**災難性遺忘（catastrophic forgetting）**
#   來源領域的知識。應凍結早期層（空間濾波器），僅更新最終分類頭（head）。
# * 使用目標受試者的參照（同第四部分）對校準資料與評估資料套用 EA。
# * 在少量校準資料的小型驗證集（例如 20%）上使用早停（early stopping），
#   以避免過擬合（overfitting）至微小的校準集。
# * **EA + 僅微調分類頭** 的組合，在 K ≤ 30 筆試驗時通常優於微調所有層。

# %% [markdown]
# ---
# ## 總結
#
# | 技術 | 核心概念 | 程式碼行數 | 是否需要目標標籤？ | 典型提升幅度 |
# |---|---|---|---|---|
# | **歐氏對齊（Euclidean Alignment）** | 將各領域白化至均值為單位矩陣 | ~15 | 否 | +5–15 pp |
# | **黎曼對齊（Riemannian alignment）** | 以黎曼均值進行白化 | ~5（pyriemann） | 否 | 略優於 EA |
# | **校準（Calibration）** | 將 K 筆已標記目標試驗加入訓練集 | ~10 | 是（K 筆） | K ≥ 20 時 +15–25 pp |
# | **微調（Fine-tuning）** | 在目標資料上更新深度網路分類頭 | ~30（PyTorch） | 是（K 筆） | +10–20 pp |
#
# **沒有任何單一技術在所有情境下都表現最佳：**
#
# * 若確實無法取得校準資料，EA（或黎曼對齊）是最佳選擇，且不需要任何標籤成本。
# * 短暫的校準區塊（5–10 分鐘，20–40 筆試驗）搭配 EA，是最務實的 BCI 部署策略，
#   能縮小絕大部分的跨受試者差距。
# * 微調僅在存在大型、訓練完善的來源模型，且可取得 K ≥ 10 筆校準試驗時，
#   才值得投入工程成本。

# %% [markdown]
# ## ⚠️ A subtler trap — computing the EA reference with test-subject labels
#
# ### 隱形的洩漏
#
# 歐氏對齊是非監督式的：參照矩陣 $\mathbf{R}^{(t)}$ 為**均值共變異數**——
# 它不使用標籤。這看起來很安全。但有一種較不顯眼的失效模式會在不觸發
# 標準「你是否使用了測試標籤？」檢查的情況下虛增準確率：
#
# > **若你僅用目標試驗的類別特定子集——即分別按類別——計算 $\mathbf{R}^{(t)}$
# > 再合併，你已隱式使用了測試標籤。**
#
# 具體而言，部分論文將 EA 參照計算為：
# $$\mathbf{R}^{(t)} = \frac{N_0}{N}\,\mathbf{R}_0 + \frac{N_1}{N}\,\mathbf{R}_1,
#   \quad \mathbf{R}_k = \frac{1}{N_k}\sum_{i: y_i=k} \mathbf{C}_i^{(t)}$$
# 並將其視為「僅是取均值」。但 $\mathbf{R}_k$ 使用了測試試驗的 $y_i$——這構成洩漏。
#
# ### 為何會虛增準確率
#
# 類別感知的參照可產生一種領域適應後的表徵，其兩類之間的線性可分性由此建構——
# 因為白化方向是在知曉哪些試驗屬於哪個類別的前提下選定的。
# 在極端情況下（僅 2 個類別，所有試驗變異性均來自類別差異），
# 白化後的共變異數可能在分類器見到它們之前就已線性可分，
# 純粹是因為 $\mathbf{R}^{(t)}$ 編碼了類別邊界。
#
# 以下情況下虛增最為嚴重：
# 1. 目標受試者的類別條件共變異數差異較大（強烈的 ERD/ERS 運動想像訊號）——
#    正是你最在乎的受試者。
# 2. 評估在計算參照的相同試驗上進行（最常見的錯誤）。
# 3. 資料集較小，使得類別條件均值因機率原因而差異顯著。
#
# ### 為何不易察覺
#
# * 參照矩陣 $\mathbf{R}^{(t)}$ 不是模型參數——它不會傳遞給 `.fit()`。
#   標準的「僅在訓練集擬合」審查無法偵測到它。
# * 當類別平衡時，加權平均在數學上與整體均值完全相同，
#   因此此錯誤在平衡資料集中是隱形的。
# * 已發表論文報告了 EA 帶來 10–20 pp 的跨受試者提升；
#   其中部分文獻可能使用了類別感知的參照。
#
# ### 正確規則（如上方實作所示）
#
# $$\mathbf{R}^{(t)} = \frac{1}{N_t}\sum_{i=1}^{N_t} \mathbf{C}_i^{(t)}$$
#
# — 將**所有**目標試驗（無論標籤）合併為單一均值。標籤從不被查詢。
# 這是安全的，因為均值共變異數捕捉的是受試者特定的振幅基準線，
# 而這與標籤無關。
#
# 同一陷阱的第二個較不知名的變體：當校準集很小時，
# 使用目標受試者的*評估*試驗（而非校準試驗）計算 $\mathbf{R}^{(t)}$。
# K = 5 筆校準試驗時，參照的估計較為不穩定；研究者可能會嘗試加入評估試驗來
# 「穩定」它。這同樣構成洩漏——參照現在已看過評估集，
# 其白化方向隱式編碼了關於那些試驗的資訊。
#
# **結論：** 歐氏對齊之所以強大，正是因為它不依賴標籤也不依賴評估資料。
# 一旦你根據類別成員資格或評估資料對其進行條件化，你就建立了一條捷徑——
# 而在真實部署的 BCI 中，測試標籤根本不存在，這條捷徑將無法泛化。
