# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — CSP 幾何與推導（CSP Geometry & Derivation）
#
# 讀完本章，你將對 *CSP 為何有效* 有嚴謹的理解，而不只是知道 *它做了什麼*：
# Rayleigh 商（Rayleigh quotient）目標、廣義特徵值（generalized eigenvalue）化簡、
# 白化（whitening）加 PCA 的幾何觀點，以及一個讓每個步驟清晰可見的合成示範。
#
# > **先備知識：** 主要章節 03 與 07。
# > **難度：** 進階 ★★★★☆
# > **不受 5 分鐘 CPU 預算限制**（本章是選修支線）。

# %%
# --- 啟動程序：向上尋找 neuro101 套件，無論從本地端或 deep-dives/_src/ 執行皆可
#     （比 notebooks/_src/ 多一層父目錄）。
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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg

rng = np.random.default_rng(0)

# 在 CI 環境中使用乾淨的非互動式後端。
matplotlib.use("Agg")

print("numpy", np.__version__, " | scipy", linalg.__name__)

# %% [markdown]
# ---
# ## 1  目標函數：最大化 Rayleigh 商（Rayleigh quotient）
#
# 給定來自**類別 1** 與**類別 2** 的有標籤 EEG 試次，令
#
# $$\Sigma_1 = \mathbb{E}[\,x x^\top \mid \text{class 1}\,],\qquad
#   \Sigma_2 = \mathbb{E}[\,x x^\top \mid \text{class 2}\,]$$
#
# 為*跨試次平均*的空間共變異數矩陣（spatial covariance matrices，大小 $C \times C$，
# 其中 $C$ 為 EEG 通道數）。
#
# **空間濾波器（spatial filter）** $w \in \mathbb{R}^C$ 將多通道訊號投影為
# 純量通道 $z = w^\top x$。$z$ 在類別 $k$ 的變異數為
#
# $$\text{Var}_k(z) = w^\top \Sigma_k\, w.$$
#
# **CSP 目標：** 找到 $w$，使類別 1 的變異數**最大化**，同時使類別 2 的變異數**最小化**。
# 形式上，最大化 **Rayleigh 商**：
#
# $$\mathcal{R}(w) = \frac{w^\top \Sigma_1\, w}{w^\top \Sigma_2\, w}.$$
#
# *為何後來要取對數變異數（log-variance）？*  變異數比值在試次之間是乘法性的；
# 取對數將其映射為加法性、近似高斯分布的特徵，線性分類器（linear classifiers）
# 對此較為偏好。

# %% [markdown]
# ---
# ## 2  化簡為廣義特徵值問題（generalized eigenvalue problem）
#
# ### 建立穩定性條件（stationarity condition）
#
# 在 $\mathcal{R}(w)$ 的極大值處，梯度為零：
#
# $$\nabla_w \mathcal{R}(w) = 0
#   \implies
#   \frac{2\Sigma_1 w}{w^\top \Sigma_2 w}
#   - \frac{w^\top \Sigma_1 w}{(w^\top \Sigma_2 w)^2}\,2\Sigma_2 w = 0.$$
#
# 兩邊乘以 $(w^\top \Sigma_2 w)/2$，並令 $\lambda = \mathcal{R}(w)$：
#
# $$\boxed{\Sigma_1\, w = \lambda\, \Sigma_2\, w}$$
#
# 這就是**廣義特徵值問題**（GEP）。Rayleigh 商的每個穩定點——極大值、極小值或鞍點——
# 都是此 GEP 的特徵對。*極大值*對應**最大**特徵值 $\lambda_1$，
# 而*極小值*（另一方向最具鑑別力）對應**最小**特徵值 $\lambda_C$。
#
# ### 白化（whitening）加 PCA 的推導（關鍵步驟，不含繁瑣證明）
#
# **步驟 1 — 對複合共變異數（composite covariance）進行白化。**
# 令 $\Sigma_c = \Sigma_1 + \Sigma_2$。由於 $\Sigma_c \succ 0$（正定，在試次足夠的前提下），
# 寫出其特徵分解 $\Sigma_c = U D U^\top$，並形成白化矩陣
# $P = D^{-1/2} U^\top$。定義 $\tilde\Sigma_k = P\,\Sigma_k\,P^\top$。
#
# **步驟 2 — 將 GEP 轉化為普通特徵值問題。**
# 將 $w = P^\top v$ 代入 $\Sigma_1 w = \lambda \Sigma_2 w$，並在左側乘以 $P$，得到：
#
# $$\tilde\Sigma_1\, v = \lambda\,\tilde\Sigma_2\, v.$$
#
# 由於白化強制 $\tilde\Sigma_1 + \tilde\Sigma_2 = I$，我們有
# $\tilde\Sigma_2 = I - \tilde\Sigma_1$。GEP 隨即化為
#
# $$\tilde\Sigma_1\, v = \lambda (I - \tilde\Sigma_1)\, v
#   \implies \tilde\Sigma_1\, v = \frac{\lambda}{1+\lambda}\, v.$$
#
# 因此 $v$ 就只是 $\tilde\Sigma_1$ 的**普通特徵向量（ordinary eigenvector）**！
#
# **步驟 3 — 映射回原始空間。**
# 原始空間中的空間濾波器為 $w = P^\top v$；對應的
# **模式（pattern）**（訊號源在頭皮上的分布）為 $a = \Sigma_c w /
# (w^\top \Sigma_c w)$——不是 $w$ 本身（這是常見錯誤）。
#
# **幾何直觀（geometric intuition）：** 白化使兩個共變異數在複合度量下都變得「圓」，
# 然後對 $\tilde\Sigma_1$ 的普通 PCA 找到類別 1 最大變異數的軸——
# 這個軸同時也是類別 2 *最小*變異數的軸（因為兩者相加等於單位矩陣）。

# %%
# --- 數值驗證：GEP 對比 scipy.linalg.eigh --------------------------------
# scipy.linalg.eigh(A, B) 求解 A v = lambda B v（對稱 A、B）。
# 我們也驗證白化路徑能得到相同的特徵向量（差一個符號）。

def csp_filters(Sigma1: np.ndarray, Sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """透過 eigh 回傳 CSP 空間濾波器（W 的各列）與特徵值。

    eigh(Sigma1, Sigma1+Sigma2) 求解 Sigma1 w = lambda*(Sigma1+Sigma2)*w。
    這等價於 Rayleigh 商 Sigma1 / Sigma2，因為
    w'Sigma1 w / w'Sigma2 w = rho  <=>  Sigma1 w = rho Sigma2 w，
    而將分母正規化為 Sigma1+Sigma2 是數值上更穩定的重新參數化
    （特徵值落在 [0,1]，lambda_new = rho/(1+rho)）。
    """
    Sigma_c = Sigma1 + Sigma2
    # eigh 以升序回傳特徵值；v 的各列為特徵向量。
    vals, vecs = linalg.eigh(Sigma1, Sigma_c)
    # 最大特徵值 = 類別 1 最多的變異數；回傳兩端結果。
    return vals, vecs


# 快速健全性檢查：隨機 3x3 SPD 矩陣
A = rng.standard_normal((3, 3)); A = A @ A.T + np.eye(3)
B = rng.standard_normal((3, 3)); B = B @ B.T + np.eye(3)

vals_eigh, vecs_eigh = linalg.eigh(A, A + B)
# 直接透過 GEP Aw = lambda Bw
vals_gep, vecs_gep = linalg.eigh(A, B)

# Rayleigh 商應相符（差 rho/(1+rho) 的轉換）
rho_gep = vals_gep        # rho = 來自 A w = lambda B w 的 lambda
rho_eigh = vals_eigh / (1 - vals_eigh + 1e-15)   # 反推 lambda_new = rho/(1+rho)
print("GEP 特徵值 (Sigma1 w = λ Sigma2 w):", np.round(rho_gep, 4))
print("從 eigh 參數化還原：                 ", np.round(rho_eigh, 4))
print("最大絕對差異：", np.max(np.abs(np.sort(rho_gep) - np.sort(rho_eigh))))

# %% [markdown]
# 兩組特徵值吻合（差捨入誤差）。`scipy.linalg.eigh(Sigma1, Sigma1+Sigma2)`
# 的參數化正是 MNE 內部所採用的——特徵值落在 $[0,1]$，避免無界比值。

# %% [markdown]
# ---
# ## 3  視覺示範：2 通道、2 類別的合成資料
#
# 我們建構兩個類別，其共變異數矩陣在*旋轉方向*上不同，但共享相同的總體散布。
# 這模擬動作想像（motor imagery）：整體 EEG「能量」相同，但空間分布因想像的
# 手而異。

# %%
# --- 合成資料集 -------------------------------------------------------
n_trials = 200      # 每類別的試次數
n_times  = 400      # 每試次的取樣點數

# 類別 1：變異數沿方向 [cos θ, sin θ] 延伸
theta1 = np.deg2rad(30)
u1 = np.array([np.cos(theta1), np.sin(theta1)])
v1 = np.array([-np.sin(theta1), np.cos(theta1)])
# 特徵值：沿 u1 較大，沿 v1 較小（通道 1 功率較高）
D1 = np.diag([4.0, 0.5])
cov1 = u1[:, None] * D1[0, 0] * u1[None, :] + v1[:, None] * D1[1, 1] * v1[None, :]

# 類別 2：變異數沿方向 [cos φ, sin φ] 延伸
theta2 = np.deg2rad(-40)
u2 = np.array([np.cos(theta2), np.sin(theta2)])
v2 = np.array([-np.sin(theta2), np.cos(theta2)])
D2 = np.diag([4.0, 0.5])
cov2 = u2[:, None] * D2[0, 0] * u2[None, :] + v2[:, None] * D2[1, 1] * v2[None, :]

# 抽取試次：每筆試次 x 的形狀為 (2, n_times)；每試次估計共變異數。
# 為簡化起見，從類別共變異數（零均值高斯）中獨立抽取時間點。
L1 = np.linalg.cholesky(cov1)
L2 = np.linalg.cholesky(cov2)

X1 = (L1 @ rng.standard_normal((2, n_trials * n_times))).reshape(2, n_trials, n_times)
X2 = (L2 @ rng.standard_normal((2, n_trials * n_times))).reshape(2, n_trials, n_times)
# X_k 形狀：(channels=2, trials, times) → 轉置為 (trials, channels, times)
X1 = X1.transpose(1, 0, 2)   # (n_trials, 2, n_times)
X2 = X2.transpose(1, 0, 2)

# 從資料估計各類別的共變異數矩陣（跨試次平均 xx'）
Sigma1_hat = np.mean([x @ x.T / n_times for x in X1], axis=0)
Sigma2_hat = np.mean([x @ x.T / n_times for x in X2], axis=0)

print("估計的 Sigma1：\n", np.round(Sigma1_hat, 3))
print("真實的 Sigma1：\n", np.round(cov1, 3))

# %%
# --- 計算 CSP 濾波器 ----------------------------------------------------
vals, W = csp_filters(Sigma1_hat, Sigma2_hat)
# W 的各列為濾波器；eigh 以升序回傳，
# 因此最後一列 = 類別 1 最大 Rayleigh 濾波器，
# 第一列 = 最小 Rayleigh（= 類別 2 最大）。
w_max = W[:, -1]   # 類別 1 最多的變異數
w_min = W[:,  0]   # 類別 1 最少的變異數（類別 2 最多）

print(f"\nCSP 特徵值（在 [0,1] 內）：{vals.round(4)}")
print(f"濾波器 w_max（偏好類別 1）：{w_max.round(4)}")
print(f"濾波器 w_min（偏好類別 2）：{w_min.round(4)}")

# 驗證 Rayleigh 商
def rayleigh(w, S1, S2):
    return (w @ S1 @ w) / (w @ S2 @ w)

print(f"\nRayleigh(w_max) = {rayleigh(w_max, Sigma1_hat, Sigma2_hat):.4f}  "
      f"（應較大）")
print(f"Rayleigh(w_min) = {rayleigh(w_min, Sigma1_hat, Sigma2_hat):.4f}  "
      f"（應較小）")

# %% [markdown]
# ### 圖 1 — 資料點雲 + CSP 濾波器方向
#
# 橢圓顯示二維通道空間中每個類別的共變異數結構。
# CSP 濾波器（箭頭）是透過變異數最大程度分離類別的方向。

# %%
def cov_ellipse(cov, center=(0, 0), n_std=2.0, n_pts=200):
    """回傳代表二維共變異數矩陣的橢圓 (x, y) 座標。"""
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.array([np.cos(t), np.sin(t)])
    L = np.linalg.cholesky(cov)
    ellipse = n_std * (L @ circle)
    return ellipse[0] + center[0], ellipse[1] + center[1]


# 抽取少數具代表性的試次均值點用於散點圖（每試次一個點 = x 的均值）
mean1 = X1.mean(axis=-1)   # (n_trials, 2)
mean2 = X2.mean(axis=-1)

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(mean1[:, 0], mean1[:, 1], s=8, alpha=0.3,
           color="#4c72b0", label="類別 1 試次均值")
ax.scatter(mean2[:, 0], mean2[:, 1], s=8, alpha=0.3,
           color="#dd8452", label="類別 2 試次均值")

# 共變異數橢圓
ex1, ey1 = cov_ellipse(Sigma1_hat)
ex2, ey2 = cov_ellipse(Sigma2_hat)
ax.plot(ex1, ey1, color="#4c72b0", lw=2, label="類別 1 共變異數橢圓 (2σ)")
ax.plot(ex2, ey2, color="#dd8452", lw=2, label="類別 2 共變異數橢圓 (2σ)")

# CSP 濾波器箭頭
scale = 2.5
ax.annotate("", xy=w_max * scale, xytext=-w_max * scale,
            arrowprops=dict(arrowstyle="<->", color="green", lw=2.5))
ax.annotate("", xy=w_min * scale, xytext=-w_min * scale,
            arrowprops=dict(arrowstyle="<->", color="purple", lw=2.5))

ax.text(*(w_max * (scale + 0.3)), "w_max\n(↑ 類別 1 變異數)", color="green",
        ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(*(w_min * (scale + 0.3)), "w_min\n(↑ 類別 2 變異數)", color="purple",
        ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_aspect("equal")
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("通道 1", fontsize=11)
ax.set_ylabel("通道 2", fontsize=11)
ax.set_title("二維 EEG 通道空間：共變異數橢圓 + CSP 濾波器方向",
             fontsize=11)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig1.png", dpi=120)
plt.show()
print("圖 1 已儲存。")

# %% [markdown]
# **閱讀圖 1。** 綠色雙向箭頭（`w_max`）穿過*藍色*（類別 1）橢圓的長軸，
# 以及橙色（類別 2）橢圓的*短*軸。將任何資料點投影到此軸上，
# 得到一個純量，其變異數對類別 1 *高*，對類別 2 *低*——這正是 Rayleigh 商目標。
# 紫色箭頭（`w_min`）則相反。
#
# 兩個濾波器都不與任何原始通道軸對齊；它們是針對鑑別性最佳化的*斜交*（oblique）混合。
# 這是 CSP 相對於單純選取一個電極的核心優勢。

# %%
# --- 將試次投影到 CSP 濾波器上並計算對數變異數特徵 -------
def logvar_features(X_class, w):
    """將每個試次投影到 w 上，回傳時間軸上的對數變異數。

    X_class : (n_trials, n_channels, n_times)
    w       : (n_channels,)
    回傳    : (n_trials,) 的對數變異數
    """
    # 對每個試次計算 w @ x[trial] -> (n_trials, n_times)
    z = np.einsum("c,tcs->ts", w, X_class)
    return np.log(z.var(axis=1) + 1e-12)


lv1_max = logvar_features(X1, w_max)   # 類別 1，濾波器 w_max
lv2_max = logvar_features(X2, w_max)   # 類別 2，濾波器 w_max
lv1_min = logvar_features(X1, w_min)   # 類別 1，濾波器 w_min
lv2_min = logvar_features(X2, w_min)   # 類別 2，濾波器 w_min

# 組合為特徵矩陣：[logvar(w_max), logvar(w_min)]
F1 = np.column_stack([lv1_max, lv1_min])   # (n_trials, 2)
F2 = np.column_stack([lv2_max, lv2_min])

# %% [markdown]
# ### 圖 2 — 對數變異數特徵散點圖（分類器的視角）
#
# 投影到兩個 CSP 濾波器後，對數變異數在二維特徵空間中清晰地分離了兩個類別。
# 此處線性分類器（虛線）幾乎可以完美分割，即使原始二維通道資料有大量重疊。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左圖：每個濾波器的一維分布
ax = axes[0]
bins = np.linspace(-2, 4, 50)
ax.hist(lv1_max, bins=bins, alpha=0.6, color="#4c72b0", label="類別 1")
ax.hist(lv2_max, bins=bins, alpha=0.6, color="#dd8452", label="類別 2")
ax.set(xlabel="對數變異數（w_max 投影）", ylabel="計數",
       title="濾波器 w_max：依類別分離的變異數")
ax.legend()
ax.axvline(0.5 * (lv1_max.mean() + lv2_max.mean()), color="k", ls="--",
           label="中點")

# 右圖：CSP 特徵空間的二維散點圖
ax = axes[1]
ax.scatter(F1[:, 0], F1[:, 1], s=12, alpha=0.5, color="#4c72b0", label="類別 1")
ax.scatter(F2[:, 0], F2[:, 1], s=12, alpha=0.5, color="#dd8452", label="類別 2")

# 簡單線性決策邊界（類別均值的垂直平分線）
m1, m2 = F1.mean(0), F2.mean(0)
mid = 0.5 * (m1 + m2)
direction = m2 - m1
normal = np.array([-direction[1], direction[0]])
t_vals = np.linspace(-3, 3, 100)
boundary = mid[:, None] + normal[:, None] * t_vals
ax.plot(boundary[0], boundary[1], "k--", lw=1.5, label="線性邊界")
ax.set(xlabel="log-var(w_max)  [類別 1 ↑]",
       ylabel="log-var(w_min)  [類別 2 ↑]",
       title="CSP 特徵空間：類別線性可分")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig2.png", dpi=120)
plt.show()
print("圖 2 已儲存。")

# %% [markdown]
# **與第三章的連結。** 第三章中我們看到共變異數矩陣的特徵向量指向最大變異數的方向。
# CSP 將此延伸：它不是對角化*一個*共變異數，而是透過廣義特徵問題同時對角化*兩個*。
# 所得濾波器是同時使類別 1 看起來「大」、使類別 2 看起來「小」的軸。

# %% [markdown]
# ---
# ## 4  白化（whitening）觀點 — CSP 即白化加 PCA（幾何視角）
#
# §2 的推導揭示了一個清晰的幾何流程：
#
# ```
# 原始通道空間
#       │
#       │  對 Σ_c = Σ₁ + Σ₂ 進行白化
#       │  （使兩個類別在複合度量下都變「圓」）
#       ▼
# 白化空間  (Σ̃₁ + Σ̃₂ = I)
#       │
#       │  對 Σ̃₁ 進行普通 PCA
#       │  （第一主成分 = 類別 1 變異數最大的方向）
#       ▼
# CSP 成分空間
# ```
#
# 白化**均衡**各方向的總功率，消除與類別區分無關的通道振幅差異。
# 然後對 $\tilde\Sigma_1$ 的 PCA 找到鑑別軸。
#
# 由於 $\tilde\Sigma_2 = I - \tilde\Sigma_1$，最大化類別 1 變異數的主成分
# *同時最小化*類別 2 的變異數。這就是 CSP 在兩個方向上同時最佳的代數原因。

# %%
# --- 視覺化白化步驟 -------------------------------------------
# 對樣本點雲做 Sigma_c 的白化，然後驗證 Sigma_tilde_1 + Sigma_tilde_2 = I。

Sigma_c = Sigma1_hat + Sigma2_hat
vals_c, U_c = np.linalg.eigh(Sigma_c)
P = (U_c / np.sqrt(vals_c)[None, :]).T          # 白化矩陣 (D^{-1/2} U')

Sigma1_tilde = P @ Sigma1_hat @ P.T
Sigma2_tilde = P @ Sigma2_hat @ P.T
identity_check = Sigma1_tilde + Sigma2_tilde

print("Σ̃₁ + Σ̃₂（應為單位矩陣）：\n", np.round(identity_check, 6))

# 對 Sigma1_tilde 求特徵向量（白化類別 1 共變異數的普通 PCA）
vals_t, V = np.linalg.eigh(Sigma1_tilde)
print(f"\nΣ̃₁ 的特徵值：{vals_t.round(4)}  (總和={vals_t.sum():.4f})")
print(f"Σ̃₂ 的特徵值：{(1-vals_t).round(4)}  (總和={(1-vals_t).sum():.4f})")

# 將特徵向量映射回原始空間
W_whitening_route = P.T @ V    # 各列 = CSP 濾波器（白化路徑）

# 與 scipy.linalg.eigh 濾波器比較（應差一個符號）
cos_sim = np.abs(np.diag(W_whitening_route.T @ W))
print(f"\n兩條路徑濾波器的餘弦相似度：{cos_sim.round(6)}")
print("（1.0 = 差一個符號的相同，如預期）")

# %% [markdown]
# 兩條路徑——直接 `scipy.linalg.eigh` 與白化加 PCA——給出相同的空間濾波器
# （餘弦相似度 = 1.0）。實作任何一種都等價於另一種；白化路徑幾何上更透明，
# 而 `eigh` 在數值上更優，因為它使用專用求解器。

# %%
# --- 圖 3：白化空間與原始空間 -----------------------------
# 顯示白化前後兩類別的點雲。

# 抽取少數具代表性的點（試次均值）
pts1 = mean1   # (n_trials, 2)
pts2 = mean2

# 白化
pts1_w = (P @ pts1.T).T
pts2_w = (P @ pts2.T).T

Sigma1_w_est = np.cov(pts1_w.T)
Sigma2_w_est = np.cov(pts2_w.T)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# 原始空間
ax = axes[0]
ax.scatter(pts1[:, 0], pts1[:, 1], s=8, alpha=0.3, color="#4c72b0")
ax.scatter(pts2[:, 0], pts2[:, 1], s=8, alpha=0.3, color="#dd8452")
ex1o, ey1o = cov_ellipse(Sigma1_hat)
ex2o, ey2o = cov_ellipse(Sigma2_hat)
ax.plot(ex1o, ey1o, "#4c72b0", lw=2, label="類別 1 橢圓")
ax.plot(ex2o, ey2o, "#dd8452", lw=2, label="類別 2 橢圓")
# 繪製軸
for label, vec, col in [("w_max", w_max, "green"), ("w_min", w_min, "purple")]:
    ax.annotate("", xy=vec * 2, xytext=-vec * 2,
                arrowprops=dict(arrowstyle="<->", color=col, lw=2))
ax.set_aspect("equal"); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
ax.set_title("原始通道空間", fontsize=11)
ax.set_xlabel("通道 1"); ax.set_ylabel("通道 2")
ax.legend(fontsize=8)

# 白化空間
ax = axes[1]
ax.scatter(pts1_w[:, 0], pts1_w[:, 1], s=8, alpha=0.3, color="#4c72b0")
ax.scatter(pts2_w[:, 0], pts2_w[:, 1], s=8, alpha=0.3, color="#dd8452")
if Sigma1_w_est.shape == (2, 2):
    ex1w, ey1w = cov_ellipse(Sigma1_w_est)
    ex2w, ey2w = cov_ellipse(Sigma2_w_est)
    ax.plot(ex1w, ey1w, "#4c72b0", lw=2, label="類別 1 橢圓（白化後）")
    ax.plot(ex2w, ey2w, "#dd8452", lw=2, label="類別 2 橢圓（白化後）")
# 在白化空間繪製 Sigma1_tilde 的 PCA 方向
for label, vec, col in [("PC1(Σ̃₁)=w_max", V[:, -1], "green"),
                         ("PC2(Σ̃₁)=w_min", V[:,  0], "purple")]:
    ax.annotate("", xy=vec * 1.5, xytext=-vec * 1.5,
                arrowprops=dict(arrowstyle="<->", color=col, lw=2))
# 繪製單位圓（白化後複合共變異數為單位矩陣）
theta_c = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta_c), np.sin(theta_c), "k:", lw=1, label="單位圓（Σ_c = I）")
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
ax.set_title("白化空間 (Σ_c = I)\nCSP = Σ̃₁ 的普通 PCA", fontsize=11)
ax.set_xlabel("白化維度 1"); ax.set_ylabel("白化維度 2")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig3.png", dpi=120)
plt.show()
print("圖 3 已儲存。")

# %% [markdown]
# **閱讀圖 3。** 左圖中，兩個類別橢圓傾斜方向不同——原始通道資料的功率依方向而異。
# 白化後（右圖），複合共變異數變成圓形（虛線單位圓），均衡了所有方向。
# 類別 1 橢圓（藍色）仍然延伸——只是現在的延伸方向*純粹*關乎類別鑑別。
# 白化後類別 1 共變異數的 PCA 軸（綠色／紫色箭頭）就是 CSP 濾波器。

# %% [markdown]
# ---
# ## 5  選修：MNE CSP 在 BCI IV 2a 真實資料上（快速版）
#
# 以下部分載入幾位真實受試者的資料，以確認幾何圖像在真實 EEG 上同樣適用。
# 使用 `NEURO101_SMOKE=1` 感知，在 CI 環境中保持快速執行。

# %%
import os
from neuro101 import io, datasets as ds

SMOKE = ds.is_smoke()
n_subj = 1 if SMOKE else 2    # 保持小規模；這是支線任務，不是基準測試

print(f"正在載入 {n_subj} 位受試者的 BCI IV 2a 資料（SMOKE={SMOKE}）……")
X_real, y_real, subj_real = io.load_bnci_2a_epochs(n_subjects=n_subj)
sf = ds.DATASETS["bnci_2a"].sfreq_hz
print(f"X={X_real.shape}  y 類別={np.bincount(y_real)}  取樣率={sf} Hz")

# %%
from neuro101.features import make_csp

csp_mne = make_csp(n_components=4)
F_csp_real = csp_mne.fit_transform(X_real, y_real)
print("MNE CSP 對數變異數特徵：", F_csp_real.shape)

# %%
# --- 圖 4：真實資料上的 MNE CSP 散點圖 ---------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
colors = {0: "#4c72b0", 1: "#dd8452"}
for cls in np.unique(y_real):
    mask = y_real == cls
    ax.scatter(F_csp_real[mask, 0], F_csp_real[mask, -1],
               s=15, alpha=0.55, color=colors[cls],
               label=f"類別 {cls} ({'左' if cls==0 else '右'}手)")

ax.set(xlabel="CSP 成分 1（對數變異數，↑ 類別 1 變異數）",
       ylabel="CSP 成分 4（對數變異數，↑ 類別 2 變異數）",
       title=f"MNE CSP 在 BCI IV 2a 上（{n_subj} 位受試者）\n"
             f"[僅示範用途，對全部資料擬合——見 §6 的陷阱]")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/csp_geometry_fig4.png", dpi=120)
plt.show()
print("圖 4 已儲存。")

# %% [markdown]
# 兩個 CSP 對數變異數特徵即使沒有分類器也已顯示類別分離。
# 散點圖與合成示範相符：成分 1 對左手想像較高、對右手較低
# （頂部 CSP 成分捕獲對側的 ERD 非對稱性），而成分 4 則相反。
#
# **模式（patterns）與濾波器（filters）——再一個細節。** 空間*濾波器* $w$ 是你乘以資料的向量。
# 空間*模式* $a$（你正在隔離的腦部訊號源在頭皮的分布）是 $(W^{-1})^\top$ 的對應列，
# 或等價地 $\Sigma_c w / (w^\top \Sigma_c w)$。在地形圖（topomap）上繪製 $w$ 是**錯的**；
# 繪製 $a$ 才正確。MNE 的 `CSP.patterns_` 已為你儲存了 $a$。

# %% [markdown]
# ---
# ## 6  ⚠️  一個更隱蔽的陷阱：試次稀少時 CSP 會悄悄過擬合
#
# 大多數從業者發現得太晚的陷阱，並不是顯而易見的「對所有資料擬合 CSP」的錯誤
# （那是第 12 章的資料洩漏，已在第 07 章標記）。
# 更隱蔽的陷阱關係到**選取的 CSP 成分數量**，而且它在一個其他方面正確的
# 交叉驗證（cross-validation）迴圈*內部*發生。
#
# ### 問題：特徵譜（eigenspectrum）的變異數
#
# CSP 求解一個基於樣本的特徵值問題。在 $T$ 次試次和 $C$ 個通道的情況下，
# 樣本共變異數 $\hat\Sigma_k$ 具有 $O(\sqrt{C/T})$ 量級的估計噪聲。
# 當 $C$ 較大（BCI IV 2a 有 22 個通道，PhysioNet 有 64 個通道）
# 且 $T$ 較小（每位受試者幾十次試次）時，特徵向量的*順序*可能翻轉：
# 在這個折（fold）的訓練集中恰好捕獲最多類別 1 變異數的濾波器，
# 可能追蹤的是噪聲方向，而非真實的動作想像訊號源。
#
# ### 揭露問題的實驗
#
# 一種常見工作流程（錯誤但容易犯）：
#
# 1. 在 CV 折中對所有有標籤訓練資料擬合 CSP。
# 2. **選取**在測試集 CSP 對數變異數分離上表現最好的 $k$ 個成分（從所有 $C$ 個中選）——
#    也就是用測試集表現來選擇 $k$。
# 3. 報告最佳測試準確率。
#
# 步驟 2 偷看了測試集並誇大了結果。正確的方法是：(a) 在迴圈前根據生物學依據固定 $k$
# （例如，始終使用 2 個極端成分），或 (b) 透過*僅使用訓練折*的內部 CV 迴圈來選取 $k$。
#
# ### 為何這並不明顯
#
# - 模型仍然只在訓練資料上擬合——標準的「無洩漏」檢查通過了。
# - 偏誤只出現在**模型選擇**步驟，感覺像超參數選擇，而不是資料切分。
# - 有 22 個通道就有 22 個 CSP 成分；當試次稀少時，
#   至少一個噪聲方向純粹偶然在某個保留折上「表現良好」的概率很高。
# - 正規化 CSP 變體（Ledoit-Wolf 收縮 $\hat\Sigma_k$，
#   `mne.decoding.CSP(reg="ledoit_wolf")`）降低了估計噪聲，
#   但不能消除選擇偏誤（selection bias）——它們仍然需要與適當的
#   內部折成分選取結合使用。
#
# ### 正確的配方
#
# ```python
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from mne.decoding import CSP
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# pipe = Pipeline([
#     ("csp", CSP(reg="ledoit_wolf", log=True)),
#     ("lda", LinearDiscriminantAnalysis()),
# ])
#
# # 內部 CV 選取 n_components；外部 CV 估計泛化能力。
# param_grid = {"csp__n_components": [2, 4, 6, 8]}
# inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#
# # 外部迴圈（對 GridSearchCV 物件呼叫 cross_val_score）：
# # scores = cross_val_score(GridSearchCV(pipe, param_grid, cv=inner_cv),
# #                          X, y, cv=outer_cv)
# ```
#
# 外部迴圈只看到*內部優化*管線的測試表現；內部迴圈絕不接觸外部測試折。
# 這種嵌套交叉驗證（nested CV）增加了計算量（對所示網格為 25 次擬合），
# 但在調整 $k$ 時，這是報告準確率的唯一誠實方式。
#
# **結論：** CSP 的隱性過擬合不在空間濾波器的擬合本身，
# 而在維度選擇上。把 CSP 成分數量視為超參數，
# 用內部交叉驗證迴圈選取它——而不是偷看測試表現。
