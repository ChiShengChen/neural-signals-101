# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深入探討 — 來源定位（逆問題）
#
# *頭皮訊號來自大腦的哪個位置？*
#
# > **先修條件：** 主要章節 01 和 02。
# > **程度：** 進階 ★★★★☆
# > **下載 fsaverage 模板（已快取）。僅限無頭式 2-D 視覺化。**

# %% [markdown]
# ## 0 — 引導程序（Bootstrap）

# %%
# --------------------------------------------------------------------------- #
# 引導程序 — 無論從儲存庫根目錄或 deep-dives/_src/ 執行，均可找到 neuro101
# --------------------------------------------------------------------------- #
import sys
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

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 非互動式後端 — 在 nbconvert 下安全執行
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds

SMOKE = ds.is_smoke()
RNG = np.random.default_rng(42)

print(f"MNE {mne.__version__}  |  smoke={SMOKE}  |  imports OK")

# %% [markdown]
# ---
# ## 1 — 全局概觀：正問題（forward）vs. 逆問題（inverse）
#
# ### 容積傳導（volume conduction）：為何 EEG 影像模糊
#
# 第 01 章說明了每個頭皮電極都是許多底層神經電流的**空間平均值**。頭骨、腦脊液（cerebrospinal fluid）與頭皮組織是歐姆導體（ohmic conductors）——它們像霧一樣擴散電流。單一皮質區塊（一個「偶極子（dipole）」）的電位會擴散至整個頭部。反之，單一電極同時接收數十個皮質區塊的貢獻。
#
# ### 正問題（sources → scalp）— 適定問題（well-posed）
#
# 若我們*已知*每個皮質區塊的位置與激發強度，就能利用馬克士威方程組（Maxwell's equations）精確計算預期的頭皮電位。MNE 將此稱為**正向解（forward solution）**或**導場矩陣（leadfield matrix）** **L** ∈ ℝ^{n_channels × 3·n_sources}：
#
# $$\mathbf{x}(t) = \mathbf{L}\, \mathbf{j}(t) + \boldsymbol{\varepsilon}(t)$$
#
# 其中 **x** 是電極資料、**j** 是來源振幅向量，**ε** 是雜訊。已知 **j** 時，計算 **x** 具有唯一性且穩定——正問題是*適定的*。
#
# ### 逆問題（scalp → sources）— 不適定問題（ill-posed）
#
# 我們觀測到 **x** 並想求 **j**。這個系統*高度欠定（underdetermined）*：64 個電極 vs. ≥10,000 個皮質來源位置。無限多種來源組合能產生完全相同的頭皮地形圖。這就是**不適定逆問題（ill-posed inverse problem）**。
#
# **正則化（Regularization）**透過加入先驗（prior）打破僵局：最小範數估計（minimum-norm estimate, MNE）偏好總來源功率最小的解。dSPM 以雜訊估計值正規化 MNE，使地圖成為類似 *z*-score 的量。sLORETA 與 LCMV 波束成形器（beamformer）則加入不同形式的先驗。這些方法都無法還原「真實值（ground truth）」——它們還原的是與資料*及*先驗一致的*最簡約解（most parsimonious solution）*。

# %% [markdown]
# ---
# ## 2 — 建立各元件（fsaverage 模板）
#
# 我們使用 FreeSurfer 的 **fsaverage** 模板大腦——無需受試者 MRI。MNE 預先建置了 fsaverage 的來源空間與 BEM 解，因此只需呼叫一次 `fetch_fsaverage()`（下載一次後即快取）。
#
# 為了進行定位的資料，我們在標準 10-20 電極配置上建構一個乾淨的**模擬誘發反應（simulated evoked response）**。這是最穩健的無頭式選擇：除了 fsaverage 模板外無需網路連線，無 BIDS 複雜性，也無 EEG 硬體特殊問題。模擬波形包含一個約在 300 ms 達到峰值的類 P300 成分。

# %%
# ------------------------------------------------------------------ #
# 2a. 取得 fsaverage（下載一次後快取至 ~/mne_data/）               #
# ------------------------------------------------------------------ #
_FSAVERAGE_OK = False
_FWD_OK = False
stc = None
stc_mne_sol = None
evoked = None

try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = str(Path(fs_dir).parent)
    print(f"fsaverage directory: {fs_dir}")
    _FSAVERAGE_OK = True
except Exception as e:
    print(f"[WARNING] Could not fetch fsaverage: {e}")
    print("         Falling back to simulated-only mode.")

# %%
# ------------------------------------------------------------------ #
# 2b. 在 standard_1020 電極配置上建立模擬誘發反應                  #
# ------------------------------------------------------------------ #
montage = mne.channels.make_standard_montage("standard_1020")
# 使用 32 個通道以提升速度；完整 10-20 配置同樣適用。
_CH_NAMES = montage.ch_names[:32]
_SFREQ = 250.0
_N_TIMES = 62 if SMOKE else 125   # 煙霧測試 250 ms，一般執行 500 ms
_T = np.linspace(0.0, _N_TIMES / _SFREQ, _N_TIMES)

# 模擬資料：白雜訊 + 300 ms 處的類 P300 高斯凸起
_data = RNG.standard_normal((len(_CH_NAMES), _N_TIMES)) * 1e-7
for _i in range(len(_CH_NAMES)):
    _amp = 3e-6 * (1.0 + 0.3 * RNG.standard_normal())
    _data[_i] += _amp * np.exp(-((_T - 0.30) ** 2) / (2 * 0.04 ** 2))

info = mne.create_info(ch_names=_CH_NAMES, sfreq=_SFREQ, ch_types="eeg")
evoked = mne.EvokedArray(_data, info, tmin=0.0, comment="simulated P300-like ERP")
evoked.set_montage(montage)
# EEG 平均參考是逆算子所必需的
evoked.set_eeg_reference(projection=True)

print(
    f"Evoked: {evoked.info['nchan']} channels × {evoked.times.size} time points  "
    f"({evoked.tmin:.2f} – {evoked.tmax:.2f} s)"
)

# %%
# ------------------------------------------------------------------ #
# 2c. 來源空間 + BEM + 正向解                                       #
# ------------------------------------------------------------------ #
if _FSAVERAGE_OK:
    try:
        _bem_dir = os.path.join(fs_dir, "bem")
        _src_file = os.path.join(_bem_dir, "fsaverage-ico-5-src.fif")
        _bem_sol_file = os.path.join(_bem_dir, "fsaverage-5120-5120-5120-bem-sol.fif")
        _trans_file = os.path.join(_bem_dir, "fsaverage-trans.fif")

        src = mne.read_source_spaces(_src_file)
        bem = mne.read_bem_solution(_bem_sol_file)
        print(
            f"Source space: {src[0]['nuse'] + src[1]['nuse']:,} vertices  |  "
            f"BEM: {len(bem['surfs'])} surfaces"
        )

        fwd = mne.make_forward_solution(
            evoked.info,
            trans=_trans_file,
            src=src,
            bem=bem,
            eeg=True,
            meg=False,
            verbose=False,
        )
        print(f"Forward solution: {fwd['nsource']:,} sources × {fwd['nchan']} channels")
        _FWD_OK = True
    except Exception as e:
        print(f"[WARNING] Forward solution failed: {e}")
        print("         Falling back to simulated-stc mode.")

# %%
# ------------------------------------------------------------------ #
# 2d. 雜訊共變異數 + 逆算子 + dSPM / MNE                           #
# ------------------------------------------------------------------ #
if _FWD_OK:
    try:
        # 特設雜訊共變異數（對角線，依感測器類型調整）
        noise_cov = mne.make_ad_hoc_cov(evoked.info, verbose=False)

        inv_op = mne.minimum_norm.make_inverse_operator(
            evoked.info, fwd, noise_cov, verbose=False
        )
        print("Inverse operator built successfully")

        _lambda2 = 1.0 / 9.0  # SNR = 3 → 標準正則化選擇
        stc = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2=_lambda2, method="dSPM", verbose=False
        )
        stc_mne_sol = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2=_lambda2, method="MNE", verbose=False
        )
        print(f"dSPM SourceEstimate: {stc.data.shape[0]:,} vertices × {stc.data.shape[1]} times")
        print(
            f"dSPM range: {stc.data.min():.2f} – {stc.data.max():.2f} (pseudo-z scores)"
        )
    except Exception as e:
        print(f"[WARNING] Inverse solution failed: {e}")

# 備用方案：若 MNE 逆運算失敗，生成虛擬的 stc 類似結構
if stc is None:
    print("[INFO] Using synthetic SourceEstimate for visualization.")
    _n_src = 500
    _stc_data = RNG.standard_normal((_n_src, _N_TIMES)) * 0.5
    # 在少數頂點的 ~300 ms 處加入一個凸起
    for _vi in [10, 42, 99]:
        _stc_data[_vi] += 5.0 * np.exp(-((_T - 0.30) ** 2) / (2 * 0.04 ** 2))

    class _FakeStc:
        data = _stc_data
        times = _T

    stc = _FakeStc()
    stc_mne_sol = _FakeStc()

# %% [markdown]
# ---
# ## 3 — dSPM vs. MNE：有何差異？
#
# 兩種方法解的是相同的欠定最小二乘問題，差異僅在後處理：
#
# | 方法 | 公式 | 解釋 |
# |--------|---------|----------------|
# | **MNE** | $\hat{\mathbf{j}} = \mathbf{W} \mathbf{x}$，其中 **W** 最小化 $\|\mathbf{j}\|^2$ | 電流振幅（A·m）；偏向表層來源 |
# | **dSPM** | MNE ÷ 每個來源的估計雜訊標準差 | 偽 *z* 分數；空間選擇性優於原始 MNE |
# | **sLORETA** | MNE ÷ 估計來源變異數 | 單一偶極子的精確零誤差定位 |
#
# **正則化參數 λ²** 控制雜訊與平滑度之間的取捨。常見選擇為 λ² = 1/SNR²，SNR = 3 → λ² ≈ 0.11。
# *較大的 λ²* → 更平滑（更擴散），雜訊較少的地圖。
# *較小的 λ²* → 更銳利但會放大雜訊。

# %% [markdown]
# ---
# ## 4 — 視覺化（僅限無頭式 2-D）
#
# `stc.plot()` 會開啟 3-D VTK/PyVista 視窗，在 nbconvert 下會崩潰。
# 我們改以 2-D 繪製來源估計：來源 × 時間影像、峰值頂點時間序列，以及錨定詮釋的頭皮地形圖。

# %%
# ------------------------------------------------------------------ #
# 圖 1 — 模擬誘發反應的頭皮地形圖                                   #
# ------------------------------------------------------------------ #
_times_plot = [t for t in [0.10, 0.20, 0.30, 0.40] if t <= evoked.tmax]
fig1 = evoked.plot_topomap(times=_times_plot, show=False)
fig1.suptitle(
    "Simulated P300-like ERP — scalp topomaps\n"
    "(input to the source localization pipeline)",
    fontsize=10, y=1.02,
)
# 注意：plot_topomap 管理其自身的版面引擎（constrained）；不可對其呼叫
# tight_layout()，否則會設定第二個衝突的引擎。
plt.show()
print("Figure 1: scalp topomaps rendered OK")

# %%
# ------------------------------------------------------------------ #
# 圖 2 — 來源 × 時間影像（dSPM）                                    #
# ------------------------------------------------------------------ #
# 僅顯示前 500 個頂點使影像保持可讀性；
# 完整的 ico-5 解有 20 484 個皮質頂點。
_n_show = min(500, stc.data.shape[0])
_vmax = np.percentile(np.abs(stc.data[:_n_show]), 99)

fig2, axes2 = plt.subplots(
    1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]},
    layout="constrained",
)

# 左圖：來源振幅的熱圖
im = axes2[0].imshow(
    stc.data[:_n_show],
    aspect="auto",
    origin="lower",
    extent=[float(stc.times[0]), float(stc.times[-1]), 0, _n_show],
    cmap="RdBu_r",
    vmin=-_vmax,
    vmax=_vmax,
)
axes2[0].axvline(x=0.30, color="k", linestyle="--", linewidth=0.8, label="t = 300 ms")
axes2[0].set_xlabel("Time (s)", fontsize=11)
axes2[0].set_ylabel(f"Cortical vertex index (first {_n_show})", fontsize=11)
axes2[0].set_title("dSPM source estimates", fontsize=12)
axes2[0].legend(fontsize=9)
fig2.colorbar(im, ax=axes2[0], label="dSPM (pseudo-z)")

# 右圖：峰值時間振幅的直方圖
_peak_t_idx = int(np.argmin(np.abs(stc.times - 0.30)))
_peak_amps = stc.data[:, _peak_t_idx]
axes2[1].hist(
    _peak_amps, bins=40, orientation="horizontal", color="steelblue", edgecolor="white", linewidth=0.4
)
axes2[1].axhline(0, color="k", linewidth=0.6)
axes2[1].set_xlabel("Count", fontsize=10)
axes2[1].set_ylabel("dSPM amplitude at t = 300 ms", fontsize=10)
axes2[1].set_title("Amplitude\ndistribution", fontsize=11)

fig2.suptitle(
    "Source space view of the dSPM inverse solution\n"
    f"({stc.data.shape[0]:,} cortical vertices, λ² = 1/9)",
    fontsize=11,
)
plt.show()
print("Figure 2: source × time image rendered OK")

# %%
# ------------------------------------------------------------------ #
# 圖 3 — 峰值頂點時間序列：dSPM vs. MNE                            #
# ------------------------------------------------------------------ #
_top_k = 5
_peak_vertices = np.argsort(np.abs(stc.data).max(axis=1))[-_top_k:]

fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

_colors = plt.cm.tab10(np.linspace(0, 0.5, _top_k))

for _rank, _vi in enumerate(_peak_vertices[::-1]):
    _lbl = f"vertex {_vi}" if _rank > 0 else f"vertex {_vi} (peak)"
    axes3[0].plot(stc.times, stc.data[_vi], color=_colors[_rank], label=_lbl, linewidth=1.2)
    axes3[1].plot(stc_mne_sol.times, stc_mne_sol.data[_vi], color=_colors[_rank], linewidth=1.2)

axes3[0].axvline(0.30, color="k", linestyle="--", linewidth=0.8)
axes3[1].axvline(0.30, color="k", linestyle="--", linewidth=0.8, label="t = 300 ms")
axes3[0].set_ylabel("dSPM (pseudo-z)", fontsize=10)
axes3[1].set_ylabel("MNE (A·m)", fontsize=10)
axes3[1].set_xlabel("Time (s)", fontsize=10)
axes3[0].set_title(f"Top-{_top_k} peak-amplitude vertices: dSPM (upper) vs. MNE (lower)", fontsize=11)
axes3[0].legend(fontsize=8, ncol=2, loc="upper left")
axes3[1].legend(fontsize=9, loc="upper left")
for _ax in axes3:
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.grid(True, alpha=0.3)

fig3.tight_layout()
plt.show()
print("Figure 3: peak-vertex time courses rendered OK")

# %% [markdown]
# ---
# ## 5 — 這些地圖的意義（與侷限）
#
# ### 如何解讀圖 2
# 每一行是一個皮質頂點；顏色編碼該時刻的估計來源振幅。大部分皮質的活動接近零（淺色/白色）。少數頂點在 t = 300 ms 附近亮起——那是最小範數解「放置」我們模擬類 P300 成分之生成器的位置。直方圖（右側）顯示被激活集合有多稀疏。
#
# ### 為何 dSPM ≠ 真實值（ground truth）
# 模擬資料根本不含真正的偶極子——活動是空間均勻的高斯雜訊加上時間高斯包絡。dSPM 仍產生看似合理的局部焦點。「定位」是*最小範數先驗*（假設最少、最小的來源）的產物，而非皮質電流的物理測量。真實的 EEG 來源定位亦然：地圖告訴你的是正則化器的*猜測*，而非神經元實際激發的位置。

# %% [markdown]
# ---
# ## ⚠️ A subtler trap
#
# It is tempting to report "we localized the P300 to the posterior parietal cortex
# at (x, y, z) mm precision." Three reasons that claim overstates the evidence:
#
# **1. The ill-posedness never goes away.**
# Regularization makes the inverse problem *solvable*, but the solution is not
# unique. Different regularizers (MNE, dSPM, sLORETA, LCMV, eLORETA, …) will
# return overlapping but shifted source estimates from identical scalp data. The
# "blob" you see is the intersection of your data and your *prior*, not a unbiased
# estimator of the true generator.
#
# **2. Millimetre precision from centimetre electrodes is physically implausible.**
# The skull smears potentials across ~3 cm. Source localization accuracy in
# empirical studies with known intracranial sources is 5–20 mm on average even
# with 64+ channels. Reporting sub-centimetre coordinates from 32-channel EEG
# is wishful thinking.
#
# **3. Source-space features do not magically remove volume-conduction leakage.**
# A common pattern in BCI papers: apply source localization, extract
# source-time-series features, feed to a classifier, celebrate. But the inverse
# operator is a *linear* combination of the electrode signals. If two cortical
# regions are correlated at the scalp (because their dipoles project similarly),
# the estimated source signals will also be correlated. The leakage moves from
# sensor space to source space; it does not disappear. True functional
# connectivity between source estimates requires additional methods (e.g.
# imaginary coherence, orthogonalization, or beamformer spatial filters with
# careful orientation selection).
#
# **In short:** source localization is a powerful hypothesis-generation tool.
# Treat its output as *probabilistic spatial priors* on where activity *might*
# be, not as anatomically verified measurements. Use it to motivate electrode
# placement or constrain group-level neuroimaging analyses — never as a
# substitute for invasive recordings or high-resolution MRI-guided source
# modelling.
