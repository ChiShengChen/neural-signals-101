# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/06_frequency_domain.ipynb)
#
# > **在 Google Colab 上執行？** 請先執行下一個儲存格——它會安裝所有套件並取得輔助程式包。**在本機執行（執行 `make setup` 之後）？** 下一個儲存格什麼也不做；直接執行並繼續即可。

# %%
# --- Colab 啟動程式：僅在 Colab 環境安裝相依套件與 neuro101 套件 ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab 設定完成——請繼續往下執行本章內容。")

# %% [markdown]
# # 第 06 章 — 頻域（Frequency Domain）
#
# 大腦節律存在於頻帶之中，因此頻率分析是神經訊號處理的核心。我們將建立 FFT、Welch PSD、頻譜圖和小波，並具體呈現**時間－頻率取捨**。
#
# ## 學習目標
# 1. **FFT**（快速傅立葉轉換）與 **PSD**（功率頻譜密度，Welch 法）：功率如何分佈在各頻率上。
# 2. **STFT／頻譜圖（spectrogram）** 與 **小波（wavelets）**：這種分佈如何隨時間變化。
# 3. **頻帶功率（band power）**（delta／theta／alpha／beta／gamma）作為特徵。
# 4. **時間－頻率取捨（time–frequency trade-off）**：無法同時在兩者上都有完美的解析度。
#
# > **先修條件：** 第 03 章和第 04 章。
# > **難度：** ★★★☆☆
#
# **執行時間：** 約 1 分鐘。

# %%
import sys
from pathlib import Path
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    _p = Path.cwd()
    for _ in range(5):
        if (_p / "src" / "neuro101").exists():
            sys.path.insert(0, str(_p / "src")); break
        _p = _p.parent
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, stft
from neuro101 import io, datasets as ds, viz
from neuro101.features import bandpower, BANDS

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. FFT：訊號作為正弦波的疊加
#
# **傅立葉轉換（Fourier transform）** 將訊號改寫為不同頻率正弦波的疊加。**FFT**（快速傅立葉轉換，Fast Fourier Transform）能快速完成這項計算。以下我們建立一個已知含有 6 Hz 和 13 Hz 成分加上雜訊的訊號，並還原這些峰值。

# %%
sf = 250.0
t = np.arange(0, 4, 1 / sf)
x = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 13 * t) + 0.4 * rng.standard_normal(t.size)

freqs = np.fft.rfftfreq(t.size, 1 / sf)
amp = np.abs(np.fft.rfft(x)) / t.size

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(t, x, lw=0.7); axes[0].set(title="訊號（時域）", xlabel="時間（秒）")
axes[1].plot(freqs, amp, color="#c44e52"); axes[1].set(xlim=(0, 40),
            title="FFT 振幅頻譜", xlabel="頻率（Hz）")
for f0 in (6, 13): axes[1].axvline(f0, ls="--", color="gray", lw=0.8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Welch PSD：穩定的功率頻譜
#
# 雜訊資料的原始 FFT 本身也充滿雜訊。**Welch 法（Welch's method）** 將訊號分割為重疊的視窗，計算每個視窗的頻譜，然後加以平均。其結果——**功率頻譜密度（power spectral density，PSD）**——更為平滑，是我們用於頻帶功率特徵的方法。讓我們比較真實 EEG 上的原始 FFT 功率與 Welch 法。

# %%
X, y, subj = io.load_physionet_mi_epochs(n_subjects=1)
sf = ds.DATASETS["physionet_mi"].sfreq_hz
sig = X[0, 0]  # 一個通道，一個試次

f_w, psd_w = welch(sig, fs=sf, nperseg=int(sf))
raw_power = (np.abs(np.fft.rfft(sig)) ** 2) / sig.size
f_raw = np.fft.rfftfreq(sig.size, 1 / sf)

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.semilogy(f_raw, raw_power, color="#bbb", lw=0.7, label="原始 FFT 功率（雜訊多）")
ax.semilogy(f_w, psd_w, color="#2e8b57", lw=1.5, label="Welch PSD（平滑）")
ax.set(xlim=(0, 60), xlabel="頻率（Hz）", ylabel="功率", title="Welch 平均法馴服頻譜")
ax.legend(); plt.show()

# %% [markdown]
# ## 3. 頻帶功率（Band Power）：從 PSD 提取特徵
#
# 我們將 PSD 彙總為經典的 EEG 頻帶。這些是最常用的 EEG 特徵之一。我們的輔助函式傳回每個通道的**對數（log）** 頻帶功率。

# %%
print("頻帶（Hz）：", BANDS)
bp = bandpower(X[:1], sf)  # (1 trial, n_channels * n_bands)
n_ch = X.shape[1]
bp_grid = bp.reshape(len(BANDS), n_ch)  # 列：頻帶，行：通道（我們的佈局）
fig, ax = plt.subplots(figsize=(9, 3))
im = ax.imshow(bp_grid, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(BANDS))); ax.set_yticklabels(list(BANDS))
ax.set(xlabel="通道索引", title="每個通道的對數頻帶功率（一個試次）")
fig.colorbar(im, ax=ax, fraction=0.025); plt.show()

# %% [markdown]
# ## 4. 時頻分析（Time–Frequency）：各頻率何時出現？
#
# PSD 丟棄了*時間*資訊。但大腦活動是非穩態的——alpha 爆發可能時有時無。**頻譜圖（spectrogram）**（短時傅立葉轉換，Short-Time Fourier Transform，STFT）沿訊號滑動一個視窗，並在每個位置計算頻譜。
#
# 我們建立一個**線性調頻（chirp）**（頻率隨時間上升的音調），使頻譜圖清楚地顯示一條斜向的脊線。
#
# > **執行前猜測：** chirp 在頻譜圖中的脊線會是什麼形狀——水平直線、從左下到右上的斜線，還是彎曲的弧線？先畫出你的預測，再與輸出結果比較。

# %%
t = np.arange(0, 4, 1 / sf)
chirp = np.sin(2 * np.pi * (5 + 8 * t) * t)  # 頻率從 ~5 Hz 開始上升
viz.plot_spectrogram(chirp, sf, title="上升 chirp 的頻譜圖（頻率隨時間增加）")
plt.show()

# %% [markdown]
# ## 5. 時間－頻率取捨（具體呈現）
#
# STFT 視窗長度迫使你做出選擇：
# - **短視窗** → 良好的*時間*解析度，差的*頻率*解析度。
# - **長視窗** → 良好的*頻率*解析度，差的*時間*解析度。
#
# 兩者無法兼得（這是訊號的不確定性原理）。以下用短視窗與長視窗分析同一個 chirp——注意脊線要麼在時間上清晰，要麼在頻率上清晰，從不同時清晰。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, nperseg, label in [(axes[0], 64, "短視窗（64）：時間清晰"),
                           (axes[1], 512, "長視窗（512）：頻率清晰")]:
    f, tt, Sxx = spectrogram(chirp, fs=sf, nperseg=nperseg)
    ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-20), shading="gouraud")
    ax.set(ylim=(0, 60), xlabel="時間（秒）", ylabel="Hz", title=label)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 小波（Wavelets）：更聰明的取捨
#
# **小波（Wavelets）** 對高頻使用*短*視窗，對低頻使用*長*視窗——與我們通常觀察大腦訊號的方式相符（快速事件需要精確的時序，慢速節律需要精確的頻率）。MNE 提供豐富的小波工具（`mne.time_frequency.tfr_morlet`）；這裡我們透過跨尺度的 STFT 幅值展示這個概念。

# %%
f_stft, t_stft, Zxx = stft(chirp, fs=sf, nperseg=128)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading="gouraud")
ax.set(ylim=(0, 60), xlabel="時間（秒）", ylabel="頻率（Hz）",
       title="chirp 的 STFT 幅值（通往小波的墊腳石）")
plt.show()

# %% [markdown]
# ## ✅ 概念確認
#
# 1. Welch 法對重疊視窗計算的頻譜取平均。將視窗數加倍（即視窗長度減半）能改善頻率或時間解析度——是哪一個？為什麼？
# 2. alpha 頻帶通常為 8–13 Hz。若你比較兩位受試者的絕對 alpha 功率，而受試者 A 的頭骨較厚，會產生什麼混淆因素？你會如何處理？
# 3. 短 STFT 視窗能提供良好的時間解析度，但頻率解析度差。請舉一個你會刻意選擇短視窗（儘管頻率解析度差）的神經科學情境。
#
# **解答：**
# 1. 視窗長度減半使頻率解析度減半（取樣點較少→頻率分箱較粗），但由於平均了更多視窗，每個估計的變異數降低。你是以頻率解析度換取統計穩定性，而非時間解析度（視窗位置決定時間解析度）。
# 2. 較厚的頭骨會衰減 EEG 訊號，因此即使受試者 A 的實際大腦 alpha 相同，其絕對功率也會偏低。使用相對頻帶功率（alpha ÷ 全頻帶總功率）來對個體間差異進行標準化。
# 3. 任何「事件發生的*時機*比*確切頻率*更重要」的情境——例如偵測運動誘發的高頻 gamma 爆發（> 70 Hz）的起始時刻，其中毫秒級的時序至關重要，而 gamma 頻帶內的確切頻率較不重要。

# %% [markdown]
# ## ⚠️ 常見錯誤／為什麼這樣做是錯的
#
# - **把單一嘈雜的 FFT 當作真相來解讀。** 請使用 Welch 法（平均）否則你會追逐雜訊峰值。報告時請說明你使用的視窗長度。
# - **跨受試者比較絕對頻帶功率。** 頭骨厚度和電極接觸狀況會改變整體振幅。比較不同人時，請使用**相對**頻帶功率（各頻帶 ÷ 總功率）——我們為此提供了 `relative=True` 參數。
# - **忽略頻譜洩漏（spectral leakage）。** 硬視窗邊緣會使功率擴散到各頻率；加視窗（Welch 法預設使用 Hann 視窗）能減輕這個問題。
# - **期望同時擁有完美的時間*和*頻率解析度。** 這在數學上是不可能的。請根據你的問題選擇合適的視窗。
# - **`nperseg` 與取樣率不匹配。** 256 個取樣點的視窗在 100 Hz 與 250 Hz 下代表不同的時間長度。請以*秒*為單位思考，再轉換。
#
# **下一章：** 第 07 章——將這些頻譜（以及更多）轉換為特徵：連通性（connectivity）、CSP 與黎曼協方差（Riemannian covariance）。
