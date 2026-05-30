# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 深度解析 — 真實 P300 與 SSVEP（MOABB）
#
# 使用公開資料集，對兩種經典 BCI（腦機介面）典範進行真實腦電圖（EEG）資料解碼，
# 涵蓋端到端可重現流程與無資料洩漏的交叉驗證。
#
# > **先備知識：** 主要章節 10 與 12。
# > **程度：** 進階 ★★★☆☆
# > **下載真實公開資料集（已快取）。不受 5 分鐘時間限制。**

# %% Bootstrap — robust upward search for neuro101 src
import sys
import os
from pathlib import Path

# 策略：先嘗試已安裝的套件，若找不到則向上搜尋目錄樹以定位 src/
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    # 從本檔案位置（或以 notebook 執行時的 cwd）向上搜尋
    _search_root = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _parent in [_search_root, *_search_root.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101" / "__init__.py").exists():
            sys.path.insert(0, str(_candidate))
            break
    try:
        import neuro101  # noqa: F401
    except ModuleNotFoundError:
        # 最後備援：假設 notebook 位於 repo 根目錄下兩層
        sys.path.insert(0, str(Path.cwd().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 無頭執行使用非互動式後端
import matplotlib.pyplot as plt
from scipy.signal import welch

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds

rng = np.random.default_rng(42)
SMOKE = ds.is_smoke()
print(f"煙霧測試模式（Smoke mode）: {SMOKE}  |  numpy {np.__version__}  |  mne {mne.__version__}")

# %% [markdown]
# ---
# ## 第一部分 — 真實 P300 解碼（BNCI2014-009）
#
# ### 為何重要
#
# 第 10 章為了控制執行時間而模擬 P300 試次。此處我們使用
# **BNCI2014-009** — 一份以 256 Hz 從 16 個電極記錄的真實 P300 拼寫器
# 資料集。十位受試者執行標準 6×6 矩陣拼寫任務。
# 該資料集可透過 MOABB 自由取得，10 位受試者共約 185 MB
# （每位受試者約 18 MB）。
#
# **類別不平衡說明（第 12 章，陷阱 #4）：** 在 6×6 拼寫器中，每個
# 「重複循環」每格閃爍一次。包含目標的列／行在每 12 次閃爍中僅被
# 標亮 2 次 → 約 **83% 非目標，17% 目標**。因此單純準確率具有誤導性；
# 我們回報**平衡準確率（balanced accuracy）**與 **ROC-AUC**。
#
# **流程：** xDAWN 空間濾波器（學習增強 ERP 的濾波器），接著
# 黎曼（Riemannian）共變異數切線空間投影，最後 LDA 分類。
# 此為 Barachant & Congedo（2014）的方法，至今仍是強力的 ERP 解碼器。

# %%
p300_section_ok = False  # 追蹤此段是否成功執行

try:
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    print("載入 BNCI2014-009 — 受試者 1（約 18 MB，首次執行後會快取）...")
    _dataset_p300 = BNCI2014_009()
    _paradigm_p300 = P300()
    X_p300, y_p300, meta_p300 = _paradigm_p300.get_data(
        dataset=_dataset_p300, subjects=[1]
    )
    # X_p300 形狀: (n_epochs, n_channels, n_times)  例如 (1728, 16, 206)
    # y_p300: 字串標籤 "Target" 或 "NonTarget"
    # meta_p300: DataFrame，含 'session' 欄位（session 0/1/2）
    print(f"已載入: {X_p300.shape}  標籤={np.unique(y_p300, return_counts=True)}")

    le_p300 = LabelEncoder()
    y_enc_p300 = le_p300.fit_transform(y_p300)  # NonTarget=0, Target=1
    target_class_idx = int(np.where(le_p300.classes_ == "Target")[0])
    print(f"標籤編碼: {dict(zip(le_p300.classes_, [0, 1]))}")
    print(f"類別分佈: {np.bincount(y_enc_p300)} → "
          f"{100*y_enc_p300.mean():.1f}% 目標（類別不平衡！）")

    p300_section_ok = True

except Exception as _e:
    print(f"警告：BNCI2014-009 無法載入（{_e}）。嘗試備援資料集...")
    try:
        # 備援：EPFLP300 資料集（較小，不同的記錄）
        from moabb.datasets import EPFLP300
        from moabb.paradigms import P300
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score

        print("備援：載入 EPFLP300 — 受試者 1...")
        _dataset_p300 = EPFLP300()
        _paradigm_p300 = P300()
        X_p300, y_p300, meta_p300 = _paradigm_p300.get_data(
            dataset=_dataset_p300, subjects=[1]
        )
        print(f"已載入: {X_p300.shape}  標籤={np.unique(y_p300, return_counts=True)}")
        le_p300 = LabelEncoder()
        y_enc_p300 = le_p300.fit_transform(y_p300)
        target_class_idx = int(np.where(le_p300.classes_ == "Target")[0])
        p300_section_ok = True

    except Exception as _e2:
        print(f"警告：兩個 P300 資料集均失敗（{_e2}）。跳過 P300 段落。")

# %% [markdown]
# ### 大平均 ERP（Grand-average ERP）：目標 vs 非目標
#
# 每個原始試次（epoch）都是*單一雜訊快照*。對多個試次取平均
# 可以消除雜訊（雜訊是隨機且均值為零），同時保留僅出現在目標試次中
# 一致的 P300 波形。P300 成分應在刺激後約 300–500 ms 達到峰值。

# %%
if p300_section_ok:
    # 從典範時間區間與樣本數重建時間軸
    t_start, t_end = _dataset_p300.interval  # 秒，例如 [0, 0.8]
    n_times_p300 = X_p300.shape[2]
    t_p300 = np.linspace(t_start, t_end, n_times_p300, endpoint=False) * 1000  # 毫秒

    # 選取中央頂葉電極（Pz）以呈現經典 P300 視圖。
    # 電極索引取決於記錄蒙太奇；若無 Pz 則使用 Cz（索引 ~0）。
    # BNCI2014-009 蒙太奇包含 Fz、Cz、Pz — 優先嘗試 Pz。
    try:
        # 預覽原始資料以取得電極名稱
        _raw_peek = _dataset_p300.get_data(subjects=[1])
        _ch_names = list(list(list(_raw_peek[1].values())[0].values())[0].info["ch_names"])
        pz_name = next((c for c in _ch_names if "Pz" in c or "pz" in c), _ch_names[0])
        pz_idx = _ch_names.index(pz_name)
    except Exception:
        pz_idx = 0
        pz_name = "ch0"

    tgt_mask = y_enc_p300 == target_class_idx
    nontgt_mask = ~tgt_mask

    # MOABB P300 典範回傳的資料已縮放至 µV — 無需轉換
    erp_tgt = X_p300[tgt_mask, pz_idx, :].mean(axis=0)    # µV
    erp_nontgt = X_p300[nontgt_mask, pz_idx, :].mean(axis=0)

    n_tgt = tgt_mask.sum()
    n_nontgt = nontgt_mask.sum()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_p300, erp_nontgt, color="#4c72b0", lw=2,
            label=f"非目標大平均 (n={n_nontgt})")
    ax.plot(t_p300, erp_tgt, color="#c44e52", lw=2.5,
            label=f"目標大平均 (n={n_tgt})")
    ax.axvline(300, ls="--", color="gray", alpha=0.7, label="300 ms")
    ax.axhline(0, color="k", lw=0.5)
    # 標記 P300 峰值的近似位置
    p300_window = (t_p300 >= 250) & (t_p300 <= 600)
    if p300_window.any():
        peak_t = t_p300[p300_window][np.argmax(erp_tgt[p300_window] - erp_nontgt[p300_window])]
        ax.axvline(peak_t, ls=":", color="#c44e52", alpha=0.8, label=f"P300 峰值 ~{peak_t:.0f} ms")
    ax.set(xlabel="時間 (ms)", ylabel="振幅 (µV)",
           title=f"P300 大平均 ERP — 真實 BNCI2014-009，受試者 1（{pz_name}）",
           xlim=(t_p300[0], t_p300[-1]))
    ax.legend(fontsize=9, loc="upper right")
    ax.fill_between(t_p300, erp_nontgt, erp_tgt, alpha=0.12, color="#c44e52",
                    label="目標–非目標差異")
    fig.tight_layout()
    plt.savefig("/tmp/p300_erp.png", dpi=120)
    plt.show()
    print(f"大平均 P300 振幅（目標 - 非目標）於 Pz: "
          f"{(erp_tgt - erp_nontgt).max():.2f} µV")
else:
    print("P300 段落已跳過 — 無可用資料。")

# %% [markdown]
# ### 解碼：xDAWN + 黎曼 + LDA（留一受試階段法）
#
# **分割策略：** BNCI2014-009 每位受試者記錄 3 個 session。我們使用
# **留一受試階段法（leave-one-session-out，LOSO）**交叉驗證，使訓練集與
# 測試集來自**完全不同的記錄回合**。這很重要，因為：
#
# 1. 來自*相同*刺激序列的 P300 試次共享相鄰試次的時間鎖定 EEG 雜訊 —
#    若隨機分割，相關樣本會在訓練集與測試集之間洩漏
#    （此微妙陷阱將在本 notebook 末尾深入探討）。
# 2. 生理非穩態性意味著 session 層級的分割能提供現實的受試者內部泛化估計。
#
# **不平衡說明：** ROC-AUC 對類別不平衡不敏感，是此處建議的指標
# （第 12 章，陷阱 #4）。為完整性起見也回報平衡準確率。

# %%
if p300_section_ok:
    pipe_p300 = make_pipeline(
        XdawnCovariances(nfilter=4, estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(),
    )

    sessions_arr = meta_p300["session"].values
    session_le = LabelEncoder()
    session_groups = session_le.fit_transform(sessions_arr)
    unique_sess = np.unique(session_groups)
    print(f"交叉驗證：在 {len(unique_sess)} 個 session 上進行留一受試階段法")

    logo = LeaveOneGroupOut()
    aucs, bal_accs = [], []

    for fold_i, (train_idx, test_idx) in enumerate(
        logo.split(X_p300, y_enc_p300, groups=session_groups)
    ):
        X_tr, X_te = X_p300[train_idx], X_p300[test_idx]
        y_tr, y_te = y_enc_p300[train_idx], y_enc_p300[test_idx]

        pipe_p300.fit(X_tr, y_tr)
        proba = pipe_p300.predict_proba(X_te)[:, target_class_idx]
        y_pred = pipe_p300.predict(X_te)

        auc = roc_auc_score(y_te, proba)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        aucs.append(auc)
        bal_accs.append(bal_acc)
        print(f"  Session 折疊 {fold_i+1}: ROC-AUC={auc:.3f}  bal_acc={bal_acc:.3f}")

    mean_auc = np.mean(aucs)
    mean_bal_acc = np.mean(bal_accs)
    print(f"\n平均 ROC-AUC : {mean_auc:.3f}  （隨機分類器 = 0.50）")
    print(f"平均平衡準確率: {mean_bal_acc:.3f}  （機率水準 = 0.50）")
    print(f"\n重點：即使在嚴重類別不平衡（約 5:1 非目標：目標）下，ROC-AUC >> 0.5。")
    print("這就是為什麼我們不依賴原始準確率（即使全部預測為非目標,")
    print("這種無意義的分類器準確率也約達 83%）。")
else:
    mean_auc = float("nan")
    mean_bal_acc = float("nan")
    print("P300 解碼已跳過。")

# %% [markdown]
# ---
# ## 第二部分 — 真實 SSVEP 解碼（Nakanishi 2015）
#
# ### 為何重要
#
# 第 10 章透過在雜訊中注入純正弦波來模擬 SSVEP（穩態視覺誘發電位）。
# 真實 SSVEP 更為複雜：背景 alpha 振盪與刺激頻率重疊、相鄰頻率彼此接近，
# 且響應強度隨注意力狀態與疲勞程度而變化。
#
# **資料集：** Nakanishi 等人（2015）— 9 位受試者，以 256 Hz 記錄 8 通道
# 枕葉 EEG，包含 12 個閃爍刺激（9.25、9.75、10.25、…、14.75 Hz，
# 間隔 0.5 Hz）。我們載入**4 個最低頻率類別**，在保持解碼可行性的同時
# 保留真實的多類別挑戰。
#
# **流程：** 對每個試次計算 Welch 功率譜密度（PSD），並提取各刺激頻率
# 的平均功率（對諧波與通道求和）。這給出 4 維特徵向量，再由邏輯迴歸分類。
# 機率水準（chance level）為 **25%**（4 個平衡類別）。

# %%
ssvep_section_ok = False

try:
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder as LE
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print("載入 Nakanishi2015 SSVEP — 受試者 1（約 15 MB，首次執行後會快取）...")
    _dataset_ssvep = Nakanishi2015()
    _paradigm_ssvep = SSVEP(n_classes=4)  # 前 4 個類別（最低頻率）
    X_ssvep, y_ssvep, meta_ssvep = _paradigm_ssvep.get_data(
        dataset=_dataset_ssvep, subjects=[1]
    )
    print(f"已載入: {X_ssvep.shape}  頻率={np.unique(y_ssvep)}")
    ssvep_section_ok = True

except Exception as _e:
    print(f"警告：Nakanishi2015 無法載入（{_e}）。嘗試備援資料集...")
    try:
        # 備援：SSVEPExo / Kalunga2016
        from moabb.datasets import Kalunga2016
        from moabb.paradigms import SSVEP
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder as LE
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        print("備援：載入 Kalunga2016 SSVEP — 受試者 1...")
        _dataset_ssvep = Kalunga2016()
        _paradigm_ssvep = SSVEP()
        X_ssvep, y_ssvep, meta_ssvep = _paradigm_ssvep.get_data(
            dataset=_dataset_ssvep, subjects=[1]
        )
        print(f"已載入: {X_ssvep.shape}  頻率={np.unique(y_ssvep)}")
        ssvep_section_ok = True

    except Exception as _e2:
        print(f"警告：兩個 SSVEP 資料集均失敗（{_e2}）。跳過 SSVEP 段落。")

# %% [markdown]
# ### PSD 視覺化：被注意的頻率主導枕葉功率
#
# 我們挑選受試者正在注意最低刺激頻率的一個試次，繪製 Welch PSD。
# 被注意的刺激頻率及其諧波應在雜訊底板之上顯示為清晰的峰值。

# %%
if ssvep_section_ok:
    # 從試次時長與樣本數推斷取樣頻率。
    # Nakanishi2015 以 256 Hz 記錄 4 秒試次 → 1025 個樣本
    # （含少量緩衝）。從原始資料中還原 sfreq。
    try:
        _raw_ssvep = _dataset_ssvep.get_data(subjects=[1])
        sfreq_ssvep = float(
            list(list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"])
            if hasattr(list(list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"], "__len__"))
            else list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"]
        )
    except Exception:
        # 啟發式：Nakanishi2015 = 256 Hz；Kalunga2016 = 256 Hz
        sfreq_ssvep = 256.0

    stim_freqs_str = np.unique(y_ssvep)
    stim_freqs = sorted([float(f) for f in stim_freqs_str])
    n_classes_ssvep = len(stim_freqs)
    print(f"刺激頻率: {stim_freqs} Hz  (sfreq={sfreq_ssvep} Hz)")

    # 選擇被注意頻率 = stim_freqs[0] 的示例試次
    attended_freq = stim_freqs[0]
    attended_label = str(attended_freq) if str(attended_freq) in y_ssvep else stim_freqs_str[0]
    attended_idx = np.where(y_ssvep == attended_label)[0][0]
    epoch_ex = X_ssvep[attended_idx]  # (n_channels, n_times)

    n_times_ssvep = epoch_ex.shape[1]
    nperseg = min(512, n_times_ssvep)
    freqs_psd, psd_ex = welch(epoch_ex, sfreq_ssvep, nperseg=nperseg, axis=-1)
    # 對枕葉通道取平均（為簡便起見使用全部通道）
    psd_mean = psd_ex.mean(axis=0)

    # 識別諧波峰值
    colors_stim = ["#c44e52", "#4c72b0", "#55a868", "#e69f00"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    ax = axes[0]
    ax.semilogy(freqs_psd, psd_mean, color="#555", lw=1, zorder=2)
    for fi, (sf, col) in enumerate(zip(stim_freqs, colors_stim)):
        for h in range(1, 4):
            fh = sf * h
            if fh > sfreq_ssvep / 2:
                break
            peak_idx = np.argmin(np.abs(freqs_psd - fh))
            label = f"{sf} Hz（被注意）" if (sf == attended_freq and h == 1) else (
                f"{sf} Hz" if h == 1 else None)
            ax.axvline(fh, color=col, lw=1.5 if sf == attended_freq else 0.8,
                       ls="-" if sf == attended_freq else "--",
                       alpha=0.9 if sf == attended_freq else 0.5,
                       label=label, zorder=1)
            if h == 1:
                ax.annotate(f"{sf}\n({h}f)",
                            xy=(fh, psd_mean[peak_idx]),
                            xytext=(fh + 0.3, psd_mean[peak_idx] * 2),
                            fontsize=7, color=col,
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax.set(xlim=(5, min(40, sfreq_ssvep/2)),
           xlabel="頻率 (Hz)", ylabel="功率譜密度（PSD）(µV²/Hz)",
           title=f"SSVEP 試次 PSD — 注意 {attended_freq} Hz\n"
                 f"（真實 Nakanishi2015，受試者 1）")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    # 右圖：單試次特徵向量（各刺激頻率的功率）
    def extract_psd_features(X, sfreq, stim_freqs, n_harmonics=3):
        """對每個試次：在各刺激頻率及其諧波處對 Welch 功率求和。"""
        n_epochs, n_ch, n_t = X.shape
        nperseg = min(512, n_t)
        feats = np.zeros((n_epochs, len(stim_freqs)))
        for i, ep in enumerate(X):
            f, psd = welch(ep, sfreq, nperseg=nperseg, axis=-1)
            for j, sf in enumerate(stim_freqs):
                power = 0.0
                for h in range(1, n_harmonics + 1):
                    fh = sf * h
                    if fh >= sfreq / 2:
                        break
                    idx = np.argmin(np.abs(f - fh))
                    power += psd[:, idx].mean()
                feats[i, j] = power
        return feats

    F_ssvep = extract_psd_features(X_ssvep, sfreq_ssvep, stim_freqs, n_harmonics=3)
    le_ssvep = LE()
    y_enc_ssvep = le_ssvep.fit_transform(y_ssvep)

    ax2 = axes[1]
    for ci, (sf, col) in enumerate(zip(stim_freqs, colors_stim)):
        mask = y_enc_ssvep == ci
        vals = F_ssvep[mask, ci]
        ax2.scatter(
            np.full(vals.shape, sf) + rng.uniform(-0.08, 0.08, vals.shape),
            vals, alpha=0.5, s=18, color=col, label=f"{sf} Hz"
        )
        ax2.hlines(vals.mean(), sf - 0.2, sf + 0.2, color=col, lw=2.5)
    ax2.set(xlabel="被注意的刺激頻率 (Hz)",
            ylabel="被注意頻率的 Welch 功率 (µV²/Hz)",
            title="特徵可分性：各真實類別在被注意頻率的功率\n（平均值 = 粗橫線）")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig("/tmp/ssvep_psd.png", dpi=120)
    plt.show()

else:
    print("SSVEP 視覺化已跳過。")

# %% [markdown]
# ### 解碼：PSD 特徵 + 邏輯迴歸（分層 5 折交叉驗證）
#
# 我們僅使用各刺激頻率（及其前兩個諧波）在全部枕葉通道上的平均頻譜功率，
# 來解碼受試者注意的頻率。4 類平衡問題的機率水準為 25%。

# %%
if ssvep_section_ok:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipe_ssvep = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    )

    n_splits_ssvep = 5
    cv_ssvep = StratifiedKFold(n_splits=n_splits_ssvep, shuffle=True, random_state=42)
    scores_ssvep = cross_val_score(
        pipe_ssvep, F_ssvep, y_enc_ssvep, cv=cv_ssvep, scoring="accuracy"
    )
    mean_acc_ssvep = scores_ssvep.mean()
    chance_ssvep = 1.0 / n_classes_ssvep

    print(f"SSVEP 4 類準確率: {mean_acc_ssvep:.3f} ± {scores_ssvep.std():.3f}")
    print(f"機率水準: {chance_ssvep:.3f}  （4 個平衡類別）")
    print(f"超越機率的提升: +{mean_acc_ssvep - chance_ssvep:.3f} "
          f"（相對提升 {100*(mean_acc_ssvep/chance_ssvep - 1):.0f}%）")
    print()
    print("注意：適中的絕對準確率反映了解碼緊密相鄰頻率（Δf = 0.5 Hz）、")
    print("短試次的困難程度,而且沒有做受試者特定校準。")
    print("P300 段落中用的黎曼方法,同樣可以改善 SSVEP 解碼")
    print("（參見 moabb.evaluations 中的基準測試）。")
else:
    mean_acc_ssvep = float("nan")
    chance_ssvep = float("nan")
    print("SSVEP 解碼已跳過。")

# %% [markdown]
# ### 結果摘要
#
# | 典範 | 資料集 | 指標 | 分數 | 機率水準 |
# |------|--------|------|------|----------|
# | P300 | BNCI2014-009 | ROC-AUC（LOSO） | — | 0.50 |
# | SSVEP | Nakanishi2015 | 準確率（5 折） | — | 0.25 |
#
# *（在執行期間由上方的儲存格填入。）*

# %%
print("=" * 55)
if p300_section_ok:
    print(f"  P300  ROC-AUC          : {mean_auc:.3f}  (chance 0.50)")
    print(f"  P300  Balanced accuracy: {mean_bal_acc:.3f}  (chance 0.50)")
else:
    print("  P300  section: SKIPPED (dataset unavailable)")
if ssvep_section_ok:
    print(f"  SSVEP accuracy         : {mean_acc_ssvep:.3f}  (chance {chance_ssvep:.2f})")
else:
    print("  SSVEP section: SKIPPED (dataset unavailable)")
print("=" * 55)

# %% [markdown]
# ---
# ## ⚠️ A subtler trap
#
# ### P300 解碼中的相關試次洩漏問題
#
# 以下是一個即使謹慎的研究人員也容易忽略的陷阱。
#
# **問題。** 在 P300 拼寫器中，刺激以結構化的*序列*呈現
#（例如矩陣的列與行，每約 100 ms 一次）。在單一「重複循環」中，
# EEG 是一段連續訊號：第 k 次閃爍的殘餘皮質反應（一個持續長達約 800 ms
# 的緩慢正波）在第 k+1 或 k+2 次閃爍到來時仍然物理存在。這意味著
# 相鄰試次**並非獨立** — 它們共享重疊的大腦響應。
#
# **為何隨機分割會失敗。** 若將試次隨機分割到訓練集／測試集
#（例如使用 `train_test_split`），訓練集中很可能包含試次 #k，
# 而測試集中包含試次 #k+1。分類器實際上透過時間上相鄰的訓練樣本
# 「看過」測試試次的大腦狀態。這會使 ROC-AUC 估計值虛增 —
# 有時高達 0.05–0.10 AUC，而研究人員渾然不覺，
# 因為資料結構一旦提取為 NumPy 陣列後便已不透明。
#
# **上述使用的修正方法。** 我們按 *session*（留一受試階段法）分割，
# 確保整個記錄區塊被完全保留。同一序列內的試次*永遠*落在同一個折疊中。
# 這是 MOABB 基準測試所建議的方法，對可重現的 ERP 解碼至關重要。
#
# **SSVEP 的類比陷阱。** SSVEP 中有一個相關陷阱：混淆*刺激偽跡*
#（刺激顯示器以恰好等於閃爍頻率的電氣雜訊洩漏到 EEG 放大器中）
# 與真正的大腦 SSVEP 響應。在受污染資料上訓練的解碼器仍可達到高準確率
# — 但只是因為它在讀取顯示器偽跡，而非受試者的注意力。
# 診斷測試：讓受試者*背對*螢幕重播刺激；受偽跡驅動的分類器仍將
# 完美「解碼」被注意的頻率，揭示它從未真正讀取大腦。
# 基準驗證：確認移除枕葉電極（真正攜帶視覺響應的電極）會導致準確率
# 顯著下降；若單獨使用額葉或顳葉電極仍能達到同樣高的準確率，
# 則應懷疑存在偽跡。

# %%
print("深度解析完成。關鍵要點：")
print("  1. 真實 P300 資料有 83% 是非目標 → 使用 ROC-AUC / 平衡準確率。")
print("  2. 相鄰 P300 試次間的相關性要求以區塊層級進行分割。")
print("  3. SSVEP「解碼」可能由刺激偽跡驅動 — 在宣稱大腦解碼前，")
print("     務必透過通道消融與離屏對照實驗加以驗證。")
