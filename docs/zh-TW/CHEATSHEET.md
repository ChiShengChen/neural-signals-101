# Neural Signals 101 — 單頁速查表

[English](../CHEATSHEET.md) · **繁體中文**

把這頁印出來。這是那 5% 你會不斷翻回來查的內容。（完整 API：`src/neuro101/` 中的 docstring；完整詞彙表：[`GLOSSARY.md`](GLOSSARY.md)。）

---

## 🧠 Array shape 心智模型
```
X = (n_trials, n_channels, n_times)     y = (n_trials,)   # y[i] 是 X[i] 的標記
X[i]      -> 一筆試驗 (channels, times)
X[i, c]   -> 一個通道的 1-D 訊號 (times,)
X.mean(axis=2) -> 對 TIME 取平均   -> (trials, channels)
X.mean(axis=0) -> 對 TRIALS 取平均 -> (channels, times)  # 即 ERP
```
**結果看起來怪怪的？先 `print(.shape)` 看看。**

---

## ✅ 評估檢查清單（貼到每個專案裡）
```text
切分方式
[ ] 時間序列 / 試驗段不做隨機打亂切分。
[ ] 整段試驗／區塊保持在切分的「同一側」。
[ ] 主要指標使用跨受試者（Leave-One-Subject-Out）評估。
資料洩漏
[ ] 所有學習型轉換（scaler/CSP/ICA/PCA/特徵選擇）放在 Pipeline 內，
    只在 TRAIN 折上擬合。
[ ] 切分前不對完整資料集計算任何統計量。
[ ] 超參數在驗證集上調整，絕不在測試集上調整。
評估指標
[ ] 不平衡資料用 balanced accuracy / F1 / ROC-AUC（不只用 accuracy）。
[ ] 看過混淆矩陣；與隨機基準及簡單基準線比較過。
穩健性
[ ] 在實際部署條件下（不同錄製次數／日期／設備／受試者）進行評估。
[ ] 固定所有亂數種子；回報多折／多種子的 mean ± std（不只一個數字）。
[ ] 在宣稱模型 A 優於模型 B 之前，先做逐折配對檢定。
```

---

## 🔑 本 Repo 的安全關鍵 API（`from neuro101.eval import ...`）
```python
# Leave-One-Subject-Out：誠實的跨受試者切分器
make_subject_split(subjects, n_splits=None)        # -> yields (train_idx, test_idx)

# 試驗／區塊感知切分，用於受試者內時間序列（永遠不隨機打亂！）
make_block_split(n_samples, groups=None, n_splits=5, gap=0)

# 將轉換器與模型包裝起來，確保每一步只在 TRAIN 上擬合（無洩漏）
leakage_safe_pipeline([("scaler", ...), ("csp", ...), ("clf", ...)])

# 對多個種子執行交叉驗證；回傳 {metric: {"mean","std","per_fold"}, ...}
evaluate_with_variance(pipe, X, y, cv=lambda: make_subject_split(subjects),
                       scoring=("accuracy","balanced_accuracy","f1_macro"),
                       seeds=(0,1,2))
```
*本 repo 刻意**不**提供隨機打亂切分器。*

## 📦 載入資料（`from neuro101 import io`）
```python
io.load_bnci_2a_epochs(n_subjects=4)        # -> X, y, subjects  (運動想像)
io.load_physionet_mi_epochs(n_subjects=2)   # -> X, y, subjects  (小型／快速)
io.load_sleep_edf_epochs(n_subjects=1)      # -> X, y, subjects  (不平衡資料)
# 環境變數：NEURO101_SMOKE=1 -> 使用最小資料切片；NEURO101_DATA -> 快取目錄
```

## 🛠 特徵提取（`from neuro101 import features as ft`）
```python
ft.bandpower(X, sfreq, relative=False)      # 各通道的 log 頻帶功率
ft.time_domain_features(X)                  # 變異數、MAV、移動性
ft.coherence_features(X, sfreq)             # 成對 coherence
ft.plv_features(X)                          # 相位鎖定值（0..1）
ft.make_csp(n_components=4)                 # 學習型空間濾波器（放在 pipeline 內擬合！）
ft.make_riemann_pipeline_steps()            # [("cov",..),("tangent",..)] Riemannian 步驟
```

---

## 📊 EEG 頻率帶（Hz）
| delta | theta | alpha | beta | gamma |
|---|---|---|---|---|
| 1–4 | 4–8 | 8–13 | 13–30 | 30–45 |
*感覺運動皮質上的 Mu（約 8–12 Hz）在（想像）動作時下降 = ERD。*

## ⚖️ 陷阱 → 一行修正（第 12 章）
| 陷阱 | 修正方式 |
|---|---|
| 對時間序列隨機打亂 | `make_block_split` |
| 合併受試者後隨機切分 | `make_subject_split`（LOSO） |
| 對全部資料擬合 scaler/CSP | 放入 `Pipeline` 內擬合 |
| 不平衡資料只看 accuracy | balanced accuracy / F1 / ROC-AUC |
| 在訓練錄製上測試 | 跨錄製次數／日期／設備測試 |
| 只跑一次，挑最好的數字 | mean ± std + 配對檢定 |

**誠實的數字幾乎永遠比你第一次得到的數字低。去追那個差距吧。**
