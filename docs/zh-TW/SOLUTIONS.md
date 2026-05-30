# 練習題與解答

[English](../SOLUTIONS.md) · **繁體中文**

每本 notebook 結尾都有一個 **✅ 概念確認**（快速問答，答案附在其中）。
本頁則為每個章節新增幾道**實作練習題**——請先自己動手試試，再展開解答。這些練習題都會用到 repo 的輔助工具（`from neuro101 import ...`）。

> 提示：在一本*草稿* notebook 中做練習，這樣可以保持章節 notebook 的整潔。

---

## 第 00 章 — 環境設定與資料

**練習題。** 載入 2 位受試者的 BCI IV 2a 資料，*不使用迴圈*，計算第 0 類別在每個頻道上的平均功率（訊號平方的平均值）。

<details><summary>解答</summary>

```python
from neuro101 import io
X, y, subj = io.load_bnci_2a_epochs(n_subjects=2)
power_c0 = (X[y == 0] ** 2).mean(axis=(0, 2))   # mean over trials and time -> (n_channels,)
print(power_c0.shape, power_c0[:5])
```
核心概念：`axis=(0, 2)` 同時對 trials 和時間軸做折疊，每個頻道留下一個數值。
</details>

## 第 01 章 — 神經訊號的本質

**練習題。** 從一位 PhysioNet 受試者的資料中，找出平均 alpha（8–13 Hz）功率最大的頻道。

<details><summary>解答</summary>

```python
import numpy as np
from neuro101 import io, datasets as ds
from neuro101.features import bandpower, BANDS
X, y, s = io.load_physionet_mi_epochs(n_subjects=1)
sf = ds.DATASETS["physionet_mi"].sfreq_hz
bp = bandpower(X, sf)                                  # (trials, channels*bands)
bp = bp.reshape(len(X), len(BANDS), X.shape[1])        # (trials, bands, channels)
alpha = bp[:, list(BANDS).index("alpha"), :].mean(0)   # mean over trials -> channels
print("strongest alpha channel index:", int(alpha.argmax()))
```
</details>

## 第 02 章 — 從零開始學機器學習

**練習題。** 在 `make_moons` 資料集上，畫出 KNN 在 `k` 從 1 到 40 時的訓練集與驗證集準確率。過擬合從哪裡開始改善？

<details><summary>解答</summary>

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = make_moons(400, noise=0.3, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
ks = range(1, 41)
tr = [KNeighborsClassifier(k).fit(Xtr, ytr).score(Xtr, ytr) for k in ks]
va = [KNeighborsClassifier(k).fit(Xtr, ytr).score(Xva, yva) for k in ks]
plt.plot(ks, tr, label="train"); plt.plot(ks, va, label="val"); plt.legend(); plt.show()
```
`k` 越小 = 過擬合（訓練集≈1.0，驗證集明顯較低）。驗證集準確率通常在 k≈10–20 時達到峰值。
</details>

## 第 03 章 — 看得見的數學

**練習題。** 建立一個 7 Hz + 24 Hz 的合成訊號，並確認 FFT 結果中恰好出現兩個峰值。

<details><summary>解答</summary>

```python
import numpy as np, matplotlib.pyplot as plt
sf = 250; t = np.arange(0, 2, 1/sf)
x = np.sin(2*np.pi*7*t) + np.sin(2*np.pi*24*t)
f = np.fft.rfftfreq(t.size, 1/sf); amp = np.abs(np.fft.rfft(x))/t.size
plt.plot(f, amp); plt.xlim(0, 40); plt.show()   # peaks at 7 and 24 Hz
```
</details>

## 第 04 章 — 數位訊號處理基礎

**練習題。** 說明一個 70 Hz 的正弦波在 100 Hz 取樣率下會產生混疊。它會以什麼頻率出現？*（答案：|70 − 100| = 30 Hz。）*

<details><summary>解答</summary>

```python
import numpy as np
f_true, fs = 70, 100
t = np.arange(0, 1, 1/fs)
# The sampled sine is indistinguishable from a 30 Hz sine:
alias = abs(f_true - fs)     # 30 Hz
print("aliases to", alias, "Hz")
```
</details>

## 第 05 章 — 前處理與去噪

**練習題。** 重新執行 ICA 示範，但這次不排除任何成分（排除數量為零）。確認處理前後的訊號波形完全相同。

<details><summary>解答</summary>

在執行 `ica.apply(...)` 之前設定 `ica.exclude = []`。不移除任何成分的情況下，`raw_ica` 會等同於輸入訊號，因此兩條疊加的波形會完全重合——這證明了你原先看到的差異，正是來自那些被排除的雜訊成分，而非 ICA 本身的影響。
</details>

## 第 06 章 — 頻域分析

**練習題。** 計算*相對* alpha 功率（alpha ÷ 總功率），並說明為什麼在比較兩位受試者時，這樣做比使用絕對功率更公平。

<details><summary>解答</summary>

```python
from neuro101.features import bandpower
rel = bandpower(X, sf, relative=True)   # each band divided by total per channel
```
相對功率可以消除整體振幅差異的影響（例如顱骨厚度、電極接觸品質），這些因素在跨受試者比較時往往會主導結果，若不加以控制便失去意義。
</details>

## 第 07 章 — 特徵工程

**練習題。** 以 block split 對一位受試者比較 CSP+LDA 與 Riemannian+LogReg 的表現。兩者的差距是否大於標準差？

<details><summary>解答</summary>

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from neuro101 import io, features as ft
from neuro101.eval import leakage_safe_pipeline, make_block_split, evaluate_with_variance
X, y, s = io.load_bnci_2a_epochs(n_subjects=1)
csp = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])
rie = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() + [("lr", LogisticRegression(max_iter=500))])
for name, p in [("CSP+LDA", csp), ("Riemann", rie)]:
    r = evaluate_with_variance(p, X, y, cv=lambda: make_block_split(len(y), n_splits=5),
                               scoring="accuracy", seeds=(0,))["accuracy"]
    print(name, f"{r['mean']:.3f} ± {r['std']:.3f}")
```
通常差距會*小於*標準差 → 無法宣稱哪個方法更好（見第 11 章）。
</details>

## 第 08 章 — 古典機器學習

**練習題。** 重現 random forest 在頻帶功率特徵上，within-subject 與 LOSO 之間的表現落差。

<details><summary>解答</summary>

對同一個 `RandomForestClassifier` pipeline，分別使用 `make_block_split`（單一受試者內部）和 `make_subject_split`（跨受試者）；LOSO 的結果會比較低。這個落差就是「對自己有效」與「對新受試者也有效」之間的差距。
</details>

## 第 09 章 — 深度學習

**練習題。** 分別訓練 EEGNet 1 個和 10 個 epochs，並畫出 held-out 準確率的變化曲線。在這個小資料集上，訓練越久一定越好嗎？

<details><summary>解答</summary>

增加 `EPOCHS` 後，你會看到準確率停滯（甚至震盪），因為資料集很小，而跨受試者的任務本來就困難。更多的 epochs 有助於降低*訓練集*的 loss，但不一定能改善 held-out 受試者上的表現——這就是過擬合，也是為什麼你需要用驗證集來做 early stopping。
</details>

## 第 10 章 — 腦機介面典範

**練習題。** 把 SSVEP「注意目標」的頻率改為 15 Hz，並確認偵測器的輸出也跟著改變。

<details><summary>解答</summary>

在 SSVEP cell 中設定 `attended = 15.0`；峰值偵測器應該會回報 15 Hz。這就是頻率標記（frequency tagging）的原理：頻譜中最大的刺激頻率峰值，就代表使用者正在注視的目標。
</details>

## 第 11 章 — 統計直覺

**練習題。** 當「受試者」人數從 N=9 減少到 N=5 時，bootstrap 95% 信賴區間會變寬還是變窄？為什麼？

<details><summary>解答</summary>

會變寬。資料越少 → 抽樣的不確定性越大 → 信賴區間越寬。這正是為什麼閱讀 N 很小的 BCI 論文時必須格外謹慎的原因。
</details>

## 第 12 章 — 評估方法與常見陷阱

**練習題。** 以陷阱 #2（受試者洩漏）為例，透過增加受試者人數，讓錯誤的數字變得更高。解釋為什麼混入更多人會讓隨機切分的結果虛高。

<details><summary>解答</summary>

受試者越多 → 以隨機切分訓練的模型可以記住更多受試者特有的模式，而這些模式會在測試折中再次出現（因為同一批人同時出現在訓練集與測試集兩側）。誠實的 LOSO 數字不會以同樣的幅度上升，因此兩者之間的差距（即虛假的增益）會越來越大。
</details>

## 第 13 章 — 神經倫理與反誇大宣傳

**練習題（不需要寫程式）。** 找一則近期「AI 可以讀取人的想法」的新聞標題。你會優先檢查第 12 章的哪個陷阱？你會問研究者哪一個關鍵問題？

<details><summary>解答</summary>

優先檢查**受試者獨立性**（陷阱 #2）：他們是否在*全新受試者*上進行測試，還是只回報了 within-subject 或混合受試者的數字？那個關鍵問題是：*「標題中的準確率是否為 Leave-One-Subject-Out 的結果？隨機猜測的基準準確率是多少？」* 如果他們無法回答，請保持懷疑。
</details>

## 第 14 章 — 期末專題

**練習題。** 只使用誠實的 LOSO，在 DEV 受試者上超越 CSP+LDA 的基準線。然後將結果**提交一次**到隱藏測試集。你的 DEV 估計值是否過於樂觀？

<details><summary>解答</summary>

幾乎百分之百是的——你的 DEV 估計值會稍微偏樂觀，因為你（不自覺地）是根據 DEV 集來選擇 pipeline 的。隱藏測試集的數字才是真相。DEV 與隱藏測試集之間的差距越小，代表你的開發流程越誠實。
</details>
