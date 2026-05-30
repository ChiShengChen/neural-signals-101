# 疑難排解 — 最常見卡關問題的急救手冊

[English](../TROUBLESHOOTING.md) · **繁體中文**

剛開始接觸、遇到問題了嗎？90% 的新手問題都出在以下幾種情況。
找到你看到的錯誤訊息，照著修正步驟處理即可。

---

## 1. 陣列形狀錯誤（EEG 最常見的 bug）

> `ValueError: could not broadcast together with shapes (22,1000) (1000,22)`
> `ValueError: Found array with dim 3. Estimator expected <= 2.`
> `IndexError: too many indices for array`

**原因：** EEG epochs 是三維陣列 `(n_trials, n_channels, n_times)`。某個函式需要不同的形狀，或是你在錯誤的軸上做了索引。請參閱第 00 章的「陣列形狀心智模型」小節。

**修正方式 — 先印出 shape，每次都要這樣做：**
```python
print(X.shape)          # (n_trials, n_channels, n_times) ?
print(X[0].shape)       # one trial -> (n_channels, n_times)
print(X[0, 0].shape)    # one channel -> (n_times,)
```
常見的形狀轉換：
```python
X.reshape(len(X), -1)   # flatten each trial to a 1-D feature vector (for plain sklearn)
X.mean(axis=2)          # average over TIME -> (n_trials, n_channels)
X[i].T                  # transpose one trial to (n_times, n_channels) if a lib wants that
```
**黃金準則：** `X` 和 `y` 在**第 0 軸**上的長度必須一致（`len(X) == len(y)`），而且如果你重新排列了 trials 的順序，`y` 也必須以同樣的方式重新排列。

**靜默的形狀錯誤（沒有報錯，但結果不對）：** NumPy 的*廣播機制*可能會讓 `(channels, 1)` 與 `(channels, times)` 相乘而不報任何錯。如果結果看起來不對，在懷疑數學之前，先把每一個 `.shape` 都印出來看看。

---

## 2. `make setup` 找不到 Python 3.11

> `ERROR: python3.11 not found`

**修正（macOS）：** `brew install python@3.11`
**修正（Ubuntu/Debian）：** `sudo apt-get install python3.11 python3.11-venv`
**修正（任何作業系統，透過 pyenv）：** `pyenv install 3.11 && pyenv local 3.11`

之後重新執行 `make setup`。這個 repo 刻意鎖定 Python 3.11，原因是某些深度學習的 wheel 目前尚未針對更新版本的 Python 進行編譯。

---

## 3. PyTorch 裝不起來 / 下載了好幾 GB 的 CUDA 套件

> `Could not find a version that satisfies the requirement torch==2.4.1`
> （或是在沒有 GPU 的筆電上觸發了好幾 GB 的 CUDA 下載）

**原因：** PyTorch 預設的 wheel 包含 GPU/CUDA 支援。這份教學**只使用 CPU**。

**修正 — 明確安裝 CPU 版本（這也是 `make setup` 的做法）：**
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. `CUDA not available` / `RuntimeError: No CUDA GPUs are available`

**這是預期行為，不需要擔心。** 每一本 notebook 都設計成在 CPU 上執行；我們刻意設定 `device = "cpu"`。你**不需要** GPU。如果有模型嘗試使用 CUDA，請確認你沒有把 `DEVICE`/`device` 改成 `"cuda"`。

---

## 5. `ModuleNotFoundError: No module named 'neuro101'`

**原因：** 輔助套件沒有安裝在你目前啟用的環境中。

**修正：**
```bash
source .venv/bin/activate      # activate the venv first!
pip install -e .               # install the package in editable mode
```
Notebooks 也有一個備用機制，會把 `../src` 加入路徑，但這只有在你從 repo 根目錄或 `notebooks/` 資料夾啟動 Jupyter 時才有效。

---

## 6. 資料集下載卡住、失敗，或出現輸入提示

> 下載卡住，或出現 `StdinNotImplementedError: raw_input was called`

**原因：** 第一次執行時會下載公開資料集（BCI IV 2a 每位受試者約 0.2 GB）；網路不穩定可能導致中斷。`raw_input` 的錯誤是 MNE 詢問是否要儲存設定路徑——這個 repo 已經預先抑制了這個提示，但自訂的環境可能沒有。

**修正：**
- 重新執行那個 cell；下載會從快取繼續（`~/neuro101_data`，可用 `NEURO101_DATA` 覆蓋路徑）。
- 若要快速測試少量資料，啟用 smoke 模式：在啟動前設定 `NEURO101_SMOKE=1`。
- 如果 MNE 在 stdin 出現提示，在 Python shell 中執行一次以下指令設定路徑：
  `import mne; mne.set_config('MNE_DATA', '~/neuro101_data')`。

---

## 7. Notebook 執行速度太慢

**修正：** 使用 smoke 模式執行，它只會載入最小的資料切片：
```bash
NEURO101_SMOKE=1 jupyter notebook
```
或在終端機中，無介面執行單一 notebook：
```bash
NEURO101_SMOKE=1 python scripts/run_all_notebooks.py 07
```

---

## 8. 圖表沒有顯示 / `FigureCanvasAgg is non-interactive`

**原因：** 使用了無顯示介面的 Agg 後端（在 CI 環境或部分終端機中屬於正常現象）。

**修正：** 在 Jupyter 中，圖表會自動內嵌顯示。如果是以腳本方式執行，可以加上 `plt.show()`（我們已經這樣做了）或使用 `plt.savefig("fig.png")` 儲存圖片。設定 `MPLBACKEND=Agg` 會強制使用非互動式渲染，這也是 CI 的做法。

---

## 還是卡住了？

1. 重新啟動 kernel，從頭到尾依序執行（殘留的狀態常常會造成難以察覺的 bug）。
2. 對每一個變數都執行 `print(.shape)`。
3. 確認你在 `.venv` 環境中（`which python` 的結果應該指向 repo 內部的路徑）。
4. 開一個 issue，附上你的作業系統、`python --version` 的輸出、問題所在的 cell，以及完整的錯誤訊息。
