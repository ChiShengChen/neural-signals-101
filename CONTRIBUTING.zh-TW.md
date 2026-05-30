# 貢獻指南

[English](CONTRIBUTING.md) · **繁體中文**

感謝你協助改善 **ML & Signal Processing on Neural Signals 101**！這份教學的目標是：*對初學者友善、可重現、誠實*。請在貢獻時維持這三項特質。

## 基本原則

1. **誠實優先。** 絕對不要在時序性資料或 epoch 資料上使用隨機打亂的切分方式，也不要在切分前對整個資料集套用需要學習的轉換。請使用 `src/neuro101/eval.py` 中的輔助函式（`make_subject_split`、`make_block_split`、`leakage_safe_pipeline`、`evaluate_with_variance`）。
2. **⚠️ 為每個錯誤示範 cell 加上標記。** 任何刻意寫錯的程式碼 cell（用於示範陷阱），都必須在其正上方加一個 markdown cell，開頭寫 `⚠️ WRONG`，避免有人誤以為可以直接複製使用。
3. **對 CPU 友善。** 每本 notebook 必須能在筆電 CPU 上於約 ≤ 5 分鐘內執行完畢。請對資料進行降采樣，並遵守 `NEURO101_SMOKE=1` 的設定（參見 `neuro101.datasets`）。
4. **固定所有隨機種子。** `random_state=0`、`np.random.seed`、`torch.manual_seed`。
5. **術語首次出現時須定義。** 假設讀者懂 Python，但不懂神經科學。

## 專案環境設定

```bash
make setup           # Python 3.11 venv + pinned deps + editable install
source .venv/bin/activate
make test-fast       # quick unit tests, no downloads
```

## Notebook 以 `.py` 格式撰寫，而非 `.ipynb`

每本 notebook 的**唯一真實來源**是位於 `notebooks/_src/*.py` 的 [jupytext](https://jupytext.readthedocs.io) percent 格式檔案：

- `# %%` 代表一個**程式碼** cell 的開始。
- `# %% [markdown]` 代表一個 **markdown** cell 的開始（其後每行以 `# ` 開頭的即為 markdown 內容）。

這樣可以讓 diff 易於閱讀，且 notebook 不包含輸出結果。用以下指令生成 `.ipynb`：

```bash
make notebooks       # converts notebooks/_src/*.py -> notebooks/*.ipynb
```

**請勿**手動編輯生成的 `.ipynb`——你的變更將會被覆蓋。

### 新增章節

1. 建立 `notebooks/_src/NN_title.py`（percent 格式）。以一個包含**學習目標**與預估執行時間的 markdown cell 開頭。
2. 至少包含一個**視覺化圖表**，以及一個結尾的 **「⚠️ 常見錯誤 / 為什麼這樣做是錯的」** markdown cell。
3. 任何共用邏輯請使用 `from neuro101 import ...`；若你撰寫了可重用的程式碼，請放入 `src/neuro101/` 並加入對應的測試。
4. 執行 `make notebooks && NEURO101_SMOKE=1 python scripts/run_all_notebooks.py NN`，確認可以正常執行。
5. 在 README 的章節表格中新增一列（包含預估執行時間）。

## 新增程式碼至 `src/neuro101/`

- 加入清楚的 docstring（NumPy 風格），在適當的地方加上 `Examples` 區塊。
- 在 `tests/` 中加入測試。需要下載資料的測試必須標記 `@pytest.mark.network`（通常也要加 `@pytest.mark.slow`）。
- 執行 `make lint`（ruff）和 `make test`。

## 執行完整測試套件

```bash
make test            # unit tests + smoke-execute every notebook (downloads data)
make lint
```

CI（`.github/workflows/ci.yml`）會在每次 push 時執行單元測試和 notebook smoke 測試。資料集在執行之間會被快取。

## 回報問題

請附上：你的作業系統、Python 版本（`python --version`）、問題所在的 notebook/cell，以及完整的錯誤訊息。如果是某個*結果看起來太漂亮*，請先對照第 09 章的六個陷阱逐一確認——大多數「bug」其實是資料洩漏。 🙂
