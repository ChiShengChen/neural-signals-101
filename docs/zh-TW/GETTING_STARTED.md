# 入門指南（完全新手版）

[English](../GETTING_STARTED.md) · **繁體中文**

從來沒開過 Jupyter notebook？不確定自己的 Python 夠不夠用？從這裡開始吧。
本頁會帶你從「零基礎」到「成功跑完第 00 章」，完全不需要先備經驗。

---

## 第 0 步 — 最快的執行方式：Google Colab（免安裝）

如果你只是想**馬上試試看**，完全不需要安裝任何東西：

1. 點選 GitHub 上任意 notebook 頂端的 **「Open in Colab」** 徽章
   （或到 [README 章節表格](../../README.md#chapters-with-estimated-runtime)）。
2. Colab 會在你的瀏覽器中開啟 notebook。
3. **執行第一個 cell**（「Colab bootstrap」那格）。它會自動安裝所有套件並下載
   輔助程式包，等待約 1–2 分鐘。
4. 接著由上而下依序執行其餘 cell。

就這樣——免費、免設定、在雲端跑的真實 Python 環境。（本機執行在重複使用時
更方便，請參考 README 的 *Install* 段落和 `make setup`。）

---

## Jupyter notebook 到底是什麼？

**notebook** 是由一格一格的 **cell** 組成的文件：

- **Code cell（程式碼格）** 裡面是可以執行的 Python，執行後輸出（數字、圖表）
  會顯示在格子正下方。
- **Markdown cell（說明文字格）** 包含格式化的說明文字（就像本頁一樣）——
  用來解釋程式碼之間的概念。

**如何執行一個 cell：** 點選它，然後按 **Shift+Enter**（執行並移到下一格），
或點選 ▶️「Run」按鈕。

**新手黃金守則：**
- **由上而下依序執行 cell。** 後面的 cell 通常依賴前面 cell 建立的變數；
  跳著執行會導致 `NameError`。
- **遇到怪事就重新啟動並全部執行。** 選單 → *Kernel → Restart & Run All*。
  大多數「幽靈 bug」都是殘留變數惹的禍。
- **程式碼格左側的 `[ ]` 數字** 是執行順序；`[*]` 表示仍在執行中。
- **輸出結果顯示在 cell 正下方。** 圖表以圖片呈現；print 的文字則以文字呈現。

---

## 準備好了嗎？5 道 Python 自我檢測題

你不需要是 Python 專家——但要能看懂並寫出基本的 Python。
如果下面這些題目感覺可以應付，那你已經準備好了。（解答在最下方。）

```python
# Q1. 這段程式碼會印出什麼？
xs = [3, 1, 2]
print(sorted(xs)[0])

# Q2. total 的值是多少？
total = 0
for n in range(1, 4):
    total += n

# Q3. 這段程式碼會印出什麼？
def square(x):
    return x * x
print(square(5))

# Q4. d["b"] 的值是什麼？
d = {"a": 1, "b": 2}

# Q5. arr 的 shape 大概是什麼？
import numpy as np
arr = np.zeros((3, 4))
```

如果不用實際跑就能回答這些題目，你的 Python 程度已經足夠跟上本教學。
**機器學習**與**訊號處理**的部分我們會從頭教起——你只需要掌握 Python
基礎：變數、list／dict、`for` 迴圈、函式，以及對 NumPy array 有初步認識即可。

### 還不懂 Python？

先花幾個小時學基礎，再回來：
- 官方 [Python 教學](https://docs.python.org/3/tutorial/)（第 3–5 章）。
- 任何「一個下午學會 Python」的課程；你只需要用到這門語言約 10% 的功能。
- NumPy 部分：[NumPy 初學者指南](https://numpy.org/doc/stable/user/absolute_beginners.html)。
  我們也在第 00 章重新說明 array 的 **shape**（*array-shape 心智模型*）——
  這是每個 EEG 初學者都會卡住的 NumPy 概念。

---

## 典型的第一次學習流程

1. 開啟 **第 00 章**（點 Colab 徽章，或執行 `jupyter notebook notebooks/00_setup_and_data.ipynb`）。
2. 由上而下依序執行所有 cell，並閱讀程式碼之間的說明文字。
3. 遇到不認識的詞彙，查閱 [`GLOSSARY.md`](GLOSSARY.md)。
4. 誠實面對 **「預測後再執行」** 的練習格——先猜，再跑。猜錯才記得住。
5. 完成結尾的 **✅ 概念確認題**。卡住了？看 [`SOLUTIONS.md`](SOLUTIONS.md)。
6. 遇到錯誤？查 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)。有「這樣正常嗎？」
   的疑問？查 [`FAQ.md`](FAQ.md)。

接著前往第 01 章，繼續往下走。你一定做得到。🧠

---

### 自我檢測解答
Q1 → `1`（最小的元素）。Q2 → `6`（1+2+3）。Q3 → `25`。Q4 → `2`。Q5 → `(3, 4)`
（3 列、4 行）。
