# Getting Started (absolute beginner edition)

**English** · [繁體中文](zh-TW/GETTING_STARTED.md)

Never opened a Jupyter notebook? Not sure if your Python is good enough? Start here.
This page gets you from *zero* to *running Chapter 00* with no prior experience.

---

## Step 0 — The fastest way to run this: Google Colab (no install)

If you just want to **try it now**, you don't need to install anything:

1. Click the **"Open in Colab"** badge at the top of any notebook on GitHub
   (or in the [README chapter table](../README.md#chapters-with-estimated-runtime)).
2. Colab opens the notebook in your browser.
3. **Run the first cell** (the "Colab bootstrap" cell). It installs everything and
   fetches the helper package. Wait ~1–2 minutes.
4. Then run the rest, top to bottom.

That's it — a real Python environment in the cloud, free, no setup. (Running locally is
better for repeated use; see the README's *Install* section and `make setup`.)

---

## What even is a Jupyter notebook?

A **notebook** is a document made of **cells**:

- **Code cells** contain Python you can run. The output (numbers, plots) appears right
  below the cell.
- **Markdown cells** contain formatted text (like this page) — explanations between the
  code.

**How to run a cell:** click it, then press **Shift+Enter** (run this cell and move to
the next). Or click the ▶️ "Run" button.

**Golden rules for beginners:**
- **Run cells top to bottom, in order.** A later cell often needs variables created by
  an earlier one. Skipping around causes `NameError`s.
- **If things get weird, restart and run all.** Menu → *Kernel → Restart & Run All*.
  Stale leftover variables cause most "ghost" bugs.
- **The number in `[ ]` left of a code cell** is the run order. `[*]` means it's still
  running.
- **Output appears under the cell.** A plot shows as an image; printed text shows as
  text.

---

## Are you ready? A 5-question Python self-check

You don't need to be a Python expert — but you should be comfortable reading and writing
basic Python. If the following feel doable, you're ready. (Answers at the bottom.)

```python
# Q1. What does this print?
xs = [3, 1, 2]
print(sorted(xs)[0])

# Q2. What is the value of total?
total = 0
for n in range(1, 4):
    total += n

# Q3. What does this print?
def square(x):
    return x * x
print(square(5))

# Q4. What does d["b"] give?
d = {"a": 1, "b": 2}

# Q5. Roughly, what is the shape of `arr`?
import numpy as np
arr = np.zeros((3, 4))
```

If you can answer these without running them, you have enough Python for this tutorial.
We teach the **machine learning** and **signal processing** from scratch — you only need
the Python basics: variables, lists/dicts, `for` loops, functions, and a first taste of
NumPy arrays.

### Don't know Python yet?

Spend a few hours on the basics first, then come back:
- The official [Python tutorial](https://docs.python.org/3/tutorial/) (sections 3–5).
- Any "Python in an afternoon" course; you need ~10% of the language to start here.
- For NumPy specifically: the [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html).
  We also re-explain array **shapes** in Chapter 00 (the *array-shape mental model*) —
  the one NumPy idea that trips up every EEG beginner.

---

## A typical first session

1. Open **Chapter 00** (Colab badge, or `jupyter notebook notebooks/00_setup_and_data.ipynb`).
2. Run the cells top to bottom. Read the markdown between them.
3. When you hit a term you don't know, check [`GLOSSARY.md`](GLOSSARY.md).
4. Try the **"predict-before-run"** cells honestly — guess *before* you run. Being wrong
   is how the intuition sticks.
5. Do the **✅ concept check** at the end. Stuck? See [`SOLUTIONS.md`](SOLUTIONS.md).
6. Hit an error? [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md). A "wait, is this normal?"
   question? [`FAQ.md`](FAQ.md).

Then move to Chapter 01, and onward. You've got this. 🧠

---

### Self-check answers
Q1 → `1` (smallest element). Q2 → `6` (1+2+3). Q3 → `25`. Q4 → `2`. Q5 → `(3, 4)`
(3 rows, 4 columns).
