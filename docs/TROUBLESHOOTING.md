# Troubleshooting — first aid for the most common time-sinks

**English** · [繁體中文](zh-TW/TROUBLESHOOTING.md)

New to this and something broke? 90% of beginner problems are one of the few below.
Find your error message and follow the fix.

---

## 1. Shape errors (the #1 EEG bug)

> `ValueError: could not broadcast together with shapes (22,1000) (1000,22)`
> `ValueError: Found array with dim 3. Estimator expected <= 2.`
> `IndexError: too many indices for array`

**Why:** EEG epochs are 3-D `(n_trials, n_channels, n_times)`. A function wanted a
different shape, or you indexed the wrong axis. See the *array-shape mental model*
section in Chapter 00.

**Fix — print the shape FIRST, every time:**
```python
print(X.shape)          # (n_trials, n_channels, n_times) ?
print(X[0].shape)       # one trial -> (n_channels, n_times)
print(X[0, 0].shape)    # one channel -> (n_times,)
```
Common conversions:
```python
X.reshape(len(X), -1)   # flatten each trial to a 1-D feature vector (for plain sklearn)
X.mean(axis=2)          # average over TIME -> (n_trials, n_channels)
X[i].T                  # transpose one trial to (n_times, n_channels) if a lib wants that
```
**Golden rule:** `X` and `y` must agree on **axis 0** (`len(X) == len(y)`), and if you
reorder trials you must reorder `y` the same way.

**Silent shape bugs (no error, wrong numbers):** NumPy *broadcasting* can multiply a
`(channels, 1)` by `(channels, times)` without complaint. If results look weird,
print every `.shape` before suspecting the maths.

---

## 2. `make setup` can't find Python 3.11

> `ERROR: python3.11 not found`

**Fix (macOS):** `brew install python@3.11`
**Fix (Ubuntu/Debian):** `sudo apt-get install python3.11 python3.11-venv`
**Fix (any OS, via pyenv):** `pyenv install 3.11 && pyenv local 3.11`

Then re-run `make setup`. The repo deliberately pins Python 3.11 because some deep-
learning wheels are not yet built for newer Python versions.

---

## 3. PyTorch won't install / pulls gigabytes of CUDA

> `Could not find a version that satisfies the requirement torch==2.4.1`
> (or a multi-GB CUDA download on a laptop with no GPU)

**Why:** the default PyTorch wheel includes GPU/CUDA. This tutorial is **CPU-only**.

**Fix — install the CPU build explicitly (what `make setup` does):**
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. `CUDA not available` / `RuntimeError: No CUDA GPUs are available`

**This is expected and fine.** Every notebook is designed to run on CPU; we set
`device = "cpu"` on purpose. You do **not** need a GPU. If a model tries to use CUDA,
make sure you didn't change `DEVICE`/`device` to `"cuda"`.

---

## 5. `ModuleNotFoundError: No module named 'neuro101'`

**Why:** the helper package isn't installed in your active environment.

**Fix:**
```bash
source .venv/bin/activate      # activate the venv first!
pip install -e .               # install the package in editable mode
```
The notebooks also have a fallback that adds `../src` to the path, but it only works
when you launch Jupyter from the repo root or the `notebooks/` folder.

---

## 6. A dataset download hangs, fails, or asks a question

> stuck on a download, or `StdinNotImplementedError: raw_input was called`

**Why:** the first run downloads public datasets (BCI IV 2a is ~0.2 GB/subject); a
flaky network can interrupt it. The `raw_input` error is MNE asking to save a config
path — the repo silences this, but a custom environment might not.

**Fix:**
- Re-run the cell; downloads resume from cache (`~/neuro101_data`, override with
  `NEURO101_DATA`).
- For a quick run on tiny data, set smoke mode: `NEURO101_SMOKE=1` before launching.
- If MNE prompts on stdin, set its path once in a Python shell:
  `import mne; mne.set_config('MNE_DATA', '~/neuro101_data')`.

---

## 7. A notebook is too slow

**Fix:** run in smoke mode — it loads the smallest data slice:
```bash
NEURO101_SMOKE=1 jupyter notebook
```
or, from a terminal, execute one notebook headless:
```bash
NEURO101_SMOKE=1 python scripts/run_all_notebooks.py 07
```

---

## 8. Plots don't show / `FigureCanvasAgg is non-interactive`

**Why:** a headless/Agg backend (normal in CI or some terminals).

**Fix:** in Jupyter the figures render inline automatically. If running as a script,
either add `plt.show()` (we do) or save with `plt.savefig("fig.png")`. Setting
`MPLBACKEND=Agg` forces non-interactive rendering, which is what CI uses.

---

## Still stuck?

1. Restart the kernel and run top-to-bottom (stale state causes phantom bugs).
2. `print(.shape)` everything.
3. Check you're in the `.venv` (`which python` should point inside the repo).
4. Open an issue with your OS, `python --version`, the cell, and the full traceback.
