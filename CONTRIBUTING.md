# Contributing

**English** · [繁體中文](CONTRIBUTING.zh-TW.md)

Thanks for helping improve **ML & Signal Processing on Neural Signals 101**! The
goal is a tutorial that is *beginner-readable, reproducible, and honest*. Please
keep those three properties intact.

## Ground rules

1. **Honesty first.** Never introduce a random-shuffle split on time-series/epoched
   data, and never fit a learned transform on the full dataset before splitting.
   Use the helpers in `src/neuro101/eval.py` (`make_subject_split`,
   `make_block_split`, `leakage_safe_pipeline`, `evaluate_with_variance`).
2. **⚠️ mark every wrong cell.** Any code cell that is intentionally incorrect (to
   demonstrate a pitfall) must have a markdown cell directly above it starting with
   `⚠️ WRONG` so nobody copies it by accident.
3. **CPU-friendly.** Every notebook must run on a laptop CPU in roughly ≤ 5 minutes.
   Subsample data and honour `NEURO101_SMOKE=1` (see `neuro101.datasets`).
4. **Seed everything.** `random_state=0`, `np.random.seed`, `torch.manual_seed`.
5. **Define terms on first use.** Assume the reader knows Python but not neuroscience.

## Project setup

```bash
make setup           # Python 3.11 venv + pinned deps + editable install
source .venv/bin/activate
make test-fast       # quick unit tests, no downloads
```

## Notebooks are written as `.py`, not `.ipynb`

The **source of truth** for each notebook is a [jupytext](https://jupytext.readthedocs.io)
percent-format file in `notebooks/_src/*.py`:

- `# %%` starts a **code** cell.
- `# %% [markdown]` starts a **markdown** cell (each following `# ` line is markdown).

This keeps diffs readable and notebooks output-free. Build the `.ipynb` with:

```bash
make notebooks       # converts notebooks/_src/*.py -> notebooks/*.ipynb
```

Do **not** hand-edit the generated `.ipynb` — your change will be overwritten.

### Adding a new chapter

1. Create `notebooks/_src/NN_title.py` (percent format). Start with a markdown cell
   containing **learning objectives** and an estimated runtime.
2. Include at least one **visualization** and a closing
   **"⚠️ Common mistakes / why this is wrong"** markdown cell.
3. Use `from neuro101 import ...` for any shared logic; if you write reusable code,
   put it in `src/neuro101/` and add a test.
4. Run `make notebooks && NEURO101_SMOKE=1 python scripts/run_all_notebooks.py NN`
   to confirm it executes.
5. Add a row to the README chapter table (with a runtime estimate).

## Adding code to `src/neuro101/`

- Add a clear docstring (NumPy style) with an `Examples` block where practical.
- Add a test in `tests/`. Tests that download data must be marked
  `@pytest.mark.network` (and usually `@pytest.mark.slow`).
- Run `make lint` (ruff) and `make test`.

## Running the full test suite

```bash
make test            # unit tests + smoke-execute every notebook (downloads data)
make lint
```

CI (`.github/workflows/ci.yml`) runs the unit tests and the notebook smoke test on
every push. Datasets are cached between runs.

## Reporting issues

Please include: your OS, Python version (`python --version`), the notebook/cell, and
the full traceback. If it's a *result* that looks too good, double-check the six
pitfalls in Chapter 09 first — most "bugs" are leakage. 🙂
