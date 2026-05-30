# ML & Signal Processing on Neural Signals 101 (Python)

> Go from **raw brain recordings → preprocessing → features → models → an honest
> score**, entirely through runnable Jupyter notebooks. Built for engineers and ML
> practitioners who are new to neural signals. Every term is defined on first use;
> no unexplained acronyms.

The single most important thing this tutorial teaches is **how not to fool
yourself**. Most "amazing" brain-decoding results quietly leak information and
collapse when reproduced. We show you exactly how that happens — and how to get
the honest number instead.

![Inflated vs honest score](docs/headline.png)

*The same model on the same data (BCI Competition IV 2a motor imagery). The only
difference is the evaluation method: the red bar pools subjects and splits
**randomly**, so it secretly tests on people it trained on; the green bar uses
**Leave-One-Subject-Out**, testing only on people it has never seen. The red bar
is a mirage. This whole repo is about earning the green bar — and Chapter 09 shows
six different ways the red bar sneaks into real projects.*

---

## What you'll be able to do

- Load real public datasets **by code** (nothing to download by hand).
- Filter, denoise (ICA), and epoch EEG correctly.
- Build time-, frequency-, connectivity-, **CSP**- and **Riemannian** features.
- Train classical models **and** deep nets (EEGNet, ShallowConvNet, DeepConvNet,
  LSTM, a tiny Transformer) — all on a **laptop CPU in minutes**.
- Evaluate **honestly**: no leakage, subject-independent headline metric,
  mean ± std, the right metric for imbalance.

---

## Install

Requires **Python 3.11** and ~3 GB of free disk for cached datasets.

```bash
git clone <this-repo> && cd <this-repo>
make setup        # creates a Python 3.11 .venv and installs everything (CPU-only torch)
source .venv/bin/activate
```

No `make`? The manual equivalent:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Then launch the notebooks:

```bash
jupyter notebook notebooks/   # or: make run-all   to execute them all headless
```

> **Reproducing the headline figure:** `make headline` (downloads BCI IV 2a on
> first run, ~0.2 GB/subject, then writes `docs/headline.png`).

---

## How to use this tutorial

- **Work through the notebooks in order** (`notebooks/00_*` → `10_*`). Each one is
  self-contained, opens with **learning objectives**, has explanatory markdown
  between every code step, at least one **visualization**, and a closing
  **"Common mistakes / why this is wrong"** cell.
- **Every ⚠️ cell is wrong on purpose.** It exists to show you a trap and the
  resulting fake-high score — never copy a ⚠️ cell into real work.
- **The shared code lives in `src/neuro101/`** and is imported by every notebook
  (and covered by tests). The evaluation helpers there — `make_subject_split`,
  `make_block_split`, `leakage_safe_pipeline`, `evaluate_with_variance` — are the
  guard rails that make honest evaluation the path of least resistance.
- **Everything runs on CPU in ~5 minutes per notebook.** Data is subsampled and we
  always tell you when. Set `NEURO101_SMOKE=1` to use the smallest slices.

### Two learning paths

| If you come from… | Suggested path |
|---|---|
| **Machine learning** (you know sklearn/PyTorch, signals are new) | Skim **Ch 1–3** for intuition, then focus on **Ch 6–9** (proper CV, deep models, and — above all — the pitfalls). |
| **Neuroscience / signals** (you know EEG, ML is new) | Focus on **Ch 1–5** (the signal & feature foundations), then read **Ch 6** and **Ch 9** carefully for the ML evaluation discipline. |

Everyone should read **Chapter 09** — it is the point of the whole tutorial.

---

## Chapters (with estimated runtime)

| # | Notebook | What it covers | First-run time* |
|---|---|---|---|
| 00 | [Setup & data](notebooks/00_setup_and_data.ipynb) | Ecosystem, file formats, load & plot every dataset | ~3–5 min |
| 01 | [What neural signals are](notebooks/01_what_are_neural_signals.ipynb) | EEG/MEG/ECoG/LFP/spikes/fNIRS/EMG; where the noise is | ~1 min |
| 02 | [DSP basics](notebooks/02_dsp_basics.ipynb) | Sampling, aliasing, quantization, filters, notch, referencing | ~1 min |
| 03 | [Preprocessing & denoising](notebooks/03_preprocessing_and_denoising.ipynb) | Artefacts, ICA, ASR-style cleaning, epoching, baseline | ~1–2 min |
| 04 | [Frequency domain](notebooks/04_frequency_domain.ipynb) | FFT, Welch PSD, STFT, wavelets, band power, time–freq trade-off | ~1 min |
| 05 | [Feature engineering](notebooks/05_feature_engineering.ipynb) | Time/freq/connectivity features, CSP, Riemannian covariance | ~1–2 min |
| 06 | [Classical ML](notebooks/06_classical_ml.ipynb) | LDA/SVM/RF/Riemann via Pipelines, **proper cross-validation** | ~2–3 min |
| 07 | [Deep learning](notebooks/07_deep_learning.ipynb) | EEGNet, ShallowConvNet, DeepConvNet, LSTM, tiny Transformer | ~3–5 min |
| 08 | [Paradigms & applications](notebooks/08_paradigms_and_applications.ipynb) | MI, P300/ERP, SSVEP, sleep staging, seizure, brain-to-text | ~2–3 min |
| 09 | [**Evaluation & pitfalls**](notebooks/09_evaluation_and_pitfalls.ipynb) ⭐ | Six WRONG→RIGHT pairs; the most important chapter | ~2–4 min |
| 10 | [Capstone](notebooks/10_capstone.ipynb) | Raw → honest report, with TODOs you fill in | ~2–4 min |

\*First run downloads & caches data; later runs are much faster.

---

## Datasets (all public, all auto-downloaded by code)

| Dataset | Used for | Approx. download |
|---|---|---|
| **MNE sample** (MEG+EEG) | First plots, ERPs (Ch 00) | ~1.5 GB (skipped in CI/smoke mode) |
| **BCI Competition IV 2a** (motor imagery, via MOABB) | Headline + Ch 05–10 | ~0.2 GB per subject (9 subjects) |
| **PhysioNet EEG Motor Movement/Imagery** | Ch 01, 03 (light demos) | ~40 MB per subject |
| **Sleep-EDF** (polysomnography) | Sleep staging & imbalance (Ch 08, 09) | ~8 MB per recording |

Downloads are cached in `~/neuro101_data` (override with the `NEURO101_DATA`
environment variable). See `src/neuro101/datasets.py` for the full registry and
`neuro101.datasets.describe()` for a printable summary.

---

## Repository layout

```
README.md                LICENSE (MIT)   CONTRIBUTING.md   requirements.txt   Makefile
src/neuro101/   io.py preprocessing.py features.py viz.py eval.py datasets.py   (importable, tested)
notebooks/      00_setup … 10_capstone  (.ipynb, built from notebooks/_src/*.py)
tests/          pytest for src/ + a smoke test that every notebook executes
scripts/        make_headline_figure.py  build_notebooks.py  run_all_notebooks.py
docs/           headline.png
.github/workflows/ci.yml  (pytest + notebook smoke test on push)
```

---

## The hard rules (enforced in code, not just prose)

This repo is opinionated so that honesty is the default:

1. **No random-shuffle splits on time series — ever.** We only provide
   `make_subject_split` (Leave-One-Subject-Out) and `make_block_split`
   (trial/block-aware), and use them everywhere.
2. **All learned preprocessing is fit on train only**, via sklearn `Pipeline`
   (`leakage_safe_pipeline`).
3. **Subject-independent results are the headline**; subject-dependent ones are
   labelled "optimistic".
4. **Everything is seeded and CPU-friendly** (`<~5 min` per notebook).
5. **Variance is always reported** (`evaluate_with_variance` → mean ± std).

---

## Development

```bash
make test        # unit tests + a fast (smoke-mode) notebook execution test
make test-fast   # unit tests only, no downloads
make lint        # ruff
make run-all     # execute every notebook end-to-end (full data)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to add a chapter or a feature.

## License

[MIT](LICENSE). Educational use encouraged — please share it.
