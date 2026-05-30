# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.0] — 2026-05-30

First public release. A complete, runnable, bilingual beginner-to-advanced tutorial
taking learners from raw public EEG recordings to honest, leakage-free evaluation.

### Added — main tutorial (spine, 16 notebooks)
- **On-ramps:** 00 Setup & data (+ array-shape mental model), 01 What neural signals
  physically are, 02 ML from zero (overfitting, why the test set is sacred),
  03 Math you can see (Fourier / covariance / CSP geometry, visual).
- **Signal & features:** 04 DSP basics, 05 Preprocessing & denoising (ICA),
  06 Frequency domain, 07 Feature engineering (tied to physiology: ERD/ERS, CSP, Riemannian).
- **Models:** 08 Classical ML (proper CV), 09 Deep learning (EEGNet/Shallow/Deep/LSTM/Transformer).
- **Applications & the honesty core:** 10 Paradigms × neurophysiology,
  11 Statistics intuition (chance levels, CIs, corrected paired tests),
  **12 Evaluation & pitfalls** (seven WRONG→RIGHT pairs incl. nested CV — the headline chapter),
  13 Neuroethics & anti-hype.
- **Integration:** 14 Capstone (hidden held-out leaderboard), 15 Real-time & hardware.
- Every chapter: learning objectives, prerequisite + difficulty badge,
  "predict-before-run" cells, a ✅ concept-check, and a closing ⚠️ common-mistakes cell.

### Added — deep-dives (advanced ceiling, 14 topics)
CSP geometry, Riemannian on small data, statistical rigor (Nadeau–Bengio), chance-level
CIs, benchmark overfitting, ICA/ASR internals, **is-it-brain-or-artifact?**,
**transfer/domain adaptation**, filter-bank CSP, interpretability + augmentation,
real P300/SSVEP (MOABB), **regression decoding**, **source localization**,
**self-supervised learning**. Each ends with a subtler, topic-specific trap.

### Added — package, tests, infra
- `src/neuro101/` importable package: `io`, `preprocessing`, `features`, `viz`, `eval`,
  `datasets`. The leakage-safe API (`make_subject_split`, `make_block_split`,
  `leakage_safe_pipeline`, `evaluate_with_variance`) is the only splitting path provided.
- pytest suite incl. **soul-guards** asserting the WRONG→RIGHT contrasts hold, and a
  **zh-TW ↔ EN code-drift guard** keeping translations in lockstep.
- GitHub Actions CI (lint + unit tests + notebook smoke test), Makefile, pinned
  `requirements.txt`, headline figure + social-preview generators.

### Added — docs & accessibility
- Bilingual **English + Traditional Chinese** mirror of every notebook and doc.
- `docs/`: CURRICULUM, GETTING_STARTED, GLOSSARY, FAQ, CHEATSHEET, TROUBLESHOOTING,
  SOLUTIONS (each with a zh-TW counterpart); README, CONTRIBUTING, CITATION.cff.
- Open-in-Colab + nbviewer paths for zero-install / preview-fallback.

### Notes
- Public datasets are auto-downloaded by code (BCI IV 2a, PhysioNet MI, Sleep-EDF,
  MNE sample, plus MOABB P300/SSVEP and fsaverage for deep-dives).
- Everything runs on **CPU**; `NEURO101_SMOKE=1` shrinks data for fast/CI runs.

[0.1.0]: https://github.com/ChiShengChen/neural-signals-101/releases/tag/v0.1.0
