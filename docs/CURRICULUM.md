# Curriculum & Roadmap (v2 — student-first redesign)

**English** · [繁體中文](zh-TW/CURRICULUM.md)

This document is the roadmap for the tutorial. **v1** targeted engineers who
already knew sklearn/PyTorch and only needed the neural-signal side. **v2**
retargets the real audience: **university students new to neuro-AI/ML**, who are
often shaky on *both* sides — ML fundamentals aren't internalised, linear-algebra
/ signal intuition is thin, and even `(n_trials, n_channels, n_times)` array
shapes are a stumbling block.

## Why the redesign (the north star, extended)

The whole tutorial teaches **"don't fool yourself."** For an engineer, the first
form of self-deception is *leakage*, so v1 could open at the evaluation chapter.
For a beginner, leakage isn't even the *first* trap. Three earlier ones come first:

1. **Number illiteracy** — staring at `0.92` without knowing what it means
   relative to chance, to `N=9` subjects, or to a confidence interval.
2. **Shape errors** — wiring an axis wrong, silent broadcasting, and *believing*
   the garbage output.
3. **Hype gullibility** — trusting "offline 95%" or "EEG mind-reading" headlines.

These happen *before* leakage, and v1 silently assumed the reader was already
immune. v2 extends the north star earlier so the evaluation chapter can actually
land.

## Pillars (priority tiers)

- 🟥 **Pillar (opens the audience):** ML-from-zero on-ramp, visual math intuition,
  neuro-physiology grounding, ethics/anti-hype. *(user priorities 1, 2, 4)*
- 🟧 **Course-engineering (textbook → course):** per-chapter prerequisite +
  difficulty badges, "predict-before-run" cells, concept-checks, and appendices
  (glossary, troubleshooting, exercises + solutions). *(user priority 3)*
- 🟨 **Enhancers:** statistics intuition, a gamified hidden-held-out capstone, and
  a real-time/low-cost-hardware motivation hook. *(user priorities 5, 6)*

## Chapter map (11 → 15 notebooks + appendices)

Status: 🆕 new · ♻️ reworked · ✅ carried over | Difficulty ★1–5

| # | Title | Status | Prereq | Diff | Pillar |
|---|---|---|---|---|---|
| 00 | Setup & first contact (plot real EEG in 5 lines · **array-shape mental model** · glossary/troubleshooting entry) | ♻️ | — | ★1 | 🟧 |
| 01 | What neural signals physically are (synaptic currents → scalp, volume conduction, µV, 10-20) | ♻️ | 00 | ★2 | 🟥 |
| 02 | **ML from zero** (toy 2-D data: overfitting, decision boundary, why test is sacred) | 🆕 | — | ★2 | 🟥 |
| 03 | **Math you can see** (Fourier = how much of each rhythm; covariance = how channels move together; eigenvectors/CSP geometry; no proofs) | 🆕 | 02 | ★3 | 🟥 |
| 04 | DSP basics (sampling, aliasing, filters, notch, referencing) | ✅←02 | 03 | ★3 | 🟧 |
| 05 | Preprocessing & denoising (artefacts, ICA, epoching) | ✅←03 | 04 | ★3 | — |
| 06 | Frequency domain (FFT, Welch, STFT, wavelets, band power) | ✅←04 | 03,04 | ★3 | — |
| 07 | Feature engineering **+ back to physiology** (ERD/ERS in mu/beta, CSP geometry from Ch03, Riemannian) | ♻️←05 | 03,06 | ★4 | 🟥 |
| 08 | Classical ML done right (Pipelines, proper CV) | ✅←06 | 02,07 | ★3 | — |
| 09 | Deep learning (EEGNet … LSTM/Transformer) | ✅←07 | 08 | ★4 | — |
| 10 | Paradigms **× neurophysiology** (oddball→P300, frequency tagging→SSVEP, ERD→MI) | ♻️←08 | 01,07 | ★3 | 🟥 |
| 11 | **Statistics intuition** (sampling variation, what mean±std means, chance≠1/k, N=9 is tiny, a small bar gap ≠ a real difference) | 🆕 | 02 | ★3 | 🟨 |
| 12 | **Evaluation & pitfalls** (six WRONG→RIGHT pairs) | ✅←09 | 02,08,11 | ★4 | 🟥 core |
| 13 | **Neuroethics & anti-hype** (privacy, consent, neuro-rights, dual-use; "offline 95% ≠ a working BCI"; mind-reading headlines = the ethical twin of leakage) | 🆕 | 10,12 | ★2 | 🟥 |
| 14 | Capstone: **hidden held-out leaderboard** (peeking at test feels good, then crashes) | ♻️←10 | 12 | ★4 | 🟨 |
| 15 | (Appendix notebook) Real-time streaming demo & low-cost hardware path (OpenBCI/Muse/Emotiv) | 🆕 | 08 | ★2 | 🟨 |

### Cross-cutting per-chapter additions (🟧)
Every chapter gains: a **prerequisite box** + **difficulty badge**, 2–3
**"predict-before-run"** cells (guess the accuracy / will this go up or down
*before* running), an end-of-chapter **concept-check** mini-quiz (with answers),
plus the existing ⚠️ *common mistakes* cell.

### Appendices (🟧)
- **`docs/GLOSSARY.md`** — every term & acronym, lookup-able (not scattered in prose).
- **`docs/TROUBLESHOOTING.md`** — shape mismatch, torch/env install, CUDA-not-found.
- **`docs/SOLUTIONS.md`** — worked answers to the per-chapter exercises.

## The only two structural moves
1. Insert **02 (ML-from-zero)** + **03 (visual math)** after the grounding and
   *before* the DSP/feature/ML track.
2. Wrap the honesty core with **11 (statistics)** and **13 (ethics)**.
Everything else is reordering plus weaving physiology into 01/07/10.

## Reading paths (v2)
- **Total beginner:** 00 → 14 in order (this is the design target).
- **Has ML, new to signals:** skim 02, do 03; focus 01,04–07,10,12,13.
- **Has neuro, new to ML:** skim 01; do 02,03,11; focus 04–09,12.
- **Just want honest evaluation:** 02 → 08 → 11 → 12 → `deep-dives/stats_rigor`.
- **Everyone reads 12 and 13.**

---

## v2.1 addendum — spine + sidequests, and depth on the honesty core

A later refinement (keeping the 00–15 numbering, no renumber) added:

- **`deep-dives/`** — the **advanced ceiling** (CSP geometry & derivation, Riemannian
  small-data math, statistical rigor, chance-level CIs, benchmark overfitting, ICA/ASR
  internals). This realises the **"spine细而慢 + sidequests深而硬"** design: the
  5-minute-CPU promise binds **only the spine**; deep-dives may be heavier and are
  **excluded from the CI smoke test**. Each ends with a subtler, topic-specific trap so
  "don't fool yourself" runs to the deepest content.
- **Chapter 12 reframe** — the headline rule is now *match the evaluation to your
  deployment claim*, not "LOSO is the only truth": leak-free within-subject evaluation is
  honest **for a calibrated personal model**, as long as it's labelled as such. (Avoid
  trading the random-split myth for a new LOSO dogma.)
- **Pitfall #7 — tuning / model-selection on the test set** (nested CV) added as the
  seventh WRONG→RIGHT pair: the #1 silent killer in practice.
- **Chapter 09 seed-variance honesty note** — single-seed runs teach *plumbing*, not
  architecture comparison; the seed-to-seed spread is shown explicitly.
- **Chapter 11 rigor** — binomial "above chance" test + the Nadeau-Bengio corrected
  resampled t-test (naive CV t-tests are anti-conservative).
- **CI soul-guard** (`tests/test_pitfalls.py`) — asserts the WRONG>RIGHT contrasts hold.
