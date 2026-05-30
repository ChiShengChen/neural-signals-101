# Deep-dive — Benchmark Overfitting: leakage at the scale of a whole field

> **Prerequisites:** main Chapter 12 (Evaluation & pitfalls).
> **Level:** advanced (conceptual — no code).

Chapter 12 taught you to avoid leakage *in your own pipeline*. This deep-dive is about
a subtler, collective version: **a whole research community can overfit a single public
benchmark**, so that "state-of-the-art" numbers drift upward without the methods getting
any better at the real task. It's the same disease as tuning-on-test (pitfall #7) — just
spread across hundreds of papers and years.

## The mechanism

A benchmark like **BCI Competition IV 2a** has a fixed test set. Now imagine the field's
workflow over a decade:

1. Hundreds of researchers try methods and **report the ones that beat the previous best
   on that test set**. Methods that don't beat it are quietly dropped (publication bias).
2. Each paper tunes architectures, preprocessing, and hyper-parameters while *watching*
   the benchmark score — even if each individual paper uses clean internal CV, the
   *community* is running an enormous, uncontrolled hyper-parameter search against the
   same test data.
3. Reviewers and leaderboards reward new highs, so the selection pressure is relentless.

The result is **adaptive overfitting**: the reported SOTA is partly fitted to the idiosyncrasies
(noise, specific subjects, specific artefacts) of *that* benchmark, not to motor imagery
in general. The number-correct on the held-out set becomes, collectively, a quantity that
was selected for — exactly the winner's curse of pitfall #7, at population scale.

## Why EEG is especially vulnerable

- **Tiny datasets.** 2a has 9 subjects. With so few subjects, the gap between "genuinely
  better" and "luckier on these 9 people" is small and easily crossed by chance.
- **High variance.** Subject-to-subject spread is large (Chapter 11), so a method can top
  a leaderboard by doing well on the two or three "easy" subjects.
- **Long benchmark lifetimes.** 2a has been the standard for 15+ years — a long time to
  accumulate adaptive overfitting.
- **Shared splits.** Everyone uses the same train/test (or the same LOSO folds), so there
  is no fresh test data to catch the inflation.

## How to tell SOTA from noise

When you read "our method reaches X% on BCI IV 2a", ask:

- **Is the improvement within the subject-to-subject std?** A +1% mean over 9 subjects
  with a ±10% std is almost certainly noise (Chapter 11; use a corrected paired test —
  see `stats_rigor.ipynb`).
- **Was a *paired, corrected* test reported across subjects?** Or just two bar heights?
- **Does it generalise to a *different* dataset** (e.g. BNCI 2b, Cho2017, Lee2019) without
  re-tuning? Cross-dataset evaluation is the antidote to single-benchmark overfitting.
- **How many design choices were explored** to get the number? Reported once, or the best
  of dozens? (The garden of forking paths — see `stats_rigor.ipynb`.)
- **Is the code released and the split exactly specified?** Unreproducible SOTA is the
  easiest place for leakage to hide.

## What good practice looks like

- **Evaluate on multiple datasets**, ideally via a standardized harness like **MOABB**
  (which runs many methods over many datasets with fixed, leak-free pipelines — the
  closest the field has to an honesty referee).
- **Report effect sizes and corrected significance**, not just leaderboard rank.
- **Hold out a dataset you never touch during development**, the field-scale analogue of
  the hidden held-out set in the Chapter 14 capstone.
- **Treat "+0.5% SOTA" with suspicion** unless it survives all of the above.

## The takeaway

> A leaderboard is a shared test set, and a shared test set that everyone optimises
> against eventually leaks — slowly, collectively, invisibly. The fix is the same as
> always: **match the evaluation to the claim, test on data nobody tuned against, and
> report uncertainty, not just the peak.** "Don't fool yourself" scales up to "don't let
> the field fool itself."

### Further reading (concepts to search)
- Adaptive data analysis / "the reusable holdout" (Dwork et al., 2015).
- Benchmark overfitting in ML (e.g. "Do ImageNet classifiers generalize to ImageNet?",
  Recht et al., 2019) — same phenomenon, different field.
- MOABB: *Mother of All BCI Benchmarks* (Jayaram & Barachant, 2018) for multi-dataset,
  leak-free EEG evaluation.
