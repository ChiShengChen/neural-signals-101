# FAQ — the questions every beginner actually asks

**English** · [繁體中文](zh-TW/FAQ.md)

This is the **conceptual** companion to [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
(which fixes error messages). Here we answer the *"wait, is this normal?"* questions
that decide whether you keep going.

---

## "My accuracy is only 65%. Did I fail?"

**Almost certainly not.** Brain signals are not handwritten digits. On the headline
dataset (BCI IV 2a, left vs right hand), a **subject-independent** accuracy in the
**0.60–0.75** range is a *normal, often good* result — many published methods live
here. Three things to internalise:

- **Compare to chance, not to 100%.** Two classes → chance is 0.50. So 0.66 means the
  model is genuinely reading the brain, not guessing.
- **Subject-independent (LOSO) numbers are *supposed* to be lower** than the
  within-subject numbers you might see in flashy demos. Lower and honest beats high
  and fake (that's the whole point of Chapter 12).
- **A 0.66 that reproduces is worth more than a 0.95 that doesn't.** If your number
  looks *too* good, suspect leakage before celebrating.

## "Why is EEG so much harder than image/audio classification?"

- **Terrible signal-to-noise ratio.** The brain signal is tens of microvolts at the
  scalp, buried under blinks, muscle, and 50/60 Hz mains hum (Chapter 01).
- **Tiny datasets.** A "big" BCI dataset is 9 subjects; ImageNet has millions of
  images. Less data → more variance, more overfitting risk.
- **Every brain is different.** A model tuned to your brain may fail on mine
  (Chapter 12, pitfall #2). Generalising across people is genuinely hard.
- **Non-stationarity.** Even the *same* person's signal drifts between sessions and
  within a session (Chapter 12, pitfall #5).

So: modest-looking numbers are the nature of the problem, not a bug in your code.

## "How much data / how many subjects do I need?"

There's no magic number, but intuition:
- **More subjects** matters more than more trials per subject for *subject-independent*
  goals — you need variety of brains.
- With **fewer than ~10 subjects**, your confidence interval is wide (Chapter 11): a
  difference of a few points between methods is probably noise.
- For a **personal** BCI (subject-dependent), tens of trials per class can already
  work — but that number does not transfer to new people.

## "Do I need a GPU?"

**No.** The entire tutorial is designed for **CPU** and runs each notebook in minutes.
On small EEG datasets, classical methods (CSP+LDA, Riemannian) often *beat* deep nets
anyway (Chapter 07/09). A GPU only helps once you scale to large datasets and big
models — beyond a 101 course.

## "Why do my numbers change a little each time I run it?"

Some randomness is expected (model initialisation, data shuffling inside folds). That's
exactly why we **seed everything** and report **mean ± std over folds/seeds**, never a
single number (Chapter 11). If your number swings *a lot*, you probably have too little
data or a too-flexible model — both are forms of high variance.

## "The 'WRONG' code gives a higher score. Why not just use it?"

Because that score is a **lie** that won't survive contact with reality. The whole of
Chapter 12 exists to show that the inflated number comes from *leakage* — the model
secretly saw the answers. Deploy it and it collapses. We even mark every wrong cell
with ⚠️ so nobody copies it by accident.

## "Is it OK that I used `train_test_split` in Chapter 02?"

Yes — **only there**, because Chapter 02 uses i.i.d. toy points (independent samples).
For EEG (time series / grouped by subject), random shuffling **leaks** and is banned in
this repo. We provide `make_block_split` and `make_subject_split` instead. The contrast
is exactly the lesson of Chapter 12.

## "What's the difference between subject-dependent and subject-independent?"

- **Subject-dependent**: train and test on the *same* people (often higher, optimistic).
  Honest *if* you only ever deploy on those same people after calibration.
- **Subject-independent (LOSO)**: test on people the model has **never seen** — the
  honest headline for "does this work on a new user?". This is the number to report.

## "Which chapter should I start with / can I skip around?"

If you're a **total beginner**, go 00 → 14 in order. If you already know ML or neuro,
see the **learning paths** in the README. Everyone should read **Chapter 12** (pitfalls)
and **Chapter 13** (ethics) — they are the soul of the course.

## "Can I use my own EEG headset (Muse / OpenBCI / Emotiv)?"

Yes — see **Chapter 15** for a low-cost hardware path and a simulated real-time demo.
Consumer devices have fewer channels and more noise, but they're a great way to record
your *own* signals (start by watching your eye-blinks appear in the waveform).

## "I want to go deeper. What's next after this tutorial?"

- Read real benchmarks with **MOABB** (the library behind our datasets).
- Explore **self-supervised / foundation models** for EEG (Chapter 09's "further
  reading").
- Re-read **Chapter 12** before trusting *any* result you produce — including your own.

---

*Didn't find your question? Open a GitHub issue — good questions become new FAQ entries.*
