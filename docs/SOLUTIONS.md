# Exercises & Solutions

**English** · [繁體中文](zh-TW/SOLUTIONS.md)

Each notebook ends with a **✅ Concept check** (quick questions, answers inline).
This page adds a few **hands-on exercises** per chapter — try them yourself first,
then expand the solution. They reuse the repo's helpers (`from neuro101 import ...`).

> Tip: do exercises in a *scratch* notebook so you keep the chapter notebooks clean.

---

## Chapter 00 — Setup & data
**Exercise.** Load 2 subjects of BCI IV 2a and, *without looping over trials*, compute
the average power (mean of squared signal) per channel for class 0 only.

<details><summary>Solution</summary>

```python
from neuro101 import io
X, y, subj = io.load_bnci_2a_epochs(n_subjects=2)
power_c0 = (X[y == 0] ** 2).mean(axis=(0, 2))   # mean over trials and time -> (n_channels,)
print(power_c0.shape, power_c0[:5])
```
Key idea: `axis=(0, 2)` collapses trials and time, leaving one number per channel.
</details>

## Chapter 01 — What neural signals are
**Exercise.** From one PhysioNet subject, find which channel has the largest alpha
(8–13 Hz) power on average.

<details><summary>Solution</summary>

```python
import numpy as np
from neuro101 import io, datasets as ds
from neuro101.features import bandpower, BANDS
X, y, s = io.load_physionet_mi_epochs(n_subjects=1)
sf = ds.DATASETS["physionet_mi"].sfreq_hz
bp = bandpower(X, sf)                                  # (trials, channels*bands)
bp = bp.reshape(len(X), len(BANDS), X.shape[1])        # (trials, bands, channels)
alpha = bp[:, list(BANDS).index("alpha"), :].mean(0)   # mean over trials -> channels
print("strongest alpha channel index:", int(alpha.argmax()))
```
</details>

## Chapter 02 — ML from zero
**Exercise.** On `make_moons`, plot train vs validation accuracy for KNN as `k` goes
1→40. Where does it stop overfitting?

<details><summary>Solution</summary>

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = make_moons(400, noise=0.3, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
ks = range(1, 41)
tr = [KNeighborsClassifier(k).fit(Xtr, ytr).score(Xtr, ytr) for k in ks]
va = [KNeighborsClassifier(k).fit(Xtr, ytr).score(Xva, yva) for k in ks]
plt.plot(ks, tr, label="train"); plt.plot(ks, va, label="val"); plt.legend(); plt.show()
```
Small `k` = overfit (train≈1.0, val lower). Validation usually peaks around k≈10–20.
</details>

## Chapter 03 — Math you can see
**Exercise.** Build a signal of 7 Hz + 24 Hz and confirm the FFT shows exactly two peaks.

<details><summary>Solution</summary>

```python
import numpy as np, matplotlib.pyplot as plt
sf = 250; t = np.arange(0, 2, 1/sf)
x = np.sin(2*np.pi*7*t) + np.sin(2*np.pi*24*t)
f = np.fft.rfftfreq(t.size, 1/sf); amp = np.abs(np.fft.rfft(x))/t.size
plt.plot(f, amp); plt.xlim(0, 40); plt.show()   # peaks at 7 and 24 Hz
```
</details>

## Chapter 04 — DSP basics
**Exercise.** Show that a 70 Hz sine sampled at 100 Hz aliases. What frequency does it
appear as? *(Answer: |70 − 100| = 30 Hz.)*

<details><summary>Solution</summary>

```python
import numpy as np
f_true, fs = 70, 100
t = np.arange(0, 1, 1/fs)
# The sampled sine is indistinguishable from a 30 Hz sine:
alias = abs(f_true - fs)     # 30 Hz
print("aliases to", alias, "Hz")
```
</details>

## Chapter 05 — Preprocessing & denoising
**Exercise.** Re-run the ICA demo but exclude *zero* components. Confirm the before/after
traces are then identical.

<details><summary>Solution</summary>

Set `ica.exclude = []` before `ica.apply(...)`. With nothing removed, `raw_ica` equals
the input, so the two overlaid traces coincide — proof that the change you saw came
specifically from the excluded artefact components, not from ICA itself.
</details>

## Chapter 06 — Frequency domain
**Exercise.** Compute *relative* alpha power (alpha ÷ total) and explain why it's fairer
than absolute power when comparing two subjects.

<details><summary>Solution</summary>

```python
from neuro101.features import bandpower
rel = bandpower(X, sf, relative=True)   # each band divided by total per channel
```
Relative power removes overall amplitude differences (skull thickness, electrode
contact) that otherwise dominate cross-subject comparisons.
</details>

## Chapter 07 — Feature engineering
**Exercise.** Compare CSP+LDA vs Riemannian+LogReg on one subject with a block split.
Is the difference bigger than the std?

<details><summary>Solution</summary>

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from neuro101 import io, features as ft
from neuro101.eval import leakage_safe_pipeline, make_block_split, evaluate_with_variance
X, y, s = io.load_bnci_2a_epochs(n_subjects=1)
csp = leakage_safe_pipeline([("csp", ft.make_csp(4)), ("lda", LDA())])
rie = leakage_safe_pipeline(ft.make_riemann_pipeline_steps() + [("lr", LogisticRegression(max_iter=500))])
for name, p in [("CSP+LDA", csp), ("Riemann", rie)]:
    r = evaluate_with_variance(p, X, y, cv=lambda: make_block_split(len(y), n_splits=5),
                               scoring="accuracy", seeds=(0,))["accuracy"]
    print(name, f"{r['mean']:.3f} ± {r['std']:.3f}")
```
Usually the gap is *smaller* than the std → you cannot claim one is better (Chapter 11).
</details>

## Chapter 08 — Classical ML
**Exercise.** Reproduce the within-subject vs LOSO gap for a random forest on band power.

<details><summary>Solution</summary>

Use `make_block_split` (within one subject) vs `make_subject_split` (across subjects)
with the same `RandomForestClassifier` pipeline; LOSO will be lower. The gap is the
"works on me" vs "works on a new person" tax.
</details>

## Chapter 09 — Deep learning
**Exercise.** Train EEGNet for 1 vs 10 epochs and plot the held-out accuracy curve.
Does more training always help on this tiny dataset?

<details><summary>Solution</summary>

Increase `EPOCHS`; you'll see accuracy plateau (or wobble) because the dataset is tiny
and the subject-independent task is hard. More epochs help the *train* loss but not
necessarily the held-out subject — that's overfitting, and why you need a validation
split for early stopping.
</details>

## Chapter 10 — Paradigms
**Exercise.** Change the SSVEP "attended" frequency to 15 Hz and confirm the detector
follows.

<details><summary>Solution</summary>

Set `attended = 15.0` in the SSVEP cell; the peak-picking detector should now report
15 Hz. This is frequency tagging: the spectrum's largest stimulus-frequency peak tells
you where the user looked.
</details>

## Chapter 11 — Statistics intuition
**Exercise.** With N=5 instead of N=9 "subjects", does the bootstrap 95% CI get wider or
narrower? Why?

<details><summary>Solution</summary>

Wider. Less data → more sampling uncertainty → a wider confidence interval. This is the
whole reason small-N BCI papers must be read cautiously.
</details>

## Chapter 12 — Evaluation & pitfalls
**Exercise.** Take pitfall #2 (subject leakage) and make the WRONG number even higher by
adding more subjects to the pool. Explain why pooling more people inflates a random split.

<details><summary>Solution</summary>

More subjects → the random-split model can memorize more subject-specific patterns that
reappear in the test fold (same people on both sides). The honest LOSO number does *not*
rise the same way, so the gap (the fake gain) widens.
</details>

## Chapter 13 — Neuroethics & anti-hype
**Exercise (no code).** Take a recent "AI reads your mind" headline. Which Chapter 12
pitfall would you check first, and what one question would you ask the authors?

<details><summary>Solution</summary>

First check **subject-independence** (pitfall #2): did they test on *new people*, or
report a within-subject/pooled number? The one question: *"Is the headline accuracy
Leave-One-Subject-Out, and what is the chance level?"* If they can't answer, be skeptical.
</details>

## Chapter 14 — Capstone
**Exercise.** Beat the CSP+LDA baseline on the DEV subjects using only honest LOSO. Then
submit to the hidden set **once**. Was your DEV estimate optimistic?

<details><summary>Solution</summary>

Almost always yes — your DEV estimate is a little optimistic because you (gently)
selected your pipeline using it. The hidden-set number is the truth. The smaller your
DEV↔hidden gap, the more honest your development process was.
</details>
