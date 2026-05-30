# Neural Signals 101 — One-Page Cheat Sheet

**English** · [繁體中文](zh-TW/CHEATSHEET.md)

Print this. It's the 5% you'll reach for constantly. (Full API: docstrings in
`src/neuro101/`; full glossary: [`GLOSSARY.md`](GLOSSARY.md).)

---

## 🧠 The array-shape mental model
```
X = (n_trials, n_channels, n_times)     y = (n_trials,)   # y[i] labels X[i]
X[i]      -> one trial (channels, times)
X[i, c]   -> one channel's 1-D signal (times,)
X.mean(axis=2) -> average over TIME   -> (trials, channels)
X.mean(axis=0) -> average over TRIALS -> (channels, times)  # an ERP
```
**When a result looks weird, `print(.shape)` first.**

---

## ✅ The evaluation checklist (copy into every project)
```text
SPLITTING
[ ] No random-shuffle split on time series / epochs.
[ ] Whole trials/blocks stay on ONE side of the split.
[ ] Headline metric is subject-INDEPENDENT (Leave-One-Subject-Out).
LEAKAGE
[ ] Every learned transform (scaler/CSP/ICA/PCA/feature-selection) is inside a
    Pipeline, fit on TRAIN folds only.
[ ] No statistic computed on the full dataset before splitting.
[ ] Hyper-parameters tuned on validation, never on test.
METRICS
[ ] Balanced accuracy / F1 / ROC-AUC for imbalanced data (not just accuracy).
[ ] Looked at the confusion matrix; compared to chance AND a simple baseline.
ROBUSTNESS
[ ] Evaluated across the shift you'll deploy under (session/day/device/subject).
[ ] Seeded all RNGs; reported mean ± std over folds/seeds (never one number).
[ ] Paired test across folds before claiming model A beats model B.
```

---

## 🔑 The repo's safety-critical API (`from neuro101.eval import ...`)
```python
# Leave-One-Subject-Out: the honest, subject-independent splitter
make_subject_split(subjects, n_splits=None)        # -> yields (train_idx, test_idx)

# Trial/block-aware split for WITHIN-subject time series (never random!)
make_block_split(n_samples, groups=None, n_splits=5, gap=0)

# Wrap transforms+model so each step is fit on TRAIN only (no leakage)
leakage_safe_pipeline([("scaler", ...), ("csp", ...), ("clf", ...)])

# Run CV over seeds; returns {metric: {"mean","std","per_fold"}, ...}
evaluate_with_variance(pipe, X, y, cv=lambda: make_subject_split(subjects),
                       scoring=("accuracy","balanced_accuracy","f1_macro"),
                       seeds=(0,1,2))
```
*There is deliberately **no** random-shuffle splitter in this repo.*

## 📦 Loading data (`from neuro101 import io`)
```python
io.load_bnci_2a_epochs(n_subjects=4)        # -> X, y, subjects  (motor imagery)
io.load_physionet_mi_epochs(n_subjects=2)   # -> X, y, subjects  (small/fast)
io.load_sleep_edf_epochs(n_subjects=1)      # -> X, y, subjects  (imbalanced)
# Env: NEURO101_SMOKE=1 -> tiniest slices ;  NEURO101_DATA -> cache dir
```

## 🛠 Features (`from neuro101 import features as ft`)
```python
ft.bandpower(X, sfreq, relative=False)      # log band power per channel
ft.time_domain_features(X)                  # variance, MAV, mobility
ft.coherence_features(X, sfreq)             # pairwise coherence
ft.plv_features(X)                          # phase-locking value (0..1)
ft.make_csp(n_components=4)                 # learned spatial filters (fit in pipeline!)
ft.make_riemann_pipeline_steps()            # [("cov",..),("tangent",..)] Riemannian
```

---

## 📊 EEG frequency bands (Hz)
| delta | theta | alpha | beta | gamma |
|---|---|---|---|---|
| 1–4 | 4–8 | 8–13 | 13–30 | 30–45 |
*Mu (~8–12 Hz) over sensorimotor cortex drops during (imagined) movement = ERD.*

## ⚖️ Pitfall → one-line fix (Chapter 12)
| Pitfall | Fix |
|---|---|
| Random shuffle on time series | `make_block_split` |
| Pool subjects + random split | `make_subject_split` (LOSO) |
| Fit scaler/CSP on all data | fit inside a `Pipeline` |
| Accuracy on imbalanced data | balanced accuracy / F1 / ROC-AUC |
| Test on the training session | test across sessions/days/devices |
| One run, best number | mean ± std + paired test |

**The honest number is almost always lower than your first number. Chase the drop.**
