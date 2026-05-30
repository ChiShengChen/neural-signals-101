# Deep-dives — the advanced ceiling 🏔️

**English** · [繁體中文](zh-TW/README.md)

These are **optional side-quests** for readers who finished the main spine
(`notebooks/00`–`15`) and want the real math, derivations, and edge cases. Unlike the
main chapters, they are **not bound by the "5-minute CPU" promise** and assume you're
comfortable with the corresponding chapter.

Each deep-dive ends with a **"⚠️ A subtler trap"** — a non-obvious, topic-specific way to
fool yourself — so the tutorial's north star ("don't fool yourself") runs all the way to
the deepest content, not just Chapter 12.

| Deep-dive | What it covers | Hooks from |
|---|---|---|
| [CSP geometry](csp_geometry.ipynb) | CSP as a generalized eigenvalue problem; the whitening + rotation view; derivation | Ch 03, 07 |
| [Riemann on small data](riemann_small_data.ipynb) | SPD manifold, geometric vs Euclidean mean ("swelling"), why covariance wins at low N | Ch 07 |
| [Statistical rigor](stats_rigor.ipynb) | Why naive CV t-tests over-reject; Nadeau-Bengio corrected test; nested CV; multiple comparisons | Ch 11 |
| [Chance level & CIs](chance_level_ci.ipynb) | Binomial test, Wilson/Clopper-Pearson intervals, Müller-Putz "above chance" thresholds | Ch 11 |
| [Benchmark overfitting](benchmark_overfitting.md) | How a whole field can overfit one benchmark — leakage at population scale (no code) | Ch 12 |
| [ICA & ASR internals](ica_asr_internals.ipynb) | Blind source separation math, a cocktail-party demo, ASR's subspace reconstruction | Ch 05 |
| [Brain or artifact?](artifact_confounds.ipynb) | Decoding confounds — EOG/EMG masquerading as a "BCI"; honesty checks (the physiological twin of leakage) | Ch 03, 05, 12 |
| [Transfer & domain adaptation](domain_adaptation.ipynb) | Euclidean/Riemannian alignment, calibration trials, fine-tuning — rescuing cross-subject/session accuracy | Ch 07, 12 |
| [Filter-bank CSP (FBCSP)](fbcsp.ipynb) | Multi-band CSP + feature selection; the classic competition-winning MI baseline | Ch 06, 07 |
| [Interpreting nets & augmentation](interpretability_augmentation.ipynb) | EEGNet gradient saliency + EEG data augmentation for small data | Ch 09 |
| [Real P300 & SSVEP (MOABB)](real_p300_ssvep.ipynb) | Real ERP/SSVEP datasets (vs the simulated demos in Ch 10) | Ch 10, 12 |
| [Regression decoding](regression_decoding.ipynb) | Predicting **continuous** targets; R²/MAE/correlation; the autocorrelated-target leakage trap | Ch 06, 08, 12 |
| [Source localization](source_localization.ipynb) | The ill-posed inverse problem; fsaverage forward + MNE/dSPM inverse (headless 2-D views) | Ch 01, 02 |
| [Self-supervised learning](self_supervised.ipynb) | Pretext pretraining (relative positioning) → fine-tune on few labels; the idea behind EEG foundation models | Ch 09 |

## Building & running

The `.ipynb` here are generated from `deep-dives/_src/*.py` (jupytext percent format),
same as the main notebooks:

```bash
make notebooks                                   # builds spine + deep-dives
python scripts/run_all_notebooks.py              # main spine only (CI smoke target)
# run a deep-dive directly:
jupyter notebook deep-dives/csp_geometry.ipynb
```

> Deep-dives are intentionally **excluded from the CI smoke test** (they're heavier and
> optional). Each was verified to execute on CPU when written; if you edit one, run it
> locally to confirm.
