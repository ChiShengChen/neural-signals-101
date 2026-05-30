# Glossary & Acronym Index

**English** · [繁體中文](zh-TW/GLOSSARY.md)

Every term and acronym used in the tutorial, in one place. Terms are also defined
on first use inside the notebooks — this page is for quick lookup. Roughly grouped;
use your browser's find (Ctrl/Cmd-F).

## Signals & neuroscience
- **EEG** (Electroencephalography) — recording the brain's electrical activity from
  electrodes on the scalp. Cheap, non-invasive, low signal-to-noise ratio.
- **MEG** (Magnetoencephalography) — records the magnetic fields produced by the same
  neural currents; needs a shielded room.
- **ECoG** (Electrocorticography) — electrodes placed directly on the cortical
  surface during surgery; high quality, invasive.
- **LFP** (Local Field Potential) — voltage from a small population of neurons,
  measured by an electrode inside tissue.
- **Spike** — a single neuron's action potential (~1 ms electrical pulse).
- **fNIRS** (functional Near-Infrared Spectroscopy) — measures blood oxygenation
  (a slow proxy for activity) with light through the scalp.
- **EMG** (Electromyography) — electrical activity of muscles; a common EEG artefact.
- **EOG** (Electrooculography) — electrical activity from eye movements/blinks; a
  major EEG artefact.
- **ECG / EKG** (Electrocardiography) — the heartbeat signal; can contaminate EEG.
- **PSP** (Post-Synaptic Potential) — the slow voltage change in a neuron that, summed
  over thousands of aligned cells, produces the EEG signal.
- **Pyramidal neuron** — the elongated cortical cell type whose synchronized PSPs
  generate most of the scalp EEG.
- **Volume conduction** — the spreading/blurring of a brain source through tissue,
  skull and scalp, which makes neighbouring electrodes correlated.
- **µV (microvolt)** — one millionth of a volt; the typical scale of EEG.
- **10-20 system** — the standard scheme for naming/placing scalp electrodes
  (e.g. Fz, Cz, C3, C4, Oz). Letters = brain region, odd numbers = left, even = right.
- **Montage** — the map from channel names to 3-D electrode positions on the head.
- **Reference** — EEG values are voltage *differences*; the reference is the point
  you subtract (e.g. average reference = mean over all electrodes).
- **SNR** (Signal-to-Noise Ratio) — how strong the signal is relative to the noise.

## Brain rhythms & responses
- **Delta / Theta / Alpha / Beta / Gamma** — standard EEG frequency bands
  (≈1–4 / 4–8 / 8–13 / 13–30 / 30–45 Hz).
- **Mu rhythm** — ~8–12 Hz rhythm over sensorimotor cortex, suppressed by movement.
- **ERD / ERS** (Event-Related De-/Synchronization) — a drop / rebound in band power
  caused by an event (e.g. contralateral mu/beta ERD during hand movement imagery).
- **ERP** (Event-Related Potential) — the brain's averaged voltage response to an
  event, revealed by averaging many trials.
- **P300** — a positive ERP peak ~300 ms after a rare, attended stimulus.
- **Oddball paradigm** — an experiment where targets are made rare so they evoke a P300.
- **SSVEP** (Steady-State Visual Evoked Potential) — the visual cortex oscillating at
  the frequency of a flickering stimulus ("frequency tagging").
- **Sleep stages** — Wake, N1, N2, N3 (deep), REM; each has signature rhythms
  (e.g. spindles/K-complexes in N2, delta in N3).
- **Seizure** — hypersynchronous, high-amplitude rhythmic neural discharge.

## Signal processing (DSP)
- **DSP** (Digital Signal Processing) — manipulating sampled signals numerically.
- **Sampling rate / `sfreq`** — samples recorded per second (Hz).
- **Nyquist frequency** — half the sampling rate; the highest representable frequency.
- **Aliasing** — a too-high frequency masquerading as a lower one when undersampled.
- **Quantization** — rounding continuous voltage to discrete digital levels.
- **Filter** — keeps/removes frequency ranges. **Low-pass / high-pass / band-pass /
  notch (band-stop)**.
- **FIR / IIR** (Finite / Infinite Impulse Response) — two filter families; FIR is
  always stable with linear phase, IIR is cheaper but can distort phase.
- **Zero-phase filtering** — applying a filter forwards and backwards (`filtfilt`) to
  cancel phase distortion.
- **FFT** (Fast Fourier Transform) — fast computation of a signal's frequency content.
- **PSD** (Power Spectral Density) — how power is distributed across frequency.
- **Welch's method** — averaging spectra of overlapping windows for a stable PSD.
- **STFT** (Short-Time Fourier Transform) / **Spectrogram** — frequency content over time.
- **Wavelet** — time-frequency analysis with frequency-dependent window length.
- **Band power** — signal energy within a frequency band (a common feature).
- **Epoch / trial** — a short, labelled segment cut from a continuous recording.
- **Baseline correction** — subtracting a pre-event mean so epochs start at zero.
- **ICA** (Independent Component Analysis) — separates a recording into independent
  sources so artefacts (blinks, heartbeat) can be removed.
- **ASR** (Artifact Subspace Reconstruction) — reconstructs corrupted segments from a
  clean subspace; we use a simplified amplitude-clipping stand-in.

## Features & models
- **Feature** — a number (or vector) summarising a trial for a model.
- **Connectivity** — how channels relate: **coherence** (frequency-domain
  correlation), **PLV** (Phase-Locking Value, phase synchrony 0–1).
- **CSP** (Common Spatial Patterns) — learns channel mixtures (spatial filters) that
  maximise the variance difference between two classes; a motor-imagery workhorse.
- **Covariance matrix** — how every pair of channels co-varies within a trial.
- **Riemannian / tangent space** — treating covariance matrices on their curved
  geometry and projecting to a flat space for ordinary classifiers; a strong baseline.
- **LDA / SVM / RF** — Linear Discriminant Analysis / Support Vector Machine / Random
  Forest: classical classifiers.
- **Pipeline** — an sklearn object chaining transforms + a model so every step is fit
  on training data only (prevents leakage).
- **EEGNet / ShallowConvNet / DeepConvNet** — convolutional neural nets designed for EEG.
- **LSTM** (Long Short-Term Memory) — a recurrent neural network for sequences.
- **Transformer** — a neural network based on self-attention.
- **Self-supervised / foundation model** — pre-training on large unlabelled data, then
  fine-tuning on a small labelled task.

## Machine-learning fundamentals
- **Features (X) / labels (y)** — inputs and the target to predict.
- **Train / validation / test** — data for fitting / tuning / a single final, honest
  estimate. **The test set is sacred** — touch it once, at the end.
- **Overfitting** — fitting noise in the training data; great train score, poor test.
- **Decision boundary** — the surface where a classifier switches its prediction.
- **Cross-validation (CV)** — repeatedly splitting into train/test to estimate
  performance more stably.
- **Chance level** — the score of a trivial baseline (NOT always 1/n_classes — for
  imbalanced data it is the majority-class fraction).
- **Accuracy / Balanced accuracy / F1 / ROC-AUC** — classification metrics;
  balanced accuracy and F1 are fairer than accuracy under class imbalance.
- **Confusion matrix** — table of true vs predicted classes.
- **mean ± std** — average and spread of a metric over folds/seeds; always report it.
- **Confidence interval (CI)** — a range that plausibly contains the true value.
- **Paired test** (t-test / Wilcoxon) — compares two models fold-by-fold before
  claiming one is better.

## Evaluation honesty (the heart of this tutorial)
- **Leakage** — test information sneaking into training (directly, or via a transform
  fit on all data); inflates scores.
- **Random-shuffle split** — splitting correlated time series randomly; leaks adjacent
  samples. **Forbidden in this repo.**
- **Block / trial-aware split** — keeping whole trials/blocks on one side (`make_block_split`).
- **Subject-dependent vs subject-independent** — testing on the same vs new people.
- **LOSO** (Leave-One-Subject-Out) — hold out all of one subject; the honest headline.
- **Domain / distribution shift** — train and test come from different conditions
  (sessions, days, devices); performance drops.
- **Non-stationarity** — signal statistics changing over time.

## BCI & datasets
- **BCI** (Brain-Computer Interface) — a system that turns brain signals into commands.
- **Motor imagery (MI)** — imagining movement to control a BCI.
- **Brain-to-text / speech neuroprosthesis** — decoding intended speech, typically from
  invasive implants.
- **Neuro-rights** — proposed rights protecting mental privacy and agency.
- **Dual-use** — technology usable for both beneficial and harmful ends.
- **BCI Competition IV 2a** — a standard 9-subject motor-imagery EEG dataset (this
  tutorial's headline data).
- **PhysioNet / Sleep-EDF / MNE sample** — the other public datasets used here.
- **MOABB** (Mother of All BCI Benchmarks) — a library for standardized BCI datasets.
- **MNE** — the standard Python library for EEG/MEG analysis.
- **LSL** (Lab Streaming Layer) — a protocol for streaming live signals; **BrainFlow** /
  **MNE-LSL** are common tools for consumer EEG (OpenBCI, Muse, Emotiv).

## Repo-specific
- **`neuro101`** — this tutorial's importable helper package (`src/neuro101/`).
- **`NEURO101_SMOKE=1`** — environment flag to load the tiniest data slices (used by CI).
- **`NEURO101_DATA`** — environment variable to override the dataset cache directory.
- **Smoke mode** — fast, subsampled execution so notebooks run in seconds.
