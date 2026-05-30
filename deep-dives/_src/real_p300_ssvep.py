# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Deep-dive — Real P300 & SSVEP (MOABB)
#
# Decoding real EEG data for two classic BCI paradigms using publicly available
# datasets, end-to-end reproducible pipelines, and leak-free cross-validation.
#
# > **Prerequisites:** main Chapters 10 and 12.
# > **Level:** advanced ★★★☆☆
# > **Downloads real public datasets (cached). Not bound by the 5-min budget.**

# %% Bootstrap — robust upward search for neuro101 src
import sys
import os
from pathlib import Path

# Strategy: try installed package first, then walk up directory tree to find src/
try:
    import neuro101  # noqa: F401
except ModuleNotFoundError:
    # Walk up from this file's location (or cwd when executed as notebook)
    _search_root = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
    for _parent in [_search_root, *_search_root.parents]:
        _candidate = _parent / "src"
        if (_candidate / "neuro101" / "__init__.py").exists():
            sys.path.insert(0, str(_candidate))
            break
    try:
        import neuro101  # noqa: F401
    except ModuleNotFoundError:
        # Last-resort fallback: assume notebook lives 2 levels below repo root
        sys.path.insert(0, str(Path.cwd().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
from scipy.signal import welch

import mne
mne.set_log_level("ERROR")

from neuro101 import datasets as ds

rng = np.random.default_rng(42)
SMOKE = ds.is_smoke()
print(f"Smoke mode: {SMOKE}  |  numpy {np.__version__}  |  mne {mne.__version__}")

# %% [markdown]
# ---
# ## Part 1 — Real P300 Decoding (BNCI2014-009)
#
# ### Why this matters
#
# Chapter 10 simulated P300 trials to keep runtime short.  Here we use
# **BNCI2014-009** — a real P300 speller dataset recorded at 256 Hz from 16
# channels.  Ten participants performed a standard 6×6 matrix speller.  The
# dataset is freely available via MOABB and weighs only ~185 MB for all 10
# subjects (one ~18 MB file per subject).
#
# **Class imbalance note (Chapter 12, pitfall #4):**  In a 6×6 speller, each
# cell is highlighted once per "repetition cycle".  The row/column that contains
# the target is highlighted only 2 out of every 12 flashes → roughly **83% non-
# target, 17% target**.  Accuracy alone is therefore misleading; we report
# **balanced accuracy** and **ROC-AUC**.
#
# **Pipeline:**  xDAWN spatial filtering (learns ERP-enhancing filters) followed
# by Riemannian covariance tangent-space projection and LDA classification.
# This is the approach of Barachant & Congedo (2014) and remains a strong
# ERP decoder.

# %%
p300_section_ok = False  # track whether this section ran successfully

try:
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    print("Loading BNCI2014-009 — subject 1 (~18 MB, cached after first run)...")
    _dataset_p300 = BNCI2014_009()
    _paradigm_p300 = P300()
    X_p300, y_p300, meta_p300 = _paradigm_p300.get_data(
        dataset=_dataset_p300, subjects=[1]
    )
    # X_p300 shape: (n_epochs, n_channels, n_times)  e.g. (1728, 16, 206)
    # y_p300: string labels "Target" or "NonTarget"
    # meta_p300: DataFrame with 'session' column (sessions 0/1/2)
    print(f"Loaded: {X_p300.shape}  labels={np.unique(y_p300, return_counts=True)}")

    le_p300 = LabelEncoder()
    y_enc_p300 = le_p300.fit_transform(y_p300)  # NonTarget=0, Target=1
    target_class_idx = int(np.where(le_p300.classes_ == "Target")[0])
    print(f"Label encoding: {dict(zip(le_p300.classes_, [0, 1]))}")
    print(f"Class balance: {np.bincount(y_enc_p300)} → "
          f"{100*y_enc_p300.mean():.1f}% Target (imbalanced!)")

    p300_section_ok = True

except Exception as _e:
    print(f"WARNING: BNCI2014-009 could not be loaded ({_e}). Trying fallback...")
    try:
        # Fallback: EPFLP300 dataset (smaller, different recording)
        from moabb.datasets import EPFLP300
        from moabb.paradigms import P300
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score

        print("Fallback: loading EPFLP300 — subject 1...")
        _dataset_p300 = EPFLP300()
        _paradigm_p300 = P300()
        X_p300, y_p300, meta_p300 = _paradigm_p300.get_data(
            dataset=_dataset_p300, subjects=[1]
        )
        print(f"Loaded: {X_p300.shape}  labels={np.unique(y_p300, return_counts=True)}")
        le_p300 = LabelEncoder()
        y_enc_p300 = le_p300.fit_transform(y_p300)
        target_class_idx = int(np.where(le_p300.classes_ == "Target")[0])
        p300_section_ok = True

    except Exception as _e2:
        print(f"WARNING: Both P300 datasets failed ({_e2}). Skipping P300 section.")

# %% [markdown]
# ### Grand-average ERP: Target vs Non-Target
#
# Each raw epoch is a *single noisy snapshot*.  Averaging across many trials
# cancels the noise (it is random and zero-mean) while preserving the
# consistent P300 waveform present only in Target trials.  The P300 component
# should peak at approximately 300–500 ms post-stimulus.

# %%
if p300_section_ok:
    # Reconstruct time axis from paradigm interval and number of samples
    t_start, t_end = _dataset_p300.interval  # seconds, e.g. [0, 0.8]
    n_times_p300 = X_p300.shape[2]
    t_p300 = np.linspace(t_start, t_end, n_times_p300, endpoint=False) * 1000  # ms

    # Select a central parietal channel (Pz) for the canonical P300 view.
    # Channel index depends on the recording montage; use Cz (index ~0) if Pz unavailable.
    # BNCI2014-009 montage has channels including Fz, Cz, Pz — try Pz first.
    try:
        # Peek at the raw to get channel names
        _raw_peek = _dataset_p300.get_data(subjects=[1])
        _ch_names = list(list(list(_raw_peek[1].values())[0].values())[0].info["ch_names"])
        pz_name = next((c for c in _ch_names if "Pz" in c or "pz" in c), _ch_names[0])
        pz_idx = _ch_names.index(pz_name)
    except Exception:
        pz_idx = 0
        pz_name = "ch0"

    tgt_mask = y_enc_p300 == target_class_idx
    nontgt_mask = ~tgt_mask

    # MOABB P300 paradigm returns data already scaled to µV — no conversion needed
    erp_tgt = X_p300[tgt_mask, pz_idx, :].mean(axis=0)    # µV
    erp_nontgt = X_p300[nontgt_mask, pz_idx, :].mean(axis=0)

    n_tgt = tgt_mask.sum()
    n_nontgt = nontgt_mask.sum()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_p300, erp_nontgt, color="#4c72b0", lw=2,
            label=f"Non-Target grand avg (n={n_nontgt})")
    ax.plot(t_p300, erp_tgt, color="#c44e52", lw=2.5,
            label=f"Target grand avg (n={n_tgt})")
    ax.axvline(300, ls="--", color="gray", alpha=0.7, label="300 ms")
    ax.axhline(0, color="k", lw=0.5)
    # Mark approximate P300 peak
    p300_window = (t_p300 >= 250) & (t_p300 <= 600)
    if p300_window.any():
        peak_t = t_p300[p300_window][np.argmax(erp_tgt[p300_window] - erp_nontgt[p300_window])]
        ax.axvline(peak_t, ls=":", color="#c44e52", alpha=0.8, label=f"P300 peak ~{peak_t:.0f} ms")
    ax.set(xlabel="Time (ms)", ylabel="Amplitude (µV)",
           title=f"P300 grand-average ERP — real BNCI2014-009, subject 1 ({pz_name})",
           xlim=(t_p300[0], t_p300[-1]))
    ax.legend(fontsize=9, loc="upper right")
    ax.fill_between(t_p300, erp_nontgt, erp_tgt, alpha=0.12, color="#c44e52",
                    label="Target–NonTarget diff")
    fig.tight_layout()
    plt.savefig("/tmp/p300_erp.png", dpi=120)
    plt.show()
    print(f"Grand-avg P300 amplitude (Target - NonTarget) at Pz: "
          f"{(erp_tgt - erp_nontgt).max():.2f} µV")
else:
    print("P300 section skipped — no data available.")

# %% [markdown]
# ### Decoding: xDAWN + Riemannian + LDA (leave-one-session-out)
#
# **Splitting strategy:** BNCI2014-009 records 3 sessions per subject.  We use
# **leave-one-session-out (LOSO)** cross-validation, so training and test sets
# come from **entirely separate recording runs**.  This matters because:
#
# 1. P300 epochs from the *same* stimulation sequence share the time-locked EEG
#    noise of neighbouring trials — splitting randomly would leak correlated
#    samples between train and test (the subtle trap explored at the end of this
#    notebook).
# 2. Physiological non-stationarity means session-level splits give a realistic
#    estimate of within-subject generalisation.
#
# **Imbalance note:** ROC-AUC is insensitive to class imbalance and is the
# recommended metric here (Chapter 12, pitfall #4).  Balanced accuracy is also
# reported for completeness.

# %%
if p300_section_ok:
    pipe_p300 = make_pipeline(
        XdawnCovariances(nfilter=4, estimator="oas"),
        TangentSpace(metric="riemann"),
        LinearDiscriminantAnalysis(),
    )

    sessions_arr = meta_p300["session"].values
    session_le = LabelEncoder()
    session_groups = session_le.fit_transform(sessions_arr)
    unique_sess = np.unique(session_groups)
    print(f"Cross-validation: leave-one-session-out over {len(unique_sess)} sessions")

    logo = LeaveOneGroupOut()
    aucs, bal_accs = [], []

    for fold_i, (train_idx, test_idx) in enumerate(
        logo.split(X_p300, y_enc_p300, groups=session_groups)
    ):
        X_tr, X_te = X_p300[train_idx], X_p300[test_idx]
        y_tr, y_te = y_enc_p300[train_idx], y_enc_p300[test_idx]

        pipe_p300.fit(X_tr, y_tr)
        proba = pipe_p300.predict_proba(X_te)[:, target_class_idx]
        y_pred = pipe_p300.predict(X_te)

        auc = roc_auc_score(y_te, proba)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        aucs.append(auc)
        bal_accs.append(bal_acc)
        print(f"  Session fold {fold_i+1}: ROC-AUC={auc:.3f}  bal_acc={bal_acc:.3f}")

    mean_auc = np.mean(aucs)
    mean_bal_acc = np.mean(bal_accs)
    print(f"\nMean ROC-AUC : {mean_auc:.3f}  (random classifier = 0.50)")
    print(f"Mean balanced accuracy: {mean_bal_acc:.3f}  (chance = 0.50)")
    print(f"\nKey: ROC-AUC >> 0.5 despite heavy class imbalance (~5:1 NonTarget:Target).")
    print("This is why we do NOT rely on raw accuracy (which would be ~83% even for a")
    print("trivial all-NonTarget classifier).")
else:
    mean_auc = float("nan")
    mean_bal_acc = float("nan")
    print("P300 decoding skipped.")

# %% [markdown]
# ---
# ## Part 2 — Real SSVEP Decoding (Nakanishi 2015)
#
# ### Why this matters
#
# Chapter 10 simulated SSVEP by injecting a pure sine wave into noise.  Real
# SSVEP is messier: background alpha oscillations overlap with stimulus
# frequencies, neighbouring frequencies are close together, and the response
# strength varies with attentional state and fatigue.
#
# **Dataset:** Nakanishi et al. (2015) — 9 subjects, 8-channel occipital EEG
# at 256 Hz, with 12 flickering stimuli (9.25, 9.75, 10.25, …, 14.75 Hz in
# 0.5 Hz steps).  We load the **4 lowest-frequency classes** to keep the
# decoding tractable while still being genuinely multi-class.
#
# **Pipeline:**  For each epoch we compute the Welch PSD and extract the mean
# power at each stimulus frequency (summed over harmonics and channels).  This
# gives a 4-dimensional feature vector that a logistic regression then
# classifies.  Chance level is **25%** (4 classes, balanced).

# %%
ssvep_section_ok = False

try:
    from moabb.datasets import Nakanishi2015
    from moabb.paradigms import SSVEP
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder as LE
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print("Loading Nakanishi2015 SSVEP — subject 1 (~15 MB, cached after first run)...")
    _dataset_ssvep = Nakanishi2015()
    _paradigm_ssvep = SSVEP(n_classes=4)  # first 4 classes (lowest freqs)
    X_ssvep, y_ssvep, meta_ssvep = _paradigm_ssvep.get_data(
        dataset=_dataset_ssvep, subjects=[1]
    )
    print(f"Loaded: {X_ssvep.shape}  freqs={np.unique(y_ssvep)}")
    ssvep_section_ok = True

except Exception as _e:
    print(f"WARNING: Nakanishi2015 could not be loaded ({_e}). Trying fallback...")
    try:
        # Fallback: SSVEPExo / Kalunga2016
        from moabb.datasets import Kalunga2016
        from moabb.paradigms import SSVEP
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder as LE
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        print("Fallback: loading Kalunga2016 SSVEP — subject 1...")
        _dataset_ssvep = Kalunga2016()
        _paradigm_ssvep = SSVEP()
        X_ssvep, y_ssvep, meta_ssvep = _paradigm_ssvep.get_data(
            dataset=_dataset_ssvep, subjects=[1]
        )
        print(f"Loaded: {X_ssvep.shape}  freqs={np.unique(y_ssvep)}")
        ssvep_section_ok = True

    except Exception as _e2:
        print(f"WARNING: Both SSVEP datasets failed ({_e2}). Skipping SSVEP section.")

# %% [markdown]
# ### PSD visualisation: the attended frequency dominates occipital power
#
# We pick one epoch where the participant was attending to the lowest stimulus
# frequency and plot the Welch PSD.  The attended stimulus frequency and its
# harmonics should appear as clear peaks above the noise floor.

# %%
if ssvep_section_ok:
    # Infer sampling frequency from epoch duration and n_times.
    # Nakanishi2015 records at 256 Hz with 4 s epochs → 1025 samples
    # (with a small buffer).  We recover sfreq from the raw data.
    try:
        _raw_ssvep = _dataset_ssvep.get_data(subjects=[1])
        sfreq_ssvep = float(
            list(list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"])
            if hasattr(list(list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"], "__len__"))
            else list(list(_raw_ssvep[1].values())[0].values())[0].info["sfreq"]
        )
    except Exception:
        # Heuristic: Nakanishi2015 = 256 Hz; Kalunga2016 = 256 Hz
        sfreq_ssvep = 256.0

    stim_freqs_str = np.unique(y_ssvep)
    stim_freqs = sorted([float(f) for f in stim_freqs_str])
    n_classes_ssvep = len(stim_freqs)
    print(f"Stimulus frequencies: {stim_freqs} Hz  (sfreq={sfreq_ssvep} Hz)")

    # Choose an example epoch for the attended frequency = stim_freqs[0]
    attended_freq = stim_freqs[0]
    attended_label = str(attended_freq) if str(attended_freq) in y_ssvep else stim_freqs_str[0]
    attended_idx = np.where(y_ssvep == attended_label)[0][0]
    epoch_ex = X_ssvep[attended_idx]  # (n_channels, n_times)

    n_times_ssvep = epoch_ex.shape[1]
    nperseg = min(512, n_times_ssvep)
    freqs_psd, psd_ex = welch(epoch_ex, sfreq_ssvep, nperseg=nperseg, axis=-1)
    # Average over occipital channels (use all channels for simplicity)
    psd_mean = psd_ex.mean(axis=0)

    # Identify harmonic peaks
    colors_stim = ["#c44e52", "#4c72b0", "#55a868", "#e69f00"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    ax = axes[0]
    ax.semilogy(freqs_psd, psd_mean, color="#555", lw=1, zorder=2)
    for fi, (sf, col) in enumerate(zip(stim_freqs, colors_stim)):
        for h in range(1, 4):
            fh = sf * h
            if fh > sfreq_ssvep / 2:
                break
            peak_idx = np.argmin(np.abs(freqs_psd - fh))
            label = f"{sf} Hz (attended)" if (sf == attended_freq and h == 1) else (
                f"{sf} Hz" if h == 1 else None)
            ax.axvline(fh, color=col, lw=1.5 if sf == attended_freq else 0.8,
                       ls="-" if sf == attended_freq else "--",
                       alpha=0.9 if sf == attended_freq else 0.5,
                       label=label, zorder=1)
            if h == 1:
                ax.annotate(f"{sf}\n({h}f)",
                            xy=(fh, psd_mean[peak_idx]),
                            xytext=(fh + 0.3, psd_mean[peak_idx] * 2),
                            fontsize=7, color=col,
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    ax.set(xlim=(5, min(40, sfreq_ssvep/2)),
           xlabel="Frequency (Hz)", ylabel="Power spectral density (µV²/Hz)",
           title=f"SSVEP epoch PSD — attending {attended_freq} Hz\n"
                 f"(real Nakanishi2015, subject 1)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    # Right panel: single-trial feature vectors (power at each stim freq)
    def extract_psd_features(X, sfreq, stim_freqs, n_harmonics=3):
        """For each epoch: sum Welch power at each stimulus frequency and its harmonics."""
        n_epochs, n_ch, n_t = X.shape
        nperseg = min(512, n_t)
        feats = np.zeros((n_epochs, len(stim_freqs)))
        for i, ep in enumerate(X):
            f, psd = welch(ep, sfreq, nperseg=nperseg, axis=-1)
            for j, sf in enumerate(stim_freqs):
                power = 0.0
                for h in range(1, n_harmonics + 1):
                    fh = sf * h
                    if fh >= sfreq / 2:
                        break
                    idx = np.argmin(np.abs(f - fh))
                    power += psd[:, idx].mean()
                feats[i, j] = power
        return feats

    F_ssvep = extract_psd_features(X_ssvep, sfreq_ssvep, stim_freqs, n_harmonics=3)
    le_ssvep = LE()
    y_enc_ssvep = le_ssvep.fit_transform(y_ssvep)

    ax2 = axes[1]
    for ci, (sf, col) in enumerate(zip(stim_freqs, colors_stim)):
        mask = y_enc_ssvep == ci
        vals = F_ssvep[mask, ci]
        ax2.scatter(
            np.full(vals.shape, sf) + rng.uniform(-0.08, 0.08, vals.shape),
            vals, alpha=0.5, s=18, color=col, label=f"{sf} Hz"
        )
        ax2.hlines(vals.mean(), sf - 0.2, sf + 0.2, color=col, lw=2.5)
    ax2.set(xlabel="Stimulus frequency attended (Hz)",
            ylabel="Welch power at attended freq (µV²/Hz)",
            title="Feature separability: power at attended freq\nby true class (mean = thick bar)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig("/tmp/ssvep_psd.png", dpi=120)
    plt.show()

else:
    print("SSVEP visualisation skipped.")

# %% [markdown]
# ### Decoding: PSD features + Logistic Regression (stratified 5-fold CV)
#
# We decode which frequency was attended using only the spectral power at each
# stimulus frequency (and its first two harmonics) averaged across all occipital
# channels.  Chance level for a 4-class balanced problem is 25 %.

# %%
if ssvep_section_ok:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipe_ssvep = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    )

    n_splits_ssvep = 5
    cv_ssvep = StratifiedKFold(n_splits=n_splits_ssvep, shuffle=True, random_state=42)
    scores_ssvep = cross_val_score(
        pipe_ssvep, F_ssvep, y_enc_ssvep, cv=cv_ssvep, scoring="accuracy"
    )
    mean_acc_ssvep = scores_ssvep.mean()
    chance_ssvep = 1.0 / n_classes_ssvep

    print(f"SSVEP 4-class accuracy: {mean_acc_ssvep:.3f} ± {scores_ssvep.std():.3f}")
    print(f"Chance level: {chance_ssvep:.3f}  (4 balanced classes)")
    print(f"Lift over chance: +{mean_acc_ssvep - chance_ssvep:.3f} "
          f"({100*(mean_acc_ssvep/chance_ssvep - 1):.0f}% relative improvement)")
    print()
    print("Note: The modest absolute accuracy reflects the difficulty of decoding")
    print("closely-spaced frequencies (Δf = 0.5 Hz) with short epochs and no")
    print("subject-specific calibration.  The Riemannian approach in the P300 section")
    print("would also improve SSVEP decoding (see moabb.evaluations for benchmarks).")
else:
    mean_acc_ssvep = float("nan")
    chance_ssvep = float("nan")
    print("SSVEP decoding skipped.")

# %% [markdown]
# ### Summary of results
#
# | Paradigm | Dataset | Metric | Score | Chance |
# |----------|---------|--------|-------|--------|
# | P300 | BNCI2014-009 | ROC-AUC (LOSO) | — | 0.50 |
# | SSVEP | Nakanishi2015 | Accuracy (5-fold) | — | 0.25 |
#
# *(Filled in at runtime by the cells above.)*

# %%
print("=" * 55)
if p300_section_ok:
    print(f"  P300  ROC-AUC          : {mean_auc:.3f}  (chance 0.50)")
    print(f"  P300  Balanced accuracy: {mean_bal_acc:.3f}  (chance 0.50)")
else:
    print("  P300  section: SKIPPED (dataset unavailable)")
if ssvep_section_ok:
    print(f"  SSVEP accuracy         : {mean_acc_ssvep:.3f}  (chance {chance_ssvep:.2f})")
else:
    print("  SSVEP section: SKIPPED (dataset unavailable)")
print("=" * 55)

# %% [markdown]
# ---
# ## A subtler trap
#
# ### The correlated-epoch leakage problem in P300 decoding
#
# Here is a pitfall that is easy to overlook even for careful researchers.
#
# **The problem.**  In a P300 speller, stimuli are presented in structured
# *sequences* (e.g. rows and columns of a matrix, one per ~100 ms).  Over a
# single "repetition cycle" the EEG is one continuous segment: the residual
# cortical response from flash #k (a slow positive wave lasting up to ~800 ms)
# is still physically present when flash #k+1 or #k+2 arrives.  This means
# neighbouring epochs are **not independent** — they share overlapping brain
# responses.
#
# **Why random splitting fails.**  If you split epochs randomly into train/test
# (e.g. with `train_test_split`), you will often place epoch #k in training and
# epoch #k+1 in the test set.  The classifier has, in effect, "seen" the brain
# state of the test epoch through its temporally adjacent training neighbour.
# This inflates ROC-AUC estimates — sometimes by as much as 0.05–0.10 AUC
# points — without the researcher noticing, because the data structure is
# opaque once it has been extracted into a NumPy array.
#
# **The fix used above.**  We split by *session* (leave-one-session-out), which
# guarantees that entire recording blocks are held out.  Epochs within the same
# sequence *always* land in the same fold.  This is the approach recommended by
# the MOABB benchmark and is essential for reproducible ERP decoding.
#
# **The SSVEP analogue.**  A related trap in SSVEP is conflating the *stimulus
# artifact* (electrical noise leaking from the stimulus display into the EEG
# amplifier at exactly the flicker frequency) with the genuine brain SSVEP
# response.  A decoder trained on contaminated data will still achieve high
# accuracy — but only because it is reading the display artifact, not the
# participant's attention.  The diagnostic test: replay the stimulus while the
# participant looks *away* from the screen; the artifact-driven classifier
# will still "decode" the attended frequency perfectly, revealing that it never
# read the brain at all.  Ground truth: verify that removing the occipital
# channels (the ones that genuinely carry the visual response) substantially
# drops accuracy; if accuracy is equally high from frontal or temporal channels
# alone, suspect an artifact.

# %%
print("Deep-dive complete.  Key takeaways:")
print("  1. Real P300 data is 83 % non-target → use ROC-AUC / balanced accuracy.")
print("  2. Correlations between neighbouring P300 epochs demand block-level splits.")
print("  3. SSVEP 'decoding' can be driven by stimulus artifact — verify with")
print("     channel ablation and off-screen controls before claiming brain decoding.")
