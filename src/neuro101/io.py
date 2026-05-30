"""Loaders for the public datasets, with caching and CPU-friendly subsampling.

Every loader **downloads on first use** and caches; nothing is bundled. Loaders
come in two flavours:

* ``load_*_raw`` — return an MNE ``Raw`` object for plotting / DSP demos.
* ``load_*_epochs`` — return ready-for-ML arrays ``(X, y, subjects)`` where
  ``X`` has shape ``(n_trials, n_channels, n_times)``, ``y`` is integer labels,
  and ``subjects`` gives the subject id of each trial (so you can build a
  subject-aware split with :func:`neuro101.eval.make_subject_split`).

These functions honour ``NEURO101_SMOKE=1`` (load the tiniest slice) via
:mod:`neuro101.datasets`.
"""
from __future__ import annotations

import warnings
from typing import Sequence

# MNE is noisy by default; keep tutorial output readable.
import mne  # noqa: E402
import numpy as np

from . import datasets as ds

mne.set_log_level("ERROR")


def _silence_mne_path_prompts() -> None:
    """Point every MNE dataset at our cache dir so MNE never prompts on stdin.

    MNE asks (via ``input()``) whether to save a download path to its config the
    first time you fetch a dataset. That prompt crashes non-interactive notebook
    execution (CI), so we set the config keys up-front to our cache directory.
    """
    cache = str(ds.cache_dir())
    for key in (
        "MNE_DATA",
        "MNE_DATASETS_EEGBCI_PATH",
        "MNE_DATASETS_SLEEP_PHYSIONET_PATH",
        "MNE_DATASETS_SAMPLE_PATH",
    ):
        try:
            if not mne.get_config(key):
                mne.set_config(key, cache, set_env=False)
        except Exception:
            pass


_silence_mne_path_prompts()


# --------------------------------------------------------------------------- #
# MNE sample dataset (single subject, MEG+EEG) — for first plots & ERPs
# --------------------------------------------------------------------------- #
def load_mne_sample_raw(preload: bool = True) -> "mne.io.Raw":
    """Load the MNE sample dataset as a ``Raw`` object (auto-downloaded once).

    Returns
    -------
    mne.io.Raw
        ~600 Hz MEG+EEG recording of one subject hearing tones / seeing
        checkerboards. See :func:`neuro101.datasets.describe`.
    """
    from mne.datasets import sample

    data_path = sample.data_path(download=True)
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=preload)
    return raw


def load_mne_sample_events(raw: "mne.io.Raw"):
    """Return ``(events, event_id)`` for the MNE sample dataset stimulus channel."""
    events = mne.find_events(raw, stim_channel="STI 014")
    event_id = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
    }
    return events, event_id


# --------------------------------------------------------------------------- #
# BCI Competition IV 2a (motor imagery) — the headline dataset, via MOABB
# --------------------------------------------------------------------------- #
def load_bnci_2a_epochs(
    subjects: Sequence[int] | None = None,
    *,
    n_subjects: int | None = None,
    tmin: float = 0.5,
    tmax: float = 2.5,
    fmin: float = 8.0,
    fmax: float = 30.0,
    classes: Sequence[str] = ("left_hand", "right_hand"),
    return_session: bool = False,
):
    """Load BCI IV 2a motor-imagery epochs as ML-ready arrays via MOABB.

    Parameters
    ----------
    subjects : sequence of int, optional
        1-based subject ids (1..9). If None, the first ``n_subjects`` are used.
    n_subjects : int, optional
        How many subjects to load when ``subjects`` is None. Resolved through
        :func:`neuro101.datasets.n_subjects_to_load` (so smoke mode shrinks it).
    tmin, tmax : float
        Epoch window in seconds, relative to cue onset. The default 0.5–2.5 s
        skips the cue transient and keeps the steady motor-imagery period.
    fmin, fmax : float
        Band-pass band (Hz) applied by the MOABB paradigm. 8–30 Hz captures the
        mu/beta rhythms that motor imagery modulates.
    classes : sequence of str
        Which classes to keep. Default is the 2-class left-vs-right problem,
        which gives the clearest, fastest demo.

    Returns
    -------
    X : np.ndarray (n_trials, n_channels, n_times), float
    y : np.ndarray (n_trials,), int   — 0/1 class labels (label order = `classes`)
    subjects : np.ndarray (n_trials,), int  — subject id per trial

    Notes
    -----
    First call downloads ~0.2 GB per subject (cached by MNE/MOABB).
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    if subjects is None:
        n = ds.n_subjects_to_load("bnci_2a", n_subjects)
        subjects = list(range(1, n + 1))

    paradigm = MotorImagery(
        events=list(classes),
        n_classes=len(classes),
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
    )
    dataset = BNCI2014_001()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=list(subjects))

    # Map string labels -> integer 0..k-1 in the requested class order.
    label_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_int[str(lbl)] for lbl in labels], dtype=int)
    subj = meta["subject"].to_numpy().astype(int)
    if return_session:
        # MOABB labels the two recording days as session strings; map to 0/1.
        sess_raw = meta["session"].to_numpy()
        _, sess = np.unique(sess_raw, return_inverse=True)
        return X.astype(np.float64), y, subj, sess.astype(int)
    return X.astype(np.float64), y, subj


def load_bnci_2a_raw(subject: int = 1):
    """Load one BCI IV 2a subject as a list of MNE ``Raw`` runs (for plotting/DSP)."""
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject])
    # data[subject][session][run] -> Raw
    sessions = data[subject]
    first_session = next(iter(sessions.values()))
    raws = list(first_session.values())
    return raws


# --------------------------------------------------------------------------- #
# PhysioNet EEG Motor Movement/Imagery — cross-subject backup
# --------------------------------------------------------------------------- #
def load_physionet_mi_raw(subject: int = 1, runs: Sequence[int] = (4, 8, 12)) -> "mne.io.Raw":
    """Load one PhysioNet subject's runs as a single continuous ``Raw`` (for DSP/ICA).

    Channels are renamed to the standard 10-05 names and a montage is set, so the
    frontal electrodes (Fp1/Fp2/Fpz) can serve as eye-blink proxies for ICA.
    """
    from mne.datasets import eegbci
    from mne.io import concatenate_raws

    fnames = eegbci.load_data(subject, list(runs), path=str(ds.cache_dir()), update_path=False)
    raw = concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in fnames])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    return raw


def load_physionet_mi_epochs(
    subjects: Sequence[int] | None = None,
    *,
    n_subjects: int | None = None,
    tmin: float = 1.0,
    tmax: float = 2.0,
    runs: Sequence[int] = (4, 8, 12),
):
    """Load PhysioNet motor-imagery epochs as ``(X, y, subjects)``.

    Uses runs 4/8/12 (imagined left vs right fist). Labels: 0 = left, 1 = right.
    First call downloads ~40 MB per subject.
    """
    from mne.datasets import eegbci
    from mne.io import concatenate_raws

    if subjects is None:
        n = ds.n_subjects_to_load("physionet_mi", n_subjects)
        subjects = list(range(1, n + 1))

    all_X, all_y, all_s = [], [], []
    for s in subjects:
        fnames = eegbci.load_data(s, list(runs), path=str(ds.cache_dir()), update_path=False)
        raws = [mne.io.read_raw_edf(f, preload=True) for f in fnames]
        raw = concatenate_raws(raws)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")
        raw.filter(8.0, 30.0, verbose="ERROR")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=0, T2=1))
        epochs = mne.Epochs(
            raw, events, event_id=dict(left=0, right=1),
            tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR",
        )
        X = epochs.get_data(copy=False)
        y = epochs.events[:, -1]
        all_X.append(X)
        all_y.append(y)
        all_s.append(np.full(len(y), s, dtype=int))

    # Channels are consistent across subjects for these runs.
    return (
        np.concatenate(all_X).astype(np.float64),
        np.concatenate(all_y).astype(int),
        np.concatenate(all_s),
    )


# --------------------------------------------------------------------------- #
# Sleep-EDF — sleep staging & class-imbalance demo
# --------------------------------------------------------------------------- #
_SLEEP_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge stage 3+4 into N3 (standard AASM practice)
    "Sleep stage R": 4,
}
SLEEP_STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def load_sleep_edf_epochs(
    subjects: Sequence[int] | None = None,
    *,
    n_subjects: int | None = None,
    epoch_sec: float = 30.0,
    channel: str = "EEG Fpz-Cz",
):
    """Load Sleep-EDF as 30-second epochs labelled by sleep stage.

    Returns ``(X, y, subjects)`` with ``X`` of shape (n_epochs, 1, n_times).
    Labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM (see :data:`SLEEP_STAGE_NAMES`).
    The class distribution is naturally imbalanced (lots of N2) — used in the
    Chapter 09 metrics demo. First call downloads ~8 MB per recording.
    """
    from mne.datasets.sleep_physionet.age import fetch_data

    if subjects is None:
        n = ds.n_subjects_to_load("sleep_edf", n_subjects)
        subjects = list(range(n))

    all_X, all_y, all_s = [], [], []
    for s in subjects:
        # One night (recording 1) per subject keeps it small.
        paths = fetch_data(subjects=[s], recording=[1], on_missing="warn")
        if not paths:
            continue
        psg, hyp = paths[0]
        raw = mne.io.read_raw_edf(psg, preload=True, stim_channel=False)
        ann = mne.read_annotations(hyp)
        raw.set_annotations(ann, emit_warning=False)
        events, _ = mne.events_from_annotations(
            raw, event_id=_SLEEP_STAGE_MAP, chunk_duration=epoch_sec, verbose="ERROR"
        )
        tmax = epoch_sec - 1.0 / raw.info["sfreq"]
        epochs = mne.Epochs(
            raw, events, event_id={n: i for n, i in
                                   ((nm, ix) for ix, nm in enumerate(SLEEP_STAGE_NAMES))},
            tmin=0.0, tmax=tmax, baseline=None, preload=True,
            picks=[channel], on_missing="ignore", verbose="ERROR",
        )
        X = epochs.get_data(copy=False)
        y = epochs.events[:, -1]
        all_X.append(X)
        all_y.append(y)
        all_s.append(np.full(len(y), s, dtype=int))

    return (
        np.concatenate(all_X).astype(np.float64),
        np.concatenate(all_y).astype(int),
        np.concatenate(all_s),
    )
