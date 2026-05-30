"""Preprocessing & denoising helpers (filters, referencing, epoching, ICA).

These wrap MNE and SciPy with beginner-friendly defaults and docstrings. Where a
function fits parameters from data (e.g. a scaler), it is designed to be used
**inside** a leakage-safe pipeline — see :mod:`neuro101.eval`. Filtering and
notch are stateless (they apply the same fixed response to any segment) so they
are safe to run before splitting.

Terms:

* **Band-pass filter** — keeps frequencies inside a band, removes the rest.
* **Notch filter** — removes a single narrow frequency (here: mains hum, 50/60 Hz).
* **Re-referencing** — EEG voltages are differences; choosing a reference (e.g.
  the average of all electrodes) changes every channel's values consistently.
* **Epoching** — cutting a continuous recording into fixed-length labelled trials.
* **ICA** — Independent Component Analysis: separates the recording into
  statistically independent sources so artefacts (blinks, heartbeat) can be removed.
"""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Array-level filters (work on plain numpy, last axis = time)
# --------------------------------------------------------------------------- #
def bandpass_filter(
    x: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied along the last axis.

    Uses ``sosfiltfilt`` (second-order sections, forward+backward) so there is
    no phase distortion — important when you care about *when* something happened.

    Parameters
    ----------
    x : np.ndarray
        Signal, time on the last axis.
    sfreq : float
        Sampling rate in Hz.
    low, high : float
        Pass-band edges in Hz.
    order : int
        Filter order (steepness). Higher = sharper but less stable.
    """
    nyq = 0.5 * sfreq
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def notch_filter(x: np.ndarray, sfreq: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove a single mains frequency (50 Hz in EU, 60 Hz in US) along last axis.

    Parameters
    ----------
    freq : float
        Mains frequency to remove. Use 50.0 in Europe/Asia, 60.0 in the Americas.
    q : float
        Quality factor: higher = narrower notch (removes less neighbouring signal).
    """
    b, a = iirnotch(freq, q, sfreq)
    return filtfilt(b, a, x, axis=-1)


# --------------------------------------------------------------------------- #
# MNE-level helpers (operate on Raw / Epochs)
# --------------------------------------------------------------------------- #
def basic_clean_raw(
    raw: "mne.io.Raw",
    *,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch: float | None = 50.0,
    reference: str = "average",
) -> "mne.io.Raw":
    """Apply a standard, conservative cleaning chain to a copy of ``raw``.

    Steps: band-pass (``l_freq``–``h_freq``), optional mains notch, then
    re-reference. Returns a **new** Raw (the input is not modified).

    Notes
    -----
    These steps are stateless w.r.t. labels, so running them on the whole
    recording before splitting does **not** leak. Anything that *learns* from
    data (ICA, scalers, CSP) must go inside a train-only pipeline instead.
    """
    raw = raw.copy().load_data()
    raw.filter(l_freq, h_freq, verbose="ERROR")
    if notch is not None:
        raw.notch_filter(freqs=[notch], verbose="ERROR")
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif reference:
        raw.set_eeg_reference([reference], verbose="ERROR")
    return raw


def make_epochs(
    raw: "mne.io.Raw",
    events: np.ndarray,
    event_id: dict,
    *,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: tuple | None = (None, 0),
) -> "mne.Epochs":
    """Cut ``raw`` into labelled epochs with baseline correction.

    ``baseline=(None, 0)`` subtracts the mean of each channel over the
    pre-stimulus window, so every epoch starts from a common zero — this removes
    slow drifts that would otherwise dominate.
    """
    return mne.Epochs(
        raw, events, event_id, tmin=tmin, tmax=tmax,
        baseline=baseline, preload=True, verbose="ERROR",
    )


def fit_ica(
    raw: "mne.io.Raw",
    *,
    n_components: int = 15,
    random_state: int = 0,
) -> "mne.preprocessing.ICA":
    """Fit ICA on (a high-pass-filtered copy of) ``raw`` for artefact removal.

    ICA expects data high-passed at ~1 Hz to behave well, so we filter a copy
    before fitting. The returned ICA can then be applied to the original.
    """
    from mne.preprocessing import ICA

    ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto")
    raw_hp = raw.copy().filter(1.0, None, verbose="ERROR")
    ica.fit(raw_hp, verbose="ERROR")
    return ica


def detect_eog_components(
    ica: "mne.preprocessing.ICA",
    raw: "mne.io.Raw",
    ch_name: str | None = None,
) -> list[int]:
    """Return indices of ICA components correlated with eye movement (EOG).

    If no dedicated EOG channel exists, MNE can use a frontal EEG channel as a
    proxy via ``ch_name``.
    """
    try:
        idx, _ = ica.find_bads_eog(raw, ch_name=ch_name, verbose="ERROR")
    except Exception:
        idx = []
    return list(idx)


# --------------------------------------------------------------------------- #
# A tiny, transparent ASR-style amplitude cleaner (educational, not production)
# --------------------------------------------------------------------------- #
def clip_extreme_amplitudes(
    x: np.ndarray,
    *,
    z_thresh: float = 5.0,
) -> np.ndarray:
    """Clip samples whose amplitude exceeds ``z_thresh`` robust std-devs.

    A simplified stand-in for Artifact Subspace Reconstruction (ASR): real ASR
    reconstructs corrupted segments from a clean subspace; here we just clip
    extreme excursions (e.g. movement spikes) using a robust (MAD-based) scale.
    Returned array is a filtered copy; shape is unchanged.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    robust_std = 1.4826 * mad + 1e-12
    limit = z_thresh * robust_std
    return np.clip(x, med - limit, med + limit)
