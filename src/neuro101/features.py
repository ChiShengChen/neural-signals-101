"""Feature extraction: time-domain, frequency-domain, connectivity, CSP, Riemannian.

All feature functions take epochs as ``X`` of shape ``(n_trials, n_channels,
n_times)`` and return a 2-D matrix ``(n_trials, n_features)`` ready for sklearn,
**or** an sklearn-compatible transformer (so the fit happens inside a leakage-safe
pipeline). The split between the two is deliberate:

* Stateless features (band power, variance) → plain functions.
* Features that *learn* from data (CSP spatial filters, covariance whitening) →
  transformers, because they must be fit on the training fold only.

Terms:

* **Band power** — how much signal energy sits in a frequency band (e.g. alpha
  8–12 Hz). Computed here from Welch's power spectral density.
* **Coherence / PLV** — measures of how synchronised two channels are.
* **CSP** — Common Spatial Patterns: learns channel mixtures that maximise the
  variance difference between two classes (a classic motor-imagery feature).
* **Covariance / Riemannian** — represent each trial by its channel covariance
  matrix and compare these matrices on their curved (Riemannian) geometry; a
  strong, now-standard BCI baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence, hilbert, welch

# Standard EEG frequency bands (Hz). Defined once, reused everywhere.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# --------------------------------------------------------------------------- #
# Time-domain features
# --------------------------------------------------------------------------- #
def time_domain_features(X: np.ndarray) -> np.ndarray:
    """Per-channel simple statistics, concatenated across channels.

    Features per channel: variance, mean absolute value, and a Hjorth mobility
    proxy (std of the derivative / std of the signal). Returns
    ``(n_trials, n_channels * 3)``.
    """
    X = np.asarray(X, dtype=float)
    var = X.var(axis=-1)
    mav = np.abs(X).mean(axis=-1)
    dx = np.diff(X, axis=-1)
    mobility = dx.std(axis=-1) / (X.std(axis=-1) + 1e-12)
    return np.concatenate([var, mav, mobility], axis=1)


# --------------------------------------------------------------------------- #
# Frequency-domain features
# --------------------------------------------------------------------------- #
def bandpower(
    X: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    *,
    relative: bool = False,
) -> np.ndarray:
    """Average band power per channel via Welch's PSD.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times)
    sfreq : float
        Sampling rate (Hz).
    bands : dict, optional
        ``{name: (low, high)}``. Defaults to :data:`BANDS`.
    relative : bool
        If True, divide each band by the total power so features sum to ~1 per
        channel (removes overall-amplitude differences between trials/subjects).

    Returns
    -------
    np.ndarray (n_trials, n_channels * n_bands)
        Band power, log-transformed (log power is closer to normally distributed
        and is what most EEG classifiers use).
    """
    X = np.asarray(X, dtype=float)
    bands = bands or BANDS
    nperseg = min(X.shape[-1], int(sfreq))  # ~1 s windows when possible
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd: (n_trials, n_channels, n_freqs)
    total = np.trapz(psd, freqs, axis=-1) + 1e-12
    feats = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs < high)
        bp = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        if relative:
            bp = bp / total
        feats.append(bp)
    out = np.concatenate(feats, axis=1)  # (n_trials, n_channels*n_bands)
    return np.log(out + 1e-12)


# --------------------------------------------------------------------------- #
# Connectivity features
# --------------------------------------------------------------------------- #
def coherence_features(X: np.ndarray, sfreq: float, fmin: float = 8.0, fmax: float = 30.0) -> np.ndarray:
    """Mean magnitude-squared coherence in [fmin, fmax] for every channel pair.

    Returns ``(n_trials, n_pairs)`` where ``n_pairs = C*(C-1)/2``. Coherence near
    1 means two channels rise and fall together at those frequencies.
    """
    X = np.asarray(X, dtype=float)
    n_trials, n_ch, n_times = X.shape
    nperseg = min(n_times, int(sfreq))
    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    out = np.zeros((n_trials, len(pairs)))
    for t in range(n_trials):
        for p, (i, j) in enumerate(pairs):
            f, cxy = coherence(X[t, i], X[t, j], fs=sfreq, nperseg=nperseg)
            band = (f >= fmin) & (f <= fmax)
            out[t, p] = cxy[band].mean() if band.any() else 0.0
    return out


def plv_features(X: np.ndarray) -> np.ndarray:
    """Phase-Locking Value for every channel pair (broadband, via Hilbert phase).

    PLV measures phase synchrony: 1 = phases are perfectly locked across the
    trial, 0 = unrelated. Band-pass the signal first if you want band-specific
    PLV. Returns ``(n_trials, n_pairs)``.
    """
    X = np.asarray(X, dtype=float)
    n_trials, n_ch, _ = X.shape
    phase = np.angle(hilbert(X, axis=-1))
    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    out = np.zeros((n_trials, len(pairs)))
    for p, (i, j) in enumerate(pairs):
        dphi = phase[:, i, :] - phase[:, j, :]
        out[:, p] = np.abs(np.exp(1j * dphi).mean(axis=-1))
    return out


# --------------------------------------------------------------------------- #
# Learned transformers (fit on TRAIN only — use inside a pipeline)
# --------------------------------------------------------------------------- #
def make_csp(n_components: int = 4, *, reg: float | None = None):
    """Return an MNE CSP transformer (Common Spatial Patterns) for 2-class MI.

    CSP **learns** spatial filters from labelled training data, so it must live
    inside a leakage-safe pipeline. Outputs log-variance features per component.

    Examples
    --------
    >>> csp = make_csp(4)
    >>> hasattr(csp, "fit") and hasattr(csp, "transform")
    True
    """
    from mne.decoding import CSP

    return CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)


def make_riemann_pipeline_steps(*, estimator: str = "oas"):
    """Return ``(Covariances, TangentSpace)`` steps for a Riemannian baseline.

    Each trial becomes a channel covariance matrix (``Covariances``); the matrices
    are then projected to a flat tangent space (``TangentSpace``) so an ordinary
    linear classifier can be used. Both steps are fit on training data only when
    placed in a pipeline. Returns a list of ``(name, transformer)`` tuples.

    Examples
    --------
    >>> steps = make_riemann_pipeline_steps()
    >>> [name for name, _ in steps]
    ['cov', 'tangent']
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    return [
        ("cov", Covariances(estimator=estimator)),
        ("tangent", TangentSpace(metric="riemann")),
    ]
