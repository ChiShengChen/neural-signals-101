"""Plotting helpers shared across notebooks.

Kept deliberately simple (matplotlib only) and dependency-light. Every function
returns the matplotlib ``Axes`` (or ``Figure``) so notebooks can tweak it. The
star of the module is :func:`plot_wrong_vs_right`, which draws the inflated-vs-
honest score contrast used by the headline figure and Chapter 09.
"""
from __future__ import annotations

# Use a non-interactive backend when running headless (CI / notebook execution).
import os as _os
from typing import Sequence

import matplotlib
import numpy as np

if _os.environ.get("NEURO101_SMOKE") == "1" or _os.environ.get("MPLBACKEND") is None:
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass

import matplotlib.pyplot as plt  # noqa: E402

# Consistent, colour-blind-friendly colours for the WRONG/RIGHT contrast.
WRONG_COLOR = "#d1495b"  # warm red  = inflated / wrong
RIGHT_COLOR = "#2e8b57"  # green      = honest / right


def plot_signal(
    x: np.ndarray,
    sfreq: float,
    *,
    ax=None,
    title: str = "Signal",
    units: str = "amplitude",
    max_seconds: float | None = None,
):
    """Plot a single-channel 1-D signal against time in seconds."""
    x = np.asarray(x).ravel()
    if max_seconds is not None:
        x = x[: int(max_seconds * sfreq)]
    t = np.arange(x.size) / sfreq
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, x, lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(units)
    ax.set_title(title)
    return ax


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    ax=None,
    label: str | None = None,
    title: str = "Power spectral density",
    logy: bool = True,
):
    """Plot a power spectral density curve (optionally log-scaled y-axis)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(freqs, psd, label=label, lw=1.2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    if label:
        ax.legend()
    return ax


def plot_spectrogram(
    x: np.ndarray,
    sfreq: float,
    *,
    ax=None,
    title: str = "Spectrogram",
    nperseg: int | None = None,
):
    """Plot a time–frequency spectrogram of a 1-D signal (STFT magnitude in dB)."""
    from scipy.signal import spectrogram

    x = np.asarray(x).ravel()
    nperseg = nperseg or min(256, x.size)
    f, t, Sxx = spectrogram(x, fs=sfreq, nperseg=nperseg)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.2))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-20), shading="gouraud")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


def plot_confusion(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    ax=None,
    title: str = "Confusion matrix",
    normalize: bool = False,
):
    """Plot a confusion matrix with cell annotations.

    A confusion matrix shows, for each true class (rows), how many examples were
    predicted as each class (columns). The diagonal is correct predictions.
    """
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    fmt = ".2f" if normalize else ".0f"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    return ax


def plot_wrong_vs_right(
    wrong_score: float,
    right_score: float,
    *,
    wrong_err: float | None = None,
    right_err: float | None = None,
    chance: float | None = None,
    metric: str = "Accuracy",
    wrong_label: str = "WRONG method\n(inflated)",
    right_label: str = "RIGHT method\n(honest)",
    title: str = "Why evaluation method matters",
    ax=None,
):
    """Draw the signature inflated-vs-honest two-bar contrast.

    This is the visual hook of the whole tutorial: one red bar for the
    deceptively high score from a leaky method, one green bar for the honest
    score from a correct method, with an arrow showing the drop. Optionally draws
    a dashed "chance level" line so the honest number has context.

    Returns the matplotlib ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    xs = [0, 1]
    heights = [wrong_score, right_score]
    errs = [wrong_err or 0.0, right_err or 0.0]
    bars = ax.bar(
        xs, heights, width=0.6,
        color=[WRONG_COLOR, RIGHT_COLOR],
        yerr=errs if (wrong_err or right_err) else None,
        capsize=6,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([wrong_label, right_label])
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)

    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontweight="bold")

    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", lw=1)
        ax.text(1.45, chance, f"chance ≈ {chance:.2f}", color="gray",
                va="center", ha="left", fontsize=8)

    # Drop annotation between the two bars.
    drop = wrong_score - right_score
    if drop > 0.01:
        ax.annotate(
            f"−{drop:.2f}\nfake gain",
            xy=(1, right_score), xytext=(0.5, (wrong_score + right_score) / 2),
            ha="center", va="center", fontsize=9, color="black",
            arrowprops=dict(arrowstyle="->", color="black"),
        )
    ax.figure.tight_layout()
    return ax
