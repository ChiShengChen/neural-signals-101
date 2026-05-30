"""Central registry of the public datasets used in this tutorial.

Everything about *where data comes from, how big it is, and how aggressively we
subsample it to stay under the CPU/time budget* lives here, so the notebooks and
``io.py`` never hard-code paths or magic numbers.

No data is bundled with this repo. Every dataset is downloaded on first use and
cached on disk (see :func:`cache_dir`). Approximate download sizes are documented
in :data:`DATASETS` so a learner knows what they are committing to.

Two environment variables control runtime:

* ``NEURO101_DATA``  — override the cache directory (default ``~/neuro101_data``).
* ``NEURO101_SMOKE`` — if set to ``1``, loaders use the *smallest* possible slice
  (few subjects, short crops). CI sets this so notebooks run in seconds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for one public dataset."""

    key: str
    name: str
    modality: str
    approx_download: str
    n_subjects: int
    sfreq_hz: float
    source: str
    description: str
    # Default subsampling to honour the "<5 min on CPU" rule. Notebooks may
    # override, but these are the safe defaults and what CI uses in smoke mode.
    smoke_subjects: int = 1
    default_subjects: int = 3
    notes: str = ""


DATASETS: dict[str, DatasetInfo] = {
    "mne_sample": DatasetInfo(
        key="mne_sample",
        name="MNE sample dataset (audio/visual MEG+EEG)",
        modality="MEG + EEG",
        approx_download="~1.5 GB (downloaded once, cached by MNE)",
        n_subjects=1,
        sfreq_hz=600.614,
        source="https://mne.tools/stable/documentation/datasets.html#sample",
        description=(
            "One subject hearing tones and seeing checkerboards. The canonical "
            "MNE teaching dataset; great for first plots of raw signals and ERPs."
        ),
        smoke_subjects=1,
        default_subjects=1,
        notes="Auto-downloaded by mne.datasets.sample.data_path().",
    ),
    "bnci_2a": DatasetInfo(
        key="bnci_2a",
        name="BCI Competition IV dataset 2a (motor imagery)",
        modality="EEG (22 ch)",
        approx_download="~1.6 GB total (9 subjects, fetched per-subject via MOABB)",
        n_subjects=9,
        sfreq_hz=250.0,
        source="https://www.bbci.de/competition/iv/  /  MOABB BNCI2014_001",
        description=(
            "9 subjects, 4-class motor imagery (left hand, right hand, feet, "
            "tongue), two sessions each. The standard BCI benchmark and the "
            "source of this tutorial's headline figure."
        ),
        smoke_subjects=2,
        default_subjects=3,
        notes="Loaded via MOABB BNCI2014_001 (a.k.a. BCI IV 2a).",
    ),
    "physionet_mi": DatasetInfo(
        key="physionet_mi",
        name="PhysioNet EEG Motor Movement/Imagery",
        modality="EEG (64 ch)",
        approx_download="~40 MB per subject (109 subjects available)",
        n_subjects=109,
        sfreq_hz=160.0,
        source="https://physionet.org/content/eegmmidb/1.0.0/",
        description=(
            "109 subjects performing/imagining hand and foot movements. Easy to "
            "download a handful of subjects; used as the cross-subject backup."
        ),
        smoke_subjects=2,
        default_subjects=4,
        notes="Auto-downloaded by mne.datasets.eegbci.load_data().",
    ),
    "sleep_edf": DatasetInfo(
        key="sleep_edf",
        name="Sleep-EDF (polysomnography, sleep staging)",
        modality="EEG + EOG + EMG",
        approx_download="~8 MB per recording (subset auto-fetched)",
        n_subjects=83,
        sfreq_hz=100.0,
        source="https://physionet.org/content/sleep-edfx/1.0.0/",
        description=(
            "Overnight recordings with expert sleep-stage labels (Wake, N1, N2, "
            "N3, REM). Used for the sleep-staging example and class-imbalance demo."
        ),
        smoke_subjects=1,
        default_subjects=2,
        notes="Auto-downloaded by mne.datasets.sleep_physionet.age.fetch_data().",
    ),
}


def is_smoke() -> bool:
    """True when ``NEURO101_SMOKE=1`` — use the smallest data slice (CI mode)."""
    return os.environ.get("NEURO101_SMOKE", "0") == "1"


def cache_dir() -> Path:
    """Return (and create) the on-disk cache directory for downloads.

    Defaults to ``~/neuro101_data``; override with the ``NEURO101_DATA`` env var.
    """
    root = os.environ.get("NEURO101_DATA")
    path = Path(root).expanduser() if root else Path.home() / "neuro101_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def n_subjects_to_load(key: str, requested: int | None = None) -> int:
    """Resolve how many subjects to load for a dataset, honouring smoke mode.

    Priority: explicit ``requested`` (capped at smoke default in smoke mode) >
    smoke default (smoke mode) > dataset default.
    """
    info = DATASETS[key]
    if is_smoke():
        return min(requested or info.smoke_subjects, info.smoke_subjects)
    return requested or info.default_subjects


def describe(key: str | None = None) -> str:
    """Human-readable summary of one dataset, or all of them."""
    keys = [key] if key else list(DATASETS)
    lines = []
    for k in keys:
        d = DATASETS[k]
        lines += [
            f"{d.name}  [{d.key}]",
            f"  modality : {d.modality}",
            f"  subjects : {d.n_subjects} (sfreq {d.sfreq_hz} Hz)",
            f"  download : {d.approx_download}",
            f"  source   : {d.source}",
            f"  {d.description}",
            "",
        ]
    return "\n".join(lines)
