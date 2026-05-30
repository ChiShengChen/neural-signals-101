"""Tests for dataset loaders.

Anything that downloads is marked ``network`` and ``slow`` so the default test
run (and CI's fast lane) can skip it with ``-m 'not network'``. The registry
tests below need no network and always run.
"""
import numpy as np
import pytest

from neuro101 import datasets as ds


# --------------------------------------------------------------------------- #
# Registry / config (no network)
# --------------------------------------------------------------------------- #
def test_registry_has_required_datasets():
    for key in ("mne_sample", "bnci_2a", "physionet_mi", "sleep_edf"):
        assert key in ds.DATASETS
        info = ds.DATASETS[key]
        assert info.approx_download  # documented size, per the spec
        assert info.source.startswith("http") or "MOABB" in info.source


def test_smoke_mode_shrinks_subjects(monkeypatch):
    monkeypatch.setenv("NEURO101_SMOKE", "1")
    n = ds.n_subjects_to_load("bnci_2a", requested=9)
    assert n == ds.DATASETS["bnci_2a"].smoke_subjects  # capped in smoke mode


def test_non_smoke_uses_requested(monkeypatch):
    monkeypatch.delenv("NEURO101_SMOKE", raising=False)
    assert ds.n_subjects_to_load("bnci_2a", requested=2) == 2


def test_cache_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURO101_DATA", str(tmp_path / "cache"))
    p = ds.cache_dir()
    assert p.exists()


def test_describe_returns_text():
    text = ds.describe("bnci_2a")
    assert "motor imagery" in text.lower()


# --------------------------------------------------------------------------- #
# Actual downloads (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.network
@pytest.mark.slow
def test_load_bnci_2a_epochs_shapes():
    from neuro101 import io

    X, y, subj = io.load_bnci_2a_epochs(subjects=[1])
    assert X.ndim == 3 and X.shape[0] == y.shape[0] == subj.shape[0]
    assert set(np.unique(y)) <= {0, 1}
    assert (subj == 1).all()


@pytest.mark.network
@pytest.mark.slow
def test_load_mne_sample_raw():
    from neuro101 import io

    raw = io.load_mne_sample_raw(preload=False)
    assert raw.info["sfreq"] > 0
