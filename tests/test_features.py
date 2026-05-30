"""Tests for feature extraction: shapes, determinism, and that band power
actually tracks the band it claims to measure."""
import numpy as np
import pytest

from neuro101 import features as ft


@pytest.fixture
def epochs():
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 5, 256)), 128.0  # 8 trials, 5 ch, 2 s @128 Hz


def test_time_domain_shape(epochs):
    X, _ = epochs
    feats = ft.time_domain_features(X)
    assert feats.shape == (8, 5 * 3)
    assert np.isfinite(feats).all()


def test_bandpower_shape_and_finiteness(epochs):
    X, sf = epochs
    feats = ft.bandpower(X, sf)
    assert feats.shape == (8, 5 * len(ft.BANDS))
    assert np.isfinite(feats).all()


def test_bandpower_detects_injected_oscillation():
    """A 10 Hz sine should put most relative power in the alpha (8-13 Hz) band."""
    sf = 128.0
    t = np.arange(int(2 * sf)) / sf
    sine = np.sin(2 * np.pi * 10 * t)
    X = np.tile(sine, (3, 1, 1))  # 3 trials, 1 channel
    feats = ft.bandpower(X, sf, relative=True)  # (3, n_bands), channel order
    band_names = list(ft.BANDS)
    alpha_idx = band_names.index("alpha")
    # alpha power should dominate the other bands.
    assert np.argmax(feats[0]) == alpha_idx


def test_coherence_and_plv_shapes(epochs):
    X, sf = epochs
    n_pairs = 5 * 4 // 2
    assert ft.coherence_features(X, sf).shape == (8, n_pairs)
    assert ft.plv_features(X).shape == (8, n_pairs)


def test_plv_bounded_zero_one(epochs):
    X, _ = epochs
    plv = ft.plv_features(X)
    assert (plv >= 0).all() and (plv <= 1.0 + 1e-9).all()


def test_csp_and_riemann_are_estimators():
    csp = ft.make_csp(2)
    assert hasattr(csp, "fit") and hasattr(csp, "transform")
    steps = ft.make_riemann_pipeline_steps()
    assert [n for n, _ in steps] == ["cov", "tangent"]
    for _, est in steps:
        assert hasattr(est, "fit")


def test_features_are_deterministic(epochs):
    X, sf = epochs
    assert np.allclose(ft.bandpower(X, sf), ft.bandpower(X, sf))
    assert np.allclose(ft.plv_features(X), ft.plv_features(X))
