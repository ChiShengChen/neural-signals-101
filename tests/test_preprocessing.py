"""Tests for preprocessing: filters actually attenuate what they target."""
import numpy as np

from neuro101 import preprocessing as pp


def _sine(freq, sf, dur=2.0):
    t = np.arange(int(dur * sf)) / sf
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_keeps_passband_rejects_stopband():
    sf = 256.0
    in_band = _sine(10, sf)    # inside 8-30 Hz
    out_band = _sine(60, sf)   # outside
    mixed = (in_band + out_band)[None, None, :]  # (1,1,T)
    filtered = pp.bandpass_filter(mixed, sf, 8, 30)[0, 0]
    # Power of the 60 Hz component should be strongly reduced; 10 Hz preserved.
    assert filtered.std() < (in_band + out_band).std()
    # Correlate with pure 10 Hz: should stay high.
    corr = np.corrcoef(filtered, in_band)[0, 1]
    assert corr > 0.9


def test_notch_attenuates_mains():
    sf = 256.0
    mains = _sine(50, sf)
    signal = _sine(10, sf)
    mixed = (mains + signal)[None, None, :]
    out = pp.notch_filter(mixed, sf, 50.0)[0, 0]
    # The 50 Hz energy should drop a lot; project onto the mains sine.
    before = np.abs((mixed[0, 0] * mains).mean())
    after = np.abs((out * mains).mean())
    assert after < 0.5 * before


def test_bandpass_preserves_shape():
    sf = 128.0
    X = np.random.default_rng(0).normal(size=(4, 6, 256))
    assert pp.bandpass_filter(X, sf, 4, 40).shape == X.shape


def test_clip_extreme_amplitudes_limits_spikes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 1, 1000))
    x[0, 0, 500] = 50.0  # a huge spike
    cleaned = pp.clip_extreme_amplitudes(x, z_thresh=5.0)
    assert cleaned.max() < 50.0
    assert cleaned.shape == x.shape
