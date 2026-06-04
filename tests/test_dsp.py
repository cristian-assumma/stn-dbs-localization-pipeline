import pytest
import numpy as np
from src.features import FeatureExtractor
from src.config import SignalConfig


@pytest.fixture
def extractor():
    """Initialize the feature extractor with the default sampling rate."""
    return FeatureExtractor(sampling_rate=SignalConfig.SAMPLING_RATE)


def test_rms_calculation(extractor):
    """
    Test the RMS (Root Mean Square) calculation.
    The RMS of a perfect square wave alternating between +3 and -3 is exactly 3.0.
    """
    # Synthetic square wave: constant amplitude in absolute value
    square_wave = np.array([3.0, -3.0] * 500)
    features = extractor.extract(square_wave)

    assert 'RMS' in features, "The 'RMS' key is missing from the feature dictionary"
    assert np.isclose(features['RMS'], 3.0), f"Expected RMS 3.0, got {features['RMS']}"


def test_curve_length_calculation(extractor):
    """
    Test the Curve Length (or Line Length) calculation.
    Measures the sum of absolute differences between consecutive samples.
    """
    # Zig-zag signal: [0, 2, 0, 2, 0]
    # Absolute differences: |2-0| + |0-2| + |2-0| + |0-2| = 2 + 2 + 2 + 2 = 8.0
    zigzag = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
    features = extractor.extract(zigzag)

    assert 'CurveLength' in features, "The 'CurveLength' key is missing"
    assert np.isclose(features['CurveLength'], 8.0), f"Expected Curve Length 8.0, got {features['CurveLength']}"


def test_num_spikes_detection(extractor):
    """
    Test the spike counter.
    Injects low-amplitude background noise and inserts an exact number of high-amplitude spikes.
    """
    # Generate 10,000 samples of very low-amplitude Gaussian noise
    # (mean 0, standard deviation 0.1)
    np.random.seed(42)
    signal = np.random.normal(0, 0.1, 10000)

    # Inject exactly 5 large spikes (100 times the standard deviation)
    # at isolated positions to avoid merging detections
    spike_indices = [1000, 3000, 5000, 7000, 9000]
    for idx in spike_indices:
        signal[idx] = 10.0

    features = extractor.extract(signal)

    assert 'NumSpikes' in features, "The 'NumSpikes' key is missing"
    assert features['NumSpikes'] == 5, f"Expected 5 spikes, detected {features['NumSpikes']}"

