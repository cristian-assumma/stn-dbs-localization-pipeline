"""
Global configuration parameters for the STN-DBS localization pipeline.
Centralizing these variables prevents hardcoding and allows rapid tuning.
"""

class SignalConfig:
    SAMPLING_RATE = 20000  # Hz
    BANDPASS_LOW = 200     # Hz
    BANDPASS_HIGH = 5000   # Hz
    FILTER_ORDER = 4

class ProcessingConfig:
    WINDOW_LENGTH_SEC = 2
    SPIKE_THRESHOLD_MULTIPLIER = 3  # mean + (multiplier * std)
    LONG_ISI_THRESHOLD = 0.05       # 50 ms expressed in seconds

class FeatureConfig:
    # Minimal feature set validated for real-time deployment
    SELECTED_FEATURES = ['RMS', 'CurveLength', 'NumSpikes']

class PathConfig:
    DATA_DIR = "data/"
