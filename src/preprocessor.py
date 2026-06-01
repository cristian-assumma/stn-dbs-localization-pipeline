import numpy as np
from scipy.signal import butter, filtfilt
from src.config import SignalConfig, ProcessingConfig

class SignalPreprocessor:
    """
    Handles Digital Signal Processing (DSP) operations for MER signals.
    Responsible for artifact rejection and frequency band isolation.
    Designed for O(N) time complexity to support low-latency requirements.
    """
    
    def __init__(self):
        """
        Initializes the Butterworth filter coefficients. 
        Pre-computing them here avoids redundant calculations during real-time data streaming.
        """
        # Calculate Nyquist frequency
        nyquist = 0.5 * SignalConfig.SAMPLING_RATE
        
        # Normalize frequencies
        low = SignalConfig.BANDPASS_LOW / nyquist
        high = SignalConfig.BANDPASS_HIGH / nyquist
        
        # Generate filter coefficients (b, a) for a digital IIR filter
        self.b, self.a = butter(SignalConfig.FILTER_ORDER, [low, high], btype='band')

    def remove_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """
        Removes excessive high-amplitude artifacts by clipping the signal dynamically.
        Applies symmetric clipping (both positive and negative spikes) based on 
        the configurable standard deviation multiplier.
        """
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # Define symmetric bounds
        upper_bound = mean_val + (ProcessingConfig.SPIKE_THRESHOLD_MULTIPLIER * std_val)
        lower_bound = mean_val - (ProcessingConfig.SPIKE_THRESHOLD_MULTIPLIER * std_val)
        
        # np.clip is highly optimized in C and avoids slow element-wise loops
        clipped_signal = np.clip(signal, lower_bound, upper_bound)
        
        return clipped_signal

    def apply_bandpass(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a zero-phase forward and reverse digital filter (filtfilt).
        This ensures no phase distortion is introduced into the neural signal.
        """
        # filtfilt applies the filter twice (forward and backward) to achieve zero phase shift
        filtered_signal = filtfilt(self.b, self.a, signal)
        
        return filtered_signal
        
    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Executes the full preprocessing pipeline in the correct sequence.
        """
        # 1. Remove artifacts to prevent them from ringing through the filter
        clean_signal = self.remove_artifacts(signal)
        
        # 2. Apply bandpass filter
        ready_signal = self.apply_bandpass(clean_signal)
        
        return ready_signal
