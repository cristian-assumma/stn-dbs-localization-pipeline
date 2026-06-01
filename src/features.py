import numpy as np
from src.config import ProcessingConfig, FeatureConfig

class FeatureExtractor:
    """
    Extracts strictly the required physiological features from MER signal segments.
    Optimized for low-latency inference by avoiding computation of unused metrics.
    """
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        # Fetch the selected features from the central configuration
        self.selected_features = FeatureConfig.SELECTED_FEATURES

    def extract(self, segment: np.ndarray) -> dict:
        """
        Routes the calculation to only compute the features defined in config.py.
        Returns a dictionary of feature names and their scalar values.
        """
        features = {}
        
        if 'RMS' in self.selected_features:
            features['RMS'] = self._compute_rms(segment)
            
        if 'CurveLength' in self.selected_features:
            features['CurveLength'] = self._compute_curve_length(segment)
            
        if 'NumSpikes' in self.selected_features:
            features['NumSpikes'] = self._compute_num_spikes(segment)
            
        return features

    def _compute_rms(self, segment: np.ndarray) -> float:
        """
        Root Mean Square: Measures the overall energy of the neural signal.
        """
        # np.mean and **2 are vectorized O(N) operations in C under the hood
        return float(np.sqrt(np.mean(segment**2)))

    def _compute_curve_length(self, segment: np.ndarray) -> float:
        """
        Curve Length: Sum of absolute differences between consecutive points.
        Captures high-frequency content and neural firing density.
        """
        return float(np.sum(np.abs(np.diff(segment))))

    def _compute_num_spikes(self, segment: np.ndarray) -> int:
        """
        Counts distinct neural spikes crossing a dynamic standard-deviation threshold.
        """
        threshold = np.std(segment) * ProcessingConfig.SPIKE_THRESHOLD_MULTIPLIER
        
        # Identify indices where the signal exceeds the positive threshold
        spike_indices = np.where(segment > threshold)[0]
        
        if len(spike_indices) == 0:
            return 0
            
        # Crucial: A single physiological spike spans multiple sampling points.
        # We must count distinct crossing events, not just every point above the line.
        # If the gap between two indices is > 1, it's a new discrete spike event.
        discrete_spikes = np.sum(np.diff(spike_indices) > 1) + 1
        
        return int(discrete_spikes)
