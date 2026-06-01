import os
import numpy as np
from scipy.io import loadmat
from src.config import PathConfig, SignalConfig

class MERDataLoader:
    """
    Handles ingestion and decoding of .mat files.
    Decouples the reading logic from MATLAB’s proprietary format.
    """
    
    def __init__(self, data_dir: str = PathConfig.DATA_DIR):
        self.data_dir = data_dir

    def load_subject(self, subject_id: str) -> dict:
        """
        Loads a patient's data and returns a clean dictionary
        with numpy tensors for the right and left hemisphere.
        """
        file_path = os.path.join(self.data_dir, f"Data_{subject_id}.mat")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}. Check the data/ directory.")

        # loadmat loads the file. squeeze_me=True removes unnecessary singleton dimensions
        # that MATLAB inserts everywhere by default.
        raw_mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        # Safe extraction of the 'Data' block
        try:
            data_struct = raw_mat['Data']
        except KeyError:
            raise KeyError(f"The file {file_path} does not contain the expected 'Data' structure.")

        # Consistency check on the sampling rate
        mat_fs = getattr(data_struct, 'Sampl_Freq', None)
        if mat_fs and mat_fs != SignalConfig.SAMPLING_RATE:
            # Execution does not stop, but in production we would raise a strong warning
            print(f"[WARNING] Sampling rate in file ({mat_fs} Hz) differs from config ({SignalConfig.SAMPLING_RATE} Hz).")

        return {
            'subject_id': subject_id,
            'RightHemisphere': self._extract_hemisphere(getattr(data_struct, 'RightHemisphere', None)),
            'LeftHemisphere': self._extract_hemisphere(getattr(data_struct, 'LeftHemisphere', None))
        }

    def _extract_hemisphere(self, hemi_struct) -> dict:
        """
        Extracts signals and targets (ground truth) for a single hemisphere.
        """
        if hemi_struct is None:
            return {'signals': {}, 'targets': np.array([])}

        signals_dict = {}
        # Extract the MERs object
        mers = getattr(hemi_struct, 'MERs', None)
        
        if mers:
            # mers._fieldnames contains the depth names (e.g., 'm6', 'm1', 'zero')
            for depth_name in mers._fieldnames:
                # To avoid loading PSD or Freq arrays if saved together,
                # we only pick fields that do not end with _PSD or _Freq
                if not depth_name.endswith('_PSD') and not depth_name.endswith('_Freq'):
                    signal_array = getattr(mers, depth_name)
                    # Force float32 to optimize RAM (MATLAB often uses float64 unnecessarily)
                    signals_dict[depth_name] = np.array(signal_array, dtype=np.float32)

        # Target is typically a 1D array
        targets = np.array(getattr(hemi_struct, 'Target', []), dtype=np.int8)

        return {
            'signals': signals_dict,
            'targets': targets
        }
