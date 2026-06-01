import numpy as np
from src.config import SignalConfig, ProcessingConfig, FeatureConfig
from src.data_loader import MERDataLoader
from src.preprocessor import SignalPreprocessor
from src.features import FeatureExtractor
from src.model import STNClassifier

# Hardcoded for the current dataset, but easily externalized
SUBJECTS = ['Subj20', 'Subj33', 'Subj38']

def extract_dataset(subjects_list: list, loader: MERDataLoader, preprocessor: SignalPreprocessor, extractor: FeatureExtractor) -> tuple:
    """
    Simulates the data ingestion and feature extraction pipeline for a list of subjects.
    Chunks the continuous MER signals into non-overlapping windows (e.g., 2 seconds)
    to match the real-time inference constraints.
    """
    X, y = [], []
    window_samples = int(ProcessingConfig.WINDOW_LENGTH_SEC * SignalConfig.SAMPLING_RATE)
    
    for subj in subjects_list:
        data = loader.load_subject(subj)
        
        for hemi in ['RightHemisphere', 'LeftHemisphere']:
            signals_dict = data[hemi]['signals']
            targets = data[hemi]['targets']
            
            # Loop through depths (e.g., 'm6', 'm1', 'zero')
            for depth_idx, depth_name in enumerate(signals_dict.keys()):
                raw_signal = signals_dict[depth_name]
                target = targets[depth_idx]
                
                # 1. Preprocess the entire depth signal (Artifact rejection + Bandpass)
                clean_signal = preprocessor.process(raw_signal)
                
                # 2. Windowing: Split into 2-second segments
                n_windows = len(clean_signal) // window_samples
                
                for w in range(n_windows):
                    start_idx = w * window_samples
                    end_idx = start_idx + window_samples
                    segment = clean_signal[start_idx:end_idx]
                    
                    # 3. Extract features for this specific window
                    features_dict = extractor.extract(segment)
                    
                    # Convert dict to a flat list respecting the order in config.py
                    feature_vector = [features_dict[feat] for feat in FeatureConfig.SELECTED_FEATURES]
                    
                    X.append(feature_vector)
                    y.append(target)
                    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

def run_loso_validation():
    """
    Executes Leave-One-Subject-Out (LOSO) cross-validation.
    Trains on N-1 subjects, tests on the 1 remaining subject.
    """
    print("Initializing STN-DBS Localization Pipeline...")
    print(f"Features in use: {FeatureConfig.SELECTED_FEATURES}\n")
    
    loader = MERDataLoader()
    preprocessor = SignalPreprocessor()
    extractor = FeatureExtractor(sampling_rate=SignalConfig.SAMPLING_RATE)
    
    # Store results for aggregation
    loso_results = []
    
    for test_subj in SUBJECTS:
        print(f"--- LOSO Fold: Testing on {test_subj} ---")
        
        # Define train and test sets
        train_subjects = [s for s in SUBJECTS if s != test_subj]
        
        print(f"Extracting training data from {train_subjects}...")
        X_train, y_train = extract_dataset(train_subjects, loader, preprocessor, extractor)
        
        print(f"Extracting testing data from [{test_subj}]...")
        X_test, y_test = extract_dataset([test_subj], loader, preprocessor, extractor)
        
        # Initialize and train model
        classifier = STNClassifier(max_depth=5) # Constrained depth to avoid overfitting on small N
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        loso_results.append(metrics)
        
        print(f"Accuracy:    {metrics['Accuracy']:.2%}")
        print(f"Sensitivity: {metrics['Sensitivity_Recall']:.2%}")
        print(f"Specificity: {metrics['Specificity']:.2%}")
        print(f"Conf Matrix: {metrics['ConfusionMatrix']}\n")

    # Aggregate global metrics
    print("=== GLOBAL LOSO RESULTS ===")
    avg_acc = np.mean([res['Accuracy'] for res in loso_results])
    avg_sens = np.mean([res['Sensitivity_Recall'] for res in loso_results])
    avg_spec = np.mean([res['Specificity'] for res in loso_results])
    
    print(f"Average Accuracy:    {avg_acc:.2%}")
    print(f"Average Sensitivity: {avg_sens:.2%}")
    print(f"Average Specificity: {avg_spec:.2%}")

if __name__ == "__main__":
    run_loso_validation()
