import argparse
import logging
import numpy as np

from src.config import SignalConfig, ProcessingConfig, FeatureConfig
from src.data_loader import MERDataLoader
from src.preprocessor import SignalPreprocessor
from src.features import FeatureExtractor
from src.model import STNClassifier

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def extract_dataset(subjects_list: list, loader: MERDataLoader, preprocessor: SignalPreprocessor,
                    extractor: FeatureExtractor) -> tuple:
    """
    Simulates the data ingestion and feature extraction pipeline for a list of subjects.
    Chunks the continuous MER signals into non-overlapping windows (e.g., 2 seconds)
    to match the real-time inference constraints.
    """
    X, y = [], []
    window_samples = int(ProcessingConfig.WINDOW_LENGTH_SEC * SignalConfig.SAMPLING_RATE)

    for subj in subjects_list:
        logger.info(f"Extracting signals and features for {subj}...")
        data = loader.load_subject(subj)

        for hemi in ['RightHemisphere', 'LeftHemisphere']:
            # Safety check in case a patient only had unilateral surgery
            if hemi not in data:
                continue

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


def run_loso_validation(subjects: list, max_depth: int):
    """
    Executes Leave-One-Subject-Out (LOSO) cross-validation.
    Trains on N-1 subjects, tests on the 1 remaining subject.
    """
    logger.info("Initializing STN-DBS Localization Pipeline...")
    logger.info(f"Features in use: {FeatureConfig.SELECTED_FEATURES}")
    logger.info(f"Active Subjects: {subjects}")
    logger.info(f"Model Configuration: Decision Tree (max_depth={max_depth})")

    try:
        loader = MERDataLoader()
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader. Ensure data/ folder is populated. Error: {e}")
        return

    preprocessor = SignalPreprocessor()
    extractor = FeatureExtractor(sampling_rate=SignalConfig.SAMPLING_RATE)

    # Store results for aggregation
    loso_results = []

    for test_subj in subjects:
        logger.info(f"--- LOSO Fold: Testing on [{test_subj}] ---")

        # Define train and test sets
        train_subjects = [s for s in subjects if s != test_subj]

        X_train, y_train = extract_dataset(train_subjects, loader, preprocessor, extractor)
        X_test, y_test = extract_dataset([test_subj], loader, preprocessor, extractor)

        # Initialize and train model
        classifier = STNClassifier(max_depth=max_depth)
        classifier.train(X_train, y_train)

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        loso_results.append(metrics)

        logger.info(
            f"Fold Results [{test_subj}] -> Accuracy: {metrics['Accuracy']:.2%} | Sensitivity: {metrics['Sensitivity_Recall']:.2%} | Specificity: {metrics['Specificity']:.2%}")
        logger.info(f"Confusion Matrix [{test_subj}]:\n{metrics['ConfusionMatrix']}")

    # Aggregate global metrics
    logger.info("=== GLOBAL LOSO RESULTS ===")
    avg_acc = np.mean([res['Accuracy'] for res in loso_results])
    avg_sens = np.mean([res['Sensitivity_Recall'] for res in loso_results])
    avg_spec = np.mean([res['Specificity'] for res in loso_results])

    logger.info(f"Average Accuracy:    {avg_acc:.2%}")
    logger.info(f"Average Sensitivity: {avg_sens:.2%}")
    logger.info(f"Average Specificity: {avg_spec:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Intraoperative MER Analysis for STN-DBS")

    # Accept a dynamic list of subjects
    parser.add_argument("--subjects", nargs='+', default=['Subj20', 'Subj33', 'Subj38'],
                        help="List of subject IDs to process (e.g., --subjects Subj20 Subj33 Subj39)")

    # Externalized hyperparameters
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum tree depth to prevent overfitting on small N.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Prevent logic errors before starting heavy processing
    if len(args.subjects) < 2:
        logger.error("Leave-One-Subject-Out validation requires at least 2 subjects. Aborting.")
        exit(1)

    run_loso_validation(subjects=args.subjects, max_depth=args.max_depth)