import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class STNClassifier:
    """
    Wrapper for the Decision Tree Classifier optimized for intraoperative STN localization.
    Provides rigorous medical-grade evaluation metrics (Sensitivity, Specificity) 
    instead of naive global accuracy.
    """
    
    def __init__(self, max_depth: int = None, random_state: int = 42):
        """
        Initializes the CART Decision Tree model.
        Fixed random_state ensures deterministic behavior across pipeline runs.
        """
        self.model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the internal decision tree model.
        Expects raw float32 features without arbitrary mathematical rounding.
        """
        if X_train.shape[0] == 0:
            raise ValueError("Training dataset is empty.")
            
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates interval-level predictions (0: Non-STN, 1: STN).
        """
        if not self.is_trained:
            raise RuntimeError("Cannot predict. Model must be trained first.")
            
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on test data and computes clinical-grade performance metrics.
        Calculates Specificity and Sensitivity to explicitly account for class imbalance.
        """
        predictions = self.predict(X_test)
        
        # Calculate standard ML metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        sens = recall_score(y_test, predictions, zero_division=0)  # Sensitivity / Recall
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Compute confusion matrix to isolate true negatives and false positives
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        
        # Specificity (True Negative Rate): Critical for ensuring we don't misclassify healthy tissue as STN
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Sensitivity_Recall': float(sens),
            'Specificity': float(spec),
            'F1_Score': float(f1),
            'ConfusionMatrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
        }

    def get_feature_importance(self, feature_names: list) -> dict:
        """
        Extracts Gini importance scores for each feature.
        Allows validation of the minimal feature set requirement.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Importance scores unavailable.")
            
        importances = self.model.feature_importances_
        
        # Map importance scores back to their respective configuration names
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
