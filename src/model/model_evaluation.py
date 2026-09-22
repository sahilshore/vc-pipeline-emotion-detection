# model_evaluation.py
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

file_handler = logging.FileHandler('Error_files/model_evaluation.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_model(model_path: str) -> Any:
    try:
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
        logger.debug(f"Model loaded successfully from {model_path}.")
        return clf
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading the model: {e}")
        raise

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        test_data = pd.read_csv(data_path)
        
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        
        logger.debug("Testing features and labels extracted successfully.")
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def evaluate_model(clf: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    try:
        logger.debug("Predicting on the test set...")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        logger.debug("Calculating evaluation metrics...")
        # Cast metrics to standard Python float to avoid JSON serialization errors with numpy.float64
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        logger.debug("Evaluation metrics calculated successfully.")
        return metrics
    except ValueError as e:
        logger.error(f"Value error during evaluation (e.g., mismatched shapes or invalid labels): {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved successfully to {metrics_path}.")
    except IOError as e:
        logger.error(f"IOError while saving metrics: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving metrics: {e}")
        raise

def main() -> None:
    try:
        # Load model
        clf = load_model(model_path='models/model.pkl')
        
        # Load testing data
        X_test, y_test = load_data(data_path='./data/features/test_tfidf.csv')
        
        # Evaluate model
        metrics_dict = evaluate_model(clf, X_test, y_test)
        
        # Save metrics
        save_metrics(metrics_dict, metrics_path='reports/metrics.json')
        
        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete model evaluation process: {e}")

if __name__ == '__main__':
    main()