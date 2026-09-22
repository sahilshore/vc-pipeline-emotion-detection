# model_building.py
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from typing import Tuple, Dict, Any
from sklearn.ensemble import GradientBoostingClassifier

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

file_handler = logging.FileHandler('Error_files/model_building.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_params(params_path: str) -> Dict[str, Any]:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        mb_params = params['model_building']
        logger.debug(f"Model parameters retrieved successfully: {mb_params}")
        return mb_params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing expected parameter in YAML: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise

def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        train_data = pd.read_csv(data_path)
        
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        
        logger.debug("Training features and labels extracted successfully.")
        return X_train, y_train
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> GradientBoostingClassifier:
    try:
        logger.debug("Initializing GradientBoostingClassifier...")
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"], 
            learning_rate=params["learning_rate"]
        )
        
        logger.debug("Training the model... (this may take a moment)")
        clf.fit(X_train, y_train)
        
        logger.debug("Model training completed successfully.")
        return clf
    except ValueError as e:
        logger.error(f"Value error during model training (e.g., mismatch in data shapes): {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise

def save_model(model: GradientBoostingClassifier, model_path: str) -> None:
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully at {model_path}.")
    except IOError as e:
        logger.error(f"IOError while saving the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving the model: {e}")
        raise

def main() -> None:
    try:
        # Load params
        params = load_params(params_path="params.yaml")
        
        # Load training data
        X_train, y_train = load_data(data_path="./data/features/train_bow.csv")
        
        # Train model
        clf = train_model(X_train, y_train, params)
        
        # Save model
        save_model(clf, model_path="models/model.pkl")
        
        logger.info("Model building pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete model building process: {e}")

if __name__ == '__main__':
    main()