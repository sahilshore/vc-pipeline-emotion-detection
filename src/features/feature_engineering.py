# feature_engineering.py
import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple 
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

file_handler = logging.FileHandler('Error_files/feature_engineering.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['feature_engineering']['max_features']
        logger.debug(f'max_features retrieved successfully: {max_features}')
        return max_features
    except FileNotFoundError:
        logger.error(f'File not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML parsing error: {e}')
        raise
    except KeyError as e:
        logger.error(f'Missing expected parameter in YAML: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading parameters: {e}')
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('Processed data loaded successfully.')
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.debug('Applying Bag of Words (TfidfVectorizer)...')
        
        # Handle NaN values to prevent errors in vectorization
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values


        # Initialize and apply TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse matrices to DataFrames
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        logger.debug('Bag of Words applied successfully.')
        return train_df, test_df
    except KeyError as e:
        logger.error(f"Missing column in dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Error applying Bag of Words: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        features_path = os.path.join(data_path, "features")
        os.makedirs(features_path, exist_ok=True)

        train_df.to_csv(os.path.join(features_path, "train_tfidf.csv"), index=False)
        test_df.to_csv(os.path.join(features_path, "test_tfidf.csv"), index=False)
        logger.debug("Features saved successfully.")
    except Exception as e:
        logger.error(f"Unexpected error saving features data: {e}")
        raise

def main() -> None:
    try:
        # Load params
        max_features = load_params(params_path='params.yaml')
        
        # Load processed data
        train_data, test_data = load_data(
            train_path="./data/processed/train_processed.csv", 
            test_path="./data/processed/test_processed.csv"
        )
        
        # Feature Engineering
        train_df, test_df = apply_bow(train_data, test_data, max_features)
        
        # Save features
        save_data(train_df, test_df, data_path="data")
        
        logger.info("Feature engineering completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete feature engineering: {e}")

if __name__ == '__main__':
    main()