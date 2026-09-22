# data_preprocessing.py
import numpy as np
import pandas as pd
import os
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple

# Download NLTK data safely
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    pass

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

file_handler = logging.FileHandler('Error_files/data_preprocessing.log')
file_handler.setLevel('DEBUG')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# --- Text processing helper functions ---

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text_list = str(text).split()
    text_list = [lemmatizer.lemmatize(y) for y in text_list]
    return " ".join(text_list)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    text_list = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text_list)

def removing_numbers(text: str) -> str:
    return ''.join([i for i in str(text) if not i.isdigit()])

def lower_case(text: str) -> str:
    # Optimized: replaced loop with simple lower()
    return str(text).lower() 

def removing_punctuations(text: str) -> str:
    text = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', str(text))
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', str(text))

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    # Optimized: Avoided slow for-loop and iloc. Used vectorized apply.
    try:
        df_copy = df.copy()
        df_copy['content'] = df_copy['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
        return df_copy
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Normalizing text data...")
        df_copy = df.copy()
        df_copy['content'] = df_copy['content'].apply(lower_case)
        df_copy['content'] = df_copy['content'].apply(remove_stop_words)
        df_copy['content'] = df_copy['content'].apply(removing_numbers)
        df_copy['content'] = df_copy['content'].apply(removing_punctuations)
        df_copy['content'] = df_copy['content'].apply(removing_urls)
        df_copy['content'] = df_copy['content'].apply(lemmatization)
        return df_copy
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

# --- Core Pipeline Functions ---

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug("Raw data loaded successfully.")
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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        processed_path = os.path.join(data_path, "processed")
        os.makedirs(processed_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_path, "test_processed.csv"), index=False)
        logger.debug("Processed data saved successfully.")
    except Exception as e:
        logger.error(f"Unexpected error saving data: {e}")
        raise

def main() -> None:
    try:
        # Load Data
        train_data, test_data = load_data("data/raw/train.csv", "data/raw/test.csv")
        
        # Normalize Data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        # Save Data
        save_data(train_processed_data, test_processed_data, data_path="data")
        
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Failed to complete data preprocessing: {e}")

if __name__ == '__main__':
    main()