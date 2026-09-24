# data_ingestion.py
"""
Data Ingestion stage.

Responsibilities:
    1. Read the raw Housing.csv dataset.
    2. Perform a basic sanity check on the data.
    3. Split it into train / test sets.
    4. Persist the splits to `data/raw/train.csv` and `data/raw/test.csv`
       so that downstream stages never touch the original raw file.
"""

import logging
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load YAML parameters from the given path."""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("Params file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error while parsing params file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading params: %s", e)
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s with shape %s", data_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except FileNotFoundError:
        logger.error("Raw data file not found at: %s", data_path)
        raise
    except Exception as e:
        logger.error("Unexpected error loading data: %s", e)
        raise


def validate_data(df: pd.DataFrame) -> None:
    """Run lightweight sanity checks on the raw dataframe."""
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' not found in dataset.")
    logger.debug("Data validation passed. Null counts:\n%s", df.isnull().sum())


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """Persist train/test splits to the given output directory."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, "train.csv")
        test_path = os.path.join(out_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.debug("Train data saved to %s", train_path)
        logger.debug("Test data saved to %s", test_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving data: %s", e)
        raise


def main():
    try:
        params = load_params("params.yaml")["data_ingestion"]
        raw_data_path = params["raw_data_path"]
        raw_dir = params["raw_dir"]
        test_size = params["test_size"]
        random_state = params["random_state"]

        df = load_data(raw_data_path)
        validate_data(df)

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        logger.info(
            "Data split into train (%d rows) and test (%d rows)",
            len(train_df),
            len(test_df),
        )

        save_data(train_df, test_df, raw_dir)
        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
