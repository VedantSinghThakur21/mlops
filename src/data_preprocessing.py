# data_preprocessing.py
"""
Data Preprocessing stage.

Responsibilities:
    1. Read the train / test splits produced by data_ingestion.
    2. Handle missing values and duplicates.
    3. Binary-encode yes/no categorical columns (mainroad, guestroom, etc.)
    4. Persist the cleaned data to `data/processed/train.csv` and
       `data/processed/test.csv`.
"""

import logging
import os
import sys

import pandas as pd
import yaml

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "data_preprocessing.log"))
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


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    try:
        df = pd.read_csv(path)
        logger.debug("Data loaded from %s with shape %s", path, df.shape)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except Exception as e:
        logger.error("Unexpected error loading data: %s", e)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and handle missing values."""
    before = df.shape[0]
    df = df.drop_duplicates()
    logger.debug("Dropped %d duplicate rows", before - df.shape[0])

    # Numerical columns: fill missing values with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled missing values in '%s' with median %s", col, median_val)

    # Categorical columns: fill missing values with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Filled missing values in '%s' with mode %s", col, mode_val)

    return df


def binary_encode(df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    """Encode yes/no columns to 1/0."""
    mapping = {"yes": 1, "no": 0}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(int)
            logger.debug("Binary encoded column '%s'", col)
        else:
            logger.warning("Binary column '%s' not found in dataframe", col)
    return df


def save_data(df: pd.DataFrame, out_path: str) -> None:
    """Persist a dataframe to CSV."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.debug("Data saved to %s", out_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving data: %s", e)
        raise


def main():
    try:
        all_params = load_params("params.yaml")
        ingestion_params = all_params["data_ingestion"]
        preprocessing_params = all_params["data_preprocessing"]

        raw_dir = ingestion_params["raw_dir"]
        processed_dir = preprocessing_params["processed_dir"]
        binary_cols = preprocessing_params["binary_cols"]

        train_df = load_data(os.path.join(raw_dir, "train.csv"))
        test_df = load_data(os.path.join(raw_dir, "test.csv"))

        train_df = clean_data(train_df)
        test_df = clean_data(test_df)

        train_df = binary_encode(train_df, binary_cols)
        test_df = binary_encode(test_df, binary_cols)

        save_data(train_df, os.path.join(processed_dir, "train.csv"))
        save_data(test_df, os.path.join(processed_dir, "test.csv"))

        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
