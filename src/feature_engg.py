# feature_engg.py
"""
Feature Engineering stage.

Responsibilities:
    1. Read the cleaned train / test data produced by data_preprocessing.
    2. One-hot encode multi-category columns (furnishingstatus).
    3. Create new engineered features (area_per_bedroom, total_rooms).
    4. Scale numerical features using StandardScaler (fit on train only).
    5. Persist the final model-ready data to `data/features/train.csv`
       and `data/features/test.csv`.
"""

import logging
import os
import pickle
import sys

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("feature_engg")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "feature_engg.log"))
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


def create_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Create new engineered features."""
    if params.get("create_area_per_bedroom", False):
        df["area_per_bedroom"] = df["area"] / df["bedrooms"].replace(0, 1)
        logger.debug("Created feature 'area_per_bedroom'")

    if params.get("create_total_rooms", False):
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        logger.debug("Created feature 'total_rooms'")

    return df


def onehot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list):
    """One-hot encode categorical columns consistently across train/test."""
    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=cols, drop_first=True)

    # cast the newly created dummy columns to int (0/1) instead of bool
    dummy_cols = [c for c in combined.columns if any(c.startswith(f"{col}_") for col in cols)]
    combined[dummy_cols] = combined[dummy_cols].astype(int)

    train_encoded = combined.xs("train")
    test_encoded = combined.xs("test")
    logger.debug("One-hot encoded columns: %s", cols)
    return train_encoded, test_encoded


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, scaler_path: str):
    """Scale numerical feature columns using StandardScaler fit on train data."""
    feature_cols = [c for c in train_df.columns if c != target_col]
    numeric_cols = train_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.debug("Scaler fit on columns %s and saved to %s", numeric_cols, scaler_path)

    return train_df, test_df


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
        preprocessing_params = all_params["data_preprocessing"]
        feature_params = all_params["feature_engineering"]
        target_col = all_params["model_building"]["target_column"]

        processed_dir = preprocessing_params["processed_dir"]
        features_dir = feature_params["features_dir"]
        onehot_cols = feature_params["onehot_cols"]

        train_df = load_data(os.path.join(processed_dir, "train.csv"))
        test_df = load_data(os.path.join(processed_dir, "test.csv"))

        train_df = create_features(train_df, feature_params)
        test_df = create_features(test_df, feature_params)

        train_df, test_df = onehot_encode(train_df, test_df, onehot_cols)

        if feature_params.get("scale_features", False):
            scaler_path = os.path.join(features_dir, "scaler.pkl")
            train_df, test_df = scale_features(train_df, test_df, target_col, scaler_path)

        save_data(train_df, os.path.join(features_dir, "train.csv"))
        save_data(test_df, os.path.join(features_dir, "test.csv"))

        logger.info("Feature engineering completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
