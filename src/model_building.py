# model_building.py
"""
Model Building stage.

Responsibilities:
    1. Read the feature-engineered training data.
    2. Split into X (features) / y (target = price).
    3. Train a RandomForestRegressor using hyperparameters from params.yaml.
    4. Persist the trained model to `models/model.pkl`.
"""

import logging
import os
import pickle
import sys

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "model_building.log"))
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


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestRegressor:
    """Train a RandomForestRegressor with the given hyperparameters."""
    try:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        logger.debug("Model training completed with params: %s", params)
        return model
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, out_dir: str, model_name: str) -> None:
    """Persist the trained model as a pickle file."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", model_path)
    except Exception as e:
        logger.error("Error occurred while saving the model: %s", e)
        raise


def main():
    try:
        all_params = load_params("params.yaml")
        feature_params = all_params["feature_engineering"]
        model_params = all_params["model_building"]

        features_dir = feature_params["features_dir"]
        target_col = model_params["target_column"]

        train_df = load_data(os.path.join(features_dir, "train.csv"))

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        model = train_model(X_train, y_train, model_params)

        save_model(model, model_params["model_dir"], model_params["model_name"])

        logger.info("Model building completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
