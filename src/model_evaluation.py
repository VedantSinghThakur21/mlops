# model_evaluation.py
"""
Model Evaluation stage.

Responsibilities:
    1. Load the trained model and the feature-engineered test data.
    2. Generate predictions and compute regression metrics
       (MAE, MSE, RMSE, R2).
    3. Persist the metrics as `metrics/metrics.json` so DVC can
       track and compare them across experiments (`dvc metrics show/diff`).
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "model_evaluation.log"))
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


def load_model(model_path: str):
    """Load a pickled model from disk."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded from %s", model_path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute regression metrics for the given model and test set."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2),
    }
    logger.debug("Computed metrics: %s", metrics)
    return metrics


def save_metrics(metrics: dict, out_dir: str, filename: str) -> None:
    """Persist metrics dictionary as a JSON file."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", out_path)
    except Exception as e:
        logger.error("Error occurred while saving metrics: %s", e)
        raise


def main():
    try:
        all_params = load_params("params.yaml")
        feature_params = all_params["feature_engineering"]
        model_params = all_params["model_building"]
        eval_params = all_params["model_evaluation"]

        features_dir = feature_params["features_dir"]
        target_col = eval_params["target_column"]

        model_path = os.path.join(model_params["model_dir"], model_params["model_name"])
        model = load_model(model_path)

        test_df = load_data(os.path.join(features_dir, "test.csv"))
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        metrics = evaluate_model(model, X_test, y_test)

        save_metrics(metrics, eval_params["metrics_dir"], eval_params["metrics_file"])

        logger.info("Model evaluation completed successfully. Metrics: %s", metrics)
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
