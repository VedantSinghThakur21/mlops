# app/main.py
"""
FastAPI inference service for the Housing Price Prediction model.

Wraps the trained RandomForestRegressor (models/model.pkl) and the fitted
StandardScaler (data/features/scaler.pkl) produced by the DVC pipeline in
`src/` (data_ingestion -> data_preprocessing -> feature_engg -> model_building
-> model_evaluation), and exposes them as an HTTP API.

Endpoints:
    GET  /            - basic service info
    GET  /health       - liveness/readiness probe (checks model & scaler are loaded)
    POST /predict      - predict the price for a single house
    POST /predict/batch - predict prices for a list of houses
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import List, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join("logs", "app.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Artifact paths can be overridden via environment variables at container
# runtime (e.g. if artifacts are mounted from a volume instead of baked into
# the image).
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "data/features/scaler.pkl")

# Columns that get binary-encoded (yes/no -> 1/0) during preprocessing.
BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

# Column that gets one-hot encoded during feature engineering.
ONEHOT_COL = "furnishingstatus"
FURNISHING_CATEGORIES = ["furnished", "semi-furnished", "unfurnished"]

# In-memory handles for the loaded artifacts, populated at startup.
ml_artifacts = {"model": None, "scaler": None}


# --------------------------------------------------------------------------- #
# Startup / shutdown
# --------------------------------------------------------------------------- #
def _load_artifacts() -> None:
    """Load the trained model and scaler from disk into memory."""
    try:
        with open(MODEL_PATH, "rb") as f:
            ml_artifacts["model"] = pickle.load(f)
        logger.info("Model loaded from %s", MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            ml_artifacts["scaler"] = pickle.load(f)
        logger.info("Scaler loaded from %s", SCALER_PATH)
    except FileNotFoundError as e:
        logger.error("Required artifact not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading artifacts: %s", e)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Housing Price Prediction API",
    description="Serves predictions from the RandomForestRegressor trained by the DVC pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
YesNo = Literal["yes", "no"]
Furnishing = Literal["furnished", "semi-furnished", "unfurnished"]


class HouseFeatures(BaseModel):
    """Raw house attributes, matching the columns in dataset/Housing.csv (minus price)."""

    area: float = Field(..., gt=0, description="Total area of the plot in square feet")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms")
    stories: int = Field(..., ge=0, description="Number of stories")
    mainroad: YesNo = Field(..., description="Whether the house faces a main road")
    guestroom: YesNo = Field(..., description="Whether the house has a guest room")
    basement: YesNo = Field(..., description="Whether the house has a basement")
    hotwaterheating: YesNo = Field(..., description="Whether the house has hot water heating")
    airconditioning: YesNo = Field(..., description="Whether the house has air conditioning")
    parking: int = Field(..., ge=0, description="Number of parking spots")
    prefarea: YesNo = Field(..., description="Whether the house is in a preferred area")
    furnishingstatus: Furnishing = Field(..., description="Furnishing status of the house")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": 7420,
                "bedrooms": 4,
                "bathrooms": 2,
                "stories": 3,
                "mainroad": "yes",
                "guestroom": "no",
                "basement": "no",
                "hotwaterheating": "no",
                "airconditioning": "yes",
                "parking": 2,
                "prefarea": "yes",
                "furnishingstatus": "furnished",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    predicted_prices: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool


# --------------------------------------------------------------------------- #
# Preprocessing (mirrors src/data_preprocessing.py + src/feature_engg.py,
# adapted for scoring a single request instead of a batch training file)
# --------------------------------------------------------------------------- #
def _binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"yes": 1, "no": 0}
    for col in BINARY_COLS:
        df[col] = df[col].map(mapping).astype(int)
    return df


def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    df["area_per_bedroom"] = df["area"] / df["bedrooms"].replace(0, 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    return df


def _onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    # Fix the category dtype so pd.get_dummies always produces the same set
    # of dummy columns, regardless of which furnishingstatus value(s) are
    # present in this particular request/batch (mirrors drop_first=True from
    # training, where "furnished" was the dropped/baseline category).
    df[ONEHOT_COL] = pd.Categorical(df[ONEHOT_COL], categories=FURNISHING_CATEGORIES)
    df = pd.get_dummies(df, columns=[ONEHOT_COL], drop_first=True)
    dummy_cols = [c for c in df.columns if c.startswith(f"{ONEHOT_COL}_")]
    df[dummy_cols] = df[dummy_cols].astype(int)
    return df


def preprocess(records: List[HouseFeatures]) -> pd.DataFrame:
    """Turn a list of raw HouseFeatures requests into a scaled feature matrix
    with columns in the exact order the model was trained on."""
    df = pd.DataFrame([r.model_dump() for r in records])

    df = _binary_encode(df)
    df = _create_features(df)
    df = _onehot_encode(df)

    scaler = ml_artifacts["scaler"]
    model = ml_artifacts["model"]

    # Align columns to what the model/scaler actually expect, in order.
    # (scaler.feature_names_in_ == model.feature_names_in_ for this pipeline)
    expected_cols = list(getattr(scaler, "feature_names_in_", model.feature_names_in_))
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns after preprocessing: {missing}")
    df = df[expected_cols]

    df[expected_cols] = scaler.transform(df[expected_cols])
    return df


def _predict(records: List[HouseFeatures]) -> List[float]:
    if ml_artifacts["model"] is None or ml_artifacts["scaler"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
    try:
        X = preprocess(records)
        preds = ml_artifacts["model"].predict(X)
        return [float(p) for p in np.atleast_1d(preds)]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def read_root():
    return {
        "service": "Housing Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check():
    return HealthResponse(
        status="ok" if ml_artifacts["model"] is not None and ml_artifacts["scaler"] is not None else "degraded",
        model_loaded=ml_artifacts["model"] is not None,
        scaler_loaded=ml_artifacts["scaler"] is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(house: HouseFeatures):
    """Predict the price for a single house."""
    prediction = _predict([house])[0]
    return PredictionResponse(predicted_price=prediction)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(houses: List[HouseFeatures]):
    """Predict prices for a batch of houses."""
    if not houses:
        raise HTTPException(status_code=400, detail="Request body must contain at least one house.")
    predictions = _predict(houses)
    return BatchPredictionResponse(predicted_prices=predictions)
