"""
FlightOnTime — Production Serving API (v6.0)
=============================================
FastAPI microservice with two model-loading strategies:

  1. **MLflow Registry** (preferred) — loads from the MLflow Model Registry.
  2. **Joblib fallback** — loads the packaged .joblib artifact.

The model source is controlled by the ``MODEL_SOURCE`` env var:
  - ``mlflow``  → load from registry (default if MLflow server is reachable)
  - ``joblib``  → load from disk

Usage:
    MODEL_SOURCE=joblib uvicorn src.serving.app:app --reload
    MODEL_SOURCE=mlflow uvicorn src.serving.app:app --reload
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import holidays
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
#  App Setup                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

app = FastAPI(
    title="FlightOnTime AI Service",
    version="6.0.0",
    description="Production-grade flight delay prediction with live weather integration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────── #
#  Model Loading                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

MODEL = None
FEATURES_LIST: list[str] = []
CAT_FEATURES: list[str] = []
AIRPORT_COORDS: dict = {}
THRESHOLD: float = 0.35
MODEL_VERSION: str = "unknown"


def _load_from_mlflow() -> bool:
    """Try loading the latest Production model from MLflow Registry."""
    global MODEL, FEATURES_LIST, CAT_FEATURES, THRESHOLD, MODEL_VERSION
    try:
        import mlflow.catboost

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        model_name = os.getenv("MLFLOW_MODEL_NAME", "FlightOnTime")
        model_uri = f"models:/{model_name}/latest"

        logger.info("Loading model from MLflow: %s", model_uri)
        MODEL = mlflow.catboost.load_model(model_uri)
        MODEL_VERSION = f"mlflow-{model_name}-latest"
        logger.info("MLflow model loaded successfully")
        return True
    except Exception as e:
        logger.warning("MLflow loading failed: %s", e)
        return False


def _load_from_joblib() -> bool:
    """Load from the packaged joblib artifact."""
    global MODEL, FEATURES_LIST, CAT_FEATURES, AIRPORT_COORDS, THRESHOLD, MODEL_VERSION
    try:
        artifact_dir = os.getenv(
            "ARTIFACT_PATH",
            os.path.join(os.path.dirname(__file__), "../../artifacts"),
        )
        # Find the latest .joblib file
        import glob

        candidates = glob.glob(os.path.join(artifact_dir, "*.joblib"))
        if not candidates:
            logger.error("No .joblib artifact found in %s", artifact_dir)
            return False

        artifact_path = max(candidates, key=os.path.getmtime)
        logger.info("Loading joblib artifact: %s", artifact_path)

        artifact = joblib.load(artifact_path)
        MODEL = artifact["model"]
        FEATURES_LIST = artifact["features"]
        CAT_FEATURES = artifact.get("cat_features", [])
        AIRPORT_COORDS = artifact.get("airport_coords", {})

        meta = artifact.get("metadata", {})
        THRESHOLD = meta.get("threshold", 0.35)
        MODEL_VERSION = meta.get("version", "joblib-unknown")

        logger.info("Joblib model loaded: version=%s", MODEL_VERSION)
        return True
    except Exception as e:
        logger.error("Joblib loading failed: %s", e)
        return False


# ── Startup ──
@app.on_event("startup")
def startup_load_model():
    source = os.getenv("MODEL_SOURCE", "joblib").lower()
    if source == "mlflow":
        if not _load_from_mlflow():
            logger.warning("Falling back to joblib …")
            _load_from_joblib()
    else:
        _load_from_joblib()


# ──────────────────────────────────────────────────────────────────────────── #
#  Utilities                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _get_live_weather(lat: float, lon: float, dt_str: str) -> tuple[float, float, str]:
    """Fetch weather from Open-Meteo. Returns (precipitation, wind_speed, source)."""
    try:
        dt = pd.to_datetime(dt_str)
        date_str = dt.strftime("%Y-%m-%d")
        hour = dt.hour

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=precipitation,wind_speed_10m"
            f"&start_date={date_str}&end_date={date_str}&timezone=UTC"
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if "hourly" in data:
                precip = float(data["hourly"]["precipitation"][hour])
                wind = float(data["hourly"]["wind_speed_10m"][hour])
                return precip, wind, "LIVE (OpenMeteo)"
    except Exception as e:
        logger.warning("Weather API error: %s", e)

    return 0.0, 5.0, "Offline/Fallback"


# ──────────────────────────────────────────────────────────────────────────── #
#  Request / Response Models                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


class FlightInput(BaseModel):
    airline: str
    origin: str
    destination: str
    departure_datetime: str
    distance_km: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_color: str
    model_version: str
    data_used: dict


# ──────────────────────────────────────────────────────────────────────────── #
#  Endpoints                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL is not None else "no_model",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Auto-distance
        dist = flight.distance_km
        orig_lat, orig_lon = 0.0, 0.0

        if flight.origin in AIRPORT_COORDS:
            orig_lat = AIRPORT_COORDS[flight.origin]["lat"]
            orig_lon = AIRPORT_COORDS[flight.origin]["long"]

        if not dist or dist == 0:
            if flight.origin in AIRPORT_COORDS and flight.destination in AIRPORT_COORDS:
                dest = AIRPORT_COORDS[flight.destination]
                dist = _haversine(orig_lat, orig_lon, dest["lat"], dest["long"])
            else:
                dist = 800.0

        # 2. Auto-weather
        precip = flight.precipitation
        wind = flight.wind_speed
        weather_source = "Manual"

        if precip is None and wind is None:
            if orig_lat != 0:
                precip, wind, weather_source = _get_live_weather(
                    orig_lat, orig_lon, flight.departure_datetime
                )
            else:
                precip, wind, weather_source = 0.0, 5.0, "No coords"
        else:
            precip = precip if precip is not None else 0.0
            wind = wind if wind is not None else 5.0

        # 3. Feature engineering
        dt = pd.to_datetime(flight.departure_datetime)
        is_holiday = 1 if dt.date() in holidays.Brazil() else 0

        input_df = pd.DataFrame(
            [
                {
                    "companhia": str(flight.airline),
                    "origem": str(flight.origin),
                    "destino": str(flight.destination),
                    "distancia_km": float(dist),
                    "hora": dt.hour,
                    "dia_semana": dt.dayofweek,
                    "mes": dt.month,
                    "is_holiday": is_holiday,
                    "precipitation": float(precip),
                    "wind_speed": float(wind),
                    "clima_imputado": 0,
                }
            ]
        )

        if FEATURES_LIST:
            input_df = input_df[FEATURES_LIST]

        # 4. Prediction
        prob = float(MODEL.predict_proba(input_df)[0][1])

        if prob < THRESHOLD:
            status, color = "ON_TIME", "green"
        elif prob < 0.70:
            status, color = "PREVENTIVE_ALERT", "yellow"
        else:
            status, color = "LIKELY_DELAYED", "red"

        return PredictionResponse(
            prediction=status,
            probability=round(prob, 4),
            risk_color=color,
            model_version=MODEL_VERSION,
            data_used={
                "distance_km": round(dist, 1),
                "precipitation_mm": precip,
                "wind_speed_kmh": wind,
                "weather_source": weather_source,
                "is_holiday": bool(is_holiday),
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
