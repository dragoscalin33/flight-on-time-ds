# FlightOnTime — Production ML Pipeline (v6.0)

> **Status:** Production-Ready | **Recall:** 90.8% | **Stack:** CatBoost + Hydra + MLflow

An AI engine for predicting flight delays in Brazilian commercial aviation. Version 6.0 evolves the monolithic notebook architecture into a **production-grade pipeline** with configuration management, experiment tracking, data validation, and cross-validation.

---

## Pipeline Overview

<p align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/rrbotlab/flight-on-time/blob/main/data-science-prod/docs/pipeline_overview.html">
    <img src="https://img.shields.io/badge/%F0%9F%94%8D_View_Interactive-Pipeline_Diagram-blueviolet?style=for-the-badge" alt="View Pipeline Diagram">
  </a>
</p>

---

## Architectural Evolution (v5.0 → v6.0)

The original system (v5.0) was a single Jupyter notebook that handled everything — data loading, cleaning, feature engineering, model training, threshold tuning, and artifact serialization — in one linear flow. While effective for research and prototyping, this approach had critical limitations for production deployment:

- **No reproducibility**: hyperparameters and paths were hardcoded across notebook cells.
- **No experiment tracking**: metrics were printed to stdout and lost between runs.
- **No validation**: malformed data could silently corrupt model training.
- **No modular testing**: everything was tightly coupled in one file.

Version 6.0 addresses all of these by introducing three production pillars:

### 1. Hydra (Configuration Management)
Every parameter — model hyperparameters, data paths, outlier rules, feature lists, threshold values — lives in **YAML config files**. The Python code reads from the config and never contains magic numbers. This means:
- Changing `iterations` from 500 to 1000 requires editing a YAML file, not code.
- Different configs can be composed (e.g., `model=catboost_tuned`) without code changes.
- CLI overrides work out of the box: `python -m src.models.train model.depth=8`.
- Every run's full config is automatically saved for reproducibility.

### 2. MLflow (Experiment Tracking & Model Registry)
Every training run logs its parameters, metrics, and artifacts to MLflow:
- **Parameters**: all model hyperparameters, split ratios, CV settings.
- **Metrics**: holdout recall/precision/F1/F2/ROC-AUC, plus per-fold CV metrics.
- **Artifacts**: confusion matrix plots, feature importance charts, threshold sensitivity curves.
- **Model Registry**: the production model is registered with lifecycle stages (Staging → Production → Archived).

### 3. Stratified K-Fold Cross-Validation
Instead of relying on a single train/test split, we run 5-fold stratified CV on the training set. Each fold preserves the class ratio (88.4% on-time / 11.6% delayed), giving a robust performance estimate with mean and standard deviation for every metric.

| Aspect | v5.0 (Notebook) | v6.0 (Production) |
|:--------|:----------------|:--------------------|
| Configuration | Hardcoded in code | YAML via Hydra |
| Experiments | print() to stdout | MLflow tracking |
| Validation | None | Pandera schemas |
| CV | None (single holdout) | Stratified K-Fold |
| Code | All in one notebook | Modules in `src/` |
| Model artifact | Manual .joblib | MLflow Registry + .joblib |
| API | Coupled to artifact format | Health check + dual loading |

---

## Project Structure

```
data-science-prod/
├── config/                        # Hydra configuration (YAML)
│   ├── config.yaml                # Main config (composes all groups)
│   ├── model/
│   │   ├── catboost.yaml          # Default: 500 iter, depth 6, balanced weights
│   │   └── catboost_tuned.yaml    # Experimental: 800 iter, depth 8, L2 reg
│   ├── data/
│   │   └── flights_v4.yaml        # Paths, scope filters, outlier rules, target def
│   ├── features/
│   │   └── weather_aware.yaml     # Feature list, categoricals, column renaming
│   └── threshold/
│       └── safety_first.yaml      # Business override (0.35), semaphore rules
├── src/                           # Source modules
│   ├── data/
│   │   ├── preprocess.py          # Load CSV, clean, filter scope, build target
│   │   └── validate.py            # Pandera schema validation (graceful fallback)
│   ├── features/
│   │   └── engineer.py            # Haversine, temporal, holidays, weather, coords
│   ├── models/
│   │   ├── train.py               # Hydra @main + MLflow + K-Fold CV (10-step pipeline)
│   │   ├── evaluate.py            # Threshold optimization, confusion matrix, feat imp
│   │   └── export.py              # Dual export: joblib artifact + MLflow Registry
│   ├── serving/
│   │   └── app.py                 # FastAPI with /health + /predict, dual model loading
│   └── utils/
│       └── haversine.py           # Vectorized geodesic distance calculation
├── notebooks/
│   └── production_pipeline.ipynb  # Interactive walkthrough of the full pipeline
├── data/                          # Datasets (git-ignored)
│   ├── raw/                       # BrFlights_Enriched_v4.csv (751MB)
│   ├── processed/                 # Intermediate datasets from earlier versions
│   └── references/                # Airport codes, weather history
├── models/                        # Exported model artifacts (git-ignored)
├── Makefile                       # Convenience commands (make train, make serve, etc.)
├── requirements.txt               # Python dependencies
└── .gitignore
```

---

## Training Pipeline (10 Steps)

When you run `python -m src.models.train`, the pipeline executes:

1. **Load** — Read the enriched CSV from the path in `cfg.data.raw_path`.
2. **Clean** — Drop duplicates, remove leakage columns, filter to completed flights only, coerce types, drop essential nulls, remove outliers, build binary target (delay > 15 min).
3. **Engineer Features** — Compute Haversine distance, extract hour/day-of-week/month, flag Brazilian holidays, validate weather columns, rename to model-ready names.
4. **Validate** — Run Pandera schema checks (type safety, range bounds, no nulls).
5. **Split** — 80/20 stratified train/test split.
6. **Cross-Validate** — 5-fold stratified CV on the training set, logging mean ± std for recall, precision, F1, F2, accuracy, and ROC-AUC.
7. **Train** — Fit CatBoost on the full training set with native categorical support and balanced class weights.
8. **Evaluate** — Holdout test metrics + threshold sensitivity analysis (F2-optimal vs. business override).
9. **Log Artifacts** — Confusion matrix, feature importance, and threshold curve plots → MLflow.
10. **Export** — Re-train on all data, register in MLflow Model Registry, save joblib artifact.

---

## Quick Start

```bash
# 1. Setup
cd data-science-prod
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train (default config)
python -m src.models.train

# 3. Train (alternative config)
python -m src.models.train model=catboost_tuned

# 4. Train (CLI override)
python -m src.models.train model.iterations=1000 cv.n_splits=10

# 5. View experiments
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000

# 6. Serve API
MODEL_SOURCE=joblib uvicorn src.serving.app:app --reload
# Or from MLflow: MODEL_SOURCE=mlflow uvicorn src.serving.app:app --reload

# 7. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"airline":"GOL","origin":"Congonhas","destination":"Santos Dumont","departure_datetime":"2025-12-24T14:00:00"}'
```

---

## Key Technical Decisions

### Why Random Split (Not Temporal)?

Each flight's delay is **conditionally independent** given the features available at prediction time (scheduled departure, route, weather forecast). We do not use any lagging or autoregressive features (like "average delay in the last 24 hours"), so knowing the outcome of a flight in March does not leak information about a flight in January. Stratified K-fold CV is the correct and more rigorous validation strategy for this feature set.

### Why CatBoost with Native Categorical Support?

Instead of manually encoding airline names and airport names (Label Encoding assigns arbitrary numbers like GOL=1, TAM=2), we pass the raw string columns directly to CatBoost. Internally, CatBoost uses **Ordered Target Encoding**, which converts categories into numbers based on their correlation with the target — without causing data leakage. This captures nuances (e.g., "Congonhas has more delays than Guarulhos") that label encoding would lose.

### Why Threshold 0.35 Instead of 0.43?

The F2-Score optimal threshold is approximately 0.43. However, the business decision is to use 0.35 — a **safety-first override**. At 0.35, the model achieves >90% Recall (catches 9 out of 10 real delays) at the cost of more false alerts. In aviation, a preventive alert that turns out unnecessary is far less costly than a missed delay that causes a passenger to miss their connection.

### Why Balanced Class Weights?

Only 11.6% of flights are delayed. A naive model predicting "on-time" for everything would achieve 88.4% accuracy but 0% Recall — completely useless for delay detection. Setting `auto_class_weights='Balanced'` makes errors on the minority class (delays) much more expensive for the optimizer, forcing it to learn delay patterns rather than defaulting to the majority class.

---

## API Reference

### `GET /health`
Returns model status, version, and threshold.

### `POST /predict`

**Request:**
```json
{
  "airline": "GOL",
  "origin": "Congonhas",
  "destination": "Santos Dumont",
  "departure_datetime": "2025-12-24T14:00:00",
  "distance_km": null,
  "precipitation": null,
  "wind_speed": null
}
```
*Note: `distance_km`, `precipitation`, and `wind_speed` are optional. If omitted, the API computes geodesic distance automatically and fetches live weather from Open-Meteo.*

**Response:**
```json
{
  "prediction": "PREVENTIVE_ALERT",
  "probability": 0.654,
  "risk_color": "yellow",
  "model_version": "catboost_native",
  "data_used": {
    "distance_km": 366.0,
    "precipitation_mm": 5.2,
    "wind_speed_kmh": 12.0,
    "weather_source": "LIVE (OpenMeteo)",
    "is_holiday": true
  }
}
```

### Risk Semaphore

| Range | Status | Action |
|:------|:-------|:-------|
| < 35% | ON_TIME (green) | No action needed |
| 35–70% | PREVENTIVE_ALERT (yellow) | Monitor flight status |
| > 70% | LIKELY_DELAYED (red) | Consider contingency plan |

---

## Technology Stack

- **Language:** Python 3.10+
- **ML Core:** CatBoost (Gradient Boosting with native categorical support)
- **Configuration:** Hydra + OmegaConf
- **Experiment Tracking:** MLflow
- **Data Validation:** Pandera
- **API:** FastAPI + Uvicorn
- **External Data:** Open-Meteo API (real-time weather)
- **Deployment:** Docker / Oracle Cloud Infrastructure

---

## Dataset

- **Source:** Flights in Brazil (2015-2017) — Kaggle
- **Weather Enrichment:** Open-Meteo Historical API
- **Size:** 2.5M raw records → 2.2M after cleaning
- **Target:** Binary (1 = delay > 15 min, 0 = on-time)
- **Class Balance:** 88.4% on-time / 11.6% delayed
