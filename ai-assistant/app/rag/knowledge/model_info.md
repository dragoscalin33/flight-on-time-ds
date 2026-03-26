# FlightOnTime ML Model Technical Details

## Model Overview

FlightOnTime uses a CatBoost gradient boosting classifier trained on 2.2 million Brazilian domestic flight records (2015-2017) enriched with historical weather data.

## Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.794 |
| Recall | 90.8% |
| Accuracy | 73.1% |
| F1-Score | 0.725 |
| Threshold | 0.35 (safety-first) |

## Risk Classification

- **Green (ON_TIME)**: delay probability < 35%
- **Yellow (PREVENTIVE_ALERT)**: delay probability 35-70%
- **Red (LIKELY_DELAYED)**: delay probability > 70%

The 35% threshold was chosen as a business decision favoring passenger safety — the model catches 90.8% of actual delays, accepting some false positives.

## What Counts as a Delay

A flight is classified as delayed if it departs more than 15 minutes after the scheduled departure time. This follows international aviation standards (US DOT definition).

## Dataset Characteristics

- **Source**: Brazilian ANAC flight records + OpenMeteo historical weather
- **Period**: January 2015 - December 2017
- **Class imbalance**: 88.4% on-time, 11.6% delayed
- **After cleaning**: 2.2M records (removed duplicates, missing values, data quality issues)

## Training Pipeline (Production v6.0)

1. Load enriched CSV dataset
2. Clean data (duplicates, leakage columns, type coercion)
3. Engineer features (Haversine distance, temporal features, holiday flags)
4. Validate with Pandera schemas
5. Stratified train/test split (80/20)
6. 5-fold stratified cross-validation
7. Train CatBoost on full training set
8. Evaluate with optimized threshold
9. Log artifacts to MLflow
10. Export model (joblib + MLflow Registry)

## Feature Importance (Top 5)

1. Origin airport - different airports have very different delay profiles
2. Hour of departure - time of day strongly correlates with delays
3. Airline - operational efficiency varies by carrier
4. Month - strong seasonal patterns
5. Precipitation - direct weather impact on operations

## Live Weather Integration

For predictions on future dates (within 16 days), the model fetches real-time weather forecasts from the OpenMeteo API. For dates beyond the forecast range, it uses conservative defaults (0mm rain, 5 km/h wind).
