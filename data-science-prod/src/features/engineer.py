"""
Feature Engineering Module
==========================
Transforms the cleaned DataFrame into the final feature matrix
ready for model training / inference.

Steps:
  1. Haversine distance calculation
  2. Temporal feature extraction (hour, day-of-week, month)
  3. Holiday detection (BR calendar)
  4. Weather integration (already numeric from preprocess)
  5. Column renaming and selection
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import holidays

from src.utils.haversine import haversine_distance

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def build_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Returns the DataFrame with all model-ready columns added
    (and raw columns still present for traceability).
    """
    feat_cfg = cfg.features

    # 1. Haversine distance
    logger.info("Computing Haversine distances …")
    df["distancia_km"] = haversine_distance(
        df["LatOrig"].values,
        df["LongOrig"].values,
        df["LatDest"].values,
        df["LongDest"].values,
    )
    df = df.dropna(subset=["distancia_km"])

    # 2. Temporal features
    logger.info("Extracting temporal features …")
    df["hora"] = df["Partida.Prevista"].dt.hour
    df["dia_semana"] = df["Partida.Prevista"].dt.dayofweek
    df["mes"] = df["Partida.Prevista"].dt.month

    # 3. Holidays
    logger.info("Computing holiday flags …")
    br_holidays = holidays.Brazil()
    df["is_holiday"] = (
        df["Partida.Prevista"]
        .dt.date
        .apply(lambda x: 1 if x in br_holidays else 0)
    )

    # 4. Ensure clima_imputado exists
    if "clima_imputado" not in df.columns:
        df["clima_imputado"] = 0

    # 5. Rename columns
    rename_map = dict(feat_cfg.rename_map)
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)
    logger.info("Renamed columns: %s", existing_renames)

    # 6. Ensure categorical columns are string type
    for col in feat_cfg.categorical:
        df[col] = df[col].astype(str).fillna("MISSING")

    # 7. Sort chronologically
    df = df.sort_values("Partida.Prevista").reset_index(drop=True)

    logger.info("Feature engineering complete — %s rows", f"{len(df):,}")
    return df


def select_features(
    df: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Select model features and target from the engineered DataFrame.

    Returns
    -------
    X : pd.DataFrame — Feature matrix.
    y : pd.Series    — Binary target.
    feature_names : list[str] — Ordered feature column names.
    cat_feature_names : list[str] — Categorical feature names.
    """
    feat_cfg = cfg.features
    feature_names = list(feat_cfg.categorical) + list(feat_cfg.numerical)
    cat_feature_names = list(feat_cfg.categorical)

    X = df[feature_names].copy()
    y = df[cfg.data.target.column].copy()

    logger.info(
        "Feature matrix: %s rows × %s features (%s categorical)",
        f"{len(X):,}",
        len(feature_names),
        len(cat_feature_names),
    )
    return X, y, feature_names, cat_feature_names


def build_airport_coords(df: pd.DataFrame) -> dict:
    """
    Extract a lookup dict {airport_name: {lat, long}} from the raw data.
    Used by the serving API for automatic distance calculation.
    """
    coords = {}

    # From origin columns
    if "Aeroporto.Origem" in df.columns:
        col_name = "Aeroporto.Origem"
    elif "origem" in df.columns:
        col_name = "origem"
    else:
        return coords

    lat_col = "LatOrig" if "LatOrig" in df.columns else None
    lon_col = "LongOrig" if "LongOrig" in df.columns else None

    if lat_col and lon_col:
        grouped = df.groupby(col_name)[[lat_col, lon_col]].first()
        for airport, row in grouped.iterrows():
            coords[airport] = {"lat": float(row[lat_col]), "long": float(row[lon_col])}

    # Fill gaps from destination columns
    dest_col = "Aeroporto.Destino" if "Aeroporto.Destino" in df.columns else "destino"
    dlat = "LatDest" if "LatDest" in df.columns else None
    dlon = "LongDest" if "LongDest" in df.columns else None

    if dlat and dlon and dest_col in df.columns:
        grouped_dest = df.groupby(dest_col)[[dlat, dlon]].first()
        for airport, row in grouped_dest.iterrows():
            if airport not in coords:
                coords[airport] = {"lat": float(row[dlat]), "long": float(row[dlon])}

    logger.info("Built coordinate lookup for %s airports", len(coords))
    return coords
