"""
Data Preprocessing Module
=========================
Loads the enriched CSV, applies scope filters, removes duplicates,
handles nulls, and computes the binary target.

All rules are driven by the Hydra config (cfg.data.*).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def load_raw(cfg: DictConfig) -> pd.DataFrame:
    """Load raw CSV from the path specified in ``cfg.data.raw_path``."""
    path = cfg.data.raw_path
    logger.info("Loading raw dataset: %s", path)
    df = pd.read_csv(path, low_memory=cfg.data.get("low_memory", False))
    logger.info("Loaded %s rows × %s cols", f"{len(df):,}", df.shape[1])
    return df


def clean(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop duplicates
      2. Drop leakage / useless columns
      3. Filter scope (Realizados only)
      4. Coerce coordinate & weather columns
      5. Parse datetimes
      6. Drop essential nulls
      7. Compute delay & duration
      8. Remove outliers
      9. Build binary target
    """
    data_cfg = cfg.data

    # 1. Duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.info("Removing %s duplicates", f"{n_dup:,}")
        df = df.drop_duplicates()

    # 2. Drop columns
    cols_drop = [c for c in data_cfg.columns_to_drop if c in df.columns]
    if cols_drop:
        df = df.drop(columns=cols_drop)
        logger.info("Dropped columns: %s", cols_drop)

    # 3. Scope filter
    scope_col = data_cfg.scope_column
    scope_val = data_cfg.scope_value
    if scope_col in df.columns:
        before = len(df)
        df = df[df[scope_col] == scope_val].copy()
        df = df.drop(columns=[scope_col])
        logger.info(
            "Scope filter (%s=%s): %s → %s rows",
            scope_col,
            scope_val,
            f"{before:,}",
            f"{len(df):,}",
        )

    # 4. Coerce numerics
    for col in data_cfg.coordinate_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in data_cfg.weather_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 5. Datetimes
    for col in data_cfg.datetime_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 6. Essential nulls
    df = df.dropna(subset=["Partida.Prevista", "Partida.Real"])

    # 7. Delay & Duration
    df["delay_minutes"] = (
        (df["Partida.Real"] - df["Partida.Prevista"]).dt.total_seconds() / 60
    )
    df["duration_minutes"] = (
        (df["Chegada.Real"] - df["Partida.Real"]).dt.total_seconds() / 60
    )

    # 8. Outlier removal
    oc = data_cfg.outliers
    mask = (
        (df["duration_minutes"] > oc.min_duration_minutes)
        & (df["delay_minutes"] > oc.min_delay_minutes)
        & (df["delay_minutes"] < oc.max_delay_minutes)
    )
    before = len(df)
    df = df[mask].copy()
    logger.info("Outlier filter: %s → %s rows", f"{before:,}", f"{len(df):,}")

    # 9. Target
    threshold_min = data_cfg.target.delay_threshold_minutes
    df[data_cfg.target.column] = np.where(
        df["delay_minutes"] > threshold_min, 1, 0
    )

    logger.info(
        "Target distribution:\n%s",
        df[data_cfg.target.column].value_counts().to_string(),
    )

    return df.reset_index(drop=True)
