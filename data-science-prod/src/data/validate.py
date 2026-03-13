"""
Data Validation Module
======================
Schema-based validation for the preprocessed DataFrame.
Uses Pandera for declarative checks; falls back to manual
assertions if Pandera is not installed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Try to use Pandera; degrade gracefully                                      #
# --------------------------------------------------------------------------- #
_HAS_PANDERA = False
try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema

    _HAS_PANDERA = True
except ImportError:
    pass


def _build_pandera_schema(cfg: DictConfig) -> "DataFrameSchema":
    """Build a Pandera schema from the Hydra config."""
    cat_cols = list(cfg.features.categorical)
    num_cols = list(cfg.features.numerical)

    columns = {}

    # Categorical — must be non-null strings
    for col in cat_cols:
        columns[col] = Column(
            dtype="object",
            nullable=False,
            checks=Check.str_length(min_value=1),
        )

    # Numerical — basic range checks
    range_checks = {
        "distancia_km": (0, 20_000),
        "hora": (0, 23),
        "dia_semana": (0, 6),
        "mes": (1, 12),
        "is_holiday": (0, 1),
        "precipitation": (0, 500),
        "wind_speed": (0, 200),
        "clima_imputado": (0, 1),
    }

    for col in num_cols:
        lo, hi = range_checks.get(col, (None, None))
        checks = []
        if lo is not None:
            checks.append(Check.ge(lo))
        if hi is not None:
            checks.append(Check.le(hi))
        columns[col] = Column(nullable=False, checks=checks)

    # Target
    columns["target"] = Column(
        dtype="int64",
        nullable=False,
        checks=Check.isin([0, 1]),
    )

    return DataFrameSchema(columns=columns, coerce=True)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


def validate(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Validate the preprocessed DataFrame.

    * If Pandera is available, runs a full schema check.
    * Otherwise, runs lightweight assertions.

    Returns the validated (and possibly coerced) DataFrame.
    """
    if _HAS_PANDERA:
        logger.info("Running Pandera schema validation …")
        schema = _build_pandera_schema(cfg)
        # Select only the columns the schema knows about
        cols_to_validate = [
            c for c in schema.columns if c in df.columns
        ]
        df_subset = df[cols_to_validate]
        schema.validate(df_subset, lazy=True)
        logger.info("Schema validation PASSED")
    else:
        logger.warning(
            "Pandera not installed — running lightweight assertions."
        )
        _manual_checks(df, cfg)

    return df


def _manual_checks(df: pd.DataFrame, cfg: DictConfig) -> None:
    """Fallback validation without Pandera."""
    all_features = list(cfg.features.categorical) + list(cfg.features.numerical)

    # Check required columns exist
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check no nulls in features
    null_counts = df[all_features].isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    if len(has_nulls) > 0:
        raise ValueError(f"Unexpected nulls in features:\n{has_nulls}")

    # Check target is binary
    assert set(df["target"].unique()).issubset({0, 1}), "Target must be binary"

    # Check categorical columns are strings
    for col in cfg.features.categorical:
        assert df[col].dtype == "object", f"{col} must be string dtype"

    logger.info("Manual validation PASSED")
