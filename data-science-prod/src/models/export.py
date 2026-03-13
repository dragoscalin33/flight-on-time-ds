"""
Model Export Module
===================
Serializes the production model as a joblib artifact with metadata,
and optionally registers it in the MLflow Model Registry.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Artifact filename template
ARTIFACT_FILENAME = "flight_classifier_{version}.joblib"


def export_production_artifact(
    model,
    feature_names: list[str],
    cat_feature_names: list[str],
    airport_coords: dict,
    threshold: float,
    metrics: dict,
    cv_metrics: dict,
    cfg: "DictConfig",
) -> str:
    """
    Package the trained model + metadata into a single joblib artifact.

    The artifact is:
      1. Saved to the configured output directory.
      2. Logged to the active MLflow run.

    Returns the path to the saved artifact.
    """
    version = cfg.model.name
    filename = ARTIFACT_FILENAME.format(version=version)

    output_dir = Path(cfg.output.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / filename

    artifact = {
        "model": model,
        "features": feature_names,
        "cat_features": cat_feature_names,
        "airport_coords": airport_coords,
        "metadata": {
            "version": version,
            "threshold": threshold,
            "holdout_metrics": metrics,
            "cv_metrics": cv_metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    }

    joblib.dump(artifact, artifact_path)
    logger.info("Artifact saved: %s", artifact_path)

    # Also log to MLflow
    mlflow.log_artifact(str(artifact_path), "production_artifact")

    return str(artifact_path)
