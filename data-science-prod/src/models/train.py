"""
Training Pipeline
=================
Hydra-configured, MLflow-tracked training with Stratified K-Fold CV.

Usage (from project root):
    python -m src.models.train
    python -m src.models.train model=catboost_tuned
    python -m src.models.train model.iterations=1000 cv.n_splits=10
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Local modules — we add project root to path so Hydra can find them
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_raw, clean
from src.data.validate import validate
from src.features.engineer import (
    build_features,
    select_features,
    build_airport_coords,
)
from src.models.evaluate import (
    optimize_threshold,
    log_confusion_matrix,
    log_feature_importance,
)
from src.models.export import export_production_artifact

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _build_catboost(cfg: DictConfig, cat_features: list[str]) -> CatBoostClassifier:
    """Instantiate a CatBoostClassifier from config."""
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params.pop("name", None)  # not a CatBoost parameter
    return CatBoostClassifier(
        **model_params,
        cat_features=cat_features,
    )


# --------------------------------------------------------------------------- #
#  Cross-Validation                                                            #
# --------------------------------------------------------------------------- #


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: DictConfig,
    cat_features: list[str],
) -> dict:
    """
    Stratified K-Fold cross-validation.

    Returns a dict with mean ± std for each metric.
    """
    cv_cfg = cfg.cv
    skf = StratifiedKFold(
        n_splits=cv_cfg.n_splits,
        shuffle=cv_cfg.shuffle,
        random_state=cv_cfg.random_state,
    )

    fold_metrics = {
        "recall": [],
        "precision": [],
        "f1": [],
        "f2": [],
        "accuracy": [],
        "roc_auc": [],
    }

    logger.info(
        "Starting %s-Fold Stratified CV (shuffle=%s, seed=%s)",
        cv_cfg.n_splits,
        cv_cfg.shuffle,
        cv_cfg.random_state,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = _build_catboost(cfg, cat_features)
        model.fit(X_tr, y_tr, verbose=0)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        fold_metrics["recall"].append(recall_score(y_val, y_pred))
        fold_metrics["precision"].append(precision_score(y_val, y_pred))
        fold_metrics["f1"].append(f1_score(y_val, y_pred))
        fold_metrics["f2"].append(fbeta_score(y_val, y_pred, beta=2))
        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))

        logger.info(
            "  Fold %s/%s — Recall: %.3f | Precision: %.3f | ROC-AUC: %.3f",
            fold_idx,
            cv_cfg.n_splits,
            fold_metrics["recall"][-1],
            fold_metrics["precision"][-1],
            fold_metrics["roc_auc"][-1],
        )

    # Aggregate
    summary = {}
    for metric_name, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[f"cv_{metric_name}_mean"] = round(mean_val, 4)
        summary[f"cv_{metric_name}_std"] = round(std_val, 4)

    logger.info("CV Summary: %s", summary)
    return summary


# --------------------------------------------------------------------------- #
#  Main Training Entrypoint                                                    #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Full training pipeline: load → clean → features → CV → train → export."""
    logger.info("=" * 60)
    logger.info("FlightOnTime Training Pipeline v6.0")
    logger.info("=" * 60)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Change to project root so relative paths work
    os.chdir(str(PROJECT_ROOT))

    # ── MLflow Setup ──
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.model.name) as run:
        # Log all config as params
        flat_cfg = dict(OmegaConf.to_container(cfg, resolve=True))
        _log_nested_params(flat_cfg, prefix="")

        # ── 1. Data Loading & Cleaning ──
        logger.info("STEP 1: Loading & Cleaning")
        df_raw = load_raw(cfg)
        # Extract airport coords BEFORE renaming columns
        airport_coords = build_airport_coords(df_raw)
        df = clean(df_raw, cfg)

        # ── 2. Feature Engineering ──
        logger.info("STEP 2: Feature Engineering")
        df = build_features(df, cfg)

        # ── 3. Feature Selection ──
        logger.info("STEP 3: Feature Selection & Validation")
        X, y, feature_names, cat_feature_names = select_features(df, cfg)
        validate(
            pd.concat([X, y], axis=1),
            cfg,
        )

        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("class_balance", y.value_counts().to_dict())

        # ── 4. Train / Test Split ──
        logger.info("STEP 4: Train/Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.split.test_size,
            random_state=cfg.split.random_state,
            stratify=y if cfg.split.stratify else None,
        )
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # ── 5. Cross-Validation on Training Set ──
        logger.info("STEP 5: Stratified K-Fold Cross-Validation")
        cv_summary = run_cv(X_train, y_train, cfg, cat_feature_names)
        mlflow.log_metrics(cv_summary)

        # ── 6. Train Final Model on Full Train Set ──
        logger.info("STEP 6: Training Final Model on Training Set")
        model = _build_catboost(cfg, cat_feature_names)
        model.fit(X_train, y_train)

        # ── 7. Evaluate on Holdout Test ──
        logger.info("STEP 7: Holdout Test Evaluation")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        holdout_metrics = {
            "holdout_recall": recall_score(y_test, y_pred),
            "holdout_precision": precision_score(y_test, y_pred),
            "holdout_f1": f1_score(y_test, y_pred),
            "holdout_f2": fbeta_score(y_test, y_pred, beta=2),
            "holdout_accuracy": accuracy_score(y_test, y_pred),
            "holdout_roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(holdout_metrics)
        logger.info("Holdout metrics: %s", holdout_metrics)
        logger.info(
            "\n%s", classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"])
        )

        # ── 8. Threshold Optimization ──
        logger.info("STEP 8: Threshold Optimization")
        threshold_results = optimize_threshold(y_test, y_proba, cfg)
        mlflow.log_metrics(
            {
                "optimal_threshold": threshold_results["business_threshold"],
                "threshold_recall": threshold_results["recall"],
                "threshold_precision": threshold_results["precision"],
            }
        )

        # ── 9. Log Artifacts (Plots) ──
        logger.info("STEP 9: Logging Artifacts")
        log_confusion_matrix(y_test, y_proba, threshold_results["business_threshold"])
        log_feature_importance(model, feature_names)

        # ── 10. Re-train on Full Data & Export ──
        logger.info("STEP 10: Production Model (Full Data) & Export")
        model_full = _build_catboost(cfg, cat_feature_names)
        model_full.fit(X, y, verbose=0)

        # Log model to MLflow
        mlflow.catboost.log_model(
            model_full,
            artifact_path="model",
            registered_model_name=(
                "FlightOnTime" if cfg.mlflow.register_model else None
            ),
        )

        # Also export as joblib artifact
        export_production_artifact(
            model=model_full,
            feature_names=feature_names,
            cat_feature_names=cat_feature_names,
            airport_coords=airport_coords,
            threshold=threshold_results["business_threshold"],
            metrics=holdout_metrics,
            cv_metrics=cv_summary,
            cfg=cfg,
        )

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("MLflow Run ID: %s", run.info.run_id)
        logger.info("=" * 60)


def _log_nested_params(d: dict, prefix: str) -> None:
    """Flatten nested dict and log each leaf as an MLflow param."""
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _log_nested_params(v, key)
        else:
            try:
                mlflow.log_param(key, v)
            except Exception:
                pass  # skip if value is too long or unsupported


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
