"""
Evaluation Module
=================
Threshold optimization, confusion matrix generation,
and feature importance logging — all MLflow-aware.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Threshold Optimization                                                      #
# --------------------------------------------------------------------------- #


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cfg: "DictConfig",
) -> dict:
    """
    Find the F-beta–optimal threshold AND apply the business override.

    Returns a dict with:
        math_threshold  — algorithmically optimal
        business_threshold — from config (safety override)
        recall, precision — at the business threshold
    """
    th_cfg = cfg.threshold
    beta = th_cfg.beta
    thresholds = np.linspace(
        th_cfg.search_range.min,
        th_cfg.search_range.max,
        th_cfg.search_range.steps,
    )

    scores_fbeta = []
    recalls = []
    precisions = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        scores_fbeta.append(fbeta_score(y_true, preds, beta=beta))
        recalls.append(recall_score(y_true, preds))
        precisions.append(precision_score(y_true, preds))

    best_idx = int(np.argmax(scores_fbeta))
    math_threshold = float(thresholds[best_idx])
    business_threshold = float(th_cfg.value)

    # Compute metrics at business threshold
    biz_preds = (y_proba >= business_threshold).astype(int)
    biz_recall = recall_score(y_true, biz_preds)
    biz_precision = precision_score(y_true, biz_preds)

    logger.info(
        "Math-optimal threshold (F%s): %.3f  |  Business override: %.3f",
        beta,
        math_threshold,
        business_threshold,
    )
    logger.info(
        "At business threshold → Recall: %.1f%% | Precision: %.1f%%",
        biz_recall * 100,
        biz_precision * 100,
    )

    # ── Plot threshold sensitivity curve ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, scores_fbeta, label=f"F{beta}-Score", color="black", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall (Safety)", color="green", linestyle="--")
    ax.plot(thresholds, precisions, label="Precision (Efficiency)", color="blue", linestyle=":")
    ax.axvline(math_threshold, color="orange", linestyle="-", alpha=0.7, label=f"Math Optimal ({math_threshold:.2f})")
    ax.axvline(business_threshold, color="red", linestyle="-", label=f"Business Override ({business_threshold:.2f})")
    ax.set_title("Threshold Sensitivity Analysis")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(f.name, "plots")
        os.unlink(f.name)
    plt.close(fig)

    return {
        "math_threshold": math_threshold,
        "business_threshold": business_threshold,
        "recall": biz_recall,
        "precision": biz_precision,
    }


# --------------------------------------------------------------------------- #
#  Confusion Matrix                                                            #
# --------------------------------------------------------------------------- #


def log_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> None:
    """Plot and log the confusion matrix at the given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(
        "Confusion Matrix (threshold=%.2f): TP=%s FP=%s FN=%s TN=%s",
        threshold,
        f"{tp:,}",
        f"{fp:,}",
        f"{fn:,}",
        f"{tn:,}",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(f.name, "plots")
        os.unlink(f.name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Feature Importance                                                          #
# --------------------------------------------------------------------------- #


def log_feature_importance(model, feature_names: list[str]) -> None:
    """Plot and log CatBoost feature importance."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="coral",
    )
    ax.set_title("Feature Importance (CatBoost Native)")
    ax.set_xlabel("Importance (%)")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(f.name, "plots")
        os.unlink(f.name)
    plt.close(fig)

    logger.info("Feature Importance:")
    for i in sorted_idx[::-1]:
        logger.info("  %s: %.2f%%", feature_names[i], importance[i])
