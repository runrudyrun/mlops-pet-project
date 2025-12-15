"""Model evaluation module for the MLOps pipeline.

This module provides functionality to evaluate trained ML models,
compute metrics, and select the best model for production.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import load_config

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path) -> BaseEstimator:
    """Load trained model from disk.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded sklearn model.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, precision, recall, and f1_score.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    logger.info(f"Computed metrics: {metrics}")
    return metrics


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Evaluate model and return metrics.

    Args:
        model: Trained sklearn model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary containing metrics and confusion matrix.
    """
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    return metrics


def select_best_model(
    metrics: dict[str, dict[str, float]],
    metric_name: str = "f1_score",
) -> str:
    """Select best model based on specified metric.

    Args:
        metrics: Dictionary mapping model names to their metrics.
        metric_name: Name of the metric to use for selection.

    Returns:
        Name of the best model.

    Raises:
        ValueError: If metrics dict is empty or metric_name not found.
    """
    if not metrics:
        raise ValueError("No models to compare")

    best_model_name = None
    best_score = -1.0

    for model_name, model_metrics in metrics.items():
        if metric_name not in model_metrics:
            raise ValueError(f"Metric '{metric_name}' not found for model '{model_name}'")

        score = model_metrics[metric_name]
        logger.info(f"{model_name} {metric_name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = model_name

    logger.info(f"Best model: {best_model_name} with {metric_name}: {best_score:.4f}")
    return best_model_name


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics to save.
        path: Path to save the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Metrics saved to {path}")



def log_metrics_to_mlflow(
    model_name: str,
    metrics: dict[str, float],
    run_id: str | None = None,
) -> None:
    """Log evaluation metrics to MLflow.

    Args:
        model_name: Name of the model being evaluated.
        metrics: Dictionary of metrics to log.
        run_id: Optional MLflow run ID to log to. If None, creates new run.
    """
    if run_id:
        with mlflow.start_run(run_id=run_id):
            for metric_name, value in metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(f"eval_{metric_name}", value)
            logger.info(f"Logged metrics for {model_name} to existing run {run_id}")
    else:
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.log_param("model_name", model_name)
            for metric_name, value in metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(metric_name, value)
            logger.info(f"Logged metrics for {model_name} to new MLflow run")


def log_model_comparison(
    all_metrics: dict[str, dict[str, float]],
    best_model_name: str,
    primary_metric: str,
) -> None:
    """Log model comparison results to MLflow.

    Args:
        all_metrics: Dictionary mapping model names to their metrics.
        best_model_name: Name of the best performing model.
        primary_metric: The metric used for comparison.
    """
    with mlflow.start_run(run_name="model_comparison"):
        # Log comparison summary
        mlflow.log_param("primary_metric", primary_metric)
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("num_models_compared", len(all_metrics))

        # Log best model's metrics
        best_metrics = all_metrics[best_model_name]
        for metric_name, value in best_metrics.items():
            if metric_name != "confusion_matrix":
                mlflow.log_metric(f"best_{metric_name}", value)

        # Log all models' primary metric for comparison
        for model_name, metrics in all_metrics.items():
            mlflow.log_metric(f"{model_name}_{primary_metric}", metrics[primary_metric])

        logger.info(f"Logged model comparison to MLflow. Best: {best_model_name}")


def _make_registry_name(best_model_name: str) -> str:
    prefix = os.getenv("MLFLOW_REGISTER_PREFIX", "wine-quality")
    safe_model_name = best_model_name.replace(" ", "_").replace("/", "_")
    append_date = os.getenv("MLFLOW_REGISTER_APPEND_DATE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not append_date:
        return f"{prefix}-{safe_model_name}"

    from datetime import datetime, timezone

    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}-{safe_model_name}-{date_part}"


def register_best_model_to_registry(
    *,
    best_model_name: str,
    model: BaseEstimator,
    metrics: dict[str, Any],
) -> None:
    registry_name = _make_registry_name(best_model_name)

    with mlflow.start_run(run_name=f"register_{best_model_name}") as run:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("registered_model_name", registry_name)

        for metric_name, value in metrics.items():
            if metric_name != "confusion_matrix":
                mlflow.log_metric(metric_name, float(value))

        model_info = mlflow.sklearn.log_model(model, name="model")

        model_uri = getattr(model_info, "model_uri", None) or f"runs:/{run.info.run_id}/models/model"
        result = mlflow.register_model(model_uri=model_uri, name=registry_name)
        logger.info(
            "Registered model '%s' version %s from run %s",
            result.name,
            result.version,
            run.info.run_id,
        )


def load_test_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load test data from processed directory.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (X_test, y_test).
    """
    processed_dir = Path(config["data"]["processed_dir"])
    test_path = processed_dir / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_test = test_df.drop(columns=["quality"])
    y_test = test_df["quality"]

    logger.info(f"Loaded {len(X_test)} test samples with {len(X_test.columns)} features")
    return X_test, y_test


def load_all_models(models_dir: str | Path) -> dict[str, BaseEstimator]:
    """Load all trained models from directory.

    Args:
        models_dir: Directory containing trained models.

    Returns:
        Dictionary mapping model names to loaded models.
    """
    models_dir = Path(models_dir)
    models = {}

    # Load the main model
    model_path = models_dir / "model.pkl"
    if model_path.exists():
        model = load_model(model_path)

        # Try to get model name from info file
        info_path = models_dir / "model_info.txt"
        if info_path.exists():
            info_content = info_path.read_text()
            for line in info_content.strip().split("\n"):
                if line.startswith("best_model:"):
                    model_name = line.split(":")[1].strip()
                    models[model_name] = model
                    break
        else:
            models["model"] = model

    logger.info(f"Loaded {len(models)} models from {models_dir}")
    return models


def setup_mlflow(config: dict) -> None:
    """Configure MLflow tracking.

    Args:
        config: Configuration dictionary.
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "")
    experiment_name = mlflow_config.get("experiment_name", "wine-quality-classification")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    else:
        logger.info("Using local MLflow tracking")

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")



def main(config_path: str = "configs/params.yaml") -> None:
    """Main entry point for evaluation stage.

    Loads test data and trained models, evaluates all models,
    selects the best one, and saves metrics.

    Args:
        config_path: Path to the configuration file.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting model evaluation...")

    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    setup_mlflow(config)

    # Load test data
    X_test, y_test = load_test_data(config)

    # Load trained models
    models_dir = Path("models")
    models = load_all_models(models_dir)

    if not models:
        raise RuntimeError("No trained models found in models/ directory")

    # Evaluate all models
    all_metrics: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[model_name] = metrics

        # Log metrics to MLflow
        log_metrics_to_mlflow(model_name, metrics)

    # Select best model
    primary_metric = config.get("evaluation", {}).get("primary_metric", "f1_score")
    best_model_name = select_best_model(all_metrics, primary_metric)

    # Log model comparison to MLflow
    log_model_comparison(all_metrics, best_model_name, primary_metric)

    register_best_model_to_registry(
        best_model_name=best_model_name,
        model=models[best_model_name],
        metrics=all_metrics[best_model_name],
    )

    # Prepare output metrics
    output_metrics = {
        "best_model": best_model_name,
        "primary_metric": primary_metric,
        "models": all_metrics,
    }

    # Save metrics to file
    metrics_path = config.get("evaluation", {}).get("metrics_path", "reports/metrics.json")
    save_metrics(output_metrics, metrics_path)

    # Save confusion matrix as CSV for DVC plots
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Create confusion matrix CSV for the best model
    best_cm = all_metrics[best_model_name].get("confusion_matrix", [])
    if best_cm:
        cm_data = []
        for i, row in enumerate(best_cm):
            for j, value in enumerate(row):
                cm_data.append({"actual": i, "predicted": j, "count": value})

        cm_df = pd.DataFrame(cm_data)
        cm_path = reports_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=False)
        logger.info(f"Confusion matrix saved to {cm_path}")

    logger.info("Model evaluation complete!")
    logger.info(f"Best model: {best_model_name} ({primary_metric}: {all_metrics[best_model_name][primary_metric]:.4f})")


if __name__ == "__main__":
    main()
