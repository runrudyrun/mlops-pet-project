"""Model training module for the MLOps pipeline.

This module provides functionality to train ML models with MLflow tracking
and support for multiple model types.
"""

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import load_config

logger = logging.getLogger(__name__)

# Supported model types mapping
MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def get_model(model_name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    """Factory function to create model instance based on model name.

    Args:
        model_name: Name of the model type (e.g., 'random_forest').
        params: Dictionary of hyperparameters for the model.

    Returns:
        Instantiated sklearn model.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Supported models: {supported}")

    model_class = MODEL_REGISTRY[model_name]
    params = params or {}

    logger.info(f"Creating {model_name} model with params: {params}")
    return model_class(**params)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    params: dict[str, Any] | None = None,
) -> tuple[BaseEstimator, float]:
    """Train a single model with given parameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_name: Name of the model type.
        params: Dictionary of hyperparameters.

    Returns:
        Tuple of (trained model, training time in seconds).
    """
    model = get_model(model_name, params)

    logger.info(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")
    return model, training_time


def train_model_with_mlflow(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    params: dict[str, Any] | None = None,
) -> tuple[BaseEstimator, str]:
    """Train a model and log to MLflow.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_name: Name of the model type.
        params: Dictionary of hyperparameters.

    Returns:
        Tuple of (trained model, MLflow run_id).
    """
    params = params or {}

    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)

        # Train model
        model, training_time = train_model(X_train, y_train, model_name, params)

        # Log training time
        mlflow.log_metric("training_time", training_time)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Logged {model_name} to MLflow run: {run.info.run_id}")
        return model, run.info.run_id


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict[str, tuple[BaseEstimator, str]]:
    """Train all enabled models specified in config.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Configuration dictionary containing model definitions.

    Returns:
        Dictionary mapping model names to (model, run_id) tuples.
    """
    models_config = config.get("models", [])
    trained_models: dict[str, tuple[BaseEstimator, str]] = {}

    enabled_models = [m for m in models_config if m.get("enabled", True)]
    logger.info(f"Training {len(enabled_models)} enabled models...")

    for model_config in enabled_models:
        model_name = model_config["name"]
        params = model_config.get("params", {})

        try:
            model, run_id = train_model_with_mlflow(X_train, y_train, model_name, params)
            trained_models[model_name] = (model, run_id)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            raise

    logger.info(f"Successfully trained {len(trained_models)} models")
    return trained_models


def save_model(model: BaseEstimator, path: str | Path) -> None:
    """Save trained model to disk using joblib.

    Args:
        model: Trained sklearn model.
        path: Path to save the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")



def load_training_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data from processed directory.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (X_train, y_train).
    """
    processed_dir = Path(config["data"]["processed_dir"])
    train_path = processed_dir / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    # Separate features and target
    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"]

    logger.info(f"Loaded {len(X_train)} training samples with {len(X_train.columns)} features")
    return X_train, y_train


def select_best_model(
    trained_models: dict[str, tuple[BaseEstimator, str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    primary_metric: str = "accuracy",
) -> tuple[str, BaseEstimator]:
    """Select the best model based on training accuracy.

    For proper evaluation, this uses training data accuracy as a proxy.
    The actual evaluation on test data happens in the evaluate stage.

    Args:
        trained_models: Dictionary of trained models.
        X_train: Training features.
        y_train: Training labels.
        primary_metric: Metric to use for selection (uses accuracy on train).

    Returns:
        Tuple of (best_model_name, best_model).
    """
    best_score = -1.0
    best_model_name = ""
    best_model = None

    for model_name, (model, _) in trained_models.items():
        score = model.score(X_train, y_train)
        logger.info(f"{model_name} training accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model

    logger.info(f"Best model: {best_model_name} with training accuracy: {best_score:.4f}")
    return best_model_name, best_model


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
    """Main entry point for training stage.

    Loads training data, trains all enabled models, and saves the best model.

    Args:
        config_path: Path to the configuration file.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting model training...")

    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    setup_mlflow(config)

    # Load training data
    X_train, y_train = load_training_data(config)

    # Train all enabled models
    trained_models = train_all_models(X_train, y_train, config)

    # Select best model based on training performance
    primary_metric = config.get("evaluation", {}).get("primary_metric", "f1_score")
    best_model_name, best_model = select_best_model(
        trained_models, X_train, y_train, primary_metric
    )

    # Save best model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pkl"
    save_model(best_model, model_path)

    # Also save model name for reference
    model_info_path = models_dir / "model_info.txt"
    model_info_path.write_text(f"best_model: {best_model_name}\n")

    logger.info("Model training complete!")


if __name__ == "__main__":
    main()
