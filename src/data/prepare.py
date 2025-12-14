"""Data preparation module for the MLOps pipeline.

This module provides functionality to load, preprocess, and split
the wine quality dataset for model training.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config

logger = logging.getLogger(__name__)


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load raw wine quality dataset from CSV.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if df.empty:
        raise pd.errors.EmptyDataError(f"Data file is empty: {data_path}")

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the wine quality data.

    Handles missing values by:
    - Dropping rows with missing target values
    - Filling missing numeric features with median values

    Args:
        df: Raw DataFrame to preprocess.

    Returns:
        Cleaned DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]

    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    logger.info("Preprocessing data...")

    # Drop rows with missing target values
    initial_rows = len(df)
    df = df.dropna(subset=["quality"])
    dropped_target = initial_rows - len(df)
    if dropped_target > 0:
        logger.warning(f"Dropped {dropped_target} rows with missing target values")

    # Fill missing numeric features with median
    feature_columns = [col for col in required_columns if col != "quality"]
    for col in feature_columns:
        if df[col].isna().any():
            median_val = df[col].median()
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {missing_count} missing values in '{col}' with median {median_val:.4f}")

    logger.info(f"Preprocessing complete. Final dataset: {len(df)} rows")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets.

    Args:
        df: DataFrame to split.
        test_size: Proportion of data for test set (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If test_size is not between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["quality"],
    )

    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    return train_df, test_df



def main(config_path: str = "configs/params.yaml") -> None:
    """Main entry point for data preparation stage.

    Loads raw data, preprocesses it, splits into train/test sets,
    and saves to the processed directory.

    Args:
        config_path: Path to the configuration file.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting data preparation...")

    # Load configuration
    config = load_config(config_path)
    data_config = config["data"]

    raw_path = data_config["raw_path"]
    processed_dir = Path(data_config["processed_dir"])
    test_size = data_config["test_size"]
    random_state = data_config["random_state"]

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df = load_data(raw_path)
    df = preprocess_data(df)

    # Split data
    train_df, test_df = split_data(df, test_size=test_size, random_state=random_state)

    # Save processed data
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved test data to {test_path}")
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
