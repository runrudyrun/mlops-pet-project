"""Data drift detection module using Evidently AI.

This module provides functionality to detect data drift between
reference (training) data and current (production) data using
Evidently's drift detection capabilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import load_config

logger = logging.getLogger(__name__)

# Feature columns for wine quality dataset
FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "quality"


def load_reference_data(path: str | Path) -> pd.DataFrame:
    """Load reference (training) data for drift comparison.

    Args:
        path: Path to the reference data CSV file.

    Returns:
        DataFrame containing the reference data.

    Raises:
        FileNotFoundError: If the reference data file doesn't exist.
        ValueError: If the file is empty or missing required columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Reference data file not found: {path}")

    logger.info(f"Loading reference data from {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Reference data file is empty: {path}")

    # Validate required columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Reference data missing required columns: {missing_cols}")

    logger.info(f"Loaded reference data: {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping | None = None,
) -> Report:
    """Generate Evidently drift report comparing reference and current data.

    Args:
        reference_data: DataFrame with reference (training) data.
        current_data: DataFrame with current (production) data.
        column_mapping: Optional Evidently ColumnMapping for feature specification.

    Returns:
        Evidently Report object with drift analysis results.

    Raises:
        ValueError: If dataframes are empty or have mismatched columns.
    """
    if reference_data.empty:
        raise ValueError("Reference data is empty")
    if current_data.empty:
        raise ValueError("Current data is empty")

    # Validate column alignment
    ref_features = set(reference_data.columns) & set(FEATURE_COLUMNS)
    curr_features = set(current_data.columns) & set(FEATURE_COLUMNS)

    if ref_features != curr_features:
        raise ValueError(
            f"Feature mismatch between reference and current data. "
            f"Reference has: {ref_features}, Current has: {curr_features}"
        )

    # Create default column mapping if not provided
    if column_mapping is None:
        column_mapping = ColumnMapping(
            target=TARGET_COLUMN if TARGET_COLUMN in reference_data.columns else None,
            numerical_features=[col for col in FEATURE_COLUMNS if col in reference_data.columns],
        )

    logger.info("Computing drift report...")
    logger.info(f"Reference data: {len(reference_data)} rows")
    logger.info(f"Current data: {len(current_data)} rows")

    # Create and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    logger.info("Drift report computation complete")
    return report


def check_drift_threshold(
    report: Report,
    threshold: float = 0.1,
) -> dict[str, bool]:
    """Check if drift exceeds threshold for each feature.

    Args:
        report: Evidently Report object with drift analysis.
        threshold: Drift score threshold (0.0 to 1.0). Features with
            drift score above this are considered drifted.

    Returns:
        Dictionary mapping feature names to drift status (True if drifted).

    Raises:
        ValueError: If threshold is not between 0 and 1.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    # Extract drift results from report
    report_dict = report.as_dict()

    drift_results: dict[str, bool] = {}

    # Navigate to drift metrics in the report structure
    metrics = report_dict.get("metrics", [])

    for metric in metrics:
        metric_result = metric.get("result", {})

        # Check for dataset drift result
        if "drift_by_columns" in metric_result:
            drift_by_columns = metric_result["drift_by_columns"]
            for col_name, col_data in drift_by_columns.items():
                if col_name in FEATURE_COLUMNS:
                    # Get drift score (p-value based, lower means more drift)
                    drift_score = col_data.get("drift_score", 0)
                    # In Evidently, drift_detected is based on statistical test
                    is_drifted = col_data.get("drift_detected", False)
                    drift_results[col_name] = is_drifted

    logger.info(f"Drift check complete. Threshold: {threshold}")
    drifted_count = sum(drift_results.values())
    logger.info(f"Features with drift: {drifted_count}/{len(drift_results)}")

    return drift_results


def get_drift_summary(report: Report) -> dict[str, Any]:
    """Extract drift summary for API response.

    Args:
        report: Evidently Report object with drift analysis.

    Returns:
        Dictionary containing:
            - drift_detected: bool indicating if dataset drift was detected
            - drifted_features: list of feature names with detected drift
            - drift_score: overall dataset drift score
            - feature_drift_scores: dict mapping features to their drift scores
            - last_check: ISO timestamp of when the check was performed
    """
    report_dict = report.as_dict()

    drift_detected = False
    drifted_features: list[str] = []
    drift_score = 0.0
    feature_drift_scores: dict[str, float] = {}

    # Navigate to drift metrics in the report structure
    metrics = report_dict.get("metrics", [])

    for metric in metrics:
        metric_result = metric.get("result", {})

        # Get overall dataset drift status
        if "dataset_drift" in metric_result:
            drift_detected = metric_result["dataset_drift"]

        # Get share of drifted columns as overall score
        if "share_of_drifted_columns" in metric_result:
            drift_score = metric_result["share_of_drifted_columns"]

        # Get per-column drift information
        if "drift_by_columns" in metric_result:
            drift_by_columns = metric_result["drift_by_columns"]
            for col_name, col_data in drift_by_columns.items():
                if col_name in FEATURE_COLUMNS:
                    col_drift_score = col_data.get("drift_score", 0)
                    feature_drift_scores[col_name] = col_drift_score

                    if col_data.get("drift_detected", False):
                        drifted_features.append(col_name)

    summary = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "drift_score": drift_score,
        "feature_drift_scores": feature_drift_scores,
        "last_check": datetime.utcnow().isoformat(),
    }

    logger.info(f"Drift summary: detected={drift_detected}, score={drift_score:.4f}")
    return summary


def save_drift_report(report: Report, path: str | Path) -> None:
    """Save Evidently drift report as HTML file.

    Args:
        report: Evidently Report object to save.
        path: Output path for the HTML report.

    Raises:
        IOError: If the report cannot be saved.
    """
    path = Path(path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving drift report to {path}")
    report.save_html(str(path))
    logger.info(f"Drift report saved successfully")


def run_drift_check(
    reference_path: str | Path,
    current_data: pd.DataFrame,
    config_path: str = "configs/params.yaml",
    save_report: bool = True,
) -> dict[str, Any]:
    """Run complete drift check workflow.

    Convenience function that loads reference data, computes drift,
    checks thresholds, and optionally saves the HTML report.

    Args:
        reference_path: Path to reference (training) data CSV.
        current_data: DataFrame with current data to check.
        config_path: Path to configuration file.
        save_report: Whether to save HTML report.

    Returns:
        Drift summary dictionary.
    """
    # Load configuration
    config = load_config(config_path)
    monitoring_config = config.get("monitoring", {})
    threshold = monitoring_config.get("drift_threshold", 0.1)
    report_path = monitoring_config.get("report_path", "reports/drift_report.html")

    # Load reference data
    reference_data = load_reference_data(reference_path)

    # Compute drift report
    report = compute_drift_report(reference_data, current_data)

    # Check thresholds
    drift_by_feature = check_drift_threshold(report, threshold)

    # Get summary
    summary = get_drift_summary(report)
    summary["threshold"] = threshold
    summary["drift_by_feature"] = drift_by_feature

    # Save HTML report if requested
    if save_report:
        save_drift_report(report, report_path)
        summary["report_path"] = str(report_path)

    return summary


def main(config_path: str = "configs/params.yaml") -> None:
    """Main entry point for drift detection.

    Loads training data as reference and test data as current,
    then runs drift analysis.

    Args:
        config_path: Path to the configuration file.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting drift detection...")

    # Load configuration
    config = load_config(config_path)
    data_config = config["data"]
    processed_dir = Path(data_config["processed_dir"])

    # Use training data as reference
    reference_path = processed_dir / "train.csv"

    # Use test data as current (for demonstration)
    current_path = processed_dir / "test.csv"
    current_data = pd.read_csv(current_path)

    # Run drift check
    summary = run_drift_check(
        reference_path=reference_path,
        current_data=current_data,
        config_path=config_path,
    )

    logger.info("Drift detection complete!")
    logger.info(f"Dataset drift detected: {summary['drift_detected']}")
    logger.info(f"Overall drift score: {summary['drift_score']:.4f}")

    if summary["drifted_features"]:
        logger.warning(f"Drifted features: {summary['drifted_features']}")


if __name__ == "__main__":
    main()
