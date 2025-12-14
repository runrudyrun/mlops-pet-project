"""Unit tests for drift detection module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    FEATURE_COLUMNS,
    check_drift_threshold,
    compute_drift_report,
    get_drift_summary,
    load_reference_data,
    save_drift_report,
)


@pytest.fixture
def sample_reference_data():
    """Create sample reference data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "fixed acidity": np.random.normal(8.0, 1.0, n_samples),
        "volatile acidity": np.random.normal(0.5, 0.1, n_samples),
        "citric acid": np.random.normal(0.3, 0.1, n_samples),
        "residual sugar": np.random.normal(2.5, 1.0, n_samples),
        "chlorides": np.random.normal(0.08, 0.02, n_samples),
        "free sulfur dioxide": np.random.normal(15.0, 5.0, n_samples),
        "total sulfur dioxide": np.random.normal(45.0, 15.0, n_samples),
        "density": np.random.normal(0.996, 0.002, n_samples),
        "pH": np.random.normal(3.3, 0.2, n_samples),
        "sulphates": np.random.normal(0.6, 0.1, n_samples),
        "alcohol": np.random.normal(10.5, 1.0, n_samples),
        "quality": np.random.randint(3, 9, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data_no_drift(sample_reference_data):
    """Create current data similar to reference (no drift)."""
    np.random.seed(123)
    n_samples = 50

    data = {
        "fixed acidity": np.random.normal(8.0, 1.0, n_samples),
        "volatile acidity": np.random.normal(0.5, 0.1, n_samples),
        "citric acid": np.random.normal(0.3, 0.1, n_samples),
        "residual sugar": np.random.normal(2.5, 1.0, n_samples),
        "chlorides": np.random.normal(0.08, 0.02, n_samples),
        "free sulfur dioxide": np.random.normal(15.0, 5.0, n_samples),
        "total sulfur dioxide": np.random.normal(45.0, 15.0, n_samples),
        "density": np.random.normal(0.996, 0.002, n_samples),
        "pH": np.random.normal(3.3, 0.2, n_samples),
        "sulphates": np.random.normal(0.6, 0.1, n_samples),
        "alcohol": np.random.normal(10.5, 1.0, n_samples),
        "quality": np.random.randint(3, 9, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data_with_drift():
    """Create current data with significant drift."""
    np.random.seed(456)
    n_samples = 50

    # Significantly different distributions to trigger drift
    data = {
        "fixed acidity": np.random.normal(12.0, 2.0, n_samples),  # Shifted mean
        "volatile acidity": np.random.normal(0.9, 0.2, n_samples),  # Shifted mean
        "citric acid": np.random.normal(0.1, 0.05, n_samples),  # Shifted mean
        "residual sugar": np.random.normal(8.0, 3.0, n_samples),  # Shifted mean
        "chlorides": np.random.normal(0.15, 0.05, n_samples),  # Shifted mean
        "free sulfur dioxide": np.random.normal(30.0, 10.0, n_samples),  # Shifted
        "total sulfur dioxide": np.random.normal(100.0, 30.0, n_samples),  # Shifted
        "density": np.random.normal(1.002, 0.005, n_samples),  # Shifted mean
        "pH": np.random.normal(2.9, 0.3, n_samples),  # Shifted mean
        "sulphates": np.random.normal(1.0, 0.2, n_samples),  # Shifted mean
        "alcohol": np.random.normal(13.0, 1.5, n_samples),  # Shifted mean
        "quality": np.random.randint(5, 9, n_samples),
    }
    return pd.DataFrame(data)


class TestLoadReferenceData:
    """Tests for load_reference_data function."""

    def test_load_valid_data(self, sample_reference_data, tmp_path):
        """Test loading valid reference data."""
        # Save sample data to temp file
        data_path = tmp_path / "reference.csv"
        sample_reference_data.to_csv(data_path, index=False)

        # Load and verify
        loaded_data = load_reference_data(data_path)
        assert len(loaded_data) == len(sample_reference_data)
        assert list(loaded_data.columns) == list(sample_reference_data.columns)

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_reference_data(tmp_path / "nonexistent.csv")

    def test_load_empty_file(self, tmp_path):
        """Test loading empty file raises ValueError."""
        data_path = tmp_path / "empty.csv"
        # Create empty CSV with headers only
        pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(data_path, index=False)

        with pytest.raises(ValueError, match="empty"):
            load_reference_data(data_path)

    def test_load_missing_columns(self, tmp_path):
        """Test loading data with missing columns raises ValueError."""
        data_path = tmp_path / "incomplete.csv"
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            "fixed acidity": [7.0, 8.0],
            "volatile acidity": [0.5, 0.6],
        })
        incomplete_data.to_csv(data_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_reference_data(data_path)


class TestComputeDriftReport:
    """Tests for compute_drift_report function."""

    def test_compute_report_valid_data(self, sample_reference_data, sample_current_data_no_drift):
        """Test computing drift report with valid data."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)

        # Verify report is generated
        assert report is not None
        report_dict = report.as_dict()
        assert "metrics" in report_dict

    def test_compute_report_empty_reference(self, sample_current_data_no_drift):
        """Test computing report with empty reference data raises ValueError."""
        empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)

        with pytest.raises(ValueError, match="Reference data is empty"):
            compute_drift_report(empty_df, sample_current_data_no_drift)

    def test_compute_report_empty_current(self, sample_reference_data):
        """Test computing report with empty current data raises ValueError."""
        empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)

        with pytest.raises(ValueError, match="Current data is empty"):
            compute_drift_report(sample_reference_data, empty_df)

    def test_compute_report_detects_drift(
        self, sample_reference_data, sample_current_data_with_drift
    ):
        """Test that drift is detected when distributions differ significantly."""
        report = compute_drift_report(sample_reference_data, sample_current_data_with_drift)

        # Get drift summary
        summary = get_drift_summary(report)

        # With significantly different data, drift should be detected
        assert summary["drift_detected"] is True
        assert len(summary["drifted_features"]) > 0

    def test_compute_report_no_drift(
        self, sample_reference_data, sample_current_data_no_drift
    ):
        """Test that no drift is detected when distributions are similar."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)

        # Get drift summary
        summary = get_drift_summary(report)

        # With similar data, drift score should be low
        # Note: Some features might still show drift due to random sampling
        assert summary["drift_score"] < 0.5  # Less than half features drifted


class TestCheckDriftThreshold:
    """Tests for check_drift_threshold function."""

    def test_check_threshold_valid(self, sample_reference_data, sample_current_data_no_drift):
        """Test threshold checking with valid report."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)
        drift_results = check_drift_threshold(report, threshold=0.1)

        # Verify results structure
        assert isinstance(drift_results, dict)
        # Should have results for feature columns
        for col in drift_results:
            assert col in FEATURE_COLUMNS
            assert isinstance(drift_results[col], bool)

    def test_check_threshold_invalid_value(
        self, sample_reference_data, sample_current_data_no_drift
    ):
        """Test that invalid threshold raises ValueError."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)

        with pytest.raises(ValueError, match="between 0 and 1"):
            check_drift_threshold(report, threshold=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            check_drift_threshold(report, threshold=-0.1)

    def test_check_threshold_detects_drifted_features(
        self, sample_reference_data, sample_current_data_with_drift
    ):
        """Test that drifted features are correctly identified."""
        report = compute_drift_report(sample_reference_data, sample_current_data_with_drift)
        drift_results = check_drift_threshold(report, threshold=0.1)

        # With significantly different data, some features should be marked as drifted
        drifted_count = sum(drift_results.values())
        assert drifted_count > 0


class TestGetDriftSummary:
    """Tests for get_drift_summary function."""

    def test_summary_structure(self, sample_reference_data, sample_current_data_no_drift):
        """Test that summary has expected structure."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)
        summary = get_drift_summary(report)

        # Verify required keys
        assert "drift_detected" in summary
        assert "drifted_features" in summary
        assert "drift_score" in summary
        assert "feature_drift_scores" in summary
        assert "last_check" in summary

        # Verify types
        assert isinstance(summary["drift_detected"], bool)
        assert isinstance(summary["drifted_features"], list)
        assert isinstance(summary["drift_score"], (int, float))
        assert isinstance(summary["feature_drift_scores"], dict)
        assert isinstance(summary["last_check"], str)

    def test_summary_with_drift(self, sample_reference_data, sample_current_data_with_drift):
        """Test summary when drift is detected."""
        report = compute_drift_report(sample_reference_data, sample_current_data_with_drift)
        summary = get_drift_summary(report)

        assert summary["drift_detected"] is True
        assert len(summary["drifted_features"]) > 0
        assert summary["drift_score"] > 0

    def test_summary_feature_scores(self, sample_reference_data, sample_current_data_no_drift):
        """Test that feature drift scores are populated."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)
        summary = get_drift_summary(report)

        # Should have scores for features
        assert len(summary["feature_drift_scores"]) > 0

        # All scores should be numeric
        for score in summary["feature_drift_scores"].values():
            assert isinstance(score, (int, float))


class TestSaveDriftReport:
    """Tests for save_drift_report function."""

    def test_save_report_creates_file(
        self, sample_reference_data, sample_current_data_no_drift, tmp_path
    ):
        """Test that HTML report is saved correctly."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)
        report_path = tmp_path / "drift_report.html"

        save_drift_report(report, report_path)

        assert report_path.exists()
        assert report_path.stat().st_size > 0

        # Verify it's valid HTML
        content = report_path.read_text()
        assert "<html" in content.lower()

    def test_save_report_creates_parent_dirs(
        self, sample_reference_data, sample_current_data_no_drift, tmp_path
    ):
        """Test that parent directories are created if they don't exist."""
        report = compute_drift_report(sample_reference_data, sample_current_data_no_drift)
        report_path = tmp_path / "nested" / "dir" / "drift_report.html"

        save_drift_report(report, report_path)

        assert report_path.exists()
