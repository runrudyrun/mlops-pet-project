"""Unit tests for model evaluation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import (
    compute_metrics,
    evaluate_model,
    load_all_models,
    load_model,
    load_test_data,
    log_metrics_to_mlflow,
    log_model_comparison,
    save_metrics,
    select_best_model,
    setup_mlflow,
)


@pytest.fixture
def sample_test_data():
    """Create sample test data for evaluation."""
    X = pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4, 7.9, 7.3, 7.8, 6.7, 7.5],
        "volatile acidity": [0.7, 0.88, 0.76, 0.28, 0.7, 0.6, 0.65, 0.58, 0.58, 0.5],
        "citric acid": [0.0, 0.0, 0.04, 0.56, 0.0, 0.06, 0.0, 0.02, 0.08, 0.36],
        "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.9, 1.6, 1.2, 2.0, 1.8, 6.1],
        "chlorides": [0.076, 0.098, 0.092, 0.075, 0.076, 0.069, 0.065, 0.073, 0.097, 0.071],
        "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 11.0, 15.0, 15.0, 9.0, 15.0, 17.0],
        "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 34.0, 21.0, 21.0, 18.0, 65.0, 102.0],
        "density": [0.9978, 0.9968, 0.997, 0.998, 0.9978, 0.9946, 0.9946, 0.9968, 0.9959, 0.9978],
        "pH": [3.51, 3.2, 3.26, 3.16, 3.51, 3.3, 3.39, 3.36, 3.28, 3.15],
        "sulphates": [0.56, 0.68, 0.65, 0.58, 0.56, 0.46, 0.47, 0.57, 0.54, 0.65],
        "alcohol": [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 10.0, 9.5, 9.2, 9.0],
    })
    y = pd.Series([5, 5, 5, 6, 5, 6, 6, 6, 5, 5], name="quality")
    return X, y


@pytest.fixture
def trained_model(sample_test_data):
    """Create a trained model for testing."""
    X, y = sample_test_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "data": {
            "processed_dir": "data/processed",
        },
        "evaluation": {
            "metrics_path": "reports/metrics.json",
            "primary_metric": "f1_score",
        },
        "mlflow": {
            "tracking_uri": "",
            "experiment_name": "test-experiment",
        },
    }


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_compute_metrics_partial_correct(self):
        """Test metrics with partial correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.5
        assert 0 < metrics["precision"] < 1
        assert 0 < metrics["recall"] < 1
        assert 0 < metrics["f1_score"] < 1

    def test_compute_metrics_returns_all_keys(self):
        """Test that all expected metric keys are returned."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_compute_metrics_with_pandas_series(self):
        """Test metrics computation with pandas Series input."""
        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0

    def test_compute_metrics_values_are_floats(self):
        """Test that all metric values are Python floats."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = compute_metrics(y_true, y_pred)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} should be float"


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_model_returns_metrics(self, trained_model, sample_test_data):
        """Test that evaluate_model returns metrics dictionary."""
        X, y = sample_test_data
        metrics = evaluate_model(trained_model, X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_evaluate_model_returns_confusion_matrix(self, trained_model, sample_test_data):
        """Test that evaluate_model returns confusion matrix."""
        X, y = sample_test_data
        metrics = evaluate_model(trained_model, X, y)

        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_evaluate_model_metrics_in_valid_range(self, trained_model, sample_test_data):
        """Test that metrics are in valid range [0, 1]."""
        X, y = sample_test_data
        metrics = evaluate_model(trained_model, X, y)

        for key in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0 <= metrics[key] <= 1, f"{key} should be between 0 and 1"



class TestSelectBestModel:
    """Tests for select_best_model function."""

    def test_select_best_model_returns_highest_score(self):
        """Test that function returns model with highest score."""
        metrics = {
            "model_a": {"accuracy": 0.8, "f1_score": 0.75},
            "model_b": {"accuracy": 0.85, "f1_score": 0.82},
            "model_c": {"accuracy": 0.78, "f1_score": 0.70},
        }

        best = select_best_model(metrics, "f1_score")
        assert best == "model_b"

    def test_select_best_model_with_accuracy(self):
        """Test selection based on accuracy metric."""
        metrics = {
            "model_a": {"accuracy": 0.9, "f1_score": 0.75},
            "model_b": {"accuracy": 0.85, "f1_score": 0.82},
        }

        best = select_best_model(metrics, "accuracy")
        assert best == "model_a"

    def test_select_best_model_single_model(self):
        """Test with single model."""
        metrics = {
            "only_model": {"accuracy": 0.8, "f1_score": 0.75},
        }

        best = select_best_model(metrics, "f1_score")
        assert best == "only_model"

    def test_select_best_model_empty_raises_error(self):
        """Test that empty metrics dict raises ValueError."""
        with pytest.raises(ValueError, match="No models to compare"):
            select_best_model({}, "f1_score")

    def test_select_best_model_missing_metric_raises_error(self):
        """Test that missing metric raises ValueError."""
        metrics = {
            "model_a": {"accuracy": 0.8},
        }

        with pytest.raises(ValueError, match="Metric 'f1_score' not found"):
            select_best_model(metrics, "f1_score")

    def test_select_best_model_default_metric(self):
        """Test default metric is f1_score."""
        metrics = {
            "model_a": {"f1_score": 0.75},
            "model_b": {"f1_score": 0.82},
        }

        best = select_best_model(metrics)
        assert best == "model_b"


class TestSaveMetrics:
    """Tests for save_metrics function."""

    def test_save_metrics_creates_file(self):
        """Test that save_metrics creates a JSON file."""
        metrics = {"accuracy": 0.85, "f1_score": 0.82}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics, path)

            assert path.exists()

    def test_save_metrics_creates_parent_dirs(self):
        """Test that save_metrics creates parent directories."""
        metrics = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "metrics.json"
            save_metrics(metrics, path)

            assert path.exists()

    def test_save_metrics_valid_json(self):
        """Test that saved file is valid JSON."""
        metrics = {"accuracy": 0.85, "models": {"rf": {"f1": 0.8}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics, path)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded == metrics

    def test_save_metrics_handles_numpy_types(self):
        """Test that numpy types are converted to Python types."""
        metrics = {
            "accuracy": np.float64(0.85),
            "count": np.int64(100),
            "array": np.array([1, 2, 3]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics, path)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["accuracy"] == 0.85
            assert loaded["count"] == 100
            assert loaded["array"] == [1, 2, 3]

    def test_save_metrics_with_string_path(self):
        """Test save_metrics with string path."""
        metrics = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/metrics.json"
            save_metrics(metrics, path)

            assert Path(path).exists()



class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_success(self, trained_model):
        """Test successful model loading."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            joblib.dump(trained_model, model_path)

            loaded = load_model(model_path)

            assert loaded is not None
            assert hasattr(loaded, "predict")

    def test_load_model_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model("/nonexistent/path/model.pkl")

    def test_load_model_with_string_path(self, trained_model):
        """Test load_model with string path."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.pkl"
            joblib.dump(trained_model, model_path)

            loaded = load_model(model_path)

            assert loaded is not None


class TestLoadTestData:
    """Tests for load_test_data function."""

    def test_load_test_data_success(self, sample_test_data):
        """Test successful loading of test data."""
        X, y = sample_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test.csv
            test_df = X.copy()
            test_df["quality"] = y
            test_path = Path(tmpdir) / "test.csv"
            test_df.to_csv(test_path, index=False)

            config = {"data": {"processed_dir": tmpdir}}
            X_loaded, y_loaded = load_test_data(config)

            assert len(X_loaded) == len(X)
            assert len(y_loaded) == len(y)
            assert "quality" not in X_loaded.columns

    def test_load_test_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        config = {"data": {"processed_dir": "/nonexistent/path"}}

        with pytest.raises(FileNotFoundError, match="Test data not found"):
            load_test_data(config)


class TestLoadAllModels:
    """Tests for load_all_models function."""

    def test_load_all_models_with_info_file(self, trained_model):
        """Test loading models with model_info.txt."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            model_path = models_dir / "model.pkl"
            info_path = models_dir / "model_info.txt"

            joblib.dump(trained_model, model_path)
            info_path.write_text("best_model: random_forest\n")

            models = load_all_models(models_dir)

            assert "random_forest" in models
            assert models["random_forest"] is not None

    def test_load_all_models_without_info_file(self, trained_model):
        """Test loading models without model_info.txt."""
        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            model_path = models_dir / "model.pkl"

            joblib.dump(trained_model, model_path)

            models = load_all_models(models_dir)

            assert "model" in models

    def test_load_all_models_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = load_all_models(tmpdir)

            assert len(models) == 0


class TestLogMetricsToMlflow:
    """Tests for log_metrics_to_mlflow function."""

    @patch("src.models.evaluate.mlflow")
    def test_log_metrics_creates_new_run(self, mock_mlflow):
        """Test that new MLflow run is created when no run_id provided."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        metrics = {"accuracy": 0.85, "f1_score": 0.82}
        log_metrics_to_mlflow("test_model", metrics)

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_param.assert_called_with("model_name", "test_model")

    @patch("src.models.evaluate.mlflow")
    def test_log_metrics_logs_all_metrics(self, mock_mlflow):
        """Test that all metrics are logged."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        metrics = {"accuracy": 0.85, "f1_score": 0.82}
        log_metrics_to_mlflow("test_model", metrics)

        # Check that log_metric was called for each metric
        call_args = [call[0] for call in mock_mlflow.log_metric.call_args_list]
        metric_names = [arg[0] for arg in call_args]

        assert "accuracy" in metric_names
        assert "f1_score" in metric_names

    @patch("src.models.evaluate.mlflow")
    def test_log_metrics_skips_confusion_matrix(self, mock_mlflow):
        """Test that confusion_matrix is not logged as metric."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        metrics = {"accuracy": 0.85, "confusion_matrix": [[10, 2], [3, 15]]}
        log_metrics_to_mlflow("test_model", metrics)

        call_args = [call[0] for call in mock_mlflow.log_metric.call_args_list]
        metric_names = [arg[0] for arg in call_args]

        assert "confusion_matrix" not in metric_names



class TestLogModelComparison:
    """Tests for log_model_comparison function."""

    @patch("src.models.evaluate.mlflow")
    def test_log_model_comparison_logs_params(self, mock_mlflow):
        """Test that comparison parameters are logged."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        all_metrics = {
            "model_a": {"accuracy": 0.8, "f1_score": 0.75},
            "model_b": {"accuracy": 0.85, "f1_score": 0.82},
        }

        log_model_comparison(all_metrics, "model_b", "f1_score")

        # Check params were logged
        param_calls = mock_mlflow.log_param.call_args_list
        param_names = [call[0][0] for call in param_calls]

        assert "primary_metric" in param_names
        assert "best_model" in param_names
        assert "num_models_compared" in param_names

    @patch("src.models.evaluate.mlflow")
    def test_log_model_comparison_logs_best_metrics(self, mock_mlflow):
        """Test that best model metrics are logged."""
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        all_metrics = {
            "model_a": {"accuracy": 0.8, "f1_score": 0.75},
            "model_b": {"accuracy": 0.85, "f1_score": 0.82},
        }

        log_model_comparison(all_metrics, "model_b", "f1_score")

        # Check metrics were logged
        metric_calls = mock_mlflow.log_metric.call_args_list
        metric_names = [call[0][0] for call in metric_calls]

        assert "best_accuracy" in metric_names
        assert "best_f1_score" in metric_names


class TestSetupMlflow:
    """Tests for setup_mlflow function."""

    @patch("src.models.evaluate.mlflow")
    def test_setup_mlflow_sets_experiment(self, mock_mlflow, sample_config):
        """Test that MLflow experiment is set."""
        setup_mlflow(sample_config)

        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.models.evaluate.mlflow")
    def test_setup_mlflow_with_tracking_uri(self, mock_mlflow):
        """Test that tracking URI is set when provided."""
        config = {
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test",
            }
        }

        setup_mlflow(config)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    @patch("src.models.evaluate.mlflow")
    def test_setup_mlflow_without_tracking_uri(self, mock_mlflow, sample_config):
        """Test that tracking URI is not set when empty."""
        setup_mlflow(sample_config)

        mock_mlflow.set_tracking_uri.assert_not_called()

    @patch("src.models.evaluate.mlflow")
    def test_setup_mlflow_default_experiment(self, mock_mlflow):
        """Test default experiment name when not specified."""
        config = {"mlflow": {}}

        setup_mlflow(config)

        mock_mlflow.set_experiment.assert_called_once_with("wine-quality-classification")
