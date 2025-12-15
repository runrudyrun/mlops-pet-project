"""Unit tests for model training module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.train import (
    MODEL_REGISTRY,
    get_model,
    load_training_data,
    save_model,
    select_best_model,
    setup_mlflow,
    train_all_models,
    train_model,
    train_model_with_mlflow,
)


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    # Create a simple dataset with 2 classes
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
def sample_config():
    """Create sample configuration for testing."""
    return {
        "data": {
            "processed_dir": "data/processed",
        },
        "models": [
            {
                "name": "random_forest",
                "enabled": True,
                "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
            },
            {
                "name": "logistic_regression",
                "enabled": True,
                "params": {"max_iter": 100, "random_state": 42},
            },
            {
                "name": "gradient_boosting",
                "enabled": False,
                "params": {"n_estimators": 10, "random_state": 42},
            },
        ],
        "evaluation": {
            "primary_metric": "f1_score",
        },
        "mlflow": {
            "tracking_uri": "",
            "experiment_name": "test-experiment",
        },
    }


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_contains_random_forest(self):
        """Test that registry contains RandomForest."""
        assert "random_forest" in MODEL_REGISTRY
        assert MODEL_REGISTRY["random_forest"] == RandomForestClassifier

    def test_registry_contains_gradient_boosting(self):
        """Test that registry contains GradientBoosting."""
        assert "gradient_boosting" in MODEL_REGISTRY
        assert MODEL_REGISTRY["gradient_boosting"] == GradientBoostingClassifier

    def test_registry_contains_logistic_regression(self):
        """Test that registry contains LogisticRegression."""
        assert "logistic_regression" in MODEL_REGISTRY
        assert MODEL_REGISTRY["logistic_regression"] == LogisticRegression


class TestGetModel:
    """Tests for get_model factory function."""

    def test_get_random_forest(self):
        """Test creating RandomForest model."""
        model = get_model("random_forest", {"n_estimators": 50, "random_state": 42})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.random_state == 42

    def test_get_gradient_boosting(self):
        """Test creating GradientBoosting model."""
        model = get_model("gradient_boosting", {"n_estimators": 50, "learning_rate": 0.05})
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05

    def test_get_logistic_regression(self):
        """Test creating LogisticRegression model."""
        model = get_model("logistic_regression", {"max_iter": 500})
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 500

    def test_get_model_no_params(self):
        """Test creating model with no parameters."""
        model = get_model("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_get_model_empty_params(self):
        """Test creating model with empty params dict."""
        model = get_model("random_forest", {})
        assert isinstance(model, RandomForestClassifier)

    def test_get_model_unknown_type(self):
        """Test that ValueError is raised for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model")

    def test_get_model_error_message_lists_supported(self):
        """Test that error message lists supported models."""
        with pytest.raises(ValueError) as exc_info:
            get_model("invalid")

        error_msg = str(exc_info.value)
        assert "random_forest" in error_msg
        assert "gradient_boosting" in error_msg
        assert "logistic_regression" in error_msg


class TestTrainModel:
    """Tests for train_model function."""

    def test_train_model_random_forest(self, sample_training_data):
        """Test training RandomForest model."""
        X, y = sample_training_data
        model, training_time = train_model(
            X, y, "random_forest", {"n_estimators": 10, "random_state": 42}
        )

        assert isinstance(model, RandomForestClassifier)
        assert training_time > 0
        # Model should be fitted
        assert hasattr(model, "classes_")

    def test_train_model_logistic_regression(self, sample_training_data):
        """Test training LogisticRegression model."""
        X, y = sample_training_data
        model, training_time = train_model(
            X, y, "logistic_regression", {"max_iter": 100, "random_state": 42}
        )

        assert isinstance(model, LogisticRegression)
        assert training_time > 0
        assert hasattr(model, "classes_")

    def test_train_model_can_predict(self, sample_training_data):
        """Test that trained model can make predictions."""
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 10, "random_state": 42})

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [5, 6] for p in predictions)

    def test_train_model_returns_positive_time(self, sample_training_data):
        """Test that training time is positive."""
        X, y = sample_training_data
        _, training_time = train_model(X, y, "random_forest", {"n_estimators": 5})

        assert training_time >= 0



class TestTrainModelWithMlflow:
    """Tests for train_model_with_mlflow function."""

    @patch("src.models.train.mlflow")
    def test_train_model_logs_params(self, mock_mlflow, sample_training_data):
        """Test that MLflow logs parameters."""
        X, y = sample_training_data
        params = {"n_estimators": 10, "random_state": 42}

        # Setup mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model, run_id = train_model_with_mlflow(X, y, "random_forest", params)

        # Verify params were logged
        mock_mlflow.log_params.assert_called_once_with(params)
        mock_mlflow.log_param.assert_called_with("model_type", "random_forest")

    @patch("src.models.train.mlflow")
    def test_train_model_logs_training_time(self, mock_mlflow, sample_training_data):
        """Test that MLflow logs training time metric."""
        X, y = sample_training_data

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        train_model_with_mlflow(X, y, "random_forest", {"n_estimators": 5})

        # Verify training_time metric was logged
        mock_mlflow.log_metric.assert_called()
        call_args = mock_mlflow.log_metric.call_args_list
        metric_names = [call[0][0] for call in call_args]
        assert "training_time" in metric_names

    @patch("src.models.train.mlflow")
    def test_train_model_logs_model_artifact(self, mock_mlflow, sample_training_data):
        """Test that MLflow logs model artifact."""
        X, y = sample_training_data

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        train_model_with_mlflow(X, y, "random_forest", {"n_estimators": 5})

        # Verify model was logged
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch("src.models.train.mlflow")
    def test_train_model_returns_run_id(self, mock_mlflow, sample_training_data):
        """Test that function returns MLflow run ID."""
        X, y = sample_training_data

        mock_run = MagicMock()
        mock_run.info.run_id = "expected-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        _, run_id = train_model_with_mlflow(X, y, "random_forest", {"n_estimators": 5})

        assert run_id == "expected-run-id"


class TestTrainAllModels:
    """Tests for train_all_models function."""

    @patch("src.models.train.mlflow")
    def test_train_all_enabled_models(self, mock_mlflow, sample_training_data, sample_config):
        """Test that only enabled models are trained."""
        X, y = sample_training_data

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        trained_models = train_all_models(X, y, sample_config)

        # Should have 2 models (random_forest and logistic_regression are enabled)
        assert len(trained_models) == 2
        assert "random_forest" in trained_models
        assert "logistic_regression" in trained_models
        assert "gradient_boosting" not in trained_models

    @patch("src.models.train.mlflow")
    def test_train_all_models_returns_models_and_run_ids(
        self, mock_mlflow, sample_training_data, sample_config
    ):
        """Test that function returns models and run IDs."""
        X, y = sample_training_data
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        
        trained_models = train_all_models(X, y, sample_config)
        
        for model_name, (model, run_id) in trained_models.items():
            assert model is not None
            assert hasattr(model, "predict")
            assert run_id == "test-run-id"

    @patch("src.models.train.mlflow")
    def test_train_all_models_empty_config(self, mock_mlflow, sample_training_data):
        """Test with empty models config."""
        X, y = sample_training_data
        config = {"models": []}
        
        trained_models = train_all_models(X, y, config)
        
        assert len(trained_models) == 0

    @patch("src.models.train.mlflow")
    def test_train_all_models_no_models_key(self, mock_mlflow, sample_training_data):
        """Test with missing models key in config."""
        X, y = sample_training_data
        config = {}
        
        trained_models = train_all_models(X, y, config)
        
        assert len(trained_models) == 0



class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_creates_file(self, sample_training_data):
        """Test that save_model creates a file."""
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 5, "random_state": 42})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            save_model(model, model_path)
            
            assert model_path.exists()

    def test_save_model_creates_parent_dirs(self, sample_training_data):
        """Test that save_model creates parent directories."""
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 5, "random_state": 42})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
            save_model(model, model_path)
            
            assert model_path.exists()

    def test_save_model_can_be_loaded(self, sample_training_data):
        """Test that saved model can be loaded and used."""
        import joblib
        
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 5, "random_state": 42})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            save_model(model, model_path)
            
            loaded_model = joblib.load(model_path)
            predictions = loaded_model.predict(X)
            
            assert len(predictions) == len(y)

    def test_save_model_with_string_path(self, sample_training_data):
        """Test save_model with string path."""
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 5, "random_state": 42})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.pkl"
            save_model(model, model_path)
            
            assert Path(model_path).exists()


class TestSelectBestModel:
    """Tests for select_best_model function."""

    def test_select_best_model_returns_highest_score(self, sample_training_data):
        """Test that function returns model with highest training accuracy."""
        X, y = sample_training_data
        
        # Create models with different performance
        rf_model, _ = train_model(X, y, "random_forest", {"n_estimators": 100, "random_state": 42})
        lr_model, _ = train_model(X, y, "logistic_regression", {"max_iter": 100, "random_state": 42})
        
        trained_models = {
            "random_forest": (rf_model, "run-1"),
            "logistic_regression": (lr_model, "run-2"),
        }
        
        best_name, best_model = select_best_model(trained_models, X, y)
        
        assert best_name in ["random_forest", "logistic_regression"]
        assert best_model is not None

    def test_select_best_model_single_model(self, sample_training_data):
        """Test with single model."""
        X, y = sample_training_data
        model, _ = train_model(X, y, "random_forest", {"n_estimators": 10, "random_state": 42})
        
        trained_models = {"random_forest": (model, "run-1")}
        
        best_name, best_model = select_best_model(trained_models, X, y)
        
        assert best_name == "random_forest"
        assert best_model is model


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_load_training_data_success(self, sample_training_data):
        """Test successful loading of training data."""
        X, y = sample_training_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create train.csv
            train_df = X.copy()
            train_df["quality"] = y
            train_path = Path(tmpdir) / "train.csv"
            train_df.to_csv(train_path, index=False)
            
            config = {"data": {"processed_dir": tmpdir}}
            X_loaded, y_loaded = load_training_data(config)
            
            assert len(X_loaded) == len(X)
            assert len(y_loaded) == len(y)
            assert "quality" not in X_loaded.columns

    def test_load_training_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        config = {"data": {"processed_dir": "/nonexistent/path"}}
        
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            load_training_data(config)


class TestSetupMlflow:
    """Tests for setup_mlflow function."""

    @patch("src.models.train.mlflow")
    def test_setup_mlflow_sets_experiment(self, mock_mlflow, sample_config):
        """Test that MLflow experiment is set."""
        setup_mlflow(sample_config)
        
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.models.train.mlflow")
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

    @patch("src.models.train.mlflow")
    def test_setup_mlflow_without_tracking_uri(self, mock_mlflow, sample_config):
        """Test that tracking URI is not set when empty."""
        setup_mlflow(sample_config)
        
        # Should not call set_tracking_uri with empty string
        mock_mlflow.set_tracking_uri.assert_not_called()

    @patch("src.models.train.mlflow")
    def test_setup_mlflow_default_experiment(self, mock_mlflow):
        """Test default experiment name when not specified."""
        config = {"mlflow": {}}
        
        setup_mlflow(config)
        
        mock_mlflow.set_experiment.assert_called_once_with("wine-quality-classification")
