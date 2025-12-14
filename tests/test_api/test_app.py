import pandas as pd
import pytest
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.linear_model import LogisticRegression

import src.api.app as app_module
from src.api.app import create_app
from src.monitoring.drift import FEATURE_COLUMNS


@pytest.fixture
def trained_model_path(tmp_path):
    X = pd.DataFrame(
        {
            col: [1.0, 2.0, 3.0, 4.0, 5.0]
            for col in FEATURE_COLUMNS
        }
    )
    y = [3, 4, 5, 6, 7]

    model = LogisticRegression(max_iter=1000, multi_class="auto")
    model.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pkl"
    dump(model, model_path)

    (models_dir / "model_info.txt").write_text("best_model: logistic_regression\n")
    return model_path


class TestHealthEndpoint:
    def test_health_model_loaded_true(self, trained_model_path):
        app = create_app(model_path=str(trained_model_path))
        with TestClient(app) as client:
            res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "logistic_regression"

    def test_health_model_loaded_false_when_missing(self, tmp_path):
        missing = tmp_path / "models" / "missing.pkl"
        app = create_app(model_path=str(missing))
        with TestClient(app) as client:
            res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["model_loaded"] is False


@pytest.mark.filterwarnings("ignore:.*feature names.*")
class TestPredictEndpoints:
    def test_predict_single(self, trained_model_path, sample_wine_features):
        app = create_app(model_path=str(trained_model_path))
        with TestClient(app) as client:
            res = client.post("/predict", json=sample_wine_features)
        assert res.status_code == 200
        body = res.json()
        assert isinstance(body["predicted_quality"], int)
        if body["confidence"] is not None:
            assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_batch(self, trained_model_path, sample_wine_features):
        app = create_app(model_path=str(trained_model_path))
        payload = {"items": [sample_wine_features, sample_wine_features]}
        with TestClient(app) as client:
            res = client.post("/predict/batch", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_predict_returns_503_when_model_not_loaded(self, tmp_path, sample_wine_features):
        missing = tmp_path / "models" / "missing.pkl"
        app = create_app(model_path=str(missing))
        with TestClient(app) as client:
            res = client.post("/predict", json=sample_wine_features)
        assert res.status_code == 503

    def test_predict_validation_error(self, trained_model_path, sample_wine_features):
        app = create_app(model_path=str(trained_model_path))
        bad = {k: v for k, v in sample_wine_features.items() if k != "alcohol"}
        with TestClient(app) as client:
            res = client.post("/predict", json=bad)
        assert res.status_code == 422


class TestDriftEndpoint:
    def test_drift_endpoint_contract(self, monkeypatch, tmp_path):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        current = pd.DataFrame([{**{c: 1.0 for c in FEATURE_COLUMNS}, "quality": 5}])
        current.to_csv(processed_dir / "test.csv", index=False)

        monkeypatch.setattr(
            app_module,
            "load_config",
            lambda _path: {"data": {"processed_dir": str(processed_dir)}},
        )

        monkeypatch.setattr(
            app_module,
            "run_drift_check",
            lambda **_kwargs: {
                "drift_detected": False,
                "drifted_features": [],
                "drift_score": 0.0,
                "feature_drift_scores": {},
                "last_check": "2025-01-01T00:00:00",
            },
        )

        app = create_app(config_path=str(tmp_path / "params.yaml"), model_path=str(tmp_path / "missing.pkl"))
        with TestClient(app) as client:
            res = client.get("/monitoring/drift")

        assert res.status_code == 200
        body = res.json()
        assert body["drift_detected"] is False
        assert body["drift_score"] == 0.0
