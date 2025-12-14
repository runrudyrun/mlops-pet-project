import pytest

from src.api.schemas import (
    BatchPredictionRequest,
    DriftResponse,
    PredictionResponse,
    WineFeatures,
)


def _validate(model_cls, data):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _dump(model, by_alias: bool = False):
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


class TestWineFeatures:
    def test_accepts_snake_case_fields(self, sample_wine_features):
        model = _validate(WineFeatures, sample_wine_features)
        assert model.fixed_acidity == sample_wine_features["fixed_acidity"]

    def test_accepts_dataset_column_aliases(self, sample_wine_features):
        alias_payload = {
            "fixed acidity": sample_wine_features["fixed_acidity"],
            "volatile acidity": sample_wine_features["volatile_acidity"],
            "citric acid": sample_wine_features["citric_acid"],
            "residual sugar": sample_wine_features["residual_sugar"],
            "chlorides": sample_wine_features["chlorides"],
            "free sulfur dioxide": sample_wine_features["free_sulfur_dioxide"],
            "total sulfur dioxide": sample_wine_features["total_sulfur_dioxide"],
            "density": sample_wine_features["density"],
            "pH": sample_wine_features["pH"],
            "sulphates": sample_wine_features["sulphates"],
            "alcohol": sample_wine_features["alcohol"],
        }

        model = _validate(WineFeatures, alias_payload)
        dumped = _dump(model, by_alias=True)
        assert "fixed acidity" in dumped
        assert dumped["fixed acidity"] == sample_wine_features["fixed_acidity"]

    def test_forbids_extra_fields(self, sample_wine_features):
        payload = {**sample_wine_features, "extra": 1}
        with pytest.raises(Exception):
            _validate(WineFeatures, payload)

    def test_rejects_negative_values(self, sample_wine_features):
        payload = {**sample_wine_features, "alcohol": -1}
        with pytest.raises(Exception):
            _validate(WineFeatures, payload)


class TestBatchPredictionRequest:
    def test_items_min_length(self, sample_wine_features):
        model = _validate(BatchPredictionRequest, {"items": [sample_wine_features]})
        assert len(model.items) == 1

        with pytest.raises(Exception):
            _validate(BatchPredictionRequest, {"items": []})


class TestPredictionResponse:
    def test_confidence_bounds(self):
        _validate(
            PredictionResponse,
            {"predicted_quality": 5, "confidence": 0.5, "model_name": "rf"},
        )

        with pytest.raises(Exception):
            _validate(
                PredictionResponse,
                {"predicted_quality": 5, "confidence": 1.5, "model_name": "rf"},
            )


class TestDriftResponse:
    def test_drift_score_bounds(self):
        _validate(
            DriftResponse,
            {
                "drift_detected": False,
                "drifted_features": [],
                "drift_score": 0.0,
                "feature_drift_scores": {},
                "last_check": "2025-01-01T00:00:00",
            },
        )

        with pytest.raises(Exception):
            _validate(
                DriftResponse,
                {
                    "drift_detected": False,
                    "drifted_features": [],
                    "drift_score": 2.0,
                    "feature_drift_scores": {},
                },
            )
