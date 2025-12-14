from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
    from pydantic import field_validator

    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    ConfigDict = Any  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    _PYDANTIC_V2 = False

    from pydantic import validator


class _APIModel(BaseModel):
    if _PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True, extra="forbid")
    else:

        class Config:
            allow_population_by_field_name = True
            extra = "forbid"


class WineFeatures(_APIModel):
    fixed_acidity: float = Field(..., alias="fixed acidity", ge=0)
    volatile_acidity: float = Field(..., alias="volatile acidity", ge=0)
    citric_acid: float = Field(..., alias="citric acid", ge=0)
    residual_sugar: float = Field(..., alias="residual sugar", ge=0)
    chlorides: float = Field(..., alias="chlorides", ge=0)
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", ge=0)
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", ge=0)
    density: float = Field(..., alias="density", ge=0)
    pH: float = Field(..., alias="pH", ge=0)
    sulphates: float = Field(..., alias="sulphates", ge=0)
    alcohol: float = Field(..., alias="alcohol", ge=0)


class PredictionRequest(WineFeatures):
    pass


class PredictionResponse(_APIModel):
    predicted_quality: int
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_name: str | None = None


class BatchPredictionRequest(_APIModel):
    items: list[WineFeatures]

    if _PYDANTIC_V2:

        @field_validator("items")
        @classmethod
        def _validate_items_non_empty(cls, v: list[WineFeatures]) -> list[WineFeatures]:
            if len(v) < 1:
                raise ValueError("items must contain at least 1 element")
            return v

    else:

        @validator("items")
        def _validate_items_non_empty(cls, v: list[WineFeatures]) -> list[WineFeatures]:
            if len(v) < 1:
                raise ValueError("items must contain at least 1 element")
            return v


class BatchPredictionResponse(_APIModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(_APIModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class DriftResponse(_APIModel):
    drift_detected: bool
    drifted_features: list[str] = Field(default_factory=list)
    drift_score: float = Field(..., ge=0, le=1)
    feature_drift_scores: dict[str, float] = Field(default_factory=dict)
    last_check: str | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)
    drift_by_feature: dict[str, bool] = Field(default_factory=dict)
    report_path: str | None = None
