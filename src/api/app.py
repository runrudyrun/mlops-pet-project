from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config import load_config
from src.models.evaluate import load_model
from src.monitoring.drift import FEATURE_COLUMNS, run_drift_check


def _model_dump(model: Any, *, by_alias: bool = False) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)
    return model.dict(by_alias=by_alias)


def _features_to_dataframe(items: list[Any]) -> pd.DataFrame:
    rows = [_model_dump(item, by_alias=True) for item in items]
    df = pd.DataFrame(rows)
    df = df[[col for col in FEATURE_COLUMNS if col in df.columns]]
    return df


def create_app(
    *,
    config_path: str = "configs/params.yaml",
    model_path: str = "models/model.pkl",
) -> FastAPI:
    app = FastAPI(
        title="Wine Quality Prediction API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config_path = config_path
    app.state.model_path = model_path
    app.state.model = None
    app.state.model_name = None

    @app.on_event("startup")
    def _startup() -> None:
        model_file = Path(app.state.model_path)
        if not model_file.exists():
            app.state.model = None
            app.state.model_name = None
            return

        app.state.model = load_model(model_file)

        info_path = model_file.parent / "model_info.txt"
        if info_path.exists():
            for line in info_path.read_text().splitlines():
                if line.startswith("best_model:"):
                    app.state.model_name = line.split(":", 1)[1].strip() or None
                    break

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=app.state.model is not None,
            model_name=app.state.model_name,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictionRequest) -> PredictionResponse:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = _features_to_dataframe([req])
        pred = app.state.model.predict(X)[0]

        confidence: float | None = None
        if hasattr(app.state.model, "predict_proba"):
            proba = app.state.model.predict_proba(X)[0]
            confidence = float(max(proba))

        return PredictionResponse(
            predicted_quality=int(pred),
            confidence=confidence,
            model_name=app.state.model_name,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = _features_to_dataframe(req.items)
        preds = app.state.model.predict(X)

        confidences: list[float | None]
        if hasattr(app.state.model, "predict_proba"):
            probs = app.state.model.predict_proba(X)
            confidences = [float(max(row)) for row in probs]
        else:
            confidences = [None] * len(preds)

        results = [
            PredictionResponse(
                predicted_quality=int(p),
                confidence=confidences[i],
                model_name=app.state.model_name,
            )
            for i, p in enumerate(preds)
        ]

        return BatchPredictionResponse(predictions=results, count=len(results))

    @app.get("/monitoring/drift", response_model=DriftResponse)
    def drift() -> DriftResponse:
        config = load_config(app.state.config_path)
        processed_dir = Path(config["data"]["processed_dir"])

        reference_path = processed_dir / "train.csv"
        current_path = processed_dir / "test.csv"
        if not current_path.exists():
            raise HTTPException(status_code=404, detail=f"Current data not found: {current_path}")

        current_data = pd.read_csv(current_path)
        summary = run_drift_check(
            reference_path=reference_path,
            current_data=current_data,
            config_path=app.state.config_path,
            save_report=False,
        )
        return DriftResponse(**summary)

    return app


app = create_app(
    config_path=os.getenv("CONFIG_PATH", "configs/params.yaml"),
    model_path=os.getenv("MODEL_PATH", "models/model.pkl"),
)
