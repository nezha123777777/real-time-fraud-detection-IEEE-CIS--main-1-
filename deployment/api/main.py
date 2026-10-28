"""FastAPI application exposing the IEEE-CIS fraud model."""

from __future__ import annotations

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from deployment.api.schemas import (
    FraudExplainResponse,
    FraudPredictionRequest,
    FraudPredictionResponse,
)
from deployment.runtime import explain_payload, load_artifacts, predict_payload

ARTIFACTS: dict[str, object] = {}
InferenceRunner = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the frozen artefacts once at startup."""
    print("Starting Fraud API -> Loading frozen artifacts...")
    ARTIFACTS.update(load_artifacts())
    print(f"Artifacts loaded. Fraud threshold: {ARTIFACTS['threshold']:.4f}")
    yield
    print("Shutting down Fraud API.")
    ARTIFACTS.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="IEEE-CIS Model Endpoint - raw transaction scoring and explanation.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


def _run_inference(
    payload: FraudPredictionRequest,
    runner: InferenceRunner,
    failure_label: str,
) -> tuple[dict[str, Any], float]:
    """Execute a runner with consistent timing and error handling."""
    start_t = time.perf_counter()

    try:
        result = runner(payload.model_dump(), ARTIFACTS)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{failure_label} failed: {exc}") from exc

    execution_time_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return result, execution_time_ms


def _as_optional_float(value: Any) -> float | None:
    """Cast optional numeric values while preserving nulls."""
    return None if value is None else float(value)


def _build_prediction_response_fields(
    result: dict[str, Any],
    execution_time_ms: float,
) -> dict[str, Any]:
    """Normalise shared prediction fields for API responses."""
    return {
        "TransactionID": str(uuid.uuid4()),
        "fraud_probability": float(result["fraud_probability"]),
        "fraud_label": int(result["fraud_label"]),
        "action": str(result["action"]),
        "execution_time_ms": execution_time_ms,
        "threshold": float(result["threshold"]),
        "review_threshold": float(result["review_threshold"]),
        "anomaly_flag": bool(result["anomaly_flag"]),
        "anomaly_score": _as_optional_float(result["anomaly_score"]),
        "model_version": str(result["model_version"]),
    }


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(payload: FraudPredictionRequest):
    result, execution_time_ms = _run_inference(payload, predict_payload, "Prediction")
    return FraudPredictionResponse(
        **_build_prediction_response_fields(result, execution_time_ms),
    )


@app.post("/predict/explain", response_model=FraudExplainResponse)
async def predict_and_explain(payload: FraudPredictionRequest):
    result, execution_time_ms = _run_inference(payload, explain_payload, "Explain")
    return FraudExplainResponse(
        **_build_prediction_response_fields(result, execution_time_ms),
        base_value=float(result["base_value"]),
        feature_count=int(result["feature_count"]),
        top_features=result["top_features"],
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": bool(ARTIFACTS.get("model")),
    }
