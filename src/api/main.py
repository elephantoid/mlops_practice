"""FastAPI service exposing the registered churn model.

Run locally with::

    uv run uvicorn src.api.main:app --reload

The model is resolved from ``MODEL_URI``. It defaults to the registry alias so a fresh
checkout works with no setup; the container image sets it to a baked-in local path so
``docker run`` needs no MLflow server reachable at boot.

The registered model was logged with ``pyfunc_predict_fn="predict_proba"``, so
``model.predict(df)`` returns ``[[p_no_churn, p_churn]]`` -- column 1 is the churn
probability. Thresholding happens here rather than in the artifact, which keeps the
decision boundary a serving concern that can change without retraining.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.features.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The default mirrors MODEL_NAME/PRODUCTION_ALIAS in src/models/train.py. Duplicated rather
# than imported: importing train.py would pull sklearn and LightGBM into the serving image.
MODEL_URI = os.environ.get("MODEL_URI", "models:/churnwatch@production")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

# Column order is pinned explicitly rather than trusting dict insertion order, so a field
# reordering in schemas.py can never silently permute the model's inputs.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Defined at module level on purpose: prometheus_client raises DuplicateTimeseries if the
# same metric name is registered twice, which is what happens if these live inside a
# handler or a factory that runs more than once.
REQUESTS = Counter("churnwatch_requests_total", "Requests handled", ["endpoint", "status"])
LATENCY = Histogram("churnwatch_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter(
    "churnwatch_predictions_total", "Prediction outcome distribution", ["outcome"]
)


def load_model(uri: str = MODEL_URI) -> tuple[Any, str]:
    """Load the serving model and resolve the version string to report.

    Returns ``(model, version)``. Defined at module level rather than inline in the
    lifespan so tests can substitute a stub without needing a populated registry.
    """
    if uri.startswith("models:/"):
        # Anchored to the repo root like src/models/train.py. A CWD-relative sqlite path
        # makes MLflow silently create a fresh empty database and then report the model as
        # not found, which reads as a registry problem rather than a path problem.
        default_tracking = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_tracking))

    model = mlflow.pyfunc.load_model(uri)

    version = os.environ.get("MODEL_VERSION", "unknown")
    if uri.startswith("models:/"):
        suffix = uri.removeprefix("models:/")
        if "@" in suffix:
            name, alias = suffix.split("@", 1)
            version = MlflowClient().get_model_version_by_alias(name, alias).version
        elif "/" in suffix:
            # models:/name/3 -- the version is already in the URI, no lookup needed.
            version = suffix.rsplit("/", 1)[1]

    return model, str(version)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup.

    A failure here is left to propagate: a service that answers /health while holding no
    model is worse than one that refuses to start.
    """
    app.state.model, app.state.model_version = load_model()
    app.state.start_time = time.time()
    logger.info("Loaded model %s (version %s)", MODEL_URI, app.state.model_version)
    yield


app = FastAPI(
    title="ChurnWatch",
    description="Telecom customer churn prediction.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_metrics(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Time every request and count it by endpoint and status.

    The recording sits in ``finally`` because ``call_next`` re-raises when a handler fails.
    Recording only on the success path would leave 500s uncounted, making error rate --
    the one metric worth alerting on -- impossible to compute, and biasing the latency
    histogram toward requests that worked.
    """
    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        # scope["route"] is only set once a route matches. Falling back to the raw path
        # would mint a new Prometheus series per URL a scanner probes, growing the
        # in-process registry without bound on a public endpoint.
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        REQUESTS.labels(endpoint=endpoint, status=status).inc()


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score one customer."""
    # by_alias renames snake_case fields to the raw training columns; reindex pins order.
    features = pd.DataFrame([payload.model_dump(by_alias=True)]).reindex(columns=FEATURE_COLUMNS)

    probability = float(request.app.state.model.predict(features)[0][1])
    outcome = "churn" if probability >= DECISION_THRESHOLD else "no_churn"
    PREDICTIONS.labels(outcome=outcome).inc()

    return PredictResponse(
        churn_probability=probability,
        prediction=outcome,
        model_version=request.app.state.model_version,
        request_id=str(uuid.uuid4()),
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness plus the version actually loaded, so a stale rollout is visible."""
    return HealthResponse(
        status="ok",
        model_version=request.app.state.model_version,
        uptime_seconds=time.time() - request.app.state.start_time,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
