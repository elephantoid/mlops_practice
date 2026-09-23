"""FastAPI service exposing the registered risk models.

Run locally with::

    uv run uvicorn src.api.main:app --reload

One model per track, resolved from ``MODEL_URI_<TRACK>`` or the registry default. The
container image sets a baked-in local path so ``docker run`` needs no MLflow server
reachable at boot.

The registered models are logged with ``pyfunc_predict_fn="predict_proba"``, so
``model.predict(df)`` returns ``[[p_negative, p_positive]]`` -- column 1 is the risk
probability. Thresholding happens here rather than in the artifact, which keeps the
decision boundary a serving concern that can change without retraining. That matters more
here than it did for churn: the operating points come from a cost-asymmetry optimisation
that will be re-run as costs change, and baking them into the artifact would mean
retraining to move a threshold.

This module imports from ``src.features.specs`` and never from ``src.data``. That keeps
pandera and the acquisition stack out of the serving image, and ``tests/test_tracks.py``
asserts it.
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
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.prediction_log import log_prediction
from src.api.schemas import CreditPredictRequest, HealthResponse, RiskResponse
from src.features.specs import get_feature_spec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The defaults mirror the track-derived names in src/data/tracks.py. Duplicated rather than
# imported: importing that module would pull pandera into the serving image, and importing
# train.py would pull sklearn and LightGBM.
MODEL_URI = os.environ.get("MODEL_URI", "")

# Which tracks this deployment serves. W1 registers credit only; fraud joins in W2, and the
# switch becomes load-bearing then -- it is here now because the deploy-shape decision
# (one track or both) is deferred to a W2 image measurement, and deferring it is only free
# if the entrypoint already reads a list rather than assuming a single model.
ENABLED_TRACKS = tuple(
    t.strip() for t in os.environ.get("ENABLED_TRACKS", "credit").split(",") if t.strip()
)

# (review_at, decline_at) per track. Placeholders: the real operating points come from the
# W2 cost-asymmetry optimisation over a swept FPR grid. A 0.5 split with a narrow band is
# stated as a placeholder rather than presented as a tuned boundary -- at a sub-1% positive
# rate a 0.5 cut is close to meaningless, which is exactly why W2 computes it properly.
DECISION_BANDS: dict[str, tuple[float, float]] = {
    "credit": (0.40, 0.60),
    "fraud": (0.40, 0.60),
}

# Column order is pinned explicitly rather than trusting dict insertion order, so a field
# reordering in schemas.py can never silently permute the model's inputs.
FEATURE_COLUMNS = list(get_feature_spec("credit").feature_columns)

# Defined at module level on purpose: prometheus_client raises DuplicateTimeseries if the
# same metric name is registered twice, which is what happens if these live inside a
# handler or a factory that runs more than once.
REQUESTS = Counter("riskwatch_requests_total", "Requests handled", ["endpoint", "status"])

# prometheus_client's default buckets start at 5ms, so every observation from this service
# would land in the first one and histogram_quantile would report ~5ms for every percentile
# regardless of real latency. Measured /predict latency sits around 3-6ms, so the buckets
# are dense through 1-10ms and coarsen above it: a quantile can only ever be as precise as
# the bucket it falls in, and there is no value in resolving the difference between 1s and
# 2s for a service that should never approach either.
LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.0075,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)
LATENCY = Histogram(
    "riskwatch_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "riskwatch_predictions_total",
    "Prediction decision distribution",
    ["track", "decision"],
)


def model_uri_for(track: str) -> str:
    """Registry URI for one track's production model.

    Per-track override first (``MODEL_URI_CREDIT``), then the shared ``MODEL_URI`` for
    single-track deployments, then the registry default. Two registered models means two
    URIs; a single ``MODEL_URI`` could only ever point at one of them.
    """
    specific = os.environ.get(f"MODEL_URI_{track.upper()}")
    if specific:
        return specific
    if len(ENABLED_TRACKS) == 1 and MODEL_URI:
        return MODEL_URI
    return f"models:/riskwatch_{track}@production"


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
    """Load one model per enabled track.

    Per-track rather than all-or-nothing. The previous policy failed the whole process if
    the single model would not load, which was right for one model and is wrong for two:
    the tracks are independent by design, so a fraud-side registry problem should not take
    the credit endpoint down -- and on Cloud Run it would fail the entire revision, taking
    the public URL with it.

    A track that fails to load is recorded and its endpoint returns 503; ``/health`` then
    reports ``degraded``. **Every** track failing is still fatal: a service holding no
    model at all has nothing to offer and should not answer.
    """
    app.state.models = {}
    app.state.model_versions = {}
    failures: dict[str, str] = {}

    for track in ENABLED_TRACKS:
        uri = model_uri_for(track)
        try:
            model, version = load_model(uri)
        except Exception as exc:  # noqa: BLE001 - recorded per track, reported by /health
            failures[track] = f"{type(exc).__name__}: {exc}"
            logger.error("Track %r failed to load from %s: %s", track, uri, exc)
            continue
        app.state.models[track] = model
        app.state.model_versions[track] = version
        logger.info("Loaded %r model %s (version %s)", track, uri, version)

    if not app.state.models:
        raise RuntimeError(f"no track loaded a model; failures: {failures}")

    app.state.load_failures = failures
    app.state.start_time = time.time()
    yield


app = FastAPI(
    title="RiskWatch",
    description="Credit-risk and fraud-detection scoring. One platform, two risk domains.",
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


def decide(probability: float, track: str) -> tuple[str, float]:
    """Map a probability onto approve / review / decline.

    Two thresholds, not one. Below ``review`` is an approve, at or above ``decline`` is a
    decline, and the band between them is routed to a human -- which is what the field
    exists for and what a binary decision cannot express.

    The W1 values are placeholders and are marked as such: the real operating points come
    from the cost-asymmetry optimisation in W2, computed against a swept FPR grid rather
    than chosen. Until then this is a 0.5 split with a narrow band around it, and no claim
    is made that it is optimal.
    """
    review_at, decline_at = DECISION_BANDS[track]
    if probability >= decline_at:
        return "decline", decline_at
    if probability >= review_at:
        return "review", decline_at
    return "approve", decline_at


def _model_for(request: Request, track: str) -> tuple[Any, str]:
    """Fetch a loaded track model, or 503 naming the track that is down."""
    model = request.app.state.models.get(track)
    if model is None:
        reason = request.app.state.load_failures.get(track, "not enabled")
        raise HTTPException(status_code=503, detail=f"track {track!r} is unavailable: {reason}")
    return model, request.app.state.model_versions[track]


@app.post("/predict/credit", response_model=RiskResponse)
async def predict_credit(payload: CreditPredictRequest, request: Request) -> RiskResponse:
    """Score one credit application."""
    model, model_version = _model_for(request, "credit")

    # by_alias renames snake_case fields to the raw training columns; reindex pins order.
    feature_values = payload.model_dump(by_alias=True)
    features = pd.DataFrame([feature_values]).reindex(columns=FEATURE_COLUMNS)

    probability = float(model.predict(features)[0][1])
    decision, threshold = decide(probability, "credit")
    PREDICTIONS.labels(track="credit", decision=decision).inc()

    request_id = str(uuid.uuid4())

    # Logs the same dict that built the DataFrame, not a re-derived copy: a second
    # model_dump could drift from what the model actually scored, and drift monitoring
    # built on a slightly different record would be quietly measuring the wrong thing.
    log_prediction(
        request_id=request_id,
        model_version=model_version,
        features=feature_values,
        risk_probability=probability,
        decision=decision,
        track="credit",
    )

    return RiskResponse(
        risk_probability=probability,
        decision=decision,
        # Empty until SHAP lands in W2. Asserted empty by a test so it cannot read as a
        # delivered capability in the meantime.
        reason_codes=[],
        threshold=threshold,
        track="credit",
        model_version=model_version,
        request_id=request_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness plus which tracks are actually serving, so a half-up service is visible."""
    loaded = dict(request.app.state.model_versions)
    status = "ok" if len(loaded) == len(ENABLED_TRACKS) else "degraded"
    return HealthResponse(
        status=status,
        models=loaded,
        uptime_seconds=time.time() - request.app.state.start_time,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
