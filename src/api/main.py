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
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from src.api.prediction_log import log_prediction
from src.api.schemas import (
    CreditPredictRequest,
    FraudPredictRequest,
    HealthResponse,
    RiskResponse,
)
from src.features.specs import FEATURE_SPECS
from src.models.costs import COST_MATRICES, decision_bands

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

# (review_at, decline_at) per track: the cost-minimising boundaries of each track's review
# band, computed rather than copied.
#
# ``src.models.costs`` is safe to import here and is the only module in ``src/models/`` that
# is -- it imports **nothing**, not even numpy, because the closed form is arithmetic on three
# floats. An earlier draft of this held hand-copied literals with a test asserting they still
# matched the cost matrices, on the grounds that importing the optimiser would pull sklearn
# into an image already fighting a 0.5 GB Artifact Registry budget. True of the optimiser,
# false of the closed form, and the distinction is the whole reason the split exists.
#
# The bands come from a **review budget**, not a guessed review price: each track states the
# fraction of traffic a human can absorb and the optimiser reports the shadow price that implies.
# credit refers 15% and lands at (0.0645, 0.0963); fraud refers 0.5% and lands at
# (0.0886, 0.1143). Both straddle their two-action Bayes threshold closely, which is the shape a
# review band is supposed to have, and all three outcomes are populated on real holdouts.
#
# The previous commit priced a review instead of budgeting it, at 0.1% of the loan, and produced
# [0.0014, 0.98] -- which sent **100.000%** of a 61,503-row credit holdout to review, leaving both
# ``approve`` and ``decline`` unreachable and this three-valued field a constant function. Not a
# bad guess so much as the known boundary case of pricing abstention: a review that cheap beats
# deciding for everybody. ``docs/debt-ledger.md`` 2-E carries it.
DECISION_BANDS: dict[str, tuple[float, float]] = {
    track: decision_bands(track) for track in COST_MATRICES
}

# Per track, and column order is pinned rather than trusting dict insertion order, so a field
# reordering in schemas.py can never silently permute a model's inputs.
#
# Keyed by track rather than being one shared list, for the reason this whole step is about: one
# name holding two tracks' worth of meaning is the defect. A single ``FEATURE_COLUMNS`` would
# have reindexed a fraud request onto credit's 26 columns, filling all 29 of its own with NaN --
# and ``reindex`` fills rather than raises, so it would have scored and returned a probability.
#
# Built from ``FEATURE_SPECS`` rather than ``ENABLED_TRACKS``: the contract of a track does not
# depend on whether this deployment happens to serve it, and deriving it from an env var would
# make the request models' validity configuration-dependent.
FEATURE_COLUMNS: Mapping[str, list[str]] = MappingProxyType(
    {track: list(spec.feature_columns) for track, spec in FEATURE_SPECS.items()}
)

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


class BakedModelMisconfigured(RuntimeError):
    """A baked deployment enabled a track it carries no artifact for.

    Its own exception type because the lifespan must treat it as a *per-track* failure. A
    bare ``RuntimeError`` would read as the all-tracks-failed one raised below it.
    """


def model_uri_for(track: str) -> str:
    """Resolve one track's model URI.

    Per-track override first (``MODEL_URI_CREDIT``), then the shared ``MODEL_URI`` for
    single-track deployments, then the registry default. Two registered models means two
    URIs; a single ``MODEL_URI`` could only ever point at one of them.

    **A baked deployment never falls back to the registry.** The condition this replaces was
    ``if len(ENABLED_TRACKS) == 1 and MODEL_URI``, which meant enabling a second track stopped
    the baked path being used **for every track** -- so an image carrying credit's artifact sent
    credit to ``models:/riskwatch_credit@production`` too, a registry it cannot reach. Measured
    in the Step 13 rehearsal: both tracks failed and the process took **204.5 s** to exit 3.

    Two things that are *not* wrong with it, recorded because the first draft of this docstring
    claimed both. MLflow's retry is **bounded** -- ``MAX_RETRY_COUNT`` is 10, sleeps are
    ``0.1 * (2**n - 1)``, so 101.3 s per engine creation and one cycle per track. And
    ``load_model`` **does** raise at the end of it, so the lifespan's per-track isolation runs
    exactly as designed; every track having failed is why the process still exits.

    What the guard buys is therefore 204.5 s -> 1.0 s and a message naming the misconfiguration,
    not the recovery of a protection that was never lost. The state being refused is one where
    the operator asked for two tracks and silently got zero.
    """
    specific = os.environ.get(f"MODEL_URI_{track.upper()}")
    if specific:
        return specific
    if MODEL_URI:
        if len(ENABLED_TRACKS) == 1:
            return MODEL_URI
        if not MODEL_URI.startswith("models:/"):
            raise BakedModelMisconfigured(
                f"track {track!r} is enabled but this deployment has no artifact for it: "
                f"MODEL_URI={MODEL_URI!r} is a local path and can only carry one model, and "
                f"MODEL_URI_{track.upper()} is unset. Set MODEL_URI_{track.upper()} to a "
                f"second baked path, or drop {track!r} from ENABLED_TRACKS."
            )
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
    if version == "unknown" and not uri.startswith("models:/"):
        # A baked local path. export.py writes MODEL_VERSION alongside the artifact for
        # exactly this case: the container has no registry to ask, so without reading the
        # file every prediction is stamped "unknown" -- and W3's deploy acceptance asks for
        # a REAL version from the public URL. Reported per prediction and in /health, so an
        # unknown version means nobody can say which model answered.
        stamp = Path(uri) / "MODEL_VERSION"
        if stamp.is_file():
            version = stamp.read_text().strip() or "unknown"
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
        # Resolution is inside the try because it can now fail: a baked image asked to serve a
        # track it carries no artifact for raises rather than handing back an unreachable
        # registry URI. Outside the try that raise would kill the whole process, which is the
        # all-or-nothing policy the per-track loop below exists to replace.
        uri = "<unresolved>"
        try:
            uri = model_uri_for(track)
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

    The boundaries come from the track's review budget -- see ``DECISION_BANDS`` above and
    ``src/models/costs.py`` for the derivation. They are no longer a 0.5 split, and the width is
    a result rather than a preference: credit's [0.0645, 0.0963] is what a 15% referral capacity
    buys against a 14:1 cost of being wrong.

    ``threshold`` in the response is ``decline_at`` for every outcome, including approvals.
    That is deliberate: it reports the cut the decision was taken *against*, so an approved
    applicant's record says how far from a decline they were. Returning ``review_at`` on an
    approval would report a boundary the applicant did not cross.
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


def _score(track: str, payload: BaseModel, request: Request) -> RiskResponse:
    """Score one request against ``track``'s model and record it.

    Shared by both endpoints rather than copied into each. The copy is what would rot: the
    metrics increment, the log write and the response construction have to agree about the same
    probability and the same request id, and two tracks maintaining that agreement separately
    means one of them eventually stops. The endpoints below keep only what genuinely differs --
    the payload type FastAPI validates against, and the track name.

    Not a route handler itself, so it is deliberately synchronous: there is no await in here, and
    declaring it async would only add a coroutine frame per request.
    """
    model, model_version = _model_for(request, track)

    # by_alias renames snake_case fields to the raw training columns; reindex pins order.
    feature_values = payload.model_dump(by_alias=True)
    features = pd.DataFrame([feature_values]).reindex(columns=FEATURE_COLUMNS[track])

    probability = float(model.predict(features)[0][1])
    decision, threshold = decide(probability, track)
    PREDICTIONS.labels(track=track, decision=decision).inc()

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
        track=track,
    )

    return RiskResponse(
        risk_probability=probability,
        decision=decision,
        # Empty until SHAP lands in W2. Asserted empty by a test so it cannot read as a
        # delivered capability in the meantime.
        reason_codes=[],
        threshold=threshold,
        track=track,
        model_version=model_version,
        request_id=request_id,
    )


@app.post("/predict/credit", response_model=RiskResponse)
async def predict_credit(payload: CreditPredictRequest, request: Request) -> RiskResponse:
    """Score one credit application."""
    return _score("credit", payload, request)


@app.post("/predict/fraud", response_model=RiskResponse)
async def predict_fraud(payload: FraudPredictRequest, request: Request) -> RiskResponse:
    """Score one card transaction.

    Same response model and the same ``approve``/``review``/``decline`` vocabulary as credit.
    The plan's target contract sketched ``allow``/``review``/``block`` for this track, and that
    was not taken: a per-track vocabulary means either a union Literal that cannot express "credit
    only ever returns approve" or a second response model whose fields are otherwise identical,
    and neither buys anything a caller reading ``track`` does not already have. The divergence is
    recorded in ``STATUS.md`` rather than left as a silent disagreement with the plan.

    ``reason_codes`` stays empty here even after SHAP lands on credit: ``V1``..``V28`` are
    unlabelled principal components, so a contribution against them explains nothing a caller can
    act on. DoD (3) is recorded credit-only for that reason.
    """
    return _score("fraud", payload, request)


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
