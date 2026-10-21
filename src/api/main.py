"""FastAPI service exposing the registered risk models.

Run locally with::

    uv run python -m uvicorn src.api.main:app --reload --log-config src/api/logging.json

Logging is configured by that file, not here: the process owner sets it up, and setting it up
at import would rewire logging in every process that imports this module, tests included.

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
import math
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from src.api.prediction_log import log_prediction
from src.api.schemas import (
    CreditPredictRequest,
    FraudPredictRequest,
    HealthResponse,
    RiskResponse,
)
from src.data.tracks import get_track
from src.features.specs import FEATURE_SPECS
from src.models.costs import COST_MATRICES, decision_bands
from src.models.explain import Explainer, build_explainer, reason_codes
from src.models.registry import configure_tracking, registry_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

# prometheus_client's default buckets start at 5ms, so every observation from this service would
# land in the first one and histogram_quantile would report ~5ms for every percentile regardless of
# real latency. A quantile is only ever as precise as the bucket it falls in, so the buckets are
# **geometric at 25% steps** from 0.5 ms to ~35 ms, then coarse above: whatever the latency in that
# range, its p50 and p95 are read to within a quarter of their value.
#
# Re-derived for Step 10 from a measurement rather than the old "3-6 ms" figure, which reason codes
# made stale. Host, real credit v6 (300 trees, 31 leaves), 2,000 requests through the full app:
# p50 2.45 ms / p99 2.87 ms without reason codes, 4.47 / 5.20 ms with them; the explanation alone
# is 1.87 ms p50, 61% of it the second transform rather than the contributions. The old buckets
# put the new p50 in a 4-5 ms bucket and any slower runtime -- a 1-vCPU Cloud Run instance is the
# next one this meets -- in 7.5-10 or 10-20 ms ones, 33% and 100% wide. The range runs to 35 ms
# so a several-fold slowdown still reads cleanly; that is a margin, not a measurement.
LATENCY_BUCKETS = tuple(round(0.0005 * 1.25**step, 6) for step in range(20)) + (
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
# Its own histogram so a latency regression can be pinned on the explanation rather than the
# score: the request histogram above sees only their sum.
EXPLAIN_LATENCY = Histogram(
    "riskwatch_explain_latency_seconds",
    "Time to compute reason codes for one scored request",
    ["track"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "riskwatch_predictions_total",
    "Prediction decision distribution",
    ["track", "decision"],
)


class ModelConfigurationError(RuntimeError):
    """This deployment cannot serve a track it was told to serve, and nothing will fix it at run
    time.

    Raised for all three ways the model-to-track wiring can be wrong: a shared local path with two
    tracks enabled and no per-track override, an artifact that does not declare its track, and a
    URI or artifact that names a *different* track's model. Configuration, not availability -- a
    registry being unreachable is an ordinary load failure and stays an ``MlflowException``.

    Its own exception type because the lifespan must treat it as a *per-track* failure. A bare
    ``RuntimeError`` would read as the all-tracks-failed one raised below it.

    Named for the whole scope rather than the first case found. It was ``BakedModelMisconfigured``
    while it only guarded baked paths, and kept that name for one commit after it started refusing
    registry URIs too -- at which point the name told a maintainer the failure was about baked
    artifacts when it was not.
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

    **Returning a shared value for a single track is not the same as trusting it.** Whether what
    it resolves to is actually this track's model is checked by :func:`assert_model_matches_track`,
    not here; this function resolves and that one verifies.
    """
    # First, so an unregistered track is named as such whichever way the deployment resolves --
    # a baked image would otherwise answer "export telco's model" for a model that cannot exist.
    registry_uri = get_track(track).production_uri
    specific = os.environ.get(f"MODEL_URI_{track.upper()}")
    if specific:
        return specific
    if MODEL_URI and len(ENABLED_TRACKS) == 1:
        return MODEL_URI
    if baked_paths():
        raise ModelConfigurationError(_no_artifact_message(track))
    return registry_uri


def baked_paths() -> dict[str, str]:
    """Local artifact paths this deployment is configured with, keyed by where they came from.

    Empty means "serve from the registry", and that is the only condition under which falling back
    to ``models:/riskwatch_<track>@production`` is right.

    **The first version of the guard asked whether the *shared* ``MODEL_URI`` was a local path**,
    which misses the shape this repo actually recommends for two tracks: one
    ``MODEL_URI_<TRACK>`` each and no shared value at all. Omit one of them and that track fell
    through to a registry the image cannot reach, so forgetting a single variable in the documented
    both-tracks configuration bought back the 101.3 s retry cycle the guard exists to prevent.
    Found by review, on the third pass over this function.
    """
    found = {}
    if MODEL_URI and not MODEL_URI.startswith("models:/"):
        found["MODEL_URI"] = MODEL_URI
    for candidate in ENABLED_TRACKS:
        name = f"MODEL_URI_{candidate.upper()}"
        value = os.environ.get(name, "")
        if value and not value.startswith("models:/"):
            found[name] = value
    return found


def _no_artifact_message(track: str) -> str:
    """Why ``track`` cannot be served here, and the two ways out.

    Names the track a baked artifact actually holds whenever one says so, because the obvious
    remediation is otherwise a trap: "drop 'credit' from ENABLED_TRACKS" leaves fraud alone and
    single-track, pointed at credit's directory. ``assert_model_matches_track`` refuses that too,
    but an error message that steers into a second error is a bad error message.
    """
    variable = f"MODEL_URI_{track.upper()}"
    paths = baked_paths()

    holders = []
    for source, path in paths.items():
        stamp = Path(path) / "MODEL_TRACK"
        if stamp.is_file():
            holders.append(f"{source}={path} holds {stamp.read_text().strip()!r}")
    inventory = "; ".join(holders) if holders else "; ".join(f"{k}={v}" for k, v in paths.items())

    return (
        f"track {track!r} is enabled and {variable} is unset, but this deployment serves from "
        f"local artifacts ({inventory}) and has none for {track!r}. Falling back to the registry "
        f"is not an option here -- a baked image has no registry to reach. Export {track!r}'s "
        f"model and point {variable} at it, or drop {track!r} from ENABLED_TRACKS."
    )


def assert_model_matches_track(track: str, uri: str) -> None:
    """Whatever ``uri`` resolves to must be *this* track's model, and must say so.

    The ownership argument that makes ``model_uri_for`` refuse a shared path across two tracks
    applies just as much to one track, and leaving it out left a hole with the same shape.
    ``ENABLED_TRACKS=fraud`` against an image carrying credit's artifact reported
    ``{"status":"ok","models":{"fraud":"5"}}`` and returned **500 on every request** -- healthy by
    its own account, wrong about which model it holds, and serving nothing. The version it
    advertised belonged to another track's model.

    It was reachable by following this service's own advice: the refusal message for the credit
    track said "drop 'credit' from ENABLED_TRACKS", which leaves ``fraud`` alone, single-track,
    and pointed at credit's directory. So the remediation text created the state.

    **Both shapes are checked, because the first version of this function only checked one.** It
    returned early for any ``models:/`` URI on the reasoning that "the name in the URI *is* the
    claim" -- true, and useless while nothing compared the claim to ``track``. So
    ``ENABLED_TRACKS=fraud`` with ``MODEL_URI=models:/riskwatch_credit@production`` still passed
    startup and reported the credit model as fraud, which is the identical defect one layer up,
    shipped in the commit that claimed to fix it. An explicit ``MODEL_URI_FRAUD`` pointing at
    credit's registered model is the same mistake by hand.

    - ``models:/riskwatch_<track>@...`` -- the registered name must match ``Track.model_name``.
      Parsed rather than trusted.
    - a local path -- must carry ``MODEL_TRACK``, which ``src/models/export.py`` writes from the
      registered model name it resolved to fetch the artifact. **Absence is a failure, not a
      pass:** an artifact that cannot say what it is provides exactly the ambiguity this check
      exists to remove, and "unlabelled means trust the caller" is the behaviour that shipped.
    """
    expected = get_track(track).model_name

    if uri.startswith("models:/"):
        name = uri.removeprefix("models:/").split("@", 1)[0].rsplit("/", 1)[0]
        if name != expected:
            raise ModelConfigurationError(
                f"{uri} names the registered model {name!r}, but this deployment is serving it "
                f"as track {track!r}, whose model is {expected!r}. Point it at {expected!r}, or "
                f"change ENABLED_TRACKS to the track {name!r} belongs to."
            )
        return

    stamp = Path(uri) / "MODEL_TRACK"
    if not stamp.is_file():
        raise ModelConfigurationError(
            f"the artifact at {uri} does not say which track it holds, so serving it as "
            f"{track!r} would be a guess. Re-export it with "
            f"`uv run python -m src.models.export --uri {get_track(track).production_uri}`"
        )

    baked = stamp.read_text().strip()
    if baked != track:
        raise ModelConfigurationError(
            f"the artifact at {uri} holds the {baked!r} model but this deployment is serving it "
            f"as {track!r}. Export {track!r}'s model, or set ENABLED_TRACKS={baked} to match the "
            f"artifact that is actually baked in."
        )


def assert_model_matches_contract(track: str, uri: str, model: Any) -> None:
    """The artifact's logged input schema must name the same features the track serves.

    Separate from :func:`assert_model_can_score` because it refuses on a criterion that function's
    name excludes: an artifact **narrower** than the contract scores its own example perfectly.
    MLflow's schema enforcement *drops* a request column the signature does not know about, with a
    warning and no error -- so a feature added to ``feature_columns`` would be silently ignored by
    an older model while ``/health`` reported ``ok``.

    **Compared as sets, not sequences.** MLflow reorders a request to the signature, so an
    order-only difference is harmless and refusing it would reject a sound artifact. An earlier
    version compared lists and produced "it does not know [] and expects []" for exactly that
    case -- a refusal whose message named nothing, under a parenthetical claiming ordering was
    already excluded.

    Silent when the artifact logs no input schema: that is missing metadata, not a mismatch, and
    :func:`assert_model_can_score` still has to pass.
    """
    metadata = getattr(model, "metadata", None)
    schema = metadata.get_input_schema() if metadata is not None else None
    if schema is None or not schema.input_names():
        return

    logged = set(schema.input_names())
    expected = set(FEATURE_COLUMNS[track])
    if logged == expected:
        return

    missing = sorted(expected - logged)
    unknown = sorted(logged - expected)
    raise ModelConfigurationError(
        f"{uri} was logged against a different feature contract than track {track!r} serves "
        f"now: it does not know {missing} and expects {unknown} that the contract no longer "
        f"has. MLflow would silently drop the columns it does not know and score without "
        f"them. Re-sweep the track."
    )


def assert_model_can_score(track: str, uri: str, model: Any) -> None:
    """Score the artifact's own logged input example, or refuse the track.

    **A model can load cleanly and still be unable to answer a request.** Removing the derived
    27th feature made every credit version up to v5 exactly that: their fitted
    ``ColumnTransformer`` still expects 27 columns, while ``_replace_sentinels`` -- cloudpickled
    *by reference*, so the artifact picks up today's code -- now emits 26.
    ``mlflow.pyfunc.load_model`` succeeds, the logged signature is 26 wide either way so
    signature enforcement passes, and the failure arrives per request as
    ``ValueError: Length mismatch: Expected axis has 26 elements, new values have 27``.

    Without this check the shape of that failure is a service reporting ``ok`` and 500ing every
    request -- the exact state Step 13 spent its review rounds eliminating, reached this time by
    moving an alias during an incident.

    Scoring the logged example rather than comparing widths, because width is not the general
    case: the logreg arm's estimator is legitimately wider than the contract after one-hot
    encoding, so any arithmetic on ``n_features_in_`` is encoder-specific. A prediction is the
    property actually wanted, and it catches every other artifact/code incompatibility too.

    Costs one prediction per track at startup, measured at under 2 ms.

    ``model.input_example`` is read **unguarded** on purpose. It is a property on
    ``mlflow.pyfunc.PyFuncModel`` that cannot itself raise, so the only thing a ``try`` around it
    would catch is the attribute ceasing to exist -- which is exactly the MLflow rename that would
    turn this whole check into a permanent silent skip. Letting an ``AttributeError`` reach the
    lifespan makes that a loud per-track failure instead, and ``tests/test_model_loading.py`` pins
    the attribute against the real class.

    A *missing* example (``None``) is still a skip with a warning: MLflow logs one for every model
    this project registers, but an artifact from elsewhere should not be unservable for lacking
    optional metadata. What that artifact loses is this check, not
    :func:`assert_model_matches_contract`.
    """
    example = model.input_example
    if example is None:
        logger.warning(
            "Track %r: %s carries no input example, so the load-time smoke score was skipped. "
            "An artifact/code incompatibility will surface per request instead of at startup.",
            track,
            uri,
        )
        return

    try:
        model.predict(example)
    except Exception as exc:
        raise ModelConfigurationError(
            f"{uri} loads but cannot score its own logged input example, so it cannot serve "
            f"track {track!r}: {type(exc).__name__}: {exc}. The artifact and this code disagree "
            f"about the feature set -- most likely the model predates a change to "
            f"`feature_columns` or to the pipeline. Promote a version trained against the "
            f"current contract, or re-sweep the track."
        ) from exc


def serve_single_threaded(model: Any) -> None:
    """Score one row per call on one thread.

    The trained estimator carries ``n_jobs=-1``, which LightGBM's sklearn wrapper turns into an
    explicit thread count on *every* ``predict`` -- one per visible CPU, 15 on the training host --
    overriding ``OMP_NUM_THREADS``. LightGBM parallelizes across rows, and the API sends one, so the
    extra threads are pure startup overhead. Measured on the host against v6, three alternating
    rounds of 1,500 requests: mean 4.95 -> 4.77 ms. Set on the same object pyfunc scores with.

    Skipped for an estimator without ``n_jobs`` (the test stubs), which has no threads to pin.
    """
    estimator = model.get_raw_model()[-1]
    if "n_jobs" in getattr(estimator, "get_params", dict)():
        estimator.set_params(n_jobs=1)


def explainer_for(track: str, uri: str, model: Any) -> Explainer | None:
    """Build ``track``'s explainer once, or refuse the track; ``None`` if it carries no reasons.

    Reason codes are part of credit's response contract, so a credit model that cannot produce
    them is refused at startup like any other unservable artifact -- not served with an empty
    list, which would read as fraud's permanent ``[]``.

    The explainer reads the unwrapped sklearn pipeline (``get_raw_model()``, a public pyfunc
    method), because pyfunc exposes only ``predict``. It is the same object pyfunc scores with,
    loaded once.

    The logged example is explained once against what the model scores it at, so a pipeline whose
    contributions do not add up is refused here rather than 500ing its first request. The sentinel
    path is not covered by that one row unless the example carries a sentinel;
    ``tests/test_explain.py`` covers it on a real pipeline, and the per-request check covers it
    in production.
    """
    spec = FEATURE_SPECS[track]
    if not spec.reason_codes:
        return None

    try:
        explainer = build_explainer(model.get_raw_model(), spec)
        example = model.input_example
        if example is not None:
            reason_codes(explainer, example, float(model.predict(example)[0][1]))
    except Exception as exc:
        raise ModelConfigurationError(
            f"{uri} serves track {track!r}, whose responses carry reason codes, but it cannot be "
            f"explained: {type(exc).__name__}: {exc}"
        ) from exc
    return explainer


def load_model(uri: str = MODEL_URI) -> tuple[Any, str]:
    """Load the serving model and resolve the version string to report.

    Returns ``(model, version)``. Defined at module level rather than inline in the
    lifespan so tests can substitute a stub without needing a populated registry.
    """
    if uri.startswith("models:/"):
        # load_model reads the global registry URI; see src/models/registry.py for why it is
        # anchored to the repo root rather than left to MLflow's cwd-relative default.
        configure_tracking()

    model = mlflow.pyfunc.load_model(uri)

    version = "unknown"
    if not uri.startswith("models:/"):
        # A baked local path. export.py writes MODEL_VERSION alongside the artifact for
        # exactly this case: the container has no registry to ask, so without reading the
        # file every prediction is stamped "unknown" -- and W3's deploy acceptance asks for
        # a REAL version from the public URL. Reported per prediction and in /health, so an
        # unknown version means nobody can say which model answered.
        #
        # The artifact's own file is the only source. A process-wide MODEL_VERSION variable
        # used to win over it, which was harmless with one baked track and wrong with two:
        # one value, stamped onto both, so fraud reported credit's version.
        stamp = Path(uri) / "MODEL_VERSION"
        if stamp.is_file():
            version = stamp.read_text().strip() or "unknown"
    if uri.startswith("models:/"):
        suffix = uri.removeprefix("models:/")
        if "@" in suffix:
            name, alias = suffix.split("@", 1)
            version = registry_client().get_model_version_by_alias(name, alias).version
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
    app.state.explainers = {}
    failures: dict[str, str] = {}

    for track in ENABLED_TRACKS:
        # Resolution is inside the try because it can now fail: a baked image asked to serve a
        # track it carries no artifact for raises rather than handing back an unreachable
        # registry URI. Outside the try that raise would kill the whole process, which is the
        # all-or-nothing policy the per-track loop below exists to replace.
        uri = "<unresolved>"
        try:
            uri = model_uri_for(track)
            assert_model_matches_track(track, uri)
            model, version = load_model(uri)
            serve_single_threaded(model)
            assert_model_matches_contract(track, uri, model)
            assert_model_can_score(track, uri, model)
            explainer = explainer_for(track, uri, model)
        except Exception as exc:  # noqa: BLE001 - recorded per track, reported by /health
            failures[track] = f"{type(exc).__name__}: {exc}"
            logger.error("Track %r failed to load from %s: %s", track, uri, exc)
            continue
        app.state.models[track] = model
        app.state.model_versions[track] = version
        if explainer is not None:
            app.state.explainers[track] = explainer
        logger.info("Loaded %r model %s (version %s)", track, uri, version)

    if not app.state.models:
        raise RuntimeError(f"no track loaded a model; failures: {failures}")

    app.state.load_failures = failures
    # What n_jobs=-1 would have meant here. Recorded because Cloud Run's visible CPU count is not
    # its vCPU allocation, and serve_single_threaded's case rests on that number.
    logger.info("Visible CPUs: os %s, joblib %s", os.cpu_count(), joblib.cpu_count())
    app.state.start_time = time.time()
    yield


app = FastAPI(
    title="RiskWatch",
    description="Credit-risk and fraud-detection scoring. One platform, two risk domains.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def reject_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 naming each failing field, without echoing the value that failed.

    FastAPI's default handler returns each error's ``input`` and ``ctx``. A rejected ``NaN`` or
    ``Infinity`` in ``input`` cannot be serialized as JSON, so the 422 itself raised and the client
    got a 500; ``ctx`` can hold a live exception object for the same reason. ``loc``, ``msg`` and
    ``type`` are what a caller needs to fix the request, and the shape stays ``{"detail": [...]}``.
    """
    detail = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


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
    """Fetch a loaded track model, or 503 naming the track that is down.

    **The body names the track and nothing else.** It used to interpolate
    ``load_failures[track]`` verbatim, which was survivable while those strings were mostly
    ``MlflowException: Registered Model ... not found`` -- and stopped being survivable when
    :class:`ModelConfigurationError` messages were written to be actionable for an operator. On a
    degraded deployment an unauthenticated ``POST /predict/fraud`` returned the baked artifact's
    filesystem path, the ``MODEL_URI_<TRACK>`` variable names, which track the artifact holds, and
    the deployment shape. Measured against a running app, not reasoned. This service is headed for
    a public Cloud Run URL, so that body is internet-facing.

    The detail is not lost, it is *addressed*: the lifespan already logs the full reason per track
    at ``ERROR`` (`src/api/main.py` in :func:`lifespan`), which is where an operator looks and
    where an anonymous caller cannot. ``app.state.load_failures`` still holds it for tests.

    Whether the track is misconfigured or merely absent is deliberately not distinguished here
    either: "not enabled" versus "enabled but broken" tells a prober which tracks a deployment was
    *meant* to serve.
    """
    model = request.app.state.models.get(track)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"track {track!r} is unavailable; see server logs for the reason",
        )
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
    # Inputs are finite by the request schema, so a non-finite score is a model fault. Refused
    # here, before anything is counted or logged: decide() would call NaN an approve, and the
    # log line would carry a bare NaN token that no JSON reader accepts.
    if not math.isfinite(probability):
        raise RuntimeError(f"{track} model returned a non-finite probability: {probability!r}")

    # Before anything is counted or logged: a request whose explanation fails is a 500, and a 500
    # was not a served prediction. Explained on ``features`` -- the frame the model just scored --
    # and checked against ``probability``, so a reason can never describe a different score.
    codes = []
    explainer = request.app.state.explainers.get(track)
    if explainer is not None:
        started = time.perf_counter()
        codes = reason_codes(explainer, features, probability)
        EXPLAIN_LATENCY.labels(track=track).observe(time.perf_counter() - started)

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
        reason_codes=codes,
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

    ``reason_codes`` is always empty here: ``V1``..``V28`` are unlabelled principal components, so a
    contribution against them explains nothing a caller can act on. ``FRAUD_FEATURES`` leaves
    ``reason_codes`` off, and DoD (3) is recorded credit-only for that reason.
    """
    return _score("fraud", payload, request)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """The bare URL is the one people share and click, and a 404 there reads as a dead service.

    Temporary (307) rather than permanent, so a browser does not cache it and the root can later
    point somewhere else. Left out of the schema: it is navigation, not part of the API.
    """
    return RedirectResponse(url="/docs")


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
