"""Training/serving skew: the API and the model must agree on the same customer.

This is the one check that catches a column-order or alias bug. Every other test in the
suite runs against a stub, so a permuted or NaN-filled feature frame still returns a
plausible-looking probability and passes. Only scoring the *real* model twice, by two
different construction paths, makes the two answers disagree.

The two paths differ where it matters: the API builds its frame with
``reindex(columns=FEATURE_COLUMNS)``, which pins order and **fills anything it cannot find
with NaN**, while this test builds the same row straight from ``model_dump(by_alias=True)``
with no reindex. A ``FEATURE_COLUMNS`` entry that no ``serialization_alias`` produces is
therefore a silently imputed NaN on one path and simply absent on the other.

Verified by mutation, not assumed: renaming ``serialization_alias="MonthlyCharges"`` to
``"MonthlyCharge"`` makes this test fail. It fails on the *direct* path, where MLflow's
signature enforcement rejects the frame outright ("Model is missing inputs
['MonthlyCharges']") before any probability is produced -- so the guard that fires is
signature validation rather than a numeric divergence. Either way the skew is caught, which
is the point; the equality assertion below covers what the signature cannot see, such as a
column reaching the model imputed rather than missing.

Unlike the rest of the suite this needs a real model, so it skips when the registry is
unreachable. Milestone 3 runs the suite in GitHub Actions where neither ``mlflow.db`` nor
``mlruns/`` exists; the check is meant to run locally, before and after any change to
``schemas.py``, ``FEATURE_COLUMNS``, or ``MODEL_URI``.

Parametrized per track from W2 Step 8. The mutation its docstring documents has to be
re-established and re-verified **once per track**, because the property is per model and not
per module -- a credit model proven skew-free says nothing about the fraud one.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from src.api import main
from src.api.schemas import (
    CREDIT_EXAMPLE_REQUEST,
    FRAUD_EXAMPLE_REQUEST,
    CreditPredictRequest,
    FraudPredictRequest,
)

# Track -> (endpoint, request model, example payload). One entry per track the API can serve,
# and both tracks are here from Step 9.
#
# The body below still **fails loudly** rather than skipping when a registered model has no entry,
# and that guard is load-bearing history rather than dead defensiveness: between Step 8, which
# registered the fraud track, and Step 9, which added `POST /predict/fraud`, this mapping was
# credit-only while a fraud model existed -- and the failure is what made the missing endpoint
# impossible to ship quietly. A registered model the API cannot serve is a model nothing checks
# for skew, which is a defect and not a missing precondition. Leave the guard in: the next track
# added will pass through the same window.
SERVING_CONTRACTS = {
    "credit": ("/predict/credit", CreditPredictRequest, CREDIT_EXAMPLE_REQUEST),
    "fraud": ("/predict/fraud", FraudPredictRequest, FRAUD_EXAMPLE_REQUEST),
}


@pytest.fixture(scope="module", params=["credit", "fraud"])
def registered_model(request):
    """The real pyfunc model for one track, or a skip if there is no registry entry.

        Parametrized by track rather than hardcoded, because skew is a per-model property: a
        credit model proven skew-free says nothing about the fraud one, and fraud is the harder
        case -- 28 near-identical ``V*`` float columns mean a column-order bug yields a
        *plausible* probability rather than an obvious one. MLflow signature enforcement catches
        missing columns; only the exact-equality assertion below catches a permutation.

        Both parameters run from Step 9, which registered the fraud model and added
    ``POST /predict/fraud``. Between those two, the fraud parameter *failed* rather than skipped --
    deliberately, because a registered model the API cannot serve is a model nothing checks for
    skew, and the body below says so loudly instead of passing quietly.
        The skip names the registry URI it could not resolve, which is the distinction that
        matters: the URI is built by ``main.model_uri_for`` -- the real resolution path -- so a
        skip here means "that model is not registered", never "this test called something
        wrong". Calling ``load_model()`` bare against the empty default ``MODEL_URI`` is exactly
        the broken call that would have reported "no registry" for the wrong reason.

        **Only absence skips.** The guard is narrowed to MLflow's ``RESOURCE_DOES_NOT_EXIST``,
        because a catch-all would report a corrupt artifact, a dependency mismatch, an auth
        failure or a transient registry error as "no model" -- and this is the one test in the
        suite that scores a real model, so a skip that swallows those is a skip that hides the
        failure of the only check that can see training/serving skew. Anything other than
        absence re-raises and fails the run.
    """
    track = request.param
    uri = main.model_uri_for(track)
    try:
        model, version = main.load_model(uri)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        pytest.skip(f"no {track} model registered at {uri} ({type(exc).__name__}: {exc})")
    return track, model, version


def test_served_probability_matches_the_model(registered_model, tmp_path, monkeypatch):
    track, model, version = registered_model
    if track not in SERVING_CONTRACTS:
        pytest.fail(
            f"{track!r} has a model registered at {main.model_uri_for(track)} but no entry in "
            f"SERVING_CONTRACTS: the API cannot serve it, so nothing checks it for skew. Add "
            f"the request model and endpoint, then add them here."
        )
    endpoint, request_model, example = SERVING_CONTRACTS[track]
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))

    # One model, two callers. Without this the lifespan would resolve the alias a second
    # time, and a comparison across two loads cannot tell a frame-construction bug from a
    # model that changed underneath it -- which is the only thing this test is here to see.
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (model, version))

    # Enable exactly the track under test. ``ENABLED_TRACKS`` defaults to credit alone because
    # the deploy shape -- one track per image or both -- is still deferred to a W2 image
    # measurement, so a fraud request against the default configuration correctly 503s. That
    # deferral is a deployment question and this is a skew test; inheriting the default would
    # make it fail for a reason it is not about.
    monkeypatch.setattr(main, "ENABLED_TRACKS", (track,))

    # Direct path: no reindex, no pinned order.
    raw_row = request_model(**example).model_dump(by_alias=True)
    direct = float(model.predict(pd.DataFrame([raw_row]))[0][1])

    # Served path: the whole chain -- validation, alias mapping, reindex, pyfunc wrapper.
    with TestClient(main.app) as client:
        response = client.post(endpoint, json=example)
        assert response.status_code == 200
        served = response.json()["risk_probability"]

    # Exact, not approximate. These are the same float from the same model; anything that
    # makes them merely close has changed what the model was fed.
    assert served == direct, f"skew: API returned {served!r}, model returned {direct!r}"
