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

Verified by mutation, per track, not assumed. Renaming one ``serialization_alias`` in
``src/api/schemas.py`` and running this file alone against models trained on the CI fixture:

- credit, ``"AMT_CREDIT"`` -> ``"AMT_CREDITS"``: **4 of 4 credit cases fail.** The probability
  test fails on the *direct* path, where MLflow's signature enforcement refuses the frame
  ("Model is missing inputs ['AMT_CREDIT']") before a probability exists; the reason-code test
  fails selecting ``spec.feature_columns`` from that frame. So the guard that fires is signature
  validation rather than a numeric divergence -- the equality assertion covers what the signature
  cannot see, such as a column reaching the model imputed rather than missing.
- fraud, ``"V1"`` -> ``"V01"``: **2 of 4 fraud cases fail**, both probability cases, the same
  way. The two reason-code cases pass by design: fraud has no reason codes
  (``FeatureSpec.reason_codes``), so they return before touching the frame.

In the full suite ``tests/test_api.py`` catches both mutations as well, through its alias-coverage
check, so a red CI run alone does not prove this file caught them; the counts above are this
file's own.

Unlike the rest of the suite this needs a real model. Locally it skips when the registry is empty.
CI trains both tracks on a synthetic schema-shaped snapshot (``scripts/make_ci_fixture.py``) first
and sets ``RISKWATCH_REQUIRE_MODELS=1``, under which that skip is a failure
(``tests/preconditions.py``): a CI run where this does not execute is not a pass.

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
from tests.preconditions import skip_or_fail

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


def payload_for(request_model, example: dict, variant: str) -> dict:
    """The example as given, or with every optional field null.

    The null variant is the common case, not an edge: 73% of credit applicants have at least one of
    the four optional fields missing, and such a request reaches pyfunc and the explainer as an
    ``object`` column holding ``None`` -- the representation the training parquet uses for nulls.
    """
    if variant == "example":
        return example
    # Fraud has no optional fields, so its null variant is the example itself. Run rather than
    # skipped: a skip on a populated checkout reads as a missing precondition, and there is none.
    optional = [
        name for name, field in request_model.model_fields.items() if not field.is_required()
    ]
    return {**example, **dict.fromkeys(optional)}


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

        **Only absence skips.** The guard is narrowed to MLflow's not-found codes,
        because a catch-all would report a corrupt artifact, a dependency mismatch, an auth
        failure or a transient registry error as "no model" -- and this is the one test in the
        suite that scores a real model, so a skip that swallows those is a skip that hides the
        failure of the only check that can see training/serving skew. Anything other than
        absence re-raises and fails the run.
    """
    from src.models.train import NOT_FOUND_CODES

    track = request.param
    uri = main.model_uri_for(track)
    try:
        model, version = main.load_model(uri)
    except MlflowException as exc:
        # The repo's definition, in one place -- ``src/models/train.py``. This compared against
        # RESOURCE_DOES_NOT_EXIST alone while the production paths have always treated
        # ENDPOINT_NOT_FOUND as absence too, so a registry answering the latter for a missing alias
        # would have failed this test rather than skipping it -- the opposite error from those
        # modules', out of the same split
        # definition.
        if exc.error_code not in NOT_FOUND_CODES:
            raise
        skip_or_fail(f"no {track} model registered at {uri} ({type(exc).__name__}: {exc})")
    return track, model, version


@pytest.mark.parametrize("variant", ["example", "optional_nulls"])
def test_served_probability_matches_the_model(registered_model, variant, tmp_path, monkeypatch):
    track, model, version = registered_model
    if track not in SERVING_CONTRACTS:
        pytest.fail(
            f"{track!r} has a model registered at {main.model_uri_for(track)} but no entry in "
            f"SERVING_CONTRACTS: the API cannot serve it, so nothing checks it for skew. Add "
            f"the request model and endpoint, then add them here."
        )
    endpoint, request_model, example = SERVING_CONTRACTS[track]
    example = payload_for(request_model, example, variant)
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))

    # One model, two callers. Without this the lifespan would resolve the alias a second
    # time, and a comparison across two loads cannot tell a frame-construction bug from a
    # model that changed underneath it -- which is the only thing this test is here to see.
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (model, version))

    # Enable exactly the track under test. ``ENABLED_TRACKS`` defaults to credit alone in code
    # (the image enables both), so a fraud request against the default correctly 503s. That is a
    # deployment question and this is a skew test; inheriting the default would make it fail for
    # a reason it is not about.
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


@pytest.mark.parametrize("variant", ["example", "optional_nulls"])
def test_served_reason_codes_match_the_model(registered_model, variant, tmp_path, monkeypatch):
    """The reasons the API returns are the real model's own contributions to the served score.

    Explained directly from the unwrapped pipeline on a frame built without the API's alias
    mapping or reindex, then compared exactly. The stubbed API tests cannot see this: they check
    that the reasons are wired through, not that they are this model's.
    """
    from src.features.specs import get_feature_spec
    from src.models import explain

    track, model, version = registered_model
    if track not in SERVING_CONTRACTS:
        pytest.fail(f"{track!r} is registered but has no SERVING_CONTRACTS entry")
    endpoint, request_model, example = SERVING_CONTRACTS[track]
    example = payload_for(request_model, example, variant)
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (model, version))
    monkeypatch.setattr(main, "ENABLED_TRACKS", (track,))

    with TestClient(main.app) as client:
        response = client.post(endpoint, json=example)
        assert response.status_code == 200
        served = response.json()["reason_codes"]

    spec = get_feature_spec(track)
    if not spec.reason_codes:
        assert served == []
        return

    row = pd.DataFrame([request_model(**example).model_dump(by_alias=True)])
    row = row[list(spec.feature_columns)]
    probability = float(model.predict(row)[0][1])
    direct = explain.reason_codes(
        explain.build_explainer(model.get_raw_model(), spec), row, probability
    )

    assert served == direct
