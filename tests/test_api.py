"""API contract tests.

These are deliberately hermetic: ``load_model`` is stubbed, so the suite needs no
``mlflow.db`` and no populated registry. Milestone 3 runs this in GitHub Actions, where
neither exists. That the *real* model serves correctly is verified by running the app
against the registry, not here -- these tests guard the request/response contract.

``TestClient`` is synchronous, so no ``pytest-asyncio`` is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main, prediction_log
from src.api.schemas import CREDIT_EXAMPLE_REQUEST, FRAUD_EXAMPLE_REQUEST

STUB_PROBABILITY = 0.73


class StubModel:
    """Stands in for the pyfunc model, returning [[p_negative, p_positive]] like the real one.

    The assertions are the point, not the return value. ``reindex`` in the handler fills any
    column it cannot find with NaN, so a one-character typo in a ``serialization_alias``
    would produce a NaN feature, get quietly imputed by the pipeline, and score normally --
    passing every other test in this file while serving wrong predictions.
    """

    def predict(self, features):
        # Matched against *some* track's contract rather than one hardcoded list, because the
        # same stub now serves both endpoints. Asserting membership rather than equality to
        # credit's columns is what lets the NaN check below stay meaningful for fraud: a fraud
        # request reindexed onto credit's 26 columns would be 29 NaNs, and this is the assertion
        # that sees it.
        columns = list(features.columns)
        assert columns in main.FEATURE_COLUMNS.values(), (
            f"alias mapping drifted: {len(columns)} columns matching no track's contract"
        )
        assert not features.isna().any().any(), "a feature arrived as NaN"
        return np.array([[1 - STUB_PROBABILITY, STUB_PROBABILITY]] * len(features))


class ExplodingModel:
    """A model that fails at request time, e.g. a corrupt artifact."""

    def predict(self, features):
        raise RuntimeError("model is broken")


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Redirect the prediction log so tests never touch the real one."""
    path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(path))
    return path


@pytest.fixture
def client(monkeypatch, log_file):
    """A client whose app loaded the stub instead of the registry model."""
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (StubModel(), "test-1"))
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def both_tracks_client(monkeypatch, log_file):
    """A client serving credit *and* fraud.

    Separate from ``client`` because ``ENABLED_TRACKS`` defaults to credit alone: the deploy shape
    -- one track per image or both in one -- is deferred to a W2 image measurement, so a fraud
    request against the default configuration 503s by design. Two fixtures keep both facts
    testable: that the fraud endpoint works when the track is enabled, and that it refuses
    clearly when it is not.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (StubModel(), "test-1"))
    with TestClient(main.app) as test_client:
        yield test_client


def test_predict_happy_path(client):
    """The full response shape, and the region STUB_PROBABILITY actually falls in.

    This assertion has moved twice, and the history is the argument for writing it this way.
    0.73 was a *decline* against the W1 placeholder band (0.40, 0.60); a *review* against the
    priced band (0.0014, 0.98), which reviewed everything; and a *decline* again against the
    budgeted band (0.0645, 0.0963). The model never changed. The boundary did, twice.

    So the membership check above the decision is the substance rather than decoration. Asserting
    an outcome alone would have stayed green through the middle state, where the band had swallowed
    the entire score distribution and this endpoint could return nothing else -- the assertion
    would have been describing the band instead of testing it. Locating 0.73 against a named edge
    fails the moment that stops being true.

    Reachability of the *other* outcomes is a different question and cannot be answered here,
    because a stub probability proves nothing about what a real model produces. That is
    ``tests/test_reachable_decisions.py``.
    """
    response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "risk_probability",
        "decision",
        "reason_codes",
        "threshold",
        "track",
        "model_version",
        "request_id",
    }
    assert body["risk_probability"] == pytest.approx(STUB_PROBABILITY)
    assert body["track"] == "credit"

    _, decline_at = main.DECISION_BANDS["credit"]
    assert STUB_PROBABILITY >= decline_at, (
        f"STUB_PROBABILITY {STUB_PROBABILITY} must sit at or above the credit decline boundary "
        f"{decline_at} for this test to be about the decline outcome"
    )
    assert body["decision"] == "decline"
    # decline_at for every outcome, including this one: the field reports the cut the
    # decision was taken against, not the nearest boundary.
    assert body["threshold"] == pytest.approx(decline_at)


def test_reason_codes_ship_empty_until_shap_lands(client):
    """DoD (3) is open, and the contract must say so rather than imply otherwise.

    The field exists now so the contract breaks once rather than twice -- there is no
    compatibility shim in this repo, so adding it in W2 would be a second breaking change.
    Asserting it is *empty* is what stops an unimplemented capability reading as delivered.
    """
    body = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()
    assert body["reason_codes"] == []


def test_decision_bands_are_three_valued(client, monkeypatch):
    """approve / review / decline, driven by the two operating points.

    The review band is the reason the field is not a boolean: the cost of wrongly
    approving and the cost of wrongly declining are not symmetric, so the middle is routed
    to a human rather than forced to one side.
    """
    review_at, decline_at = main.DECISION_BANDS["credit"]
    seen = {}

    for probability, expected in (
        (review_at - 0.05, "approve"),
        ((review_at + decline_at) / 2, "review"),
        (decline_at + 0.05, "decline"),
    ):
        decision, threshold = main.decide(probability, "credit")
        seen[expected] = decision
        assert decision == expected, f"p={probability} should be {expected}, got {decision}"
        assert threshold == decline_at

    assert set(seen) == {"approve", "review", "decline"}


def test_health_reports_degraded_when_a_track_fails_to_load(monkeypatch, log_file):
    """One track down must not take the other with it.

    The previous all-or-nothing lifespan failed the whole process if the single model
    would not load. With two independent tracks that would mean a fraud-side registry
    problem taking the credit endpoint down -- and on Cloud Run, failing the entire
    revision and with it the public URL that DoD (1) depends on.

    This is the live state of the repo as of W2 Step 8, not a hypothetical: ``riskwatch_fraud``
    is a registered *track* with no registered *model* yet, so the URI below is exactly what
    a real boot resolves and fails on. The URI is derived through ``model_uri_for`` rather
    than matched on the substring ``"fraud"``, so the stub cannot pass while the real
    resolution is broken.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    fraud_uri = main.model_uri_for("fraud")
    assert fraud_uri == "models:/riskwatch_fraud@production"

    def half_broken(uri: str = ""):
        if uri == fraud_uri:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST: riskwatch_fraud has no version")
        return StubModel(), "test-1"

    monkeypatch.setattr(main, "load_model", half_broken)

    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["models"] == {"credit": "test-1"}

        # The healthy track still serves.
        assert client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).status_code == 200


def test_a_shared_baked_path_is_refused_for_every_track_when_two_are_enabled(monkeypatch):
    """Enabling a second track must not silently un-bake the first.

    Found in the Step 13 deploy rehearsal against a real container. The condition this replaces
    was ``if len(ENABLED_TRACKS) == 1 and MODEL_URI``, so adding a second track stopped the
    baked path being used **for every track**: an image carrying credit's artifact sent credit
    to ``models:/riskwatch_credit@production`` as well, a registry it cannot reach. Measured,
    both tracks failed and the process took **204.5 s** to exit 3.

    **Not a hang, and the distinction was checked rather than assumed.**
    ``mlflow.store.db.utils.MAX_RETRY_COUNT`` is 10 with sleeps ``0.1 * (2**n - 1)``, so 101.3 s
    per engine creation and one cycle per track; ``load_model`` then raises and the lifespan's
    per-track isolation runs as designed. An earlier draft of this docstring called it unbounded
    on the strength of a 45-second poll against a 204-second failure.

    So what the guard buys is 204.5 s -> 1.0 s plus a message naming the misconfiguration -- not
    the recovery of a protection that was never lost.

    **Both tracks are refused, including the one the artifact actually holds, and that is the
    correct answer rather than a limitation.** A bare path carries no claim about whose model
    it is. `build/model/registered_model_meta` happens to name one, but resolving ownership
    from it would make the serving contract depend on MLflow's artifact layout, and it would
    still be guessing which track the *operator* meant. An image with one artifact and two
    enabled tracks is a configuration that cannot work; serving half of it, chosen by the API,
    is worse than refusing it with a message that names both ways out.
    """
    monkeypatch.setattr(main, "MODEL_URI", "/app/model")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_CREDIT", raising=False)
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    for track in ("credit", "fraud"):
        with pytest.raises(main.ModelConfigurationError) as raised:
            main.model_uri_for(track)
        # The operator reading this is looking at a revision that will not start, so the
        # message has to name both exits rather than only the symptom.
        assert f"MODEL_URI_{track.upper()}" in str(raised.value)
        assert "ENABLED_TRACKS" in str(raised.value)


def test_the_misconfigured_deployment_refuses_to_start_immediately(monkeypatch):
    """No track resolves, so the process must exit -- and at resolution, not after 204 s.

    Distinct from the degraded path deliberately. ``degraded`` is for "a track I was told to
    serve is missing", which leaves something worth answering with. Here *nothing* resolves, and
    the lifespan's own rule is that every track failing stays fatal.

    The exit is not what changed -- the pre-guard configuration exited too, with the same code,
    after two 101.3 s MLflow retry cycles. What this pins is that the refusal happens during URI
    resolution, with no registry contacted, so the cost is a function call rather than 3.4
    minutes of billable startup. A test cannot observe the timing difference, so it observes the
    mechanism: ``model_uri_for`` raising is what the lifespan converts into the failure below.
    """
    monkeypatch.setattr(main, "MODEL_URI", "/app/model")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_CREDIT", raising=False)
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def _baked(dir_path: Path, track: str, version: str = "5") -> Path:
    """A minimal stand-in for an exported artifact: the two files the API reads off disk."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "MODEL_TRACK").write_text(f"{track}\n")
    (dir_path / "MODEL_VERSION").write_text(f"{version}\n")
    return dir_path


def test_two_baked_artifacts_serve_two_tracks(monkeypatch, log_file, tmp_path):
    """The supported both-tracks shape: one explicit path per track, no shared ``MODEL_URI``.

    This is what the rehearsal's measurement argues for -- the fraud artifact is 348 KB against
    a 363.5 MB dependency layer, so a second model is free and the deploy shape is "both". The
    test pins the wiring that makes it work, so the Dockerfile change landing later has a
    contract to satisfy rather than one to invent.

    Real directories rather than string paths, because each artifact now has to *declare* its
    track and both declarations are checked at startup.
    """
    credit_dir = _baked(tmp_path / "credit", "credit")
    fraud_dir = _baked(tmp_path / "fraud", "fraud")

    monkeypatch.setattr(main, "MODEL_URI", str(tmp_path))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_CREDIT", str(credit_dir))
    monkeypatch.setenv("MODEL_URI_FRAUD", str(fraud_dir))

    assert main.model_uri_for("credit") == str(credit_dir)
    assert main.model_uri_for("fraud") == str(fraud_dir)

    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))

    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["models"] == {"credit": "5", "fraud": "5"}

        assert client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).status_code == 200
        assert client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST).status_code == 200


def test_a_single_enabled_track_cannot_serve_another_tracks_artifact(monkeypatch, tmp_path):
    """``ENABLED_TRACKS=fraud`` against credit's baked artifact must refuse, not report ok.

    The hole this closes had the same shape as the two-track one and was left open by the same
    omission: ``model_uri_for`` returns a shared ``MODEL_URI`` unconditionally when one track is
    enabled, and a bare directory carries no claim about whose model it is. Measured against a
    real container before the check existed: ``/health`` reported
    ``{"status":"ok","models":{"fraud":"5"}}`` -- advertising another track's version -- and every
    ``POST /predict/fraud`` returned **500**, because MLflow's signature enforcement rejected a
    29-column fraud frame against a 26-column credit model.

    Worse, this service's own error message steered operators into it: the refusal for the credit
    track said "drop 'credit' from ENABLED_TRACKS", which leaves fraud alone and pointed at
    credit's directory.
    """
    artifact = _baked(tmp_path / "model", "credit")

    monkeypatch.setattr(main, "MODEL_URI", str(artifact))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("fraud",))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    # Resolution still hands back the shared path: resolving and verifying are separate jobs.
    assert main.model_uri_for("fraud") == str(artifact)

    with pytest.raises(main.ModelConfigurationError) as raised:
        main.assert_model_matches_track("fraud", str(artifact))
    assert "'credit'" in str(raised.value), "the message must name what the artifact holds"
    assert "ENABLED_TRACKS=credit" in str(raised.value), "and the configuration that would match"

    # End to end: the startup refuses rather than reporting a healthy fraud deployment.
    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))
    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_an_unlabelled_baked_artifact_is_refused(monkeypatch, tmp_path):
    """No ``MODEL_TRACK`` is a failure, not a pass.

    "Unlabelled means trust the caller" is precisely the behaviour that shipped, so treating a
    missing marker as permission would leave the hole open for every artifact exported before the
    marker existed -- which is all of them.
    """
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "MODEL_VERSION").write_text("5\n")

    monkeypatch.setattr(main, "MODEL_URI", str(artifact))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))

    with pytest.raises(main.ModelConfigurationError, match="does not say which track"):
        main.assert_model_matches_track("credit", str(artifact))


def test_a_matching_artifact_and_a_registry_uri_both_pass(monkeypatch, tmp_path):
    """The two shapes that are correct must not raise: a labelled match, and a matching name."""
    artifact = _baked(tmp_path / "model", "credit")

    main.assert_model_matches_track("credit", str(artifact))
    main.assert_model_matches_track("fraud", "models:/riskwatch_fraud@production")
    # The version form takes a different branch of the same parse.
    main.assert_model_matches_track("fraud", "models:/riskwatch_fraud/3")


def test_a_registry_uri_naming_another_tracks_model_is_refused(monkeypatch):
    """``ENABLED_TRACKS=fraud`` with ``MODEL_URI=models:/riskwatch_credit@production`` must refuse.

    This is the identical defect to the baked-path one, one layer up, and it **shipped in the
    commit that claimed to fix that one.** The first version of
    ``assert_model_matches_track`` returned early for any ``models:/`` URI, reasoning that "the
    name in the URI *is* the claim" -- true, and worth nothing while nothing compared the claim to
    the track being served. Found by review, not by me.

    Reachable two ways, both plausible: a single-track deployment inherits the shared
    ``MODEL_URI`` unconditionally, and an explicit ``MODEL_URI_FRAUD`` can simply be pointed at
    the wrong registered model by hand.
    """
    monkeypatch.setattr(main, "MODEL_URI", "models:/riskwatch_credit@production")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("fraud",))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    # Resolution hands it back: a models:/ value is a legitimate single-track MODEL_URI, and
    # whether it is the *right* model is the verifier's question.
    assert main.model_uri_for("fraud") == "models:/riskwatch_credit@production"

    with pytest.raises(main.ModelConfigurationError) as raised:
        main.assert_model_matches_track("fraud", main.model_uri_for("fraud"))
    message = str(raised.value)
    assert "riskwatch_credit" in message, "the message must name the model the URI points at"
    assert "riskwatch_fraud" in message, "and the one this track expects"

    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))
    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_an_explicit_per_track_override_is_verified_too(monkeypatch):
    """``MODEL_URI_FRAUD`` pointing at credit's model must not be trusted because it is explicit.

    An explicit override is the operator being specific, not the operator being right, and this is
    the one path with no shared-``MODEL_URI`` ambiguity to blame -- so if verification only ran on
    the fallbacks, the most deliberate misconfiguration would be the one that got through.
    """
    monkeypatch.setattr(main, "MODEL_URI", "")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_FRAUD", "models:/riskwatch_credit@production")

    with pytest.raises(main.ModelConfigurationError, match="riskwatch_credit"):
        main.assert_model_matches_track("fraud", main.model_uri_for("fraud"))


def test_a_registry_deployment_still_resolves_both_tracks(monkeypatch, log_file):
    """A registry-backed ``MODEL_URI`` must fall through to the per-track default, not raise.

    Narrowing the guard to non-``models:/`` values is what keeps it from breaking a deployment
    shape it was not about: a ``models:/`` value is not a baked artifact, and the registry behind
    it holds every track, so resolving each track to its own registry URI is correct.

    **Not the current compose configuration**, which is worth stating because an earlier version
    of this docstring claimed it was. `docker-compose.yml` sets `MODEL_URI_CREDIT` and no
    `ENABLED_TRACKS`, so it serves credit alone and never reaches this branch. This is the
    manually configured two-track registry case — the shape a registry-backed both-tracks
    deployment would take, pinned before anything is wired to produce it.
    """
    monkeypatch.setattr(main, "MODEL_URI", "models:/riskwatch_credit@production")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    assert main.model_uri_for("fraud") == "models:/riskwatch_fraud@production"


def test_health_is_ok_when_only_one_of_the_two_tracks_is_enabled(monkeypatch, log_file):
    """``degraded`` means "something I was told to serve is missing", not "I serve one track".

    This is the complement of the test above and it is the half that became load-bearing
    when the second track registered. Both tracks now exist in the registry, so a
    single-track deployment is a real configuration -- W2 Step 15 decides whether the image
    carries one model or two off a measured size, and the 0.5 GB Artifact Registry budget
    may well decide it carries one.

    Reporting that deployment as ``degraded`` would make the signal useless in the direction
    that matters: Cloud Run's health check would flag a correctly configured revision, and an
    operator would learn to ignore the field that is supposed to tell them a model is down.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))

    def only_credit(uri: str = ""):
        assert uri == main.model_uri_for("credit"), f"a disabled track was loaded from {uri}"
        return StubModel(), "test-1"

    monkeypatch.setattr(main, "load_model", only_credit)

    with TestClient(main.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["models"] == {"credit": "test-1"}, "a disabled track must not appear as loaded"


def test_enabled_tracks_is_parsed_from_the_environment():
    """The switch is what defers the one-model-or-two deploy decision to a measurement.

    Parsed rather than assumed: the env var is a comma-separated list, and a deployment
    setting ``ENABLED_TRACKS="credit, fraud ,"`` -- a stray space and a trailing comma, both
    ordinary in a YAML env block -- must enable two tracks rather than one named ``" fraud"``
    resolving to ``models:/riskwatch_ fraud@production``.

    Run in a subprocess because the value is read at import time. ``importlib.reload`` is the
    obvious alternative and it does not work here: re-executing the module re-registers the
    Prometheus collectors and raises ``DuplicateTimeseries``, which is the reason those live
    at module level in the first place.
    """
    import os
    import subprocess
    import sys

    probe = "import src.api.main as m; print(m.ENABLED_TRACKS, m.model_uri_for('fraud'))"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "ENABLED_TRACKS": "credit, fraud ,"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "('credit', 'fraud')" in result.stdout, result.stdout
    assert "models:/riskwatch_fraud@production" in result.stdout, result.stdout


def test_predict_request_ids_are_unique(client):
    """Two identical requests must be individually traceable in the prediction log."""
    first = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()["request_id"]
    second = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()["request_id"]
    assert first != second


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in CREDIT_EXAMPLE_REQUEST.items() if k != "amt_credit"}
    response = client.post("/predict/credit", json=payload)
    assert response.status_code == 422
    assert any(err["loc"][-1] == "amt_credit" for err in response.json()["detail"])


def test_predict_out_of_range_returns_422(client):
    response = client.post("/predict/credit", json={**CREDIT_EXAMPLE_REQUEST, "amt_credit": -5})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "greater_than"


def test_predict_rejects_a_positive_days_birth(client):
    """DAYS_BIRTH is a negative offset from the application date.

    A caller passing an age in years would otherwise get a confident prediction from a
    value the model has never seen -- 34 where every training row is around -12000.
    """
    response = client.post("/predict/credit", json={**CREDIT_EXAMPLE_REQUEST, "days_birth": 34})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "less_than_equal"


def test_predict_unknown_category_returns_422(client):
    """A category the model never saw would be encoded as -1 and scored anyway."""
    response = client.post(
        "/predict/credit", json={**CREDIT_EXAMPLE_REQUEST, "name_contract_type": "Barter"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "literal_error"


def test_predict_rejects_camelcase_input(client):
    """The public contract is snake_case only -- the model's column names must not leak."""
    payload = {**CREDIT_EXAMPLE_REQUEST}
    payload["AMT_CREDIT"] = payload.pop("amt_credit")
    response = client.post("/predict/credit", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_field(client):
    """extra='forbid': silently ignoring a field would mislead the caller."""
    response = client.post(
        "/predict/credit", json={**CREDIT_EXAMPLE_REQUEST, "customerID": "1234-ABCDE"}
    )
    assert response.status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["models"] == {"credit": "test-1"}
    assert body["uptime_seconds"] >= 0


def test_metrics_exposes_prediction_counter(client):
    client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=1.0.0; charset=utf-8"
    assert "riskwatch_predictions_total" in response.text
    assert "riskwatch_request_latency_seconds" in response.text


def test_unmatched_paths_share_one_metric_label(client):
    """Labelling by raw path would mint a series per URL a scanner probes."""
    for path in ("/nope", "/.env", "/wp-admin/setup-config.php"):
        assert client.get(path).status_code == 404

    metrics = client.get("/metrics").text
    assert 'endpoint="unmatched"' in metrics
    assert ".env" not in metrics
    assert "wp-admin" not in metrics


def test_prediction_is_logged(client, log_file):
    response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert set(record) == {
        "timestamp",
        "request_id",
        "model_version",
        "track",
        "input_hash",
        "risk_probability",
        "decision",
        "features",
    }
    assert record["track"] == "credit"
    # The log must agree with what the caller was told, or analysis built on it is fiction.
    assert record["request_id"] == response["request_id"]
    assert record["risk_probability"] == response["risk_probability"]
    assert record["model_version"] == "test-1"


def test_logged_features_match_the_model_contract(client, log_file):
    """Evidently compares these against training data, so the names must be identical."""
    client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)

    record = json.loads(log_file.read_text().splitlines()[0])
    assert sorted(record["features"]) == sorted(main.FEATURE_COLUMNS["credit"])


def test_each_prediction_appends_one_line(client, log_file):
    for _ in range(3):
        client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
    assert len(log_file.read_text().splitlines()) == 3


def test_input_hash_is_order_independent_and_discriminating():
    a = prediction_log.input_hash({"AMT_CREDIT": 406597.5, "CNT_CHILDREN": 0})
    reordered = prediction_log.input_hash({"CNT_CHILDREN": 0, "AMT_CREDIT": 406597.5})
    different = prediction_log.input_hash({"AMT_CREDIT": 406598.5, "CNT_CHILDREN": 0})

    assert a == reordered
    assert a != different


def test_logging_failure_does_not_break_prediction(client, monkeypatch, tmp_path):
    """Observability must degrade, not take down serving.

    input_hash runs during record construction, before either sink. If that raised outside
    a guard it would propagate into the handler and turn a logging fault into a 500, which
    is the exact inversion this module exists to prevent.
    """
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "nope" / "x.jsonl"))
    monkeypatch.setattr(
        prediction_log, "input_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
    assert response.status_code == 200
    assert response.json()["risk_probability"] == pytest.approx(STUB_PROBABILITY)


def test_prediction_is_emitted_to_stdout_as_json(client, capfd):
    """The stdout sink is the only one that survives the deployed environment.

    Cloud Run has no durable filesystem -- the container-local log is discarded on every
    scale-to-zero -- and this record is the substrate drift, PSI, decomposition and the
    incident write-up all read. It must be one bare parseable JSON object per line, with
    no level prefix, because Cloud Logging parses a bare JSON line into structured fields
    and a prefixed line stays an opaque string.
    """
    response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()

    emitted = [
        json.loads(line)
        for line in capfd.readouterr().out.splitlines()
        if line.startswith('{"log_type"')
    ]

    assert len(emitted) == 1, "expected exactly one structured prediction line on stdout"
    assert emitted[0]["request_id"] == response["request_id"]
    assert emitted[0]["track"] == "credit"
    assert emitted[0]["risk_probability"] == pytest.approx(STUB_PROBABILITY)


def test_stdout_sink_survives_a_broken_file_sink(client, monkeypatch, tmp_path, capfd):
    """The two sinks fail independently.

    A container with an unwritable /app/logs -- the classic missing-chown case -- must
    still be observable in Cloud Logging, which is the sink that matters once deployed.
    """
    unwritable = tmp_path / "blocked"
    unwritable.write_text("not a directory")
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(unwritable / "predictions.jsonl"))

    assert client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).status_code == 200

    emitted = [
        line for line in capfd.readouterr().out.splitlines() if line.startswith('{"log_type"')
    ]
    assert len(emitted) == 1, "stdout sink must survive a file-sink failure"


def test_failed_prediction_is_counted(monkeypatch):
    """A 500 must still land in the metrics, or error rate is uncomputable."""
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (ExplodingModel(), "test-1"))
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
        assert response.status_code == 500

        metrics = client.get("/metrics").text
        assert 'riskwatch_requests_total{endpoint="/predict/credit",status="500"}' in metrics


def test_baked_model_path_reports_its_real_version(tmp_path):
    """A baked artifact must report its version, not "unknown".

    The container has no registry to ask, so export.py writes MODEL_VERSION next to the
    artifact. Without reading it every prediction and every /health response is stamped
    "unknown" -- and W3's deploy acceptance asks for a REAL version from the public URL,
    which is exactly this path.

    Exercised against the real exported artifact rather than a mock: the version file is
    written by export.py, so a test that stubs the loader would pass even if export stopped
    writing it. Skips when nothing has been exported yet, since that is a local-state
    precondition rather than a defect.
    """
    exported = Path(__file__).resolve().parents[1] / "build" / "model"
    if not (exported / "MLmodel").is_file():
        pytest.skip("no exported artifact at build/model; run `python -m src.models.export`")
    # Integration counterpart to test_baked_version_is_read_hermetically below, which runs
    # everywhere. Kept because only this one proves export.py actually writes the file.

    stamp = exported / "MODEL_VERSION"
    assert stamp.is_file(), "export.py must write MODEL_VERSION beside the artifact"

    _, version = main.load_model(str(exported))

    assert version == stamp.read_text().strip()
    assert version != "unknown", "a baked artifact must not serve predictions as 'unknown'"


def test_baked_artifact_is_not_stale_against_the_alias():
    """The baked artifact must be the version ``@production`` points at, not an older one.

    The test above proves the baked version is *readable*; it cannot see whether it is
    *current*, because a stale export reports its own version perfectly confidently. That gap
    shipped: ``build/model`` held credit v4 while the alias had moved to v5, every test was
    green, and a ``docker build`` would have baked the wrong model into an image whose
    ``/health`` announced a real-looking version. Nothing compared the two numbers.

    It matters at the deploy rather than here. The image is built from this directory and the
    container has no registry to check itself against, so the last moment the comparison is
    possible is on the host, before the build -- and `src/models/export.py` resolving the alias
    correctly does not help when the export simply was not re-run.

    **Which registry gets compared is the resolver's business, not this test's.**
    ``resolve_version`` anchors tracking to ``PROJECT_ROOT/mlflow.db`` itself, so this reads the
    same backend the export wrote from regardless of the cwd pytest was launched in. Setting the
    URI here instead would have made the test pass while leaving the CLI free to inspect whatever
    database the caller happened to be standing in.

    Skips on the two local-state preconditions, each named: no export, or no registry entry to
    compare against. A version mismatch is a defect and fails.
    """
    from mlflow.exceptions import MlflowException

    from src.models import export
    from src.models.train import NOT_FOUND_CODES

    exported = Path(__file__).resolve().parents[1] / "build" / "model"
    stamp = exported / "MODEL_VERSION"
    # MLmodel as well as the stamp, matching the integration test above. A leftover or
    # half-written build/model can hold a version file and no loadable model, and comparing its
    # number against the alias would pass while the image has nothing to serve -- a green test
    # for a directory that cannot answer a request.
    if not (stamp.is_file() and (exported / "MLmodel").is_file()):
        pytest.skip("no exported artifact at build/model; run `python -m src.models.export`")

    try:
        current = export.resolve_version(export.DEFAULT_MODEL_URI)
    except MlflowException as exc:
        # The repo's definition of absence, not a narrower local one: a registry answering
        # ENDPOINT_NOT_FOUND for a missing alias would fail this test on a clean checkout instead
        # of taking the skip it is entitled to.
        if exc.error_code not in NOT_FOUND_CODES:
            raise
        pytest.skip(f"nothing registered at {export.DEFAULT_MODEL_URI} to compare against")

    baked = stamp.read_text().strip()
    assert baked == current, (
        f"build/model holds version {baked} but {export.DEFAULT_MODEL_URI} resolves to "
        f"{current}: re-run `uv run python -m src.models.export` before docker build, or the "
        f"image serves a model the registry no longer promotes"
    )


def test_baked_version_is_read_hermetically(tmp_path, monkeypatch):
    """The same behaviour as the integration test above, but runs in CI.

    ``build/model`` is gitignored and the suite performs no export, so the real-artifact test
    skips in a clean checkout -- a regression in the local-path branch of ``load_model``
    would leave CI green. This stubs only the loader, so every line of the version-resolution
    path still executes.
    """
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "MODEL_VERSION").write_text("11\n")

    monkeypatch.delenv("MODEL_VERSION", raising=False)
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    _, version = main.load_model(str(artifact))

    assert version == "11", "the baked version file must win over the 'unknown' default"


def test_baked_path_without_a_version_file_reports_unknown(tmp_path, monkeypatch):
    """Absent is reported honestly rather than guessed at."""
    artifact = tmp_path / "model"
    artifact.mkdir()

    monkeypatch.delenv("MODEL_VERSION", raising=False)
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    _, version = main.load_model(str(artifact))

    assert version == "unknown"


def test_an_explicit_env_version_wins_over_the_file(tmp_path, monkeypatch):
    """MODEL_VERSION is the documented override; the file is the fallback for a baked image."""
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "MODEL_VERSION").write_text("11\n")

    monkeypatch.setenv("MODEL_VERSION", "99")
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    _, version = main.load_model(str(artifact))

    assert version == "99"


# --- The fraud endpoint (Step 9) ----------------------------------------------------------


def test_predict_fraud_happy_path(both_tracks_client):
    """The second track's endpoint, on the same response model as the first.

    The plan's target contract sketched ``allow``/``review``/``block`` for fraud; that was not
    taken, and this assertion is where the divergence is visible. A per-track vocabulary means
    either a union Literal that cannot express "credit only ever returns approve" or a second
    response model identical but for one field, and a caller that reads ``track`` already knows
    which domain answered.
    """
    response = both_tracks_client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST)
    assert response.status_code == 200

    body = response.json()
    assert body["track"] == "fraud"
    assert body["risk_probability"] == pytest.approx(STUB_PROBABILITY)
    assert body["decision"] in {"approve", "review", "decline"}
    _, decline_at = main.DECISION_BANDS["fraud"]
    assert body["threshold"] == pytest.approx(decline_at)


def test_fraud_reason_codes_ship_empty_and_will_stay_empty(both_tracks_client):
    """Empty here is permanent, not pending, and that is a different claim from credit's.

    Credit's ``reason_codes`` are empty until SHAP lands. Fraud's stay empty afterwards, because
    ``V1``..``V28`` are unlabelled principal components -- the ULB researchers never published the
    loadings -- so a contribution against them explains nothing a caller can act on. DoD (3) is
    recorded credit-only for exactly this reason.
    """
    body = both_tracks_client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST).json()
    assert body["reason_codes"] == []


def test_fraud_refuses_the_column_it_validates_but_does_not_model(both_tracks_client):
    """``Time`` is under schema contract at ingest and is not a request field.

    A caller sending it has misread the contract rather than merely added a field, so 422 is the
    right answer -- and ``extra="forbid"`` is what makes it one. Accepting and ignoring it would
    let a caller believe a value reached the model.
    """
    response = both_tracks_client.post(
        "/predict/fraud", json={**FRAUD_EXAMPLE_REQUEST, "time": 0.0}
    )
    assert response.status_code == 422


def test_fraud_endpoint_refuses_clearly_when_the_track_is_not_enabled(client):
    """The default deployment serves credit only, and must say so rather than fail obscurely.

    ``ENABLED_TRACKS`` defaults to credit because the deploy shape is deferred to a W2 image
    measurement. Until that decision lands, ``/predict/fraud`` exists and 503s naming the track --
    which is a different failure from a 404, and the difference matters: the route is real, the
    model is simply not loaded here.
    """
    response = client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST)
    assert response.status_code == 503
    assert "fraud" in response.json()["detail"]


def test_fraud_prediction_is_logged_with_its_own_track_and_columns(both_tracks_client, log_file):
    """One log, two tracks, and the record must say which -- and carry that track's columns.

    Drift monitoring reads this file per track. A fraud record carrying credit's column names, or
    no track at all, would be compared against the wrong reference distribution and report drift
    that is really a schema mix-up.
    """
    both_tracks_client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST)

    record = json.loads(log_file.read_text().splitlines()[0])
    assert record["track"] == "fraud"
    assert sorted(record["features"]) == sorted(main.FEATURE_COLUMNS["fraud"])


@pytest.mark.parametrize("track", ["credit", "fraud"])
def test_request_aliases_cover_the_feature_contract(track):
    """A request model's serialization aliases must equal its track's feature columns exactly.

    This is the guard that lets ``FraudPredictRequest`` declare 28 near-identical
    ``serialization_alias="V17"`` lines by hand. Written-out aliases keep the contract readable
    and statically typed; an off-by-one among them (``V17`` where ``V18`` belongs) would produce
    one NaN column and one column the model never asked for, and ``reindex`` fills rather than
    raises -- so it would score, return a plausible probability, and pass every test that does not
    compare the two lists.

    Order is asserted too, not just membership. The served frame is reindexed onto
    ``feature_columns``, so order is not load-bearing at request time -- but ``tests/test_skew.py``
    builds its direct-path frame straight from ``model_dump(by_alias=True)`` with no reindex, and
    there order is exactly what MLflow's signature enforcement sees.
    """
    from src.api.schemas import CreditPredictRequest, FraudPredictRequest
    from src.features.specs import get_feature_spec

    models = {"credit": CreditPredictRequest, "fraud": FraudPredictRequest}
    examples = {"credit": CREDIT_EXAMPLE_REQUEST, "fraud": FRAUD_EXAMPLE_REQUEST}

    aliased = list(models[track](**examples[track]).model_dump(by_alias=True))
    assert aliased == list(get_feature_spec(track).feature_columns), (
        f"{track}: request aliases do not match the feature contract. "
        f"Extra in request: {set(aliased) - set(get_feature_spec(track).feature_columns)}; "
        f"missing: {set(get_feature_spec(track).feature_columns) - set(aliased)}"
    )
