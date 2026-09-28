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
from src.api.schemas import CREDIT_EXAMPLE_REQUEST

STUB_PROBABILITY = 0.73


class StubModel:
    """Stands in for the pyfunc model, returning [[p_negative, p_positive]] like the real one.

    The assertions are the point, not the return value. ``reindex`` in the handler fills any
    column it cannot find with NaN, so a one-character typo in a ``serialization_alias``
    would produce a NaN feature, get quietly imputed by the pipeline, and score normally --
    passing every other test in this file while serving wrong predictions.
    """

    def predict(self, features):
        assert list(features.columns) == main.FEATURE_COLUMNS, "alias mapping drifted"
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


def test_predict_happy_path(client):
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
    # STUB_PROBABILITY sits above the credit decline threshold.
    assert body["decision"] == "decline"


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
    assert sorted(record["features"]) == sorted(main.FEATURE_COLUMNS)


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
