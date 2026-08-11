"""API contract tests.

These are deliberately hermetic: ``load_model`` is stubbed, so the suite needs no
``mlflow.db`` and no populated registry. Milestone 3 runs this in GitHub Actions, where
neither exists. That the *real* model serves correctly is verified by running the app
against the registry, not here -- these tests guard the request/response contract.

``TestClient`` is synchronous, so no ``pytest-asyncio`` is involved.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main, prediction_log
from src.api.schemas import EXAMPLE_REQUEST

STUB_PROBABILITY = 0.73


class StubModel:
    """Stands in for the pyfunc model, returning [[p_no_churn, p_churn]] like the real one.

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
    response = client.post("/predict", json=EXAMPLE_REQUEST)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"churn_probability", "prediction", "model_version", "request_id"}
    assert body["churn_probability"] == pytest.approx(STUB_PROBABILITY)
    assert body["prediction"] == "churn"
    assert body["model_version"] == "test-1"
    assert body["request_id"]


def test_predict_request_ids_are_unique(client):
    """Two identical requests must be individually traceable in the prediction log."""
    first = client.post("/predict", json=EXAMPLE_REQUEST).json()["request_id"]
    second = client.post("/predict", json=EXAMPLE_REQUEST).json()["request_id"]
    assert first != second


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in EXAMPLE_REQUEST.items() if k != "tenure"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(err["loc"][-1] == "tenure" for err in response.json()["detail"])


def test_predict_out_of_range_returns_422(client):
    response = client.post("/predict", json={**EXAMPLE_REQUEST, "tenure": -5})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "greater_than_equal"


def test_predict_unknown_category_returns_422(client):
    """A category the model never saw would be encoded as -1 and scored anyway."""
    response = client.post("/predict", json={**EXAMPLE_REQUEST, "contract": "Yearly"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "literal_error"


def test_predict_rejects_camelcase_input(client):
    """The public contract is snake_case only -- the model's column names must not leak."""
    payload = {**EXAMPLE_REQUEST}
    payload["MonthlyCharges"] = payload.pop("monthly_charges")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_field(client):
    """extra='forbid': silently ignoring a field would mislead the caller."""
    response = client.post("/predict", json={**EXAMPLE_REQUEST, "customerID": "1234-ABCDE"})
    assert response.status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-1"
    assert body["uptime_seconds"] >= 0


def test_metrics_exposes_prediction_counter(client):
    client.post("/predict", json=EXAMPLE_REQUEST)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=1.0.0; charset=utf-8"
    assert "churnwatch_predictions_total" in response.text
    assert "churnwatch_request_latency_seconds" in response.text


def test_unmatched_paths_share_one_metric_label(client):
    """Labelling by raw path would mint a series per URL a scanner probes."""
    for path in ("/nope", "/.env", "/wp-admin/setup-config.php"):
        assert client.get(path).status_code == 404

    metrics = client.get("/metrics").text
    assert 'endpoint="unmatched"' in metrics
    assert ".env" not in metrics
    assert "wp-admin" not in metrics


def test_prediction_is_logged(client, log_file):
    response = client.post("/predict", json=EXAMPLE_REQUEST).json()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert set(record) == {
        "timestamp",
        "request_id",
        "model_version",
        "input_hash",
        "churn_probability",
        "prediction",
        "features",
    }
    # The log must agree with what the caller was told, or analysis built on it is fiction.
    assert record["request_id"] == response["request_id"]
    assert record["churn_probability"] == response["churn_probability"]
    assert record["model_version"] == "test-1"


def test_logged_features_match_the_model_contract(client, log_file):
    """Evidently compares these against training data, so the names must be identical."""
    client.post("/predict", json=EXAMPLE_REQUEST)

    record = json.loads(log_file.read_text().splitlines()[0])
    assert sorted(record["features"]) == sorted(main.FEATURE_COLUMNS)


def test_each_prediction_appends_one_line(client, log_file):
    for _ in range(3):
        client.post("/predict", json=EXAMPLE_REQUEST)
    assert len(log_file.read_text().splitlines()) == 3


def test_input_hash_is_order_independent_and_discriminating():
    a = prediction_log.input_hash({"tenure": 24, "MonthlyCharges": 65.5})
    reordered = prediction_log.input_hash({"MonthlyCharges": 65.5, "tenure": 24})
    different = prediction_log.input_hash({"tenure": 25, "MonthlyCharges": 65.5})

    assert a == reordered
    assert a != different


def test_logging_failure_does_not_break_prediction(client, monkeypatch, tmp_path):
    """Observability must degrade, not take down serving."""
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "nope" / "x.jsonl"))
    monkeypatch.setattr(
        prediction_log, "input_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    response = client.post("/predict", json=EXAMPLE_REQUEST)
    assert response.status_code == 200
    assert response.json()["churn_probability"] == pytest.approx(STUB_PROBABILITY)


def test_failed_prediction_is_counted(monkeypatch):
    """A 500 must still land in the metrics, or error rate is uncomputable."""
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (ExplodingModel(), "test-1"))
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=EXAMPLE_REQUEST)
        assert response.status_code == 500

        metrics = client.get("/metrics").text
        assert 'churnwatch_requests_total{endpoint="/predict",status="500"}' in metrics
