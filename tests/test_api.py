"""API contract tests.

These are deliberately hermetic: ``load_model`` is stubbed, so the suite needs no
``mlflow.db`` and no populated registry. Milestone 3 runs this in GitHub Actions, where
neither exists. That the *real* model serves correctly is verified by running the app
against the registry, not here -- these tests guard the request/response contract.

``TestClient`` is synchronous, so no ``pytest-asyncio`` is involved.
"""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main, prediction_log
from src.api.schemas import CREDIT_EXAMPLE_REQUEST, FRAUD_EXAMPLE_REQUEST, CreditPredictRequest
from src.features.specs import CREDIT_FEATURES
from tests.stubs import STUB_PROBABILITY, ExplodingModel, StubModel, StubRawPipeline


def test_a_credit_score_above_the_decline_cut_is_declined_with_the_full_response(client):
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


def test_credit_carries_exactly_three_reason_codes_named_for_display(client):
    """DoD (3): three reasons, in words a reviewer can read, with the applicant's value shown.

    The stub's contributions are weighted toward the first contract columns, so the expected
    three are known without recomputing a ranking here.
    """
    body = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()

    columns = CREDIT_FEATURES.feature_columns[:3]
    assert [code["feature"] for code in body["reason_codes"]] == [
        CREDIT_FEATURES.display_name(column) for column in columns
    ]
    assert all(code["contribution"] > 0 for code in body["reason_codes"]), (
        "a score above 0.5 must be explained by contributions that raise risk"
    )
    sent = CreditPredictRequest(**CREDIT_EXAMPLE_REQUEST).model_dump(by_alias=True)
    assert [code["value"] for code in body["reason_codes"]] == [
        CREDIT_FEATURES.display_value(column, sent[column]) for column in columns
    ]


def test_reason_codes_are_given_for_an_approval_too(monkeypatch, log_file):
    """Every credit decision carries three, approve included: one contract shape, not two."""
    low = 0.01

    class Approves(StubModel):
        def predict(self, features):
            return np.array([[1 - low, low]] * len(features))

        def get_raw_model(self):
            return StubRawPipeline(low)

    monkeypatch.setattr(main, "load_model", lambda *a, **k: (Approves(), "test-1"))
    with TestClient(main.app) as client:
        body = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()

    assert body["decision"] == "approve"
    assert len(body["reason_codes"]) == 3
    assert all(code["contribution"] < 0 for code in body["reason_codes"])


def test_a_failed_explanation_is_a_counted_500_rather_than_an_empty_list(monkeypatch, log_file):
    """Credit's ``[]`` would read like fraud's permanent ``[]``, so a failure has to be loud.

    The model passes the startup check on its logged example, then serves a probability its own
    contributions do not reconstruct -- the shape of a reason computed on a different frame from
    the one scored. Nothing is logged as served, because nothing was.
    """

    class DisagreesOnLiveTraffic(StubModel):
        def predict(self, features):
            if features.equals(self.input_example):
                return super().predict(features)
            return np.array([[0.8, 0.2]] * len(features))

    monkeypatch.setattr(main, "load_model", lambda *a, **k: (DisagreesOnLiveTraffic(), "test-1"))
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST)
        metrics = client.get("/metrics").text

    assert response.status_code == 500
    assert 'riskwatch_requests_total{endpoint="/predict/credit",status="500"}' in metrics
    assert not log_file.exists() or log_file.read_text() == ""


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


def test_predict_request_ids_are_unique(client):
    """Two identical requests must be individually traceable in the prediction log."""
    first = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()["request_id"]
    second = client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).json()["request_id"]
    assert first != second


def test_predict_rejects_a_missing_field(client):
    payload = {k: v for k, v in CREDIT_EXAMPLE_REQUEST.items() if k != "amt_credit"}
    response = client.post("/predict/credit", json=payload)
    assert response.status_code == 422
    assert any(err["loc"][-1] == "amt_credit" for err in response.json()["detail"])


def test_predict_rejects_an_out_of_range_value(client):
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


def test_predict_rejects_an_unknown_category(client):
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


def test_health_is_ok_and_lists_each_loaded_track_version(client):
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


def test_latency_buckets_read_any_quantile_from_half_a_millisecond_to_35_to_within_25_percent():
    """The derivation rule, not the measurement: a quantile is only as precise as its bucket.

    Measured p50s -- 2.45 ms without reason codes, 4.47 ms with them -- sit mid-range, so a
    slower runtime moves them along the ladder rather than off it.
    """
    buckets = main.LATENCY_BUCKETS
    assert list(buckets) == sorted(set(buckets)), "buckets must be strictly increasing"

    fine = [b for b in buckets if b <= 0.035]
    assert fine[0] <= 0.0005 and fine[-1] >= 0.034
    widest = max(upper / lower for lower, upper in pairwise(fine))
    assert widest <= 1.25 + 1e-3, f"a bucket below 35 ms is {widest:.0%} of its lower edge"


def test_unmatched_paths_share_one_metric_label(client):
    """Labelling by raw path would mint a series per URL a scanner probes."""
    for path in ("/nope", "/.env", "/wp-admin/setup-config.php"):
        assert client.get(path).status_code == 404

    metrics = client.get("/metrics").text
    assert 'endpoint="unmatched"' in metrics
    assert ".env" not in metrics
    assert "wp-admin" not in metrics


def test_the_logged_prediction_agrees_with_the_response(client, log_file):
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


# --- The fraud endpoint (Step 9) ----------------------------------------------------------


def test_fraud_answers_on_the_same_response_model_as_credit(both_tracks_client):
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


def test_fraud_carries_no_reason_codes(both_tracks_client):
    """Empty here is permanent, and a different claim from a failed credit explanation.

    ``V1``..``V28`` are unlabelled principal components -- the ULB researchers never published the
    loadings -- so a contribution against them explains nothing a caller can act on. DoD (3) is
    recorded credit-only for exactly this reason, and ``FRAUD_FEATURES.reason_codes`` says so.
    """
    body = both_tracks_client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST).json()
    assert body["reason_codes"] == []


def test_fraud_rejects_the_column_it_validates_but_does_not_model(both_tracks_client):
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
