"""What the lifespan does with a model before it serves one.

A track whose model fails to load, cannot score its own logged example, or was logged against a
different feature contract is refused at startup, one track at a time, and ``/health`` reports
the service as degraded rather than ok. Split out of ``tests/test_api.py``, which keeps the
request/response contract; the stubs both use live in ``tests/stubs.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.schemas import CREDIT_EXAMPLE_REQUEST, FRAUD_EXAMPLE_REQUEST
from tests.stubs import StubModel, baked, example_frame


def test_an_unavailable_track_503s_without_disclosing_the_reason(monkeypatch, log_file, tmp_path):
    """The 503 body must name the track and nothing about the deployment.

    It used to interpolate `load_failures[track]` verbatim. That was survivable while those
    strings were mostly `MlflowException: Registered Model ... not found`, and stopped being
    survivable once `ModelConfigurationError` messages were written to be actionable for an
    operator: an unauthenticated request then returned the baked artifact's filesystem path, the
    `MODEL_URI_<TRACK>` variable names, which track the artifact holds, and the deployment shape.
    This service is headed for a public Cloud Run URL.

    The reason is not lost, it is addressed — the lifespan logs it per track at ERROR, and
    `app.state.load_failures` still carries it.
    """
    credit_dir = baked(tmp_path / "credit", "credit")

    monkeypatch.setattr(main, "MODEL_URI", "")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_CREDIT", str(credit_dir))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (StubModel(), "6"))

    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        response = client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST)
        assert response.status_code == 503

        detail = response.json()["detail"]
        assert "fraud" in detail, "the caller is still told which track is down"
        for leak in (
            str(credit_dir),
            "MODEL_URI_FRAUD",
            "MODEL_URI_CREDIT",
            "ModelConfigurationError",
        ):
            assert leak not in detail, f"the 503 body disclosed {leak!r}"

        # Addressed, not discarded: the operator-facing reason is still recorded server-side.
        assert "MODEL_URI_FRAUD" in client.app.state.load_failures["fraud"]


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


def test_the_lifespan_refuses_a_track_whose_model_cannot_score_its_example(monkeypatch, log_file):
    """The check must be *wired in*, not merely present.

    Review found that every startup test here routed `assert_model_can_score` through its
    skip-and-warn branch, because `StubModel` had no `input_example` -- so deleting the lifespan's
    call to it left the suite green. A guard against a healthy-looking service that 500s every
    request was itself unguarded.

    This asserts at the lifespan level: a model that loads, carries an example, and raises on it
    must take the track down at startup rather than at the first request.
    """

    class LoadsButCannotScore:
        def __init__(self):
            self.input_example = example_frame("credit")

        def predict(self, features):
            raise ValueError("Length mismatch: Expected axis has 26 elements, new values have 27")

    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (LoadsButCannotScore(), "5"))

    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_the_lifespan_serves_a_track_whose_model_can_score_its_example(monkeypatch, log_file):
    """The other half: the check must not refuse a healthy model.

    Without this, satisfying the test above by refusing everything would pass.

    ``load_model`` is stubbed like every other test in this file. The first version of this test
    left it real, so it passed in a populated checkout and failed in a fresh clone with
    ``Registered Model with name=riskwatch_credit not found`` -- breaking the hermeticity this
    module's docstring promises, and it would have failed in CI. Found by measuring a clean clone
    rather than by running the suite where the registry happens to exist.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (StubModel(), "6"))

    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["models"] == {"credit": "6"}


def test_a_model_that_cannot_score_its_own_example_is_refused_at_startup(monkeypatch):
    """Loading cleanly is not the same as being able to answer, and the gap is reachable.

    Removing the derived 27th feature left credit v1-v5 in exactly that state: their fitted
    `ColumnTransformer` expects 27 columns while the pipeline step -- cloudpickled *by
    reference*, so the artifact picks up today's code -- now emits 26. `load_model` succeeds,
    the logged signature is 26 wide either way so signature enforcement passes, and every
    request fails with a length mismatch. An operator moving the alias back during an incident
    gets a service reporting `ok` and 500ing everything.
    """

    class LoadsButCannotScore:
        input_example = "whatever the artifact logged"

        def predict(self, features):
            raise ValueError("Length mismatch: Expected axis has 26 elements, new values have 27")

    with pytest.raises(main.ModelConfigurationError, match="cannot score its own logged input"):
        main.assert_model_can_score("credit", "models:/riskwatch_credit/5", LoadsButCannotScore())


def test_input_example_is_a_real_pyfunc_attribute():
    """The one thing the stub tests cannot check: that our attribute name is MLflow's.

    `assert_model_can_score` reads `model.input_example` unguarded, so an MLflow rename becomes a
    loud per-track failure rather than a silent skip -- but nothing else in this file touches the
    real class, so every stub test would stay green while production refused every track. This
    pins the name against `PyFuncModel` itself.
    """
    import mlflow.pyfunc

    assert isinstance(getattr(mlflow.pyfunc.PyFuncModel, "input_example", None), property), (
        "PyFuncModel.input_example is gone or is no longer a property; "
        "assert_model_can_score reads it by that name"
    )


def _schema_model(names, *, track: str = "credit"):
    """A stand-in whose logged input schema names ``names``."""
    from mlflow.types import ColSpec, Schema

    class WithSchema:
        def __init__(self):
            self.input_example = example_frame(track)
            self.metadata = type(
                "Meta",
                (),
                {
                    "get_input_schema": lambda _self: Schema(
                        [ColSpec("double", name) for name in names]
                    )
                },
            )()

        def predict(self, features):
            return np.array([[0.5, 0.5]])

    return WithSchema()


def test_a_model_logged_against_a_different_contract_is_refused(monkeypatch):
    """Scoring its own example only proves the artifact agrees with itself.

    An artifact *narrower* than today's contract passes the smoke score perfectly, because MLflow
    drops a request column its signature does not know about -- with a warning and no error. A
    feature added to `feature_columns` would then be silently ignored by an older model while
    `/health` reported `ok`.

    Lives in its own function rather than inside `assert_model_can_score`, because it refuses on a
    criterion that name excludes -- the artifact here *can* score.
    """
    short = list(main.FEATURE_COLUMNS["credit"])[:-1]

    with pytest.raises(main.ModelConfigurationError, match="different feature contract") as raised:
        main.assert_model_matches_contract(
            "credit", "models:/riskwatch_credit/1", _schema_model(short)
        )
    dropped = list(main.FEATURE_COLUMNS["credit"])[-1]
    assert dropped in str(raised.value), "the message must name the feature the artifact lacks"


def test_the_lifespan_refuses_a_track_whose_model_predates_the_contract(monkeypatch, log_file):
    """The contract check must be *wired in*, like the smoke score beside it.

    Splitting `assert_model_can_score` in two reproduced the defect review caught last pass on the
    original: the new function had unit coverage and no test observed the lifespan calling it, so
    deleting the call site left the suite green. Third appearance of this class in this branch --
    a guard is not wired until a test fails when the call goes away.
    """
    short = list(main.FEATURE_COLUMNS["credit"])[:-1]

    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (_schema_model(short), "1"))

    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_a_schema_differing_only_in_order_is_accepted(monkeypatch):
    """MLflow reorders a request to the signature, so order alone is not a mismatch.

    The first version of this check compared lists, so a reordered schema was refused with
    "it does not know [] and expects []" -- a refusal naming nothing, under a parenthetical
    claiming ordering was already excluded. It would have rejected a sound artifact.
    """
    columns = list(main.FEATURE_COLUMNS["credit"])
    reordered = [columns[1], columns[0], *columns[2:]]
    assert reordered != columns

    main.assert_model_matches_contract(
        "credit", "models:/riskwatch_credit/6", _schema_model(reordered)
    )


def test_a_model_with_no_logged_schema_is_not_refused(monkeypatch):
    """Missing metadata is not a mismatch; the smoke score still has to pass."""

    class NoSchema:
        input_example = None
        metadata = None

    main.assert_model_matches_contract("credit", "/app/model", NoSchema())


def test_an_artifact_without_an_input_example_is_served_with_a_warning(monkeypatch, caplog):
    """No example is a missing check, not a broken model.

    Refusing here would make an older artifact unservable for lacking metadata it was never
    required to carry. The warning is what keeps the skip from being invisible.
    """

    class NoExample:
        input_example = None

        def predict(self, features):  # pragma: no cover - must never be reached
            raise AssertionError("predict must not be called when there is no example")

    with caplog.at_level("WARNING"):
        main.assert_model_can_score("credit", "/app/model", NoExample())

    assert "carries no input example" in caplog.text


def test_a_model_that_scores_its_example_is_accepted(monkeypatch):
    """The ordinary path must not cost a raise, and the example must actually be scored."""
    scored = []

    class Healthy:
        input_example = "example"

        def predict(self, features):
            scored.append(features)
            return [[0.5, 0.5]]

    main.assert_model_can_score("credit", "models:/riskwatch_credit@production", Healthy())

    assert scored == ["example"], "the logged example must be the thing scored"
