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
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.schemas import EXAMPLE_REQUEST, PredictRequest


@pytest.fixture(scope="module")
def registered_model():
    """The real pyfunc model, or a skip if there is no registry to load it from."""
    try:
        model, version = main.load_model()
    except Exception as exc:  # noqa: BLE001 -- any failure to resolve the URI means "skip"
        pytest.skip(f"no model at {main.MODEL_URI} ({type(exc).__name__}: {exc})")
    return model, version


def test_served_probability_matches_the_model(registered_model, tmp_path, monkeypatch):
    model, version = registered_model
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))

    # One model, two callers. Without this the lifespan would resolve the alias a second
    # time, and a comparison across two loads cannot tell a frame-construction bug from a
    # model that changed underneath it -- which is the only thing this test is here to see.
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (model, version))

    # Direct path: no reindex, no pinned order.
    raw_row = PredictRequest(**EXAMPLE_REQUEST).model_dump(by_alias=True)
    direct = float(model.predict(pd.DataFrame([raw_row]))[0][1])

    # Served path: the whole chain -- validation, alias mapping, reindex, pyfunc wrapper.
    with TestClient(main.app) as client:
        response = client.post("/predict", json=EXAMPLE_REQUEST)
        assert response.status_code == 200
        served = response.json()["churn_probability"]

    # Exact, not approximate. These are the same float from the same model; anything that
    # makes them merely close has changed what the model was fed.
    assert served == direct, f"skew: API returned {served!r}, model returned {direct!r}"
