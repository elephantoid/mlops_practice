"""Fixtures shared by the API test modules: a redirected prediction log and stubbed clients."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import main
from tests.stubs import StubModel


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

    Separate from ``client`` because ``ENABLED_TRACKS`` defaults to credit alone in code; the
    serving image enables both (Step 14), but a fraud request against the code default still 503s
    by design. Two fixtures keep both facts
    testable: that the fraud endpoint works when the track is enabled, and that it refuses
    clearly when it is not.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setattr(main, "load_model", lambda *a, **k: (StubModel(), "test-1"))
    with TestClient(main.app) as test_client:
        yield test_client
