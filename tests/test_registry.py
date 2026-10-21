"""Which MLflow backend ``src/models/registry.py`` hands out.

Both tests stay off the repo's own ``mlflow.db``: constructing a client opens its database
eagerly, and on a fresh clone that would create the empty registry 4c4384c removed.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import _model_registry, _tracking_service

from src.data.tracks import PROJECT_ROOT
from src.models import registry


@pytest.fixture(autouse=True)
def restore_mlflow_globals(monkeypatch):
    """Put MLflow's process-wide URIs back, so later tests do not inherit a tmp database.

    The environment is recorded first because ``mlflow.set_tracking_uri`` also writes
    ``MLFLOW_TRACKING_URI``: a test that moves the global before its own ``setenv`` would
    otherwise have monkeypatch "restore" the tmp database as the original value.
    """
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"):
        # setenv records the current value -- absence included -- for teardown; a delenv on an
        # unset variable records nothing, and MLflow's own write would then outlive the test.
        monkeypatch.setenv(var, os.environ.get(var, ""))
        if not os.environ[var]:
            monkeypatch.delenv(var)
    monkeypatch.setattr(
        _tracking_service.utils, "_tracking_uri", _tracking_service.utils._tracking_uri
    )
    monkeypatch.setattr(_model_registry.utils, "_registry_uri", _model_registry.utils._registry_uri)


def test_the_default_backend_is_the_repo_registry_whatever_the_cwd(monkeypatch, tmp_path):
    """Anchored to the repo root, not MLflow's ``sqlite:///<cwd>/mlflow.db`` default.

    From a fresh worktree that default read an empty database and reported the model as not
    found (4c4384c). Only the resolved string is checked, so no database is opened.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    assert registry.configure_tracking() == f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def test_a_registry_client_reads_the_configured_backend_whatever_the_process_set_before(
    monkeypatch, tmp_path
):
    """Registry reads land in the configured database, not in one the process was left pointing at.

    Two ways a client goes elsewhere, both set up here: the process-wide URI still names another
    database (a bare ``MlflowClient()`` built before anyone configured tracking reads it --
    54cbf98's ``promote_best`` relied on someone having configured it), and
    ``MLFLOW_REGISTRY_URI`` names another, which MLflow prefers over the tracking URI for
    registry operations.
    """
    own: Path = tmp_path / "own.db"
    other: Path = tmp_path / "other.db"
    mlflow.set_tracking_uri(f"sqlite:///{other}")  # noqa: TID251 - the stale state under test
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{own}")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"sqlite:///{other}")

    registry.registry_client().search_registered_models()

    assert own.exists()
    assert not other.exists(), "a registry read went to a database nobody configured"
