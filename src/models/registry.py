"""The one way to reach the MLflow backend: which database, and a client bound to it.

MLflow's own default is ``sqlite:///<cwd>/mlflow.db``, which is this repo's registry only when
the cwd happens to be the repo root. Run from anywhere else and a lookup silently inspects a
different backend: from a fresh git worktree ``resolve_version`` reported ``Registered Model ...
not found`` -- loud, but naming the wrong cause -- and left an empty ``mlflow.db`` behind for
the next command to find as a valid, empty registry (4c4384c).

A bare ``MlflowClient()`` reads whatever URI the process last set, so it is only right when some
earlier call happened to configure it; ``promote_best`` once relied on exactly that (54cbf98).
:func:`registry_client` names its backend instead. ruff's TID251 rejects ``MlflowClient`` and
``set_tracking_uri`` everywhere but here.

The global still matters: ``mlflow.pyfunc.load_model``, ``register_model`` and the run/experiment
APIs read it, so entry points that use those call :func:`configure_tracking` too.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from src.data.tracks import PROJECT_ROOT

DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

# For annotations, so callers need not import the banned name to type a parameter.
RegistryClient = MlflowClient


def configure_tracking() -> str:
    """Point MLflow at the tracking backend and return the resolved URI.

    Every entry point calls this rather than relying on ``train()`` having done it once.
    Airflow runs each task in a separate process, so ``task_evaluate`` and ``task_promote``
    would otherwise fall back to the local sqlite default and read an empty registry --
    reporting "no incumbent" on a project that has been promoting models for weeks.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def registry_client() -> RegistryClient:
    """A client bound to the configured backend, for tracking and registry operations alike.

    Both URIs are passed: MLflow resolves the registry from ``MLFLOW_REGISTRY_URI`` before it
    falls back to the tracking URI, so naming only the tracking backend would still let the
    environment send registry reads somewhere else.
    """
    uri = configure_tracking()
    return MlflowClient(tracking_uri=uri, registry_uri=uri)
