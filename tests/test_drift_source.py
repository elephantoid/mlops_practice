"""The drift module's source contract.

Hermetic: every case here is rejected before any file is read, so no reference parquet and
no prediction log are needed.

This exists because Milestone 4 opened a second way into ``drift.run()``. The CLI is
constrained by argparse ``choices``, but the retrain DAG passes ``dag_run.conf`` through,
and the dispatch used to read ``if source == "synthetic" else logged_current()`` -- so any
value that was not exactly ``"synthetic"`` silently compared against the prediction log
instead. Validation that lives in only one of two entry points is not validation.
"""

from __future__ import annotations

import pytest

from src.monitoring import drift


@pytest.mark.parametrize("bad", ["log", "Synthetic", "SYNTHETIC", "logs ", "", "csv"])
def test_unknown_source_is_rejected(bad):
    """Rejected loudly rather than falling back to the other source."""
    with pytest.raises(ValueError, match="Unknown drift source"):
        drift.run(source=bad, push=False)


def test_the_accepted_values_are_the_ones_the_cli_offers():
    """One source of truth: argparse choices and the runtime guard cannot disagree."""
    assert drift.DRIFT_SOURCES == ("synthetic", "logs")


def test_missing_prediction_log_is_insufficient_data_not_an_error(tmp_path):
    """ "No traffic yet" must be distinguishable from a failure.

    The weekly DAG defaults to the logs source, so on a freshly deployed service this path
    is hit every run until real traffic accumulates. Failing the task there would train
    whoever is on call to ignore it.
    """
    with pytest.raises(drift.InsufficientCurrentData):
        drift.logged_current(tmp_path / "does-not-exist.jsonl")


def test_empty_prediction_log_is_insufficient_data(tmp_path):
    log = tmp_path / "predictions.jsonl"
    log.write_text("\n  \n")
    with pytest.raises(drift.InsufficientCurrentData):
        drift.logged_current(log)


def test_insufficient_data_is_not_caught_by_generic_error_handling():
    """It must stay distinguishable from ValueError, which the DAG lets fail."""
    assert issubclass(drift.InsufficientCurrentData, RuntimeError)
    assert not issubclass(drift.InsufficientCurrentData, ValueError)
