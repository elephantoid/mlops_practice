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

import json
from datetime import UTC, datetime, timedelta

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


class TestPredictionLogWindow:
    """The logs source must be bounded in time.

    Milestone 4 made `logs` the scheduled default, and the read was unbounded: every weekly
    run parsed the whole append-only log. Two consequences, the second worse than the first
    -- cost grows forever, and traffic from months ago keeps outvoting current behaviour, so
    a drift signal that has already been dealt with never clears.
    """

    @staticmethod
    def _write(path, entries):
        path.write_text(
            "".join(
                json.dumps({"timestamp": ts.isoformat(), "features": feats}) + "\n"
                for ts, feats in entries
            )
        )
        return path

    def test_rows_outside_the_window_are_dropped(self, tmp_path):
        now = datetime(2026, 9, 14, tzinfo=UTC)
        log = self._write(
            tmp_path / "p.jsonl",
            [
                (now - timedelta(days=90), {"tenure": 1}),
                (now - timedelta(days=8), {"tenure": 2}),
                (now - timedelta(days=1), {"tenure": 3}),
            ],
        )
        frame = drift.logged_current(log, window_days=7, now=now)
        assert frame["tenure"].tolist() == [3]

    def test_an_all_stale_log_reads_as_no_data_not_as_no_drift(self, tmp_path):
        """Silence must be distinguishable from "nothing changed".

        If a stale log returned an empty frame, drift would be computed over nothing and the
        run would report a number. It has to say "no data" instead.
        """
        now = datetime(2026, 9, 14, tzinfo=UTC)
        log = self._write(tmp_path / "p.jsonl", [(now - timedelta(days=60), {"tenure": 1})])
        with pytest.raises(drift.InsufficientCurrentData):
            drift.logged_current(log, window_days=7, now=now)

    def test_max_rows_keeps_the_newest(self, tmp_path):
        """The cap is a memory bound, and drift is a question about recent behaviour."""
        now = datetime(2026, 9, 14, tzinfo=UTC)
        log = self._write(
            tmp_path / "p.jsonl",
            [(now - timedelta(hours=n), {"tenure": n}) for n in range(10, 0, -1)],
        )
        frame = drift.logged_current(log, window_days=7, max_rows=3, now=now)
        assert frame["tenure"].tolist() == [3, 2, 1]

    def test_rows_without_a_timestamp_are_skipped(self, tmp_path):
        """Pre-timestamp records must not sneak the unbounded history back in."""
        now = datetime(2026, 9, 14, tzinfo=UTC)
        log = tmp_path / "p.jsonl"
        log.write_text(
            json.dumps({"features": {"tenure": 1}})
            + "\n"
            + json.dumps({"timestamp": now.isoformat(), "features": {"tenure": 2}})
            + "\n"
        )
        frame = drift.logged_current(log, window_days=7, now=now)
        assert frame["tenure"].tolist() == [2]

    def test_the_window_matches_the_dag_schedule(self):
        """7 days because the DAG is @weekly -- the window covers traffic since the last
        run. A number chosen to match something real, not picked."""
        assert drift.DEFAULT_WINDOW_DAYS == 7
