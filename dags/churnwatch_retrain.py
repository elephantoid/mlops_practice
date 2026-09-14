"""Weekly retraining loop: ingest -> train -> evaluate -> promote -> monitor.

Milestone 4. This file is deliberately thin. Every decision it makes lives in
``src/pipelines/retrain.py``, which the hermetic test suite can import; Airflow is not
installed in the uv venv (see ``CLAUDE.md``), so anything written *here* is untestable by
``make check``. Wiring belongs here, judgement does not.

Two Airflow-specific shapes worth knowing before editing:

* **Project imports happen inside the task bodies, not at module scope.** The scheduler
  re-parses every DAG file on a short loop; a top-level ``import lightgbm`` would drag
  MLflow, LightGBM and Evidently into each parse and turn a metadata operation into a
  multi-second one. The tasks run in their own processes, where the cost is paid once.
* **``src`` resolves because the repo is mounted and on ``PYTHONPATH``**
  (``docker-compose.airflow.yml`` sets ``PYTHONPATH=/opt/airflow/project``). This DAG
  cannot run against a bare ``apache/airflow`` image.

Run parameters, via ``dag_run.conf``:

``drift_source``
    ``"logs"`` (the default for scheduled runs) or ``"synthetic"``. Validated by the drift
    module, which rejects anything else rather than quietly falling back to one of them.
    Pass ``"synthetic"`` by hand to prove the detector still fires on a known-positive
    batch; do not schedule it, for the reason on :data:`SCHEDULED_DRIFT_SOURCE`.
``triggered_by_drift``
    Set automatically on a drift-triggered run. Its only job is to stop that run from
    triggering another; see :func:`task_monitor`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

DAG_ID = "churnwatch_retrain"

# What a *scheduled* run compares against. Deliberately not "synthetic": that batch shifts
# MonthlyCharges by construction, so it always reports drift on a watched column, so every
# weekly run would trigger a second full 14-config sweep over identical data -- twice the
# compute, forever, on a signal that was manufactured rather than observed. Synthetic is a
# known-positive fixture for proving the detector works, which is a manual act; monitoring
# production means looking at production traffic.
SCHEDULED_DRIFT_SOURCE = "logs"

DEFAULT_ARGS = {
    # One retry: the failures worth retrying here are transient (the MLflow server not yet
    # up, a Pushgateway blip). A genuinely broken sweep fails the same way twice and costs
    # one extra run of the grid, which is minutes, not hours.
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id=DAG_ID,
    schedule="@weekly",
    # A fixed past date, not days_ago(): a start_date that moves with the clock makes the
    # first scheduled interval unreproducible. Timezone-aware because Airflow warns on
    # naive datetimes and resolves them against its own configured tz, not yours.
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    # No backfill. Retraining "the week of 2026-03-01" is meaningless -- ingest always reads
    # the current raw CSV, so every backfilled run would train on today's data and race to
    # promote into the same alias.
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churnwatch", "mlops", "m4"],
    doc_md=__doc__,
)
def churnwatch_retrain() -> None:
    """Retrain, promote if it earned it, then check whether the data has moved."""

    @task
    def task_ingest() -> str:
        """Validate the raw CSV into a fresh timestamped parquet snapshot."""
        from src.data.ingest import ingest

        return str(ingest())

    @task
    def task_train(data_path: str) -> list[str]:
        """Run the 14-config sweep. Promotes nothing -- that is task_promote's call."""
        from pathlib import Path

        from src.models.train import sweep

        return sweep(Path(data_path))

    @task
    def task_evaluate(run_ids: list[str]) -> dict[str, Any]:
        """Score the sweep's winner against the model currently serving production."""
        from src.models.train import best_finished_run
        from src.pipelines.retrain import incumbent_auc, should_promote

        best = best_finished_run(run_ids)
        candidate = float(best["metrics.cv_auc_mean"])
        incumbent = incumbent_auc()

        return {
            "run_ids": run_ids,
            "candidate_auc": candidate,
            "incumbent_auc": incumbent,
            "promote": should_promote(candidate, incumbent),
        }

    @task
    def task_promote(decision: dict[str, Any]) -> str:
        """Move the production alias, or skip loudly.

        Skipped rather than failed: a candidate that did not beat the incumbent is the
        system working correctly. Failing here would page someone every time the model
        held steady.
        """
        from src.models.train import promote_best

        if not decision["promote"]:
            raise AirflowSkipException(
                f"Candidate cv_auc {decision['candidate_auc']:.4f} did not beat incumbent "
                f"{decision['incumbent_auc']} by the required margin; alias unchanged."
            )

        version = promote_best(decision["run_ids"])
        logger.info("Promoted %s v%s to @production", version.name, version.version)
        return str(version.version)

    @task.short_circuit(
        # NONE_FAILED, not the default ALL_SUCCESS: task_promote *skips* whenever the
        # candidate did not earn the alias, and a skip propagates. Drift is worth measuring
        # either way -- a model held steady is exactly when you want to know the data moved.
        #
        # Not NONE_FAILED_MIN_ONE_SUCCESS either: that additionally demands one upstream
        # success, and task_promote is the only upstream, so a skipped promotion would take
        # the monitoring half of the pipeline down with it.
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    def task_monitor(**context: Any) -> bool:
        """Compute drift, push the gauges, and decide whether to retrain again.

        Returns whether to trigger a follow-up run; short-circuiting skips the trigger
        downstream when it returns False.

        **The guard matters.** This DAG *is* the retraining, and retraining does not move
        the reference distribution -- so a run that triggers itself on drift would find the
        same drift next time and trigger again, forever. A drift-triggered run therefore
        never triggers another: one hop, then the weekly schedule takes over.
        """
        from src.monitoring.drift import InsufficientCurrentData, run
        from src.pipelines.retrain import should_retrain

        conf = (context["dag_run"].conf or {}) if context.get("dag_run") else {}
        # `or` rather than a .get default: a conf carrying an explicit None must fall back
        # too, and a scheduled run has no conf at all.
        source = conf.get("drift_source") or SCHEDULED_DRIFT_SOURCE

        try:
            summary = run(source=source, push=True)
        except InsufficientCurrentData as exc:
            # Expected on a service that has not served enough traffic yet. Not a failure,
            # and not a reason to retrain -- there is simply nothing to compare against.
            logger.info("No drift verdict this run: %s", exc)
            return False

        if not should_retrain(summary):
            return False

        if conf.get("triggered_by_drift"):
            logger.info("Drift persists, but this run was itself drift-triggered; not chaining.")
            return False

        return True

    retrain_on_drift = TriggerDagRunOperator(
        task_id="task_trigger_retrain",
        trigger_dag_id=DAG_ID,
        # drift_source is carried across rather than left to default. A run started with
        # drift_source="logs" detects drift in real traffic; without this the follow-up it
        # triggers would silently measure the *synthetic* batch instead, so the retrain
        # would be justified by one dataset and verified against another. `conf` is a
        # templated field, which is what lets this read the current run's value.
        conf={
            "triggered_by_drift": True,
            # The fallback must be SCHEDULED_DRIFT_SOURCE, not a second literal. Hardcoding
            # one here is how this bug came back: a scheduled run has no conf, so the
            # template falls back -- and a stale literal would hand the follow-up a
            # different source than the run that triggered it, which is exactly the defect
            # propagating drift_source was meant to fix.
            "drift_source": (
                f"{{{{ dag_run.conf.get('drift_source', '{SCHEDULED_DRIFT_SOURCE}') }}}}"
            ),
        },
        # Fire and forget. Waiting would hold a worker slot for the whole sweep, and
        # max_active_runs=1 means the new run cannot start until this one ends anyway.
        wait_for_completion=False,
    )

    snapshot = task_ingest()
    run_ids = task_train(snapshot)
    decision = task_evaluate(run_ids)
    promoted = task_promote(decision)

    promoted >> task_monitor() >> retrain_on_drift


churnwatch_retrain()
