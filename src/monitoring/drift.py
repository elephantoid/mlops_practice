"""Feature drift detection with Evidently, exported to Prometheus via the Pushgateway.

Run from the repo root::

    uv run python -m src.monitoring.drift --source synthetic
    uv run python -m src.monitoring.drift --source logs --push

Two comparison sources:

``synthetic``
    The blueprint's documented simulation -- ``MonthlyCharges`` shifted +15% and sampled
    to 500 rows, standing in for a price-hike event. This is what Milestone 5 asks for.
``logs``
    The real request features recorded by ``src/api/prediction_log.py``. Drift here is
    genuine traffic drift rather than a simulation.

The gauge reaches Prometheus through a Pushgateway because this is a batch job: it runs,
computes, and exits, so there is no long-lived target for Prometheus to scrape.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# nltk 3.10.1 installs an import hook that blocks nltk-initiated imports resolving from the
# current working directory. Because .venv lives inside this repo, a legitimately installed
# `regex` looks like a cwd import and `import evidently` dies. It is a false positive
# caused by the venv layout, not a real shadowing attempt. PYTHONSAFEPATH -- the fix the
# error message suggests -- does not help and would break `import src.…` as well.
# Set before the evidently import, and in the module rather than the environment so the
# script behaves identically from a shell, a container, or an Airflow task.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "latest.parquet"
PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")
PUSH_JOB = "churnwatch-drift"

# Excluded from both frames. customerID is unique per row, so a high-cardinality identifier
# always registers as drifted and inflates the share (0.095 vs 0.048 measured). Churn is the
# label, absent from the prediction log, and keeping it would make the two sources
# incomparable.
NON_FEATURE_COLUMNS = ["customerID", "Churn"]

DRIFT_COUNT_METRIC = "evidently:metric_v2:DriftedColumnsCount"
VALUE_DRIFT_METRIC = "evidently:metric_v2:ValueDrift"

MONTHLY_CHARGES_UPLIFT = 1.15
SYNTHETIC_SAMPLE_SIZE = 500
RANDOM_STATE = 42

# The only accepted values. Validated rather than pattern-matched: the dispatch below used
# to read ``if source == "synthetic" else logged_current()``, so every other string -- a
# typo like "log", or a capitalised "Synthetic" -- silently compared against the wrong
# dataset and reported a number answering a different question. The CLI was safe because
# argparse constrains it; the retrain DAG passes ``dag_run.conf`` straight through and was
# not.
DRIFT_SOURCES = ("synthetic", "logs")


class InsufficientCurrentData(RuntimeError):
    """Not enough current data to compute a meaningful comparison.

    Separate from an ordinary failure because the weekly DAG must be able to tell the two
    apart. On a freshly deployed service "no traffic logged yet" is the expected state, not
    an error, and failing the monitor task every week until the first hundred requests
    arrive would train whoever is on call to ignore it. Everything else still raises
    normally.
    """


# Drift on a tiny sample is not a weak signal, it is a wrong one. Measured: 40 identical
# requests report drift_share 1.0 (every column is a point mass, so every column looks
# maximally shifted), while 300 varied rows drawn from the training set report 0.0. Since
# Milestone 4 retrains when drift_share > 0.2, an unguarded run over a quiet hour would
# trigger a spurious retrain on nothing.
MIN_CURRENT_ROWS = 100


def load_reference(path: Path = REFERENCE_PATH) -> pd.DataFrame:
    """Load the training snapshot, reduced to the columns the model actually sees."""
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found at {path}. Run the ingest step.")

    frame = pd.read_parquet(path)
    return frame.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")


def synthetic_current(reference: pd.DataFrame) -> pd.DataFrame:
    """Simulate a price-hike batch: MonthlyCharges +15%, sampled to 500 rows.

    Documented simulation, not fabrication -- the shift is deliberate and reproducible so
    the drift detector has a known-positive case to prove itself against.
    """
    drifted = reference.copy()
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * MONTHLY_CHARGES_UPLIFT
    return drifted.sample(SYNTHETIC_SAMPLE_SIZE, random_state=RANDOM_STATE)


def logged_current(path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """Rebuild a feature frame from the prediction log."""
    if not path.exists():
        raise InsufficientCurrentData(
            f"No prediction log at {path}. Serve some traffic before comparing against it."
        )

    rows = [json.loads(line)["features"] for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        # An empty frame would produce a report full of NaNs that looks like a result.
        raise InsufficientCurrentData(f"Prediction log at {path} is empty; nothing to compare.")

    return pd.DataFrame(rows).drop(columns=NON_FEATURE_COLUMNS, errors="ignore")


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Any:
    """Run the drift report. Returns an Evidently Snapshot.

    In evidently 0.7 ``Report.run`` returns the Snapshot and the Report itself holds no
    results -- the 0.4 pattern of running then reading the report is gone. ``include_tests``
    belongs on the Report; setting it only on the preset leaves ``tests`` empty.
    """
    report = Report(metrics=[DataDriftPreset()], include_tests=True)
    return report.run(current_data=current, reference_data=reference)


def summarise(snapshot: Any) -> dict[str, Any]:
    """Pull the numbers worth exporting out of a Snapshot.

    ``as_dict()`` does not exist in 0.7; the shape is
    ``dict()["metrics"][i]["value"]["share"]`` on the DriftedColumnsCount entry.
    """
    metrics = snapshot.dict()["metrics"]

    drift_counts = [m for m in metrics if m["config"]["type"] == DRIFT_COUNT_METRIC]
    if not drift_counts:
        raise RuntimeError("DriftedColumnsCount missing from the report; Evidently API changed.")

    value = drift_counts[0]["value"]

    # Per-column scores come back as numpy floats, which prometheus_client will not accept.
    drifted_columns = {
        m["config"]["column"]: float(m["value"])
        for m in metrics
        if m["config"]["type"] == VALUE_DRIFT_METRIC and m["value"] > m["config"]["threshold"]
    }

    return {
        "drift_share": float(value["share"]),
        "drifted_count": int(value["count"]),
        "drifted_columns": drifted_columns,
    }


def push_metrics(summary: dict[str, Any], gateway: str = PUSHGATEWAY_URL) -> None:
    """Push the drift gauges to the Pushgateway.

    A fresh registry per call rather than the process-global default: a batch job should
    publish exactly what it just computed, with no leftover collectors from imports.
    """
    registry = CollectorRegistry()

    Gauge(
        "churnwatch_drift_share",
        "Fraction of features detected as drifted",
        registry=registry,
    ).set(summary["drift_share"])

    Gauge(
        "churnwatch_drifted_features_total",
        "Number of features detected as drifted",
        registry=registry,
    ).set(summary["drifted_count"])

    # Without this a stale drift job is indistinguishable from a healthy one reporting no
    # drift: the Pushgateway keeps serving the last value forever, so "0.0" could mean
    # "nothing drifted" or "this stopped running a week ago".
    Gauge(
        "churnwatch_drift_last_run_timestamp_seconds",
        "Unix timestamp of the last completed drift run",
        registry=registry,
    ).set(time.time())

    push_to_gateway(gateway, job=PUSH_JOB, registry=registry)
    logger.info("Pushed drift metrics to %s as job=%s", gateway, PUSH_JOB)


def run(
    source: str,
    push: bool,
    out_dir: Path = REPORTS_DIR,
    min_rows: int = MIN_CURRENT_ROWS,
) -> dict[str, Any]:
    """Compute drift for ``source``, write the HTML report, optionally push metrics."""
    if source not in DRIFT_SOURCES:
        raise ValueError(f"Unknown drift source {source!r}; expected one of {list(DRIFT_SOURCES)}")

    reference = load_reference()
    current = synthetic_current(reference) if source == "synthetic" else logged_current()

    if len(current) < min_rows:
        raise InsufficientCurrentData(
            f"Only {len(current)} current rows; need at least {min_rows} for a meaningful "
            f"comparison. Below this the sample is too homogeneous and every feature reads "
            f"as drifted -- which would trip the retrain threshold on no real signal."
        )

    logger.info("reference %s rows, current %s rows (%s)", len(reference), len(current), source)

    snapshot = compute_drift(reference, current)
    summary = summarise(snapshot)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"drift_{source}_{stamp}.html"
    snapshot.save_html(str(report_path))

    logger.info(
        "drift_share %.4f across %d/%d features | drifted: %s",
        summary["drift_share"],
        summary["drifted_count"],
        len(reference.columns),
        ", ".join(summary["drifted_columns"]) or "none",
    )
    logger.info("Report written to %s", report_path)

    if push:
        push_metrics(summary)

    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Detect feature drift against training data.")
    parser.add_argument(
        "--source",
        choices=DRIFT_SOURCES,
        default="synthetic",
        help="synthetic: simulated +15%% MonthlyCharges batch. logs: real served requests.",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push gauges to the Pushgateway.",
    )
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="Report output directory.")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_CURRENT_ROWS,
        help="Refuse to compute drift on fewer current rows than this.",
    )
    args = parser.parse_args()

    run(source=args.source, push=args.push, out_dir=args.out, min_rows=args.min_rows)


if __name__ == "__main__":
    main()
