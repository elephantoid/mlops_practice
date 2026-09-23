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
from collections import deque
from datetime import UTC, datetime, timedelta
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

from src.api.prediction_log import LOG_TYPE
from src.features.specs import get_feature_spec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")
# One job per track. push_to_gateway REPLACES every metric under a job name, so a single
# shared job would mean each track's push silently erased the other's -- the last writer
# would look like the only track being monitored.
PUSH_JOB_PREFIX = "riskwatch-drift"

# Excluded from both frames. customerID is unique per row, so a high-cardinality identifier
# always registers as drifted and inflates the share (0.095 vs 0.048 measured). Churn is the
# label, absent from the prediction log, and keeping it would make the two sources
# incomparable.
# Derived per track from FeatureSpec rather than hardcoded. The measurement above is the
# reason this must not be a constant: it was taken on Telco columns, and a hardcoded
# ["customerID", "Churn"] against a credit frame drops nothing at all -- SK_ID_CURR stays
# in, registers as drifted on every run, and inflates the share that the retrain trigger
# reads.
DEFAULT_TRACK = "credit"

DRIFT_COUNT_METRIC = "evidently:metric_v2:DriftedColumnsCount"
VALUE_DRIFT_METRIC = "evidently:metric_v2:ValueDrift"

# The documented simulation standard, now expressed as (column, multiplier) per track
# instead of a hardcoded MonthlyCharges shift. AGENTS.md licenses a deliberate,
# reproducible perturbation applied ON TOP OF REAL DATA so a detector has a known-positive
# case to prove itself against; it does not license synthesising the base data or the
# class balance. Widening it from one Telco column to a per-track choice stays inside that
# licence and is recorded as a dated revision rather than inherited silently.
SYNTHETIC_UPLIFT = 1.15
SYNTHETIC_SAMPLE_SIZE = 500
RANDOM_STATE = 42

# Column each track perturbs for its synthetic batch. Chosen for being continuous, densely
# populated, and economically meaningful -- a shift in it is a plausible real-world event
# (a repricing, an income distribution moving) rather than an artefact.
SYNTHETIC_DRIFT_COLUMN = {
    "credit": "AMT_CREDIT",
    "fraud": "Amount",
}

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

# How far back the prediction log is read. Tied to the DAG's @weekly schedule rather than
# picked: the window should cover the traffic since the last run and no more. Reading the
# whole append-only log instead -- which is what it did until review -- grows without bound
# and, worse, keeps a drift signal alive forever: traffic from months ago would keep
# outvoting current behaviour long after the thing that caused it was fixed.
DEFAULT_WINDOW_DAYS = 7

# A hard ceiling regardless of the window, so one busy week cannot exhaust memory. Newest
# rows win: drift is a question about recent behaviour.
MAX_CURRENT_ROWS = 50_000


def reference_path(track: str = DEFAULT_TRACK) -> Path:
    """Per-track training snapshot. Flat single-track paths do not survive two tracks."""
    return PROCESSED_DIR / track / "latest.parquet"


# The default track's snapshot, as a module constant. Retained because the retrain DAG
# imports it by name to resolve a pre-ingest baseline, and because a caller that has no
# opinion about tracks should not have to form one. Anything that does care calls
# reference_path(track) instead -- this is the one-track convenience, not the source of
# truth.
REFERENCE_PATH = reference_path()


def push_job(track: str = DEFAULT_TRACK) -> str:
    """Pushgateway job name for one track."""
    return f"{PUSH_JOB_PREFIX}-{track}"


def non_feature_columns(track: str = DEFAULT_TRACK) -> list[str]:
    """Columns to drop before comparison, derived from the track's FeatureSpec.

    The id column is the one that matters. It is unique per row, so leaving it in makes it
    register as drifted on every single run -- measured at 0.095 against 0.048 on the
    Telco data -- and the retrain trigger reads exactly that number.
    """
    spec = get_feature_spec(track)
    return list(spec.non_feature_columns)


def load_reference(path: Path | None = None, track: str = DEFAULT_TRACK) -> pd.DataFrame:
    """Load the training snapshot, reduced to the columns the model actually sees.

    The reduction is not cosmetic. Evidently compares whatever columns it is given, and
    the prediction log carries only model inputs -- so a reference still holding the id,
    the label, or any unmodeled column would be compared against a narrower current frame
    and report drift for columns the model never sees.
    """
    resolved = path if path is not None else reference_path(track)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Reference data for track {track!r} not found at {resolved}. Run the ingest step."
        )

    frame = pd.read_parquet(resolved)
    frame = frame.drop(columns=non_feature_columns(track), errors="ignore")

    # Reduce to exactly the track's feature set. Home Credit's raw table carries 122
    # columns against a modeled subset of 26, and the schema permits rather than drops the
    # rest -- so without this the reference would be ~120 columns wide against a 26-column
    # prediction log.
    modeled = [c for c in get_feature_spec(track).feature_columns if c in frame.columns]
    return frame[modeled]


def synthetic_current(reference: pd.DataFrame, track: str = DEFAULT_TRACK) -> pd.DataFrame:
    """Simulate a shifted batch: one column lifted 15%, sampled to 500 rows.

    Documented simulation, not fabrication -- the shift is deliberate and reproducible so
    the drift detector has a known-positive case to prove itself against. The perturbed
    column is per track (AMT_CREDIT for credit, Amount for fraud) rather than the Telco
    MonthlyCharges this replaced.

    Raises rather than silently returning an unshifted frame when the column is absent: a
    "synthetic drift" batch with no drift in it would make the detector look broken when
    the harness was.
    """
    column = SYNTHETIC_DRIFT_COLUMN.get(track)
    if column is None:
        raise KeyError(
            f"no synthetic drift column configured for track {track!r}; "
            f"configured: {sorted(SYNTHETIC_DRIFT_COLUMN)}"
        )
    if column not in reference.columns:
        raise KeyError(
            f"synthetic drift column {column!r} is not in the {track!r} reference frame; "
            f"available: {sorted(reference.columns)[:10]}"
        )

    drifted = reference.copy()
    drifted[column] = drifted[column] * SYNTHETIC_UPLIFT

    # A sample larger than the frame raises in pandas; clamp so a small reference still
    # produces a usable batch rather than an error about sampling.
    size = min(SYNTHETIC_SAMPLE_SIZE, len(drifted))
    return drifted.sample(size, random_state=RANDOM_STATE)


def logged_current(
    path: Path = PREDICTION_LOG_PATH,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_rows: int = MAX_CURRENT_ROWS,
    now: datetime | None = None,
    track: str = DEFAULT_TRACK,
) -> pd.DataFrame:
    """Rebuild a feature frame from the recent tail of the prediction log.

    Bounded on purpose, in two ways. ``window_days`` keeps the comparison about *current*
    behaviour -- an unbounded read lets traffic from months ago keep a resolved drift
    signal alive indefinitely -- and ``max_rows`` caps memory when a single window is
    unusually busy. Rows are streamed rather than read whole so the file size does not
    have to fit in memory.

    ``now`` is injectable so the window is testable without waiting for the clock.

    Filters to the requested track: the log interleaves every track's predictions, and
    comparing a credit reference against a frame containing fraud rows would report drift
    on a schema difference rather than a distribution shift.
    """
    if not path.exists():
        raise InsufficientCurrentData(
            f"No prediction log at {path}. Serve some traffic before comparing against it."
        )

    cutoff = (now or datetime.now(UTC)) - timedelta(days=window_days)
    rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    undated = 0
    other_track = 0

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            # Skip anything that is not a prediction record. A GCS export of the deployed
            # service's stdout interleaves application logs with these lines, so the
            # marker is what makes that stream readable here without a separate sink.
            if "log_type" in record and record["log_type"] != LOG_TYPE:
                continue
            # Records written before the track field existed are treated as the default
            # track rather than dropped, so an older log is still usable.
            if record.get("track", DEFAULT_TRACK) != track:
                other_track += 1
                continue
            stamp = record.get("timestamp")
            if stamp is None:
                # Written before the timestamp field existed. Counted and dropped rather
                # than silently included, which would reintroduce the unbounded history.
                undated += 1
                continue
            if datetime.fromisoformat(stamp) >= cutoff:
                rows.append(record["features"])

    if undated:
        logger.warning("Skipped %d prediction log rows with no timestamp", undated)

    if not rows:
        # An empty frame would produce a report full of NaNs that looks like a result.
        # The message distinguishes "no traffic at all" from "traffic, but another
        # track's" -- on a two-track service those need different responses, and a single
        # message would send whoever is on call looking in the wrong place.
        detail = (
            f"No {track!r} predictions logged in {path} within the last {window_days} days"
            if not other_track
            else (
                f"No {track!r} predictions logged in {path} within the last {window_days} "
                f"days, though {other_track} row(s) for other tracks were skipped"
            )
        )
        raise InsufficientCurrentData(f"{detail}; nothing to compare.")

    return pd.DataFrame(list(rows)).drop(columns=non_feature_columns(track), errors="ignore")


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


def push_metrics(
    summary: dict[str, Any], gateway: str = PUSHGATEWAY_URL, track: str = DEFAULT_TRACK
) -> None:
    """Push the drift gauges to the Pushgateway.

    A fresh registry per call rather than the process-global default: a batch job should
    publish exactly what it just computed, with no leftover collectors from imports.
    """
    registry = CollectorRegistry()

    Gauge(
        "riskwatch_drift_share",
        "Fraction of features detected as drifted",
        registry=registry,
    ).set(summary["drift_share"])

    Gauge(
        "riskwatch_drifted_features_total",
        "Number of features detected as drifted",
        registry=registry,
    ).set(summary["drifted_count"])

    # Without this a stale drift job is indistinguishable from a healthy one reporting no
    # drift: the Pushgateway keeps serving the last value forever, so "0.0" could mean
    # "nothing drifted" or "this stopped running a week ago".
    Gauge(
        "riskwatch_drift_last_run_timestamp_seconds",
        "Unix timestamp of the last completed drift run",
        registry=registry,
    ).set(time.time())

    job = push_job(track)
    push_to_gateway(gateway, job=job, registry=registry)
    logger.info("Pushed %s drift metrics to %s as job=%s", track, gateway, job)


def run(
    source: str,
    push: bool,
    out_dir: Path = REPORTS_DIR,
    min_rows: int = MIN_CURRENT_ROWS,
    track: str = DEFAULT_TRACK,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Compute drift for ``track`` from ``source``, write the report, optionally push.

    ``reference_path`` is explicit because the default moves. ``latest.parquet`` is a
    symlink that ``ingest()`` repoints, and the retrain DAG ingests *before* it monitors --
    so taking the default here compared live traffic against data the serving model had
    never seen, and called the result drift. The DAG resolves the pre-ingest snapshot and
    passes it in; the CLI still gets the current one, which is what a human running this by
    hand means.

    ``None`` rather than a module constant as the default: the path is now per-track, so
    it cannot be resolved at import time without pinning a track.
    """
    if source not in DRIFT_SOURCES:
        raise ValueError(f"Unknown drift source {source!r}; expected one of {list(DRIFT_SOURCES)}")

    reference = load_reference(path=reference_path, track=track)
    current = (
        synthetic_current(reference, track=track)
        if source == "synthetic"
        else logged_current(track=track)
    )

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
    report_path = out_dir / f"drift_{track}_{source}_{stamp}.html"
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
        push_metrics(summary, track=track)

    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Detect feature drift against training data.")
    parser.add_argument(
        "--track",
        default=DEFAULT_TRACK,
        help="Which risk track to compute drift for (credit, fraud).",
    )
    parser.add_argument(
        "--source",
        choices=DRIFT_SOURCES,
        default="synthetic",
        help="synthetic: documented +15%% shift on the track's drift column. "
        "logs: real served requests for that track.",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push gauges to the Pushgateway.",
    )
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="Report output directory.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Training snapshot to compare against. Defaults to the selected track's "
            "current latest.parquet, resolved at run time -- it cannot be a fixed default "
            "because the path is per-track."
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_CURRENT_ROWS,
        help="Refuse to compute drift on fewer current rows than this.",
    )
    args = parser.parse_args()

    run(
        source=args.source,
        push=args.push,
        out_dir=args.out,
        min_rows=args.min_rows,
        track=args.track,
        reference_path=args.reference,
    )


if __name__ == "__main__":
    main()
