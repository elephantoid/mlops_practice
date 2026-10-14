"""The daily PSI row: one UTC day of the deployed service's log, read from GCS, one line per track.

Run by ``.github/workflows/evidence.yml`` on a runner with no ``data/``::

    uv run python -m src.monitoring.daily --out evidence/reports/psi_daily.jsonl
    uv run python -m src.monitoring.daily --check evidence/reports/psi_daily.jsonl

Writing and judging are separate invocations because the workflow must commit the row and
generate the next day's traffic *before* it fails on an alert. A job that failed first would skip
both, and the next morning would find an empty day and nothing to compare.

A day that cannot be compared is a row too, with its reason. It also fails ``--check``: on this
service an empty day means the generator, the sink or the reader is broken, and a monitor that
says nothing when its input disappears is the failure it exists to catch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.specs import FEATURE_SPECS
from src.monitoring import gcs_log, psi, traffic
from src.monitoring.drift import MODEL_VERSION_COLUMN, InsufficientCurrentData

logger = logging.getLogger(__name__)


def row(track: str, day: date, summary: dict[str, Any], current: pd.DataFrame | str) -> dict:
    """One track's line for ``day``. ``current`` is the day's frame, or why there is none."""
    drill = traffic.drill_on(day, track)
    line: dict[str, Any] = {
        "date": day.isoformat(),
        "track": track,
        "reference_model_version": summary["model_version"],
        "generator": traffic.GENERATOR,
        "drill": drill.name if drill else None,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    if isinstance(current, str):
        return {**line, "status": "insufficient", "rows": 0, "reason": current, "alert": True}
    try:
        results = psi.psi_by_feature(summary, current)
    except InsufficientCurrentData as error:
        return {
            **line,
            "status": "insufficient",
            "rows": len(current),
            "reason": str(error),
            "alert": True,
        }
    worst = max(results, key=results.get)
    versions = current[MODEL_VERSION_COLUMN].astype(str).value_counts().to_dict()
    # Served by a model the reference is not for: the score PSI is skipped and the feature PSI is
    # against the wrong training set, so an "ok" here would be a quiet lie. Rebuild the reference
    # (psi.py --write-reference) after a re-promotion.
    stale = set(versions) != {summary["model_version"]}
    return {
        **line,
        "status": "stale-reference" if stale else "ok",
        "rows": len(current),
        "model_versions": versions,
        "max_feature": worst,
        "max_psi": round(results[worst], 4),
        "score_psi": round(results[psi.SCORE_FEATURE], 4) if psi.SCORE_FEATURE in results else None,
        "alert": stale or results[worst] > psi.PSI_ALERT,
        "psi": {k: round(v, 4) for k, v in sorted(results.items())},
    }


def run(day: date, out: Path, client: Any = None, bucket: str | None = None) -> list[dict]:
    """Append ``day``'s row for every track to ``out``."""
    since, until = gcs_log.day_window(day)
    lines = []
    for track in sorted(FEATURE_SPECS):
        summary = psi.load_summary(track)
        try:
            current: pd.DataFrame | str = gcs_log.logged_current(
                since, until, track, include_score=True, bucket=bucket, client=client
            )
        except InsufficientCurrentData as error:
            current = str(error)
        lines.append(row(track, day, summary, current))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return lines


def alerts(path: Path, day: date) -> list[dict]:
    """The latest row per track for ``day`` that alerts. A track with no row at all alerts."""
    latest: dict[str, dict] = {}
    if path.exists():
        for text in path.read_text().splitlines():
            if text.strip():
                line = json.loads(text)
                if line["date"] == day.isoformat():
                    latest[line["track"]] = line
    missing = [
        {"date": day.isoformat(), "track": t, "status": "missing", "alert": True}
        for t in sorted(FEATURE_SPECS)
        if t not in latest
    ]
    return [line for line in latest.values() if line["alert"]] + missing


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None, help="UTC day; default yesterday"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="Append the day's rows here.")
    action.add_argument("--check", type=Path, help="Exit 1 if the day's rows alert.")
    args = parser.parse_args()
    day = args.date or datetime.now(UTC).date() - timedelta(days=1)

    if args.out:
        for line in run(day, args.out):
            detail = (
                f"max {line['max_feature']} {line['max_psi']} score {line['score_psi']}"
                if "max_psi" in line
                else line["reason"]
            )
            print(f"{line['date']} {line['track']}: {line['status']} rows={line['rows']} {detail}")
        return

    failing = alerts(args.check, day)
    for line in failing:
        print(f"ALERT {line['date']} {line['track']}: {json.dumps(line)}")
    raise SystemExit(1 if failing else 0)


if __name__ == "__main__":
    main()
