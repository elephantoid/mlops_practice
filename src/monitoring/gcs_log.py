"""The deployed service's prediction log, read back from the GCS sink.

Cloud Run puts each stdout line the service prints into Cloud Logging, which parses a bare JSON
line into ``jsonPayload``. The sink ``riskwatch-predictions`` (``ops/evidence/setup.sh``) routes
the prediction records into a bucket as hourly shards::

    run.googleapis.com/stdout/YYYY/MM/DD/HH:00:00_HH:59:59_S0.json      one LogEntry per line
    run.googleapis.com/stdout/YYYY/MM/DD/HH:00:00_HH:59:59_S0_A0.json   late entries, appended

A shard can arrive two to three hours after its hour closes. A day is therefore read the next
day, not as it happens.

The payload is exactly the record ``prediction_log.log_prediction`` printed, so filtering is
``drift.records_frame``'s and nothing here interprets it. The folder an entry lands in is not
used as its date: the days on either side are listed as well and every record is filtered on
its own ``timestamp``, so an entry filed under the neighbouring day is neither lost nor counted
twice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd

from src.monitoring import drift

BUCKET_ENV = "RISKWATCH_EVIDENCE_BUCKET"
LOG_DIR = "run.googleapis.com/stdout"


def bucket_name() -> str:
    """The evidence bucket, from the environment. No default: a wrong guess reads nothing."""
    name = os.environ.get(BUCKET_ENV)
    if not name:
        raise RuntimeError(f"{BUCKET_ENV} is not set; it names the bucket the sink writes to")
    return name


def day_window(day: date) -> tuple[datetime, datetime]:
    """``[day 00:00, day+1 00:00)`` in UTC."""
    start = datetime.combine(day, time(), tzinfo=UTC)
    return start, start + timedelta(days=1)


def day_prefixes(since: datetime, until: datetime) -> list[str]:
    """Folder prefixes for every UTC day the window touches, plus one either side."""
    first = since.astimezone(UTC).date() - timedelta(days=1)
    # ``until`` is exclusive: a window ending at midnight does not touch the day it ends on.
    last = (until.astimezone(UTC) - timedelta(microseconds=1)).date() + timedelta(days=1)
    days = (last - first).days + 1
    return [f"{LOG_DIR}/{first + timedelta(days=n):%Y/%m/%d}/" for n in range(days)]


def iter_records(
    since: datetime, until: datetime, bucket: str | None = None, client: Any = None
) -> Iterator[dict[str, Any]]:
    """The ``jsonPayload`` of every entry under the window's folders, in shard order.

    ``client`` is a ``google.cloud.storage.Client`` or anything with its ``list_blobs``; tests
    pass a fake. The import is deferred so the local path never needs the library configured.
    """
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    bucket = bucket or bucket_name()
    for prefix in day_prefixes(since, until):
        for blob in sorted(client.list_blobs(bucket, prefix=prefix), key=lambda b: b.name):
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line).get("jsonPayload")
                if payload is not None:
                    yield payload


def logged_current(
    since: datetime,
    until: datetime,
    track: str,
    include_score: bool = False,
    max_rows: int = drift.MAX_CURRENT_ROWS,
    bucket: str | None = None,
    client: Any = None,
) -> pd.DataFrame:
    """``drift.logged_current``'s frame, from the deployed service's log in ``[since, until)``."""
    bucket = bucket or bucket_name()
    return drift.records_frame(
        iter_records(since, until, bucket=bucket, client=client),
        since=since,
        until=until,
        max_rows=max_rows,
        track=track,
        include_score=include_score,
        where=f"gs://{bucket} in [{since.isoformat()}, {until.isoformat()})",
    )
