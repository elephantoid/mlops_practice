"""Append-only JSONL log of every prediction the service makes.

This is the substrate two later pieces read:

* Evidently compares the ``features`` recorded here against the training distribution to
  detect drift on real traffic.
* The A/B analysis splits these records by ``model_version`` and runs a KS-test on
  ``risk_probability``.

The blueprint's Week 9 schema records only an ``input_hash``, but a hash cannot
reconstruct a feature distribution, so drift monitoring on live requests would be
impossible. The record here is a superset: it keeps ``input_hash`` for de-duplication and
request matching, and adds the feature values that Evidently needs.

**Two sinks, deliberately.** The file is the local plane; stdout is the deployed one.
Cloud Run has no durable filesystem -- the container-local log is discarded on every
scale-to-zero -- so a file-only sink means the deployed service produces no evidence at
all, while drift monitoring, PSI, drift decomposition and the incident record all read
this log. Cloud Run captures stdout into Cloud Logging with no added dependency, which is
why the second sink is a ``print`` and not a client library.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prometheus_client import Counter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

# Because log_prediction swallows every exception, a broken sink produces no signal at
# all -- a full disk or a bad mount looks exactly like a healthy service. This counter is
# the signal. Defined here rather than in main.py: main already imports this module, so
# the reverse would be a circular import.
LOG_WRITES = Counter(
    "riskwatch_prediction_log_total",
    "Prediction log write outcomes",
    ["sink", "status"],
)


def log_path() -> Path:
    """Resolve the log destination.

    Read per call rather than captured at import so tests can redirect it with monkeypatch
    and the container can override it without a rebuild.
    """
    override = os.environ.get("PREDICTION_LOG_PATH")
    return Path(override) if override else DEFAULT_LOG_PATH


def input_hash(features: dict[str, Any]) -> str:
    """Stable short digest of a feature payload.

    ``sort_keys`` makes the digest independent of dict ordering, so the same customer
    hashes identically no matter how the request was serialised. Truncated to 16 hex chars:
    this identifies duplicate requests in analysis, it is not a security boundary.
    """
    canonical = json.dumps(features, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def log_prediction(
    request_id: str,
    model_version: str,
    features: dict[str, Any],
    risk_probability: float,
    decision: str,
    track: str,
) -> None:
    """Append one prediction record to both sinks. Never raises.

    Observability must not be able to take down serving: a full disk, a read-only mount or
    a missing directory degrades the log, not availability. Every failure is swallowed
    after being reported.

    The two sinks fail independently. A container with an unwritable ``/app/logs`` -- the
    classic missing-chown case -- still emits to stdout and is still observable in Cloud
    Logging, which is the sink that matters once deployed.
    """
    # Built inside its own guard. Record construction calls input_hash, which serialises
    # the payload -- an unserialisable value there would otherwise raise outside both sink
    # guards and propagate into the request handler, turning a logging fault into a 500.
    # The whole point of this module is that observability cannot take down serving.
    try:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "model_version": model_version,
            "track": track,
            "input_hash": input_hash(features),
            "risk_probability": risk_probability,
            "decision": decision,
            "features": features,
        }
    except Exception:
        LOG_WRITES.labels(sink="file", status="failed").inc()
        LOG_WRITES.labels(sink="stdout", status="failed").inc()
        logger.exception("Failed to build prediction log record for request %s", request_id)
        return

    try:
        # Serialise fully before opening the file, then write in a single call. POSIX
        # guarantees appends below PIPE_BUF (4096 bytes) are atomic, and a record is around
        # 600 bytes -- so concurrent uvicorn workers cannot interleave partial lines.
        # Writing incrementally would give that up.
        line = json.dumps(record, default=str) + "\n"

        destination = log_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line)

        LOG_WRITES.labels(sink="file", status="written").inc()

    except Exception:
        LOG_WRITES.labels(sink="file", status="failed").inc()
        logger.exception("Failed to write prediction log for request %s", request_id)

    try:
        # print rather than logger: this must be one parseable JSON object per line with
        # no level prefix or formatter in front of it, because Cloud Logging parses a bare
        # JSON line into structured fields and a prefixed line stays an opaque string.
        print(json.dumps({"prediction_log": record}, default=str), flush=True)
        LOG_WRITES.labels(sink="stdout", status="written").inc()
    except Exception:
        LOG_WRITES.labels(sink="stdout", status="failed").inc()
        logger.exception("Failed to emit prediction log to stdout for request %s", request_id)
