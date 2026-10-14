"""Generated traffic for the public service, drawn from the committed PSI reference.

Run from the repo root::

    uv run python -m src.monitoring.traffic --dry-run
    uv run python -m src.monitoring.traffic --url https://riskwatch-46336906432.us-central1.run.app

**Why it exists.** The service has no users, and Step 18's daily Action has to read a day of
prediction log from GCS on a runner with no ``data/``. Something has to put that day there.

**What it is.** Every feature is drawn *independently* from its reference histogram: a bin by the
reference's counts, then a value inside the bin. PSI only sees bin membership, so in expectation
each feature's PSI reproduces the reference exactly -- the daily row is a no-shift baseline, and
its spread is the noise floor. The joint distribution is not reproduced. Two consequences, both
recorded in ``STATUS.md`` rather than corrected:

* The score PSI has a floor. Independent marginals put the model off its training manifold, so
  ``__score__`` sits near 0.05 (credit) and 0.035 (fraud) on any day, at any sample size.
* A domain classifier (``decompose.py``) would separate this traffic from the reference at once.
  It is not pointed at it.

**Drills.** ``DRILLS`` is a pre-registered schedule of days on which one column is distorted. It
is committed before the day, so a day that alerts was chosen in advance and not picked out of the
log afterwards. A drill proves the alert path -- GCS, the Action, PSI, the failed job, the mail --
end to end. It says nothing about detecting real drift; that is Step 12's evaluation.
"""

from __future__ import annotations

import argparse
import logging
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import annotated_types
import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.api.schemas import CREDIT_CATEGORIES, CreditPredictRequest, FraudPredictRequest
from src.features.specs import FEATURE_SPECS, get_feature_spec
from src.monitoring import psi

logger = logging.getLogger(__name__)

GENERATOR = "reference-marginals"
SERVICE_URL = "https://riskwatch-46336906432.us-central1.run.app"
REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "credit": CreditPredictRequest,
    "fraud": FraudPredictRequest,
}

# Per day. Above psi.required_rows (720 credit, 400 fraud at the current references) with room for
# a few failed requests; the daily row refuses a short day rather than reporting its noise.
DAILY_ROWS = {"credit": 800, "fraud": 500}
CONCURRENCY = 4
TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Drill:
    """On ``day``, ``column`` of ``track``'s generated rows is multiplied by ``factor``."""

    day: date
    track: str
    column: str
    factor: float

    @property
    def name(self) -> str:
        return f"drill-{self.day.isoformat()}"


# Pre-registered 2026-10-14. One day, credit only: AMT_CREDIT x1.5 moves every decile, so the
# feature PSI clears PSI_ALERT by a wide margin and the alert has to fire the next morning.
DRILLS: tuple[Drill, ...] = (Drill(date(2026, 11, 10), "credit", "AMT_CREDIT", 1.5),)


def drill_on(day: date, track: str) -> Drill | None:
    """The drill scheduled for ``track`` on ``day``, if any."""
    return next((d for d in DRILLS if d.day == day and d.track == track), None)


@dataclass(frozen=True)
class _Field:
    integer: bool
    optional: bool
    low: float | None
    low_open: bool
    high: float | None
    literal: tuple[str, ...] | None


def _field(model: type[BaseModel], column: str) -> _Field:
    """The request field behind a feature column: its type, optionality, bounds, choices."""
    info = model.model_fields[column.lower()]
    annotation = info.annotation
    optional = False
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(members) < len(typing.get_args(annotation))
        (annotation,) = members
    low = high = None
    low_open = False
    for constraint in info.metadata:
        if isinstance(constraint, annotated_types.Ge):
            low = float(constraint.ge)
        elif isinstance(constraint, annotated_types.Gt):
            low, low_open = float(constraint.gt), True
        elif isinstance(constraint, annotated_types.Le):
            high = float(constraint.le)
        elif isinstance(constraint, annotated_types.Lt):
            high = float(constraint.lt)
    literal = (
        typing.get_args(annotation) if typing.get_origin(annotation) is typing.Literal else None
    )
    return _Field(annotation is int, optional, low, low_open, high, literal)


def _bin_probabilities(counts: list[int], dropped: list[int]) -> np.ndarray:
    """The reference's bin shares, with ``dropped`` bins' mass spread over the rest."""
    weights = np.asarray(counts, dtype="float64")
    weights[dropped] = 0.0
    return weights / weights.sum()


def _numeric(entry: dict[str, Any], field: _Field, n: int, rng: np.random.Generator) -> list:
    edges = entry["edges"]
    k = len(edges)
    null, sentinel = k + 1, k + 2
    # A required field cannot carry None. Its null bin is a training-side artefact anyway: the
    # request contract is what makes a value required.
    bins = rng.choice(
        psi.n_bins(entry),
        size=n,
        p=_bin_probabilities(entry["counts"], [] if field.optional else [null]),
    )
    # The open end bins get the width of their neighbour, clipped to the contract's bound.
    first = edges[1] - edges[0] if k > 1 else 1.0
    last = edges[-1] - edges[-2] if k > 1 else 1.0
    out: list = [None] * n
    for b in np.unique(bins):
        rows = np.flatnonzero(bins == b)
        if b == null:
            continue
        if b == sentinel:
            value = entry["sentinel"]
            for r in rows:
                out[r] = int(value) if field.integer else value
            continue
        if b == 0:
            hi = edges[0]
            lo = hi - first if field.low is None else max(hi - first, field.low)
        elif b == k:
            lo = edges[-1]
            hi = lo + last if field.high is None else min(lo + last, field.high)
        else:
            lo, hi = edges[b - 1], edges[b]
        if field.integer:
            # Integers in (lo, hi]; the first bin is closed at a contract bound it reaches.
            lo_i = int(np.floor(lo)) + 1
            if b == 0 and field.low is not None and lo <= field.low:
                lo_i = int(np.floor(field.low)) + 1 if field.low_open else int(np.ceil(field.low))
            values = rng.integers(lo_i, int(np.floor(hi)) + 1, size=len(rows))
            for r, v in zip(rows, values, strict=True):
                out[r] = int(v)
        else:
            # (lo, hi]: a draw of exactly lo would fall into the bin below.
            values = rng.uniform(lo, hi, size=len(rows))
            values = np.where(values <= lo, hi, values)
            for r, v in zip(rows, values, strict=True):
                out[r] = float(v)
    return out


def _categorical(
    entry: dict[str, Any], field: _Field, column: str, n: int, rng: np.random.Generator
) -> list:
    kept = entry["categories"]
    other, null = len(kept), len(kept) + 1
    universe = field.literal if field.literal is not None else CREDIT_CATEGORIES.get(column, ())
    rest = [c for c in universe if c not in kept]
    dropped = ([] if field.optional else [null]) + ([] if rest else [other])
    bins = rng.choice(psi.n_bins(entry), size=n, p=_bin_probabilities(entry["counts"], dropped))
    return [
        None if b == null else rest[rng.integers(len(rest))] if b == other else kept[b]
        for b in bins
    ]


def generate(track: str, rows: int, seed: int, drill: Drill | None = None) -> pd.DataFrame:
    """``rows`` feature rows for ``track``, in the spec's column order, with ``drill`` applied."""
    summary = psi.load_summary(track)
    model = REQUEST_MODELS[track]
    rng = np.random.default_rng(seed)
    columns = {}
    for column, entry in summary["features"].items():
        field = _field(model, column)
        columns[column] = (
            _numeric(entry, field, rows, rng)
            if entry["kind"] == "numeric"
            else _categorical(entry, field, column, rows, rng)
        )
    frame = pd.DataFrame(columns, dtype=object)[list(get_feature_spec(track).feature_columns)]
    if drill is not None:
        frame[drill.column] = [None if v is None else v * drill.factor for v in frame[drill.column]]
    return frame


def payloads(track: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Request bodies, each validated against the track's request model before it is sent."""
    model = REQUEST_MODELS[track]
    bodies = []
    for row in frame.to_dict(orient="records"):
        body = {column.lower(): value for column, value in row.items()}
        model.model_validate(body)
        bodies.append(body)
    return bodies


def send(url: str, track: str, bodies: list[dict[str, Any]]) -> tuple[int, int]:
    """POST every body to ``/predict/<track>``. Returns (succeeded, failed)."""
    import httpx

    def one(client: httpx.Client, body: dict[str, Any]) -> bool:
        for _ in range(2):
            try:
                if client.post(f"/predict/{track}", json=body).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
        return False

    with (
        httpx.Client(base_url=url, timeout=TIMEOUT_SECONDS) as client,
        ThreadPoolExecutor(CONCURRENCY) as pool,
    ):
        results = list(pool.map(lambda body: one(client, body), bodies))
    return sum(results), len(results) - sum(results)


def seed_for(day: date, track: str) -> int:
    """One seed per day and track, so a day's traffic can be regenerated exactly."""
    return day.toordinal() * 10 + sorted(FEATURE_SPECS).index(track)


def main() -> None:
    """CLI entry point. Exits non-zero if any request fails."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=SERVICE_URL)
    parser.add_argument("--track", action="append", choices=sorted(FEATURE_SPECS))
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="UTC; default today")
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate; send none.")
    args = parser.parse_args()

    day = args.date or datetime.now(UTC).date()
    failed = 0
    for track in args.track or sorted(FEATURE_SPECS):
        drill = drill_on(day, track)
        frame = generate(track, DAILY_ROWS[track], seed_for(day, track), drill)
        bodies = payloads(track, frame)
        label = f"{track} {day} {GENERATOR}" + (f" {drill.name}" if drill else "")
        if args.dry_run:
            print(f"{label}: {len(bodies)} valid requests, none sent")
            continue
        ok, bad = send(args.url, track, bodies)
        print(f"{label}: {ok} sent, {bad} failed")
        failed += bad
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
