"""Population Stability Index per feature (CSI) and on the score, against a committed reference.

Run from the repo root::

    uv run python -m src.monitoring.psi --track credit --write-reference
    uv run python -m src.monitoring.psi --track credit --source synthetic --push
    uv run python -m src.monitoring.psi --track credit --source logs --push

**Why a second drift measure beside ``drift.py``.** Evidently picks a statistical test per column
and the choice is recorded nowhere (``.claude/rules/monitoring.md``), so its numbers can only be
reproduced by re-running it against the training rows. PSI here is fixed bins and a fixed
formula, so the *reference side* reduces to a few KB of bin edges and counts --
``src/monitoring/reference/<track>_psi_reference.json``, committed. Anything that can read that
file and a current frame can compute the same number, including a runner with no ``data/``,
which is what Step 18's hosted job needs. The file holds aggregates only, no rows: Home Credit's
competition rules bar redistributing the data itself.

**Bins.** Numeric: deciles of the reference (interior edges rounded to six significant digits
and de-duplicated, so a discrete column such as ``CNT_CHILDREN`` gets fewer bins), right-closed,
open at both ends; a null bin; and a bin of its own for the track's sentinel (``DAYS_EMPLOYED``
365243), which would otherwise drag the top decile edge to a value no real applicant has.
Categorical: every category holding at least 1% of the reference, one ``__other__`` bin for the
rest and for anything unseen, and a null bin. The 1% floor matters twice: a 58-category column
has a large no-shift PSI on small samples (see ``required_rows``), and a category of 2 rows has
no business in a committed file called a summary.

**Bands.** 0.1 / 0.25 are the industry convention for scorecards (Siddiqi, *Credit Risk
Scorecards*, 2006), not values derived from this data.

**The score.** ``__score__`` is the PSI of the model's own output. Its reference is the served
model's scores on the training holdout (``holdout_split``) -- live scores are out-of-sample by
definition, so the reference is too. Measured on credit v6 the in-sample/holdout difference is a
PSI of 0.0003, so the choice is principle, not a correction of a visible bias. The score is
computed from ``--source logs`` only, on rows logged by the same model version as the reference;
a synthetic batch has no score to compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.features.specs import FEATURE_SPECS, FeatureSpec, get_feature_spec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "build" / "model"

SCHEMA_VERSION = 1
N_QUANTILE_BINS = 10
SIGNIFICANT_DIGITS = 6
RARE_CATEGORY_SHARE = 0.01
OTHER = "__other__"
SCORE_FEATURE = "__score__"

# Floor on each bin's proportion. Without it a bin empty on one side gives ln(0) and the index is
# infinite; with it the contribution is bounded. It is also a choice of sensitivity: a bin at 0% in
# the reference reaching 10% in the current batch contributes 0.1 * ln(0.1 / 1e-4) = 0.69, past
# the alert band on its own. That is intended -- mass appearing where the reference had none is
# the clearest shift there is.
EPS = 1e-4

PSI_WATCH = 0.1
PSI_ALERT = 0.25

# How much no-shift PSI a current batch may carry, in expectation, before its size is refused.
# PSI on two samples of one distribution is not 0: it is about (B - 1) * (1/n + 1/N) for B bins
# (Yurdakul, 2018). Measured on credit with no shift at all, 100 rows gave ORGANIZATION_TYPE a PSI
# of 0.657 -- an alert on noise. A quarter of the watch band keeps the noise an order of magnitude
# under the alert line. The reference term 1/N is left out: N is the whole training snapshot.
NOISE_BUDGET = PSI_WATCH / 4

SYNTHETIC_ROWS = 5_000
RANDOM_STATE = 42

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")
# Not drift.py's job name. push_to_gateway replaces every metric under a job, so sharing
# ``riskwatch-drift-<track>`` would make each module's push erase the other's.
PUSH_JOB_PREFIX = "riskwatch-psi"


def push_job(track: str) -> str:
    """Pushgateway job name for one track's PSI."""
    return f"{PUSH_JOB_PREFIX}-{track}"


def reference_file(track: str, directory: Path = REFERENCE_DIR) -> Path:
    """Where a track's committed reference summary lives."""
    return directory / f"{track}_psi_reference.json"


def band(value: float) -> str:
    """``stable`` / ``watch`` / ``shift`` for one PSI value."""
    if value > PSI_ALERT:
        return "shift"
    return "watch" if value > PSI_WATCH else "stable"


def psi(reference_counts: Any, current_counts: Any, eps: float = EPS) -> float:
    """PSI between two count vectors over the same bins: sum of (q - p) * ln(q / p)."""
    reference = np.asarray(reference_counts, dtype="float64")
    current = np.asarray(current_counts, dtype="float64")
    if reference.shape != current.shape:
        raise ValueError(f"bin counts differ in length: {reference.shape} vs {current.shape}")
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("cannot compute PSI against an empty side")
    p = np.maximum(reference / reference.sum(), eps)
    q = np.maximum(current / current.sum(), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def _round_significant(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def numeric_entry(values: pd.Series, sentinel: float | None) -> dict[str, Any]:
    """Decile bins of ``values`` plus null and sentinel bins, with the reference's counts."""
    numbers = pd.to_numeric(values, errors="raise").astype("float64")
    keep = numbers.notna()
    if sentinel is not None:
        keep &= numbers != sentinel
    body = numbers[keep]
    if len(body):
        quantiles = np.quantile(
            body.to_numpy(), np.linspace(0, 1, N_QUANTILE_BINS + 1)[1:-1], method="linear"
        )
        edges = sorted({_round_significant(q) for q in quantiles})
    else:
        edges = []
    entry: dict[str, Any] = {"kind": "numeric", "edges": edges, "sentinel": sentinel}
    entry["counts"] = bin_counts(entry, values)
    return entry


def categorical_entry(values: pd.Series) -> dict[str, Any]:
    """One bin per category at or above ``RARE_CATEGORY_SHARE``, then other and null."""
    present = values.dropna().astype(str)
    counts = present.value_counts()
    kept = sorted(c for c, n in counts.items() if n / len(values) >= RARE_CATEGORY_SHARE)
    entry: dict[str, Any] = {"kind": "categorical", "categories": kept}
    entry["counts"] = bin_counts(entry, values)
    return entry


def bin_counts(entry: dict[str, Any], values: Any) -> list[int]:
    """Counts of ``values`` in ``entry``'s bins.

    Layout, numeric: one bin per interval ``(-inf, e1], (e1, e2], ..., (ek, inf)``, then null, then
    the sentinel if the entry has one. Categorical: one bin per kept category, then ``__other__``,
    then null. Step 18's reader relies on this order; ``tests/test_psi.py`` pins it against an
    independent ``pd.cut`` implementation.
    """
    series = pd.Series(values)
    if entry["kind"] == "numeric":
        numbers = pd.to_numeric(series, errors="raise").astype("float64").to_numpy()
        null = np.isnan(numbers)
        sentinel = entry["sentinel"]
        is_sentinel = numbers == sentinel if sentinel is not None else np.zeros_like(null)
        body = numbers[~null & ~is_sentinel]
        edges = np.asarray(entry["edges"], dtype="float64")
        # side="left": a value equal to an edge lands in the bin that edge closes on the right.
        index = np.searchsorted(edges, body, side="left")
        counts = np.bincount(index, minlength=len(edges) + 1).tolist() + [int(null.sum())]
        if sentinel is not None:
            counts.append(int(is_sentinel.sum()))
        return [int(c) for c in counts]

    if entry["kind"] == "categorical":
        null = series.isna()
        text = series[~null].astype(str)
        categories = entry["categories"]
        kept = text[text.isin(categories)].value_counts()
        counts = [int(kept.get(c, 0)) for c in categories]
        return counts + [int((~text.isin(categories)).sum()), int(null.sum())]

    raise ValueError(f"unknown bin kind {entry['kind']!r}")


def summarise_frame(frame: pd.DataFrame, spec: FeatureSpec) -> dict[str, dict[str, Any]]:
    """One bin entry per modeled feature, in the spec's column order."""
    entries: dict[str, dict[str, Any]] = {}
    for column in spec.feature_columns:
        if column in spec.numeric_features:
            entries[column] = numeric_entry(frame[column], spec.sentinels.get(column))
        else:
            entries[column] = categorical_entry(frame[column])
    return entries


def required_rows(summary: dict[str, Any]) -> int:
    """Smallest current batch whose expected no-shift PSI stays within ``NOISE_BUDGET``.

    Taken over the widest entry, since the alert reads the maximum over features.
    """
    entries = list(summary["features"].values())
    if summary.get("score"):
        entries.append(summary["score"])
    widest = max(len(entry["counts"]) for entry in entries)
    return math.ceil((widest - 1) / NOISE_BUDGET)


def psi_by_feature(
    summary: dict[str, Any], current: pd.DataFrame, min_rows: int | None = None
) -> dict[str, float]:
    """PSI of each feature in ``current`` against the summary, plus ``__score__`` when possible.

    ``__score__`` needs ``risk_probability`` and ``model_version`` columns
    (``drift.logged_current(include_score=True)``) and enough rows logged by the reference's own
    model version; another version's scores are a different function of the same inputs.
    """
    from src.monitoring.drift import (
        MODEL_VERSION_COLUMN,
        SCORE_COLUMN,
        InsufficientCurrentData,
    )

    needed = required_rows(summary) if min_rows is None else min_rows
    if len(current) < needed:
        raise InsufficientCurrentData(
            f"Only {len(current)} current rows; PSI needs {needed} for its no-shift noise to stay "
            f"under {NOISE_BUDGET} on the widest feature."
        )

    results = {
        column: psi(entry["counts"], bin_counts(entry, current[column]))
        for column, entry in summary["features"].items()
    }

    if summary.get("score") and SCORE_COLUMN in current.columns:
        same_model = current[current[MODEL_VERSION_COLUMN].astype(str) == summary["model_version"]]
        if len(same_model) >= needed:
            results[SCORE_FEATURE] = psi(
                summary["score"]["counts"], bin_counts(summary["score"], same_model[SCORE_COLUMN])
            )
        else:
            logger.warning(
                "score PSI skipped: %d rows from model version %s, need %d",
                len(same_model),
                summary["model_version"],
                needed,
            )
    return results


def dumps(summary: dict[str, Any]) -> str:
    """JSON with one line per scalar list, so a reference diff reads per feature, not per number."""
    inline: dict[str, str] = {}

    def collapse(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: collapse(value) for key, value in node.items()}
        if isinstance(node, list) and not any(isinstance(x, dict | list) for x in node):
            key = f"@@{len(inline)}@@"
            inline[key] = json.dumps(node)
            return key
        return node

    text = json.dumps(collapse(summary), indent=2)
    for key, value in inline.items():
        text = text.replace(json.dumps(key), value, 1)
    return text + "\n"


def load_summary(track: str, directory: Path = REFERENCE_DIR) -> dict[str, Any]:
    """Read a committed reference summary, refusing a schema this code does not write."""
    path = reference_file(track, directory)
    summary = json.loads(path.read_text())
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {summary.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    return summary


def build_reference(track: str) -> dict[str, Any]:
    """Summarise the snapshot the *served* model was trained on, from the exported model.

    The snapshot is resolved from the model's run tag ``data_ingested_at``, not from
    ``latest.parquet``: ingest repoints that symlink, and a reference built from data the served
    model never saw would call the difference drift.
    """
    import mlflow
    import sklearn
    from mlflow.tracking import MlflowClient

    from src.features.pipeline import holdout_split, split_features_target
    from src.models.train import configure_tracking, model_name_for

    spec = get_feature_spec(track)
    model_dir = MODEL_DIR / track
    version = (model_dir / "MODEL_VERSION").read_text().strip()

    configure_tracking()
    client = MlflowClient()
    run_id = client.get_model_version(model_name_for(track), version).run_id
    stamp = client.get_run(run_id).data.tags["data_ingested_at"]
    snapshot = PROCESSED_DIR / track / f"{track}_{stamp}.parquet"

    frame = pd.read_parquet(snapshot)
    features, target = split_features_target(frame, spec)
    _, x_test, _, _ = holdout_split(features, target)
    scores = np.asarray(mlflow.pyfunc.load_model(str(model_dir)).predict(x_test))[:, 1]

    return {
        "schema_version": SCHEMA_VERSION,
        "track": track,
        "model_version": version,
        "run_id": run_id,
        "snapshot": snapshot.name,
        "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "rows": len(frame),
        "holdout_rows": len(x_test),
        "sklearn_version": sklearn.__version__,
        "eps": EPS,
        "closed": "right",
        "quantile_method": "linear",
        "quantile_bins": N_QUANTILE_BINS,
        "significant_digits": SIGNIFICANT_DIGITS,
        "rare_category_share": RARE_CATEGORY_SHARE,
        "features": summarise_frame(frame, spec),
        "score": numeric_entry(pd.Series(scores), None),
    }


def snapshot_path(summary: dict[str, Any]) -> Path:
    """The local parquet a summary was built from. Absent on a fresh clone."""
    return PROCESSED_DIR / summary["track"] / summary["snapshot"]


def synthetic_current(
    summary: dict[str, Any], rows: int = SYNTHETIC_ROWS, shift: float | None = None
) -> pd.DataFrame:
    """Rows sampled from the reference snapshot, with the track's drift column multiplied.

    ``shift`` defaults to drift.py's documented +15%. ``1.0`` gives a no-shift batch, which is
    the noise floor. Anything else is a delivery test of the alert path, not evidence about the
    detector: the person choosing the multiplier is choosing whether it clears the line.
    """
    from src.monitoring import drift

    track = summary["track"]
    reference = drift.load_reference(path=snapshot_path(summary), track=track)
    batch = reference.sample(min(rows, len(reference)), random_state=RANDOM_STATE)
    column = drift.SYNTHETIC_DRIFT_COLUMN[track]
    batch[column] = batch[column] * (drift.SYNTHETIC_UPLIFT if shift is None else shift)
    return batch


def push_metrics(results: dict[str, float], track: str, gateway: str = PUSHGATEWAY_URL) -> None:
    """Push one gauge per feature, and the run's timestamp, under this track's PSI job."""
    registry = CollectorRegistry()
    gauge = Gauge(
        "riskwatch_psi",
        "Population Stability Index of one feature, or __score__, against the committed reference",
        ["track", "feature"],
        registry=registry,
    )
    for feature, value in results.items():
        gauge.labels(track=track, feature=feature).set(value)
    # A Pushgateway serves the last value forever; without this a PSI job that stopped a month ago
    # is indistinguishable from one reporting stable.
    Gauge(
        "riskwatch_psi_last_run_timestamp_seconds",
        "Unix timestamp of the last completed PSI run",
        ["track"],
        registry=registry,
    ).labels(track=track).set(time.time())
    push_to_gateway(gateway, job=push_job(track), registry=registry)
    logger.info("Pushed %d PSI values for %s as job=%s", len(results), track, push_job(track))


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", required=True, choices=sorted(FEATURE_SPECS))
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Rebuild the committed summary from the served model's training snapshot.",
    )
    parser.add_argument("--source", choices=("synthetic", "logs"), default="synthetic")
    parser.add_argument("--rows", type=int, default=SYNTHETIC_ROWS, help="Synthetic batch size.")
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Synthetic multiplier on the track's drift column (default: drift.py's 1.15).",
    )
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.write_reference:
        path = reference_file(args.track)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(build_reference(args.track)))
        print(f"{path} written")
        return

    summary = load_summary(args.track)
    if args.source == "synthetic":
        current = synthetic_current(summary, rows=args.rows, shift=args.shift)
    else:
        from src.monitoring.drift import logged_current

        current = logged_current(track=args.track, include_score=True)

    results = psi_by_feature(summary, current, min_rows=args.min_rows)
    for feature, value in sorted(results.items(), key=lambda item: -item[1]):
        print(f"{value:8.4f}  {band(value):6}  {feature}")
    if args.push:
        push_metrics(results, args.track)


if __name__ == "__main__":
    main()
