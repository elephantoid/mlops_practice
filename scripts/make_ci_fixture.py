"""Write a synthetic, schema-shaped training snapshot for one track, for CI.

CI has no data: Home Credit's competition rules bar redistributing it, and `data/` is gitignored
anyway. Without a trained model the skew test skips and the image cannot be built, because the
Dockerfile copies `build/model/<track>/`. This makes a snapshot the real training command accepts:

    uv run python -m scripts.make_ci_fixture --track credit --out build/ci-fixture/credit.parquet
    uv run python -m src.models.train --track credit --data build/ci-fixture/credit.parquet

**Every column is derived from the request model, not written out here.** The request model is
what builds the frame the served model sees, so its annotations are the dtypes the model must be
trained on: a column trained as int64 gets an MLflow signature of `long`, and the API's `1.0` for
it is then refused as an unsafe conversion. Bounds come from the field's `Ge`/`Gt`/`Le`
constraints, categories from its `Literal` (or `credit_categories.json` for the two free-text
fields), and optional fields are null at a fixed rate. A column added to the contract is therefore
generated without touching this file.

Nothing here is drawn from the real data, which is the point: the values are uniform within the
contract's bounds, and the target is a planted rule over three features, so a model can learn
*something* and promotion has a non-degenerate run to pick. It says nothing about model quality.
Written to `build/`, never `data/processed/`, where `tests/test_reachable_decisions.py` would
pick it up and judge thresholds against noise.
"""

from __future__ import annotations

import argparse
import json
import types
import typing
from pathlib import Path

import annotated_types
import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.api.schemas import CreditPredictRequest, FraudPredictRequest
from src.data.ingest import _write_parquet_with_metadata
from src.features.specs import get_feature_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = json.loads((PROJECT_ROOT / "src/data/schemas/credit_categories.json").read_text())

REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "credit": CreditPredictRequest,
    "fraud": FraudPredictRequest,
}

NULL_RATE = 0.1
SENTINEL_RATE = 0.15
POSITIVE_RATE = 0.2
# Width used when a field is bounded on one side only, or not at all.
OPEN_SPAN = 1_000.0


def _bounds(metadata: list[object]) -> tuple[float | None, float | None]:
    low = high = None
    for constraint in metadata:
        if isinstance(constraint, annotated_types.Ge | annotated_types.Gt):
            low = float(getattr(constraint, "ge", getattr(constraint, "gt", 0.0)))
        elif isinstance(constraint, annotated_types.Le | annotated_types.Lt):
            high = float(getattr(constraint, "le", getattr(constraint, "lt", 0.0)))
    return low, high


def _column(
    alias: str, annotation: object, metadata: list[object], rows: int, rng: np.random.Generator
) -> pd.Series:
    """Values for one field, in the dtype the request model will hand the served model."""
    optional = False
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(members) < len(typing.get_args(annotation))
        (annotation,) = members

    if typing.get_origin(annotation) is typing.Literal:
        values = pd.Series(rng.choice(list(typing.get_args(annotation)), rows), dtype=object)
    elif annotation is str:
        values = pd.Series(rng.choice(CATEGORIES[alias], rows), dtype=object)
    elif annotation in (int, float):
        low, high = _bounds(metadata)
        if low is None and high is None:
            low, high = -OPEN_SPAN, OPEN_SPAN
        elif low is None:
            low = high - OPEN_SPAN
        elif high is None:
            high = low + OPEN_SPAN
        drawn = rng.uniform(low, high, rows)
        values = pd.Series(np.round(drawn).astype("int64") if annotation is int else drawn)
    else:
        raise TypeError(f"{alias}: no generator for annotation {annotation!r}")

    if optional:
        values = values.astype(object if values.dtype == object else "float64")
        values[rng.random(rows) < NULL_RATE] = None
    return values


def synthetic_snapshot(track: str, rows: int, seed: int) -> pd.DataFrame:
    """A frame with the track's id, features and target, shaped as ingest would write it."""
    spec = get_feature_spec(track)
    rng = np.random.default_rng(seed)
    # Fields are matched to columns by *name* (`amt_credit` <-> `AMT_CREDIT`), never by
    # `serialization_alias`. The alias is the thing the skew test exists to check: matching on it
    # made a mutated alias crash this script with a KeyError, so CI went red before the skew test
    # ever ran -- and a real alias bug would have been reported as a broken fixture.
    fields = REQUEST_MODELS[track].model_fields
    unmatched = [c for c in spec.feature_columns if c.lower() not in fields]
    if unmatched:
        raise KeyError(f"{track}: no request field named like {unmatched}; extend the matching")
    frame = pd.DataFrame(
        {
            column: _column(
                column,
                fields[column.lower()].annotation,
                fields[column.lower()].metadata,
                rows,
                rng,
            )
            for column in spec.feature_columns
        }
    )
    for column, sentinel in spec.sentinels.items():
        hit = rng.random(rows) < SENTINEL_RATE
        frame.loc[hit, column] = frame[column].dtype.type(sentinel)

    # The planted rule: the first three required numeric features, standardised, plus noise,
    # cut so POSITIVE_RATE of rows are positive -- enough for a stratified split and 5-fold CV.
    numeric = [c for c in spec.numeric_features if frame[c].notna().all()][:3]
    signal = sum((frame[c] - frame[c].mean()) / frame[c].std() for c in numeric)
    score = signal + rng.normal(0, 1.0, rows)
    positive = score > np.quantile(score, 1 - POSITIVE_RATE)

    # Both tracks ship an integer 0/1 target (`FeatureSpec.positive_label`); a track that does not
    # needs its own encoding here rather than a guess.
    if spec.positive_label != 1:
        raise ValueError(f"{track}: positive_label {spec.positive_label!r} is not 1")
    frame.insert(0, spec.id_column, np.arange(rows))
    frame[spec.target_column] = positive.astype("int64")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", required=True, choices=sorted(REQUEST_MODELS))
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = synthetic_snapshot(args.track, args.rows, args.seed)
    positive_rate = float(frame[get_feature_spec(args.track).target_column].mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The same metadata keys ingest writes, so the training run tags itself as synthetic and the
    # CI registry says what its models were trained on.
    _write_parquet_with_metadata(
        frame,
        args.out,
        {
            b"source_used": b"synthetic_ci",
            b"is_fallback": b"False",
            b"track": args.track.encode(),
            b"rows": str(len(frame)).encode(),
            b"positive_rate": f"{positive_rate:.6f}".encode(),
        },
    )
    print(f"{args.out}: {len(frame)} rows, positive rate {positive_rate:.3f}")


if __name__ == "__main__":
    main()
