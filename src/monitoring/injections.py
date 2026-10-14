"""Labelled drift injections on real rows, for evaluating ``decompose.py``.

Every injection starts from rows of the training snapshot and perturbs them in one declared way,
so the cause of the difference between the batch and the reference is known by construction.
Three causes, the stage-10 plan's taxonomy:

- **feature** -- one column's marginal moves: ``add_sigma`` (plus k reference standard deviations),
  ``multiply``, or ``unit`` (a currency or day count restated in another unit). A unit change
  presents as a shifted marginal and is scored as one.
- **population** -- the mix of an *observed* segment changes: rows of one segment are drawn with
  weight k, without replacement, so every row in the batch is a real reference row and nothing
  sits off the data manifold.
- **upstream** -- the pipeline damages values: ``null`` (random masking), ``quantize`` (rounding to
  a lattice), ``duplicate`` (rows repeated).

``label`` -- reweighting by the target -- is **not** a population cause here. It is prior shift,
and a detector that reads features only cannot see it unless the features move with the label
(measured on fraud: Class x20 gave a domain-classifier AUC of 0.503). It is kept as a negative
control whose expected outcome is "nothing claimed", rather than counted as a population case the
detector fails. dtype changes and unseen categories are out of scope: ingest's pandera schema and
the request models refuse them before any batch exists.

**Pre-registration.** ``EVAL_SEED``, the families below and ``MIXED`` are committed before the
detector is written. The evaluation draws its cases from these families with that seed; nobody
picks the individual cases, and a later change here is visible in the history as a change to the
experiment rather than to its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.specs import FeatureSpec

CAUSES = ("feature", "population", "upstream")
# Not a cause: label reweighting, the negative control described above.
PRIOR = "prior"

KINDS = {
    "add_sigma": "feature",
    "multiply": "feature",
    "unit": "feature",
    "segment": "population",
    "label": PRIOR,
    "null": "upstream",
    "quantize": "upstream",
    "duplicate": "upstream",
}

# The first detector (424240c) was evaluated on 20261014 and failed credit's diagonal: a segment
# nearly independent of the other columns reads as a feature shift under leave-one-feature-out
# (reports/decompose_eval_v1.json). The revision it prompted is judged on this seed, committed
# before the revision was written; 20261014's cases were not used to develop it. The families are
# small enough that a new seed redraws some of the same injections: 4 of credit's 26 distinct
# single-cause injections and 7 of fraud's 19 also appear under 20261014, two of them among the
# cases the first detector got wrong. So "a fresh seed" is not "unseen cases" throughout.
FIRST_EVAL_SEED = 20261014
EVAL_SEED = 20261015
CASES_PER_CAUSE = 10
NEGATIVE_CASES = 4

# Strength grids. Each spans "probably below what a 5,000-row batch can show" to "obvious", so the
# matrix reports detection as well as attribution; a grid of only strong shifts would flatter both.
ADD_SIGMA = (0.1, 0.25, 0.5, 1.0)
MULTIPLY = (1.05, 1.1, 1.25, 1.5)
SEGMENT_WEIGHT = (1.5, 2.0, 4.0)
NULL_SHARE = (0.05, 0.1, 0.3)
# Lattice step as a multiple of the column's reference standard deviation.
QUANTIZE_STEP = (0.1, 0.25, 1.0)
DUPLICATE_SHARE = (0.05, 0.1, 0.3)

# Columns whose unit is a fact about the domain rather than about the data: currency amounts can be
# restated in thousands, day counts in years. The divisor is the realistic restatement.
UNITS: dict[str, dict[str, float]] = {
    "credit": {
        "AMT_INCOME_TOTAL": 1000.0,
        "AMT_CREDIT": 1000.0,
        "AMT_ANNUITY": 1000.0,
        "AMT_GOODS_PRICE": 1000.0,
        "DAYS_BIRTH": 365.25,
        "DAYS_EMPLOYED": 365.25,
        "DAYS_REGISTRATION": 365.25,
        "DAYS_ID_PUBLISH": 365.25,
    },
    # Euros restated in cents, the other direction.
    "fraud": {"Amount": 0.01},
}

# Fraud has no categorical column. Its segments are cut on Amount, the only raw business attribute;
# V1..V28 are PCA components, and "the customers with a high V7" is not a segment anyone has.
# The cut points are reference quantiles, so they are computed, not chosen.
NUMERIC_SEGMENTS: dict[str, dict[str, tuple[float, ...]]] = {
    "credit": {},
    "fraud": {"Amount": (0.5, 0.75, 0.9)},
}
# Credit's segments: each categorical column's two largest reference categories.
TOP_CATEGORIES = 2

LABEL_WEIGHT = {"credit": (2.0, 4.0), "fraud": (10.0, 20.0)}


@dataclass(frozen=True)
class Injection:
    """One perturbation. ``value`` is the segment's category, or its numeric cut (``> value``)."""

    kind: str
    column: str | None
    strength: float
    value: Any = None

    @property
    def cause(self) -> str:
        return KINDS[self.kind]

    def describe(self) -> str:
        if self.kind == "segment":
            where = f"> {self.value:.6g}" if isinstance(self.value, float) else f"= {self.value}"
            return f"segment {self.column} {where} x{self.strength:g}"
        if self.kind == "label":
            return f"label x{self.strength:g}"
        if self.kind == "duplicate":
            return f"duplicate {self.strength:.0%}"
        return f"{self.kind} {self.column} {self.strength:g}"


@dataclass(frozen=True)
class Scenario:
    """Injections applied together to one batch. No injections is the no-drift case."""

    name: str
    injections: tuple[Injection, ...]
    seed: int

    @property
    def causes(self) -> frozenset[str]:
        return frozenset(i.cause for i in self.injections if i.cause in CAUSES)

    def describe(self) -> str:
        return " + ".join(i.describe() for i in self.injections) or "no drift"


# Strengths chosen *before* the detector existed, from the plan review's measurements, so that
# neither constituent dominates. The acceptance floor and ratio are fixed here with them.
MIXED_FLOOR = 0.3
MIXED_MAX_RATIO = 2.0
MIXED: dict[str, tuple[Scenario, ...]] = {
    "credit": (
        Scenario(
            "mixed-feature-population",
            (
                Injection("multiply", "AMT_CREDIT", 1.05),
                Injection("segment", "NAME_INCOME_TYPE", 4.0, "Pensioner"),
            ),
            seed=101,
        ),
        Scenario(
            "mixed-feature-upstream",
            (Injection("multiply", "AMT_CREDIT", 1.1), Injection("null", "EXT_SOURCE_2", 0.3)),
            seed=102,
        ),
    ),
    "fraud": (
        Scenario(
            "mixed-feature-population",
            (Injection("add_sigma", "V14", 0.3), Injection("segment", "Amount", 4.0, 100.0)),
            seed=201,
        ),
        Scenario(
            "mixed-feature-upstream",
            (Injection("add_sigma", "V14", 0.5), Injection("null", "V4", 0.3)),
            seed=202,
        ),
    ),
}


def _segment_mask(pool: pd.DataFrame, injection: Injection) -> np.ndarray:
    column = pool[injection.column]
    if isinstance(injection.value, float):
        return (column > injection.value).to_numpy()
    return (column == injection.value).to_numpy()


def _body(series: pd.Series, spec: FeatureSpec, column: str) -> pd.Series:
    """The values a feature transform touches: not null, not the column's sentinel."""
    mask = series.notna()
    if column in spec.sentinels:
        mask &= series != spec.sentinels[column]
    return mask


def apply(scenario: Scenario, pool: pd.DataFrame, spec: FeatureSpec, rows: int) -> pd.DataFrame:
    """Draw ``rows`` from ``pool`` and apply ``scenario``. Returns the feature columns only.

    Order: reweighting decides which rows are drawn; the value transforms -- feature moves, nulls,
    quantization -- run in the scenario's own order (``MIXED`` lists feature moves first);
    duplicates are written last, so a duplicate repeats a row as the rest of the pipeline left it. ``pool`` must carry the target column for ``label``.
    """
    rng = np.random.default_rng(scenario.seed)
    weights = np.ones(len(pool))
    for injection in scenario.injections:
        if injection.kind == "segment":
            weights[_segment_mask(pool, injection)] *= injection.strength
        elif injection.kind == "label":
            positive = (pool[spec.target_column] == spec.positive_label).to_numpy()
            weights[positive] *= injection.strength
    batch = pool.sample(
        rows, weights=weights, replace=False, random_state=int(rng.integers(2**31))
    ).reset_index(drop=True)
    batch = batch[list(spec.feature_columns)].copy()

    for injection in scenario.injections:
        column = injection.column
        if injection.kind in ("add_sigma", "multiply", "unit", "quantize"):
            body = _body(batch[column], spec, column)
            values = batch.loc[body, column].astype("float64")
            sigma = pool.loc[_body(pool[column], spec, column), column].std()
            if injection.kind == "add_sigma":
                values = values + injection.strength * sigma
            elif injection.kind == "multiply":
                values = values * injection.strength
            elif injection.kind == "unit":
                values = values / injection.strength
            else:
                step = injection.strength * sigma
                values = (values / step).round() * step
            batch[column] = batch[column].astype("float64")
            batch.loc[body, column] = values
        elif injection.kind == "null":
            batch.loc[rng.random(rows) < injection.strength, column] = np.nan

    for injection in scenario.injections:
        if injection.kind == "duplicate":
            copies = round(injection.strength * rows)
            originals = batch.iloc[: rows - copies]
            repeated = originals.sample(copies, replace=True, random_state=int(rng.integers(2**31)))
            batch = pd.concat([originals, repeated], ignore_index=True)
    return batch


def families(track: str, spec: FeatureSpec, pool: pd.DataFrame) -> dict[str, list[Injection]]:
    """Every injection the evaluation may draw, per kind. The grid, not a selection from it."""
    numeric = list(spec.numeric_features)
    segments = []
    for column in spec.categorical_features:
        for category in pool[column].value_counts().index[:TOP_CATEGORIES]:
            segments += [Injection("segment", column, w, category) for w in SEGMENT_WEIGHT]
    for column, quantiles in NUMERIC_SEGMENTS[track].items():
        for q in quantiles:
            cut = float(pool[column].quantile(q))
            segments += [Injection("segment", column, w, cut) for w in SEGMENT_WEIGHT]
    return {
        "add_sigma": [Injection("add_sigma", c, s) for c in numeric for s in ADD_SIGMA],
        "multiply": [Injection("multiply", c, s) for c in numeric for s in MULTIPLY],
        "unit": [Injection("unit", c, s) for c, s in UNITS[track].items()],
        "segment": segments,
        "null": [Injection("null", c, s) for c in spec.feature_columns for s in NULL_SHARE],
        "quantize": [Injection("quantize", c, s) for c in numeric for s in QUANTIZE_STEP],
        "duplicate": [Injection("duplicate", None, s) for s in DUPLICATE_SHARE],
    }


def evaluation_suite(
    track: str, spec: FeatureSpec, pool: pd.DataFrame, seed: int = EVAL_SEED
) -> list[Scenario]:
    """Single-cause cases drawn from the families, the negatives, the prior control, the mixed."""
    rng = np.random.default_rng(seed)
    grid = families(track, spec, pool)
    scenarios: list[Scenario] = []
    for cause in CAUSES:
        # Kind first, then a member of it: drawing from the pooled grid would make ``duplicate``,
        # which has no column to vary, three cases in a hundred.
        kinds = [k for k, c in KINDS.items() if c == cause]
        for n in range(CASES_PER_CAUSE):
            members = grid[kinds[rng.integers(len(kinds))]]
            injection = members[rng.integers(len(members))]
            scenarios.append(Scenario(f"{cause}-{n}", (injection,), seed=int(rng.integers(2**31))))
    for n in range(NEGATIVE_CASES):
        scenarios.append(Scenario(f"none-{n}", (), seed=int(rng.integers(2**31))))
    for n, weight in enumerate(LABEL_WEIGHT[track]):
        scenarios.append(
            Scenario(f"prior-{n}", (Injection("label", None, weight),), int(rng.integers(2**31)))
        )
    return scenarios + list(MIXED[track])
