"""Which cause moved a batch: feature, population, or upstream -- a score per cause, not a label.

Run from the repo root (needs the training snapshot)::

    uv run python -m src.monitoring.decompose --track credit --evaluate
    uv run python -m src.monitoring.decompose --track credit --case population-2

Three causes, defined by how ``injections.py`` makes them: one column's marginal moves
(**feature**), the mix of an observed segment moves while every row stays a real one
(**population**), or the pipeline damages values -- nulls, a lattice, repeated rows (**upstream**).
The output is ``{"feature": s_f, "population": s_p, "upstream": s_u}``, each in [0, 1] and not
summing to 1: a batch with no drift must score low on all three, which a softmax cannot do. A cause
is *claimed* at ``CLAIM`` (0.5).

**Order.** Upstream rules run first, because their damage would otherwise reach the classifier
and be charged to the other two causes (30% repeated rows alone gave an AUC of 0.57-0.59 in the
plan review). Columns the rules flag are dropped and the batch is de-duplicated before the domain
classifier sees it.

1. **Rules** (``s_u``, 0 or 1 -- they are deterministic tests, not degrees). Duplicate share; per
   column a null rate in excess of the reference; per numeric column collisions (values repeating
   another) in excess -- a lattice. Each excess must hold *both* marginally and conditionally,
   which is what keeps the other two causes out. Pensioners have no ``OCCUPATION_TYPE`` (99.99%
   null), so drawing more of them raises that column's null rate while ``P(null | the other
   columns)`` stays where it was; Revolving loans have round limits, so drawing more of them adds
   ``AMT_CREDIT`` collisions that the pool reweighted to the batch's mix reproduces. A feature
   shift moves what the conditional models predict but not the marginal counts.

   **Every rule also needs a minimum count** (``_binomial_floor``, ``_poisson_floor``). The
   empirical gates are quantiles of 200 reference draws, and where the reference never shows the
   event they collapse: fraud has no nulls, so its null gates sat below zero and one null claimed
   upstream; credit's reference has no repeated rows, so one repeated row did. The floor is the
   fewest events whose probability, at the reference's own rate with a rule-of-succession pseudo-
   count, is at most the rule's share of ``1 - LEVEL`` -- split again across columns for the
   per-column null rules. A batch claims only past both. The floor binds only where the empirical
   gate degenerates: fraud's nulls, credit's duplicates.
2. **Domain classifier.** LightGBM, reference sample vs batch, AUC on a held-out half. It is
   compared with the AUC of the same classifier between two *reference* samples, drawn
   ``NULL_DRAWS`` times: drift is detected when the batch's AUC is above that null's ``LEVEL``
   quantile. Without the null the classifier is not a detector -- on pure reference data its AUC
   exceeds 0.5 in about four draws in ten (``evaluate`` reports the measured share).
3. **Attribution, a column at a time** (``_attribute``). The most important column either moved
   (feature) or is the attribute whose mix moved (population). Population means real reference
   rows, reweighted on that column's PSI bins, reproduce the batch; feature means they cannot, and
   only dropping the column removes the separation. Each round takes the reading with the smaller
   residual and carries the residual into the next, until it is inside the null.

**Two detectors were evaluated.** The first (424240c) attributed by leave-one-feature-out alone: a
column carrying all the separation was a feature shift. Its pre-registered evaluation failed
credit's diagonal, because a segment nearly independent of the other columns -- Tuesdays, car
owners -- also puts all its separation in one column (``reports/decompose_eval_v1.json``). The
reweighting reading replaced it, developed on cases outside that evaluation and judged on a seed
committed before it was written (``injections.EVAL_SEED``).

**No constant here was fitted to an injection.** The thresholds are quantiles of distributions
measured between reference samples, at the one level ``LEVEL``; the attribution margin is derived
from ``CLAIM``; the bins are the committed PSI bins; the classifier's settings are the plan
review's and were not varied.

**The gate is not an alarm.** The classifier sees joint structure PSI cannot: ``AMT_CREDIT`` x1.02
gave an AUC of 0.78 at a PSI of 0.003, because the ratio to ``AMT_GOODS_PRICE`` broke. It decides
whether there is anything to attribute. Paging on it would page on every batch; PSI owns the alert.

**Identifiability limit.** A shift inside a column's own support, in a column independent of the
rest, *is* a reweighting of that column: no detector reading the batch can tell the two apart.
There the two readings in ``_attribute`` leave residuals that differ by less than the null's noise,
and the smaller one wins with no margin -- so which cause is claimed is close to a coin toss.
``EXT_SOURCE_3`` x1.1 is the measured example: 0.48 feature, 0.52 population. Both residuals are
kept in the evidence so a reader can see how close a call was.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson
from sklearn.metrics import roc_auc_score

from src.features.specs import FEATURE_SPECS, FeatureSpec, get_feature_spec
from src.monitoring import injections, psi

logger = logging.getLogger(__name__)

REPORT_PATH = psi.PROJECT_ROOT / "reports" / "decompose_eval.json"

ROWS = 5_000
NULL_DRAWS = 200
# The one significance level. Every threshold below is a quantile at this level of a distribution
# measured on reference data, or derived from it.
LEVEL = 0.99
CLAIM = 0.5
# Three upstream rules each tested at LEVEL would claim upstream on ~3% of clean batches.
RULES = 3
# One rule's share of the false-claim rate.
ALPHA = (1 - LEVEL) / RULES
# Reweighted resamples per batch for the expected collision count. A precision setting -- more
# narrows the estimate, none moves the gate.
RESAMPLES = 20
SPLIT_SEED = 1

CLASSIFIER = {
    "n_estimators": 100,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "n_jobs": 4,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


@dataclass
class Calibration:
    """Everything the detector compares a batch against, measured on reference data only."""

    track: str
    spec: FeatureSpec
    reference: pd.DataFrame
    pool: pd.DataFrame
    rows: int
    null_auc: np.ndarray
    auc_gate: float
    null_models: dict[str, Any]
    null_gate: float
    collision_mean: pd.Series
    collision_sd: pd.Series
    collision_gate: float
    duplicate_gate: float
    marginal_null_gate: float
    reference_null_rate: pd.Series
    # Repeated-row pairs in the reference sample before it was de-duplicated, and its size: the
    # duplicate floor's rate.
    reference_pairs: int
    reference_rows: int
    bins: dict[str, dict[str, Any]]

    def auc_gate_at(self, rows: int, reference_rows: float | None = None) -> float:
        """The AUC gate for a batch of ``rows`` against ``reference_rows`` (effective) rows.

        The null was drawn at ``self.rows`` against the whole reference sample. Under no shift an
        AUC's sampling error scales with ``sqrt(1/n_reference + 1/n_current)`` (Hanley & McNeil,
        1982), so the gate's excess over 0.5 is rescaled rather than re-drawn for every batch size.
        An approximation, not a derivation: the null's mean is 0.497, not 0.5, and for a
        reweighted reference Kish's effective size stands in for the variance of a weighted AUC
        and for what weighted training does to overfitting -- neither of which was measured.
        """
        n = len(self.reference)
        m = n if reference_rows is None else reference_rows
        scale = math.sqrt((1 / m + 1 / rows) / (1 / n + 1 / self.rows))
        return 0.5 + (self.auc_gate - 0.5) * scale


@dataclass
class Attribution:
    """Scores per cause, the causes claimed, and the evidence behind them."""

    scores: dict[str, float]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def claimed(self) -> list[str]:
        return [c for c in injections.CAUSES if self.scores[c] >= CLAIM]


# --- Data ----------------------------------------------------------------------------------------


def split(frame: pd.DataFrame, rows: int = ROWS, seed: int = SPLIT_SEED):
    """Reference sample and the disjoint pool batches are drawn from.

    Disjoint on purpose: a batch resampled from the reference sample itself shares rows with it,
    and the classifier memorises them -- the plan review measured an AUC of 0.388 there.
    """
    order = np.random.default_rng(seed).permutation(len(frame))
    reference = frame.iloc[order[:rows]].reset_index(drop=True)
    pool = frame.iloc[order[rows:]].reset_index(drop=True)
    return reference, pool


def load_snapshot(track: str) -> pd.DataFrame:
    """The snapshot the served model was trained on, as the PSI reference names it."""
    return pd.read_parquet(psi.snapshot_path(psi.load_summary(track)))


def _encode(frames: list[pd.DataFrame], spec: FeatureSpec) -> list[pd.DataFrame]:
    """Categoricals as one shared ``category`` dtype across ``frames``, nulls left as nulls."""
    categorical = [c for c in spec.categorical_features if c in frames[0].columns]
    out = [f.copy() for f in frames]
    for column in categorical:
        levels = sorted(set().union(*(f[column].dropna().astype(str) for f in frames)))
        for f in out:
            f[column] = pd.Categorical(f[column].astype("string"), categories=levels)
    return out


def _classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(random_state=seed, **CLASSIFIER)


# --- Upstream rules ------------------------------------------------------------------------------


def _collisions(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Per numeric column, values that repeat an earlier value -- ignoring nulls and the sentinel.

    A lattice shows up as thousands of collisions in a column that had a handful. ``frame`` must
    already be de-duplicated, *after* dropping any column the null rule flagged: repeated rows add
    a collision to every column at once, which is the duplicate rule's finding; and a natural
    duplicate pair that masking made differ in one column would otherwise survive the dedupe and
    collide everywhere else (seen on fraud, whose pool repeats 3.17% of rows on its features).
    """
    counts = {}
    for column in spec.numeric_features:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if column in spec.sentinels:
            values = values[values != spec.sentinels[column]]
        counts[column] = len(values) - values.nunique()
    return pd.Series(counts, dtype="float64")


def _fit_null_models(reference: pd.DataFrame, spec: FeatureSpec, seed: int) -> dict[str, Any]:
    """Per column with nulls in the reference: P(null | the other columns)."""
    models: dict[str, Any] = {}
    for column in spec.feature_columns:
        target = reference[column].isna().to_numpy()
        if target.any() and not target.all():
            (x,) = _encode([reference.drop(columns=[column])], spec)
            models[column] = _classifier(seed).fit(x, target)
    return models


def _null_z(
    cal_reference: pd.DataFrame, models, spec, frame: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Per column: (observed nulls - expected nulls) / sd, expected from the conditional model.

    A column with no nulls in the reference expects ``1 / (n + 2)`` per row -- the rule of
    succession -- rather than 0, so one null does not read as an infinite excess. Also returns
    each column's mean expected rate, which the count floor is taken at.
    """
    floor = 1 / (len(cal_reference) + 2)
    z, rate = {}, {}
    for column in spec.feature_columns:
        observed = frame[column].isna().sum()
        if column in models:
            (x,) = _encode(
                [cal_reference.drop(columns=[column]), frame.drop(columns=[column])], spec
            )[1:]
            p = models[column].predict_proba(x)[:, 1]
        else:
            p = np.full(len(frame), cal_reference[column].isna().mean())
        p = np.clip(p, floor, 1 - floor)
        z[column] = (observed - p.sum()) / math.sqrt(np.sum(p * (1 - p)))
        rate[column] = float(p.mean())
    return pd.Series(z, dtype="float64"), pd.Series(rate, dtype="float64")


def _marginal_null_z(reference_rate: pd.Series, frame: pd.DataFrame, n_reference: int) -> pd.Series:
    """Per column: the two-sample binomial z of the batch's null rate against the reference's."""
    floor = 1 / (n_reference + 2)
    p = reference_rate.clip(floor, 1 - floor)
    observed = frame[reference_rate.index].isna().mean()
    return (observed - p) / np.sqrt(p * (1 - p) * (1 / len(frame) + 1 / n_reference))


def _expected_collisions(
    cal: Calibration, batch: pd.DataFrame, column: str, seed: int
) -> tuple[float, float]:
    """Collisions ``column`` would show if only the batch's *mix* had moved, with their spread.

    The pool is reweighted towards the batch by a classifier's density ratio ``p / (1 - p)`` and
    resampled at the batch's size. The classifier never sees ``column``: a population shift shows
    in the *other* columns, so the reweighted pool reproduces it, ``column``'s collisions included
    -- Revolving loans have round credit limits, and drawing more of them adds collisions to
    ``AMT_CREDIT`` that no pipeline caused. A shift or a lattice in ``column`` itself is invisible
    to the weights, so the expectation stays the pool's: a shift keeps the batch's collisions where
    they were, a lattice multiplies them.

    Training on ``column`` too was tried first and is wrong the other way: the weights then chase
    a shifted column into a range where the pool's values are spread differently, and an honest
    feature shift reads as a lattice.
    """
    spec = cal.spec
    others = [c for c in batch.columns if c != column]
    reference, current = _encode([cal.reference[others], batch[others]], spec)
    x = pd.concat([reference, current], ignore_index=True)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    model = _classifier(seed).fit(x, y)
    (pool,) = _encode([cal.pool[others]], spec)
    p = model.predict_proba(pool.astype(x.dtypes.to_dict()))[:, 1]
    weights = p / (1 - p)
    columns = list(batch.columns)
    counts = [
        _collisions(
            cal.pool[columns]
            .sample(len(batch), weights=weights, random_state=seed + k)
            .drop_duplicates(),
            spec,
        )[column]
        for k in range(RESAMPLES)
    ]
    return float(np.mean(counts)), float(np.std(counts, ddof=1))


def _duplicate_share(frame: pd.DataFrame) -> float:
    return float(frame.duplicated().mean())


def _rule_gate(values: list[float]) -> float:
    """The null quantile one rule is tested at, Bonferroni-corrected over the ``RULES`` rules."""
    return float(np.quantile(values, 1 - ALPHA))


def _binomial_floor(n: int, p: float, alpha: float) -> int:
    """Fewest events ``k`` with ``P(Binom(n, p) >= k) <= alpha``.

    For the conditional null rule the count is Poisson-binomial -- each row its own ``p_i`` -- and
    ``p`` is their mean. Hoeffding (1956, Theorem 4): for ``k - 1 >= n * p`` its upper tail is at
    most the binomial's, so the floor is conservative there; at these alphas ``k`` is always past
    that point, and the check keeps it so. The guarantee is nominal all the same: the
    conditional models are overconfident out of sample, and the empirical gate is what holds.
    """
    k = int(binom.isf(alpha, n, p)) + 1
    if k - 1 < n * p:
        raise ValueError(f"floor {k} for n={n}, p={p} is short of Hoeffding's condition k-1 >= np")
    return k


def _poisson_floor(rate: float, alpha: float) -> int:
    """Fewest events ``k`` with ``P(Poisson(rate) >= k) <= alpha``."""
    return int(poisson.isf(alpha, rate)) + 1


def _pairs(frame: pd.DataFrame) -> int:
    """Pairs of identical rows: a group of ``c`` copies is ``c * (c - 1) / 2`` of them."""
    sizes = frame.value_counts(dropna=False).to_numpy()
    return int((sizes * (sizes - 1) // 2).sum())


def _duplicate_floor(cal: Calibration, rows: int) -> int:
    """Fewest repeated rows a batch of ``rows`` needs before the duplicate rule may claim.

    Duplicates are pair events, so the rate is per pair of rows, not per row: the reference's own
    pairs plus one (the rule of succession, as the null floor) over its ``N(N-1)/2`` pairs, scaled to
    the batch's ``n(n-1)/2``. The batch's repeated-row count is at most its pair count, so testing
    it against the pairs' tail is conservative. Pools with groups of three or more copies (fraud
    has some of 76) are overdispersed against a Poisson; the empirical gate covers the batch size it
    was calibrated at, and the floor binds only where that gate is zero.
    """
    n, total = rows, cal.reference_rows
    rate = (cal.reference_pairs + 1) * n * (n - 1) / (total * (total - 1))
    return _poisson_floor(rate, ALPHA)


def _null_floors(cal: Calibration, rows: int, conditional_rate: pd.Series) -> pd.Series:
    """Per column, the fewest nulls either null rule needs at ``rows``. Alpha is split across
    columns: the rules are tested on every column and the largest excess is the one read."""
    alpha = ALPHA / len(cal.spec.feature_columns)
    floor = 1 / (len(cal.reference) + 2)
    marginal = cal.reference_null_rate.clip(floor, 1 - floor)
    return pd.Series(
        {
            c: max(
                _binomial_floor(rows, float(marginal[c]), alpha),
                _binomial_floor(rows, float(conditional_rate[c]), alpha),
            )
            for c in cal.spec.feature_columns
        }
    )


# --- Domain classifier ---------------------------------------------------------------------------


def domain_auc(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    spec: FeatureSpec,
    seed: int = 0,
    importance: bool = False,
    weights: np.ndarray | None = None,
) -> tuple[float, pd.Series | None]:
    """Held-out AUC of reference-vs-current, and each feature's permutation AUC drop on it.

    ``weights`` reweight the reference rows, in training and in the held-out AUC alike, so the
    AUC is that of the batch against the reference *as reweighted*.
    """
    x = pd.concat(_encode([reference, current], spec), ignore_index=True)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    w = np.ones(len(x))
    if weights is not None:
        w[: len(reference)] = weights * len(reference) / weights.sum()
    order = np.random.default_rng(seed).permutation(len(x))
    train, test = order[: len(x) // 2], order[len(x) // 2 :]
    model = _classifier(seed).fit(x.iloc[train], y[train], sample_weight=w[train])
    held = x.iloc[test].reset_index(drop=True)
    auc = roc_auc_score(y[test], model.predict_proba(held)[:, 1], sample_weight=w[test])
    if not importance:
        return float(auc), None

    rng = np.random.default_rng(seed)
    drops = {}
    for column in held.columns:
        shuffled = held.copy()
        # iloc then set_axis keeps the column's dtype; a bare numpy array would turn a category
        # column into object, which LightGBM refuses.
        shuffled[column] = held[column].iloc[rng.permutation(len(held))].set_axis(held.index)
        score = model.predict_proba(shuffled)[:, 1]
        drops[column] = auc - roc_auc_score(y[test], score, sample_weight=w[test])
    return float(auc), pd.Series(drops)


def _bin_ratio(
    entry: dict[str, Any], reference: pd.Series, weights: np.ndarray, current: pd.Series
) -> np.ndarray:
    """Per reference row, the factor that gives the reweighted reference the batch's bin mix.

    The bins are the committed PSI bins, so "reweight on this column" means the same coarse
    resolution everywhere it is used. A bin the batch fills and the reference left empty cannot be
    reproduced by any weighting; its mass stays unexplained, which is the point.
    """
    reference_bins = psi.bin_index(entry, reference)
    size = psi.n_bins(entry)
    p = np.bincount(reference_bins, weights=weights, minlength=size) / weights.sum()
    q = np.bincount(psi.bin_index(entry, current), minlength=size) / len(current)
    ratio = np.divide(q, p, out=np.zeros(size), where=p > 0)
    return ratio[reference_bins]


def _effective_rows(weights: np.ndarray) -> float:
    """Kish's effective sample size of a weighted sample."""
    return float(weights.sum() ** 2 / np.sum(weights**2))


# --- Calibration and decomposition ---------------------------------------------------------------


def calibrate(
    track: str,
    reference: pd.DataFrame,
    pool: pd.DataFrame,
    rows: int = ROWS,
    draws: int = NULL_DRAWS,
) -> Calibration:
    """Measure every threshold on ``draws`` reference-vs-reference comparisons."""
    spec = get_feature_spec(track)
    features = list(spec.feature_columns)
    ref = reference[features].drop_duplicates().reset_index(drop=True)
    null_models = _fit_null_models(ref, spec, seed=0)
    reference_null_rate = ref.isna().mean()

    aucs, null_t, marginal_t, collisions, duplicates = [], [], [], [], []
    for b in range(draws):
        draw = pool.sample(rows, random_state=b)[features].reset_index(drop=True)
        deduped = draw.drop_duplicates()
        duplicates.append(_duplicate_share(draw))
        collisions.append(_collisions(deduped, spec))
        null_t.append(_null_z(ref, null_models, spec, draw)[0].max())
        marginal_t.append(_marginal_null_z(reference_null_rate, draw, len(ref)).max())
        aucs.append(domain_auc(ref, deduped, spec, seed=b)[0])

    collision_frame = pd.DataFrame(collisions)
    collision_mean = collision_frame.mean()
    # A count cannot be resolved finer than one: a continuous column with no collisions in any
    # draw would otherwise have a standard deviation of 0 and flag on its first coincidence.
    collision_sd = collision_frame.std().clip(lower=1.0)
    collision_t = ((collision_frame - collision_mean) / collision_sd).max(axis=1)

    null_auc = np.asarray(aucs)
    return Calibration(
        track=track,
        spec=spec,
        reference=ref,
        pool=pool[features].reset_index(drop=True),
        rows=rows,
        null_auc=null_auc,
        auc_gate=float(np.quantile(null_auc, LEVEL)),
        null_models=null_models,
        null_gate=_rule_gate(null_t),
        marginal_null_gate=_rule_gate(marginal_t),
        collision_mean=collision_mean,
        collision_sd=collision_sd,
        collision_gate=_rule_gate(list(collision_t)),
        duplicate_gate=_rule_gate(duplicates),
        reference_null_rate=reference_null_rate,
        reference_pairs=_pairs(reference[features]),
        reference_rows=len(reference),
        bins=psi.load_summary(track)["features"],
    )


def decompose(cal: Calibration, current: pd.DataFrame, seed: int = 0) -> Attribution:
    """Score ``current`` against the calibration. See the module docstring for the method."""
    spec = cal.spec
    batch = current[list(spec.feature_columns)].reset_index(drop=True)
    evidence: dict[str, Any] = {"rows": len(batch)}

    # 1. Upstream rules.
    duplicate_share = _duplicate_share(batch)
    conditional, conditional_rate = _null_z(cal.reference, cal.null_models, spec, batch)
    marginal = _marginal_null_z(cal.reference_null_rate, batch, len(cal.reference))
    null_floor = _null_floors(cal, len(batch), conditional_rate)
    nulls = batch.isna().sum()
    past_gates = [
        c
        for c in spec.feature_columns
        if conditional[c] > cal.null_gate and marginal[c] > cal.marginal_null_gate
    ]
    nulled = [c for c in past_gates if nulls[c] >= null_floor[c]]
    # Past both gates but short of the floor: what the floor exists to stop, kept visible.
    held = {c: [int(nulls[c]), int(null_floor[c])] for c in past_gates if c not in nulled}
    deduped = batch.drop(columns=nulled).drop_duplicates().reset_index(drop=True)
    observed = _collisions(deduped, spec)
    # Marginal AND conditional, as for nulls. Against the plain pool, a population shift adds
    # collisions (Revolving loans' round limits); against the pool reweighted to the batch's mix,
    # it does not, and a lattice still does. The marginal test runs first because it is cheap and
    # the conditional one costs a classifier per column.
    marginal_collision = (observed - cal.collision_mean[observed.index]) / cal.collision_sd
    collision: dict[str, float] = {}
    for column in marginal_collision.index[marginal_collision > cal.collision_gate]:
        expected, spread = _expected_collisions(cal, deduped, column, seed)
        # The larger spread of the resamples' and the null's: a handful of resamples
        # under-estimates it, and the gate was calibrated against the null's.
        z = (observed[column] - expected) / max(spread, cal.collision_sd[column])
        if z > cal.collision_gate:
            collision[column] = z
    quantized = list(collision)
    duplicate_floor = _duplicate_floor(cal, len(batch))
    duplicated = (
        duplicate_share > cal.duplicate_gate and int(batch.duplicated().sum()) >= duplicate_floor
    )
    evidence["upstream"] = {
        "duplicate_share": round(duplicate_share, 4),
        "duplicate_gate": round(cal.duplicate_gate, 4),
        "duplicate_floor": duplicate_floor,
        "nulled": {c: round(float(conditional[c]), 1) for c in nulled},
        "nulls_held_by_floor": held,
        "quantized": {c: round(float(z), 1) for c, z in collision.items()},
    }
    s_u = 1.0 if (nulled or quantized or duplicated) else 0.0

    # 2. Domain classifier, on what the rules left.
    excluded = sorted(set(nulled) | set(quantized))
    kept = [c for c in spec.feature_columns if c not in excluded]
    clean = batch[kept].drop_duplicates().reset_index(drop=True)
    gate = cal.auc_gate_at(len(clean))
    auc, drops = domain_auc(cal.reference[kept], clean, spec, seed=seed, importance=True)
    evidence.update(
        excluded=excluded, rows_after_dedupe=len(clean), auc=round(auc, 4), auc_gate=round(gate, 4)
    )
    s_f = s_p = 0.0
    evidence["detected"] = auc > gate
    # Below this margin the null's noise alone could move a share across CLAIM: a share's error
    # is at most (gate - 0.5) / (auc - 0.5).
    evidence["attributable"] = auc - 0.5 >= (gate - 0.5) / CLAIM
    if evidence["detected"] and evidence["attributable"]:
        s_f, s_p, evidence["rounds"] = _attribute(cal, clean, kept, auc, drops, seed)
    evidence["psi_top"] = _psi_top(cal.track, batch)
    return Attribution({"feature": s_f, "population": s_p, "upstream": s_u}, evidence)


def _attribute(
    cal: Calibration,
    batch: pd.DataFrame,
    columns: list[str],
    auc: float,
    drops: pd.Series,
    seed: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Split the classifier's separation between feature and population, a column at a time.

    Each round takes the most important column and asks two questions of the current
    comparison. *Reweighted* on that column's bins, how much of the separation is left? What is
    removed is population: real reference rows, drawn in the batch's proportions of one observed
    attribute, reproduce it. With the column also *dropped*, how much is left? The column carried
    the difference; the part of it reweighting did not remove is feature -- values that moved, or
    joints that broke, which no reweighting of real rows produces. The next round starts from
    what is left, with the column dropped and the weights kept, and the rounds stop when that
    residual is inside the null.

    The plan review's leave-one-feature-out alone read a segment independent of every other column
    as a feature shift, because its drift is all in one column (first evaluation,
    ``reports/decompose_eval_v1.json``). Reweighting separates them: drawing more Tuesdays is a
    reweighting; adding 0.5 sigma to an hour is not.
    """
    spec = cal.spec
    weights = np.ones(len(cal.reference))
    remaining, s_f, s_p, rounds = 1.0, 0.0, 0.0, []
    current = auc
    while columns:
        top = str(drops.idxmax())
        rest = [c for c in columns if c != top]
        separation = current - 0.5
        reweighted = weights * _bin_ratio(cal.bins[top], cal.reference[top], weights, batch[top])
        # Two readings of the top column. Population: real rows reweighted on its bins reproduce
        # the batch, so what is left after reweighting -- and after dropping it -- is the rest.
        # Feature: its values moved, so reweighting on them only distorts every column correlated
        # with it (a unit change puts the whole batch in one bin, and the reweighted reference is
        # then the lowest earners); what is left is what dropping it leaves. The reading with the
        # smaller residual is taken, with no margin: where the two cannot be told apart -- a
        # segment independent of everything, a shift inside the column's own support -- the
        # residuals differ by noise and the call is close to arbitrary. Both are recorded.
        if rest:
            auc_lofo, drops_lofo = domain_auc(
                cal.reference[rest], batch[rest], spec, seed, importance=True, weights=weights
            )
        else:
            auc_lofo, drops_lofo = 0.5, None
        # No reference row falls in the batch's bins: no reweighting reproduces anything, so only
        # the feature reading is available.
        auc_rw, auc_rest, drops_rest = current, 1.0, None
        if reweighted.sum() > 0:
            auc_rw, _ = domain_auc(
                cal.reference[columns], batch[columns], spec, seed=seed, weights=reweighted
            )
            if rest:
                auc_rest, drops_rest = domain_auc(
                    cal.reference[rest],
                    batch[rest],
                    spec,
                    seed,
                    importance=True,
                    weights=reweighted,
                )
            else:
                auc_rest, drops_rest = 0.5, None
        removed_by_weights = float(np.clip(1 - (auc_rw - 0.5) / separation, 0, 1))
        reading = "population" if auc_rest <= auc_lofo else "feature"
        if reading == "population":
            carried = float(np.clip(1 - (auc_rest - 0.5) / separation, 0, 1))
            population = min(removed_by_weights, carried)
            weights, current, drops = reweighted, auc_rest, drops_rest
        else:
            carried = float(np.clip(1 - (auc_lofo - 0.5) / separation, 0, 1))
            population = 0.0
            current, drops = auc_lofo, drops_lofo
        s_p += remaining * population
        s_f += remaining * (carried - population)
        rounds.append(
            {
                "column": top,
                "reading": reading,
                "share": round(remaining * carried, 3),
                "population": round(remaining * population, 3),
                "auc_reweighted": round(auc_rw, 4),
                "auc_reweighted_without": round(auc_rest, 4),
                "auc_without": round(auc_lofo, 4),
            }
        )
        remaining *= 1 - carried
        columns = rest
        if current <= cal.auc_gate_at(len(batch), _effective_rows(weights)):
            break
    return s_f, s_p, rounds


def _psi_top(track: str, batch: pd.DataFrame, k: int = 3) -> dict[str, float]:
    """The three largest per-feature PSIs, as evidence only -- they feed no score."""
    from src.monitoring.drift import InsufficientCurrentData

    try:
        values = psi.psi_by_feature(psi.load_summary(track), batch, min_rows=1)
    except InsufficientCurrentData as error:  # a column nulled to nothing has no PSI
        return {"unavailable": str(error)}
    top = sorted(values.items(), key=lambda item: -item[1])[:k]
    return {c: round(v, 4) for c, v in top}


# --- Evaluation ----------------------------------------------------------------------------------

NOT_ATTRIBUTED = "detected, not attributed"
NOTHING = "nothing"


def _group(name: str) -> str:
    return name.split("-", 1)[0]


def matrix(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Injected cause x claimed cause, over the single-cause cases.

    A case claiming two causes counts in both columns. Detected-but-unclaimed and undetected are
    columns of their own, so a weak injection shows as missed rather than vanishing.
    """
    columns = [*injections.CAUSES, NOT_ATTRIBUTED, NOTHING]
    table = {cause: dict.fromkeys(columns, 0) for cause in injections.CAUSES}
    for case in cases:
        if _group(case["name"]) not in injections.CAUSES:
            continue
        row = table[case["causes"][0]]
        for cause in case["claimed"]:
            row[cause] += 1
        if not case["claimed"]:
            row[NOT_ATTRIBUTED if case["evidence"].get("detected") else NOTHING] += 1
    return table


def acceptance(report: dict[str, Any]) -> dict[str, bool]:
    """The pre-registered criteria (plan, ``injections.MIXED_FLOOR`` / ``MIXED_MAX_RATIO``)."""
    cases = {case["name"]: case for case in report["cases"]}
    table = report["matrix"]
    checks = {
        # Each injected cause is claimed more often than any wrong cause is.
        "diagonal_dominates": all(
            row[cause] > max(row[other] for other in injections.CAUSES if other != cause)
            for cause, row in table.items()
        ),
        "negatives_claim_nothing": all(
            not case["claimed"] for name, case in cases.items() if _group(name) == "none"
        ),
        # The uncalibrated reading of the classifier, AUC > 0.5, fires on pure reference data.
        "uncalibrated_fires_on_reference": report["null"]["share_above_0.5"] > 0,
    }
    for name, case in cases.items():
        if _group(name) == "mixed":
            scores = [case["scores"][cause] for cause in case["causes"]]
            low, high = min(scores), max(scores)
            checks[name] = (
                low >= injections.MIXED_FLOOR and high <= injections.MIXED_MAX_RATIO * low
            )
    return checks


def _case(scenario: injections.Scenario, attribution: Attribution) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "injected": scenario.describe(),
        "causes": sorted(scenario.causes),
        "scores": {c: round(v, 3) for c, v in attribution.scores.items()},
        "claimed": attribution.claimed,
        "evidence": attribution.evidence,
    }


def evaluate(
    track: str, draws: int = NULL_DRAWS, seed: int = injections.EVAL_SEED
) -> dict[str, Any]:
    """Calibrate on the snapshot, then run every pre-registered case through the detector."""
    reference, pool = split(load_snapshot(track))
    cal = calibrate(track, reference, pool, draws=draws)
    cases = [
        _case(scenario, decompose(cal, injections.apply(scenario, pool, cal.spec, ROWS)))
        for scenario in injections.evaluation_suite(track, cal.spec, pool, seed=seed)
    ]
    null = cal.null_auc
    report = {
        "track": track,
        "seed": seed,
        "rows": ROWS,
        "null": {
            "draws": len(null),
            "mean": round(float(null.mean()), 4),
            "sd": round(float(null.std()), 4),
            "q95": round(float(np.quantile(null, 0.95)), 4),
            "q99": round(cal.auc_gate, 4),
            "max": round(float(null.max()), 4),
            "share_above_0.5": round(float(np.mean(null > 0.5)), 3),
            "share_above_0.55": round(float(np.mean(null > 0.55)), 3),
        },
        "rule_gates": {
            "conditional_null_z": round(cal.null_gate, 2),
            "marginal_null_z": round(cal.marginal_null_gate, 2),
            "collision_z": round(cal.collision_gate, 2),
            "duplicate_share": round(cal.duplicate_gate, 4),
            # The count floors at ROWS: nulls in a column the reference never has null, and
            # repeated rows. A batch claims only at or past these as well as past the gates.
            "null_floor_k": _binomial_floor(
                ROWS, 1 / (len(cal.reference) + 2), ALPHA / len(cal.spec.feature_columns)
            ),
            "duplicate_floor_k": _duplicate_floor(cal, ROWS),
        },
        "cases": cases,
        "matrix": matrix(cases),
    }
    report["acceptance"] = acceptance(report)
    return report


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", choices=sorted(FEATURE_SPECS), action="append")
    parser.add_argument("--evaluate", action="store_true", help=f"Write {REPORT_PATH.name}.")
    parser.add_argument("--case", help="Run one pre-registered case by name, e.g. population-2.")
    parser.add_argument("--draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--seed", type=int, default=injections.EVAL_SEED, help="Evaluation seed.")
    args = parser.parse_args()
    tracks = args.track or sorted(FEATURE_SPECS)

    if args.case:
        for track in tracks:
            reference, pool = split(load_snapshot(track))
            cal = calibrate(track, reference, pool, draws=args.draws)
            suite = {s.name: s for s in injections.evaluation_suite(track, cal.spec, pool)}
            scenario = suite[args.case]
            result = decompose(cal, injections.apply(scenario, pool, cal.spec, ROWS))
            print(json.dumps(_case(scenario, result), indent=2))
        return

    reports = {track: evaluate(track, draws=args.draws, seed=args.seed) for track in tracks}
    for track, report in reports.items():
        print(
            f"\n{track}: null q99 {report['null']['q99']}, AUC > 0.5 on "
            f"{report['null']['share_above_0.5']:.0%} of reference-vs-reference draws"
        )
        for cause, row in report["matrix"].items():
            print(f"  {cause:10} " + "  ".join(f"{k}={v}" for k, v in row.items()))
        for case in report["cases"]:
            if _group(case["name"]) not in injections.CAUSES:
                print(f"  {case['name']:26} {case['scores']} claimed={case['claimed']}")
        print(f"  acceptance {report['acceptance']}")
    if args.evaluate:
        REPORT_PATH.write_text(json.dumps(reports, indent=1) + "\n")
        print(f"{REPORT_PATH} written")


if __name__ == "__main__":
    main()
