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
   column a null rate that is higher than the reference *both* marginally and conditionally on the
   other columns; per numeric column a distinct-value share that has fallen. The conditional test
   is what keeps a population shift from reading as upstream: Pensioners have no
   ``OCCUPATION_TYPE`` (99.99% null), so drawing more of them raises that column's null rate while
   leaving ``P(null | the other columns)`` exactly where it was. The marginal test is what keeps a
   feature shift out: moving one column changes what the conditional model predicts for the
   others, but not how many nulls there are.
2. **Domain classifier.** LightGBM, reference sample vs batch, AUC on a held-out half. It is
   compared with the AUC of the same classifier between two *reference* samples, drawn
   ``NULL_DRAWS`` times: drift is detected when the batch's AUC is above that null's ``LEVEL``
   quantile. Without the null the classifier is not a detector -- on pure reference data its AUC
   exceeds 0.5 in about half the draws (``evaluate`` reports the measured share).
3. **Leave one feature out.** Retrain without the most important feature (permutation importance
   on the holdout). ``explained = 1 - (AUC_without - 0.5) / (AUC - 0.5)``: the share of the
   separation that one column carries. A feature shift is, by definition, one column, so its
   explained share is near 1; a population shift moves every column correlated with the segment,
   and the rest still separate the batch. ``s_f = explained``; ``s_p = 1 - explained`` when what is
   left is itself above the null, 0 when it is not.

**No constant here was fitted to an injection.** The thresholds are quantiles of distributions
measured between reference samples, at the one level ``LEVEL``; the attribution margin below is
derived from ``CLAIM``; the classifier's settings are the plan review's and were not varied. The
evaluation cases come from the families committed in ``injections.py`` before this file existed.

**The gate is not an alarm.** The classifier sees joint structure PSI cannot: ``AMT_CREDIT`` x1.02
gave an AUC of 0.78 at a PSI of 0.003, because the ratio to ``AMT_GOODS_PRICE`` broke. It decides
whether there is anything to attribute. Paging on it would page on every batch; PSI owns the alert.

**Identifiability limit.** A segment independent of every other column (``CODE_GENDER`` is close)
moves only its own marginal, which is what a feature shift is. No detector reading this batch can
separate the two; the measured explained share there is 0.43-0.61, reported as mixed.
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

    def auc_gate_at(self, rows: int) -> float:
        """The AUC gate for a batch of ``rows`` after de-duplication.

        The null was drawn at ``self.rows``. An AUC's sampling error scales with
        ``sqrt(1/n_reference + 1/n_current)`` (Hanley & McNeil, 1982), so the gate's excess over
        0.5 is rescaled rather than re-drawn for every batch size.
        """
        n = len(self.reference)
        scale = math.sqrt((1 / n + 1 / rows) / (1 / n + 1 / self.rows))
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
    collide everywhere else (seen on fraud, whose snapshot is 0.25% duplicates).
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


def _null_z(cal_reference: pd.DataFrame, models, spec, frame: pd.DataFrame) -> pd.Series:
    """Per column: (observed nulls - expected nulls) / sd, expected from the conditional model.

    A column with no nulls in the reference expects ``1 / (n + 2)`` per row -- the rule of
    succession -- rather than 0, so one null does not read as an infinite excess.
    """
    floor = 1 / (len(cal_reference) + 2)
    z = {}
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
    return pd.Series(z, dtype="float64")


def _marginal_null_z(reference_rate: pd.Series, frame: pd.DataFrame, n_reference: int) -> pd.Series:
    """Per column: the two-sample binomial z of the batch's null rate against the reference's."""
    floor = 1 / (n_reference + 2)
    p = reference_rate.clip(floor, 1 - floor)
    observed = frame[reference_rate.index].isna().mean()
    return (observed - p) / np.sqrt(p * (1 - p) * (1 / len(frame) + 1 / n_reference))


def _expected_collisions(
    cal: Calibration, batch: pd.DataFrame, seed: int
) -> tuple[pd.Series, pd.Series]:
    """Collisions the batch would show if only its *mix* had moved, with their spread.

    The pool is reweighted towards the batch by a classifier's density ratio ``p / (1 - p)`` and
    resampled at the batch's size. A population shift is covariate shift, so the reweighted pool
    reproduces it, collisions included -- Revolving loans have round credit limits, and drawing
    more of them adds collisions to ``AMT_CREDIT`` that no pipeline caused. A lattice is not
    reproduced: trees split on thresholds and cannot learn that values now sit on multiples of a
    step, so the weights carry at most the lattice's coarse footprint and the fine structure stays
    the pool's.
    """
    spec = cal.spec
    columns = list(batch.columns)
    reference, current = _encode([cal.reference[columns], batch], spec)
    x = pd.concat([reference, current], ignore_index=True)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    model = _classifier(seed).fit(x, y)
    (pool,) = _encode([cal.pool[columns]], spec)
    p = model.predict_proba(pool.astype(x.dtypes.to_dict()))[:, 1]
    weights = p / (1 - p)
    draws = [
        _collisions(
            cal.pool[columns]
            .sample(len(batch), weights=weights, random_state=seed + k)
            .drop_duplicates(),
            spec,
        )
        for k in range(RESAMPLES)
    ]
    frame = pd.DataFrame(draws)
    return frame.mean(), frame.std()


def _duplicate_share(frame: pd.DataFrame) -> float:
    return float(frame.duplicated().mean())


def _rule_gate(values: list[float]) -> float:
    """The null quantile one rule is tested at, Bonferroni-corrected over the ``RULES`` rules."""
    return float(np.quantile(values, 1 - (1 - LEVEL) / RULES))


# --- Domain classifier ---------------------------------------------------------------------------


def domain_auc(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    spec: FeatureSpec,
    seed: int = 0,
    importance: bool = False,
) -> tuple[float, pd.Series | None]:
    """Held-out AUC of reference-vs-current, and each feature's permutation AUC drop on it."""
    x = pd.concat(_encode([reference, current], spec), ignore_index=True)
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    order = np.random.default_rng(seed).permutation(len(x))
    train, test = order[: len(x) // 2], order[len(x) // 2 :]
    model = _classifier(seed).fit(x.iloc[train], y[train])
    held = x.iloc[test].reset_index(drop=True)
    auc = roc_auc_score(y[test], model.predict_proba(held)[:, 1])
    if not importance:
        return float(auc), None

    rng = np.random.default_rng(seed)
    drops = {}
    for column in held.columns:
        shuffled = held.copy()
        # iloc then set_axis keeps the column's dtype; a bare numpy array would turn a category
        # column into object, which LightGBM refuses.
        shuffled[column] = held[column].iloc[rng.permutation(len(held))].set_axis(held.index)
        drops[column] = auc - roc_auc_score(y[test], model.predict_proba(shuffled)[:, 1])
    return float(auc), pd.Series(drops)


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
        null_t.append(_null_z(ref, null_models, spec, draw).max())
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
    )


def decompose(cal: Calibration, current: pd.DataFrame, seed: int = 0) -> Attribution:
    """Score ``current`` against the calibration. See the module docstring for the method."""
    spec = cal.spec
    batch = current[list(spec.feature_columns)].reset_index(drop=True)
    evidence: dict[str, Any] = {"rows": len(batch)}

    # 1. Upstream rules.
    duplicate_share = _duplicate_share(batch)
    conditional = _null_z(cal.reference, cal.null_models, spec, batch)
    marginal = _marginal_null_z(cal.reference_null_rate, batch, len(cal.reference))
    nulled = [
        c
        for c in spec.feature_columns
        if conditional[c] > cal.null_gate and marginal[c] > cal.marginal_null_gate
    ]
    deduped = batch.drop(columns=nulled).drop_duplicates().reset_index(drop=True)
    expected, spread = _expected_collisions(cal, deduped, seed)
    # The spread is the larger of the resamples' and the null's: a handful of resamples
    # under-estimates it, and the gate was calibrated against the null's.
    spread = np.maximum(spread, cal.collision_sd[spread.index])
    observed = _collisions(deduped, spec)
    # Marginal AND conditional, as for nulls, and for the mirror-image reason. Against the plain
    # pool, a population shift adds collisions (Revolving loans' round limits). Against the
    # reweighted pool, a feature shift does: the weights chase the moved column's new range, where
    # the pool's values are spread differently, while the batch kept its own collisions. A lattice
    # is in excess of both.
    marginal_collision = (observed - cal.collision_mean[observed.index]) / cal.collision_sd
    collision = (observed - expected) / spread
    quantized = [
        c
        for c in collision.dropna().index
        if collision[c] > cal.collision_gate and marginal_collision[c] > cal.collision_gate
    ]
    duplicated = duplicate_share > cal.duplicate_gate
    evidence["upstream"] = {
        "duplicate_share": round(duplicate_share, 4),
        "duplicate_gate": round(cal.duplicate_gate, 4),
        "nulled": {c: round(float(conditional[c]), 1) for c in nulled},
        "quantized": {c: round(float(collision[c]), 1) for c in quantized},
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
    if auc > gate:
        # 3. Leave the top feature out. Below this margin the null's noise alone could move
        # ``explained`` across CLAIM: its error is at most (gate - 0.5) / (auc - 0.5).
        attributable = auc - 0.5 >= (gate - 0.5) / CLAIM
        top = str(drops.idxmax())
        rest = [c for c in kept if c != top]
        auc_without, _ = domain_auc(cal.reference[rest], clean[rest], spec, seed=seed)
        explained = float(np.clip(1 - (auc_without - 0.5) / (auc - 0.5), 0, 1))
        evidence.update(
            detected=True,
            attributable=attributable,
            top_feature=top,
            auc_without_top=round(auc_without, 4),
            explained=round(explained, 3),
        )
        if attributable:
            s_f = explained
            s_p = 1 - explained if auc_without > gate else 0.0
    else:
        evidence["detected"] = False

    evidence["psi_top"] = _psi_top(cal.track, batch)
    return Attribution({"feature": s_f, "population": s_p, "upstream": s_u}, evidence)


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
    return name.rsplit("-", 1)[0] if name.startswith("mixed") else name.split("-", 1)[0]


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


def evaluate(track: str, draws: int = NULL_DRAWS) -> dict[str, Any]:
    """Calibrate on the snapshot, then run every pre-registered case through the detector."""
    reference, pool = split(load_snapshot(track))
    cal = calibrate(track, reference, pool, draws=draws)
    cases = [
        _case(scenario, decompose(cal, injections.apply(scenario, pool, cal.spec, ROWS)))
        for scenario in injections.evaluation_suite(track, cal.spec, pool)
    ]
    null = cal.null_auc
    report = {
        "track": track,
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

    reports = {track: evaluate(track, draws=args.draws) for track in tracks}
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
