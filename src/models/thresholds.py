"""Operating points under cost asymmetry, and the selection metric each track is scored on.

Two jobs, and they meet in one place. The first is choosing **where** to cut a probability
into approve / review / decline. The second is choosing **which** number a model is
selected on at all, which is a per-track decision because ROC-AUC and PR-AUC disagree
about what a good model is at a 0.17% positive rate.

## Why a 2x2 cost matrix cannot answer the first question

With two costs -- a missed positive and a wrongly flagged negative -- expected cost has
exactly one crossing::

    E[approve]  = p * C_FN
    E[decline]  = (1 - p) * C_FP
    crossing    = C_FP / (C_FP + C_FN)

One crossing is one threshold, and the serving contract needs **two** so the middle band
can be routed to a human. The third action is what creates the second boundary, and it has
a cost of its own: a reviewer's time, paid whatever the applicant turns out to be::

    E[review]   = C_R                       (flat in p)

    approve | review  boundary:  p_lower = C_R / C_FN
    review  | decline boundary:  p_upper = 1 - C_R / C_FP
    the band exists iff          C_R / C_FN + C_R / C_FP < 1

That last condition is the interesting one: when review costs more than the best a decision
can do at the 2x2 crossing, the band collapses and the three-valued contract degenerates to
the single threshold above. :func:`analytic_bands` returns the collapsed pair in that case
rather than an inverted band, because an inverted band is not a configuration a caller can
be expected to notice.

## Why the optimiser is O(n log n) and not a 2-D grid

The obvious implementation sweeps both boundaries together, which is quadratic in the number
of candidate thresholds. It is unnecessary. The three regions *partition* the score axis, so
with ``Below(t) = #{s < t}`` and ``AtOrAbove(t) = N - Below(t)``::

    cost(lo, hi) = C_FN * PosBelow(lo)
                 + C_R  * (N - Below(lo) - AtOrAbove(hi))
                 + C_FP * NegAtOrAbove(hi)

                 = [C_FN * PosBelow(lo) - C_R * Below(lo)]      <- depends on lo only
                 + [C_FP * NegAtOrAbove(hi) - C_R * AtOrAbove(hi)]  <- depends on hi only
                 + C_R * N                                      <- constant

The two brackets are minimised **independently**. Walking the second bracket's difference
between adjacent candidates recovers the pointwise Bayes rule -- a score group moves from
review to approve exactly when its empirical positive rate falls below ``C_R / C_FN`` -- so
on calibrated scores the empirical minimum converges to the closed form above. That
agreement is what ``tests/test_thresholds.py`` asserts, and it is the reason both forms
exist here rather than one.

The minimum is taken over *all* candidates rather than by stopping at the first group that
fails the rule. Empirical positive rates are not monotone in the score, so a stopping rule
finds a local minimum and the argmin finds the global one.

## Why serving gets the analytic band, not the empirical one

``src/api/main.py``'s ``DECISION_BANDS`` are the **analytic** values, imported from
:mod:`src.models.costs` -- which exists precisely so that import costs the serving image
nothing. Three reasons they are the analytic ones rather than fitted:

1. ``CLAUDE.md``: thresholds live in serving, not in the artifact, so the decision boundary
   can move without retraining. A band fitted to one model version's score distribution is
   coupled to that artifact in everything but name -- redeploying a retrained model would
   silently invalidate it.
2. The closed form needs no data, so the boundary is computable in a process that has no
   registry, no parquet and no sklearn.
3. It is honest about what is assumed. The analytic band is correct for *calibrated*
   probabilities. LightGBM under ``class_weight="balanced"`` is not calibrated, so the
   empirical optimum will differ -- and that gap is a **calibration** finding to record,
   not a band to quietly refit. Refitting the band would hide the miscalibration inside a
   number nobody would think to question.

## The cost numbers themselves are an assumption, not a measurement

:data:`src.models.costs.COST_MATRICES` carries ratios estimated from consumer-credit and
card-fraud loss structure. **They are estimates, not measured values, and they are not sourced
to published figures** -- recorded as the DoD (5) cost-matrix assumption in
``docs/debt-ledger.md``, because presenting them as derived is the circular reasoning that
ledger entry 2-C exists to refuse: the same person choosing both the threshold and the
distribution it is judged against.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.features.specs import get_feature_spec
from src.models.costs import (
    COST_MATRICES,
    REVIEW_BUDGETS,
    SERVING_PRECISION,
    CostMatrix,
    analytic_bands,
    decision_bands,
)

# Name -> scorer, for the per-track selection metric declared on ``FeatureSpec``. The names
# are the ones that appear in MLflow metric keys (``cv_roc_auc_mean``, ``cv_pr_auc_mean``),
# so this mapping and the key naming in ``src/features/specs.py`` are the same vocabulary
# read from two places.
#
# average_precision_score rather than auc(recall, precision): the latter interpolates
# linearly between operating points, which is optimistic on a PR curve because the curve is
# not piecewise-linear in that space. The same choice is made and argued in
# ``src.models.train.evaluate``; this is the selection-time entry point to it, not a second
# implementation.
SELECTION_SCORERS: Mapping[str, Callable[[ArrayLike, ArrayLike], float]] = MappingProxyType(
    {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
    }
)

# Swept FPR targets, spanning four orders of magnitude. The low end is not decoration: at a
# 0.001727 positive rate, a 1% false-positive rate means ~2,848 false alarms against 492
# true positives, so the operating points a fraud reviewer could actually staff live below
# 0.005. A grid that started at 0.01 would report only unusable points.
DEFAULT_FPR_GRID: tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20)

# FN:FP ratios for the sensitivity sweep -- two orders of magnitude, against the plan's
# requirement of at least one.
DEFAULT_COST_RATIOS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


@dataclass(frozen=True)
class OperatingPoint:
    """One point on the ROC curve, reported in the terms a reviewer staffs against."""

    threshold: float
    fpr: float
    recall: float
    precision: float
    # Fraction of all rows at or above the threshold. What "flagged volume" means in a
    # capacity conversation, and not recoverable from fpr alone without the base rate.
    flagged_share: float


@dataclass(frozen=True)
class BandSolution:
    """The two boundaries the optimiser chose, and what they cost.

    ``expected_cost`` is per row, so it is comparable across the sensitivity sweep's rows
    even though each one scores the same population under a different cost matrix.
    """

    review_at: float
    decline_at: float
    expected_cost: float
    approve_share: float
    review_share: float
    decline_share: float
    recall_at_decline: float
    precision_at_decline: float
    # False when the cost matrix leaves no room for a review band and both boundaries
    # collapsed onto the 2x2 crossing.
    has_band: bool


def selection_scorer(name: str) -> Callable[[ArrayLike, ArrayLike], float]:
    """Resolve a selection-metric name to its scorer.

    Raises ``KeyError`` naming the registered metrics rather than returning ``None``, so a
    typo in a ``FeatureSpec`` fails at the first score rather than silently selecting on
    whatever a ``None`` fell through to.
    """
    try:
        return SELECTION_SCORERS[name]
    except KeyError:
        registered = ", ".join(sorted(SELECTION_SCORERS))
        raise KeyError(f"unknown selection metric {name!r}; registered: {registered}") from None


def selection_score(track: str, y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Score ``y_score`` on whatever metric ``track`` is selected on."""
    return float(selection_scorer(get_feature_spec(track).selection_metric)(y_true, y_score))


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve, as average precision.

    A named alias rather than new arithmetic. It exists so the selection registry above and
    ``tests/test_thresholds.py`` have one symbol to refer to; the reason average precision
    is the right estimator of this area is argued in the module docstring.
    """
    return float(average_precision_score(y_true, y_score))


# Re-exported from src.models.costs so callers that already import this module do not need to
# know the split. ``serving_bands`` is the old name for it, kept because the report below and
# the tests read better with it.
serving_bands = decision_bands


def _as_arrays(
    y_true: ArrayLike, y_score: ArrayLike
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Validate and coerce the label/score pair every function here takes."""
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score must be the same length; got {labels.shape} and {scores.shape}"
        )
    if labels.size == 0:
        raise ValueError("y_true is empty")
    unique = np.unique(labels)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"y_true must be binary 0/1; found {unique.tolist()}")
    return labels.astype(int), scores


def recall_at_fpr(y_true: ArrayLike, y_score: ArrayLike, max_fpr: float) -> float:
    """Highest achievable recall without exceeding ``max_fpr``.

    Monotone non-decreasing in ``max_fpr`` by construction: the feasible set of ROC points
    only grows as the constraint loosens, so the maximum over it cannot fall. That is the
    property ``tests/test_thresholds.py`` asserts, and it is the reason this is expressed as
    a maximum over a feasible set rather than as "the recall at the point where FPR crosses
    the budget" -- the ROC curve is a step function, and the crossing point is not always a
    point the classifier can actually be operated at.

    Returns 0.0 when no threshold meets the budget, which happens only for ``max_fpr`` below
    the smallest non-zero FPR the scores can produce.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0, 1]; got {max_fpr}")
    labels, scores = _as_arrays(y_true, y_score)
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("recall_at_fpr needs both classes present")

    fpr, tpr, _ = roc_curve(labels, scores)
    feasible = fpr <= max_fpr
    return float(tpr[feasible].max()) if feasible.any() else 0.0


def sweep_fpr_grid(
    y_true: ArrayLike,
    y_score: ArrayLike,
    fpr_grid: Sequence[float] = DEFAULT_FPR_GRID,
) -> list[OperatingPoint]:
    """Precision, recall and flagged volume at each swept false-positive-rate budget.

    Reports precision alongside recall because recall alone is not a decision: at a 0.17%
    positive rate an FPR of 1% buys ~2,848 false alarms, and a reviewer told only "recall
    0.85" has no way to see that. Precision is reconstructed from the ROC point and the
    class counts (``TP = tpr * P``, ``FP = fpr * N``) rather than re-thresholding, so the
    numbers in a row are guaranteed to describe the same point.

    Grid entries no threshold can satisfy are skipped rather than reported with zeros -- a
    row of zeros reads as a measured result, and this is an absence.
    """
    labels, scores = _as_arrays(y_true, y_score)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("sweep_fpr_grid needs both classes present")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    points: list[OperatingPoint] = []
    for budget in fpr_grid:
        feasible = np.flatnonzero(fpr <= budget)
        if feasible.size == 0:
            continue
        # The last feasible index is the highest recall within budget: roc_curve returns
        # both arrays non-decreasing, so walking to the end of the feasible prefix maximises
        # tpr without exceeding the constraint.
        index = int(feasible[-1])
        true_positives = tpr[index] * positives
        false_positives = fpr[index] * negatives
        flagged = true_positives + false_positives
        points.append(
            OperatingPoint(
                threshold=float(thresholds[index]),
                fpr=float(fpr[index]),
                recall=float(tpr[index]),
                precision=float(true_positives / flagged) if flagged > 0 else 0.0,
                flagged_share=float(flagged / labels.size),
            )
        )
    return points


def _boundary_candidates(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thresholds worth evaluating, given that the rule is ``s >= t``.

    The distinct scores, plus one value above the maximum. The extra candidate is the
    "nobody is at or above this" end of the range, without which the optimiser cannot
    express "approve everyone" or "decline nobody" -- and those are the correct answers for
    cost matrices near the degenerate edge.
    """
    distinct = np.unique(scores)
    return np.append(distinct, np.nextafter(distinct[-1], np.inf))


def review_benefit(y_score: ArrayLike, costs: CostMatrix) -> NDArray[np.float64]:
    """What reviewing each row is worth, if the reviewer gets it right.

    ``min(p * C_FN, (1 - p) * C_FP)`` -- the expected cost of the decision a two-action policy
    would be forced into, which reviewing avoids. Rows near
    :attr:`CostMatrix.forced_choice_threshold` are the ones where that forced decision is most
    likely to be the wrong one, and the function peaks there and falls away monotonically on
    both sides.

    That unimodality is why a review *budget* yields an interval rather than an arbitrary set:
    taking the highest-benefit rows takes a contiguous window straddling the peak. It is also
    what makes the budgeted solution the same closed form as the priced one -- see
    :func:`resolve_review_cost`.
    """
    scores = np.asarray(y_score, dtype=float).ravel()
    return np.minimum(scores * costs.false_negative, (1.0 - scores) * costs.false_positive)


def resolve_review_cost(y_score: ArrayLike, costs: CostMatrix, budget: float) -> float:
    """The shadow price of a review budget: what one review has to cost to justify ``budget``.

    Reviewing is treated as free but rationed. Rank rows by :func:`review_benefit` and take the
    top ``budget`` fraction; the benefit of the marginal row is the Lagrange multiplier
    ``lambda``, and the selected interval's edges are exactly ``lambda / C_FN`` and
    ``1 - lambda / C_FP``. Substituting ``lambda`` for ``C_R`` in :func:`analytic_bands`
    therefore reproduces the budgeted solution -- the priced and budgeted formulations are the
    same arithmetic read in opposite directions.

    **Which direction matters.** Pricing a review means asserting a number nobody measured.
    Budgeting one means stating a team's capacity, which is a fact, and *reading back* the price
    it implies -- a number you can hold against reality and reject. The first attempt at this
    step priced a credit review at 0.1% of the loan and produced a band that reviewed 100.000%
    of traffic; the budget form says a 15% referral rate implies 4.52%, which is implausibly
    high for a review and therefore says capacity, not cost, is what binds.

    Returns ``lambda``. A caller that wants bands should use :func:`bands_for_budget`.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be a fraction strictly between 0 and 1; got {budget}")
    benefit = review_benefit(y_score, costs)
    if benefit.size == 0:
        raise ValueError("cannot resolve a review price from an empty score array")
    # The (1 - budget) quantile: the benefit level above which exactly ``budget`` of the mass
    # sits. Higher budget -> lower bar -> smaller lambda -> wider band.
    return float(np.quantile(benefit, 1.0 - budget))


def bands_for_budget(
    y_score: ArrayLike,
    costs: CostMatrix,
    budget: float,
) -> tuple[float, float]:
    """``(review_at, decline_at)`` for a review budget rather than a review price.

    Equal to ``analytic_bands(costs.with_review(resolve_review_cost(...)))`` by construction;
    written as its own function because that equality is the claim and
    ``tests/test_thresholds.py`` asserts it rather than assuming it.
    """
    return analytic_bands(costs.with_review(resolve_review_cost(y_score, costs, budget)))


def optimise_bands(
    y_true: ArrayLike,
    y_score: ArrayLike,
    costs: CostMatrix,
) -> BandSolution:
    """Cost-minimising ``(review_at, decline_at)`` over the observed score distribution.

    Separable in the two boundaries -- see the module docstring for the algebra -- so each
    is an independent ``argmin`` over the distinct scores and the whole thing is one sort.

    On calibrated probabilities the result converges to :func:`analytic_bands`, which is
    what makes the closed form testable against an implementation rather than merely
    asserted. On uncalibrated scores it will not, and the size of that gap is a measurement
    of the miscalibration.

    When the independent minima come out inverted -- ``review_at`` above ``decline_at``,
    which the observed distribution can produce even where :attr:`CostMatrix.band_exists`
    holds in the limit -- the band is dropped and both boundaries fall back to the single
    two-action optimum. The alternative is returning a pair that ``decide`` would read as
    "review everything and approve nothing".
    """
    labels, scores = _as_arrays(y_true, y_score)
    total = labels.size
    candidates = _boundary_candidates(scores)

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # For each candidate t: how many rows, and how many positives, sit strictly below it.
    below = np.searchsorted(sorted_scores, candidates, side="left")
    positives_cumulative = np.concatenate(([0], np.cumsum(sorted_labels)))
    positives_below = positives_cumulative[below]
    negatives_below = below - positives_below

    at_or_above = total - below
    positives_total = int(sorted_labels.sum())
    negatives_total = total - positives_total
    negatives_at_or_above = negatives_total - negatives_below

    # The two separable brackets. Each is minimised over all candidates rather than by a
    # stopping rule: empirical positive rates are not monotone in the score, so a stopping
    # rule finds the first local minimum.
    approve_term = costs.false_negative * positives_below - costs.review * below
    decline_term = costs.false_positive * negatives_at_or_above - costs.review * at_or_above

    low_index = int(np.argmin(approve_term))
    high_index = int(np.argmin(decline_term))
    has_band = low_index <= high_index

    if not has_band:
        # Two actions only. Minimise C_FN * PosBelow(t) + C_FP * NegAtOrAbove(t) directly --
        # not approve_term + decline_term, whose review credits would double-count rows that
        # no longer go to review at all.
        forced = (
            costs.false_negative * positives_below + costs.false_positive * negatives_at_or_above
        )
        collapsed = int(np.argmin(forced))
        low_index = high_index = collapsed

    review_at = float(candidates[low_index])
    decline_at = float(candidates[high_index])

    approved = int(below[low_index])
    declined = int(at_or_above[high_index])
    reviewed = total - approved - declined

    if has_band:
        cost = float(approve_term[low_index] + decline_term[high_index] + costs.review * total)
    else:
        cost = float(
            costs.false_negative * positives_below[low_index]
            + costs.false_positive * negatives_at_or_above[high_index]
        )

    declined_positives = positives_total - int(positives_below[high_index])
    return BandSolution(
        review_at=review_at,
        decline_at=decline_at,
        expected_cost=cost / total,
        approve_share=approved / total,
        review_share=reviewed / total,
        decline_share=declined / total,
        recall_at_decline=(declined_positives / positives_total) if positives_total else 0.0,
        precision_at_decline=(declined_positives / declined if declined else 0.0),
        has_band=has_band,
    )


@dataclass(frozen=True)
class CostSweepRow:
    """One FN:FP ratio, and where it put the boundaries."""

    ratio: float
    review_at: float
    decline_at: float
    review_share: float
    recall_at_decline: float
    precision_at_decline: float
    expected_cost: float


def sweep_cost_ratio(
    y_true: ArrayLike,
    y_score: ArrayLike,
    costs: CostMatrix,
    ratios: Sequence[float] = DEFAULT_COST_RATIOS,
) -> list[CostSweepRow]:
    """Re-solve the boundaries across FN:FP, holding ``false_positive`` and ``review``.

    The point of the sweep is not to find a better ratio. It is to show how much of the
    conclusion rests on a number nobody measured: if the operating point barely moves across
    two orders of magnitude, the un-sourced estimate in :data:`COST_MATRICES` is not load
    bearing, and if it moves a lot, that is the honest caveat on DoD (5).

    Expect ``decline_at`` to be constant down the rows. That is arithmetic, not a bug --
    :meth:`CostMatrix.with_ratio` explains which term it drops out of.
    """
    if len(ratios) < 2:
        raise ValueError("a sensitivity sweep needs at least two ratios")
    span = max(ratios) / min(ratios)
    if span < 10:
        raise ValueError(
            f"ratios span {span:.1f}x; the plan requires at least one order of magnitude. "
            "A sweep narrower than the uncertainty in the estimate it probes says nothing."
        )

    rows: list[CostSweepRow] = []
    for ratio in ratios:
        solution = optimise_bands(y_true, y_score, costs.with_ratio(ratio))
        rows.append(
            CostSweepRow(
                ratio=ratio,
                review_at=solution.review_at,
                decline_at=solution.decline_at,
                review_share=solution.review_share,
                recall_at_decline=solution.recall_at_decline,
                precision_at_decline=solution.precision_at_decline,
                expected_cost=solution.expected_cost,
            )
        )
    return rows


def metrics_at_operating_point(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Threshold-dependent metrics at a stated cut, for ``src.models.train.evaluate``.

    Returns the threshold alongside the metrics it was taken at. That is not redundancy: the
    MLflow table previously carried ``precision_at_0.5``, where the cut was in the key, and
    moving the cut to an operating point would have left a key whose name was a lie. Logging
    the number makes the row self-describing at any cut.
    """
    labels, scores = _as_arrays(y_true, y_score)
    flagged = scores >= threshold
    true_positives = int((flagged & (labels == 1)).sum())
    false_positives = int((flagged & (labels == 0)).sum())
    positives = int(labels.sum())

    precision = true_positives / (true_positives + false_positives) if flagged.any() else 0.0
    recall = true_positives / positives if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "operating_threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_share": float(flagged.mean()),
    }


def _report(track: str, data_path: str | None) -> None:
    """Print the operating-point evidence for one track against its promoted model.

    A reporting entry point rather than a notebook, because every number this prints ends up
    quoted in ``STATUS.md`` and ``docs/debt-ledger.md`` -- and a number quoted from a shell
    session nobody can re-run is a number taken on trust. Run it as::

        uv run python -m src.models.thresholds --track fraud

    Imports are local on purpose. ``src.models.train`` imports *this* module, so a top-level
    import here would be a cycle; and mlflow and pandas have no business being loaded when
    something only wants :func:`analytic_bands`.
    """
    import mlflow
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.features.pipeline import RANDOM_STATE, split_features_target
    from src.models.train import (
        TEST_SIZE,
        configure_tracking,
        model_name_for,
        processed_path_for,
    )

    spec = get_feature_spec(track)
    costs = COST_MATRICES[track]
    configure_tracking()

    frame = pd.read_parquet(data_path or processed_path_for(track))
    features, target = split_features_target(frame, spec)
    # The same split as the sweep, so these numbers describe the same holdout the MLflow
    # table reports on rather than a second, differently drawn one.
    _, x_test, _, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )

    model = mlflow.pyfunc.load_model(f"models:/{model_name_for(track)}@production")
    scores = np.asarray(model.predict(x_test))[:, 1]

    review_at, decline_at = analytic_bands(costs)
    print(f"\n=== {track} | {len(y_test):,} held-out rows | positive rate {y_test.mean():.6f} ===")
    print(
        f"selection metric : {spec.selection_metric} = {selection_score(track, y_test, scores):.4f}"
    )
    # Both are always printed, including the one this track is not selected on: the gap
    # between them is the whole argument for the per-track selection metric, and it is only
    # visible side by side.
    print(f"roc_auc          : {roc_auc_score(y_test, scores):.4f}")
    print(f"pr_auc           : {pr_auc(y_test, scores):.4f}")
    print(
        f"cost matrix      : C_FN={costs.false_negative} C_FP={costs.false_positive} "
        f"C_R={costs.review} (FN:FP = {costs.ratio:.0f}:1)"
    )
    print(
        f"analytic band    : [{review_at:.4f}, {decline_at:.4f}] -> serving {serving_bands(track)}"
    )

    print("\n-- what moving off 0.5 did, at the decline boundary --")
    print(f"{'cut':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'flagged':>9}")
    for label, cut in (("0.5", 0.5), (f"{decline_at:.4f}", decline_at)):
        m = metrics_at_operating_point(y_test, scores, cut)
        print(
            f"{label:>8} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['flagged_share']:>9.5f}"
        )

    print("\n-- swept FPR grid --")
    print(f"{'fpr':>8} {'threshold':>10} {'precision':>10} {'recall':>8} {'flagged':>9}")
    for point in sweep_fpr_grid(y_test, scores):
        print(
            f"{point.fpr:>8.5f} {point.threshold:>10.4f} {point.precision:>10.4f} "
            f"{point.recall:>8.4f} {point.flagged_share:>9.5f}"
        )

    print("\n-- cost-ratio sensitivity (decline_at is expected to be constant) --")
    print(
        f"{'FN:FP':>7} {'review_at':>10} {'decline_at':>11} {'review%':>9} {'recall':>8} {'prec':>8}"
    )
    for row in sweep_cost_ratio(y_test, scores, costs):
        print(
            f"{row.ratio:>7.0f} {row.review_at:>10.4f} {row.decline_at:>11.4f} "
            f"{row.review_share:>9.4f} {row.recall_at_decline:>8.4f} "
            f"{row.precision_at_decline:>8.4f}"
        )

    budget = REVIEW_BUDGETS[track]
    print(f"\n-- review budget {budget.share:.3%} -> implied review price --")
    print(
        f"{'budget':>8} {'lambda':>10} {'lam/C_FP':>9} {'review_at':>10} {'decline_at':>11} "
        f"{'approve%':>9} {'review%':>8} {'decline%':>9} {'recall':>7} {'prec':>7}"
    )
    grid = sorted({budget.share, *(budget.share * m for m in (0.2, 0.5, 2.0, 5.0))})
    for share in (b for b in grid if 0.0 < b < 1.0):
        lam = resolve_review_cost(scores, costs, share)
        low, high = analytic_bands(costs.with_review(lam))
        low, high = round(low, SERVING_PRECISION), round(high, SERVING_PRECISION)
        if low >= high:
            print(f"{share:>8.4f} {lam:>10.6f}   band collapses")
            continue
        declined = scores >= high
        caught = int((declined & (y_test.to_numpy() == 1)).sum())
        print(
            f"{share:>8.4f} {lam:>10.6f} {lam / costs.false_positive:>9.4f} {low:>10.4f} "
            f"{high:>11.4f} {(scores < low).mean():>9.4%} "
            f"{((scores >= low) & (scores < high)).mean():>8.4%} {declined.mean():>9.4%} "
            f"{caught / int(y_test.sum()):>7.4f} "
            f"{(caught / int(declined.sum()) if declined.sum() else 0):>7.4f}"
        )
    print(f"committed lambda {costs.review} resolved against: {budget.resolved_against}")
    drift = abs(resolve_review_cost(scores, costs, budget.share) - costs.review)
    print(f"re-resolved here differs from the committed value by {drift:.2e}")

    print("\n-- are all three outcomes reachable at the served band? --")
    for name, mask in (
        ("approve", scores < review_at),
        ("review", (scores >= review_at) & (scores < decline_at)),
        ("decline", scores >= decline_at),
    ):
        count = int(mask.sum())
        print(
            f"  {name:<8} {count:>7,} rows ({mask.mean():>8.4%}){'' if count else '   <- UNREACHABLE'}"
        )

    empirical = optimise_bands(y_test, scores, costs)
    print("\n-- empirical optimum vs the analytic band it is served at --")
    print(f"analytic  [{review_at:.4f}, {decline_at:.4f}]")
    print(
        f"empirical [{empirical.review_at:.4f}, {empirical.decline_at:.4f}]  "
        f"has_band={empirical.has_band}  review_share={empirical.review_share:.4f}"
    )
    print("A large gap is a calibration finding, not a band to refit -- see the module docstring.")


def main() -> None:
    """CLI entry point for the operating-point report."""
    import argparse

    parser = argparse.ArgumentParser(description="Operating-point evidence for one track.")
    parser.add_argument("--track", default="credit", choices=sorted(COST_MATRICES))
    parser.add_argument(
        "--budget-sweep",
        action="store_true",
        help="alias for the default report, which always includes the budget sweep; kept so the "
        "regeneration command named in src/models/costs.py is literally runnable",
    )
    parser.add_argument("--data", default=None, help="processed parquet; defaults to the snapshot")
    args = parser.parse_args()
    _report(args.track, args.data)


if __name__ == "__main__":
    main()
