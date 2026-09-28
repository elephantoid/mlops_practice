"""Tests for thresholds, boundaries and selection metrics.

These tests enforce the arithmetic and invariants of cost-based thresholding,
ensuring that boundaries match the closed-form math and the correct metric is
used per track.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.features.specs import SELECTION_METRICS, FeatureSpec, get_feature_spec
from src.models import costs as costs_module
from src.models import thresholds
from src.models.thresholds import (
    CostMatrix,
    analytic_bands,
    metrics_at_operating_point,
    optimise_bands,
    pr_auc,
    recall_at_fpr,
    serving_bands,
    sweep_cost_ratio,
    sweep_fpr_grid,
)
from src.models.train import cv_metric_key, cv_std_key


def test_pr_auc_perfect_separation():
    """A perfect ranking must score 1.0.

    If it doesn't, the metric implementation itself is broken or clamped.
    """
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_random_scores_collapse_to_base_rate():
    """PR-AUC baseline is the positive rate, not 0.5.

    This is the core reason PR-AUC is used for highly imbalanced data like fraud.
    ROC-AUC for a random classifier is 0.5, but PR-AUC collapses to the prior.
    """
    rng = np.random.default_rng(42)
    y_true = np.array([1] * 100 + [0] * 9900)  # 1% positive rate
    y_score = rng.uniform(0, 1, size=len(y_true))

    auc = pr_auc(y_true, y_score)
    assert 0.005 < auc < 0.02, "Random PR-AUC should be near the 0.01 base rate"


def test_pr_auc_worsens_when_scores_are_inverted():
    """Inverting the scores must reduce the area.

    A model outputting (1 - score) is actively wrong, and the metric must penalize it.
    """
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    good_auc = pr_auc(y_true, y_score)
    bad_auc = pr_auc(y_true, 1 - y_score)
    assert bad_auc < good_auc


def test_selection_metrics_vocabulary_is_identical():
    """Both modules must agree on the allowed selection metrics.

    specs.py defines the strings, thresholds.py maps them to callables. A mismatch
    would mean a metric could be configured but not scored, or scored but not allowed.
    """
    assert set(thresholds.SELECTION_SCORERS.keys()) == SELECTION_METRICS


def test_recall_at_fpr_is_monotone_non_decreasing():
    """Loosening the constraint cannot reduce the maximum achievable recall.

    The feasible set of thresholds only grows as max_fpr increases.
    """
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.uniform(0, 1, size=1000)

    recalls = [recall_at_fpr(y_true, y_score, fpr) for fpr in np.linspace(0.0, 1.0, 20)]
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i + 1]


def test_recall_at_fpr_boundaries():
    """Extreme inputs must return sensible boundary values or raise.

    max_fpr=1.0 allows all thresholds, meaning a recall of 1.0 must be reachable.
    """
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    assert recall_at_fpr(y_true, y_score, 1.0) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        recall_at_fpr(y_true, y_score, 1.1)

    with pytest.raises(ValueError):
        recall_at_fpr(y_true, y_score, -0.1)

    with pytest.raises(ValueError):
        recall_at_fpr(np.array([1, 1]), np.array([0.5, 0.6]), 0.5)


def _calibrated_data(size: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that are exactly calibrated probabilities, by construction.

    A row scoring ``s`` is positive with probability ``s``, so the empirical positive rate in
    any score neighbourhood converges to the score itself. That is the only regime in which
    :func:`analytic_bands` is the right answer, so it is the regime the agreement has to be
    tested in -- on an uncalibrated ranker the two forms *should* disagree.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=size)
    labels = (rng.uniform(size=scores.size) < scores).astype(int)
    return labels, scores


def _sample_cost(
    labels: np.ndarray, scores: np.ndarray, costs: CostMatrix, review_at: float, decline_at: float
) -> float:
    """Per-row cost of operating this sample at these two boundaries.

    Written out here rather than read off ``BandSolution.expected_cost`` on purpose: the
    point of the caller is to compare the optimiser's choice against a boundary the optimiser
    did not pick, so the cost has to be computable for an arbitrary pair.
    """
    approve = scores < review_at
    decline = scores >= decline_at
    review = ~approve & ~decline
    total = (
        costs.false_negative * int((labels[approve] == 1).sum())
        + costs.review * int(review.sum())
        + costs.false_positive * int((labels[decline] == 0).sum())
    )
    return total / labels.size


@pytest.mark.parametrize(
    "matrix",
    [
        CostMatrix(false_negative=1.0, false_positive=0.1, review=0.05),
        CostMatrix(false_negative=0.8, false_positive=0.2, review=0.01),
        CostMatrix(false_negative=5.0, false_positive=1.0, review=0.1),
    ],
)
def test_optimise_bands_is_never_worse_than_the_closed_form(matrix):
    """The sharp half of the analytic-optimum check, and it is not statistical.

    The optimiser minimises the *sample* objective over every distinct score. The analytic
    boundary falls between two scores, so the sample cost there equals the sample cost at the
    nearest candidate below it -- which the optimiser also considered. The empirical cost can
    therefore never exceed the analytic one, for any sample, any seed, any cost matrix. A
    strict inequality is normal and means only that the sample optimum is not the population
    optimum; a violation means the search is not finding its own minimum, which is the defect
    a location-based tolerance check can mask by passing on luck.
    """
    y_true, y_score = _calibrated_data()

    solution = optimise_bands(y_true, y_score, matrix)
    analytic_lower, analytic_upper = analytic_bands(matrix)

    empirical_cost = _sample_cost(y_true, y_score, matrix, solution.review_at, solution.decline_at)
    analytic_cost = _sample_cost(y_true, y_score, matrix, analytic_lower, analytic_upper)
    assert empirical_cost <= analytic_cost + 1e-12, (
        "the optimiser returned boundaries costing more on its own sample than the closed "
        "form does, so it did not find the minimum of the objective it claims to minimise"
    )
    assert solution.expected_cost == pytest.approx(empirical_cost, rel=1e-9)

    total_share = solution.approve_share + solution.review_share + solution.decline_share
    assert total_share == pytest.approx(1.0)


@pytest.mark.parametrize(
    "matrix",
    [
        CostMatrix(false_negative=1.0, false_positive=0.1, review=0.05),
        CostMatrix(false_negative=0.8, false_positive=0.2, review=0.01),
        CostMatrix(false_negative=5.0, false_positive=1.0, review=0.1),
    ],
)
def test_optimise_bands_lands_near_the_closed_form(matrix):
    """And the boundaries themselves converge -- with a tolerance that was measured.

    ``abs=0.025`` is not a guess. The objective is *flat* at its optimum by construction: the
    marginal cost of moving a boundary is proportional to ``p - p_boundary``, which is zero
    there, so the argmin has no curvature to pin it down and its scatter falls like
    ``n**(-1/3)`` rather than ``n**(-1/2)``. Measured over six seeds on this fixture, the
    decline boundary's RMS deviation is 0.0113 at n=50k, 0.0099 at n=200k, 0.0038 at n=800k
    and 0.0020 at n=3.2M, with a mean deviation of roughly zero -- scatter, not bias.

    A tolerance of 0.01 at n=200k therefore fails on some seeds and passes on others, which
    is the worst outcome available: it reads as a correctness check and behaves as a coin
    toss. The tight, deterministic statement lives in the cost comparison above; this test
    exists to catch a boundary landing in the *wrong region* entirely.
    """
    y_true, y_score = _calibrated_data()

    solution = optimise_bands(y_true, y_score, matrix)
    analytic_lower, analytic_upper = analytic_bands(matrix)

    assert solution.review_at == pytest.approx(analytic_lower, abs=0.025)
    assert solution.decline_at == pytest.approx(analytic_upper, abs=0.025)


def test_optimise_bands_scatter_shrinks_with_sample_size():
    """The deviation above is sampling scatter, and this is what says so.

    Without this, ``abs=0.025`` is an unexplained magic number and a systematic bias in the
    optimiser -- an off-by-one in the candidate array, say -- would hide inside it forever.
    Bias does not shrink with n; scatter does.
    """
    matrix = CostMatrix(false_negative=1.0, false_positive=0.1, review=0.05)
    _, analytic_upper = analytic_bands(matrix)

    def rms_deviation(size: int) -> float:
        deviations = []
        for seed in range(4):
            y_true, y_score = _calibrated_data(size=size, seed=seed)
            deviations.append(optimise_bands(y_true, y_score, matrix).decline_at - analytic_upper)
        return float(np.sqrt(np.mean(np.square(deviations))))

    small = rms_deviation(25_000)
    large = rms_deviation(400_000)
    assert large < small, f"scatter did not shrink: {small:.5f} -> {large:.5f}"


def test_analytic_bands_math():
    """The closed form must exactly match the documented arithmetic.

    p_lower = C_R / C_FN
    p_upper = 1 - C_R / C_FP
    """
    costs = CostMatrix(false_negative=0.8, false_positive=0.2, review=0.04)
    lower, upper = analytic_bands(costs)
    assert lower == pytest.approx(0.05)
    assert upper == pytest.approx(0.8)


def test_band_collapse():
    """When review is too expensive, the band must collapse to a single threshold.

    An inverted band (review_at > decline_at) would cause the decision function
    to review everything and approve nothing. It must collapse to the 2x2 crossing.
    """
    costs = CostMatrix(false_negative=0.8, false_positive=0.2, review=0.5)
    assert not costs.band_exists

    lower, upper = analytic_bands(costs)
    assert lower == upper
    assert lower == pytest.approx(0.2)

    y_true, y_score = _calibrated_data()
    solution = optimise_bands(y_true, y_score, costs)
    assert not solution.has_band
    assert solution.review_at == solution.decline_at


def test_cost_matrix_validation():
    """Invalid cost matrices must be rejected immediately.

    Zero error costs make the corresponding boundary degenerate.
    """
    with pytest.raises(ValueError):
        CostMatrix(false_negative=0, false_positive=0.1, review=0.05)
    with pytest.raises(ValueError):
        CostMatrix(false_negative=1.0, false_positive=-0.1, review=0.05)
    with pytest.raises(ValueError):
        CostMatrix(false_negative=1.0, false_positive=0.1, review=-0.05)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["false_negative", "false_positive", "review"])
def test_cost_matrix_rejects_non_finite_costs(field, bad):
    """Non-finite costs must raise here, because nothing downstream will.

    The sign checks cannot see a NaN -- every comparison against it is False, so
    ``false_negative=nan`` passes ``<= 0``, reaches ``analytic_bands``, and yields
    ``(nan, nan)``. ``src/api/main.py``'s ``decide`` then evaluates ``probability >= nan``,
    which is False for every probability, and **approves every request**. No exception, no log
    line, and a service that has stopped making decisions while still returning 200s.

    Infinity is quieter: it satisfies positivity and drives ``review / false_negative`` to 0.0,
    widening the review band to everything. Parametrized over all three fields because the
    guard has to be a loop over the fields rather than a check on the one that happened to be
    reported.
    """
    costs = {"false_negative": 0.7, "false_positive": 0.05, "review": 0.001} | {field: bad}
    with pytest.raises(ValueError, match="finite"):
        CostMatrix(**costs)


def test_the_reported_cut_is_the_served_cut():
    """``evaluate``'s cut and the API's decline boundary must be the same number.

    ``operating_threshold_for`` used to read the *unrounded* ``analytic_bands`` while the API
    serves ``decision_bands``, which rounds. Both are 0.98 for credit today, so the bug was
    invisible -- and the first cost matrix with more than ``SERVING_PRECISION`` decimals would
    have made the MLflow table report precision and recall at a cut the service does not use,
    showing up as two numbers that disagree for no visible reason.
    """
    from src.api.main import DECISION_BANDS
    from src.models.train import operating_threshold_for

    for track in costs_module.COST_MATRICES:
        assert operating_threshold_for(track) == DECISION_BANDS[track][1]


def test_cost_matrix_ratio():
    """The ratio property must be exactly FN/FP."""
    costs = CostMatrix(false_negative=0.8, false_positive=0.2, review=0.04)
    assert costs.ratio == pytest.approx(4.0)


def test_scale_invariance():
    """Scaling all costs by a positive constant must not change the solution.

    Only the ratios between costs matter to the boundaries.
    """
    y_true, y_score = _calibrated_data()

    costs1 = CostMatrix(false_negative=1.0, false_positive=0.1, review=0.05)
    costs2 = CostMatrix(false_negative=10.0, false_positive=1.0, review=0.5)

    sol1 = optimise_bands(y_true, y_score, costs1)
    sol2 = optimise_bands(y_true, y_score, costs2)

    assert sol1.review_at == sol2.review_at
    assert sol1.decline_at == sol2.decline_at


def test_selection_metrics_per_track():
    """Fraud and credit use different metrics by design.

    Fraud's base rate is so low that ROC-AUC is misleading.
    """
    assert get_feature_spec("fraud").selection_metric == "pr_auc"
    assert get_feature_spec("credit").selection_metric == "roc_auc"

    assert cv_metric_key("fraud") == "cv_pr_auc_mean"
    assert cv_metric_key("credit") == "cv_roc_auc_mean"
    assert cv_std_key("fraud") == "cv_pr_auc_std"
    assert cv_std_key("credit") == "cv_roc_auc_std"


def test_invalid_selection_metric_raises():
    """A typo in the metric name must fail loud, not silently ignore it."""
    with pytest.raises(ValueError):
        FeatureSpec(
            id_column="id",
            target_column="target",
            positive_label=1,
            numeric_features=("x",),
            categorical_features=(),
            selection_metric="정체불명",
        )


def test_selection_score_delegation():
    """The selection_score function must match the explicit scorers.

    It should also demonstrate that the two metrics yield different values on the same data.
    """
    rng = np.random.default_rng(99)
    y_true = np.array([1] * 10 + [0] * 990)
    y_score = rng.uniform(0, 1, size=1000)

    fraud_score = thresholds.selection_score("fraud", y_true, y_score)
    credit_score = thresholds.selection_score("credit", y_true, y_score)

    assert fraud_score == pytest.approx(pr_auc(y_true, y_score))
    assert credit_score == pytest.approx(roc_auc_score(y_true, y_score))
    assert fraud_score != credit_score


def test_sweep_fpr_grid():
    """FPR grid sweep must skip impossible constraints and be monotone.

    A skipped point is better than a zero point because it signals that no threshold
    can satisfy the constraint, whereas zero might look like a measured result.
    """
    rng = np.random.default_rng(100)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.uniform(0, 1, size=1000)

    grid = [0.0, 0.0001, 0.1, 0.5, 1.0]
    points = sweep_fpr_grid(y_true, y_score, grid)

    assert len(points) <= len(grid)

    for i in range(len(points) - 1):
        assert points[i].recall <= points[i + 1].recall
        assert points[i].fpr <= points[i + 1].fpr


def test_sweep_cost_ratio():
    """Sensitivity sweep holds decline boundary constant.

    As C_FN scales, only the lower edge of the review band moves, because the
    decline boundary is fixed by C_R / C_FP.
    """
    y_true, y_score = _calibrated_data()
    costs = CostMatrix(false_negative=1.0, false_positive=0.1, review=0.05)
    ratios = [1.0, 5.0, 20.0, 100.0]

    rows = sweep_cost_ratio(y_true, y_score, costs, ratios)

    with pytest.raises(ValueError):
        sweep_cost_ratio(y_true, y_score, costs, [1.0, 2.0])

    decline_ats = [row.decline_at for row in rows]
    assert len(set(decline_ats)) == 1


def test_metrics_at_operating_point():
    """Values must be computable and correct on a small fixed dataset."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = metrics_at_operating_point(y_true, y_score, threshold=0.5)

    assert metrics["operating_threshold"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["flagged_share"] == 0.5
    assert metrics["f1"] == 1.0


def test_serving_bands_matches_analytic_bands():
    """Serving bands are exactly the analytic bands rounded to SERVING_PRECISION.

    This ensures the constants defined in the serving layer are in sync with the
    cost assumptions without needing to couple the two at runtime.
    """
    for track in ["credit", "fraud"]:
        sb = serving_bands(track)
        ab = analytic_bands(costs_module.COST_MATRICES[track])

        assert sb[0] == pytest.approx(round(ab[0], costs_module.SERVING_PRECISION))
        assert sb[1] == pytest.approx(round(ab[1], costs_module.SERVING_PRECISION))


# --- The seam: serving's bands against the configured cost matrices -----------------------
#
# src/api/main.py imports src.models.costs -- which imports nothing at all, so it costs the
# serving image nothing -- and builds DECISION_BANDS from it. There is no copy to keep in step
# any more, so these tests guard the arrangement rather than a copy: that serving still
# *derives* its boundaries, and that the set of served tracks still matches the configured one.
#
# The copy they were written for existed because this module imports sklearn and main.py must
# not. That is true of the optimiser and false of the closed form, and nobody checked which of
# the two serving actually wanted until a review asked.


def test_serving_derives_its_bands_rather_than_restating_them():
    """``DECISION_BANDS`` must be *computed* from the cost matrices, not written out.

    This looks tautological now, and it is -- deliberately. An earlier draft held the
    boundaries as hand-copied literals in ``src/api/main.py``, and the defect that invites is
    silent and total: change a cost, forget the copy, and the service keeps deciding at the old
    boundary while every report, sweep and ledger entry describes the new one. Nothing fails
    and nothing logs. The literals were removed once it turned out the closed form needs no
    sklearn and so can simply be imported; this test is what fails if somebody reintroduces
    them, which is the only way that defect can come back.
    """
    from src.api import main

    for track in costs_module.COST_MATRICES:
        assert main.DECISION_BANDS[track] == costs_module.decision_bands(track), (
            f"{track}: serving holds {main.DECISION_BANDS[track]} but the configured cost "
            f"matrix implies {costs_module.decision_bands(track)}"
        )


def test_every_served_track_has_a_cost_matrix():
    """A track in ``DECISION_BANDS`` with no cost matrix has an unexplained boundary.

    Asserted in this direction as well as the other because the failure is asymmetric: an
    extra cost matrix is unused configuration, while an extra band is a decision boundary
    with no recorded derivation -- which is exactly the state the W1 placeholders were in.
    Still worth keeping after the literals became a comprehension over ``COST_MATRICES``: the
    comprehension makes the two sets equal by construction, and this is the test that notices
    if anyone adds a track to one side by hand.
    """
    from src.api import main

    assert set(main.DECISION_BANDS) == set(costs_module.COST_MATRICES)


def test_default_cost_ratios_span_an_order_of_magnitude():
    """The plan's sensitivity requirement, asserted against the default rather than a caller.

    ``sweep_cost_ratio`` rejects a narrow explicit grid, but nothing stopped the *default*
    from being narrowed later -- and the default is what every report will actually use.
    """
    ratios = thresholds.DEFAULT_COST_RATIOS
    assert max(ratios) / min(ratios) >= 10, f"span is only {max(ratios) / min(ratios)}x"


# --- Bounded abstention: budget in, review price out --------------------------------------
#
# The step's first attempt priced a review and got a credit band of [0.0014, 0.98], which sent
# 100.000% of a 61,503-row holdout to review -- approve and decline both unreachable, and a
# three-valued contract reduced to a constant. These tests cover the replacement: state the
# capacity, measure the price it implies.


def _uniform_scores(n: int = 100_000, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)


def test_review_benefit_peaks_at_the_forced_choice_threshold():
    """The benefit of reviewing must be largest where the forced decision is least certain.

    Everything about the budgeted form rests on this: if the benefit were not unimodal with its
    peak at ``C_FP / (C_FP + C_FN)``, taking the highest-benefit rows would select a scattered
    set and there would be no *band* to serve -- just a mask no two thresholds can express.
    """
    costs = CostMatrix(false_negative=0.70, false_positive=0.05, review=0.02)
    grid = np.linspace(0.0, 1.0, 20_001)
    benefit = thresholds.review_benefit(grid, costs)

    peak = float(grid[int(np.argmax(benefit))])
    assert peak == pytest.approx(costs.forced_choice_threshold, abs=1e-3)
    # Unimodal: non-decreasing up to the peak, non-increasing after it.
    left, right = benefit[: int(np.argmax(benefit)) + 1], benefit[int(np.argmax(benefit)) :]
    assert np.all(np.diff(left) >= -1e-12)
    assert np.all(np.diff(right) <= 1e-12)


@pytest.mark.parametrize("budget", [0.005, 0.05, 0.15, 0.40])
def test_resolved_price_delivers_the_requested_budget(budget):
    """The whole point: the band reviews what the budget says, not what a guess implies."""
    costs = CostMatrix(false_negative=0.70, false_positive=0.05, review=0.02)
    scores = _uniform_scores()

    low, high = thresholds.bands_for_budget(scores, costs, budget)
    achieved = float(((scores >= low) & (scores < high)).mean())
    assert achieved == pytest.approx(budget, abs=0.002)


def test_the_budgeted_band_straddles_the_forced_choice_threshold():
    """A review band must contain the cut it exists to hedge, or it is two unrelated cuts."""
    costs = CostMatrix(false_negative=0.70, false_positive=0.05, review=0.02)
    low, high = thresholds.bands_for_budget(_uniform_scores(), costs, 0.15)
    assert low < costs.forced_choice_threshold < high


def test_bands_for_budget_is_the_closed_form_at_the_resolved_price():
    """The budgeted and priced forms are one arithmetic read in two directions.

    Asserted rather than assumed, because it is the claim that lets the closed form -- and the
    analytic-optimum test above it -- survive the switch from pricing to budgeting.
    """
    costs = CostMatrix(false_negative=0.70, false_positive=0.05, review=0.02)
    scores = _uniform_scores()

    resolved = thresholds.resolve_review_cost(scores, costs, 0.15)
    assert thresholds.bands_for_budget(scores, costs, 0.15) == analytic_bands(
        costs.with_review(resolved)
    )


def test_a_bigger_budget_buys_a_cheaper_price_and_a_wider_band():
    """Monotone, and in the direction that says the multiplier is a shadow price.

    More capacity means the marginal reviewed row is less valuable, so lambda falls and the band
    widens. A violation would mean the quantile is being read from the wrong tail -- which would
    still produce plausible-looking bands, just ones that shrink as capacity grows.
    """
    costs = CostMatrix(false_negative=0.70, false_positive=0.05, review=0.02)
    scores = _uniform_scores()

    prices, widths = [], []
    for budget in (0.02, 0.05, 0.15, 0.30):
        prices.append(thresholds.resolve_review_cost(scores, costs, budget))
        low, high = thresholds.bands_for_budget(scores, costs, budget)
        widths.append(high - low)

    assert np.all(np.diff(prices) < 0), f"lambda must fall as budget rises: {prices}"
    assert np.all(np.diff(widths) > 0), f"the band must widen as budget rises: {widths}"


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_review_budget_rejects_a_share_outside_the_open_unit_interval(bad):
    """0 has no band and 1 reviews everything; neither needs an optimiser."""
    with pytest.raises(ValueError, match="between 0 and 1|in \\(0, 1\\)"):
        costs_module.ReviewBudget(share=bad, implied_review_cost=0.02, resolved_against="test")


def test_every_committed_review_price_carries_its_provenance():
    """A measured number with no record of what it was measured against is a guess again.

    ``implied_review_cost`` depends on the score distribution it was resolved from, so the
    version and snapshot that produced it are part of the value. This is the test that stops the
    next person pasting in a number.
    """
    for track, budget in costs_module.REVIEW_BUDGETS.items():
        assert budget.resolved_against.strip(), f"{track} has no provenance for its review price"
        assert f"riskwatch_{track}" in budget.resolved_against
        assert costs_module.COST_MATRICES[track].review == budget.implied_review_cost, (
            f"{track}: the cost matrix's review price must come from the budget, not beside it"
        )
