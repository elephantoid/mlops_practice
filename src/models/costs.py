"""Decision costs and the closed-form operating points they imply.

Split out of :mod:`src.models.thresholds` for one reason: **this module imports nothing.**
Not numpy, not sklearn, not pandas. Everything here is arithmetic on three floats.

That matters because ``src/api/main.py`` needs the operating points, and the serving image is
fighting a 0.5 GB Artifact Registry budget. The first version of this step kept
``DECISION_BANDS`` as hand-copied literals in ``main.py`` with a test asserting they still
matched, on the stated grounds that importing the optimiser would drag sklearn into the
image. The grounds were wrong: the *optimiser* needs sklearn, the *closed form* does not, and
nobody checked which one the serving path actually wanted. The literals are gone and
``main.py`` imports from here, so the two numbers cannot disagree -- there is only one of
them now.

The arithmetic, once, for reference:

    E[approve] = p * C_FN        E[decline] = (1 - p) * C_FP        E[review] = C_R

    approve | review  boundary:  p_lower = C_R / C_FN
    review  | decline boundary:  p_upper = 1 - C_R / C_FP
    the band exists iff          C_R / C_FN + C_R / C_FP < 1

Why three costs rather than two, why the empirical optimiser converges to this, and why
serving is handed these analytic values rather than boundaries fitted to a model's own score
distribution, are all argued in :mod:`src.models.thresholds`. This module is the half of that
argument with no dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# Decimal places the operating points are rounded to before serving reads them. Rounding at
# all is a readability choice -- 0.0014285714285714288 in a log line or a response body tells
# a reader nothing the fourth decimal does not -- and it is done in one place so no caller
# invents its own convention.
SERVING_PRECISION = 4


@dataclass(frozen=True)
class CostMatrix:
    """Relative cost of each way of being wrong, plus the cost of not deciding.

    Units are arbitrary and only ratios matter -- the boundaries are invariant to scaling all
    three by a positive constant. They are written as fractions of the exposed amount
    (``false_negative=0.70`` reads "a missed default loses 70% of the loan") because that is
    the form the estimate was made in, not because the arithmetic cares.

    ``review`` is flat in the probability, and that is what makes the second boundary exist. A
    reviewer costs the same whether the applicant turns out to have defaulted or not.

    **``review`` is not an estimate.** It is the shadow price of the review budget, measured --
    see :class:`ReviewBudget`. It was a guess for one commit, and that guess collapsed the
    three-valued contract to a single value; the history is in ``docs/debt-ledger.md`` 2-E.
    """

    false_negative: float
    false_positive: float
    review: float

    def __post_init__(self) -> None:
        # Finiteness first, because the sign checks below cannot see a NaN: every comparison
        # against NaN is False, so ``false_negative=nan`` passes ``<= 0`` and lands in
        # :func:`analytic_bands`, which returns ``(nan, nan)``. ``src/api/main.py``'s ``decide``
        # then evaluates ``probability >= nan`` -- False for every probability -- and **approves
        # every request**, with no exception anywhere. Infinity is quieter still: it satisfies
        # positivity and drives ``review / false_negative`` to 0.0, silently widening the review
        # band to everything.
        #
        # ``src/pipelines/retrain.py`` already learned this about the AUC tag ("float() happily
        # accepts nan, inf and -inf, so parsing is not the same as being usable"). The same
        # lesson, one module over: a value that arithmetic accepts is not a value a decision
        # boundary can be built from.
        for name, value in (
            ("false_negative", self.false_negative),
            ("false_positive", self.false_positive),
            ("review", self.review),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite; got {value!r}. A non-finite cost does not raise "
                    "anywhere downstream -- it produces a decision boundary that silently "
                    "sends every request to one outcome."
                )

        if self.false_negative <= 0 or self.false_positive <= 0:
            raise ValueError(
                "false_negative and false_positive must both be positive; got "
                f"{self.false_negative} and {self.false_positive}. A zero error cost makes "
                "the corresponding boundary degenerate rather than merely cheap."
            )
        if self.review < 0:
            raise ValueError(f"review cost cannot be negative; got {self.review}")

    @property
    def ratio(self) -> float:
        """FN:FP, the number the sensitivity sweep varies."""
        return self.false_negative / self.false_positive

    @property
    def band_exists(self) -> bool:
        """Is there a probability range where reviewing beats both deciding either way?

        False when review costs at least as much as the best a forced decision can do at the
        two-action crossing. The three-valued contract then has nothing to express and
        :func:`analytic_bands` collapses both boundaries onto that crossing.
        """
        return self.review / self.false_negative + self.review / self.false_positive < 1.0

    def with_review(self, review: float) -> CostMatrix:
        """The same error costs at a different review price.

        Used to turn a resolved shadow price into a matrix, and by the budget sweep, which
        walks review prices rather than error ratios.
        """
        return CostMatrix(
            false_negative=self.false_negative,
            false_positive=self.false_positive,
            review=review,
        )

    @property
    def forced_choice_threshold(self) -> float:
        """The two-action Bayes threshold, ``C_FP / (C_FP + C_FN)``.

        Where the cut would be with no review option at all, and the point the review band is
        centred on: the value of reviewing a row is ``min(p * C_FN, (1 - p) * C_FP)``, which
        peaks exactly here and falls away on both sides. So a band derived from a review budget
        always straddles this, and a band that does not is not a review band -- it is two
        unrelated cuts.
        """
        return self.false_positive / (self.false_positive + self.false_negative)

    def with_ratio(self, ratio: float) -> CostMatrix:
        """The same matrix at a different FN:FP, holding ``false_positive`` and ``review``.

        Scaling ``false_negative`` rather than both sides has a consequence worth stating
        outright: ``p_upper = 1 - C_R / C_FP`` does not mention ``C_FN``, so **a sensitivity
        sweep over FN:FP moves only the lower edge of the review band.** The decline boundary
        is fixed by the cost of a wrong decline against the cost of a review, and no amount of
        fear about missed positives moves it. Anyone reading a sweep and expecting both
        columns to move is reading a bug that is not there.
        """
        if ratio <= 0:
            raise ValueError(f"cost ratio must be positive; got {ratio}")
        return CostMatrix(
            false_negative=ratio * self.false_positive,
            false_positive=self.false_positive,
            review=self.review,
        )


@dataclass(frozen=True)
class ReviewBudget:
    """How much traffic a human can absorb, and what that scarcity is worth.

    This is the replacement for guessing the cost of a review, and the difference is which
    direction the unmeasured number points.

    Setting ``CostMatrix.review`` directly means asserting a number nobody measured -- and the
    first attempt at that put credit's band at ``[0.0014, 0.98]``, which sent **100.000%** of a
    61,503-row holdout to review: both ``approve`` and ``decline`` were unreachable and the
    three-valued contract was a constant function. The cause was structural rather than a bad
    guess. A review costing 0.1% of a loan against a missed default costing 70% means almost no
    probability is confident enough to beat asking a person, so cost minimisation correctly
    answers "ask a person about everyone".

    Bounded abstention inverts it. State the capacity -- which is a fact about a team, not an
    estimate -- and let the optimiser report the price that capacity implies. Reviewing is free
    but rationed, so the rows worth reviewing are the ones where deciding is most likely wrong,
    ranked by ``min(p * C_FN, (1 - p) * C_FP)``. That function peaks at
    :attr:`CostMatrix.forced_choice_threshold` and decreases away from it, so taking the top
    ``share`` selects an *interval* straddling that point, whose edges are
    ``lambda / C_FN`` and ``1 - lambda / C_FP`` for the marginal benefit ``lambda`` at the
    budget boundary.

    That is the same closed form as before with ``lambda`` where the guess used to be, and
    ``lambda`` is a Lagrange multiplier: **the price at which reviewing the marginal row breaks
    even.** It comes out of a measurement, and it is auditable in a way a guess is not -- you
    can read it back and ask whether a review really costs that much. The literature is
    classification with a reject option (Chow 1970) for the closed form, and bounded-abstention
    for the budgeted version; the collapse above is that method's known boundary case, not a
    discovery.

    ``implied_review_cost`` is therefore **measured, not chosen**, and it is only as good as
    ``resolved_against`` says. Regenerate it with::

        uv run python -m src.models.thresholds --track <track> --budget-sweep

    **Read the implied cost before trusting the band.** Credit's says a review is worth 4.52%
    of the exposure, where an underwriting review plausibly costs nearer 0.1%. The gap is the
    point: at a 15% referral budget the shadow price of review capacity sits far above what
    review actually costs, so **capacity is the binding constraint and buying more of it has
    positive expected value.** That is a conclusion about staffing, and it is the kind of thing
    the cost formulation could not express at all.
    """

    share: float
    implied_review_cost: float
    resolved_against: str

    def __post_init__(self) -> None:
        if not 0.0 < self.share < 1.0:
            raise ValueError(
                f"review budget share must be in (0, 1); got {self.share}. A budget of 0 has no "
                "review band and a budget of 1 reviews everything -- neither needs an optimiser."
            )


# Per-track policy. ``share`` is the dial; ``implied_review_cost`` is what it resolved to.
#
# credit 15% -- consumer lending commonly refers 10-20% of applications to manual underwriting.
# fraud 0.5% -- a manual review queue on card transactions is a small fraction of volume; 0.5%
# of 56,962 is already 283 reviews in the holdout window.
REVIEW_BUDGETS: Mapping[str, ReviewBudget] = MappingProxyType(
    {
        "credit": ReviewBudget(
            share=0.15,
            implied_review_cost=0.045183,
            resolved_against=(
                "riskwatch_credit v1 @production, credit_20260928T032717Z.parquet, "
                "61,503 holdout rows, 2026-09-28"
            ),
        ),
        "fraud": ReviewBudget(
            share=0.005,
            implied_review_cost=0.088571,
            resolved_against=(
                "riskwatch_fraud v1 @production, fraud_20260928T051222Z.parquet, "
                "56,962 holdout rows, 2026-09-28"
            ),
        ),
    }
)

# Error costs per track. **These two are still estimates** and are recorded as the DoD (5) cost
# matrix assumption in ``docs/debt-ledger.md``; only the review price is measured.
#
# credit -- a defaulted unsecured consumer loan recovers poorly, so a missed default costs most
# of the principal, and a declined good applicant costs the margin that loan would have earned.
# The 14:1 ratio is in line with published work on this dataset, which uses LGD 0.65 against a
# 0.12 margin for 5.4:1; the difference is a more conservative margin, and the two-action
# threshold each implies (0.0667 against 0.1558) sits well inside the score distribution either
# way. **The error ratio was never the problem** -- see the ledger.
#
# fraud -- a missed fraudulent transaction costs the full amount plus the chargeback fee; a
# blocked legitimate transaction costs its margin plus the customer friction that follows.
#
# They are **unit ratios held constant across rows**, deliberately. Weighting each row by its
# own ``AMT_CREDIT`` or ``Amount`` would make the boundary a function of the transaction size,
# and the serving contract takes two scalars per track.
_ERROR_COSTS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        # track: (false_negative, false_positive)
        "credit": (0.70, 0.05),
        "fraud": (1.00, 0.10),
    }
)

# Assembled rather than written out, so a review price can never be pasted in beside an error
# cost without going through a budget that says where it came from.
COST_MATRICES: Mapping[str, CostMatrix] = MappingProxyType(
    {
        track: CostMatrix(
            false_negative=false_negative,
            false_positive=false_positive,
            review=REVIEW_BUDGETS[track].implied_review_cost,
        )
        for track, (false_negative, false_positive) in _ERROR_COSTS.items()
    }
)


def analytic_bands(costs: CostMatrix) -> tuple[float, float]:
    """The closed-form cost-minimising boundaries, for calibrated probabilities.

    Returns ``(review_at, decline_at)``. When the cost matrix admits no review band both
    values are the two-action crossing ``C_FP / (C_FP + C_FN)`` -- a degenerate band rather
    than an inverted one, because ``src/api/main.py``'s ``decide`` would read an inverted pair
    as a band covering everything and never approve.
    """
    if not costs.band_exists:
        crossing = costs.false_positive / (costs.false_positive + costs.false_negative)
        return crossing, crossing
    return (
        costs.review / costs.false_negative,
        1.0 - costs.review / costs.false_positive,
    )


def decision_bands(track: str) -> tuple[float, float]:
    """``(review_at, decline_at)`` for ``track``, rounded the way serving reports them.

    The single definition of a track's operating points. ``src/api/main.py`` calls this at
    import time and ``src/models/train.py`` calls it to pick the cut ``evaluate`` reports at,
    so the boundary the service decides on and the boundary the MLflow table describes are the
    same number by construction rather than by a test.
    """
    review_at, decline_at = analytic_bands(COST_MATRICES[track])
    return round(review_at, SERVING_PRECISION), round(decline_at, SERVING_PRECISION)
