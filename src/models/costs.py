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
    """

    false_negative: float
    false_positive: float
    review: float

    def __post_init__(self) -> None:
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


# Per-track cost assumptions. **These are estimates, not measurements, and they are not
# sourced to published figures** -- recorded as the DoD (5) cost-matrix assumption in
# ``docs/debt-ledger.md`` rather than presented as derived.
#
# credit -- a defaulted unsecured consumer loan recovers poorly, so a missed default costs
# most of the principal; a declined good applicant costs the margin that loan would have
# earned; one underwriting review is cheap against the loan size. The resulting band is very
# wide, and that is a finding rather than a tuning failure: when review is nearly free
# relative to the exposure, almost no probability is confident enough to beat asking a human.
#
# fraud -- a missed fraudulent transaction costs the full amount plus the chargeback fee; a
# blocked legitimate transaction costs its margin plus the customer friction that follows; one
# manual review is *not* cheap against a mean transaction of 88, which is why the fraud band
# is narrower than credit's despite a lower FN:FP ratio.
#
# They are **unit ratios held constant across rows**, deliberately. Weighting each row by its
# own ``AMT_CREDIT`` or ``Amount`` would make the cost-optimal boundary a function of the
# transaction size, and the serving contract takes two scalars per track.
COST_MATRICES: Mapping[str, CostMatrix] = MappingProxyType(
    {
        "credit": CostMatrix(false_negative=0.70, false_positive=0.05, review=0.001),
        "fraud": CostMatrix(false_negative=1.00, false_positive=0.10, review=0.003),
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
