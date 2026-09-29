"""Every decision outcome a track serves must be reachable by some real applicant.

This file exists because a review found a hollow pass. ``tests/test_api.py`` asserts that
``decide`` returns all three outcomes, and it does so by constructing probabilities *relative to
the band* -- ``review_at - 0.05``, the midpoint, ``decline_at + 0.05``. That test is green for
any band whatsoever, including one no model can land outside of.

It was green while credit's served band was ``[0.0014, 0.98]``, a band the credit model cannot
reach either end of: its holdout scores run 0.0027 to 0.7816, so **100.000% of 61,503 applicants
were routed to review** and both ``approve`` and ``decline`` were dead code paths. The service
returned 200s, every test passed, and the three-valued contract had quietly become a constant
function. Nothing in the suite could see it, because seeing it requires a score distribution.

So this check cannot be hermetic, and it is the only test in the suite that is unhermetic on
purpose rather than incidentally. It needs a registered model and an ingested snapshot, and it
skips -- naming what is missing -- when either is absent. ``tests/test_skew.py`` makes the same
trade for the same reason: some properties are only visible against a real model, and the answer
is to gate them rather than to pretend a stub can stand in.

**Why this is worth an unhermetic test rather than a note in STATUS.md.** The band is derived
from a review budget resolved against one model version. Promote a differently-calibrated model
and the same budget resolves to a different price; keep the committed price and the band drifts
away from the distribution it was fitted to. That drift is silent, it is the exact failure this
step already shipped once, and a document cannot fail a build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.api import main
from src.features.pipeline import RANDOM_STATE, split_features_target
from src.features.specs import get_feature_spec
from src.models.costs import COST_MATRICES, REVIEW_BUDGETS
from src.models.thresholds import resolve_review_cost
from src.models.train import TEST_SIZE, processed_path_for

# How far the re-resolved review price may sit from the committed one before the committed value
# is stale. Generous, because the two are resolved from the same holdout and should agree to
# floating-point noise -- the value is there to catch a *changed distribution*, not rounding.
PRICE_TOLERANCE = 1e-3


@pytest.fixture(scope="module", params=sorted(COST_MATRICES))
def holdout_scores(request) -> tuple[str, np.ndarray, np.ndarray]:
    """One track's holdout scores from its promoted model, or a skip naming what is missing.

    The split is reproduced with the same constants the sweep uses, so these are the rows the
    MLflow table already reports on rather than a second, differently drawn sample.
    """
    track = request.param
    snapshot = processed_path_for(track)
    if not snapshot.exists():
        pytest.skip(f"no {track} snapshot at {snapshot}; run `python -m src.data.ingest`")

    from src.models.train import NOT_FOUND_CODES

    uri = main.model_uri_for(track)
    try:
        model, _ = main.load_model(uri)
    except MlflowException as exc:
        # Only absence skips. A corrupt artifact or a registry outage reported as "no model"
        # would turn this guard off exactly when something is wrong. The code set is the repo's
        # single definition rather than a local copy -- see the note on ``NOT_FOUND_CODES``.
        if exc.error_code not in NOT_FOUND_CODES:
            raise
        pytest.skip(f"no {track} model registered at {uri} ({type(exc).__name__}: {exc})")

    from sklearn.model_selection import train_test_split

    spec = get_feature_spec(track)
    features, target = split_features_target(pd.read_parquet(snapshot), spec)
    _, x_test, _, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )
    scores = np.asarray(model.predict(x_test))[:, 1]
    return track, scores, y_test.to_numpy()


def test_all_three_decision_outcomes_are_reachable(holdout_scores):
    """approve, review and decline must each be some real applicant's answer.

    An unreachable outcome is not a latent edge case. It means the field promises a distinction
    the service never makes, and every downstream consumer -- the Prometheus ``decision`` label,
    the prediction log, whatever reads them -- carries a dimension that is constant.
    """
    track, scores, _ = holdout_scores
    review_at, decline_at = main.DECISION_BANDS[track]

    populations = {
        "approve": int((scores < review_at).sum()),
        "review": int(((scores >= review_at) & (scores < decline_at)).sum()),
        "decline": int((scores >= decline_at).sum()),
    }
    unreachable = [name for name, count in populations.items() if count == 0]
    assert not unreachable, (
        f"{track}: {unreachable} unreachable at band ({review_at}, {decline_at}) over "
        f"{len(scores):,} rows scoring {scores.min():.4f}..{scores.max():.4f}. "
        f"Populations: {populations}. The band is outside what this model can produce."
    )


def test_the_review_band_delivers_roughly_its_budget(holdout_scores):
    """The served band must review about what the budget says it will.

    Two ways this fails and both matter. The band could be right and the budget stale, in which
    case the documented capacity is a fiction; or the committed review price could have been
    resolved against a different model, in which case the band is fitted to a distribution that
    is no longer being served. The tolerance is loose because the price is rounded to four
    decimals before serving -- it is checking the same order of magnitude, not the sixth digit.
    """
    track, scores, _ = holdout_scores
    review_at, decline_at = main.DECISION_BANDS[track]
    budget = REVIEW_BUDGETS[track].share

    achieved = float(((scores >= review_at) & (scores < decline_at)).mean())
    assert achieved == pytest.approx(budget, rel=0.15), (
        f"{track}: band ({review_at}, {decline_at}) reviews {achieved:.4%} against a stated "
        f"budget of {budget:.4%}. Re-resolve with "
        f"`python -m src.models.thresholds --track {track} --budget-sweep`."
    )


def test_the_committed_review_price_still_resolves_here(holdout_scores):
    """The measured number must still be the number this model and snapshot produce.

    ``implied_review_cost`` is a measurement, so it has a shelf life: it was resolved against one
    model version and one snapshot, both named in ``resolved_against``. Promote a differently
    calibrated model and the same budget implies a different price. Nothing else in the suite
    notices, because every other test takes the committed value as given.
    """
    track, scores, _ = holdout_scores
    committed = COST_MATRICES[track].review
    resolved = resolve_review_cost(scores, COST_MATRICES[track], REVIEW_BUDGETS[track].share)

    assert resolved == pytest.approx(committed, abs=PRICE_TOLERANCE), (
        f"{track}: committed review price {committed} but this model and snapshot imply "
        f"{resolved:.6f} at a {REVIEW_BUDGETS[track].share:.3%} budget. The committed value was "
        f"resolved against: {REVIEW_BUDGETS[track].resolved_against}"
    )
