"""``MIN_CURRENT_ROWS``, checked against the real snapshots it was derived from.

Needs ``data/processed/<track>/latest.parquet``, which a fresh clone and CI do not have -- a data
skip, not a model skip, so it stays ``pytest.skip`` (``tests/preconditions.py``). Locally, a change
to the floor, the retrain share, the watched columns or the Evidently version re-checks the floor
here; CI does not.
"""

from __future__ import annotations

import pytest

from src.features.specs import FEATURE_SPECS
from src.monitoring import drift
from src.pipelines.retrain import should_retrain, watched_columns

# Seeds disjoint from those used to derive the floor (1000+s, 2000+s), so this is not the
# derivation re-run. 9 is not among them: its sample holds the reference's one 117,000,000
# income row, which is a different defect -- see the xfail below.
SEEDS = (7, 8, 10)
OUTLIER_SEED = 9
# A watched column at the floor must sit this far under its drift threshold. Over 100 seeds the
# derivation's largest watched score was 0.074 of a 0.1 threshold, so a regression that brings
# noise near the threshold fails here before it starts retraining.
WATCHED_MARGIN = 0.8


def _reference(track: str):
    path = drift.reference_path(track)
    if not path.exists():
        pytest.skip(f"needs {path}")
    return drift.load_reference(track=track)


def _column_scores(snapshot) -> dict[str, tuple[float, float]]:
    return {
        m["config"]["column"]: (float(m["value"]), float(m["config"]["threshold"]))
        for m in snapshot.dict()["metrics"]
        if m["config"]["type"] == drift.VALUE_DRIFT_METRIC
    }


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
@pytest.mark.parametrize("seed", SEEDS)
def test_an_undrifted_batch_at_the_floor_does_not_retrain(track, seed):
    """Rows drawn from the track's own reference are no shift at all, so they must not retrain.

    At the Telco-derived 100 this failed on every seed of both tracks: Evidently's distance tests
    are not calibrated to the current sample's size.
    """
    reference = _reference(track)
    current = reference.sample(drift.MIN_CURRENT_ROWS, random_state=seed)

    snapshot = drift.compute_drift(reference, current)
    summary = drift.summarise(snapshot)

    assert not should_retrain(summary, track=track), summary
    scores = _column_scores(snapshot)
    for column in watched_columns(track):
        value, threshold = scores[column]
        assert value < WATCHED_MARGIN * threshold, (column, value, threshold)


@pytest.mark.xfail(
    strict=True,
    reason="open: one extreme AMT_INCOME_TOTAL row moves normalised Wasserstein past 0.1 alone",
)
def test_one_extreme_income_does_not_retrain_credit():
    """A single applicant is not drift -- but today one row of 117,000,000 retrains credit.

    The reference's maximum is 117,000,000 against a median of 147,150 and a std of 237,123.
    Normalised Wasserstein divides by that std, so one such row in a 1500-row batch adds about
    0.33 to the score on its own; seed 9 draws it (any batch does with probability ~0.5%), and
    the API's ``gt=0`` sets no upper bound to keep it out of live traffic. More rows do not help
    until about 4,900. Strict, so whoever fixes the detector is told to delete this marker.
    """
    reference = _reference("credit")
    current = reference.sample(drift.MIN_CURRENT_ROWS, random_state=OUTLIER_SEED)
    assert current["AMT_INCOME_TOTAL"].max() == reference["AMT_INCOME_TOTAL"].max()

    summary = drift.summarise(drift.compute_drift(reference, current))

    assert not should_retrain(summary, track="credit"), summary


def test_the_credit_synthetic_batch_still_retrains_at_the_floor():
    """Raising the floor raised the synthetic sample with it; the one demonstrable case survives."""
    reference = _reference("credit")

    summary = drift.summarise(
        drift.compute_drift(reference, drift.synthetic_current(reference, track="credit"))
    )

    assert set(summary["drifted_columns"]) == {drift.SYNTHETIC_DRIFT_COLUMN["credit"]}, summary
    assert should_retrain(summary, track="credit")
