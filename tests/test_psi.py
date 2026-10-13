"""PSI: the formula, the bins, the committed reference summaries.

Everything above the last section is hermetic and runs in CI. The last section needs the real
training snapshot the summaries were built from, which a fresh clone and CI do not have -- that is
a data skip, not a model skip, so it stays ``pytest.skip`` (``tests/preconditions.py``).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.make_ci_fixture import synthetic_snapshot
from src.features.specs import FEATURE_SPECS, get_feature_spec
from src.monitoring import drift, psi


@pytest.fixture(scope="module")
def credit_frame() -> pd.DataFrame:
    """A credit-shaped frame with nulls and DAYS_EMPLOYED sentinels, from the CI generator."""
    return synthetic_snapshot("credit", rows=4_000, seed=11)


def _summary(frame: pd.DataFrame, track: str = "credit") -> dict:
    return {
        "track": track,
        "model_version": "7",
        "features": psi.summarise_frame(frame, get_feature_spec(track)),
    }


# --- The formula ---------------------------------------------------------------------------------


def test_psi_matches_the_hand_computed_value():
    """[0.5, 0.5] -> [0.7, 0.3]: 0.2 * ln(1.4) - 0.2 * ln(0.6) = 0.2 * ln(7/3).

    Catches a flipped sign or a swapped ratio, which a self-PSI of 0 cannot.
    """
    assert psi.psi([50, 50], [70, 30]) == pytest.approx(0.16945957207744075, abs=1e-12)


def test_psi_is_symmetric_in_its_arguments():
    assert psi.psi([10, 30, 60], [30, 30, 40]) == pytest.approx(psi.psi([30, 30, 40], [10, 30, 60]))


def test_self_psi_is_exactly_zero_for_every_credit_feature(credit_frame):
    summary = _summary(credit_frame)
    for column, entry in summary["features"].items():
        assert psi.psi(entry["counts"], psi.bin_counts(entry, credit_frame[column])) == 0.0, column


def test_a_one_sigma_shift_exceeds_the_alert_band():
    rng = np.random.default_rng(3)
    entry = psi.numeric_entry(pd.Series(rng.normal(0, 1, 20_000)), None)
    shifted = psi.bin_counts(entry, rng.normal(1, 1, 5_000))
    assert psi.psi(entry["counts"], shifted) > psi.PSI_ALERT


def test_psi_stays_finite_when_either_side_has_an_empty_bin():
    assert math.isfinite(psi.psi([10, 0, 10], [0, 20, 0]))
    assert math.isfinite(psi.psi([0, 20, 0], [10, 0, 10]))


def test_mass_where_the_reference_had_none_crosses_the_alert_on_its_own():
    """The EPS comment's claim: 10% landing in a bin the reference left empty is ~0.69."""
    assert psi.psi([100, 0], [90, 10]) > psi.PSI_ALERT


def test_psi_refuses_mismatched_or_empty_count_vectors():
    with pytest.raises(ValueError, match="differ in length"):
        psi.psi([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="empty"):
        psi.psi([0, 0], [1, 1])


# --- The bins ------------------------------------------------------------------------------------


def test_numeric_bins_agree_with_an_independent_pd_cut():
    """The layout Step 18's reader depends on, recomputed without psi.py's searchsorted.

    The current batch includes every edge exactly, so a right/left-closed slip moves a count.
    """
    rng = np.random.default_rng(5)
    reference = pd.Series(np.round(rng.normal(100, 30, 5_000)))
    reference[:50] = np.nan
    reference[50:120] = 365243.0
    entry = psi.numeric_entry(reference, 365243.0)

    current = pd.Series(
        np.concatenate([rng.normal(110, 30, 2_000), entry["edges"], [np.nan] * 7, [365243.0] * 4])
    )
    body = current[current.notna() & (current != 365243.0)]
    cut = pd.cut(body, bins=[-np.inf, *entry["edges"], np.inf], right=True)
    expected = cut.value_counts(sort=False).tolist() + [7, 4]

    assert psi.bin_counts(entry, current) == expected


def test_sentinel_and_null_get_their_own_bins_rather_than_bending_the_deciles():
    values = pd.Series([float(i) for i in range(1, 101)] + [365243.0] * 30 + [np.nan] * 5)
    entry = psi.numeric_entry(values, 365243.0)
    assert max(entry["edges"]) < 101
    assert entry["counts"][-2:] == [5, 30]
    assert sum(entry["counts"]) == len(values)


def test_rare_and_unseen_categories_share_the_other_bin():
    reference = pd.Series(["A"] * 600 + ["B"] * 395 + ["rare"] * 3 + [None] * 2)
    entry = psi.categorical_entry(reference)
    assert entry["categories"] == ["A", "B"]
    assert entry["counts"] == [600, 395, 3, 2]
    assert psi.bin_counts(entry, ["A", "never-seen", None, "rare"]) == [1, 0, 2, 1]


# --- The summary and the run ---------------------------------------------------------------------


def test_dumps_round_trips_exactly(credit_frame):
    summary = _summary(credit_frame)
    assert json.loads(psi.dumps(summary)) == json.loads(json.dumps(summary))


def test_required_rows_follows_the_widest_entry():
    summary = {"features": {"a": {"counts": [1] * 5}, "b": {"counts": [1] * 21}}, "score": None}
    assert psi.required_rows(summary) == math.ceil(20 / psi.NOISE_BUDGET)


def test_a_batch_below_the_noise_floor_is_refused(credit_frame):
    summary = _summary(credit_frame)
    with pytest.raises(drift.InsufficientCurrentData, match="PSI needs"):
        psi.psi_by_feature(summary, credit_frame.head(10))


def test_score_psi_uses_only_rows_from_the_reference_model_version(credit_frame):
    rng = np.random.default_rng(9)
    summary = _summary(credit_frame)
    summary["score"] = psi.numeric_entry(pd.Series(rng.uniform(0, 1, 4_000)), None)
    current = credit_frame.copy()
    current[drift.SCORE_COLUMN] = rng.uniform(0, 1, len(current))
    current[drift.MODEL_VERSION_COLUMN] = "7"
    assert psi.SCORE_FEATURE in psi.psi_by_feature(summary, current, min_rows=100)

    current[drift.MODEL_VERSION_COLUMN] = "6"
    assert psi.SCORE_FEATURE not in psi.psi_by_feature(summary, current, min_rows=100)


def test_psi_pushes_under_its_own_job_not_drifts():
    """Same job name would make each module's push erase the other's metrics."""
    assert psi.push_job("credit") != drift.push_job("credit")


def test_logged_current_carries_the_score_only_when_asked(tmp_path):
    log = tmp_path / "predictions.jsonl"
    record = {
        "timestamp": "2026-10-13T00:00:00+00:00",
        "track": "credit",
        "model_version": "6",
        "risk_probability": 0.12,
        "features": {"AMT_CREDIT": 1.0},
    }
    log.write_text(json.dumps(record) + "\n")
    now = pd.Timestamp("2026-10-13T01:00:00Z").to_pydatetime()

    plain = drift.logged_current(path=log, now=now, track="credit")
    assert list(plain.columns) == ["AMT_CREDIT"]
    scored = drift.logged_current(path=log, now=now, track="credit", include_score=True)
    assert scored.loc[0, drift.SCORE_COLUMN] == 0.12
    assert scored.loc[0, drift.MODEL_VERSION_COLUMN] == "6"


# --- The committed reference summaries -----------------------------------------------------------


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_committed_reference_matches_the_feature_contract(track):
    """A column added to the contract without regenerating the reference fails here, in CI."""
    summary = psi.load_summary(track)
    spec = get_feature_spec(track)

    assert summary["track"] == track
    assert list(summary["features"]) == list(spec.feature_columns)
    for column, entry in summary["features"].items():
        expected_kind = "numeric" if column in spec.numeric_features else "categorical"
        assert entry["kind"] == expected_kind, column
        assert sum(entry["counts"]) == summary["rows"], column
        if expected_kind == "numeric":
            assert entry["edges"] == sorted(set(entry["edges"])), column
            width = len(entry["edges"]) + 2 + (entry["sentinel"] is not None)
            assert entry["sentinel"] == spec.sentinels.get(column), column
        else:
            width = len(entry["categories"]) + 2
        assert len(entry["counts"]) == width, column
    assert sum(summary["score"]["counts"]) == summary["holdout_rows"]


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_committed_reference_regenerates_byte_for_byte(track):
    """The summary is reproducible from its snapshot and the served model, and still current.

    Needs the real snapshot; CI and a fresh clone skip. Fails when the exported model is no
    longer the one the summary was built against -- a re-promotion that forgot to rebuild it.
    """
    committed = psi.reference_file(track).read_text()
    summary = json.loads(committed)
    snapshot = psi.snapshot_path(summary)
    version_file = psi.MODEL_DIR / track / "MODEL_VERSION"
    if not snapshot.exists() or not version_file.exists():
        pytest.skip(f"needs {snapshot.name} and an exported {track} model")

    assert version_file.read_text().strip() == summary["model_version"], (
        f"build/model/{track} is not model version {summary['model_version']}; "
        f"regenerate with `python -m src.monitoring.psi --track {track} --write-reference`"
    )
    assert psi.dumps(psi.build_reference(track)) == committed
