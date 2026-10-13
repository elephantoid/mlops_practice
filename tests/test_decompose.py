"""Drift injections and the cause decomposition.

Everything above the last section is hermetic and runs in CI, on the CI fixture's synthetic
credit frame. Its columns are independent of one another, so it can check the injector, the rules,
the negative case and the null -- but not the feature/population distinction, which lives in the
correlations a synthetic frame does not have. That is the last section's job: the pre-registered
evaluation on the real snapshot, a data skip (``tests/preconditions.py``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.make_ci_fixture import synthetic_snapshot
from src.features.specs import FEATURE_SPECS, get_feature_spec
from src.monitoring import decompose, injections, psi
from src.monitoring.injections import Injection, Scenario

SPEC = get_feature_spec("credit")
ROWS = 1_000


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return synthetic_snapshot("credit", rows=20_000, seed=11)


@pytest.fixture(scope="module")
def calibration(pool) -> decompose.Calibration:
    reference, rest = decompose.split(pool, rows=ROWS)
    return decompose.calibrate("credit", reference, rest, rows=ROWS, draws=40)


def _apply(pool, *injected: Injection, seed: int = 3) -> pd.DataFrame:
    return injections.apply(Scenario("t", injected, seed), pool, SPEC, ROWS)


# --- The injector --------------------------------------------------------------------------------


def test_a_batch_is_the_feature_columns_only(pool):
    assert list(_apply(pool).columns) == list(SPEC.feature_columns)


def test_null_masks_about_the_stated_share(pool):
    before = _apply(pool)["AMT_CREDIT"].isna().mean()
    after = _apply(pool, Injection("null", "AMT_CREDIT", 0.3))["AMT_CREDIT"].isna().mean()
    assert after - before == pytest.approx(0.3, abs=0.05)


def test_duplicate_repeats_exactly_the_stated_share(pool):
    batch = _apply(pool, Injection("duplicate", None, 0.3))
    assert len(batch) == ROWS
    assert batch.duplicated().mean() == pytest.approx(0.3)


def test_no_other_injection_repeats_a_row(pool):
    batch = _apply(pool, Injection("segment", "NAME_INCOME_TYPE", 4.0, "Pensioner"))
    assert not batch.duplicated().any()


def test_unit_divides_the_same_rows_and_leaves_the_sentinel(pool):
    plain = _apply(pool)
    moved = _apply(pool, Injection("unit", "DAYS_EMPLOYED", 365.25))
    sentinel = plain["DAYS_EMPLOYED"] == SPEC.sentinels["DAYS_EMPLOYED"]
    assert sentinel.any()
    assert (moved.loc[sentinel, "DAYS_EMPLOYED"] == SPEC.sentinels["DAYS_EMPLOYED"]).all()
    body = ~sentinel & plain["DAYS_EMPLOYED"].notna()
    np.testing.assert_allclose(
        moved.loc[body, "DAYS_EMPLOYED"], plain.loc[body, "DAYS_EMPLOYED"] / 365.25
    )


def test_quantize_puts_every_value_on_the_lattice(pool):
    batch = _apply(pool, Injection("quantize", "AMT_CREDIT", 1.0))
    sigma = pool["AMT_CREDIT"].std()
    steps = batch["AMT_CREDIT"].dropna() / sigma
    np.testing.assert_allclose(steps, steps.round(), atol=1e-9)


def test_segment_raises_its_share(pool):
    share = (_apply(pool)["NAME_INCOME_TYPE"] == "Pensioner").mean()
    weighted = _apply(pool, Injection("segment", "NAME_INCOME_TYPE", 4.0, "Pensioner"))
    assert (weighted["NAME_INCOME_TYPE"] == "Pensioner").mean() > 2 * share


def test_the_evaluation_suite_is_fixed_by_its_seed(pool):
    first = injections.evaluation_suite("credit", SPEC, pool)
    assert first == injections.evaluation_suite("credit", SPEC, pool)
    assert first != injections.evaluation_suite("credit", SPEC, pool, seed=1)
    singles = [s for s in first if s.name.split("-")[0] in injections.CAUSES]
    assert len(singles) == len(injections.CAUSES) * injections.CASES_PER_CAUSE
    assert all(s.causes == {s.name.split("-")[0]} for s in singles)
    assert sum(not s.injections for s in first) == injections.NEGATIVE_CASES
    assert all(not s.causes for s in first if s.name.startswith("prior"))
    assert set(injections.MIXED["credit"]) <= set(first)


def test_the_revision_is_judged_on_a_seed_it_was_not_developed_on():
    assert injections.EVAL_SEED != injections.FIRST_EVAL_SEED


# --- The detector, on independent synthetic columns ----------------------------------------------


def test_an_uncalibrated_classifier_fires_on_reference_data(calibration):
    """On reference-vs-reference draws, AUC > 0.5 -- the naive reading -- happens."""
    assert (calibration.null_auc > 0.5).any()
    assert calibration.auc_gate > 0.5


def test_a_batch_without_drift_claims_nothing(calibration, pool):
    attribution = decompose.decompose(calibration, _apply(calibration.pool, seed=99))
    assert attribution.claimed == []


@pytest.mark.parametrize(
    "injection",
    [
        Injection("null", "AMT_CREDIT", 0.3),
        Injection("quantize", "AMT_CREDIT", 1.0),
        Injection("duplicate", None, 0.3),
    ],
    ids=lambda i: i.kind,
)
def test_upstream_damage_is_claimed_as_upstream_alone(calibration, injection):
    attribution = decompose.decompose(calibration, _apply(calibration.pool, injection))
    assert attribution.claimed == ["upstream"]


def test_a_flagged_column_is_kept_from_the_classifier(calibration):
    batch = _apply(calibration.pool, Injection("null", "AMT_CREDIT", 0.3))
    evidence = decompose.decompose(calibration, batch).evidence
    assert "AMT_CREDIT" in evidence["upstream"]["nulled"]
    assert evidence["excluded"] == ["AMT_CREDIT"]


def test_a_unit_change_is_a_feature_shift(calibration):
    batch = _apply(calibration.pool, Injection("unit", "AMT_INCOME_TOTAL", 1000.0))
    assert decompose.decompose(calibration, batch).claimed == ["feature"]


def test_a_reweighted_independent_segment_is_population_not_feature(calibration):
    """The tie the first detector lost: all the drift is in one column, and it is a reweighting.

    On this frame every column is independent, so leave-one-feature-out alone would call it a
    feature shift.
    """
    segment = Injection("segment", "NAME_INCOME_TYPE", 4.0, "Pensioner")
    assert decompose.decompose(calibration, _apply(calibration.pool, segment)).claimed == [
        "population"
    ]


def test_the_gate_widens_for_a_smaller_or_reweighted_comparison(calibration):
    full = calibration.auc_gate_at(ROWS)
    assert full == pytest.approx(calibration.auc_gate)
    assert calibration.auc_gate_at(ROWS // 2) > full
    assert calibration.auc_gate_at(ROWS, reference_rows=ROWS / 4) > full


# --- The pre-registered evaluation, on the real snapshot -----------------------------------------


@pytest.fixture(scope="module")
def evaluation() -> dict:
    for track in FEATURE_SPECS:
        snapshot = psi.snapshot_path(psi.load_summary(track))
        if not snapshot.exists():
            pytest.skip(f"needs {snapshot.name}")
    return {track: decompose.evaluate(track) for track in sorted(FEATURE_SPECS)}


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_the_evaluation_meets_every_pre_registered_criterion(evaluation, track):
    failed = [name for name, ok in evaluation[track]["acceptance"].items() if not ok]
    assert not failed, f"{track}: {failed}\n{json.dumps(evaluation[track]['matrix'], indent=1)}"


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_the_committed_report_is_this_code_s_output(evaluation, track):
    """A detector change that forgot to rerun ``--evaluate`` fails here.

    Only the claims and the cause columns are compared. Whether an unclaimed case was "detected,
    not attributed" or "nothing" turns on AUC differences of 0.0004 near the gate, and the report
    was written on macOS while the container is Linux.
    """
    committed = json.loads(decompose.REPORT_PATH.read_text())[track]
    assert committed["seed"] == injections.EVAL_SEED
    for cause in injections.CAUSES:
        for claimed in injections.CAUSES:
            assert (
                committed["matrix"][cause][claimed] == evaluation[track]["matrix"][cause][claimed]
            ), (cause, claimed)
    assert [c["claimed"] for c in committed["cases"]] == [
        c["claimed"] for c in evaluation[track]["cases"]
    ]
