"""Reason codes against a real fitted pipeline, with no registry.

A small LightGBM pipeline is trained once on a synthetic frame shaped like the credit contract.
Everything here runs the real ``pred_contrib`` path; the API tests stub the model and check the
wiring, and ``tests/test_skew.py`` checks the registered model itself.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_pipeline
from src.features.specs import CREDIT_FEATURES
from src.models import explain

SENTINEL = CREDIT_FEATURES.sentinels["DAYS_EMPLOYED"]
CATEGORIES = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "CODE_GENDER": ["M", "F"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "Commercial associate"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents"],
    "OCCUPATION_TYPE": ["Laborers", "Core staff", "Drivers"],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed"],
    "WEEKDAY_APPR_PROCESS_START": ["MONDAY", "FRIDAY"],
}


def _credit_frame(rows: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Credit-shaped features with a target the trees can actually split on.

    Every fifth applicant carries the employment sentinel, so the sentinel step is on the path
    the contributions are computed through -- the case that silently breaks if the explainer
    transforms with the preprocessor alone.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.uniform(30_000, 300_000, rows),
            "AMT_CREDIT": rng.uniform(50_000, 1_000_000, rows),
            "AMT_ANNUITY": rng.uniform(5_000, 60_000, rows),
            "AMT_GOODS_PRICE": rng.uniform(40_000, 900_000, rows),
            "DAYS_BIRTH": rng.integers(-25_000, -7_000, rows),
            "DAYS_EMPLOYED": np.where(
                np.arange(rows) % 5 == 0, SENTINEL, rng.integers(-10_000, -30, rows)
            ),
            "DAYS_REGISTRATION": rng.uniform(-15_000, 0, rows),
            "DAYS_ID_PUBLISH": rng.integers(-6_000, 0, rows),
            "CNT_CHILDREN": rng.integers(0, 4, rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, rows).astype(float),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, rows),
            "EXT_SOURCE_1": rng.uniform(0, 1, rows),
            "EXT_SOURCE_2": rng.uniform(0, 1, rows),
            "EXT_SOURCE_3": rng.uniform(0, 1, rows),
            "HOUR_APPR_PROCESS_START": rng.integers(0, 24, rows),
            **{column: rng.choice(values, rows) for column, values in CATEGORIES.items()},
        }
    )[list(CREDIT_FEATURES.feature_columns)]
    risk = 2.5 * (0.5 - frame["EXT_SOURCE_2"]) + (frame["DAYS_EMPLOYED"] == SENTINEL) * 0.8
    target = pd.Series((risk + rng.normal(0, 0.5, rows) > 0).astype(int))
    return frame, target


@pytest.fixture(scope="module")
def fitted():
    features, target = _credit_frame()
    pipeline = build_pipeline("lightgbm", spec=CREDIT_FEATURES, n_estimators=30, verbose=-1)
    pipeline.fit(features, target)
    return pipeline, features


@pytest.fixture(scope="module")
def explainer(fitted):
    pipeline, _ = fitted
    return explain.build_explainer(pipeline, CREDIT_FEATURES)


def _probability(pipeline, row: pd.DataFrame) -> float:
    return float(pipeline.predict_proba(row)[0][1])


def test_contributions_reconstruct_the_model_logit_on_every_row(fitted, explainer):
    """Additivity, including on sentinel rows -- the rows a preprocessor-only transform gets wrong."""
    pipeline, features = fitted
    sentinel_rows = features["DAYS_EMPLOYED"] == SENTINEL
    assert sentinel_rows.any(), "the fixture must carry sentinel rows"

    contributions, base = explain.contributions(explainer, features)
    reconstructed = 1 / (1 + np.exp(-(contributions.sum(axis=1) + base)))

    np.testing.assert_allclose(
        reconstructed, pipeline.predict_proba(features)[:, 1], atol=explain.ADDITIVITY_TOLERANCE
    )


def test_contributions_are_reported_against_the_request_columns(explainer):
    assert explainer.columns == CREDIT_FEATURES.feature_columns


def test_exactly_three_reason_codes_ordered_by_magnitude_with_their_sign(fitted, explainer):
    pipeline, features = fitted
    row = features.iloc[[3]]

    codes = explain.reason_codes(explainer, row, _probability(pipeline, row))
    contributions, _ = explain.contributions(explainer, row)
    expected = sorted(contributions[0], key=abs, reverse=True)[:3]

    assert len(codes) == explain.TOP_N == 3
    assert [code["contribution"] for code in codes] == pytest.approx(expected)


def test_tied_contributions_break_in_contract_order():
    """Equal magnitudes list in the contract's column order, not the sort algorithm's.

    Six tied at the tail, with mixed signs: numpy's default quicksort returns ``[25, 23, 22]``
    here, so a tie order that is merely deterministic would pass a weaker fixture and still
    name reasons in an order nobody chose.
    """
    phi = np.zeros(len(CREDIT_FEATURES.feature_columns))
    phi[20:] = [0.4, -0.4, 0.4, -0.4, 0.4, -0.4]

    assert explain.top_indices(phi, 3) == [20, 21, 22]


def test_reason_codes_carry_display_names_and_unit_converted_values(fitted, explainer):
    pipeline, features = fitted
    row = features.iloc[[1]]

    codes = explain.reason_codes(explainer, row, _probability(pipeline, row))

    for code in codes:
        column = next(
            c
            for c in CREDIT_FEATURES.feature_columns
            if CREDIT_FEATURES.display_name(c) == code["feature"]
        )
        assert code["value"] == CREDIT_FEATURES.display_value(column, row.iloc[0][column])
    assert all(not code["feature"].isupper() for code in codes), "raw column names leaked"


def test_a_probability_the_contributions_do_not_reconstruct_raises(fitted, explainer):
    """The per-request guard: reasons computed on a different frame than the one scored."""
    pipeline, features = fitted
    row = features.iloc[[0]]

    with pytest.raises(explain.ExplanationError, match="do not reconstruct"):
        explain.reason_codes(explainer, row, _probability(pipeline, row) + 1e-3)


def test_integer_columns_sent_as_floats_explain_identically(fitted, explainer):
    """pyfunc casts to the logged signature and the raw pipeline does not; the two must agree."""
    _, features = fitted
    row = features.iloc[[2]]
    as_floats = row.astype({"CNT_CHILDREN": float, "DAYS_BIRTH": float})

    original, _ = explain.contributions(explainer, row)
    cast, _ = explain.contributions(explainer, as_floats)

    np.testing.assert_array_equal(original, cast)


def test_the_same_row_explains_to_identical_reason_codes_on_every_call(fitted, explainer):
    """Exact equality, not ``approx``: a reason that drifts between calls is not a reason."""
    pipeline, features = fitted
    row = features.iloc[[4]]
    probability = _probability(pipeline, row)

    first = explain.reason_codes(explainer, row, probability)

    for _ in range(5):
        assert explain.reason_codes(explainer, row, probability) == first


def test_refitting_on_the_same_data_reproduces_every_contribution(fitted):
    """Row and column subsampling are on, so the seed is what makes the refit identical.

    With LightGBM's defaults nothing is sampled and any seed would pass -- measured, seeds 1 and
    999 give identical scores. The sampling is what lets this fail when the seed varies between
    builds. ``random_state=None`` would still pass: LightGBM falls back to fixed C++ seeds.
    """
    _, features = fitted
    _, target = _credit_frame()
    sampled = {"subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7}

    refits = [
        build_pipeline(
            "lightgbm", spec=CREDIT_FEATURES, n_estimators=30, verbose=-1, **sampled
        ).fit(features, target)
        for _ in range(2)
    ]
    first, second = (
        explain.contributions(explain.build_explainer(p, CREDIT_FEATURES), features) for p in refits
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_the_explainer_refuses_a_logreg_pipeline(fitted):
    """One-hot output names differ from the contract, and logreg has no ``pred_contrib``."""
    _, features = fitted
    _, target = _credit_frame()
    logreg = build_pipeline("logreg", spec=CREDIT_FEATURES).fit(features, target)

    with pytest.raises(explain.ExplanationError, match="cannot explain"):
        explain.build_explainer(logreg, CREDIT_FEATURES)


def test_the_explainer_refuses_a_pipeline_whose_output_names_differ_from_the_contract(fitted):
    """Contributions are mapped to columns by position, so the positions must be the contract's."""
    pipeline, _ = fitted
    reordered = replace(
        CREDIT_FEATURES, numeric_features=tuple(reversed(CREDIT_FEATURES.numeric_features))
    )

    with pytest.raises(explain.ExplanationError, match="feature order"):
        explain.build_explainer(pipeline, reordered)
