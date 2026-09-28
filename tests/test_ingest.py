"""Tests for the credit track's ingest contract.

Hermetic: every case builds a small frame in memory. The real 307,511-row archive is
exercised by running the CLI, not by the suite -- a test that needs an 847 MB download is a
test that does not run in CI.

What these cover is the *failure* behaviour, because that is what ingest exists for. A
happy path that reads a good file proves little; the value is in what happens to a batch
that upstream changed without telling anyone.
"""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from src.data import credit
from src.data.credit import SchemaFingerprintError
from src.features.specs import get_feature_spec


def _valid_row() -> dict:
    """One row that satisfies every bound, so each test can break exactly one thing."""
    return {
        "SK_ID_CURR": 100001,
        "TARGET": 0,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637,
        "DAYS_REGISTRATION": -3648.0,
        "DAYS_ID_PUBLISH": -2120,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 1.0,
        "REGION_POPULATION_RELATIVE": 0.018801,
        "EXT_SOURCE_1": 0.083037,
        "EXT_SOURCE_2": 0.262949,
        "EXT_SOURCE_3": 0.139376,
        "HOUR_APPR_PROCESS_START": 10,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
    }


def _frame(**overrides) -> pd.DataFrame:
    row = _valid_row() | overrides
    return pd.DataFrame([row])


# --- Fingerprint: structural change the schema cannot see -------------------------------


def test_fingerprint_passes_on_the_real_manifest(tmp_path):
    manifest = tmp_path / "cols.txt"
    manifest.write_text("A\nB\nC\n")
    credit.assert_fingerprint(pd.DataFrame(columns=["A", "B", "C"]), path=manifest)


def test_fingerprint_names_a_renamed_column(tmp_path):
    """The plan's acceptance criterion, verbatim: a rename must name the column.

    "The schema changed" sends the reader diffing 122 names by hand.
    """
    manifest = tmp_path / "cols.txt"
    manifest.write_text("AMT_CREDIT\nTARGET\n")

    with pytest.raises(SchemaFingerprintError) as excinfo:
        credit.assert_fingerprint(pd.DataFrame(columns=["AMT_CREDIT_V2", "TARGET"]), path=manifest)

    message = str(excinfo.value)
    assert "AMT_CREDIT" in message, "the missing name must appear"
    assert "AMT_CREDIT_V2" in message, "the new name must appear too"
    assert "rename" in message.lower(), "a same-count swap should be called out as a rename"


def test_fingerprint_reports_a_dropped_column(tmp_path):
    manifest = tmp_path / "cols.txt"
    manifest.write_text("A\nB\n")

    with pytest.raises(SchemaFingerprintError, match="missing.*'B'"):
        credit.assert_fingerprint(pd.DataFrame(columns=["A"]), path=manifest)


def test_fingerprint_reports_an_added_column(tmp_path):
    manifest = tmp_path / "cols.txt"
    manifest.write_text("A\n")

    with pytest.raises(SchemaFingerprintError, match="unexpected.*'B'"):
        credit.assert_fingerprint(pd.DataFrame(columns=["A", "B"]), path=manifest)


def test_fingerprint_ignores_column_order(tmp_path):
    """Order is not a contract anyone upstream promised, and every consumer selects by name."""
    manifest = tmp_path / "cols.txt"
    manifest.write_text("A\nB\n")
    credit.assert_fingerprint(pd.DataFrame(columns=["B", "A"]), path=manifest)


def test_missing_manifest_raises_rather_than_passing(tmp_path):
    """An absent manifest must not make the check a no-op that still reports success."""
    with pytest.raises(SchemaFingerprintError, match="manifest missing"):
        credit.assert_fingerprint(pd.DataFrame(columns=["A"]), path=tmp_path / "nope.txt")


def test_empty_manifest_raises(tmp_path):
    manifest = tmp_path / "cols.txt"
    manifest.write_text("\n\n")
    with pytest.raises(SchemaFingerprintError, match="is empty"):
        credit.assert_fingerprint(pd.DataFrame(columns=["A"]), path=manifest)


def test_the_committed_manifest_matches_the_feature_contract():
    """Every modeled column must exist in the manifest, or the two disagree about the data."""
    names = set(credit.load_manifest())
    spec = get_feature_spec("credit")

    missing = [c for c in spec.feature_columns if c not in names]
    assert not missing, f"modeled columns absent from the manifest: {missing}"
    assert spec.id_column in names
    assert spec.target_column in names


# --- Validation: bad data in a known shape ----------------------------------------------


def test_validation_accepts_a_good_frame():
    assert len(credit.validate(_frame())) == 1


def test_validation_reports_every_violation_at_once():
    """The plan's acceptance criterion: a corrupted column yields ALL violations.

    Without lazy=True a broken upstream export surfaces one failure per run, so diagnosing
    it becomes a sequence of ingest attempts instead of one report.
    """
    broken = _frame(
        AMT_INCOME_TOTAL=-1.0,  # gt=0
        DAYS_BIRTH=500,  # le=0
        CODE_GENDER="Z",  # isin
        HOUR_APPR_PROCESS_START=99,  # le=23
        REGION_POPULATION_RELATIVE=2.0,  # le=1
    )

    with pytest.raises(pandera.errors.SchemaErrors) as excinfo:
        credit.validate(broken)

    reported = str(excinfo.value)
    for column in (
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "CODE_GENDER",
        "HOUR_APPR_PROCESS_START",
        "REGION_POPULATION_RELATIVE",
    ):
        assert column in reported, f"{column} missing from the lazy report"


def test_validation_permits_the_days_employed_sentinel():
    """365243 is a legitimate value in this column, on 18% of rows.

    An upper bound here would reject exactly the population the sentinel describes.
    """
    assert len(credit.validate(_frame(DAYS_EMPLOYED=credit.DAYS_EMPLOYED_SENTINEL))) == 1


def test_validation_permits_xna_gender():
    """4 rows of 307,511. The serving contract accepts it, so ingest must too.

    Rejecting it here would make ingest and the API disagree about a valid applicant.
    """
    assert len(credit.validate(_frame(CODE_GENDER="XNA"))) == 1


def test_validation_permits_measured_nulls():
    """EXT_SOURCE_1 is missing on 56% of real rows; OCCUPATION_TYPE on 31%."""
    frame = _frame()
    for column in ("EXT_SOURCE_1", "EXT_SOURCE_3", "OCCUPATION_TYPE", "AMT_ANNUITY"):
        frame[column] = None
    assert len(credit.validate(frame)) == 1


def test_validation_rejects_a_duplicate_id():
    doubled = pd.concat([_frame(), _frame()], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaErrors, match="SK_ID_CURR"):
        credit.validate(doubled)


def test_validation_rejects_a_non_binary_target():
    with pytest.raises(pandera.errors.SchemaErrors, match="TARGET"):
        credit.validate(_frame(TARGET=2))


# --- clean(): record the quirks, do not erase them ---------------------------------------


def test_clean_flags_the_sentinel_without_removing_it():
    """The flag is derived here; the value is normalised inside the pipeline.

    Converting to NaN here would leave training seeing NaN while a live request carried
    365243 -- the same applicant scoring differently by path.
    """
    frame = pd.concat(
        [_frame(SK_ID_CURR=1, DAYS_EMPLOYED=credit.DAYS_EMPLOYED_SENTINEL), _frame(SK_ID_CURR=2)],
        ignore_index=True,
    )

    cleaned = credit.clean(frame)

    assert cleaned["DAYS_EMPLOYED_ANOMALY"].tolist() == [1, 0]
    assert cleaned.loc[0, "DAYS_EMPLOYED"] == credit.DAYS_EMPLOYED_SENTINEL, (
        "the sentinel must survive clean() -- the pipeline owns the conversion"
    )


def test_clean_leaves_xna_in_place():
    cleaned = credit.clean(_frame(CODE_GENDER="XNA"))
    assert cleaned.loc[0, "CODE_GENDER"] == "XNA"


def test_clean_does_not_mutate_its_input():
    frame = _frame()
    credit.clean(frame)
    assert "DAYS_EMPLOYED_ANOMALY" not in frame.columns


# --- downcast(): smaller, and provably lossless ------------------------------------------


def test_downcast_does_not_narrow_dtypes():
    """Narrowing here breaks serving, and this pins the decision not to.

    Whatever dtypes reach ``fit()`` become the signature MLflow enforces at serving time.
    The API builds its frame from JSON, where numbers arrive 64-bit, so a column stored as
    float32 or int32 made the logged signature demand the narrow type and every request
    failed schema enforcement. Training passed, serving 500'd.

    Asserting the no-op rather than deleting the test: someone will reasonably think to
    shrink this parquet again, and the failure it causes appears two stages away.
    """
    frame = _frame()
    result = credit.downcast(frame)

    pd.testing.assert_frame_equal(result, frame)
    assert frame["DAYS_BIRTH"].dtype == "int64", "integers must stay 64-bit for the signature"
    assert frame["AMT_CREDIT"].dtype == "float64", "floats must stay 64-bit for the signature"
