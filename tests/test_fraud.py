"""Tests for the fraud track's ingest contract.

Hermetic: every case builds a small frame in memory, except the two that read the real
archive or the real snapshot and skip when they are absent (``data/`` is gitignored and CI
has no copy).

What these cover is the *failure* behaviour, because that is what ingest exists for. The two
properties specific to this track are the derived surrogate key -- the source ships no
identifier and ``Time`` is not one -- and the fact that ``Time`` is validated while
deliberately not modeled.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from src.data import fraud
from src.data.fingerprint import SchemaFingerprintError
from src.data.tracks import get_track
from src.features.specs import get_feature_spec

SOURCE_COLUMNS = ("Time", *fraud.V_COLUMNS, "Amount", "Class")


def _valid_row(**overrides) -> dict:
    """One row that satisfies every bound, so each test can break exactly one thing."""
    row: dict[str, object] = {"Time": 0.0}
    # Real component values, taken from the first row of the archive rather than invented,
    # so a bound that rejects genuine data fails here rather than in production.
    row.update(dict.fromkeys(fraud.V_COLUMNS, 0.0))
    row["V1"] = -1.359807
    row["V2"] = -0.072781
    row["V14"] = -0.311169
    row["Amount"] = 149.62
    row["Class"] = 0
    return row | overrides


def _frame(rows: int = 1, **overrides) -> pd.DataFrame:
    """``rows`` valid transactions, each a second apart, with ``overrides`` applied to all."""
    return pd.DataFrame(
        [_valid_row(Time=float(index), **overrides) for index in range(rows)],
        columns=list(SOURCE_COLUMNS),
    )


# --- clean(): derive the key, change nothing else -----------------------------------------


def test_clean_keys_rows_that_are_otherwise_byte_identical():
    """1,081 rows in the real archive are exact duplicates of another row.

    Without a positional key those rows are indistinguishable: nothing can point at one of
    them in a report, and no consumer can tell them apart. This is the case the key exists
    for, so it is the case that proves it.
    """
    duplicated = pd.concat([_frame(), _frame(), _frame()], ignore_index=True)
    assert duplicated.duplicated().sum() == 2, "the fixture must actually contain duplicates"

    cleaned = fraud.clean(duplicated)

    assert cleaned[fraud.INDEX_COLUMN].tolist() == [0, 1, 2]
    # And the key survives its own uniqueness check, which is the executable form of
    # "TransactionIndex actually keys this frame".
    fraud.validate(cleaned)


def test_clean_ignores_a_gapped_or_repeated_incoming_index():
    """Derived from position, not from ``frame.index``.

    A frame arriving pre-filtered or concatenated carries a gapped or repeated pandas index.
    Inheriting it -- the one-character version of this line -- yields a key that is not
    unique, and the schema then rejects the whole batch for a reason that has nothing to do
    with the data.
    """
    frame = _frame(rows=3)
    frame.index = pd.Index([5, 5, 7])

    cleaned = fraud.clean(frame)

    assert cleaned[fraud.INDEX_COLUMN].tolist() == [0, 1, 2]
    fraud.validate(cleaned)


def test_clean_adds_only_the_key_and_alters_no_source_value():
    """clean() records the source's shape; it does not edit the source.

    The credit track's equivalent rule cost a training run: a column added by cleaning that
    then reached ``fit()`` produced a 27-wide signature against a 26-column request
    contract. Here nothing is derived *as a feature* at all, and this is what pins that.
    """
    frame = _frame(rows=4)

    cleaned = fraud.clean(frame)

    assert list(cleaned.columns) == [*SOURCE_COLUMNS, fraud.INDEX_COLUMN]
    pd.testing.assert_frame_equal(cleaned[list(SOURCE_COLUMNS)], frame)
    assert get_feature_spec("fraud").derived_features == (), (
        "the key is an id, not a derived feature; a derived feature would reach fit()"
    )


def test_clean_does_not_mutate_its_argument():
    frame = _frame(rows=2)
    fraud.clean(frame)
    assert fraud.INDEX_COLUMN not in frame.columns


# --- validate(): bad data in a known shape ------------------------------------------------


def test_validate_reports_every_violation_at_once():
    """``lazy=True``, and it matters more here than on the credit track.

    28 of these columns are indistinguishable to a reader, so one failure per ingest run
    would mean 28 round trips to learn that the whole V block shifted.
    """
    frame = fraud.clean(_frame(rows=2))
    frame.loc[0, "Amount"] = -1.0
    frame.loc[0, "Class"] = 2
    frame.loc[1, "V9"] = 9_000.0

    with pytest.raises(pandera.errors.SchemaErrors) as excinfo:
        fraud.validate(frame)

    message = str(excinfo.value)
    for column in ("Amount", "Class", "V9"):
        assert column in message, f"{column} violated a bound but was not reported"


def test_validate_rejects_a_duplicated_key():
    """The uniqueness check is the assertion that the key actually keys the frame.

    It is also what would fire if the derivation were ever switched to ``Time``, which is
    the obvious candidate and the obvious mistake: 160,215 of 284,807 real rows share a
    second with another row.
    """
    frame = fraud.clean(_frame(rows=3))
    frame[fraud.INDEX_COLUMN] = 0

    with pytest.raises(pandera.errors.SchemaErrors, match=fraud.INDEX_COLUMN):
        fraud.validate(frame)


def test_validate_rejects_a_non_finite_component():
    """An infinity survives a float dtype check and then poisons every downstream statistic."""
    frame = fraud.clean(_frame())
    frame.loc[0, "V3"] = math.inf

    with pytest.raises(pandera.errors.SchemaErrors, match="V3"):
        fraud.validate(frame)


def test_validate_rejects_a_null_component():
    """The real archive has zero nulls, so a null is upstream change rather than sparsity."""
    frame = fraud.clean(_frame())
    frame.loc[0, "V17"] = None

    with pytest.raises(pandera.errors.SchemaErrors, match="V17"):
        fraud.validate(frame)


def test_validate_rejects_a_component_arriving_at_a_raw_scale():
    """What V_ABS_BOUND actually buys.

    The components have no domain meaning, so there is no "implausible" value -- only an
    impossible one. A raw un-transformed amount landing under a V name is the realistic
    encoding change, and 25,000 is inside Amount's real range and far outside any
    component's.
    """
    frame = fraud.clean(_frame())
    frame.loc[0, "V5"] = 25_000.0

    with pytest.raises(pandera.errors.SchemaErrors, match="V5"):
        fraud.validate(frame)


def test_validate_accepts_the_measured_extremes():
    """The bounds must reject the impossible without rejecting the tails.

    The positives are 0.17% of rows and they live in exactly these extremes, so a bound
    pinned to the observed maximum would throw away the signal the model exists to find.
    Every value here was measured against the archive.
    """
    frame = fraud.clean(_frame(V7=120.589494, V5=-113.743307, V1=-56.407510, Amount=25_691.16))
    frame.loc[0, "Time"] = 172_792.0

    fraud.validate(frame)


def test_validate_accepts_a_zero_amount():
    """Card verification transactions post at 0.00.

    ``ge`` rather than ``gt``, unlike credit's ``AMT_CREDIT``: a zero-credit loan
    application is a broken record, a zero-amount card transaction is a real event.
    """
    fraud.validate(fraud.clean(_frame(Amount=0.0)))


def test_validate_rejects_a_negative_time():
    """``ge=0`` encodes the "elapsed from the first transaction" convention.

    An upstream switching to absolute epoch seconds would pass a bare float check and
    silently rescale the column; a negative offset is the same class of change.
    """
    frame = fraud.clean(_frame())
    frame.loc[0, "Time"] = -1.0

    with pytest.raises(pandera.errors.SchemaErrors, match="Time"):
        fraud.validate(frame)


def test_validate_requires_time_even_though_it_is_not_modeled():
    """Dropped from the model on judgement, still under contract at ingest.

    The distinction is the point: an upstream that stops shipping ``Time`` has changed the
    data, and that must fail loudly even though no model reads the column.
    """
    frame = fraud.clean(_frame()).drop(columns=["Time"])

    with pytest.raises(pandera.errors.SchemaErrors, match="Time"):
        fraud.validate(frame)


def test_validate_tolerates_unmodeled_extra_columns():
    """``strict=False``: the structural check owns the full column set, not the schema."""
    frame = fraud.clean(_frame())
    frame["SomethingNew"] = 1

    fraud.validate(frame)


# --- downcast(): the no-op that is load-bearing -------------------------------------------


def test_downcast_does_not_narrow_dtypes():
    """The single largest frame in the repo, and narrowing it is still wrong.

    284,807 x 30 float64 is the one place ``float32`` looks free. Whatever dtypes reach
    ``fit()`` become the signature MLflow enforces at serving time, and the API builds its
    frame from JSON where every number arrives 64-bit -- so a narrowed parquet makes the
    signature demand a type no request can satisfy. It cost the credit track two registered
    versions before ``tests/test_skew.py`` could catch it.
    """
    frame = fraud.clean(_frame(rows=3))
    before = frame.copy()

    result = fraud.downcast(frame)

    pd.testing.assert_frame_equal(result, before)
    for column in result.select_dtypes(include=["number"]).columns:
        assert result[column].dtype.itemsize == 8, (
            f"{column} is {result[column].dtype}; a narrowed dtype becomes a signature the "
            f"JSON API cannot satisfy"
        )


# --- Fingerprint: structural change the schema cannot see ---------------------------------


def test_fingerprint_accepts_the_real_source_shape():
    """The committed manifest must accept the frame the source actually ships."""
    fraud.assert_fingerprint(pd.DataFrame(columns=list(SOURCE_COLUMNS)))


def test_fingerprint_defaults_to_the_fraud_manifest_not_the_credit_one():
    """Each track's module binds its own manifest as the default.

    A shared default is the defect ``ingest``'s comment warns about: every track
    fingerprinting against the credit column list and passing or failing for reasons having
    nothing to do with its own data. A fraud frame checked against 122 Home Credit names
    fails on all 153 of them.
    """
    credit_spec = get_feature_spec("credit")

    with pytest.raises(SchemaFingerprintError) as excinfo:
        fraud.assert_fingerprint(pd.DataFrame(columns=[credit_spec.id_column]))

    assert "ulb_fraud_columns.txt" in str(excinfo.value)


def test_fingerprint_names_a_renamed_component():
    """A rename is not a row-level problem and the fix is not to clean the batch."""
    renamed = [c if c != "V14" else "V14_NEW" for c in SOURCE_COLUMNS]

    with pytest.raises(SchemaFingerprintError) as excinfo:
        fraud.assert_fingerprint(pd.DataFrame(columns=renamed))

    message = str(excinfo.value)
    assert "V14" in message
    assert "V14_NEW" in message
    assert "likely a rename" in message


def test_fingerprint_catches_a_dropped_component():
    """28 identical-looking columns is exactly where a silent drop hides."""
    with pytest.raises(SchemaFingerprintError, match="V28"):
        fraud.assert_fingerprint(pd.DataFrame(columns=[c for c in SOURCE_COLUMNS if c != "V28"]))


def test_the_committed_manifest_matches_the_feature_contract():
    """Every modeled column must exist in the manifest, or the two disagree about the data."""
    names = set(fraud.load_manifest())
    spec = get_feature_spec("fraud")

    missing = [c for c in spec.feature_columns if c not in names]
    assert not missing, f"modeled columns absent from the manifest: {missing}"
    assert spec.target_column in names
    assert spec.id_column not in names, "the key is derived, so the raw frame cannot have it"


# --- ingest(): the whole chain, on a synthetic frame --------------------------------------


def test_ingest_writes_the_model_contract_and_nothing_else(tmp_path, monkeypatch):
    """Drives the real ``ingest()`` with only acquisition and the CSV read stubbed.

    Three contracts in one pass, all of which have broken before on the credit track:

    * the descriptor's manifest is used, not a module default -- so a second track is
      fingerprinted against its own column list;
    * ``Time`` does not reach the parquet, because it is not in the feature contract;
    * dtypes stay 64-bit through to disk, because the parquet is what ``fit()`` reads and
      what reaches ``fit()`` becomes the enforced serving signature.
    """
    from src.data import ingest as ingest_module
    from src.data import tracks as tracks_module
    from src.data.kaggle_source import Acquisition

    raw = _frame(rows=5)
    monkeypatch.setattr(tracks_module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        ingest_module,
        "acquire",
        lambda *a, **k: Acquisition(
            path=tmp_path / "raw.csv", source_used="test", from_cache=True, is_fallback=False
        ),
    )
    monkeypatch.setattr(ingest_module, "load_raw", lambda path: raw)

    snapshot = ingest_module.ingest("fraud")

    frame = pd.read_parquet(snapshot)
    spec = get_feature_spec("fraud")

    assert list(frame.columns) == [spec.id_column, spec.target_column, *spec.feature_columns]
    assert "Time" not in frame.columns, "Time is not modeled, so it must not reach the parquet"
    assert frame[spec.id_column].is_unique
    for column in frame.select_dtypes(include=["number"]).columns:
        assert frame[column].dtype.itemsize == 8, f"{column} was narrowed on the way to disk"

    metadata = ingest_module.read_metadata(snapshot)
    assert metadata["track"] == "fraud"
    assert metadata["rows"] == "5"


def test_ingest_fails_when_the_source_loses_a_column(tmp_path, monkeypatch):
    """The structural check runs first, so a changed shape is not reported as bad rows."""
    from src.data import ingest as ingest_module
    from src.data import tracks as tracks_module
    from src.data.kaggle_source import Acquisition

    monkeypatch.setattr(tracks_module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        ingest_module,
        "acquire",
        lambda *a, **k: Acquisition(
            path=tmp_path / "raw.csv", source_used="test", from_cache=True, is_fallback=False
        ),
    )
    monkeypatch.setattr(ingest_module, "load_raw", lambda path: _frame(rows=3).drop(columns=["V2"]))

    with pytest.raises(SchemaFingerprintError, match="V2"):
        ingest_module.ingest("fraud")


# --- The measured archive, when it is present ---------------------------------------------


def test_the_real_snapshot_matches_the_measured_acceptance():
    """The Step 8 acceptance numbers, read off the snapshot rather than asserted in prose.

    284,807 rows and a 0.001727 positive rate were measured against
    ``creditcard.csv``. Skips rather than fails without a snapshot: ``data/`` is gitignored,
    so a fresh clone and CI have neither the archive nor the parquet.
    """
    snapshot = get_track("fraud").processed_path
    if not snapshot.exists():
        pytest.skip("no fraud snapshot; run `uv run python -m src.data.ingest --track fraud`")

    frame = pd.read_parquet(snapshot)
    spec = get_feature_spec("fraud")

    assert len(frame) == fraud.MEASURED_ROWS
    assert frame[spec.target_column].mean() == pytest.approx(fraud.MEASURED_POSITIVE_RATE, abs=1e-4)
    assert frame[spec.id_column].is_unique
    assert frame.isna().sum().sum() == 0, "the source has no nulls; ingest must not invent any"
    assert list(frame.columns) == [spec.id_column, spec.target_column, *spec.feature_columns]


def test_the_cached_archive_holds_exactly_what_the_descriptor_declares():
    """``archive_members`` is what ``is_cached`` checks, so it must name real members.

    A declared member the archive does not hold makes every run a cache miss and every
    extraction fail; the reverse leaves a half-extracted archive reading as a cache hit.
    """
    import zipfile

    archive = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraud" / "creditcardfraud.zip"
    if not archive.is_file():
        pytest.skip("no cached fraud archive; data/ is gitignored")

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())

    for member in get_track("fraud").source.archive_members:
        assert member in names, f"{member!r} is declared but the archive holds {sorted(names)}"
