"""Validation contract and cleaning for the fraud track's ULB card-transaction source.

Same two layers as the credit track -- a pandera schema with measured bounds, and a
structural fingerprint against a committed column manifest -- but almost nothing else is
shared, because the data is nothing like it. 284,807 transactions, 31 columns, **492
positives (0.001727)**, zero nulls, zero categoricals.

``V1``..``V28`` are principal components. The ULB researchers ran PCA over the original
transaction fields to publish the set at all, and the loadings were never released, so
**what each component means is not recoverable**. That is why this module and
``FRAUD_FEATURES`` carry no display names for them and no per-column domain bounds: there
is no domain to bound. Inventing "Merchant risk score" for ``V14`` would be fabrication
dressed as interpretability, and DoD (3)'s reason codes are better served by an honest
``V14`` than a confident lie. ``Time``, ``Amount`` and ``Class`` are the three columns whose
meaning survived, and they are the three this module actually reasons about.

Two decisions here are not obvious and are argued where they are made:

* ``TransactionIndex`` -- a derived surrogate key, because the source ships no identifier
  and ``Time`` is not one. See :func:`clean`.
* ``Time`` is validated but **not modeled**. See :data:`~src.features.specs.FRAUD_FEATURES`
  and the note on :data:`TIME_COLUMN` below.

Every number in this module was measured against the real archive
(``data/raw/fraud/creditcardfraud.zip``, ``creditcard.csv``), not assumed: 284,807 rows x 31
columns, ``Class`` positive on 492 rows, ``Time`` spanning 0..172,792 seconds and monotonic
non-decreasing, ``Amount`` 0.0..25,691.16, the widest component excursion
``|V7| = 120.59``, no nulls anywhere, and 1,081 rows that are exact duplicates of another
row.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd
import pandera.pandas as pa

from src.data.fingerprint import assert_column_manifest, read_manifest

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
COLUMN_MANIFEST = SCHEMA_DIR / "ulb_fraud_columns.txt"

# The 28 anonymized principal components, generated rather than enumerated. Writing 28
# near-identical lines by hand is how a duplicated V8 or a missing V19 gets in, and the
# fingerprint would not catch it -- the manifest and the schema would simply agree with each
# other about the wrong set. Derived from the range, "all 28 present" is structural.
V_COLUMNS: Final[tuple[str, ...]] = tuple(f"V{index}" for index in range(1, 29))

TIME_COLUMN: Final = "Time"
AMOUNT_COLUMN: Final = "Amount"
TARGET_COLUMN: Final = "Class"

# Derived in clean(), see the argument there. Named "Index" rather than "Id" on purpose:
# it is a position in this snapshot, not an identifier the upstream issued, and a reader
# who assumes otherwise would try to join on it.
INDEX_COLUMN: Final = "TransactionIndex"

# Measured widest excursion across all 28 components is |V7| = 120.59. The bound is set well
# above it because the components have no domain meaning, so there is no such thing as an
# "implausible" value here -- only an impossible one. What this rejects is a non-finite
# value (inf survives a float dtype check and then poisons the scaler) and a gross scale or
# encoding change, such as raw un-transformed amounts arriving under a V name. A bound
# pinned to the observed maximum would instead reject the tails, and the tails are where the
# fraud signal lives: the positives are 0.17% of rows and sit in exactly those extremes.
V_ABS_BOUND: Final = 500.0

# Measured 0.0..25,691.16. Zero is legitimate -- card verification transactions post at
# 0.00 -- so the bound is ge, not gt. No upper bound: an unusually large transaction is the
# single most interesting row in a fraud dataset and rejecting it at ingest would drop the
# case the model exists to catch.
#
# Contrast with credit, where AMT_CREDIT is gt=0: a zero-credit loan application is a broken
# record, while a zero-amount card transaction is a real event.

# Positive rate, measured. Logged rather than asserted: see clean().
MEASURED_POSITIVE_RATE: Final = 0.001727
MEASURED_ROWS: Final = 284_807

# The discriminator between "seconds elapsed from the first transaction" and "absolute epoch
# seconds". ``ge(0)`` alone does not tell them apart -- 1_700_000_000 is a perfectly
# non-negative float -- so an upstream switching to epoch timestamps would pass validation,
# and because Time is not modeled, ingest would then drop the column and hide the change
# entirely. Nothing downstream would ever notice.
#
# 1e9 is not arbitrary and does not reintroduce the upper bound rejected above: an elapsed
# offset below it still covers **31.7 years** of extract window, so any legitimately longer
# snapshot passes. Every absolute epoch timestamp since September 2001 is above it. The
# bound therefore separates the two encodings without constraining the data.
EPOCH_FLOOR_SECONDS: Final = 1e9


class TransactionOrderError(RuntimeError):
    """The source's rows are no longer in chronological order.

    Separate from a pandera failure and from a fingerprint failure because it is a third kind
    of upstream change: every column is present, every value is in range, and the *ordering*
    the derived key depends on has gone. That is not bad data and not a changed shape, and
    cleaning the batch does not fix it.
    """


FraudSchema: Final = pa.DataFrameSchema(
    {
        # The surrogate key clean() derives. unique=True is the check that matters: it is
        # the assertion that the key actually keys the frame, and it is what would fire if
        # clean() were ever changed to derive it from something non-unique -- Time being
        # the obvious candidate, and the obvious mistake.
        INDEX_COLUMN: pa.Column(int, pa.Check.ge(0), unique=True, nullable=False),
        # Seconds elapsed from the first transaction in the snapshot. Bounded at both ends,
        # and the upper bound is the interesting one: it is NOT the extract's observed
        # ceiling (172,792s, about 48h), which is an artefact of how long this particular
        # extract ran rather than a property of the data. It is EPOCH_FLOOR_SECONDS, which
        # separates an elapsed offset from an absolute epoch timestamp -- see its definition
        # for why ge=0 alone cannot, and why this rejects nothing legitimate.
        TIME_COLUMN: pa.Column(
            float,
            [pa.Check.ge(0), pa.Check.lt(EPOCH_FLOOR_SECONDS)],
            nullable=False,
        ),
        AMOUNT_COLUMN: pa.Column(float, pa.Check.ge(0), nullable=False),
        # Already integer 0/1 in the source -- which is the contract FeatureSpec's
        # positive_label exists for. isin rather than in_range so a 2 or a -1 fails as a
        # category error rather than being silently accepted as "not 1".
        TARGET_COLUMN: pa.Column(int, pa.Check.isin((0, 1)), nullable=False),
        **{
            column: pa.Column(
                float,
                pa.Check.in_range(-V_ABS_BOUND, V_ABS_BOUND),
                nullable=False,
            )
            for column in V_COLUMNS
        },
    },
    # Matches the credit track: unmodeled columns pass through and are dropped downstream by
    # ingest's keep list. The full set is checked structurally by assert_fingerprint.
    strict=False,
    coerce=True,
    name="ULB credit-card transactions",
)
# A DataFrameSchema object rather than a DataFrameModel class, unlike HomeCreditSchema. The
# credit model earns its class: 26 columns with 26 different domain bounds read better as
# annotated fields. Here 28 of 31 columns take the identical check, and a class would mean
# 28 hand-copied lines whose only purpose is to be identical. SchemaSpec.model stays None
# for both tracks anyway -- validation is reached through the module, not the descriptor.


def load_manifest(path: Path = COLUMN_MANIFEST) -> list[str]:
    """Read this track's committed column manifest.

    A thin binding of the shared reader to the fraud manifest. The shared function takes no
    default path on purpose -- a shared default is what made every track fingerprint against
    the credit manifest -- so the default belongs next to the manifest it names.
    """
    return read_manifest(path)


def assert_fingerprint(frame: pd.DataFrame, path: Path = COLUMN_MANIFEST) -> None:
    """Compare the frame's full column set against the fraud manifest.

    Structural, and it runs on the **raw** frame before :func:`clean` -- so the manifest
    holds the 31 source columns and deliberately does *not* hold ``TransactionIndex``. A
    manifest listing the derived key would demand a column the source has never shipped and
    fail every ingest.

    This is also the check that makes the OpenML fallback's declared equivalence enforced
    rather than trusted: if data id 1597 arrives with different names, ingest fails here
    naming them, instead of training a model on a frame nothing downstream expects.
    """
    assert_column_manifest(frame, path)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the surrogate key and record the source's shape. Changes no source value.

    **Why a derived key at all.** Every consumer in this repo needs an id column that is not
    a feature: ``split_features_target`` drops it so a tree cannot memorize row identity,
    ``drift.non_feature_columns`` drops it so a per-row-unique column does not register as
    drifted on every run and inflate the share the retrain trigger reads (measured 0.095
    against 0.048 on the retired Telco frame), and ``ingest`` writes it so a row in the
    parquet can be pointed at. The ULB extract ships no such column.

    **Why not ``Time``.** It is the only candidate and it fails on both counts. It is not
    unique -- 160,215 of 284,807 rows share a second with another row, leaving 124,592
    distinct values -- so it cannot key the frame, and the schema's ``unique=True`` says so
    executably. Worse, a ``FeatureSpec`` forbids the id column from also being a feature, so
    electing ``Time`` would silently remove it from the contract rather than merely label
    it. And 1,081 rows are exact duplicates across all 31 source columns, so no combination
    of source columns keys this frame either.

    **Why a positional index is sound here rather than a fudge.** ``Time`` is monotonic
    non-decreasing over the file (verified against the archive), so row order *is* arrival
    order, and the index is therefore a meaningful ordinal rather than an arbitrary label:
    it preserves the chronology that a later time-ordered split needs after ``Time`` itself
    is dropped from the modeled set. It is honest about its scope -- valid within one
    snapshot, never a join key across snapshots, which is what the ``Index`` in its name is
    for.

    It is emphatically **not** a derived *feature*. ``derived_features`` is empty for this
    track, so the key never reaches ``fit()`` and never appears in the request contract.
    The rule it would otherwise break -- the one that cost the credit track a 27-wide
    signature against a 26-column contract -- is about model inputs; the id column is the
    one non-feature ``ingest`` has always persisted.

    Raises :class:`TransactionOrderError` when the chronology the key's meaning rests on is
    not actually there.
    """
    out = frame.copy()

    # The invariant is CHECKED here, not merely cited. Everything above rests on row order
    # being arrival order, and that was originally established by measuring the archive once
    # -- which says nothing about the next extract. A reordered export passes the manifest
    # (same columns) and the schema (same values), receives a positional key that no longer
    # means what its name and docstring claim, and a later time-ordered split would then use
    # row order while believing it used transaction time. Silent, and wrong in the direction
    # that inflates a fraud model's measured performance.
    #
    # Refuses rather than warns. A warning is the wrong instrument for an assumption a
    # derived column's whole meaning depends on, and the sort order is not something ingest
    # can repair on the caller's behalf: sorting here would change which row gets which key
    # between runs, so two snapshots of the same data would disagree about row 41,234.
    time = out[TIME_COLUMN]
    if not time.is_monotonic_increasing:
        first_break = int((time.diff() < 0).idxmax())
        raise TransactionOrderError(
            f"{TIME_COLUMN} is not monotonic non-decreasing (first decrease at position "
            f"{first_break}), so row order is no longer arrival order and a positional "
            f"{INDEX_COLUMN} would not mean what its name says. The upstream extract has "
            f"been reordered: either restore source order, or stop deriving chronology from "
            f"the index and key on something else."
        )

    # Positional, from a clean 0..n-1 range rather than the incoming index: a frame arriving
    # pre-filtered or concatenated can carry a non-unique or gapped index, and inheriting it
    # would break the key's own uniqueness check.
    out[INDEX_COLUMN] = range(len(out))

    positives = int(out[TARGET_COLUMN].sum())
    duplicated = int(out.duplicated(subset=list(frame.columns)).sum())
    logger.info(
        "%d rows | %d positive (%.6f) | %d exact duplicate row(s) across the %d source "
        "columns -- %s is positional, so those stay individually addressable",
        len(out),
        positives,
        positives / max(len(out), 1),
        duplicated,
        frame.shape[1],
        INDEX_COLUMN,
    )
    # Logged, never asserted. The measured rate is 0.001727 and the acceptance for this step
    # reads it out of the ingest log and the parquet metadata -- but a hard assertion here
    # would reject a legitimately refreshed extract, which is upstream *change*, the thing
    # the fingerprint and the drift decomposer are for. An ingest that refuses new data
    # because the class balance moved is an ingest that cannot be used again.
    if not out[TARGET_COLUMN].between(0, 1).all():
        # Left to validate() to report properly; noted here because the ratio above would
        # otherwise be quietly meaningless.
        logger.warning(
            "%s holds values outside {0, 1}; the rate above is not a rate", TARGET_COLUMN
        )

    return out


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate against :data:`FraudSchema`, reporting every violation at once.

    ``lazy=True`` is the point, and it matters more here than on the credit track. 28 of
    these columns are indistinguishable to a reader, so diagnosing a broken export one
    failure per ingest run would mean 28 round trips to learn that the whole V block
    shifted.
    """
    return FraudSchema.validate(frame, lazy=True)


def downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged, on purpose. Kept as the documented place not to do this.

    The temptation is stronger here than anywhere else in the repo -- 284,807 rows x 30
    float64 columns is the largest frame the project handles, and ``float32`` would halve it
    for free. It is still wrong, for a reason that has nothing to do with size.

    Whatever dtypes reach ``fit()`` become the signature MLflow enforces at serving time.
    The API builds its frame from JSON, where every number arrives 64-bit, so a ``float32``
    parquet makes the logged signature demand the narrow type and **every request fails
    schema enforcement** -- on a dtype nobody chose deliberately. It cost the credit track
    two registered model versions (v3 and v4, still in the registry as the record) before
    ``tests/test_skew.py`` caught it, and it could only catch it once a model existed.

    So this is a no-op in both track modules, and a test pins it in both. A function rather
    than a deleted call site, because ingest's stages read as a pipeline and the reason this
    one does nothing is worth more inline than in a commit message. If the parquet footprint
    ever genuinely matters, narrow at the *storage* boundary and widen on read, rather than
    letting storage dictate the serving contract.
    """
    return frame
