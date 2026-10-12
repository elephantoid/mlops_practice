"""Validation contract and cleaning for the credit track's Home Credit source.

Two layers, because 122 columns make full enumeration impractical where the retired Telco
schema enumerated all 21:

* :class:`HomeCreditSchema` enumerates only the **modeled** subset with real domain bounds,
  validated lazily so a corrupted batch reports every violation at once rather than the
  first one.
* :func:`assert_fingerprint` compares the frame's full column set against a committed
  manifest. That catches added, dropped and renamed columns -- upstream structural change
  the schema cannot see, because a schema with ``strict=False`` is silent about columns it
  was never told about. The comparison itself lives in :mod:`src.data.fingerprint`, which
  is track-agnostic; what is here is this track's manifest and the default that binds to
  it. The limits of that check are documented there.

Every bound and every category list here was measured against the real archive, not
guessed: 307,511 rows, positive rate 0.0807, ``DAYS_EMPLOYED == 365243`` on 18.0% of rows,
``CODE_GENDER == "XNA"`` on exactly 4, and ``EXT_SOURCE_1`` missing on 56%.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from src.data.fingerprint import assert_column_manifest, read_manifest
from src.features.specs import CREDIT_FEATURES

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
COLUMN_MANIFEST = SCHEMA_DIR / "home_credit_columns.txt"

# "Never employed", encoded as roughly a thousand years in the future. On 18.0% of rows, so
# it is a population rather than an error.
#
# Converted to NaN by the `sentinels` step inside the fitted pipeline, NOT here. That
# placement is deliberate and was a review finding: the pipeline is the only thing both
# training and serving pass through, so normalising in ingest alone would leave training
# seeing NaN while a request carried the raw sentinel, and the same applicant would score
# differently depending on which path it arrived by. Ingest's job is to *record* that the
# sentinel is present, not to erase it.
# Read from the spec rather than repeated, because the only remaining consumer here is a log
# line while the spec's value is what actually normalises. Two unlinked literals would let the
# sentinel change in one place, leave normalisation correct, compute zero sentinel rows, and
# have `if sentinel_rows:` suppress the log entirely -- silently deleting the observability this
# module keeps the value in place for.
DAYS_EMPLOYED_SENTINEL: Final = CREDIT_FEATURES.sentinels["DAYS_EMPLOYED"]

# 4 rows of 307,511. Kept as a category the model was fit on rather than dropped or
# imputed: at that frequency either choice is noise, and rejecting it at the API boundary
# while training on it would be a contract mismatch. Recorded in the debt ledger.
GENDER_CATEGORIES: Final = ("F", "M", "XNA")

CONTRACT_TYPES: Final = ("Cash loans", "Revolving loans")
WEEKDAYS: Final = (
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
)


class HomeCreditSchema(pa.DataFrameModel):
    """The modeled subset of ``application_train.csv``, with measured bounds.

    ``strict=False`` because the raw table carries 122 columns and only 26 are modeled;
    the rest are permitted through and dropped downstream. The full set is checked
    structurally by :func:`assert_fingerprint` instead.

    Bounds reject the impossible, not the merely unusual. An unusually large loan should
    ingest; a negative one should not.
    """

    SK_ID_CURR: Series[int] = pa.Field(unique=True, gt=0)
    TARGET: Series[int] = pa.Field(isin=(0, 1))

    # Money. gt=0 on income and credit: a zero-income or zero-credit application is not a
    # data point about default risk, it is a broken record.
    AMT_INCOME_TOTAL: Series[float] = pa.Field(gt=0)
    AMT_CREDIT: Series[float] = pa.Field(gt=0)
    AMT_ANNUITY: Series[float] = pa.Field(ge=0, nullable=True)
    AMT_GOODS_PRICE: Series[float] = pa.Field(ge=0, nullable=True)

    # Negative day offsets from the application date. le=0 encodes that sign convention, so
    # an upstream switch to positive ages fails here rather than training on a fiction.
    DAYS_BIRTH: Series[int] = pa.Field(le=0, ge=-30000)
    # NOT bounded above: the 365243 sentinel is a legitimate value in this column. Bounding
    # it would reject 18% of the population the sentinel describes.
    DAYS_EMPLOYED: Series[int] = pa.Field(ge=-30000)
    DAYS_REGISTRATION: Series[float] = pa.Field(le=0, ge=-30000)
    DAYS_ID_PUBLISH: Series[int] = pa.Field(le=0, ge=-30000)

    CNT_CHILDREN: Series[int] = pa.Field(ge=0, le=25)
    CNT_FAM_MEMBERS: Series[float] = pa.Field(ge=0, le=25, nullable=True)
    REGION_POPULATION_RELATIVE: Series[float] = pa.Field(ge=0, le=1)

    # Normalised external credit scores, and the strongest single predictors here.
    # EXT_SOURCE_1 is missing on 56% of rows and EXT_SOURCE_3 on 20% -- nullable is the
    # measured reality, not a convenience.
    EXT_SOURCE_1: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    EXT_SOURCE_2: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    EXT_SOURCE_3: Series[float] = pa.Field(ge=0, le=1, nullable=True)

    HOUR_APPR_PROCESS_START: Series[int] = pa.Field(ge=0, le=23)

    NAME_CONTRACT_TYPE: Series[str] = pa.Field(isin=CONTRACT_TYPES)
    CODE_GENDER: Series[str] = pa.Field(isin=GENDER_CATEGORIES)
    FLAG_OWN_CAR: Series[str] = pa.Field(isin=("Y", "N"))
    FLAG_OWN_REALTY: Series[str] = pa.Field(isin=("Y", "N"))
    WEEKDAY_APPR_PROCESS_START: Series[str] = pa.Field(isin=WEEKDAYS)

    # Free-ish categoricals. Membership is enforced at the API boundary against a committed
    # vocabulary; here they are only required to be present and non-null where the source
    # is. OCCUPATION_TYPE is missing on 31% of rows.
    NAME_INCOME_TYPE: Series[str] = pa.Field()
    NAME_EDUCATION_TYPE: Series[str] = pa.Field()
    NAME_FAMILY_STATUS: Series[str] = pa.Field()
    NAME_HOUSING_TYPE: Series[str] = pa.Field()
    OCCUPATION_TYPE: Series[str] = pa.Field(nullable=True)
    ORGANIZATION_TYPE: Series[str] = pa.Field()

    class Config:
        strict = False
        coerce = True


def load_manifest(path: Path = COLUMN_MANIFEST) -> list[str]:
    """Read this track's committed column manifest.

    A thin binding of the shared reader to the credit manifest. The shared function takes
    no default path on purpose -- a shared default is what made every track fingerprint
    against *this* manifest, so the default belongs next to the manifest it names.
    """
    return read_manifest(path)


def assert_fingerprint(frame: pd.DataFrame, path: Path = COLUMN_MANIFEST) -> None:
    """Compare the frame's full column set against the credit manifest.

    ``ingest()`` reaches this through ``Track.validation_module()``, so every track exposes
    the same name with its own manifest bound as the default. The comparison is shared; the
    manifest is not.
    """
    assert_column_manifest(frame, path)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Observe the source's quirks without changing anything. Value-identical to its input.

    It is a no-op on the data by design and returns a copy only so a caller cannot be surprised
    by aliasing -- the same explicitness ``downcast()`` documents about itself. What it does is
    **log**, and the logs are the deliverable.

    **Reports** the ``DAYS_EMPLOYED`` sentinel population; it no longer adds a column for it.
    The ``DAYS_EMPLOYED_ANOMALY`` flag this used to write was dropped before the parquet by
    ``ingest.py`` anyway -- only the log line survived -- and the flag itself measured as a
    99.9967% duplicate of ``NAME_INCOME_TYPE``, so the pipeline no longer derives one either.
    The log line is the point: it is how the sentinel population stays observable at ingest
    time, which is where a change in its share would first show.

    The sentinel value itself is deliberately **left in place**. It is converted to NaN by
    the ``sentinels`` step inside the fitted pipeline, which is the only place both
    training and serving pass through. Converting here would mean training saw NaN while a
    live request carried 365243, and the same applicant would score differently by path --
    training/serving skew that no test at either end would catch.

    ``CODE_GENDER == "XNA"`` (4 rows) is left as-is and logged. At that frequency dropping
    or imputing is noise, and the serving contract accepts it, so rejecting it here would
    make ingest and the API disagree about what a valid applicant looks like.
    """
    out = frame.copy()

    sentinel_rows = int((out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum())
    if sentinel_rows:
        logger.info(
            "DAYS_EMPLOYED sentinel (%d) on %d rows (%.1f%%) -- normalised to NaN inside the "
            "pipeline, not here, so training and serving see the same thing. Not flagged: "
            "NAME_INCOME_TYPE already identifies this population",
            DAYS_EMPLOYED_SENTINEL,
            sentinel_rows,
            100 * sentinel_rows / max(len(out), 1),
        )

    xna = int((out["CODE_GENDER"] == "XNA").sum())
    if xna:
        logger.info(
            "CODE_GENDER == 'XNA' on %d row(s); kept as a trained-on category rather than "
            "imputed -- see docs/debt-ledger.md",
            xna,
        )

    return out


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate against :class:`HomeCreditSchema`, reporting every violation at once.

    ``lazy=True`` is the point. Without it a corrupted batch surfaces one failure per run,
    so diagnosing a broken upstream export becomes a sequence of ingest attempts instead of
    one report.
    """
    return HomeCreditSchema.validate(frame, lazy=True)


def downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged, on purpose. Kept as the documented place not to do this.

    Narrowing dtypes here is a trap, and it cost two failed training runs to see why.

    Whatever dtypes reach ``fit()`` become the signature MLflow enforces at serving time.
    The API builds its frame from JSON, where numbers arrive as 64-bit, so storing a column
    as ``float32`` or ``int32`` made the logged signature demand the narrow type and every
    request failed schema enforcement -- on a dtype nobody chose deliberately, to save a few
    MB on an 11 MB file. Training passed, serving 500'd, and each half looked correct alone.

    ``tests/test_skew.py`` is what caught it: the one test that sends the same applicant
    down both paths and compares. It could only catch it once a model was actually
    registered, which is why this survived until the first real training run.

    A function rather than a deleted call site: ingest's stages read as a pipeline, and the
    reason this one is a no-op is worth more inline than in a commit message. If the parquet
    footprint ever genuinely matters, narrow it at the *storage* boundary and widen on read,
    rather than letting storage dictate the serving contract.
    """
    return frame
