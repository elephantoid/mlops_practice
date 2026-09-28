"""Validation contract and cleaning for the credit track's Home Credit source.

Two layers, because 122 columns make full enumeration impractical where the retired Telco
schema enumerated all 21:

* :class:`HomeCreditSchema` enumerates only the **modeled** subset with real domain bounds,
  validated lazily so a corrupted batch reports every violation at once rather than the
  first one.
* :func:`assert_fingerprint` compares the frame's full column set against a committed
  manifest. That catches added, dropped and renamed columns -- upstream structural change
  the schema cannot see, because a schema with ``strict=False`` is silent about columns it
  was never told about.

The fingerprint is **not** a total upstream detector and is not sold as one. It cannot see
a dtype change, a unit change, a semantic change under a stable name, a null-rate jump, a
recode, or truncation. Those need the data-quality rules that land with the drift
decomposer. Overselling this would make DoD (6)'s upstream cause class look stronger than
it is.

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
DAYS_EMPLOYED_SENTINEL: Final = 365243

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


class SchemaFingerprintError(RuntimeError):
    """The raw frame's column set does not match the committed manifest.

    Separate from a pandera failure because the two mean different things: a pandera
    violation is bad *data* in a known shape, this is a changed *shape*. Upstream renaming
    a column is not a row-level problem and the fix is not to clean the batch.
    """


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
    """Read the committed column manifest.

    Raises rather than returning an empty list: an empty manifest would make
    :func:`assert_fingerprint` a no-op that still reports success, which is worse than
    having no fingerprint at all.
    """
    if not path.is_file():
        raise SchemaFingerprintError(
            f"Column manifest missing at {path}. It is written from the archive during "
            f"ingest setup; without it the structural check cannot run."
        )
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise SchemaFingerprintError(f"Column manifest at {path} is empty.")
    return names


def assert_fingerprint(frame: pd.DataFrame, path: Path = COLUMN_MANIFEST) -> None:
    """Compare the frame's full column set against the manifest.

    Names the specific columns that moved. "The schema changed" sends whoever reads it
    diffing 122 names by hand; "ORGANIZATION_TYPE was renamed" does not.

    Order is deliberately not checked. Column order is not a contract anyone upstream
    promised, and a reorder breaks nothing downstream because every consumer selects by
    name.
    """
    expected = set(load_manifest(path))
    actual = set(frame.columns)

    missing = sorted(expected - actual)
    added = sorted(actual - expected)

    if not missing and not added:
        return

    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if added:
        parts.append(f"unexpected: {added}")

    # A same-size swap is the renaming case, and saying so saves the reader the inference.
    hint = ""
    if missing and added and len(missing) == len(added):
        hint = " (same count missing and added -- likely a rename upstream)"

    raise SchemaFingerprintError(
        f"Raw column set does not match {path.name}: {'; '.join(parts)}{hint}. "
        f"Expected {len(expected)} columns, got {len(actual)}."
    )


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Record the source's quirks without erasing them.

    Adds ``DAYS_EMPLOYED_ANOMALY`` -- a 0/1 flag marking the "never employed" population.
    The flag is informative in its own right: those applicants have no employment history
    to score, which is different from having a short one.

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
    out["DAYS_EMPLOYED_ANOMALY"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype("int8")
    if sentinel_rows:
        logger.info(
            "DAYS_EMPLOYED sentinel (%d) on %d rows (%.1f%%) -- flagged as "
            "DAYS_EMPLOYED_ANOMALY; the value is normalised to NaN inside the pipeline, "
            "not here, so training and serving see the same thing",
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
