"""Ingest the raw IBM Telco churn CSV into a validated, versioned parquet snapshot.

.. warning::

   **This module has not been retargeted and does not work with either riskwatch track.**

   It still reads ``data/raw/telco.csv`` against ``TelcoRawSchema`` and writes to a flat
   ``data/processed/`` rather than the per-track layout the rest of the pipeline now
   expects. Every other module -- ``features/pipeline.py``, ``models/train.py``,
   ``monitoring/drift.py`` -- takes a ``Track`` or a ``FeatureSpec``; this one does not.

   Retargeting it is plan Step 4, which is now unblocked: the Home Credit archive landed
   on 2026-09-22 and `data/raw/credit/application_train.csv` is cached, with the 122-name
   column manifest written from it at `src/data/schemas/home_credit_columns.txt`. Step 4
   is the next unit of work; until it lands, this module is the reference implementation
   of the pattern it must follow, not a working component.

   Left standing rather than deleted so the working reference implementation is visible
   while Step 4 is written against it. Do not call it expecting riskwatch behaviour.

Run directly with::

    uv run python src/data/ingest.py

The schema below is the input contract for the whole project: ``features/pipeline.py``
and ``models/train.py`` both read the parquet this module writes, so anything that
violates the contract must fail here rather than surface as a confusing model error.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LATEST_NAME = "latest.parquet"

# Observed value sets. Three columns carry sentinel strings rather than nulls:
# "No phone service" / "No internet service" encode a dependency on another column.
YES_NO = ["Yes", "No"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]
YES_NO_PHONE = ["Yes", "No", "No phone service"]
GENDER = ["Female", "Male"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]


class TelcoRawSchema(pa.DataFrameModel):
    """Contract for the IBM Telco Customer Churn dataset: 7,043 rows, 21 columns.

    ``TotalCharges`` is declared ``float`` even though the raw CSV types it as
    ``object`` -- :func:`clean` must run before validation.

    Bounds here are domain invariants only (a negative tenure or charge is impossible),
    not the observed range of this particular snapshot. Values that are merely unusual --
    a tenure past the current 72-month maximum, say -- are a distribution shift, which is
    Evidently's job to report, not a reason to fail the retraining pipeline.
    """

    customerID: Series[str] = pa.Field(unique=True)
    gender: Series[str] = pa.Field(isin=GENDER)
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=YES_NO)
    Dependents: Series[str] = pa.Field(isin=YES_NO)
    tenure: Series[int] = pa.Field(ge=0)
    PhoneService: Series[str] = pa.Field(isin=YES_NO)
    MultipleLines: Series[str] = pa.Field(isin=YES_NO_PHONE)
    InternetService: Series[str] = pa.Field(isin=INTERNET_SERVICE)
    OnlineSecurity: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    OnlineBackup: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    DeviceProtection: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    TechSupport: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    StreamingTV: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    StreamingMovies: Series[str] = pa.Field(isin=YES_NO_INTERNET)
    Contract: Series[str] = pa.Field(isin=CONTRACT)
    PaperlessBilling: Series[str] = pa.Field(isin=YES_NO)
    PaymentMethod: Series[str] = pa.Field(isin=PAYMENT_METHOD)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0)
    Churn: Series[str] = pa.Field(isin=YES_NO)

    class Config:
        strict = True
        coerce = True


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV verbatim.

    No ``na_values`` handling here on purpose: the blank ``TotalCharges`` sentinel is a
    single space rather than an empty string, so pandas leaves it as an ``object``
    column. :func:`clean` deals with it explicitly instead of hiding it in a parser flag.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Download it first.")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``TotalCharges`` to float, zero-filling customers who were never billed.

    Eleven rows store a single space instead of a number. Every one of them has
    ``tenure == 0``, meaning the customer signed up but has not been billed a cycle yet
    -- so 0.0 is the semantically correct value, not an imputation, and the rows stay.

    A blank on a row with ``tenure > 0`` would be a genuine data anomaly, so it raises
    rather than being silently zeroed alongside the legitimate cases.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    unbilled = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[unbilled, "TotalCharges"] = 0.0
    logger.info("TotalCharges: zero-filled %d unbilled rows (tenure == 0)", int(unbilled.sum()))

    still_missing = out["TotalCharges"].isna()
    if still_missing.any():
        offenders = out.loc[still_missing, ["customerID", "tenure"]].to_dict("records")
        raise ValueError(f"Non-numeric TotalCharges on billed customers: {offenders}")

    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against :class:`TelcoRawSchema`, reporting every violation at once.

    ``lazy=True`` collects all failures instead of raising on the first one, which makes
    a broken upstream file one debugging round-trip instead of many.
    """
    return TelcoRawSchema.validate(df, lazy=True)


def ingest(raw_path: Path = RAW_PATH, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Load, clean, validate, and write a timestamped parquet snapshot.

    Each run writes a new ``telco_<UTC timestamp>.parquet`` and repoints
    ``latest.parquet`` at it. Keeping the history lets the Airflow retrain DAG hold a
    per-run artifact and lets drift monitoring compare any two batches later.

    Returns the path of the snapshot just written.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = validate(clean(load_raw(raw_path)))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = processed_dir / f"telco_{stamp}.parquet"
    df.to_parquet(out_path, index=False)

    # Swap the pointer atomically: build the link under a temp name, then rename over
    # the old one. unlink-then-symlink would leave a window in which latest.parquet does
    # not exist, and a crash there strands every downstream reader with FileNotFoundError.
    # The target stays relative so the link survives data/processed/ being mounted at a
    # different path inside the container.
    latest = processed_dir / LATEST_NAME
    pending = processed_dir / f".{LATEST_NAME}.tmp"
    pending.unlink(missing_ok=True)
    pending.symlink_to(out_path.name)
    pending.replace(latest)

    logger.info("Wrote %d rows x %d columns to %s", len(df), df.shape[1], out_path)
    logger.info("%s -> %s", LATEST_NAME, out_path.name)
    return out_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest()


if __name__ == "__main__":
    main()
