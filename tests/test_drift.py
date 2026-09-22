"""Tests for the drift module's track-derived contracts.

This module had **no test coverage at all** before the retarget, which is how it kept a
full set of Telco constants through a rename: `NON_FEATURE_COLUMNS = ["customerID",
"Churn"]`, a hardcoded `MonthlyCharges` perturbation, a flat single-track reference path,
and a shared Pushgateway job. Nothing here asserted any of it, so nothing caught it.

These tests are hermetic -- they build frames in memory and never read `data/` or call
Evidently -- so they run in a fresh clone with no archives present.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.monitoring import drift


def _credit_frame(rows: int = 600) -> pd.DataFrame:
    """A frame shaped like the credit reference, including columns that must be dropped."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(100000, 100000 + rows),
            "TARGET": [0, 1] * (rows // 2),
            "AMT_CREDIT": [400000.0 + i for i in range(rows)],
            "AMT_INCOME_TOTAL": [200000.0 + i for i in range(rows)],
            "CNT_CHILDREN": [i % 3 for i in range(rows)],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * rows,
            # Present in the raw table, absent from the modeled subset. Home Credit ships
            # 122 columns against 26 modeled, so this is the common case, not an edge one.
            "FLAG_MOBIL": [1] * rows,
        }
    )


def test_non_feature_columns_come_from_the_track_not_a_constant():
    """The Telco constant dropped customerID and Churn, which do not exist here.

    Left hardcoded against a credit frame it would drop nothing: SK_ID_CURR would stay in,
    register as drifted on every run because it is unique per row, and inflate the share
    the retrain trigger reads. The module's own comment measures that effect at 0.095
    against 0.048.
    """
    columns = drift.non_feature_columns("credit")

    assert "SK_ID_CURR" in columns
    assert "TARGET" in columns
    assert "customerID" not in columns
    assert "Churn" not in columns


def test_reference_and_push_job_are_per_track():
    """Flat single-track paths and a shared job name do not survive two tracks."""
    assert drift.reference_path("credit") != drift.reference_path("fraud")
    assert drift.reference_path("credit").parts[-2] == "credit"

    # push_to_gateway REPLACES all metrics under a job name, so a shared job would mean
    # each track's push silently erased the other's.
    assert drift.push_job("credit") != drift.push_job("fraud")
    assert "credit" in drift.push_job("credit")


def test_load_reference_reduces_to_the_modeled_feature_set(tmp_path):
    """A wide reference against a narrow log reports drift on a schema difference."""
    frame = _credit_frame()
    path = tmp_path / "latest.parquet"
    frame.to_parquet(path)

    loaded = drift.load_reference(path=path, track="credit")

    assert "SK_ID_CURR" not in loaded.columns, "id column must be dropped"
    assert "TARGET" not in loaded.columns, "label must be dropped"
    assert "FLAG_MOBIL" not in loaded.columns, "unmodeled columns must be dropped"
    assert "AMT_CREDIT" in loaded.columns
    assert len(loaded) == len(frame)


def test_load_reference_names_the_track_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="credit"):
        drift.load_reference(path=tmp_path / "absent.parquet", track="credit")


def test_synthetic_current_shifts_the_tracks_column():
    """The documented simulation, now per track rather than hardcoded to MonthlyCharges."""
    reference = _credit_frame()[["AMT_CREDIT", "AMT_INCOME_TOTAL", "CNT_CHILDREN"]]

    drifted = drift.synthetic_current(reference, track="credit")

    assert len(drifted) == drift.SYNTHETIC_SAMPLE_SIZE
    # The shifted column moved by exactly the documented factor.
    ratio = drifted["AMT_CREDIT"].mean() / reference["AMT_CREDIT"].loc[drifted.index].mean()
    assert ratio == pytest.approx(drift.SYNTHETIC_UPLIFT)
    # Nothing else moved -- a batch where everything shifted proves nothing about a
    # detector's ability to localise drift.
    untouched = drifted["AMT_INCOME_TOTAL"].mean()
    assert untouched == pytest.approx(reference["AMT_INCOME_TOTAL"].loc[drifted.index].mean())


def test_synthetic_current_raises_rather_than_returning_an_unshifted_frame():
    """A 'synthetic drift' batch with no drift in it makes the detector look broken.

    The old code indexed a hardcoded column and would KeyError; the failure mode worth
    guarding is the quieter one where a missing column silently yields a clean frame.
    """
    frame = _credit_frame()[["AMT_INCOME_TOTAL"]]

    with pytest.raises(KeyError, match="AMT_CREDIT"):
        drift.synthetic_current(frame, track="credit")


def test_synthetic_current_rejects_an_unconfigured_track():
    with pytest.raises(KeyError, match="no synthetic drift column"):
        drift.synthetic_current(_credit_frame(), track="telco")


def test_synthetic_current_clamps_to_the_frame_size():
    """A reference smaller than the sample size must not raise inside pandas."""
    small = _credit_frame(rows=10)[["AMT_CREDIT", "AMT_INCOME_TOTAL"]]
    assert len(drift.synthetic_current(small, track="credit")) == 10


def test_logged_current_filters_by_track(tmp_path):
    """The log interleaves tracks; comparing across them measures schema, not drift."""
    path = tmp_path / "predictions.jsonl"
    records = [
        {"track": "credit", "features": {"AMT_CREDIT": 400000.0, "SK_ID_CURR": 1}},
        {"track": "fraud", "features": {"Amount": 12.5, "Time": 3600}},
        {"track": "credit", "features": {"AMT_CREDIT": 410000.0, "SK_ID_CURR": 2}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    credit = drift.logged_current(path=path, track="credit")

    assert len(credit) == 2
    assert "Amount" not in credit.columns, "a fraud row leaked into the credit frame"
    assert "SK_ID_CURR" not in credit.columns, "id column must be dropped here too"


def test_logged_current_treats_untracked_records_as_the_default_track(tmp_path):
    """Records written before the track field existed stay usable."""
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"features": {"AMT_CREDIT": 400000.0}}) + "\n")

    assert len(drift.logged_current(path=path, track=drift.DEFAULT_TRACK)) == 1


def test_logged_current_raises_when_no_rows_match_the_track(tmp_path):
    """An empty frame produces a report full of NaNs that looks like a result."""
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"track": "fraud", "features": {"Amount": 12.5}}) + "\n")

    with pytest.raises(ValueError, match="no 'credit' rows"):
        drift.logged_current(path=path, track="credit")


def test_min_current_rows_is_still_telco_derived():
    """MIN_CURRENT_ROWS = 100 was measured on Telco and is carried unexamined.

    It is re-derivation debt alongside drift_share > 0.2 and AUC delta > 0.01. This test
    does not assert the value is right -- it cannot be, before the data exists -- it
    asserts the guard is still wired, so removing it is a deliberate act rather than an
    accident.
    """
    assert drift.MIN_CURRENT_ROWS == 100
