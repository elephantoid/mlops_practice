"""The hosted evidence plane: the GCS reader, the traffic generator and the daily PSI row.

Hermetic. GCS is a fake with the client's ``list_blobs`` and the blob's ``download_as_text``; the
generator reads only the committed PSI references, which is the point -- the Action that runs it
has no ``data/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.specs import FEATURE_SPECS
from src.monitoring import daily, drift, gcs_log, psi, traffic

DAY = date(2026, 10, 20)
SINCE, UNTIL = gcs_log.day_window(DAY)


@dataclass
class _Blob:
    name: str
    text: str

    def download_as_text(self) -> str:
        return self.text


class _Client:
    def __init__(self, entries_by_day: dict[date, list[dict]]):
        self.blobs = [
            _Blob(
                f"{gcs_log.LOG_DIR}/{day:%Y/%m/%d}/00:00:00_00:59:59_S0.json",
                "".join(json.dumps(entry) + "\n" for entry in entries),
            )
            for day, entries in entries_by_day.items()
        ]
        self.prefixes: list[str] = []

    def list_blobs(self, bucket: str, prefix: str):
        self.prefixes.append(prefix)
        return [b for b in self.blobs if b.name.startswith(prefix)]


def _entry(stamp: datetime, track: str = "credit", **features) -> dict:
    """A LogEntry as the sink writes it: the printed record under ``jsonPayload``."""
    return {
        "logName": "projects/p/logs/run.googleapis.com%2Fstdout",
        "jsonPayload": {
            "log_type": "prediction",
            "timestamp": stamp.isoformat(),
            "track": track,
            "model_version": "6",
            "risk_probability": 0.1,
            "features": features or {"AMT_CREDIT": 1.0},
        },
    }


def _read(client: _Client, track: str = "credit") -> pd.DataFrame:
    return gcs_log.logged_current(SINCE, UNTIL, track, bucket="b", client=client)


# --- The reader ----------------------------------------------------------------------------------


def test_the_reader_lists_the_day_and_one_either_side():
    client = _Client({})
    with pytest.raises(drift.InsufficientCurrentData):
        _read(client)
    assert client.prefixes == [
        f"{gcs_log.LOG_DIR}/2026/10/19/",
        f"{gcs_log.LOG_DIR}/2026/10/20/",
        f"{gcs_log.LOG_DIR}/2026/10/21/",
    ]


def test_a_day_is_its_records_timestamps_not_its_folder():
    """A record filed under the next day's folder counts; one stamped outside the day does not."""
    client = _Client(
        {
            DAY: [_entry(SINCE + timedelta(hours=1), AMT_CREDIT=1.0)],
            DAY + timedelta(days=1): [
                _entry(UNTIL - timedelta(seconds=1), AMT_CREDIT=2.0),
                _entry(UNTIL, AMT_CREDIT=3.0),
            ],
            DAY - timedelta(days=1): [_entry(SINCE - timedelta(seconds=1), AMT_CREDIT=4.0)],
        }
    )
    assert sorted(_read(client)["AMT_CREDIT"]) == [1.0, 2.0]


def test_the_reader_skips_other_tracks_and_entries_without_a_payload():
    client = _Client(
        {
            DAY: [
                _entry(SINCE, AMT_CREDIT=1.0),
                _entry(SINCE, track="fraud", Amount=5.0),
                {"textPayload": "INFO: Uvicorn running"},
                {"jsonPayload": {"log_type": "access", "track": "credit"}},
            ]
        }
    )
    assert _read(client)["AMT_CREDIT"].tolist() == [1.0]


def test_an_empty_day_raises_insufficient_data_naming_the_bucket_and_window():
    with pytest.raises(drift.InsufficientCurrentData, match=r"gs://b in \[2026-10-20"):
        _read(_Client({DAY: [_entry(SINCE, track="fraud")]}))


def test_the_reader_refuses_to_guess_a_bucket(monkeypatch):
    monkeypatch.delenv(gcs_log.BUCKET_ENV, raising=False)
    with pytest.raises(RuntimeError, match=gcs_log.BUCKET_ENV):
        gcs_log.bucket_name()


# --- The generator -------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=sorted(FEATURE_SPECS))
def generated(request) -> tuple[str, pd.DataFrame]:
    return request.param, traffic.generate(request.param, 5_000, seed=1)


def test_every_generated_row_passes_the_request_contract(generated):
    track, frame = generated
    assert len(traffic.payloads(track, frame)) == len(frame)


def test_generated_traffic_reproduces_every_reference_histogram(generated):
    """Each feature's PSI against its own reference stays inside the noise budget."""
    track, frame = generated
    summary = psi.load_summary(track)
    worst = max(
        psi.psi(entry["counts"], psi.bin_counts(entry, pd.Series(frame[column].tolist())))
        for column, entry in summary["features"].items()
    )
    assert worst < psi.NOISE_BUDGET


def test_generated_credit_traffic_reaches_the_sentinel_null_and_other_bins():
    frame = traffic.generate("credit", 5_000, seed=1)
    kept = psi.load_summary("credit")["features"]["ORGANIZATION_TYPE"]["categories"]
    assert (frame["DAYS_EMPLOYED"] == 365243).any()
    assert frame["EXT_SOURCE_1"].isna().any()
    assert not frame["ORGANIZATION_TYPE"].isin(kept).all()


def test_the_registered_drill_clears_the_alert_line_at_a_day_s_volume():
    (drill,) = [d for d in traffic.DRILLS if d.track == "credit"]
    assert traffic.drill_on(drill.day, "credit") == drill
    assert traffic.drill_on(drill.day, "fraud") is None
    frame = traffic.generate("credit", traffic.DAILY_ROWS["credit"], seed=1, drill=drill)
    entry = psi.load_summary("credit")["features"][drill.column]
    shifted = psi.psi(entry["counts"], psi.bin_counts(entry, pd.Series(frame[drill.column])))
    assert shifted > psi.PSI_ALERT


def test_a_day_s_traffic_is_fixed_by_its_date():
    assert traffic.seed_for(DAY, "credit") != traffic.seed_for(DAY, "fraud")
    assert traffic.seed_for(DAY, "credit") != traffic.seed_for(DAY + timedelta(days=1), "credit")


# --- The daily row -------------------------------------------------------------------------------


def _generated_day(track: str, rows: int, model_version: str | None = None) -> list[dict]:
    """A day of generated requests, with scores drawn from the reference's own score bins."""
    summary = psi.load_summary(track)
    frame = traffic.generate(track, rows, seed=2)
    probability = traffic._Field(False, False, 0.0, False, 1.0, None)
    scores = traffic._numeric(summary["score"], probability, rows, np.random.default_rng(2))
    return [
        {
            "jsonPayload": {
                "log_type": "prediction",
                "timestamp": (SINCE + timedelta(seconds=n)).isoformat(),
                "track": track,
                "model_version": model_version or summary["model_version"],
                "risk_probability": score,
                "features": record,
            }
        }
        for n, (record, score) in enumerate(
            zip(frame.to_dict(orient="records"), scores, strict=True)
        )
    ]


def test_a_generated_day_writes_a_stable_row_and_an_empty_track_writes_an_alerting_one(tmp_path):
    out = tmp_path / "psi_daily.jsonl"
    client = _Client({DAY: _generated_day("credit", traffic.DAILY_ROWS["credit"])})
    lines = {line["track"]: line for line in daily.run(DAY, out, client=client, bucket="b")}

    assert lines["credit"]["status"] == "ok"
    assert lines["credit"]["rows"] == traffic.DAILY_ROWS["credit"]
    assert not lines["credit"]["alert"]
    assert lines["fraud"]["status"] == "insufficient"
    assert lines["fraud"]["alert"]
    assert [line["track"] for line in daily.alerts(out, DAY)] == ["fraud"]


def test_a_day_served_by_a_model_the_reference_is_not_for_alerts(tmp_path):
    client = _Client({DAY: _generated_day("credit", traffic.DAILY_ROWS["credit"], "999")})
    (credit, _) = daily.run(DAY, tmp_path / "psi_daily.jsonl", client=client, bucket="b")
    assert credit["status"] == "stale-reference"
    assert credit["alert"]
    assert credit["score_psi"] is None


def test_a_short_day_is_an_alerting_row_with_its_reason():
    line = daily.row(
        "credit",
        DAY,
        psi.load_summary("credit"),
        pd.DataFrame({"AMT_CREDIT": [1.0]}),
    )
    assert line["status"] == "insufficient"
    assert line["alert"]
    assert "PSI needs" in line["reason"]


def test_a_drill_day_is_labelled_in_its_row():
    (drill,) = traffic.DRILLS
    line = daily.row(drill.track, drill.day, psi.load_summary(drill.track), "no data")
    assert line["drill"] == drill.name
    assert line["generator"] == traffic.GENERATOR


def test_the_check_reads_the_latest_row_per_track_and_alerts_on_a_missing_one(tmp_path):
    out = tmp_path / "psi_daily.jsonl"
    rows = [
        {"date": DAY.isoformat(), "track": "credit", "alert": True},
        {"date": DAY.isoformat(), "track": "credit", "alert": False},
        {"date": (DAY - timedelta(days=1)).isoformat(), "track": "fraud", "alert": False},
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert daily.alerts(out, DAY) == [
        {"date": DAY.isoformat(), "track": "fraud", "status": "missing", "alert": True}
    ]


def test_the_day_window_is_one_utc_calendar_day():
    assert gcs_log.day_window(DAY) == (
        datetime(2026, 10, 20, tzinfo=UTC),
        datetime(2026, 10, 21, tzinfo=UTC),
    )
