"""Acquire, validate and snapshot a track's raw source into versioned parquet.

Run directly with::

    uv run python -m src.data.ingest --track credit

This is the input contract for everything downstream: ``features/pipeline.py``,
``models/train.py`` and ``monitoring/drift.py`` all read the parquet this writes, so
anything violating the contract must fail here rather than surface later as a confusing
model error or a drift number that answers the wrong question.

**Track-driven, not domain-specific.** Acquisition comes from
:mod:`src.data.kaggle_source`, the schema and cleaning from the track's own module
(:mod:`src.data.credit` today), and the paths from the ``Track`` descriptor. Adding the
fraud track is one new module and one registry entry; nothing here changes.

Snapshots are timestamped and the ``latest.parquet`` symlink is repointed atomically. The
retrain DAG depends on that: ``task_preflight`` resolves the symlink to a concrete file
*before* ingest runs, so drift compares live traffic against the snapshot the serving model
was actually trained on rather than the one ingest just produced.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.kaggle_source import acquire
from src.data.tracks import get_track, registered_track_names
from src.features.specs import get_feature_spec

logger = logging.getLogger(__name__)

LATEST_NAME = "latest.parquet"

# There is deliberately no TRACK_MODULES registry here. The validation module is carried by
# the track's own SchemaSpec and resolved through Track.validation_module(), because a second
# registry meant a track could be registered in TRACKS, accepted by the CLI, and still fail
# at dispatch -- and adding fraud would need edits in two places rather than the one the
# seam promises.


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw table verbatim, every column.

    Deliberately NOT restricted to the modeled subset. The fingerprint compares the frame's
    full column set against the manifest, so reading a subset would make the structural
    check compare a selection against itself and pass unconditionally -- the added, dropped
    and renamed columns it exists to catch are all outside the modeled subset by definition.

    Peak memory is the price. At 307,511 rows x 122 columns that is a few hundred MB for the
    duration of one call, which is affordable; if it stops being affordable, read the header
    alone for the fingerprint and then re-read the subset, rather than weakening the check.
    """
    return pd.read_csv(path, low_memory=False)


def ingest(track: str = "credit", *, allow_fallback: bool = True) -> Path:
    """Acquire, validate, clean and snapshot one track. Returns the snapshot path.

    Returns a ``Path`` rather than a frame because every stage downstream re-reads from
    disk. That is what lets the DAG pass a filename through XCom instead of pickling a
    300k-row frame between processes.
    """
    descriptor = get_track(track)
    spec = get_feature_spec(track)
    module = descriptor.schema.validation_module()

    acquisition = acquire(descriptor.source, descriptor.raw_dir, allow_fallback=allow_fallback)
    if acquisition.is_fallback:
        logger.warning(
            "Ingesting %r from the FALLBACK source %r, not the primary %r. These are "
            "different datasets: the schema and feature contract below were written for "
            "the primary, so validation is expected to fail unless they were retargeted "
            "too. source_used is recorded in the parquet metadata.",
            track,
            acquisition.source_used,
            descriptor.source.source_ref,
        )
    logger.info(
        "source_used=%s (cache_hit=%s) -> %s",
        acquisition.source_used,
        acquisition.from_cache,
        acquisition.path,
    )

    raw = load_raw(acquisition.path)
    logger.info("read %d rows x %d columns", len(raw), raw.shape[1])

    # Structural check first. A renamed column makes every row-level message downstream
    # misleading -- "AMT_CREDIT is missing" reads as bad data when the column was renamed.
    # The descriptor's manifest path, not the module's default. SchemaSpec carries a
    # per-track manifest_path and omitting it here meant every track fingerprinted against
    # src.data.credit's manifest through the function default -- so a second track would
    # have validated its structure against the credit column list and passed or failed for
    # reasons having nothing to do with its own data. The registry was consolidated to one
    # source of truth; this is that source actually being read.
    if descriptor.schema.manifest_path is not None:
        module.assert_fingerprint(raw, path=descriptor.schema.manifest_path)
    else:
        module.assert_fingerprint(raw)

    cleaned = module.clean(raw)
    validated = module.validate(cleaned)

    # Reduce to what is modeled, plus the id, the target and any derived flags. Carrying all
    # 122 columns into the parquet would make drift compare a ~120-column reference against
    # a 26-column prediction log and report every unmodeled column as drifted.
    # Derived columns are NOT written. The pipeline computes them from the raw sentinel, so
    # storing them here would put a column in the parquet that the request contract does not
    # have -- and anything training off this frame would log a model signature wider than
    # the API can satisfy. clean() still computes the flag, because its log line is how the
    # sentinel population is observable at ingest time.
    keep = [spec.id_column, spec.target_column, *spec.feature_columns]
    frame = module.downcast(validated[keep])

    positive_rate = float(frame[spec.target_column].mean())
    logger.info(
        "%d rows | %d columns | positive rate %.4f",
        len(frame),
        frame.shape[1],
        positive_rate,
    )

    processed_dir = descriptor.processed_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot = processed_dir / f"{track}_{stamp}.parquet"

    # source_used rides in the parquet's own metadata rather than a sidecar file: a model
    # trained on the fallback is otherwise indistinguishable downstream from one trained on
    # the primary, and for credit those are genuinely different datasets.
    table_metadata = {
        b"source_used": acquisition.source_used.encode(),
        b"is_fallback": str(acquisition.is_fallback).encode(),
        b"track": track.encode(),
        b"rows": str(len(frame)).encode(),
        b"positive_rate": f"{positive_rate:.6f}".encode(),
        b"ingested_at": stamp.encode(),
    }
    _write_parquet_with_metadata(frame, snapshot, table_metadata)

    _repoint_latest(processed_dir / LATEST_NAME, snapshot)
    logger.info("wrote %s (latest -> %s)", snapshot.name, snapshot.name)
    return snapshot


def _write_parquet_with_metadata(
    frame: pd.DataFrame, path: Path, metadata: dict[bytes, bytes]
) -> None:
    """Write parquet carrying key-value metadata alongside the data."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **metadata})
    pq.write_table(table, path, compression="snappy")


def read_metadata(path: Path) -> dict[str, str]:
    """Read back the key-value metadata a snapshot was written with.

    Exists so the provenance is queryable rather than merely stored -- ``source_used`` is
    only useful if something can ask for it.
    """
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    return {
        key.decode(): value.decode() for key, value in raw.items() if not key.startswith(b"pandas")
    }


def _repoint_latest(link: Path, target: Path) -> None:
    """Point ``latest.parquet`` at ``target``, atomically.

    Written to a temporary name and renamed rather than unlinked and recreated: a reader
    between the two steps would otherwise find no ``latest.parquet`` at all, and the DAG's
    preflight resolves exactly this link. ``os.replace`` is atomic on the same filesystem.

    Relative target so the tree stays movable -- an absolute link breaks the moment the
    repo is bind-mounted at a different path, which is precisely what the Airflow overlay
    does.
    """
    temporary = link.with_name(f"{link.name}.swap")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(target.name)
    os.replace(temporary, link)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest a track's raw source into parquet.")
    parser.add_argument(
        "--track",
        default="credit",
        choices=sorted(registered_track_names()),
        help="Which risk track to ingest.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "Fail instead of substituting the auth-free fallback source. The credit "
            "fallback is a different dataset, so a silent substitution would train a model "
            "nothing downstream expects."
        ),
    )
    args = parser.parse_args()

    ingest(args.track, allow_fallback=not args.no_fallback)


if __name__ == "__main__":
    main()
