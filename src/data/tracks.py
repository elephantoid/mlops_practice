"""Per-track acquisition and validation contracts.

A *track* is one risk domain end to end: where its raw data comes from, how that data is
validated, and what a model for it consumes. Two tracks are planned -- ``credit``
(Home Credit application defaults) and ``fraud`` (ULB card transactions) -- and they are
never joined. They share no key, no entity, no time base, and no feature space; their
positive rates differ by roughly 46x. What they share is this platform.

The seam is deliberately a dataclass and a registry, not a plugin framework. The test of
whether it is right: adding the fraud track in W2 must cost one new module and one
registry entry. If it needs an abstract base class hierarchy, the seam is over-built; if
it needs edits across six modules, it is under-built.

**This module imports pandera. `src/features/specs.py` does not, and must not.** The
serving path reaches feature contracts through ``src.features.specs`` directly, so
importing this module is never on the API's critical path. ``tests/test_tracks.py``
asserts that invariant rather than leaving it to convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import pandera as pa

from src.features.specs import FeatureSpec, get_feature_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

SourceKind = Literal["kaggle_competition", "kaggle_dataset", "openml", "url"]


@dataclass(frozen=True)
class SourceSpec:
    """Where a track's raw data comes from, and what to do when that fails.

    ``source_kind`` matters more than it looks. Kaggle exposes two download modes that
    fail in different ways, and conflating them produces a misdiagnosis:

    - ``kaggle_dataset`` needs only an API token.
    - ``kaggle_competition`` needs the token **and** browser acceptance of that
      competition's rules. Without the acceptance it returns a 403 that reads like an
      authentication failure but is actually a consent failure.

    That is why the fraud track (a dataset) is fetched before the credit track (a
    competition): proving the token in isolation is what makes a subsequent 403
    diagnosable as missing consent rather than a bad key.

    ``fallback`` is a code path, not a prose note. The deploy clock has no slack for an
    auth stall, so every track carries an auth-free alternative that actually runs.
    """

    source_kind: SourceKind
    source_ref: str
    primary_table: str
    fallback: SourceSpec | None = None
    archive_members: tuple[str, ...] = ()
    # Set where the fallback is a genuinely different dataset rather than the same data
    # by another route, so a silent substitution can never be mistaken for equivalence.
    equivalent_to_primary: bool = True

    def __post_init__(self) -> None:
        if not self.source_ref:
            raise ValueError("SourceSpec requires a non-empty source_ref")
        if not self.primary_table:
            raise ValueError(f"SourceSpec {self.source_ref!r} requires a primary_table")

    @property
    def requires_rule_acceptance(self) -> bool:
        """Whether this source needs a browser action no API call can perform."""
        return self.source_kind == "kaggle_competition"


@dataclass(frozen=True)
class SchemaSpec:
    """How a track's raw frame is validated.

    Two layers, because Home Credit's 122 columns make full enumeration impractical where
    the Telco schema enumerated all 21:

    - ``model`` is a pandera model enumerating only the modeled subset, with real domain
      bounds, validated with ``lazy=True`` so every violation is reported at once.
    - ``manifest_path`` points at a committed list of the full column set. Comparing
      against it detects added, dropped, and renamed columns.

    The manifest is **not** a total upstream-change detector, and the plan says so
    explicitly: it cannot see dtype changes, unit changes, semantic changes under a
    stable name, null-rate jumps, recodes, or truncation. Those need the data-quality
    rules that land with the drift decomposer. Overselling this as total would make
    DoD (6)'s upstream cause class look stronger than it is.
    """

    model: type[pa.DataFrameModel] | None
    manifest_path: Path | None = None

    @property
    def has_manifest(self) -> bool:
        return self.manifest_path is not None


@dataclass(frozen=True)
class Track:
    """One risk domain: acquisition, validation, and feature contract composed.

    Three concerns with three change rates and three consumer sets, held together but not
    fused. ``features`` is reached directly by the serving path; ``source`` and ``schema``
    are touched only by ingest.
    """

    name: str
    source: SourceSpec
    schema: SchemaSpec
    features: FeatureSpec

    @property
    def raw_dir(self) -> Path:
        """Cache location for downloaded archives. Never re-downloaded once populated."""
        return RAW_DIR / self.name

    @property
    def processed_path(self) -> Path:
        """Validated parquet snapshot for this track."""
        return PROCESSED_DIR / self.name / "latest.parquet"

    @property
    def model_name(self) -> str:
        """MLflow registered-model name.

        Track-derived rather than a shared constant: two registered models with
        independent schemas, thresholds, and retrain cadence are what make DoD (7)'s
        "drift in one track retrains only that track" an honest demonstration.
        """
        return f"riskwatch_{self.name}"

    @property
    def experiment_name(self) -> str:
        return f"riskwatch_{self.name}"


CREDIT_SOURCE = SourceSpec(
    source_kind="kaggle_competition",
    source_ref="home-credit-default-risk",
    primary_table="application_train.csv",
    # Selective extraction: the competition archive carries eight relational tables and
    # roughly 166 MB. W1 models the application table alone; the bureau/previous-application
    # tables are a feature-engineering exercise, not an MLOps one.
    archive_members=("application_train.csv",),
    fallback=SourceSpec(
        source_kind="openml",
        source_ref="42477",
        primary_table="default-of-credit-card-clients",
    ),
    # The fallback is a DIFFERENT dataset -- UCI Taiwan credit-card default, 30k rows and
    # 24 columns against Home Credit's 307k and 122. Taking it means rewriting the schema
    # and the feature spec, which the plan prices at 6-8h. It is a deadline escape hatch,
    # not a drop-in swap, and this flag is what stops it being mistaken for one.
    equivalent_to_primary=False,
)

CREDIT_SCHEMA = SchemaSpec(
    # The pandera model and its column manifest land with the ingest retarget (plan
    # Step 4). Declared here as unbuilt rather than silently omitted: get_track("credit")
    # must not read as fully wired when validation is not yet in place.
    model=None,
    manifest_path=SCHEMA_DIR / "home_credit_columns.txt",
)

CREDIT = Track(
    name="credit",
    source=CREDIT_SOURCE,
    schema=CREDIT_SCHEMA,
    features=get_feature_spec("credit"),
)

# Only the credit track is registered in W1. The fraud track arrives in W2 with its own
# data module (plan Step 8); registering it early would hand callers a Track whose source
# cannot be fetched, which fails later and less clearly than a KeyError here.
TRACKS: Mapping[str, Track] = MappingProxyType({"credit": CREDIT})


def get_track(name: str) -> Track:
    """Look up a registered track.

    Raises ``KeyError`` naming what *is* registered, so an unregistered track fails at the
    lookup with a readable message instead of surfacing as an attribute error downstream.
    """
    try:
        return TRACKS[name]
    except KeyError:
        registered = ", ".join(sorted(TRACKS)) or "none"
        raise KeyError(f"unknown track {name!r}; registered tracks: {registered}") from None


def registered_track_names() -> tuple[str, ...]:
    """Registered track names, sorted."""
    return tuple(sorted(TRACKS))
