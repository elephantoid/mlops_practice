"""Per-track feature contracts.

This module is deliberately dependency-light. It imports nothing beyond the standard
library, and in particular it does **not** import pandera.

That is the whole reason it exists as a separate module from :mod:`src.data.tracks`.
Python imports at module granularity: if the pandera-bearing ``SchemaSpec`` and the
feature lists lived in one module, then ``import src.data.tracks`` would execute
``import pandera``, and because ``src/api/main.py`` needs the feature lists to pin its
column order, pandera would be pulled into the serving image. The image already contends
with a 0.5 GB Artifact Registry budget, so composition alone does not solve this --
only placement does.

The dependency direction is therefore **data -> features, never the reverse**:
``src/data/tracks.py`` imports from here, and ``src/api/main.py`` imports from here.
Nothing in this module may ever import from ``src.data``. ``tests/test_tracks.py``
asserts the resulting invariant directly rather than trusting the convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class FeatureSpec:
    """What a model for one track consumes, and how to say it in a human sentence.

    ``positive_label`` is the value of the target column that means "the risk event
    happened" -- default on a credit application, fraud on a card transaction. It exists
    because the Telco pipeline hardcoded a ``{"Yes": 1, "No": 0}`` mapping that raised on
    anything else, and both riskwatch tracks ship targets that are **already** integer
    0/1. That raise is the first thing the credit track would have hit.

    ``display_names`` backs DoD (3)'s reason codes. A SHAP contribution reported against
    the raw column ``DAYS_EMPLOYED`` is not a reason a human can act on; the same
    contribution reported as "Years employed" is. Columns absent from the mapping fall
    back to their raw name, so the map can grow incrementally without breaking callers.
    """

    id_column: str
    target_column: str
    positive_label: object
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    display_names: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.numeric_features and not self.categorical_features:
            raise ValueError(
                f"FeatureSpec for target {self.target_column!r} has no features at all"
            )

        overlap = set(self.numeric_features) & set(self.categorical_features)
        if overlap:
            raise ValueError(f"columns claimed as both numeric and categorical: {sorted(overlap)}")

        for column in (self.id_column, self.target_column):
            if column in self.numeric_features or column in self.categorical_features:
                raise ValueError(f"{column!r} is a non-feature column but is listed as a feature")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Model input columns in a pinned order.

        Numerics first, then categoricals. Order is pinned rather than derived from a set
        or a dict so that adding a field can never silently permute the model's inputs --
        the serving path reindexes onto this exact sequence.
        """
        return tuple(self.numeric_features) + tuple(self.categorical_features)

    @property
    def non_feature_columns(self) -> tuple[str, ...]:
        """Columns to drop before drift comparison.

        ``src/monitoring/drift.py`` needs this. Leaving a unique identifier in the frame
        makes it register as drifted on every run and inflates the drift share -- the
        Telco module measured 0.095 with the id column against 0.048 without it, and the
        retrain trigger rests on exactly that number.
        """
        return (self.id_column, self.target_column)

    def display_name(self, column: str) -> str:
        """Human-facing name for ``column``, falling back to the raw column name."""
        return self.display_names.get(column, column)


# Home Credit application_train.csv. The modeled subset is deliberately narrow: the raw
# table carries 122 columns, most of them sparse normalized-credit-bureau aggregates, and
# enumerating all of them would buy schema noise rather than signal. The full 122-name set
# is asserted structurally by the column manifest in src/data/tracks.py instead.
CREDIT_FEATURES = FeatureSpec(
    id_column="SK_ID_CURR",
    target_column="TARGET",
    # Already integer 0/1 in the source; 1 means the applicant defaulted.
    positive_label=1,
    numeric_features=(
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "HOUR_APPR_PROCESS_START",
    ),
    categorical_features=(
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
    ),
    display_names=MappingProxyType(
        {
            "AMT_INCOME_TOTAL": "Annual income",
            "AMT_CREDIT": "Credit amount",
            "AMT_ANNUITY": "Loan annuity",
            "AMT_GOODS_PRICE": "Goods price",
            # The DAYS_* columns are negative day counts measured backwards from the
            # application date. Reporting "-15238" as a reason code is useless; the
            # display layer converts to years at the point of presentation.
            "DAYS_BIRTH": "Age (years)",
            "DAYS_EMPLOYED": "Years employed",
            "DAYS_REGISTRATION": "Years since registration",
            "DAYS_ID_PUBLISH": "Years since ID issued",
            "CNT_CHILDREN": "Number of children",
            "CNT_FAM_MEMBERS": "Family size",
            "REGION_POPULATION_RELATIVE": "Region population density",
            "EXT_SOURCE_1": "External credit score 1",
            "EXT_SOURCE_2": "External credit score 2",
            "EXT_SOURCE_3": "External credit score 3",
            "HOUR_APPR_PROCESS_START": "Application hour",
            "NAME_CONTRACT_TYPE": "Contract type",
            "CODE_GENDER": "Gender",
            "FLAG_OWN_CAR": "Owns a car",
            "FLAG_OWN_REALTY": "Owns property",
            "NAME_INCOME_TYPE": "Income type",
            "NAME_EDUCATION_TYPE": "Education",
            "NAME_FAMILY_STATUS": "Family status",
            "NAME_HOUSING_TYPE": "Housing",
            "OCCUPATION_TYPE": "Occupation",
            "ORGANIZATION_TYPE": "Employer industry",
            "WEEKDAY_APPR_PROCESS_START": "Application weekday",
        }
    ),
)

# Track name -> feature contract. The fraud track lands in W2 (plan Step 8); registering
# it here before its data module exists would let get_track("fraud") return a Track whose
# source cannot be fetched, which is a worse failure than a KeyError.
FEATURE_SPECS: Mapping[str, FeatureSpec] = MappingProxyType({"credit": CREDIT_FEATURES})


def get_feature_spec(track_name: str) -> FeatureSpec:
    """Look up a track's feature contract.

    Raises ``KeyError`` naming the registered tracks, rather than returning ``None`` for
    a caller to trip over later.
    """
    try:
        return FEATURE_SPECS[track_name]
    except KeyError:
        registered = ", ".join(sorted(FEATURE_SPECS)) or "none"
        raise KeyError(f"unknown track {track_name!r}; registered tracks: {registered}") from None
