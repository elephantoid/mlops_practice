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

# The metrics a track may be selected on. Declared here, in the module that imports nothing,
# because ``FeatureSpec`` has to validate against the set and cannot reach the scorers: those
# live in ``src/models/thresholds.py`` behind an sklearn import that must never enter the
# serving image. ``tests/test_thresholds.py`` asserts the two stay in step, so the split is
# two views of one vocabulary rather than two lists that can drift.
SELECTION_METRICS: frozenset[str] = frozenset({"roc_auc", "pr_auc"})


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
    # Column -> the magic value that means "not applicable" rather than a measurement.
    # Normalised to NaN inside the *pipeline*, which is the only place both training and
    # serving pass through. Doing it in ingest alone would leave training seeing NaN while
    # serving sent the raw sentinel -- the same applicant scoring differently by path,
    # which is precisely the training/serving skew this project has a test suite for.
    sentinels: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    #
    # There is deliberately no ``derived_features`` any more. It held exactly one entry,
    # credit's ``DAYS_EMPLOYED_ANOMALY``, and that column measured as a **99.9967% duplicate of
    # ``NAME_INCOME_TYPE``** -- `Pensioner` and `Unemployed` are the sentinel population and
    # nothing else is. LightGBM gave it 0.02% of total gain, logistic regression a coefficient
    # 1% of its largest, and cross-validated ROC-AUC was identical to five decimals with and
    # without it in both arms. A derived feature has to be justified against the features
    # already in the contract; this one never was.
    #
    # Removing it made the estimator's feature space equal the request contract -- 26 either
    # side -- which is the property that makes a training/serving skew in *column set* not
    # merely absent but unrepresentable. See ``docs/debt-ledger.md``.
    # Which number this track's models are *selected* on, out of SELECTION_METRICS. A track
    # setting rather than a module constant because ROC-AUC and PR-AUC disagree about what a
    # good model is under extreme imbalance: at a 0.001727 positive rate ROC-AUC can read
    # 0.97 while almost every positive prediction is wrong, because its false-positive rate
    # has the enormous negative class in the denominator. Average precision has no such
    # denominator and collapses toward the base rate instead.
    #
    # It defaults to ROC-AUC rather than requiring every track to state one, because that is
    # what every model in the registry was in fact selected on -- a default that matched
    # history is what let this land without rewriting the credit track's meaning.
    selection_metric: str = "roc_auc"

    def __post_init__(self) -> None:
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"unknown selection_metric {self.selection_metric!r}; "
                f"registered: {', '.join(sorted(SELECTION_METRICS))}"
            )

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

    # There is deliberately no `model_columns` property combining features and derived
    # columns. One existed and had no callers, and its docstring claimed the preprocessor
    # selected on it -- which was false. `build_preprocessor` composes the two lists at the
    # point of use instead, because that is the only place the combination is correct: the
    # request contract, the parquet, the drift frame and the API reindex must all stay on
    # `feature_columns`, and a convenient combined property is an invitation to reach for it
    # in one of those four places. That mistake produces a 27-wide model signature against a
    # 26-column contract, which is exactly the defect this design exists to prevent.

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

    # The two properties below name an MLflow metric, which looks out of place in a module
    # about feature contracts. They are here because the alternative is worse: the only other
    # candidate is ``src/models/train.py``, and ``run_experiment`` there is handed a
    # ``FeatureSpec`` rather than a track name -- so the rule would have to be written once
    # against a spec and once against a track, in the same module, for the same string. One
    # rule in the place that owns ``selection_metric`` beats two rules anywhere else. It
    # costs nothing: these are string formatting, so the no-imports invariant holds.

    @property
    def cv_metric_key(self) -> str:
        """MLflow metric key for the cross-validated selection score.

        ``cv_roc_auc_mean`` or ``cv_pr_auc_mean``. The metric is *in* the key so two tracks
        selected on different metrics have no shared field to be compared through.
        """
        return f"cv_{self.selection_metric}_mean"

    @property
    def cv_std_key(self) -> str:
        """MLflow metric key for the fold-to-fold spread of the selection score."""
        return f"cv_{self.selection_metric}_std"


# Home Credit application_train.csv. The modeled subset is deliberately narrow: the raw
# table carries 122 columns, most of them sparse normalized-credit-bureau aggregates, and
# enumerating all of them would buy schema noise rather than signal. The full 122-name set
# is asserted structurally by the column manifest in src/data/tracks.py instead.
CREDIT_FEATURES = FeatureSpec(
    id_column="SK_ID_CURR",
    target_column="TARGET",
    # Already integer 0/1 in the source; 1 means the applicant defaulted.
    positive_label=1,
    # DAYS_EMPLOYED uses 365243 (roughly a thousand years in the future) for "never
    # employed" -- about 18% of rows. Left as a number it is an extreme outlier that drags
    # any scaler and splits trees on a fiction.
    sentinels=MappingProxyType({"DAYS_EMPLOYED": 365243.0}),
    # The claim that used to sit here -- "an applicant with no employment history is a different
    # case from one with a short history" -- justified a derived ``DAYS_EMPLOYED_ANOMALY`` flag.
    # The claim is true and the flag was still redundant: that distinction is already carried by
    # ``NAME_INCOME_TYPE`` below, which the caller sends. Measured rather than argued; see the
    # note on the dataclass above.
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

# ULB credit-card transactions (creditcard.csv). 284,807 rows, 31 source columns, 492
# positives -- a 0.001727 positive rate against credit's 0.0807, roughly 47x apart. The two
# tracks share no key, no entity, no time base and no feature space; they share this
# platform, and nothing else.
#
# 29 of the 31 source columns are modeled. The two that are not are argued below.
_V_COMPONENTS: tuple[str, ...] = tuple(f"V{index}" for index in range(1, 29))

FRAUD_FEATURES = FeatureSpec(
    # Derived by src/data/fraud.py's clean() as a positional index, because the source ships
    # no identifier and ``Time`` is not one -- 160,215 of 284,807 rows share a second with
    # another row, and 1,081 rows are exact duplicates across all 31 source columns, so no
    # combination of source columns keys this frame either. The full argument, including why
    # a positional key is sound rather than a fudge here, is in that function's docstring.
    id_column="TransactionIndex",
    target_column="Class",
    # Already integer 0/1 in the source; 1 means the transaction was fraudulent.
    positive_label=1,
    # ``Time`` is deliberately NOT here, and it is the only source column dropped from the
    # model on judgement rather than because it is the id or the target. Three reasons, any
    # one of which would be enough:
    #
    # 1. No live caller can produce it. It is seconds elapsed from the first transaction of
    #    *this extract*, so every real request would carry a value beyond the entire training
    #    range (0..172,792) or an arbitrarily re-based one. A feature whose served values are
    #    guaranteed to fall outside the training support is not a feature.
    # 2. It is monotonic non-decreasing over the file, so it encodes row position. A tree
    #    handed it can learn *when in this particular 48 hours* the frauds were, which is
    #    memorisation wearing a timestamp.
    # 3. Drift. Live traffic's Time distribution differs from the snapshot's by construction,
    #    so it would register as drifted on every single run and inflate the share the
    #    retrain trigger reads -- the same defect the id column caused, measured at 0.095
    #    against 0.048.
    #
    # It is still validated at ingest (presence, and bounds that separate an elapsed offset
    # from an absolute epoch timestamp), because an upstream that stops shipping it or
    # switches encoding is a change worth failing on -- and since the column is dropped
    # rather than modeled, ingest is the ONLY place that change is visible at all. A
    # time-of-day feature derived from it -- which a caller genuinely can supply -- is real
    # feature engineering, out of scope here, and recorded in docs/debt-ledger.md.
    numeric_features=(*_V_COMPONENTS, "Amount"),
    # Zero categoricals, and that is the whole of the special handling required: sklearn's
    # ColumnTransformer skips an empty column selection internally, so build_preprocessor
    # needs no branch for it. tests/test_pipeline.py guards the case in both model arms.
    categorical_features=(),
    # No sentinels: the source has no nulls and no magic values. And therefore no derived
    # features -- there is nothing to flag.
    #
    # PR-AUC, not ROC-AUC, and this is the track the distinction was introduced for. 492
    # positives in 284,807 rows means the negative class is 578x the positive one, so a
    # model can move thousands of false positives without ROC-AUC noticing -- the ratio it
    # reports has all 284,315 negatives underneath it. Average precision measures the
    # positive class against what was actually flagged, which is the quantity a fraud
    # reviewer's queue is made of.
    selection_metric="pr_auc",
    display_names=MappingProxyType(
        {
            "Amount": "Transaction amount",
            # V1..V28 are deliberately absent, and this is the honest answer rather than a
            # gap to fill in later. They are principal components; the ULB researchers ran
            # PCA to publish the data at all and never released the loadings, so what each
            # one measures is not recoverable. display_name() falls back to the raw column,
            # so DoD (3)'s reason codes will read "V14" -- which tells a reader exactly as
            # much as is actually known. Naming it "Merchant risk score" would tell them
            # more than is known, which is worse than telling them nothing.
        }
    ),
)

# Track name -> feature contract. Two tracks from W2 Step 8; the second one is what makes
# every registry lookup, per-track path and ENABLED_TRACKS switch in this repo load-bearing
# rather than ceremonial.
FEATURE_SPECS: Mapping[str, FeatureSpec] = MappingProxyType(
    {"credit": CREDIT_FEATURES, "fraud": FRAUD_FEATURES}
)


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
