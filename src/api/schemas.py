"""Request and response models for the prediction API.

The public contract is snake_case, but the trained pipeline's columns are the raw dataset
names (``AMT_INCOME_TOTAL``, ``NAME_CONTRACT_TYPE``, ...). ``serialization_alias`` bridges
the two: requests are validated against snake_case, and ``model_dump(by_alias=True)`` emits
exactly the column names the model was fit on.

``serialization_alias`` rather than ``alias``: a plain ``alias`` would make the raw spelling
the *input* form and advertise it in the OpenAPI schema, which is the opposite of the
documented contract. This way the public API stays snake_case in both the schema and the
request body, and the raw column names never leak past this module.

The ``Literal`` value sets must stay identical to the category checks in
``src/data/ingest.py``. A category accepted here but unseen in training is encoded as -1 by
the OrdinalEncoder, which produces a confident-looking prediction from a value the model
has never observed.

There is no Telco/churn contract left here and no compatibility shim for it. The repo's
doctrine is to delete obsolete paths rather than carry them, and this service has no
external consumers to break.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Category vocabularies too large to inline as Literals, extracted from the training
# archive rather than typed by hand. Loaded once at import: the file is a few KB, and a
# per-request read would put disk I/O on the serving path for a value that cannot change
# without a retrain.
_CATEGORIES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "schemas" / "credit_categories.json"
)
try:
    CREDIT_CATEGORIES: dict[str, list[str]] = json.loads(_CATEGORIES_PATH.read_text())
except FileNotFoundError:
    # Absent in a fresh clone, where the archive has not been fetched. Membership checks
    # then pass through rather than rejecting everything -- a missing manifest must not
    # turn every request into a 422.
    CREDIT_CATEGORIES = {}

# Field name -> the raw column its vocabulary is keyed by.
SERIALIZATION_ALIASES = {
    "occupation_type": "OCCUPATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
}

# The three-valued decision is the whole point of the contract change. A binary
# approve/decline cannot express the review band, and the review band is where the
# cost-asymmetry optimisation actually lands: the cost of wrongly approving a default and
# the cost of wrongly rejecting a good applicant are not symmetric, so the operating point
# is two thresholds, not one. Real credit and fraud systems route the middle to a human.
Decision = Literal["approve", "review", "decline"]

TrackName = Literal["credit", "fraud"]

CREDIT_EXAMPLE_REQUEST = {
    "amt_income_total": 202500.0,
    "amt_credit": 406597.5,
    "amt_annuity": 24700.5,
    "amt_goods_price": 351000.0,
    "days_birth": -9461,
    "days_employed": -637,
    "days_registration": -3648,
    "days_id_publish": -2120,
    "cnt_children": 0,
    "cnt_fam_members": 1,
    "region_population_relative": 0.018801,
    "ext_source_1": 0.0830,
    "ext_source_2": 0.2629,
    "ext_source_3": 0.1393,
    "hour_appr_process_start": 10,
    "name_contract_type": "Cash loans",
    "code_gender": "M",
    "flag_own_car": "N",
    "flag_own_realty": "Y",
    "name_income_type": "Working",
    "name_education_type": "Secondary / secondary special",
    "name_family_status": "Single / not married",
    "name_housing_type": "House / apartment",
    "occupation_type": "Laborers",
    "organization_type": "Business Entity Type 3",
    "weekday_appr_process_start": "WEDNESDAY",
}


class CreditPredictRequest(BaseModel):
    """One credit application, in the public snake_case spelling.

    ``extra="forbid"`` is deliberate: silently ignoring an unrecognised field would let a
    caller believe they were influencing the prediction when they were not.

    The ``days_*`` fields are negative day counts measured backwards from the application
    date -- that is how Home Credit ships them, and converting at the boundary would mean
    the API and the model disagreed about what the number means. ``le=0`` encodes the sign
    convention so a caller passing a positive age gets a 422 rather than a confident
    prediction from a value the model has never seen.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [CREDIT_EXAMPLE_REQUEST]}
    )

    # Numeric. Bounds reject the impossible, not the merely unusual, so an unusually large
    # loan still scores rather than 422-ing.
    amt_income_total: float = Field(gt=0, serialization_alias="AMT_INCOME_TOTAL")
    amt_credit: float = Field(gt=0, serialization_alias="AMT_CREDIT")
    amt_annuity: float = Field(ge=0, serialization_alias="AMT_ANNUITY")
    amt_goods_price: float = Field(ge=0, serialization_alias="AMT_GOODS_PRICE")
    days_birth: int = Field(le=0, serialization_alias="DAYS_BIRTH")
    # Not bounded by le=0: the source encodes "never employed" as the sentinel 365243,
    # about 18% of rows. Rejecting positives here would 422 exactly the population that
    # sentinel describes.
    #
    # It is normalised to NaN by the `sentinels` step inside the fitted pipeline -- not by
    # ingest. That placement is deliberate: the pipeline is the only thing both training
    # and serving pass through, so a request carrying the raw sentinel and a training row
    # carrying it are treated identically. Normalising in ingest alone would have made the
    # same applicant score differently depending on which path it arrived by.
    days_employed: int = Field(serialization_alias="DAYS_EMPLOYED")
    days_registration: float = Field(le=0, serialization_alias="DAYS_REGISTRATION")
    days_id_publish: int = Field(le=0, serialization_alias="DAYS_ID_PUBLISH")
    cnt_children: int = Field(ge=0, serialization_alias="CNT_CHILDREN")
    cnt_fam_members: float = Field(ge=0, serialization_alias="CNT_FAM_MEMBERS")
    region_population_relative: float = Field(
        ge=0, le=1, serialization_alias="REGION_POPULATION_RELATIVE"
    )
    # The EXT_SOURCE_* columns are normalised external credit scores and are the strongest
    # single predictors in this dataset. They are also frequently missing in the source.
    ext_source_1: float | None = Field(default=None, ge=0, le=1, serialization_alias="EXT_SOURCE_1")
    ext_source_2: float | None = Field(default=None, ge=0, le=1, serialization_alias="EXT_SOURCE_2")
    ext_source_3: float | None = Field(default=None, ge=0, le=1, serialization_alias="EXT_SOURCE_3")
    hour_appr_process_start: int = Field(ge=0, le=23, serialization_alias="HOUR_APPR_PROCESS_START")

    # Categorical.
    name_contract_type: Literal["Cash loans", "Revolving loans"] = Field(
        serialization_alias="NAME_CONTRACT_TYPE"
    )
    # "XNA" appears on 4 rows of 307,511 in the source. It is kept rather than rejected so
    # the serving contract matches what the model was actually fit on.
    code_gender: Literal["M", "F", "XNA"] = Field(serialization_alias="CODE_GENDER")
    flag_own_car: Literal["Y", "N"] = Field(serialization_alias="FLAG_OWN_CAR")
    flag_own_realty: Literal["Y", "N"] = Field(serialization_alias="FLAG_OWN_REALTY")
    name_income_type: Literal[
        "Businessman",
        "Commercial associate",
        "Maternity leave",
        "Pensioner",
        "State servant",
        "Student",
        "Unemployed",
        "Working",
    ] = Field(serialization_alias="NAME_INCOME_TYPE")
    name_education_type: Literal[
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ] = Field(serialization_alias="NAME_EDUCATION_TYPE")
    name_family_status: Literal[
        "Civil marriage",
        "Married",
        "Separated",
        "Single / not married",
        "Unknown",
        "Widow",
    ] = Field(serialization_alias="NAME_FAMILY_STATUS")
    name_housing_type: Literal[
        "Co-op apartment",
        "House / apartment",
        "Municipal apartment",
        "Office apartment",
        "Rented apartment",
        "With parents",
    ] = Field(serialization_alias="NAME_HOUSING_TYPE")
    # Validated against the committed vocabulary rather than a Literal: 18 and 58 members
    # respectively, which inline would bury the rest of the model. Membership is enforced
    # by the validator below -- leaving them bare `str` meant an unseen value reached the
    # OrdinalEncoder, which encodes it as -1 and scores it anyway, so a typo produced a
    # confident answer about a category the model has never observed.
    occupation_type: str | None = Field(default=None, serialization_alias="OCCUPATION_TYPE")
    organization_type: str = Field(serialization_alias="ORGANIZATION_TYPE")

    @field_validator("occupation_type", "organization_type")
    @classmethod
    def _known_category(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Reject a category the training data never contained.

        ``None`` passes for the nullable field: absence is a real state in the source
        (roughly a third of rows have no occupation) and the pipeline imputes it.
        """
        if value is None:
            return value
        allowed = CREDIT_CATEGORIES.get(SERIALIZATION_ALIASES[info.field_name])
        if allowed and value not in allowed:
            raise ValueError(
                f"{value!r} is not a category seen in training; "
                f"expected one of {len(allowed)} known values"
            )
        return value

    weekday_appr_process_start: Literal[
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ] = Field(serialization_alias="WEEKDAY_APPR_PROCESS_START")


class ReasonCode(BaseModel):
    """One signed contribution to a score, named in words a human can act on.

    ``feature`` is the display name, not the raw column: "Years employed" rather than
    ``DAYS_EMPLOYED``. A reason code a reviewer cannot read is not a reason code, and the
    regulatory case for this field is that a declined applicant is owed an explanation.
    """

    feature: str = Field(description="Human-readable feature name")
    contribution: float = Field(description="Signed SHAP contribution; positive raises risk")
    value: str | None = Field(default=None, description="The applicant's value, as displayed")


class RiskResponse(BaseModel):
    """A risk probability, the decision it implies, and the reasons behind it."""

    risk_probability: float = Field(ge=0, le=1, description="P(risk event) from the model")
    decision: Decision = Field(
        description="approve / review / decline, from the track's two operating points"
    )
    # Ships EMPTY in W1 and is asserted empty by a test. This is the one field that exists
    # ahead of its capability, and it is here deliberately: SHAP arrives in W2, and adding
    # the field then would be a second breaking change to a contract that has no shims. An
    # empty list that is *tested* as empty cannot be mistaken for a delivered feature.
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Top contributions to this score. Empty until SHAP lands (W2).",
    )
    threshold: float = Field(
        ge=0, le=1, description="Decline threshold the decision was taken against"
    )
    track: TrackName = Field(description="Which risk model scored this request")
    model_version: str = Field(description="MLflow model version currently serving")
    request_id: str = Field(description="Correlates this response with the prediction log")


class HealthResponse(BaseModel):
    """Liveness plus enough detail to tell *which* models are live.

    ``degraded`` exists because the two tracks are independent by design. With a single
    ``Literal["ok"]`` the only way to report a half-loaded service was to fail the whole
    process, which would mean a fraud-side problem taking the credit endpoint down with it
    -- and on Cloud Run, failing the whole revision. A three-state field lets one track be
    down and be *seen* to be down.
    """

    status: Literal["ok", "degraded"] = "ok"
    models: dict[str, str] = Field(
        default_factory=dict,
        description="Track name -> loaded model version, for each track that loaded",
    )
    uptime_seconds: float
