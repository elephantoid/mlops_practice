"""Request and response models for the prediction API.

The public contract is snake_case, but the trained pipeline's columns are the raw dataset
names (``MonthlyCharges``, ``Contract``, ...). ``serialization_alias`` bridges the two:
requests are validated against snake_case, and ``model_dump(by_alias=True)`` emits exactly
the column names the model was fit on.

``serialization_alias`` rather than ``alias``: a plain ``alias`` would make CamelCase the
*input* spelling and advertise CamelCase in the OpenAPI schema, which is the opposite of the
documented contract. This way the public API stays snake_case in both the schema and the
request body, and the CamelCase names never leak past this module.

The ``Literal`` value sets must stay identical to the ``isin`` checks in
``src/data/ingest.py``. A category accepted here but unseen in training is encoded as -1 by
the OrdinalEncoder, which produces a confident-looking prediction from a value the model
has never observed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Mirrors the sentinel-bearing columns in the training data: "No internet service" and
# "No phone service" are genuine categories the model was fit on, not missing values.
YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]
YesNoPhone = Literal["Yes", "No", "No phone service"]

EXAMPLE_REQUEST = {
    "senior_citizen": 0,
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "gender": "Female",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "No",
    "device_protection": "Yes",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "One year",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


class PredictRequest(BaseModel):
    """One customer's features, in the public snake_case spelling.

    ``extra="forbid"`` is deliberate: silently ignoring an unrecognised field would let a
    caller believe they were influencing the prediction when they were not.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    # Numeric. Bounds mirror the domain invariants asserted in src/data/ingest.py -- they
    # reject the impossible, not the merely unusual, so an unusually long tenure still
    # scores rather than 422-ing.
    senior_citizen: int = Field(ge=0, le=1, serialization_alias="SeniorCitizen")
    tenure: int = Field(ge=0, serialization_alias="tenure")
    monthly_charges: float = Field(ge=0, serialization_alias="MonthlyCharges")
    total_charges: float = Field(ge=0, serialization_alias="TotalCharges")

    # Categorical.
    gender: Literal["Female", "Male"] = Field(serialization_alias="gender")
    partner: YesNo = Field(serialization_alias="Partner")
    dependents: YesNo = Field(serialization_alias="Dependents")
    phone_service: YesNo = Field(serialization_alias="PhoneService")
    multiple_lines: YesNoPhone = Field(serialization_alias="MultipleLines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        serialization_alias="InternetService"
    )
    online_security: YesNoInternet = Field(serialization_alias="OnlineSecurity")
    online_backup: YesNoInternet = Field(serialization_alias="OnlineBackup")
    device_protection: YesNoInternet = Field(serialization_alias="DeviceProtection")
    tech_support: YesNoInternet = Field(serialization_alias="TechSupport")
    streaming_tv: YesNoInternet = Field(serialization_alias="StreamingTV")
    streaming_movies: YesNoInternet = Field(serialization_alias="StreamingMovies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        serialization_alias="Contract"
    )
    paperless_billing: YesNo = Field(serialization_alias="PaperlessBilling")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(serialization_alias="PaymentMethod")


class PredictResponse(BaseModel):
    """A churn probability plus the metadata needed to trace it back to a model version."""

    churn_probability: float = Field(ge=0, le=1, description="P(churn) from the model")
    prediction: Literal["churn", "no_churn"] = Field(
        description="churn_probability thresholded at 0.5"
    )
    model_version: str = Field(description="MLflow model version currently serving")
    request_id: str = Field(description="Correlates this response with the prediction log")


class HealthResponse(BaseModel):
    """Liveness plus enough detail to tell *which* model is live."""

    status: Literal["ok"] = "ok"
    model_version: str
    uptime_seconds: float
