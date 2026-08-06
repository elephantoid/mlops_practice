"""Preprocessing + model pipelines for churn prediction.

Everything a model needs to go from a raw feature row to a probability lives inside the
:class:`~sklearn.pipeline.Pipeline` built here. That is deliberate: ``train.py`` serializes
the whole object with ``mlflow.sklearn.log_model`` and the FastAPI service reloads it with
``mlflow.pyfunc.load_model``. Any preprocessing left outside the pipeline would have to be
reimplemented in the API, which is how training/serving skew gets in.

Preprocessing is per-model rather than shared. LightGBM splits on ordinal codes without
caring about their spacing, so it takes an OrdinalEncoder and no scaler. LogisticRegression
reads those same codes as magnitudes -- it would infer that ``Electronic check`` is twice
``Mailed check`` -- so it gets one-hot encoding and standardized numerics instead. Sharing
one transformer would handicap the baseline for encoding reasons rather than model reasons
and make the MLflow comparison meaningless.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ModelType = Literal["lightgbm", "logreg"]

RANDOM_STATE = 42

ID_COLUMN = "customerID"
TARGET = "Churn"

# SeniorCitizen is already a 0/1 flag, so it rides along with the numerics -- one-hotting a
# binary indicator would only add a perfectly collinear column.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an ingested frame into the feature matrix and the binary target.

    ``customerID`` is dropped rather than passed through: it is unique per row, so a tree
    model handed that column would happily memorize the training set.

    Returns ``(X, y)`` where ``y`` is 1 for churned customers and 0 otherwise.
    """
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise KeyError(f"Input frame is missing expected columns: {sorted(missing)}")

    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    # Check the mapping rather than letting .astype(int) fail on the resulting NaNs: that
    # raises IntCastingNaNError, which names neither the column nor the offending value.
    # An already-encoded 0/1 target is the likeliest way to land here.
    mapped = df[TARGET].map({"Yes": 1, "No": 0})
    if mapped.isna().any():
        unmappable = sorted(df.loc[mapped.isna(), TARGET].unique(), key=repr)
        raise ValueError(f"{TARGET} must be 'Yes'/'No'; got unmappable values: {unmappable}")

    return features, mapped.astype(int)


def build_preprocessor(model_type: ModelType) -> ColumnTransformer:
    """Build the ColumnTransformer appropriate to ``model_type``.

    The imputers are defensive rather than load-bearing -- ``ingest.py`` guarantees zero
    nulls today, but a future batch may not, and silently failing at serving time is worse
    than imputing. Both encoders tolerate unseen categories so an unfamiliar value in a
    request degrades gracefully instead of raising inside the API.
    """
    if model_type == "lightgbm":
        numeric = SimpleImputer(strategy="median")
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
    elif model_type == "logreg":
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}; expected 'lightgbm' or 'logreg'")

    # Default numpy output is intentional. With set_output(transform="pandas"), one-hot
    # names such as "PaymentMethod_Bank transfer (automatic)" trip LightGBM's rejection of
    # JSON-special characters in feature names.
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model_type: ModelType = "lightgbm", **model_params: object) -> Pipeline:
    """Build an unfitted preprocessing + classifier pipeline.

    ``model_params`` is forwarded to the estimator and overrides the defaults set here, which
    is how ``train.py`` sweeps ``num_leaves`` / ``learning_rate`` / ``class_weight`` across
    MLflow runs without this module knowing anything about the search space.

    ``random_state`` defaults to :data:`RANDOM_STATE` so the same data and params always
    produce the same artifact, but a caller may override it to measure seed variance.
    """
    preprocessor = build_preprocessor(model_type)

    # Merge rather than pass alongside: `LGBMClassifier(random_state=..., **model_params)`
    # raises TypeError on a duplicate key, so a sweep that varies the seed or raises
    # max_iter would crash instead of taking the value.
    if model_type == "lightgbm":
        # verbose=-1 reaches the C++ booster through **kwargs; it is the only thing that
        # silences LightGBM's stdout chatter, which Python's warnings filters cannot touch.
        defaults = {"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        classifier = LGBMClassifier(**(defaults | model_params))
    else:
        defaults = {"max_iter": 1000, "random_state": RANDOM_STATE}
        classifier = LogisticRegression(**(defaults | model_params))

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
