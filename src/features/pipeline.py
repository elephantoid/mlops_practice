"""Preprocessing + model pipelines for risk scoring.

Everything a model needs to go from a raw feature row to a probability lives inside the
:class:`~sklearn.pipeline.Pipeline` built here. That is deliberate: ``train.py`` serializes
the whole object with ``mlflow.sklearn.log_model`` and the FastAPI service reloads it with
``mlflow.pyfunc.load_model``. Any preprocessing left outside the pipeline would have to be
reimplemented in the API, which is how training/serving skew gets in.

Preprocessing is per-model rather than shared. LightGBM splits on ordinal codes without
caring about their spacing, so it takes an OrdinalEncoder and no scaler. LogisticRegression
reads those same codes as magnitudes -- it would infer that ``Working`` is twice
``Pensioner`` -- so it gets one-hot encoding and standardized numerics instead. Sharing
one transformer would handicap the baseline for encoding reasons rather than model reasons
and make the MLflow comparison meaningless.

Every function here takes a :class:`~src.features.specs.FeatureSpec` rather than reading
module-level column constants. The constants are gone deliberately: they were the Telco
domain baked into the shared feature path, and hardcoding one domain there is what would
have forced the fraud track to arrive as ``if track ==`` branches spread across six
modules.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.features.specs import FeatureSpec

ModelType = Literal["lightgbm", "logreg"]

RANDOM_STATE = 42


def split_features_target(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Split an ingested frame into the feature matrix and the binary target.

    The id column is dropped rather than passed through: it is unique per row, so a tree
    model handed it would happily memorize the training set.

    Returns ``(X, y)`` where ``y`` is 1 for the risk event described by
    ``spec.positive_label`` -- default on a credit application, fraud on a transaction --
    and 0 otherwise.
    """
    missing = set(spec.feature_columns) | {spec.target_column}
    missing -= set(df.columns)
    if missing:
        raise KeyError(f"Input frame is missing expected columns: {sorted(missing)}")

    # This selection is the model's input contract, and it is the same list the API reindexes
    # a request onto. Whatever columns reach fit() become the signature MLflow enforces at
    # serving time, so anything selected here that a caller cannot send fails every request --
    # a 27-wide signature against a 26-column contract is exactly what shipped once, and
    # training passed while serving 500'd because each half looked correct alone.
    #
    # It is a positional selection rather than a filter, so a frame carrying extra columns --
    # the parquet's id, or anything a future ``clean()`` adds -- cannot widen the signature by
    # accident. `tests/test_pipeline.py` asserts the equality; `tests/test_skew.py` is what
    # catches it against a real registered model, and it has caught it twice.
    features = df[list(spec.feature_columns)].copy()

    raw_target = df[spec.target_column]
    if raw_target.isna().any():
        null_count = int(raw_target.isna().sum())
        raise ValueError(
            f"{spec.target_column!r} contains {null_count} null value(s); "
            "a null target is silently dropped by most estimators, so it is rejected here"
        )

    # Equality against positive_label rather than a hardcoded {"Yes": 1, "No": 0} map.
    # The Telco version raised on anything outside that map, and both riskwatch tracks
    # ship targets that are ALREADY integer 0/1 -- so that raise was the first thing the
    # credit track would have hit, in W1, on the very first train call.
    target = (raw_target == spec.positive_label).astype(int)

    # A target that is constant after encoding cannot train a classifier, and the error
    # sklearn raises for it names neither the column nor the value that collapsed it.
    # The likeliest cause is a positive_label that does not appear in the data at all.
    if target.nunique() < 2:
        observed = sorted(raw_target.unique(), key=repr)[:10]
        raise ValueError(
            f"{spec.target_column!r} collapsed to a single class against "
            f"positive_label={spec.positive_label!r}; observed values: {observed}"
        )

    return features, target


def _replace_sentinels(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Turn each column's "not applicable" magic number into NaN. Nothing else.

    Inside the pipeline on purpose. Both ``train.py`` and the serving path build their
    frames through here, so this is the one place a normalisation reaches both -- doing it
    in ingest would leave training seeing NaN while a request carried the raw sentinel,
    and the same applicant would score differently depending on how it arrived.

    The imputer downstream then fills these the same way it fills genuine nulls.

    **It used to do three jobs under this name** -- derive ``<column>_ANOMALY`` flags from the
    raw sentinel, replace the sentinel, and backfill any derived column the first step could
    not produce with zeros. The flag it derived for credit turned out to be a 99.9967%
    duplicate of ``NAME_INCOME_TYPE`` and contributed nothing measurable, so it is gone; with
    it went the only reason for the other two jobs. The width of this step's output now
    equals the width of its input, which is why ``build_pipeline`` no longer has to teach
    sklearn otherwise.

    It delegates the column-order question to :func:`_canonicalise_columns`, which is not
    tidiness: ``feature_names_out="one-to-one"`` makes a reordered frame silently mislabelled,
    and the flag's removal is what exposed that. See that function.

    The sentinel column is indexed without an ``in out.columns`` guard on purpose. The guard
    that used to be here skipped a spec naming a column the frame does not have, which meant a
    mistyped ``sentinels`` key normalised nothing and logged nothing -- the same silent-no-op
    shape as the derived-flag naming bug this change removed. ``FeatureSpec.__post_init__``
    now rejects that spec outright, so by the time a frame gets here the column is guaranteed
    present by :func:`_canonicalise_columns` above.
    """
    out = _canonicalise_columns(frame, spec)
    if not spec.sentinels:
        return out

    out = out.copy()
    for column, sentinel in spec.sentinels.items():
        out[column] = out[column].replace(sentinel, np.nan)
    return out


def _canonicalise_columns(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Put the frame in ``spec.feature_columns`` order, or refuse it.

    **This exists because ``feature_names_out="one-to-one"`` makes a reordered frame silent.**
    sklearn only raises when the function *renames* its columns. When the names merely arrive
    in a different order it takes the rename branch instead: it stamps the fitted order's
    labels onto arrival-order data and **does not move the values**. Measured on a three-column
    fixture, ``AMT_CREDIT``'s value came back under the ``DAYS_EMPLOYED`` label, the
    ColumnTransformer's own name check then passed because the names matched fit exactly, and
    the estimator returned a plausible probability from scrambled features.

    Before the derived flag was removed, the step widened its input and needed a custom
    ``feature_names_out`` callable -- which made that same frame raise a length mismatch. So
    dropping the flag silently removed a guard, and this restores it as a property of the step
    rather than of one caller: ``src/api/main.py`` pins order with ``reindex`` and pyfunc
    reorders to the signature, but a direct sklearn caller had nothing.

    A wrong column *set* raises; a wrong *order* is corrected. Reindexing would have done both
    at once and filled a genuinely missing column with NaN -- trading a loud failure for a
    quiet one, which is the trade this whole change exists to stop making.
    """
    expected = list(spec.feature_columns)
    actual = list(frame.columns)
    if actual == expected:
        return frame

    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing or extra:
        raise ValueError(
            f"frame does not carry this track's feature columns: missing {missing}, "
            f"unexpected {extra}. The pipeline will not guess which column is which."
        )
    return frame[expected]


def build_preprocessor(model_type: ModelType, spec: FeatureSpec) -> ColumnTransformer:
    """Build the ColumnTransformer appropriate to ``model_type`` and ``spec``.

    The imputers are defensive rather than load-bearing -- ``ingest.py`` guarantees zero
    nulls today, but a future batch may not, and silently failing at serving time is worse
    than imputing. Both encoders tolerate unseen categories so an unfamiliar value in a
    request degrades gracefully instead of raising inside the API.

    A spec with **no categorical features** needs no special handling here: sklearn's
    ColumnTransformer already skips empty column selections internally. That matters for
    the fraud track, whose features are 30 anonymized PCA components with no categoricals
    at all. It was briefly flagged in review as the known W2 breakage and then withdrawn
    on inspection -- ``tests/test_pipeline.py`` keeps a guard on it for documentation
    value, not because a fix was needed.
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
            ("num", numeric, list(spec.numeric_features)),
            ("cat", categorical, list(spec.categorical_features)),
        ],
        remainder="drop",
    )


def build_pipeline(
    model_type: ModelType = "lightgbm",
    spec: FeatureSpec | None = None,
    **model_params: object,
) -> Pipeline:
    """Build an unfitted preprocessing + classifier pipeline.

    ``model_params`` is forwarded to the estimator and overrides the defaults set here, which
    is how ``train.py`` sweeps ``num_leaves`` / ``learning_rate`` / ``class_weight`` across
    MLflow runs without this module knowing anything about the search space.

    ``random_state`` defaults to :data:`RANDOM_STATE` so the same data and params always
    produce the same artifact, but a caller may override it to measure seed variance.

    ``spec`` is keyword-optional only so ``model_params`` can stay the trailing
    ``**kwargs`` the sweep relies on; it is required in practice and raises if omitted.
    """
    if spec is None:
        raise TypeError("build_pipeline requires a FeatureSpec; pass spec=get_feature_spec(...)")

    preprocessor = build_preprocessor(model_type, spec)

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

    # The sentinel step runs first, so the imputer downstream treats a "never employed"
    # marker exactly like a genuine null. It is part of the serialized artifact, which is
    # what makes it apply identically at training and at serving time -- the property the
    # skew test exists to protect.
    return Pipeline(
        [
            (
                "sentinels",
                FunctionTransformer(
                    _replace_sentinels,
                    kw_args={"spec": spec},
                    # One-to-one: the step rewrites values in place and adds no columns, so the
                    # estimator sees exactly the request contract's features. It used to add a
                    # derived flag and needed a custom name callable to tell sklearn its output
                    # was wider than its input.
                    feature_names_out="one-to-one",
                ),
            ),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
