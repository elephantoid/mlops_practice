"""Tests for the retargeted feature pipeline.

These cover the two plan Step 5 acceptance criteria that were otherwise unverified:
an already-encoded integer target must pass, an unmappable one must raise naming the
column and the offending values, and a zero-categorical track must fit without raising.

``tests/test_tracks.py`` asserts that the credit spec *declares* ``positive_label=1``.
That is a different claim from the pipeline actually honouring it -- a spec can be right
while the code ignores it -- so these exercise the functions themselves.

Hermetic: small in-memory frames, no ``data/``, no registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_pipeline, build_preprocessor, split_features_target
from src.features.specs import FeatureSpec

# A track with categoricals, shaped like credit but tiny.
MIXED = FeatureSpec(
    id_column="SK_ID_CURR",
    target_column="TARGET",
    positive_label=1,
    numeric_features=("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    categorical_features=("NAME_CONTRACT_TYPE",),
)

# A track with no categoricals at all -- the fraud shape: 30 anonymized PCA components
# plus Time and Amount, every one of them numeric.
NUMERIC_ONLY = FeatureSpec(
    id_column="row_id",
    target_column="Class",
    positive_label=1,
    numeric_features=("V1", "V2", "Amount"),
    categorical_features=(),
)


def _mixed_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(rows),
            "TARGET": [i % 2 for i in range(rows)],
            "AMT_CREDIT": rng.normal(400000, 5000, rows),
            "AMT_INCOME_TOTAL": rng.normal(200000, 3000, rows),
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"] * (rows // 2),
        }
    )


def _numeric_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "row_id": range(rows),
            "Class": [i % 2 for i in range(rows)],
            "V1": rng.normal(0, 1, rows),
            "V2": rng.normal(0, 1, rows),
            "Amount": rng.gamma(2, 20, rows),
        }
    )


def test_already_encoded_integer_target_passes():
    """The Telco map raised on anything outside {"Yes", "No"}.

    Both riskwatch tracks ship integer 0/1 targets, so that raise was the first thing the
    credit track would have hit -- in W1, on the first train call, not as a W2 surprise.
    """
    features, target = split_features_target(_mixed_frame(), MIXED)

    assert set(target.unique()) == {0, 1}
    assert target.dtype.kind in "iu"
    assert list(features.columns) == list(MIXED.feature_columns)
    # The id and the label must not reach the model.
    assert "SK_ID_CURR" not in features.columns
    assert "TARGET" not in features.columns


def test_string_target_works_when_positive_label_matches():
    """positive_label is a contract, not an integer assumption.

    A track whose label ships as a string is handled by declaring that string, rather than
    by special-casing it in the shared pipeline.
    """
    frame = _mixed_frame()
    frame["TARGET"] = ["default" if v else "repaid" for v in frame["TARGET"]]
    spec = FeatureSpec(
        id_column=MIXED.id_column,
        target_column=MIXED.target_column,
        positive_label="default",
        numeric_features=MIXED.numeric_features,
        categorical_features=MIXED.categorical_features,
    )

    _, target = split_features_target(frame, spec)

    assert set(target.unique()) == {0, 1}
    assert target.sum() == (frame["TARGET"] == "default").sum()


def test_target_that_collapses_to_one_class_raises_naming_the_label_and_values():
    """A constant target cannot train a classifier.

    sklearn's own error for this names neither the column nor the value that collapsed it,
    and the likeliest cause -- a positive_label that appears nowhere in the data -- is
    exactly what the message needs to surface.
    """
    frame = _mixed_frame()
    spec = FeatureSpec(
        id_column=MIXED.id_column,
        target_column=MIXED.target_column,
        positive_label=99,  # appears nowhere
        numeric_features=MIXED.numeric_features,
        categorical_features=MIXED.categorical_features,
    )

    with pytest.raises(ValueError) as excinfo:
        split_features_target(frame, spec)

    message = str(excinfo.value)
    assert "TARGET" in message, "the error must name the column"
    assert "99" in message, "the error must name the positive_label that matched nothing"
    assert "observed values" in message, "the error must show what was actually there"


def test_null_target_is_rejected_rather_than_silently_dropped():
    """A null label is dropped by most estimators without comment, shrinking the training
    set invisibly. Better to fail with a count."""
    frame = _mixed_frame()
    frame.loc[0, "TARGET"] = None

    with pytest.raises(ValueError, match="null value"):
        split_features_target(frame, MIXED)


def test_missing_feature_column_names_what_is_missing():
    frame = _mixed_frame().drop(columns=["AMT_CREDIT"])

    with pytest.raises(KeyError, match="AMT_CREDIT"):
        split_features_target(frame, MIXED)


def test_zero_categorical_track_fits_without_raising():
    """The fraud shape: all-numeric features, no categoricals.

    This was raised in review as the known W2 breakage and then withdrawn -- sklearn's
    ColumnTransformer already skips empty column selections. The guard stays because the
    claim is worth holding to evidence rather than to a recollection of a review thread,
    and because it is the one assumption the seam's design rests on.
    """
    frame = _numeric_frame()
    features, target = split_features_target(frame, NUMERIC_ONLY)

    pipeline = build_pipeline("lightgbm", spec=NUMERIC_ONLY, n_estimators=5, verbose=-1)
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)
    assert probabilities.shape == (len(frame), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_zero_categorical_track_fits_under_logreg_too():
    """The one-hot arm has a different empty-selection path from the ordinal arm."""
    frame = _numeric_frame()
    features, target = split_features_target(frame, NUMERIC_ONLY)

    pipeline = build_pipeline("logreg", spec=NUMERIC_ONLY)
    pipeline.fit(features, target)

    assert pipeline.predict_proba(features).shape == (len(frame), 2)


def test_mixed_track_fits_and_preprocessor_covers_every_feature():
    frame = _mixed_frame()
    features, target = split_features_target(frame, MIXED)

    pipeline = build_pipeline("lightgbm", spec=MIXED, n_estimators=5, verbose=-1)
    pipeline.fit(features, target)

    transformed = pipeline.named_steps["preprocessor"].transform(features)
    assert transformed.shape[0] == len(frame)
    # Ordinal encoding is 1:1, so the column count is preserved for the lightgbm arm.
    assert transformed.shape[1] == len(MIXED.feature_columns)


def test_build_pipeline_requires_a_spec():
    """spec is keyword-optional only so **model_params can stay trailing.

    Defaulting it to a hardcoded track would reintroduce exactly the module-level domain
    constant the seam removed.
    """
    with pytest.raises(TypeError, match="requires a FeatureSpec"):
        build_pipeline("lightgbm")


def test_preprocessor_column_selections_come_from_the_spec():
    """A preprocessor built for one track must not carry another's columns."""
    mixed = build_preprocessor("lightgbm", MIXED)
    numeric_only = build_preprocessor("lightgbm", NUMERIC_ONLY)

    mixed_columns = {c for _, _, cols in mixed.transformers for c in cols}
    numeric_columns = {c for _, _, cols in numeric_only.transformers for c in cols}

    assert mixed_columns == set(MIXED.feature_columns)
    assert numeric_columns == set(NUMERIC_ONLY.feature_columns)
    assert not (mixed_columns & numeric_columns), "tracks must not share feature columns"


def test_unknown_category_at_transform_time_does_not_raise():
    """An unfamiliar value in a request must degrade, not 500 inside the API."""
    frame = _mixed_frame()
    features, target = split_features_target(frame, MIXED)
    pipeline = build_pipeline("lightgbm", spec=MIXED, n_estimators=5, verbose=-1)
    pipeline.fit(features, target)

    unseen = features.iloc[[0]].copy()
    unseen["NAME_CONTRACT_TYPE"] = "Barter"

    assert pipeline.predict_proba(unseen).shape == (1, 2)


SENTINEL_SPEC = FeatureSpec(
    id_column="SK_ID_CURR",
    target_column="TARGET",
    positive_label=1,
    numeric_features=("DAYS_EMPLOYED", "AMT_CREDIT"),
    categorical_features=(),
    sentinels={"DAYS_EMPLOYED": 365243.0},
)


def test_sentinel_is_normalised_inside_the_pipeline():
    """The sentinel must be handled where both training and serving pass through.

    365243 means "never employed" -- roughly 18% of Home Credit rows -- not a thousand
    years of future employment. Left as a number it is an extreme outlier that drags any
    scaler and splits trees on a fiction.

    Normalising it in ingest alone would leave training seeing NaN while a request carried
    the raw value, so the same applicant would score differently depending on the path it
    arrived by. That is training/serving skew, and it is invisible: both halves work.
    """
    rng = np.random.default_rng(3)
    rows = 40
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(rows),
            "TARGET": [i % 2 for i in range(rows)],
            "DAYS_EMPLOYED": [365243 if i % 4 == 0 else -1000 - i for i in range(rows)],
            "AMT_CREDIT": rng.normal(400000, 5000, rows),
        }
    )
    features, target = split_features_target(frame, SENTINEL_SPEC)

    pipeline = build_pipeline("lightgbm", spec=SENTINEL_SPEC, n_estimators=5, verbose=-1)
    pipeline.fit(features, target)

    transformed = pipeline.named_steps["sentinels"].transform(features)
    assert not (transformed["DAYS_EMPLOYED"] == 365243).any(), "sentinel survived the step"
    assert transformed["DAYS_EMPLOYED"].isna().sum() == 10, "one NaN per sentinel row"
    # Real values are untouched.
    assert (transformed["DAYS_EMPLOYED"].dropna() < 0).all()


def test_serving_and_training_agree_on_a_sentinel_row():
    """The same applicant must score identically whichever path built its frame.

    The skew this guards is not hypothetical: it is what normalising in ingest rather than
    in the pipeline would have produced.
    """
    rng = np.random.default_rng(4)
    rows = 40
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": range(rows),
            "TARGET": [i % 2 for i in range(rows)],
            "DAYS_EMPLOYED": [365243 if i % 4 == 0 else -1000 - i for i in range(rows)],
            "AMT_CREDIT": rng.normal(400000, 5000, rows),
        }
    )
    features, target = split_features_target(frame, SENTINEL_SPEC)
    pipeline = build_pipeline("lightgbm", spec=SENTINEL_SPEC, n_estimators=5, verbose=-1)
    pipeline.fit(features, target)

    # A request carrying the raw sentinel, exactly as the API would serialize it.
    from_request = pd.DataFrame([{"DAYS_EMPLOYED": 365243, "AMT_CREDIT": 400000.0}])
    # The same applicant as ingest would have stored it, had ingest normalised.
    from_ingest = pd.DataFrame([{"DAYS_EMPLOYED": np.nan, "AMT_CREDIT": 400000.0}])

    assert pipeline.predict_proba(from_request)[0][1] == pipeline.predict_proba(from_ingest)[0][1]
