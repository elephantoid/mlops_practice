"""Hyperparameter sweep with MLflow tracking and Model Registry promotion.

Run from the repo root with::

    uv run python -m src.models.train

The ``-m`` form is required, not stylistic: ``python src/models/train.py`` puts
``src/models/`` on ``sys.path`` rather than the repo root, so ``src.features.pipeline``
would not resolve.

Models are **selected** on mean cross-validated ROC-AUC and **reported** on a test set that
is touched exactly once per run. Selecting on the test score would leak the holdout and
inflate every number in the MLflow table.

Three MLflow 3.x details this module depends on, each verified against the installed 3.15.1
rather than assumed:

* The ``file:`` tracking backend now raises on startup ("maintenance mode"), so the default
  tracking URI is ``sqlite:///mlflow.db``. Artifacts still land in ``./mlruns``.
* ``serialization_format`` defaults to ``"skops"``, which rejects this pipeline outright --
  ``LGBMClassifier`` is not a trusted sklearn type. ``cloudpickle`` is pinned explicitly.
* ``mlflow.pyfunc`` wraps ``predict`` by default, which returns class labels. The serving
  contract needs probabilities, so the model is logged with
  ``pyfunc_predict_fn="predict_proba"`` and the API applies its own threshold.

Model versions are promoted with a registry **alias**, not a stage: stages have been
deprecated since MLflow 2.9 and emit a ``FutureWarning`` pointing at the migration guide.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.features.pipeline import RANDOM_STATE, ModelType, build_pipeline, split_features_target

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "latest.parquet"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

EXPERIMENT_NAME = "churnwatch"
MODEL_NAME = "churnwatch"
PRODUCTION_ALIAS = "production"

TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

# cloudpickle rather than the 3.x default of skops: skops refuses to load the pipeline
# because LGBMClassifier is not on its trusted-types list.
SERIALIZATION_FORMAT = "cloudpickle"

# Hand-specified rather than a product() sweep, so every row in the MLflow table has a
# reason. The blueprint asks for 10+ runs varying num_leaves, learning_rate and
# class_weight; this is 14.
LIGHTGBM_GRID: list[dict[str, Any]] = [
    # Capacity sweep at the stock learning rate. num_leaves=31 is the overfitting default,
    # kept deliberately as the control to measure the others against.
    {"num_leaves": 8, "n_estimators": 100, "learning_rate": 0.1},
    {"num_leaves": 15, "n_estimators": 100, "learning_rate": 0.1},
    {"num_leaves": 31, "n_estimators": 100, "learning_rate": 0.1},
    # Slower learning, more trees.
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05},
    {"num_leaves": 15, "n_estimators": 300, "learning_rate": 0.05},
    {"num_leaves": 31, "n_estimators": 300, "learning_rate": 0.05},
    # Explicit regularization at the expected-best capacity.
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05, "reg_lambda": 5.0},
    {
        # subsample is inert in LightGBM unless subsample_freq > 0 -- a silent no-op that
        # makes bagging look ineffective when it was simply never applied.
        "num_leaves": 8,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
    },
    # class_weight arm. AUC is a ranking metric and barely moves under reweighting; the
    # effect should show up in recall and F1 instead. That contrast is the point.
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05, "class_weight": "balanced"},
    {"num_leaves": 15, "n_estimators": 100, "learning_rate": 0.1, "class_weight": "balanced"},
    {"num_leaves": 31, "n_estimators": 100, "learning_rate": 0.1, "class_weight": "balanced"},
]

LOGREG_GRID: list[dict[str, Any]] = [
    {"C": 1.0},
    {"C": 0.1},
]


def configure_tracking() -> str:
    """Point MLflow at the tracking backend and return the resolved URI.

    Every entry point calls this rather than relying on ``train()`` having done it once.
    Airflow runs each task in a separate process, so ``task_evaluate`` and ``task_promote``
    would otherwise fall back to the local sqlite default and read an empty registry --
    reporting "no incumbent" on a project that has been promoting models for weeks.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Score a fitted pipeline.

    ROC-AUC drives promotion because it is threshold-independent. The rest are logged at the
    fixed 0.5 cut point the blueprint specifies -- useful for reading the class-imbalance
    story, but not for selecting between models.
    """
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "f1": f1_score(target, predictions),
        "precision_at_0.5": precision_score(target, predictions, zero_division=0),
        "recall_at_0.5": recall_score(target, predictions),
        "log_loss": log_loss(target, probabilities),
    }


def cross_val_auc(
    model_type: ModelType,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[float, float]:
    """Mean and standard deviation of ROC-AUC over stratified folds.

    A fresh pipeline is built per fold so the imputers, encoders and scaler are fit inside
    the fold. Fitting preprocessing once on all of ``features`` would leak held-out
    statistics into every fold and quietly inflate the score.
    """
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in splitter.split(features, target):
        pipeline = build_pipeline(model_type, **params)
        pipeline.fit(features.iloc[train_idx], target.iloc[train_idx])
        probabilities = pipeline.predict_proba(features.iloc[valid_idx])[:, 1]
        scores.append(roc_auc_score(target.iloc[valid_idx], probabilities))
    return float(np.mean(scores)), float(np.std(scores))


def run_experiment(
    model_type: ModelType,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Execute one MLflow run: cross-validate, refit, score on test, log the model.

    Returns the run id.
    """
    run_name = f"{model_type}-" + "-".join(f"{k}={v}" for k, v in sorted(params.items()))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params(params)

        cv_mean, cv_std = cross_val_auc(model_type, params, X_train, y_train)
        mlflow.log_metrics({"cv_auc_mean": cv_mean, "cv_auc_std": cv_std})

        pipeline = build_pipeline(model_type, **params)
        pipeline.fit(X_train, y_train)

        train_metrics = evaluate(pipeline, X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # The gap between train and CV is the overfitting signal the sweep exists to close.
        mlflow.log_metric("overfit_gap", train_metrics["roc_auc"] - cv_mean)

        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            serialization_format=SERIALIZATION_FORMAT,
        )

        logger.info(
            "%-46s cv %.4f +/- %.4f | test %.4f | gap %.4f",
            run_name,
            cv_mean,
            cv_std,
            test_metrics["roc_auc"],
            train_metrics["roc_auc"] - cv_mean,
        )
        return run.info.run_id


def best_finished_run(run_ids: list[str], experiment_name: str = EXPERIMENT_NAME) -> pd.Series:
    """Return the highest cross-validated FINISHED run among *this sweep's* ``run_ids``.

    The search is constrained to ``run_ids`` rather than the whole experiment for two
    reasons, both of which bite on the second invocation:

    * An unconstrained search ranks every run ever logged. When Milestone 4's DAG retrains
      weekly into this same experiment, a stale run that happened to score well on *older
      data* would outrank the fresh sweep and be re-promoted -- so the pipeline could never
      improve past its historical high-water mark, silently.
    * ``search_runs`` filters on lifecycle stage, not status, so FAILED runs are returned.
      A run that logged ``cv_auc_mean`` and then died before ``log_model`` would rank first
      forever and make every future promotion raise. Hence the explicit status filter.

    Split out from :func:`promote_best` so the retrain DAG can read the candidate's score
    in ``task_evaluate`` and decide whether promotion is warranted *before* anything
    touches the production alias.
    """
    if not run_ids:
        raise ValueError("best_finished_run() requires at least one run_id")

    configure_tracking()
    quoted_ids = ",".join(f"'{run_id}'" for run_id in run_ids)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"attributes.run_id IN ({quoted_ids}) and attributes.status = 'FINISHED'",
        order_by=["metrics.cv_auc_mean DESC"],
    )
    if runs.empty:
        raise RuntimeError(f"No FINISHED runs among {len(run_ids)} in {experiment_name!r}")

    return runs.iloc[0]


def _registered_version_for_run(client: MlflowClient, run_id: str) -> ModelVersion | None:
    """Find an existing ``MODEL_NAME`` version already registered from ``run_id``.

    Returns ``None`` when there is none, and also when the lookup itself fails -- the
    caller then registers as it always did, so a failure here costs a duplicate version at
    worst rather than blocking the promotion outright.
    """
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    except MlflowException:
        # Most often the registered model does not exist yet, i.e. the first promotion.
        return None
    return versions[0] if versions else None


def promote_best(run_ids: list[str], experiment_name: str = EXPERIMENT_NAME) -> ModelVersion:
    """Register the best run of *this sweep* and move the production alias onto it.

    An alias rather than a stage: ``transition_model_version_stage`` has been deprecated
    since MLflow 2.9 and is slated for removal.

    Unconditional by design -- the caller decides whether promotion is deserved. The CLI
    always promotes; the retrain DAG gates this behind an AUC-delta check in
    ``src/pipelines/retrain.py``.

    Configures tracking itself rather than inheriting it from ``best_finished_run`` below.
    The call is idempotent, and the guarantee needs to be local: reordering these two lines
    would otherwise register the model into whatever backend happened to be set, which for
    an unconfigured process is the local sqlite file -- a silent write to the wrong registry
    rather than a failure.
    """
    configure_tracking()
    best = best_finished_run(run_ids, experiment_name)

    client = MlflowClient()

    # Registration is the one step here that is not naturally idempotent: setting an alias
    # or a tag twice is a no-op, but register_model() mints a NEW version every call. The
    # DAG runs this task with retries=1, and a transient failure anywhere after this line
    # -- setting the alias, writing a tag, the re-fetch below -- would otherwise re-enter
    # with the same run and leave a duplicate version behind, with the alias moved twice.
    # Reuse the existing version for this run when there is one.
    existing = _registered_version_for_run(client, best.run_id)
    if existing is not None:
        logger.info(
            "Run %s is already registered as %s v%s; reusing it instead of re-registering",
            best.run_id,
            MODEL_NAME,
            existing.version,
        )
        version = existing
    else:
        version = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)

    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version.version)
    for key, value in {
        "model_type": best["tags.model_type"],
        "cv_auc_mean": f"{best['metrics.cv_auc_mean']:.4f}",
        "test_roc_auc": f"{best['metrics.test_roc_auc']:.4f}",
    }.items():
        client.set_model_version_tag(name=MODEL_NAME, version=version.version, key=key, value=value)

    logger.info(
        "Promoted %s v%s (%s, cv_auc %.4f) to @%s",
        MODEL_NAME,
        version.version,
        best["tags.model_type"],
        best["metrics.cv_auc_mean"],
        PRODUCTION_ALIAS,
    )
    # Re-fetch: mlflow.register_model returns a snapshot taken before the alias and tags
    # were applied, so `version.tags` on that object is empty. Callers -- the DAG's
    # task_promote in particular -- need the populated version.
    return client.get_model_version(MODEL_NAME, version.version)


def sweep(
    data_path: Path = DEFAULT_DATA_PATH,
    experiment_name: str = EXPERIMENT_NAME,
) -> list[str]:
    """Run every configuration in the grid. Returns the run ids, promoting nothing.

    Separated from promotion so the retrain DAG can put its AUC-delta gate between the
    two: ``task_train`` calls this, ``task_evaluate`` scores the winner against the
    incumbent, and only then does ``task_promote`` move the alias. A sweep that trains a
    worse model must be able to end without production noticing.
    """
    configure_tracking()
    mlflow.set_experiment(experiment_name)

    features, target = split_features_target(pd.read_parquet(data_path))
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "train %d rows | test %d rows | churn rate %.4f", len(X_train), len(X_test), target.mean()
    )

    configs: list[tuple[ModelType, dict[str, Any]]] = [
        ("logreg", params) for params in LOGREG_GRID
    ] + [("lightgbm", params) for params in LIGHTGBM_GRID]

    return [
        run_experiment(model_type, params, X_train, y_train, X_test, y_test)
        for model_type, params in configs
    ]


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    experiment_name: str = EXPERIMENT_NAME,
) -> ModelVersion:
    """Run the full sweep and promote the winner. Returns the promoted model version.

    The unconditional path, used by the CLI. The retrain DAG deliberately does *not* call
    this: it composes :func:`sweep`, :func:`best_finished_run` and :func:`promote_best`
    itself so it can refuse a promotion that has not earned one.
    """
    return promote_best(sweep(data_path, experiment_name), experiment_name)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()


if __name__ == "__main__":
    main()
