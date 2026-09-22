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
from src.features.specs import FeatureSpec, get_feature_spec

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "latest.parquet"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

# Track-derived, not a shared constant. Two registered models with independent
# schemas, thresholds and retrain cadence are what make DoD (7)'s "drift in one track
# retrains only that track" an honest demonstration -- a single shared name would
# couple the two retrain cycles and the serving lookup resolves per track anyway.
DEFAULT_TRACK = "credit"


def experiment_name_for(track: str = DEFAULT_TRACK) -> str:
    """MLflow experiment name for one track."""
    return f"riskwatch_{track}"


def model_name_for(track: str = DEFAULT_TRACK) -> str:
    """Registered-model name for one track. Must match Track.model_name."""
    return f"riskwatch_{track}"


PRODUCTION_ALIAS = "production"

# SHAP's TreeExplainer is the only explainer with a supported handle on these pipelines,
# so while DoD (3) requires reason codes the promoted artifact must be a tree model.
# Lifting this is a deliberate act -- it trades reason codes for whatever the alternative
# model wins on -- which is why it is a named flag rather than an inline condition.
TREE_MODELS_ONLY = True
TREE_MODEL_TYPES = frozenset({"lightgbm"})

TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

# cloudpickle rather than the 3.x default of skops: skops refuses to load the pipeline
# because LGBMClassifier is not on its trusted-types list.
SERIALIZATION_FORMAT = "cloudpickle"

# Three configs, not fourteen. The original sweep existed to satisfy a retired Telco
# milestone that asked for "10+ runs"; the riskwatch rollup asks for a registered model
# and says nothing about a run count, so the sweep was retired rather than deferred --
# deferring it would only have re-created the cost in W2, the week the pre-mortem names
# as least able to absorb it.
#
# What survives is the contrast worth having in the MLflow table: a capacity arm, a
# regularized arm at that capacity, and a class_weight arm. Credit defaults run ~8%
# positive and fraud ~0.17%, so the reweighting arm is the one that speaks to the
# imbalance the whole project is about -- ROC-AUC barely moves under reweighting because
# it is a ranking metric, and the effect shows up in recall and F1 instead.
LIGHTGBM_GRID: list[dict[str, Any]] = [
    # Baseline capacity.
    {"num_leaves": 31, "n_estimators": 300, "learning_rate": 0.05},
    # Same capacity, explicitly regularized.
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05, "min_child_samples": 50},
    # The imbalance arm.
    {"num_leaves": 8, "n_estimators": 300, "learning_rate": 0.05, "class_weight": "balanced"},
]

# One baseline, not two. A second regularization strength on a model that exists only
# as a sanity floor is a run whose result nobody acts on.
LOGREG_GRID: list[dict[str, Any]] = [
    {"C": 1.0},
]


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
    spec: FeatureSpec,
) -> tuple[float, float]:
    """Mean and standard deviation of ROC-AUC over stratified folds.

    A fresh pipeline is built per fold so the imputers, encoders and scaler are fit inside
    the fold. Fitting preprocessing once on all of ``features`` would leak held-out
    statistics into every fold and quietly inflate the score.
    """
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, valid_idx in splitter.split(features, target):
        pipeline = build_pipeline(model_type, spec=spec, **params)
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
    spec: FeatureSpec,
) -> str:
    """Execute one MLflow run: cross-validate, refit, score on test, log the model.

    Returns the run id.
    """
    run_name = f"{model_type}-" + "-".join(f"{k}={v}" for k, v in sorted(params.items()))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params(params)

        cv_mean, cv_std = cross_val_auc(model_type, params, X_train, y_train, spec)
        mlflow.log_metrics({"cv_auc_mean": cv_mean, "cv_auc_std": cv_std})

        pipeline = build_pipeline(model_type, spec=spec, **params)
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


def promote_best(
    run_ids: list[str],
    experiment_name: str | None = None,
    track: str = DEFAULT_TRACK,
) -> ModelVersion:
    """Register the highest cross-validated run of *this sweep* and move the production alias.

    An alias rather than a stage: ``transition_model_version_stage`` has been deprecated
    since MLflow 2.9 and is slated for removal.

    The search is constrained to ``run_ids`` rather than the whole experiment for two
    reasons, both of which bite on the second invocation:

    * An unconstrained search ranks every run ever logged. When Milestone 4's DAG retrains
      weekly into this same experiment, a stale run that happened to score well on *older
      data* would outrank the fresh sweep and be re-promoted -- so the pipeline could never
      improve past its historical high-water mark, silently.
    * ``search_runs`` filters on lifecycle stage, not status, so FAILED runs are returned.
      A run that logged ``cv_auc_mean`` and then died before ``log_model`` would rank first
      forever and make every future promotion raise. Hence the explicit status filter.
    """
    if not run_ids:
        raise ValueError("promote_best() requires at least one run_id")

    experiment = experiment_name or experiment_name_for(track)
    model_name = model_name_for(track)

    quoted_ids = ",".join(f"'{run_id}'" for run_id in run_ids)
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"attributes.run_id IN ({quoted_ids}) and attributes.status = 'FINISHED'",
        order_by=["metrics.cv_auc_mean DESC"],
    )
    if runs.empty:
        raise RuntimeError(f"No FINISHED runs among {len(run_ids)} in {experiment!r}")

    # Gate promotion to tree models while DoD (3) is in force. The ordering above is by
    # cv_auc_mean across BOTH grids with nothing constraining model flavour, so a logreg
    # could win on a thin margin -- and SHAP's TreeExplainer has no handle on it. The
    # reason codes DoD (3) promises would then be unbuildable against the promoted
    # artifact, and the failure would surface in W2 as "SHAP does not support this model"
    # rather than here as a promotion decision.
    if TREE_MODELS_ONLY:
        tree_runs = runs[runs["tags.model_type"].isin(TREE_MODEL_TYPES)]
        if tree_runs.empty:
            raise RuntimeError(
                f"No FINISHED tree-model run among {len(run_ids)} in {experiment!r}. "
                f"Promotion is gated to {sorted(TREE_MODEL_TYPES)} while reason codes "
                f"(DoD 3) are in force: SHAP's TreeExplainer cannot explain the "
                f"alternatives. Observed model types: "
                f"{sorted(runs['tags.model_type'].dropna().unique())}"
            )
        if len(tree_runs) < len(runs):
            skipped = len(runs) - len(tree_runs)
            logger.info(
                "Promotion gate: skipped %d non-tree run(s); best tree run cv_auc %.4f "
                "against overall best %.4f",
                skipped,
                tree_runs.iloc[0]["metrics.cv_auc_mean"],
                runs.iloc[0]["metrics.cv_auc_mean"],
            )
        runs = tree_runs

    best = runs.iloc[0]
    version = mlflow.register_model(f"runs:/{best.run_id}/model", model_name)

    client = MlflowClient()
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version.version)
    for key, value in {
        "model_type": best["tags.model_type"],
        "cv_auc_mean": f"{best['metrics.cv_auc_mean']:.4f}",
        "test_roc_auc": f"{best['metrics.test_roc_auc']:.4f}",
    }.items():
        client.set_model_version_tag(name=model_name, version=version.version, key=key, value=value)

    logger.info(
        "Promoted %s v%s (%s, cv_auc %.4f) to @%s",
        model_name,
        version.version,
        best["tags.model_type"],
        best["metrics.cv_auc_mean"],
        PRODUCTION_ALIAS,
    )
    # Re-fetch: mlflow.register_model returns a snapshot taken before the alias and tags
    # were applied, so `version.tags` on that object is empty. Callers -- Milestone 4's
    # task_train in particular -- need the populated version.
    return client.get_model_version(model_name, version.version)


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    experiment_name: str | None = None,
    track_name: str = DEFAULT_TRACK,
) -> ModelVersion:
    """Run the full sweep and promote the winner. Returns the promoted model version.

    Returning the ``ModelVersion`` rather than ``None`` gives Milestone 4's ``task_train``
    something to hand downstream via XCom.
    """
    experiment = experiment_name or experiment_name_for(track_name)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)

    spec = get_feature_spec(track_name)
    features, target = split_features_target(pd.read_parquet(data_path), spec)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "train %d rows | test %d rows | positive rate %.4f",
        len(X_train),
        len(X_test),
        target.mean(),
    )

    configs: list[tuple[ModelType, dict[str, Any]]] = [
        ("logreg", params) for params in LOGREG_GRID
    ] + [("lightgbm", params) for params in LIGHTGBM_GRID]

    run_ids = [
        run_experiment(model_type, params, X_train, y_train, X_test, y_test, spec)
        for model_type, params in configs
    ]
    return promote_best(run_ids, experiment, track=track_name)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()


if __name__ == "__main__":
    main()
