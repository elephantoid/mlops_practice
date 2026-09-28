"""Tests for the data-independent half of the training module.

The sweep shape and the promotion gate are both decisions expressed in code, so they are
checkable without a populated registry or any archive on disk. What genuinely needs data
-- the metric values, the ``cv_roc_auc_mean > 0.6`` floor, the logged positive rate -- is not
covered here and stays blocked on plan Step 0.

The promotion gate is the one worth holding to a test. ``promote_best`` orders by the track's
selection metric across both grids with nothing constraining model flavour, so a
LogisticRegression can win on a thin margin -- and SHAP's ``TreeExplainer`` has no handle
on it. The reason codes DoD (3) promises would then be unbuildable against the promoted
artifact, and the failure would surface in W2 as "SHAP does not support this model"
rather than here as a promotion decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.specs import get_feature_spec
from src.models import train as train_module
from src.models.train import (
    LIGHTGBM_GRID,
    LOGREG_GRID,
    TREE_MODEL_TYPES,
    TREE_MODELS_ONLY,
    cv_metric_key,
    experiment_name_for,
    model_name_for,
    operating_threshold_for,
    promote_best,
)


def _runs_frame(rows: list[tuple[str, str, float]], track: str = "credit") -> pd.DataFrame:
    """Build what mlflow.search_runs returns, ordered by the track's selection metric DESC.

    The metric column is derived through ``cv_metric_key`` rather than spelled out. A
    hardcoded ``metrics.cv_auc_mean`` here would keep passing after the production key was
    renamed, because this frame is the only thing ``promote_best`` reads -- the fixture would
    be testing itself.
    """
    metric_column = f"metrics.{cv_metric_key(track)}"
    frame = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "tags.model_type": model_type,
                metric_column: auc,
                # promote_best tags the registered version with this, so a frame without
                # it would exercise a narrower path than production takes.
                "metrics.test_roc_auc": auc - 0.01,
            }
            for run_id, model_type, auc in rows
        ]
    )
    return frame.sort_values(metric_column, ascending=False).reset_index(drop=True)


def test_sweep_is_within_the_planned_range():
    """The 14-config sweep was retired, not deferred.

    It existed to satisfy a Telco milestone asking for "10+ runs". The riskwatch rollup
    asks for a registered model and says nothing about a count, so deferring it would only
    have re-created the cost in W2 -- the week the pre-mortem names as least able to
    absorb it.
    """
    total = len(LIGHTGBM_GRID) + len(LOGREG_GRID)
    assert 2 <= total <= 4, f"plan asks for 2-4 configs, found {total}"


def test_sweep_keeps_the_class_weight_arm():
    """The imbalance arm is the one that speaks to what this project is about.

    Credit runs ~8% positive and fraud ~0.17%. ROC-AUC barely moves under reweighting
    because it is a ranking metric, so the effect shows up in recall and F1 -- that
    contrast is the reason to keep the arm when cutting the sweep.
    """
    assert any("class_weight" in config for config in LIGHTGBM_GRID)


def test_promotion_gate_skips_a_better_scoring_non_tree_model(monkeypatch):
    """A logreg winning on cv_auc must not be promoted while reason codes are in force."""
    captured = {}

    runs = _runs_frame(
        [
            ("logreg-run", "logreg", 0.79),  # highest score
            ("lgbm-run", "lightgbm", 0.78),
        ]
    )
    monkeypatch.setattr(train_module.mlflow, "search_runs", lambda **kw: runs)

    def fake_register(uri: str, name: str):
        captured["uri"] = uri
        captured["name"] = name
        return type("V", (), {"version": "1"})()

    monkeypatch.setattr(train_module.mlflow, "register_model", fake_register)
    monkeypatch.setattr(train_module, "MlflowClient", lambda: _StubClient())

    promote_best(["logreg-run", "lgbm-run"], track="credit")

    assert "lgbm-run" in captured["uri"], "the tree model must be promoted despite a lower score"
    assert captured["name"] == "riskwatch_credit"


def test_promotion_gate_raises_when_no_tree_run_exists(monkeypatch):
    """Failing loudly beats promoting an unexplainable model.

    The message must name the gate and what was observed, or the next reader sees only
    "no runs" against an experiment that plainly has runs.
    """
    runs = _runs_frame([("logreg-run", "logreg", 0.81)])
    monkeypatch.setattr(train_module.mlflow, "search_runs", lambda **kw: runs)

    with pytest.raises(RuntimeError) as excinfo:
        promote_best(["logreg-run"], track="credit")

    message = str(excinfo.value)
    assert "tree" in message.lower()
    assert "logreg" in message, "the error must report what model types were actually there"
    assert "DoD 3" in message or "reason codes" in message.lower()


def test_promotion_gate_is_a_named_flag_not_an_inline_condition():
    """Lifting the gate trades reason codes for something else -- a deliberate act."""
    assert TREE_MODELS_ONLY is True
    assert "lightgbm" in TREE_MODEL_TYPES
    assert "logreg" not in TREE_MODEL_TYPES


def test_promote_best_rejects_an_empty_run_list():
    with pytest.raises(ValueError, match="at least one run_id"):
        promote_best([])


def test_names_are_track_derived():
    assert model_name_for("credit") == "riskwatch_credit"
    assert model_name_for("fraud") == "riskwatch_fraud"
    assert experiment_name_for("credit") == "riskwatch_credit"


class _StubClient:
    """Minimal MlflowClient stand-in: records calls, resolves nothing."""

    def search_model_versions(self, *a, **k) -> list:
        """No existing version for any run.

        promote_best() consults this before registering, so that a DAG retry reuses the
        version it already minted instead of creating a duplicate. Returning empty drives
        the first-promotion path, which is what these tests exercise.
        """
        return []

    def set_registered_model_alias(self, *a, **k) -> None:
        return None

    def set_model_version_tag(self, *a, **k) -> None:
        return None

    def get_model_version(self, name, version):
        return type("V", (), {"version": version, "name": name, "tags": {}})()


# --- PR-AUC (plan Step 6: "Log PR-AUC and Recall@FPR alongside ROC-AUC") ---------------
#
# average_precision_score is a pure function of two arrays, so this is testable on
# synthetic data with no archive and no training run -- the same way promote_best is
# tested against synthetic run frames above. What needs real data is the VALUE; what the
# plan asked for is the logging.


class _FixedProbaPipeline:
    """Returns predetermined probabilities, so evaluate() is tested and not the model."""

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack([1 - self._probabilities, self._probabilities])


def test_evaluate_emits_pr_auc():
    from src.models.train import evaluate

    target = pd.Series([0, 0, 1, 1])
    pipeline = _FixedProbaPipeline([0.1, 0.2, 0.8, 0.9])

    # Any cut will do: every assertion below is on a threshold-free metric. That evaluate
    # now demands one explicitly is the point -- the old signature defaulted to 0.5 and let
    # a caller report precision at a cut they never chose.
    metrics = evaluate(pipeline, pd.DataFrame(index=target.index), target, 0.5)

    assert "pr_auc" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    # A perfect ranking scores 1.0 on both.
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_pr_auc_diverges_from_roc_auc_under_extreme_imbalance():
    """This divergence is the entire reason the plan asked for PR-AUC.

    At a sub-1% positive rate ROC-AUC's false-positive rate has the enormous negative
    class in its denominator, so a model can look excellent while almost every positive
    prediction it makes is wrong. Average precision has no such denominator.

    Constructed to mirror the fraud track: 1000 rows, 5 positives (0.5%), a model that
    ranks the positives highly but buries them under a wall of high-scoring negatives.
    """
    from src.models.train import evaluate

    rng = np.random.default_rng(0)
    n, n_pos = 1000, 5
    target = pd.Series([1] * n_pos + [0] * (n - n_pos))

    probabilities = np.concatenate(
        [
            rng.uniform(0.70, 0.85, n_pos),  # positives score well...
            rng.uniform(0.60, 0.95, 60),  # ...but 60 negatives score as well or better
            rng.uniform(0.00, 0.30, n - n_pos - 60),
        ]
    )

    metrics = evaluate(
        _FixedProbaPipeline(probabilities), pd.DataFrame(index=target.index), target, 0.5
    )

    assert metrics["roc_auc"] > 0.85, "ROC-AUC should look reassuring here"
    assert metrics["pr_auc"] < 0.35, "PR-AUC should not"
    assert metrics["roc_auc"] - metrics["pr_auc"] > 0.5, (
        "the gap between them is the signal that the negative class is swamping ROC-AUC"
    )


def test_pr_auc_collapses_toward_the_base_rate_for_a_useless_model():
    """A random ranker's average precision approaches the positive rate, not 0.5."""
    from src.models.train import evaluate

    rng = np.random.default_rng(1)
    n, n_pos = 2000, 20  # 1% positive
    target = pd.Series(rng.permutation([1] * n_pos + [0] * (n - n_pos)))
    probabilities = rng.uniform(0, 1, n)

    metrics = evaluate(
        _FixedProbaPipeline(probabilities), pd.DataFrame(index=target.index), target, 0.5
    )

    assert metrics["pr_auc"] < 0.10, "a useless model must not score near 0.5 on PR-AUC"
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.15)


# --- Provenance: written by ingest, consumed here ----------------------------------------
#
# ingest writes source_used into the parquet metadata so a model trained on the credit
# FALLBACK -- a genuinely different dataset -- is distinguishable from one trained on Home
# Credit. It was written and never read, which made the provenance claim false at the only
# boundary that matters: the registry.


def test_provenance_is_read_from_the_snapshot(tmp_path):
    from src.data.ingest import _write_parquet_with_metadata
    from src.models.train import _snapshot_provenance

    path = tmp_path / "snap.parquet"
    _write_parquet_with_metadata(
        pd.DataFrame({"a": [1]}),
        path,
        {
            b"source_used": b"42477",
            b"is_fallback": b"True",
            b"track": b"credit",
            b"rows": b"30000",
            b"positive_rate": b"0.221000",
            b"ingested_at": b"20260928T000000Z",
        },
    )

    provenance = _snapshot_provenance(path)

    assert provenance["source_used"] == "42477"
    assert provenance["is_fallback"] == "True"
    assert provenance["rows"] == "30000"


def test_provenance_ignores_keys_ingest_does_not_write(tmp_path):
    """Only the known keys become run tags.

    A future metadata addition must not silently turn into a tag nobody chose.
    """
    from src.data.ingest import _write_parquet_with_metadata
    from src.models.train import _snapshot_provenance

    path = tmp_path / "snap.parquet"
    _write_parquet_with_metadata(
        pd.DataFrame({"a": [1]}),
        path,
        {b"source_used": b"home-credit-default-risk", b"something_else": b"ignore me"},
    )

    provenance = _snapshot_provenance(path)

    assert provenance == {"source_used": "home-credit-default-risk"}


def test_missing_provenance_degrades_instead_of_failing(tmp_path):
    """A snapshot predating provenance must not stop a training run."""
    from src.models.train import _snapshot_provenance

    assert _snapshot_provenance(tmp_path / "does-not-exist.parquet") == {}


def test_run_experiment_tags_every_provenance_key(monkeypatch):
    """The tags must actually reach the run, prefixed so they cannot collide with params."""
    import src.models.train as train_module

    tags: dict[str, str] = {}
    monkeypatch.setattr(train_module.mlflow, "set_tag", lambda k, v: tags.__setitem__(k, v))
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "cross_val_selection_score", lambda *a, **k: (0.75, 0.01))
    # Both metric names, because run_experiment reads ``train_metrics[spec.selection_metric]``
    # and this stub must not silently constrain the test to whichever track happens to be the
    # default. A single-key stub passes here and raises KeyError the day the spec changes.
    monkeypatch.setattr(train_module, "evaluate", lambda *a, **k: {"roc_auc": 0.75, "pr_auc": 0.31})

    class _Run:
        info = type("I", (), {"run_id": "r1"})()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda *a, **k: _Run())

    class _Fittable(_FixedProbaPipeline):
        def fit(self, *a, **k):
            return self

    monkeypatch.setattr(train_module, "build_pipeline", lambda *a, **k: _Fittable([0.5, 0.5]))
    monkeypatch.setattr(train_module, "infer_signature", lambda *a, **k: None)

    frame = pd.DataFrame({"x": [1.0, 2.0]})
    target = pd.Series([0, 1])
    train_module.run_experiment(
        "lightgbm",
        {},
        frame,
        target,
        frame,
        target,
        get_feature_spec("credit"),
        operating_threshold_for("credit"),
        {"source_used": "42477", "is_fallback": "True"},
    )

    assert tags["data_source_used"] == "42477"
    assert tags["data_is_fallback"] == "True"
    assert tags["model_type"] == "lightgbm"


# --- The selection metric, and the cut evaluate reports at --------------------------------


def test_metric_keys_name_the_metric_they_hold():
    """``cv_auc_mean`` did not say *which* area, and that is why it was renamed.

    A shared key is not merely untidy: a fraud PR-AUC of 0.31 read out of a field called
    "auc" looks like a model worse than random, when at a 0.001727 positive rate it is a good
    one. The two keys must also differ from each other, or the cross-track comparison the
    rename exists to make impossible is still possible.
    """
    assert cv_metric_key("credit") == "cv_roc_auc_mean"
    assert cv_metric_key("fraud") == "cv_pr_auc_mean"
    assert cv_metric_key("credit") != cv_metric_key("fraud")

    from src.models.train import cv_std_key

    assert cv_std_key("credit") == "cv_roc_auc_std"
    assert cv_std_key("fraud") == "cv_pr_auc_std"


def test_operating_threshold_is_the_tracks_decline_boundary():
    """evaluate's cut has to be the boundary the service actually declines at.

    Reporting precision at any other point describes a decision this service does not make.
    Asserted against ``src.api.main``'s literal rather than recomputed, so a drift between the
    reported cut and the served one fails here as well as in tests/test_thresholds.py.
    """
    from src.api.main import DECISION_BANDS

    for track in ("credit", "fraud"):
        _, decline_at = DECISION_BANDS[track]
        assert operating_threshold_for(track) == pytest.approx(decline_at, abs=5e-5)


def test_evaluate_reports_at_the_threshold_it_is_given():
    """The retired ``precision_at_0.5`` keys had the cut baked into their names.

    Two cuts on the same fitted model must produce different precision and recall and the
    same ROC-AUC -- that contrast is what says the threshold argument is actually reaching the
    metrics rather than being accepted and ignored, which a default would have hidden.
    """
    from src.models.train import evaluate

    target = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = _FixedProbaPipeline([0.05, 0.20, 0.55, 0.45, 0.80, 0.95])

    strict = evaluate(pipeline, pd.DataFrame(index=target.index), target, 0.90)
    loose = evaluate(pipeline, pd.DataFrame(index=target.index), target, 0.40)

    assert strict["operating_threshold"] == 0.90
    assert loose["operating_threshold"] == 0.40
    assert strict["recall"] < loose["recall"], "a higher cut must catch fewer positives"
    assert strict["flagged_share"] < loose["flagged_share"]
    assert strict["roc_auc"] == pytest.approx(loose["roc_auc"]), (
        "ranking metrics must not move with the cut"
    )
    assert "precision_at_0.5" not in strict, "the cut must not be back in a key name"


def test_run_experiment_tags_the_selection_metric(monkeypatch):
    """The metric name is logged as a tag as well as being in the key.

    The key tells a reader what a number is; the tag lets a query filter runs by it without
    parsing key strings. Without the tag, "show me every run selected on PR-AUC" is a string
    match over column names.
    """
    import src.models.train as train_module

    tags: dict[str, str] = {}
    metrics: dict[str, float] = {}
    monkeypatch.setattr(train_module.mlflow, "set_tag", lambda k, v: tags.__setitem__(k, v))
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow, "log_metrics", lambda d: metrics.update(d))
    monkeypatch.setattr(train_module.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "cross_val_selection_score", lambda *a, **k: (0.31, 0.02))
    monkeypatch.setattr(train_module, "evaluate", lambda *a, **k: {"pr_auc": 0.31})
    monkeypatch.setattr(train_module, "infer_signature", lambda *a, **k: None)

    class _Run:
        info = type("I", (), {"run_id": "r1"})()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda *a, **k: _Run())

    class _Fittable(_FixedProbaPipeline):
        def fit(self, *a, **k):
            return self

    monkeypatch.setattr(train_module, "build_pipeline", lambda *a, **k: _Fittable([0.5, 0.5]))

    frame = pd.DataFrame({"x": [1.0, 2.0]})
    target = pd.Series([0, 1])
    train_module.run_experiment(
        "lightgbm",
        {},
        frame,
        target,
        frame,
        target,
        get_feature_spec("fraud"),
        operating_threshold_for("fraud"),
    )

    assert tags["selection_metric"] == "pr_auc"
    assert metrics["cv_pr_auc_mean"] == pytest.approx(0.31)
    assert metrics["cv_pr_auc_std"] == pytest.approx(0.02)
    assert "cv_auc_mean" not in metrics, "the retired key must not be written alongside"
