"""Tests for the data-independent half of the training module.

The sweep shape and the promotion gate are both decisions expressed in code, so they are
checkable without a populated registry or any archive on disk. What genuinely needs data
-- the metric values, the ``cv_auc_mean > 0.6`` floor, the logged positive rate -- is not
covered here and stays blocked on plan Step 0.

The promotion gate is the one worth holding to a test. ``promote_best`` orders by
``cv_auc_mean`` across both grids with nothing constraining model flavour, so a
LogisticRegression can win on a thin margin -- and SHAP's ``TreeExplainer`` has no handle
on it. The reason codes DoD (3) promises would then be unbuildable against the promoted
artifact, and the failure would surface in W2 as "SHAP does not support this model"
rather than here as a promotion decision.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import train as train_module
from src.models.train import (
    LIGHTGBM_GRID,
    LOGREG_GRID,
    TREE_MODEL_TYPES,
    TREE_MODELS_ONLY,
    experiment_name_for,
    model_name_for,
    promote_best,
)


def _runs_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build what mlflow.search_runs returns, already ordered by cv_auc_mean DESC."""
    frame = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "tags.model_type": model_type,
                "metrics.cv_auc_mean": auc,
                # promote_best tags the registered version with this, so a frame without
                # it would exercise a narrower path than production takes.
                "metrics.test_roc_auc": auc - 0.01,
            }
            for run_id, model_type, auc in rows
        ]
    )
    return frame.sort_values("metrics.cv_auc_mean", ascending=False).reset_index(drop=True)


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

    def set_registered_model_alias(self, *a, **k) -> None:
        return None

    def set_model_version_tag(self, *a, **k) -> None:
        return None

    def get_model_version(self, name, version):
        return type("V", (), {"version": version, "name": name, "tags": {}})()
