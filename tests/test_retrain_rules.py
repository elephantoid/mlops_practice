"""Decision rules for the retraining DAG.

Hermetic like ``test_api.py``: no MLflow registry, no ``mlflow.db``, no Airflow. The DAG
module itself cannot be imported here -- Airflow is deliberately not in the uv venv -- so
these tests are the only automated check on the two judgements the DAG makes. They matter
more than usual for that reason.
"""

from __future__ import annotations

import pytest
from mlflow.exceptions import MlflowException

from src.pipelines import retrain

# One drifted column out of the 19 the model sees. This exact number is why the rule is
# not a bare share comparison: see test_watched_column_fires_below_the_share_threshold.
# One drifted column out of the credit track's 26 modeled columns. The Telco frame was
# 19 wide, so this number moved with the retarget -- the point it makes did not.
ONE_COLUMN_SHARE = 1 / 26


def summary(share: float, **columns: float) -> dict:
    """Build the dict shape ``src.monitoring.drift.run`` returns."""
    return {"drift_share": share, "drifted_count": len(columns), "drifted_columns": columns}


class TestShouldPromote:
    def test_empty_registry_promotes(self):
        """Something has to be first, or the API never gets a model to serve."""
        assert retrain.should_promote(0.80, None) is True

    def test_clear_improvement_promotes(self):
        assert retrain.should_promote(0.8500, 0.8300) is True

    def test_marginal_gain_is_refused(self):
        """+0.005 is inside run-to-run CV noise; promoting on it churns the alias weekly."""
        assert retrain.should_promote(0.8450, 0.8400) is False

    def test_delta_exactly_at_the_threshold_is_refused(self):
        """The rule is ``> min_delta``, not ``>=`` -- pin which side of the boundary wins."""
        assert retrain.should_promote(0.8500, 0.8400, min_delta=0.01) is False

    def test_regression_is_refused(self):
        assert retrain.should_promote(0.7900, 0.8400) is False


class TestShouldRetrain:
    def test_watched_column_fires_below_the_share_threshold(self):
        """The case the inherited ``drift_share > 0.2`` rule got wrong.

        A +15% shift on the track's drift column is the scenario this repo ships, and it
        moves exactly one column -- a share of 0.0385 on credit, nowhere near 0.20. Under
        the old rule the retrain trigger could never fire on the only drift the project
        can demonstrate.
        """
        assert retrain.should_retrain(summary(ONE_COLUMN_SHARE, AMT_CREDIT=0.31)) is True

    def test_unwatched_column_at_the_same_share_does_not_fire(self):
        """Same magnitude, different column: the rule is about *which*, not only how many."""
        assert retrain.should_retrain(summary(ONE_COLUMN_SHARE, FLAG_OWN_CAR=0.31)) is False

    def test_broad_drift_fires_without_any_watched_column(self):
        """The catch-all arm: broad movement is worth retraining on even with nothing watched.

        **Six columns, not five.** On Telco's 19-column frame the threshold of 0.20 meant
        "4 or more"; on credit's 26 it means "6 or more", because 5/26 = 0.192 does not
        clear it. That is a real behavioural change the retarget caused, and it was
        invisible while this test kept the 5/19 denominator alongside retargeted column
        names -- the share still cleared the threshold, so the test passed and read as
        fully converted.
        """
        moved = dict.fromkeys(
            (
                "FLAG_OWN_CAR",
                "FLAG_OWN_REALTY",
                "CNT_CHILDREN",
                "OCCUPATION_TYPE",
                "NAME_HOUSING_TYPE",
                "WEEKDAY_APPR_PROCESS_START",
            ),
            0.4,
        )
        assert retrain.should_retrain(summary(6 / 26, **moved)) is True

    def test_five_of_twenty_six_columns_does_not_reach_the_catch_all(self):
        """The boundary the retarget moved, asserted rather than implied.

        Five drifted columns cleared 0.20 on the Telco frame and does not on credit's.
        Without this, the change lives only in a comment.
        """
        moved = dict.fromkeys(
            (
                "FLAG_OWN_CAR",
                "FLAG_OWN_REALTY",
                "CNT_CHILDREN",
                "OCCUPATION_TYPE",
                "NAME_HOUSING_TYPE",
            ),
            0.4,
        )
        assert retrain.should_retrain(summary(5 / 26, **moved)) is False

    def test_share_exactly_at_the_threshold_does_not_fire(self):
        assert retrain.should_retrain(summary(0.20, FLAG_OWN_CAR=0.3)) is False

    def test_no_drift_does_not_fire(self):
        assert retrain.should_retrain(summary(0.0)) is False

    def test_missing_keys_do_not_raise(self):
        """A malformed summary must read as "no drift", never crash the monitor task."""
        assert retrain.should_retrain({}) is False

    @pytest.mark.parametrize(
        "malformed",
        [
            pytest.param({"drift_share": None, "drifted_columns": {}}, id="share-is-none"),
            pytest.param({"drift_share": 0.1, "drifted_columns": None}, id="columns-are-none"),
            pytest.param({"drift_share": "n/a", "drifted_columns": {}}, id="share-not-numeric"),
            pytest.param({"drift_share": None, "drifted_columns": None}, id="both-none"),
        ],
    )
    def test_present_but_null_fields_do_not_raise(self, malformed):
        """A key present and explicitly ``None`` is not the same as a key that is absent.

        The original guard used ``summary.get(key, default)``, which only covers the absent
        case -- ``float(None)`` and ``set(None)`` both raise. Caught in review on PR #6.
        By the time should_retrain() runs, the drift report and its Prometheus gauges are
        already written; raising here would fail the task and lose them to a retry.
        """
        assert retrain.should_retrain(malformed) is False


class TestIncumbentAuc:
    """``incumbent_auc`` must answer ``None`` rather than raise: every failure mode here
    should let a promotion through, not wedge the DAG."""

    @pytest.fixture(autouse=True)
    def no_tracking_setup(self, monkeypatch):
        """Stop the real ``configure_tracking`` touching mlflow.db."""
        monkeypatch.setattr(retrain, "configure_tracking", lambda: "stub://")

    def _client(self, monkeypatch, result):
        class StubClient:
            def get_model_version_by_alias(self, name, alias):
                if isinstance(result, Exception):
                    raise result
                return result

        monkeypatch.setattr(retrain, "MlflowClient", StubClient)

    def test_reads_the_tag_promote_best_writes(self, monkeypatch):
        version = type("V", (), {"version": "7", "tags": {retrain.AUC_TAG: "0.8412"}})()
        self._client(monkeypatch, version)
        assert retrain.incumbent_auc() == pytest.approx(0.8412)

    def test_missing_alias_is_none(self, monkeypatch):
        missing = MlflowException("no such alias")
        missing.error_code = "RESOURCE_DOES_NOT_EXIST"
        self._client(monkeypatch, missing)
        assert retrain.incumbent_auc() is None

    @pytest.mark.parametrize(
        "code", ["INTERNAL_ERROR", "TEMPORARILY_UNAVAILABLE", "PERMISSION_DENIED", None]
    )
    def test_registry_failures_propagate_instead_of_reading_as_empty(self, monkeypatch, code):
        """An outage is not an empty registry. Caught in review on PR #6.

        This is the most dangerous of the "treat it as absent" fallbacks, because the
        fallback is permissive in both directions: ``None`` means "no incumbent", which
        makes ``should_promote()`` approve unconditionally, and the promotion that follows
        replaces production without the AUC gate ever running. A tracking-server blip would
        have silently disabled the safety check this whole module exists to provide. Airflow
        should retry instead.
        """
        failure = MlflowException("server on fire")
        failure.error_code = code
        self._client(monkeypatch, failure)
        with pytest.raises(MlflowException):
            retrain.incumbent_auc()

    def test_version_without_the_tag_is_none(self, monkeypatch):
        version = type("V", (), {"version": "3", "tags": {}})()
        self._client(monkeypatch, version)
        assert retrain.incumbent_auc() is None

    @pytest.mark.parametrize("corrupt", ["not-a-number", "", "0.84.1"])
    def test_unreadable_tag_is_none(self, monkeypatch, corrupt):
        """A tag that will not parse must not wedge the DAG. Caught in review on PR #6.

        promote_best() writes this tag as a 4-decimal string, so an unparseable value means
        it was hand-edited in the MLflow UI. Raising here would fail task_evaluate at the
        exact moment the right answer is "no incumbent, let the promotion through".
        """
        version = type("V", (), {"version": "4", "tags": {retrain.AUC_TAG: corrupt}})()
        self._client(monkeypatch, version)
        assert retrain.incumbent_auc() is None

    @pytest.mark.parametrize("hostile", ["nan", "inf", "-inf", "1.5", "-0.2"])
    def test_unusable_but_parseable_tag_is_none(self, monkeypatch, hostile):
        """Parsing is not the same as being usable. Caught in review on PR #6.

        ``float()`` accepts "nan", "inf" and "-inf" without complaint, so the earlier
        try/except let them straight through. NaN is the dangerous one: every comparison
        against it is False, so ``should_promote`` would refuse every candidate for the rest
        of the project's life while the DAG kept reporting success. An AUC outside [0, 1]
        is meaningless on its own terms.
        """
        version = type("V", (), {"version": "5", "tags": {retrain.AUC_TAG: hostile}})()
        self._client(monkeypatch, version)
        assert retrain.incumbent_auc() is None

    def test_nan_incumbent_would_have_blocked_every_promotion(self):
        """Why the guard above matters, stated as the behaviour it prevents."""
        assert retrain.should_promote(0.99, float("nan")) is False
