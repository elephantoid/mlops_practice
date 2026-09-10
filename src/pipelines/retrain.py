"""Decision rules for the weekly retraining DAG.

These live in ``src/`` rather than beside the DAG on purpose. ``CLAUDE.md`` forbids
installing Airflow into the uv venv, so anything imported by ``dags/churnwatch_retrain.py``
is unreachable from ``tests/``. Keeping the judgements here and the wiring there is the
only arrangement in which the judgements can be tested at all.

Two questions are answered here, and nothing else:

* Has the model just trained earned the production alias?
* Has the data moved enough to be worth retraining on?
"""

from __future__ import annotations

import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.train import MODEL_NAME, PRODUCTION_ALIAS, configure_tracking

logger = logging.getLogger(__name__)

# AGENTS.md: promote only when the candidate beats the incumbent by more than this. A
# delta rather than a bare comparison because cross-validated AUC wobbles by a few
# thousandths between identical runs -- promoting on noise churns the production alias
# weekly while changing nothing.
MIN_AUC_DELTA = 0.01

# The columns whose drift actually changes the churn decision. Named rather than counted:
# see the module comment on RETRAIN_SHARE_THRESHOLD for why a share alone cannot work here.
WATCHED_COLUMNS = ("MonthlyCharges", "tenure", "Contract")

# The catch-all arm, kept at the value AGENTS.md specifies. It is deliberately NOT the
# only arm. With 19 features one drifted column is a share of 0.0526, so 0.20 means "4 or
# more columns at once" -- and the drift scenario this project ships (MonthlyCharges +15%,
# a single column) can never reach it. A rule that cannot fire on the one case the repo
# can demonstrate is not a rule, so the watched-column arm below carries the real signal
# and this one only catches broad, many-column shift.
RETRAIN_SHARE_THRESHOLD = 0.20

# The tag promote_best() writes on every model version it promotes. Reading it back is
# cheaper and more honest than re-scoring the incumbent: it is the number the incumbent
# was actually selected on.
AUC_TAG = "cv_auc_mean"


def should_promote(
    candidate_auc: float,
    incumbent_auc: float | None,
    min_delta: float = MIN_AUC_DELTA,
) -> bool:
    """Decide whether ``candidate_auc`` deserves the production alias.

    With no incumbent -- a fresh registry, or the very first DAG run -- promotion is
    unconditional: some model has to be first, and refusing here would leave the API with
    nothing to serve forever.
    """
    if incumbent_auc is None:
        logger.info("No incumbent in the registry; promoting candidate at %.4f", candidate_auc)
        return True

    # Rounded before comparing: 0.85 - 0.84 is 0.010000000000000009 in binary floating
    # point, which clears a 0.01 threshold on representation error alone. Six places is far
    # finer than the inputs justify anyway -- promote_best() stores cv_auc_mean as a
    # 4-decimal string, so the incumbent is only known to that precision.
    delta = round(candidate_auc - incumbent_auc, 6)
    verdict = delta > min_delta
    logger.info(
        "candidate %.4f vs incumbent %.4f | delta %+.4f (need > %.4f) -> %s",
        candidate_auc,
        incumbent_auc,
        delta,
        min_delta,
        "promote" if verdict else "keep incumbent",
    )
    return verdict


def should_retrain(
    summary: dict[str, Any],
    watched: tuple[str, ...] = WATCHED_COLUMNS,
    share_threshold: float = RETRAIN_SHARE_THRESHOLD,
) -> bool:
    """Decide whether a drift summary warrants retraining.

    Takes the dict :func:`src.monitoring.drift.run` already returns, so the monitoring
    module needs no changes to support this.

    Either arm can fire: any *watched* column drifting, or the overall share clearing
    ``share_threshold``. The first catches a targeted shift in something that matters; the
    second catches broad movement across columns nobody thought to watch.
    """
    # `or {}` rather than a default argument: the key can be present and explicitly None,
    # which a default only covers when the key is absent entirely.
    drifted = set(summary.get("drifted_columns") or {})

    raw_share = summary.get("drift_share")
    try:
        share = float(raw_share) if raw_share is not None else 0.0
    except (TypeError, ValueError):
        # Read as "no drift" rather than raising. By the time this runs the report and its
        # gauges are already written; crashing here would fail the monitor task, lose that
        # work to a retry, and turn a malformed number into an outage.
        logger.warning("Unreadable drift_share %r; treating as no drift", raw_share)
        share = 0.0

    hits = sorted(drifted.intersection(watched))
    if hits:
        logger.info("Retrain: watched column(s) drifted -- %s", ", ".join(hits))
        return True

    if share > share_threshold:
        logger.info("Retrain: drift share %.4f exceeds %.4f", share, share_threshold)
        return True

    logger.info(
        "No retrain: share %.4f <= %.4f, no watched column among %s",
        share,
        share_threshold,
        ", ".join(sorted(drifted)) or "none drifted",
    )
    return False


def incumbent_auc(
    model_name: str = MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> float | None:
    """Read the cross-validated AUC of the model currently holding ``@alias``.

    ``None`` means "no comparable incumbent" and is a normal answer, not an error: an
    empty registry, or a version promoted before the tag existed. Both should let a
    promotion through rather than block one, so neither raises.
    """
    configure_tracking()
    try:
        version = MlflowClient().get_model_version_by_alias(model_name, alias)
    except MlflowException:
        logger.info("No @%s alias on %s; treating as no incumbent", alias, model_name)
        return None

    raw = version.tags.get(AUC_TAG)
    if raw is None:
        logger.warning(
            "%s v%s holds @%s but carries no %s tag; treating as no incumbent",
            model_name,
            version.version,
            alias,
            AUC_TAG,
        )
        return None

    return float(raw)
