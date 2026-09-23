"""Decision rules for the weekly retraining DAG.

These live in ``src/`` rather than beside the DAG on purpose. ``CLAUDE.md`` forbids
installing Airflow into the uv venv, so anything imported by ``dags/riskwatch_retrain.py``
is unreachable from ``tests/``. Keeping the judgements here and the wiring there is the
only arrangement in which the judgements can be tested at all.

Two questions are answered here, and nothing else:

* Has the model just trained earned the production alias?
* Has the data moved enough to be worth retraining on?
"""

from __future__ import annotations

import logging
import math
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.train import DEFAULT_TRACK, PRODUCTION_ALIAS, configure_tracking, model_name_for

logger = logging.getLogger(__name__)

# AGENTS.md: promote only when the candidate beats the incumbent by more than this. A
# delta rather than a bare comparison because cross-validated AUC wobbles by a few
# thousandths between identical runs -- promoting on noise churns the production alias
# weekly while changing nothing.
MIN_AUC_DELTA = 0.01

# The columns whose drift actually changes the decision, per track. Named rather than
# counted: see the comment on RETRAIN_SHARE_THRESHOLD for why a share alone cannot work.
#
# Retargeted from the Telco set (MonthlyCharges, tenure, Contract) when the project moved
# to credit and fraud. Leaving those names in place would have been worse than a stale
# comment: none of them exists in either track's frame, so the watched arm could never
# match a column and the trigger would have been silently dead -- while still reading as
# implemented.
#
# Credit: the loan size and the strongest external score, which is what the default
# decision actually turns on, plus income. Fraud: transaction amount and the two PCA
# components that carry the most separation in the ULB set.
WATCHED_COLUMNS_BY_TRACK: dict[str, tuple[str, ...]] = {
    "credit": ("AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"),
    "fraud": ("Amount", "V14", "V17"),
}

# The default track's watched columns, for callers with no opinion about tracks.
WATCHED_COLUMNS = WATCHED_COLUMNS_BY_TRACK[DEFAULT_TRACK]


def watched_columns(track: str = DEFAULT_TRACK) -> tuple[str, ...]:
    """Watched columns for one track, raising on an unknown one.

    Raises rather than returning an empty tuple: an empty watch list disables the arm that
    carries the real signal, and it would do so silently.
    """
    try:
        return WATCHED_COLUMNS_BY_TRACK[track]
    except KeyError:
        known = ", ".join(sorted(WATCHED_COLUMNS_BY_TRACK))
        raise KeyError(
            f"no watched columns configured for track {track!r}; known: {known}"
        ) from None


# The catch-all arm, kept at the value AGENTS.md specifies. It is deliberately NOT the
# only arm. The share a single drifted column produces depends on the frame width -- on
# Telco's 19 features it was 0.0526, on credit's 26 modeled columns it is 0.0385 -- so
# 0.20 has always meant "several columns at once" and can never be reached by the
# single-column drift scenario this project ships. A rule that cannot fire on the one case
# the repo can demonstrate is not a rule, so the watched-column arm carries the real
# signal and this one only catches broad, many-column shift.
#
# The 0.20 figure is itself Telco-derived and is logged as re-derivation debt in
# docs/debt-ledger.md; it has not yet been re-measured against credit or fraud.
RETRAIN_SHARE_THRESHOLD = 0.20

# The tag promote_best() writes on every model version it promotes. Reading it back is
# cheaper and more honest than re-scoring the incumbent: it is the number the incumbent
# was actually selected on.
AUC_TAG = "cv_auc_mean"

# MLflow reports a genuinely missing model, version or alias with one of these. Anything
# else -- 5xx, auth, transport -- is a failure and must not be mistaken for absence.
_NOT_FOUND_CODES = frozenset({"RESOURCE_DOES_NOT_EXIST", "ENDPOINT_NOT_FOUND"})


def _is_not_found(exc: MlflowException) -> bool:
    """Is this MlflowException a confirmed "it does not exist", rather than a failure?

    The distinction matters more than it looks: everywhere this module treats an exception
    as "absent", the fallback is permissive. Mistaking an outage for an absence turns a
    safety gate off silently.
    """
    return getattr(exc, "error_code", None) in _NOT_FOUND_CODES


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
    watched: tuple[str, ...] | None = None,
    share_threshold: float = RETRAIN_SHARE_THRESHOLD,
    track: str = DEFAULT_TRACK,
) -> bool:
    """Decide whether a drift summary warrants retraining.

    Takes the dict :func:`src.monitoring.drift.run` already returns, so the monitoring
    module needs no changes to support this.

    Either arm can fire: any *watched* column drifting, or the overall share clearing
    ``share_threshold``. The first catches a targeted shift in something that matters; the
    second catches broad movement across columns nobody thought to watch.
    """
    resolved_watched = watched if watched is not None else watched_columns(track)

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

    hits = sorted(drifted.intersection(resolved_watched))
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
    model_name: str | None = None,
    alias: str = PRODUCTION_ALIAS,
    track: str = DEFAULT_TRACK,
) -> float | None:
    """Read the cross-validated AUC of the model currently holding ``@alias``.

    ``model_name`` defaults to the track's registered name rather than a module constant:
    there are two registered models with independent retrain cadence, so a single default
    could only ever name one of them, and the DAG would gate the fraud track's promotion
    against the credit track's incumbent.

    ``None`` means "no comparable incumbent" and is a normal answer, not an error: an
    empty registry, a version promoted before the tag existed, or a tag that cannot be
    read as a number. All of them should let a promotion through rather than block one,
    so none of them raises -- the alternative is wedging the retrain exactly when it
    needs to fall back to "no incumbent".
    """
    configure_tracking()
    resolved_name = model_name or model_name_for(track)
    try:
        version = MlflowClient().get_model_version_by_alias(resolved_name, alias)
    except MlflowException as exc:
        if not _is_not_found(exc):
            # Only a confirmed absence means "no incumbent". A tracking-server outage, an
            # auth failure or a transport error would otherwise be reported as an empty
            # registry, should_promote() would approve unconditionally, and the next
            # promote_best() would replace production without the AUC gate ever running.
            # Let Airflow retry instead.
            raise
        logger.info("No @%s alias on %s; treating as no incumbent", alias, resolved_name)
        return None

    raw = version.tags.get(AUC_TAG)
    if raw is None:
        logger.warning(
            "%s v%s holds @%s but carries no %s tag; treating as no incumbent",
            resolved_name,
            version.version,
            alias,
            AUC_TAG,
        )
        return None

    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        # The tag is written by promote_best() as a 4-decimal string, so a value that will
        # not parse means it was edited by hand in the MLflow UI or otherwise corrupted.
        # Loud in the log, harmless to the run.
        logger.warning(
            "%s v%s has an unreadable %s tag (%r); treating as no incumbent",
            resolved_name,
            version.version,
            AUC_TAG,
            raw,
        )
        return None

    # float() happily accepts "nan", "inf" and "-inf", so parsing is not the same as being
    # usable. A NaN incumbent is the worst case: every comparison against it is False, so
    # should_promote() would refuse every candidate for the rest of the project's life and
    # the DAG would look like it was working. An AUC outside [0, 1] is meaningless too.
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        logger.warning(
            "%s v%s has an out-of-range %s tag (%r); treating as no incumbent",
            resolved_name,
            version.version,
            AUC_TAG,
            raw,
        )
        return None

    return parsed
