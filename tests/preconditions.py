"""Skip on a missing model locally; fail on it in CI.

The skew test and the baked-artifact tests need a registered model and an export, which a fresh
clone does not have, so locally they skip and say why. CI trains one first. There, the same skip
would mean the training or export step silently did not produce what the next step reads -- a
wrong ``MLFLOW_TRACKING_URI`` in one step is enough -- and the only check that scores the real
model would go green by not running. ``RISKWATCH_REQUIRE_MODELS=1`` turns those skips into
failures.

Only the model-dependent skips use this. The ones that need the real competition data stay skips
everywhere: CI cannot have that data, by licence.
"""

from __future__ import annotations

import os

import pytest


def skip_or_fail(reason: str) -> None:
    """Skip with ``reason``, or fail with it when ``RISKWATCH_REQUIRE_MODELS`` is set."""
    if os.environ.get("RISKWATCH_REQUIRE_MODELS") == "1":
        pytest.fail(f"RISKWATCH_REQUIRE_MODELS=1 but {reason}")
    pytest.skip(reason)
