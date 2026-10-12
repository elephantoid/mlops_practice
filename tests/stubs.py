"""Stand-ins for the served model, shared by the API test modules.

A plain module rather than ``conftest.py`` because these are imported by name, and pytest
discourages importing from a conftest. The fixtures that use them are in ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.api import main

STUB_PROBABILITY = 0.73


def example_frame(track: str = "credit") -> pd.DataFrame:
    """A one-row frame shaped like ``track``'s contract, standing in for a logged input example.

    Track-shaped rather than arbitrary because ``StubModel.predict`` asserts its input matches
    some track's column list -- the assertion that catches an alias-mapping drift. An example
    that did not match would break every test that scores through the stub, for a reason having
    nothing to do with what those tests are about.
    """
    return pd.DataFrame([{column: 0.0 for column in main.FEATURE_COLUMNS[track]}])


class StubModel:
    """Stands in for the pyfunc model, returning [[p_negative, p_positive]] like the real one.

    The assertions are the point, not the return value. ``reindex`` in the handler fills any
    column it cannot find with NaN, so a one-character typo in a ``serialization_alias``
    would produce a NaN feature, get quietly imputed by the pipeline, and score normally --
    passing every other test in this file while serving wrong predictions.

    **It carries an ``input_example`` because the lifespan scores one.** Without it, every
    startup test in this file routed ``assert_model_can_score`` through its skip-and-warn branch,
    so deleting the lifespan's call to it left the suite green -- the check written to stop a
    healthy-looking service from 500ing every request was itself unguarded.
    """

    def __init__(self, track: str = "credit") -> None:
        self.input_example = example_frame(track)

    def predict(self, features):
        # Matched against *some* track's contract rather than one hardcoded list, because the
        # same stub now serves both endpoints. Asserting membership rather than equality to
        # credit's columns is what lets the NaN check below stay meaningful for fraud: a fraud
        # request reindexed onto credit's 26 columns would be 29 NaNs, and this is the assertion
        # that sees it.
        columns = list(features.columns)
        assert columns in main.FEATURE_COLUMNS.values(), (
            f"alias mapping drifted: {len(columns)} columns matching no track's contract"
        )
        assert not features.isna().any().any(), "a feature arrived as NaN"
        return np.array([[1 - STUB_PROBABILITY, STUB_PROBABILITY]] * len(features))


class ExplodingModel:
    """A model that passes the startup smoke score and then fails on live traffic.

    The tests using it are about the 500 being *counted*, which needs the request to reach the
    handler -- so it has to survive `assert_model_can_score`. Scoring the logged example and
    failing on anything else is not a contrivance: a lazily-initialised resource or a
    row-dependent bug behaves exactly like this, and it is the only failure mode a load-time smoke
    check cannot catch. Worth having a test that says so.
    """

    def __init__(self, track: str = "credit") -> None:
        self.input_example = example_frame(track)

    def predict(self, features):
        if features.equals(self.input_example):
            return np.array([[0.5, 0.5]])
        raise RuntimeError("model is broken")


def baked(dir_path: Path, track: str, version: str = "5") -> Path:
    """A minimal stand-in for an exported artifact: the two files the API reads off disk."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "MODEL_TRACK").write_text(f"{track}\n")
    (dir_path / "MODEL_VERSION").write_text(f"{version}\n")
    return dir_path
