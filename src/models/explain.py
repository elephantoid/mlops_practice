"""Per-request reason codes: LightGBM's own TreeSHAP, read back through the feature contract.

**Not the ``shap`` package, and that is not an approximation.** For a LightGBM model,
``shap.TreeExplainer`` computes its values by calling ``booster.predict(X, pred_contrib=True)``
(``shap/explainers/_tree.py``, the ``lightgbm`` branch). Calling it directly gives the same
numbers -- measured identical, bias column included -- and leaves out ``shap``'s dependency on
``numba`` and ``llvmlite``, 193 MB of the serving image for a wrapper around one function call.

The contributions are in **log-odds**, the only space where they add up: the base value plus
every feature's contribution is the model's raw score, and its sigmoid is the served
probability. :func:`reason_codes` checks that identity on every request, because a reason
computed on a different frame from the one that was scored is worse than no reason.

The model is the unwrapped sklearn ``Pipeline`` (``PyFuncModel.get_raw_model()``), since pyfunc
exposes only ``predict``. Everything before the classifier -- the sentinel step *and* the
preprocessor -- is the transform. Transforming with the preprocessor alone skips the sentinel
step, and the sentinel population (about 18% of credit applicants) then explains a probability
the model never served.

Imports numpy and pandas only. The fitted estimator brings sklearn and LightGBM with it when the
model is loaded; nothing here needs to import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.specs import FeatureSpec

TOP_N = 3

# In probability space. The identity holds to float rounding -- measured 1e-15 -- so anything
# this size is a real disagreement between the explained frame and the scored one.
ADDITIVITY_TOLERANCE = 1e-9


class ExplanationError(RuntimeError):
    """A model cannot be explained, or an explanation does not add up to what was served."""


@dataclass(frozen=True)
class Explainer:
    """Everything needed to explain one track's model, resolved once at startup."""

    spec: FeatureSpec
    transform: Any
    model: Any

    @property
    def columns(self) -> tuple[str, ...]:
        return self.spec.feature_columns


def build_explainer(pipeline: Any, spec: FeatureSpec) -> Explainer:
    """Split ``pipeline`` into transform and model, and refuse what cannot be read back.

    Two refusals, both about mapping a contribution to a column by position. The model must be
    a LightGBM booster, since ``pred_contrib`` is what produces the contributions. And the
    transform's output names, prefixes stripped, must be the contract's columns **in order**:
    the logreg arm one-hot encodes, so its output is wider than the contract, and a reordered
    contract would attach every contribution to the wrong name without any error at all.
    """
    transform, model = pipeline[:-1], pipeline[-1]

    if not hasattr(model, "booster_"):
        raise ExplanationError(
            f"cannot explain a {type(model).__name__}: reason codes come from LightGBM's "
            f"pred_contrib, and only tree models are promoted"
        )

    produced = [name.split("__", 1)[-1] for name in transform.get_feature_names_out()]
    if produced != list(spec.feature_columns):
        raise ExplanationError(
            f"cannot explain this pipeline: its output columns do not match the contract's "
            f"feature order, so contributions would be named after the wrong features. "
            f"Produced {produced}, contract {list(spec.feature_columns)}"
        )

    return Explainer(spec=spec, transform=transform, model=model)


def contributions(explainer: Explainer, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature log-odds contributions, one row per row of ``frame``, and the base value.

    ``pred_contrib`` returns one column per feature plus a final column holding the expected
    value, so the split is positional.
    """
    raw = explainer.model.predict(explainer.transform.transform(frame), pred_contrib=True)
    raw = np.asarray(raw, dtype=float)
    return raw[:, :-1], raw[:, -1]


def top_indices(phi: np.ndarray, n: int) -> list[int]:
    """The ``n`` largest contributions by magnitude, ties broken by contract order.

    A stable sort on ``-|phi|`` keeps equal magnitudes in column order, so two identical
    requests can never list the same reasons in a different sequence.
    """
    return [int(i) for i in np.argsort(-np.abs(phi), kind="stable")[:n]]


def reason_codes(explainer: Explainer, row: pd.DataFrame, probability: float) -> list[dict]:
    """The top three reasons behind ``probability``, which must be what ``row`` was scored at.

    Raises :class:`ExplanationError` if the contributions do not reconstruct ``probability``.
    The API turns that into a 500 rather than an empty list: credit returning ``[]`` would be
    indistinguishable from fraud, whose ``[]`` is permanent and means "there is nothing to
    explain", not "the explanation failed".
    """
    phi, base = contributions(explainer, row)
    phi, base = phi[0], float(base[0])

    reconstructed = 1.0 / (1.0 + np.exp(-(phi.sum() + base)))
    if abs(reconstructed - probability) > ADDITIVITY_TOLERANCE:
        raise ExplanationError(
            f"contributions do not reconstruct the served probability: "
            f"sigmoid(sum) = {reconstructed!r}, served {probability!r}"
        )

    spec, values = explainer.spec, row.iloc[0]
    codes = []
    for i in top_indices(phi, TOP_N):
        column = explainer.columns[i]
        codes.append(
            {
                "feature": spec.display_name(column),
                "contribution": float(phi[i]),
                "value": spec.display_value(column, values[column]),
            }
        )
    return codes
