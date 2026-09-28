"""Structural comparison of a raw frame's column set against a committed manifest.

Track-agnostic, and that is the whole point of it living here rather than inside a track
module. The check has no domain content: it reads a list of expected names, compares it
against what arrived, and names what moved. Home Credit's 122 columns and ULB's 31 are the
same problem.

W1 wrote this inside :mod:`src.data.credit` because there was only one track. Adding the
second one made the duplication explicit -- two copies of a ~45-line comparison, two places
to fix a message, and two chances for them to disagree about what "renamed" means. Moved
here instead, with both track modules keeping a thin wrapper that supplies their own
manifest as the default so ``ingest()``'s ``module.assert_fingerprint(frame, path=...)``
contract is unchanged.

``path`` is **required** here, with no module-level default. That is deliberate: a shared
default manifest is precisely the defect ingest.py's comment warns about -- every track
fingerprinting against the credit column list and passing or failing for reasons having
nothing to do with its own data. A required argument makes that mistake unrepresentable in
the shared layer; the per-track default lives with the per-track manifest.

What this catches: added, dropped and renamed columns. What it does **not** catch: a dtype
change, a unit change, a semantic change under a stable name, a null-rate jump, a recode,
or truncation. Those need the data-quality rules that land with the drift decomposer.
Overselling this as a total upstream detector would make DoD (6)'s upstream cause class look
stronger than it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SchemaFingerprintError(RuntimeError):
    """The raw frame's column set does not match the committed manifest.

    Separate from a pandera failure because the two mean different things: a pandera
    violation is bad *data* in a known shape, this is a changed *shape*. Upstream renaming
    a column is not a row-level problem and the fix is not to clean the batch.
    """


def read_manifest(path: Path) -> list[str]:
    """Read a committed column manifest, preserving order.

    Raises rather than returning an empty list: an empty manifest would make
    :func:`assert_column_manifest` a no-op that still reports success, which is worse than
    having no fingerprint at all.
    """
    if not path.is_file():
        raise SchemaFingerprintError(
            f"Column manifest missing at {path}. It is written from the archive during "
            f"ingest setup; without it the structural check cannot run."
        )
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise SchemaFingerprintError(f"Column manifest at {path} is empty.")
    return names


def assert_column_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Compare ``frame``'s full column set against the manifest at ``path``.

    Names the specific columns that moved. "The schema changed" sends whoever reads it
    diffing 122 names by hand; "ORGANIZATION_TYPE was renamed" does not.

    Order is deliberately not checked. Column order is not a contract anyone upstream
    promised, and a reorder breaks nothing downstream because every consumer selects by
    name.
    """
    expected = set(read_manifest(path))
    actual = set(frame.columns)

    missing = sorted(expected - actual)
    added = sorted(actual - expected)

    if not missing and not added:
        return

    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if added:
        parts.append(f"unexpected: {added}")

    # A same-size swap is the renaming case, and saying so saves the reader the inference.
    hint = ""
    if missing and added and len(missing) == len(added):
        hint = " (same count missing and added -- likely a rename upstream)"

    raise SchemaFingerprintError(
        f"Raw column set does not match {path.name}: {'; '.join(parts)}{hint}. "
        f"Expected {len(expected)} columns, got {len(actual)}."
    )
