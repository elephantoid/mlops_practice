"""The DAG's calls into ``src``, checked by reading it.

Airflow is not in the venv, so no test can import ``dags/`` and a wrong call there surfaces
only when the scheduler runs the task. 9446f7c found six calls that left the track to each
callee's default -- right only while the one DAG retrains credit -- and missed ``ingest()``
and the monitor's fallback baseline, which stayed on credit until this file existed.

So every call to a function imported from ``src`` is bound against that function's real
signature, and a callee that takes ``track`` must be handed the DAG's ``TRACK``: not left to
a default, and not a literal that a second DAG stamped out for fraud would carry over.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import types
from pathlib import Path

import pytest

from src.data.tracks import PROJECT_ROOT

DAG_FILES = sorted((PROJECT_ROOT / "dags").glob("*.py"))
TRACK_NAME = "TRACK"
# Values, not functions, the DAG may import from src. Anything else is refused: a constant
# can carry a track baked in -- drift.REFERENCE_PATH was credit's baseline -- and a value has
# no signature to check. Add one here only if no track went into it.
TRACK_FREE_CONSTANTS = {"DRIFT_SOURCES"}


def wiring_problems(source: str) -> tuple[list[str], set[str]]:
    """Problems with ``source``'s calls into ``src``, and the names of the callables checked."""
    tree = ast.parse(source)
    imported: dict[str, object] = {}
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "src":
                    # Calls through a module attribute would not be checked below.
                    problems.append(f"line {node.lineno}: import functions from {alias.name}")
            continue
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src")):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            name = alias.asname or alias.name
            if not hasattr(module, alias.name):
                problems.append(f"line {node.lineno}: {node.module} has no {alias.name}")
            elif isinstance(getattr(module, alias.name), types.ModuleType):
                problems.append(f"line {node.lineno}: import functions from {name}, not the module")
            elif not callable(value := getattr(module, alias.name)) and (
                alias.name not in TRACK_FREE_CONSTANTS
            ):
                problems.append(f"line {node.lineno}: {alias.name} is a value; call for it instead")
            else:
                imported[name] = value

    checked: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        fn = imported.get(node.func.id)
        if fn is None or inspect.isclass(fn) or not callable(fn):
            continue
        checked.add(node.func.id)
        call = f"line {node.lineno}: {ast.unparse(node)}"
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            problems.append(f"{call} -- */** arguments cannot be checked")
            continue
        try:
            bound = inspect.signature(fn).bind(
                *node.args, **{k.arg: k.value for k in node.keywords if k.arg}
            )
        except TypeError as exc:
            problems.append(f"{call} -- {exc}")
            continue
        if "track" in bound.signature.parameters:
            given = bound.arguments.get("track")
            if not (isinstance(given, ast.Name) and given.id == TRACK_NAME):
                problems.append(f"{call} -- the track must be {TRACK_NAME}")
    return problems, checked


@pytest.mark.parametrize("dag", DAG_FILES, ids=lambda p: p.name)
def test_every_call_from_the_dag_into_src_binds_and_names_the_dags_track(dag: Path):
    problems, checked = wiring_problems(dag.read_text())

    assert not problems, "\n".join(problems)
    assert checked, f"{dag.name} calls nothing from src -- the check would be vacuous"


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        # 9446f7c's shape. Its callees had defaults; now a missing track no longer binds.
        ("from src.models.train import sweep\nsweep(p)", "missing a required argument: 'track'"),
        ("from src.data.ingest import ingest\ningest('credit')", "the track must be TRACK"),
        ("from src.models.train import sweep\nsweep(p, nope=1, track=TRACK)", "unexpected"),
        ("from src.models.train import model_name_for", "has no model_name_for"),
        ("from src.models import train", "not the module"),
        ("import src.models.train as t\nt.sweep(p)", "import functions from src.models.train"),
        ("from src.data.tracks import PROJECT_ROOT", "is a value"),
        ("from src.data.tracks import get_track\nget_track('fraud')", "the track must be TRACK"),
        ("from src.models.train import sweep\nsweep(*a, track=TRACK)", "cannot be checked"),
    ],
    ids=[
        "default-track",
        "literal-track",
        "bad-keyword",
        "deleted-symbol",
        "module",
        "module-import",
        "constant",
        "literal-lookup",
        "starred",
    ],
)
def test_the_wiring_check_rejects_each_kind_of_unchecked_call(snippet: str, expected: str):
    problems, _ = wiring_problems(snippet)

    assert len(problems) == 1 and expected in problems[0], problems
