"""Which artifact each track is served from, and whether it belongs to that track.

``model_uri_for`` resolution across baked paths, per-track overrides and the registry; the
``MODEL_TRACK`` marker that stops one track serving another's artifact; and the version a baked
artifact reports. Split out of ``tests/test_api.py``; the stubs live in ``tests/stubs.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.schemas import CREDIT_EXAMPLE_REQUEST, FRAUD_EXAMPLE_REQUEST
from src.features.specs import FEATURE_SPECS
from tests.stubs import StubModel, baked


def test_a_shared_baked_path_is_refused_for_every_track_when_two_are_enabled(monkeypatch):
    """Enabling a second track must not silently un-bake the first.

    Found in the Step 13 deploy rehearsal against a real container. The condition this replaces
    was ``if len(ENABLED_TRACKS) == 1 and MODEL_URI``, so adding a second track stopped the
    baked path being used **for every track**: an image carrying credit's artifact sent credit
    to ``models:/riskwatch_credit@production`` as well, a registry it cannot reach. Measured,
    both tracks failed and the process took **204.5 s** to exit 3.

    **Not a hang, and the distinction was checked rather than assumed.**
    ``mlflow.store.db.utils.MAX_RETRY_COUNT`` is 10 with sleeps ``0.1 * (2**n - 1)``, so 101.3 s
    per engine creation and one cycle per track; ``load_model`` then raises and the lifespan's
    per-track isolation runs as designed. An earlier draft of this docstring called it unbounded
    on the strength of a 45-second poll against a 204-second failure.

    So what the guard buys is 204.5 s -> 1.0 s plus a message naming the misconfiguration -- not
    the recovery of a protection that was never lost.

    **Both tracks are refused, including the one the artifact actually holds, and that is the
    correct answer rather than a limitation.** A bare path carries no claim about whose model
    it is. `build/model/registered_model_meta` happens to name one, but resolving ownership
    from it would make the serving contract depend on MLflow's artifact layout, and it would
    still be guessing which track the *operator* meant. An image with one artifact and two
    enabled tracks is a configuration that cannot work; serving half of it, chosen by the API,
    is worse than refusing it with a message that names both ways out.
    """
    monkeypatch.setattr(main, "MODEL_URI", "/app/model")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_CREDIT", raising=False)
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    for track in ("credit", "fraud"):
        with pytest.raises(main.ModelConfigurationError) as raised:
            main.model_uri_for(track)
        # The operator reading this is looking at a revision that will not start, so the
        # message has to name both exits rather than only the symptom.
        assert f"MODEL_URI_{track.upper()}" in str(raised.value)
        assert "ENABLED_TRACKS" in str(raised.value)


def test_the_misconfigured_deployment_refuses_to_start_immediately(monkeypatch):
    """No track resolves, so the process must exit -- and at resolution, not after 204 s.

    Distinct from the degraded path deliberately. ``degraded`` is for "a track I was told to
    serve is missing", which leaves something worth answering with. Here *nothing* resolves, and
    the lifespan's own rule is that every track failing stays fatal.

    The exit is not what changed -- the pre-guard configuration exited too, with the same code,
    after two 101.3 s MLflow retry cycles. What this pins is that the refusal happens during URI
    resolution, with no registry contacted, so the cost is a function call rather than 3.4
    minutes of billable startup. A test cannot observe the timing difference, so it observes the
    mechanism: ``model_uri_for`` raising is what the lifespan converts into the failure below.
    """
    monkeypatch.setattr(main, "MODEL_URI", "/app/model")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_CREDIT", raising=False)
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_two_baked_artifacts_serve_two_tracks(monkeypatch, log_file, tmp_path):
    """The supported both-tracks shape: one explicit path per track, no shared ``MODEL_URI``.

    This is what the rehearsal's measurement argues for -- the fraud artifact is 348 KB against
    a 363.5 MB dependency layer, so a second model is free and the deploy shape is "both". The
    test pins the wiring that makes it work, so the Dockerfile change landing later has a
    contract to satisfy rather than one to invent.

    Real directories rather than string paths, because each artifact now has to *declare* its
    track and both declarations are checked at startup.
    """
    credit_dir = baked(tmp_path / "credit", "credit")
    fraud_dir = baked(tmp_path / "fraud", "fraud")

    monkeypatch.setattr(main, "MODEL_URI", str(tmp_path))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_CREDIT", str(credit_dir))
    monkeypatch.setenv("MODEL_URI_FRAUD", str(fraud_dir))

    assert main.model_uri_for("credit") == str(credit_dir)
    assert main.model_uri_for("fraud") == str(fraud_dir)

    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))

    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["models"] == {"credit": "5", "fraud": "5"}

        assert client.post("/predict/credit", json=CREDIT_EXAMPLE_REQUEST).status_code == 200
        assert client.post("/predict/fraud", json=FRAUD_EXAMPLE_REQUEST).status_code == 200


def test_a_single_enabled_track_cannot_serve_another_tracks_artifact(monkeypatch, tmp_path):
    """``ENABLED_TRACKS=fraud`` against credit's baked artifact must refuse, not report ok.

    The hole this closes had the same shape as the two-track one and was left open by the same
    omission: ``model_uri_for`` returns a shared ``MODEL_URI`` unconditionally when one track is
    enabled, and a bare directory carries no claim about whose model it is. Measured against a
    real container before the check existed: ``/health`` reported
    ``{"status":"ok","models":{"fraud":"5"}}`` -- advertising another track's version -- and every
    ``POST /predict/fraud`` returned **500**, because MLflow's signature enforcement rejected a
    29-column fraud frame against a 26-column credit model.

    Worse, this service's own error message steered operators into it: the refusal for the credit
    track said "drop 'credit' from ENABLED_TRACKS", which leaves fraud alone and pointed at
    credit's directory.
    """
    artifact = baked(tmp_path / "model", "credit")

    monkeypatch.setattr(main, "MODEL_URI", str(artifact))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("fraud",))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    # Resolution still hands back the shared path: resolving and verifying are separate jobs.
    assert main.model_uri_for("fraud") == str(artifact)

    with pytest.raises(main.ModelConfigurationError) as raised:
        main.assert_model_matches_track("fraud", str(artifact))
    assert "'credit'" in str(raised.value), "the message must name what the artifact holds"
    assert "ENABLED_TRACKS=credit" in str(raised.value), "and the configuration that would match"

    # End to end: the startup refuses rather than reporting a healthy fraud deployment.
    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))
    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_an_unlabelled_baked_artifact_is_refused(monkeypatch, tmp_path):
    """No ``MODEL_TRACK`` is a failure, not a pass.

    "Unlabelled means trust the caller" is precisely the behaviour that shipped, so treating a
    missing marker as permission would leave the hole open for every artifact exported before the
    marker existed -- which is all of them.
    """
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "MODEL_VERSION").write_text("5\n")

    monkeypatch.setattr(main, "MODEL_URI", str(artifact))
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))

    with pytest.raises(main.ModelConfigurationError, match="does not say which track"):
        main.assert_model_matches_track("credit", str(artifact))


def test_a_matching_artifact_and_a_registry_uri_both_pass(monkeypatch, tmp_path):
    """The two shapes that are correct must not raise: a labelled match, and a matching name."""
    artifact = baked(tmp_path / "model", "credit")

    main.assert_model_matches_track("credit", str(artifact))
    main.assert_model_matches_track("fraud", "models:/riskwatch_fraud@production")
    # The version form takes a different branch of the same parse.
    main.assert_model_matches_track("fraud", "models:/riskwatch_fraud/3")


def test_a_registry_uri_naming_another_tracks_model_is_refused(monkeypatch):
    """``ENABLED_TRACKS=fraud`` with ``MODEL_URI=models:/riskwatch_credit@production`` must refuse.

    This is the identical defect to the baked-path one, one layer up, and it **shipped in the
    commit that claimed to fix that one.** The first version of
    ``assert_model_matches_track`` returned early for any ``models:/`` URI, reasoning that "the
    name in the URI *is* the claim" -- true, and worth nothing while nothing compared the claim to
    the track being served. Found by review, not by me.

    Reachable two ways, both plausible: a single-track deployment inherits the shared
    ``MODEL_URI`` unconditionally, and an explicit ``MODEL_URI_FRAUD`` can simply be pointed at
    the wrong registered model by hand.
    """
    monkeypatch.setattr(main, "MODEL_URI", "models:/riskwatch_credit@production")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("fraud",))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    # Resolution hands it back: a models:/ value is a legitimate single-track MODEL_URI, and
    # whether it is the *right* model is the verifier's question.
    assert main.model_uri_for("fraud") == "models:/riskwatch_credit@production"

    with pytest.raises(main.ModelConfigurationError) as raised:
        main.assert_model_matches_track("fraud", main.model_uri_for("fraud"))
    message = str(raised.value)
    assert "riskwatch_credit" in message, "the message must name the model the URI points at"
    assert "riskwatch_fraud" in message, "and the one this track expects"

    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))
    with pytest.raises(RuntimeError, match="no track loaded a model"), TestClient(main.app):
        pass


def test_the_documented_both_tracks_shape_refuses_a_missing_override(monkeypatch, tmp_path):
    """One local override present, the other forgotten, no shared ``MODEL_URI`` -- must refuse.

    **This is the configuration this repo recommends for two tracks**, which is what makes the gap
    worth a test rather than a footnote: one ``MODEL_URI_<TRACK>`` per track and no shared value.
    Forget one and, before this fix, that track fell through to
    ``models:/riskwatch_fraud@production`` -- a registry a baked image cannot reach -- buying back
    the 101.3 s retry cycle the guard exists to prevent.

    The first guard asked whether the *shared* ``MODEL_URI`` was a local path, so it saw nothing
    here. The question is whether **any** configured artifact path is local, which is what
    ``baked_paths()`` answers. Found by review, on the third pass over this function.
    """
    credit_dir = baked(tmp_path / "credit", "credit")

    monkeypatch.setattr(main, "MODEL_URI", "")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_CREDIT", str(credit_dir))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    assert main.model_uri_for("credit") == str(credit_dir)

    with pytest.raises(main.ModelConfigurationError) as raised:
        main.model_uri_for("fraud")
    message = str(raised.value)
    assert "MODEL_URI_FRAUD" in message
    assert "'credit'" in message, "the message must inventory what the deployment does hold"
    assert "models:/" not in message, (
        "the message must not suggest the registry, which is the fallback being refused"
    )

    # And the whole deployment refuses rather than serving credit while fraud 503s forever.
    monkeypatch.setattr(main, "load_model", lambda uri="": (StubModel(), "5"))
    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["models"] == {"credit": "5"}


def test_a_pure_registry_deployment_is_not_treated_as_baked(monkeypatch):
    """No local path anywhere means the registry fallback is correct and must still happen.

    The guard keys off ``baked_paths()`` being non-empty, so this pins the other side of that
    predicate: per-track ``models:/`` overrides, or none at all, leave every track resolving to its
    own registry URI. Without this, widening the guard would have broken the compose deployment
    and the DAG, neither of which has a baked artifact.
    """
    monkeypatch.setattr(main, "MODEL_URI", "")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_CREDIT", "models:/riskwatch_credit@production")
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    assert main.baked_paths() == {}
    assert main.model_uri_for("fraud") == "models:/riskwatch_fraud@production"

    monkeypatch.delenv("MODEL_URI_CREDIT", raising=False)
    assert main.baked_paths() == {}
    assert main.model_uri_for("credit") == "models:/riskwatch_credit@production"


def test_an_explicit_per_track_override_is_verified_too(monkeypatch):
    """``MODEL_URI_FRAUD`` pointing at credit's model must not be trusted because it is explicit.

    An explicit override is the operator being specific, not the operator being right, and this is
    the one path with no shared-``MODEL_URI`` ambiguity to blame -- so if verification only ran on
    the fallbacks, the most deliberate misconfiguration would be the one that got through.
    """
    monkeypatch.setattr(main, "MODEL_URI", "")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.setenv("MODEL_URI_FRAUD", "models:/riskwatch_credit@production")

    with pytest.raises(main.ModelConfigurationError, match="riskwatch_credit"):
        main.assert_model_matches_track("fraud", main.model_uri_for("fraud"))


def test_a_registry_deployment_resolves_each_track_to_its_own_model(monkeypatch, log_file):
    """A registry-backed ``MODEL_URI`` must fall through to the per-track default, not raise.

    Narrowing the guard to non-``models:/`` values is what keeps it from breaking a deployment
    shape it was not about: a ``models:/`` value is not a baked artifact, and the registry behind
    it holds every track, so resolving each track to its own registry URI is correct.

    **Not the current compose configuration**, which is worth stating because an earlier version
    of this docstring claimed it was. `docker-compose.yml` sets `MODEL_URI_CREDIT` and no
    `ENABLED_TRACKS`, so it serves credit alone and never reaches this branch. This is the
    manually configured two-track registry case — the shape a registry-backed both-tracks
    deployment would take, pinned before anything is wired to produce it.
    """
    monkeypatch.setattr(main, "MODEL_URI", "models:/riskwatch_credit@production")
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit", "fraud"))
    monkeypatch.delenv("MODEL_URI_FRAUD", raising=False)

    assert main.model_uri_for("fraud") == "models:/riskwatch_fraud@production"


def test_health_is_ok_when_only_one_of_the_two_tracks_is_enabled(monkeypatch, log_file):
    """``degraded`` means "something I was told to serve is missing", not "I serve one track".

    This is the complement of the test above and it is the half that became load-bearing
    when the second track registered. Both tracks now exist in the registry, so a
    single-track deployment is a real configuration -- W2 Step 15 decides whether the image
    carries one model or two off a measured size, and the 0.5 GB Artifact Registry budget
    may well decide it carries one.

    Reporting that deployment as ``degraded`` would make the signal useless in the direction
    that matters: Cloud Run's health check would flag a correctly configured revision, and an
    operator would learn to ignore the field that is supposed to tell them a model is down.
    """
    monkeypatch.setattr(main, "ENABLED_TRACKS", ("credit",))

    def only_credit(uri: str = ""):
        assert uri == main.model_uri_for("credit"), f"a disabled track was loaded from {uri}"
        return StubModel(), "test-1"

    monkeypatch.setattr(main, "load_model", only_credit)

    with TestClient(main.app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["models"] == {"credit": "test-1"}, "a disabled track must not appear as loaded"


def test_enabled_tracks_is_parsed_from_the_environment():
    """The switch is what defers the one-model-or-two deploy decision to a measurement.

    Parsed rather than assumed: the env var is a comma-separated list, and a deployment
    setting ``ENABLED_TRACKS="credit, fraud ,"`` -- a stray space and a trailing comma, both
    ordinary in a YAML env block -- must enable two tracks rather than one named ``" fraud"``
    resolving to ``models:/riskwatch_ fraud@production``.

    Run in a subprocess because the value is read at import time. ``importlib.reload`` is the
    obvious alternative and it does not work here: re-executing the module re-registers the
    Prometheus collectors and raises ``DuplicateTimeseries``, which is the reason those live
    at module level in the first place.
    """
    import os

    probe = "import src.api.main as m; print(m.ENABLED_TRACKS, m.model_uri_for('fraud'))"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "ENABLED_TRACKS": "credit, fraud ,"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "('credit', 'fraud')" in result.stdout, result.stdout
    assert "models:/riskwatch_fraud@production" in result.stdout, result.stdout


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_baked_model_path_reports_its_real_version(track):
    """A baked artifact must report its version, not "unknown".

    The container has no registry to ask, so export.py writes MODEL_VERSION next to the
    artifact. Without reading it every prediction and every /health response is stamped
    "unknown" -- and W3's deploy acceptance asks for a REAL version from the public URL,
    which is exactly this path.

    Exercised against the real exported artifact rather than a mock: the version file is
    written by export.py, so a test that stubs the loader would pass even if export stopped
    writing it. Skips when nothing has been exported yet, since that is a local-state
    precondition rather than a defect.
    """
    exported = Path(__file__).resolve().parents[1] / "build" / "model" / track
    if not (exported / "MLmodel").is_file():
        pytest.skip(f"no exported artifact at build/model/{track}; run `make image`")
    # Integration counterpart to test_baked_version_is_read_hermetically below, which runs
    # everywhere. Kept because only this one proves export.py actually writes the file.

    stamp = exported / "MODEL_VERSION"
    assert stamp.is_file(), "export.py must write MODEL_VERSION beside the artifact"

    _, version = main.load_model(str(exported))

    assert version == stamp.read_text().strip()
    assert version != "unknown", "a baked artifact must not serve predictions as 'unknown'"


@pytest.mark.parametrize("track", sorted(FEATURE_SPECS))
def test_baked_artifact_is_not_stale_against_the_alias(track):
    """The baked artifact must be the version ``@production`` points at, not an older one.

    The test above proves the baked version is *readable*; it cannot see whether it is
    *current*, because a stale export reports its own version perfectly confidently. That gap
    shipped: ``build/model`` held credit v4 while the alias had moved to v5, every test was
    green, and a ``docker build`` would have baked the wrong model into an image whose
    ``/health`` announced a real-looking version. Nothing compared the two numbers.

    It matters at the deploy rather than here. The image is built from this directory and the
    container has no registry to check itself against, so the last moment the comparison is
    possible is on the host, before the build -- and `src/models/export.py` resolving the alias
    correctly does not help when the export simply was not re-run.

    **Which registry gets compared is the resolver's business, not this test's.**
    ``resolve_version`` anchors tracking to ``PROJECT_ROOT/mlflow.db`` itself, so this reads the
    same backend the export wrote from regardless of the cwd pytest was launched in. Setting the
    URI here instead would have made the test pass while leaving the CLI free to inspect whatever
    database the caller happened to be standing in.

    Skips on the two local-state preconditions, each named: no export, or no registry entry to
    compare against. A version mismatch is a defect and fails.
    """
    from mlflow.exceptions import MlflowException

    from src.models import export
    from src.models.train import NOT_FOUND_CODES

    uri = f"models:/{export.MODEL_NAME_PREFIX}{track}@production"
    exported = Path(__file__).resolve().parents[1] / "build" / "model" / track
    stamp = exported / "MODEL_VERSION"
    # MLmodel as well as the stamp, matching the integration test above. A leftover or
    # half-written build/model can hold a version file and no loadable model, and comparing its
    # number against the alias would pass while the image has nothing to serve -- a green test
    # for a directory that cannot answer a request.
    if not (stamp.is_file() and (exported / "MLmodel").is_file()):
        pytest.skip(f"no exported artifact at build/model/{track}; run `make image`")

    try:
        current = export.resolve_version(uri)
    except MlflowException as exc:
        # The repo's definition of absence, not a narrower local one: a registry answering
        # ENDPOINT_NOT_FOUND for a missing alias would fail this test on a clean checkout instead
        # of taking the skip it is entitled to.
        if exc.error_code not in NOT_FOUND_CODES:
            raise
        pytest.skip(f"nothing registered at {uri} to compare against")

    baked = stamp.read_text().strip()
    assert baked == current, (
        f"build/model/{track} holds version {baked} but {uri} resolves to "
        f"{current}: re-run `make image`, or the "
        f"image serves a model the registry no longer promotes"
    )


def test_baked_version_is_read_hermetically(tmp_path, monkeypatch):
    """The same behaviour as the integration test above, but runs in CI.

    ``build/model`` is gitignored and the suite performs no export, so the real-artifact test
    skips in a clean checkout -- a regression in the local-path branch of ``load_model``
    would leave CI green. This stubs only the loader, so every line of the version-resolution
    path still executes.
    """
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "MODEL_VERSION").write_text("11\n")

    monkeypatch.delenv("MODEL_VERSION", raising=False)
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    _, version = main.load_model(str(artifact))

    assert version == "11", "the baked version file must win over the 'unknown' default"


def test_two_baked_artifacts_each_report_their_own_version(tmp_path, monkeypatch):
    """One process, two baked tracks: each version comes from its own artifact.

    A process-wide ``MODEL_VERSION`` used to win over the artifact's file. With one baked track
    the two agreed; with two, the image built with ``--build-arg MODEL_VERSION=6`` would have
    stamped fraud as version 6 as well. The variable is set here to prove it is ignored.
    """
    credit = baked(tmp_path / "credit", "credit", version="6")
    fraud = baked(tmp_path / "fraud", "fraud", version="1")

    monkeypatch.setenv("MODEL_VERSION", "6")
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    assert main.load_model(str(credit))[1] == "6"
    assert main.load_model(str(fraud))[1] == "1"


def test_baked_path_without_a_version_file_reports_unknown(tmp_path, monkeypatch):
    """Absent is reported honestly rather than guessed at."""
    artifact = tmp_path / "model"
    artifact.mkdir()

    monkeypatch.delenv("MODEL_VERSION", raising=False)
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: StubModel())

    _, version = main.load_model(str(artifact))

    assert version == "unknown"
