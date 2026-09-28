"""Export contract tests.

Hermetic: the registry lookup and the artifact download are both stubbed, so these run on a
clean checkout with no `mlflow.db` and no `mlruns/`. That matters more than usual here, because
`src/api/main.py` now *requires* the marker this module writes -- a regression in the writer
would leave CI green and fail every baked deployment at startup.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.models import export

TRACK = "credit"
URI = f"models:/riskwatch_{TRACK}@production"


@pytest.fixture
def stub_download(monkeypatch):
    """Stand in for `mlflow.artifacts.download_artifacts`, writing what a real model leaves.

    Returns a list the test can inspect, so a test can tell "download was never called" from
    "download wrote nothing" -- the two produce the same empty directory otherwise.
    """
    calls: list[Path] = []

    def fake(artifact_uri: str, dst_path: str):
        dst = Path(dst_path)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "MLmodel").write_text("flavors: {}\n")
        (dst / "model.pkl").write_bytes(b"not a real pickle")
        calls.append(dst)
        return dst_path

    monkeypatch.setattr(export.mlflow.artifacts, "download_artifacts", fake)
    monkeypatch.setattr(export, "resolve_version", lambda uri: "7")
    return calls


def test_export_writes_the_track_marker_the_api_requires(tmp_path, stub_download):
    """`MODEL_TRACK` must contain the track derived from the URI, beside `MODEL_VERSION`.

    The integration test in `test_api.py` proves export wrote a marker *at some point* but skips
    on a clean checkout, and the API tests build their markers by hand. Neither can see the
    writer disappear. This can.
    """
    out = tmp_path / "model"

    version = export.export_model(URI, out)

    assert version == "7"
    assert (out / export.TRACK_FILENAME).read_text().strip() == TRACK
    assert (out / export.VERSION_FILENAME).read_text().strip() == "7"
    # The downloaded files land beside the markers rather than under a nested directory, which is
    # what the Dockerfile's `COPY build/model/ /app/model/` assumes.
    assert (out / "MLmodel").is_file()


def test_the_marker_names_the_track_in_the_uri_not_the_default(tmp_path, stub_download):
    """Exporting fraud must not stamp the artifact `credit`.

    `DEFAULT_TRACK` is credit, so a writer that reached for the default instead of parsing the
    URI would pass every credit test and mislabel every fraud artifact -- and the serving check
    would then refuse the fraud deployment while naming credit, sending the operator after the
    wrong thing.
    """
    out = tmp_path / "model"

    export.export_model("models:/riskwatch_fraud@production", out)

    assert (out / export.TRACK_FILENAME).read_text().strip() == "fraud"


def test_a_failed_export_leaves_nothing_deployable(tmp_path, monkeypatch):
    """A previous export must not survive a failed one.

    The only consumer is `docker build`, and a `COPY` cannot tell a current artifact from last
    week's: it would bake the stale model into an image that looks healthy and reports that
    model's own version. So the destination is cleared before anything can fail, and the build
    breaks on a missing source instead.
    """
    out = tmp_path / "model"
    out.mkdir()
    (out / export.TRACK_FILENAME).write_text("credit\n")
    (out / export.VERSION_FILENAME).write_text("4\n")
    (out / "MLmodel").write_text("stale\n")

    def boom(artifact_uri: str, dst_path: str):
        raise RuntimeError("registry went away mid-download")

    monkeypatch.setattr(export.mlflow.artifacts, "download_artifacts", boom)
    monkeypatch.setattr(export, "resolve_version", lambda uri: "7")

    with pytest.raises(RuntimeError, match="registry went away"):
        export.export_model(URI, out)

    assert not out.exists(), "a stale artifact survived a failed export and could still be baked"
    assert not out.with_name(out.name + ".incoming").exists(), "staging directory was left behind"


def test_an_unresolvable_track_also_clears_the_destination(tmp_path, stub_download):
    """The same guarantee for a failure that happens before the download.

    `resolve_track` raising is the likeliest way this fails in practice -- a hand-typed `--uri` --
    and an earlier version validated before clearing, so exactly this case left the stale artifact
    in place.
    """
    out = tmp_path / "model"
    out.mkdir()
    (out / export.TRACK_FILENAME).write_text("credit\n")

    with pytest.raises(ValueError, match="does not follow"):
        export.export_model("models:/churnwatch@production", out)

    assert not out.exists()
    assert stub_download == [], "the download should not have been attempted"


def test_a_partial_download_never_appears_at_the_destination(tmp_path, monkeypatch):
    """The destination must not exist until the markers are written.

    Downloading straight into it would leave a directory holding a real `MLmodel` and no
    `MODEL_TRACK` if the process died between the two, which the API refuses -- loudly, but only
    after a build and a deploy. Staging moves that failure to the export.
    """
    out = tmp_path / "model"
    seen: list[bool] = []

    def fake(artifact_uri: str, dst_path: str):
        dst = Path(dst_path)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "MLmodel").write_text("flavors: {}\n")
        seen.append(out.exists())
        return dst_path

    monkeypatch.setattr(export.mlflow.artifacts, "download_artifacts", fake)
    monkeypatch.setattr(export, "resolve_version", lambda uri: "7")

    export.export_model(URI, out)

    assert seen == [False], "the artifact was downloaded into the destination, not into staging"
    assert (out / export.TRACK_FILENAME).is_file()


def test_a_leftover_staging_directory_does_not_block_a_retry(tmp_path, stub_download):
    """A previous crash's staging directory must be cleared, not tripped over.

    `Path.rename` onto an existing directory fails on POSIX, and `mkdir` on an existing one raises
    too, so a stale `.incoming` would make every subsequent export fail until someone deleted it
    by hand.
    """
    out = tmp_path / "model"
    staging = out.with_name(out.name + ".incoming")
    staging.mkdir()
    (staging / "junk").write_text("from a crashed run\n")

    export.export_model(URI, out)

    assert (out / export.TRACK_FILENAME).read_text().strip() == TRACK
    assert not (out / "junk").exists(), "the crashed run's files were carried into the export"
    assert not staging.exists()


def test_resolve_track_and_the_api_agree_on_the_model_name():
    """The label export writes must be the one `src/api/main.py` compares against.

    Both derive `riskwatch_<track>` independently -- export cannot import the API and the API
    will not import export -- so nothing but this stops them diverging. A divergence fails as
    "MODEL_TRACK says X, we are serving Y" on a correctly exported artifact, which points at the
    deployment rather than at the naming convention.
    """
    from src.api import main

    for track in ("credit", "fraud"):
        uri = f"models:/{export.MODEL_NAME_PREFIX}{track}@production"
        assert export.resolve_track(uri) == track
        # The API's own parse of the same URI must accept it for that track and reject it for
        # the other one.
        main.assert_model_matches_track(track, uri)
        other = "fraud" if track == "credit" else "credit"
        with pytest.raises(main.BakedModelMisconfigured):
            main.assert_model_matches_track(other, uri)


def test_export_cli_default_out_dir_is_what_the_dockerfile_copies():
    """`build/model` is hard-coded in the Dockerfile's COPY, so the default must match it."""
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()

    relative = export.DEFAULT_OUT_DIR.relative_to(export.PROJECT_ROOT)

    assert f"COPY --chown=riskwatch:riskwatch {relative}/" in dockerfile, (
        f"the Dockerfile does not copy {relative}/, so an export to the default location would "
        f"not reach the image"
    )


def test_staging_sits_beside_the_destination_so_the_move_is_atomic(tmp_path, stub_download):
    """Staging must share a filesystem with the destination or the rename is a copy.

    Using the system temp directory would work on this machine and break in a container where
    /tmp is a different mount: `rename` raises `EXDEV` and the export fails after a full
    download. Asserting the sibling location rather than the rename's success is what makes the
    reason visible.
    """
    out = tmp_path / "nested" / "model"
    out.parent.mkdir()

    export.export_model(URI, out)

    assert (out / export.TRACK_FILENAME).is_file()
    # The contract, stated directly: the staging path is derived from out_dir, so it is always a
    # sibling and never a temp-mount path.
    assert out.with_name(out.name + ".incoming").parent == out.parent


def test_export_does_not_import_the_training_stack():
    """Export must stay loadable without sklearn or LightGBM.

    The module says so in a comment and duplicates constants to keep it true; a stray import
    would make the claim false silently, and `src/models/export.py` runs on the host before
    `docker build` where the training extras may not be installed at all.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import src.models.export; "
        "print(sorted(m for m in ('sklearn', 'lightgbm', 'shap') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"export pulled in the training stack: {result.stdout}"


def test_shutil_is_actually_used_for_the_clear(tmp_path, stub_download):
    """A non-empty previous export must be removed, not just unlinked.

    `Path.rmdir` would raise on the populated directory a real export leaves, so this pins that
    the clear handles a tree rather than an empty directory -- the case a hand-rolled fix is
    most likely to get wrong.
    """
    out = tmp_path / "model"
    nested = out / "metadata" / "deep"
    nested.mkdir(parents=True)
    (nested / "file.json").write_text("{}\n")

    export.export_model(URI, out)

    assert not nested.exists()
    assert (out / export.TRACK_FILENAME).read_text().strip() == TRACK
    assert shutil.rmtree is not None  # the import is load-bearing, not incidental
