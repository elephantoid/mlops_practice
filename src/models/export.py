"""Export the production model from the registry into a plain directory.

Run from the repo root before ``docker build``::

    uv run python -m src.models.export

This exists because the container loads its model from a local path rather than the
registry, which is what lets ``docker run`` work with no MLflow server reachable. The
export cannot happen inside the Docker build: the registry lives in ``mlflow.db`` and
``mlruns/``, both gitignored and both excluded from the build context. So it runs on the
host first and the Dockerfile copies the result.

It also writes the resolved version to ``MODEL_VERSION`` beside the model. The image bakes
that in, because a local-path ``MODEL_URI`` gives the API no way to look the version up --
without it ``/health`` reports "unknown" in exactly the deployment where knowing which
model is live matters most.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Track-derived rather than a flat name. Nothing registers a bare "riskwatch" -- train.py
# registers riskwatch_credit and riskwatch_fraud -- so a singular default resolves to a
# model that does not exist, and the failure reads as a missing registry rather than a
# stale constant. Duplicated from train.py rather than imported because importing it here
# would pull sklearn and LightGBM into anything that touches export; the agreement is
# asserted by a test instead.
DEFAULT_TRACK = "credit"
DEFAULT_MODEL_URI = f"models:/riskwatch_{DEFAULT_TRACK}@production"
DEFAULT_OUT_DIR = PROJECT_ROOT / "build" / "model"
VERSION_FILENAME = "MODEL_VERSION"

# Which track's model this artifact holds. Written for the same reason as MODEL_VERSION -- the
# container has no registry to ask -- but it answers a different and more dangerous question.
#
# A bare directory path carries no claim about whose model it is, so ``MODEL_URI=/app/model``
# will load for *any* enabled track and ``/health`` will report it as that track at that
# version. Serving credit's artifact under ``ENABLED_TRACKS=fraud`` produced
# ``{"status":"ok","models":{"fraud":"5"}}`` and a 500 on every request: healthy by its own
# report, wrong about which model it holds, and useless.
#
# MLflow writes ``registered_model_meta`` into the artifact and it names the model, but reading
# it would tie the serving contract to MLflow's artifact layout. This is our own file, written
# by the step that already knows the answer because it resolved the alias to get here.
TRACK_FILENAME = "MODEL_TRACK"
MODEL_NAME_PREFIX = "riskwatch_"
# Anchored to the repo root, exactly as ``src/models/train.py`` and ``src/api/main.py`` do, and
# duplicated for the same reason the track name above is: importing ``train.py`` would pull
# sklearn and LightGBM into anything that touches export.
#
# This is not a nicety. MLflow's own default resolves to ``sqlite:///<cwd>/mlflow.db``, which
# coincides with this repo's registry **only when the cwd happens to be the repo root** -- the
# module docstring asks for that as a convention, and a convention is not a guarantee. Run from
# anywhere else and ``MlflowClient()`` silently inspects a different backend: from a fresh git
# worktree it reported ``Registered Model ... not found``, which is loud but names the wrong
# cause, and created an empty ``mlflow.db`` there for the next command to find as a valid,
# empty registry.
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def configure_tracking() -> str:
    """Point MLflow at the repo's tracking backend and return the resolved URI.

    Every entry point does this rather than inheriting MLflow's cwd-relative default; see
    :data:`DEFAULT_TRACKING_URI` for what inheriting it actually costs.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def resolve_version(uri: str) -> str:
    """Resolve the registry version a ``models:/`` URI points at.

    Mirrors the parsing in ``src/api/main.py`` so the baked version and the one the API
    would have reported from the registry agree.

    Configures tracking itself rather than trusting the caller. It is reachable from a test and
    from the CLI, and a lookup against the wrong backend does not fail loudly -- it reports the
    model as absent, which reads as "nothing is registered" rather than "you asked the wrong
    database".
    """
    if not uri.startswith("models:/"):
        return "unknown"

    suffix = uri.removeprefix("models:/")
    if "@" in suffix:
        name, alias = suffix.split("@", 1)
        configure_tracking()
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    if "/" in suffix:
        return suffix.rsplit("/", 1)[1]
    return "unknown"


def resolve_track(uri: str) -> str:
    """Resolve which track's model ``uri`` names, from the registered model name.

    ``models:/riskwatch_credit@production`` -> ``credit``. Raises rather than guessing: an
    exported artifact that cannot say which track it belongs to is the exact ambiguity
    :data:`TRACK_FILENAME` exists to remove, and writing it as ``"unknown"`` would hand the
    serving side a label it has to treat as "trust the caller" anyway.

    Parsed from the name rather than taken as a separate ``--track`` argument, so the label can
    never disagree with the artifact that was actually downloaded.
    """
    if not uri.startswith("models:/"):
        raise ValueError(
            f"cannot tell which track {uri!r} holds: only a models:/ URI names a registered "
            f"model, and a baked artifact must declare its track"
        )

    name = uri.removeprefix("models:/").split("@", 1)[0].rsplit("/", 1)[0]
    if not name.startswith(MODEL_NAME_PREFIX) or name == MODEL_NAME_PREFIX:
        raise ValueError(
            f"registered model {name!r} does not follow {MODEL_NAME_PREFIX}<track>, so the "
            f"track cannot be derived from it"
        )
    return name.removeprefix(MODEL_NAME_PREFIX)


def export_model(uri: str = DEFAULT_MODEL_URI, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Download ``uri`` into ``out_dir`` and record its version and track. Returns the version.

    **Either this leaves a complete artifact or it leaves none.** A failed export must not leave a
    previous one behind, because the only consumer is ``docker build`` and a `COPY` cannot tell a
    current artifact from last week's -- it would bake the stale model and the image would look
    entirely healthy, reporting that model's own version. Deleting the destination first is what
    makes the failure loud: the build fails on a missing `COPY` source instead of succeeding with
    the wrong model.

    The download goes to a sibling staging directory and is moved into place once the markers are
    written, so an interrupted download cannot be mistaken for a finished export either. There is
    no window in which a *partial* artifact sits at the destination.
    """
    # Before the download as well as inside resolve_version, because the version form
    # (``models:/name/3``) needs no registry lookup and would otherwise reach
    # download_artifacts with MLflow's cwd-relative default still in force.
    configure_tracking()

    staging = out_dir.with_name(out_dir.name + ".incoming")
    for path in (out_dir, staging):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir(parents=True)

    try:
        # Resolution is inside the try, and after the destination is cleared, so a URI whose track
        # or version cannot be resolved also leaves nothing deployable behind.
        track = resolve_track(uri)
        version = resolve_version(uri)

        # download_artifacts writes into dst_path directly and returns that same path, so the
        # destination must be the final directory rather than its parent.
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(staging))

        (staging / VERSION_FILENAME).write_text(f"{version}\n")
        (staging / TRACK_FILENAME).write_text(f"{track}\n")
    except BaseException:
        # BaseException so a Ctrl-C mid-download is cleaned up too; a half-downloaded artifact is
        # exactly as dangerous as a stale one.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    staging.rename(out_dir)

    logger.info("Exported %s (%s version %s) to %s", uri, track, version, out_dir)
    return version


def main() -> None:
    """CLI entry point. Prints the version so a build script can capture it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_MODEL_URI, help="MLflow model URI to export")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Destination directory")
    args = parser.parse_args()

    print(export_model(args.uri, args.out))


if __name__ == "__main__":
    main()
