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
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_URI = "models:/riskwatch@production"
DEFAULT_OUT_DIR = PROJECT_ROOT / "build" / "model"
VERSION_FILENAME = "MODEL_VERSION"


def resolve_version(uri: str) -> str:
    """Resolve the registry version a ``models:/`` URI points at.

    Mirrors the parsing in ``src/api/main.py`` so the baked version and the one the API
    would have reported from the registry agree.
    """
    if not uri.startswith("models:/"):
        return "unknown"

    suffix = uri.removeprefix("models:/")
    if "@" in suffix:
        name, alias = suffix.split("@", 1)
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    if "/" in suffix:
        return suffix.rsplit("/", 1)[1]
    return "unknown"


def export_model(uri: str = DEFAULT_MODEL_URI, out_dir: Path = DEFAULT_OUT_DIR) -> str:
    """Download ``uri`` into ``out_dir`` and record its version. Returns the version.

    The destination is cleared first: leaving a previous export in place risks the image
    picking up a stale mix of two models' files.
    """
    version = resolve_version(uri)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # download_artifacts writes into dst_path directly and returns that same path, so the
    # destination must be the final directory rather than its parent.
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(out_dir))

    (out_dir / VERSION_FILENAME).write_text(f"{version}\n")

    logger.info("Exported %s (version %s) to %s", uri, version, out_dir)
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
