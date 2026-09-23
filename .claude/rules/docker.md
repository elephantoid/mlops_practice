---
paths:
  - "Dockerfile"
  - "docker-compose.yml"
  - ".dockerignore"
---

# Image packaging

- **No `libomp`.** LightGBM needs it on macOS only; Linux base images ship `libgomp`.
  Adding it cargo-cults a host-specific fix into the image.
- **No dev dependencies.** `pytest`, `ruff`, and `pre-commit` have no place in a serving
  image.
- **`uv` only** — never `pip install` (project-wide constraint).
- **`MODEL_VERSION` must be baked in alongside `MODEL_URI=/app/model`.** With a local-path
  model URI the API cannot query the registry for a version, so `/health` reports
  `"unknown"` — in exactly the deployment where knowing the live version matters most.
  `src/models/export.py` emits the resolved version for this reason.
- The registry lives in `mlflow.db` + `mlruns/`, both gitignored and both excluded from the
  build context. The export step therefore runs **on the host**, before the build.

## Where a model's artifacts live depends on *who trained it*

Verified during Milestone 4, by training the same sweep both ways:

| Trained by | `artifact_location` | Loadable from the host with `sqlite:///mlflow.db`? |
|---|---|---|
| `uv run python -m src.models.train` | absolute local path | yes |
| the Airflow DAG (via `http://mlflow:5000`) | `mlflow-artifacts:/1` | **no** |

The DAG logs through the tracking server, so MLflow records a proxied URI and the client
refuses to resolve it without an http tracking URI — "When an mlflow-artifacts URI was
supplied, the tracking URI must be a valid http or https URI". The files are still under
`mlruns/`; only the addressing changed.

This breaks the assumption in the `api` service comment in `docker-compose.yml` that
artifact resolution is always a `LocalArtifactRepository` read. It holds for host-trained
models only. Anything that must work from the host against a **DAG-trained** model —
`src/models/export.py` before a `docker build`, `tests/test_skew.py` — needs a reachable
MLflow server, not the sqlite file.

## The Airflow image is not the serving image, and breaks two assumptions

`docker/airflow/Dockerfile` extends `apache/airflow`, which is laid out unlike any other
Python image here. Both of these were hit during Milestone 4, and both **succeeded** before
failing later — the expensive kind.

- **`uv pip install --system` installs where nothing reads.** The base image runs Airflow
  from a virtualenv at `/home/airflow/.local` created with
  `include-system-site-packages = false`, so `/usr/local/lib/python3.12/site-packages` is
  not on `sys.path`. The install reports success; the first task dies with
  `ModuleNotFoundError`. Name the venv interpreter explicitly:
  `uv pip install --python /home/airflow/.local/bin/python`.
- **Anything written after `USER airflow` must be owned by `airflow`.** A `COPY` before the
  user switch leaves a root-owned directory, and writing into it — `uv export --output-file`,
  for instance — fails with a bare `exit code 2` that names no path. Use `COPY --chown`.

The final `RUN python -c "import airflow, mlflow, lightgbm, evidently"` exists because both
failures above were invisible at build time otherwise. Do not remove it.

## Dependencies come from `uv.lock`, in every image

The serving image uses `uv sync --locked`. The Airflow image cannot — it installs into an
existing venv, not a new one — so it uses `uv export --frozen` to turn the same lockfile
into pinned requirements. Installing from `pyproject.toml` ranges instead re-resolves on
every build, which silently swaps MLflow, Evidently or LightGBM under an image that was
verified with different versions. `apache-airflow` is installed *after*, pinned, so a real
incompatibility fails the build rather than producing a scheduler that will not start.
