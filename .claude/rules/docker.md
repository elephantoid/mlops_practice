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
