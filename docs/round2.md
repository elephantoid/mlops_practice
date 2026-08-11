# Milestone 2 · Round 2 — Docker packaging

Status: **not started**. Prerequisite: [round1.md](round1.md).

Closes Milestone 2. Acceptance criterion from `AGENT.md`:

> `docker run -p 8000:8000 churnwatch:latest` + `curl -X POST /predict` returns
> `{ churn_probability, prediction, model_version, request_id }`

Note the criterion is **standalone `docker run`** — no compose, no MLflow server reachable. That single constraint drives most of the design below.

---

## Files to create

| File | Purpose |
|---|---|
| `src/models/export.py` | Export `models:/churnwatch@production` to a plain directory the image can copy |
| `.dockerignore` | Keep `data/`, `mlruns/`, `mlflow.db`, `.venv/`, `.git/` out of the build context |
| `Dockerfile` | Multi-stage, non-root, `HEALTHCHECK` |
| `docker-compose.yml` | `api` + `mlflow` services, shared artifact volume |

---

## Why an export step exists

`MODEL_URI=/app/model` (a local path) is what makes standalone `docker run` work. Something has to put a model at that path.

The export cannot happen *inside* the Docker build: the registry lives in `mlflow.db` + `mlruns/`, both gitignored and both excluded from the build context. So `export.py` runs on the host first, writing to a directory the `COPY` instruction picks up.

`export.py` must also emit the resolved **version string**, which the Dockerfile bakes in as `MODEL_VERSION`. Without it `/health` reports `"unknown"` — see the table in round1.md.

---

## Dockerfile constraints

- **Multi-stage.** Builder installs dependencies with `uv`; the runtime stage copies only the virtualenv and `src/`. `AGENT.md` requires multi-stage explicitly.
- **Non-root user.** Required by `AGENT.md`; also a Cloud Run good practice.
- **`HEALTHCHECK`** hitting `GET /health`.
- **No `libomp`.** LightGBM needs it on macOS only — Linux images ship `libgomp`. Adding it would be cargo-culting a host-specific fix into the image.
- **Do not install dev dependencies.** `pytest`, `ruff`, `pre-commit` have no place in a serving image.
- **`uv` only** — never `pip install` (project constraint).

Environment baked at build time:

```
MODEL_URI=/app/model
MODEL_VERSION=<resolved at export>
```

## docker-compose constraints

- `api` + `mlflow` services with a shared artifact volume (`AGENT.md`).
- Compose is the *registry-backed* configuration: it overrides `MODEL_URI=models:/churnwatch@production` and points `MLFLOW_TRACKING_URI` at the `mlflow` service.
- Airflow is **not** added here — that is Milestone 4.

---

## Open questions to resolve before implementing

1. **MLflow service backend in compose.** MLflow 3.15 refuses the `file:` store ("maintenance mode"), so the compose `mlflow` service needs sqlite or postgres. The local dev setup uses `sqlite:///mlflow.db`.
2. **Does the compose `mlflow` service get a populated registry?** A fresh container has an empty registry, so `MODEL_URI=models:/churnwatch@production` would fail unless `mlflow.db` and `mlruns/` are mounted from the host.
3. **Image size.** mlflow + scikit-learn + LightGBM + pandas is heavy. `mlflow` is imported by `main.py` purely to load the model — worth measuring before optimising.

---

## Verification

1. `uv run python -m src.models.export` writes a loadable model directory and prints the version.
2. `docker build -t churnwatch:latest .` succeeds.
3. `docker run -p 8000:8000 churnwatch:latest` starts with **no** MLflow server running.
4. `curl /health` → 200, `model_version` is the **real** version, not `"unknown"`.
5. `curl -X POST /predict` returns all four fields, and the probability **matches the host-side `predict_proba` for the same row** — the skew check from round1.md, re-run against the container.
6. `docker inspect` shows the container running as a non-root user.
7. `HEALTHCHECK` reports `healthy`.
8. `docker compose up` brings up `api` + `mlflow`; the API loads from the registry rather than the baked path.
9. Image size recorded.
