# STATUS — ChurnWatch

**Last updated: 2026-09-09.**

This is the only file in the repo that records what is done. `CLAUDE.md` describes how to
work here, `AGENTS.md` describes what was planned — neither says where the project stands,
deliberately, because state kept in more than one place drifts and this repo has the scars.

The record of *how* each piece was built is the git history. Pull request bodies carry the
verified environment facts, the bugs found in review, and the rejected alternatives; they
are timestamped, append-only, and have never been wrong. Read them before re-deriving
anything. This file is the index — what is true now — and nothing more.

## Shipped

| Milestone | What landed | Evidence |
|---|---|---|
| **M1** — training | pandera-validated ingest → versioned parquet; per-model preprocessing; 14-config sweep promoting the best CV AUC to `models:/churnwatch@production` | PR #1 |
| **M2** — serving | FastAPI `POST /predict`, `GET /health`, `GET /metrics`; pydantic v2 snake_case contract; multi-stage non-root Dockerfile; host-side model export | PR #2 |
| *(unplanned)* — local observability | prediction JSONL log; Prometheus + Grafana; Evidently drift via Pushgateway | PR #3 |
| **M4** — orchestration | 5-task weekly Airflow DAG: ingest → train → evaluate → promote → monitor, with an AUC-delta promotion gate and a drift-based retrain trigger; Airflow image + compose overlay | this branch |

31 tests. 30 are hermetic and run anywhere; `tests/test_skew.py` needs a populated registry
and skips without one.

M4 was run end to end twice via `airflow dags test`, both `state=success`:

| | run 1 (empty registry) | run 2 (incumbent at 0.8476) |
|---|---|---|
| `task_promote` | promoted, `@production` → v1 | **skipped** — `delta -0.0000, need > 0.0100` |
| `task_monitor` | success | success **despite the upstream skip** |
| drift | `0.0526`, `MonthlyCharges` | identical |

Afterwards: 28 runs, **one** registered version, alias still v1. The gate refuses before
registering, so a rejected candidate leaves no junk version behind.

## Not built

- `README.md` — **0 bytes.** An M3/M7 deliverable; deliberately left empty rather than
  written before there is a live demo URL to put in it.
- `notebooks/ab_analysis.ipynb` — a 243-byte stub with zero cells.
- No Cloud Run deployment, no GitHub Actions workflow, no GCS upload, no A/B middleware.

## Where this diverges from the plan in AGENTS.md

`AGENTS.md` says "Implement in order. Do not skip ahead." That has already been overridden
once, on purpose. It is left unedited so the original reasoning survives; the deviations are
recorded here instead.

- **M3 (Cloud Run) was deferred.** PR #3 made the local stack observable instead.
- **M5 (monitoring) was partly done early**, and not to spec: `src/monitoring/drift.py` uses
  `DataDriftPreset` **only**. There is no `ClassificationPreset` and no GCS upload. M5 is not
  closed.
- **`docker-compose.yml` is not what M2 described.** Services are `api`, `mlflow`,
  `prometheus`, `grafana`, `pushgateway`. Airflow and PostgreSQL arrived with M4 but live in
  a **separate overlay**, `docker-compose.airflow.yml`, so a plain `docker compose up` stays
  the light API-only path. The full stack is
  `docker compose -f docker-compose.yml -f docker-compose.airflow.yml up`.
- **M4's retrain trigger is not `drift_share > 0.2`.** That number was inherited from a spec
  written before any data existed and, with 19 features, means "4 or more columns at once" —
  which the `MonthlyCharges` +15% scenario this repo ships can never reach. Implementing it
  verbatim would have shipped a trigger that provably never fires. `should_retrain()` in
  `src/pipelines/retrain.py` fires on a **named watched column** (`MonthlyCharges`, `tenure`,
  `Contract`) *or* the 0.20 share as a catch-all for broad shift.
- **M4's DAG does not chain retrains.** `AGENTS.md` says `task_monitor` triggers a retrain,
  but this DAG *is* the retrain and retraining does not move the reference distribution — so
  an unguarded self-trigger loops forever. A drift-triggered run never triggers another.
- **`train()` was split.** `sweep()` runs the grid and promotes nothing; `promote_best()` is
  unchanged and still unconditional. The DAG composes them with its own gate in between. The
  CLI (`uv run python -m src.models.train`) behaves exactly as before.
- **Models trained by the DAG are logged as `mlflow-artifacts:` URIs, not local paths.**
  Discovered during M4 verification, and it invalidates an assumption written into
  `docker-compose.yml`. The DAG logs *through* the tracking server, so the experiment's
  `artifact_location` is `mlflow-artifacts:/1` and a model version's source is
  `models:/m-<id>` — the files are on disk under `mlruns/`, but resolving them needs an
  http tracking URI. Host-side tooling that assumed a plain local path
  (`src/models/export.py`, `tests/test_skew.py`) therefore cannot load a DAG-trained model
  with `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`; it skips or raises. A host-trained model
  (`uv run python -m src.models.train`) is unaffected — that path still writes local URIs.
  **Not resolved.** Pointing the host at `http://localhost:5000` instead returns 403: the
  compose `mlflow` service rejects the Host header even though `localhost:5000` is in
  `MLFLOW_SERVER_ALLOWED_HOSTS`. That 403 predates M4 and was not investigated further.

## Before you can run anything

Everything needed to train, serve from the registry, or compute drift is gitignored:
`data/raw/`, `data/processed/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `build/`. **A fresh
clone has none of it** and must run ingest and training first. The hermetic tests are the
only thing that works out of the box.

Machine-local as of this update: in the primary checkout at
`~/Documents/projects/mlops_practice` the registry is empty (0 runs, 0 registered models)
and `data/raw/` is absent. Two worktrees are populated instead:

- `spookfish` (`~/orca/workspaces/mlops_practice/spookfish`) — 14 runs, the `churnwatch`
  registered model, `data/raw/telco.csv`.
- `horseshoe` (this one) — populated by the two M4 verification runs above: 28 runs,
  `churnwatch` v1 on `@production`, plus `data/raw/telco.csv` copied in from `spookfish`.

Point `MLFLOW_TRACKING_URI` at one of those `mlflow.db` files, or retrain, before expecting
the registry-backed paths to work.

The DAG needs nothing extra: the Airflow overlay bind-mounts the repo, so `task_ingest`
writes `data/processed/` back into the working tree and the sweep logs through the `mlflow`
service into the same `mlflow.db` the host reads.
