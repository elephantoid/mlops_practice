# STATUS — ChurnWatch

**Last updated: 2026-09-03.**

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

17 tests. 16 are hermetic and run anywhere; `tests/test_skew.py` needs a populated registry
and skips without one.

## Not built

- `dags/churnwatch_retrain.py` — **0 bytes.** The `drift_share > 0.2` retrain trigger exists
  only as a red band on a Grafana panel and a number in `AGENTS.md`. Nothing implements it.
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
  `prometheus`, `grafana`, `pushgateway` — there is no Airflow service and no PostgreSQL.
  Airflow arrives with M4, if it arrives.

## Before you can run anything

Everything needed to train, serve from the registry, or compute drift is gitignored:
`data/raw/`, `data/processed/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `build/`. **A fresh
clone has none of it** and must run ingest and training first. The hermetic tests are the
only thing that works out of the box.

Machine-local as of this update: in the primary checkout at
`~/Documents/projects/mlops_practice` the registry is empty (0 runs, 0 registered models)
and `data/raw/` is absent. The populated copy lives in the `spookfish` git worktree at
`~/orca/workspaces/mlops_practice/spookfish` — 14 runs, the `churnwatch` registered model,
and `data/raw/telco.csv`. Point `MLFLOW_TRACKING_URI` at that worktree's `mlflow.db`, or
retrain, before expecting the registry-backed paths to work.
