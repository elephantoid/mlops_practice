# CLAUDE.md — ChurnWatch

ChurnWatch is a solo MLOps portfolio project: telecom customer churn prediction served via FastAPI on GCP Cloud Run, with MLflow experiment tracking, Airflow-orchestrated retraining, and Evidently drift monitoring.

## Commands

| Command | Action |
|---------|--------|
| `make check` | `pytest tests/ -v` |
| `make lint` | `ruff check` + `ruff format --check` (read-only) |
| `make fmt` | `ruff check --fix` + `ruff format` (rewrites) |
| `make clean` | remove `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.pyc` |
| `uv sync --dev` | install all deps including dev group |
| `uv run <cmd>` | run any command inside the virtualenv |
| `uv run uvicorn src.api.main:app --reload` | local dev server |
| `docker compose up` | start api + mlflow + airflow + postgresql |

## Project Layout

```
src/
├── data/ingest.py        pandera schema validation on raw CSV
├── features/pipeline.py  sklearn ColumnTransformer (imputer → encoder → scaler)
├── models/train.py       MLflow experiment logging, model registry promotion
├── api/main.py           FastAPI app — POST /predict, GET /health, GET /metrics
├── api/schemas.py        pydantic v2 request/response models
└── monitoring/drift.py   Evidently DataDriftPreset + ClassificationPreset

dags/churnwatch_retrain.py  Airflow DAG: ingest → train → evaluate → promote → monitor
tests/test_api.py
notebooks/ab_analysis.ipynb  A/B KS-test analysis
```

## Key Contracts

```
POST /predict    { tenure, monthly_charges, contract, ... }
              →  { churn_probability, prediction, model_version, request_id }

GET  /health  →  { status, model_version, uptime_seconds }
GET  /metrics →  Prometheus counters (requests, latency p50/p95, prediction distribution)
```

## Conventions

- **Package manager:** `uv` only — never `pip install`, `poetry`, or `conda`
- **Formatter/linter:** `ruff` — `line-length = 100`, `target-version = "py312"`
- **Pre-commit:** hooks run on every commit (trailing whitespace, ruff check, ruff format)
- **Async tests:** `pytest-asyncio` in `STRICT` mode — mark async tests with `@pytest.mark.asyncio`
- **Secrets:** never hardcode — copy `.env.example` to `.env` and fill in values
- **Airflow:** DAG files in `dags/` only — do not install Airflow into the uv venv

## Stack

```
Training:      scikit-learn Pipeline + LightGBM + MLflow (self-hosted)
Serving:       FastAPI + Uvicorn + pydantic v2 → Docker → GCP Cloud Run
Orchestration: Airflow 2.x via Docker Compose (DAG code only, not installed in venv)
Monitoring:    Evidently AI (drift reports → GCS)
Storage:       GCS for model artifacts + drift reports
CI/CD:         GitHub Actions → GCR → Cloud Run
A/B:           FastAPI middleware + JSONL log + KS-test notebook
```

## Current State

**Milestone 1 (Week 1–2) complete.** The training path runs end to end:

- `src/data/ingest.py` — validates `data/raw/telco.csv` against a pandera schema, writes
  timestamped parquet to `data/processed/`, `latest.parquet` points at the newest snapshot.
- `src/features/pipeline.py` — `build_pipeline(model_type, **params)`, per-model
  preprocessing (LightGBM: OrdinalEncoder; LogReg: OneHotEncoder + StandardScaler).
- `src/models/train.py` — `uv run python -m src.models.train` runs a 14-config sweep and
  promotes the best by CV AUC to `models:/churnwatch@production`.

`src/api/`, `src/monitoring/`, `dags/`, `tests/` are still empty stubs. **Next: Milestone 2**
— `src/api/schemas.py`, `src/api/main.py`, `Dockerfile`, `docker-compose.yml`.

Environment notes worth knowing before you start:

- MLflow 3.x **rejects the `file:` backend**; tracking uses `sqlite:///mlflow.db`.
- LightGBM needs `brew install libomp` on macOS (not needed in Linux containers).
- Models are logged with `serialization_format="cloudpickle"` (skops rejects `LGBMClassifier`)
  and `pyfunc_predict_fn="predict_proba"`, so the served artifact returns probabilities.
- Registry promotion uses **aliases**, not stages — stages are deprecated since MLflow 2.9.

Full spec and milestones: `../blueprint/track-e2e/`
