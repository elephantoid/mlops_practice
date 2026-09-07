# CLAUDE.md — ChurnWatch

ChurnWatch is a solo MLOps portfolio project: telecom customer churn prediction served via FastAPI on GCP Cloud Run, with MLflow experiment tracking, Airflow-orchestrated retraining, and Evidently drift monitoring.

## Where things are written down

| File | Holds | Changes |
|---|---|---|
| **`STATUS.md`** | **What is done, what is not, and where the build diverged from the plan.** The only place project state is recorded — nothing else in this repo may claim it. | often |
| `CLAUDE.md` (this file) | How to work here: commands, layout, contracts, conventions | rarely |
| `AGENTS.md` | The plan: M1–M7 acceptance criteria and the architecture decisions behind them | almost never |
| `.claude/rules/*.md` | Traps that recur in one area of the code. Loaded only when a matching file is read | as they are found |
| git history + PR bodies | Why each change was made, what broke, what was rejected | append-only |

**Read `STATUS.md` first.** Read `AGENTS.md` when starting a milestone or questioning a
stack decision — it is not loaded automatically and it is long, so open it deliberately
rather than by habit.

## Commands

**Development happens inside the Linux container.** Get there first:

```bash
make shell        # docker compose run --rm dev bash
```

Everything below runs unchanged in either place — every target goes through `uv run`, which
behaves the same on both sides. The container is where you stand, not a different command
set. Running on the macOS host still works and is the fallback when Docker is not up; it is
also the only place `brew install libomp` matters.

| Command | Action |
|---------|--------|
| `make check` | `pytest tests/ -v` |
| `make lint` | `ruff check` + `ruff format --check` (read-only) |
| `make fmt` | `ruff check --fix` + `ruff format` (rewrites) |
| `make clean` | remove `__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.pyc` |
| `uv sync --dev` | install all deps including dev group |
| `uv run <cmd>` | run any command inside the virtualenv |
| `uv run uvicorn src.api.main:app --reload` | local dev server |
| `uv run python -m src.monitoring.drift --source synthetic --push` | drift report + metrics |

Container-only:

| Command | Action |
|---------|--------|
| `make shell` | bash in the dev container (Linux, Python 3.12, dev deps) |
| `docker compose up` | api + mlflow + prometheus + grafana + pushgateway — **not** `dev`, which sits behind a profile |
| `docker compose run --rm --service-ports dev` | as above, but publishing :8000; collides with a running `api` |

The working tree is bind-mounted at **the host's own absolute path**, not `/workspace`.
MLflow writes absolute artifact locations into `mlflow.db`, so a model trained on either
side has to resolve on the other; the same mount keeps `PROJECT_ROOT` — which every module
derives from its own file location — pointing at the same `data/processed/`, `logs/` and
`reports/`.

The container's virtualenv is at `/opt/venv`, outside the mount, because the repo's own
`.venv` holds macOS wheels that would otherwise be found first. VS Code can attach to the
same container through `.devcontainer/devcontainer.json`.

## Project Layout

```
src/
├── data/ingest.py        pandera schema validation on raw CSV
├── features/pipeline.py  sklearn ColumnTransformer (imputer → encoder → scaler)
├── models/train.py       MLflow experiment logging, model registry promotion
├── models/export.py      registry → plain directory, for the Docker build
├── api/main.py           FastAPI app — POST /predict, GET /health, GET /metrics
├── api/schemas.py        pydantic v2 request/response models
├── api/prediction_log.py append-only JSONL log of every served prediction
└── monitoring/drift.py   Evidently DataDriftPreset, exported via Pushgateway

dags/churnwatch_retrain.py  Airflow DAG: ingest → train → evaluate → promote → monitor
monitoring/                 Prometheus config + provisioned Grafana dashboard
tests/                      test_api.py (hermetic) · test_skew.py (needs a registry)
notebooks/ab_analysis.ipynb A/B KS-test analysis
```

## Key Contracts

```
POST /predict    { tenure, monthly_charges, contract, ... }   # 19 fields, snake_case
              →  { churn_probability, prediction, model_version, request_id }

GET  /health  →  { status, model_version, uptime_seconds }
GET  /metrics →  Prometheus exposition format
```

## Conventions

- **Package manager:** `uv` only — never `pip install`, `poetry`, or `conda`. `uv.lock` is
  committed; do not gitignore it.
- **Formatter/linter:** `ruff` — `line-length = 100`, `target-version = "py312"`. Type hints
  on public functions.
- **Pre-commit:** hooks run on every commit (whitespace, ruff, gitleaks, nbstripout, …)
- **Async tests:** `pytest-asyncio` in `STRICT` mode — mark async tests with
  `@pytest.mark.asyncio`. STRICT is the library's own 1.4 default; there is no
  `[tool.pytest.ini_options]` in `pyproject.toml`, so the behaviour is correct but unpinned.
- **pydantic models** live in `src/api/schemas.py` only — import them into `src/api/main.py`.
- **MLflow experiment name** is the constant `"churnwatch"`, not a string scattered in code.
- **Secrets:** never hardcode — copy `.env.example` to `.env` and fill in values
- **Airflow:** DAG files in `dags/` only — do not install Airflow into the uv venv
- **No ML data in git:** `data/raw/`, `data/processed/`, `mlruns/`, `mlflow.db`, `build/`
  are gitignored. A fresh clone cannot train or serve until they are rebuilt.

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

## Environment notes

- MLflow 3.x **rejects the `file:` backend**; tracking uses `sqlite:///mlflow.db`, anchored
  to the repo root rather than the CWD — a relative path makes MLflow silently create an
  empty database and then report the model as missing.
- Registry promotion uses **aliases**, not stages — stages are deprecated since MLflow 2.9.
- Models are logged with `serialization_format="cloudpickle"` (skops rejects
  `LGBMClassifier`) and `pyfunc_predict_fn="predict_proba"`, so the served artifact returns
  probabilities and the API applies its own threshold.
- LightGBM needs `brew install libomp` on macOS. Linux images ship `libgomp` — this must
  **not** appear in the Dockerfile. Working in the dev container sidesteps the split
  entirely, which is why it exists.

Area-specific traps live in `.claude/rules/` and load when you open the matching file.
