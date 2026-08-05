# AGENT.md — ChurnWatch

Full context for AI agents working on this codebase. Read CLAUDE.md first for commands and layout.

## What this project is

A solo MLOps portfolio project demonstrating end-to-end ML capability: raw CSV data → trained model → production REST API → drift monitoring → automated retraining. The primary goal is to eliminate the "notebook scientist" signal from the resume by producing concrete deployable artifacts.

**Dataset:** IBM Telco Customer Churn — 7,043 rows, 21 features, binary target `Churn`.
**Model:** LightGBM (primary) + LogisticRegression (baseline).

## Milestone Acceptance Criteria

Implement in order. Do not skip ahead.

### Milestone 1 — Week 1–2: Training Pipeline (current target)

Files to implement:
- `src/data/ingest.py` — load `data/raw/telco.csv`, validate with pandera schema, write versioned parquet to `data/processed/`
- `src/features/pipeline.py` — `build_pipeline()` returning a fitted `sklearn.pipeline.Pipeline` with `ColumnTransformer` (imputer + OrdinalEncoder for categoricals, StandardScaler for numerics) → `LGBMClassifier`
- `src/models/train.py` — `train(data_path, experiment_name)` that logs params/metrics (AUC-ROC, F1, precision@0.5) to MLflow and registers the best model to MLflow Model Registry with tag `"Production"`

Done when: `mlflow ui` shows 10+ experiment runs and Model Registry shows a Production artifact.

### Milestone 2 — Week 3–4: FastAPI Serving + Docker

Files to implement:
- `src/api/schemas.py` — pydantic v2 `PredictRequest`, `PredictResponse`, `HealthResponse`
- `src/api/main.py` — FastAPI app with lifespan (load model on startup), `POST /predict`, `GET /health`, `GET /metrics` (Prometheus format via `prometheus_client`)
- `Dockerfile` — multi-stage build, non-root user, `HEALTHCHECK`
- `docker-compose.yml` — `api` + `mlflow` services with shared artifact volume
- `tests/test_api.py` — happy path, missing field (422), out-of-range value

Done when: `docker run -p 8000:8000 churnwatch:latest` + `curl -X POST /predict` returns `{ churn_probability, prediction, model_version, request_id }`.

### Milestone 3 — Week 5: GCP Cloud Run Deployment

- `.github/workflows/deploy.yml` — build image → push to GCR → `gcloud run deploy` on push to `main`
- `README.md` — architecture diagram, `docker-compose up` quickstart, live demo URL

Done when: `curl https://churnwatch-<hash>.a.run.app/health` returns 200 from any device.

### Milestone 4 — Week 6–7: Airflow Orchestration

- `dags/churnwatch_retrain.py` — 5-task DAG: `task_ingest → task_train → task_evaluate → task_promote → task_monitor`
  - `task_promote`: promote Staging → Production if AUC delta > 0.01
  - `task_monitor`: Evidently drift check, trigger retrain if `drift_share > 0.2`
  - Schedule: `@weekly`

Done when: DAG runs end-to-end, new model version appears in MLflow Model Registry.

### Milestone 5 — Week 8: Monitoring

- `src/monitoring/drift.py` — `generate_drift_report(reference_df, current_df) -> Path` using `Evidently DataDriftPreset + ClassificationPreset`, saves HTML to `reports/`
- Airflow `task_monitor` calls this and uploads to GCS

Done when: `reports/drift_report.html` shows `MonthlyCharges` as drifted.

### Milestone 6 — Week 9: A/B Testing

- `src/api/main.py` — add middleware: route 10% of requests (or `X-Model-Version: challenger` header) to a second "Challenger" model; log both to `logs/predictions.jsonl`
- `notebooks/ab_analysis.ipynb` — load log, split by version, KS-test on `churn_probability`, compute ₩50,000 cost-per-false-negative, document winner

Done when: notebook shows KS-test result and a written promotion decision.

### Milestone 7 — Week 10: Polish

- `README.md` complete with mermaid architecture diagram, demo URL, YouTube screen-capture link
- `DECISIONS.md` — why GCP over AWS, LightGBM over XGBoost, Evidently over custom monitoring
- Resume bullet ready for `facts/projects.md` in cv-agent repo

## Architecture Decisions (do not revisit without strong reason)

| Decision | Choice | Rejected alternative | Reason |
|---|---|---|---|
| Cloud provider | GCP Cloud Run | AWS Lambda / ECS | Free tier predictable, GCR integration, one-command deploy |
| Orchestration | Airflow (Docker Compose) | Prefect, Kubeflow | 4/9 JDs name Airflow specifically |
| Experiment tracking | MLflow (self-hosted) | W&B | 4/9 JDs, free, no SaaS dependency |
| Monitoring | Evidently AI | Grafana + custom | Python-native, HTML reports as portfolio artifacts |
| Serving | FastAPI | Flask | Async, pydantic v2, OpenAPI auto-docs |
| Model | LightGBM | XGBoost, CatBoost | 70%+ Korean DS JDs; fast; SHAP interpretable |
| Container orchestration | None / Cloud Run | Kubernetes | Solo build; Cloud Run sufficient for the story |

## Constraints

- **No Kubernetes** — mention in interviews but do not implement
- **No Flask** — FastAPI only
- **No pip/poetry/conda** — `uv` only; add deps with `uv add <package>`
- **Airflow not in venv** — DAG `.py` files only; Airflow itself runs in Docker
- **No ML data committed to git** — `data/raw/`, `data/processed/`, `mlruns/` are gitignored
- **uv.lock is committed** — do not add it to `.gitignore`
- **Secrets via .env** — GCP_PROJECT_ID, GCS_BUCKET, MLFLOW_TRACKING_URI; never hardcode

## File Authoring Rules

- All Python: `ruff`-clean, `line-length = 100`, type hints on public functions
- pydantic models in `src/api/schemas.py` only — import them into `main.py`
- MLflow experiment name: `"churnwatch"` (constant, not a magic string scattered in code)
- Model artifact path pattern: `mlflow.sklearn.log_model(pipeline, "model")` — load via `mlflow.pyfunc.load_model`
- Prometheus metrics: define counters/histograms at module level in `main.py`, not inside route handlers

## Drift Simulation

The synthetic drifted batch is **not fabrication** — it is a documented simulation standard in MLOps demos:

```python
drifted = train_df.copy()
drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.15
drifted = drifted.sample(500, random_state=42)
```

Document this in comments and in the Evidently report title.
