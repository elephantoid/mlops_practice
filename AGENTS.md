# AGENTS.md — ChurnWatch

The plan: acceptance criteria per milestone, and the stack decisions behind them.
Read `STATUS.md` for where the project actually stands, and `CLAUDE.md` for commands
and conventions. This file is the plan as written — deviations in *execution* are recorded in `STATUS.md`,
not edited in here. A decision that has been **revisited and overturned** is the one
exception: the original row stays untouched and the revision is appended beneath the table.
A decision table that contradicts the decision actually in force misleads every later
reader, and leaving it wrong costs more than the rule protects.

## What this project is

A solo MLOps portfolio project demonstrating end-to-end ML capability: raw CSV data → trained model → production REST API → drift monitoring → automated retraining. The primary goal is to eliminate the "notebook scientist" signal from the resume by producing concrete deployable artifacts.

**Dataset:** IBM Telco Customer Churn — 7,043 rows, 21 features, binary target `Churn`.
**Model:** LightGBM (primary) + LogisticRegression (baseline).

## Milestone Acceptance Criteria

Implement in order. Do not skip ahead.

### Milestone 1 — Week 1–2: Training Pipeline

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
- `README.md` — summarise the rationale from the decision table below: GCP over AWS,
  LightGBM over XGBoost, Evidently over custom monitoring. (There is no `DECISIONS.md`;
  it was deleted as a second copy of that table.)
- Resume bullet ready for `facts/projects.md` in cv-agent repo

## Architecture Decisions (do not revisit without strong reason)

| Decision | Choice | Rejected alternative | Reason |
|---|---|---|---|
| Cloud provider | GCP Cloud Run — **revised 2026-09-22, see below** | AWS Lambda / ECS | Free tier predictable, GCR integration, one-command deploy |
| Orchestration | Airflow (Docker Compose) | Prefect, Kubeflow | 4/9 JDs name Airflow specifically — **reason replaced 2026-09-22, see below** |
| Experiment tracking | MLflow (self-hosted) | W&B | 4/9 JDs, free, no SaaS dependency |
| Monitoring | Evidently AI | Grafana + custom | Python-native, HTML reports as portfolio artifacts |
| Serving | FastAPI | Flask | Async, pydantic v2, OpenAPI auto-docs |
| Model | LightGBM | XGBoost, CatBoost | 70%+ Korean DS JDs; fast; SHAP interpretable |
| Container orchestration | None / Cloud Run | Kubernetes | Solo build; Cloud Run sufficient for the story |

### Revised 2026-09-22 — Cloud provider

Revisited under the clause in the heading above. The original row is left standing so the
first reasoning survives. **This section records why the choice was made and remade; it
makes no claim about where the project stands — that is `STATUS.md`'s, and putting a second
copy here is how the two came to contradict each other in the first place.**

**Why the original reason did not survive.** "Free tier predictable, GCR integration,
one-command deploy" was never a serving argument. Cold start against model load time, the
concurrency model, and image size limits are the arguments, and none of them can be settled
from documentation — they have to be measured against a built image. So the row is to be
rewritten from those measurements rather than from what the free tier advertises.

**A deferral was argued for, then withdrawn the same day.** The case for deferring was that
Cloud Run can bill and that nothing through M2 needs a public URL to be demonstrable. The
case against, which won once the Q4 plan and real pricing were read: Cloud Run scales to
zero and does not bill idle time, the request-based free tier is far above anything this
service generates, and the one charge that bites is Artifact Registry above 0.5 GB — an
image-size problem, not a reason to stay local. The plan schedules the deploy for
10/5-10/11 alongside the DAG and asks for 75+ days of uptime, so a deferral until the whole
local loop was finished would have put that out of reach.

Both positions are kept because the withdrawn one names the risk the surviving one accepts.
The conditions attached to the move, and whether any of them are met, are in `STATUS.md`;
the reasoning behind each is in `docs/debt-ledger.md`.

### Revised 2026-09-22 — Orchestration, and the JD figures this table rests on

**Airflow stands. The reason does not.**

The reason in force is that the constraints Airflow imposes are the concepts worth
learning: tasks run as separate processes, XCom carries small values rather than frames,
a schedule implies backfill semantics, and top-level DAG code is re-parsed by the
scheduler. All of that transfers to any orchestrator. Prefect hides it behind decorators
and ordinary function calls — easier to stand up, and it teaches less. For a portfolio
whose stated purpose is evidence of having operated a pipeline, the friction is the
curriculum.

The decision was cheap to take because `src/` had been shaped that way before any DAG
existed: `ingest()` returns a `Path` rather than a frame, `train()` re-reads it with
`read_parquet`, and `src/monitoring/drift.py` loads from disk. Nothing is passed in memory
between stages, which is what made the DAG thin wrappers rather than a restructuring — the
argument for choosing Airflow, not a report on the code's present shape.

**The JD figures in this table are superseded.** "4/9 JDs" came from a 9-posting survey
dated 2026-07-29, whose source postings were not kept. A later analysis in
`~/Documents/career/` collected 222 postings and verified 78 as core AI/ML roles; there
Airflow appears in **6%** of them (9% of the 57 closest to this profile) and is marked as
**declining**. The old figure overstated it by roughly five times. Two other rows —
experiment tracking and model choice — cite the same superseded survey and have not yet
been revisited. Use the 222-posting analysis, not this table, for any market claim.

## Drift Simulation

The synthetic drifted batch is **not fabrication** — it is a documented simulation standard in MLOps demos:

```python
drifted = train_df.copy()
drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.15
drifted = drifted.sample(500, random_state=42)
```

Document this in comments and in the Evidently report title.
