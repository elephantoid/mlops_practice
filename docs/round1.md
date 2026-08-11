# Milestone 2 · Round 1 — FastAPI serving layer

Status: **complete**. Files: `src/api/schemas.py`, `src/api/main.py`, `tests/test_api.py`.

Read this before Round 2 — several things here directly constrain the Dockerfile.

---

## What the service does

```
POST /predict  { tenure, monthly_charges, contract, ... }   # 19 fields, snake_case
            →  { churn_probability, prediction, model_version, request_id }

GET  /health   →  { status, model_version, uptime_seconds }
GET  /metrics  →  Prometheus exposition format
```

Run locally:

```bash
uv run uvicorn src.api.main:app --reload
```

---

## Decisions that Round 2 depends on

### Model loading is env-driven

`load_model()` in `src/api/main.py` resolves `MODEL_URI`:

| `MODEL_URI` | Behaviour | Version reported by `/health` |
|---|---|---|
| `models:/churnwatch@production` (default) | Queries the registry by alias | Real version, e.g. `1` |
| `models:/churnwatch/3` | Loads pinned version | `3`, parsed from the URI |
| `/app/model` (a local path) | Loads the directory directly | **`MODEL_VERSION` env, else `"unknown"`** |

**The container will use the local-path form.** That means the image build MUST also set `MODEL_VERSION`, or `/health` reports `"unknown"` in exactly the deployment where knowing the live version matters most.

### The API is snake_case; the model is CamelCase

The trained pipeline's columns are the raw dataset names (`MonthlyCharges`, `Contract`). The public contract is snake_case. `pydantic`'s **`serialization_alias`** bridges them.

This is `serialization_alias`, **not** `alias` — verified difference:

| | input spelling accepted | OpenAPI schema shows |
|---|---|---|
| `alias=` | CamelCase **only** | CamelCase |
| `serialization_alias=` | snake_case only | **snake_case** ✅ |

`alias=` would have inverted the documented contract while still passing any test written around it.

### Thresholding lives in the API, not the artifact

The model was logged with `pyfunc_predict_fn="predict_proba"`, so `model.predict(df)` returns `[[p_no_churn, p_churn]]`. `main.py` takes column 1 and thresholds at `DECISION_THRESHOLD` (env, default `0.5`). The decision boundary can therefore change without retraining.

---

## Environment facts (verified by execution, not docs)

fastapi **0.141.1** · pydantic **2.13.4** · starlette **1.3.1** · prometheus_client **0.26.0** · pytest **9.1.1** · mlflow **3.15.1**

- **`CONTENT_TYPE_LATEST` is `text/plain; version=1.0.0; charset=utf-8`** — not the `version=0.0.4` every tutorial hardcodes. Import the constant.
- `generate_latest()` returns **bytes**.
- Lifespan is `@asynccontextmanager` + `FastAPI(lifespan=...)`. `@app.on_event("startup")` emits a real `DeprecationWarning`.
- **`prometheus_client` raises `DuplicateTimeseries`** if a metric name registers twice. Metrics are therefore defined once at module level — do not move them into a factory or handler.
- `TestClient` emits `StarletteDeprecationWarning` about `httpx2` on import. Harmless — but **never add `filterwarnings = ["error"]`** to pytest config; it would fail the suite at import time.
- **libomp**: LightGBM needs `brew install libomp` on macOS. Linux base images ship `libgomp` — this must **not** appear in the Dockerfile.

---

## Bugs found in review, and why they mattered

All three were caught by an adversarial review pass and are fixed. They are documented because the failure modes recur.

**1. 500s were invisible to metrics.** `await call_next(request)` re-raises when a handler fails, so both the latency observation and the request counter were skipped. Error rate — the one metric worth alerting on — was uncomputable, and the latency histogram was biased toward successes. Fixed with `try/finally` and a status defaulting to `"500"`.

**2. Unbounded Prometheus label cardinality.** `request.scope["route"]` is only populated once a route matches. Unmatched paths fell back to the raw URL, so scanner traffic (`/.env`, `/wp-admin/...`) minted a permanent time series each — unbounded memory growth in-process. Now every unmatched path collapses to `endpoint="unmatched"`. **This matters more once the service is on a public Cloud Run URL.**

**3. A defensive `reindex` was masking alias typos.** `reindex(columns=FEATURE_COLUMNS)` was added to pin column order, but it fills a missing column with `NaN` rather than raising. A one-character typo in any `serialization_alias` produced a NaN feature, got quietly imputed by the pipeline, and scored normally. Verified: the typo passed all nine tests. `StubModel.predict` in `tests/test_api.py` now asserts both column identity and absence of NaN — a mutation test confirmed it fails with `a feature arrived as NaN`.

> Generalisable: defensive code that *fills gaps* converts loud failures into silent wrong answers. For an ML service, wrong answers look plausible.

---

## Tests

11 tests, all **hermetic** — `load_model` is monkeypatched, so the suite needs no `mlflow.db` and no populated registry. This is deliberate: Milestone 3 runs it in GitHub Actions where neither exists.

```bash
uv run pytest tests/ -v
```

Note: `CLAUDE.md` says *"pytest-asyncio in STRICT mode"*. STRICT is active, but as **pytest-asyncio 1.4's built-in default** — there is no `[tool.pytest.ini_options]` anywhere. The behaviour is correct but unpinned; a library default change would break it silently.

---

## Proof there is no training/serving skew

The check that actually matters. Same customer row, both paths:

```
API /predict            → 0.06532931936364869
pipeline.predict_proba  → 0.0653293194
```

Identical. This exercises the whole chain — alias mapping, DataFrame construction, column reindex, pyfunc wrapper. **Re-run this after any change to schemas, feature columns, or the model URI.** A column-order bug still returns a plausible-looking probability, so nothing else catches it.
