---
paths:
  - "src/api/**"
  - "tests/test_api.py"
  - "tests/test_model_loading.py"
  - "tests/test_model_resolution.py"
  - "tests/stubs.py"
  - "tests/conftest.py"
---

# Serving layer

## `serialization_alias`, never `alias`

The trained pipeline's columns are the raw dataset names (`AMT_INCOME_TOTAL`, `NAME_CONTRACT_TYPE`).
The public contract is snake_case. Verified difference:

| | input spelling accepted | OpenAPI schema shows |
|---|---|---|
| `alias=` | raw column name **only** | raw column name |
| `serialization_alias=` | snake_case only | snake_case |

`alias=` would invert the documented contract while still passing any test written around it.

## Never fill a missing feature column

A defensive `reindex(columns=FEATURE_COLUMNS)` fills a missing column with `NaN` rather than
raising. A one-character typo in any `serialization_alias` then produces a NaN feature, gets
quietly imputed by the pipeline, and scores normally — verified: the typo passed all nine
tests that existed at the time.

Two tests split that guard, and both must stay. `StubModel.predict` in `tests/stubs.py` asserts
column identity and no NaN in a **required** column, because a required NaN can only be a
reindex-filled typo. An optional column may legitimately be null (73% of credit applicants have
one), so an alias typo there is caught by `test_request_aliases_cover_the_feature_contract` in
`tests/test_api.py`, which compares every alias to the contract.

> Defensive code that *fills gaps* converts loud failures into silent wrong answers. For an
> ML service, wrong answers look plausible.

## Prometheus

- Metrics are defined **once at module level**. `prometheus_client` raises
  `DuplicateTimeseries` if a name registers twice — do not move them into a factory or a
  handler.
- `CONTENT_TYPE_LATEST` is `text/plain; version=1.0.0; charset=utf-8`, **not** the
  `version=0.0.4` every tutorial hardcodes. Import the constant.
- `generate_latest()` returns **bytes**.
- Every unmatched path must collapse to `endpoint="unmatched"`. `request.scope["route"]` is
  only populated once a route matches, so scanner traffic (`/.env`, `/wp-admin/...`) would
  otherwise mint a permanent time series each — unbounded in-process memory growth, and the
  service is headed for a public Cloud Run URL.
- The middleware needs `try/finally`. `await call_next(request)` re-raises when a handler
  fails, so without it both the latency observation and the request counter are skipped and
  the error rate becomes uncomputable.

## Tests

The suite is **hermetic** — `load_model` is monkeypatched, so it needs no `mlflow.db` and no
populated registry. Keep new tests hermetic where you can. A test that needs a real model skips
through `tests.preconditions.skip_or_fail`, **never a bare `pytest.skip`**: CI trains both tracks
on synthetic data and runs with `RISKWATCH_REQUIRE_MODELS=1`, so a missing model there is a
failure, and a bare skip would let it pass green. Only a test that needs the real competition
data uses `pytest.skip` — CI cannot have that data.

**Never add `filterwarnings = ["error"]`** to the pytest config. `TestClient` emits a
`StarletteDeprecationWarning` about `httpx2` on import and the suite would fail at import
time.

`pytest-asyncio` STRICT mode is active as the library's own 1.4 default — there is no
`[tool.pytest.ini_options]` anywhere. Correct, but unpinned.
