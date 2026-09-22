# STATUS — RiskWatch

**Last updated: 2026-09-22.**

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
| **M1** — training | pandera-validated ingest → versioned parquet; per-model preprocessing; 14-config sweep promoting the best CV AUC to `models:/churnwatch@production` *(Telco-era; the name is historical — see the retarget section below)* | PR #1 |
| **M2** — serving | FastAPI `POST /predict`, `GET /health`, `GET /metrics`; pydantic v2 snake_case contract; multi-stage non-root Dockerfile; host-side model export | PR #2 |
| *(unplanned)* — local observability | prediction JSONL log; Prometheus + Grafana; Evidently drift via Pushgateway | PR #3 |

17 tests. 16 are hermetic and run anywhere; `tests/test_skew.py` needs a populated registry
and skips without one.

Both run inside the dev container: `make check` reports the same 16 passed / 1 skipped
there as on the host, and `lightgbm`, `evidently`, `mlflow` and `sklearn` all import on
Linux. **Training and serving have not been exercised in it** — this checkout has no
`data/raw/` and an empty registry, so there is nothing to train on or serve. That the
same-absolute-path mount keeps MLflow's artifact locations resolvable from both sides
therefore remains a design argument, not a measurement.

## Not built

- `dags/riskwatch_retrain.py` — **0 bytes.** The `drift_share > 0.2` retrain trigger exists
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
- **Development moved into a Linux container.** `make shell` is now the way in; the macOS
  host still works and is documented as the fallback. Added rather than planned — `AGENTS.md`
  never asked for it. The motivation was the host/target split the docs kept having to warn
  about (`libomp` vs `libgomp`).
- **Cloud Run is deferred with named triggers (2026-09-22).** It was undecided; it is now a
  decision. Cloud Run can bill and nothing through M2 needs a public URL, so the pipeline is
  finished and exercised locally first and the deploy becomes a verification step. The
  decision table in `AGENTS.md` previously said "chosen" while this file said "undecided";
  the table has been revised and the two now agree. The condition that ends the deferral is
  below; the reasoning is in `docs/debt-ledger.md`.
- **An nginx ingress was considered and dropped.** It would have fronted the five services
  on one port, which is a real pattern, but nothing in this stack needs it today: no TLS to
  terminate, no static files, no second backend. Recorded so it is not re-proposed as new.
- **`docker-compose.yml` is not what M2 described.** Services are `api`, `mlflow`,
  `prometheus`, `grafana`, `pushgateway` — there is no Airflow service and no PostgreSQL.
  Airflow arrives with M4, if it arrives.

## What has to be true before the Cloud Run move

Set 2026-09-22 and revised the same day, when the Q4 plan in `~/Documents/career/` was
read. The move is now scheduled for 10/5-10/11, **concurrent with the retraining DAG
rather than after it**: that plan's definition of done asks for a public URL with 75+ days
of uptime, and the clock has to start in early October to be true by mid-December. The
earlier framing had the deploy waiting on everything local, which would have made that
DoD unreachable.

**Cost was checked rather than assumed.** Cloud Run scales to zero and does not bill idle
time unless minimum instances are set above zero. The request-based free tier is 180,000
vCPU-seconds, 360,000 GiB-seconds and 2 million requests a month, which this service will
not approach even with the scanner traffic `tests/test_api.py` already accounts for. The
charge that does apply is **Artifact Registry, free only to 0.5 GB** - and a runtime image
carrying scikit-learn, LightGBM, MLflow and pyarrow clears that on its own, with every
rebuild adding another version. Image size and a registry cleanup policy are real
constraints here, not housekeeping. Egress is 1 GiB free per month in North America.

Before the deploy:

- [ ] `data/raw/` populated, `src/data/ingest.py` writing validated parquet
- [ ] `src/models/train.py` populating the registry and promoting the production alias
- [ ] `docker compose up` serving from the registry; the six API panels fill under load
- [ ] `src/monitoring/drift.py --push` filling the seventh panel, **Data drift share**
- [ ] the baked path exercised: export, `docker build`, `docker run`, and `/health`
      reporting a real version rather than "unknown"
- [ ] runtime image measured, and trimmed if one version puts the registry over 0.5 GB

Concurrent, not a prerequisite: `dags/riskwatch_retrain.py` - **0 bytes today** - running
all five tasks end to end.

**The artifact-path question is now live.** It was deferred earlier the same day with three
triggers, the first being the Cloud Run move; that trigger now fires in about two weeks.
The answer is forced rather than open: Cloud Run has no host filesystem to mirror, so the
registry path - which resolves through the absolute `artifact_location` MLflow writes into
`mlflow.db` when the experiment is created - cannot follow the service there. What deploys
is the baked path: `src/models/export.py` into `build/model/`, copied into the image, and
already implemented. Keeping it in the checklist above is what makes the move a deploy
rather than a redesign.

One thing this file does not yet reflect in full: the Q4 plan retargets this project from
Telco churn to credit and fraud during W1-W2, so the service deployed in W3 is the
retargeted one, not the Telco service described above.

## Retarget to riskwatch — in progress, started 2026-09-22

The consensus-approved plan is at
`.gjc/_session-01a0c6e0-c8f0-7303-ad50-3139990a0154/plans/ralplan/01a0c6e0-c8f0-7303-ad50-3139990a0154/pending-approval.md`
(Architect `CLEAR`/`APPROVE` + Critic `OKAY` after three review passes).

**Landed:** the atomic rename `churnwatch` → `riskwatch` across code, config, container,
Prometheus/Grafana, and tests, plus `kaggle` as a declared dependency. `uv.lock` carries
`riskwatch`. The inherited suite still reports 16 passed / 1 skipped.

**Blocked:** data acquisition. Kaggle needs two browser-only actions from the user — an API
token at `kaggle.com/settings`, and acceptance of the `home-credit-default-risk` competition
rules. Neither has an API path. No raw data directory exists in this checkout yet, so
training, serving from the registry, and drift all remain unrunnable.

### Open decision — credit data source (gate fires end of 2026-09-22)

The plan's Step 0 gate and its decision table both say **"decided by end of D2 (9/22) —
Step 4 does not start undecided"**. That is today. This is a decision, not a wait: the two
branches cost different things and the difference grows the longer it is deferred.

| | **A — Home Credit** (plan of record) | **B — OpenML 42477 / UCI Taiwan** (fallback) |
|---|---|---|
| Access | Kaggle token **+ browser rules acceptance** | Auth-free, verified reachable |
| Shape | 307,511 × 122 | 30,000 × 24 |
| Positive rate | ~8.07% | ~22.1% |
| Cost to take | 2 browser actions, minutes | **6–8h rewrite** |
| What the rewrite discards | — | Step 4's schema, the 26-column `FeatureSpec` in `src/features/specs.py`, and the `CreditPredictRequest` fields in `src/api/schemas.py`, all of which are written against Home Credit columns |
| Résumé recognisability | High | Lower |

**A is still recommended** — the cost is two browser clicks against a 6–8h rewrite, and
the work already committed is written against Home Credit's columns. The gate exists so
that the choice is *made* rather than defaulted into by inaction: every day A is not
chosen is a day the 6–8h rewrite gets harder to absorb against the W3 deploy date, which
is immovable because DoD ① needs 75+ days of uptime from early October.

Note: the fraud track is unaffected. Its source (`mlg-ulb/creditcardfraud`) is a Kaggle
*dataset* needing only the token, and OpenML 1597 is an auth-free equivalent — the same
data by another route, unlike the credit fallback, which is a different dataset.

**To take A**, in this order:
1. kaggle.com/settings → "Create New Token" → save to `~/.kaggle/kaggle.json`, `chmod 600`
2. <https://www.kaggle.com/c/home-credit-default-risk/rules> → Accept
3. `kaggle datasets download -d mlg-ulb/creditcardfraud` (proves the token alone)
4. `kaggle competitions download -c home-credit-default-risk`

Step 3 before step 4 deliberately: a dataset needs only the token, so proving it in
isolation is what makes a subsequent 403 diagnosable as *consent* rather than credentials.
`src/data/kaggle_source.py` encodes that distinction — `KaggleConsentError` carries the
rules URL, `KaggleAuthError` sends you to settings.

**Names in force after the retarget:** two registered models, `riskwatch_credit` and
`riskwatch_fraud`, with independent schemas, thresholds, and retrain cadence. The `churnwatch`
registered model in the `spookfish` worktree is Telco-era and is now an orphaned historical
artifact.

## Before you can run anything

Everything needed to train, serve from the registry, or compute drift is gitignored:
`data/raw/`, `data/processed/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `build/`. **A fresh
clone has none of it** and must run ingest and training first. The hermetic tests are the
only thing that works out of the box.

Machine-local as of this update: in the primary checkout at
`~/Documents/projects/mlops_practice` the registry is empty (0 runs, 0 registered models)
and `data/raw/` is absent. The populated copy lives in the `spookfish` git worktree at
`~/orca/workspaces/mlops_practice/spookfish` — 14 runs, the `churnwatch` registered model,
and `data/raw/telco.csv`. That copy is **Telco-era and historical**: after the retarget the
names in force are `riskwatch_credit` and `riskwatch_fraud`, so pointing
`MLFLOW_TRACKING_URI` at that worktree resolves the old model only. It is kept as a record,
not as a working registry.
