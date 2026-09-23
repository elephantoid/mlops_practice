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
| **M4** — orchestration | 5-task weekly Airflow DAG: ingest → train → evaluate → promote → monitor, with an AUC-delta promotion gate and a drift-based retrain trigger; Airflow image + compose overlay | PR #6 |

**155 tests, 1 skipped.** The suite grew from the 17 the Telco milestones left behind as the
retarget landed; all but one are hermetic and run anywhere. `tests/test_skew.py` needs a
populated registry and skips without one, naming `riskwatch_credit` in its skip reason.

Both run inside the dev container, and `lightgbm`, `evidently`, `mlflow` and `sklearn` all
import on Linux. **Training and serving have still not been exercised in it** — the raw
archives are now cached (see the retarget section below), but nothing has been ingested or
trained, so the registry is empty and there is nothing to serve. That the
same-absolute-path mount keeps MLflow's artifact locations resolvable from both sides
therefore remains a design argument, not a measurement.

### M4 was demonstrated end to end — on Telco

`main` ran the retraining DAG via `airflow dags test` at each review round, before the
riskwatch retarget existed:

| | empty registry | incumbent at 0.8476 | after the round-4 rework |
|---|---|---|---|
| `task_preflight` | — (added in round 4) | — | pinned `telco_20260909T094309Z.parquet` |
| `task_promote` | promoted, `@production` → v1 | **skipped** — `delta -0.0000` | **skipped** — same |
| `task_monitor` | success | success **despite the upstream skip** | success — no prediction log, read as "no verdict" |
| drift source | synthetic | synthetic | `logs` (the new scheduled default) |

A bad `drift_source` was also confirmed to fail in `task_preflight`, before ingest. Across
all of it: three sweeps, 42 runs, and still **one** registered version with the alias on
v1. The gate refuses before registering, so a rejected candidate leaves nothing behind —
which is the property the promote-then-roll-back alternative would not have had.

**That evidence is Telco-era and does not carry over.** The DAG now points at
`riskwatch_credit`, its watched columns were retargeted, and nothing has re-run it against
credit data. Re-demonstrating M4 on the retargeted stack is outstanding work, not a
completed milestone.

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
  `prometheus`, `grafana`, `pushgateway`. Airflow and PostgreSQL arrived with M4 but live in
  a **separate overlay**, `docker-compose.airflow.yml`, so a plain `docker compose up` stays
  the light API-only path. The full stack is
  `docker compose -f docker-compose.yml -f docker-compose.airflow.yml up`.
- **M4's retrain trigger is not `drift_share > 0.2`.** That number was inherited from a spec
  written before any data existed and, with 19 features, means "4 or more columns at once" —
  which the single-column +15% scenario this repo ships can never reach. Implementing it
  verbatim would have shipped a trigger that provably never fires. `should_retrain()` in
  `src/pipelines/retrain.py` fires on a **named watched column** (credit: `AMT_CREDIT`,
  `AMT_INCOME_TOTAL`, `EXT_SOURCE_2`; fraud: `Amount`, `V14`, `V17`) *or* the 0.20 share
  as a catch-all for broad shift. On credit's 26-column frame that share means six or more
  columns at once, not the four it meant on Telco's 19.
- **M4's DAG does not chain retrains.** `AGENTS.md` says `task_monitor` triggers a retrain,
  but this DAG *is* the retrain and retraining does not move the reference distribution — so
  an unguarded self-trigger loops forever. A drift-triggered run never triggers another.
- **Drift compares against the pre-ingest snapshot, not `latest.parquet`.** `ingest()`
  repoints that symlink and runs *first*, so reading the default meant comparing live
  traffic against data the serving model had never seen and calling the difference drift.
  `task_preflight` resolves the concrete snapshot before anything mutates and passes it to
  `drift.run(reference_path=...)`. The same task validates `drift_source` up front — it was
  previously checked in the last task, so a typo could ingest, sweep, evaluate and **promote
  a model** before failing on a bad string.
- **The prediction log is read over a 7-day window, not in full.** Unbounded reads grow
  forever and, worse, let months-old traffic keep a resolved drift signal alive. Seven days
  because the DAG is `@weekly`: the window covers traffic since the last run.
- **A scheduled run measures drift against the prediction log, not the synthetic batch.**
  `SCHEDULED_DRIFT_SOURCE = "logs"`. The synthetic batch shifts the track's drift column by
  construction, so it always reports drift on a watched column — scheduling it would have
  made every weekly run trigger a second full sweep over identical data, forever,
  on a manufactured signal. Synthetic is now a manual known-positive fixture for proving the
  detector still fires. When the log holds too little traffic the drift check is skipped
  cleanly (`InsufficientCurrentData`) rather than failing the task: on a fresh deployment
  "no traffic yet" is the expected state, not an incident.
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

- [x] raw archives cached (2026-09-22) — but `src/data/ingest.py` is **not** yet writing
      validated parquet; that is plan Step 4 and is the next unit of work
- [ ] `src/models/train.py` populating the registry and promoting the production alias
- [ ] `docker compose up` serving from the registry; the six API panels fill under load
- [ ] `src/monitoring/drift.py --push` filling the seventh panel, **Data drift share**
- [ ] the baked path exercised: export, `docker build`, `docker run`, and `/health`
      reporting a real version rather than "unknown"
- [ ] runtime image measured, and trimmed if one version puts the registry over 0.5 GB

Concurrent, not a prerequisite: `dags/riskwatch_retrain.py` - implemented by PR #6 and
retargeted in this merge, but blocked on Step 4's ingest - running
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
`riskwatch`. The inherited suite still reports 16 passed / 1 skipped; the full suite, with
everything the retarget added and M4's suite from `main`, reports 155 passed / 1 skipped.

**Data acquired 2026-09-22.** The credit-source decision resolved to **A (Home Credit)**:
the user supplied a Kaggle credential and accepted the `home-credit-default-risk`
competition rules, so the OpenML/UCI fallback was not taken and the 6-8h rewrite it would
have cost was not incurred. The 26-column `FeatureSpec` and the `CreditPredictRequest`
fields, both written against Home Credit columns, stand.

| | Cached at | Verified |
|---|---|---|
| `data/raw/credit/application_train.csv` | 688 MB archive, 1 of 10 members extracted | **307,511 x 122, positive rate 0.0807**, `SK_ID_CURR` unique -- matching the figures the plan was designed against |
| `data/raw/fraud/creditcardfraud.zip` | 66 MB | not yet extracted; the fraud track lands in W2 |

Both are gitignored, so a fresh clone still cannot train or serve until they are
re-downloaded. `src/data/schemas/home_credit_columns.txt` carries the 122-name manifest,
written from the archive rather than from memory.

The downloads ran dataset-before-competition deliberately: a Kaggle *dataset* needs only a
token, so proving it in isolation is what makes a subsequent competition 403 diagnosable as
*consent* rather than credentials. `src/data/kaggle_source.py` encodes that distinction --
`KaggleConsentError` carries the rules URL, `KaggleAuthError` points at settings.

One defect surfaced the moment real credentials existed, and it is the reason this section
is worth reading: `credentials_available()` hardcoded `~/.kaggle/kaggle.json`, while the
credential on this machine is `~/.kaggle/access_token`. The CLI authenticates from either.
`acquire()` consulted its own check, declared no credential, and fell back to OpenML --
silently substituting a *different dataset* while the correct archive sat extracted on
disk. It now accepts either credential filename the CLI reads from `~/.kaggle`, plus the
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment pair.
The `equivalent_to_primary=False` flag on the credit fallback is what made the
substitution visible rather than reading as a routine retry.

**Still blocked, and why:** plan Step 4 (retargeting `src/data/ingest.py`, adding a
HomeCreditSchema and a per-track credit loader) and the measurement half of Step 6 (metric
values, the
`cv_auc_mean > 0.6` floor, `source_used` tagging) are **not done**. They were blocked on
the data until today and are the next unit of work. `src/data/ingest.py` is still entirely
Telco and says so in its own module docstring. Nothing has trained yet, so
`models:/riskwatch_credit@production` does not resolve and `tests/test_skew.py` correctly
skips naming it.

**Names in force after the retarget:** two registered models, `riskwatch_credit` and
`riskwatch_fraud`, with independent schemas, thresholds, and retrain cadence. The `churnwatch`
registered model in the `spookfish` worktree is Telco-era and is now an orphaned historical
artifact.

## Before you can run anything

Everything needed to train, serve from the registry, or compute drift is gitignored:
`data/raw/`, `data/processed/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `build/`. **A fresh
clone has none of it** and must run ingest and training first. The hermetic tests are the
only thing that works out of the box.

Machine-local as of this update: the primary checkout at
`~/Documents/projects/mlops_practice` now holds both raw archives — the credit one
extracted to its application table, the fraud one still zipped — but nothing has been
ingested or trained, so there is nothing to serve. The registry holds **0 runs and 0 model
versions**; `registered_models` carries a single `riskwatch_credit` row with no versions
and no aliases, an empty shell left by a partial run, which is why
`models:/riskwatch_credit@production` does not resolve and `tests/test_skew.py` skips.

Two worktrees are populated instead, and **both are Telco-era**:

- `spookfish` (`~/orca/workspaces/mlops_practice/spookfish`) — 14 runs, the `churnwatch`
  registered model, `data/raw/telco.csv`.
- `horseshoe` — populated by the M4 verification runs above: 42 runs, `churnwatch` v1 on
  `@production`, plus `data/raw/telco.csv` copied in from `spookfish`.

Pointing `MLFLOW_TRACKING_URI` at either resolves the **old** `churnwatch` model only;
after the retarget the names in force are `riskwatch_credit` and `riskwatch_fraud`. They
are kept as a record of how M4 was demonstrated, not as working registries.

The DAG needs no extra *wiring*: the Airflow overlay bind-mounts the repo, so `task_ingest`
writes `data/processed/` back into the working tree and the sweep logs through the `mlflow`
service into the same `mlflow.db` the host reads.

**But it cannot complete a run today, and that is expected.** `task_ingest` calls
`src/data/ingest.py`, which is still entirely Telco and says so in its own docstring: it
reads `data/raw/telco.csv`, which is not in this checkout. A scheduled run reaches task 2
of 5 and stops with `FileNotFoundError`. Retargeting that module is plan Step 4, the next
unit of work, and the DAG is correct the moment it lands -- every other task already
threads the track through, and `task_preflight` resolves the per-track baseline rather
than the flat path nothing writes.
