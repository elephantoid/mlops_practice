# STATUS — RiskWatch

**Last updated: 2026-09-28** (post-Step 9 state check — both tracks now registered in the
primary checkout, and the baked artifact caught one version behind the alias).

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

**313 collected; 301 passed / 12 skipped on a machine with no data and no registry**
(measured 2026-09-28 after Step 9, `uv run pytest -q` and `pytest --collect-only -q`). The
suite grew from the 17 the Telco milestones left behind as the retarget landed.

**The previous figure in this slot — "233 passed, 1 skipped" — was wrong by 7, and the error is
in the total rather than in the environment.** At the Step 8 commit `pytest --collect-only`
reports **241**, and collection does not depend on whether data or a registry is present
(verified both ways). 233 + 1 = 234 cannot be a run of 241 tests. This is the second time a
hand-maintained count in this file has been wrong, so the line now records the collected total
alongside the run: collection is the number that can be rechecked without reproducing an
environment.

The twelve skips are entirely gitignored state, and each names what is missing: `build/model`
absent (1), no fraud snapshot or cached archive (3), neither track's model registered (2), and
`tests/test_reachable_decisions.py` needing both a snapshot and a model for each track (6).
The predicted fully-populated count in this slot was **310 passed / 3 skipped**; measured
2026-09-28 in the primary checkout with both models registered, both snapshots present and an
export on disk, it is **314 collected, 314 passed, nothing skipped and nothing failing**. The
three residual skips that figure predicted were not a floor — they assumed no export and no
cached archives, which this checkout has. `tests/test_skew.py` reads **2 passed, 0 skipped**,
which is the number the plan's W2 gate asks for. That file failed on purpose for part of Step 9,
until `POST /predict/fraud` landed — see the Step 9 section.

The collected total moved 313 → 314 with the staleness check recorded at the end of this
file; on a machine with no data, no registry and no export it is one more skip than the twelve
above, gated on the same `build/model` that gates the export test beside it.

Both run inside the dev container, and `lightgbm`, `evidently`, `mlflow` and `sklearn` all
import on Linux. **Training and serving have now been exercised on the host** — ingest wrote
a validated snapshot, a sweep registered `riskwatch_credit` v5 on `@production`, and
`POST /predict/credit` served a real prediction off it. **Not yet inside the container**,
which is W2 Step 13's protected deploy rehearsal. That the
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

- [x] raw archives cached (2026-09-22); both tracks ingested to validated parquet
      (credit 2026-09-28 Step 4, fraud 2026-09-28 Step 8)
- [x] `src/models/train.py` populating the registry and promoting the production alias —
      **both tracks, in the primary checkout** as of 2026-09-28: `riskwatch_credit` v5 and
      `riskwatch_fraud` v1, each on `@production`. Step 9 had trained fraud only in its own
      worktree's registry (`mlruns/` and `mlflow.db` are gitignored, so a registry is per
      working tree), which left four tests skipping here for want of a model that existed
      elsewhere. The sweep was re-run against this tree's snapshot and reproduced Step 9's
      number exactly — `cv_pr_auc_mean` **0.8186**, the same `lightgbm-class_weight=balanced-
      learning_rate=0.05-n_estimators=300-num_leaves=8` configuration winning the same grid.
      A figure measured twice in two registries off the same archive is the one kind of
      reproducibility this project can claim cheaply, so it is recorded rather than assumed
- [ ] `docker compose up` serving from the registry; the six API panels fill under load
- [ ] `src/monitoring/drift.py --push` filling the seventh panel, **Data drift share**
- [ ] the baked path exercised: export, `docker build`, `docker run`, and `/health`
      reporting a real version rather than "unknown"
- [ ] runtime image measured, and trimmed if one version puts the registry over 0.5 GB

Concurrent, not a prerequisite: `dags/riskwatch_retrain.py` - implemented by PR #6 and
retargeted in this merge - running all five tasks end to end. No longer blocked: Step 4
landed the track-driven ingest it calls.

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
everything the retarget added and M4's suite from `main`, reports 180 passed / 0 skipped.

**Data acquired 2026-09-22.** The credit-source decision resolved to **A (Home Credit)**:
the user supplied a Kaggle credential and accepted the `home-credit-default-risk`
competition rules, so the OpenML/UCI fallback was not taken and the 6-8h rewrite it would
have cost was not incurred. The 26-column `FeatureSpec` and the `CreditPredictRequest`
fields, both written against Home Credit columns, stand.

| | Cached at | Verified |
|---|---|---|
| `data/raw/credit/application_train.csv` | 688 MB archive, 1 of 10 members extracted | **307,511 x 122, positive rate 0.0807**, `SK_ID_CURR` unique -- matching the figures the plan was designed against |
| `data/raw/fraud/creditcardfraud.zip` | 66 MB, `data/raw/fraud/creditcard.csv` extracted 2026-09-28 | **284,807 x 31, positive rate 0.001727** (492 positives), zero nulls, `Time` monotonic non-decreasing and **not** unique -- 124,592 distinct values over 284,807 rows |

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

**Step 4 and Step 6 landed 2026-09-28, closing the W1 exit criteria.**
`src/data/credit.py` carries `HomeCreditSchema` (the modeled subset, validated lazily so a
corrupted batch reports every violation at once) plus a fingerprint against the committed
122-name manifest. `src/data/ingest.py` is track-driven and writes
`data/processed/<track>/` with `source_used` in the parquet metadata. A real sweep ran:
**`riskwatch_credit` v5 on `@production`, `cv_auc_mean` 0.7524** — promoted through the
tree-model gate, which skipped the logreg arm. (v1–v4 are the earlier attempts; see the
registry note near the end of this file for why they are still there.)

Measured on the real archive, matching the plan's figures exactly: 307,511 rows, positive
rate 0.0807, the `DAYS_EMPLOYED` sentinel on 18.0% of rows, `CODE_GENDER == "XNA"` on 4.

What the first real training run bought, beyond the model, was two defects that **only a
registered model could expose** — both caught by `tests/test_skew.py`, both the same class
of mistake:

- The derived `DAYS_EMPLOYED_ANOMALY` column reached `fit()`, so the logged signature was
  27 wide against a 26-column request contract. Every request would have failed validation
  for omitting a column the caller cannot know. The derivation moved inside the pipeline,
  where both paths reach it.
- `downcast()` narrowed dtypes, so the signature demanded `float`/`integer` while the API —
  building frames from JSON, where numbers arrive 64-bit — sent `double`/`long`. It is now
  a documented no-op: storage must not dictate the serving contract.

Both would have passed every test on either side alone. That is what the skew test is for,
and it could not do its job until something was registered.

Also fixed: a baked artifact reported `model_version: "unknown"`, because the container has
no registry to ask and nothing read the `MODEL_VERSION` file `src/models/export.py` writes
beside the artifact. W3's deploy acceptance asks for a real version from the public URL.

**Verified end to end locally:** `POST /predict/credit` with the example request returns
200 and `risk_probability` 0.4234.

**The version that record originally claimed was wrong, and the correction is the point.** It
read `model_version: "5"` — *read from the artifact's own `MODEL_VERSION` file* — and both halves
cannot be true at once. **5 is the registry's alias; the artifact on disk held 4.** `build/model`
was exported at 11:39 and credit v5 was registered at 12:08, so anything reading that file, which
is exactly what a baked container does instead of asking a registry, would have reported and
served **4**. The end-to-end claim above stands; the version attached to it was the alias, not
the artifact. See the staleness check at the end of this file.

That probability has since been three different decisions without the model changing, which is
worth keeping as a record of what a band is: `review` against the W1 placeholder (0.40, 0.60),
`review` again against Step 9's first priced band (0.0014, 0.98) — where *everything* reviewed —
and **`decline`** against the budgeted band (0.0645, 0.0963) in force now. Same model, same
applicant, same number. Only the boundary moved.

**Names in force after the retarget:** two registered models, `riskwatch_credit` and
`riskwatch_fraud`, with independent schemas, thresholds, and retrain cadence. Both now exist
in the primary checkout's registry. The Telco-era `churnwatch`
registered model lived in the `spookfish` worktree, which no longer exists — see the note near
the end of this file.

**Step 8 landed 2026-09-28 — the fraud track is registered, and the seam held.**

Both tracks are now in `TRACKS` and `FEATURE_SPECS`. Measured off the real archive and the
snapshot it produced, not asserted:

| | credit | fraud |
|---|---|---|
| rows x source columns | 307,511 x 122 | **284,807 x 31** |
| positive rate | 0.0807 | **0.001727** (492 positives) |
| modeled features | 26 (15 numeric + 11 categorical) | **29, all numeric, zero categoricals** |
| id column | `SK_ID_CURR`, from the source | **`TransactionIndex`, derived** — the source ships none |
| parquet | 11 MB | **72.9 MB** |
| registered model | v5 on `@production` | none yet (Step 9 onward) |

**What adding a track actually cost.** The W1 seam claimed one new module plus one registry
entry. Measured, with two deviations both reported rather than worked around:

1. One new data module (`src/data/fraud.py`), its registry entry and feature contract — as
   claimed.
2. **One extraction.** The column-manifest comparison W1 had written inside
   `src/data/credit.py` is track-agnostic, so it moved to `src/data/fingerprint.py` and both
   track modules now bind their own manifest to it. The alternative was a second copy of a
   45-line comparison with two places to fix a message. The shared function takes **no
   default path**, which makes the "every track fingerprints against credit's manifest"
   defect unrepresentable rather than merely absent.
3. **One log message in `src/data/ingest.py`** — found by Copilot review, and the most
   interesting thing the tripwire surfaced. The fallback warning stated "these are different
   datasets, validation is expected to fail" unconditionally, which was true of the only
   fallback that existed when it was written (credit's UCI substitute) and false for fraud's
   OpenML 1597. A fraud fallback would have logged a false operational diagnosis to whoever
   was already debugging an acquisition failure. It now reads `equivalent_to_primary`, which
   carried the distinction all along.

Tripwire, exactly: `git diff --numstat origin/main` over `src/data/ingest.py`,
`src/features/pipeline.py`, `src/models/train.py` and `src/data/kaggle_source.py` is
`29 9 src/data/ingest.py` and nothing else — **zero changed `def` lines**, so no shared
*contract* moved; one diagnostic inside one function body did. Leaving a message that is now
false in place purely to keep the diff empty would have been the forced workaround the
tripwire exists to prevent.

**Two design calls the source forced.**

- **The id column is derived**, because the ULB extract has no identifier and `Time` is not
  one: 160,215 of 284,807 rows share a second with another row, and **1,081 rows are exact
  duplicates across all 31 source columns**, so no combination of source columns keys the
  frame either. `clean()` assigns a positional `TransactionIndex`. That is sound rather than
  arbitrary here because `Time` is monotonic non-decreasing over the file (verified), so row
  order *is* arrival order — the index carries the chronology a later time-ordered split
  needs. It is an id, not a derived feature: `derived_features` is empty, so it never reaches
  `fit()` and never appears in a request.
- **`Time` is validated but not modeled.** No live caller can produce "seconds since the
  first transaction of this extract"; it is monotonic in row position, so a tree handed it
  memorises *when* in this particular 48 hours the frauds were; and its live distribution
  differs from the snapshot's by construction, so it would register as drifted on every run
  and inflate the share the retrain trigger reads. It stays under schema contract because an
  upstream that drops it or switches to absolute timestamps is a change worth failing on. The
  rejected alternative — a time-of-day feature derived from it, which a caller genuinely can
  send — is in `docs/debt-ledger.md`.

**No display names for `V1`..`V28`.** They are PCA components; ULB never published the
loadings, so what each measures is not recoverable. `display_name()` falls back to the raw
column, so a reason code reads `V14`. `Amount` is mapped because its meaning survived.

**A finding, recorded rather than tuned away.** `drift --track fraud --source synthetic` runs
and writes a report, which is the Step 8 acceptance — but the documented +15% batch is **not**
a known-positive on this track. The lift on `Amount` scores 0.0622 against Evidently's 0.100
normalised-Wasserstein threshold and goes undetected, while an untouched `V15` scores 0.1007
on 500-row sampling noise. Amount's tail is the cause: mean 88.35, std 250.12, so +15% is
0.053 sigma. No fraud watched column fires and the share stays 0.0345, so that batch cannot
trigger a retrain. Raising the multiplier until it fires would be the same person choosing
both the perturbation and the threshold it must clear; it is re-derivation debt for W3 Step 17
alongside `drift_share > 0.2` and `MIN_CURRENT_ROWS`. On fraud, that command currently proves
the drift path runs end to end and nothing about the detector's sensitivity.

**Step 9 landed 2026-09-28 — operating points from a cost matrix, and the metric key stopped lying.**

Two changes that meet in one place. `src/models/thresholds.py` (new) computes the operating
points bounding the review band from a cost matrix; `src/models/costs.py` (new) holds the cost
matrices and the closed form, and imports **nothing** so `src/api/main.py` can use it.
`FeatureSpec.selection_metric` makes the model-selection metric a track setting — credit
ROC-AUC, fraud PR-AUC — and the MLflow key now names the metric it holds.

**Why two thresholds need three costs.** A 2x2 cost matrix has exactly one crossing, so it
yields one threshold; the three-valued contract needs two. The third action supplies the second
boundary and carries its own flat cost:

```
p_lower = C_R / C_FN        p_upper = 1 - C_R / C_FP        band exists iff sum < 1
```

`optimise_bands` finds the same point empirically. The objective separates into a term in
`lo` and a term in `hi`, so it is two independent argmins over the distinct scores — O(n log n),
not a 2-D grid. On calibrated scores the two forms agree; `tests/test_thresholds.py` asserts the
exact, non-statistical half of that (the empirical cost can never exceed the analytic one on
its own sample) after the location comparison turned out to be a coin toss, for a measured
reason: the objective is flat at its optimum, so the argmin's scatter falls like `n**(-1/3)`
— RMS 0.0113 at n=50k, 0.0099 at 200k, 0.0038 at 800k, 0.0020 at 3.2M over six seeds.

**Measured, on real models trained in this worktree** (`uv run python -m src.models.thresholds
--track <t>` reproduces every number below):

| | credit | fraud |
|---|---|---|
| selection metric | `cv_roc_auc_mean` **0.7524** | `cv_pr_auc_mean` **0.8186** |
| holdout ROC-AUC / PR-AUC | 0.7567 / 0.2486 | **0.9817 / 0.7623** |
| error costs (C_FN : C_FP) | 0.70 : 0.05 (14:1) | 1.00 : 0.10 (10:1) |
| **review budget** | **15%** | **0.5%** |
| implied review price (measured) | **0.045183** = 4.52% of exposure | **0.088571** = 8.86% |
| two-action Bayes cut `C_FP/(C_FP+C_FN)` | 0.0667 | 0.0909 |
| served band | **(0.0645, 0.0963)** | **(0.0886, 0.1143)** |
| approve / review / decline share | 58.41% / 15.00% / 26.60% | 97.57% / 0.50% / 1.93% |
| precision / recall at the decline boundary | 0.1840 / 0.6066 | 0.0819 / 0.9184 |
| precision / recall at 0.5, for contrast | 0.5890 / 0.0173 | 0.3718 / 0.8878 |

Credit's ROC-AUC 0.7567 against PR-AUC 0.2486 is the divergence the per-track metric exists
for, and fraud's 0.9817 against 0.7623 is the same gap at a 47x lower positive rate.

**The review band is budgeted, not priced — and the first attempt at pricing it collapsed.**

The operating points started as a *priced* review: `C_R` set to 0.001, "one underwriting review
costs 0.1% of the loan", a number nobody measured. That produced a credit band of
`[0.0014, 0.98]` against a model whose holdout scores run 0.0027 to 0.7816 — so **100.000% of
61,503 applicants routed to review, with both `approve` and `decline` unreachable.** The
three-valued contract was a constant function and every test was green, because
`tests/test_api.py` builds its probabilities *relative to the band* and is therefore green for
any band at all.

It was not a bad guess so much as the known boundary case of the method. Pricing abstention is
classification with a reject option (Chow 1970) and the closed form here is its asymmetric
two-class version; when the reject price is small against the misclassification costs, the
optimal policy is to reject everything. A review at 0.1% of a loan against a missed default at
70% is exactly that regime. **The error ratio was never the problem:** 14:1 is in line with
published work on this dataset (LGD 0.65 against a 0.12 margin, 5.4:1), and the two-action cut
those numbers imply — 0.0667 for ours, 0.1558 for theirs — sits comfortably inside the score
distribution either way. The collapse came from the third cost alone.

**The fix inverts which quantity is guessed.** Bounded abstention states the *capacity* a human
team has, which is a fact, and reports the price that capacity implies. Reviewing is free but
rationed, so the rows worth reviewing are the ones where a forced decision is most likely wrong,
ranked by `min(p·C_FN, (1−p)·C_FP)`. That function peaks at the two-action cut and falls away on
both sides, so the top `budget` fraction is an *interval* straddling it, with edges `λ/C_FN` and
`1 − λ/C_FP` — the same closed form, with a Lagrange multiplier where the guess used to be. The
analytic-optimum acceptance test is untouched by the switch, and
`tests/test_thresholds.py::test_bands_for_budget_is_the_closed_form_at_the_resolved_price`
asserts the equivalence rather than assuming it.

**And the implied price is now the interesting number.** A 15% credit referral budget implies a
review is worth **4.52% of the exposure** — far above the ~0.1% an underwriting review plausibly
costs. That gap is a result, not an error: at 15% the shadow price of review capacity sits well
above what review actually costs, so **capacity is the binding constraint and buying more of it
has positive expected value.** The priced formulation could not express that conclusion at all;
it could only answer "review everybody". Regenerate with
`uv run python -m src.models.thresholds --track <t> --budget-sweep`, which prints the
budget→price→band sweep and a reachability check.

**Two findings, neither tuned away.**

1. **`tests/test_api.py::test_decision_bands_are_three_valued` was a hollow pass**, found by
   Copilot review: it constructs probabilities relative to the band, so it is green for a band no
   model can reach either end of — which is precisely the state that shipped for one commit.
   Reachability needs a score distribution and therefore cannot be hermetic, so
   `tests/test_reachable_decisions.py` is the suite's one deliberately unhermetic file: it gates
   on a registered model plus an ingested snapshot, asserts all three outcomes are populated,
   asserts the band delivers roughly its stated budget, and asserts the **committed review price
   still resolves against the model being served** — because that price is a measurement with a
   shelf life, and promoting a differently calibrated model silently invalidates it.
2. **`tests/test_skew.py[fraud]` failed rather than skipped for part of this step, and that is
   what bought `POST /predict/fraud`.** Registering a fraud model while `SERVING_CONTRACTS` had no
   fraud entry is the exact state that test was written to refuse: *a registered model the API
   cannot serve is a model nothing checks for skew.* It was a tripwire laid in Step 8 firing on
   the first step that could trip it. The endpoint was in the plan's target API contract, assigned
   to no step, and outside Step 9's stated file list; the test is what made the omission
   impossible to ship quietly. It now reads **2 passed, 0 skipped**.

**A calibration gap, measured rather than assumed.** Serving is handed the *analytic* band, not
one fitted to a model's scores — a fitted band is coupled to one artifact, which contradicts
this repo's rule that thresholds are a serving concern that moves without retraining. The size
of the gap is therefore a measurement of miscalibration rather than a number to close:

| | served (budgeted) | unconstrained cost optimum on the holdout |
|---|---|---|
| credit | (0.0645, 0.0963) | (0.0618, 0.0987), review share 17.68% |
| fraud | (0.0886, 0.1143) | collapsed to a single cut at 0.8879, review share 0% |

Credit's two now agree closely, which is the other half of the story the priced band obscured:
the credit model is **well calibrated** — mean score 0.080443 against a 0.080728 positive rate,
ECE 0.00147 over ten bins, against 0.0031 in the published reference project on this dataset. So
its budgeted band and its unconstrained optimum land in the same place, and the budget is only
mildly binding (15% against 17.68%).

Fraud's disagree completely, and that is the calibration finding: the unconstrained optimum
reviews nobody and cuts at 0.8879, because `class_weight="balanced"` inflates the scores. At a
cut of 0.5 the fraud model flags 0.411% of rows when the true positive rate is 0.172% —
over-flagging by 2.4x. The fix is calibration, not band-fitting, and the band is served from the
budget rather than fitted to those scores precisely so a miscalibrated model cannot move the
decision boundary without anyone choosing to.

**The metric key rename, and the promotion gate it could have switched off.** `cv_auc_mean`
became `cv_roc_auc_mean` / `cv_pr_auc_mean`. The name is derived from the track, so two tracks
selected on different metrics have no shared field to be compared through. `riskwatch_credit`
v1-v5 in the primary checkout carry the old tag and one of them holds `@production`, so reading
only the new key would have made `incumbent_auc()` report "no incumbent" — which
`should_promote()` reads as grounds to promote unconditionally, with no exception and no error
log. `src/pipelines/retrain.py:incumbent_metric_tags()` reads the new name then the old one, and
offers the fallback only to tracks selected on ROC-AUC, since a `cv_auc_mean` tag on a PR-AUC
track would be a different quantity wearing the same name. **The registry is not migrated and
dual-write was rejected**: it would keep one value under two names indefinitely with nothing
stating when the second stops being written. The fallback has a removal condition, in
`docs/debt-ledger.md`.

`evaluate()` now takes the cut as a required argument and reports at the track's decline
boundary. `precision_at_0.5` and `recall_at_0.5` are gone — the cut was in the key name, and the
moment the cut moved the name would have been false.

**`POST /predict/fraud` landed here, and the decision vocabulary diverges from the plan.**

`FraudPredictRequest` carries 29 fields — `v1`..`v28` and `amount`, aliased to the raw `V1`..`V28`
and `Amount`. `Time` is deliberately not a request field: it is under schema contract at ingest
and not modeled, so a request sending it 422s on `extra="forbid"`, which is the right answer for a
caller who has misread the contract rather than merely added a field.

**The plan's target contract sketched `allow`/`review`/`block` for fraud. That was not taken.**
Both tracks use `approve`/`review`/`decline` on one `RiskResponse`. A per-track vocabulary needs
either a union `Literal` — which cannot express "credit only ever returns approve" and so loosens
the contract rather than tightening it — or a second response model identical but for one field.
Neither buys a caller anything that reading `track` does not already give them. Recorded here as a
divergence rather than left as a silent disagreement with the plan.

Measured live with `ENABLED_TRACKS="credit,fraud"`: `/health` reports `ok` with both models,
credit's example scores 0.423420 → `decline` at threshold 0.0963, fraud's scores 0.005326 →
`approve` at 0.1143, and `riskwatch_predictions_total` carries one series per `(track, decision)`.
**`ENABLED_TRACKS` still defaults to credit alone** — the deploy shape is a W2 image measurement
that has not happened — so `/predict/fraud` 503s naming the track on a default deployment, and a
test asserts that as contract rather than leaving it to be discovered.

**The 28 hand-written `serialization_alias` lines are guarded rather than generated.** Generating
them from `FRAUD_FEATURES.feature_columns` would make an off-by-one unrepresentable, but leaves no
readable contract and no static types; `tests/test_api.py::test_request_aliases_cover_the_feature_contract`
asserts the aliases equal the feature columns exactly, in order, for both tracks.
**Verified by mutation:** `serialization_alias="V17"` → `"V18"` kills five tests — the alias guard,
the skew test on MLflow's signature check, the fraud log-columns test, and both fraud endpoint
tests, which trip the stub's NaN assertion because `reindex` fills the orphaned column rather than
raising. Written-out aliases plus that assertion beats either alone.

**The rename touched eight places, not the seven that were enumerated.** The eighth is the
summary log line in `src/models/train.py`, which never mentions `cv_auc_mean` — it reads
`roc_auc` out of `evaluate()`'s return dict. A rename like this propagates along two axes, the
MLflow key and the metric name, and grepping the key finds only one of them. A test found the
other.


## Before you can run anything

Everything needed to train, serve from the registry, or compute drift is gitignored:
`data/raw/`, `data/processed/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `build/`. **A fresh
clone has none of it** and must run ingest and training first. The hermetic tests are the
only thing that works out of the box.

Machine-local as of this update: the primary checkout at
`~/Documents/projects/mlops_practice` holds both raw archives, both extracted — the credit
one to its application table, the fraud one to `data/raw/fraud/creditcard.csv` — and **both tracks have been
ingested** to `data/processed/<track>/latest.parquet`, and **both have been trained**. Read out
of `mlflow.db` rather than asserted: **27 runs, 2 registered models, 6 versions, 2 aliases** —
`riskwatch_credit` v5 on `@production` at `cv_auc_mean` 0.7524 and `riskwatch_fraud` v1 on
`@production` at `cv_pr_auc_mean` 0.8186. Both `models:/riskwatch_*@production` URIs resolve,
so `tests/test_skew.py` runs both parameters and `tests/test_reachable_decisions.py` runs all
six rather than skipping three.

The credit versions carry the **pre-Step-9 `cv_auc_mean`** key, not `cv_roc_auc_mean`: they
were registered before the rename and nothing has re-swept credit since. That is the exact
condition `incumbent_metric_tags()` in `src/pipelines/retrain.py` reads the legacy key for, so
the promotion gate has a readable incumbent — verified here rather than assumed, because one of
those five versions holds `@production` and a gate that reads nothing off it would promote
unconditionally.

Five versions for one model because the first three were re-registered while fixing the two
signature defects the skew test caught — v1 and v2 carried the 27-wide signature, v3 and v4
the narrowed dtypes. They are left in place rather than deleted: the registry is the record
of what happened, and a promotion history that only shows the version that worked hides the
fact that two did not.

**The two Telco-era worktrees are gone, and with them the registries behind the M4 evidence.**
Until 2026-09-28 this section named two populated working trees: `spookfish` (14 runs, the
`churnwatch` registered model, `data/raw/telco.csv`) and `horseshoe` (the M4 verification runs —
42 runs, `churnwatch` v1 on `@production`). Both directories, both worktree registrations and both
branches — `spookfish` and `archive/m4-telco-evidence` — are absent as of this update, removed
outside the session that noticed it.

**They were archived rather than discarded**, and the archive is outside the repo because all of
it was gitignored:

    ~/Documents/projects/_reference/m4-telco-evidence-20260928.tar.gz   (7.4 MB)
    ~/Documents/projects/_reference/m4-telco-evidence-20260928.HANDOFF.md

Verified by listing it: `mlflow.db`, 382 entries under `mlruns/`, `data/raw/telco.csv`, and the two
Evidently drift reports. So the M4 table above is still re-queryable — extract the tarball and
point `MLFLOW_TRACKING_URI` at its `mlflow.db` — but it is **no longer a working tree**, and that
is the difference this note exists to record.

Why archived instead of re-derivable: retraining brings the model back but not the run ids or the
timestamps, so the correspondence with the M4 verification table above — four rounds of
`airflow dags test` — breaks. The model is reproducible; the evidence is not.

Recorded rather than quietly dropped because this file is cited elsewhere as the evidence for
DoD ⑦, and "in a tarball outside the repo" is a materially different claim from "in a populated
worktree". Re-demonstrating M4 on the **retargeted** stack remains outstanding either way; the
archived evidence is Telco-era and does not carry over.

The DAG needs no extra *wiring*: the Airflow overlay bind-mounts the repo, so `task_ingest`
writes `data/processed/` back into the working tree and the sweep logs through the `mlflow`
service into the same `mlflow.db` the host reads.

**The block on it is gone, and it has still not been run.** The previous note here said
`task_ingest` called a Telco-only `src/data/ingest.py` that read `data/raw/telco.csv`, so a
scheduled run stopped at task 2 of 5 with `FileNotFoundError`. Step 4 retargeted that module
-- ingest is track-driven, both tracks resolve, and every other task already threads the
track through while `task_preflight` resolves the per-track baseline. What is outstanding is
**re-demonstrating the DAG on the retargeted stack**, which nothing has done: the M4 evidence
in this file is Telco-era. That is a run, not a fix.

## The baked artifact had drifted off the alias, and the suite could not see it

Found 2026-09-28 while checking state, not while looking for it. `build/model` held
`riskwatch_credit` **v4** while `models:/riskwatch_credit@production` resolved to **v5** --
exported 11:39, alias moved 12:08, and nothing between those two facts compared them.

The damage is bounded here and the mechanism is not. All five credit versions share
`cv_auc_mean` 0.7524, so the stale artifact scores identically; what a `docker build` would have
produced is an image serving a model the registry no longer promotes, with `/health` reporting
**"5"** if the env var were set and **"4"** from the file if it were not -- a real-looking number
in both cases. "Reports a real version rather than `unknown`" is the pre-deploy criterion in the
checklist above, and a stale export satisfies it perfectly.

**Why no test caught it.** `test_baked_model_path_reports_its_real_version` asserts the version
is *readable* and not `"unknown"`; it cannot see whether it is *current*, because a stale export
reports its own version with complete confidence. The three hermetic tests beside it stub the
loader and write their own `MODEL_VERSION`, so they never touch a registry. Nothing in the suite
held both numbers at once.

`tests/test_api.py::test_baked_artifact_is_not_stale_against_the_alias` now does, and it fails
with the mismatch spelled out (`build/model holds version 4 but ... resolves to 5`) before
`build/model` was re-exported to v5. It skips on the two local-state preconditions -- no export,
nothing registered -- and names which one, because both are ordinary states rather than defects.

**The host is the last place the comparison is possible.** The container has no registry to check
itself against, and `src/models/export.py` resolving the alias correctly is no help when the
export simply was not re-run. So this belongs before `docker build`, which is where Step 13's rehearsal will
run it.
