# STATUS — RiskWatch

**Last updated: 2026-10-12** (W3 Step 18, the watch only — an uptime check and email alert **verified against a deliberately degraded revision**, a ~$5 budget, traffic pinned to a named revision; **Steps 11 and 12 were never built**, recorded under "Not built"; revision 4 — the app's INFO logs reach Cloud Run, which shows 2 CPUs to a 1-vCPU instance; test counts re-measured at `0931027`: 406; W3 Step 14 — **deployed to Cloud Run 2026-10-12 11:42 UTC; the uptime clock starts here**, both tracks in one image; W2 Step 10 — reason codes on credit from LightGBM's own TreeSHAP, `shap` and its 193 MB dropped from the serving dependencies, image re-measured at 362.3 MB compressed (67.5% of the 0.5 GiB free tier) and `--cpus=1` latency taken, latency buckets re-derived from a measurement; 2026-10-01 the legacy credit versions v1–v5 deleted, leaving v6 alone; 2026-09-30 the 27th feature removed as redundant, credit re-swept to v6; W2 Step 13 — the deploy rehearsal: image measured at 75.8% of the
free tier, the baked path served in a container, and a second enabled track found to silently
un-bake the first).

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

**406 collected at `0931027` (2026-10-12): 406 passed on the populated checkout; 385 passed /
21 skipped on a clean copy of that commit with no data, no registry and no export.** Re-check
with `uv run pytest --collect-only -q | tail -1`. That gives today's number, and this line records
the number at the named commit. A count in this file has drifted from the code three times, each
time by being carried forward instead of re-run, so every count here names its commit.

The 21 skips are all gitignored state, and each names what is missing:

| skips | file | needs |
|---|---|---|
| 8 | `tests/test_skew.py` | a registered model per track (2 tests × 2 tracks × example / optional-nulls) |
| 6 | `tests/test_reachable_decisions.py` | a snapshot and a model per track |
| 4 | `tests/test_model_resolution.py` | `build/model/<track>/` (`make image`) |
| 2 | `tests/test_fraud.py` | the fraud snapshot and cached archive |
| 1 | `tests/test_tracks.py` | the fraud archive |

Both figures are measured, not reasoned from an earlier one. The clean copy was a `git archive`
of the commit with nothing else added.

Both run inside the dev container, and `lightgbm`, `evidently`, `mlflow` and `sklearn` all
import on Linux. **Training and serving have now been exercised on the host** — ingest wrote
a validated snapshot, a sweep registered `riskwatch_credit` on `@production` (v5 then, v6 now
after the 26-feature re-sweep), and
`POST /predict/credit` served a real prediction off it. **Serving has now also been exercised
inside a container**, in Step 13's rehearsal, off the baked artifact rather than the registry —
same model, same request, byte-identical probability. The *registry* path in a container is
still unmeasured, so the claim that a same-absolute-path mount keeps MLflow's artifact
locations resolvable from both sides remains a design argument; the baked path no longer
depends on it.

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
- **Step 11 (PSI + Grafana alert rule) and Step 12 (drift-injection harness + drift decomposer,
  DoD ⑥ — the signature) were not built.** W2 went from Step 10 to the Step 13 rehearsal and
  W3 to the Step 14 deploy, both protected by the plan for the uptime clock; nothing recorded
  the skip until now. Order from here: Step 18's watch (done, below) → Step 16 CI → Step 11 →
  Step 12. The rest of Step 18 — log sink to GCS, the committed PSI reference, the `--source gcs`
  reader, Workload Identity Federation, the scheduled traffic generator — consumes Step 11's
  binned reference, so it waits for it. Until then DoD ② has no hosted producer.
- No GitHub Actions workflow, no GCS upload, no A/B middleware. Cloud Run is deployed — see
  the Step 14 section.

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

Before the deploy. **The deploy went ahead on 2026-10-12 with the two open items below still
open:** both are local observability checks, and waiting on them would have cost uptime days
that cannot be recovered.


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
- [x] the baked path exercised: export, `docker build`, `docker run`, and `/health`
      reporting a real version rather than "unknown" — **Step 13, 2026-09-28**, and it also
      served a real prediction. See the rehearsal section at the end of this file
- [x] runtime image **measured**: 2.1 GB uncompressed, **407 MB compressed = 75.8% of the
      0.5 GiB free tier**; **362.3 MB = 67.5%** after Step 10 dropped `shap`. The trim is
      *not* done and is now a hard prerequisite rather than housekeeping — the measurement says trimming alone cannot fit two versions, so a
      retention policy is mandatory. Numbers and ownership map in the rehearsal section

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
  where both paths reach it. **The column itself was removed on 2026-09-30** — it was
  redundant, and the section at the end of this file records what that measured out to. The
  fix above was correct for the column's placement and never asked whether it should exist.
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
  needs. It is an id, not a feature: it is absent from `feature_columns`, so it never reaches
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
v1-v5 carried the old tag and one of them held `@production`, so reading only the new key would
have made `incumbent_auc()` report "no incumbent" — which `should_promote()` reads as grounds to
promote unconditionally, with no exception and no error log. That was paid for with a two-key
fallback in `src/pipelines/retrain.py:incumbent_metric_tags()`, offered only to tracks selected on
ROC-AUC, since a `cv_auc_mean` tag on a PR-AUC track would be a different quantity wearing the
same name. **The registry was not migrated and dual-write was rejected**: it would keep one value
under two names indefinitely with nothing stating when the second stops being written.

**Repaid 2026-10-01** by deleting v1-v5 — the fallback's own stated condition. See the registry
section below; `docs/debt-ledger.md` section 3 records it.

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
of `mlflow.db` rather than asserted: **31 runs, 2 registered models, 2 versions, 2 aliases** —
`riskwatch_credit` **v6** on `@production` at `cv_roc_auc_mean` 0.7524 and `riskwatch_fraud` v1
on `@production` at `cv_pr_auc_mean` 0.8186. Both `models:/riskwatch_*@production` URIs resolve,
so `tests/test_skew.py` runs both parameters and `tests/test_reachable_decisions.py` runs all
six rather than skipping three.

**One version per model, as of 2026-10-01.** credit v1-v5 were deleted. They were
re-registrations from active development rather than production history — v1 and v2 carried the
27-wide signature the skew test caught, v3 and v4 the narrowed dtypes, v5 was superseded by the
26-feature re-sweep — and they had become unservable when the derived column was removed, since
their pickled internals expect a column the pipeline no longer produces. Nothing was ever going
to be rolled back to them.

This file previously argued for keeping them as "the record of what happened". That record is the
**31 runs**, which are untouched: every sweep, every metric, every artifact path is still
queryable. A registry version is a pointer with an alias slot, not the history.

Deleting them **repaid the legacy-metric-key fallback**, on exactly the condition
`src/pipelines/retrain.py` had named for it. `incumbent_metric_tags()` used to return two names
for credit so a `cv_auc_mean`-tagged incumbent stayed readable; with no such version left in the
registry there is nothing to alias that the fallback would have been needed for, so the failure it
guarded — an unreadable incumbent silently turning the promotion gate off — is now
unrepresentable rather than guarded against. `LEGACY_CV_METRIC_KEY`, `LEGACY_METRIC_NAME` and
`AUC_TAG` are gone with it.

> **Rolling back a credit model now means re-sweeping, not moving the alias.** That was already
> true before the deletion, for a different reason: `assert_model_can_score()` in
> `src/api/main.py` scores each artifact's own logged input example at load time, so a pre-26
> artifact is refused at boot rather than serving a `/health` of `ok` and 500ing every request.
> Reproduced against v5 and v6 directly before v5 was removed.

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

`tests/test_model_resolution.py::test_baked_artifact_is_not_stale_against_the_alias` now does, and it fails
with the mismatch spelled out (`build/model holds version 4 but ... resolves to 5`) before
`build/model` was re-exported to v5. It skips on the two local-state preconditions -- no export,
nothing registered -- and names which one, because both are ordinary states rather than defects.

**The host is the last place the comparison is possible.** The container has no registry to check
itself against, and `src/models/export.py` resolving the alias correctly is no help when the
export simply was not re-run. So this belongs before `docker build`, which is where Step 13's rehearsal will
run it.

## Step 13 — the deploy rehearsal, run 2026-09-28

Run in a dedicated worktree (`~/orca/workspaces/mlops_practice/Step13`, branch
`feat/step13-deploy-rehearsal`) rather than the primary checkout, so nothing about the
image depended on state the primary tree happens to hold.

**Acceptance met.** `/health` reports `{"status":"ok","models":{"credit":"5"}}` — a real
version, not the literal `"unknown"`. Two paths to that number were checked separately,
because only one of them survives a forgotten build argument:

- `--build-arg MODEL_VERSION=5` → `"5"` from the environment.
- `-e MODEL_VERSION=unknown`, which is what the `ARG` defaults to if the build argument is
  omitted → still `"5"`, read from the artifact's own `MODEL_VERSION` file. The fallback
  works, so forgetting the build argument is not a way to ship `"unknown"`.

**Serving inside a container, measured for the first time.** `POST /predict/credit` with the
example request returned `risk_probability` **0.4234197225110793** and `decision: "decline"`
at `threshold` 0.0963 — the host's 0.4234 to every digit it printed. Same artifact, same
frame construction, no registry reachable.

### Size: one version fits at 76%, and the second one is the question

| | |
|---|---|
| uncompressed | **2.1 GB** |
| compressed (what Artifact Registry stores and bills) | **407.2 MB = 0.379 GiB = 75.8% of the 0.5 GiB free tier** |

**Every percentage of the free tier in this file is on a GiB basis.** Google bills storage in
JEDEC GB = 2^30 bytes, and the Artifact Registry free tier is 0.5 GiB = 536.9 MB; sizes here are
bytes / 10^6. An earlier revision of this file mixed bases, dividing by 500 in some places and
512 in others, and was corrected on 2026-10-12.

Read from the image manifest's layer descriptors — the `size` field there *is* the stored size,
so this is the registry's own number rather than a proxy for it. Worth stating because the first
measurement here came from `docker save | gzip | wc -c`, which gave 404.4 MB: right to within
0.7%, by the roundabout route of re-compressing blobs that were already compressed.

| layer | compressed | contents |
|---|---|---|
| venv | **363.5 MB** | `/app/.venv`, 1.41 GB on disk |
| base rootfs | 26.8 MB | debian bookworm-slim arm64 |
| python | 13.0 MB | interpreter install |
| apt / ca-certs | 3.2 MB | `libgomp1` and friends |
| `src/` + model | 0.4 MB | the only layers a normal rebuild touches |

`src/` is 381 kB and the model 1.11 MB, both uncompressed — rounding error.

**So rebuild count is not the constraint; dependency churn is.** Layers dedupe, so a rebuild
that changes only `src/` or `build/model/` adds ~0.4 MB and the 363.5 MB venv blob is shared. A
change to `uv.lock` produces a *second* 363.5 MB blob: 771 MB stored, **44% over the free tier on
one dependency bump**. That is the shape of the bill, and it is not what "every rebuild adds
another version" in the section above implies.

### The trim buys less than the plan assumes, and the ownership map is why

Step 15 is "the serving image contains no kaggle, evidently or pandera". Measured against the
actual dependency graph (`uv tree --no-dev --invert`), that removes less than it sounds like:

| MB | package | owned by | leaves with evidently? |
|---|---|---|---|
| 187 | `plotly` | **evidently only** | **yes** |
| 159 | `llvmlite` | `shap` | no |
| 143 | `pyarrow` | **`mlflow`, directly** | **no** |
| 149 | `scipy` + `scipy.libs` | `scikit-learn` | no |
| 77 | `pandas` | serving | no |
| 62 | `mlflow` | pyfunc load | no |
| 57 | `sklearn` | serving | no |
| 56 | `statsmodels` | **evidently only** | **yes** |
| 38 | `numpy` | everything | no |
| 37 | `matplotlib` | **`mlflow`, directly** | **no** |
| 34 | `numba` | `shap` | no |
| 34 | `evidently` | offline | **yes** |
| 28 | `fontTools` | `matplotlib` ← `mlflow` | **no** |

Two corrections fall out of that table, and both were worth the ten minutes:

1. **`matplotlib` and `fontTools` are `mlflow`'s, not evidently's.** 65 MB that looks trimmable
   and is not, because the serving image needs `mlflow` to load a pyfunc model at all.
2. **`pyarrow` is a direct `mlflow` dependency**, not only pandas' parquet extra. The serving
   path never reads a parquet file and still cannot drop 143 MB.

So the trim is `plotly` + `statsmodels` + `evidently` = **277 MB of 1412, about 20%**. Scaling the
compressed venv layer by that leaves roughly **336 MB stored, 63% of the free tier**, for one
version — and **628 MB, 117%, for two. Trimming cannot make room for a second version.** The
scaling assumes these packages compress like the layer's average, which is the weakest number in
this section; it is directionally safe because the conclusion needs only "well above 50%". An Artifact Registry cleanup policy is therefore
mandatory rather than advisable, and that is a Step 18 deliverable that nothing currently owns.

### Step 10 and Step 15 collided — resolved by Step 10 dropping `shap`

`shap` owned `llvmlite` (159 MB) and `numba` (34 MB) — **193 MB, 14% of the venv**, already inside
the 407.2 MB measured above because `shap` was a `[project]` dependency from W1. The risk recorded
here was a re-inflation: Step 15's trim (kaggle, evidently, pandera) never touched `shap`, and
Step 10 building `TreeExplainer` at lifespan would have made those 193 MB permanent. Step 10 took
the other way out — reason codes come from LightGBM's `pred_contrib`, which is what
`TreeExplainer` calls for a LightGBM model anyway — so `shap`, `numba`, `llvmlite` and `slicer`
are gone from `uv.lock`. Re-measured after Step 10: **362.3 MB compressed, 67.5%** — see the Step 10 section.

### One-track-or-both: both. The blocker was never size

The fraud artifact exported to **348 KB** (`model.pkl` 312 KB) against credit's 1.0 MB. Against a
363.5 MB dependency layer a second model is **0.085% of the compressed image** — the deferred
question turns out not to have had a cost side. What actually blocks it is wiring: the Dockerfile
bakes one directory to `/app/model` and sets `MODEL_URI` to it, and `MODEL_URI` can only name one
model. The supported shape is one baked directory per track plus `MODEL_URI_CREDIT` /
`MODEL_URI_FRAUD`, which `tests/test_model_resolution.py::test_two_baked_artifacts_serve_two_tracks` now pins
so the Dockerfile change has a contract to satisfy rather than one to invent.

### A baked artifact now says which track it holds, because review found the hole I left

The ownership argument below — a bare directory carries no claim about whose model it is — was
applied to the two-track case and **not** to the one-track case, where `model_uri_for` returned a
shared `MODEL_URI` unconditionally. Copilot review found it, rated it high, and it reproduced
exactly as described.

`ENABLED_TRACKS=fraud` against the image carrying credit's artifact:

| | |
|---|---|
| `/health` | `{"status":"ok","models":{"fraud":"5"}}` — healthy, and advertising another track's version |
| `POST /predict/fraud` | **500 on every request**, MLflow's signature enforcement rejecting a 29-column fraud frame against a 26-column credit model |

So: reports healthy, serves nothing, and the version number it publishes belongs to a different
model. **And this service's own error message routed operators into it** — the two-track refusal
said "drop `'credit'` from ENABLED_TRACKS", which leaves `fraud` alone, single-track, and pointed
at credit's directory.

Fixed by making the artifact self-describing rather than by rewording the message.
`src/models/export.py` writes **`MODEL_TRACK`** beside `MODEL_VERSION`, derived from the
registered model name it resolved to fetch the thing — parsed from the name rather than passed as
a `--track` flag, so the label cannot disagree with what was downloaded.
`assert_model_matches_track()` checks it at startup, and **a missing marker fails**: "unlabelled
means trust the caller" is the behaviour that shipped, so treating absence as permission would
leave the hole open for every artifact exported before the marker existed, which is all of them.

**The first version of that check had the same hole one layer up, and review found it again.** It
returned early for any `models:/` URI on the reasoning that "the name in the URI *is* the claim" —
true, and worth nothing while nothing compared the claim to the track being served. So
`ENABLED_TRACKS=fraud` with `MODEL_URI=models:/riskwatch_credit@production` still passed startup
and reported the credit model as fraud, shipped inside the commit that claimed to close exactly
that defect. An explicit `MODEL_URI_FRAUD` aimed at credit's model is the same mistake by hand,
and it is the path with no ambiguity to blame. Both are now parsed and refused; measured in a
container, the registry case exits 3 naming both model names.

**Export is now all-or-nothing, which it was not.** Validation ran before the destination was
cleared, so a bad `--uri` left the previous `build/model` in place — and the only consumer is
`docker build`, where a `COPY` cannot tell a current artifact from last week's. It would have
baked the stale model into an image that looks healthy and reports that model's own version. The
destination is now cleared before anything can fail and the download is staged in a sibling
directory that is moved into place only once both markers are written, so neither a stale nor a
partial artifact can be built. Verified for real: `--uri models:/churnwatch@production` fails and
`build/model` is gone rather than stale.

**And `tests/test_export.py` is new, because the marker is now mandatory at startup.** Nothing
hermetic covered the writer — the real-artifact test skips on a clean checkout and the API tests
build markers by hand — so a regression that stopped exporting the marker would have left CI
green and failed every baked deployment. Eleven tests with the download stubbed, covering the
marker's content, the fraud-vs-default label, both failure paths leaving nothing deployable, the
staging directory's sibling location (a temp-mount path would make the move a cross-device copy
and fail after a full download), and the name agreement between export and the API that nothing
else enforces.

MLflow already writes `registered_model_meta`, which names the model, and this file argued
earlier against reading it. That argument stands — it would tie the serving contract to MLflow's
artifact layout. Writing our own marker from the step that already resolved the alias does not.

Measured in a container after the fix:

| configuration | before | after |
|---|---|---|
| `ENABLED_TRACKS=fraud`, credit artifact | `ok` + 500 on every request | **exit 3**, naming what the artifact holds and `ENABLED_TRACKS=credit` as the matching config |
| `ENABLED_TRACKS=credit,fraud` | 204.5 s, then exit 3 | 1.0 s, and the message now says "set ENABLED_TRACKS=credit" instead of the trap |
| default (credit) | `ok`, version 5 | unchanged — `ok`, version 5, `risk_probability` 0.4234197225110793 |

**Artifacts exported before this change have no marker and will be refused.** That is the
intended consequence; re-export is one command.

### The rehearsal's real find: enabling a second track silently un-bakes the first

`ENABLED_TRACKS=credit,fraud` against the one-model image took **204.5 seconds to fail**, and
both tracks failed — including credit, whose artifact the image carries.

That is the defect, and it is a condition rather than a hang. The old resolution read
`if len(ENABLED_TRACKS) == 1 and MODEL_URI: return MODEL_URI`, so the moment a second track was
enabled the baked path stopped being used **for every track**, credit included. Both fell through
to `models:/riskwatch_<track>@production` — a registry the image cannot reach — and the log says
so for both: `Track 'credit' failed to load from models...`, `Track 'fraud' failed to load...`.

**Two claims written here in the first draft of this section were wrong, and the measurement is
what corrected them.** They are left visible because the second one was the whole stated
motivation for the fix:

1. *"Retries with exponential backoff and no ceiling."* It is bounded.
   `mlflow.store.db.utils.MAX_RETRY_COUNT` is **10**, a module constant with no environment
   override, and the sleeps are `0.1 * (2**n - 1)`: 0.1, 0.3, 0.7, 1.5, 3.1, 6.3, 12.7, 25.5,
   51.1 — **101.3 s** per engine creation. Two tracks, one cycle each, 18 warnings in the log:
   **204.5 s** measured start to exit.
2. *"`load_model` never returned and never raised, so the per-track protection was
   unreachable."* It raised. The lifespan caught it per track exactly as designed, recorded both
   failures, and — because *every* track had failed — exited on its own rule with **code 3**.
   Nothing about the isolation was broken.

The first draft said "hangs forever" on the strength of a 45-second poll against a 204-second
failure. Polling for less time than the thing takes is not evidence of unboundedness.

**The fix is still worth having, for the reason the corrected reading gives rather than the
original one.** 204.5 s → **1.0 s**, and the message changes from a generic connection failure
against a path nobody configured to one naming the misconfiguration and both exits. On Cloud Run
the startup probe caps at 240 s, so a 204 s failure is inside the window only by accident; each
attempt bills 3.4 minutes of CPU and reports a timeout rather than a cause. More importantly, the
state being refused is one where **the operator asked for two tracks and silently got zero**,
with the artifact for one of them sitting in the image.

Fixed at the condition, not by capping MLflow's retries — which are already capped, and whose
ceiling is not the problem. **A baked deployment never falls back to the registry.**
`model_uri_for` raises `ModelConfigurationError` when `MODEL_URI` is a local path, more than one
track is enabled, and the track has no explicit override; resolution moved inside the lifespan's
`try` so it counts as a per-track failure rather than a process-level one.

**Both tracks are refused, including the one the artifact actually holds, and that is the answer
rather than a limitation.** A bare path carries no claim about whose model it is.
`registered_model_meta` happens to name one, but reading it would couple the serving contract to
MLflow's artifact layout and would still be guessing what the *operator* meant. One artifact plus
two enabled tracks cannot work; serving half of it, chosen by the API, is worse than refusing with
a message that names both exits.

Verified in the container, not only in tests: **exit code 3 after 1 second**, against 204.5 s
measured on the same image before the fix, with both tracks' failures and both ways out in the
message. The default single-track image is unaffected — `/health` still `ok` at version 5, and the
prediction above still 0.4234197225110793.

### Two smaller things the rehearsal surfaced — the first is now fixed

**`src/models/export.py` did not call `configure_tracking()`**, unlike every other entry point —
`src/models/train.py`'s docstring for that function says every entry point calls it precisely so a
separate process cannot fall back to a local default. MLflow's own default resolves to
`sqlite:///<cwd>/mlflow.db`, which coincides with the repo's registry **only when the cwd is the
repo root**. From a fresh worktree the export failed with `Registered Model with
name=riskwatch_credit not found` — loud, correctly, but naming the wrong cause — and left an empty
`mlflow.db` behind, so a second command would find a valid, empty registry.

Recorded here as debt and then fixed rather than left: Copilot review made the point that a
lookup against the wrong backend does not raise, it reports the model as *absent*, so the failure
reads as "nothing is registered" instead of "you asked the wrong database". The module now
anchors to `PROJECT_ROOT/mlflow.db` itself, duplicated for the same stated reason its track name
already is. Verified by running it from `/tmp`: resolves version 5 against the repo's registry and
creates no `mlflow.db` in the working directory.

**The staleness check skips in a fresh worktree**, naming the reason: the registry lives in the
primary checkout, so `models:/riskwatch_credit@production` does not resolve there. Correct
behaviour, and worth knowing before Step 16 puts that file in CI, where it will always skip.

## The 27th feature is gone, and it was never carrying anything (2026-09-30)

The estimator took 27 features while the request contract had 26. That asymmetry has been in
place since W1 and was treated as a placement problem — the derived `DAYS_EMPLOYED_ANOMALY`
column had to be computed inside the pipeline so training and serving agreed, which is correct
and is what the earlier fix did. **Nobody asked whether the column should exist.** It should
not.

### What it measured out to

`models:/riskwatch_credit@production`, 307,511 rows, the promoted config
(`num_leaves=31, n_estimators=300, lr=0.05`), same 3 folds:

| | ROC-AUC | flag's share of the model |
|---|---|---|
| median-impute **+ flag** (what shipped) | 0.75373 ± 0.00240 | **0.02%** of total gain |
| median-impute, no flag | 0.75373 ± 0.00240 | — |
| NaN passthrough, no flag | 0.75410 ± 0.00209 | — |
| NaN passthrough + flag | 0.75410 ± 0.00209 | **0.00%** of total gain |

Identical to five decimals with and without it, in both regimes. The logreg arm agrees:
ROC-AUC delta **-0.00001**, coefficient **+0.0096** against the model's largest at 1.0057.

The NaN-passthrough rows are *not* evidence that dropping the imputer is better: +0.00036
against a fold spread of 0.0021 is inside the noise. The only claim the numbers support is the
one about the flag.

### Why it was redundant, which is not the reason first guessed

The first explanation was that median imputation creates a point mass the tree can split on —
55,432 rows land exactly on the imputed median and only 58 of them are genuine, so the imputed
value is a 99.90% proxy for the flag. That is true, and it is not the reason: it would not
explain the logreg arm, where a linear model cannot isolate a point mass and the coefficient
was still zero.

The actual reason is upstream of every preprocessing choice:

| NAME_INCOME_TYPE | rows | sentinel share |
|---|---|---|
| Pensioner | 55,362 | **100.0%** |
| Unemployed | 22 | **100.0%** |
| Working / Commercial associate / State servant / Student / Businessman / Maternity leave | 252,127 | 0.0% |

`NAME_INCOME_TYPE` alone predicts the flag on **99.9967%** of rows — 10 disagreements in
307,511. The flag was a categorical the caller already sends, re-expressed as 0/1. The spec's
justification for it ("an applicant with no employment history is a different case from one with
a short history") is true and was already satisfied by a feature in the contract.

The population is genuinely different, incidentally: sentinel rows default at 0.0540 against
0.0866 for the rest. Pensioners default less. The signal is real and already carried.

### What changed

`DAYS_EMPLOYED` → NaN normalisation **stays**, inside the pipeline, for the reason it always
had: 365243 is a thousand-year employment history and both paths must see the same thing.

Removed: `FeatureSpec.derived_features` and its single entry, the derive-from-sentinel loop and
the fill-with-zero fallback in `_replace_sentinels`, the `numeric_features + derived_features`
special case in `build_preprocessor`, the custom `feature_names_out` callable that existed only
because the step widened its input, and the column `clean()` computed for a parquet that
`ingest` then selected it out of.

**That last one was dead output.** `clean()` computed the flag, `ingest` dropped it before
writing, and the pipeline re-derived it — so `_replace_sentinels`' comment claiming "ingest
already writes this column for training rows" described a data flow that did not exist, and the
`flag not in out.columns` guard deferred to a producer with no output.

**And the three-way coupling it left behind was a silent-failure mode.** The flag appeared only
when `spec.sentinels`, `spec.derived_features` and a hardcoded `f"{column}_ANOMALY"` convention
all agreed. Setting `derived_features=("DAYS_EMPLOYED_FLAG",)` produced a **constant-zero
column** — no error, no log — because the fallback loop filled what the convention could not
build. A one-character naming slip would have trained on a dead feature.

### Verification

`riskwatch_credit` **v6** on `@production`, `cv_roc_auc_mean` **0.7524** — the same number v5
scored with the flag, which is what the measurement predicted. The estimator now takes **26**
features, `get_feature_names_out()` returns **26** names, and every one maps to a column the
caller sends.

The re-sweep was not optional and the suite said so: before it, `tests/test_skew.py[credit]` and
all three `tests/test_reachable_decisions.py[credit]` parameters failed with
`Length mismatch: Expected axis has 26 elements, new values have 27`. Those tests exist to
compare a real registered model against the serving path, and this is the second time they have
caught a contract change that no hermetic test could see.

**Seven** tests in `tests/test_pipeline.py` guarded the derived column — not six, as the first
version of this paragraph and the commit message both said. They were replaced by seven that assert
the invariant directly: the model input frame equalling the contract *when the source frame
carries an extra column*, the sentinel step neither widening nor narrowing, the sentinel still
becoming NaN, a reordered frame being corrected rather than mislabelled, a wrong column set being
refused, a duplicated column being refused, and the ColumnTransformer selecting exactly the
declared features. A skew in *column
set* is now unrepresentable rather than merely tested for.

Review rejected three of the first five as written. `n_features_in_ == len(feature_columns)`
restated the ColumnTransformer assertion and is **false for the logreg arm**, where one-hot
encoding makes the estimator wider than the contract by design. An assertion on
`get_feature_names_out()` could not fail, since `one-to-one` returns `feature_names_in_`
verbatim. And the "scores the same from either path" test compared a frame with a copy of itself
— the old version asserted the derived flag was 1 on a frame that lacked it, so deleting the
flag deleted the real half and kept the tautology.

**Consequence for Step 10:** the open design decision is gone. There is no longer a transformed
feature without a request-side counterpart, so no reason code can name a column the caller never
sent and cannot act on.

### What review caught that the change itself had broken (2026-09-30)

Copilot's quota was exhausted, so the review of record was the `architect` agent plus direct
repro. Verdict CHANGES-REQUESTED, twelve findings, five blocking. Three were defects the change
introduced rather than documentation drift, and the first is the one that mattered:

**Rollback silently died and this file said it had not.** Covered in the registry section above.
Found by review; the live repro and the load-time smoke check that now catches it are both
recorded there.

**Removing the flag removed a guard, silently.** The step used to widen its input, which forced a
custom `feature_names_out` callable. With the flag gone, `feature_names_out="one-to-one"` is
correct — and it made a *reordered* frame silent. sklearn raises only when a function **renames**
its columns; when the same names merely arrive in a different order it takes the rename branch,
stamps the fitted order's labels onto arrival-order data, and **does not move the values**.
Measured on a three-column fixture: `AMT_CREDIT`'s value came back under the `DAYS_EMPLOYED`
label, the ColumnTransformer's own name check passed because the names matched fit exactly, and
the estimator returned a plausible probability from scrambled features. Before the commit that
same frame raised a length mismatch and printed both column lists.

Not reachable through the API — `_score` pins order with `reindex` and pyfunc reorders to the
signature — but reachable by any direct sklearn caller, and permutation is the failure class
`tests/test_skew.py` is written around. `_canonicalise_columns()` now reorders to
`spec.feature_columns` and **raises on a wrong column set**, so the guard is a property of the
step rather than of one caller. Reindexing would have handled both at once and filled a genuinely
missing column with NaN, which trades a loud failure for a quiet one.

**The silent-no-op class was not closed, only halved.** The claim above — that a one-character
naming slip can no longer train on a dead feature — was true of `derived_features` and false of
`sentinels`, three lines away in the same function. `sentinels={"DAYS_EMPLOYD": 365243.0}` was
accepted, the `if column in out.columns` guard skipped it, nothing logged, and 365243 scored as a
thousand-year employment history in training *and* serving identically. `FeatureSpec.__post_init__`
now rejects a sentinel key that is not a feature column, and the guard it made unnecessary is
gone.

The remaining findings were things this repo asserts about itself that had stopped being true: a
comment above the one line the 26-vs-26 invariant rests on still described the deleted
derivation, `src/features/specs.py` cited a debt-ledger entry that was never written, the ledger still named
the removed mechanism as the live blocker for a rejected `Hour` feature, and two files claimed
`derived_features` was empty for an attribute that no longer exists. Plus the count corrections
recorded above.

`clean()` is now documented as observation-only and value-identical, and
`DAYS_EMPLOYED_SENTINEL` reads from the spec instead of being a second literal — as two unlinked
numbers, changing the sentinel in `src/features/specs.py` alone would have kept normalisation correct,
computed zero sentinel rows, and let `if sentinel_rows:` suppress the log entirely, deleting the
observability the raw value is kept in place for.

### Pass two: the remedy for the first blocker was itself unguarded (2026-09-30)

A second `architect` pass on the fixes above returned CHANGES-REQUESTED again — one HIGH, four
MEDIUM, four LOW. The HIGH is the one worth recording.

**`assert_model_can_score` existed, worked, and nothing pinned the lifespan's call to it.**
`StubModel` in `tests/test_api.py` had no `input_example`, so every startup test routed the new
check through its skip-and-warn branch. Deleting the call site left the suite green. A guard
written to stop a healthy-looking service from 500ing every request could be removed without one
test noticing — the same "test that cannot fail" shape that this file already records twice.

Fixed by giving the stub a track-shaped example and adding two lifespan-level tests, both
directions. Verified by deleting the call site and watching one fail.

`ExplodingModel` needed an example too, and giving it one changed what it models: it now scores
the logged example and fails on anything else. That is not a contrivance — a lazily-initialised
resource or a row-dependent bug behaves exactly that way, and it is the failure mode a load-time
smoke check inherently cannot catch, which is worth a test saying out loud.

**Scoring the example only proves the artifact agrees with itself.** An artifact *narrower* than
the contract passes perfectly, because MLflow **drops** a request column its signature does not
know about — with a warning and no error. Adding a feature to `feature_columns` would leave an
older model silently scoring without it while `/health` said `ok`. The logged input schema is now
compared to `FEATURE_COLUMNS[track]` before the smoke score.

**The `try` around `model.input_example` was the hazard, not the protection.** It is a property
that cannot raise, so the only thing that `except` could catch is the attribute ceasing to exist —
the exact MLflow rename that would turn the whole check into a permanent silent skip. Read
unguarded now, and `test_input_example_is_a_real_pyfunc_attribute` pins the name against
`mlflow.pyfunc.PyFuncModel`, which is the one thing a stub test cannot check.

**Fixing half the rollback claim exposed the other half.** `src/pipelines/retrain.py`,
`docs/debt-ledger.md` and `tests/test_retrain_rules.py` all still said v1-v5 hold `@production`,
and the legacy-key fallback's own stated removal condition had now fired. It stays, condition
restated, because **unservable is not unaliasable**: an operator can still alias v5, and then the
API refuses the track loudly *while* a DAG run in the same window reads no incumbent and promotes
silently. The condition is now "when those versions cannot be aliased at all", i.e. deleted.

**A hole in the new guard, found by probing it rather than by reading it.**
`_canonicalise_columns` compared set membership, so a *duplicated* column name was neither
missing nor unexpected: `frame[expected]` then returned 27 columns for a 26-name selection and
sklearn caught it one step later as a feature-name length mismatch. Refused by name now.

**And the clean-clone measurement caught a defect in one of the new tests.** The
healthy-lifespan test did not stub `load_model`, so it passed here and failed in a fresh clone
with `Registered Model with name=riskwatch_credit not found` — breaking the hermeticity
`tests/test_api.py`'s own docstring promises, and it would have failed in CI. Measuring a clean
clone is the practice that caught it; running the suite where the registry happens to exist never
would have.

Five guards were verified red before green: the deleted call site, the neutered canonicalisation,
the neutered schema comparison, the removed `clean()` copy, and the duplicated column.

### Pass three: a multi-axis review, and the same wiring defect a third time (2026-09-30)

Copilot's quota is still exhausted. This pass used the `code-review-and-quality` skill instead of
the `architect` agent — a different method rather than a different model, and the difference is
what it found. The two `architect` passes were adversarial-correctness: they read for what breaks.
This one reviews five axes with sizing and dead-code hygiene, and three of its four findings are
in axes the earlier passes never looked at.

**A false refusal with a message naming nothing.** `assert_model_can_score` compared the logged
input schema to the contract as *lists*, so an artifact whose schema had the same names in a
different order was refused with `it does not know [] and expects []` — under a parenthetical
claiming ordering was already excluded. MLflow reorders a request to the signature, so order alone
is harmless and that refusal would have rejected a sound artifact. Compared as sets now.

**The same defect this PR fixed in `_replace_sentinels`, reproduced two commits later.** That
function did three jobs under a name covering one, and fixing it is half this branch's rationale.
Then `assert_model_can_score` grew a second, unrelated check — a *narrower* artifact scores its own
example perfectly, which is precisely why the schema comparison had to exist, and precisely why it
is not a scoring question. Split into `assert_model_matches_contract` and
`assert_model_can_score`, each with one job and one argument in its docstring.

**And splitting it reproduced the P1 from pass two.** The new function had unit coverage and
nothing observed the lifespan calling it, so deleting the call site left the suite green — exactly
what pass two caught on the original. Third appearance of this class in one branch. Both call
sites are now pinned by lifespan-level tests, verified by deleting each in turn.

The pattern is worth naming, since three occurrences is not bad luck: **a guard added in response
to a review gets unit-tested against its own signature, and the wiring is assumed.** The test that
matters is the one that fails when the call disappears, and it has to be written in the same commit
as the guard.

**`tests/test_api.py` is 1177 lines**, past the skill's ~1000 inspection signal, and this PR put
183 of them there. It holds two distinct clusters — the request/response contract its docstring
claims, and model resolution plus startup validation — sharing only two helpers. The extraction
boundary is recorded as `docs/debt-ledger.md` 2-G with a trigger, deliberately not done here: the
same skill requires refactoring and feature work to be separate changes, and this one is already
922 insertions across 15 files.

`AUC_TAG` in `src/pipelines/retrain.py` is dead in `src/` — its own comment says nothing in the
promotion path should use it — and survives only because three tests reference it as a convenient
symbol. Pre-existing, recorded here rather than fixed in a branch about something else.

**And the two axes pass three had skipped turned out to hold the worst finding.** The first run of
the skill covered correctness, readability and architecture and stopped there. Security and
performance were on the checklist and never executed.

**Security: an unauthenticated 503 disclosed the deployment's internals.** `_model_for`
interpolated `load_failures[track]` verbatim into the HTTP body. That was survivable while those
strings were mostly `MlflowException: Registered Model ... not found` — and stopped being
survivable the moment `ModelConfigurationError` messages were written to be *actionable for an
operator*. Measured against a running app on a degraded deployment, a valid unauthenticated
`POST /predict/fraud` returned:

> `track 'fraud' is unavailable: ModelConfigurationError: track 'fraud' is enabled and
> MODEL_URI_FRAUD is unset, but this deployment serves from local artifacts
> (MODEL_URI_CREDIT=/app/model/credit holds 'credit') and has none for 'fraud'. Falling back to
> the registry is not an option here — a baked image has no registry to reach. …`

The baked artifact's filesystem path, both `MODEL_URI_<TRACK>` variable names, which track the
artifact holds, and the deployment shape — to an anonymous caller, on a service whose entire W3
goal is a **public** Cloud Run URL. The mechanism predates this branch; what this branch did was
make the payload worth reading.

The body now names the track and points at the logs. The reason is not lost but *addressed*: the
lifespan already logs it per track at `ERROR`, where an operator looks and an anonymous caller
cannot, and `app.state.load_failures` still carries it for tests. Whether a track is misconfigured
or simply not enabled is deliberately not distinguished either — that distinction tells a prober
which tracks the deployment was *meant* to serve. Verified by restoring the old interpolation and
watching the new test name the leaked path.

**Performance: no findings, measured rather than assumed.** `_canonicalise_columns` runs on every
transform, so it is in the request path: **0.7 µs** for credit's 26 in-order columns against a
2880 µs endpoint, 0.02% of a request. The reorder branch costs 55 µs and only fires on a misuse it
is correcting. The duplicate scan's `list.count()` is O(n²) — 2.8 µs against 0.5 µs for
`pandas.duplicated()` at this width, which is a Nit and only on the reorder branch.

**Dead code, listed rather than deleted.** `AUC_TAG` has no consumer in `src/` outside its own
definition — its comment says nothing in the promotion path should use it — and survives only as a
convenient symbol in three tests. `LEGACY_CV_METRIC_KEY` and `LEGACY_METRIC_NAME` are live, and
deliberately so: the fallback stays until v1-v5 cannot be aliased. No code references
`derived_features` any more; the three remaining mentions are prose recording why it is gone.

**All three were deleted the next day** — see the section below. The condition named above was met
deliberately rather than waited for.

### The legacy versions are gone, and the fallback with them (2026-10-01)

The review above listed `AUC_TAG` as dead and recorded the legacy-key fallback as live-until-v1-v5
cannot be aliased. Both entries rested on keeping five registry versions that nobody was ever
going to use.

They were re-registrations from active development, not production history: **v1 and v2** carried
the 27-wide signature the skew test caught, **v3 and v4** the narrowed dtypes, and **v5** was
superseded by the 26-feature re-sweep. All five had already become *unservable* when the derived
column was removed — their pickled internals expect a column the pipeline no longer produces — so
the "rollback target" they were being preserved as did not exist.

Deleted, leaving `riskwatch_credit` with one version. **The 31 runs are untouched**: every sweep,
every metric and every artifact path is still queryable. This file previously argued the registry
was "the record of what happened"; the record is the runs, and a registry version is a pointer
with an alias slot.

That met the fallback's own stated removal condition, so it went too —
`incumbent_metric_tags()` returns one name per track, and `LEGACY_CV_METRIC_KEY`,
`LEGACY_METRIC_NAME` and `AUC_TAG` are gone. **What makes that safe is not care but arithmetic:**
with no `cv_auc_mean`-tagged version left in the registry, the situation the fallback existed for
cannot be constructed. The gate-off failure is unrepresentable rather than defended against.

The tests followed the same logic. `test_reads_a_version_tagged_before_the_metric_rename` asserted
the fallback worked and is replaced by `test_a_legacy_tagged_version_reads_as_no_incumbent`,
which states the consequence plainly: such a version reads as no incumbent on either track. That
is safe *because the registry holds none*, not because reading one would be harmless — and the
test says so, for whoever re-registers a pre-rename version later.

A backup of `mlflow.db` was taken before the deletion, and the delete refused to run unless the
alias was on v6.

## Step 10 — reason codes on credit (2026-10-12)

Credit responses carry **exactly three** reason codes on every decision, approvals included;
fraud's are always `[]`. `src/models/explain.py` computes them, `src/api/main.py` builds one
explainer per track at startup, and `CREDIT_FEATURES.reason_codes` / `FRAUD_FEATURES.reason_codes`
say which track has them. Against the registered credit v6, the example applicant is explained by
its three external credit scores (+0.98, +0.62, +0.48 log-odds).

### Where this diverges from the Step 10 spec

- **LightGBM's `pred_contrib`, not `shap.TreeExplainer`.** Not an algorithm change: for a LightGBM
  model `TreeExplainer` calls `booster.predict(X, pred_contrib=True)` itself
  (the `lightgbm` branch of shap 0.52's tree explainer), and the two were measured identical,
  bias column included. Calling it directly drops 193 MB of `numba` and `llvmlite` from the
  serving image. The spec named the library; the acceptance criteria name the values, and those
  are met.
- **`ReasonCode` has three fields, not the spec's five.** The spec sketched
  `{feature, display_name, value, shap_value, direction}`; the W1 schema shipped
  `{feature, contribution, value}`, with `feature` already the display name and the sign of
  `contribution` carrying direction. Step 10 is the first step to fill the field, so it is the
  one that inherits the difference. Nothing is lost: a raw column name in a response is the
  thing the display map exists to keep out of it.
- **Contributions are in log-odds.** It is the only space where they add up: base value plus
  every contribution is the logit of `risk_probability`. The schema says so.
- **Three on approvals too.** One response shape for every decision; a consumer that wants only
  risk-raising reasons filters on the sign. Decided by the user over the adverse-action reading,
  which would require reasons on declines only.

### What fails, and how loudly

- A credit model that cannot be explained — not a LightGBM booster, output columns that differ
  from the contract's order, contributions that do not reconstruct its own logged example — is
  **refused at startup**, per track, like every other unservable artifact. Fraud keeps serving.
- A request whose contributions do not reconstruct the served probability to 1e-9 is a
  **counted 500**, raised before the prediction is counted or logged. Credit's `[]` would read
  like fraud's permanent one, so an empty list is never the failure mode.
- The explainer transforms with everything before the classifier — the sentinel step **and** the
  preprocessor. The preprocessor alone skips the sentinel step and mis-explains the ~18% of
  applicants carrying `DAYS_EMPLOYED = 365243` by 0.18-0.27 in probability;
  `tests/test_explain.py` fails on exactly that mutation, and a sentinel request against the
  real v6 was served with its additivity check passing.

### Measured, host, real credit v6 (300 trees, 31 leaves), 2,000 iterations

| | p50 | p95 | p99 |
|---|---|---|---|
| `/predict/credit`, no reason codes | 2.45 ms | 2.65 | 2.87 |
| `/predict/credit`, with reason codes | **4.47 ms** | 4.90 | 5.20 |
| explanation alone | 1.87 | 2.11 | 2.69 |
| — of which the second transform | 1.15 | 1.34 | 1.67 |
| — of which `pred_contrib` | 0.71 | 0.83 | 0.99 |

The explanation's cost is mostly the **second transform**, not the contributions. Reusing the
first one would mean scoring through the raw pipeline instead of pyfunc and losing signature
enforcement, so it was not done; +2 ms is the price. Startup cost per explained track: 4.0 ms.
`LATENCY_BUCKETS` were re-derived from these numbers — geometric at 25% steps from 0.5 to 35 ms —
and a test pins the rule. A separate `riskwatch_explain_latency_seconds{track}` histogram
separates the explanation's cost from the score's.

### Measured in a container, after merge (2026-10-12, `687fdea`)

Image built from `main` with credit v6 baked, read from the image manifest's layer descriptors
exactly as in Step 13.

| | Step 13 | Step 10 | change |
|---|---|---|---|
| uncompressed | 2.1 GB | 1.84 GB | −260 MB, against `shap`'s 193 MB; the rest is not explained |
| venv layer, compressed | 363.5 MB | **314.3 MB** | **−49.2 MB** |
| base rootfs + python layers | 39.8 MB | 44.0 MB | +4.2 MB — the base image moved, not this repo |
| **total compressed** | 407.2 MB = 75.8% | **362.3 MB = 67.5% of 0.5 GiB** | −44.9 MB |

The prediction was "about 50 MB" and the venv layer came in at 49.2. `shap`, `numba`,
`llvmlite` and `slicer` are absent from the image's `site-packages`. **One `uv.lock` change still
means a second 314 MB blob: about 676 MB stored, 126% of the free tier** — a projection that
assumes the base and `src/` layers are shared, not a measurement. The retention policy
stays mandatory.

`--cpus=1`, in-process `TestClient` inside the container, 2,000 requests after 200 warm-up, two
runs:

| | run 1 p50 / p99 | run 2 p50 / p99 |
|---|---|---|
| no reason codes | 3.25 / 6.45 ms | 2.95 / 4.90 ms |
| with reason codes | 5.30 / 8.51 ms | 5.59 / **18.97** ms |

p50 costs about +2–2.6 ms, the same as on the host. p99 is noisy at one CPU: the 18.97 ms is
one run, and CFS quota throttling is the likely cause. It is still inside the buckets, which
run to 35 ms at 25% steps. The order was not alternated: "with" ran first in both runs.

The reason codes in the container matched the host to the printed digits: External credit
score 3 / 1 / 2 at +0.98 / +0.62 / +0.48. The container runs Linux arm64 and the host runs
macOS, so the same model gives the same reasons on both.

## Step 14 — deployed to Cloud Run (2026-10-12)

**Live at `https://riskwatch-46336906432.us-central1.run.app` since 2026-10-12 11:42:53 UTC.
The 75-day uptime clock starts here.** That is 64 days before a mid-December review, so the
plan's 75 days by then is no longer reachable; it was lost to the date, not to this step.

| | |
|---|---|
| project | `project-e9695c4a-c0d1-4e50-84f`, free-trial billing ($300, 90 days — ends about 2027-01-10) |
| region | us-central1, where the Cloud Run free tier applies |
| image | `us-central1-docker.pkg.dev/<project>/riskwatch/api:c6-f1-0931027` — credit v6 + fraud v1 at commit `0931027`, linux/amd64, revision 4. The models have not changed since revision 1; revisions 2–4 changed code only (root redirect, review follow-ups, log config) |
| registry | Artifact Registry repo `riskwatch`, cleanup policy from day one: keep the 2 newest versions, delete the rest |
| service | 1 GiB, 1 vCPU, min 0, **max 2 instances**, unauthenticated |

Verified from the public URL, off the LAN:

- `/health` → 200, `{"status":"ok","models":{"credit":"6","fraud":"1"}}` — two real versions,
  each read from its own artifact.
- `POST /predict/credit` → 0.4234197225110793, `decline`, External credit score 3 / 1 / 2 at
  +0.98 / +0.62 / +0.48 — the same as the host and the arm64 container, to every printed digit.
- `POST /predict/fraud` → 0.005326298672912457, `approve`, `[]`.
- A scanner path (`/.env`) → 404, counted under `endpoint="unmatched"`.
- The root URL → 307 to `/docs`. Revision 1 answered it with `{"detail":"Not Found"}`, and the
  bare URL is the one that gets shared and clicked.
- `docker run -e PORT=9000` (arm64, locally) served on 9000 and its own healthcheck reported
  `healthy`; `docker stop` took 0.66 s, so SIGTERM reaches uvicorn rather than a shell.

**Cold start is about 20 s**, from Cloud Run's log: instance start → uvicorn 11 s (imports),
→ both models loaded 8.5 s. A request to a scaled-to-zero service waits that long.

### What the step changed

- **One process-wide `MODEL_VERSION` stamped every baked track with the same version.** With
  one track it agreed with the artifact; with two, an image built with
  `--build-arg MODEL_VERSION=6` would have reported fraud as 6. The variable is gone: a baked
  artifact's version comes from its own file, and a test sets the variable to prove it is
  ignored. It failed on the old code with fraud reported as `'6'`.
- Export writes `build/model/<track>/`; the image bakes both and sets `ENABLED_TRACKS=credit,fraud`
  with `MODEL_URI_CREDIT` / `MODEL_URI_FRAUD`. Compose pins `ENABLED_TRACKS=credit`, so it does
  not serve credit from the registry and fraud from the baked copy at once.
- `CMD` and `HEALTHCHECK` read `${PORT:-8000}`. `EXPOSE` stays 8000: it is resolved at build
  time and Cloud Run ignores it, so the plan's third site was documentation only.
- `make image [PLATFORM=linux/amd64]` exports both tracks and builds.

### Three things worth knowing

1. **The amd64 image cannot be smoke-tested on this Mac.** Rancher Desktop emulates amd64 with
   QEMU (Rosetta off): `import numpy` works, and `import pandas`, `sklearn`, `lightgbm` and
   `mlflow` each die with a segmentation fault. The same image started first time on Cloud
   Run, so the failure is the emulator's. Which instruction it mishandles was not pinned down.
2. **Startup logs a dependency mismatch warning** — `llvmlite (current: uninstalled,
   required: llvmlite==0.45.1)`. Both models were logged while `shap` was still a dependency,
   so the requirements file MLflow wrote beside each model lists it. Serving is unaffected; the next retrain clears it.
3. **Not in this step:** uptime check and alert, log sink to GCS, budget alert, CI (Steps 16
   and 18). The free-trial account is the spending cap until then: it cannot be charged unless
   upgraded by hand.

### Revision 3: the review follow-ups, deployed from `main` (2026-10-12)

Today's code reviews of #19–#24 found one live bug and several gaps, fixed in #25 and deployed as
`riskwatch-00003-rrc`, image `c6-f1-c85a42f`, built from `main` after the merge so no unmerged
code reached the public URL. Checked on the public URL:

- `Infinity` in `amt_credit`, `NaN` in `ext_source_1` and `-Infinity` in fraud `v1` each return 422
  "Input should be a finite number". Before #25 the first scored with a 200, the second
  returned a 500, and the log line carried a bare `Infinity` token.
- A credit request with all four optional fields null returns 200 with 3 reasons. That is the
  common request: 73% of applicants have an optional field missing.
- The example requests score as before: credit 0.4234197225110793 with the same three reasons,
  fraud 0.0053 `approve`.

**Found while checking: the app's own INFO logs never reach Cloud Run.** Nothing configures
Python logging, so `logger.info` lines are dropped and only WARNING and above get out, through
`logging.lastResort`. That covers "Loaded 'credit' model" and #25's startup line naming the
visible CPU count. It has been true since the first deploy. Fixed in revision 4, below.

### Revision 4: the app's logs reach Cloud Run (2026-10-12)

#27 starts uvicorn with `--log-config src/api/logging.json`: `src.*` at INFO, libraries at WARNING.
Deployed as `riskwatch-00004-rnz`, image `c6-f1-0931027`, built from `main` after the merge.
The first lines Cloud Run has ever received from the app itself (17:28 UTC):

```
INFO:     Loaded 'credit' model /app/model/credit (version 6)
INFO:     Loaded 'fraud' model /app/model/fraud (version 1)
INFO:     Visible CPUs: os 2, joblib 2
```

**Cloud Run shows 2 CPUs to an instance configured with 1 vCPU.** So `n_jobs=-1` would have meant
two LightGBM threads sharing one vCPU of quota. #25 pinned the served estimator to one thread
based on host measurements only. This confirms its premise, two threads on one vCPU, on
production; the latency gain is still a host number and was not measured on Cloud Run.

The lines arrive with severity DEFAULT, not INFO: Cloud Logging reads plain text without a level.
Filtering or alerting by severity needs JSON logs with a `severity` field, which is Step 18 work.

Checked on the public URL: `/health` reports credit 6 and fraud 1; the credit example scores
0.4234197225110793 with the same three reasons; fraud scores 0.0053 `approve`; `NaN` returns 422.

### Open from the reviews, not done

- reason codes are not in the prediction log, so reproducing a decline's reasons needs the exact
  model artifact;
- the additivity check compares in probability space; in log-odds space it would not loosen for
  a track with a very low base rate;
- test names: `raises` still appears where the rule says `rejects`/`refuses`, and so do
  `passes`/`works_when` where it says `accepts`, and two names carry a status code;
- about ten inline `TestClient` setups in `tests/test_model_loading.py` and
  `tests/test_model_resolution.py` that a shared fixture would remove.

## Step 18, the watch only (2026-10-12)

The service had run six and a half hours with nothing watching it. Made by
`ops/monitoring/setup.sh`; the alert policy is `ops/monitoring/alert-policy.json`.

| | |
|---|---|
| uptime check | `riskwatch-health-QObE7vdXJ8I` — `GET /health`, every 5 min from usa-oregon / usa-iowa / usa-virginia, 60 s timeout, passes only on `$.status == "ok"` |
| alert policy | `9414181861483259973` — `check_passed` false in **2+ regions** over a 600 s window for 60 s; runbook in the policy's documentation, so it arrives in the mail |
| channel | email, `riskwatch-ops` |
| budget | `a9a4b2b2-…` — **7,000 KRW** (~$5) a month on this project; mail at 50 / 90 / 100 % actual and 100 % forecast |
| traffic | **pinned to `riskwatch-00004-rnz`**; a deploy no longer moves it — see `CLAUDE.md`, Deploy |

**Why `status == "ok"` and not a 2xx.** `/health` answers 200 with `"degraded"` when one track
failed to load. A status-code check would call a half-down service up.

**The check was proved against a deliberately degraded revision, on the public URL.**
`riskwatch-00006-tef` is the same image with `MODEL_URI_FRAUD=/app/model/nonexistent`: fraud fails
to load, credit serves, `/health` says `degraded` with a 200. A revision that fails to *start* could
not test this — Cloud Run never routes traffic to it. Times UTC:

| | |
|---|---|
| 18:17:42 | traffic → `00006-tef` (T0). Credit kept serving; fraud answers 503 by code (`src/api/main.py:667`), not observed during the test |
| 18:18:40 | first failing check (iowa) |
| 18:21:30 | virginia and oregon failing |
| **18:24:30** | **alert mail received — T0 + 6 min 48 s** |
| 18:25:33 | traffic → `00004-rnz`, no rebuild |
| 18:27:46 | resolved mail (two regions passing closes it; iowa caught up at 18:28:40) |

That is **about 8 minutes of planned `degraded` against the uptime clock**, by this check's own
definition. Runbook step 2 was run against the incident and returned the cause in one line:
`Track 'fraud' failed to load from /app/model/nonexistent …` on `00006-tef`.

**What the test cost to clean up.** `--update-env-vars` is written to the service, not the
revision, so every later deploy would have inherited the broken variable. A `services update
--remove-env-vars MODEL_URI_FRAUD --no-traffic` made `riskwatch-00006-npz`, which carries no
variable and no traffic. Revisions are numbered by service generation, which is why there are two
`00006`s and no `00005` (the 18:10:43 traffic pin took that number). `00006-tef` is kept: it
serves nothing, costs nothing, and is the evidence.

**Found on the way: a cold start took 36.0 s, not the 20 s recorded in Step 14.** The first check
ever run (18:11:25, oregon) hit a scaled-to-zero service and timed out at the 30 s first set. One
region failing does not alert; two regions inside the same cold start would have. The timeout is
now the 60 s maximum.

**The budget amount is in KRW because the billing account is.** `--budget-amount=5USD` is refused
with a bare `INVALID_ARGUMENT` that does not say why.

**Not verified: that the budget measures what it is meant to.** It counts `FREE_TIER` credits as
credits and everything else, the $300 trial included, as spend — so it should read "what this would
cost once the trial ends". Counting the trial credit would hold the net at 0 forever; counting no
credits would flag usage the free tier covers. Whether Cloud Run's free tier is booked as a
`FREE_TIER` credit is not documented for Cloud Run specifically. **Check in Billing → Reports,
grouped by credit type, after the first full month (2026-11-01).** No budget mail has fired, so the
path from budget to mail is also unexercised.

**Owed monthly (W4–W9):** confirm the check is green and record the date here. Next: 2026-11-12.
