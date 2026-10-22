---
paths:
  - "src/monitoring/**"
---

# Drift monitoring

## Evidently is 0.7.x, and almost nothing from 0.4 transfers

`pyproject.toml` names a floor, not the installed version. Verified against **0.7.21**:

- `Report.run` **returns** a Snapshot that owns the results. The 0.4 pattern of running a
  report and then reading the report is gone.
- `save_html` lives on the Snapshot, not the Report.
- `as_dict()` does not exist. The shape is `snapshot.dict()["metrics"][i]["value"]["share"]`.
- `ColumnMapping` was replaced by `DataDefinition`.
- `include_tests` belongs on the `Report`. Setting it only on the preset leaves `tests` empty.

The 0.4 surface survives under `evidently.legacy` with a different API. It was reported to
disagree numerically (c3ddea4, no figure given); on today's synthetic batch the two agree, so
treat the disagreement as unmeasured. Do not mix the two in one comparison either way.

## `import evidently` fails from the repo root

nltk 3.10.1 ships an import hook that blocks nltk-initiated imports resolving from the
working directory. Because `.venv` lives inside this repo, a legitimately installed `regex`
trips it. It is a false positive from the venv layout, not a shadowing attempt.

`PYTHONSAFEPATH` — what the error message suggests — does not fix it and would break
`import src.` as well. `NLTK_DISABLE_IMPORT_SECURITY` is set in the module rather than the
environment so the script behaves the same from a shell, a container, or an Airflow task.

## The drift thresholds are not derived from this data

Four numbers, with different provenances. Know which is which before
changing one:

| Threshold | Where it came from |
|---|---|
| `WATCHED_COLUMNS_BY_TRACK` (retrain trigger) | **The rule that fires.** Credit: `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `EXT_SOURCE_2`. Fraud: `Amount`, `V14`, `V17`. Chosen because they are what the decision turns on, not measured. Retargeted from the Telco set in the riskwatch merge — those names exist in neither track, so leaving them would have left the arm silently dead while still reading as implemented. |
| `RETRAIN_SHARE_THRESHOLD = 0.20` | Inherited from the original spec, written before any data existed. Kept only as the **catch-all second arm**, for broad shift across columns nobody watched. It cannot fire on this project's own single-column drift scenario. Still Telco-derived and unre-measured — see `docs/debt-ledger.md`. |
| Per-column drift decision | **Evidently's default.** `DataDriftPreset()` is constructed with no arguments; `summarise()` reads `m["config"]["threshold"]` rather than setting one. Which stattest gets picked per column is not recorded anywhere. |
| `MIN_CURRENT_ROWS = 1500` | **Measured on the real snapshots (2026-10-22).** Evidently's distance tests are not calibrated to the current sample's size, so a small batch drawn from the reference itself reads as drift: at the Telco-era 100 it retrained every seed of both tracks. The derivation table is beside the constant in `src/monitoring/drift.py`; `tests/test_drift_floor.py` re-checks it locally (CI has no snapshot). It removes **sampling noise only** — see below — and not a single extreme value: one 117,000,000 `AMT_INCOME_TOTAL` row (the reference's max) retrains credit on its own (strict xfail in that test). `SYNTHETIC_SAMPLE_SIZE` is tied to it. |

Both retrain thresholds live in `src/pipelines/retrain.py`, not in the DAG — Airflow is not
installed in the uv venv, so a rule written in `dags/` cannot be tested. `tests/test_retrain_rules.py`
covers them.

One drifted column is `1/26 = 0.0385` of credit's modeled columns and `1/29 = 0.0345` of
fraud's, so a `0.2` share needs **six** drifted columns on either track. The synthetic batch
(`SYNTHETIC_DRIFT_COLUMN` lifted 15%, `MIN_CURRENT_ROWS` rows) reaches `1/26` on credit —
`AMT_CREDIT` alone — and on fraud does not even detect the column it shifted. Neither comes
near the share, which is why `should_retrain()` fires on a *named column* first and only falls
back to the share: on credit, `AMT_CREDIT` is watched, so the one drift case this repo can
demonstrate fires; on fraud neither arm does. Do not collapse it back to a single share
comparison: that reinstates a trigger which never fires on that case.

## Both frames hold the model's columns, and nothing else

`load_reference` reduces the snapshot to `FeatureSpec.feature_columns`, and `records_frame`
drops `FeatureSpec.non_feature_columns` (`SK_ID_CURR`/`TARGET`, `TransactionIndex`/`Class`). The
label is absent from the prediction log, so keeping it would make the two sources incomparable;
the id is not a model input, so its drift says nothing about the model — and on live traffic,
where ids only grow, it would always drift. (The retired Telco `customerID` inflated the share
even on a sample of the training rows; credit's integer `SK_ID_CURR` does not on the synthetic
batch, but that batch is drawn from the reference, so it proves nothing about live traffic.)

## PSI (`src/monitoring/psi.py`) is a separate measure, with its own traps

- **Its own Pushgateway job** (`riskwatch-psi-<track>`). `push_to_gateway` replaces everything under
  a job, so sharing drift's job would make each module erase the other's metrics.
- **The 0.1 / 0.25 bands are the scorecard convention** (Siddiqi 2006), not derived here.
- **Batch size is not optional.** No-shift PSI ≈ `(B − 1)/n`: 100 rows put `ORGANIZATION_TYPE` at
  0.657. `required_rows()` derives the floor from the widest entry; do not lower it to drift's 100.
- **The reference is tied to a model version.** After a re-promotion run `--write-reference`;
  `tests/test_psi.py` fails until you do. Never point it at `latest.parquet`.
- **In Grafana alert rules, annotations are raw and labels are not.** Annotation templates take
  `$labels` as written, and a `$$` escape arrives in the email verbatim. Rule labels go through
  environment expansion, so a `{{ $labels.x }}` there becomes empty. Contact-point settings expand
  `${NAME}` — that is how the address comes from `.env`.

## Drift decomposition (`src/monitoring/decompose.py`) is judged on a committed seed

- **Do not tune the detector against `reports/decompose_eval.json`.** The cases come from
  `src/monitoring/injections.py`'s families and `EVAL_SEED`, both committed before the detector they judge. A
  change to the detector that is prompted by the evaluation needs a new seed, committed first —
  that is how the first detector's failure was handled (`FIRST_EVAL_SEED`). Develop on hand-built
  cases instead.
- **Every threshold comes from the reference-vs-reference null at `LEVEL`.** A new constant that is
  not a quantile of it, or derived from `CLAIM`, is a tuning knob; justify it or don't add it.
- **After any change to either module, rerun `--evaluate`** and commit the report;
  `tests/test_decompose.py` compares it with a fresh run whenever the snapshot is present.
- **A rule claims past its empirical gate AND at a count floor.** Where the reference never shows an
  event its quantile gate collapses to zero; the floor (`_binomial_floor`, `_duplicate_floor`) is
  derived from the reference and `ALPHA`. Never take a floor's rate from the calibration draws:
  `NULL_DRAWS` is a precision setting, and a threshold must not move with it.
- **The classifier gate is not an alarm.** It fires on joints PSI cannot see (`AMT_CREDIT` ×1.02:
  AUC 0.78, PSI 0.003). Alert on PSI.
- **Label reweighting is a control, not a population case.** A features-only detector cannot see
  prior shift; counting it as a miss would push the detector to read noise.

## The hosted evidence plane (`src/monitoring/gcs_log.py`, `src/monitoring/traffic.py`, `src/monitoring/daily.py`)

- **The sink is hourly and late.** Shards land up to 2–3 h after their hour, late entries as
  `_A0:<unix>` files. Read a day the next morning, and filter on each record's own `timestamp` — never on
  the folder it was filed in.
- **Generated traffic reproduces marginals, not joints.** Feature PSI on it is a no-shift baseline;
  the score PSI has a floor near 0.05. Never point `src/monitoring/decompose.py` at it. **Not a
  no-shift baseline for Evidently:** at 1500 rows it retrains fraud on every seed (share 0.55–0.72,
  watched columns firing) and puts credit at 0.15–0.19, one column short of the share — a bias of
  the bin-level marginal reproduction, which more rows do not remove. One real day of the fraud
  source does the same (contiguous 1500-row slices: share 0.55–0.83). `MIN_CURRENT_ROWS` cannot
  help with either; this is input to re-deriving the 0.20 share, and it is what Step 20 meets the
  day the DAG reads the deployed log.
- **A drill is added before its day, never after.** `traffic.DRILLS` is the pre-registration; a
  drill added to explain a day that already alerted is a fabricated incident.
- **The Action judges last.** Commit and traffic steps run under `always()`; only the final step
  fails the job. Reorder them and one alert silences every day after it.
- **The federation condition is ids, not names.** Changing it to `assertion.repository` reopens
  the repository-name takeover the id closes.
