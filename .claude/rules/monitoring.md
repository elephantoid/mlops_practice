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

The 0.4 surface survives under `evidently.legacy` and **disagrees numerically** — 0.0952 vs
0.0526 on the same data. The two must never be mixed in one comparison.

## `import evidently` fails from the repo root

nltk 3.10.1 ships an import hook that blocks nltk-initiated imports resolving from the
working directory. Because `.venv` lives inside this repo, a legitimately installed `regex`
trips it. It is a false positive from the venv layout, not a shadowing attempt.

`PYTHONSAFEPATH` — what the error message suggests — does not fix it and would break
`import src.` as well. `NLTK_DISABLE_IMPORT_SECURITY` is set in the module rather than the
environment so the script behaves the same from a shell, a container, or an Airflow task.

## The drift thresholds are not derived from this data

Three distinct numbers, with three different provenances. Know which is which before
changing one:

| Threshold | Where it came from |
|---|---|
| `drift_share > 0.2` (retrain trigger) | Inherited from the original spec, written before any data existed. **Not implemented** — `dags/churnwatch_retrain.py` is empty. It exists only as a red band on the Grafana panel. |
| Per-column drift decision | **Evidently's default.** `DataDriftPreset()` is constructed with no arguments; `summarise()` reads `m["config"]["threshold"]` rather than setting one. Which stattest gets picked per column is not recorded anywhere. |
| `MIN_CURRENT_ROWS = 100` | The guard's existence is measured (40 identical rows → `drift_share` 1.0; 300 varied rows → 0.0). The value 100 is a round number inside that bracket and was never measured. |

With 19 features, one drifted feature is `0.0526`. A `0.2` trigger therefore means "4 or more
features" — the synthetic `MonthlyCharges` +15% scenario this project ships **cannot** reach
it. Re-derive the number before wiring the Milestone 4 retrain task to it.

## Both frames drop `customerID` and `Churn`

`customerID` is unique per row, so a high-cardinality identifier always registers as drifted
and inflates the share from 0.0526 to 0.0952. `Churn` is the label and is absent from the
prediction log, so keeping it would make the two sources incomparable.
