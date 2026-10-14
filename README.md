# evidence

Written by `.github/workflows/evidence.yml` on `main`, never by hand. One line per track per UTC
day in `reports/psi_daily.jsonl`: the PSI of the deployed service's logged predictions for that
day, read from the GCS sink, against the committed reference in
`src/monitoring/reference/` on `main`. The row format is `src/monitoring/daily.py`'s `row()`.

The traffic is generated (`src/monitoring/traffic.py`, `generator: reference-marginals`), not
real users. A row with `drill` set is a pre-registered drill and is meant to alert.
