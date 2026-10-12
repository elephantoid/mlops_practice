#!/usr/bin/env bash
# Creates the watch on the public service: email channel, uptime check, alert policy, $5 budget.
#
# Create-only and NOT idempotent: run it once against an empty project. A second run makes
# duplicates. What exists today is recorded in STATUS.md ("Step 18 — the watch"); this file is how
# it was made, so it can be made again.
#
#   ops/monitoring/setup.sh <alert email>
set -euo pipefail

EMAIL=${1:?usage: setup.sh <alert email>}
PROJECT=project-e9695c4a-c0d1-4e50-84f
PROJECT_NUMBER=46336906432
BILLING_ACCOUNT=013E75-1269F8-D3BB82
HOST=riskwatch-${PROJECT_NUMBER}.us-central1.run.app
HERE=$(cd "$(dirname "$0")" && pwd)

gcloud services enable billingbudgets.googleapis.com --project "$PROJECT"

# No GA gcloud command creates a notification channel, so the REST API does; an email channel
# needs no verification.
CHANNEL=$(curl -sSf -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://monitoring.googleapis.com/v3/projects/$PROJECT/notificationChannels" \
  -d "{\"type\":\"email\",\"displayName\":\"riskwatch-ops\",\"labels\":{\"email_address\":\"$EMAIL\"},\"enabled\":true}" |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])')

# Matches "status": "ok", not just a 2xx: /health answers 200 with "degraded" when one track failed
# to load. The content is quoted because a JSON-path exact match compares JSON values. The timeout
# is the 60 s maximum: the first check ever run hit a cold start, took 36.0 s, and failed at 30 s.
CHECK=$(gcloud monitoring uptime create riskwatch-health --project "$PROJECT" \
  --resource-type=uptime-url \
  --resource-labels="host=$HOST,project_id=$PROJECT" \
  --protocol=https --path=/health --status-classes=2xx \
  --matcher-type=matches-json-path --json-path='$.status' \
  --json-path-matcher-type=exact-match --matcher-content='"ok"' \
  --period=5 --timeout=60 --regions=usa-oregon,usa-iowa,usa-virginia \
  --format='value(name)')

POLICY=$(mktemp)
sed -e "s|__CHECK_ID__|${CHECK##*/}|" -e "s|__CHANNEL__|$CHANNEL|" \
  "$HERE/alert-policy.json" >"$POLICY"
gcloud monitoring policies create --project "$PROJECT" --policy-from-file="$POLICY"

# Counts the free tier as free and the trial credit as spend: with the trial credit included the
# net cost is $0 forever and nothing would ever fire. Alerts; does not cap.
# The amount must be in the billing account's currency, which is KRW; 5USD is refused with a bare
# INVALID_ARGUMENT that does not say why. 7000 KRW is about $5.
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
  --display-name="riskwatch ~\$5" \
  --budget-amount=7000KRW --calendar-period=month \
  --filter-projects="projects/$PROJECT_NUMBER" \
  --credit-types-treatment=include-specified-credits --filter-credit-types=FREE_TIER \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend \
  --notifications-rule-monitoring-notification-channels="$CHANNEL"
