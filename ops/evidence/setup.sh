#!/usr/bin/env bash
# Creates the hosted evidence plane: the bucket the prediction log is routed to, the sink that
# routes it, and the keyless federation the daily Action reads it through.
#
# Create-only and NOT idempotent: run it once against a project that has none of these. What
# exists is recorded in STATUS.md (Step 18, the evidence plane).
#
#   ops/evidence/setup.sh
set -euo pipefail

PROJECT=project-e9695c4a-c0d1-4e50-84f
PROJECT_NUMBER=46336906432
BUCKET=riskwatch-evidence-${PROJECT_NUMBER}
# GitHub's numeric ids, not names: a deleted repository's name can be re-registered by anyone,
# and a condition on the name would then admit them. The ids are never reused.
REPOSITORY_ID=1324779696

gcloud services enable iam.googleapis.com sts.googleapis.com iamcredentials.googleapis.com \
  --project "$PROJECT"

# Retention, unlocked: nothing written can be deleted for 120 days, which covers the 75-day run
# plus the write-up. Unlocked means the policy itself can be removed -- the only way to tear the
# bucket down early. It also pins any stray real request for the same 120 days.
gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location us-central1 \
  --default-storage-class STANDARD --uniform-bucket-level-access --public-access-prevention \
  --retention-period 120d

# The service writes one JSON line per prediction to stdout; Cloud Logging parses it into
# jsonPayload. The sink writes hourly shards under run.googleapis.com/stdout/YYYY/MM/DD/.
gcloud logging sinks create riskwatch-predictions "storage.googleapis.com/$BUCKET" \
  --project "$PROJECT" \
  --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="riskwatch" AND jsonPayload.log_type="prediction"'
WRITER=$(gcloud logging sinks describe riskwatch-predictions --project "$PROJECT" \
  --format='value(writerIdentity)')
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "$WRITER" \
  --role roles/storage.objectCreator

# No service account and no key: the federated principal is granted read on this bucket and
# nothing else. Only workflows on main of this repository get a token.
gcloud iam workload-identity-pools create github --project "$PROJECT" --location global \
  --display-name github
gcloud iam workload-identity-pools providers create-oidc github-actions --project "$PROJECT" \
  --location global --workload-identity-pool github \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping 'google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref' \
  --attribute-condition "assertion.repository_id=='$REPOSITORY_ID' && assertion.ref=='refs/heads/main'"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository_id/$REPOSITORY_ID" \
  --role roles/storage.objectViewer
