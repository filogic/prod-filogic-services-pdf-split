#!/bin/bash
# update-service.sh — Update Cloud Run service configuration
#
# Run this ONCE to apply the correct resource settings.
# These settings persist across redeployments — Cloud Run only
# updates the container image on subsequent deploys, not the
# service config.
#
# Usage:
#   chmod +x update-service.sh
#   ./update-service.sh
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-filogic-opentms}"
REGION="${GCP_REGION:-europe-west4}"
SERVICE_NAME="prod-filogic-services-pdf-split"
GCS_BUCKET="${GCS_BUCKET:-filogic-opentms-tmp}"

echo "Updating ${SERVICE_NAME} in ${PROJECT_ID} (${REGION})..."

gcloud run services update "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 1 \
    --execution-environment gen2 \
    --cpu-boost \
    --min-instances 0 \
    --timeout 300s \
    --max-instances 10 \
    --update-env-vars "GCS_BUCKET=${GCS_BUCKET}"

echo ""
echo "Done. Current config:"
gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format="table(
        spec.template.spec.containers[0].resources.limits.memory,
        spec.template.spec.containers[0].resources.limits.cpu,
        spec.template.metadata.annotations['autoscaling.knative.dev/maxScale'],
        spec.template.metadata.annotations['run.googleapis.com/execution-environment'],
        spec.template.spec.containerConcurrency
    )"
