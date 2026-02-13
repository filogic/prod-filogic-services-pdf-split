#!/bin/bash
# deploy.sh — Deploy PDF Splitter to Cloud Run
#
# Cloud Run detecteert automatisch de Dockerfile en bouwt + deployt in één stap.
# Tesseract, Poppler en alle Python dependencies zitten in het image.
#
# Usage:
#   export GCP_PROJECT_ID=my-project
#   export GCS_BUCKET=pdf-splitter-output
#   ./deploy.sh
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-filogic-opentms}"
REGION="${GCP_REGION:-europe-west4}"
GCS_BUCKET="${GCS_BUCKET:-filogic-opentms-tmp}"
SERVICE_NAME="prod-filogic-services-pdf-split"

# ── GCS bucket met auto-cleanup ────────────────────────────
echo "📦 Ensuring GCS bucket: ${GCS_BUCKET}"
gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --project="${PROJECT_ID}" --location="${REGION}" 2>/dev/null || true

gcloud storage buckets update "gs://${GCS_BUCKET}" \
    --lifecycle-file=<(cat <<'EOF'
{
  "rule": [{
    "action": {"type": "Delete"},
    "condition": {"age": 7}
  }]
}
EOF
)

# ── Deploy (bouwt image automatisch vanuit Dockerfile) ─────
echo "🚀 Deploying ${SERVICE_NAME}..."
gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 1 \
    --execution-environment gen2 \
    --cpu-boost \
    --min-instances 0 \
    --timeout 300s \
    --max-instances 10 \
    --set-env-vars "GCS_BUCKET=${GCS_BUCKET}"

URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" --region "${REGION}" \
    --format='value(status.url)')

echo ""
echo "✅ ${URL}"
echo ""
echo "Test:"
echo "  curl -X POST ${URL} \\"
echo "    -F 'file=@vrachtbrief.pdf' \\"
echo "    -F 'reference_pattern=Vrachtbrief\s+(\d+)'"
