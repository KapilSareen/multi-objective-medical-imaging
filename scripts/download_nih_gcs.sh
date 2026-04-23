#!/bin/bash
# Download NIH ChestX-ray14 from Google Cloud Storage (public bucket)
# Source: gs://gcs-public-data--healthcare-nih-chest-xray

set -e

echo "============================================================================"
echo "DOWNLOADING NIH CHESTX-RAY14 FROM GOOGLE CLOUD STORAGE"
echo "============================================================================"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw/nih_images"
mkdir -p "$DATA_DIR"

echo "📋 Dataset Info:"
echo "   Source: Google Cloud Storage (public bucket)"
echo "   Total: ~112,000 PNG images (~45GB)"
echo "   Target: $DATA_DIR"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "❌ gsutil not found. Installing Google Cloud SDK..."
    echo ""
    echo "Run these commands:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
    echo "  gcloud init"
    echo ""
    echo "Or use snap:"
    echo "  sudo snap install google-cloud-sdk --classic"
    echo ""
    exit 1
fi

echo "✅ gsutil found"
echo ""
echo "📥 Downloading NIH ChestX-ray14 images..."
echo "   This will take 1-3 hours depending on network speed"
echo ""

# Note: This is a "Requester Pays" bucket, but we'll try anonymous access first
# Download all PNG files
gsutil -m cp -r \
  "gs://gcs-public-data--healthcare-nih-chest-xray/png/*.png" \
  "$DATA_DIR/" \
  || echo "⚠️  Requester Pays error - see below for solution"

# Check if download succeeded
IMAGE_COUNT=$(find "$DATA_DIR" -name "*.png" 2>/dev/null | wc -l)

if [ "$IMAGE_COUNT" -gt 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo "   Downloaded: $IMAGE_COUNT images"
    echo "   Location: $DATA_DIR"
else
    echo ""
    echo "❌ Download failed (Requester Pays bucket)"
    echo ""
    echo "SOLUTION: This bucket requires a Google Cloud project for billing."
    echo ""
    echo "Option 1 - Use your GCP project (costs ~\$1-2 for egress):"
    echo "  1. Create GCP project: https://console.cloud.google.com"
    echo "  2. Enable billing"
    echo "  3. Run: gcloud auth login"
    echo "  4. Run: gsutil -u YOUR_PROJECT_ID -m cp -r gs://gcs-public-data--healthcare-nih-chest-xray/png/*.png $DATA_DIR/"
    echo ""
    echo "Option 2 - Use Kaggle CLI (no GCP needed):"
    echo "  See: download_nih_kaggle_cli.sh"
    echo ""
    echo "Option 3 - Use Academic Torrents:"
    echo "  See: download_nih_torrents.sh"
    echo ""
    exit 1
fi
