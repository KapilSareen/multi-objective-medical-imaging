#!/bin/bash
# Download NIH ChestX-ray14 using Kaggle CLI
# This downloads the actual ZIP files, not just metadata

set -e

echo "============================================================================"
echo "DOWNLOADING NIH CHESTX-RAY14 VIA KAGGLE CLI"
echo "============================================================================"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$PROJECT_ROOT/data/raw"
mkdir -p "$DOWNLOAD_DIR"

cd "$DOWNLOAD_DIR"

echo "📋 Dataset Info:"
echo "   Source: Kaggle (nih-chest-xrays/data)"
echo "   Total: 12 ZIP files (~42GB)"
echo "   Target: $DOWNLOAD_DIR"
echo ""

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check for Kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "✅ Kaggle CLI configured"
echo ""
echo "📥 Downloading NIH ChestX-ray14 dataset..."
echo "   This will take 1-3 hours (42GB total)"
echo ""

# Download the full dataset using Kaggle CLI
# This downloads ALL files including the 12 image ZIPs
kaggle datasets download -d nih-chest-xrays/data -p "$DOWNLOAD_DIR" --unzip

# Check what was downloaded
echo ""
echo "📦 Checking downloaded files..."
ZIP_COUNT=$(ls -1 images_*.zip 2>/dev/null | wc -l)

if [ "$ZIP_COUNT" -eq 12 ]; then
    echo "✅ All 12 ZIP files downloaded!"
    echo ""
    echo "📦 Extracting images..."
    
    for i in {1..12}; do
        ZIP_FILE=$(printf "images_%03d.zip" $i)
        if [ -f "$ZIP_FILE" ]; then
            echo "  [$i/12] Extracting $ZIP_FILE..."
            unzip -q "$ZIP_FILE"
            rm "$ZIP_FILE"  # Delete ZIP after extraction to save space
        fi
    done
    
    IMAGE_COUNT=$(find images_*/images -name "*.png" 2>/dev/null | wc -l)
    echo ""
    echo "✅ Extraction complete!"
    echo "   Total images: $IMAGE_COUNT"
    echo "   Location: $DOWNLOAD_DIR/images_*/images/"
    
else
    echo "⚠️  Expected 12 ZIP files, found: $ZIP_COUNT"
    echo ""
    echo "Files downloaded:"
    ls -lh
    echo ""
    echo "If ZIPs are missing, the kaggle download might have failed."
    echo "Try running again or download manually from:"
    echo "  https://www.kaggle.com/datasets/nih-chest-xrays/data"
fi

echo ""
echo "============================================================================"
