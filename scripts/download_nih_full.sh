#!/bin/bash
# Helper script to download full NIH ChestX-ray14 dataset
# Run this ONLY if you need the full dataset for Option 1 in 04_download_coreset.py
# Size: ~50GB total (12 tar.gz files)

set -e  # Exit on error

echo "============================================================================"
echo "DOWNLOADING NIH CHESTX-RAY14 FULL DATASET"
echo "============================================================================"
echo "⚠️  WARNING: This will download ~50GB of data"
echo "   Make sure you have sufficient storage and bandwidth"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled"
    exit 0
fi

# Create output directory
OUTDIR="../data/raw/nih_full_dataset"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Start timer
START_TIME=$(date +%s)

echo ""
echo "📥 Downloading 12 image archives..."
echo ""

# Download all 12 parts
# URLs from: https://nihcc.app.box.com/v/ChestXray-NIHCC

declare -a URLS=(
    "https://nihcc.app.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz:images_001.tar.gz"
    "https://nihcc.app.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz:images_002.tar.gz"
    "https://nihcc.app.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz:images_003.tar.gz"
    "https://nihcc.app.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz:images_004.tar.gz"
    "https://nihcc.app.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz:images_005.tar.gz"
    "https://nihcc.app.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz:images_006.tar.gz"
    "https://nihcc.app.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz:images_007.tar.gz"
    "https://nihcc.app.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz:images_008.tar.gz"
    "https://nihcc.app.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz:images_009.tar.gz"
    "https://nihcc.app.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz:images_010.tar.gz"
    "https://nihcc.app.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w8sevk4q.gz:images_011.tar.gz"
    "https://nihcc.app.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz:images_012.tar.gz"
)

# Download each file
for i in "${!URLS[@]}"; do
    IFS=':' read -r URL FILENAME <<< "${URLS[$i]}"
    PART=$((i + 1))
    
    echo "[$PART/12] Downloading $FILENAME..."
    
    # Skip if exists
    if [ -f "$FILENAME" ]; then
        echo "   ⏭️  Already exists, skipping"
        continue
    fi
    
    # Download with wget (shows progress)
    wget --no-check-certificate -O "$FILENAME" "$URL"
    
    if [ $? -eq 0 ]; then
        SIZE=$(du -h "$FILENAME" | cut -f1)
        echo "   ✅ Downloaded $FILENAME ($SIZE)"
    else
        echo "   ❌ Failed to download $FILENAME"
        exit 1
    fi
done

echo ""
echo "📦 Extracting archives..."
echo ""

# Extract all tar.gz files
mkdir -p images
for f in images_*.tar.gz; do
    echo "Extracting $f..."
    tar -xzf "$f" -C images/
    echo "   ✅ Extracted $f"
done

# Calculate time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

echo ""
echo "============================================================================"
echo "DOWNLOAD COMPLETE"
echo "============================================================================"
echo "📁 Location: $(pwd)/images"
echo "📊 Total images: $(find images -type f | wc -l)"
echo "💾 Total size: $(du -sh images | cut -f1)"
echo "⏱️  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "📌 Next: Go back and run 04_download_coreset.py to copy selected images"
echo "============================================================================"
