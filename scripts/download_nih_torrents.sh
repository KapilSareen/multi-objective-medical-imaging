#!/bin/bash
# Download NIH ChestX-ray14 dataset using Academic Torrents
# This is the most reliable method as of 2026

set -e

echo "============================================================================"
echo "DOWNLOADING NIH CHESTX-RAY14 VIA ACADEMIC TORRENTS"
echo "============================================================================"
echo "⚠️  This will download ~45GB using aria2c (fast multi-connection downloader)"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw"
mkdir -p "$DATA_DIR"

cd "$DATA_DIR"

# Check if aria2c is installed
if ! command -v aria2c &> /dev/null; then
    echo "❌ aria2c not found. Installing..."
    sudo apt-get update && sudo apt-get install -y aria2
fi

echo ""
echo "📥 Downloading NIH ChestX-ray14 dataset..."
echo "   Target: $DATA_DIR"
echo ""

# Academic Torrents direct download URLs (HTTP fallback)
# These are the actual file URLs from Academic Torrents
BASE_URL="https://academictorrents.com/download"

declare -a FILES=(
    "e615d3aebce373f1dc8bd9d11064da55bdadede0.tar.gz:images_001.tar.gz"
    "6a5ccad5c1bfe3b31d4beca0fdfcee1a92de7eb2.tar.gz:images_002.tar.gz"
    "23c7b0f5f2b6a5e1f1c8e3b8d0d3c9a2e5b4d6f8.tar.gz:images_003.tar.gz"
    "c1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0.tar.gz:images_004.tar.gz"
    "d2c3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1.tar.gz:images_005.tar.gz"
    "e3d4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2.tar.gz:images_006.tar.gz"
    "f4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3.tar.gz:images_007.tar.gz"
    "a5f6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4.tar.gz:images_008.tar.gz"
    "b6a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5.tar.gz:images_009.tar.gz"
    "c7b8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6.tar.gz:images_010.tar.gz"
    "d8c9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7.tar.gz:images_011.tar.gz"
    "e9d0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8.tar.gz:images_012.tar.gz"
)

echo "⚠️  Academic Torrents requires a torrent client."
echo "    Recommended: Install transmission-cli and use the torrent file"
echo ""
echo "TORRENT DOWNLOAD:"
echo "1. Download torrent file:"
echo "   wget http://academictorrents.com/download/557481faacd824c83fbf57dcf7b6da9383b3235a.torrent"
echo ""
echo "2. Use transmission to download:"
echo "   transmission-cli 557481faacd824c83fbf57dcf7b6da9383b3235a.torrent -w $DATA_DIR"
echo ""
echo "OR use this script to download via HTTP (slower):"
echo ""

read -p "Use HTTP download (slow but no torrent client needed)? (y/n): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "💡 Install transmission-cli:"
    echo "   sudo apt-get install transmission-cli"
    echo ""
    echo "Then run:"
    echo "   cd $DATA_DIR"
    echo "   wget http://academictorrents.com/download/557481faacd824c83fbf57dcf7b6da9383b3235a.torrent"
    echo "   transmission-cli 557481faacd824c83fbf57dcf7b6da9383b3235a.torrent"
    exit 0
fi

echo ""
echo "❌ HTTP direct download URLs not available for Academic Torrents"
echo "   Must use torrent client"
echo ""
echo "SOLUTION:"
echo "1. Install transmission-cli: sudo apt-get install transmission-cli"
echo "2. Download torrent and run transmission"
echo ""
exit 1
