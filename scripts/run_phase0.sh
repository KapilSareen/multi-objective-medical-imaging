#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PHASE_START=$(date +%s)

echo "================================================================================"
echo -e "${BLUE}PHASE 0: ZCORE CORESET SELECTION${NC}"
echo "================================================================================"
echo "Working directory: $PROJECT_ROOT"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

source venv/bin/activate

python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Step 1
echo ""
echo "================================================================================"
echo -e "${BLUE}STEP 1/5: DOWNLOAD METADATA${NC}"
echo "================================================================================"
python scripts/01_download_metadata.py || exit 1

# Step 1.5 - NEW
echo ""
echo "================================================================================"
echo -e "${BLUE}STEP 1.5/5: DOWNLOAD FULL NIH DATASET${NC}"
echo "================================================================================"
echo -e "${YELLOW}⚠️  This downloads ~50GB and takes 1-3 hours${NC}"
python scripts/01.5_download_nih_images.py || exit 1

# Step 2
echo ""
echo "================================================================================"
echo -e "${BLUE}STEP 2/5: GENERATE CLIP EMBEDDINGS${NC}"
echo "================================================================================"
echo -e "${YELLOW}⚠️  This takes 2-3 hours with GPU${NC}"
python scripts/02_generate_embeddings.py --non-interactive || exit 1

# Step 3
echo ""
echo "================================================================================"
echo -e "${BLUE}STEP 3/5: RUN ZCORE SELECTION${NC}"
echo "================================================================================"
python scripts/03_run_zcore.py || exit 1

# Step 4
echo ""
echo "================================================================================"
echo -e "${BLUE}STEP 4/5: CREATE CORESET DIRECTORY${NC}"
echo "================================================================================"
python scripts/04_download_coreset.py || exit 1

PHASE_END=$(date +%s)
PHASE_DURATION=$((PHASE_END - PHASE_START))
PHASE_HOURS=$((PHASE_DURATION / 3600))
PHASE_MINUTES=$(( (PHASE_DURATION % 3600) / 60 ))

echo ""
echo "================================================================================"
echo -e "${GREEN}PHASE 0 COMPLETE!${NC}"
echo "================================================================================"
echo "Total time: ${PHASE_HOURS}h ${PHASE_MINUTES}m"
echo ""
echo "📦 OUTPUT:"
echo "   • data/raw/nih_coreset_images/ (~10GB, 22K images)"
echo "   • data/processed/nih_coreset.csv"
echo "   • data/cache/nih_embeddings.npy"
echo ""
echo "📌 NEXT: Begin Phase 1 (Model Training)"
echo "================================================================================"
